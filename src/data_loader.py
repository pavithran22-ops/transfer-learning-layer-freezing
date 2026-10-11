"""
src/data_loader.py
──────────────────
Handles dataset discovery, validation, preprocessing, augmentation, and
splitting into train / validation / test TensorFlow datasets.
"""

import os
import sys
import random
import pathlib
from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(data_dir: str = config.DATA_DIR) -> List[str]:
    """
    Check that *data_dir* exists and contains at least two sub-folders (classes),
    each holding at least one supported image.

    Returns
    -------
    class_names : list[str]
        Sorted list of class names (sub-folder names).

    Raises
    ------
    SystemExit
        Printed error + exit when the dataset is missing or malformed.
    """
    if not os.path.isdir(data_dir):
        print(
            f"\n[ERROR] Dataset directory not found:\n"
            f"        {data_dir}\n\n"
            f"  Please follow the instructions in data/README.md to set up\n"
            f"  your dataset before running the project.\n"
        )
        sys.exit(1)

    # Collect sub-directories
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if len(class_names) < 2:
        print(
            f"\n[ERROR] Expected at least 2 class sub-folders inside:\n"
            f"        {data_dir}\n"
            f"  Found: {class_names}\n\n"
            f"  Each sub-folder must be named after a class and contain images.\n"
        )
        sys.exit(1)

    # Verify each class has at least one valid image
    empty_classes = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        images = [
            f for f in os.listdir(cls_dir)
            if pathlib.Path(f).suffix.lower() in config.VALID_EXTENSIONS
        ]
        if not images:
            empty_classes.append(cls)

    if empty_classes:
        print(
            f"\n[ERROR] The following class folders contain no supported images\n"
            f"        ({', '.join(config.VALID_EXTENSIONS)}):\n"
            f"        {empty_classes}\n"
        )
        sys.exit(1)

    print(f"[INFO] Dataset validated. Classes found ({len(class_names)}): {class_names}")
    return class_names


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Collect all image paths and labels
# ─────────────────────────────────────────────────────────────────────────────

def collect_image_paths(
    data_dir: str,
    class_names: List[str],
) -> Tuple[List[str], List[int]]:
    """
    Walk *data_dir* and return parallel lists of image paths and integer labels.
    Skips unreadable files gracefully.
    """
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if pathlib.Path(fname).suffix.lower() not in config.VALID_EXTENSIONS:
                continue
            # Quick sanity check: file must be non-empty
            try:
                if os.path.getsize(fpath) == 0:
                    print(f"[WARN] Skipping empty file: {fpath}")
                    continue
            except OSError as exc:
                print(f"[WARN] Cannot access {fpath}: {exc}")
                continue
            paths.append(fpath)
            labels.append(idx)

    print(f"[INFO] Total images collected: {len(paths)}")
    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Train / validation / test split (no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    paths: List[str],
    labels: List[int],
    train_split: float = config.TRAIN_SPLIT,
    val_split: float   = config.VAL_SPLIT,
    seed: int          = config.RANDOM_SEED,
) -> Tuple[List, List, List, List, List, List]:
    """
    Stratified split so each split contains roughly the same class proportions.
    Returns (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels).
    """
    from collections import defaultdict

    # Group by class
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    rng = random.Random(seed)

    train_p, train_l = [], []
    val_p,   val_l   = [], []
    test_p,  test_l  = [], []

    for lbl, idxs in class_indices.items():
        rng.shuffle(idxs)
        n     = len(idxs)
        n_tr  = max(1, int(n * train_split))
        n_val = max(1, int(n * val_split))

        for i in idxs[:n_tr]:
            train_p.append(paths[i]); train_l.append(labels[i])
        for i in idxs[n_tr:n_tr + n_val]:
            val_p.append(paths[i]);   val_l.append(labels[i])
        for i in idxs[n_tr + n_val:]:
            test_p.append(paths[i]);  test_l.append(labels[i])

    # Shuffle training set
    combined = list(zip(train_p, train_l))
    rng.shuffle(combined)
    train_p, train_l = zip(*combined) if combined else ([], [])

    print(
        f"[INFO] Split → train: {len(train_p)}, "
        f"val: {len(val_p)}, test: {len(test_p)}"
    )
    return (
        list(train_p), list(train_l),
        list(val_p),   list(val_l),
        list(test_p),  list(test_l),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TensorFlow dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_path: tf.Tensor, label: tf.Tensor, num_classes: int) -> Tuple:
    """Load, decode, resize, and MobileNetV2-preprocess one image."""
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple:
    """Apply random augmentations to training images only."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Random rotation ±15° via crop-and-resize trick
    image = tf.keras.layers.RandomRotation(0.05)(image, training=True)
    return image, label


def build_dataset(
    paths: List[str],
    labels: List[int],
    num_classes: int,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = config.RANDOM_SEED,
) -> tf.data.Dataset:
    """
    Create a batched, prefetched tf.data.Dataset from lists of paths & labels.

    Parameters
    ----------
    paths       : list of image file paths
    labels      : list of integer class labels
    num_classes : total number of classes
    batch_size  : mini-batch size
    augment     : if True, apply random augmentation (training only)
    shuffle     : if True, shuffle the dataset each epoch
    seed        : random seed for shuffling
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _preprocess(p, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public convenience function
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    data_dir: str  = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE_A,
) -> Tuple:
    """
    Full pipeline: validate → collect → split → build tf.data datasets.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    class_names               : list[str]
    (test_paths, test_labels) : raw lists for evaluation
    """
    class_names = validate_dataset(data_dir)
    num_classes = len(class_names)

    paths, labels = collect_image_paths(data_dir, class_names)

    (train_p, train_l,
     val_p,   val_l,
     test_p,  test_l) = split_data(paths, labels)

    train_ds = build_dataset(
        train_p, train_l, num_classes, batch_size,
        augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_p, val_l, num_classes, batch_size,
        augment=False, shuffle=False,
    )
    test_ds = build_dataset(
        test_p, test_l, num_classes, batch_size,
        augment=False, shuffle=False,
    )

    return train_ds, val_ds, test_ds, class_names, (test_p, test_l)
