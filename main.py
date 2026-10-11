"""
main.py
--------
Entry point for the Transfer Learning with Layer Freezing project.

Usage
-----
# Full training + evaluation pipeline
python main.py

# Predict the class of a single image
python main.py --predict path/to/image.jpg --model a
python main.py --predict path/to/image.jpg --model b
"""

import argparse
import os
import sys
import pathlib

# -- silence TF progress bars at startup ---------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import config
from src.data_loader   import load_data
from src.train         import train_model_a, train_model_b
from src.evaluate      import evaluate_model, compare_models, load_trained_model
from src.visualize     import generate_all_plots
from src.model_builder import count_params


# -----------------------------------------------------------------------------
# Prediction helper
# -----------------------------------------------------------------------------

def predict_image(image_path: str, model_choice: str, class_names: list):
    """
    Predict the class of a single image.

    Parameters
    ----------
    image_path   : str   Path to the image file.
    model_choice : str   'a' or 'b'.
    class_names  : list  Class names from the dataset.
    """
    # -- validate image path ------------------------------------------------
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    suffix = pathlib.Path(image_path).suffix.lower()
    if suffix not in config.VALID_EXTENSIONS:
        print(
            f"[ERROR] Unsupported image format '{suffix}'.\n"
            f"  Supported: {config.VALID_EXTENSIONS}"
        )
        sys.exit(1)

    # -- load the requested model -------------------------------------------
    model_path = config.MODEL_A_PATH if model_choice == "a" else config.MODEL_B_PATH
    model_label = "Model A (Frozen)" if model_choice == "a" else "Model B (Fine-Tuned)"

    try:
        model = load_trained_model(model_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # -- preprocess --------------------------------------------------------
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, 0)    # add batch dimension

    # -- predict -----------------------------------------------------------
    preds = model.predict(image, verbose=0)

    pred_idx   = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])
    pred_class = class_names[pred_idx]

    print("\n" + "="*50)
    print(f"  Prediction Result")
    print("="*50)
    print(f"  Model used  : {model_label}")
    print(f"  Image       : {image_path}")
    print(f"  Predicted   : {pred_class}")
    print(f"  Confidence  : {confidence*100:.2f}%")
    print("="*50 + "\n")


# -----------------------------------------------------------------------------
# Full training + evaluation pipeline
# -----------------------------------------------------------------------------

def run_pipeline():
    """Execute the complete workflow from data loading to results generation."""

    print("\n" + "*"*60)
    print("  Transfer Learning with Layer Freezing")
    print("  MobileNetV2 - Frozen vs Fine-Tuned")
    print("*"*60)

    # -- 1. Create output directories ---------------------------------------
    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # -- 2. Load and preprocess data ----------------------------------------
    print("\n[STEP 1/7] Loading and preprocessing dataset ...")
    train_ds, val_ds, test_ds, class_names, _ = load_data(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE_A,
    )
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

    # -- 3. Train Model A ---------------------------------------------------
    print("\n[STEP 2/7] Training Model A (Frozen Transfer Learning) ...")
    model_a, history_a, duration_a = train_model_a(train_ds, val_ds, num_classes)

    # -- 4. Train Model B ---------------------------------------------------
    print("\n[STEP 3/7] Training Model B (Fine-Tuned Transfer Learning) ...")
    model_b, history_b_head, history_b_fine, duration_b = train_model_b(
        train_ds, val_ds, num_classes
    )

    # -- 5. Evaluate both models --------------------------------------------
    print("\n[STEP 4/7] Evaluating Model A on test set ...")
    metrics_a = evaluate_model(
        model_a, test_ds, class_names,
        model_name="Model A (Frozen)",
        duration=duration_a,
    )

    print("\n[STEP 5/7] Evaluating Model B on test set ...")
    metrics_b = evaluate_model(
        model_b, test_ds, class_names,
        model_name="Model B (Fine-Tuned)",
        duration=duration_b,
    )

    # -- 6. Compare models --------------------------------------------------
    print("\n[STEP 6/7] Comparing models and saving CSV ...")
    compare_models(metrics_a, metrics_b)

    # -- 7. Generate plots --------------------------------------------------
    print("\n[STEP 7/7] Generating plots ...")
    generate_all_plots(
        history_a,
        [history_b_head, history_b_fine],
        metrics_a,
        metrics_b,
        class_names,
    )

    # -- Final summary ------------------------------------------------------
    print("\n" + "="*60)
    print("  EXPERIMENT COMPLETE")
    print("="*60)
    print(f"  Dataset      : {config.DATA_DIR}")
    print(f"  Classes      : {class_names}")
    print(f"  Saved models : {config.MODELS_DIR}")
    print(f"  Results      : {config.RESULTS_DIR}")
    print("\n  Output files:")
    for fname in sorted(os.listdir(config.RESULTS_DIR)):
        print(f"    results/{fname}")
    print("="*60 + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transfer Learning with Layer Freezing -- MobileNetV2"
    )
    parser.add_argument(
        "--predict", metavar="IMAGE_PATH",
        help="Path to an image for single-image prediction (skips training).",
    )
    parser.add_argument(
        "--model", choices=["a", "b"], default="b",
        help="Which trained model to use for prediction: 'a' (frozen) or 'b' (fine-tuned). Default: b",
    )
    args = parser.parse_args()

    if args.predict:
        # -- Prediction mode -----------------------------------------------
        # Derive class names from the dataset directory (must exist)
        from src.data_loader import validate_dataset
        class_names = validate_dataset(config.DATA_DIR)
        predict_image(args.predict, args.model, class_names)
    else:
        # -- Full pipeline -------------------------------------------------
        run_pipeline()


if __name__ == "__main__":
    main()
