"""
src/model_builder.py
─────────────────────
Builds Model A (frozen transfer learning) and Model B (fine-tuned transfer learning)
using MobileNetV2 pretrained on ImageNet.
"""

import sys
import os
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the shared classification head
# ─────────────────────────────────────────────────────────────────────────────

def _build_head(base: keras.Model, num_classes: int, model_name: str) -> keras.Model:
    """
    Attach a Global Average Pooling → Dropout → Dense head to *base*.

    The output layer uses:
    - sigmoid  (binary)    when num_classes == 2
    - softmax  (multiclass) when num_classes > 2
    """
    x = base.output
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)

    if num_classes == 2:
        outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=base.input, outputs=outputs, name=model_name)
    return model


def _loss_and_metrics(num_classes: int):
    """Return appropriate loss function and metrics list."""
    if num_classes == 2:
        loss    = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.AUC(name="auc")]
    else:
        loss    = "categorical_crossentropy"
        metrics = ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top3_acc")]
    return loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model A — Frozen Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

def build_model_a(num_classes: int) -> keras.Model:
    """
    Build Model A: MobileNetV2 base fully frozen + classification head.

    The entire pretrained convolutional base is frozen — only the newly added
    Dense layer is trained.

    Parameters
    ----------
    num_classes : int  Number of target classes.

    Returns
    -------
    model : compiled keras.Model
    """
    base = keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    # ── Freeze the entire base ─────────────────────────────────────────────
    base.trainable = False

    model = _build_head(base, num_classes, model_name="ModelA_Frozen")

    loss, metrics = _loss_and_metrics(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR_A),
        loss=loss,
        metrics=metrics,
    )

    # Report trainable vs total parameters
    trainable     = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_trainable = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(
        f"[Model A] Total params: {trainable + non_trainable:,} | "
        f"Trainable: {trainable:,} | Non-trainable: {non_trainable:,}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model B — Fine-Tuned Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

def build_model_b_phase1(num_classes: int) -> keras.Model:
    """
    Build Model B — Phase 1: identical to Model A (base fully frozen).
    Train only the classification head to a good initialisation before
    unfreezing deeper layers.

    Returns
    -------
    model : compiled keras.Model  (base frozen)
    """
    base = keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # start frozen — same as Model A

    model = _build_head(base, num_classes, model_name="ModelB_FineTuned")

    loss, metrics = _loss_and_metrics(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR_B_HEAD),
        loss=loss,
        metrics=metrics,
    )

    trainable     = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_trainable = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(
        f"[Model B Phase-1] Total params: {trainable + non_trainable:,} | "
        f"Trainable: {trainable:,} | Non-trainable: {non_trainable:,}"
    )
    return model


def unfreeze_for_finetuning(model: keras.Model, num_classes: int) -> keras.Model:
    """
    Phase 2: unfreeze MobileNetV2 layers from index *config.FINE_TUNE_AT* onward
    while keeping BatchNormalization layers frozen for training stability.

    Recompiles the model with a lower learning rate.

    Parameters
    ----------
    model       : the Model B returned by build_model_b_phase1 (already partly trained)
    num_classes : int

    Returns
    -------
    model : recompiled keras.Model  (selected layers now trainable)
    """
    # The sub-model called 'mobilenetv2_...' is the base
    base = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base = layer
            break

    if base is None:
        raise RuntimeError("Could not locate the MobileNetV2 base inside Model B.")

    # Unfreeze the entire base first, then selectively re-freeze
    base.trainable = True

    for layer in base.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False  # keep early layers frozen

    # Always freeze BatchNormalization layers to prevent instability
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    # Recompile with smaller learning rate
    loss, metrics = _loss_and_metrics(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR_B_FINE),
        loss=loss,
        metrics=metrics,
    )

    trainable     = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_trainable = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(
        f"[Model B Phase-2] Total params: {trainable + non_trainable:,} | "
        f"Trainable: {trainable:,} | Non-trainable: {non_trainable:,}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Utility: count parameters
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: keras.Model) -> dict:
    """Return a dict with trainable and non-trainable parameter counts."""
    trainable     = int(sum(tf.size(v).numpy() for v in model.trainable_variables))
    non_trainable = int(sum(tf.size(v).numpy() for v in model.non_trainable_variables))
    return {
        "trainable":     trainable,
        "non_trainable": non_trainable,
        "total":         trainable + non_trainable,
    }
