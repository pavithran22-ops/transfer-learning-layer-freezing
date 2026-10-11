"""
src/train.py
─────────────
Training routines for Model A (frozen) and Model B (fine-tuned).
Returns training histories and wall-clock durations.
"""

import os
import sys
import time

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model_builder import build_model_a, build_model_b_phase1, unfreeze_for_finetuning


# ─────────────────────────────────────────────────────────────────────────────
# Shared callbacks factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_callbacks(checkpoint_path: str, monitor: str = "val_accuracy") -> list:
    """
    Return a list of standard Keras callbacks:
    - ModelCheckpoint  — saves the best model weights
    - EarlyStopping    — halts training when *monitor* stops improving
    - ReduceLROnPlateau — halves LR when plateau is detected
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=config.LR_FACTOR,
            patience=config.PATIENCE_LR,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Model A — Frozen Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

def train_model_a(
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    num_classes: int,
) -> tuple:
    """
    Build and train Model A (frozen base).

    Returns
    -------
    model    : trained keras.Model
    history  : keras.callbacks.History
    duration : float  wall-clock seconds
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  MODEL A — Frozen Transfer Learning")
    print("="*60)

    model = build_model_a(num_classes)

    callbacks = _make_callbacks(config.MODEL_A_PATH)

    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_A,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.time() - t_start

    # Persist the best model
    model.save(config.MODEL_A_PATH)
    print(f"\n[Model A] Training complete in {duration:.1f}s — saved to {config.MODEL_A_PATH}")

    return model, history, duration


# ─────────────────────────────────────────────────────────────────────────────
# Model B — Fine-Tuned Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

def train_model_b(
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    num_classes: int,
) -> tuple:
    """
    Build and train Model B in two phases:
      Phase 1 — frozen base, head only.
      Phase 2 — selected deeper layers unfrozen, low LR.

    Returns
    -------
    model         : trained keras.Model
    history_head  : History from Phase 1
    history_fine  : History from Phase 2
    duration      : float  total wall-clock seconds
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  MODEL B — Fine-Tuned Transfer Learning")
    print("="*60)

    # ── Phase 1: train head only ───────────────────────────────────────────
    print("\n--- Phase 1: Training classification head (base frozen) ---")
    model = build_model_b_phase1(num_classes)

    ckpt_phase1 = os.path.join(config.MODELS_DIR, "model_b_phase1_best.keras")
    callbacks_p1 = _make_callbacks(ckpt_phase1)

    t_start = time.time()
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_B_HEAD,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ── Phase 2: fine-tune deeper layers ──────────────────────────────────
    print("\n--- Phase 2: Fine-tuning deeper layers (low LR) ---")
    model = unfreeze_for_finetuning(model, num_classes)

    callbacks_p2 = _make_callbacks(config.MODEL_B_PATH)

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_B_FINE,
        callbacks=callbacks_p2,
        verbose=1,
    )
    duration = time.time() - t_start

    # Persist the best model
    model.save(config.MODEL_B_PATH)
    print(f"\n[Model B] Training complete in {duration:.1f}s — saved to {config.MODEL_B_PATH}")

    return model, history_head, history_fine, duration
