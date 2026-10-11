"""
src/evaluate.py
────────────────
Evaluation utilities: test-set metrics, confusion matrix, classification report,
and comparison summary saved as CSV.
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Collect ground-truth and prediction arrays from a tf.data.Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset, num_classes: int):
    """
    Run *model* over *dataset* and return (y_true, y_pred) as integer arrays.
    """
    y_true_list, y_pred_list = [], []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)

        y_pred_list.extend(pred_classes.tolist())
        y_true_list.extend(true_classes.tolist())

    return np.array(y_true_list), np.array(y_pred_list)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:       tf.keras.Model,
    test_ds:     tf.data.Dataset,
    class_names: list,
    model_name:  str,
    duration:    float,
) -> dict:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix on *test_ds*.

    Returns a metrics dict with all scalar values.
    """
    num_classes = len(class_names)
    y_true, y_pred = get_predictions(model, test_ds, num_classes)

    avg = "binary" if num_classes == 2 else "weighted"

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    cr   = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Count trainable parameters
    trainable_params = int(sum(tf.size(v).numpy() for v in model.trainable_variables))

    metrics = {
        "model_name":        model_name,
        "test_accuracy":     round(acc  * 100, 4),
        "precision":         round(prec * 100, 4),
        "recall":            round(rec  * 100, 4),
        "f1_score":          round(f1   * 100, 4),
        "training_time_s":   round(duration, 2),
        "trainable_params":  trainable_params,
        "confusion_matrix":  cm,
        "classification_report": cr,
        "y_true":            y_true,
        "y_pred":            y_pred,
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} — Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Train time: {duration:.1f}s")
    print(f"\nClassification Report:\n{cr}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Compare and save summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(metrics_a: dict, metrics_b: dict) -> pd.DataFrame:
    """
    Build and print a side-by-side comparison table, save it as CSV.

    Returns the DataFrame.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    rows = []
    for m in (metrics_a, metrics_b):
        rows.append({
            "Model":               m["model_name"],
            "Test Accuracy (%)":   m["test_accuracy"],
            "Precision (%)":       m["precision"],
            "Recall (%)":          m["recall"],
            "F1-Score (%)":        m["f1_score"],
            "Training Time (s)":   m["training_time_s"],
            "Trainable Params":    m["trainable_params"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_CSV, index=False)
    print(f"\n[INFO] Comparison table saved → {config.RESULTS_CSV}")

    print("\n" + "="*70)
    print("  MODEL COMPARISON SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # ── Announce the winner ───────────────────────────────────────────────
    acc_a = metrics_a["test_accuracy"]
    acc_b = metrics_b["test_accuracy"]

    if acc_b > acc_a:
        winner = metrics_b["model_name"]
        margin = acc_b - acc_a
        print(f"\n[WINNER] {winner} achieved higher test accuracy by {margin:.2f}%.")
    elif acc_a > acc_b:
        winner = metrics_a["model_name"]
        margin = acc_a - acc_b
        print(f"\n[WINNER] {winner} achieved higher test accuracy by {margin:.2f}%.")
    else:
        print("\n[TIE] Both models achieved identical test accuracy.")

    return df


# -----------------------------------------------------------------------------
# 4.  Load trained model for inference
# -----------------------------------------------------------------------------

def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model from *model_path* with clear error handling."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at:\n  {model_path}\n"
            "Please run `python main.py` first to train the models."
        )
    print(f"[INFO] Loading model from {model_path} ...")
    return tf.keras.models.load_model(model_path)
