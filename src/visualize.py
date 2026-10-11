"""
src/visualize.py
─────────────────
Generates and saves all plots:
  - Training/validation accuracy curves
  - Training/validation loss curves
  - Confusion matrices (heatmaps)
  - Bar charts comparing accuracy and training time
"""

import os
import sys
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
PALETTE = {"A": "#4A90D9", "B": "#E87040"}   # blue for Model A, orange for Model B


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_results_dir():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str):
    """Save *fig* to results/ and close it."""
    _ensure_results_dir()
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Training curves (accuracy + loss)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metric(history_objects: list, key: str) -> list:
    """
    Concatenate *key* from one or more History objects into a single list.
    Handles both 'accuracy' and 'val_accuracy' safely.
    """
    values = []
    for h in history_objects:
        if key in h.history:
            values.extend(h.history[key])
    return values


def plot_training_curves(
    history_a,              # keras.callbacks.History  (single, Model A)
    history_b_list: list,   # list of History objects  (two phases, Model B)
    filename_prefix: str = "",
):
    """
    Plot accuracy and loss curves for Model A and Model B side by side.

    history_b_list = [history_head, history_fine]
    """
    # ── Accuracy ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Model A vs Model B", fontsize=14, fontweight="bold")

    def _plot_pair(ax, train_a, val_a, train_b, val_b, title, ylabel):
        epochs_a = range(1, len(train_a) + 1)
        epochs_b = range(1, len(train_b) + 1)
        ax.plot(epochs_a, train_a, color=PALETTE["A"], linestyle="-",  label="A train")
        ax.plot(epochs_a, val_a,   color=PALETTE["A"], linestyle="--", label="A val")
        ax.plot(epochs_b, train_b, color=PALETTE["B"], linestyle="-",  label="B train")
        ax.plot(epochs_b, val_b,   color=PALETTE["B"], linestyle="--", label="B val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    acc_a_tr  = _extract_metric([history_a], "accuracy")
    acc_a_val = _extract_metric([history_a], "val_accuracy")
    acc_b_tr  = _extract_metric(history_b_list, "accuracy")
    acc_b_val = _extract_metric(history_b_list, "val_accuracy")

    loss_a_tr  = _extract_metric([history_a], "loss")
    loss_a_val = _extract_metric([history_a], "val_loss")
    loss_b_tr  = _extract_metric(history_b_list, "loss")
    loss_b_val = _extract_metric(history_b_list, "val_loss")

    _plot_pair(axes[0], acc_a_tr, acc_a_val, acc_b_tr, acc_b_val,
               "Accuracy over Epochs", "Accuracy")
    _plot_pair(axes[1], loss_a_tr, loss_a_val, loss_b_tr, loss_b_val,
               "Loss over Epochs", "Loss")

    fname = f"{filename_prefix}training_curves.png" if filename_prefix else "training_curves.png"
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
    model_name:  str,
    filename:    str,
):
    """Plot and save a normalised confusion matrix heatmap."""
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalise

    n = len(class_names)
    fig_size = max(6, n)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_n, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": max(8, 12 - n)},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Comparison bar charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bars(metrics_a: dict, metrics_b: dict):
    """
    Two bar charts side by side:
      Left  — Test accuracy comparison
      Right — Training time comparison
    """
    names  = [metrics_a["model_name"], metrics_b["model_name"]]
    colors = [PALETTE["A"], PALETTE["B"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    # Accuracy
    accs = [metrics_a["test_accuracy"], metrics_b["test_accuracy"]]
    bars = axes[0].bar(names, accs, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Test Accuracy (%)"); axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%)")
    for bar, val in zip(bars, accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[0].grid(axis="y", alpha=0.3)

    # Training time
    times = [metrics_a["training_time_s"], metrics_b["training_time_s"]]
    bars2 = axes[1].bar(names, times, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    axes[1].set_title("Training Time (seconds)"); axes[1].set_ylabel("Time (s)")
    for bar, val in zip(bars2, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[1].grid(axis="y", alpha=0.3)

    _save(fig, "comparison_bar_charts.png")


# -----------------------------------------------------------------------------
# 4.  Sample images visualizations
# -----------------------------------------------------------------------------

def plot_sample_predictions(
    model,
    test_ds,
    class_names: list,
    num_samples: int = 6,
    filename: str = "sample_predictions.png",
):
    """Display a grid of test images with their true and predicted labels."""
    if model is None or test_ds is None:
        return

    images_list, labels_list = [], []
    for imgs, lbls in test_ds:
        images_list.append(imgs.numpy())
        labels_list.append(lbls.numpy())
        if sum(len(x) for x in images_list) >= num_samples:
            break

    if not images_list:
        return

    all_images = np.concatenate(images_list, axis=0)[:num_samples]
    all_labels = np.concatenate(labels_list, axis=0)[:num_samples]

    preds = model.predict(all_images, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    true_indices = np.argmax(all_labels, axis=1)
    confidences  = np.max(preds, axis=1)

    n = len(all_images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten()

    for i in range(n):
        # Reverse MobileNetV2 preprocessing: [-1, 1] -> [0, 1]
        img = (all_images[i] + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)

        true_cls = class_names[true_indices[i]]
        pred_cls = class_names[pred_indices[i]]
        conf = confidences[i] * 100

        is_correct = (true_indices[i] == pred_indices[i])
        color = "#2E7D32" if is_correct else "#C62828"

        axes_flat[i].imshow(img)
        axes_flat[i].axis("off")
        title_text = f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)"
        axes_flat[i].set_title(title_text, fontsize=10, fontweight="bold", color=color)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Sample Test Predictions (Model B -- Fine-Tuned)", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, filename)


def plot_dataset_samples(
    data_dir: str,
    class_names: list,
    samples_per_class: int = 3,
    filename: str = "sample_dataset_images.png",
):
    """Show representative sample images from each class folder."""
    from PIL import Image

    num_classes = len(class_names)
    if num_classes == 0:
        return

    fig, axes = plt.subplots(
        num_classes, samples_per_class,
        figsize=(3.5 * samples_per_class, 3.2 * num_classes),
    )

    axes_grid = np.array(axes).reshape(num_classes, samples_per_class)

    for row_idx, cls in enumerate(class_names):
        cls_folder = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        files = [
            f for f in sorted(os.listdir(cls_folder))
            if os.path.splitext(f)[1].lower() in config.VALID_EXTENSIONS
        ][:samples_per_class]

        for col_idx in range(samples_per_class):
            ax = axes_grid[row_idx, col_idx]
            if col_idx < len(files):
                img_path = os.path.join(cls_folder, files[col_idx])
                try:
                    im = Image.open(img_path).convert("RGB")
                    ax.imshow(im)
                    if col_idx == 0:
                        ax.set_ylabel(cls, fontsize=11, fontweight="bold")
                except Exception:
                    pass
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Sample Dataset Images by Class", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, filename)


# -----------------------------------------------------------------------------
# 5.  All-in-one entry point called from main.py
# -----------------------------------------------------------------------------

def generate_all_plots(
    history_a,
    history_b_list: list,
    metrics_a: dict,
    metrics_b: dict,
    class_names: list,
    model_for_samples=None,
    test_ds=None,
    data_dir: str = config.DATA_DIR,
):
    """
    Convenience wrapper that generates every plot and saves them to results/.
    """
    print("\n[INFO] Generating visualisations ...")

    plot_training_curves(history_a, history_b_list)

    plot_confusion_matrix(
        metrics_a["y_true"], metrics_a["y_pred"],
        class_names, metrics_a["model_name"],
        "confusion_matrix_model_a.png",
    )
    plot_confusion_matrix(
        metrics_b["y_true"], metrics_b["y_pred"],
        class_names, metrics_b["model_name"],
        "confusion_matrix_model_b.png",
    )

    plot_comparison_bars(metrics_a, metrics_b)

    if model_for_samples is not None and test_ds is not None:
        plot_sample_predictions(model_for_samples, test_ds, class_names)

    if data_dir and os.path.isdir(data_dir):
        plot_dataset_samples(data_dir, class_names)

    print("[INFO] All plots saved to results/")
