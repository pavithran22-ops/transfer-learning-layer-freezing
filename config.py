"""
config.py — Central configuration for the Transfer Learning project.
All hyperparameters and paths are defined here for easy modification.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "custom_dataset")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
SRC_DIR         = os.path.join(BASE_DIR, "src")

# Model save paths
MODEL_A_PATH    = os.path.join(MODELS_DIR, "model_a_frozen.keras")
MODEL_B_PATH    = os.path.join(MODELS_DIR, "model_b_finetuned.keras")

# Results paths
RESULTS_CSV     = os.path.join(RESULTS_DIR, "comparison_results.csv")

# ─── Image Settings ───────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)          # MobileNetV2 native input size
IMG_CHANNELS    = 3
INPUT_SHAPE     = IMG_SIZE + (IMG_CHANNELS,)

# ─── Dataset Split ────────────────────────────────────────────────────────────
TRAIN_SPLIT     = 0.70               # 70 % training
VAL_SPLIT       = 0.15               # 15 % validation
TEST_SPLIT      = 0.15               # 15 % test
RANDOM_SEED     = 42                 # fixed seed for reproducibility

# ─── Training — Model A (Frozen) ──────────────────────────────────────────────
BATCH_SIZE_A    = 32
EPOCHS_A        = 5
LR_A            = 1e-3               # Adam learning rate for head training

# ─── Training — Model B (Fine-Tuned) ──────────────────────────────────────────
BATCH_SIZE_B    = 32
# Phase 1 (head-only) — same as Model A
EPOCHS_B_HEAD   = 3
LR_B_HEAD       = 1e-3
# Phase 2 (fine-tuning deeper layers)
EPOCHS_B_FINE   = 5
LR_B_FINE       = 1e-5               # lower LR for fine-tuning
FINE_TUNE_AT    = 100                # unfreeze MobileNetV2 layers from this index onward

# ─── Callbacks ────────────────────────────────────────────────────────────────
PATIENCE_ES     = 5                  # early-stopping patience
PATIENCE_LR     = 3                  # ReduceLROnPlateau patience
LR_FACTOR       = 0.5                # factor to reduce LR
MIN_LR          = 1e-7               # minimum learning rate

# ─── Regularisation ───────────────────────────────────────────────────────────
DROPOUT_RATE    = 0.3

# ─── Supported image formats ──────────────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
