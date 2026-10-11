# 🚀 Transfer Learning with Layer Freezing

> **Comparing Frozen vs Fine-Tuned MobileNetV2 for Custom Image Classification**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [What is Transfer Learning?](#what-is-transfer-learning)
5. [What is Layer Freezing?](#what-is-layer-freezing)
6. [What is Fine-Tuning?](#what-is-fine-tuning)
7. [Difference Between the Two Approaches](#difference-between-the-two-approaches)
8. [Technologies Used](#technologies-used)
9. [Project Folder Structure](#project-folder-structure)
10. [Dataset Preparation](#dataset-preparation)
11. [Installation](#installation)
12. [Running the Project](#running-the-project)
13. [Making Predictions](#making-predictions)
14. [Expected Output Files](#expected-output-files)
15. [Example Comparison Table](#example-comparison-table)
16. [Configuration](#configuration)
17. [Conclusion](#conclusion)
18. [Future Enhancements](#future-enhancements)
19. [References](#references)

---

## Introduction

This project demonstrates and compares two popular **transfer learning** strategies for image classification using **MobileNetV2** pretrained on ImageNet:

| Approach | Description |
|---|---|
| **Model A — Frozen** | The entire pretrained CNN base is frozen. Only a new classification head is trained. |
| **Model B — Fine-Tuned** | Initially trained like Model A, then selected deeper layers are unfrozen and re-trained with a smaller learning rate. |

Both models are trained and evaluated on the **same dataset** using the **same train/val/test splits** so the comparison is completely fair.

---

## Problem Statement

Training a deep CNN from scratch requires:
- **Thousands of labeled images** per class
- **Days of GPU time**
- **Expertise** in architecture design

Transfer learning solves this by reusing knowledge already encoded in pretrained models. However, there is a design choice:

> *Should we keep the pretrained layers completely fixed, or should we partially adapt them to our new dataset?*

This project answers that question empirically.

---

## Objectives

- ✅ Implement **frozen transfer learning** (Model A) with MobileNetV2
- ✅ Implement **fine-tuned transfer learning** (Model B) with MobileNetV2
- ✅ Apply proper **data augmentation**, **early stopping**, and **model checkpointing**
- ✅ Evaluate both models with **accuracy, precision, recall, F1-score**
- ✅ Visualise **confusion matrices** and **training curves**
- ✅ Generate a **comparison report** (CSV + charts)
- ✅ Provide a **prediction CLI** for new images

---

## What is Transfer Learning?

**Transfer learning** is a machine learning technique where a model trained on a large dataset (like ImageNet with 1.2 million images and 1000 classes) is reused as the starting point for a different but related task.

```
┌─────────────────────────────────┐
│   ImageNet Pretrained Model     │  ← 1.2 million images, 1000 classes
│   (General feature extractor)   │
└──────────────┬──────────────────┘
               │  Transfer weights
               ▼
┌─────────────────────────────────┐
│   Your Custom Dataset           │  ← Your images, your classes
│   (Specific classification)     │
└─────────────────────────────────┘
```

**Why it works:** The early layers of a CNN learn universal features (edges, textures, shapes) that are useful across many image tasks. These don't need to be re-learned from scratch.

**Benefits:**
- 🕐 **Faster training** — start from learned features instead of random weights
- 📦 **Less data needed** — works well even with a small dataset
- 🎯 **Better accuracy** — especially when your dataset is small

---

## What is Layer Freezing?

When we **freeze** a layer, we prevent its weights from being updated during training. The layer's parameters stay fixed at their pretrained values.

```
MobileNetV2 Architecture:
─────────────────────────────────────────────────
Layer 1  (edges, basic patterns)    → ❄️ FROZEN
Layer 2  (textures, shapes)         → ❄️ FROZEN
Layer 3  (object parts)             → ❄️ FROZEN
   ...
Layer N  (high-level features)      → ❄️ FROZEN
─────────────────────────────────────────────────
Global Average Pooling              → ✅ TRAINED
Dropout                             → ✅ TRAINED
Dense (output)                      → ✅ TRAINED  ← New layer
─────────────────────────────────────────────────
```

**Why freeze?**
- Preserves the pretrained knowledge
- Avoids destroying useful features with random gradients from the new task
- Reduces training time significantly
- Prevents overfitting on small datasets

---

## What is Fine-Tuning?

**Fine-tuning** takes freezing one step further. After initially training just the classification head (like Model A), we **selectively unfreeze** some of the deeper layers and continue training with a **very small learning rate**.

```
Fine-Tuning Phase:
─────────────────────────────────────────────────
Layer 1  (edges, basic patterns)    → ❄️ Still FROZEN (kept general)
Layer 2  (textures)                 → ❄️ Still FROZEN
   ...
Layer 100 (mid-level features)      → ❄️ Still FROZEN  ← fine_tune_at = 100
─────────────────────────────────────────────────
Layer 101 (task-specific features)  → 🔥 UNFROZEN
Layer 102 (high-level patterns)     → 🔥 UNFROZEN
   ...
Layer N                             → 🔥 UNFROZEN
─────────────────────────────────────────────────
Global Average Pooling              → 🔥 UNFROZEN
Dropout                             → 🔥 UNFROZEN
Dense (output)                      → 🔥 UNFROZEN
─────────────────────────────────────────────────
Note: BatchNormalization layers → ❄️ Always FROZEN (for stability)
```

**Why fine-tune?**
- Allows deeper layers to **specialise** for your specific domain
- Can improve accuracy when the target dataset is **very different** from ImageNet
- Requires care: learning rate must be small to avoid "catastrophic forgetting"

---

## Difference Between the Two Approaches

| Feature | Model A (Frozen) | Model B (Fine-Tuned) |
|---|---|---|
| Pretrained base | Completely frozen | Partially unfrozen |
| Layers trained | Head only | Head + selected deep layers |
| Training phases | 1 | 2 |
| Learning rate | 0.001 (head) | 0.001 (head) → 0.00001 (fine-tune) |
| Training time | ⚡ Faster | 🐢 Slower |
| Risk of overfitting | Lower | Higher (needs more data) |
| Potential accuracy | Good | Potentially higher |
| Best for | Small datasets | Medium–large datasets |

> ⚠️ **Note:** Fine-tuning does **not** always produce better results. If your dataset is small or very similar to ImageNet, the frozen approach may generalise better.

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.8+ | Core programming language |
| **TensorFlow** | 2.12+ | Deep learning framework |
| **Keras** | Built-in | Model building API |
| **MobileNetV2** | ImageNet weights | Pretrained backbone |
| **NumPy** | 1.23+ | Numerical computation |
| **Matplotlib** | 3.7+ | Plotting training curves |
| **Seaborn** | 0.12+ | Confusion matrix heatmaps |
| **Scikit-learn** | 1.2+ | Metrics and evaluation |
| **Pandas** | 2.0+ | CSV export and data tables |
| **Pillow** | 9.5+ | Image I/O support |

---

## Project Folder Structure

```
transfer-learning-layer-freezing/
│
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── .gitignore                  ← Files excluded from Git
├── LICENSE                     ← MIT License
├── main.py                     ← 🚀 Entry point — run this!
├── config.py                   ← All hyperparameters and paths
│
├── data/
│   ├── README.md               ← Dataset setup instructions
│   └── custom_dataset/         ← 📂 Place YOUR images here
│       ├── class_1/
│       ├── class_2/
│       └── ...
│
├── models/
│   ├── .gitkeep
│   ├── model_a_frozen.keras    ← Saved Model A (generated)
│   └── model_b_finetuned.keras ← Saved Model B (generated)
│
├── results/
│   ├── .gitkeep
│   ├── training_curves.png            ← Accuracy & loss plots
│   ├── confusion_matrix_model_a.png   ← Confusion matrix A
│   ├── confusion_matrix_model_b.png   ← Confusion matrix B
│   ├── comparison_bar_charts.png      ← Accuracy & time bars
│   └── comparison_results.csv         ← Summary table
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← Dataset loading, splitting, augmentation
│   ├── model_builder.py        ← MobileNetV2 model construction
│   ├── train.py                ← Training loops with callbacks
│   ├── evaluate.py             ← Metrics, reports, comparison
│   └── visualize.py            ← All plots and charts
│
└── notebooks/
    └── README.md               ← Jupyter notebook guide
```

---

## Dataset Preparation

### Step 1: Create the dataset folder

```bash
mkdir -p data/custom_dataset
```

### Step 2: Organise your images by class

```
data/custom_dataset/
├── cats/
│   ├── cat001.jpg
│   ├── cat002.jpg
│   └── ...
├── dogs/
│   ├── dog001.jpg
│   └── ...
└── birds/
    ├── bird001.jpg
    └── ...
```

### Step 3: Guidelines

- **Minimum**: 2 classes, at least 10 images per class (more is better)
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`
- **Images are auto-resized** to 224×224 — no manual resizing needed
- **Naming**: folder names become your class labels

### Example Public Datasets

| Dataset | Classes | Size | Source |
|---|---|---|---|
| Cats vs Dogs | 2 | 25,000 | [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) |
| Flowers | 5 | 4,323 | [TF Datasets](https://www.tensorflow.org/datasets/catalog/tf_flowers) |
| Intel Images | 6 | 25,000 | [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- (Optional but recommended) a virtual environment

### 1. Clone the repository

```bash
git clone https://github.com/pavithran22-ops/transfer-learning-layer-freezing.git
cd transfer-learning-layer-freezing
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (optional):** For faster training, install TensorFlow with GPU support:
> ```bash
> pip install tensorflow[and-cuda]
> ```

---

## Running the Project

### Full training + evaluation pipeline

```bash
python main.py
```

This runs all 7 steps automatically:

1. ✅ Validate dataset structure
2. ✅ Load and preprocess images
3. ✅ Split into train / validation / test sets
4. ✅ Build + train **Model A** (frozen)
5. ✅ Build + train **Model B** (fine-tuned, two phases)
6. ✅ Evaluate both models on the test set
7. ✅ Generate charts, confusion matrices, and comparison CSV

---

## Making Predictions

After training, you can classify any image:

```bash
# Using Model A (frozen)
python main.py --predict path/to/your/image.jpg --model a

# Using Model B (fine-tuned) — default
python main.py --predict path/to/your/image.jpg --model b
```

**Example output:**

```
==================================================
  Prediction Result
==================================================
  Model used  : Model B (Fine-Tuned)
  Image       : test_image.jpg
  Predicted   : cats
  Confidence  : 97.34%
==================================================
```

---

## Expected Output Files

After running `python main.py`, the following files are generated:

| File | Description |
|---|---|
| `results/training_curves.png` | Accuracy & loss curves for both models |
| `results/confusion_matrix_model_a.png` | Confusion matrix for Model A |
| `results/confusion_matrix_model_b.png` | Confusion matrix for Model B |
| `results/comparison_bar_charts.png` | Bar charts for accuracy & training time |
| `results/comparison_results.csv` | Summary comparison table |
| `models/model_a_frozen.keras` | Saved Model A weights |
| `models/model_b_finetuned.keras` | Saved Model B weights |

---

## Example Comparison Table

> ⚠️ **Illustrative only.** Actual results depend on your dataset, hardware, and training settings. Do not interpret these as guaranteed values.

| Model | Test Accuracy | Precision | Recall | F1-Score | Training Time | Trainable Params |
|---|---|---|---|---|---|---|
| Model A (Frozen) | ~88% | ~87% | ~88% | ~87% | ~120s | ~1,280 |
| Model B (Fine-Tuned) | ~93% | ~93% | ~93% | ~93% | ~480s | ~500,000 |

**Key observations (illustrative):**

- Model B may achieve higher accuracy because fine-tuning allows the model to specialise deeper features for the target domain.
- Model A trains much faster since only a tiny classification head is updated.
- On small datasets (< 500 images per class), Model A may sometimes generalise **better** due to lower risk of overfitting.

---

## Configuration

Edit `config.py` to customise the experiment:

```python
# Image size (must be 224x224 for MobileNetV2)
IMG_SIZE = (224, 224)

# Dataset split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# Model A training
BATCH_SIZE_A = 32
EPOCHS_A     = 20
LR_A         = 1e-3

# Model B — Phase 1 (head only)
EPOCHS_B_HEAD = 10
LR_B_HEAD     = 1e-3

# Model B — Phase 2 (fine-tuning)
EPOCHS_B_FINE = 20
LR_B_FINE     = 1e-5        # much smaller LR
FINE_TUNE_AT  = 100         # unfreeze from layer 100 onward

# Regularisation
DROPOUT_RATE  = 0.3
```

---

## Conclusion

This project demonstrates that:

1. **Frozen transfer learning** is a quick, reliable baseline — it trains fast, requires minimal data, and delivers solid accuracy.
2. **Fine-tuned transfer learning** can potentially improve accuracy further by adapting the deeper pretrained features to the target domain, but requires more data, more compute, and careful tuning of the learning rate.
3. **Neither approach is universally superior.** The best choice depends on your dataset size, its similarity to ImageNet, and your computational budget.

The project provides a reusable, modular codebase that can be adapted to any image classification task simply by placing a new dataset in `data/custom_dataset/`.

---

## Future Enhancements

- [ ] Add support for additional pretrained backbones (EfficientNet, ResNet50, VGG16)
- [ ] Implement Grad-CAM visualisation to explain model predictions
- [ ] Add a web-based prediction interface (Flask / Streamlit)
- [ ] Support TFRecord dataset format for large-scale experiments
- [ ] Add class-imbalance handling (class weights / oversampling)
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add mixed-precision training for faster GPU utilisation
- [ ] Export models to TensorFlow Lite for mobile deployment

---

## References

1. Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
2. Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
3. Yosinski, J., et al. (2014). *How transferable are features in deep neural networks?* [NIPS 2014](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html)
4. TensorFlow Transfer Learning Guide: [tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
5. Keras Applications: [keras.io/api/applications](https://keras.io/api/applications/)
6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)

---

## 📤 Uploading to GitHub

```bash
# 1. Initialize git (already done if cloned)
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "Initial commit: Transfer Learning with Layer Freezing"

# 4. Add remote (replace with your URL)
git remote add origin https://github.com/pavithran22-ops/transfer-learning-layer-freezing.git

# 5. Push
git branch -M main
git push -u origin main
```

---

<div align="center">

**Made with ❤️ for the ML community**

*If this project helped you, please ⭐ the repository!*

</div>
