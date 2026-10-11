# Dataset Directory

Place your custom image dataset here with the following structure:

```
data/
└── custom_dataset/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image1.jpg
    │   └── ...
    └── class_n/
        ├── image1.jpg
        └── ...
```

## Requirements

| Requirement | Details |
|---|---|
| **Minimum classes** | 2 (binary) or more |
| **Minimum images per class** | At least 10 recommended |
| **Supported formats** | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff` |
| **Images are auto-resized** | 224 × 224 px (MobileNetV2 input size) |

## Dataset Split

The project automatically splits images into:

| Split | Default Ratio |
|---|---|
| Training set | 70% |
| Validation set | 15% |
| Test set | 15% |

Splitting is **stratified** — every split keeps the same class proportions —
and uses a fixed random seed for reproducibility.

## Why is the dataset not included?

To keep the repository lightweight and avoid violating dataset licenses, the
actual images are excluded from Git via `.gitignore`. Each user supplies their
own dataset.

## Example Public Datasets

You can download one of these free datasets and place it inside
`data/custom_dataset/`:

| Dataset | Classes | Source |
|---|---|---|
| Cats vs Dogs | 2 | [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) |
| Flowers 102 | 102 | [Kaggle](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset) |
| Intel Image Classification | 6 | [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) |
| Food-101 | 101 | [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101) |

## Notes

- The `data/custom_dataset/` folder is listed in `.gitignore` and will **not** be pushed to GitHub.
- Do **not** mix class images in the root `custom_dataset/` folder — each class must have its own subfolder.
