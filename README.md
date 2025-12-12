# MIL Path — Multiple Instance Learning for Whole Slide Images (WSI)

A small, focused repository demonstrating a Multiple Instance Learning (MIL) pipeline for Whole Slide Images (WSI). This project contains utilities to create CSV manifests for WSIs, a Dataset/DataLoader that converts WSIs into bags of tissue patches, simple MIL model implementations (Attention MIL and Additive MIL), and a solver module for training and evaluation.

This repository is intended as a lightweight starting point for MIL experiments on WSI data. It is educational and meant to be adapted to production-grade pipelines (WSI tiling libraries, multi-resolution inputs, efficient IO, etc.).

Contents
- datautils.py — utilities to create CSV manifests (wsi_id, wsi_file_path, wsi_label).
- dataset.py — WSIBagDataset that segments tissue, extracts patches, and builds bag tensors; dataloader factory.
- model.py — Two MIL model families:
  - AttentionMIL (embedding-level gated attention)
  - AdditiveMIL (instance-level additive pooling)
- solver.py — MILSolver: training loop, validation, checkpointing, and convenience train_mil_model.
- main.py — example script that strings together CSV creation, dataloaders, model creation, and training.
- wsi_train_labels.csv, wsi_val_labels.csv — small example CSVs (edit with your real paths).

Quickstart

1. Create CSV manifests
- Prepare a list of WSIs with label(s) in the format:
  - columns: wsi_id, wsi_file_path, wsi_label
- Use datautils.create_wsi_csv or create_train_val_csvs to write CSVs.

Example (interactive):
```python
from datautils import create_train_val_csvs

train_data = [
    ("WSI_train_001", "/path/to/train/wsi_001.png", 0),
    ("WSI_train_002", "/path/to/train/wsi_002.png", 1),
]
val_data = [
    ("WSI_val_001", "/path/to/val/wsi_001.png", 1),
]
create_train_val_csvs(train_data, val_data,
                      train_csv_path="wsi_train_labels.csv", 
                      val_csv_path="wsi_val_labels.csv")
```

2. Create dataloaders
- The WSIBagDataset loads each WSI from the CSV, performs a simple tissue segmentation on the saturation channel (Otsu), crops to the tissue bounding box, and extracts non-overlapping patches with a tissue coverage threshold.

Example:
```python
from dataset import create_mil_dataloaders

train_loader, val_loader = create_mil_dataloaders(
    train_csv="wsi_train_labels.csv",
    val_csv="wsi_val_labels.csv",
    patch_size=128,
    tissue_threshold=0.5,
    batch_size=1,
    num_workers=0
)
```
Notes:
- batch_size is typically 1 for MIL (one bag per batch).
- transform can be passed to apply torchvision or custom preprocessing per patch.

3. Create a model
- Use the factory function to choose Attention or Additive MIL:

```python
from model import create_mil_model

model = create_mil_model(model_type='attention', num_classes=1)
# or
model = create_mil_model(model_type='additive', num_classes=1)
```

4. Train
- Use the convenience train_mil_model which wraps the MILSolver.

```python
from solver import train_mil_model

trained_model, history = train_mil_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    learning_rate=1e-3,
    device='cuda'
)
```

Key design decisions and behavior
- WSIBagDataset
  - Loads whole-slide image files with PIL (expects an image file readable by PIL).
  - Tissue segmentation is performed on the saturation channel using Otsu thresholding.
  - Small objects/holes are removed using skimage.morphology with default thresholds; you may need to tune area_threshold/min_size for your data.
  - Non-overlapping patch tiling is used; patches are included if patch_mask.mean() > tissue_threshold.
  - Patches are returned as float tensors in [C, H, W] normalized to [0, 1] unless a custom transform is provided.

- model.py
  - Uses torchvision's ResNet-18 pretrained backbone as feature extractor.
  - AttentionMIL implements gated attention pooling (embedding-level).
  - AdditiveMIL computes per-instance logits then sums them for bag logits (instance-level).
  - Both models assume single-output binary classification by default (num_classes=1). Adjust num_classes for multi-class tasks.

- solver.py
  - MILSolver uses BCEWithLogitsLoss for binary tasks. For multi-class classification, update criterion and label encoding appropriately.
  - Checkpointing saves model and optimizer state in `checkpoints/`.
  - The predict method returns a single scalar probability and binary prediction. For multi-class, adapt accordingly.

Dependencies
- Python 3.8+
- PyTorch (tested with 1.12+)
- torchvision
- numpy
- pandas
- Pillow (PIL)
- scikit-image (skimage)
- OpenCV (cv2)
- tqdm

Install via pip (example):
```bash
pip install torch torchvision numpy pandas pillow scikit-image opencv-python tqdm
```

Usage tips and pitfalls
- Input WSI formats: This repo uses PIL to open WSI image files (e.g., PNG, JPEG). For true large WSI formats (e.g., .svs, .ndpi), integrate an OME/openslide-based reader instead of PIL to read regions efficiently.
- Memory: The dataset loads an entire WSI image into memory and extracts patches — for large WSIs this may be infeasible. Consider tiling with on-disk readers or streaming patches.
- Patch ordering and sampling: Current code uses deterministic tiling. For large bags, you may wish to sample a subset of patches per bag during training or use a more sophisticated sampler.
- Multi-GPU: DataParallel/DistributedDataParallel not configured in examples — add standard PyTorch wrappers for multi-GPU training.

Example: Quick test (main.py)
- main.py contains a minimal example tying CSV creation, dataloaders, model creation, and training invocation. Edit file paths and parameters before running.

Extending this repository
- Use an openslide-based patch reader to handle large WSI formats and multi-resolution inputs.
- Add stronger augmentations and patch-level transforms.
- Add logging (tensorboard, wandb), better checkpoint management, learning rate schedulers, and metrics beyond accuracy.
- Add unit tests and CI for reproducibility.

Acknowledgements
- Concepts implemented here are inspired by standard MIL literature (e.g., Attention-based MIL) and common WSI preprocessing heuristics.
