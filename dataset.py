"""
dataset.py
Dataset and DataLoader implementation for Multiple Instance Learning (MIL)
with Whole Slide Images (WSI).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from skimage import filters, morphology
from PIL import Image
import pandas as pd


class WSIBagDataset(Dataset):
    """
    Dataset for loading Whole Slide Images (WSI) as bags of patches.
    Each WSI is segmented into tissue regions and divided into patches.
    """
    
    def __init__(self, csv_file, patch_size=128, tissue_threshold=0.5, transform=None):
        """
        Args:
            csv_file (str): Path to CSV with columns [wsi_id, wsi_file_path, wsi_label]
            patch_size (int): Size of patches to extract from WSI
            tissue_threshold (float): Minimum tissue coverage for patch inclusion (0-1)
            transform (callable, optional): Optional transform to apply to each patch
        """
        self.df = pd.read_csv(csv_file)
        self.patch_size = patch_size
        self.tissue_threshold = tissue_threshold
        self.transform = transform
        
        print(f"Loaded {len(self.df)} WSIs from {csv_file}")

    def __len__(self):
        return len(self.df)
    
    def _tissue_segmentation(self, wsi):
        """
        Segment tissue regions from WSI using Otsu thresholding on saturation channel.
        
        Args:
            wsi: WSI image as numpy array (H, W, 3)
            
        Returns:
            mask: Binary mask of tissue regions
        """
        # Convert to HSV and use saturation channel
        hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)
        S = hsv[:, :, 1]
        
        # Otsu thresholding
        thresh_val = filters.threshold_otsu(S)
        mask = S > thresh_val
        
        # Morphological operations to clean mask
        mask = morphology.remove_small_holes(mask, area_threshold=2000)
        mask = morphology.remove_small_objects(mask, min_size=2000)
        
        return mask
    
    def _extract_patches(self, wsi, mask):
        """
        Extract patches from tissue regions of the WSI.
        
        Args:
            wsi: WSI image as numpy array
            mask: Binary tissue mask
            
        Returns:
            patches: List of tissue patches as tensors
        """
        H, W, _ = wsi.shape
        patches = []
        
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                patch_mask = mask[i:i+self.patch_size, j:j+self.patch_size]
                
                # Check tissue coverage
                coverage = patch_mask.mean()
                if coverage > self.tissue_threshold:
                    patch = wsi[i:i+self.patch_size, j:j+self.patch_size]
                    
                    if self.transform:
                        patch = self.transform(patch)
                    else:
                        # Convert to float tensor [C, H, W]
                        patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                    
                    patches.append(patch)
        
        return patches

    def __getitem__(self, idx):
        """
        Get a bag of patches for a single WSI.
        
        Returns:
            dict with keys:
                - patches: Tensor of shape [num_patches, 3, patch_size, patch_size]
                - label: Bag-level label (float)
                - wsi_id: WSI identifier (string)
        """
        row = self.df.iloc[idx]
        wsi_path = row["wsi_file_path"]
        label = row["wsi_label"]
        wsi_id = row["wsi_id"]

        # Load WSI
        wsi = Image.open(wsi_path)
        wsi = np.array(wsi).astype(np.uint8)

        # Tissue segmentation
        mask = self._tissue_segmentation(wsi)

        # Crop to tissue bounding box
        coords = np.column_stack(np.where(mask))
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        wsi_cropped = wsi[y0:y1+1, x0:x1+1]
        mask_cropped = mask[y0:y1+1, x0:x1+1]

        # Extract patches
        patches = self._extract_patches(wsi_cropped, mask_cropped)
        
        if len(patches) == 0:
            raise ValueError(f"No tissue patches found for WSI: {wsi_id}")
        
        patches = torch.stack(patches)
        label = torch.tensor(label).float()

        return {
            "patches": patches,
            "label": label,
            "wsi_id": wsi_id
        }


def create_mil_dataloaders(train_csv, val_csv, 
                          patch_size=128, 
                          tissue_threshold=0.5,
                          batch_size=1,
                          num_workers=0,
                          transform=None):
    """
    Create train and validation dataloaders for MIL.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        patch_size: Size of patches
        tissue_threshold: Minimum tissue coverage
        batch_size: Batch size (typically 1 for MIL)
        num_workers: Number of dataloader workers
        transform: Optional transforms
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = WSIBagDataset(
        csv_file=train_csv,
        patch_size=patch_size,
        tissue_threshold=tissue_threshold,
        transform=transform
    )
    
    val_dataset = WSIBagDataset(
        csv_file=val_csv,
        patch_size=patch_size,
        tissue_threshold=tissue_threshold,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("Testing WSIBagDataset...")
    
    # Create dataloaders (requires CSV files to exist)
    try:
        train_loader, val_loader = create_mil_dataloaders(
            train_csv="wsi_train_labels.csv",
            val_csv="wsi_val_labels.csv",
            patch_size=128,
            tissue_threshold=0.5
        )
        
        # Test loading one batch
        batch = next(iter(train_loader))
        print(f"\nBatch loaded successfully!")
        print(f"  Patches shape: {batch['patches'].shape}")
        print(f"  Label: {batch['label'].item()}")
        print(f"  WSI ID: {batch['wsi_id'][0]}")
        
    except FileNotFoundError:
        print("\nCSV files not found. Please create them using datautils.py first.")