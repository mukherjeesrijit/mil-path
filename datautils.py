"""
datautils.py
Utilities for creating CSV files from WSI images for MIL training.
Users should modify this to create their own CSV files with their WSI data.
"""

import pandas as pd
import random
from pathlib import Path


def create_wsi_csv(wsi_data, output_csv_path):
    """
    Create a CSV file with WSI information.
    
    Args:
        wsi_data: List of tuples [(wsi_id, wsi_file_path, wsi_label), ...]
        output_csv_path: Path where the CSV file will be saved
    
    Example:
        wsi_data = [
            ("WSI_001", "/path/to/wsi_001.png", 0),
            ("WSI_002", "/path/to/wsi_002.png", 1),
        ]
        create_wsi_csv(wsi_data, "train_labels.csv")
    """
    df = pd.DataFrame(wsi_data, columns=["wsi_id", "wsi_file_path", "wsi_label"])
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file created: {output_csv_path}")
    print(f"Total WSIs: {len(df)}")
    return df


def create_train_val_csvs(train_wsi_data, val_wsi_data, 
                          train_csv_path="wsi_train_labels.csv",
                          val_csv_path="wsi_val_labels.csv"):
    """
    Create both training and validation CSV files.
    
    Args:
        train_wsi_data: List of tuples for training data
        val_wsi_data: List of tuples for validation data
        train_csv_path: Output path for training CSV
        val_csv_path: Output path for validation CSV
    
    Returns:
        tuple: (train_df, val_df)
    """
    train_df = create_wsi_csv(train_wsi_data, train_csv_path)
    val_df = create_wsi_csv(val_wsi_data, val_csv_path)
    return train_df, val_df


def example_csv_creation():
    """
    Example function showing how to create CSV files.
    Users should replace this with their own data.
    """
    # Example: Create dummy data
    train_data = [
        ("WSI_train_001", "/path/to/train/wsi_001.png", 0),
        ("WSI_train_002", "/path/to/train/wsi_002.png", 1),
        ("WSI_train_003", "/path/to/train/wsi_003.png", 0),
    ]
    
    val_data = [
        ("WSI_val_001", "/path/to/val/wsi_001.png", 1),
        ("WSI_val_002", "/path/to/val/wsi_002.png", 0),
    ]
    
    # Create CSV files
    train_df, val_df = create_train_val_csvs(train_data, val_data)
    
    return train_df, val_df


if __name__ == "__main__":
    print("=" * 60)
    print("WSI CSV Creation Utility")
    print("=" * 60)
    print("\nThis utility helps create CSV files for MIL training.")
    print("Modify the example_csv_creation() function with your own data.\n")
    
    # Run example
    example_csv_creation()
    
    print("\nCSV Format:")
    print("  - wsi_id: Unique identifier for each WSI")
    print("  - wsi_file_path: Full path to the WSI image file")
    print("  - wsi_label: Binary label (0 or 1) for the WSI")