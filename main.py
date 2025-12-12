# 1. Create CSV files (modify with your data)
from datautils import create_train_val_csvs
train_data = [("wsi_1", "example\wsi_train.png", 0)]
val_data = [("wsi_val_1", "example\wsi_val.png", 1)]
create_train_val_csvs(train_data, val_data)

# 2. Create dataloaders
from dataset import create_mil_dataloaders
train_loader, val_loader = create_mil_dataloaders(
    "wsi_train_labels.csv", "wsi_val_labels.csv"
)

# 3. Create model
from model import create_mil_model
model = create_mil_model(model_type='attention', num_classes=1)

# 4. Train
from solver import train_mil_model
trained_model, history = train_mil_model(
    model, train_loader, val_loader, num_epochs=10
)