# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os

from model import CelebAMaskLiteUNet
from dataset import CelebAMaskDataset

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths (update with your actual paths)
    image_dir = "/home/msai/nadir001/advc_data/train/images"
    mask_dir = "/home/msai/nadir001/advc_data/train/masks"

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225])
    ])

    # Create Dataset and DataLoader
    full_dataset = CelebAMaskDataset(image_dir, mask_dir, transform)

    # Spliting train and validation set
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Setting dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = CelebAMaskLiteUNet(base_channels=30, num_classes=19).to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Learning Rate Scheduler (Reduce LR when validation loss stops improving)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)


    num_epochs = 50  # Adjust as needed

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Reduce LR on Plateau
        scheduler.step(avg_val_loss)

    # Save the trained model
    checkpoint_path = "ckpt.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Save the trained model
    checkpoint_path = "ckpt.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
