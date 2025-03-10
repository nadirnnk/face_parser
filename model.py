

import torch
import torch.nn as nn

class CelebAMaskLiteUNet(nn.Module):
    """Lightweight U-Net for CelebAMask-HQ segmentation (~1.7M params)."""
    def __init__(self, in_channels=3, num_classes=19, base_channels=30):
        super(CelebAMaskLiteUNet, self).__init__()
        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(                   # Level 1 encoder
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True)
        )  # Output: base_channels (e.g. 30) filters, same HxW
        self.enc2 = nn.Sequential(                   # Level 2 encoder
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True)
        )  # Output: 2*base (e.g. 60) filters, H/2 x W/2 (after pooling)
        self.enc3 = nn.Sequential(                   # Level 3 encoder
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True)
        )  # Output: 4*base (e.g. 120) filters, H/4 x W/4 (after pooling)
        self.enc4 = nn.Sequential(                   # Level 4 encoder (bottleneck)
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True)
        )  # Output: 8*base (e.g. 240) filters, H/8 x W/8 (after pooling)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pooling layer to downsample

        # Decoder (Upsampling path)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        # After upsampling, we'll concatenate with enc3 output (skip connection), so input to dec3 convs = base*4 + base*4 channels
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True)
        )  # Output: 4*base (e.g. 120) filters, H/4 x W/4
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True)
        )  # Output: 2*base (e.g. 60) filters, H/2 x W/2
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True)
        )  # Output: base (e.g. 30) filters, H x W
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        # Output: num_classes segmentation map, H x W

    def forward(self, x):
        # Encoder: downsample and save feature maps for skip connections
        e1 = self.enc1(x)             # Shape: (N, base, H, W)
        e2 = self.enc2(self.pool(e1)) # Shape: (N, base*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2)) # Shape: (N, base*4, H/4, W/4)
        e4 = self.enc4(self.pool(e3)) # Shape: (N, base*8, H/8, W/8) - bottleneck features

        # Decoder: upsample and concatenate with encoder features
        d3 = self.up3(e4)                       # Upsample bottleneck to H/4 x W/4, shape: (N, base*4, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)         # Concatenate with encoder level 3 (skip connection), shape: (N, base*8, H/4, W/4)
        d3 = self.dec3(d3)                      # Decode: two convs, output shape: (N, base*4, H/4, W/4)
        d2 = self.up2(d3)                       # Upsample to H/2 x W/2, shape: (N, base*2, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)         # Skip connection from encoder level 2, shape: (N, base*4, H/2, W/2)
        d2 = self.dec2(d2)                      # Output: (N, base*2, H/2, W/2)
        d1 = self.up1(d2)                       # Upsample to H x W, shape: (N, base, H, W)
        d1 = torch.cat([d1, e1], dim=1)         # Skip connection from encoder level 1, shape: (N, base*2, H, W)
        d1 = self.dec1(d1)                      # Output: (N, base, H, W)
        out = self.final_conv(d1)               # Final 1x1 conv to get class scores, shape: (N, num_classes, H, W)
        return out

# Instantiate model and print total parameter count for verification
model = CelebAMaskLiteUNet(base_channels=30, num_classes=19)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")  # Expect ~1.7 million


