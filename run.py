#run.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
from model import CelebAMaskLiteUNet

def run_inference(image_path, model, device):
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return predicted_mask

def main(input_path, output_path, weights):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = CelebAMaskLiteUNet(base_channels=30, num_classes=19).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    
    # Check if input is a directory
    os.makedirs(args.output, exist_ok=True)  # Ensure output folder exists
        
    for filename in os.listdir(args.input):
        input_path = os.path.join(args.input, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"  # Ensure PNG output
        output_path = os.path.join(args.output, output_filename)
                
                # Generate mask
        mask = run_inference(input_path, model, device)
                
                # Save mask as PNG
        mask_img = Image.fromarray(mask.astype('uint8'))
        mask_img.save(output_path, format="PNG")
        print(f"Processed {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)

