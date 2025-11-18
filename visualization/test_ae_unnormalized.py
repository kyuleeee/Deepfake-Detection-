"""
Test if the autoencoder works better with unnormalized inputs
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.genconvit import GenConViT
from models.config import load_config


def main():
    # Load config and model
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model on {device}...")
    model = GenConViT(config, 'genconvit_ed_inference', 'genconvit_vae_inference', 'ed', False)
    model = model.to(device)
    model.eval()
    print("Model loaded!\n")

    # Get sample image
    data_dir = 'data'
    image_path = os.path.join(data_dir, 'sample_image_1.png')

    # Load original
    original_img = Image.open(image_path).convert('RGB')

    # Test 1: Normalized input (what we've been doing)
    print("Test 1: Using NORMALIZED input (ImageNet stats)")
    transform_normalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_norm = transform_normalized(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get just encoder output
        enc_output = model.model_ed.encoder(img_norm)
        print(f"  Encoder output shape: {enc_output.shape}")
        print(f"  Encoder output range: [{enc_output.min():.3f}, {enc_output.max():.3f}]")

        # Get full AE reconstruction
        recon_norm = model.extract_AE(img_norm)
        print(f"  AE output range: [{recon_norm.min():.3f}, {recon_norm.max():.3f}]")
        print(f"  Unique values: {torch.unique(recon_norm).shape[0]}\n")

    # Test 2: Unnormalized input (just resize and convert to tensor)
    print("Test 2: Using UNNORMALIZED input (raw [0,1] range)")
    transform_unnormalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_unnorm = transform_unnormalized(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get just encoder output
        enc_output = model.model_ed.encoder(img_unnorm)
        print(f"  Encoder output shape: {enc_output.shape}")
        print(f"  Encoder output range: [{enc_output.min():.3f}, {enc_output.max():.3f}]")

        # Get full AE reconstruction
        recon_unnorm = model.extract_AE(img_unnorm)
        print(f"  AE output range: [{recon_unnorm.min():.3f}, {recon_unnorm.max():.3f}]")
        print(f"  Unique values: {torch.unique(recon_unnorm).shape[0]}\n")

    # Visualize both
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    recon_norm_vis = recon_norm.squeeze(0).cpu().permute(1, 2, 0).numpy()
    axes[1].imshow(recon_norm_vis, vmin=0, vmax=1)
    axes[1].set_title(f'AE Output (Normalized Input)\nRange: [{recon_norm.min():.3f}, {recon_norm.max():.3f}]',
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')

    recon_unnorm_vis = recon_unnorm.squeeze(0).cpu().permute(1, 2, 0).numpy()
    axes[2].imshow(recon_unnorm_vis, vmin=0, vmax=1)
    axes[2].set_title(f'AE Output (Unnormalized Input)\nRange: [{recon_unnorm.min():.3f}, {recon_unnorm.max():.3f}]',
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    output_dir = 'ae_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'normalized_vs_unnormalized.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
