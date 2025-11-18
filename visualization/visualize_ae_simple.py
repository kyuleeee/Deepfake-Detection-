"""
Simple script to visualize autoencoder reconstructions
Usage: python visualize_ae_simple.py
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.genconvit import GenConViT
from models.config import load_config


def main():
    # Configuration
    net_type = 'ed'  # Options: 'ed', 'vae', 'both'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'
    fp16 = False
    num_samples = 3  # Number of sample images to show

    # Load config and model
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading {net_type} model on {device}...")
    model = GenConViT(config, ed_weight, vae_weight, net_type, fp16)
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Get sample images
    data_dir = 'data'
    valid_extensions = ('.png', '.jpg', '.jpeg')
    all_images = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(valid_extensions)
    ]
    image_paths = sorted(all_images)[:num_samples]

    if not image_paths:
        print(f"No images found in {data_dir}/")
        return

    print(f"\nProcessing {len(image_paths)} images...")

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create figure
    fig, axes = plt.subplots(len(image_paths), 2, figsize=(10, 5 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)

    # Process each image
    for idx, img_path in enumerate(image_paths):
        print(f"  Processing: {os.path.basename(img_path)}")

        # Load and preprocess
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(original_img).unsqueeze(0).to(device)

        # Get reconstruction
        with torch.no_grad():
            reconstructed_tensor = model.extract_AE(img_tensor)

        # Denormalize reconstructed image
        reconstructed = reconstructed_tensor.squeeze(0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        reconstructed = reconstructed * std + mean
        reconstructed = torch.clamp(reconstructed, 0, 1)
        reconstructed_np = reconstructed.permute(1, 2, 0).numpy()

        # Plot
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original: {os.path.basename(img_path)}',
                              fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(reconstructed_np)
        axes[idx, 1].set_title('Reconstructed (AE Output)',
                              fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

    plt.tight_layout()

    # Save and show
    output_dir = 'ae_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reconstruction_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")

    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()
