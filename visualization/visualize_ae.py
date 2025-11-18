import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from models.genconvit import GenConViT
from models.config import load_config


def load_image(image_path, img_size=224):
    """Load and preprocess a single image"""
    img = Image.open(image_path).convert('RGB')

    # Define the same preprocessing as used in training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_reconstruction(original_img, reconstructed_tensor, save_path=None):
    """Visualize original and reconstructed images side by side"""
    # Denormalize the reconstructed image
    reconstructed = denormalize(reconstructed_tensor.squeeze(0).cpu())
    reconstructed_np = reconstructed.permute(1, 2, 0).numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Reconstructed image
    axes[1].imshow(reconstructed_np)
    axes[1].set_title('Reconstructed Image (After AE)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_multiple_images(image_paths, model, device, save_dir='ae_visualizations'):
    """Visualize multiple images in a grid"""
    os.makedirs(save_dir, exist_ok=True)

    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(image_paths):
        # Load image
        img_tensor, original_img = load_image(img_path)
        img_tensor = img_tensor.to(device)

        # Get reconstruction from autoencoder
        with torch.no_grad():
            reconstructed = model.extract_AE(img_tensor)

        # Denormalize for visualization
        reconstructed_vis = denormalize(reconstructed.squeeze(0).cpu())
        reconstructed_np = reconstructed_vis.permute(1, 2, 0).numpy()

        # Plot original
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original: {os.path.basename(img_path)}',
                              fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        # Plot reconstructed
        axes[idx, 1].imshow(reconstructed_np)
        axes[idx, 1].set_title(f'Reconstructed', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

    plt.tight_layout()

    # Save the grid
    save_path = os.path.join(save_dir, 'ae_reconstruction_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid visualization to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize AutoEncoder reconstruction')
    parser.add_argument('--images', type=str, nargs='+',
                       help='Paths to image files (space-separated)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing sample images')
    parser.add_argument('--net', type=str, default='ed',
                       choices=['ed', 'vae', 'both'],
                       help='Network type: ed, vae, or both')
    parser.add_argument('--ed_weight', type=str, default='genconvit_ed_inference',
                       help='ED model weight name')
    parser.add_argument('--vae_weight', type=str, default='genconvit_vae_inference',
                       help='VAE model weight name')
    parser.add_argument('--fp16', action='store_true',
                       help='Use half precision (FP16)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample images to visualize (if using data_dir)')
    parser.add_argument('--output_dir', type=str, default='ae_visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {args.net} model...")
    model = GenConViT(config, args.ed_weight, args.vae_weight, args.net, args.fp16)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Get image paths
    if args.images:
        image_paths = args.images
    else:
        # Get sample images from data directory
        valid_extensions = ('.png', '.jpg', '.jpeg')
        all_images = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.lower().endswith(valid_extensions)
        ]
        image_paths = sorted(all_images)[:args.num_samples]

    if not image_paths:
        print(f"No images found! Please provide images via --images or check {args.data_dir}")
        return

    print(f"\nProcessing {len(image_paths)} images...")

    # Visualize images
    if len(image_paths) == 1:
        # Single image visualization
        img_tensor, original_img = load_image(image_paths[0])
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            reconstructed = model.extract_AE(img_tensor)

        save_path = os.path.join(args.output_dir, 'ae_reconstruction_single.png')
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_reconstruction(original_img, reconstructed, save_path)
    else:
        # Multiple images grid
        visualize_multiple_images(image_paths, model, device, args.output_dir)

    print(f"\nVisualization complete! Check the '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()
