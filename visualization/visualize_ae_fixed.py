"""
Fixed visualization script for autoencoder reconstruction
Handles the decoder output properly without normalization issues
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
    net_type = 'ed'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'
    fp16 = False
    num_samples = 3

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

    # Preprocessing transform WITH normalization
    transform_normalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Preprocessing transform WITHOUT normalization (for comparison)
    transform_unnormalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create figure with 4 columns: Original, Input (normalized), Reconstructed (raw), Reconstructed (scaled)
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)

    # Process each image
    for idx, img_path in enumerate(image_paths):
        print(f"  Processing: {os.path.basename(img_path)}")

        # Load original
        original_img = Image.open(img_path).convert('RGB')

        # Create normalized and unnormalized versions
        img_tensor_normalized = transform_normalized(original_img).unsqueeze(0).to(device)
        img_tensor_unnormalized = transform_unnormalized(original_img).unsqueeze(0).to(device)

        # Get reconstruction from normalized input
        with torch.no_grad():
            reconstructed_tensor = model.extract_AE(img_tensor_normalized)

        # Convert to numpy for visualization
        reconstructed_np = reconstructed_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Create a scaled version that uses the full dynamic range
        reconstructed_scaled = reconstructed_np.copy()
        reconstructed_scaled = (reconstructed_scaled - reconstructed_scaled.min()) / (reconstructed_scaled.max() - reconstructed_scaled.min() + 1e-8)

        # Denormalize the input to see what went in
        input_denorm = img_tensor_normalized.squeeze(0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_denorm = input_denorm * std + mean
        input_denorm = torch.clamp(input_denorm, 0, 1).permute(1, 2, 0).numpy()

        # Plot all versions
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original\n{os.path.basename(img_path)}', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(input_denorm)
        axes[idx, 1].set_title(f'Input\n(After Normalization)', fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(reconstructed_np, vmin=0, vmax=1)
        axes[idx, 2].set_title(f'AE Output (Raw)\nRange: [{reconstructed_np.min():.3f}, {reconstructed_np.max():.3f}]',
                              fontsize=10, fontweight='bold')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(reconstructed_scaled)
        axes[idx, 3].set_title(f'AE Output (Scaled)\nNormalized to [0, 1]', fontsize=10, fontweight='bold')
        axes[idx, 3].axis('off')

    plt.tight_layout()

    # Save
    output_dir = 'ae_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reconstruction_detailed.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nDetailed results saved to: {output_path}")

    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()
