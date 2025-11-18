"""
Corrected visualization using the proper VAE extraction method
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.genconvit import GenConViT
from models.config import load_config


def get_reconstruction(model, img_tensor, net_type):
    """Get reconstruction using the correct method for each network type"""
    with torch.no_grad():
        if net_type == 'ed':
            # ED has extract_AE method
            reconstruction = model.model_ed.extract_AE(img_tensor)
        elif net_type == 'vae':
            # VAE has extract_VAE method (NOT extract_AE!)
            reconstruction = model.model_vae.extract_VAE(img_tensor)
        else:  # both
            recon_ed = model.model_ed.extract_AE(img_tensor)
            recon_vae = model.model_vae.extract_VAE(img_tensor)
            reconstruction = (recon_ed + recon_vae) / 2

    return reconstruction


def main():
    # Configuration
    num_samples = 3

    # Load config
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("CORRECTED AE VISUALIZATION - ED vs VAE COMPARISON")
    print("="*70)

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

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create figure: 3 columns (Original, ED reconstruction, VAE reconstruction)
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(15, 5 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)

    # Load both models
    print("\nLoading ED model...")
    model_ed = GenConViT(config, 'genconvit_ed_inference', 'genconvit_vae_inference', 'ed', False)
    model_ed = model_ed.to(device)
    model_ed.eval()

    print("\nLoading VAE model...")
    model_vae = GenConViT(config, 'genconvit_ed_inference', 'genconvit_vae_inference', 'vae', False)
    model_vae = model_vae.to(device)
    model_vae.eval()

    print(f"\nProcessing {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        print(f"  {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")

        # Load original
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(original_img).unsqueeze(0).to(device)

        # Get ED reconstruction
        recon_ed = get_reconstruction(model_ed, img_tensor, 'ed')
        ed_np = recon_ed.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Get VAE reconstruction
        recon_vae = get_reconstruction(model_vae, img_tensor, 'vae')
        vae_np = recon_vae.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Print stats
        print(f"    ED  - Range: [{recon_ed.min():.4f}, {recon_ed.max():.4f}], Std: {recon_ed.std():.6f}")
        print(f"    VAE - Range: [{recon_vae.min():.4f}, {recon_vae.max():.4f}], Std: {recon_vae.std():.6f}")

        # Plot original
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original\n{os.path.basename(img_path)}',
                               fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')

        # Plot ED reconstruction
        axes[idx, 1].imshow(ed_np, vmin=0, vmax=1)
        axes[idx, 1].set_title(f'ED Reconstruction\nStd: {recon_ed.std():.6f}',
                               fontsize=11, fontweight='bold')
        axes[idx, 1].axis('off')

        # Plot VAE reconstruction
        axes[idx, 2].imshow(vae_np, vmin=0, vmax=1)
        axes[idx, 2].set_title(f'VAE Reconstruction\nStd: {recon_vae.std():.6f}',
                               fontsize=11, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.tight_layout()

    # Save
    output_dir = 'ae_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ed_vs_vae_corrected.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    plt.show()


if __name__ == "__main__":
    main()
