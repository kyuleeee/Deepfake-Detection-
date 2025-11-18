"""
Quick test to check if VAE produces better reconstructions than ED
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.genconvit import GenConViT
from models.config import load_config


def test_reconstruction(net_type='vae'):
    """Test reconstruction quality for a given network type"""
    # Load config and model
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nTesting {net_type.upper()} reconstruction...")
    print(f"Device: {device}")

    try:
        model = GenConViT(config, 'genconvit_ed_inference', 'genconvit_vae_inference', net_type, False)
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Get sample image
    data_dir = 'data'
    image_path = os.path.join(data_dir, 'sample_image_1.png')

    # Load and preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # Get reconstruction
    with torch.no_grad():
        try:
            reconstructed = model.extract_AE(img_tensor)

            # Statistics
            print(f"Input shape:  {img_tensor.shape}")
            print(f"Output shape: {reconstructed.shape}")
            print(f"Output range: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
            print(f"Output mean:  {reconstructed.mean():.6f}")
            print(f"Output std:   {reconstructed.std():.6f}")
            print(f"Unique vals:  {torch.unique(reconstructed).shape[0]:,}")

            # Check if it's actually reconstructing
            if reconstructed.std() < 0.01:
                print("⚠️  WARNING: Very low std - likely producing near-constant output")
            else:
                print("✅ Good variation in output - likely real reconstruction")

            return {
                'net_type': net_type,
                'original': original_img,
                'reconstructed': reconstructed.cpu(),
                'stats': {
                    'min': reconstructed.min().item(),
                    'max': reconstructed.max().item(),
                    'mean': reconstructed.mean().item(),
                    'std': reconstructed.std().item(),
                }
            }

        except Exception as e:
            print(f"❌ Reconstruction failed: {e}")
            return None


def main():
    print("="*70)
    print("COMPARING ED VS VAE RECONSTRUCTION QUALITY")
    print("="*70)

    # Test ED
    ed_result = test_reconstruction('ed')

    # Test VAE
    vae_result = test_reconstruction('vae')

    # Visualize comparison
    if ed_result and vae_result:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Denormalize for visualization
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
            return torch.clamp(tensor, 0, 1)

        # ED row
        axes[0, 0].imshow(ed_result['original'])
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')

        ed_recon = ed_result['reconstructed'].squeeze(0)
        ed_recon_vis = ed_recon.permute(1, 2, 0).numpy()
        axes[0, 1].imshow(ed_recon_vis, vmin=0, vmax=1)
        axes[0, 1].set_title(f'ED Reconstruction (Raw)\nStd: {ed_result["stats"]["std"]:.6f}', fontweight='bold')
        axes[0, 1].axis('off')

        ed_recon_norm = (ed_recon_vis - ed_recon_vis.min()) / (ed_recon_vis.max() - ed_recon_vis.min() + 1e-8)
        axes[0, 2].imshow(ed_recon_norm)
        axes[0, 2].set_title('ED Reconstruction (Scaled)', fontweight='bold')
        axes[0, 2].axis('off')

        # VAE row
        axes[1, 0].imshow(vae_result['original'])
        axes[1, 0].set_title('Original Image', fontweight='bold')
        axes[1, 0].axis('off')

        vae_recon = vae_result['reconstructed'].squeeze(0)
        vae_recon_vis = vae_recon.permute(1, 2, 0).numpy()
        axes[1, 1].imshow(vae_recon_vis, vmin=0, vmax=1)
        axes[1, 1].set_title(f'VAE Reconstruction (Raw)\nStd: {vae_result["stats"]["std"]:.6f}', fontweight='bold')
        axes[1, 1].axis('off')

        vae_recon_norm = (vae_recon_vis - vae_recon_vis.min()) / (vae_recon_vis.max() - vae_recon_vis.min() + 1e-8)
        axes[1, 2].imshow(vae_recon_norm)
        axes[1, 2].set_title('VAE Reconstruction (Scaled)', fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()

        output_dir = 'ae_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'ed_vs_vae_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n{'='*70}")
        print(f"Comparison saved to: {output_path}")
        print(f"{'='*70}")

        plt.show()

    elif ed_result:
        print("\n⚠️  VAE test failed, only showing ED results")
    elif vae_result:
        print("\n⚠️  ED test failed, only showing VAE results")
    else:
        print("\n❌ Both tests failed")


if __name__ == "__main__":
    main()
