"""
Check if the encoder/decoder weights are actually loaded
"""

import torch
from models.genconvit import GenConViT
from models.config import load_config


def main():
    config = load_config()

    print("Loading GenConViT model...")
    model = GenConViT(config, 'genconvit_ed_inference', 'genconvit_vae_inference', 'ed', False)

    print("\n=== Checking Encoder weights ===")
    encoder_params = list(model.model_ed.encoder.parameters())
    print(f"Number of encoder parameter tensors: {len(encoder_params)}")

    if len(encoder_params) > 0:
        first_param = encoder_params[0]
        print(f"First encoder layer shape: {first_param.shape}")
        print(f"First encoder layer stats:")
        print(f"  Mean: {first_param.mean().item():.6f}")
        print(f"  Std: {first_param.std().item():.6f}")
        print(f"  Min: {first_param.min().item():.6f}")
        print(f"  Max: {first_param.max().item():.6f}")
        print(f"  All zeros?: {torch.all(first_param == 0).item()}")

    print("\n=== Checking Decoder weights ===")
    decoder_params = list(model.model_ed.decoder.parameters())
    print(f"Number of decoder parameter tensors: {len(decoder_params)}")

    if len(decoder_params) > 0:
        first_param = decoder_params[0]
        print(f"First decoder layer shape: {first_param.shape}")
        print(f"First decoder layer stats:")
        print(f"  Mean: {first_param.mean().item():.6f}")
        print(f"  Std: {first_param.std().item():.6f}")
        print(f"  Min: {first_param.min().item():.6f}")
        print(f"  Max: {first_param.max().item():.6f}")
        print(f"  All zeros?: {torch.all(first_param == 0).item()}")

    print("\n=== Checking what's in the checkpoint ===")
    checkpoint_path = 'model/GenConViT/genconvit_ed_inference.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", checkpoint.keys())
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        print(f"\nNumber of keys in state_dict: {len(state_dict.keys())}")
        print("\nFirst 20 keys in checkpoint:")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")

        # Check if encoder/decoder keys exist
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k.lower()]

        print(f"\n Keys with 'encoder': {len(encoder_keys)}")
        print(f"Keys with 'decoder': {len(decoder_keys)}")

        if encoder_keys:
            print("\nEncoder keys found:")
            for key in encoder_keys[:5]:
                print(f"  {key}")

        if decoder_keys:
            print("\nDecoder keys found:")
            for key in decoder_keys[:5]:
                print(f"  {key}")


if __name__ == "__main__":
    main()
