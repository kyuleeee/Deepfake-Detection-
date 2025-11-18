"""
Comprehensive analysis of weights for each architecture in the checkpoint
"""

import torch
import numpy as np
from collections import defaultdict


def analyze_tensor(tensor, name):
    """Analyze a single tensor and return statistics"""
    if not hasattr(tensor, 'numel'):
        return None

    # Skip non-parameter tensors (like indices, masks)
    if tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.bool]:
        return {
            'name': name,
            'shape': tuple(tensor.shape),
            'num_params': tensor.numel(),
            'dtype': str(tensor.dtype),
            'is_index_or_mask': True,
        }

    flat = tensor.flatten()

    # Convert to float if needed
    if not tensor.is_floating_point():
        flat = flat.float()

    return {
        'name': name,
        'shape': tuple(tensor.shape),
        'num_params': tensor.numel(),
        'mean': flat.mean().item(),
        'std': flat.std().item(),
        'min': flat.min().item(),
        'max': flat.max().item(),
        'abs_mean': flat.abs().mean().item(),
        'all_zeros': torch.all(tensor == 0).item(),
        'near_zero_ratio': (flat.abs() < 1e-6).sum().item() / tensor.numel(),
    }


def group_by_architecture(state_dict):
    """Group parameters by architecture component"""
    #일단 빈 딕셔너리 만들기 
    groups = defaultdict(list)

    #state_dict의 구조가 key-value 쌍이므로, key들을 하나씩 살펴보기    
    #state_dict의 key들은 모델의 각 층(layer)이나 파라미터(parameter)를 나타냄
    #ex) encoder.layer1.weight : 30, encoder.layer1.bias : 30, decoder.layer1.weight : 30 ...
    for key in state_dict.keys():
        if 'encoder.' in key:
            groups['encoder'].append(key)
        elif 'decoder.' in key:
            groups['decoder'].append(key)
        elif 'backbone.patch_embed.backbone' in key:
            groups['embedder_in_backbone'].append(key)
        elif 'backbone.' in key:
            groups['backbone'].append(key)
        elif 'embedder.' in key:
            groups['embedder'].append(key)
        elif 'fc' in key.lower():
            groups['fc_layers'].append(key)
        else:
            groups['other'].append(key)

    return groups


def print_architecture_summary(state_dict, group_name, keys):
    """Print summary statistics for an architecture component"""
    print(f"\n{'='*80}")
    print(f"  {group_name.upper()}")
    print(f"{'='*80}")
    print(f"Number of parameter tensors: {len(keys)}")

    if len(keys) == 0:
        print("  (No parameters found)")
        return

    # Collect statistics
    total_params = 0
    all_stats = []
    index_tensors = []

    
    
    for key in keys:
        tensor = state_dict[key]
        stats = analyze_tensor(tensor, key)
        if stats:
            if stats.get('is_index_or_mask', False):
                index_tensors.append(stats)
            else:
                all_stats.append(stats)
            total_params += stats['num_params']

    # Print summary
    print(f"Total parameters: {total_params:,}")

    # Check if entire component is zeros
    all_zero_count = sum(1 for s in all_stats if s['all_zeros'])
    if all_zero_count == len(all_stats):
        print(f"\n⚠️  WARNING: ALL PARAMETERS ARE ZEROS! (Untrained)")
    elif all_zero_count > 0:
        print(f"\n⚠️  WARNING: {all_zero_count}/{len(all_stats)} tensors are all zeros")

    # Overall statistics
    if all_stats:
        all_means = [s['abs_mean'] for s in all_stats]
        all_stds = [s['std'] for s in all_stats]

        print(f"\nOverall statistics:")
        print(f"  Average absolute mean: {np.mean(all_means):.6f}")
        print(f"  Average std dev:       {np.mean(all_stds):.6f}")
        print(f"  Min value (global):    {min(s['min'] for s in all_stats):.6f}")
        print(f"  Max value (global):    {max(s['max'] for s in all_stats):.6f}")
        print(f"  zero tensors:         {all_zero_count}/{len(all_stats)}")

    # Show first few layers
    print(f"\nFirst {min(5, len(all_stats))} layers:")
    print(f"{'Name':<60} {'Shape':<20} {'Mean(abs)':<12} {'Std':<12} {'Zeros?'}")
    print("-" * 120)

    for stats in all_stats[:5]:
        shape_str = str(stats['shape'])
        zeros_marker = "⚠️ YES" if stats['all_zeros'] else "No"
        print(f"{stats['name']:<60} {shape_str:<20} {stats['abs_mean']:<12.6f} {stats['std']:<12.6f} {zeros_marker}")

    if len(all_stats) > 5:
        print(f"... ({len(all_stats) - 5} more layers)")

    # Note about index tensors
    if index_tensors:
        print(f"\nNote: {len(index_tensors)} index/mask tensors (e.g., position indices) not shown in statistics")


def main():
    # 체크포인트 경로 확인해보기! 
    ed_checkpoint_path = 'model/GenConViT/genconvit_ed_inference.pth'
    vae_checkpoint_path = 'model/GenConViT/genconvit_vae_inference.pth'

    print("="*80)
    print("GENCONVIT CHECKPOINT WEIGHT ANALYSIS")
    print("="*80)

    # Analyze ED checkpoint
    print(f"\n\n{'#'*80}")
    print(f"# ANALYZING: {ed_checkpoint_path}")
    print(f"{'#'*80}")

    try:
        #체크포인트 로드하기 
        checkpoint_ed = torch.load(ed_checkpoint_path, map_location=torch.device('cpu'))

        
        if isinstance(checkpoint_ed, dict) and 'state_dict' in checkpoint_ed:
            #체크포인트의 state_dict 추출하기
            # state_dict를 추출하는 이유는, 모델의 가중치들이 state_dict에 저장되어 있기 때문! 
            state_dict_ed = checkpoint_ed['state_dict']
            print(f"\nCheckpoint metadata keys: {list(checkpoint_ed.keys())}")
            if 'epoch' in checkpoint_ed:
                print(f"Training epoch: {checkpoint_ed['epoch']}")
            if 'best_acc' in checkpoint_ed:
                print(f"Best accuracy: {checkpoint_ed['best_acc']}")
        else:
            #체크포인트가 바로 state_dict인 경우
            state_dict_ed = checkpoint_ed

        print(f"\nTotal keys in state_dict: {len(state_dict_ed)}")

        # state_dict의 key들을 그룹화하기 (ex. encoder, decoder 등)
        groups_ed = group_by_architecture(state_dict_ed)

        print(f"\nArchitecture components found:")
        #groups_ed 의 예시 - { encoder :encoder.layer1.weight, encoder.layer1.bias ... } 
        for group_name, keys in groups_ed.items():
            print(f"  - {group_name}: {len(keys)} parameters")

        #groups_ed 의 예시 - { 'encoder': [encoder.layer1.weight, encoder.layer1.bias], 'decoder': [decoder.layer1.weight, decoder.layer1.bias] ... }  
        #groups_ed['encoder'] = [encoder.layer1.weight, encoder.layer1.bias ... ]
        for group_name in ['encoder', 'decoder', 'backbone', 'embedder_in_backbone', 'embedder', 'fc_layers', 'other']:
            if group_name in groups_ed:
                print_architecture_summary(state_dict_ed, group_name, groups_ed[group_name])

    except FileNotFoundError:
        print(f"❌ ED checkpoint not found: {ed_checkpoint_path}")

    # Analyze VAE checkpoint
    print(f"\n\n{'#'*80}")
    print(f"# ANALYZING: {vae_checkpoint_path}")
    print(f"{'#'*80}")

    try:
        checkpoint_vae = torch.load(vae_checkpoint_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint_vae, dict) and 'state_dict' in checkpoint_vae:
            state_dict_vae = checkpoint_vae['state_dict']
            print(f"\nCheckpoint metadata keys: {list(checkpoint_vae.keys())}")
            if 'epoch' in checkpoint_vae:
                print(f"Training epoch: {checkpoint_vae['epoch']}")
            if 'best_acc' in checkpoint_vae:
                print(f"Best accuracy: {checkpoint_vae['best_acc']}")
        else:
            state_dict_vae = checkpoint_vae

        print(f"\nTotal keys in state_dict: {len(state_dict_vae)}")

        # Group by architecture
        groups_vae = group_by_architecture(state_dict_vae)

        print(f"\nArchitecture components found:")
        for group_name, keys in groups_vae.items():
            print(f"  - {group_name}: {len(keys)} parameters")

        # Analyze each architecture
        for group_name in ['encoder', 'decoder', 'backbone', 'embedder_in_backbone', 'embedder', 'fc_layers', 'other']:
            if group_name in groups_vae:
                print_architecture_summary(state_dict_vae, group_name, groups_vae[group_name])

    except FileNotFoundError:
        print(f"❌ VAE checkpoint not found: {vae_checkpoint_path}")

    print(f"\n\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
