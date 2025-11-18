import torch
import os
from collections import OrderedDict


# Helper to load local weights robustly (no network)
def load_local_weights(model: torch.nn.Module, ckpt_path: str, allow_partial: bool = False) -> None:
    """
    Load weights from a local checkpoint file without any network access.
    Supports plain state_dict, or dicts with keys like 'state_dict' or 'model'.
    If allow_partial is False, loads with strict=True.
    If allow_partial is True, loads only matching keys with matching shapes, strict=False.
    """
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict):
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        elif 'model' in state and isinstance(state['model'], dict):
            state = state['model']
    # strip optional DistributedDataParallel 'module.' prefix
    cleaned = OrderedDict()
    for k, v in state.items():
        if k.startswith('module.'):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    if not allow_partial:
        model.load_state_dict(cleaned, strict=True)
        print(f"[load_local_weights] Loaded {len(cleaned)} keys (strict)")
    else:
        model_state = model.state_dict()
        filtered = OrderedDict()
        skipped = 0
        for k, v in cleaned.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped += 1
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"[load_local_weights] Loaded {len(filtered)} keys, skipped {skipped} keys (partial)")
        if missing:
            print(f"[load_local_weights] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[load_local_weights] Unexpected keys: {len(unexpected)}")


def resolve_ckpt_path(path: str, base_dir: str) -> str:
    path_expanded = os.path.expanduser(path)
    if os.path.isabs(path_expanded) and os.path.isfile(path_expanded):
        return path_expanded
    # relative to current working directory
    cwd_path = os.path.abspath(path)
    if os.path.isfile(cwd_path):
        return cwd_path
    # relative to base_dir
    base_path = os.path.abspath(os.path.join(base_dir, path))
    if os.path.isfile(base_path):
        return base_path
    # relative to parent of base_dir
    parent_path = os.path.abspath(os.path.join(base_dir, '..', path))
    if os.path.isfile(parent_path):
        return parent_path
    raise FileNotFoundError(f"Checkpoint path not found: {path}\n"
                            f"CWD: {os.getcwd()}\n"
                            f"Base dir: {base_dir}")
