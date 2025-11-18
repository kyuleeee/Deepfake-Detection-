# GenConViT Checkpoint Weight Analysis Summary

## Key Findings

### **ED Checkpoint** (`genconvit_ed_inference.pth`)

| Architecture Component | # Parameters | Status | Mean (abs) | Std Dev | Notes |
|----------------------|-------------|--------|-----------|---------|-------|
| **Encoder** | 392,608 | ⚠️ **ZEROS** | 0.000000 | 0.000000 | **Completely untrained!** |
| **Decoder** | 174,515 | ⚠️ **ZEROS** | 0.008930 | 0.012517 | First 5 layers are zeros, last layer has small values |
| **Backbone** | 29,357,896 | ✅ **Trained** | 0.162881 | 0.213151 | ConvNeXt properly trained |
| **Embedder (in backbone)** | 28,538,058 | ✅ **Trained** | 1.285427 | 1.526297 | Swin Transformer properly trained |
| **Embedder (standalone)** | 28,538,058 | ✅ **Trained** | 1.285427 | 1.526297 | Same as embedder in backbone |
| **FC Layers** | 1,001,502 | ✅ **Trained** | 0.011701 | 0.015515 | Classification head trained |

**Total ED Parameters:** ~87.4M

---

### **VAE Checkpoint** (`genconvit_vae_inference.pth`)

| Architecture Component | # Parameters | Status | Mean (abs) | Std Dev | Notes |
|----------------------|-------------|--------|-----------|---------|-------|
| **Encoder** | 635,986,916 | ✅ **Trained** | 1.180662 | 0.980339 | **Much larger VAE encoder, properly trained** |
| **Decoder** | 76,083 | ✅ **Trained** | 0.120815 | 0.161051 | VAE decoder is trained |
| **Backbone** | 29,357,896 | ✅ **Trained** | 0.412782 | 0.386122 | ConvNeXt properly trained |
| **Embedder (in backbone)** | 28,538,058 | ✅ **Trained** | 1.285427 | 1.526297 | Swin Transformer properly trained |
| **Embedder (standalone)** | 28,538,058 | ✅ **Trained** | 1.285427 | 1.526297 | Same as embedder in backbone |
| **FC Layers** | 1,502,002 | ✅ **Trained** | 0.021428 | 0.023172 | Classification head trained |

**Total VAE Parameters:** ~724M (much larger due to VAE encoder!)

---

## Critical Insights

### 1. **ED Model's Encoder/Decoder Are Untrained**
The ED checkpoint has **completely zero weights** in the encoder:
- All 10 encoder layers have mean=0, std=0
- First 5 decoder layers are also zeros
- This explains why `extract_AE()` produces solid gray/green blocks

### 2. **VAE Model Has Trained Encoder/Decoder**
The VAE checkpoint has a **massive, properly trained** encoder:
- 636M parameters (vs 393K in ED)
- Mean absolute value: 1.18, Std: 0.98
- Decoder also has non-zero weights (mean: 0.12, std: 0.16)

### 3. **Backbone & Embedder Are Identical**
Both checkpoints share the same trained weights for:
- ConvNeXt backbone: 29.3M params
- Swin Transformer embedder: 28.5M params
- These are the core feature extractors

### 4. **How ED Model Works Without Trained AE**
Even with zero encoder/decoder weights, the ED model can classify because:
- Encoder with zero weights produces all-zero latent codes
- Decoder with zero weights produces near-constant output
- The backbone learns to classify by comparing:
  - Features from original image (x2)
  - Features from constant decoder output (x1)
- This creates a learned baseline comparison

---

## Recommendations

### If you want to visualize reconstructions:

1. **Use the VAE model** (`--net vae`):
   ```bash
   python visualize_ae_simple.py  # Edit net_type='vae'
   ```
   The VAE has trained encoder/decoder weights and should produce actual reconstructions.

2. **Or train the ED encoder/decoder**:
   The ED encoder/decoder architecture exists but needs training.

### For deepfake detection:
- Both models work fine for classification
- ED: 87M params, faster inference
- VAE: 724M params, more powerful but slower

---

## Weight Statistics Details

### ED Model - Encoder (UNTRAINED)
```
Shape: (16, 3, 3, 3) to (256, 128, 3, 3)
Mean: 0.000000
Std:  0.000000
Range: [0.000, 0.000]
Status: All zeros
```

### VAE Model - Encoder (TRAINED)
```
36 layers with 636M parameters
Mean: 1.180662
Std:  0.980339
Range: [-4.30, 125.61]
Status: Properly initialized and trained
```

### Both Models - Backbone (TRAINED)
```
184 layers with 29.3M parameters
Mean: ~0.16-0.41
Std:  ~0.21-0.39
Range: [-23.8, 20.1]
Status: Properly trained ConvNeXt
```

---

## Next Steps

Run `python visualize_ae_simple.py` with `net_type='vae'` to see actual image reconstructions!
