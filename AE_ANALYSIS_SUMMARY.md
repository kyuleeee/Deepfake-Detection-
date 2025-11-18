# Autoencoder Analysis Summary

## What We Found

The `extract_AE()` method in GenConViT uses a simple encoder-decoder architecture defined in `models/genconvit_ed.py`:
- **Encoder**: CNN that compresses images from 224x224x3 to 7x7x256
- **Decoder**: Transposed CNN that reconstructs images from 7x7x256 back to 224x224x3

## The Problem

The encoder/decoder weights in the checkpoint **are all zeros** (or extremely small values):
- Encoder output range: `[0.000, 0.000]` (all zeros!)
- This causes the decoder to output nearly constant values: `[0.000, 0.200]`
- Only 12 unique values across the entire 224x224x3 reconstruction
- Result: Solid gray/green blocks instead of actual image reconstructions

## Why This Happens

The standalone encoder-decoder autoencoder was **never properly trained**. The checkpoint contains:
1. ✅ Backbone (ConvNeXt) weights - properly trained
2. ✅ Embedder (Swin Transformer) weights - properly trained
3. ❌ Encoder/Decoder weights - all zeros (untrained)

The model's main deepfake detection works by comparing:
- Features from the original image
- Features from backbone + embedder
- But NOT using the encoder/decoder reconstruction

## How the Model Actually Works

Looking at `models/genconvit_ed.py:119-131`, the forward pass:

```python
def forward(self, images):
    encimg = self.encoder(images)  # Encodes image
    decimg = self.decoder(encimg)   # Decodes (but with zero weights!)

    x1 = self.backbone(decimg)      # Features from reconstructed
    x2 = self.backbone(images)      # Features from original

    x = torch.cat((x1, x2), dim=1)  # Concatenate both
    x = self.fc2(self.relu(self.fc(self.relu(x))))  # Classify
    return x
```

Since the encoder/decoder weights are zeros:
- `decimg` is essentially a constant tensor
- `x1 = self.backbone(decimg)` produces features from this constant
- The model learns to classify based on the difference between features from the original and features from the constant

This is an unusual architecture, but it seems to work for deepfake detection by learning what "original" features look like compared to a constant baseline.

## Visualizations Created

1. **`ae_visualizations/reconstruction_results.png`**
   - Shows original images vs AE reconstruction
   - Demonstrates the reconstruction failure

2. **`ae_visualizations/reconstruction_detailed.png`**
   - Shows original, normalized input, raw AE output, and scaled AE output
   - Reveals the limited range [0.000, 0.200]

3. **`ae_visualizations/normalized_vs_unnormalized.png`**
   - Compares normalized vs unnormalized inputs
   - Both produce the same poor reconstruction

## Scripts Created

- **`visualize_ae_simple.py`** - Simple visualization with no arguments
- **`visualize_ae.py`** - Full-featured with CLI options
- **`visualize_ae_fixed.py`** - Detailed multi-view visualization
- **`debug_ae_output.py`** - Debug script showing tensor statistics
- **`test_ae_unnormalized.py`** - Tests different input normalizations
- **`check_model_weights.py`** - Inspects model weights and checkpoint

## Conclusion

The `extract_AE()` method cannot produce meaningful image reconstructions because the encoder/decoder were never trained. If you want to visualize actual autoencoder reconstructions, you would need to:

1. Train the encoder/decoder separately, OR
2. Use the VAE version (`net='vae'`) if it has trained weights, OR
3. Focus on the model's actual purpose: deepfake detection via feature comparison

The model achieves deepfake detection not through reconstruction quality, but through learned feature differences between original images and a constant baseline produced by the untrained autoencoder.
