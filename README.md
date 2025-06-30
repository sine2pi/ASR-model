

NLP/ASR multimodal pitch aware model. Research model.

<img width="780" alt="cc5" src="https://github.com/user-attachments/assets/ce9417de-a892-4811-b151-da612f31c0fb"  />

**This plot illustrates the pattern similiarity of pitch and spectrogram. (librispeech

To highlight the relationship between pitch and rotary embeddings, the model implements three complementary pitch-based enhancements:

1. **Pitch-modulated theta:** Pitch (f0) is used to modify the theta parameter, dynamically adjusting the rotary frequency.
2. **Direct similarity bias:** A pitch-based similarity bias is added directly to the attention mechanism.
3. **Variable radii in torch.polar:** The unit circle radius (1.0) in the `torch.polar` calculation is replaced with variable radii derived from f0. This creates acoustically-weighted positional encodings, so each position in the embedding space reflects the acoustic prominence in the original speech. This approach effectively adds phase information without significant computational overhead.


<img width="780" alt="cc4" src="https://github.com/user-attachments/assets/165a3f18-659a-4e2e-a154-a3456b667bae"  />

Each figure shows 4 subplots (one for each of the first 4 dimensions of your embeddings in the test run). These visualizations show how pitch information modifies position encoding patterns in the model.

In each subplot:

- **Thick solid lines**: Standard RoPE rotations for even dimensions (no F0 adaptation)
- **Thick dashed lines**: Standard RoPE rotations for odd dimensions (no F0 adaptation)
- **Thin solid lines**: F0-adapted RoPE rotations for even dimensions
- **Thin dashed lines**: F0-adapted RoPE rotations for odd dimensions


1. **Differences between thick and thin lines**: This shows how much the F0 information is modifying the standard position encodings. Larger differences indicate stronger F0 adaptation.

2. **Pattern changes**: The standard RoPE (thick lines) show regular sinusoidal patterns, while the F0-adapted RoPE (thin lines) show variations that correspond to the audio's pitch contour.

3. **Dimension-specific effects**: Compared across four subplots to see if F0 affects different dimensions differently.

4. **Position-specific variations**: In standard RoPE, frequency decreases with dimension index, but F0 adaptation modify this pattern.

The patterns below show how positions "see" each other in relation to theta and f0. 

Bright diagonal line: Each position matches itself perfectly.
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

<img width="680" alt="cc" src="https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8"  />
<img width="680" alt="cc2" src="https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1"  />

----

#### Diagnostic test run where 1 epoch = 1000 steps = 1000 samples:

<img width="680" alt="1555" src="https://github.com/user-attachments/assets/5bed0421-e32f-4234-ab55-51d64eb927ef" />

