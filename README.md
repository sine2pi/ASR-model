

NLP/ASR multimodal pitch aware model. Research model.

<img width="780" alt="cc5" src="https://github.com/user-attachments/assets/ce9417de-a892-4811-b151-da612f31c0fb"  />

**This plot illustrates the pattern similiarity of pitch and spectrogram. (librispeech)

Hypotheses 

By modulating the RoPE frequencies based on pitch (F0), we are essentially telling the model to pay attention to the acoustic features relate to sequence position in a way that's proportional to the voice characteristics.  This approach creates a more speech-aware positional representation that helps the model better understand the relationship between acoustic features and text.


To highlight the relationship between pitch and rotary embeddings, the model implements three complementary pitch-based enhancements:

1. **Pitch-modulated theta:** Pitch (f0) is used to modify the theta parameter, dynamically adjusting the rotary frequency.
2. **Direct similarity bias:** A pitch-based similarity bias is added directly to the attention mechanism.
3. **Variable radii in torch.polar:** The unit circle radius (1.0) in the `torch.polar` calculation is replaced with variable radii derived from f0. This creates acoustically-weighted positional encodings, so each position in the embedding space reflects the acoustic prominence in the original speech. This approach effectively adds phase information without significant computational overhead.


This accurately describes the mechanisms in your code and their intended effects.

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

<img width="780" alt="cc" src="https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8"  />
<img width="780" alt="cc2" src="https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1"  />


```python


def theta_freqs(self, theta):
    freq = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.dim // 2, device=device,               dtype=dtype) / 2595) - 1) / 1000
    freqs = nn.Parameter(torch.tensor(freq, device=device, dtype=dtype), requires_grad=True)        
    return freqs

### then theta is updated in the forward:

    def forward(self, x=None, enc=None, layer=None, feature_type="audio") -> Tensor:
        f0 = enc.get("f0") if enc is not None else None 
        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 2:
            batch, ctx = x.shape
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape
        t = torch.arange(ctx, device=device, dtype=dtype)

        if f0 is not None and f0.dim() == 2:
            if f0.shape[0] == 1: 
                f0 = f0.squeeze(0)  
            else:
                f0 = f0.view(-1)        

        if f0 is not None: # take the mean of f0 for theta since it's a scalar anyway.
            f0_mean = f0.mean()
            theta = f0_mean + self.theta
        else:
            theta = self.theta 
        freqs = self.theta_freqs(theta)

        freqs = t[:, None] * freqs[None, :]
        if self.radii and f0 is not None: # Here we need to maintain the pitch chracteristics across dimensions.
            radius = f0.to(device, dtype)
            L = radius.shape[0]
            if L != ctx:
                F = L / ctx
                idx = torch.arange(ctx, device=f0.device)
                idx = (idx * F).long().clamp(0, L - 1)
                radius = radius[idx]
                rad = radius
            radius = radius.unsqueeze(-1).expand(-1, freqs.shape[-1])
            radius = torch.sigmoid(radius)
        else:
            radius = torch.ones_like(freqs) 
        freqs = torch.polar(radius, freqs)

        self.counter += 1
        return freqs.unsqueeze(0)

    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        if x1.ndim == 2:
            x1 = x1.unsqueeze(0)
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)

```
