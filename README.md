NLP/ASR multimodal modal with f0 modulated relative positional embeddings. 
For research/testing.

Moving beyond literal transcription toward something more artistic and creative.

      On a road to build a "creative" speech-to-text model that can:
      
      -Generate stories from audio
      -Make poetic associations
      -Fill in gaps with imagination
      -Create richer, more expressive text..
      -Seperate the sad Morrissey songs from the two that arn't.. 


----

Questions:

   -How can we make attention mechanisms aware of speech-specific properties?
   
   -Can we incorporate acoustic information directly into positional encodings?
   
   -Does pitch-conditioning improve speech recognition?

   Standard RoPE was designed for text: Text doesn't have pitch, timing, or acoustic properties.
   
---



<img width="780" alt="cc5" src="https://github.com/user-attachments/assets/106ebe75-f1db-4f85-bdae-818b114fedd2"  />

This plot illustrates the pattern similiarity of pitch waveform and spectrogram. (librispeech - clean).

To explore the relationship between pitch and rotary embeddings, the model implements three complementary pitch based enhancements:

1. Pitch modulated theta Pitch (f0) is used to modify the theta parameter, dynamically adjusting the rotary frequency.
2. Direct similarity bias: A pitch based similarity bias is added directly to the attention mechanism.
3. Variable radii in torch.polar: The unit circle radius (1.0) in the torch.polar calculation is replaced with variable radii derived from f0. This creates acoustically-weighted positional encodings, so each position in the embedding space reflects the acoustic prominence in the original speech. This approach effectively adds phase and amplitutde information without significant computational overhead.

The function `torch.polar` constructs a complex tensor from polar coordinates:

````python
# torch.polar(magnitude, angle) returns:
result = magnitude * (torch.cos(angle) + 1j * torch.sin(angle))
````

So, for each element:
- magnitude is the modulus (radius, r)
- angle is the phase (theta, in radians)
- The result is: `r * exp(i * theta) = r * (cos(theta) + i * sin(theta))`

Reference: [PyTorch Documentation - torch.polar](https://pytorch.org/docs/stable/generated/torch.polar.html)

Here are the abbreviated steps for replacing theta and radius in the rotary forward:

```python

f0 = f0.to(device, dtype) # feature extracted during processing
if f0 is not None:
    if f0.dim() == 2:
        f0 = f0.squeeze(0) 
    theta = f0 + self.theta  
else:
    theta = self.theta 

## In text, theta=10,000 sets the base frequency for positional encoding, ensuring a wide range of periodicities for long sequences. I'm not sure if the specific number 10k was experimentally derived.
## For audio, especially speech, the relevant periodicities are determined by the pitch (f0 neighborhood or f0 per frame) might be more meaningful. 

freqs = (theta.unsqueeze(-1) / 220.0) * 700 * (
    torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
            self.dim // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000

## This seems to give better results compared to the standard freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)).
## I thought a mel-scale version might be more perceptually meaningful for audio.. ie. using mel-scale to create a perceptually-relevant distance metric instead of Euclidean distance.

t = torch.arange(ctx, device=device, dtype=dtype)
freqs = t[:, None] * freqs  # dont repeat or use some other method here 

if self.radii and f0 is not None:
    radius = f0.to(device, dtype)
    freqs = torch.polar(radius.unsqueeze(-1), freqs)
else:
    radius = torch.ones_like(freqs)
    freqs = torch.polar(radius, freqs)

```

A closer look at whats going on. Here is a slice of the actual radius values for one step
      
      [encoder] [Radius] torch.Size([454]) 92.32 [Theta] 10092.01 [f0] torch.Size([454]) [Freqs] torch.Size([454, 64]) 2.17+1.17j [ctx] 454
      
     [encoder] [Radius] tensor([  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
           283.7590, 260.6043, 257.3410, 261.8319, 249.1852, 257.2541, 263.8165,
           272.2421, 277.6960, 286.9628, 303.8460, 305.1561, 319.5129, 330.6942,
           362.0888, 355.8571, 352.8432, 336.9354, 313.0566, 319.9086, 303.4355,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 220.4299,
           254.7619, 239.6506, 228.8830, 227.3063, 225.6784, 225.7169, 211.7767,
           223.6572, 223.4174, 222.4496, 225.1645, 228.7840, 231.8760, 228.9148,
           230.6227,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000, 205.7097, 202.8816, 182.0329, 181.6536, 180.2186,
           177.8911, 176.0775, 171.3846, 173.9602, 170.4824, 171.5723, 172.0810,
           174.3897, 177.3261, 188.3212, 188.9799, 186.7493, 221.3487,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 304.4458,
           263.3122, 251.7635, 211.8467, 207.5651, 195.3680, 184.0717, 206.3800,
           197.8661,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000, 197.6399, 195.0042, 190.7016, 187.0234, 183.5980,
           183.6842, 185.0038, 185.5778, 187.4167, 185.5085, 183.4160,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000, 190.3175, 209.7377, 206.2731, 211.9862, 219.2756, 214.3068,
           202.6881, 192.1823, 210.3404, 235.5456, 230.7845, 234.5441, 234.9773,
           241.1199, 241.9640, 237.0773, 231.6952, 238.0375, 257.9242, 264.4094,
           265.3747, 251.0286, 245.7093,   0.0000, 274.9167, 273.4767, 271.6227,
           256.5457, 245.8942, 251.3361, 240.1572, 228.9316,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000, 202.6190, 217.7865, 212.3347, 208.2926,
           209.9206, 209.3961, 210.3909, 211.6021, 202.8511, 205.1674, 211.7455,
           217.9954,   0.0000, 264.9778, 229.7112, 200.8905, 182.4680, 179.4812,
           175.4307, 172.7844, 173.6305, 172.1901, 170.5743, 167.2979, 166.7781,
           166.7783, 170.8816, 173.0406, 176.2869, 181.9142, 212.7904, 170.4449,
           173.1710, 168.3079, 154.1663,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000, 211.3942, 202.3412, 203.6764, 198.4441,
           186.2687, 209.0010, 209.5012, 214.6487, 203.8741, 195.8432, 180.9673,
             0.0000,   0.0000,   0.0000, 197.7340, 198.9476, 204.5347, 209.5858,
           204.5406, 195.1048, 198.1545, 199.8559, 207.3548, 217.9402, 217.2366,
           216.4711, 212.4731, 217.5183, 218.0658, 208.7833,   0.0000, 243.7485,
           215.1998, 235.4733, 215.3242, 215.1489, 212.6266, 203.9319, 191.8531,
           197.2219, 202.7850,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 224.1991, 167.9602,
           190.8241, 178.5659, 175.4639, 172.6353, 173.5884, 173.2250,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 209.6374, 196.2949,
           216.4672, 236.3051, 195.2339, 241.1573,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000, 195.8783, 145.3826,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],
          device='cuda:0')
          
What the Radius Values Tell Us:

1. Speech Structure is Clear
   
         Zeros: Silence/unvoiced segments (no pitch)
         Non-zero values: Voiced speech segments (with pitch)
         Pattern: 0.0000 → 283.7590 → 0.0000 → 220.4299

2. Pitch Range is Realistic
   
         Range: ~145-365 Hz
         Typical speech: 80-400 Hz for most speakers
         Model values: 145-365 Hz

3. Temporal Dynamics
   
         Clusters: Pitch values cluster together (natural speech)
         Transitions: Smooth changes between values
         Silence gaps: Natural pauses in speech

         Silence detection: 0.0000 = no pitch (silence/unvoiced)
         Pitch extraction: 283.7590 = actual f0 values
         Speech segmentation: Clear boundaries between voiced/unvoiced
         
         Realistic values: 145-365 Hz is normal speech range
         Proper structure: Matches natural speech patterns
         Variable radius: Working as intended

The Complex Frequency Result:

      [Freqs] torch.Size([454, 64]) 2.17+1.17j

This means:

      Magnitude: sqrt(2.17² + 1.17²) ≈ 2.5
      Phase: atan2(1.17, 2.17) ≈ 0.49 radians
      
      Variable radius: Each frame has different magnitude

   Pitch Conditioning is Working:

      Silence frames: radius ≈ 0 → freqs ≈ 0
      Voiced frames: radius ≈ 200-300 → freqs ≈ 2-3
      
      Variable attention: Important frames get more attention

The Pattern Makes Sense:

      Silence: No acoustic prominence → low radius
      Speech: High acoustic prominence → high radius
      Transitions: Natural pitch changes

----

Approximation methods like using cos/sin projections or fixed rotation matrices typically assume a unit circle (radius=1.0) or only rotate, not scale. When we introduce a variable radius, those approximations break down and can't represent the scaling effect, only the rotation. When using a variable radius, we should use true complex multiplication to get correct results. Approximations that ignore the radius or scale after the rotation don't seem to capture the intended effect, leading to degraded or incorrect representations from my tests so far.

```python

### Do not approximate:
#     radius = radius.unsqueeze(-1).expand_as(x_rotated[..., ::2])
#     x_rotated[..., ::2] = x_rotated[..., ::2] * radius
#     x_rotated[..., 1::2] = x_rotated[..., 1::2] * radius

### 
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
This approach respects both the rotation (phase) and the scaling (radius) for each token/head, so the rotary embedding is applied when the radius varies.

<img width="780" alt="cc4" src="https://github.com/user-attachments/assets/165a3f18-659a-4e2e-a154-a3456b667bae"  />

Each figure shows 4 subplots one for each of the first 4 dimensions of your embeddings in the test run. These visualizations show how pitch information modifies position encoding patterns in the model.

In each subplot:

- Thick solid lines: Standard RoPE rotations for even dimensions (no F0 adaptation)
- Thick dashed lines: Standard RoPE rotations for odd dimensions (no F0 adaptation)
- Thin solid lines: F0 RoPE rotations for even dimensions
- Thin dashed lines: F0 RoPE rotations for odd dimensions

1. Differences between thick and thin lines: This shows how much the F0 information is modifying the standard position encodings. Larger differences indicate stronger F0 adaptation.

2. Pattern changes: The standard RoPE (thick lines) show regular sinusoidal patterns, while the F0 RoPE (thin lines) show variations that correspond to the audio's pitch contour.

3. Dimension specific effects: Compared across four subplots to see if F0 affects different dimensions differently.

4. Position specific variations: In standard RoPE, frequency decreases with dimension index, but F0 adaptation modify this pattern.

The patterns below show how positions "see" each other in relation to theta and f0. 

Bright diagonal line: Each position matches itself perfectly.
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

<img width="680" alt="cc" src="https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8"  />
<img width="680" alt="cc2" src="https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1"  />

----
https://huggingface.co/Sin2pi/Echo17/tensorboard?params=scalars

----

This model sometimes uses :

https://github.com/sine2pi/Maxfactor

MaxFactor is a custom PyTorch optimizer with adaptive learning rates and specialized handling for matrix parameters. I wrote it for the model in the asr_model repository. I needed something that performs well and has a light memory foot print since I do everything from my laptop.
