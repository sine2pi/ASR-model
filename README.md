Echo

#### Relationship Between Pitch and Rotary Embeddings
The code implements two complementary pitch-based enhancements:

1. The first uses pitch to modify theta (rotary frequency)
2. The second adds direct similarity bias to attention

#### Intuition.

The rotary embeddings (RoPE) work by encoding positions using complex numbers with different frequencies. The theta parameter essentially controls how quickly these rotary patterns change across positions. This has a natural relationship to audio pitch:

```python
perceptual_factor = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
f0_theta = self.min_theta + perceptual_factor * (self.max_theta - self.min_theta)
```

This relationship works because:

1. **Both are frequency-based concepts**: Pitch is a frequency, and theta controls frequency of position encodings
2. **Both follow logarithmic perception**: The code uses a logarithmic scaling that matches how humans perceive pitch differences
3. **Both need different ranges for different content**: Just as low-pitched voices need different analysis than high-pitched ones, different content needs different position encoding patterns

Using pitch to adjust theta helps the model:

1. **Adapt to speaker characteristics**: Different speakers (bass, tenor, alto, soprano) have fundamentally different pitch ranges
2. **Process frequency-dependent information**: Formants and other speech features shift based on fundamental frequency
3. **Maintain consistent perceptual distances**: The logarithmic scaling ensures consistent representation across the pitch spectrum

Echos rotary implementation maps the perceptual properties of audio to the mathematical properties of the rotary embeddings, creating a more adaptive and context-aware representation system.

#### Pitch bias

The pitch bias implementation creates an attention bias matrix:

```python
def get_pitch_bias(self, f0):
    if f0 is None:
        return None
        
    f0_flat = f0.squeeze().float()
    f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
    f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                               f0_norm.unsqueeze(1)) * self.pitch_scale)
    return f0_sim.unsqueeze(0).unsqueeze(0)
```

This makes tokens with similar pitch attend to each other more, which helps:

- Track speaker consistency
- Maintain coherent pitch patterns
- Group harmonically related segments

The learned `pitch_scale` parameter lets the model tune how much to rely on pitch similarity.
