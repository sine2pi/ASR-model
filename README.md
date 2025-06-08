## Echo

### Leveraging Silence for Natural Generation Stopping

#### Intuition: Help the model learn to naturally stop at silence.

By scaling attention scores related to pad/silence tokens down to near zero, we are creating a consistent pattern that the model can learn:

```python
token_ids = k[:, :, :, 0]
zscale = torch.ones_like(token_ids)
fzero = torch.clamp(F.softplus(self.fzero), self.min, self.max)
zscale[token_ids.float() == self.pad_token] = fzero.to(q.device, q.dtype)
```

This creates a direct correspondence between:
- Log mel spectrograms near zero in silent regions
- Attention scores that are scaled down for pad tokens

The benefits of this approach:

1. **Consistent signals**: The model gets similar signals for silence in both input data and attention patterns
2. **Content-based stopping**: Model can learn to end generation based on acoustic properties rather than just position
3. **Learnable behavior**: Using a learnable parameter (`self.fzero`) lets the model find the optimal scaling factor

This might be particularly useful for speech models where natural pauses and silences should guide generation boundaries.
The model also ignores 0 in the loss calculation and uses 0 for all special tokens.
Anything not near zero (or not zero) is an audio feature or the corresponding tokenized transcription of the feature.

#### Token and Value Handling in the Model

1. **Zero in loss calculation**:
   ```python
   loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
   ```
   The model explicitly ignores index 0 in the loss calculation using `ignore_index=0` parameter.

2. **Zero for special tokens**:
   ```python
   tokenizer.pad_token_id = 0
   tokenizer.eos_token_id = 0
   tokenizer.bos_token_id = 0
   ```
3. **Attention scaling for zero tokens**:
   ```python
   zscale = torch.ones_like(token_ids)
   fzero = torch.clamp(F.softplus(self.fzero), self.min, self.max)
   zscale[token_ids.float() == self.pad_token] = fzero.to(q.device, q.dtype)
   ```
   This specifically scales attention scores for pad tokens (0) down to near-zero values.

4. **Non-zero values represent meaningful content**:
   In the processing pipeline, actual audio features (after processing) and text tokens (after tokenization) are represented by non-zero values, making them stand out from the padded/silent regions.

The approach of scaling down attention for padding/silent regions helps the model distinguish between content and non-content.


### Relationship Between Pitch and Rotary Embeddings
The code implements two complementary pitch-based enhancements:

1. The first uses pitch to modify theta (rotary frequency)
2. The second adds direct similarity bias to attention

#### Intuition: The rotary embeddings (RoPE) work by encoding positions using complex numbers with different frequencies. The theta parameter essentially controls how quickly these rotary patterns change across positions. This has a natural relationship to audio pitch:

```python
perceptual_factor = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
f0_theta = self.min_theta + perceptual_factor * (self.max_theta - self.min_theta)
```

This relationship should work because:

1. **Both are frequency-based concepts**: Pitch is a frequency, and theta controls frequency of position encodings
2. **Both follow logarithmic perception**: The code uses a logarithmic scaling that matches how humans perceive pitch differences
3. **Both need different ranges for different content**: Just as low-pitched voices need different analysis than high-pitched ones, different content needs different position encoding patterns

Using pitch to adjust theta helps the model:

1. **Adapt to speaker characteristics**: Different speakers (bass, tenor, alto, soprano) have fundamentally different pitch ranges
2. **Process frequency-dependent information**: Formants and other speech features shift based on fundamental frequency
3. **Maintain consistent perceptual distances**: The logarithmic scaling ensures consistent representation across the pitch spectrum

Echos rotary implementation maps the perceptual properties of audio to the mathematical properties of the rotary embeddings, creating a more adaptive and context-aware representation system. Pitch is optionally extracted from audio in the data processing pipeline and can be used for an additional feature along with spectrograms and or used to inform the rotary and or pitch bias.

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

```python

class rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, variable_radius=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = []):
        super().__init__()
        self.use_pbias = True 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        dtype = torch.float32 
        self.dtype = dtype
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        
        if learned_freq:
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)),
                requires_grad=learned_freq)
        
        if learned_theta:
            self.theta = nn.Parameter(
                torch.tensor(float(theta)), requires_grad=learned_theta)
            self.min_theta = nn.Parameter(
                torch.tensor(800.0), requires_grad=learned_theta)
            self.max_theta = nn.Parameter(
                torch.tensor(10000.0), requires_grad=learned_theta)
        
        if learned_pitch:
            self.pitch_scale = nn.Parameter(torch.tensor(1.0), 
                                            requires_grad=learned_pitch)
    
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius)

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
            
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def add_to_rotary(self):
        def get_sim(self, freqs):
            real = freqs.real.squeeze(0)
            imag = freqs.imag.squeeze(0)
            vecs = torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], dim=-1)
            vecs = vecs.squeeze(-2)
            return F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
            
        def fwd_sim(self, x=None, f0=None):
            freqs = self.forward(x, f0)
            sim = get_sim(self, freqs)
            return freqs, sim
            
        rotary.get_sim = get_sim
        rotary.fwd_sim = fwd_sim

    def forward(self, x = None, f0=None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=self.device).float()
        else:
            t = x.float().to(self.inv_freq.device)

        if f0 is not None:
            f0_tensor = f0.squeeze(0) if f0.ndim == 3 else f0
            if f0_tensor.ndim > 1:
                f0_tensor = f0_tensor.squeeze()
            f0_mean = f0_tensor.mean()
            f0_mean = torch.clamp(f0_mean, min=80.0, max=600.0)
            perceptual_factor = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
            # min_theta, max_theta = 800.0, 10000.0
            # f0_theta = min_theta + perceptual_factor * (max_theta - min_theta)
            f0_theta = self.min_theta + perceptual_factor * (self.max_theta - self.min_theta)
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=self.device) / self.dims))
        else:
            inv_freq = self.inv_freq
        freqs = torch.einsum('i,j->ij', t, inv_freq)

        freqs = freqs.float()
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)
        
        if "rotary" in self.debug:
            if self._counter == 1:
                print(f'ROTA -- freqs: {freqs.shape}, x: {x},  {t.shape if x is not None else None}', freqs.shape, t.shape)
            if f0 is not None and self._counter % 100 == 0:
                print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
            self._counter += 1
            
        return freqs

    @staticmethod
    def apply_rotary(x, freqs):
        multihead_format = len(freqs.shape) == 4
        if multihead_format:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)

        else:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            
            if x.ndim == 2:  
  
                x1 = x1.unsqueeze(0)
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs
                x1 = torch.view_as_real(x1).flatten(-2)
                x1 = x1.squeeze(0)  
                return torch.cat([x1.type_as(x), x2], dim=-1)
            else:  
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs
                x1 = torch.view_as_real(x1).flatten(-2)
                return torch.cat([x1.type_as(x), x2], dim=-1)
```

The other steps take place in the attention layer, auxiliary embedding blocks and during data processing which are included with model.py.
