## Echo
### Zero-Value Processing ASR model with Voice-modulated Rotary Position Encoding. (vRoPE)

(All of this is currently being tested and none of it is confirmed to work at all)

Leveraging Silence for Natural Generation Stopping
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

1. Consistent signals: The model gets similar signals for silence in both input data and attention patterns
2. Content-based stopping: Model can learn to end generation based on acoustic properties rather than just position
3. Learnable behavior: Using a learnable parameter (`self.fzero`) lets the model find the optimal scaling factor

Token and Value Handling in the Model
1. Zero in loss calculation:
   ```python
   loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
   ```
   The model explicitly ignores index 0 in the loss calculation using `ignore_index=0` parameter.
2. Zero for special tokens:
   ```python
   tokenizer.pad_token_id = 0
   tokenizer.eos_token_id = 0
   tokenizer.bos_token_id = 0
   ```
3. Attention scaling for zero tokens:
   ```python
   zscale = torch.ones_like(token_ids)
   fzero = torch.clamp(F.softplus(self.fzero), self.min, self.max)
   zscale[token_ids.float() == self.pad_token] = fzero.to(q.device, q.dtype)
   ```
   This specifically scales attention scores for pad tokens (0) down to near-zero values.

In standard usage, RoPE encodes relative positional information by applying frequency-based rotations to token embeddings. What this does is creates a meaningful bridge between two domains:

1. Token positions (sequential information)
2. Pitch information (acoustic properties)

By modulating the RoPE frequencies based on pitch (F0), we are essentially telling the model: "pay attention to how these acoustic features relate to sequence position in a way that's proportional to the voice characteristics."

The theoretical foundation:
- Both position and pitch can be represented as frequencies
- Speech has inherent rhythmic and tonal patterns that correlate with semantic content
- Varying the rotation frequency based on pitch creates a more speech-aware positional encoding

This approach creates a more speech-aware positional representation that helps the model better understand the relationship between acoustic features and text.

Relationship Between Pitch and Rotary Embeddings
The code implements two complementary pitch-based enhancements:

1. The first uses pitch to modify theta (rotary frequency)
2. The second adds direct similarity bias to attention

The patterns show how positions "see" each other:
Bright diagonal line: Each position matches itself perfectly
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

![2](https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8)
![1](https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1)

plot code:
```python

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def test_rot(ctx=12, dims=120, f0s=[10, 100, 200, 300, 400, 500, 600]):
   
    results = {}
    t = torch.arange(ctx).float()
    
    theta = 5000
    inv_f = 1.0 / (theta ** (torch.arange(0, dims, 2) / dims))
    
    base_f = torch.einsum('i,j->ij', t, inv_f)
    base_emb = torch.polar(torch.ones_like(base_f), base_f)
    
    real = base_emb.real
    imag = base_emb.imag
    vecs = torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], dim=-1).squeeze(-2)
    base_sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
    
    results["base"] = base_sim
    results["rope"] = base_sim  # Standard RoPE with θ=10000
    
    for f0 in f0s:
        f0c = torch.clamp(torch.tensor(f0), min=80.0, max=600.0)
        fac = torch.log(1 + f0c / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
        f0_t = 600.0 + fac * (2400.0 - 600.0)
        o_inv = 1.0 / (f0_t ** (torch.arange(0, dims, 2) / dims))
        o_freq = torch.einsum('i,j->ij', t, o_inv)
        o_emb = torch.polar(torch.ones_like(o_freq), o_freq)
        
        h_fac = torch.log(1 + f0c / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
        h_freq = base_f * (1.0 + 0.3 * h_fac)
        h_emb = torch.polar(torch.ones_like(h_freq), h_freq)
        
        for name, emb in [("orig"+str(f0), o_emb), ("hyb"+str(f0), h_emb)]:
            r = emb.real
            i = emb.imag
            v = torch.cat([r.unsqueeze(-2), i.unsqueeze(-2)], dim=-1).squeeze(-2)
            sim = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)
            results[name] = sim
    
    return results

sims = test_rot()
rows = 2
cols = len(sims) // rows + (1 if len(sims) % rows else 0)
fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
axes = axes.flatten()

for i, (key, sim) in enumerate(sims.items()):
    axes[i].imshow(sim.numpy(), cmap='viridis')
    axes[i].set_title(key)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

def test_mod(ctx=10, dims=64, mods=[0.1, 0.3, 0.5, 0.7]):
    t = torch.arange(ctx).float()
    theta = 10000
    inv_f = 1.0 / (theta ** (torch.arange(0, dims, 2) / dims))
    base_f = torch.einsum('i,j->ij', t, inv_f)
    
    f0s = [100, 1000, 10000]
    res = {)

    base = torch.polar(torch.ones_like(base_f), base_f)
    r = base.real
    i = base.imag
    v = torch.cat([r.unsqueeze(-2), i.unsqueeze(-2)], dim=-1).squeeze(-2)
    res["base"] = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)
    
    for f0 in f0s:
        f0c = torch.clamp(torch.tensor(f0), min=80.0, max=600.0)
        fac = torch.log(1 + f0c / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
        
        for m in mods:
            freq = base_f * (1.0 + m * fac)
            emb = torch.polar(torch.ones_like(freq), freq)
            
            r = emb.real
            i = emb.imag
            v = torch.cat([r.unsqueeze(-2), i.unsqueeze(-2)], dim=-1).squeeze(-2)
            sim = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)
            
            res[f"f{f0}_m{m}"] = sim
    
    return res

sims = test_mod()
n = len(sims)
r = 3
c = (n + r - 1) // r
fig, ax = plt.subplots(r, c, figsize=(12, 9))
ax = ax.flatten()

for i, (k, s) in enumerate(sims.items()):
    ax[i].imshow(s.numpy(), cmap='viridis')
    ax[i].set_title(k)
    ax[i].axis('off')

plt.tight_layout()
plt.show()
```

Echos rotary implementation maps the perceptual properties of audio to the mathematical properties of the rotary embeddings, creating a more adaptive and context-aware representation system. Pitch is optionally extracted from audio in the data processing pipeline and can be used for an additional feature along with spectrograms and or used to inform the rotary and or pitch bias.

Pitch bias

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
    _seen = set()  
    def __init__(self, dims, max_ctx=1500, theta=5000, learned_freq=False, variable_radius=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = []):
        super().__init__()
        self.use_pbias = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        dtype = torch.float32 
        self.dtype = dtype
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        
        self.inv_freq = nn.Parameter(
                1.0 / (5000  (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)),
                requires_grad=learned_freq)
        self.theta = nn.Parameter(
            torch.tensor(float(theta)), requires_grad=learned_theta)
        self.min_theta = nn.Parameter(
            torch.tensor(600.0), requires_grad=learned_theta)
        self.max_theta = nn.Parameter(
            torch.tensor(2400.0), requires_grad=learned_theta)
        
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
            inv_freq = 1.0 / (f0_theta  (torch.arange(0, self.dims, 2, device=self.device) / self.dims))
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
            if f0 is not None:
                key = f"{self._counter}_{f0_theta:.2f}"
                if key not in rotary._seen:
                    if not hasattr(self, '_prev_f0_theta'):
                        self._prev_f0_theta = f0_theta
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                    elif abs(self._prev_f0_theta - f0_theta) > 200.0:
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                        self._prev_f0_theta = f0_theta
                    rotary._seen.add(key)
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
