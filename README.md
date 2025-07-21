ASR model that uses audio frequencies instead of spectrograms + pitch aware relative positional embeddings. 



Questions:

   -How can we make attention mechanisms aware of speech-specific properties?
   
   -Can we incorporate acoustic information directly into positional encodings?
   
   -Does pitch-conditioning improve speech recognition?

---




To explore the relationship between pitch and rotary embeddings, the model implements three complementary pitch based enhancements:

1. Pitch modulated theta Pitch f0 is used to modify the theta parameter, dynamically adjusting the rotary frequency.
2. Direct similarity bias: A pitch based similarity bias is added directly to the attention mechanism.
3. Variable radii in torch.polar: The unit circle radius 1.0 in the torch.polar calculation is replaced with variable radii derived from f0. This creates acoustically-weighted positional encodings, so each position in the embedding space reflects the acoustic prominence in the original speech. This approach effectively adds phase and amplitutde information without significant computational overhead.

4. Initial findings suggest that f0 is a superior input to spectrograms for ASR models.

The function `torch.polar` constructs a complex tensor from polar coordinates:

````python
# torch.polarmagnitude, angle returns:
result = magnitude * torch.cosangle + 1j * torch.sinangle
````

So, for each element:
- magnitude is the modulus radius, r
- angle is the phase theta, in radians
- The result is: `r * expi * theta = r * costheta + i * sintheta`

Reference: [PyTorch Documentation - torch.polar]https:pytorch.orgdocsstablegeneratedtorch.polar.html

Here are the abbreviated steps for replacing theta and radius in the rotary forward:

```python

f0 = f0.todevice, dtype # feature extracted during processing
if f0 is not None:
    if f0.dim == 2:
        f0 = f0.squeeze0 
    theta = f0 + self.theta  
else:
    theta = self.theta 


freqs = theta.unsqueeze-1  220.0 * 700 * 
    torch.pow10, torch.linspace0, 2595 * torch.log10torch.tensor1 + 8000700, 
            self.dim  2, device=theta.device, dtype=theta.dtype  2595 - 1  1000


t = torch.arangectx, device=device, dtype=dtype
freqs = t[:, None] * freqs  # dont repeat or use some other method here 

if self.radii and f0 is not None:
    radius = f0.todevice, dtype
    freqs = torch.polarradius.unsqueeze-1, freqs
else:
    radius = torch.ones_likefreqs
    freqs = torch.polarradius, freqs

```

            
            For f0: Phase Accumulation
            
            - F0 as Instantaneous Frequency:  
              Your F0 track F0_{raw,t} represents the fundamental frequency at each time step t.
            
            - Angular Frequency:  
              Convert F0,t to angular frequency omega_t using the relationship:  
              omega_t = 2pi F0,t  
              This gives you radians per second.
            
            - Phase Change:  
              To get the phase change Delta phi_t between the current frame t and the previous frame t-1, you multiply the angular    
              frequency by the time duration of your frame T_{frame}:  
              Delta phi_t = omega_t cdot T_{frame} = 2pi F0{raw,t} cdot T_{frame}.
            
            - Accumulated Phase:  
              The actual phase angle phi_t for the current frame t would then be the accumulated sum of these phase changes:  
              phi_t = phi_{t-1} + Delta phi_t.
            
            ---
            
            Practical Equation for the Rotation Angle phi:
            
            Let:  
            - F0{raw,t} be the fundamental frequency in Hz at time t.  
            - T{frame} be the duration of each time frame in seconds e.g., 10 ms = 0.01 seconds.  
            - phi_0 be an initial phase offset could be 0 or learned.
            
            Then, the rotation angle phi_t for each time step t can be calculated as:  
            [
            phi_t = phi_{t-1} + 2pi F0{raw,t} cdot T_{frame}
            ]
            

```python

    def accumulate_phase(self, f0, t_frame, phi0=0.0):
        omega = 2 * torch.pi * f0
        dphi = omega * t_frame
        phi = torch.cumsum(dphi, dim=0) + phi0
        phi = torch.remainder(phi, 2 * torch.pi) 
        return phi
```

A closer look at whats going on. Here is a slice of the actual radius values for one step
      
      [encoder] [Radius] torch.Size[454] 92.32 [Theta] 10092.01 [f0] torch.Size[454] [Freqs] torch.Size[454, 64] 2.17+1.17j [ctx] 454
      
     [encoder] [Radius] tensor[  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
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
          device='cuda:0'
          
What the Radius Values Tell Us:

1. Speech Structure is Clear
   
         Zeros: Silenceunvoiced segments no pitch
         Non-zero values: Voiced speech segments with pitch
         Pattern: 0.0000 → 283.7590 → 0.0000 → 220.4299

2. Pitch Range is Realistic
   
         Range: ~145-365 Hz
         Typical speech: 80-400 Hz for most speakers
         Model values: 145-365 Hz

3. Temporal Dynamics
   
         Clusters: Pitch values cluster together natural speech
         Transitions: Smooth changes between values
         Silence gaps: Natural pauses in speech

         Silence detection: 0.0000 = no pitch silenceunvoiced
         Pitch extraction: 283.7590 = actual f0 values
         Speech segmentation: Clear boundaries between voicedunvoiced
         
         Realistic values: 145-365 Hz is normal speech range
         Proper structure: Matches natural speech patterns
         Variable radius: Working as intended

The Complex Frequency Result:

      [Freqs] torch.Size[454, 64] 2.17+1.17j



      Magnitude: sqrt2.17² + 1.17² ≈ 2.5
      Phase: atan21.17, 2.17 ≈ 0.49 radians
      
      Variable radius: Each frame has different magnitude



      Silence frames: radius ≈ 0 → freqs ≈ 0
      Voiced frames: radius ≈ 200-300 → freqs ≈ 2-3
      
      Variable attention: Important frames get more attention

      Silence: No acoustic prominence → low radius
      Speech: High acoustic prominence → high radius
      Transitions: Natural pitch changes

----

Approximation methods like using cossin projections or fixed rotation matrices typically assume a unit circle radius=1.0 or only rotate, not scale. When we introduce a variable radius, those approximations break down and can't represent the scaling effect, only the rotation. When using a variable radius, we should use true complex multiplication to get correct results. Approximations that ignore the radius or scale after the rotation don't seem to capture the intended effect, leading to degraded or incorrect representations from my tests so far.

```python

### Do not approximate:
#     radius = radius.unsqueeze-1.expand_asx_rotated[..., ::2]
#     x_rotated[..., ::2] = x_rotated[..., ::2] * radius
#     x_rotated[..., 1::2] = x_rotated[..., 1::2] * radius

### 
    def apply_rotaryx, freqs:
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        if x1.ndim == 2:
            x1 = x1.unsqueeze0
        x1 = x1.float.reshape*x1.shape[:-1], -1, 2.contiguous
        x1 = torch.view_as_complexx1 * freqs
        x1 = torch.view_as_realx1.flatten-2
        x1 = x1.vieworig_shape
        return torch.cat[x1.type_asx, x2], dim=-1
```
This approach respects both the rotation phase and the scaling radius for each tokenhead, so the rotary embedding is applied when the radius varies.

<img width="780" alt="cc4" src="https:github.comuser-attachmentsassets165a3f18-659a-4e2e-a154-a3456b667bae"  >


----
[https:huggingface.coSin2piEcho17tensorboard?params=scalars](https://huggingface.co/Sin2pi/Echo3/tensorboard?params=scalars)

----

This model sometimes uses :

https:github.comsine2piMaxfactor

MaxFactor is a custom PyTorch optimizer with adaptive learning rates and specialized handling for matrix parameters.

** this model deviates in a lot of ways from standard transformer models.


```python
import os
import math
import warnings
import logging
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from tensordict import TensorDict
from typing import Optional, Dict, Union, List, Tuple
import numpy as np
from functools import partial
from datetime import datetime
from tensordict import TensorDict
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from echoutils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class rotary(nn.Module):
    def __init__(self, dims, head, max_ctx=1500, radii=False, debug: List[str] = [], use_pbias=False, axial=False, spec_shape=None):

        super(rotary, self).__init__()
        self.use_pbias = use_pbias
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.radii = radii
        self.debug = debug
        self.counter = 0
        self.last_theta = None
        self.axial = axial

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2), requires_grad=True if use_pbias else False)
        theta = (torch.tensor(10000, device=device, dtype=dtype))
        self.theta = nn.Parameter(theta, requires_grad=True)    
        self.theta_values = []

        if axial and spec_shape is not None:
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('time_freqs', time_freqs)
            
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('freq_freqs', freq_freqs)

    def pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)))
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def theta_freqs(self, theta):
        if theta.dim() == 0:
            theta = theta.unsqueeze(0)
        freq = (theta.unsqueeze(-1) / 220.0) * 700 * (
            torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
                    self.head_dim // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000
        return freq

    def _apply_radii(self, freqs, f0, ctx):
        if self.radii and f0 is not None:
            radius = f0.to(device, dtype)
            L = radius.shape[0]
            if L != ctx:
                F = L / ctx
                idx = torch.arange(ctx, device=f0.device)
                idx = (idx * F).long().clamp(0, L - 1)
                radius = radius[idx]
                return torch.polar(radius.unsqueeze(-1), freqs), radius
            else:
                return torch.polar(radius.unsqueeze(-1), freqs), radius
        else:
            return torch.polar(torch.ones_like(freqs), freqs), None

    def check_f0(self, f0, f0t, ctx):
        if f0 is not None and f0.shape[1] == ctx:
            return f0
        elif f0t is not None and f0t.shape[1] == ctx:
            return f0t
        else:
            return None         

    def axial_freqs(self, ctx):
        if not self.axial:
            return None
        time_frames = self.time_frames
        freq_bins = self.freq_bins
    
        t = torch.arange(ctx, device=device, dtype=dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, self.time_freqs)
        freqs_y = torch.outer(t_y, self.freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    def forward(self, x=None, en=None, f=None, layer=None) -> Tensor:
        ctx=x
        f0 = en.get("f0") if en is not None else None 
        f0t = en.get("f0t") if en is not None else None 

        f0 = self.check_f0(f0, f0t, ctx)
        if f0 is not None:
            if f0.dim() == 2:
                f0 = f0.squeeze(0) 
            theta = f0 + self.theta  
        else:
            theta = self.theta 
        freqs = self.theta_freqs(theta)
        t = torch.arange(ctx, device=device, dtype=dtype)
        freqs = t[:, None] * freqs
        freqs, radius = self._apply_radii(freqs, f0, ctx)

        if self.axial and f == "spectrogram":
            freqs_2d = self.axial_freqs(ctx)
            if freqs_2d is not None:
                return freqs_2d.unsqueeze(0)

        if "radius" in self.debug and self.counter == 10:
            print(f"  [{layer}] [Radius] {radius.shape if radius is not None else None} {radius.mean() if radius is not None else None} [Theta] {theta.mean() if theta is not None else None} [f0] {f0.shape if f0 is not None else None} [Freqs] {freqs.shape} {freqs.mean():.2f} [ctx] {ctx}")
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

class MultiheadA(nn.Module):

    rbf = False
    def __init__(self, dims: int, head: int, rotary_emb: bool = True, 
                 zero_val: float = 1e-7, minz: float = 1e-8, maxz: float = 1e-6, debug: List[str] = [], optim_attn=False, use_pbias=False):
        super(MultiheadA, self).__init__()

        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self.counter = 0
        self.use_pbias = use_pbias

        self.q = nn.Linear(dims, dims).to(device, dtype)
        self.k = nn.Linear(dims, dims, bias=False).to(device, dtype)
        self.v = nn.Linear(dims, dims).to(device, dtype)
        self.o = nn.Linear(dims, dims).to(device, dtype)

        self.pad_token = 0
        self.rotary_emb = rotary_emb
        self.minz = minz
        self.maxz = maxz
        self.zero_val = zero_val
        self.optim_attn = optim_attn        
        self.fzero = nn.Parameter(torch.tensor(zero_val, device=device, dtype=dtype), requires_grad=False)
        
        if rotary_emb:
            self.rope = rotary(
                dims=dims,
                head=head,
                debug=debug,
                radii=False,
                )
        else:
            self.rope = None

    def cos_sim(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out

    def rbf_scores(self, q, k, rbf_sigma=1.0, rbf_ratio=0.0):
        scale = (self.dims // self.head) ** -0.25
        dot_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if rbf_ratio <= 0.0:
            return dot_scores
        q_norm = q.pow(2).sum(dim=-1, keepdim=True)
        k_norm = k.pow(2).sum(dim=-1, keepdim=True)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk
        rbf_scores = torch.exp(-dist_sq / (2 * rbf_sigma**2))
        return (1 - rbf_ratio) * dot_scores + rbf_ratio * rbf_scores
          
    def forward(self, x: Tensor, xa = None, mask = None, en= None, layer = None, f=None) -> tuple:

        x = x.to(device, dtype)
        if xa is not None:
            xa = xa.to(device, dtype)
        scale = (self.dims // self.head) ** -0.25
        
        z = default(xa, x).to(device, dtype)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)

        if self.rotary_emb:   
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            q2 = q.shape[2]
            k2 = k.shape[2]

            q = self.rope.apply_rotary(q, (self.rope(x=q2, en=en, f=f, layer=layer)))
            k = self.rope.apply_rotary(k, (self.rope(x=k2, en=en, f=f, layer=layer)))
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if self.rbf:
            qk = self.rbf_scores(q * scale, k * scale, rbf_sigma=1.0, rbf_ratio=0.3)
        if self.use_pbias:
            pbias = self.rope.pitch_bias(f0 = en.get("f0", None) if en is not None else None) 
            if pbias is not None:
                qk = qk + pbias[:,:,:q2,:q2]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if mask is not None:
            if mask.dim() == 4:
                mask = mask[0, 0]
            mask = mask[:q2, :k2] if xa is not None else mask[:q2, :q2]
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if "multihead" in self.debug and self.counter % 100 == 0:
            print(f"MHA: q={q.shape}, k={k.shape}, v={v.shape} - {qk.shape}, wv shape: {wv.shape}")
        self.counter += 1        
        return self.o(wv), qk

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]

class t_gate(nn.Module):
    def __init__(self, dims, num_types=4, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.gate_projections = nn.ModuleList([
            nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            for _ in range(num_types)])
        self.type_classifier = nn.Sequential(
            Linear(dims, num_types),
            nn.Softmax(dim=-1))
    def forward(self, x):
        if not self.enabled:
            return None
        type_probs = self.type_classifier(x)
        gates = torch.stack([gate(x) for gate in self.gate_projections], dim=-1)
        comb_gate = torch.sum(gates * type_probs.unsqueeze(2), dim=-1)
        return comb_gate

class m_gate(nn.Module):
    def __init__(self, dims, mem_size=64, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.m_key = nn.Parameter(torch.randn(mem_size, dims))
            self.m_val = nn.Parameter(torch.randn(mem_size, 1))
            self.gate_proj = nn.Sequential(Linear(dims, dims//2), nn.SiLU(), Linear(dims//2, 1))
            
    def forward(self, x):
        if not self.enabled:
            return None
        d_gate = torch.sigmoid(self.gate_proj(x))
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, self.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

class c_gate(nn.Module):
    def __init__(self, dims, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.s_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.w_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.p_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.e_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.ph_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.integ = Linear(dims*5, dims)
        
    def forward(self, x, features):
        if not self.enabled:
            return None
        s_feat = features.get("spectrogram", x)
        w_feat = features.get("waveform", x)
        p_feat = features.get("pitch", x)
        e_feat = features.get("envelope", x)
        ph_feat = features.get("phase", x)
        s = self.s_gate(x) * s_feat
        w = self.w_gate(x) * w_feat
        p = self.p_gate(x) * p_feat
        e = self.e_gate(x) * e_feat
        ph = self.ph_gate(x) * ph_feat
        comb = torch.cat([s, w, p, e, ph], dim=-1)
        return self.integ(comb)

class mlp_gate(nn.Module):
    def __init__(self, dims, head, enabled=True, one_shot=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

    def forward(self, x, xa=None, f=None):
        if not self.enabled:
            return None
        return self.gate(x)

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, ctx, dims, head, act, debug: List[str] = [], 
                 tgate=True, mgate=False, cgate=False, mem_size=512, features=None, one_shot=False):
        super().__init__()
        
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.features = features
        self.debug = debug
        self.counter = 0
        self.dropout = 0.01
        self.one_shot = one_shot

        self.blend = nn.Parameter(torch.tensor(0.5)) 
        act_fn = get_activation(act)
        self.attn = MultiheadA(dims, head, rotary_emb=True, debug=debug)
        self.curiosity = curiosity(dims, head)

        if not any([tgate, mgate, cgate]):
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        else:
            self.mlp_gate = None
        
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), act_fn, Linear(mlp, dims))
        
        self.t_gate = t_gate(dims=dims, num_types=4*2, enabled=tgate)
        self.m_gate = m_gate(dims=dims, mem_size=mem_size, enabled=mgate)
        self.c_gate = c_gate(dims=dims, enabled=cgate)
        self.mlp_gate = mlp_gate(dims=dims, head=head, enabled=not any([tgate, mgate, cgate]), one_shot=True)
        
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims)
        self.lnc = RMSNorm(dims)

    def forward(self, x, xa=None, mask=None, en=None, layer=None, f=None) -> Tensor:
 
        b = torch.sigmoid(self.blend)
        ax = x + self.attn(self.lna(x), xa=xa, mask=mask, en=en, layer=layer, f=f)[0]
        bx = b * ax + (1 - b) * x
        cx = self.lnb(bx)
        dx = self.mlp(cx)
        ex = self.t_gate(cx) if not None else self.default(self.m_gate(cx), self.mlp_gate(cx))
        fx = x + ex + dx
        gx = self.lnc(fx)
        return gx
            
class OneShot(nn.Module):
    def __init__(self, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        self.head  = head
        self.hdim  = dims // head
        self.scale = scale                      
        self.q_proj = Linear(dims, dims)
        self.k_proj = Linear(dims, dims)

    def forward(self, x: Tensor, guide: Tensor, f=None) -> Tensor | None:
        B, Q, _ = x.shape
        K       = guide.size(1)
        q = self.q_proj(x ).view(B, Q, self.head, self.hdim).transpose(1,2)
        k = self.k_proj(guide).view(B, K, self.head, self.hdim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.hdim)
        return bias

class curiosity(nn.Module):
    def __init__(self, d, h, bias=True):
        super().__init__()
        self.h  = h
        self.dh = d // h
        self.qkv = nn.Linear(d, d * 3, bias=bias)
        self.qkv_aux = nn.Linear(d, d * 3, bias=bias)
        self.o  = nn.Linear(d, d, bias=bias)
        self.g  = nn.Parameter(torch.zeros(h))

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x, xa, mask=None):
        q, k, v   = self.qkv(x).chunk(3, -1)
        qa, ka, va = self.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(self.split, (q, k, v))
        qa, ka, va = map(self.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / self.dh**0.5
        dots_aux  = (q @ ka.transpose(-2, -1)) / self.dh**0.5
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)
        h_main = p  @ v
        h_aux  = pa @ va
        g = torch.sigmoid(self.g).view(1, -1, 1, 1)
        out = self.merge(h_main * (1 - g) + h_aux * g)
        return self.o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, dims, ctx):
        super(PositionalEncoding, self).__init__()
        self.dims = dims
        self.ctx = ctx
        self.pe = self.get_positional_encoding(max_ctx=ctx)

    def get_positional_encoding(self, max_ctx):
        pe = torch.zeros(max_ctx, self.dims)
        position = torch.arange(0, max_ctx, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dims, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.to(device)

    def forward(self, x):
        ctx = x.size(1)
        pe = self.pe[:, :ctx, :]
        x = x * math.sqrt(self.dims)
        x = x + pe
        return x

class FEncoder(nn.Module):
    def __init__(self, mels, dims, head, layer, kernel_size, act, stride=1, use_rope=False, spec_shape=None, debug=[]):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.dims = dims
        self.debug = debug
        act_fn = get_activation(act)
        self.attend_pitch = False

        if self.attend_pitch:
            self.q, self.k, self.v, self.o, self.scale = qkv_init(dims, head)
            self.mlp = nn.Sequential(
                nn.Linear(dims, dims),
                nn.ReLU(),
                nn.Linear(dims, dims),
            )
        else:
            self.q, self.k, self.v, self.o, self.scale = None, None, None, None, None
            self.mlp = None

        self.encoder = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, en=None, f=None, layer="audio"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, en=en, f=f, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)

        return x

    def forward(self, x: Tensor, en=None, f=None, layer = None):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, en=en, f=f, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)

        if self.mlp is not None:
            x = self.mlp(x)

        if self.attend_pitch:
            xa = en["input_ids"]
            if xa is not None:
                q, k, v = create_qkv(self.q, self.k, self.v, x=xa, xa=x, head=self.head)
                out, _ = calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True)
                out = self.o(out)
                x = x + out

        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x

class WEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False, debug=[], spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        self.debug = debug
        act_fn = get_activation(act)
        self.target_length = None
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)
            
        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, en=None, f=None, layer="audio"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, en=en, f=f, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x: Tensor, en= None, f=None, layer = None):
        x = self.encoder(x).permute(0, 2, 1)
        if self.target_length and x.shape[1] != self.target_length:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.target_length).transpose(1, 2)
        if self.use_rope:
            x = self.apply_rope_to_features(x, en=en, f=f, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.ln(x)
        print(f"X: {x.shape} {f}") if "encoder" in self.debug else None
        return self.norm(x)

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=True, debug=[], one_shot=False, spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dims = dims
        self.dropout = 0.01
        self.use_rope = use_rope
        self.debug = debug
        act_fn = get_activation(act)
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=7, stride=1, padding=3), act_fn,
            Conv1d(dims, dims, kernel_size=5, stride=1, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if use_rope:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        
        self.norm = RMSNorm(dims)
        
    def rope_to_feature(self, x, en=None, f="pitch", layer="PEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, en=en, f=f, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x: Tensor, en= None, f="pitch", layer="PEncoder"):

        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.rope_to_feature(x, en=en, f=f, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        print(f"X: {x.shape} {f}") if "PEncoder" in self.debug else None
        return x

class theBridge(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, 
                debug: List[str], features: List[str], act: str = "gelu"): 
        super(theBridge, self).__init__()
    
        tgate = True
        mgate = False
        cgate = False

        self.debug = debug
        self.counter = 0
        self.dropout = 0.01 
        self.features = features
        self.do_blend = "no_blend" not in self.debug
        self.sequential = "sequential" in self.debug
        self.layer = layer

        self.token = nn.Embedding(vocab, dims, device=device, dtype=dtype)
        self.positional = nn.Parameter(torch.empty(ctx, dims, device=device, dtype=dtype), requires_grad=True)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)
        self.norm = RMSNorm(dims)
        self.sinusoid_pos = lambda length, dims, max_tscale: sinusoids(length, dims, 10000)
        self.rotary = rotary(dims=dims,  head=head, debug=debug, radii=False)

        with torch.no_grad():
            self.token.weight[0].zero_()
        
        act_fn = get_activation(act)
        if features == ["spectrogram", "waveform", "pitch"]:
            cgate=True
        else:
            cgate = False
    
        self.blockA = nn.ModuleDict()
        self.blockA["waveform"] = nn.ModuleList(
            [WEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features) 
            for _ in range(layer)] if "waveform" in features else None) 

        for feature_type in ["spectrogram", "aperiodic", "harmonic"]:
            if feature_type in features:
                self.blockA[feature_type] = nn.ModuleList(
                    [FEncoder(mels=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
                    [Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features) for _ in range(layer)] if feature_type in features else None)
            else:
                self.blockA[feature_type] = None

        for feature_type in ["pitch", "phase"]:
            if feature_type in features:
                self.blockA[feature_type] = nn.ModuleList(
                    [PEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act_fn)] + 
                    [Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features) for _ in range(layer)] if feature_type in features else None)
            else:
                self.blockA[feature_type] = None

        self.blockB = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features)
            for _ in range(layer)])

        self.modal = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features)
            for _ in range(layer)]) 

        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0) 
        self.register_buffer("mask", mask, persistent=False)

        self.norm = RMSNorm(dims)

    def forward(self, x, xa, en, f, sequential=False) -> Tensor:
        mask = self.mask[:x.shape[1], :x.shape[1]] 
        x = self.token(x.long()) + self.positional[:x.shape[1]]

        out = {}
        out["input_ids"] = x
        out.update(en)

        for b in chain(self.blockA[f] or []):
            xa = b(x=xa, en=out, f=f, layer="en")

        for b in chain(self.blockB or []):
            x = b(x=x, xa=None, mask=mask, en=out, f=f, layer="dec")
            y = b(x, xa=xa, mask=None, en=out, f=f, layer="cross")
            if sequential:
                x = y
            else:
                a = torch.sigmoid(self.blend)
                x = a * y + (1 - a) * x 
        for b in self.modal:
            xc = b(x=torch.cat([x, xa], dim=1), xa=None, mask=None, en=out, f=f, layer="modal")    
            xm = b(x=xc[:, :x.shape[1]], xa=xc[:, x.shape[1]:], mask=None, en=out, f=f, layer="modal")
            if sequential:
                x = xm
            else:
                a = torch.sigmoid(self.blend)
                x = a * x + (1 - a) * xm

        if self.counter < 1 and "encoder" in self.debug:      
                shapes = {k: v.shape for k, v in en.items()}
                print(f"Step {self.counter}: mode: {list(en.keys()) }: shapes: {shapes}")
        self.counter += 1

        x = self.norm(x)
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()

        return x
   
class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        
        self.processor = theBridge(
            vocab=param.vocab,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            features=param.features,
            act=param.act,
            debug=param.debug,
            )       
        
    def forward(self,
        labels=None,
        input_ids=None,
        waveform: Optional[torch.Tensor]=None,
        spectrogram: Optional[torch.Tensor]=None,
        pitch: Optional[torch.Tensor]=None,
        f0: Optional[torch.Tensor]=None,
        f0t: Optional[torch.Tensor]=None,
        harmonic: Optional[torch.Tensor]=None,
        aperiodic: Optional[torch.Tensor]=None,
        phase: Optional[torch.Tensor]=None,
        ) -> Dict[str, Optional[torch.Tensor]]:

        en= TensorDict(batch_size=[1], device=self.device, dtype=self.dtype)

        en= {}
        if f0 is not None:
            en["f0"] = f0
        if f0t is not None:
            en["f0t"] = f0t
        if harmonic is not None:
            en["harmonic"] = harmonic
        if aperiodic is not None:
            en["aperiodic"] = aperiodic
        if phase is not None:
            en["phase"] = phase
        if pitch is not None:
            en["pitch"] = pitch
        if waveform is not None:
            en["waveform"] = waveform
        if spectrogram is not None:
            en["spectrogram"] = spectrogram

        x = input_ids
        for f, xa in en.items():

            logits = self.processor(x, xa, en, f)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
        
        return {"logits": logits, "loss": loss} 

    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype
```
