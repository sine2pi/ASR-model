## Zero-Value Processing in Speech Attention

### The Significance of Zeros in Audio Processing

In log-mel spectrograms, zero or near-zero values represent critical information:
- Silent regions between speech
- Low-amplitude acoustic events
- Sub-threshold background environments

The model also extracts f0 contour and energy contours.
We use them as features and we then inject the frequency into the rotary at the same time as the sample hits the multihead and encoder. Frerquency is a learnable parameter.

### Multiplicative Soft Masking: Technical Implementation

```python
token_ids = k[:, :, :, 0].to(q.device, q.dtype)
scaled_zero = torch.ones_like(token_ids).to(q.device, q.dtype)
scaled_zero[token_ids == 0] = 0.000001
scaling_factors = scaled_mask.unsqueeze(0) * scaled_zero.unsqueeze(-2).expand(qk.shape)
```

1. **Semantic Preservation of Silence**: Unlike conventional `-inf` masking that eliminates attention, this approach maintains minimal attention flow (0.000001) for silence tokens, preserving their semantic value.

2. **Prosodic Pattern Recognition**: By allowing minimal attention to silent regions, the model can learn timing, rhythm, and prosodic features critical for speech understanding.

3. **Cross-Modal Representation Unification**: Using identical attention mechanisms for both audio silence and text padding creates a unified approach across modalities, simplifying the architecture.

4. **Training Stability**: This multiplicative masking approach maintains gradient flow through all positions, potentially improving convergence stability.

5. **Natural Boundary Learning**: By processing silence regions with reduced but non-zero attention, the model learns natural speech boundaries without requiring explicit BOS/EOS tokens.

- Force the model to not rely on start tokens
- Create a cleaner boundary between "meaningful" and "non-meaningful" tokens
- Simplify the attention mechanism's behavior

This explicitly creates a system where:
- Zero-valued tokens get minimal attention (multiplied by 0.000001)
- All non-zero tokens get normal attention weights

1. **All padding uses 0** - Which gets minimal attention (good)
2. **All start tokens use 0** - Also gets minimal attention (intentional)
3. **All meaningful content uses non-zero tokens** - Gets normal attention (good)

## Potential Benefits

This approach could:
- Force the model to not rely on start tokens
- Create a cleaner boundary between "meaningful" and "non-meaningful" tokens
- Simplify the attention mechanism's behavior

The critical question is: Does a model need the start token to initialize proper generation? In most transformer models, the first token (often a start token) provides essential context for generating the first meaningful token.

This design intentionally minimizes the start token's influence.. if the model performs well, then this could be a novel and interesting approach to sequence generation.

## Adaptive Audio Feature Fusion

The model incorporates a learnable parameter with sigmoid activation to adaptively blend waveform and spectrogram encodings. Initial findings demonstrate significant WER (Word Error Rate) reduction compared to single-representation approaches, with minimal computational overhead increase.

This adaptive fusion addresses the longstanding waveform-vs-spectrogram debate in ASR by allowing the model to determine the optimal representation mix for different acoustic contexts. While feature fusion has been explored in research settings, this learnable parameter approach provides an elegant solution that maintains computational efficiency.



```python
# class MultiheadB(nn.Module):
#     ...
#     def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None, decoder: Optional[bool] = False, f0: Optional[Tensor] = None):
#         ...
#         # Instead of qf = self.rope(q.size(1)), use f0 if provided
#         if f0 is not None:
#             # f0 should be shape (batch, seq_len)
#             qf = self.rope(f0)
#             kf = self.rope(f0)
#         else:
#             qf = self.rope(q.size(1))
#             kf = self.rope(k.size(1))
#         ...
#         q = self.rope.apply_rotary(q, qf)
#         k = self.rope.apply_rotary(k, kf)

# class Echo(nn.Module):
#     ...
#     def forward(self, ..., parosody=None, ...):
#         ...
#         if parosody is not None:
#             to_encoderA["f0"] = parosody  # or batch["f0_contour"]
#         encoder_output = self.encoderA(**to_encoderA)
#         ...

```


```python


    # --- Parosody ---
    if extract_parosody:
        hop_length = extractor.hop_length
        win_length = 256
        wav = wav.unsqueeze(0) if wav.ndim == 1 else wav  
        device = wav.device if wav.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav = wav.to(device)
        f0 = torchcrepe.predict(wav, sampling_rate, model="tiny")
        # f0 = torchyin.estimate(wav, sample_rate=sampling_rate, frame_stride=hop_length / sampling_rate)
        f0 = torch.nan_to_num(f0, nan=0.0)
        f0 = (f0 - f0.mean()) / (f0.std() + 1e-8)
        num_frames = f0.shape[-1]

        energies = []
        for i in range(num_frames):
            start = int(i * hop_length)
            end = int(start + win_length)
            frame = wav[0, start:end]
            if frame.numel() == 0:
                rms = 0.0
                power = 0.0
            else:
                rms = torch.sqrt(torch.mean(frame ** 2) + 1e-8).item()
                hann = torch.hann_window(frame.numel(), device=frame.device)
                windowed = frame * hann
                power = torch.sum(windowed ** 2).item()
            energies.append([rms, power])
        energy = torch.tensor(energies, dtype=torch.float32, device=f0.device).T  # shape: (2, num_frames)

        max_len = max(f0.shape[-1], energy.shape[-1])
        if f0.shape[-1] < max_len:
            f0 = F.pad(f0, (0, max_len - f0.shape[-1]), value=0.0)
        if energy.shape[-1] < max_len:
            energy = F.pad(energy, (0, max_len - energy.shape[-1]), value=0.0)

        blend = torch.sigmoid(torch.tensor(0.5, device=f0.device))
        parosody = blend * f0 + (1 - blend) * energy
        target_len = current_features.shape[-1]
        parosody = match_length(parosody, target_len)
        batch["parosody"] = parosody#.cpu()
```
The F0 contour and the energy contour can be used together to analyze the prosody of speech, including intonation and loudness. The F0 contour follows the lowest frequency with the most energy, which is indicated by bright colors towards the bottom of the image. 
In summary: F0 contour represents pitch variation over time, while energy contour represents sound intensity across frequencies over time. They both play a crucial role in understanding speech prosody and can be used together to analyze emotional expressions and grammatical structures within speech. 



```python

import os
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import numpy as np
from typing import Optional, Dict, Union, List, Tuple
from functools import partial
import gzip
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor, WhisperTokenizer
import evaluate
import transformers
from dataclasses import dataclass
from itertools import chain
import sys
import random

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0")
dtype = torch.float32

torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}

extractor = None
tokenizer = None
optimizer = None 
scheduler = None
model = None

def set_model(model_):
    global model
    if isinstance(model_, str):
        model = torch.hub.load(model_, 'model')
    elif isinstance(model_, nn.Module):
        model = model_
    else:
        raise ValueError(f"Invalid model type: {type(model_)}. Use a string or nn.Module.")
    model.to(device)
    model.eval()

def set_device_and_dtype(device_: str, dtype_: str):
    global device, dtype
    if device_ == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif device_ == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Invalid device: {device_}. Use 'cuda' or 'cpu'.")
    if dtype_ == "float32":
        dtype = torch.float32
    elif dtype_ == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype_}. Use 'float32' or 'float16'.")
    torch.set_default_dtype(dtype)

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_extractor_and_tokenizer(extractor_, tokenizer_):
    global extractor, tokenizer
    extractor = extractor_
    tokenizer = tokenizer_

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    decoder_idx: int
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    encoder_idx: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    act: str 
    debug: bool
    cross_attention: bool

    def __post_init__(self):
        self.vocab = int(self.vocab)
        self.text_ctx = int(self.text_ctx)
        self.text_dims = int(self.text_dims)
        self.text_head = int(self.text_head)
        self.decoder_idx = int(self.decoder_idx)
        self.mels = int(self.mels)
        self.audio_ctx = int(self.audio_ctx)
        self.audio_dims = int(self.audio_dims)
        self.audio_head = int(self.audio_head)
        self.encoder_idx = int(self.encoder_idx)
        self.pad_token_id = int(self.pad_token_id)
        self.eos_token_id = int(self.eos_token_id)
        self.decoder_start_token_id = int(self.decoder_start_token_id)
        self.act = str(self.act)
        self.debug = bool(self.debug)
        self.cross_attention = bool(self.cross_attention)

    def __repr__(self):
        return f"Dimensions(vocab={self.vocab}, text_ctx={self.text_ctx}, text_dims={self.text_dims}, text_head={self.text_head}, decoder_idx={self.decoder_idx}, mels={self.mels}, audio_ctx={self.audio_ctx}, audio_dims={self.audio_dims}, audio_head={self.audio_head}, encoder_idx={self.encoder_idx}, pad_token_id={self.pad_token_id}, eos_token_id={self.eos_token_id}, decoder_start_token_id={self.decoder_start_token_id}, act='{self.act}', debug={self.debug}, cross_attention={self.cross_attention})"
    
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print(f"{exc_type.__name__}: {exc_value}")
    
    tb = exc_traceback
    while tb.tb_next:
        tb = tb.tb_next
    
    frame = tb.tb_frame
    filename = frame.f_code.co_filename
    lineno = tb.tb_lineno
    func_name = frame.f_code.co_name
    print(f"Error occurred in file '{filename}', line {lineno}, in function '{func_name}'")

def get_tracked_parameters(model, param_paths=None):
    if param_paths is None:
        param_paths = {
            "sw": "encoder.sw",
        }
    result = {}
    for name, path in param_paths.items():
        parts = path.split('.')
        param = model
        for part in parts:
            param = getattr(param, part)
        try:
            if isinstance(param, torch.Tensor):
                if param.numel() == 1:
                    result[name] = param if not param.requires_grad else param
                else:
                    result[name] = param.sum()
            else:
                result[name] = float(param) if hasattr(param, "__float__") else str(param)
        except Exception as e:
            result[name] = f"Error: {str(e)}"
    return result

def plot_waveform_and_spectrogram(x=None, w=None, sample_idx=0, sr=16000, title="Waveform and Spectrogram"):
    if x is not None and w is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
    
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        axs[0].plot(t, w_np, color="tab:blue")
        axs[0].set_title("Waveform")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[1].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma")
        axs[1].set_title("Spectrogram")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Mel Bin")
        plt.tight_layout()
        plt.show()

    elif x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.imshow(x_np.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Spectrogram")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel Bin")
        plt.tight_layout()
        plt.show()     

    elif w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(t, w_np, color="tab:blue")
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("No data to plot. Please provide at least one input tensor.")

def shift_with_zeros(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    input_ids = labels.new_zeros(labels.shape)
    input_ids[:, 1:] = labels[:, :-1].clone()   
    return input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.RMSNorm):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], eps = 1e-8, elementwise_affine = True, device=torch.device(device="cuda:0"), dtype=torch.float32):
        tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}
        if isinstance(dims, int):
            self.normalized_shape = (dims,)  
        else:
            self.normalized_shape = tuple(dims) 
        super().__init__(normalized_shape=dims, eps=eps, elementwise_affine=elementwise_affine)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **tox))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
            
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.device, x.dtype), None if self.bias is None else self.bias.to(x.device, x.dtype))
    
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)      

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class Rotary(nn.Module):
    def __init__(self, dims, ctx=1500, learned_freq=False, variable_radius=False, learned_radius=False, use_xpos=False, xpos_scale_base=1.0, debug=False):
        super().__init__()

        self._counter = 0
        self.debug = debug
        self.dims = dims
        max_ctx = ctx
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius       
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
            requires_grad=learned_freq
        )
        
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius
            )
        
        if use_xpos:
            scale = (torch.arange(0, dims, 2) + 0.4 * dims) / (1.4 * dims)
            self.scale_base = xpos_scale_base
            self.register_buffer('scale', scale, persistent=False)

        if use_xpos:
            self.register_buffer('cached_scales', torch.zeros(ctx, dims), persistent=False)
            self.cached_scales_ctx = 0

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        
    def get_seq_pos(self, ctx, device, dtype, offset=0):
        return (torch.arange(ctx, device=device, dtype=dtype) + offset) / self.interpolate_Factor
    
    def get_scale(self, t, ctx=None, offset=0):
        from einops import repeat
        assert self.use_xpos
        should_cache = (self.cache_if_possible and
            exists(ctx) and (offset + ctx) <= self.max_ctx)
        
        if (should_cache and exists(self.cached_scales) and
            (ctx + offset) <= self.cached_scales_ctx):
            return self.cached_scales[offset:(offset + ctx)]
            
        power = (t - len(t) // 2) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = repeat(scale, 'n d -> n (d r)', r=2)
        
        if should_cache and offset == 0:
            self.cached_scales[:ctx] = scale.detach()
            self.cached_scales_ctx = ctx
            
        return scale

    def forward(self, x = None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=self.inv_freq.device).float()
        else:
            t = x.float().to(self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)

        if self.debug:
            if self._counter == 1:
                print(f'ROTA -- freqs: {freqs.shape}, x: {x.shape if x is not None else None}', freqs.shape, x.shape)
            self._counter += 1

        return freqs
    
    def _reshape_for_multihead(self, freqs, head, head_dim):
        ctx = freqs.shape[0]
        complex_per_head = head_dim // 2
        if complex_per_head * head > freqs.shape[1]:
            freqs = freqs[:, :complex_per_head * head]
        elif complex_per_head * head < freqs.shape[1]:
            padding = torch.zeros(
                (ctx, complex_per_head * head - freqs.shape[1]), 
                device=freqs.device, 
                dtype=freqs.dtype
            )
            freqs = torch.cat([freqs, padding], dim=1)
        freqs = freqs.view(ctx, head, complex_per_head)
        return freqs.permute(2, 1, 0, 2).unsqueeze(0)

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
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)

            return torch.cat([x1.type_as(x), x2], dim=-1)

class MultiheadA(nn.Module):
    def __init__(self, dims: int, head: int, debug):
        super().__init__()
        self._counter = 0
        self.debug = debug
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.out = Linear(dims, dims)

        self.Rotary = Rotary(dims=self.head_dim, learned_freq=True)
        self.Factor = nn.Parameter(torch.tensor(0.00005))

    def forward(self, x: Tensor, xa: Optional[Tensor] = None,  mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None, decoder: bool = False):

        batch, ctx, dims = x.size()
        if xa is not None:
            Batch, Ctx, D = xa.size()
        
        head_dim = self.head_dim
        head = self.head
        freq = self.Rotary(ctx)
        scale = (dims // head) ** -0.25

        q = self.q(x)

        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x)
            v = self.v(x)
            if kv_cache is not None:
                kv_cache[self.k] = k
                kv_cache[self.v] = v
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]

        q = self.Rotary.apply_rotary(q, freq)
        k = self.Rotary.apply_rotary(k, freq)

        q = q.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous() 
        k = k.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()  
        v = v.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous() 

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        token_ids = k[:, :, :, 0]
        scaled_zero = torch.ones_like(token_ids)
        zero_Factor = torch.clamp(F.softplus(self.Factor), min=0.0, max=0.001)
        scaled_zero[token_ids.float() == 0] = zero_Factor.to(q.device, q.dtype)
        
        if mask is not None:
            qk = qk * mask.unsqueeze(0).unsqueeze(0)
        qk = qk * scaled_zero.unsqueeze(-2)
        
        qk = qk.float()
        w = F.softmax(qk, dim=-1)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if self._counter < 1 and self.debug: 
            print("---------")
            print("MULTIHEAD (First Call)")
            print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}, w: {w.shape}, wv: {wv.shape}, qk: {qk.shape}")
            print(f"Input x shape (first call): {x.shape}")
            print(f"Input xa shape (first call): {xa.shape if xa is not None else None}")
            print(f"Mask shape (first call): {mask.shape if mask is not None else None}")

        if self._counter % 100 == 0 and self.debug and mask is not None: 
            print(f"Counter: {self._counter} (Periodic Log)")
            print(f"Input x shape (periodic): {x.shape}")
            print(f"Input xa shape (periodic): {xa.shape if xa is not None else None}")
            print(f"Mask shape (periodic): {mask.shape}")

        self._counter += 1
        return self.out(wv), qk.detach()

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, act: str = "gelu", cross_attention: bool = False, debug: bool = False):
        super().__init__()
        
        self._counter = 0
        self.dropout = 0.1
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.cross_attention = cross_attention
        self.debug = debug

        self.blend_xa = nn.Parameter(torch.tensor(0.5))
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head, debug)
        self.attnb = (MultiheadA(dims, head, debug) if cross_attention else None)
            
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))
        self.lna = RMSNorm(dims)    
        self.lnb = RMSNorm(dims) if cross_attention else None
        self.lnc = RMSNorm(dims) 

    def forward(self, x, xa=None, mask=None, kv_cache=None, decoder=False):

        r = x
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache, decoder=decoder)[0]
        if self.attnb and xa is not None:
            cross_out = self.attnb(self.lnb(x), xa, kv_cache=kv_cache, decoder=decoder)[0]
            blend = torch.sigmoid(self.blend_xa)
            x = blend * x + (1 - blend) * cross_out
        x = x + self.mlp(self.lnc(x))
        x = x + r

        if self._counter < 1 and self.debug: 
            print("--------")
            print("RESIDUAL")
            print(f"Is decoder: {decoder}")
            print(f"kv_cache: {kv_cache is not None}")
            print(f"Input x shape: {x.shape}")
            print(f"Input xa shape: {xa.shape if xa is not None else None}")
            print(f"Mask: {mask.shape if mask is not None else None}")
                
        self._counter += 1
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class AudioEncoder(nn.Module):
    def __init__(self, mels: int, layer: int, dims: int, head: int, ctx: int, act: str = "gelu", debug: bool = False):
        super().__init__()
        self._counter = 0
        self.debug = debug
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.dropout = 0.1

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.sw = nn.Parameter(torch.tensor(0.5))
        self.ln_enc = RMSNorm(dims, **tox)
        self.positional_embedding = lambda length: sinusoids(length, dims)

        self.se = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, padding=1), self.act,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),
            Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
            nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1))
        
        self.we = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5), # Allows variable length output
            nn.GELU(), nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2), # Allows variable length output
            nn.GELU())

        self.blockA = (nn.ModuleList([
            Residual(dims=dims, head=head, act=act, cross_attention=False, debug=debug) for _ in range(layer)]) if layer > 0 else None)
            
    def forward(self, x, w, decoder=False) -> Tensor:
        """
        batch, ctx, dims = B, L, D 
        x is spectrogram (B, Mels, L)
        w is waveform (B, 1, L)
        """
        x = x.to(device=device) if x is not None else None
        w = w.to(device=device) if w is not None else None
        
        blend = torch.sigmoid(self.sw) 

        if self._counter < 1 and self.debug:
            plot_waveform_and_spectrogram(x=x, w=w)
            print(f"Initial Spectrogram tensor shape: {x.shape if x is not None else None}, Initial Waveform tensor shape: {w.shape if w is not None else None}")

        if x is not None:
            x = self.se(x).permute(0, 2, 1) # (B, L, D)
            x = x + self.positional_embedding(x.shape[1]).to(x.device, x.dtype)

        if w is not None: # self.we output shape: (B, D, L)           
            w = self.we(w).permute(0, 2, 1) # (B, L, D)

        if x is not None:
            if w is not None:
                if w.shape[1] != x.shape[1]:
                    w = w.permute(0, 2, 1) # (B, L, D) -> (B, D, L)
                    w = F.interpolate(w, size=x.shape[1], mode='linear', align_corners=False) #  w_L to x_L 
                    w = w.permute(0, 2, 1) # (B, D, L) -> (B, L, D)
                else:
                    w = w
                w = w + self.positional_embedding(w.shape[1]).to(w.device, w.dtype)
                x = blend * x + (1 - blend) * w
            else:
                x = x
        elif w is not None:
            w = w + self.positional_embedding(w.shape[1]).to(w.device, w.dtype)
            x = w
        else:
            raise ValueError("You have to provide either x (spectrogram) or w (waveform)")

        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for block in chain(self.blockA or []):
            x = block(x, decoder=decoder) 

        if self._counter < 1 and self.debug: 
            print("-------")
            print("ENCODER")
            print(f"Features to Residual Blocks shape: {x.shape}")     
            print(f"Original input x shape: {x.shape if x is not None else None}")
            print(f"Original input w shape: {w.shape if w is not None else None}")
            print(f"Positional embedding for a length of {x.shape[1]} would be: {self.positional_embedding(x.shape[1]).shape}")
            
        self._counter += 1
        return self.ln_enc(x)

class TextDecoder(nn.Module):
    def __init__(self, vocab, ctx, dims, head, layer, cross_attention=False, rotary=False, debug=False):
        super().__init__()
        self._counter = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self.dropout = 0.1
        
        self.Rotary = Rotary(dims=dims, ctx=ctx, learned_freq=False, variable_radius=False, learned_radius=False)    
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        self.ln_dec = RMSNorm(dims=dims)
    
        self.blockA = (nn.ModuleList([
            Residual(dims=dims, head=head, act="gelu", cross_attention=cross_attention, debug=debug) for _ in range(layer)]) if layer > 0 else None)

    def forward(self, x, xa, kv_cache=None, decoder=True) -> Tensor:

        """
        input:
        x : torch.LongTensor, shape = (batch, ctx)  the text tokens
        xa : torch.Tensor, shape = (batch, audio_ctx, audio_dims) the encoded audio features to be attended on
        output:
        logits : torch.FloatTensor, shape = (batch, ctx, vocab) the logits for each token in the vocabulary
        x : torch.FloatTensor, shape = (batch, ctx, dims) the text features
        xa : torch.FloatTensor, shape = (batch, audio_ctx, dims) the encoded audio features to be attended on
        """

        x = x.to(device=device)
        if xa is not None:
            xa = xa.to(device=device)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]]) # positional_embedding = (ctx, dims)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        ctx = x.shape[1]
        freqs = self.Rotary(ctx)
        x = self.Rotary.apply_rotary(x, freqs)

        mask = torch.where((torch.triu(torch.ones(ctx, ctx), diagonal=1)) == 1, torch.tensor(0.0), torch.tensor(1.0)).to(x.device, x.dtype)

        for block in chain(self.blockA or []):
            x = block(x, xa=xa, mask=mask, kv_cache=kv_cache, decoder=decoder)
        
        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        if self._counter < 1 and self.debug: 
            print("-------")
            print("DECODER")
            print(f"Token embedding shape: {self.token_embedding.weight.shape}")
            print(f"Positional embedding shape: {self.positional_embedding.shape}")
            print(f"Rotary frequency shape: {freqs.shape}")
            print(f"Input x shape: {x.shape}")
            print(f"Input xa shape: {xa.shape if xa is not None else None}")
            print(f"Mask: {mask.shape if mask is not None else None}")
            print(f"CTX: {ctx}, Offset: {offset}")
            print(f"Logits shape: {logits.shape}")

        self._counter += 1
        return logits

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param  

        self.shared = nn.ModuleDict({
            "rotary": Rotary(dims=param.audio_dims // param.audio_head),       
            "rotary_encoder": Rotary(dims=param.audio_dims // param.audio_head),
            "rotary_decoder": Rotary(dims=param.text_dims // param.text_head), 

        })

        self.param_tracking_paths = {
            "RotationA": "encoder.blockA.0.attna.Rotary.inv_freq",
            "RotationB": "decoder.Rotary.inv_freq",
            "Silence": "encoder.blockA.0.attna.Factor",
        }
        
        self.encoder = AudioEncoder(
            debug=param.debug,
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layer=param.encoder_idx,
            act=param.act,
        )

        self.decoder = TextDecoder(
            debug=param.debug,
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.decoder_idx,
            cross_attention=param.cross_attention,

            # rotary=param.rotary,
            # Factor=param.Factor,
            # rotary_config=param.rotary_config,
        )

        all_head = torch.zeros(self.param.decoder_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.decoder_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.decoder_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, input_features: torch.Tensor):
        return self.encoder(input_features)

    def logits(self,input_ids: torch.Tensor, encoder_output: torch.Tensor):
        return self.decoder(input_ids, encoder_output)
    
    def forward(self, 
        decoder_input_ids=None,
        labels=None,
        input_features: torch.Tensor=None, 
        waveform: Optional[torch.Tensor]=None,
        input_ids=None, 
    ) -> Dict[str, torch.Tensor]:

        if labels is not None:
            if labels.shape[1] > self.param.text_ctx:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.param.text_ctx} tokens."
                )
            if input_ids is None:
                input_ids = shift_with_zeros(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id
                ).to('cuda')
            decoder_input_ids = input_ids
            if input_ids.shape[1] > self.param.text_ctx:
                raise ValueError(
                    f"Input IDs' sequence length {input_ids.shape[1]} cannot exceed the maximum allowed length of {self.param.text_ctx} tokens."
                )

        if input_features is not None:    
            if waveform is not None:
                encoder_output = self.encoder(x=input_features, w=waveform)
            else:
                encoder_output = self.encoder(x=input_features, w=None)
        elif waveform is not None:
            encoder_output = self.encoder(x=None, w=waveform)
        else:
            raise ValueError("You have to provide either input_features or waveform")
        logits = self.decoder(input_ids, encoder_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
            
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, "Residual": 0,
                            "Multihead": 0, "MultiheadA": 0, "MultiheadB": 0, "MultiheadC": 0}

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                self.init_counts["LayerNorm"] += 1
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            elif isinstance(module, MultiheadA):
                self.init_counts["MultiheadA"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, SEBlock):
                nn.init.ones_(module.fc[0].weight)
                nn.init.zeros_(module.fc[0].bias)
                nn.init.ones_(module.fc[2].weight)
                nn.init.zeros_(module.fc[2].bias)
                self.init_counts["SEBlock"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                nn.init.xavier_uniform_(module.se[0].weight)
                nn.init.zeros_(module.se[0].bias)
                nn.init.xavier_uniform_(module.se[2].weight)
                nn.init.zeros_(module.se[2].bias)
                nn.init.xavier_uniform_(module.se[4].weight)
                nn.init.zeros_(module.se[4].bias)
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1    
                                                 
    def init_weights(self):
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            print(f"{module_type}: {count}")
 
metric = evaluate.load(path="wer")

@dataclass
class DataCollator:

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        global extractor, tokenizer
        decoder_start_token_id = tokenizer.bos_token_id
        pad_token_id = tokenizer.pad_token_id
        input_feature_pad_value = 0.0
        waveform_pad_value = 0.0

        if decoder_start_token_id is None:
            raise ValueError("The tokenizer does not have a bos_token_id. Please set it manually.")        
        batch = {}

        if "input_features" in features[0] and features[0]["input_features"] is not None:
            input_features_list = [f["input_features"] for f in features]
            max_len_feat = max(f.shape[1] for f in input_features_list)
            
            padded_input_features = []
            for feat in input_features_list:
                num_mels, current_len = feat.shape
                padding_needed = max_len_feat - current_len
                if padding_needed > 0:
                    padded_feat = F.pad(feat, (0, padding_needed), mode='constant', value=input_feature_pad_value)
                else:
                    padded_feat = feat
                padded_input_features.append(padded_feat)
            batch["input_features"] = torch.stack(padded_input_features)

        if "waveform" in features[0] and features[0]["waveform"] is not None:
            waveform_list = [f["waveform"] for f in features]
            max_len_wav = max(w.shape[1] for w in waveform_list)

            padded_waveforms = []
            for wav in waveform_list:
                _, current_len = wav.shape
                padding_needed = max_len_wav - current_len
                if padding_needed > 0:
                    padded_wav = F.pad(wav, (0, padding_needed), mode='constant', value=waveform_pad_value)
                else:
                    padded_wav = wav
                padded_waveforms.append(padded_wav)
            batch["waveform"] = torch.stack(padded_waveforms)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt", padding="longest", pad_to_multiple_of=None)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), 0)
        if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch["input_ids"] = shift_with_zeros(labels, pad_token_id, decoder_start_token_id)
        return batch

def prepare_dataset(batch, input_features=True, waveform=True):
    global extractor, tokenizer
    audio = batch["audio"]
    wav = torch.tensor(audio["array"]).float() 

    if waveform:
        batch["waveform"] = wav.unsqueeze(0)

    if input_features:
        features_np = extractor(wav.numpy(), sampling_rate=audio["sampling_rate"], padding="longest").input_features[0]
        current_features = torch.tensor(features_np)
        pad_val = current_features.min().item() 
        current_features = torch.where(current_features == pad_val, torch.tensor(0.0, dtype=current_features.dtype), current_features)
        batch["input_features"] = current_features
        
    batch["labels"] = tokenizer(batch["transcription"], add_special_tokens=False).input_ids
    return batch

def compute_metrics(eval_pred, compute_result: bool = True):
    global extractor, tokenizer, model, optimizer, scheduler

    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if hasattr(pred_logits, "cpu"):
        pred_logits = pred_logits.cpu()
    if hasattr(label_ids, "cpu"):
        label_ids = label_ids.cpu()

    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits

    if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
        if not isinstance(pred_ids, torch.Tensor):
            pred_ids = torch.tensor(pred_ids)
        pred_ids = pred_ids.argmax(dim=-1)
        pred_ids = pred_ids.tolist()
    elif hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()

    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()

    label_ids = [
        [tokenizer.pad_token_id if token == -100 else token for token in seq]
        for seq in label_ids
    ]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    print("--------------------------------")
    print(f"Prediction: {pred_str[0]}")
    print(f"Label: {label_str[0]}")

    pred_flat = list(chain.from_iterable(pred_ids))
    labels_flat = list(chain.from_iterable(label_ids))
    mask = [i != tokenizer.pad_token_id for i in labels_flat]

    acc = accuracy_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m]
    )
    pre = precision_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    rec = recall_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    f1 = f1_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    tracked_params = get_tracked_parameters(model, model.param_tracking_paths)
    
    RotationA = tracked_params["RotationA"]
    RotationB = tracked_params["RotationB"]
    Silence = tracked_params["Silence"]
          
    metrics = {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "RotationA": RotationA,
        "RotationB": RotationB,
        "Silence": Silence,
    }
        # metrics.update(tracked_params)
    return metrics

def create_model(param):
    model = Echo(param).to('cuda')
    model.init_weights()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    return model

def setup_tokenizers(token):
    global extractor, tokenizer
    
    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", 
        token=token, 
        feature_size=128,
        sampling_rate=16000, 
        do_normalize=True, 
        return_tensors="pt", 
        chunk_length=15, 
        padding_value=0.0,
        padding=False,
    )

    tokenizer = WhisperTokenizer.from_pretrained(
        "./tokenizer", 
        pad_token="0", 
        bos_token="0", 
        eos_token="0", 
        unk_token="0",
        token=token, 
        local_files_only=True, 
        pad_token_id=0
    )
    
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decoder_start_token_id = 0

def prepare_datasets(token):
    dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    def filter_func(x):
        return (0 < len(x["transcription"]) < 512 and
                len(x["audio"]["array"]) > 0 and
                len(x["audio"]["array"]) < 1500 * 160)
    dataset = dataset.filter(filter_func).shuffle(seed=4)
    print("Dataset size:", dataset["train"].num_rows, dataset["test"].num_rows)

    prepare_fn = partial(prepare_dataset, input_features=True, waveform=True)

    dataset = dataset.map(function=prepare_fn, remove_columns=list(next(iter(dataset.values())).features)
    ).with_format(type="torch")
    train_dataset = dataset["train"].shuffle(seed=4).flatten_indices()
    test_dataset = dataset["test"].shuffle(seed=4).take(200).flatten_indices()
    return train_dataset, test_dataset

def get_training_args(log_dir):
    return Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=1000,
        save_steps=1001,
        eval_steps=100,
        warmup_steps=0,
        num_train_epochs=1,
        logging_steps=100,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=0.0025,
        weight_decay=0.25,
        save_safetensors=False,
        eval_on_start=True,
        batch_eval_metrics=True,
    )

if __name__ == "__main__":
    
    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        encoder_idx=4,
        audio_dims=512,
        vocab=51865,
        text_ctx=256,
        text_head=4,
        decoder_idx=4,
        text_dims=512,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=0,
        act="swish",
        debug=False,
        cross_attention=True,

    )
    
    sys.excepthook = custom_excepthook
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)
    
    setup_tokenizers(token)
    model = create_model(param)
    train_dataset, test_dataset = prepare_datasets(token)
    training_args = get_training_args(log_dir)
   
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        # processing_class=extractor,
    )
        
    trainer.train()
        

# from tensorboard import program
# log_dir = "./output/logs" 
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', log_dir])
# url = tb.launch()
# print(f"TensorBoard started at {url}")







```
