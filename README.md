## Zero-Value Processing in Speech Attention

### The Significance of Zeros in Audio Processing

In log-mel spectrograms, zero or near-zero values represent critical information:
- Silent regions between speech
- Low-amplitude acoustic events
- Sub-threshold background environments

### Multiplicative Soft Masking: Technical Implementation

```python
token_ids = k[:, :, :, 0].to(q.device, q.dtype)
scaled_zero = torch.ones_like(token_ids).to(q.device, q.dtype)
scaled_zero[token_ids == 0] = 0.000001
scaling_factors = scaled_mask.unsqueeze(0) * scaled_zero.unsqueeze(-2).expand(qk.shape)
```

## Key Innovations and Benefits

1. **Semantic Preservation of Silence**: Unlike conventional `-inf` masking that eliminates attention, this approach maintains minimal attention flow (0.000001) for silence tokens, preserving their semantic value.

2. **Prosodic Pattern Recognition**: By allowing minimal attention to silent regions, the model can learn timing, rhythm, and prosodic features critical for speech understanding.

3. **Cross-Modal Representation Unification**: Using identical attention mechanisms for both audio silence and text padding creates a unified approach across modalities, simplifying the architecture.

4. **Training Stability**: This multiplicative masking approach maintains gradient flow through all positions, potentially improving convergence stability.

5. **Natural Boundary Learning**: By processing silence regions with reduced but non-zero attention, the model learns natural speech boundaries without requiring explicit BOS/EOS tokens.

## Adaptive Audio Feature Fusion

The model incorporates a learnable parameter with sigmoid activation to adaptively blend waveform and spectrogram encodings. Initial findings demonstrate significant WER (Word Error Rate) reduction compared to single-representation approaches, with minimal computational overhead increase.

This adaptive fusion addresses the longstanding waveform-vs-spectrogram debate in ASR by allowing the model to determine the optimal representation mix for different acoustic contexts. While feature fusion has been explored in research settings, this learnable parameter approach provides an elegant solution that maintains computational efficiency.


```python
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Dict
import gzip
import base64
import transformers
from dataclasses import dataclass
from itertools import chain

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

def shift_with_zeros(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()   
    return shifted_input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.RMSNorm):       
    def forward(self, x: Tensor) -> Tensor:

        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        eps = self.eps if self.eps is not None else torch.finfo(x_float.dtype).eps
        x_normalized = x_float * torch.rsqrt(variance + eps).to(x.device, x.dtype)
        if self.weight is not None:
            return (x_normalized * self.weight).to(x.device, x.dtype)
        return x_normalized.to(x.device, x.dtype)
    
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, 
            self.weight.to(x.device, x.dtype),
            None if self.bias is None else self.bias.to(x.device, x.dtype)
        )
    
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)      

class Rotary(nn.Module):
    def __init__(self, dim, max_ctx=4096, learned_freq=True):
        super().__init__()
        self.dim = dim
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=learned_freq
        )
        self.bias = nn.Parameter(torch.zeros(max_ctx, dim // 2))  

    def forward(self, positions):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.inv_freq.device).float()
        else:
            t = positions.float().to(self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)
    
class Multihead(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        head_dim = dims // head
        self.query = Linear(dims, dims)
        self.key = Linear(dims, dims, bias=False)
        self.value = Linear(dims, dims)
        self.out = Linear(dims, dims)
        self.rotary = Rotary(dim=head_dim, learned_freq=True)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None,  mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):

        q = self.query(x).to(x.device, x.dtype)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        wv, qk = self._attention(q, k, v, mask)
        return self.out(wv), qk
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v = None, mask = None):
        batch, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        freq = self.rotary(ctx)
        q = self.rotary.apply_rotary(q, freq)
        k = self.rotary.apply_rotary(k, freq)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        scaled_mask = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(1.0)).to(q.device, q.dtype)

        token_ids = k[:, :, :, 0].to(q.device, q.dtype)
        scaled_zero = torch.ones_like(token_ids).to(q.device, q.dtype)
        scaled_zero[token_ids == 0] = 0.000001
        scaling_factors = scaled_mask.unsqueeze(0) * scaled_zero.unsqueeze(-2).expand(qk.shape)
        qk *= scaling_factors
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.device, q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, qk.detach()

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, cross_attention: bool = False, act = "relu"):
        super().__init__()
        self.dims = dims
        self.head = head
        self.cross_attention = cross_attention
        self.dropout = 0.1

        self.blend_xa = nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        self.blend = torch.sigmoid(self.blend_xa)

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.attna = Multihead(dims=dims, head=head)
        self.attnb = Multihead(dims=dims, head=head) if cross_attention else None
    
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))
        self.lna = RMSNorm(normalized_shape=dims)    
        self.lnb = RMSNorm(normalized_shape=dims) if cross_attention else None
        self.lnc = RMSNorm(normalized_shape=dims) 

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        r = x
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache)[0]
        if self.attnb and xa is not None:
            cross_out = self.attnb(self.lnb(x), xa, kv_cache=kv_cache)[0]
            x = self.blend * x + (1 - self.blend) * cross_out
        x = x + self.mlp(self.lnc(x))
        x = x + r
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
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer, act, cross_attention = False):
        super().__init__()

        self.dropout = 0.1

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(),
                   "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.blend_sw = nn.Parameter(torch.tensor(0.5, device=tox["device"], dtype=tox["dtype"]), requires_grad=False)
        self.blend = torch.sigmoid(self.blend_sw)
        self.ln_enc = RMSNorm(normalized_shape=dims, **tox)
        self.register_buffer("positional_embedding", sinusoids(ctx, dims))

        self.se = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, padding=1), self.act,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),
            Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
            nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1)
        )
        self.we = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(ctx),
        )

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=cross_attention)
                    for _ in range(layer)]) if layer > 0 else None)
        
    def forward(self, x, w) -> Tensor:
            
        if x is not None:
            if w is not None:
                x = self.se(x).permute(0, 2, 1)
                w = self.we(w).permute(0, 2, 1)
                x = (x + self.positional_embedding).to(x.device, x.dtype)
                x = self.blend * x + (1 - self.blend) * w
            else:
                x = self.se(x)
                x = x.permute(0, 2, 1)
                assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                x = (x + self.positional_embedding).to(x.device, x.dtype)
        else:
            assert w is not None, "You have to provide either x or w"
            x = self.we(w).permute(0, 2, 1)
            assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            x = (x + self.positional_embedding).to(x.device, x.dtype)

        for block in chain(self.blockA or []):
            x = block(x)

        return self.ln_enc(x)
        
class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer, cross_attention = False):
        super().__init__()     
        self.dropout = 0.1

        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        self.ln_dec = RMSNorm(normalized_shape=dims)
        self.rotary = Rotary(dim=dims, learned_freq=True)
        
        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=cross_attention) for _ in range(layer)]) if layer > 0 else None)

        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, kv_cache=None) -> Tensor:

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        ctx = x.shape[1]
        freqs = self.rotary(ctx)
        x = self.rotary.apply_rotary(x, freqs)
        x = x.to(xa.dtype)

        for block in chain(self.blockA or []):
            x = block(x, xa=xa, mask=self.mask, kv_cache=kv_cache)
            
        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param  

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layer=param.encoder_idx,
            act=param.act,
        )

        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.decoder_idx,
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
    
    @torch.autocast(device_type="cuda")
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
        logits = self.decoder(input_ids, encoder_output).to('cuda')
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
            elif isinstance(module, Multihead):
                self.init_counts["Multihead"] += 1
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

param = Dimensions(
    mels=128,
    audio_ctx=1500,
    audio_head=4,
    encoder_idx=4,
    audio_dims=512,
    vocab=51865,
    text_ctx=512,
    text_head=4,
    decoder_idx=4,
    text_dims=512,
    decoder_start_token_id=0,
    pad_token_id=0,
    eos_token_id=0,
    act="gelu",
)

model = Echo(param).to('cuda')
model.init_weights()


    







```
