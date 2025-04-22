

import os
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from BetweennessRoPE import BetweennessRoPE
import numpy as np
from typing import Optional, Dict
import gzip
import base64
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor, WhisperTokenizerFast
from typing import Union, List, Any
import evaluate
import transformers
from dataclasses import dataclass
from mask import UniversalMask
from itertools import chain

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
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

def visualize_mask(mask, title="Attention Mask"):
    import matplotlib.pyplot as plt
    if mask.dim() == 4:
        mask_vis = mask[0, 0].cpu().detach().numpy()
    else:
        mask_vis = mask.cpu().detach().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_vis, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.RMSNorm):       
    def forward(self, x: Tensor) -> Tensor:
        """Preserve the input dtype throughout the normalization"""
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        eps = self.eps if self.eps is not None else torch.finfo(x_float.dtype).eps
        x_normalized = x_float * torch.rsqrt(variance + eps)
        if self.weight is not None:
            return (x_normalized * self.weight).type(x.dtype)
        return x_normalized.type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype))
    
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

class ParameterCycler:
    def __init__(self, parameters):
        self.parameters = parameters
        self.current_idx = 0

    def toggle_requires_grad(self):
        for i, param in enumerate(self.parameters):
            param.requires_grad = i == self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.parameters)

def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)      

class Rotary(nn.Module):
    def __init__(self, dim, base_theta=10000.0, learned_freq=False, device=None):
        super().__init__()
        self.dim = dim
        self.base_theta = base_theta
        self.device = device

        inv_freq = 1.0 / (base_theta ** (torch.arange(0, dim, 2).float() / dim))
        if learned_freq:
            self.inv_freq = nn.Parameter(inv_freq, requires_grad=True)
        else:
            self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs.float()), freqs.float())  # (seq_len, dim//2)
        return freqs_cis

    @staticmethod
    def apply_rotary(x, freqs_cis): # freqs_cis: (seq_len, dim//2) - x: (..., seq_len, dim)
        x1 = x[..., :freqs_cis.shape[-1]*2]
        x2 = x[..., freqs_cis.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2)
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs_cis
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)

class Multihead(nn.Module):
    blend = False
    mag = False
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.dims = dims
        self.head = head
        head_dim = dims // head
        self.head_dim = head_dim
        self.dropout = 0.1

        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)

        self.use_betweenness = False  
        if self.use_betweenness:
            self.betweenness_rope = BetweennessRoPE(dim=head_dim)
        else:
            self.rotary = Rotary(dim=head_dim, learned_freq=True)

        if Multihead.blend:
            self.factor = nn.Parameter(torch.tensor(0.5, **tox))

    def compute_cosine_attention(self, q: Tensor, k: Tensor, v: Tensor, mask):
        ctx = q.shape[1]
        qn = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        kn = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk = torch.matmul(qn, kn.transpose(-1, -2))

        if Multihead.mag:
            qm = torch.norm(q, dim=-1, keepdim=True)
            km = torch.norm(k, dim=-1, keepdim=True)
            ms = (qm * km.transpose(-1, -2)) ** 0.5
            ms = torch.clamp(ms, min=1e-8)
            qk = qk * ms

        if mask is not None:
            qk = qk + mask[:ctx, :ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        w = F.dropout(w, p=self.dropout, training=self.training)
        out = torch.matmul(w, v)
        return out, qk

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask = None, kv_cache = None):
        q = self.q(x)
        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]
        out, qk = self._forward(q, k, v, mask)
        return self.o(out), qk

    def _forward(self, q: Tensor, k: Tensor, v: Tensor, mask = None):
        ctx = q.shape[1]
        dims = self.dims
        scale = (dims // self.head) ** -0.25

        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        if self.use_betweenness:
            betweenness = self.betweenness_rope.compute_betweenness(q)
            freqs_cos, freqs_sin = self.betweenness_rope.precompute_freqs_cis(ctx, q.device)
            q = self.betweenness_rope.apply_rope(q, freqs_cos, freqs_sin, betweenness)
            k = self.betweenness_rope.apply_rope(k, freqs_cos, freqs_sin, betweenness)
        else:
            freqs_cis = self.rotary(ctx)
            q = self.rotary.apply_rotary(q, freqs_cis)
            k = self.rotary.apply_rotary(k, freqs_cis)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if mask is not None:
            qk = qk + mask[:ctx, :ctx]
        qk = qk.float()
        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        w = F.dropout(w, p=self.dropout, training=self.training)
        out = torch.matmul(w, v)

        if Multihead.blend:
            cos_w, cos_qk = self.compute_cosine_attention(q, k, v, mask)
            blend = torch.sigmoid(self.factor)
            out = blend * cos_w + (1 - blend) * out
            qk = blend * cos_qk + (1 - blend) * qk
        
        out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach() if self.training else qk
        return out, qk

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, cross_attention: bool = False, act = "relu"):
        super().__init__()
        self.dims = dims
        self.head = head
        self.cross_attention = cross_attention
        self.dropout = 0.1

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
        mask = mask if isinstance(self, TextDecoder) else None
        r = x
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache)[0]
        if self.attnb and xa is not None:
            x = x + self.attnb(self.lnb(x), xa, kv_cache=kv_cache)[0] 
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
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer, act: str = "relu"):
        super().__init__()

        self.dropout = 0.1
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), 
                   "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.blend_sw = nn.Parameter(torch.tensor(0.5), requires_grad=True) # 

        self.se = nn.Sequential(Conv1d(mels, dims, kernel_size=3, padding=1), self.act, # spectrogram encoder
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2), 
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),     
            Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
            nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1))
        
        self.we = nn.Sequential( # waveform encoder
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(ctx),
        )

        self.register_buffer("positional_embedding", sinusoids(ctx, dims))       

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=False, act = "relu")
                    for _ in range(layer)]) if layer > 0 else None)
        
        self.ln_enc = RMSNorm(normalized_shape=dims)
        self.se_norm = RMSNorm(normalized_shape=dims)
        self.we_norm = RMSNorm(normalized_shape=dims)

    def forward(self, x, w) -> Tensor:
        """ x : torch.Tensor, shape = (batch, mels, ctx) the mel spectrogram of the audio input"""
        """ w : torch.Tensor, shape = (batch, 1, ctx) the waveform of the audio input """
        """ x : torch.Tensor, shape = (batch, ctx, dims) the encoder output """
        if x is not None:
            if w is not None:
                se_x = self.se(x)
                se_x = se_x.permute(0, 2, 1)
                se_x = self.se_norm(se_x)
                x = (se_x + 0.1 * self.positional_embedding).to(se_x.dtype)
                we_w = self.we(w).permute(0, 2, 1)
                we_w = self.we_norm(we_w)
                blend = torch.sigmoid(self.blend_sw)
                x = blend * x + (1 - blend) * we_w
            else:
                x = self.se(x) 
                x = x.permute(0, 2, 1) 
                assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                x = (x + self.positional_embedding).to(x.dtype)
        else:
            assert w is not None, "You have to provide either x or w"
            x = self.we(w).permute(0, 2, 1)
            assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            x = (x + self.positional_embedding).to(x.dtype)

        for block in chain(self.blockA or []):
            x = block(x)
        return self.ln_enc(x)

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer):
        super().__init__()
        self.dropout = 0.1
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims))
        self.ln_dec = RMSNorm(normalized_shape=dims)

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=False) for _ in range(layer)]) if layer > 0 else None)

        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, kv_cache=None) -> Tensor:
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
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

    def logits(self,input_ids: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(input_ids, audio_features)

    def forward(
        self, 
        input_features: torch.Tensor=None, 
        waveform: Optional[torch.Tensor]=None,
        input_ids=None, 
        labels=None, 
        decoder_inputs_embeds=None,
        ) -> Dict[str, torch.Tensor]:
        
        if input_ids is None and decoder_inputs_embeds is None:
            if labels is not None:
                input_ids = shift_tokens_right(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id)
            else:
                raise ValueError("You have to provide either decoder_input_ids or labels")
            
        if input_features is not None:    
            if waveform is not None:
                encoded_audio = self.encoder(x=input_features, w=waveform)
            else:
                encoded_audio = self.encoder(x=input_features, w=None)
        elif waveform is not None:
            encoded_audio = self.encoder(x=None, w=waveform)
        else:
            raise ValueError("You have to provide either input_features or waveform")

        logits = self.decoder(input_ids, encoded_audio)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "labels": labels, "input_ids": input_ids, "audio_features": encoded_audio}

    @property
    def device(self):
        return next(self.parameters()).device

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []
        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.param.text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def save_adaptive_output(module, _, output):
            if isinstance(output, tuple) and len(output) == 2:
                tensor_output, cache_updates = output
                module_k = f"{module}_k"
                module_v = f"{module}_v"
                if module_k not in cache or tensor_output.shape[1] > self.param.text_ctx:
                    cache[module_k] = cache_updates["k_cache"]
                    cache[module_v] = cache_updates["v_cache"]
                else:
                    cache[module_k] = torch.cat([cache[module_k], cache_updates["k_cache"]], dim=1).detach()
                    cache[module_v] = torch.cat([cache[module_v], cache_updates["v_cache"]], dim=1).detach()
                return tensor_output
            return output
        
        def install_hooks(layer: nn.Module):
            if isinstance(layer, Multihead):
                hooks.append(layer.k.register_forward_hook(save_to_cache))
                hooks.append(layer.v.register_forward_hook(save_to_cache))
            self.encoder.apply(install_hooks)
        self.decoder.apply(install_hooks)
        return cache, hooks
    
    @staticmethod
    def create_attention_mask(batch_size, ctx, num_heads, is_causal=False, 
                            device=None, dtype=None, xa_ctx=None):
        mask_type = "cross_attention" if xa_ctx is not None else "combined"
        
        return UniversalMask.create(
            batch_size=batch_size,
            ctx=ctx,
            ctx_kv=xa_ctx if xa_ctx is not None else ctx,
            num_heads=num_heads,
            mask_type=mask_type,
            is_causal=is_causal,
            padding_mask=None,
            device=device,
            dtype=dtype
        )
    
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
                nn.init.xavier_uniform_(module.q.weight)
                nn.init.zeros_(module.q.bias)
                nn.init.xavier_uniform_(module.k.weight)
                nn.init.xavier_uniform_(module.v.weight)
                nn.init.xavier_uniform_(module.o.weight)
                if module.o.bias is not None:
                    nn.init.zeros_(module.o.bias)
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
                # nn.init.xavier_uniform_(module.token_embedding.weight)
                # nn.init.xavier_uniform_(module.positional_embedding)
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
                nn.init.xavier_uniform_(module.attna.q.weight)
                nn.init.zeros_(module.attna.q.bias)
                nn.init.xavier_uniform_(module.attna.k.weight)
                nn.init.xavier_uniform_(module.attna.v.weight)
                nn.init.xavier_uniform_(module.attna.o.weight)
                if module.attna.o.bias is not None:
                    nn.init.zeros_(module.attna.o.bias)
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
    extractor: Any
    tokenizer: Any
    decoder_start_token_id: Any

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        batch = {}

        if "input_features" in features[0]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch["input_features"] = self.extractor.pad(input_features, return_tensors="pt")["input_features"]
            # print("input_features shape:", batch["input_features"].shape)
        if "waveform" in features[0]:
            waveforms = [f["waveform"] for f in features]
            fixed_len = 3000 * 160  # Set a fixed length for all waveforms (e.g., 480000 samples for 30 seconds at 16kHz)
            padded_waveforms = []
            for w in waveforms:
                if w.shape[-1] < fixed_len:
                    w = F.pad(w, (0, fixed_len - w.shape[-1]))
                else:
                    w = w[..., :fixed_len]
                padded_waveforms.append(w)
            batch["waveform"] = torch.stack(padded_waveforms)
            # print("waveform shape:", batch["waveform"].shape)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def prepare_dataset(batch, input_features=True, waveform=True):
    audio = batch["audio"]
    fixed_len = 3000 * 160
    wav = torch.tensor(audio["array"]).float()
    if wav.shape[-1] < fixed_len:
        wav = F.pad(wav, (0, fixed_len - wav.shape[-1]))
    else:
        wav = wav[..., :fixed_len]
    if waveform:
        batch["waveform"] = wav.unsqueeze(0)  # (1, N)
    if input_features:
        batch["input_features"] = extractor(wav.numpy(), sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

def compute_metrics(eval_pred):
    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    pred_flat = pred_ids.flatten()
    labels_flat = label_ids.flatten()
    mask = labels_flat != tokenizer.pad_token_id
  
    # samp = [0] # sample = random.randint(0, len(pred_str)-1) sample = random.sample(range(10), 1)
    # for idx in samp:
    #     print("-" * 10)
    #     print(f"Step: {trainer.state.global_step}")
    #     print(f"Prediction: {pred_str[idx]}")
    #     print(f"Label: {label_str[idx]}")

    acc = accuracy_score(y_true=labels_flat[mask], y_pred=pred_flat[mask])
    pre = precision_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    rec = recall_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    f1 = f1_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    
    return {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1
        }

if __name__ == "__main__":

    param = Dimensions(
        mels=80,
        audio_ctx=3000,
        audio_head=4,
        encoder_idx=4,
        audio_dims=512,
        vocab=51865,
        text_ctx=512,
        text_head=4,
        decoder_idx=4,
        text_dims=512,
        decoder_start_token_id = 50258,
        pad_token_id = 50257,
        eos_token_id = 50257,   
        act = "gelu",
        )

    model = Echo(param).to('cuda')

    token=""
    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", token=token, feature_size=80, sampling_rate=16000, do_normalize=True, return_tensors="pt")
    tokenizer = WhisperTokenizerFast.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe", token=token)
    data_collator = DataCollator(extractor=extractor,
        tokenizer=tokenizer, decoder_start_token_id=50258)

    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)

    dataset = DatasetDict()
    dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    dataset = dataset.map(function=prepare_dataset,
        remove_columns=list(next(iter(dataset.values())).features)).with_format(type="torch")

    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=100000,
        save_steps=100000,
        eval_steps=2000,
        warmup_steps=1000,
        num_train_epochs=1,
        logging_steps=10,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        eval_on_start=False,
        optim="adafactor",
        save_safetensors=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=extractor,
    )
    model.init_weights()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    trainer.train(resume_from_checkpoint=False)

    from tensorboard import program
    log_dir = "./output/logs" 
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"TensorBoard started at {url}")


