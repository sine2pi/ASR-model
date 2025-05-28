


import os
import warnings
import logging
import torch, torchaudio
import torchcrepe
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from itertools import chain
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
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperTokenizer
import evaluate
import transformers
from dataclasses import dataclass
import pyworld as pw
from torch import nn, Tensor

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

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    text_idx: int
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    audio_idx: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    act: str
    debug: bool
    cross_attn: bool
    features: List[str]
    f0_rotary: bool

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

def analyze_gate_patterns(model, dataloader):
    gate_patterns = []
    for batch in dataloader:
        outputs = model(**batch)
        layer_gates = []
        for layer in model.decoder.self_attn_layers:
            mlp_gate = layer.mlp_gate(layer.lnc(x))
            layer_gates.append(mlp_gate.detach().cpu())
        gate_patterns.append(layer_gates)
    return gate_patterns

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def exists(v):
    return v is not None

def default(v, b):
    return v if exists(v) else b

class rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, variable_radius=False,
                 learned_radius=False, learned_theta=False, debug=False):
        super().__init__()
        self.debug = False
        self.interpolate_factor = 10.0
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        
        self.inv_freq = nn.Parameter(1.0 / (10000 ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)), requires_grad=learned_freq)
        self.theta = nn.Parameter(torch.tensor(float(theta)), requires_grad=learned_theta)
        self.min_theta = nn.Parameter(torch.tensor(800.0), requires_grad=learned_theta)
        self.max_theta = nn.Parameter(torch.tensor(10000.0), requires_grad=learned_theta)
        
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius)
            
    def forward(self, x = None, f0=None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=device).float()
        else:
            t = x.float().to(self.inv_freq.device)

        if f0 is not None:
            f0_tensor = f0.squeeze(0) if f0.ndim == 3 else f0
            if f0_tensor.ndim > 1:
                f0_tensor = f0_tensor.squeeze()
            f0_mean = f0_tensor.mean()
            f0_mean = torch.clamp(f0_mean, min=80.0, max=600.0)
            perceptual_factor = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
            f0_theta = self.min_theta + perceptual_factor * (self.max_theta - self.min_theta)
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=device) / self.dims))
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
        if self.debug:
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

class MultiheadA(nn.Module):
    def __init__(self, dims: int, head: int, debug=False):
        super().__init__()

        self.count = 0
        self.debug = debug
        self.pad_token = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)
        self.fzero = nn.Parameter(torch.tensor(0.0001))
        
        self.rope = rotary(
            dims=self.head_dim,
            max_ctx = 1500,
            theta = 10000,
            learned_freq = False,
            variable_radius = False,
            learned_radius = False,
            debug = False)

    def forward(self, x: Tensor, xa = None, mask = None, return_attn=False, f0=None):

        z = default(xa, x)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)

        if f0 is not None:
            qf = self.rope(q.size(1), f0=f0)
            kf = self.rope(k.size(1), f0=f0)
        else:
            qf = self.rope(q.size(1))
            kf = self.rope(k.size(1))

        bat, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        q = self.rope.apply_rotary(q, qf)
        k = self.rope.apply_rotary(k, kf)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), min=0.00001, max=0.001)
        zscale[token_ids.float() == self.pad_token] = fzero.to(q.device, q.dtype)
        if mask is not None:
            mask = mask[:ctx, :ctx]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)
        if return_attn:
            return qk, v
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        if self.debug and self.count % 100 == 0:
            print(f"Step {self.count}: x: {x.shape}, xa: {xa.shape if xa is not None else None}, mask: {mask.shape if mask is not None else None}")
        self.count += 1
        return self.o(wv), qk.detach()
    
class Residual(nn.Module):
    def __init__(self, dims: int, head: int, ctx, act, cross_attn=True, debug=False):
        super().__init__()
        self.ctx = ctx
        self._counter = 0
        self.dropout = 0.1
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.cross_attn = cross_attn
        self.debug = debug
        self.blend_xa = nn.Parameter(torch.tensor(0.5))
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                  "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), 
                  "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), 
                  "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head)
        self.attnb = (MultiheadA(dims, head) if cross_attn else None)
        
        mlp = dims * 4
        self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(Linear(dims, mlp), act_fn, Linear(mlp, dims))
        
 
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims) if cross_attn else None
        self.lnc = RMSNorm(dims)

    def forward(self, x, xa=None, mask=None, f0=None):
        x = x + self.attna(self.lna(x), mask=mask, f0=f0)[0]
        if self.attnb and xa is not None:
            cross = self.attnb(self.lnb(x), xa, f0=f0)[0]
            blend = torch.sigmoid(self.blend_xa)
            x = blend * x + (1 - blend) * cross
        mlp_out = self.mlp(self.lnc(x))
        if hasattr(self, 'mlp_gate'):
            mlp_gate = self.mlp_gate(self.lnc(x)) 
            x = x + mlp_gate * mlp_out
        else:
            x = x + mlp_out
        return x

class FeatureEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, stride=1):
        super().__init__()
        self.head_dim = dims // head  
        self.dropout = 0.1 
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=stride, padding=kernel_size//2), act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims), act_fn)
        
        self.positional_embedding = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)
        self._norm = RMSNorm(dims)

    def forward(self, x, f0=None):
        x = self.encoder(x).permute(0, 2, 1)
        x = x + self.positional_embedding(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._norm(x)
        return x
    
class AudioEncoder(nn.Module):
    def __init__(self, mels, dims, head, ctx, layer, act, debug, features, f0_rotary):
   
        super().__init__()
        self.debug = debug
        self.features = features
        self._counter = 0
        self.dropout = 0.1
        self.f0_rotary = f0_rotary

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
                
        self.processors = nn.ModuleDict({
            "spectrogram": nn.ModuleList(
            [FeatureEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(dims=dims, head=head, ctx=ctx, act=act, debug=debug) for _ in range(layer)] if "spectrogram" in features else None
            ),  

            "waveform": nn.ModuleList(
            [FeatureEncoder(input_dims=64, dims=dims, head=head, layer=layer, kernel_size=7, act=act_fn, stride=1)] +
            [Residual(dims=dims, head=head, ctx=ctx, act=act, debug=debug) for _ in range(layer)] if "waveform" in features else None
            ),

            "pitch": nn.ModuleList(
            [FeatureEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act, stride=2)] +
            [Residual(dims=dims, head=head, ctx=ctx, act=act, debug=debug) for _ in range(layer)] if "pitch" in features else None
            )
            })

    def forward(self, x):
                   
        out = {}
        if self.f0_rotary:
            f0 = x.get("pitch")
        else:
            f0 = None
        for feature in self.features:
            if feature in x:
                feat = x[feature]
                for blk in chain(self.processors[feature] or []):
                    feat = blk(feat, f0=f0)
                out[feature] = feat
                               
        return out

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, layer: int, dims: int, head: int, ctx: int, cross_attn, features, debug, sequential=False): 
        super().__init__()
        self._counter = 0
        self.dropout = 0.1
        self.debug = debug
        self.sequential = sequential
        self.features = features
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        
        self.self_attn_layers = nn.ModuleList([
            Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attn=cross_attn, debug=debug)
            for _ in range(layer)])
        
        self.processors = nn.ModuleDict({
        f: nn.ModuleList([Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attn=cross_attn, debug=debug)
            for _ in range(layer)]) for f in features})
        
        self.feature_blend = nn.ParameterDict({f: nn.Parameter(torch.tensor(0.5)) for f in features})
        
        self.ln_dec = RMSNorm(dims, **tox)
        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0)        
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, encoder_outputs, feature_order=None) -> Tensor:
        
        if feature_order is None:
            feature_order = self.features
            
        x = x.to(device=device)
        mask = self.mask[:x.shape[1], :x.shape[1]]
        x = (self.token_embedding(x) + self.positional_embedding[:x.shape[1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        for layer in self.self_attn_layers:
            x = layer(x, mask=mask)

        for feature in feature_order:
            if feature in encoder_outputs:
                encoding = encoder_outputs[feature]
                output = x
                for layer in self.processors[feature]:
                    output = layer(output, xa=encoding, mask=mask)
                if self.sequential:
                    x = output
                else:
                    alpha = torch.sigmoid(self.feature_blend[feature])
                    x = alpha * output + (1 - alpha) * x
                    
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
            layer=param.audio_idx,
            act=param.act,
            debug=param.debug,
            features=param.features,
            f0_rotary=param.f0_rotary,
            )
        
        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.text_idx,
            cross_attn=param.cross_attn,
            debug=param.debug,
            features=param.features,
            )

        all_head = torch.zeros(self.param.text_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.text_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.text_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, spectrogram: torch.Tensor):
        return self.encoder(spectrogram)

    def logits(self,input_ids: torch.Tensor, encoder_output: torch.Tensor):
        return self.decoder(input_ids, encoder_output)
    
    def forward(self,
        decoder_input_ids=None,
        labels=None,
        waveform: Optional[torch.Tensor]=None,
        input_ids=None,
        spectrogram: torch.Tensor=None,
        pitch: Optional[torch.Tensor]=None,
    ) -> Dict[str, torch.Tensor]:

        decoder_input_ids = input_ids
        encoder_inputs = {}
        if spectrogram is not None:
            encoder_inputs["spectrogram"] = spectrogram
        if waveform is not None:
            encoder_inputs["waveform"] = waveform
        if pitch is not None:
            encoder_inputs["pitch"] = pitch

        encoder_outputs = self.encoder(encoder_inputs)
        logits = self.decoder(input_ids, encoder_outputs)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = loss_fn(
                shifted_logits.view(-1, shifted_logits.shape[-1]), 
                shifted_labels.view(-1))

           
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_outputs} 

    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, 
            "Residual": 0, "MultiheadA": 0, "MultiheadB - Cross Attention": 0, 
            "MultiheadC": 0, "MultiheadD": 0, "FeatureEncoder": 0,
            "WaveformEncoder": 0, "PitchEncoder": 0
                            }

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
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1
            elif isinstance(module, FeatureEncoder):
                self.init_counts["FeatureEncoder"] += 1

    def init_weights(self):
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")


def analyze_gate_behavior(gate_stats):
    print("\nGate Activation Analysis:")
    
    all_means = []
    all_stds = []
    
    for layer_name, stats_list in gate_stats.items():
        if not stats_list:
            continue
            
        means = [s['mean'] for s in stats_list]
        stds = [s['std'] for s in stats_list]
        mins = [s['min'] for s in stats_list]
        maxs = [s['max'] for s in stats_list]
        
        all_means.extend(means)
        all_stds.extend(stds)
        
        print(f"\nLayer {layer_name}:")
        print(f"  Mean gate value: {sum(means)/len(means):.4f}")
        print(f"  Mean std deviation: {sum(stds)/len(stds):.4f}")
        print(f"  Range: {min(mins):.4f} to {max(maxs):.4f}")
        
        if len(means) > 10:
            early_mean = sum(means[:10])/10
            late_mean = sum(means[-10:])/10
            print(f"  Trend: {'Increasing' if late_mean > early_mean else 'Decreasing'} gate activation")
            print(f"  Early mean: {early_mean:.4f}, Late mean: {late_mean:.4f}")
    
    if all_means:
        print("\nOverall Gate Statistics:")
        print(f"  Average gate value: {sum(all_means)/len(all_means):.4f}")
        print(f"  Average std deviation: {sum(all_stds)/len(all_stds):.4f}")

class SelfCriticalTraining(nn.Module):
    
    def __init__(self, model, tokenizer, max_length=256, pad_token_id=0, 
                 eos_token_id=50256, temperature=1.0, top_k=0, top_p=0.9,
                 scheduled_sampling_ratio=0.5, monitor_gates=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.scheduled_sampling_ratio = scheduled_sampling_ratio
        self.monitor_gates = monitor_gates
        self.gate_stats = {}
        
    def _sample_from_logits(self, logits):
        logits = logits / max(self.temperature, 1e-5)
        
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_token
    
    def _collect_gate_stats(self, layer, batch_idx):

        if hasattr(layer, 'mlp_gate') and self.monitor_gates:
            x = layer.lnc(layer.attna(layer.lna(layer._last_input))[0])
            gate_values = layer.mlp_gate(x)
            
            stats = {
                'mean': gate_values.mean().item(),
                'std': gate_values.std().item(),
                'min': gate_values.min().item(),
                'max': gate_values.max().item(),
                'zeros': (gate_values < 0.01).float().mean().item(),
                'ones': (gate_values > 0.99).float().mean().item()
            }
            
            layer_name = f"layer_{id(layer)}"
            if layer_name not in self.gate_stats:
                self.gate_stats[layer_name] = []
            
            self.gate_stats[layer_name].append(stats)
                
    def forward(self,
        decoder_input_ids=None,
        labels=None,
        waveform: Optional[torch.Tensor]=None,
        input_ids=None,
        spectrogram: torch.Tensor=None,
        pitch: Optional[torch.Tensor]=None,
        use_teacher_forcing=None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        encoder_inputs = {}
        if spectrogram is not None:
            encoder_inputs["spectrogram"] = spectrogram
        if waveform is not None:
            encoder_inputs["waveform"] = waveform
        if pitch is not None:
            encoder_inputs["pitch"] = pitch
            
        for key in kwargs:
            if key in ['spectrogram', 'waveform', 'pitch'] and kwargs[key] is not None:
                encoder_inputs[key] = kwargs[key]
        
        if not encoder_inputs:
            print(f"WARNING: No encoder inputs found. Using base model forward.")
            return self.model(
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                input_ids=input_ids,
                **kwargs
            )
        
        batch_size = next(iter(encoder_inputs.values())).size(0)
        device = next(iter(encoder_inputs.values())).device
        
        decoder_input = torch.full(
            (batch_size, 1), 
            50258,
            dtype=torch.long, 
            device=device
        )
        
        encoder_outputs = self.model.encoder(encoder_inputs)
              
        hooks = []
        all_logits = []
        generated_tokens = []
        
        if self.monitor_gates:
            for i, layer in enumerate(self.model.decoder.self_attn_layers):
                layer._last_input = None
                def get_hook_fn(target_layer):
                    def hook_fn(layer, input, output):
                        target_layer._last_input = input[0].detach()
                    return hook_fn
                
                hook = layer.register_forward_hook(get_hook_fn(layer))
                hooks.append(hook)
                
                for feature_layers in self.model.decoder.processors.values():
                    for j, cross_layer in enumerate(feature_layers):
                        cross_layer._last_input = None
                        hook = cross_layer.register_forward_hook(get_hook_fn(cross_layer))
                        hooks.append(hook)
        
        for i in range(self.max_length - 1):
            logits = self.model.decoder(decoder_input, encoder_outputs)
            
            all_logits.append(logits[:, -1:, :])
            
            use_tf = use_teacher_forcing
            if use_tf is None:
                use_tf = (torch.rand(1).item() < self.scheduled_sampling_ratio)
            
            if labels is not None and i < labels.size(1) - 1 and use_tf:
                next_token = labels[:, i+1:i+2]
            else:
                next_token = self._sample_from_logits(logits[:, -1, :])
                next_token = next_token.unsqueeze(-1)
            
            if self.monitor_gates:
                for j, layer in enumerate(self.model.decoder.self_attn_layers):
                    self._collect_gate_stats(layer, i)
                
                for feature_layers in self.model.decoder.processors.values():
                    for j, cross_layer in enumerate(feature_layers):
                        self._collect_gate_stats(cross_layer, i)
                pass
            generated_tokens.append(next_token)
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            if (next_token == self.eos_token_id).all():
                break
        
        all_logits = torch.cat(all_logits, dim=1)
        generated_tokens = torch.cat(generated_tokens, dim=1)
        
        loss = None
        if labels is not None:
            shifted_labels = labels[:, 1:].contiguous()
            
            seq_length = min(all_logits.size(1), shifted_labels.size(1))
            
            loss_input = all_logits[:, :seq_length, :].reshape(-1, all_logits.size(-1))
            loss_target = shifted_labels[:, :seq_length].reshape(-1)
            
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fn(loss_input, loss_target)
        
        for hook in hooks:
            hook.remove()
        
        return_dict = {
            "loss": loss,
            "logits": all_logits
        }
        
        return return_dict

metric = evaluate.load(path="wer")

@dataclass
class DataCollator:
    tokenizer: WhisperTokenizer
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        global extractor
        spec_pad = 0
        wav_pad = 0
        batch = {}

        if "spectrogram" in features[0] and features[0]["spectrogram"] is not None:
            spectrogram_list = [f["spectrogram"] for f in features]
            max_len_feat = max(f.shape[-1] for f in spectrogram_list)
            pad_spectrogram = []
            for feat in spectrogram_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=spec_pad)
                else:
                    pad_feat = feat
                pad_spectrogram.append(pad_feat)
            batch["spectrogram"] = torch.stack(pad_spectrogram)

        if "waveform" in features[0] and features[0]["waveform"] is not None:
            waveform_list = [f["waveform"] for f in features]
            max_len_wav = max(w.shape[-1] for w in waveform_list)
            pad_waveforms = []
            for wav in waveform_list:
                current_len = wav.shape[-1]
                padding = max_len_wav - current_len
                if padding > 0:
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    pad_wav = F.pad(wav, (0, padding), mode='constant', value=wav_pad)
                else:
                    pad_wav = wav
                pad_waveforms.append(pad_wav)
            batch["waveform"] = torch.stack(pad_waveforms)

        if "labels" in features[0] and features[0]["labels"] is not None:
            labels_list = [f["labels"] for f in features]
            max_len_labels = max(len(l) for l in labels_list)            
            all_input_ids = []
            all_labels = []

            for label in labels_list:
                label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                decoder_input = [50258] + label_list                
                label_with_eos = label_list + [50256]
                input_padding_len = max_len_labels + 1 - len(decoder_input)
                label_padding_len = max_len_labels + 1 - len(label_with_eos)                
                padded_input = decoder_input + [0] * input_padding_len
                padded_labels = label_with_eos + [0] * label_padding_len                
                all_input_ids.append(padded_input)
                all_labels.append(padded_labels)        
                    
            batch["input_ids"] = torch.tensor(all_input_ids, dtype=torch.long)
            batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

        if "pitch" in features[0] and features[0]["pitch"] is not None:
            pitch_list = [f["pitch"] for f in features]
            max_len_pitch = max(e.shape[-1] for e in pitch_list)
            pad_pitch = []
            for pitch in pitch_list:
                current_len = pitch.shape[-1]
                padding = max_len_pitch - current_len
                if padding > 0:
                    pad_pitch_item = F.pad(pitch, (0, padding), mode='constant', value=spec_pad)
                else:
                    pad_pitch_item = pitch
                pad_pitch.append(pad_pitch_item)
            batch["pitch"] = torch.stack(pad_pitch)
        return batch

def downsample(wav, output_dims=64):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(1)
    input_dims = wav.shape[1]
    layer1 = nn.Conv1d(input_dims, 16, kernel_size=15, stride=8, padding=7)
    layer2 = nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3)
    layer3 = nn.Conv1d(32, output_dims, kernel_size=9, stride=5, padding=4)
    x = F.relu(layer1(wav))
    x = F.relu(layer2(x))
    x = F.relu(layer3(x))
    return x

def extract_features(batch, tokenizer, spectrogram=True, waveforms=True, pitch=True, downsample=True,
                     hop_length=128, fmin=0, fmax=8000, n_mels=128, n_fft=1024, sampling_rate=16000, pad_value=0.0,
                     pad_mode="constant", center=True, power=2.0, window_fn=torch.hann_window, mel_scale="htk", 
                     norm=None, normalized=False, debug=False):
    
    global model, extractor

    dtype = torch.float32
    device = torch.device("cuda:0")
    audio = batch["audio"]
    sampling_rate = audio["sampling_rate"]
        
    wav = torch.tensor(audio["array"]).float()
    sr = audio["sampling_rate"]
    
    if sr != sampling_rate:
        original_length = wav.shape[-1]
        target_length = int(original_length * (sampling_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = resampler(wav)
        
        if abs(wav.shape[-1] - target_length) > 1:
            new_waveform = torch.zeros((wav.shape[0], target_length), dtype=dtype, device=device)
            copy_length = min(wav.shape[1], target_length)
            new_waveform[:, :copy_length] = wav[:, :copy_length]
            wav = new_waveform

    if spectrogram:
        transform = torchaudio.transforms.MelSpectrogram(
            f_max=fmax,
            f_min=fmin,
            n_mels=n_mels,
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            norm='slaney',
            normalized=False,
            power=2.0,
            center=True, 
            mel_scale="htk",
            window_fn=torch.hann_window,  
            )
    
        mel_spectrogram = transform(wav)      
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spec = (log_mel + 4.0) / 4.0
        spec = torch.tensor(spec)

    wav_1d = wav.unsqueeze(0)
    
    if waveforms:
        if downsample:
            output_dims = 64
            downsampled_wav = downsample(wav_1d, output_dims=output_dims)
            batch["waveform"] = downsampled_wav
        else:
            downsampling_factor = 160
            downsampled_wav = wav_1d[:, ::downsampling_factor]
            batch["waveform"] = downsampled_wav

    if pitch:
        pit = torchcrepe.predict(
            wav_1d, 
            sampling_rate, 
            hop_length,
            fmin=150,
            fmax=600,
            model="tiny",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=False, 
            device=device, 
            pad=False
        )
        batch["pitch"] = pit
        
    if spectrogram:
        batch["spectrogram"] = spec
        
    batch["labels"] = tokenizer.encode(batch["transcription"], add_special_tokens=False)
    return batch

def compute_metrics(eval_pred, compute_result: bool = True, print_pred: bool = False, 
                    num_samples: int = 0, tokenizer=None, include_special_tokens: bool = True):
    global extractor, model, optimizer, scheduler
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
        display_pred_ids = pred_ids.clone()
        if include_special_tokens:
            display_pred_ids = display_pred_ids.tolist()
        else:
            display_pred_ids = display_pred_ids[:, 1:].tolist()
        metric_pred_ids = pred_ids[:, 1:].tolist()
    elif hasattr(pred_ids, "tolist"):
        if isinstance(pred_ids, torch.Tensor) and pred_ids.ndim >= 2:
            display_pred_ids = pred_ids.clone()
            if include_special_tokens:
                display_pred_ids = display_pred_ids.tolist()
            else:
                display_pred_ids = display_pred_ids[:, 1:].tolist()
            metric_pred_ids = pred_ids[:, 1:].tolist()
        else:
            display_pred_ids = pred_ids.tolist()
            metric_pred_ids = pred_ids.tolist()
    else:
        display_pred_ids = pred_ids
        metric_pred_ids = pred_ids
    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()
    label_ids = [[tokenizer.pad_token_id if token == -100 else token for token in seq] for seq in label_ids]
    display_str = tokenizer.batch_decode(display_pred_ids, skip_special_tokens=not include_special_tokens)
    metric_pred_str = tokenizer.batch_decode(metric_pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    if print_pred:
        for i in range(min(num_samples, len(display_pred_ids))):
            full_pred = tokenizer.decode(display_pred_ids[i], skip_special_tokens=False)
            clean_pred = tokenizer.decode(display_pred_ids[i], skip_special_tokens=True)
            print(f"Preds: {clean_pred}")
            print(f"Label: {label_str[i]}")
            print(f"preds: {display_pred_ids[i]}")
            print(f"label: {label_ids[i]}")
            print("")
    all_true = []
    all_pred = []
    for pred_seq, label_seq in zip(metric_pred_ids, label_ids):
        valid_indices = [i for i, token in enumerate(label_seq) if token != tokenizer.pad_token_id]
        valid_labels = [label_seq[i] for i in valid_indices if i < len(label_seq)]
        valid_preds = [pred_seq[i] for i in valid_indices if i < len(pred_seq) and i < len(valid_indices)]
        if len(valid_preds) < len(valid_labels):
            valid_preds.extend([tokenizer.pad_token_id] * (len(valid_labels) - len(valid_preds)))
        all_true.extend(valid_labels)
        all_pred.extend(valid_preds[:len(valid_labels)])
    acc = accuracy_score(all_true, all_pred) if all_true else 0
    pre = precision_score(all_true, all_pred, average='weighted', zero_division=0) if all_true else 0
    rec = recall_score(all_true, all_pred, average='weighted', zero_division=0) if all_true else 0
    f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0) if all_true else 0
    wer = 100 * metric.compute(predictions=metric_pred_str, references=label_str)
    metrics = {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1}
    return metrics

logger = logging.getLogger(__name__)

def create_model(param: Dimensions) -> Echo:
    model = Echo(param).to('cuda')
    model.init_weights()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    return model

def setup_tokenizer(token: str, local_tokenizer_path: str = "./tokenizer") -> WhisperTokenizer:
    tokenizer = WhisperTokenizer.from_pretrained(
        local_tokenizer_path,
        pad_token="<|pad|>",
        token=token,
        local_files_only=True)

    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.bos_token_id = 50258
   
    return tokenizer

def prepare_datasets(tokenizer: WhisperTokenizer, token: str, sanity_check: bool = False, dataset_config: Optional[Dict] = None) -> Tuple[any, any]:
    if dataset_config is None:
        dataset_config = {
            "spectrogram": True,
            "waveforms": True,
            "pitch": True,
            "downsample": True,
            "hop_length": 128,
            "fmin": 50,
            "fmax": 2000,
            "n_mels": 128,
            "n_fft": 1024,
            "sampling_rate": 16000,
        }
    
    dataset = load_dataset(
        "google/fleurs", 
        "en_us", 
        token=token, 
        trust_remote_code=True,
        streaming=False
    )
    
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    
    if sanity_check:
        dataset = dataset["test"].take(10)
        dataset = dataset.select_columns(["audio", "transcription"])
        logger.info(f"Sanity dataset size: {dataset.num_rows}") 
        print(f"Sanity dataset size: {dataset.num_rows}")
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        
        dataset = dataset.map(
            function=prepare_fn, 
            remove_columns=["audio", "transcription"]
        ).with_format(type="torch")
        train_dataset = dataset
        test_dataset = dataset
    else:
        def filter_func(x):
            return (0 < len(x["transcription"]) < 512 and
                   len(x["audio"]["array"]) > 0 and
                   len(x["audio"]["array"]) < 1500 * 160)
        
        dataset = dataset.filter(filter_func).shuffle(seed=4)
        logger.info(f"Dataset size: {dataset['train'].num_rows}, {dataset['test'].num_rows}")
        print(f"Dataset size: {dataset['train'].num_rows}, {dataset['test'].num_rows}")
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        columns_to_remove = list(next(iter(dataset.values())).features)
        train_dataset = train_dataset.map(
            function=prepare_fn, 
            remove_columns=columns_to_remove
        ).with_format(type="torch")
        
        test_dataset = test_dataset.map(
            function=prepare_fn, 
            remove_columns=columns_to_remove
        ).with_format(type="torch")
    
    return train_dataset, test_dataset

def get_training_args(
    log_dir: str,
    batch_eval_metrics: bool = False,
    max_steps: int = 10,
    save_steps: int = 10001,
    eval_steps: int = 100,
    warmup_steps: int = 0,
    num_train_epochs: int = 1,
    logging_steps: int = 10,
    eval_on_start: bool = False,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
) -> Seq2SeqTrainingArguments:

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
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        logging_dir=log_dir,
        logging_strategy="steps",
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_safetensors=False,
        eval_on_start=eval_on_start,
        batch_eval_metrics=batch_eval_metrics,
    )

class AudioSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer that correctly handles audio-to-text models with self-critical training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Ensure all inputs are correctly passed to the model"""
        
        batch = {}
        for key in inputs.keys():
            batch[key] = inputs[key]
            
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter < 3:
            print(f"Batch keys available: {list(batch.keys())}")
            
        outputs = model(**batch)
        
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
        return (loss, outputs) if return_outputs else loss

def main():
    
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)
    tokenizer = setup_tokenizer(token)

    def sanity(sanity: bool):

        if sanity:
            training_args = get_training_args(
            log_dir,
            batch_eval_metrics = False,
            max_steps = 10,
            save_steps = 0,
            eval_steps = 1,
            warmup_steps = 0,
            logging_steps = 10,
            eval_on_start = True,
            learning_rate = 1e-4,
            weight_decay = 0.01,
            )
        else:
            training_args = get_training_args(
            log_dir,
            batch_eval_metrics = False,
            max_steps = 10000,
            save_steps = 5000,
            eval_steps = 1000,
            warmup_steps = 500,
            logging_steps = 100,
            eval_on_start = False,
            learning_rate = 1e-4,
            weight_decay = 0.01,
            )

        return training_args
        
    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        audio_dims=512,
        audio_idx=4,
        vocab=len(tokenizer),
        text_ctx=256,
        text_head=4,
        text_dims=512,
        text_idx=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=0,
        act="gelu",
        debug=False,
        cross_attn=True,
        f0_rotary=False,
        features = ["spectrogram"],
        )
    
    sanity_check = True
    training_args = sanity(sanity_check)

    dataset_config = {
        "spectrogram": True,
        "waveforms": False,
        "pitch": False,
        "downsample": True,
        "hop_length": 128,
        "fmin": 150,
        "fmax": 2000,
        "n_mels": 128,
        "n_fft": 1024,
        "sampling_rate": 16000,
        "pad_value": 0.0,
        "pad_mode": "constant",
        "center": True, 
        "power": 2.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False,
        }
    

    metrics_fn = partial(compute_metrics, print_pred=False, num_samples=1, 
                    tokenizer=tokenizer, include_special_tokens=False)
    
    print(f"{'Sanity check' if sanity_check else 'Training'} mode")
    
    train_dataset, test_dataset = prepare_datasets(
        tokenizer=tokenizer,
        token=token,
        sanity_check=sanity_check,
        dataset_config=dataset_config)
    
    model = create_model(param)
    
    
    
    wrapped_model = SelfCriticalTraining(
        model=model,
        tokenizer=tokenizer,
        scheduled_sampling_ratio=0.7,
        monitor_gates=True,
        max_length=param.text_ctx,
        pad_token_id=param.pad_token_id,
        eos_token_id=param.eos_token_id
    )
    
    trainer = AudioSeq2SeqTrainer(
        args=training_args,
        model=wrapped_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
        compute_metrics=metrics_fn)
    
    example_batch = next(iter(trainer.get_train_dataloader()))
    print("Example batch keys:", list(example_batch.keys()))
    
    
    trainer.train()
    
    gate_stats = wrapped_model.gate_stats
    if gate_stats:
        analyze_gate_behavior(gate_stats)

    
    
