ASR encoder-decoder model with optional blending of spectrogram and waveform input. Full script with tranining loop compatable with hugging face. For testing.

This model's learnable blend (with a sigmoid-mixed parameter) between waveform and spectrogram encodings is a novel and practical way to let the model decide the optimal mix. This form of adaptive fusion is less common in open-source ASR codebases.

Blending waveform and spectrogram features has been explored in some research, but is not standard in ASR pipelines.
This learnable blend is a modern, under-explored approach addressing the waveform spectrogram debate. Initial findings of the pilot run suggest that the blending of the two significantly decreases WER compared to standalone waveform and spectrogram without significantly increasing overhead. 

This model uses 0 for padding masking and silence and no special tokens, as such the attention mechanism uses multiplicative masking instead of additive. The 0.001 is so that the model can still learn to identify silence. This gives silence tokens a tiny but meaningful attention weight rather than completely masking them out. This is conceptually sound because:

- Silence/pauses in speech carry rhythmic and semantic information.
- The 0.001 factor means silence is "whispered" to the model rather than "shouted".
- The model can learn timing patterns where pauses are meaningful.

```python


def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        scaling_factors_causal = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(1.0))

        token_ids = k[:, :, :, 0]
        scaling_factors_silence = torch.ones_like(token_ids).to(q.device, q.dtype)
        scaling_factors_silence[token_ids == 0] = 0.001
        scaling_factors_silence = scaling_factors_silence.unsqueeze(-2).expand(qk.shape).to(q.device, q.dtype)

        combined_scaling_factors = scaling_factors_causal.unsqueeze(0).to(q.device, q.dtype)  * scaling_factors_silence.to(q.device, q.dtype)
        qk *= combined_scaling_factors
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, qk.detach()
```
3 tests: spectrogram, waveform, spectrogram+waveform.
### IN PROGRESS

Full model and loop(s).. hugging face trainer and pytorch loop (for those who dont like HF).
```python


import os
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Dict, Tuple, Any, Union, List
import gzip
import base64
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor, WhisperTokenizer
import evaluate
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

def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)      

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=4096, learned_freq=True):
        super().__init__()
        self.dim = dim
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=learned_freq
        )
        self.bias = nn.Parameter(torch.zeros(max_seq_len, dim // 2))  

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

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x).to(x.device, x.dtype)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self._attention(q, k, v, mask)
        return self.out(wv), qk
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        freq = self.rotary(ctx)
        q = self.rotary.apply_rotary(q, freq)
        k = self.rotary.apply_rotary(k, freq)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if mask is None:
            causal_mask = torch.triu(torch.ones(ctx, ctx, device=q.device), diagonal=1)
        else:
            causal_mask = mask.to(q.device)

        scaling_factors_causal = torch.where(causal_mask == 1, torch.tensor(0.0), torch.tensor(1.0))

        token_ids = k[:, :, :, 0]
        scaling_factors_silence = torch.ones_like(token_ids).to(q.device, q.dtype)
        scaling_factors_silence[token_ids == 0] = 0.001
        scaling_factors_silence = scaling_factors_silence.unsqueeze(-2).expand(qk.shape).to(q.device, q.dtype)

        combined_scaling_factors = scaling_factors_causal.unsqueeze(0).to(q.device, q.dtype)  * scaling_factors_silence.to(q.device, q.dtype)
        qk *= combined_scaling_factors
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
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

class EncoderBottleneck(nn.Module):
    def __init__(self, dims, latent_dim, iterations=2):
        super().__init__()
        self.to_latent = nn.Linear(dims, latent_dim, **tox)
        self.to_decoder = nn.Linear(latent_dim, dims, **tox)
        self.to_latent_dec = nn.Linear(dims, latent_dim, **tox)
        self.to_encoder = nn.Linear(latent_dim, dims, **tox)
        self.iterations = iterations

    def forward(self, enc_x=None, dec_x=None):
        if enc_x is not None:
            if enc_x.dim() == 3:
                enc_latent = self.to_latent(enc_x.mean(dim=1))
            else:
                enc_latent = self.to_latent(enc_x)
        else:
            assert dec_x is not None, "You have to provide either enc_x or dec_x"
            enc_latent = self.to_latent(dec_x)
    
        if dec_x is not None:
            if dec_x.dim() == 3:
                dec_latent = self.to_latent_dec(dec_x.mean(dim=1))
            else:
                dec_latent = self.to_latent_dec(dec_x)
        else:
            dec_latent = torch.zeros_like(enc_latent, **tox)

        for _ in range(self.iterations):
            enc_latent = enc_latent + 0.5 * dec_latent
            dec_latent = dec_latent + 0.5 * enc_latent
        enc_to_dec = self.to_decoder(enc_latent)
        dec_to_enc = self.to_encoder(dec_latent)
        return enc_to_dec, dec_to_enc

class AudioEncoder(nn.Module):
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer, act, cross_attention = False):
        super().__init__()

        self.debug = False
        self._counter = 0
        self.dropout = 0.1

        self.bottleneck = EncoderBottleneck(dims, latent_dim=128)  
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

        _, dec_to_enc = self.bottleneck(enc_x=x)
        x = x + dec_to_enc

        for block in chain(self.blockA or []):
            x = block(x)
        self._counter += 1
        if self.debug:
            print(f"Encoder output shape: {x.shape}, x shape: {x.shape}, w shape: {w.shape}")
       
        return self.ln_enc(x)
    
class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer, cross_attention = False):
        super().__init__()
        self.debug = False
        self.dropout = 0.1

        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        self.ln_dec = RMSNorm(normalized_shape=dims)
        self.rotary = Rotary(dim=dims, learned_freq=True)
        
        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=cross_attention) for _ in range(layer)]) if layer > 0 else None)

        self.bottleneck = EncoderBottleneck(dims, latent_dim=128)  

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

        enc_to_dec, _ = self.bottleneck(dec_x=x)
        x = x + enc_to_dec

        for block in chain(self.blockA or []):
            x = block(x, xa=xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        if self.debug:
            print(f"Logits shape: {logits.shape}, x shape: {x.shape}, xa shape: {xa.shape}")

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
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id
                ).to('cuda')
            input_ids = decoder_input_ids
            if input_ids.shape[1] > self.param.text_ctx:
                raise ValueError(
                    f"Input IDs' sequence length {input_ids.shape[1]} cannot exceed the maximum allowed length of {self.param.text_ctx} tokens."
                )

        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

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

def shift_with_zeros(input_ids: torch.Tensor) -> torch.Tensor:
    """Shift tokens right, using zeros as both start and end markers."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()   
    return shifted_input_ids

metric = evaluate.load(path="wer")

@dataclass
class DataCollator:
    extractor: Any
    tokenizer: Any
    decoder_start_token_id: Any

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        batch = {}
        self.debug = False

        if "input_features" in features[0]:
            batch["input_features"] = torch.stack([f["input_features"] for f in features])
        if "waveform" in features[0]:
            batch["waveform"] = torch.stack([f["waveform"] for f in features])
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), 0)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch["decoder_input_ids"] = shift_with_zeros(labels)

        return batch
    
def prepare_dataset(batch, input_features=True, waveform=True):
    audio = batch["audio"]
    fixed_len = 1500 * 160
    wav = torch.tensor(audio["array"]).float()
    if wav.shape[-1] < fixed_len:
        wav = F.pad(wav, (0, fixed_len - wav.shape[-1]))
    else:
        wav = wav[..., :fixed_len]
    if waveform:
        batch["waveform"] = wav.unsqueeze(0)

    if input_features:
        features = extractor(wav.numpy(), sampling_rate=audio["sampling_rate"]).input_features[0]
        features = torch.tensor(features)
        pad_val = features.min().item() 
        features = torch.where(features == pad_val, torch.tensor(0.0, dtype=features.dtype), features)
        target_shape = (128, 1500)
        padded = torch.zeros(target_shape, dtype=features.dtype)
        padded[:, :features.shape[1]] = features[:, :target_shape[1]]
        batch["input_features"] = padded
        batch["labels"] = tokenizer(batch["transcription"], add_special_tokens=False).input_ids
    return batch

def compute_metrics(eval_pred, compute_result: bool = True):
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

    pred_flat = list(chain.from_iterable(pred_ids))
    labels_flat = list(chain.from_iterable(label_ids))
    mask = [l != tokenizer.pad_token_id for l in labels_flat]

    acc = accuracy_score(
        [l for l, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m]
    )
    pre = precision_score(
        [l for l, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    rec = recall_score(
        [l for l, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    f1 = f1_score(
        [l for l, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }

if __name__ == "__main__":

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
        decoder_start_token_id = 0,
        pad_token_id = 0,
        eos_token_id = 0,
        act = "gelu",
        )

    model = Echo(param).to('cuda')

    token=""
    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", token=token, feature_size=128, sampling_rate=16000, do_normalize=True, return_tensors="pt", chunk_length=15, padding_value=0.0)
    
    tokenizer = WhisperTokenizer.from_pretrained(
        "./tokenizer", 
        pad_token="0", bos_token="0", eos_token="0", unk_token="0",
        token=token, local_files_only=True, pad_token_id=0
    )

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decoder_start_token_id = 0

    data_collator = DataCollator(extractor=extractor,
        tokenizer=tokenizer, decoder_start_token_id=tokenizer.decoder_start_token_id)

    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)

    dataset = DatasetDict()
    dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)

    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    dataset = dataset.filter(lambda x: len(x["transcription"]) < 512)
    dataset = dataset.filter(lambda x: len(x["transcription"]) > 0)
    dataset = dataset.filter(lambda x: len(x["audio"]["array"]) > 0)
    dataset = dataset.filter(lambda x: len(x["audio"]["array"]) < 1500 * 160)
    print("Dataset size:", dataset["train"].num_rows, dataset["test"].num_rows)

    dataset = dataset.map(function=prepare_dataset,
        remove_columns=list(next(iter(dataset.values())).features)).with_format(type="torch")
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"].select(range(100))

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
        max_steps=10000,
        save_steps=10000,
        eval_steps=1000,
        warmup_steps=1000,
        num_train_epochs=1,
        logging_steps=1,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],

        optim="adamw_torch",
        learning_rate=0.00025,
        weight_decay=0.2,
        save_safetensors=True,
        eval_on_start=False,
        include_num_input_tokens_seen=True,
        include_tokens_per_second=True,
        batch_eval_metrics=True,
        group_by_length=False,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    trainer.train()





```
