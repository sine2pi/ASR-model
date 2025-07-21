import torch
import os
import pyworld as pw
import numpy as np
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset
from datasets import Audio
from dataclasses import dataclass
from typing import Any, List, Dict
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from typing import Any, List, Dict, Optional, Union, Tuple
from torch.nn.functional import scaled_dot_product_attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# def shape(tensor: torch.Tensor, head: int, head_dim: int, batch: int, ctx: int): 
#     return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()

# def reshape_to_output(attn_output, head: int, head_dim: int, batch: int, ctx: int, dims: int):
#     return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, dims).contiguous()

def shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def reshape_to_output(self, attn_output, batch, ctx):
    return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims).contiguous()

def create_attention_mask(batch_size, ctx, is_causal=True, padding_mask=None, device=None):
    if is_causal:
        mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=0)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, ctx, ctx)
    else:
        mask = torch.zeros((batch_size, 1, ctx, ctx), device=device)
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        mask = mask | (~padding_mask)
    return mask

def mask_win(text_ctx, aud_ctx):
    mask = torch.tril(torch.ones(text_ctx, text_ctx, device=device, dtype=dtype), diagonal=0)
    audio_mask = torch.tril(torch.ones(text_ctx, aud_ctx - text_ctx, device=device, dtype=dtype))
    full_mask = torch.cat([mask, audio_mask], dim=-1)
    return full_mask

def maskc(ctx, device):
    return torch.tril(torch.ones(ctx, ctx, device=device, dtype=dtype), diagonal=0)
    
def qkv_init(dims: int, head: int):
    head_dim = dims // head
    scale = head_dim ** -0.5
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    return q, k, v, o, scale

def create_qkv(q, k, v, x, xa=None, head=8):
    head_dim = q.out_features // head
    scale = head_dim ** -0.5
    q = q(x) * scale
    k = k(xa if xa is not None else x) * scale
    v = v(xa if xa is not None else x)
    batch, ctx, _ = q.shape
    def _shape(tensor):
        return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
    return _shape(q), _shape(k), _shape(v)

def calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True):
    # q, k, v = create_qkv(q, k, v, dims, head)

    batch, head, ctx, dims = q.shape
    attn_mask = None
    if mask is not None:
        if mask.dim() <= 3:
            attn_mask = create_attention_mask(
                batch_size=batch,
                ctx=ctx,
                is_causal=is_causal,
                padding_mask=mask if mask.dim() > 1 else None,
                device=device)
        else:
            attn_mask = mask
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5
    a = scaled_dot_product_attention(scaled_q, k, v, attn_mask=attn_mask, is_causal=is_causal if attn_mask is None else False)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return out, None

# example
# class MyAttention(nn.Module):
#     def __init__(self, dims, head):
#         super().__init__()
#         self.q, self.k, self.v, self.o, self.scale = qkv_init(dims, head)
#         self.head = head

#     def forward(self, x, xa=None, mask=None, temperature=1.0):
#         q, k, v = create_qkv(self.q, self.k, self.v, x, xa, head=self.head)
#         out, _ = calculate_attention(q, k, v, mask=mask, temperature=temperature)
#         out = self.o(out)  # Final projection
#         return out

def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)

def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()

def trace_x(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            print(f"  {func.__name__} returned shape: {result.shape}")
        return result
    return wrapper

def track_x(new_x, operation=""): 
    """ track_x(x, "x") """
    x_id = [id(new_x)]
    if new_x is None:
        return new_x
    current_id = id(new_x)
    if current_id != x_id[0]:
        print(f"x FLOW: {x_id[0]} → {current_id} in {operation}")
        x_id[0] = current_id
    else:
        print(f"x REUSE: {current_id} in {operation}")
    return new_x

def track_xa(new_xa, operation=""): 
    """ track_xa(xa, "xa - decoder") """
    xa_id = [id(new_xa)] if new_xa is not None else [None]
    if new_xa is None:
        return new_xa
    current_id = id(new_xa)
    if current_id != xa_id[0]:
        print(f"xa FLOW: {xa_id[0]} → {current_id} in {operation}")
        xa_id[0] = current_id
    else:
        print(f"xa REUSE: {current_id} in {operation}")
    return new_xa

def get_activation(act: str) -> nn.Module:
    """Get activation function by name."""
    act_map = {
        "gelu": nn.GELU(), 
        "relu": nn.ReLU(), 
        "sigmoid": nn.Sigmoid(), 
        "tanh": nn.Tanh(), 
        "swish": nn.SiLU(), 
        "tanhshrink": nn.Tanhshrink(), 
        "softplus": nn.Softplus(), 
        "softshrink": nn.Softshrink(), 
        "leaky_relu": nn.LeakyReLU(), 
        "elu": nn.ELU()
    }
    return act_map.get(act, nn.GELU())

@dataclass
class Dimensions:
    vocab: int
    mels: int
    ctx: int
    dims: int
    head: int
    layer: int
    act: str
    debug: List[str]
    features: List[str]

def get_generation_config(param):
    return GenerationConfig(    # type: ignore
        max_length=param.text_ctx,
        pad_token_id=getattr(param, "pad_token_id", 0),
        bos_token_id=getattr(param, "bos_token_id", 1),
        eos_token_id=getattr(param, "eos_token_id", 2),
        do_sample=False,
        num_beams=1,
        early_stopping=False,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        temperature=1.0,
        decoder_start_token_id=1,
        is_multilingual=False,
        use_cache=False,
        return_timestamps=False)

def plot_waveform(x=None, w=None, p=None, per=None, sample_idx=0, sr=16000, hop_length=160, 
                                 title="", markers=None, marker_labels=None, 
                                 show_voiced_regions=True, show_energy=False):
    num_plots = sum([x is not None, w is not None, p is not None, per is not None])
    if num_plots == 0:
        raise ValueError("No data to plot. Please provide at least one input tensor.")
    t_spans = []
    
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t_spans.append(len(w_np) / sr)
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        t_spans.append(x_np.shape[0] * hop_length / sr)
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_spans.append(len(p_np) * hop_length / sr)
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_spans.append(len(per_np) * hop_length / sr)
    max_t = max(t_spans) if t_spans else 0
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 4*num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]
    if show_voiced_regions and per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        threshold = 0.5
        for ax in axs:
            for i in range(len(per_np)-1):
                if per_np[i] > threshold:
                    ax.axvspan(t_per[i], t_per[i+1], color='lightblue', alpha=0.2, zorder=0)
    cu_ax = 0
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        axs[cu_ax].plot(t, w_np, color="tab:blue")
        
        if show_energy:
            frame_length = hop_length
            hop_length_energy = hop_length // 2
            energy = []
            for i in range(0, len(w_np)-frame_length, hop_length_energy):
                frame = w_np[i:i+frame_length]
                energy.append(np.sqrt(np.mean(frame**2)))
            energy = np.array(energy)
            energy = energy / np.max(energy) * 0.8 * max(abs(w_np.min()), abs(w_np.max()))  
            t_energy = np.arange(len(energy)) * hop_length_energy / sr
            axs[cu_ax].plot(t_energy, energy, color="red", alpha=0.7, label="Energy")
            axs[cu_ax].legend(loc='upper right')
        axs[cu_ax].set_title("Waveform")
        axs[cu_ax].set_ylabel("Amplitude")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        cu_ax += 1
    
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        axs[cu_ax].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma", 
                                   extent=[0, x_np.shape[0]*hop_length/sr, 0, x_np.shape[1]])
        axs[cu_ax].set_title("Spectrogram")
        axs[cu_ax].set_ylabel("Mel Bin")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        cu_ax += 1
    
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_p = np.arange(len(p_np)) * hop_length / sr
        axs[cu_ax].plot(t_p, p_np, color="tab:green")
        axs[cu_ax].set_title("Pitch")
        axs[cu_ax].set_ylabel("Frequency (Hz)")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[cu_ax].set_ylim([0, min(1000, p_np.max() * 1.2)])
        cu_ax += 1
    
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        axs[cu_ax].plot(t_per, per_np, color="tab:red")
        axs[cu_ax].set_title("Period (Voice Activity)")
        axs[cu_ax].set_ylabel("periodocity")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[cu_ax].set_ylim([-0.05, 1.05])
        axs[cu_ax].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    
    if markers is not None:
        for i, t in enumerate(markers):
            label = marker_labels[i] if marker_labels and i < len(marker_labels) else None
            for ax in axs:
                ax.axvline(x=t, color='k', linestyle='-', alpha=0.7, label=label if i == 0 else None)
        if marker_labels:
            axs[0].legend(loc='upper right', fontsize='small')
    axs[-1].set_xlabel("t (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # type: ignore
    plt.show()
    return fig

def valid(default_value, *items):
    """Get first non-None item"""
    for item in items:
        if item is not None:
            return item
    return default_value

def dict_to(d, device, dtype=dtype):
    return {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}
    
def exists(v):
    return v is not None

def default(v, b):
    return v if exists(v) else b

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class RMSNorm(nn.Module):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], 
                 eps = 1e-8, elementwise_affine = True):
        super(RMSNorm, self).__init__()
        if isinstance(dims, int):
            self.normalized_shape = (dims,)
        else:
            self.normalized_shape = tuple(dims)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))  # type: ignore
            init.ones_(self.weight)  
        else:
            self.register_parameter("weight", None)
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)  # type: ignore
    
def LayerNorm(x: Tensor, normalized_shape: Union[int, Tensor, List, Tuple],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)  # type: ignore

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dtype():
    return torch.float32 if torch.cuda.is_available() else torch.float64

def tox():
    return {"device": get_device(), "dtype": get_dtype()}

class Sinusoids(nn.Module):
    def __init__(self, ctx: int, dims: int):
        super().__init__()

        position = torch.arange(start=0, end=ctx, dtype=dtype).unsqueeze(dim=1)
        div_term = torch.exp(input=torch.arange(start=0, end=dims, step=2, dtype=dtype) * -(math.log(10000.0) / dims))
        features = torch.zeros(ctx, dims)
        features[:, 0::2] = torch.sin(position * div_term)
        features[:, 1::2] = torch.cos(position* div_term)
        self.register_buffer('sinusoid', tensor=features)
        self.positional_embeddings = nn.Parameter(self.sinusoid.clone())
    def forward(self, positions):
        position_embeddings = self.positional_embeddings[positions]
        return position_embeddings

def sinusoids(length, channels, max_tscale=10000):
    assert channels % 2 == 0
    log_tscale_increment = np.log(max_tscale) / (channels // 2 - 1)
    inv_tscales = torch.exp(-log_tscale_increment * torch.arange(channels // 2))
    scaled_t = torch.arange(length)[:, np.newaxis] * inv_tscales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_t), torch.cos(scaled_t)], dim=1)

def clean_ids(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(id) for id in ids if id != -100 and id != pad_token_id and id != bos_token_id and id != eos_token_id]

def clean_batch(batch_ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    return [clean_ids(seq, pad_token_id, bos_token_id, eos_token_id) for seq in batch_ids]

def setup_tokenizer(dir: str):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{dir}/tokenizer.json")
    orig_encode = tokenizer.encode
    orig_decode = tokenizer.decode

    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]
            ids = [id for id in ids if id not in sp_ids]
        return ids

    def bdec(ids_list, pad_token_id=0, bos_token_id=1, eos_token_id=2, skip_special_tokens=True):
        results = []
        if isinstance(ids_list, torch.Tensor):
            ids_list = ids_list.tolist()
        elif isinstance(ids_list, np.ndarray):
            ids_list = ids_list.tolist()
        for ids in ids_list:
            ids = [int(id) for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id, -100)]
            results.append(orig_decode(ids))
        return results

    def dec(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        ids = [int(id) for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id, -100)]
        return orig_decode(ids)

    def save_pretrained(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(f"{save_dir}/tokenizer.json")

    tokenizer.encode = enc
    tokenizer.batch_decode = bdec
    tokenizer.decode = dec
    tokenizer.save_pretrained = save_pretrained
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer

def tokenize_pitch(pitch_features, target_length):
    pitch_len = pitch_features.shape[-1]
    token_len = target_length
    if pitch_len > token_len:
        pitch_tokens = F.adaptive_avg_pool1d(pitch_features, token_len)
    else:
        pitch_tokens = F.interpolate(pitch_features, token_len)
    return pitch_tokens

def load_wave(wave_data, sample_rate=16000):

    if isinstance(wave_data, str):
        waveform, sample_rate = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sample_rate = wave_data["sampling_rate"]  # noqa: F841
    else:
        raise TypeError("Invalid wave_data format.")
    return waveform

def world_to_mel(sp, ap, sample_rate=16000, n_mels=128):
    import librosa
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float()
    sp_mel = torch.matmul(sp, mel_basis.T)  # (frames, 128)
    ap_mel = torch.matmul(ap, mel_basis.T)  # (frames, 128)
    return sp_mel, ap_mel

def extract_features(batch, tokenizer, waveform=False, spec=False, f0=False, f0t=False, pitch=False, harmonics=False, sample_rate=16000, hop_length=256, mode="mean", debug=False, phase_mod=False, crepe=False, aperiodics=False):

    audio = batch["audio"]
    sample_rate = audio["sampling_rate"]
    labels = tokenizer.encode(batch["transcription"])
    wav = load_wave(wave_data=audio, sample_rate=sample_rate)

    spectrogram_config = {
        "hop_length": 256,
        "f_min": 150,
        "f_max": 2000,
        "n_mels": 128,
        "n_fft": 1024,
        "sample_rate": 16000,
        "pad_mode": "constant",
        "center": True, 
        "power": 1.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False,
    }

    if crepe:
        time, frequency, confidence, activation = crepe.predict(wav, sample_rate, viterbi=True)
        crepe_time = torch.from_numpy(time)
        crepe_frequency = torch.from_numpy(frequency)
        crepe_confidence = torch.from_numpy(confidence)
        crepe_activation = torch.from_numpy(activation)
    else:
        crepe_time = None
        crepe_frequency = None
        crepe_confidence = None
        crepe_activation = None

    if spec:
        transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        mel_spectrogram = transform(wav)
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        # spectrogram_tensor = mel_spectrogram.log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spectrogram_tensor = (log_mel + 4.0) / 4.0
        spectrogram_tensor = torch.tensor(spectrogram_tensor)
    else:
        spectrogram_tensor = None

    if f0 or f0t or pitch or harmonics or aperiodics:
        wavnp = wav.numpy().astype(np.float64)
        f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)

    if f0:
        f0_tensor = torch.from_numpy(f0_np)
    else:
        f0_tensor = None

    if f0t:
        wav = torch.from_numpy(wavnp)
        t2 = torch.from_numpy(t)
        audio_duration = len(wav) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        start_idx = torch.searchsorted(t2, token_starts, side="left")
        end_idx = torch.searchsorted(t2, token_ends, side="right")
        pitch_tok = torch.zeros(T, dtype=torch.float32)
        for i in range(T):
            lo, hi = start_idx[i], max(start_idx[i]+1, end_idx[i])
            segment = f0_np[lo:hi]
            if mode == "mean":
                pitch_tok[i] = segment.mean()
            elif mode == "median":
                pitch_tok[i] = torch.median(segment)
            else:
                pitch_tok[i] = segment[-1]
        pitch_tok[pitch_tok < 100.0] = 0.0
        bos_pitch = pitch_tok[0] if len(pitch_tok) > 0 else 0.0
        f0t_tensor = torch.cat([torch.tensor([bos_pitch]), pitch_tok])
        # f0t_tensor = torch.where(f0t_tensor == 0.0, torch.zeros_like(f0t_tensor), (f0t_tensor - 71.0) / (500.0 - 71.0))
    else:
        f0t_tensor = None

    if phase_mod:
        tframe = torch.mean(t2[1:] - t2[:-1])
        phi0 = 0.0
        omega = 2 * torch.pi * f0_tensor
        dphi = omega * tframe
        phi = torch.cumsum(dphi, dim=0) + phi0
        phase = torch.remainder(phi, 2 * torch.pi)
    else:
        phase = None

    if pitch:
        p_tensor = torch.from_numpy(f0_np)
    else:
        p_tensor = None

    if harmonics or aperiodics:
        spnp = pw.cheaptrick(wavnp, f0_np, t, sample_rate, fft_size=256)
        apnp = pw.d4c(wavnp, f0_np, t, sample_rate, fft_size=256)
        harmonic_tensor = torch.from_numpy(spnp)
        aperiodic_tensor = torch.from_numpy(apnp)
        harmonic_tensor = harmonic_tensor[:, :128].contiguous().T
        aperiodic_tensor = aperiodic_tensor[:, :128].contiguous().T
        harmonic_tensor = torch.where(harmonic_tensor == 0.0, torch.zeros_like(harmonic_tensor), harmonic_tensor / 1.0)
        aperiodic_tensor = torch.where(aperiodic_tensor == 0.0, torch.zeros_like(aperiodic_tensor), aperiodic_tensor / 1.0)
    else:
        harmonic_tensor = None
        aperiodic_tensor = None

    if waveform:
        wave_tensor = wav
    else:
        wave_tensor = None

    if debug:
      
        print(f"['f0']: {f0_tensor.shape if f0 else None}")
        print(f"['f0t']: {f0t_tensor.shape if f0t else None}")
        print(f"['harmonic']: {harmonic_tensor.shape if harmonics else None}")
        print(f"['aperiodic']: {aperiodic_tensor.shape if aperiodics else None}")
        print(f"['spectrogram']: {spectrogram_tensor.shape if spec else None}")
        print(f"['waveform']: {wave_tensor.shape if waveform else None}")
        print(f"['labels']: {len(labels) if labels else None}")
        print(f"['phase']: {phase.shape if phase else None}")
        print(f"['pitch']: {p_tensor.shape if pitch else None}")
        print(f"['crepe_time']: {crepe_time.shape if crepe else None}")        
        print(f"['crepe_frequency']: {crepe_frequency.shape if crepe else None}")
        print(f"['crepe_confidence']: {crepe_confidence.shape if crepe else None}")
        print(f"['crepe_activation']: {crepe_activation.shape if crepe else None}")

    return {
        "waveform": wave_tensor if waveform else None,
        "spectrogram": spectrogram_tensor if spec else None,
        "f0": f0_tensor if f0 else None,
        "f0t": f0t_tensor if f0t else None,
        "pitch": p_tensor if pitch else None,
        "harmonic": harmonic_tensor if harmonics else None,
        "aperiodic": aperiodic_tensor if aperiodics else None,  
        "labels": labels,
        "phase": phase if phase_mod else None,
        "crepe_time": crepe_time if crepe else None,
        "crepe_frequency": crepe_frequency if crepe else None,
        "crepe_confidence": crepe_confidence if crepe else None,
        "crepe_activation": crepe_activation if crepe else None,
    }

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=None, max_ctx=2048):

    if extract_args is None:
        extract_args = {
        "waveform": False,
        "spec": False,
        "f0": False,
        "f0t": False,
        "pitch": False,
        "harmonic": False,
        "aperiodic": False,
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
        "phase_mod": False,
        "crepe": False,
        }

    if load_saved:
        if cache_dir is None:
            cache_dir = "./processed_datasets"
        else:
            cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        cache_file_train = os.path.join(cache_dir, "train.arrow")
        cache_file_test = os.path.join(cache_dir, "test.arrow")

        if os.path.exists(cache_file_train) and os.path.exists(cache_file_test):
            from datasets import Dataset
            train_dataset = Dataset.load_from_disk(cache_file_train)
            test_dataset = Dataset.load_from_disk(cache_file_test)
            return train_dataset, test_dataset   

    if sanity_check:
        test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).cast_column("audio", Audio(sampling_rate=sample_rate)).take(1)

        dataset = test.map(
            lambda x: extract_features(x, tokenizer, **extract_args),
            remove_columns=test.column_names)

        train_dataset = dataset
        test_dataset = dataset
        return train_dataset, test_dataset
 
    else:

        def filter_func(x):
            return (0 < len(x["transcription"]) < max_ctx and
                    len(x["audio"]["array"]) > 0 and
                    len(x["audio"]["array"]) < max_ctx * 160)

        raw_train = load_dataset(
            "google/fleurs", "en_us", token=token, split="train", trust_remote_code=True, streaming=streaming).take(1000)
        raw_test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).take(100)

        raw_train = raw_train.filter(filter_func)
        raw_test = raw_test.filter(filter_func)
        raw_train = raw_train.cast_column("audio", Audio(sampling_rate=sample_rate))
        raw_test = raw_test.cast_column("audio", Audio(sampling_rate=sample_rate))

        train_dataset = raw_train.map(
            lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=raw_train.column_names)

        test_dataset = raw_test.map(
            lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=raw_test.column_names)
        train_dataset.save_to_disk(cache_file_train) if save_dataset is True else None
        test_dataset.save_to_disk(cache_file_test) if save_dataset is True else None

        return train_dataset, test_dataset

@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        batch = {}
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        bos_token_id = getattr(self.tokenizer, 'bos_token_id', 1)
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2)

        for key in all_keys:
            if key == "labels":
                labels_list = [f["labels"] for f in features]
                max_len = max(len(l) for l in labels_list)  # noqa: E741
                all_ids, all_labels = [], []
                for label in labels_list:
                    label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                    decoder_input = [bos_token_id] + label_list
                    label_eos = label_list + [eos_token_id]
                    input_len = max_len + 1 - len(decoder_input)
                    label_len = max_len + 1 - len(label_eos)
                    padded_input = decoder_input + [pad_token_id] * input_len
                    padded_labels = label_eos + [pad_token_id] * label_len
                    all_ids.append(padded_input)
                    all_labels.append(padded_labels)
                batch["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

            elif key in ["spectrogram", "waveform", "pitch", "harmonic", "aperiodic", "f0t", "f0", "phase", "crepe_time", "crepe_frequency", "crepe_confidence", "crepe_activation"]:
                items = [f[key] for f in features if key in f]
                items = [item for item in items if item is not None]
                if not items:  
                    continue
                items = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in items]
                max_len = max(item.shape[-1] for item in items)
                padded = []
                for item in items:
                    pad_width = max_len - item.shape[-1]
                    if pad_width > 0:
                        pad_item = F.pad(item, (0, pad_width), mode='constant', value=pad_token_id)
                    else:
                        pad_item = item
                    padded.append(pad_item)
                batch[key] = torch.stack(padded)
                # if key == "spectrogram":
                #     batch["spectrogram"] = batch[key]
        return batch

def levenshtein(reference_words, hypothesis_words):
    m, n = len(reference_words), len(hypothesis_words)
    dist_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m+1):
        dist_matrix[i][0] = i
    for j in range(n+1):
        dist_matrix[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if reference_words[i-1] == hypothesis_words[j-1]:
                dist_matrix[i][j] = dist_matrix[i-1][j-1]
            else:
                substitution = dist_matrix[i-1][j-1] + 1
                insertion = dist_matrix[i][j-1] + 1
                deletion = dist_matrix[i-1][j] + 1
                dist_matrix[i][j] = min(substitution, insertion, deletion)
    return dist_matrix[m][n]

def wer_batch(references, hypotheses):
    total_errors = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.lower().split()
        errors = levenshtein(ref_words, hyp.lower().split()) 
        total_errors += errors
        total_words += len(ref_words)
    return (total_errors / total_words) * 100 if total_words > 0 else 0.0

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0):
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(i) for i in seq if i not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(i) for i in ids if i not in (-100, pad_token_id, bos_token_id, eos_token_id)]

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    if not isinstance(pred_ids, torch.Tensor):
        pred_ids = torch.tensor(pred_ids)

    label_ids = clean(label_ids)
    pred_ids = clean(pred_ids)
    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(label_ids)

    if print_pred:
        for i in range(min(num_samples, len(pred_ids))):

            print(f"Pred tokens: {pred_ids[i]}")
            print(f"Label tokens: {label_ids[i]}")
            print(f"Pred: '{pred_str[i]}'")
            print(f"Label: '{label_str[i]}'")
            print("-" * 40)
            
    wer = wer_batch(label_str, pred_str)
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        efficiency_score = (100 - wer) / trainable_params if trainable_params > 0 else 0.0
    else:
        trainable_params = 0.0
        efficiency_score = 0.0

    return {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
    }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels