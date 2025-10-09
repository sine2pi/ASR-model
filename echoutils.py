import torch, torchaudio, torch.nn.functional as F
import os, math
import pyworld as pw
from einops.layers.torch import Rearrange
from einops import rearrange
from torch import Tensor, nn, einsum
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from functools import lru_cache
from subprocess import CalledProcessError, run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def l2norm(t):
    return F.normalize(t, dim = -1)

def have(a):
    return a is not None
    
def Sequential(*modules):
    return nn.Sequential(*filter(have, modules))    

def aorb(a, b):
    return a if have(a) else b

class InstanceRMS(nn.Module):
    def __init__(n, dims, eps=1e-6):
        super().__init__()
        n.instance_norm = nn.InstanceNorm1d(
            dims,
            eps=eps,
            affine=False,
            track_running_stats=False
        )
        n.gamma = nn.Parameter(torch.ones(dims))

    def forward(n, x):
        normalized_x = n.instance_norm(x)
        gamma_reshaped = n.gamma.view(1, -1, 1)
        output = normalized_x * gamma_reshaped
        return output

def get_norm(n_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if n_type in ["batchnorm", "instancenorm"] and dims is None:
        raise ValueError(f"'{n_type}' requires 'dims'.")
    if n_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{n_type}' requires 'num_groups'.")

    norm_map = {
        "layernorm": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        "instanceRMS": lambda: InstanceRMS(dims=dims),   
        "instancenorm": lambda: nn.InstanceNorm1d(num_features=dims, affine=False, track_running_stats=False),     
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=dims),
        "instancenorm2d": lambda: nn.InstanceNorm2d(num_features=dims),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=dims),
        }
   
    norm_func = norm_map.get(n_type)
    if norm_func:
        return norm_func()
    else:
        print(f"Warning: Norm type '{n_type}' not found. Returning LayerNorm.")
        return nn.LayerNorm(dims) 

def get_activation(act: str) -> nn.Module:

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
        "elu": nn.ELU(),
    }
    return act_map.get(act, nn.GELU())

def sinusoids(ctx, dims, theta=10000):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, n_type, norm=False, enc=False):
        super().__init__()
        
        n.act_fn = get_activation(act)
        n.theta = nn.Parameter(torch.tensor(10000.0), requires_grad=True)

        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        
        n.encoder = nn.Sequential(
            n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), 
            n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   
        
        n.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()
        n.ln = get_norm(n_type, dims) if norm else nn.Identity()
               
    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)
        x = n.encoder(x).permute(0, 2, 1).contiguous().to(device, dtype)
        x = x + n.audio(x.shape[1], x.shape[-1], n.theta).to(device, dtype)
        x = n.ln(x)
        return n.EncoderLayer(x)

class TextEncoder(nn.Module):
    def __init__(n, tokens, dims, head, n_type, norm=False):
        super().__init__()

        n.norm = norm
        n.embedding = nn.Embedding(tokens, dims)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True)

        if n.norm:
            n.ln = get_norm(n_type, dims) 
        else: 
            n.ln = None 

    def forward(n, x):
        x = n.embedding(x)
        if n.norm:
            x = n.ln(x)
        return n.EncoderLayer(x)

class ScaledPositions(nn.Module):
    def __init__(n, ctx, dims, layer):
        super().__init__()

        n.scales = [(10000.0 * (i + 1)) for i in range(layer)]
        n.encodings = nn.ParameterList([nn.Parameter(n._sinusoids(ctx, dims, scale), requires_grad=True) for scale in n.scales])
    
    def _sinusoids(n, length, dims, theta):
        tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * 
        torch.arange(dims // 2, device=device, dtype=torch.float32))
        scaled = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
    
    def forward(n, x, layer_idx):
        ctx = x.shape[1]
        return x + n.encodings[layer_idx][:ctx, :]

def apply_radii(freqs, x, ctx):
    F = x.shape[0] / ctx
    idx = torch.arange(ctx, device=device)
    idx = (idx * F).long().clamp(0, x.shape[0] - 1)
    x = x[idx]
    return torch.polar(x.unsqueeze(-1), freqs)

def compute_freqs_wideband(dims, head):
    head_dim = dims // head
    max_freq = 8000.0
    mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
    mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 700 * mel_scale / 1000

def compute_freqs_variable(dims, head, max_freq=10000.0):
    head_dim = dims // head
    mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
    mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 700 * mel_scale / 1000

def compute_freqs(dims, head):
    head_dim = dims // head
    mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), head_dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 200 * mel_scale / 1000

def compute_freqs_gammatone(dims, head, min_freq=200.0, max_freq=8000.0):
    head_dim = dims // head
    freqs = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
    return freqs / 1000

class FEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, n_type, norm=False):
        super().__init__()

        n.norm = norm
        n.act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        n.encoder = nn.Sequential(n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), n.act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   
        
        if n.norm:
            n.ln = get_norm(n_type=n_type, dims=dims) 
        else: 
            n.ln = None

    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)
        if n.norm:
            x = n.ln(x)
        return n.encoder(x).permute(0, 2, 1).contiguous().to(device=device, dtype=dtype)

class WEncoder(nn.Module):
    def __init__(n, dims, head, act, downsample=[]):
        super().__init__()
        
        n.head = head
        n.head_dim = dims // head
        n.dims = dims
        act_fn = get_activation(act)

        if downsample == "1":
            n.encoder = nn.Sequential(
                nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
                nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
                nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if downsample == "2":
            n.encoder = nn.Sequential(
                nn.Conv1d(1, dims, kernel_size=127, stride=64, bias=False), act_fn,
                nn.Conv1d(dims, 2 * dims, kernel_size=7, stride=3), act_fn,
                nn.Conv1d(2 * dims, dims, kernel_size=3, stride=2), act_fn,
                nn.GroupNorm(num_groups=1, num_channels=dims, eps=1e-5))
            
        if downsample == "3":
            n.encoder = nn.Sequential(
               nn.Conv1d(1, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
               nn.Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
               nn.Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)
                
    def forward(n, x, target_length=0, use_interpolate=False):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = n.encoder(x).permute(0, 2, 1).contiguous()
        if x.shape[1] > target_length > 0:
            if use_interpolate:
                x = F.interpolate(x.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
            else:
                x = F.adaptive_avg_pool1d(x.transpose(1, 2), target_length).transpose(1, 2)
        return x

def clean_ids(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(id) for id in ids if id != -100 and id != pad_token_id and id != bos_token_id and id != eos_token_id]

def clean_batch(batch_ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    return [clean_ids(seq, pad_token_id, bos_token_id, eos_token_id) for seq in batch_ids]

def setup_tokenizer(dir: str):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{dir}")
    orig_encode = tokenizer.encode
    orig_decode = tokenizer.decode

    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<UNK>, <PAD>", "<BOS>", "<EOS>"]]
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

def tokenize_feature(audio, labels):
    if isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            ctx = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            ctx = audio.unsqueeze(1)
        elif audio.dim() == 3:
            ctx = audio
    target_length = len(labels)
    current_length = ctx.shape[-1]
    if current_length > target_length:
        tokens = F.adaptive_avg_pool1d(ctx, target_length)
    else:
        tokens = F.interpolate(ctx, size=target_length, mode='linear', align_corners=False)
    return tokens

def exact_div(x, y):
    assert x % y == 0
    return x // y

def shrink_wav(audio, sample_rate=16000, hop_length=160):
    frames_per_second = exact_div(sample_rate, hop_length)
    target_length = int((len(audio) / sample_rate) * frames_per_second)
    current_length = audio.shape[-1]    

    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)  
    if current_length > target_length:
        waveform = F.adaptive_avg_pool1d(audio, target_length)
    else:
        waveform = F.interpolate(audio, size=target_length, mode='linear', align_corners=False)    
    if waveform.dim() == 3 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    return waveform

def load_wave(audio, sample_rate=16000):
    if isinstance(audio, str):
        waveform, sample_rate = torchaudio.load(uri=audio, normalize=True, backend="ffmpeg")
    elif isinstance(audio, dict):
        waveform = torch.tensor(data=audio["array"]).float()
        sample_rate = audio["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    return waveform, sample_rate

def load_audio(file: str, sr: int = 16000):
    cmd = [ "ffmpeg", "-nostdin", "-threads", "0", "-q", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-" ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array, length: int = 30, *, axis: int = -1):
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = "D:/newmodel/mod6/mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def mel_spectrogram_b(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    n_fft = 400,
    hop_length = 160,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    s_tensor = (log_spec + 4.0) / 4.0
    return s_tensor

def harmonics_and_aperiodics(audio, sample_rate, hop_length):
    import pyworld as pw
    wav_np = audio.numpy().astype(np.float64)
    f0_np, t = pw.dio(wav_np, sample_rate, frame_period=hop_length / sample_rate * 1000)
    f0_np = pw.stonemask(wav_np, f0_np, t, sample_rate)     
    sp = pw.cheaptrick(wav_np, f0_np, t, sample_rate, fft_size=256)
    ap = pw.d4c(wav_np, f0_np, t, sample_rate, fft_size=256)
    h_tensor = torch.from_numpy(sp)
    a_tensor = torch.from_numpy(ap)
    h_tensor = h_tensor[:, :128].contiguous().T
    a_tensor = a_tensor[:, :128].contiguous().T
    h_tensor = torch.where(h_tensor == 0.0, torch.zeros_like(h_tensor), h_tensor / 1.0)
    a_tensor = torch.where(a_tensor == 0.0, torch.zeros_like(a_tensor), a_tensor / 1.0)
    return h_tensor, a_tensor

def mfcc(audio, sample_rate, n_mels, n_fft, hop_length, window_fn=torch.hann_window):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mels,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window_fn": window_fn,
            "n_mels": n_mels,
            "center": True,
            "pad_mode": "reflect",
            "norm": None,
            "mel_scale": "htk",
        }
    )
    mfcc_tensor = transform(audio)
    return mfcc_tensor

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def pitch_tokens(audio, labels, sample_rate=16000, hop_length=160, mode="mean", audio_bos=None):
    f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
    f0 = pw.stonemask(audio.numpy().astype(np.float64), f0, t, sample_rate)
    duration = len(audio) / sample_rate
    T = len(labels)
    tok_dur = duration / T
    starts = torch.arange(T) * tok_dur
    ends = starts + tok_dur
    start = torch.searchsorted(torch.from_numpy(t), starts, side="left")
    end = torch.searchsorted(torch.from_numpy(t), ends, side="right")
    ptok = torch.zeros(T, dtype=torch.float32)
    for q in range(T):
        lo, hi = start[q], max(start[q]+1, end[q])
        seg = torch.from_numpy(f0)[lo:hi]
        if mode == "mean":
            ptok[q] = seg.mean()
        elif mode == "median":
            ptok[q] = torch.median(seg)
        else:
            ptok[q] = seg[-1]
    ptok[ptok < 100.0] = 0.0
    bos_token = audio_bos if audio_bos is not None else (ptok[0] if len(ptok) > 0 else 0.0)
    tensor = torch.cat([torch.tensor([bos_token]), ptok])
    return torch.where(tensor == 0.0, torch.zeros_like(tensor), (tensor - 71.0) / (500.0 - 71.0))

def pitch_tokens2(audio: torch.Tensor, sample_rate: int, labels: list, hop_length: int, mode: str = "mean"):
    wavnp = audio.numpy().astype(np.float64)
    f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
    f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)

    audio_duration = len(audio) / sample_rate
    T = len(labels)
    tok_dur_sec = audio_duration / T
    token_starts = torch.arange(T) * tok_dur_sec
    token_ends = token_starts + tok_dur_sec
    
    f0_tensor = torch.from_numpy(f0_np)
    t_tensor = torch.from_numpy(t)
    start_idx = torch.searchsorted(t_tensor, token_starts, side="left")
    end_idx = torch.searchsorted(t_tensor, token_ends, side="right")
    pitch_tok = torch.zeros(T, dtype=torch.float32)

    for q in range(T):
        lo, hi = start_idx[q], max(start_idx[q] + 1, end_idx[q])
        segment = f0_tensor[lo:hi]
        
        voiced_segment = segment[segment > 0]
        
        if len(voiced_segment) > 0:
            if mode == "mean":
                pitch_tok[q] = voiced_segment.mean()
            elif mode == "median":
                pitch_tok[q] = torch.median(voiced_segment)
            else:
                pitch_tok[q] = voiced_segment[-1]

    mean_pitch = pitch_tok[pitch_tok > 0].mean() if (pitch_tok > 0).any() else 0.0
    std_pitch = pitch_tok[pitch_tok > 0].std() if (pitch_tok > 0).any() else 1.0
    pt_tensor = (pitch_tok - mean_pitch) / (std_pitch + 1e-6)
    bos_pitch = pt_tensor[0] if len(pt_tensor) > 0 else 0.0
    pt_tensor = torch.cat([torch.tensor([bos_pitch]), pt_tensor])
    return pt_tensor

def mel_spectrogram(audio):

    spectrogram_config = {
        "hop_length": 128,
        "f_min": 150,
        "f_max": 2000,
        "n_mels": 128,
        "n_fft": 512,
        "sample_rate": 16000,
        "pad_mode": "constant",
        "center": True, 
        "power": 2.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False,
    }

    transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
    mel_spectrogram = transform(audio)
    log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    s_tensor = (log_mel + 4.0) / 4.0
    return s_tensor
    
reduced_spec_config = {
    "hop_length": 160,
    "f_min": 50,
    "f_max": 800,
    "n_mels": 12,
    "n_fft": 2048,
    "sample_rate": 16000,
    "pad_mode": "reflect",
    "center": True,
    "power": 2.0,
    "window_fn": torch.hann_window,
    "mel_scale": "htk",
    "norm": None,
    "normalized": False,
}

def get_reduced_log_mel_spectrogram(audio):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(**reduced_spec_config)(audio)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spectrogram)
    return log_mel_spectrogram

def hilbert_transform(x):
    N = x.shape[-1]
    xf = torch.fft.rfft(x)
    h = torch.zeros(N // 2 + 1, device=device, dtype=dtype)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return torch.fft.irfft(xf * h, n=N)

def analytic_signal(x):
    return x + 1j * hilbert_transform(x)

def process_spectrogram_with_hilbert(spec):
    analytic = spec + 1j * hilbert_transform_true_2d(spec)
    envelope = torch.abs(analytic)
    phase = torch.angle(analytic)
    return envelope, phase

def hilbert_transform_2d(x, dim=-1):
    N = x.shape[dim]
    if dim == -1 or dim == len(x.shape) - 1:
        xf = torch.fft.rfft(x)
    else:
        xf = torch.fft.rfft(x, dim=dim)
    h_shape = [1] * len(x.shape)
    h_shape[dim] = N // 2 + 1
    h = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
    if dim == -1 or dim == len(x.shape) - 1:
        if N % 2 == 0:
            h[..., 0] = h[..., -1] = 1
            h[..., 1:-1] = 2
        else:
            h[..., 0] = 1
            h[..., 1:] = 2
    else:
        pass
    return torch.fft.irfft(xf * h, n=N, dim=dim)

def hilbert_transform_true_2d(x):
    xf = torch.fft.rfft2(x)
    h1 = torch.fft.fftfreq(x.shape[-2]) * 2 - 1  
    h2 = torch.fft.rfftfreq(x.shape[-1]) * 2 - 1 
    h1, h2 = torch.meshgrid(h1, h2, indexing='ij')
    h = -1j / (math.pi * (h1 + 1j * h2))
    h = h.to(x.device)
    h[0, 0] = 0
    return torch.fft.irfft2(xf * h, s=(x.shape[-2], x.shape[-1]))

def hilbert_transform(x):
    N = x.shape[-1]
    xf = torch.fft.rfft(x)
    xf = xf.to(device, dtype)
    h = torch.zeros(N // 2 + 1, device=device, dtype=dtype)
    h[0] = 1
    if N % 2 == 0:
        h[-1] = 1
    h[1:-1] = 2
    return torch.fft.irfft(xf * h, n=N)

def get_hilbert_features(x):
    x = x.to(device, dtype)
    analytic_signal = torch.complex(x, hilbert_transform(x))
    amplitude = torch.abs(analytic_signal)
    phase = torch.angle(analytic_signal)
    unwrapped_phase = torch.zeros_like(phase)
    unwrapped_phase[..., 0] = phase[..., 0]
    diff = torch.diff(phase, dim=-1)
    jumps = torch.where(diff > torch.pi, -2 * torch.pi, torch.where(diff < -torch.pi, 2 * torch.pi, 0))
    unwrapped_phase[..., 1:] = phase[..., 1:] + torch.cumsum(jumps, dim=-1)
    frequency = torch.diff(unwrapped_phase, prepend=unwrapped_phase[..., :1])
    return amplitude, frequency

def extract_features(batch, tokenizer, spectrogram=False, pitch=False, waveform=False, 
                    harmonics=False, aperiodics=False, phase=False, hilbert=False, pitch_tokens=False, hop_length=160, sample_rate=16000):
    
    mode = "mean"
    debug = False
    dummy_audio = False
    dummy_text = False

    if dummy_text:
        labels = [1] * 32
    else:
        labels = tokenizer.encode(batch["transcription"])

    if dummy_audio:
        dummy, _ = load_wave(batch["audio"], sample_rate)
        audio = torch.zeros_like(dummy)
    else:
        audio, _ = load_wave(batch["audio"], sample_rate)

    if pitch_tokens:
        pt_tensor = pitch_tokens2(audio, sample_rate, labels, hop_length, mode=mode).to(device, dtype)

    if harmonics:
        h_tensor, a_tensor = harmonics_and_aperiodics(audio, sample_rate, hop_length)
        h_tensor = h_tensor.to(device, dtype)
        a_tensor = a_tensor.to(device, dtype)

    if pitch:
        f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0 = pw.stonemask(audio.numpy().astype(np.float64), f0, t, sample_rate)
        p_tensor = torch.from_numpy(f0).unsqueeze(0).to(device, dtype)
            
    if phase:
        wavnp = audio.numpy().astype(np.float64)
        f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
        tensor = torch.from_numpy(f0_np)
        t2 = torch.from_numpy(t)
        tframe = torch.mean(t2[1:] - t2[:-1])
        phi0 = 0.0
        omega = 2 * torch.pi * tensor
        dphi = omega * tframe
        phi = torch.cumsum(dphi, dim=0) + phi0
        ph_tensor = torch.remainder(phi, 2 * torch.pi).to(device, dtype) 

    if spectrogram:
        s_tensor = mel_spectrogram(audio.float()).to(device, dtype)

    if hilbert:
        phase_tensor, _ = get_hilbert_features(audio)
        phase_tensor = phase_tensor.to(device, dtype)

    if waveform:
        current = audio.shape[-1]  
        target = int((len(audio) / sample_rate) * exact_div(sample_rate, hop_length))
            
        if audio.dim() == 1:
            aud = audio.unsqueeze(0).unsqueeze(0) 
        elif audio.dim() == 2:
            aud = audio.unsqueeze(0)  

        if current > target:
            w_tensor = F.adaptive_avg_pool1d(aud, target)
        else:
            w_tensor = F.interpolate(aud, size=target, mode='linear', align_corners=False)

        if w_tensor.dim() == 3 and w_tensor.shape[0] == 1:
            w_tensor = w_tensor.squeeze(0).to(device, dtype)
        else:
            w_tensor = w_tensor.to(device, dtype)

    if debug:
        print(f"['pitch_tokens']: {pt_tensor.shape if pitch_tokens else None}")
        print(f"['harmonic']: {h_tensor.shape if harmonics else None}")
        print(f"['aperiodic']: {a_tensor.shape if aperiodics else None}")
        print(f"['spectrogram']: {s_tensor.shape if spectrogram else None}")
        print(f"['waveform']: {w_tensor.shape if waveform else None}")
        print(f"['labels']: {len(labels) if labels else None}")
        print(f"['phase']: {ph_tensor.shape if phase else None}")
        print(f"['pitch']: {p_tensor.shape if pitch else None}")
        print(f"['hilbert']: {phase_tensor.shape if hilbert else None}")

    return {
        "waveform": w_tensor if waveform else None,
        "spectrogram": s_tensor if spectrogram else None,
        "pitch_tokens": pt_tensor if pitch_tokens else None,
        "pitch": p_tensor if pitch else None,
        "harmonic": h_tensor if harmonics else None,
        "aperiodic": a_tensor if aperiodics else None,
        "labels": labels if labels is not None else None,
        "phase": ph_tensor if phase else None,
        "hilbert": phase_tensor if hilbert else None,
    }

@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(n, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        batch = {}
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

        for key in all_keys:
            if key == "labels":
                labels_list = [f["labels"] for f in features]
                max_len = max(len(l) for l in labels_list)
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

            elif key in ["spectrogram", "waveform", "pitch", "harmonic", "aperiodic", "pitch_tokens", "hilbert", "phase"]:

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

        return batch

def levenshtein(reference_words, hypothesis_words):
    m, n = len(reference_words), len(hypothesis_words)
    dist_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for q in range(m+1):
        dist_matrix[q][0] = q
    for k in range(n+1):
        dist_matrix[0][k] = k
    for q in range(1, m+1):
        for k in range(1, n+1):
            if reference_words[q-1] == hypothesis_words[k-1]:
                dist_matrix[q][k] = dist_matrix[q-1][k-1]
            else:
                substitution = dist_matrix[q-1][k-1] + 1
                insertion = dist_matrix[q][k-1] + 1
                deletion = dist_matrix[q-1][k] + 1
                dist_matrix[q][k] = min(substitution, insertion, deletion)
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

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, logits=None, compute_result = False):
    
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(q) for q in seq if q not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(q) for q in ids if q not in (-100, pad_token_id, bos_token_id, eos_token_id)]

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
        for q in range(min(num_samples, len(pred_ids))):

            print(f"Pred tokens: {pred_ids[q]}")
            print(f"Label tokens: {label_ids[q]}")
            print(f"Pred: '{pred_str[q]}'")
            print(f"Label: '{label_str[q]}'")
            print("-" * 40)

    wer = wer_batch(label_str, pred_str)
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
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

class attention(nn.Module):
    def __init__(n, dims: int, head: int, layer: int, n_type):
        super().__init__()

        n.q   = nn.Sequential(get_norm(n_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(dims, dims), nn.Dropout(0.01))
        
        n.ln = get_norm(n_type, dims // head)
        
    def forward(n, x, xa=None, mask=None):

        q = n.q(x)
        k, v = n.kv(aorb(xa, x))
        _, _, c, d = q.shape 
        scale = d ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', n.ln(q), n.ln(k)) * scale
        if have(mask):
            qk = qk + mask[:c, :c]
        qk = F.softmax(qk, dim=-1, dtype=dtype)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        return n.out(wv)

class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        n.head  = head
        n.hdim  = dims // head
        n.scale = scale                      
        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)
    def forward(n, x: torch.Tensor, xa: torch.Tensor) -> torch.Tensor | None:
        B, Q, _ = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, Q, n.head, n.hdim).transpose(1,2)  
        k = n.k(xa).view(B, K, n.head, n.hdim).transpose(1,2)
        return (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.hdim) 
    
from transformers.trainer_seq2seq import Seq2SeqTrainer
class GradNormTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oneshot_modules = [m for m in self.model.modules() if isinstance(m, OneShot)]
        self.grad_history = []
        self.step_counter = 0
        
    def training_step(self, model, inputs):
        """Override to track gradient norm and adjust OneShot modules"""
        loss = super().training_step(model, inputs)
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.grad_history.append(total_norm)
        
        if len(self.grad_history) > 10:
            recent_avg = sum(self.grad_history[-5:]) / 5
            prev_avg = sum(self.grad_history[-10:-5]) / 5
            
            for module in self.oneshot_modules:
                if recent_avg > prev_avg * 1.2:
                    module.scale *= 0.9
                elif recent_avg < prev_avg * 0.8:
                    module.scale *= 1.1
                    
            if len(self.grad_history) > 100:
                self.grad_history = self.grad_history[-100:]
                
        self.step_counter += 1
        
        if self.step_counter % 10 == 0:
            self.log({"oneshot_scale": self.oneshot_modules[0].scale if self.oneshot_modules else 0})
            
        return loss

class GradientMonitor:
    def __init__(self, model, oneshot_modules):
        self.model = model
        self.oneshot_modules = oneshot_modules
        self.grad_norms = []
        self.hooks = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.hooks.append(param.register_hook(
                    lambda grad, param_name=name: self._hook_fn(grad, param_name)
                ))
    
    def _hook_fn(self, grad, name):
        if grad is not None:
            norm = grad.norm().item()
            self.grad_norms.append((name, norm))
        return grad
    
    def step(self):
        """Call this after backward() but before optimizer.step()"""
        if not self.grad_norms:
            return
            
        total_norm = sum(norm for _, norm in self.grad_norms)
        
        for module in self.oneshot_modules:
            if total_norm > 10.0:
                module.scale *= 0.9
            else:
                module.scale *= 1.1
                
        self.grad_norms = []
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, 
                num_epochs=5, grad_accum_steps=1, device="cuda"):
    model.to(device)
    oneshot_modules = [m for m in model.modules() if isinstance(m, OneShot)]
    grad_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs["loss"] / grad_accum_steps
            loss.backward()
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_history.append(total_norm)
            
            if len(grad_history) > 10:
                recent_avg = sum(grad_history[-5:]) / 5
                prev_avg = sum(grad_history[-10:-5]) / 5
                
                for module in oneshot_modules:
                    if recent_avg > prev_avg * 1.2:
                        module.scale *= 0.9
                    elif recent_avg < prev_avg * 0.8:
                        module.scale *= 1.1
            
            if (steps + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            steps += 1
            total_loss += loss.item() * grad_accum_steps
            
            if steps % 10 == 0:
                print(f"Epoch {epoch+1}, Step {steps}, Loss: {total_loss/steps:.4f}, "
                      f"OneShot scale: {oneshot_modules[0].scale if oneshot_modules else 0:.4f}")
        
        model.eval()
        
def train_and_evaluate(
    model,
    tokenizer,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    loss_fn,
    max_steps=10000,
    device="cuda",
    accumulation_steps=1,
    clear_cache=True,
    log_interval=10,
    eval_interval=100,
    save_interval=1000,
    checkpoint_dir="checkpoint_dir",
    log_dir="log_dir",
):
    import time
    import logging        
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    
    oneshot_modules = [m for m in model.modules() if isinstance(m, OneShot)]
    grad_history = []
    
    global_step = 0
    scaler = torch.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0

    progress_bar = tqdm(total=max_steps, desc="Training Progress", leave=True, colour="green")
    progress_bar.update(global_step)
    model.train()
    optimizer.zero_grad()

    while global_step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            dataset_epochs += 1
            print(f"Starting dataset epoch {dataset_epochs}")

            if step_in_report > 0:
                avg_loss = total_loss / step_in_report
                logging.info(
                    f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}"
                )
                total_loss = 0
                step_in_report = 0

        start_time = time.time()

        input_features = batch["input_features"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].long().to(device)

        with torch.autocast(device_type="cuda"):
            input_features_encoded = model.encoder(input_features)
            decoder_output = model.decoder(input_ids, input_features_encoded)
            logits = decoder_output.view(-1, decoder_output.size(-1))
            active_logits = logits.view(-1, decoder_output.size(-1))
            active_labels = labels.view(-1)
            active_mask = active_labels != -100
            active_logits = active_logits[active_mask]
            active_labels = active_labels[active_mask]
            loss = loss_fn(active_logits, active_labels)
        
        total_loss += loss.item()
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            grad_history.append(total_norm)
            
            if len(grad_history) > 10:
                recent_avg = sum(grad_history[-5:]) / 5
                prev_avg = sum(grad_history[-10:-5]) / 5
                
                for module in oneshot_modules:
                    if recent_avg > prev_avg * 1.2:
                        module.scale *= 0.9
                        logging.info(f"Reducing OneShot scale to {module.scale:.4f} (grad norm: {total_norm:.2f})")
                    elif recent_avg < prev_avg * 0.8:
                        module.scale *= 1.1
                        logging.info(f"Increasing OneShot scale to {module.scale:.4f} (grad norm: {total_norm:.2f})")
                
                if len(grad_history) > 100:
                    grad_history = grad_history[-100:]

            if global_step % log_interval == 0:
                writer.add_scalar("GradNorm", total_norm, global_step)
                if oneshot_modules:
                    writer.add_scalar("OneShot/scale", oneshot_modules[0].scale, global_step)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = len(batch["input_features"]) / (end_time - start_time)

        if global_step % log_interval == 0:
            writer.add_scalar(
                tag="Loss/train",
                scalar_value=total_loss / (global_step + 1),
                global_step=global_step,
            )
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar(
                tag="LearningRate", scalar_value=lr, global_step=global_step
            )
            writer.add_scalar(
                tag="SamplesPerSec",
                scalar_value=samples_per_sec,
                global_step=global_step,
            )

        if global_step % eval_interval == 0:
            model.eval()
            eval_start_time = time.time()
            eval_loss = 0
            all_predictions = []
            all_labels = []
            batch_count = 0
            total_samples = 0

            with torch.no_grad():
                for eval_batch in eval_loader:
                    input_features = eval_batch["input_features"].to(device)
                    input_ids = eval_batch["input_ids"].to(device)
                    labels = eval_batch["labels"].long().to(device)

                    batch = input_features.size(0)
                    total_samples += batch

                    input_features_encoded = model.encoder(input_features)
                    decoder_output = model.decoder(input_ids, input_features_encoded)
                    logits = decoder_output.view(-1, decoder_output.size(-1))
                    loss = loss_fn(logits, labels.view(-1))
                    eval_loss += loss.item()
                    all_predictions.extend(
                        torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist()
                    )
                    all_labels.extend(labels.cpu().numpy().tolist())
                    batch_count += 1

            eval_time = time.time() - eval_start_time
            loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {
                "predictions": np.array(all_predictions, dtype=object),
                "label_ids": np.array(all_labels, dtype=object),
            }
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

            writer.add_scalar("Loss/eval", loss_avg, global_step)
            writer.add_scalar("WER", metrics["wer"], global_step)
            writer.add_scalar("EvalSamples", total_samples, global_step)
            writer.add_scalar("EvalTimeSeconds", eval_time, global_step)
      
            if oneshot_modules:
                writer.add_scalar("OneShot/eval_scale", oneshot_modules[0].scale, global_step)
            
            lr = scheduler.get_last_lr()[0]

            print(
                f"• STEP:{global_step} • samp:{samples_per_sec:.1f} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}"
                + (f" • OneShot:{oneshot_modules[0].scale:.4f}" if oneshot_modules else "")
            )

            logging.info(
                f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}"
                + (f", OneShot: {oneshot_modules[0].scale:.4f}" if oneshot_modules else "")
            )
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_step_{global_step}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        global_step += 1
        step_in_report += 1

        avg_loss = total_loss / (global_step + 1)
        postfix_dict = {
            "loss": f"{avg_loss:.4f}",
            "lr": f"{lr:.6f}",
            "samp": f"{samples_per_sec:.1f}",
        }
        if oneshot_modules:
            postfix_dict["os"] = f"{oneshot_modules[0].scale:.4f}"
            
        progress_bar.set_postfix(postfix_dict, refresh=True)
        progress_bar.update(1)

    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed after {global_step} steps. Final model saved to {final_model_path}")
    writer.close()
    progress_bar.close()

THETA = 30000.0
def sinusoids(ctx, dims, theta=THETA):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding    

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, n_type, norm=False, enc=False):
        super().__init__()
        
        n.act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        
        n.encoder = nn.Sequential(
            n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), 
            n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   

        theta = nn.Parameter(torch.tensor(THETA), requires_grad=True)
        n.audio = lambda length, dims: sinusoids(length, dims, theta)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()
        n.ln = get_norm(n_type, dims) if norm else nn.Identity()
               
    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)
        x = n.encoder(x).permute(0, 2, 1).contiguous().to(device, dtype)
        x = x + n.audio(x.shape[1], x.shape[-1]).to(device, dtype)
        x = n.ln(x)
        return n.EncoderLayer(x)

class cgate(nn.Module):
    def __init__(n, dims, enabled=True):
        super().__init__()
        n.enabled = enabled
        if enabled:
            n.s_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.w_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.p_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.e_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.ph_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.integ = nn.Linear(dims*5, dims)
            n.reset_parameters()
        
    def forward(n, x, enc):
        if not n.enabled:
            return None
        s_feat = enc.get("spectrogram", x)
        w_feat = enc.get("waveform", x)
        p_feat = enc.get("pitch", x)
        e_feat = enc.get("envelope", x)
        ph_feat = enc.get("phase", x)
        s = n.s_gate(x) * s_feat
        w = n.w_gate(x) * w_feat
        p = n.p_gate(x) * p_feat
        e = n.e_gate(x) * e_feat
        ph = n.ph_gate(x) * ph_feat
        comb = torch.cat([s, w, p, e, ph], dim=-1)
        return n.integ(comb)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class mgate(nn.Module):
    def __init__(n, dims, mem=64):
        super().__init__()
        n.mk = nn.Parameter(torch.randn(mem, dims))
        n.mv = nn.Parameter(torch.randn(mem, 1))
        n.mg = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        n.reset_parameters()

    def forward(n, x, cos=False):
        if cos:
            key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mk, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        else:
            key = F.softmax(torch.matmul(x, n.mk.transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        return 0.5 * (torch.sigmoid(n.mg(x)) + torch.sigmoid(torch.matmul(key, n.mv)))

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tmgate(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        n.dims=dims
        n.mkeys = {}

        n.xa_proj = nn.Linear(dims, dims)
        n.activation = nn.Sigmoid()
        n.pattern = lambda length: sinusoids(length, dims=dims)  
        n.primer = torch.ones(1, 512, dims)
        n.reset_parameters()

    def forward(n, x) -> torch.Tensor: 
        if x is None:
            cur = n.primer
            n.key = cur
        else:
            cur = n.pattern(x.shape[1]).to(device, dtype)
    
        n.mkeys["last"] = n.key
        cur = n.xa_proj(cur.mean(dim=1)) 

        for b in range(cur.size(0)):
            cur_xa = cur[b]
            score = -1.0
            best = None
            for last in n.mkeys.items():
                last = n.mkeys["last"]
                similarity = F.cosine_similarity(cur_xa, last, dim=0).mean()

                if similarity > score:
                    score = similarity
                    best = best

            gating_value = n.activation(torch.tensor(score))
            if gating_value > n.threshold and best is not None:
                n.key = cur
            else:
                n.key = last
            threshold = apply_ste_threshold(x, n.threshold)
        return threshold, n.key

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class StraightThroughThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x, threshold)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste_threshold = StraightThroughThreshold.apply

def sinusoidz(length, dims):
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(10000.0) / dims))
    pe = torch.zeros(length, dims)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class LinearGate(nn.Linear):
    def __init__(n, in_features, out_features, act="swish", n_type=None, context = 4, num_types=4, top=4):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        n.context = context
        n.top = top
        n.num_types = num_types
        n.act=act

        n.gate = nn.Sequential(get_norm(n_type, in_features), nn.Linear(in_features, out_features, bias=False), nn.Softmax(dim=-1))
        n.context = nn.Parameter(torch.ones(context), requires_grad=True)
        n.bias2 = nn.Parameter(torch.zeros(context, in_features), requires_grad=True)
        n.reset_parameters()

    def forward(n, x, top=4):
        x = x.unsqueeze(-2)
        x = x * (1 + get_activation(n.act)((rearrange(n.context, 'n -> 1 1 n 1') * n.gate(x).squeeze(-1)).mean(-1, keepdim=True)))
        x = x + n.bias2.unsqueeze(0)
        _, indices = torch.topk(x, n.top, dim=-2, sorted=False)
        x = torch.gather(x, -2, indices).mean(dim=-2)
        x = super().forward(x)
        return x

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class FeatureGate(nn.Module):
    def __init__(n, dims: int, expand: int, adapt_dim: int):
        super().__init__()

        n.steps = nn.ModuleList
        (
            [
            LinearGate(dims, dims, adapt_dim) 
            for _ in range(expand)
            ]
        )
        n.gating = nn.Linear(adapt_dim, expand)
        n.reset_parameters()

    def forward(n, xa, xb):
        scores = F.softmax(n.gating(xa), dim=-1)
        output = sum(scores[:, q].unsqueeze(1) * gate(xa, xb) for q, gate in enumerate(n.steps))
        return output

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class dgate(nn.Module):
    def __init__(n, dims: int, head: int, threshold: float = 0.8):
        super().__init__()
        n.dims = dims
        
        n.register_buffer('last_key', torch.zeros(1, dims))
        n.xa_proj = nn.Linear(dims, dims)
        n.activation = nn.Sigmoid()
        n.register_buffer('primer', torch.ones(1, 512, dims))
        n.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32), requires_grad=True)
        n.reset_parameters()

    def forward(n, x: torch.Tensor): 
        device, dtype = x.device, x.dtype if x is not None else n.primer.device, n.primer.dtype
        if x is None:
            cur_representation = n.primer.to(device, dtype).mean(dim=1)
        else:
            pattern_tensor = n.pattern(x.shape[1]).to(device, dtype)
            cur_representation = pattern_tensor.mean(dim=1)

        cur_proj = n.xa_proj(cur_representation)
        expanded_last_key = n.last_key.expand(cur_proj.shape[0], -1) 
        similarity_scores = F.cosine_similarity(cur_proj, expanded_last_key, dim=-1).unsqueeze(-1)
        gating_value = n.activation(similarity_scores)
        decision = apply_ste_threshold(gating_value, n.threshold)
        n.last_key.copy_(cur_representation.detach()) 
        return decision, gating_value

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class m_gate(nn.Module):
    def __init__(n, dims, mem_size=64):
        super().__init__()

        n.m_key = nn.Parameter(torch.randn(mem_size, dims))
        n.m_val = nn.Parameter(torch.randn(mem_size, 1))
        n.gate_proj = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        n.reset_parameters()
            
    def forward(n, x):
        d_gate = torch.sigmoid(n.gate_proj(x))
        attention = torch.matmul(x, n.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, n.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class lgate(nn.Module):
    def __init__(n, dims):
        super().__init__()
        n.gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        n.reset_parameters()

    def forward(n, x):
        return n.gate(x)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tgate(nn.Module):
    def __init__(n, dims, num_types=2):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))

    def forward(n, x):
        types = n.classifier(x)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        return  torch.sum(gates * types.unsqueeze(2), dim=-1)
        
class tgate_hybrid(nn.Module):
    def __init__(n, dims, num_types=10, k=2):
        super().__init__()
        n.num_types = num_types
        n.k = k
        
        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.sparse_classifier = nn.Linear(dims, num_types)
        n.alpha = nn.Parameter(torch.ones(1))

    def forward(n, x):
        soft_types = n.classifier(x)
        logits = n.sparse_classifier(x)
        top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
        sparse_gates_values = F.softmax(top_k_logits, dim=-1)
        sparse_types = torch.zeros_like(soft_types)
        sparse_types.scatter_(-1, top_k_indices, sparse_gates_values)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        mixed_types = torch.sigmoid(n.alpha) * sparse_types + (1 - torch.sigmoid(n.alpha)) * soft_types
        return torch.sum(gates * mixed_types.unsqueeze(2), dim=-1)
  
class tgate_conditional(nn.Module):
    def __init__(n, dims, num_types=2, k=1, use_sparse=False):
        super().__init__()
        n.num_types = num_types
        n.k = k
        n.use_sparse = use_sparse
        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        
        if n.use_sparse:
            n.classifier = nn.Linear(dims, num_types)
        else:
            n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))

    def forward(n, x):
        if n.use_sparse:
            logits = n.classifier(x)
            top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
            sparse_gates_values = F.softmax(top_k_logits, dim=-1)
            types = torch.zeros_like(logits)
            types.scatter_(-1, top_k_indices, sparse_gates_values)
        else:
            types = n.classifier(x)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        return torch.sum(gates * types.unsqueeze(2), dim=-1)

class tgate_topk(nn.Module):
    def __init__(n, dims, num_experts=4, k=2):
        super().__init__()
        n.num_experts = num_experts
        n.k = k
        n.classifier = nn.Linear(dims, num_experts)
        n.experts = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_experts)])
    
    def forward(n, x):
        logits = n.classifier(x)
        top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
        mask = torch.zeros_like(logits, requires_grad=False)
        mask.scatter_(-1, top_k_indices, 1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        gates = torch.zeros_like(logits)
        gates.scatter_(-1, top_k_indices, top_k_gates)
        expert_outputs = torch.stack([expert(x) for expert in n.experts], dim=-1)
        return torch.sum(expert_outputs * gates.unsqueeze(2), dim=-1)

class BiasingGateRefactored(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super(BiasingGateRefactored, n).__init__()
        n.memory_size = memory_size
        n.threshold = threshold
        
        n.register_buffer("mkeys_pattern", sinusoids(memory_size, dims))
        n.mkeys_bias = nn.Parameter(torch.randn(memory_size, head))
        
        n.xa_projection = nn.Linear(dims, dims)
        n.activation = nn.Sigmoid()
        
    def forward(n, x, xa) -> torch.Tensor:
        if x is None:
            return None
        
        xa_projected = n.xa_projection(xa.mean(dim=1))
        similarity = F.cosine_similarity(xa_projected.unsqueeze(1), n.mkeys_pattern.unsqueeze(0), dim=-1)
        
        best_scores, best_indices = torch.max(similarity, dim=1)
        best_bias_weights = n.mkeys_bias[best_indices]
        gating_value = n.activation(best_scores)
        
        mask = (gating_value > n.threshold).float().unsqueeze(-1)
        
        shot_bias = x
        scaled_bias = shot_bias * best_bias_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        final_bias = mask * scaled_bias + (1 - mask) * shot_bias
        return final_bias
  
class SimpleMoE(nn.Module):
    def __init__(n, num_experts: int, expert_size: int, output_size: int):
        super().__init__()

        n.experts = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
        n.gate = nn.Linear(expert_size, num_experts)

    def forward(n, x: Tensor) -> Tensor:
  
        gate_outputs = F.softmax(n.gate(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in n.experts], dim=1)
        return torch.einsum("ab,abc->ac", gate_outputs, expert_outputs)

class LessSimpleMoE(nn.Module):
    def __init__(n, dims, num_experts=5, num_modalities=5, k=1, memory_size=64, threshold=0.8):
        super().__init__()
        n.dims = dims
        n.num_experts = num_experts
        n.num_modalities = num_modalities
        n.k = k
        n.memory_size = memory_size
        n.threshold = threshold
        
        n.soft_router = nn.Sequential(nn.Linear(dims, num_experts), nn.Softmax(dim=-1))
        n.sparse_router = nn.Linear(dims, num_experts)

        n.m_key = nn.Parameter(torch.randn(memory_size, dims))
        n.m_val = nn.Parameter(torch.randn(memory_size, num_experts))
        n.direct_gate = nn.Sequential(nn.Linear(dims, num_experts), nn.Sigmoid())

        n.experts = nn.ModuleList( [nn.Sequential(nn.Linear(dims, dims), nn.SiLU(), nn.Linear(dims, dims)) for _ in range(num_experts)] )
        
        n.fusion_gates = nn.ModuleList( [nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_modalities)] )
        n.fusion_projection = nn.Linear(dims * num_modalities, dims)
        
        n.register_buffer("alpha_moe", torch.tensor(0.5))
        n.register_buffer("alpha_fusion", torch.tensor(0.5))
        n.reset_parameters()
    
    def forward(n, x, enc):

        soft_expert_weights = n.soft_router(x)
        
        logits = n.sparse_router(x)
        top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
        sparse_gates = F.softmax(top_k_logits, dim=-1)
        
        sparse_expert_weights = torch.zeros_like(logits)
        sparse_expert_weights.scatter_(-1, top_k_indices, sparse_gates)
        
        attention = torch.matmul(x, n.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(n.dims), dim=-1)

        memory_weights = torch.matmul(attention, n.m_val)
        memory_weights = F.softmax(memory_weights, dim=-1)

        mixed_expert_weights = (n.alpha_moe * sparse_expert_weights + (1 - n.alpha_moe) * soft_expert_weights)
        final_expert_weights = (mixed_expert_weights + memory_weights) / 2
        expert_outputs = torch.stack([expert(x) for expert in n.experts], dim=-1)
        moe_output = torch.sum(expert_outputs * final_expert_weights.unsqueeze(1), dim=-1)

        features = [enc.get("spectrogram", x), enc.get("waveform", x), enc.get("pitch", x), enc.get("envelope", x), enc.get("phase", x)]
        gates = [gate(x) for gate in n.fusion_gates]
        
        scaled_features = [gates[q] * features[q] for q in range(len(features))]
        fused_output = n.fusion_projection(torch.cat(scaled_features, dim=-1))
        
        return (n.alpha_fusion * fused_output) + ((1 - n.alpha_fusion) * moe_output)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class BiasingGateB(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        n.dims = dims
        n.head = head
        n.memory_size = memory_size
        n.threshold = threshold
        n.mkeys = {}
        n.p = nn.Linear(dims, dims)
        n.tgate = nn.Sigmoid()

        n.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
        n.one_shot = OneShot(dims, head)
        n.reset_parameters()

        for _ in range(memory_size):
            pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
            bias_weight = OneShot(dims, head)
            n.mkeys[tuple(pattern.tolist())] = bias_weight

    def forward(n, x, xa) -> torch.Tensor:
        B, T, _ = x.shape
        input = n.p(x.mean(dim=1))
        batch_gate_biases = []
        for b in range(B):
            cur_input = input[b]
            score = -1.0
            best_match = None
            for pattern, gate_bias in n.mkeys.items():
                pattern_tensor = torch.tensor(pattern).to(cur_input.device)
                similarity = F.cosine_similarity(cur_input, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    best_match = gate_bias
            gating_value = n.tgate(score.unsqueeze(0))
            if gating_value > n.threshold and best_match is not None:
                batch_gate_biases.append(best_match.unsqueeze(0))
            else:
                batch_gate_biases.append(torch.zeros(1, n.head).to(x.device))
        return torch.cat(batch_gate_biases, dim=0)

class m2gate(nn.Module):
    def __init__(n, dims, mem=64, thresh=0.5):
        super().__init__()

        n.mkeys = nn.ParameterList([
            nn.Parameter(torch.randn(dims)),
            nn.Parameter(torch.randn(dims)),
        ])
        
        n.key_matrix = nn.Parameter(torch.randn(mem, dims))
        n.val_matrix = nn.Parameter(torch.randn(mem, 1))
        n.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
        n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False).to(device)
        n.concat_layer = nn.Linear(2, 1, device=device, dtype=dtype)
        n.tgate_activation = nn.Sigmoid() 
        n.xa_projection = nn.Linear(dims, dims)

        n.register_buffer('previous_best_pattern', None)

    def forward(n, x):
        x_processed = n.xa_projection(x.mean(dim=1))
        skip_indicators = torch.ones(x_processed.size(0), dtype=torch.float32, device=x.device)

        previous_input_pattern_in_batch = None 
        for b in range(x_processed.size(0)):
            cur_x_element = x_processed[b]
            score = -1.0
            current_best_pattern = None 

            for pattern_tensor in n.mkeys:
                similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)

                if similarity > score:
                    score = similarity
                    current_best_pattern = pattern_tensor

            if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
                skip_indicators[b] = 1.0
            else:
                skip_indicators[b] = 0.0
            previous_input_pattern_in_batch = current_best_pattern 
        change_scores = torch.zeros(x_processed.size(0), dtype=torch.float32, device=x.device)

        previous_input_pattern_in_batch = None 
        
        for b in range(x_processed.size(0)):
            cur_x_element = x_processed[b]
            score = -1.0
            current_best_pattern = None

            for pattern_tensor in n.mkeys:
                similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    current_best_pattern = pattern_tensor

            if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
                change_scores[b] = 1.0
            else:
                change_scores[b] = 0.0
            previous_input_pattern_in_batch = current_best_pattern

        scalar = apply_ste_threshold(change_scores.unsqueeze(-1), n.threshold)
        key = F.softmax(torch.matmul(F.normalize(x_processed, p=2, dim=-1), F.normalize(n.key_matrix, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x_processed.shape[-1]), dim=-1)
        gate = n.concat_layer(torch.cat((torch.matmul(key, n.val_matrix),  n.mlp(x_processed)), dim=-1))
        return scalar, gate

class Memory(nn.Module):
    def __init__( n, new_dims: int, old_dims: int):
        super().__init__()

        n.new_dims = new_dims
        n.old_dims = old_dims
        n.new = nn.Linear(new_dims, old_dims)
        n.old = nn.Linear(old_dims, old_dims)
        n.activation = nn.Tanh()

    def forward(n, x: Tensor, y: Tensor) -> Tensor:
        mix = n.activation(n.new(x) + n.old(y))
        return mix

    def initialize_old(n, batch: int, device: torch.device) -> Tensor:
        old_dims = torch.zeros(batch, n.old_dims).to(device)
        return old_dims

class MixtureOfMemories(nn.Module):
    def __init__(n, dims, head, num_experts: int, expert_size: int, output_size: int):
        super().__init__()
        n.mems = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
        n.gate = nn.Linear(expert_size, num_experts)
        n.attention = nn.MultiheadAttention(dims, head)
        n.memory = Memory(dims, dims)
        n.layernorm = nn.LayerNorm(dims)

    def forward(n, x: Tensor, y: Tensor) -> Tensor:
        output, _ = n.attention(x, x, x)
        y = n.memory(output.mean(dim=0), y)
        gate_outputs = F.softmax(n.gate(x), dim=1)
        _outputs = torch.stack([mem(x) for mem in n.mems], dim=1)
        _output = torch.einsum("ab,abc->ac", gate_outputs, _outputs)
        output = n.layernorm(output + _output.unsqueeze(0))
        return output

class ContextualBiasingGate2(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int, threshold: float = 0.8):
        super().__init__()
        n.dims = dims
        n.head = head
        n.memory_size = memory_size
        n.threshold = threshold
        n.mkeys = {}
        n.xa_projection = nn.Linear(dims, dims)
        n.tgate_activation = nn.Sigmoid()
        n.head_bias_weights = nn.Parameter(torch.ones(head))
        n.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
        n.embedding = nn.Embedding(n.pattern, dims)
        n.one_shot = OneShot(dims, head)
        n.reset_parameters()

        for _ in range(memory_size):
            pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
            bias_weight = OneShot(dims, head)
            n.mkeys[tuple(pattern.tolist())] = bias_weight

    def forward(n, shot_bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = n.xa_projection(x.mean(dim=1))
        last_x = []
        previous_best = None 

        for b in range(x.size(0)):
            cur_x = x[b]
            score = -1.0
            current_best = None  

            for pattern, bias in n.mkeys.items():
                pattern_tensor = torch.tensor(pattern).to(cur_x.device)
                similarity = F.cosine_similarity(cur_x, pattern_tensor, dim=0)

                if similarity > score:
                    score = similarity
                    current_best = bias

            gating_value = n.tgate_activation(score.unsqueeze(0))
            current_shot_bias = shot_bias[b]

            if gating_value > n.threshold and current_best is not None:
                if previous_best is None or not torch.equal(current_best, previous_best):
                    scaled_bias = current_shot_bias * (current_best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                    last_x.append(scaled_bias)
                else: 
                    last_x.append(current_shot_bias)
            else: 
                last_x.append(current_shot_bias)
            previous_best = current_best 
        return torch.stack(last_x, dim=0)
    
class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        if features is None:    
            features = ["spectrogram", "waveform", "pitch", "aperiodic", "harmonics"]
        n.head = head
        n.head_dim = dims // head
        n.scale = 1.0 // len(features) if features else scale

        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)

    def forward(n, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
        B, L, D = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, L, n.head, n.head_dim).transpose(1,2)
        k = n.k(xa).view(B, K, n.head, n.head_dim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.head_dim)
        return bias

class curiosity(nn.Module):
    def __init__(n, d, h, bias=True):
        super().__init__()
        n.h  = h
        n.dh = d // h
        n.qkv = nn.Linear(d, d * 3, bias=bias)
        n.qkv_aux = nn.Linear(d, d * 3, bias=bias)
        n.o  = nn.Linear(d, d, bias=bias)
        n.g  = nn.Parameter(torch.zeros(h))

    def split(n, x):
        b, t, _ = x.shape
        return x.view(b, t, n.h, n.dh).transpose(1, 2)

    def merge(n, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(n, x, xa, mask=None):

        q, k, v   = n.qkv(x).chunk(3, -1)
        qa, ka, va = n.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(n.split, (q, k, v))
        qa, ka, va = map(n.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / n.dh**0.5
        dots_aux  = (q @ ka.transpose(-2, -1)) / n.dh**0.5
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)
        h_main = p  @ v
        h_aux  = pa @ va
        g = torch.sigmoid(n.g).view(1, -1, 1, 1)
        out = n.merge(h_main * (1 - g) + h_aux * g)
        return n.o(out)
