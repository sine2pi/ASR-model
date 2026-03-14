# bare necessities, old mother natures recipes
import torch, torchaudio, torch.nn.functional as F
import os, math, logging, time
import pyworld as pw
from torch import Tensor, nn
import numpy as np
from torch.nn.functional import embedding, scaled_dot_product_attention as SDPA
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
from tensordict import TensorDict 
import librosa
from tensorboardX import SummaryWriter
from tqdm import tqdm
import soundfile as sf
import pandas as pd

import logging
from typing import Tuple, Optional, Dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
THETA = 30000.0

def l2norm(t):
    return F.normalize(t, dim = -1)

def have(a):
    return a is not None
    
def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, b) if not have(c) else c

def no_none(xa):
    return xa.apply(lambda tensor: tensor if tensor is not None else None)

def exact_div(x, y):
    assert x % y == 0
    return x // y

class LocalNorm(nn.Module):
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, mode: str = '1', threshold: float = 0.8):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.mode = mode
        self.threshold = threshold

    def forward(self, input: Tensor, confidence=None) -> Tensor:
        if input.numel() == 0:
            return input

        div = input.mul(input).unsqueeze(1) 
        
        pad_len = self.size // 2

        # We use 1D pooling to avoid the 'interpolation' feeling of 2D on a line
        if self.mode == "1":
            # MODE 1: Standard Average (Original LRN behavior over time)
            div = F.avg_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)

        elif self.mode == "2":
            # MODE 2: Intensity-based Switch (Lateral Inhibition vs smoothing)
            avg_d = F.avg_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            max_d = F.max_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            # If there's a significant spike, switch to Max-based suppression
            condition = (max_d > 2.0 * avg_d).float()
            div = (condition * max_d) + ((1 - condition) * avg_d)

        elif self.mode == "3":
            # MODE 3: PyWorld Confidence Gate
            avg_d = F.avg_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            max_d = F.max_pool1d(div, kernel_size=self.size, stride=1, padding=pad_len)
            
            if confidence is None:
                div = avg_d
            else:
                conf_mask = (confidence > self.threshold).float().unsqueeze(1)
                div = (conf_mask * avg_d) + ((1 - conf_mask) * max_d)

        div = div.narrow(2, 0, input.size(1)).squeeze(1)
        denom = div.mul(self.alpha).add(self.k).pow(self.beta)
        return input / denom

class GlobalNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class LinearNorm(nn.Module):
    def __init__(n, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, n).__init__()
        n.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(n.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(n, x):
        return n.linear_layer(x)

class LayerNorm(nn.Module):
    def __init__(n, dims, eps=1e-5):
        super().__init__()
        n.dims = dims
        n.eps = eps
        n.gamma = nn.Parameter(torch.ones(dims))
        n.beta = nn.Parameter(torch.zeros(dims))

    def forward(n, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (n.dims,), n.gamma, n.beta, n.eps)
        return x.transpose(1, -1)

class AdaLN(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.norm = nn.LayerNorm(dims, elementwise_affine=False)

        self.mlp = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(), 
            nn.Linear(dims, 2 * dims)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition=None):
        if condition is None:
             return self.norm(x)

        scale_bias = self.mlp(condition)
        gamma, beta = torch.chunk(scale_bias, 2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return self.norm(x) * (1 + gamma) + beta


def get_norm(n_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if n_type in ["batchnorm", "instancenorm"] and dims is None:
        raise ValueError(f"'{n_type}' requires 'dims'.")
    if n_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{n_type}' requires 'num_groups'.")

    norm_map = {
        "layernorm": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        "layernorm": lambda: LayerNorm(dims=dims),
        "linearnorm": lambda: LinearNorm(in_dim=dims, out_dim=dims, bias=False),
        "adanorm": lambda: AdaLN(dims=dims),
        "instancenorm": lambda: nn.InstanceNorm1d(num_features=dims, affine=False, track_running_stats=False),     
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=dims),
        "instancenorm2d": lambda: nn.InstanceNorm2d(num_features=dims),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_dims=dims),
        "localnorm": lambda: LocalNorm(size=5),
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

def gammatone(dims, head, min_freq=200.0, max_freq=8000.0):
    head_dim = dims // head
    f = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
    return f / 1000

def wideband(dims, head, max_freq=8000.0):
    head_dim = dims // head
    mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
    mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 700 * mel_scale / 1000

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

def exact_div(x, y):
    assert x % y == 0
    return x // y

def load_wave(audio, sample_rate=16000):
    if isinstance(audio, str):
        wp, sample_rate = sf.read(audio, dtype='float32')
        
        if wp.ndim > 1:
            abs_max = wp.max(axis=0)  
            wp = wp / abs_max if any(abs_max > 0) else wp
            waveform = torch.from_numpy(wp.T)
        else:
            abs_max = max(abs(wp))
            wp = wp / abs_max if abs_max > 0 else wp
            waveform = torch.from_numpy(wp)
            
    elif isinstance(audio, dict):
        waveform = torch.tensor(data=audio["array"]).float()
        sample_rate = audio["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    return waveform, sample_rate

def get_audio(audio, sample_rate=16000):
    import soundfile as sf
    import torch
    if isinstance(audio, str):
        wp, sample_rate = sf.read(audio, dtype='float32')
        
        if wp.ndim > 1:
            abs_max = wp.max(axis=0)  
            wp = wp / abs_max if any(abs_max > 0) else wp
            waveform = torch.from_numpy(wp.T)
        else:
            abs_max = max(abs(wp))
            wp = wp / abs_max if abs_max > 0 else wp
            waveform = torch.from_numpy(wp)
            
    elif isinstance(audio, dict):
        waveform = torch.tensor(data=audio["array"]).float()
        sample_rate = audio["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    
    if sample_rate != sample_rate:
        import librosa
        waveform = librosa.resample(waveform.numpy(), orig_sr=sample_rate, target_sr=sample_rate)
        waveform = torch.from_numpy(waveform)

    audio = waveform.numpy()
    duration = len(audio) / sample_rate
    return {
        "raw": audio,
        "sampling_rate": sample_rate,
    }, duration

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

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def pitch_toks(audio: torch.Tensor, sample_rate: int, labels: list, hop_length: int, mode: str = "mean"):
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

def quantize_pitch(pt_tensor: torch.Tensor, p_tensor: torch.Tensor = None, dims: int = 512, num_bins: int = 256, v_min: float = -2.0, v_max: float = 2.0, embedding=False) -> torch.Tensor:
    indices = ((pt_tensor - v_min) / (v_max - v_min) * (num_bins - 1)).round().long()
    tensor = torch.clamp(indices, 0, num_bins - 1)
    tensor = torch.polar(p_tensor, tensor) if p_tensor is not None else tensor
    tensor = torch.view_as_real(tensor) if p_tensor is not None else tensor
    return nn.Embedding(tensor, dims) if embedding else tensor

def extract_features(batch, tokenizer=None, spectrogram=False, pitch=False, waveform=False, 
                    harmonics=False, aperiodics=False, phase=False, hilbert=False, pitch_tokens=False, 
                    hop_length=160, sample_rate=16000, mels=128):
    
    mode = "mean"
    debug = False
    dummy_audio = False
    dummy_text = False

    if dummy_text:
        labels = [1] * 32
    else:
        labels = tokenizer.encode(batch["transcription" if "transcription" in batch else "sentence"] )

    if dummy_audio:
        dummy, _ = load_wave(batch["audio"], sample_rate)
        audio = torch.zeros_like(dummy)
    else:
        audio, sample_rate = load_wave(batch["audio"], sample_rate)

    if pitch_tokens:
        pt_tensor = pitch_toks(audio, sample_rate, labels, hop_length, mode=mode).to(device, dtype)
        # print(f"Pitch Tokens: {pt_tensor.shape}")

    if harmonics:
        h_tensor, a_tensor = harmonics_and_aperiodics(audio, sample_rate, hop_length)
        h_tensor = h_tensor.to(device, dtype)
        a_tensor = a_tensor.to(device, dtype)

    if pitch:
        frame_period = hop_length / sample_rate * 1000
        f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period)
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
        spectrogram_config = {
            "hop_length": hop_length,
            "f_min": 50,
            "f_max": 8000,
            "n_mels": mels,
            "n_fft": 1024,
            "sample_rate": sample_rate,
            "pad_mode": "constant",
            "center": True, 
            "power": 2.0,
            "window_fn": torch.hann_window,
            "mel_scale": "htk",
            "norm": None,
            "normalized": False,
        }

        transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        mel_spectrogram = transform(audio.float())
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        s_tensor = (log_mel + 4.0) / 4.0
        s_tensor = s_tensor.to(device, dtype)

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

    return {
        "waveform": w_tensor if waveform else None,
        "spectrogram": s_tensor if spectrogram else None,
        "pitch_tokens": pt_tensor if pitch_tokens else None, 
        "pitch": p_tensor if pitch else None,
        "harmonic": h_tensor if harmonics else None,
        "aperiodic": a_tensor if aperiodics else None,
        "labels": labels if labels is not None else None,
        "phase": ph_tensor if phase else None,
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
                batch["text_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

            elif key in ["spectrogram", "waveform", "pitch", "pitch_tokens"]:

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

def track_grad_norms(model):
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm(2).item()
            grad_norms[name] = norm
    return grad_norms

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, logits=None, compute_result = False):
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(q) for q in seq if q not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(q) for q in ids if q not in (-100, pad_token_id, bos_token_id, eos_token_id)]

    if isinstance(pred, dict):
        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
    else:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    else:
        pred_ids = pred_ids

    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

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
        per_layer_norms = track_grad_norms(model)
        if not isinstance(per_layer_norms, dict) or not per_layer_norms:
            per_layer_norms = {}
    else:
        trainable_params = 0.0
        efficiency_score = 0.0
        per_layer_norms = {}

    result = {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
    }

    if isinstance(per_layer_norms, dict):
        for k, v in per_layer_norms.items():
            result[f"per_layer_norms_{k}"] = float(v)
    return result
        
class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        if features is None:    
            features = ["spectrogram", "waveform", "pitch", "pitch_tokens"]
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

def spectral_entropy(grad_tensor, n_bands=64):
    if grad_tensor is None or grad_tensor.numel() < n_bands:
        return 1.0
    
    sample_size = min(grad_tensor.numel(), 2048)
    flat_grad = grad_tensor.flatten()[:sample_size].float()
    
    freq_repr = torch.fft.rfft(flat_grad)
    psd = torch.abs(freq_repr)**2

    psd_norm = psd / (psd.sum() + 1e-8)
    entropy = -torch.sum(psd_norm * torch.log(psd_norm + 1e-8))

    max_entropy = math.log(psd_norm.shape[0])
    return (entropy / max_entropy).item()

def train_and_evaluate(
    model,
    tokenizer,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    loss_fn,
    metric_fn,
    max_steps=1000,
    device="cuda",
    acc_steps=1,
    clear_cache=False,
    log_interval=10,
    eval_interval=100,
    save_interval=1000,
    warmup_interval=100,
    checkpoint_dir="checkpoint_dir",
    log_dir="log_dir",
    generate=False,
    clip_grad_norm=0.0,
):

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
            logging.info(f"Starting dataset epoch {dataset_epochs}")

            if step_in_report > 0:
                avg_loss = total_loss / step_in_report
                logging.info(f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0
                step_in_report = 0

        start_time = time.time()

        features = TensorDict({k: v.to(device) for k, v in batch.items() if k in ["spectrogram","waveform","pitch","pitch_tokens"] and v is not None}, batch_size=batch["text_ids"].shape[0]).to(device)
        text_ids = batch["text_ids"].to(device)
        labels = batch["labels"].long().to(device)

        output = model(text_ids=text_ids, labels=labels, spectrogram=features.get("spectrogram"), pitch=features.get("pitch"), waveform=features.get("waveform"), pitch_tokens=features.get("pitch_tokens"))
        loss = output["loss"]

        total_loss += loss.item()
        loss = loss / acc_steps

        scaler.scale(loss).backward()

        per_layer_norms = track_grad_norms(model)
        for k, v in per_layer_norms.items():
            writer.add_scalar(f"train/per_layer_norms_{k}", v, global_step)

        if (global_step + 1) % acc_steps == 0:
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
                    elif recent_avg < prev_avg * 0.8:
                        module.scale *= 1.1
                for module in oneshot_modules:
                    module.scale = float(max(0.05, min(2.0, module.scale)))
                
                if len(grad_history) > 100:
                    grad_history = grad_history[-100:]

            if global_step % log_interval == 0:
                writer.add_scalar("GradNorm", total_norm, global_step)
                if oneshot_modules:
                    writer.add_scalar("OneShot/scale", oneshot_modules[0].scale, global_step)

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = len(batch["text_ids"]) / (end_time - start_time)

        if global_step % log_interval == 0:
            writer.add_scalar(
                tag="Loss/train",
                scalar_value=total_loss / (global_step + 1),
                global_step=global_step,
            )
            lr = scheduler.get_last_lr()[0]

            writer.add_scalar(
                tag="LearningRate", 
                scalar_value=lr, 
                global_step=global_step
            )
            writer.add_scalar(
                tag="SamplesPerSec",
                scalar_value=samples_per_sec,
                global_step=global_step,
            )

        if global_step % log_interval == 0:
            for name, param in model.named_parameters():
                if param.grad is not None and param.numel() > 1024:
                    s_entropy = spectral_entropy(param.grad)
                    writer.add_scalar(f"Entropy/{name}", s_entropy, global_step)
                    writer.add_scalar(f"GradNorm/{name}", param.grad.norm(2), global_step)

        if eval_interval > 0 and warmup_interval < global_step or global_step == max_steps - 1: 
            eval_interval = eval_interval if eval_interval > 0 else 1
            if global_step % (eval_interval) == 0:
                model.eval()
                start = time.time()
                eval_loss = 0
                all_p = []
                all_l = []
                batch = 0
                total = 0
                
                with torch.no_grad():
                    for eval_batch in eval_loader:

                        features = TensorDict({k: v.to(device) for k, v in eval_batch.items() if k in ["spectrogram","waveform","pitch","pitch_tokens"] and v is not None}, batch_size=eval_batch["text_ids"].shape[0]).to(device)
                        text_ids = eval_batch["text_ids"].to(device)
                        labels = eval_batch["labels"].long().to(device)

                        batch_size = text_ids.size(0)
                        total += batch_size

                        output = model(
                            text_ids=text_ids, 
                            labels=labels, 
                            spectrogram=features.get("spectrogram"),
                            pitch=features.get("pitch"), 
                            waveform=features.get("waveform"),
                            pitch_tokens=features.get("pitch_tokens")
                        )

                        loss = output["loss"]
                        eval_loss += loss.item()
                        batch += 1

                        if generate:
                            generated_ids = model.generate(
                                spectrogram=features.get("spectrogram"),
                                pitch=features.get("pitch"),
                                waveform=features.get("waveform"),
                                pitch_tokens=features.get("pitch_tokens"), 
                                max_new_tokens=labels.shape[1] 
                            ) 
                            all_p.extend(generated_ids.cpu().numpy().tolist())
                            all_l.extend(labels.cpu().numpy().tolist())

                        else: 
                            all_p.extend(
                                torch.argmax(output["logits"], dim=-1).cpu().numpy().tolist()
                            )
                            all_l.extend(labels.cpu().numpy().tolist())

                eval_time = time.time() - start
                loss_avg = eval_loss / batch if batch > 0 else 0

                p = {
                    "predictions": np.array(all_p, dtype=object),
                    "label_ids": np.array(all_l, dtype=object),
                }
                
                metrics = metric_fn(p, tokenizer=tokenizer, model=model)
                writer.add_scalar("Loss/eval", loss_avg, global_step)
                writer.add_scalar("WER", metrics["wer"], global_step)
                writer.add_scalar("EvalSamples", total, global_step)
                writer.add_scalar("EvalTimeSeconds", eval_time, global_step)
        
                if "per_layer_norms" in metrics:
                    for layer, norm in metrics["per_layer_norms"].items():
                        writer.add_scalar(f"eval/per_layer_norms/{layer}", norm, global_step)

                if oneshot_modules:
                    writer.add_scalar("OneShot/eval_scale", oneshot_modules[0].scale, global_step)
                
                lr = scheduler.get_last_lr()[0]

                print(
                    f"EVALUATION STEP:{global_step} • samp:{samples_per_sec:.1f} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}"
                    + (f" • OneShot:{oneshot_modules[0].scale:.4f}" if oneshot_modules else ""))

                logging.info(
                    f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}"
                    + (f", OneShot: {oneshot_modules[0].scale:.4f}" if oneshot_modules else ""))
                
                model.train()

        else:
            model.train()

        if save_interval > 0:
            if global_step > 0 and global_step % save_interval == 0:
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
            "samp/s": f"{samples_per_sec:.1f}",
        }   

        if oneshot_modules:
            postfix_dict["os"] = f"{oneshot_modules[0].scale:.4f}"
            
        progress_bar.set_postfix(postfix_dict, refresh=True)
        progress_bar.update(1)

    if save_interval > 0:
        path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save(model.state_dict(), path)
        logging.info(f"Final model saved to {path}")
        print(f"Final model saved to {path}")
        
    logging.info(f"Training completed after {global_step} steps.")

    writer.close()
    progress_bar.close()
    return model

class prepare_datasets(torch.utils.data.Dataset):
    def __init__(n, metadata_file, data_dir, tokenizer=None, extract_args=None):
        n.metadata = pd.read_csv(metadata_file)
        n.data_dir = data_dir
        n.tokenizer = tokenizer
        n.extract_args = extract_args if extract_args is not None else {}

    def __len__(n):
        return len(n.metadata)

    def __getitem__(n, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = n.metadata.iloc[idx]
        audio_path = os.path.join(n.data_dir, row["audio"])
        
        batch_input = {
            "audio": audio_path,
            "transcription": row["sentence"]
        }
        
        features = extract_features(
            batch_input, 
            tokenizer=n.tokenizer, 
            **n.extract_args
        )
        
        return features

def generate_predictions(model, spectrogram=None, pitch=None, waveform=None, tokenizer=None, batch_size=None, max_new_tokens=None):
    decoder_start_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    generated_ids = torch.full(size=(batch_size, 1), fill_value=decoder_start_token_id, dtype=torch.long, device=device)
    
    max_length = max_new_tokens + 1
    for i in range(max_length - 1):
        with torch.no_grad():
            curr_output = model.processor(generated_ids, spectrogram=spectrogram, pitch=pitch, waveform=waveform)
        next_token_logits = curr_output[:, -1, :]
        
        if i < max_new_tokens:
            next_token_logits[:, eos_token_id] = float('-inf')
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
        if (next_tokens == eos_token_id).all() and i >= max_new_tokens:
            break
    return generated_ids

def save_model_checkpoint(model, optimizer, scheduler, checkpoint_dir, global_step):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step
    }, checkpoint_path)
    logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")

def evaluate_model(model, tokenizer, eval_loader, loss_fn, device, global_step, writer, eval_steps):
    model.eval()
    eval_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []

    eval_start_time = time.time()
    
    with torch.no_grad():
        for eval_batch in tqdm(eval_loader, desc=f"Evaluating (Step {global_step})", leave=False):
            if batch_count >= eval_steps:
                break
                
            input_features = eval_batch['input_features'].to(device)
            input_ids = eval_batch['input_ids'].to(device) 
            labels = eval_batch['labels'].long().to(device)
            
            encoder_output = model.encoder(input_features)
            decoder_output = model.decoder(input_ids, encoder_output)
            logits = decoder_output.view(-1, decoder_output.size(-1))
            
            active_labels = labels.reshape(-1)
            active_mask = active_labels != -100
            loss = loss_fn(logits[active_mask], active_labels[active_mask])
            eval_loss += loss.item()
 
            batch_size = input_features.size(0)
            
            generated_ids = generate_predictions(
                model=model,
                input_features_encoded=encoder_output,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                min_length=5
            )
        
            all_preds.append(generated_ids)
            all_labels.append(labels)
            
            batch_count += 1
            
    # metrics = load_metric("wer")
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # wer = metrics.compute(predictions=preds_text, references=labels_text)
    avg_loss = eval_loss / max(1, batch_count)
    eval_time = time.time() - eval_start_time
    
    return {
        "loss": avg_loss,
        # "wer": wer,
        "preds": preds_text,
        "labels": labels_text,
        "eval_time": eval_time,
    }

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

