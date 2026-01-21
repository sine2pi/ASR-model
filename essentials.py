import torch, torchaudio, torch.nn.functional as F
import os, math, logging, time
import pyworld as pw
from torch import Tensor, nn
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
from tensordict import TensorDict 
import librosa
from tensorboardX import SummaryWriter
from tqdm import tqdm
import soundfile as sf
import pandas as pd

import numpy as np
import logging
from typing import Tuple, Optional, Dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
THETA = 30000.0

try:
    import scipy.signal as signal
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioFeatureExtractor: # -extremely- inefficient but works as a fallback
    @staticmethod
    def extract_pitch_contour(audio: np.ndarray, sr: int,
                             fmin: float = 50, fmax: float = 500,
                             frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:

        try:
            if LIBROSA_AVAILABLE:
                f0 = librosa.pyin(
                    audio,
                    fmin=fmin,
                    fmax=fmax,
                    sr=sr,
                    frame_length=frame_length
                )[0]
                nans = np.isnan(f0)
                if not nans.all():
                    x = np.arange(len(f0))
                    f0[nans] = np.interp(x[nans], x[~nans], f0[~nans])
                else:
                    f0 = np.full_like(f0, 150.0)

                times = librosa.frames_to_time(
                    np.arange(len(f0)),
                    sr=sr,
                    hop_length=frame_length // 4
                )
            else:
                hop_length = frame_length // 4
                num_frames = 1 + (len(audio) - frame_length) // hop_length
                f0 = np.zeros(num_frames)

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + frame_length
                    frame = audio[start:end]

                    corr = np.correlate(frame, frame, mode='full')
                    corr = corr[len(corr)//2:]

                    min_lag = int(sr / fmax)
                    max_lag = int(sr / fmin)

                    if max_lag < len(corr):
                        search_range = corr[min_lag:max_lag]
                        if len(search_range) > 0:
                            peak = np.argmax(search_range) + min_lag
                            if peak > 0:
                                f0[i] = sr / peak
                            else:
                                f0[i] = 150.0
                        else:
                            f0[i] = 150.0
                    else:
                        f0[i] = 150.0

                times = np.arange(num_frames) * hop_length / sr

            return f0, times

        except Exception as e:

            num_frames = max(1, len(audio) // (frame_length // 4))
            return np.full(num_frames, 150.0), np.linspace(0, len(audio)/sr, num_frames)

    @staticmethod
    def extract_formants(audio: np.ndarray, sr: int,
                        num_formants: int = 4) -> np.ndarray:

        try:
            if not SCIPY_AVAILABLE:

                num_frames = max(1, len(audio) // 512)
                defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
                return np.tile(defaults, (num_frames, 1))

            frame_length = 1024
            hop_length = frame_length // 2
            num_frames = 1 + (len(audio) - frame_length) // hop_length

            formants = np.zeros((num_frames, num_formants))
            lpc_order = 2 + sr // 1000

            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]

                emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

                windowed = emphasized * np.hamming(len(emphasized))

                try:
                    r = np.correlate(windowed, windowed, mode='full')
                    r = r[len(r)//2:]
                    r = r[:lpc_order + 1]

                    a = np.zeros(lpc_order + 1)
                    a[0] = 1.0
                    e = r[0]

                    for k in range(1, lpc_order + 1):
                        alpha = -np.sum(a[:k] * r[k:0:-1]) / e
                        a_new = np.zeros(k + 1)
                        a_new[0] = 1.0
                        a_new[1:k] = a[1:k] + alpha * a[k-1:0:-1]
                        a_new[k] = alpha
                        a = a_new
                        e = e * (1 - alpha * alpha)

                    roots = np.roots(a)

                    angles = np.angle(roots)
                    freqs = angles * (sr / (2 * np.pi))

                    freqs = freqs[freqs > 0]
                    freqs = np.sort(freqs)

                    if len(freqs) >= num_formants:
                        formants[i, :] = freqs[:num_formants]
                    else:
                        defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
                        formants[i, :len(freqs)] = freqs
                        formants[i, len(freqs):] = defaults[len(freqs):]

                except Exception as lpc_error:
                    defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
                    formants[i, :] = defaults

            return formants

        except Exception as e:
          
            num_frames = max(1, len(audio) // 512)
            defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
            return np.tile(defaults, (num_frames, 1))

    @staticmethod
    def extract_spectral_envelope(audio: np.ndarray, sr: int,
                                  n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:

        try:
            hop_length = n_fft // 4

            if LIBROSA_AVAILABLE:
                D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                mag = np.abs(D)
                envelope = np.zeros_like(mag)
                quefrency_limit = int(sr / 50)

                for i in range(mag.shape[1]):
                    spectrum = mag[:, i]
                    log_spectrum = np.log(spectrum + 1e-10)

                    cepstrum = np.fft.ifft(log_spectrum).real

                    liftered = np.copy(cepstrum)
                    if quefrency_limit < len(liftered):
                        liftered[quefrency_limit:-quefrency_limit] = 0

                    smoothed = np.exp(np.fft.fft(liftered).real[:len(spectrum)])
                    envelope[:, i] = smoothed

                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            else:
                num_frames = 1 + (len(audio) - n_fft) // hop_length
                envelope = np.zeros((n_fft // 2 + 1, num_frames))

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + n_fft
                    frame = audio[start:end]

                    windowed = frame * np.hanning(len(frame))

                    spectrum = np.abs(np.fft.rfft(windowed))
                    envelope[:, i] = spectrum

                freqs = np.fft.rfftfreq(n_fft, 1/sr)

            return envelope, freqs

        except Exception as e:

            num_frames = max(1, len(audio) // (n_fft // 4))
            envelope = np.ones((n_fft // 2 + 1, num_frames))
            freqs = np.fft.rfftfreq(n_fft, 1/sr)
            return envelope, freqs

    @staticmethod
    def extract_amplitude_envelope(audio: np.ndarray, sr: int,
                                   frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:

        try:
            hop_length = frame_length // 4

            if LIBROSA_AVAILABLE:
                rms = librosa.feature.rms(
                    y=audio,
                    frame_length=frame_length,
                    hop_length=hop_length
                )[0]
                times = librosa.frames_to_time(
                    np.arange(len(rms)),
                    sr=sr,
                    hop_length=hop_length
                )
            else:
                num_frames = 1 + (len(audio) - frame_length) // hop_length
                rms = np.zeros(num_frames)

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + frame_length
                    frame = audio[start:end]
                    rms[i] = np.sqrt(np.mean(frame ** 2))

                times = np.arange(num_frames) * hop_length / sr

            return rms, times

        except Exception as e:

            num_frames = max(1, len(audio) // (frame_length // 4))
            return np.ones(num_frames), np.linspace(0, len(audio)/sr, num_frames)

    @staticmethod
    def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:

        try:
            if LIBROSA_AVAILABLE:
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            else:
                n_fft = 2048
                hop_length = n_fft // 4
                num_frames = 1 + (len(audio) - n_fft) // hop_length
                mfcc = np.zeros((n_mfcc, num_frames))

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + n_fft
                    frame = audio[start:end]
                    windowed = frame * np.hanning(len(frame))
                    spectrum = np.abs(np.fft.rfft(windowed))
                    log_spectrum = np.log(spectrum + 1e-10)
                    from scipy.fftpack import dct
                    cepstrum = dct(log_spectrum, type=2, norm='ortho')
                    mfcc[:, i] = cepstrum[:n_mfcc]
            return mfcc

        except Exception as e:
            num_frames = max(1, len(audio) // 512)
            return np.zeros((n_mfcc, num_frames))

    @staticmethod
    def extract_all_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:

        try:
       
            features = {}
            f0, f0_times = AudioFeatureExtractor.extract_pitch_contour(audio, sr)
            features['pitch'] = f0
            features['pitch_times'] = f0_times

            formants = AudioFeatureExtractor.extract_formants(audio, sr)
            features['formants'] = formants

            spec_env, spec_freqs = AudioFeatureExtractor.extract_spectral_envelope(audio, sr)
            features['spectral_envelope'] = spec_env
            features['spectral_freqs'] = spec_freqs

            amp_env, amp_times = AudioFeatureExtractor.extract_amplitude_envelope(audio, sr)
            features['amplitude_envelope'] = amp_env
            features['amplitude_times'] = amp_times

            mfcc = AudioFeatureExtractor.extract_mfcc(audio, sr)
            features['mfcc'] = mfcc
            return features

        except Exception as e:
            return {}

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

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dims))
        self.beta = nn.Parameter(torch.zeros(dims))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.dims,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, dims, eps=1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.fc = nn.Linear(style_dim, dims*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.dims,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

def get_norm(n_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if n_type in ["batchnorm", "instancenorm"] and dims is None:
        raise ValueError(f"'{n_type}' requires 'dims'.")
    if n_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{n_type}' requires 'num_groups'.")

    norm_map = {
        "layernorm": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        # "layernorm": lambda: LayerNorm(dims=dims),
        "linearnorm": lambda: LinearNorm(in_dim=dims, out_dim=dims, bias=False),
        "adalayernorm": lambda: AdaLayerNorm(style_dim=dims, dims=dims),
        "instancenorm": lambda: nn.InstanceNorm1d(num_features=dims, affine=False, track_running_stats=False),     
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=dims),
        "instancenorm2d": lambda: nn.InstanceNorm2d(num_features=dims),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_dims=dims),
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

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

def apply_radii(freqs, x, ctx):
    F = x.shape[0] / ctx
    idx = torch.arange(ctx, device=device)
    idx = (idx * F).long().clamp(0, x.shape[0] - 1)
    x = x[idx]
    return torch.polar(x.unsqueeze(-1), freqs)

def compute_freqs(dims, head):
    head_dim = dims // head
    mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), head_dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 200 * mel_scale / 1000

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

def extract_pitch_contour(audio: np.ndarray, sr: int,
                            fmin: float = 50, fmax: float = 500,
                            frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]: # -extremely- inefficient fallback

    try:
        if LIBROSA_AVAILABLE:
            f0 = librosa.pyin(
                audio,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length
            )[0]

            nans = np.isnan(f0)
            if not nans.all():
                x = np.arange(len(f0))
                f0[nans] = np.interp(x[nans], x[~nans], f0[~nans])
            else:
                f0 = np.full_like(f0, 50.0)  

            times = librosa.frames_to_time(
                np.arange(len(f0)),
                sr=sr,
                hop_length=frame_length // 4
            )
        else:
            hop_length = frame_length // 4
            num_frames = 1 + (len(audio) - frame_length) // hop_length
            f0 = np.zeros(num_frames)

            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]

                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(corr)//2:]

                min_lag = int(sr / fmax)
                max_lag = int(sr / fmin)

                if max_lag < len(corr):
                    search_range = corr[min_lag:max_lag]
                    if len(search_range) > 0:
                        peak = np.argmax(search_range) + min_lag
                        if peak > 0:
                            f0[i] = sr / peak
                        else:
                            f0[i] = 150.0
                    else:
                        f0[i] = 150.0
                else:
                    f0[i] = 150.0

            times = np.arange(num_frames) * hop_length / sr
        return f0, times

    except Exception as e:

        num_frames = max(1, len(audio) // (frame_length // 4))
        return np.full(num_frames, 50.0), np.linspace(0, len(audio)/sr, num_frames)

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
        f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0 = pw.stonemask(audio.numpy().astype(np.float64), f0, t, sample_rate)
        # f0, t = extract_pitch_contour(audio.numpy().astype(np.float64), sample_rate, frame_length=hop_length*4)
        p_tensor = torch.from_numpy(f0).unsqueeze(0).to(device, dtype)
        # tensor = torch.polar(p_tensor, pt_tensor)

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

def sinusoids(ctx, dims, theta=THETA):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding    

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, n_type, norm=False, enc=False):
        super().__init__()
        
        act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        
        n.encoder = nn.Sequential(
            act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1),
            act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()
        n.ln = get_norm(n_type, dims) if norm else nn.Identity()

    def _process_feature(n, x):
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
               
    def forward(n, x):
        if isinstance(x, TensorDict):
            return x.apply(n._process_feature)
        else:
            return n._process_feature(x)
  
class prepare_datasets(torch.utils.data.Dataset):
    def __init__(self, metadata_file, data_dir, tokenizer=None, extract_args=None):
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.extract_args = extract_args if extract_args is not None else {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.data_dir, row["audio"])
        
        batch_input = {
            "audio": audio_path,
            "transcription": row["sentence"]
        }
        
        features = extract_features(
            batch_input, 
            tokenizer=self.tokenizer, 
            **self.extract_args
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

class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)
    
class SineGen(nn.Module):
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines

    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

