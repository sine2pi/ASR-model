
import os
PATH = 'E:/hf'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import os
import math
import time
import logging
import warnings
import gzip
import base64
from functools import partial
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Tuple, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import numpy as np
import pyworld as pw
from datasets import load_dataset, Audio
import evaluate
import aiohttp

def ctx_to_samples(audio_ctx, hop_length):
    samples_token = hop_length * 2
    n_samples = audio_ctx * samples_token
    return n_samples

def load_wave(wave_data, sample_rate):
    if isinstance(wave_data, str):
        waveform, sr = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sr = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")

    if sr != sample_rate:
        original_length = waveform.shape[1]
        target_length = int(original_length * (sample_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        
    return waveform

def pad(array, target_length, axis=-1, dtype: torch.dtype = torch.float32):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(dtype)
    if torch.is_tensor(array):
        if array.shape[axis] > target_length:
            array = array.index_select(
                dim=axis,
                index=torch.arange(
                    end=target_length, device=array.device, dtype=torch.long
                ),
            )
        if array.shape[axis] < target_length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, target_length - array.shape[axis])
            array = F.pad(
                input=array, pad=[pad for sizes in pad_widths[::-1] for pad in sizes]
            )
        array = array.to(dtype=dtype)
    else:
        raise TypeError(
            f"Unsupported input type: {type(array)}. Expected torch.Tensor or np.ndarray."
        )
    return array

def exact_div(x, y):
    assert x % y == 0
    return x // y

metric = evaluate.load(path="wer")

def hilbert_transform(x):
    N = x.shape[-1]
    xf = torch.fft.rfft(x)
    h = torch.zeros(N // 2 + 1, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return torch.fft.irfft(xf * h, n=N)

def analytic_signal(x):
    return x + 1j * hilbert_transform(x)

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
    h1, h2 = torch.meshgrid(
        torch.fft.rfftfreq(x.shape[-2]) * 2 - 1,
        torch.fft.rfftfreq(x.shape[-1]) * 2 - 1,
        indexing='ij')
    h = -1j / (math.pi * (h1 + 1j*h2))
    h[0, 0] = 0 
    return torch.fft.irfft2(xf * h.to(x.device))

def process_spectrogram_with_hilbert(spec):
    analytic = spec + 1j * hilbert_transform(spec)
    envelope = torch.abs(analytic)
    phase = torch.angle(analytic)
    return envelope, phase
        
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

        for key in all_keys:
            if key == "label":
                labels_list = [f["label"] for f in features]
                max_len = max(len(l) for l in labels_list)
                all_ids, all_labels = [], []
                for label in labels_list:
                    label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                    decoder_input = [bos_token_id] + label_list
                    label_eos = label_list + [pad_token_id]
                    input_len = max_len + 1 - len(decoder_input)
                    label_len = max_len + 1 - len(label_eos)
                    padded_input = decoder_input + [pad_token_id] * input_len
                    padded_labels = label_eos + [pad_token_id] * label_len
                    all_ids.append(padded_input)
                    all_labels.append(padded_labels)
                batch["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)
            elif key in ["spectrogram", "input_features", "waveform", "pitch", "f0", "envelope", "phase"]:
                items = [f[key] for f in features if key in f]
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
                if key == "spectrogram":
                    batch["input_features"] = batch[key]
        return batch

def extract_features(batch, tokenizer, spectrogram, waveforms, pitch, frequency=False,
                     hop_length=128, fmin=0, fmax=8000, n_mels=128, n_fft=1024, sampling_rate=16000,
                     pad_mode="constant", center=True, power=2.0, window_fn=torch.hann_window, mel_scale="htk", 
                     norm=None, normalized=False, downsamples=False, period=False, hilbert=False):

    dtype = torch.float32
    device = torch.device("cuda:0")
    audio = batch["audio"]
    sampling_rate = audio["sampling_rate"]
    sr = audio["sampling_rate"]
    wav = load_wave(wave_data=audio, sample_rate=sr)

    if spectrogram:
        transform = torchaudio.transforms.MelSpectrogram(
            f_max=fmax,
            f_min=fmin,
            n_mels=n_mels,
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            norm=norm,
            normalized=normalized,
            power=power,
            center=center, 
            mel_scale=mel_scale,
            window_fn=window_fn,
            pad_mode=pad_mode)
        
        mel_spectrogram = transform(wav)      
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spec = (log_mel + 4.0) / 4.0
        spec = torch.tensor(spec)
        batch["spectrogram"] = spec
        
    if hilbert:
        envelope_list = []
        phase_list = []
        
        for ch_idx in range(spec.shape[0]):
            envelope, phase = process_spectrogram_with_hilbert(spec[ch_idx])
            envelope_list.append(envelope)
            phase_list.append(phase)
            
        batch["envelope"] = torch.stack(envelope_list)
        batch["phase"] = torch.stack(phase_list)
        
    wav_1d = wav.unsqueeze(0)
    
    if waveforms:
        batch["waveform"] = wav_1d
            
    if pitch:
        wav_np = wav.numpy().astype(np.float64)  
        f0, t = pw.dio(wav_np, sampling_rate, 
                    frame_period=hop_length/sampling_rate*1000)
        f0 = pw.stonemask(wav_np, f0, t, sampling_rate)
        f0 = torch.from_numpy(f0)
        batch["pitch"] = f0.unsqueeze(0)
        
    if frequency:
        wav_np = wav.numpy().astype(np.float64)  
        f0, t = pw.dio(wav_np, sampling_rate, frame_period=hop_length/sampling_rate*1000)
        f0 = pw.stonemask(wav_np, f0, t, sampling_rate)
        f0 = torch.from_numpy(f0)  
        batch["f0"] = f0
                  
    if spectrogram and waveforms and pitch:
        spec_mean = batch["spectrogram"].mean()
        spec_std = batch["spectrogram"].std() + 1e-6
        batch["spectrogram"] = (batch["spectrogram"] - spec_mean) / spec_std
        
        wav_mean = batch["waveform"].mean()
        wav_std = batch["waveform"].std() + 1e-6
        batch["waveform"] = (batch["waveform"] - wav_mean) / wav_std
        
        if batch["pitch"].max() > 1.0:
            pitch_min = 50.0
            pitch_max = 500.0
            batch["pitch"] = (batch["pitch"] - pitch_min) / (pitch_max - pitch_min)
            
    batch["label"] = tokenizer.encode(batch["transcription"], add_special_tokens=False)
    return batch

def compute_metrics(pred, compute_result: bool = True, 
                    print_pred: bool = False, num_samples: int = 0, tokenizer=None, pitch=None, model=None):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    if not isinstance(pred_ids, torch.Tensor):
        pred_ids = torch.tensor(pred_ids)
    if not isinstance(label_ids, torch.Tensor):
        label_ids = torch.tensor(label_ids)
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(dim=-1)
    pred_ids = pred_ids.tolist()
    label_ids = label_ids.tolist()
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    label_ids = [[pad_token_id if token == -100 else token for token in seq] for seq in label_ids]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    if print_pred:
        for i in range(min(num_samples, len(pred_str))):
            print(f"Pred:  {pred_str[i]}")
            print(f"Label: {label_str[i]}")
            print("-" * 30)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

logger = logging.getLogger(__name__)

def create_model(param: Dimensions) -> Echo:
    model = Echo(param).to('cuda')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    return model

def setup_tokenizer(token: str, local_tokenizer_path: str = "D:/newmodel/model/tokenn/"):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{local_tokenizer_path}/tokenizer.json")
    orig_encode = tokenizer.encode
    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]
            ids = [id for id in ids if id not in sp_ids]
        return ids

    def bdec(ids_list, skip_special_tokens=True):
        results = []
        for ids in ids_list:
            if not isinstance(ids, list):
                ids = ids.tolist()
            if skip_special_tokens:
                ids = [id for id in ids if id not in [0, 1, 2]]
            results.append(tokenizer.decode(ids))
        return results        
    def save_pretrained(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(f"{save_dir}/tokenizer.json")
    tokenizer.encode = enc
    tokenizer.batch_decode = bdec
    tokenizer.save_pretrained = save_pretrained
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer

def prepare_datasets(tokenizer, token: str, sanity_check: bool = False, dataset_config: Optional[Dict] = None) -> Tuple[any, any]:

    if sanity_check:

        dataset = load_dataset(
            "./librispeech_asr.py", "clean", "train.100",
            storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
            token=token, trust_remote_code=True, streaming=False)

        dataset = dataset.rename_column("text", "transcription")
        dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000)).select_columns(["audio", "transcription"])

        dataset = dataset["test"].take(10)
        dataset = dataset.select_columns(["audio", "transcription"])
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        dataset = dataset.map(function=prepare_fn, remove_columns=["audio", "transcription"]).with_format(type="torch")
        train_dataset = dataset
        test_dataset = dataset
    else:
        cache_dir = "./processed_datasets"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_train = os.path.join(cache_dir, "train.arrow")
        cache_file_test = os.path.join(cache_dir, "test.arrow")

        if os.path.exists(cache_file_train) and os.path.exists(cache_file_test):
            from datasets import Dataset
            train_dataset = Dataset.load_from_disk(cache_file_train)
            test_dataset = Dataset.load_from_disk(cache_file_test)
            return train_dataset, test_dataset   
    
        if dataset_config is None:
            dataset_config = {
                "spectrogram": True,
                "waveforms": True,
                "pitch": True,
                "frequency": True,
                "downsamples": True,
                "hop_length": 128,
                "fmin": 50,
                "fmax": 2000,
                "n_mels": 128,
                "n_fft": 1024,
                "sampling_rate": 16000,
            }

        dataset = load_dataset(
            "./librispeech_asr.py", "clean", "train.100",
            storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
            token=token, trust_remote_code=True, streaming=False)

        dataset = dataset.rename_column("text", "transcription")
        dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000)).select_columns(["audio", "transcription"])

        def filter_func(x):
            return (0 < len(x["transcription"]) < 512 and
                   len(x["audio"]["array"]) > 0 and
                   len(x["audio"]["array"]) < 1500 * 160)
        
        dataset = dataset.filter(filter_func)
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)

        train_dataset = dataset["train.100"].take(10000)
        test_dataset = dataset["test"].take(1000)
        train_dataset = train_dataset.map(
            function=prepare_fn, 
            remove_columns=["audio", "transcription"]
        ).with_format(type="torch")
        
        test_dataset = test_dataset.map(
            function=prepare_fn, 
            remove_columns=["audio", "transcription"]
        ).with_format(type="torch")

        train_dataset.save_to_disk(cache_file_train)
        test_dataset.save_to_disk(cache_file_test)

    return train_dataset, test_dataset

from typing import List, Dict, Any

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

        for key in all_keys:
            if key == "label":
                labels_list = [f["label"] for f in features]
                max_len = max(len(l) for l in labels_list)
                all_ids, all_labels = [], []
                for label in labels_list:
                    label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                    decoder_input = [bos_token_id] + label_list
                    label_eos = label_list + [pad_token_id]
                    input_len = max_len + 1 - len(decoder_input)
                    label_len = max_len + 1 - len(label_eos)
                    padded_input = decoder_input + [pad_token_id] * input_len
                    padded_labels = label_eos + [pad_token_id] * label_len
                    all_ids.append(padded_input)
                    all_labels.append(padded_labels)
                batch["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)
            elif key in ["spectrogram", "input_features", "waveform", "pitch", "f0", "envelope", "phase"]:
                items = [f[key] for f in features if key in f]
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
                if key == "spectrogram":
                    batch["input_features"] = batch[key]
        return batch

def train_and_evaluate(
    model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn,
    max_steps=10000, device='cuda', accumulation_steps=1, clear_cache=True,
    log_interval=10, eval_interval=100, save_interval=1000,
    checkpoint_dir="checkpoint_dir", log_dir="log_dir"
):
    model.to(device)
    global_step = 0
    scaler = torch.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0

    progress_bar = tqdm(total=max_steps, desc="Training Progress", leave=True, colour='green')

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
                logging.info(f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0
                step_in_report = 0

        start_time = time.time()

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.autocast(device_type="cuda"):
            output = model(**batch) if hasattr(model, '__call__') else model.forward(**batch)
            logits = output["logits"] if isinstance(output, dict) and "logits" in output else output
            labels = batch["labels"]
            active_logits = logits.view(-1, logits.size(-1))
            active_labels = labels.view(-1)
            active_mask = active_labels != 0
            active_logits = active_logits[active_mask]
            active_labels = active_labels[active_mask]
            loss = loss_fn(active_logits, active_labels)
        total_loss += loss.item()
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = batch["input_features"].size(0) / (end_time - start_time)

        if global_step % log_interval == 0:
            writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar(tag='LearningRate', scalar_value=lr, global_step=global_step)
            writer.add_scalar(tag='SamplesPerSec', scalar_value=samples_per_sec, global_step=global_step)

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
                    eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
                    output = model(**eval_batch) if hasattr(model, '__call__') else model.forward(**eval_batch)
                    logits = output["logits"] if isinstance(output, dict) and "logits" in output else output
                    labels = eval_batch["labels"]
                    batch_size = logits.size(0)
                    total_samples += batch_size
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    eval_loss += loss.item()
                    all_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    batch_count += 1

            eval_time = time.time() - eval_start_time
            loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

            writer.add_scalar('Loss/eval', loss_avg, global_step)
            writer.add_scalar('WER', metrics['wer'], global_step)
            writer.add_scalar('EvalSamples', total_samples, global_step)
            writer.add_scalar('EvalTimeSeconds', eval_time, global_step)

            lr = scheduler.get_last_lr()[0]
            print(f"• STEP:{global_step} • samp:{samples_per_sec:.1f} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}")
            logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}")
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        global_step += 1
        step_in_report += 1

        avg_loss = total_loss / (global_step + 1)
        postfix_dict = {
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.6f}',
            'samp': f'{samples_per_sec:.1f}'
        }
        progress_bar.set_postfix(postfix_dict, refresh=True)
        progress_bar.update(1)

    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed after {global_step} steps. Final model saved to {final_model_path}")
    writer.close()
    progress_bar.close()

def get_optimizer(model, lr=5e-4, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6, betas=(0.9, 0.98))

def get_scheduler(optimizer, total_steps=10000):
    return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=total_steps, last_epoch=-1)

def get_loss_fn():
    return torch.nn.CrossEntropyLoss(ignore_index=0)

def main():
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer(token)

    param = Dimensions(
        mels=128, aud_ctx=1500, aud_head=4, aud_dims=512, aud_idx=4,
        vocab=40000, text_ctx=512, text_head=4, text_dims=512, text_idx=4,
        act="swish", debug={}, cross_attn=True, features=["spectrogram"]
    )

    dataset_config = {
        "spectrogram": True, "waveforms": False, "pitch": False, "downsamples": False,
        "frequency": True, "hilbert": False, "hop_length": 128, "fmin": 150, "fmax": 2000,
        "n_mels": 128, "n_fft": 1024, "sampling_rate": 16000, "pad_mode": "constant",
        "center": True, "power": 2.0, "window_fn": torch.hann_window, "mel_scale": "htk",
        "norm": None, "normalized": False
    }

    model = create_model(param)
    train_dataset, test_dataset = prepare_datasets(
        tokenizer=tokenizer, token=token, sanity_check=False, dataset_config=dataset_config
    )

    collator = DataCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collator, num_workers=0)
    eval_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator, num_workers=0)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    loss_fn = get_loss_fn()

    train_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        max_steps=10000,
        device='cuda',
        accumulation_steps=1,
        clear_cache=False,
        log_interval=10,
        eval_interval=500,
        save_interval=10000,
        checkpoint_dir="./checkpoints",
        log_dir=log_dir
    )

if __name__ == "__main__":
    main()

