import pyworld as pw
import os
import math
import logging
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Any
from functools import partial
from datetime import datetime
from datasets import load_dataset, Audio, concatenate_datasets
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
import evaluate
from dataclasses import dataclass

extractor = None
tokenizer = None
optimizer = None
scheduler = None
model = None
Residual = None
MultiheadA = None
Echo = None
     
metric = evaluate.load(path="wer")

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    text_idx: int
    mels: int
    aud_ctx: int
    aud_dims: int
    aud_head: int
    aud_idx: int
    act: str
    debug: List[str]
    cross_attn: bool
    features: List[str]
    f0_rotary: bool

def align_f0(f0, ctx):
    ctx = torch.tensor(ctx)
    bat, length = f0.shape
    if length == ctx:
        return f0 
    frames = length / ctx
    idx = torch.arange(ctx, device=f0.device)
    idx = (idx * frames).long()
    batch_idx = torch.arange(bat, device=f0.device).unsqueeze(1)
    return f0[batch_idx, idx.unsqueeze(0).expand(bat, -1)]
    
@dataclass
class DataCollator:
    tokenizer: Any
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1
        
        batch = {}

        if "spectrogram" in features[0] and features[0]["spectrogram"] is not None:
            spectrogram_list = [f["spectrogram"] for f in features]
            max_len_feat = max(f.shape[-1] for f in spectrogram_list)
            pad_spectrogram = []
            for feat in spectrogram_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=pad_token_id)
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
                    pad_wav = F.pad(wav, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_wav = wav
                pad_waveforms.append(pad_wav)
            batch["waveform"] = torch.stack(pad_waveforms)

        if "label" in features[0] and features[0]["label"] is not None:
            labels_list = [f["label"] for f in features]
            max_len = max(len(l) for l in labels_list)
            all_ids = []
            all_labels = []

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

        if "pitch" in features[0] and features[0]["pitch"] is not None:
            pitch_list = [f["pitch"] for f in features]
            max_len_pitch = max(e.shape[-1] for e in pitch_list)
            pad_pitch = []
            for pitch in pitch_list:
                current_len = pitch.shape[-1]
                padding = max_len_pitch - current_len
                if padding > 0:
                    pad_pitch_item = F.pad(pitch, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_pitch_item = pitch
                pad_pitch.append(pad_pitch_item)
            batch["pitch"] = torch.stack(pad_pitch)

        if "f0" in features[0] and features[0]["f0"] is not None:
            input_ids_batch = batch.get("input_ids", None)  
            if input_ids_batch is not None:
                target_length = input_ids_batch.shape[-1] 
                aligned_list = []
                original_list = []
                for feature in features:
                    f0 = feature["f0"]
                    original_list.append(f0)
                    if f0.shape[-1] != target_length:
                        aligned_f0 = align_f0(f0.unsqueeze(0), target_length).squeeze(0)
                    else:
                        aligned_f0 = f0
                    aligned_list.append(aligned_f0)
                batch["f0d"] = torch.stack(aligned_list)  
                batch["f0"] = torch.stack(original_list)  

        if "envelope" in features[0] and features[0]["envelope"] is not None:
            env_list = [f["envelope"] for f in features]
            max_len = max(f.shape[-1] for f in env_list)
            pad_env = []
            for feat in env_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_feat = feat
                pad_env.append(pad_feat)
            batch["envelope"] = torch.stack(pad_env)

        if "phase" in features[0] and features[0]["phase"] is not None:
            ph_list = [f["phase"] for f in features]
            max_len = max(f.shape[-1] for f in ph_list)
            pad_ph = []
            for feat in ph_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_feat = feat
                pad_ph.append(pad_feat)
            batch["phase"] = torch.stack(pad_ph) 
        return batch

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
        
def load_wave(wave_data, sample_rate):
    if isinstance(wave_data, str):
        waveform, sr = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sr = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if sr != sample_rate:
        original_length = waveform.shape[1]
        target_length = int(original_length * (sample_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        
    return waveform.flatten()

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
        f0 = torch.from_numpy(f0).float()
        batch["pitch"] = f0.unsqueeze(0)  
        
    if frequency:
        wav_np = wav.numpy().astype(np.float64)  
        f0, t = pw.dio(wav_np, sampling_rate, 
                    frame_period=hop_length/sampling_rate*1000)
        f0 = pw.stonemask(wav_np, f0, t, sampling_rate)
        f0 = f0 
        batch["f0"] = torch.from_numpy(f0).float()  
                  
    if spectrogram and waveforms and pitch:
        spec_mean = batch["spectrogram"].mean()
        spec_std = batch["spectrogram"].std() + 1e-6
        batch["spectrogram"] = (batch["spectrogram"] - spec_mean) / spec_std
        
        wav_mean = batch["waveform"].mean()
        wav_std = batch["waveform"].std() + 1e-6
        batch["waveform"] = (batch["waveform"] - wav_mean) / wav_std
        
        if batch["pitch"].max() > 1.0:
            pitch_min = 50.0
            pitch_max = 600.0
            batch["pitch"] = (batch["pitch"] - pitch_min) / (pitch_max - pitch_min)
            
    batch["label"] = tokenizer.encode(batch["transcription"], add_special_tokens=False)
    return batch

def compute_metrics(eval_pred, compute_result: bool = True, 
                    print_pred: bool = False, num_samples: int = 0, tokenizer=None, pitch=None, model=None):
    
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
            
    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()
        
    label_ids = [[0 if token == -100 else token for token in seq] for seq in label_ids]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

    if print_pred:
        for i in range(min(num_samples, len(pred_str))):
            print(f"Preds: {pred_str[i]}")
            print(f"Label: {label_str[i]}")
            print(f"preds: {pred_ids[i]}")
            print(f"label: {label_ids[i]}")
            print("--------------------------------")  
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    if model is None:
        global global_model
        if 'global_model' in globals():
            model = global_model
    
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        if trainable_params > 0:
            efficiency_score = (100 - wer) / trainable_params
        else:
            print("Warning: Zero trainable parameters detected")
            efficiency_score = 0.0
    else:
        print("Warning: Model not available for parameter counting")
        trainable_params = 0.0
        efficiency_score = 0.0
    
    if hasattr(wer, "item"):
        wer = wer.item()
    
    metrics = {
        "wer": float(wer),
        "trainable_params_M": float(trainable_params),
        "efficiency_score": float(efficiency_score),
    }
    
    return metrics

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
        "google/fleurs", 
        "en_us", 
        token=token, 
        trust_remote_code=True,
        streaming=False)
    
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000)).select_columns(["audio", "transcription"])
    
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
        columns_to_remove = list(next(iter(dataset.values())).features)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"].take(50)
        logger.info(f"Train dataset size: {train_dataset.num_rows}, Test dataset size: {test_dataset.num_rows}")
        
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
    save_steps: int = 1000,
    eval_steps: int = 1,
    warmup_steps: int = 0,
    num_train_epochs: int = 1,
    logging_steps: int = 1,
    eval_on_start: bool = False,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
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
        max_grad_norm=max_grad_norm,     
    )

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
            logging_steps = 1,
            eval_on_start = False,
            learning_rate = 5e-6,
            weight_decay = 0.01,
            )
        else:
            training_args = get_training_args(
            log_dir,
            batch_eval_metrics = False,
            max_steps = 1000,
            save_steps = 1000,
            eval_steps = 100,   
            warmup_steps = 100,
            logging_steps = 10,
            eval_on_start = False,
            learning_rate = 2.5e-4,
            weight_decay = 0.01,
            )

        return training_args
        
    param = Dimensions(
        mels=128,
        aud_ctx=1500,
        aud_head=4,
        aud_dims=512,
        aud_idx=4,
        vocab=40000,
        text_ctx=512,
        text_head=4,
        text_dims=512,
        text_idx=4,
        act="swish",
        debug={},#{"encoder", "decoder", "residual", "rotary"},
        cross_attn=True,
        f0_rotary=False, 
        features = ["spectrogram"]#, "waveform", "pitch", "f0", "envelope", "phase"],
        )

    sanity_check = False
    training_args = sanity(sanity_check)
    dataset_config = {
        "spectrogram": True,
        "waveforms": False,
        "pitch": False,
        "downsamples": False,
        "frequency": True,
        "hilbert": False,
        "hop_length": 128,
        "fmin": 150,
        "fmax": 2000,
        "n_mels": 128,
        "n_fft": 1024,
        "sampling_rate": 16000,
        "pad_mode": "constant",
        "center": True, 
        "power": 2.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False}
    
    model = create_model(param)
    
    global global_model
    global_model = model
    
    metrics_fn = partial(compute_metrics, print_pred=False, num_samples=5, 
                    tokenizer=tokenizer, model=model)
    
    print(f"{'Sanity check' if sanity_check else 'Training'} mode")
    train_dataset, test_dataset = prepare_datasets(
        tokenizer=tokenizer,
        token=token,
        sanity_check=sanity_check,
        dataset_config=dataset_config)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
        compute_metrics=metrics_fn,
        ) 
       
    model.init_weights()
    trainer.train()

if __name__ == "__main__":
    main()

