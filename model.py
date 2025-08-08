
import warnings
import logging
from itertools import chain
import torch
from torch import nn, Tensor, einsum
import numpy as np
from dataclasses import dataclass
from einops import rearrange
from datetime import datetime
from echoutils import *
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

def sinusoids(ctx, dims, max_tscale=10000):
    assert dims % 2 == 0
    pos = torch.log(torch.tensor(float(max_tscale))) / (dims // 2 - 1)
    tscales = torch.exp(-pos * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    position = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) 
    positional_embedding = nn.Parameter(position, requires_grad=True)
    return positional_embedding

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
        "elu": nn.ELU()
    }
    return act_map.get(act, nn.GELU())

def there_is_a(val):
    return val is not None

@dataclass
class Dimensions:
    vocab: int
    mels: int
    ctx: int
    dims: int
    head: int
    layer: int
    act: str

class rotary(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head

        self.theta = nn.Parameter((torch.tensor(16000, device=device, dtype=dtype)), requires_grad=True)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x) -> Tensor:
        freqs = (self.theta / 220.0) * self.freqs_base 
        pos = torch.arange(x.shape[2], device=device, dtype=dtype) 
        freqs = pos[:, None] * freqs
        freqs = torch.polar(torch.ones_like(freqs), freqs)

        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)

class attentiona(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()

        self.head = head
        self.dims = dims
        self.head_dim = dims // head

        self.pad_token = 0
        self.zmin = 1e-6
        self.zmax = 1e-5     
        self.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)

        self.q = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)

        self.lna = nn.LayerNorm(dims) 
        self.rope = rotary(dims, head) 

    def forward(self, x, xa = None, mask = None):
        zero = self.zero

        q = self.q(self.lna(x))
        k, v = self.kv(self.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        q = self.rope(q)
        k = self.rope(k)

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale 

        scale = torch.ones_like(k[:, :, :, 0])
        zero = torch.clamp(F.softplus(zero), 1e-6, 1e-5)
        scale[k[:, :, :, 0].float() == 0] = zero
   
        if there_is_a(mask):
            i, j = qk.shape[-2:]
            mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
            qk = qk.masked_fill(mask,  -torch.finfo(qk.dtype).max) * scale.unsqueeze(-2).expand(qk.shape)
            qk = F.sigmoid(qk)

        qk = qk * scale.unsqueeze(-2)
        qk = taylor_softmax(qk, order=2)

        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out

class tgate(nn.Module):
    def __init__(self, dims, num_types=4):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, dims), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
    def forward(self, x):
        types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        cgate = torch.sum(gates * types.unsqueeze(2), dim=-1)
        return cgate

class residual(nn.Module): 
    def __init__(self, dims: int, head: int, act: str = "silu"):
        super().__init__()

        self.lna = nn.LayerNorm(dims, bias=False)           
        self.atta = attentiona(dims, head)
   
        self.tgate = tgate(dims, num_types=1)
        self.mlp = nn.Sequential(nn.Linear(dims, dims*4), get_activation(act), nn.Linear(dims*4, dims))

    def forward(self, x: Tensor, xa = None, mask = None):

        out = self.atta(x, mask=mask)
        if  x.shape == out.shape:
            x = x + out
        else:
            x = out
        if xa is not None:
            x = x + self.atta(x, xa, mask=None)

        x = x + self.tgate(x)
        x = x + self.mlp(self.lna(x)) 
        return x

class processor(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, act: str = "gelu", modal=True): 
        super(processor, self).__init__()

        self.ln = nn.LayerNorm(dims)        
        self.token = nn.Embedding(vocab, dims)
        self.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)     

        self.positions = nn.Parameter(torch.empty(ctx, dims), requires_grad=True)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)

        act_fn = get_activation(act)        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        self.block = nn.ModuleList([residual(dims, head, act_fn) for _ in range(layer)])
        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)  
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, enc=None, sequential=False, modal=True, blend=False, kv_cache=None) -> Tensor:    
        mask = self.mask[:x.shape[1], :x.shape[1]]

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token(x.long()) + self.positions[offset : offset + x.shape[-1]])

        xa = self.encoder(xa).permute(0, 2, 1)
        xa = xa + self.audio(xa.shape[1], xa.shape[-1], 36000.0).to(device, dtype)

        for block in chain(self.block or []):
            xa = block(xa, mask=None)
            x  = block(x, mask=mask)
            x  = block(x, xa, mask=None)
            if blend:
                if sequential:
                    y = x
                else:
                    a = torch.sigmoid(self.blend)
                    x = a * x + (1 - a) * y

            xm = block(torch.cat([x, xa], dim=1), mask=mask) if modal else None    
            x  = block(xm[:, :x.shape[1]], xm[:, x.shape[1]:], mask=None) if modal else x
            if blend:
                if sequential:
                    y = x
                else:
                    a = torch.sigmoid(self.blend)
                    x = a * x + (1 - a) * y

        x = nn.functional.dropout(x, p=0.001, training=self.training)
        x = self.ln(x)        
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()
        return x 

class Model(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        self.processor = processor(
            vocab=param.vocab,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            act=param.act)       
        
    def forward(self, labels=None, input_ids=None, pitch=None, pitch_tokens=None, spectrogram=None, waveform=None):

        x = input_ids
        xa = pitch  
 
        enc = {}
        if spectrogram is not None:
            enc["spectrogram"] = spectrogram
        if waveform is not None:
            enc["waveform"] = waveform
        if pitch is not None:
            enc["pitch"] = pitch

        logits = self.processor(x, xa, enc)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)

        return {"logits": logits, "loss": loss} 

    def _init_weights(self, module):
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
        for name, module in self.named_modules():
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                self.init_counts["LayerNorm"] += 1                
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, residual):
                self.init_counts["Residual"] += 1
            elif isinstance(module, processor):
                self.init_counts["processor"] += 1

    def init_weights(self):
        print("Initializing model weights...")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=True, load_saved=False, save_dataset=True, cache_dir='E:/hf', extract_args=None, max_ctx=2048):

    if load_saved:
        if cache_dir is None:
            cache_dir = cache_dir
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

    def filter_func(x):
        return (0 < len(x["transcription"]) < max_ctx and
                len(x["audio"]["array"]) > 0 and
                len(x["audio"]["array"]) < max_ctx * 160)

    raw_train = load_dataset(
        "google/fleurs", "en_us", token=token, split="train", streaming=streaming).take(1000)
    raw_test = load_dataset(
        "google/fleurs", "en_us", token=token, split="test", streaming=streaming).take(100)

    raw_train = raw_train.filter(filter_func).cast_column("audio", Audio(sampling_rate=sample_rate))
    raw_test = raw_test.filter(filter_func).cast_column("audio", Audio(sampling_rate=sample_rate))
    train_dataset = raw_train.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
    test_dataset = raw_test.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
    train_dataset.save_to_disk(cache_file_train) if save_dataset is True else None
    test_dataset.save_to_disk(cache_file_test) if save_dataset is True else None
    return train_dataset, test_dataset

def main():
    token = ""
    log_dir = os.path.join('D:/newmodel/output/logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer("D:/newmodel/mod5/tokenizer.json") 

    extract_args = {
        "waveform": False,
        "spec": False,
        "pitch_tokens": False,
        "pitch": True,
        "harmonics": False,
        "aperiodics": False,
        "phase_mod": False,
        "crepe": False,        
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
    }

    param = Dimensions(vocab=40000, mels=128, ctx=2048, dims=512, head=4, layer=4, act="swish")
    
    train_dataset, test_dataset = prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=extract_args, max_ctx=param.ctx)

    model = Model(param).to('cuda')
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    from functools import partial
    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1000,
        eval_steps=100,
        save_steps=100,
        warmup_steps=10,
        logging_steps=10,
        logging_dir=log_dir,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="no",
        report_to=["tensorboard"],
        push_to_hub=False,
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        eval_on_start=False,
        batch_eval_metrics=False,
        disable_tqdm=False, 
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        learning_rate=0.00025,
        weight_decay=0.025,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-10, weight_decay=training_args.weight_decay, betas=(0.9, 0.999), amsgrad=False, foreach=False, fused=False, capturable=False, differentiable=False, maximize=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_args.max_steps, eta_min=1e-9, last_epoch=-1)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=metrics_fn,
        optimizers=(optimizer, scheduler)
    )

    model.init_weights()
    trainer.train()

if __name__ == "__main__":
    main()

