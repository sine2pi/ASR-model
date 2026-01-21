import math
import torch

class MaxFactor(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, w_decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if beta_decay <= -1.0 or beta_decay >= 1.0:
            raise ValueError("beta_decay must be in [-1, 1]")
        if d <= 0.0:
            raise ValueError("d must be positive")
        if w_decay < 0.0:
            raise ValueError("w_decay must be non-negative")
        if gamma <= 0.0 or gamma >= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if max not in [True, False]:
            raise ValueError("max must be True or False")
        if bias not in [0, 1, 2]:
            raise ValueError("bias must be 0, 1 or 2")

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, w_decay=w_decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_v, col_v, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_v.append(state.get("row_var", None))
                col_v.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_v[i], col_v[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
                beta_t = step_float ** group["beta_decay"]

                state["RMS"] = self._rms(param).item()
                
                # rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                rho_t = max(group["min_lr"], min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["w_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["w_decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                # update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

#######

                if group["bias"] == 1: 
                    scale = update.abs().max(dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale
                elif group["bias"] == 2: 
        
                    scale = torch.median(update.abs(), dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale
                else: 
            
                    final_direction = update

                step_size = alpha / denom
                      
                if group["clip"]:
                    param_rms = torch.norm(param) / (param.numel() ** 0.5)
                    max_allowed_step = param_rms * group["cap"]
                    
                    update_rms = (torch.norm(final_direction * step_size) / 
                                 (final_direction.numel() ** 0.5))
                    if update_rms > max_allowed_step:
                        step_size = step_size * (max_allowed_step / (update_rms + 1e-8))
                param.add_(final_direction, alpha=-step_size)

        return loss

                # if group["bias"] == 1: 
                #     param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                # elif group["bias"] == 2: 
                #     param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                # else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch, in theory.
                #     if param.dim() > 1:
                #         max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                #         param.add_(-alpha / denom * update.sign() * max_vals)
                #     else:
                #         param.add_(-alpha / denom * update.sign())

        #     param.add_(update, alpha=-group["lr"])
             
        # return loss               

class FAMScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for the linear warmup
        max_epochs: Total number of epochs
        warmup_start_lr: Initial learning rate for warmup
        eta_min: Minimum learning rate after cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8, eta_min=1e-8, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                                (self.max_epochs - self.warmup_epochs))) / 2
                   for base_lr in self.base_lrs]

class SimpleFAM(torch.optim.Optimizer):
    """
    Simplified Frequency-Adaptive Momentum optimizer
    
    A lightweight implementation that focuses on the core concepts
    without complex debugging or parameter-specific handlers.
    """
    def __init__(self, params, lr=0.001, alpha=0.9, beta=0.99):
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super(SimpleFAM, self).__init__(params, defaults)
        print(f"SimpleFAM initialized with lr={lr}, alpha={alpha}")
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                
                state['step'] += 1
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                
                # Only apply FAM to large tensors
                if p.numel() > 1000 and state['step'] > 100:
                    # Simple frequency analysis
                    grad_sample = p.grad.flatten()[:min(1000, p.numel())]
                    freq = torch.fft.rfft(grad_sample.float())
                    power = torch.abs(freq)
                    
                    # Calculate high vs low frequency ratio
                    half = power.shape[0] // 2
                    high_ratio = power[half:].sum() / (power.sum() + 1e-8)
                    
                    # Adjust momentum based on frequency
                    effective_alpha = min(0.98, alpha + 0.05 * high_ratio)
                    exp_avg.mul_(effective_alpha).add_(p.grad, alpha=1-effective_alpha)
                else:
                    # Standard momentum
                    exp_avg.mul_(alpha).add_(p.grad, alpha=1-alpha)
                
                # Update weights
                p.add_(exp_avg, alpha=-group['lr'])
        
        return loss

class FAMScheduler2(torch.optim.lr_scheduler._LRScheduler):
    """
    Step-based learning rate scheduler for FAM optimizer
    with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps=1000, total_steps=100000, 
                 decay_start_step=None, warmup_start_lr=1e-6, eta_min=1e-6, 
                 last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_step = decay_start_step if decay_start_step is not None else warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler2, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                    for base_lr in self.base_lrs]
        
        elif self.last_epoch < self.decay_start_step:
            # Optional plateau phase (constant LR between warmup and decay)
            return self.base_lrs
        
        else:
            # Cosine annealing decay phase
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.decay_start_step) / 
                                (self.total_steps - self.decay_start_step))) / 2 + 1e-8
                   for base_lr in self.base_lrs]

    
