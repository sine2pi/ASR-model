import math, json, os
import torch
import numpy as np
from datetime import datetime
# this is what versioning hell looks like
class MaxFactorA1(torch.optim.Optimizer):
    __version__ = "0.01"

    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-10, 1e-3), d=1.0, weight_decay=0.01, gamma=0.99, max=False):
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, max=max)
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
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
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

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = step_float ** group["beta2_decay"]
                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])

        return loss


class MaxFactor2(torch.optim.Optimizer):
    __version__ = "2.0"
    
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-10, 1e-3), d=1.0, weight_decay=0.01, gamma=0.99, max=False, min_lr=1e-9):
        
        print(f"Using MaxFactor optimizer v{self.__version__}")
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, max=max, min_lr=min_lr)
        super().__init__(params=params, defaults=defaults)

    def get_lr(self):
        param_specific_lrs = []

        for group in self.param_groups:
            group_lrs = []
            min_lr = group.get("min_lr", 1e-7)
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    continue
                step_float = state["step"].item()

                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                param_norm = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
                alpha = max(eps2, param_norm) * rho_t
                group_lrs.append(alpha)
            if group_lrs:
                param_specific_lrs.append(sum(group_lrs) / len(group_lrs))
            else:
                param_specific_lrs.append(group["lr"])
        return param_specific_lrs
    
    def get_last_lr(self):
        return self.get_lr()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.dim() > 1:
                        row_shape, col_shape = list(p.shape), list(p.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = p.new_zeros(row_shape)
                        state["col_var"] = p.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = self.state[param]

                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var, col_var, vi = row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, max(0.001, step_float ** group["beta2_decay"]))
                
                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, (param.norm() / (param.numel() ** 0.5 + 1e-12)).item()) * rho_t

                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                
                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)
                
                state["step"] = step_t
                
        return loss

class MaxFactor3(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, w_decay=0.025, gamma=0.99, max=False, bias=1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, w_decay=w_decay, 
                        gamma=gamma, max=max, bias=bias)

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
            p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
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

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
                beta_t = step_float ** group["beta_decay"]

                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                # rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
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
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                # param.add_(update, alpha=-group["lr"])

# For a 1D parameter (like a bias vector), update.abs().max(dim=-1, keepdim=True)[0] finds the single largest absolute 
# value in the entire update vector and broadcasts it. This means **every element of the 1D parameter is updated by the same 
# magnitude, determined by the most extreme value. Acts as a strong form of regularization, forcing all biases in a layer to move in unison.
# for the 2D weight matrix, update.abs().max(dim=-1, keepdim=True)[0] finds the maximum absolute value per row.
# The direction of the update for each individual bias term (+ or -) is still determined by its own gradient, via update.sign(). This creates a small bias for outliers.
# The "outliers" that the max update amplifies are not statistical noise; they are the most information-rich, crucial parts of the pitch signal. (good for pitch bad for spectrograms)
# The median update, by design, filters these critical signals out (good for spectrograms bad for pitch).
# The max update latches onto the single largest gradient signal from these critical events and forces the entire group of related parameters 
# (all biases in a layer) to react strongly. It treats these spikes as the most important thing to learn from in that step.
# The median update looks at all the gradients for a parameter group and chooses the middle value. The critical "spike" from the pitch event is treated as an outlier and ignored. 
# The update is instead based on the more numerous, less important gradients from stable or unvoiced parts of the audio. 
 
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                if group["bias"] == 1: 
                    param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                elif group["bias"] == 2: 
                    param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch?.
                    if param.dim() > 1:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        param.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        param.add_(-alpha / denom * update.sign())
             
        return loss    

class MaxFactorA(torch.optim.Optimizer):
    __version__ = "0.5"

    def __init__(self, params, lr=0.025, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, w_decay=0.025, gamma=0.99, max=False, bias=1, min_lr=0.0):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, w_decay=w_decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr)

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
            p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group["min_lr"]
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

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                beta_t = step_float ** group["b_decay"]

                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                # rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
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
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                param.add_(update, alpha=-group["lr"])

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

                if group["bias"] == 1: 
                    param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                elif group["bias"] == 2: 
                    param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch
                    if param.dim() > 1:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        param.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        param.add_(-alpha / denom * update.sign())
             
        return loss               


class MaxFactor(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, w_decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

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

# class MaxFactor2(torch.optim.Optimizer):
#     __version__ = "2.0"
    
#     def __init__(self, params, lr=0.01, beta_decay=-0.8, eps=(1e-10, 1e-3), d=1.0, w_decay=0.01, gamma=0.99, max=False, min_lr=1e-9):
        
#         print(f"Using MaxFactor optimizer v{self.__version__}")
        
#         defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, w_decay=w_decay, 
#                         gamma=gamma, max=max, min_lr=min_lr)
#         super().__init__(params=params, defaults=defaults)

#     def get_lr(self):
#         param_lr = []

#         for group in self.param_groups:
#             group_lrs = []
#             min_lr = group.get("min_lr", 1e-7)
#             eps1, eps2 = group["eps"]
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 state = self.state[p]
#                 if "step" not in state:
#                     continue
#                 step_float = state["step"].item()

#                 rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
#                 param_norm = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
#                 alpha = max(eps2, param_norm) * rho_t
#                 group_lrs.append(alpha)
#             if group_lrs:
#                 param_lr.append(sum(group_lrs) / len(group_lrs))
#             else:
#                 param_lr.append(group["lr"])
#         return param_lr
    
#     def get_last_lr(self):
#         return self.get_lr()

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             p_grad, grads, row_v, col_v, v, state_steps = [], [], [], [], [], []
#             eps1, eps2 = group["eps"]
#             min_lr = group.get("min_lr", 1e-7)
            
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
                    
#                 grad = p.grad
#                 if grad.dtype in {torch.float16, torch.bfloat16}:
#                     grad = grad.float()

#                 state = self.state[p]
#                 if len(state) == 0:
#                     state["step"] = torch.tensor(0.0, dtype=torch.float32)
#                     if p.grad.dim() > 1:
#                         row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
#                         row_shape[-1], col_shape[-2] = 1, 1
#                         state["row_var"] = p.grad.new_zeros(row_shape)
#                         state["col_var"] = p.grad.new_zeros(col_shape)
#                     state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

#                 row_v.append(state.get("row_var", None))
#                 col_v.append(state.get("col_var", None))
#                 v.append(state["v"])
#                 state_steps.append(state["step"])
#                 p_grad.append(p)
#                 grads.append(grad)

#             for i, param in enumerate(p_grad):
#                 grad = grads[i]
#                 state = self.state[param]

#                 if group["max"]:
#                     grad = -grad
                    
#                 step_t = state_steps[i]
#                 row_var, col_var, vi = row_v[i], col_v[i], v[i]

#                 if eps1 is None:
#                     eps1 = torch.finfo(param.dtype).eps
                
#                 step_t += 1
#                 step_float = step_t.item()
                
#                 beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
                
#                 rho_t = min(group["lr"], 1 / (step_float ** 0.5))
#                 # rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
#                 alpha = max(eps2, (param.norm() / (param.numel() ** 0.5)).item()) * rho_t

#                 if group["w_decay"] > 0:
#                     param.mul_(1 - group["lr"] * group["w_decay"])

#                 if grad.dim() > 1:
#                     row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
#                     row_mean.div_(grad.size(-1) + eps1)
                    
#                     row_var.lerp_(row_mean, beta_t)
                    
#                     col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
#                     col_mean.div_(grad.size(-2) + eps1)
                    
#                     col_var.lerp_(col_mean, beta_t)
                    
#                     var_est = row_var @ col_var
#                     max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
#                     var_est.div_(max_row_var.clamp_(min=eps1))
#                 else:
#                     vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
#                     var_est = vi

#                 update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
#                 inf_norm = torch.norm(update, float('inf'))
#                 if inf_norm > 0:
#                     update.div_(inf_norm.clamp_(min=eps1))
                
#                 # denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                
#                 denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
#                 param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])

#                 # if param.dim() > 1:
#                 #     max_vals = update.abs().max(dim=-1, keepdim=True)[0]
#                 #     param.add_(-alpha / denom * update.sign() * max_vals)
#                 # else:
#                 #     param.add_(-alpha / denom * update)
                
#                 state["step"] = step_t
                
#         return loss
    



class FrequencyHandler:
    """Base class for parameter-specific frequency analysis functions"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        """Default frequency analysis implementation"""
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        
        # Divide into bands
        band_size = freq_power.shape[0] // n_bands
        if band_size <= 0:
            return [0.0] * n_bands
            
        band_powers = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i+1) * band_size, freq_power.shape[0])
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Default adaptive momentum calculation"""
        n_bands = len(band_values)
        high_freq_activity = sum(band_values[n_bands//2:])
        
        if high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha

class ConvFrequencyHandler(FrequencyHandler):
    """Specialized handler for convolutional layers"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        # More precise sampling for convolutional layers
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        
        # Use logarithmically spaced bands for convolution layers
        # to better capture both low and high frequency patterns
        band_powers = []
        total_freqs = freq_power.shape[0]
        
        for i in range(n_bands):
            # Log-spaced indices
            start_idx = int((total_freqs ** (i/n_bands)) - 1)
            end_idx = int((total_freqs ** ((i+1)/n_bands)) - 1)
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, total_freqs)
            
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Convolutional layers benefit from more smoothing in mid-frequencies"""
        n_bands = len(band_values)
        
        # Calculate band distribution 
        mid_freq_activity = sum(band_values[n_bands//4:(3*n_bands)//4])
        high_freq_activity = sum(band_values[(3*n_bands)//4:])
        
        # Increase momentum more for mid-frequency noise that often appears in conv layers
        if mid_freq_activity > 0.4:
            return min(0.97, base_alpha + 0.07)
        elif high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha

class AttentionFrequencyHandler(FrequencyHandler):
    """Specialized handler for attention layers"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        # Standard frequency analysis but with more bands in higher frequencies
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        
        # Attention layers often have important high-frequency patterns
        # Use more bands in high frequencies
        band_powers = []
        half_bands = n_bands // 2
        
        # Low frequency bands (first half)
        low_band_size = (freq_power.shape[0] // 2) // half_bands
        for i in range(half_bands):
            start_idx = i * low_band_size
            end_idx = min((i+1) * low_band_size, freq_power.shape[0] // 2)
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        # High frequency bands (second half with more detail)
        high_band_size = (freq_power.shape[0] - (freq_power.shape[0] // 2)) // (n_bands - half_bands)
        for i in range(half_bands, n_bands):
            start_idx = (freq_power.shape[0] // 2) + (i - half_bands) * high_band_size
            end_idx = min((freq_power.shape[0] // 2) + (i - half_bands + 1) * high_band_size, freq_power.shape[0])
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Custom adaptive momentum for attention layers"""
        n_bands = len(band_values)
        
        # Get band with maximum energy
        max_band_idx = np.argmax(band_values)
        
        # Attention matrices often benefit from lower momentum for low frequencies
        if max_band_idx < n_bands // 4:
            # Dominant low frequency - less smoothing
            return max(0.85, base_alpha - 0.05)
        elif max_band_idx > 3*n_bands // 4:
            # Dominant high frequency - more smoothing
            return min(0.98, base_alpha + 0.08)
        return base_alpha

class EmbeddingFrequencyHandler(FrequencyHandler):
    """Specialized handler for embedding layers"""
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Embeddings often benefit from very stable updates"""
        n_bands = len(band_values)
        
        # More aggressive smoothing for high-frequency components in embeddings
        high_freq_activity = sum(band_values[(3*n_bands)//4:])
        if high_freq_activity > 0.2:  # Lower threshold for embeddings
            return min(0.98, base_alpha + 0.08)
        return base_alpha

class FAMOptimizer(torch.optim.Optimizer):
    """
    Frequency-Adaptive Momentum optimizer with parameter-specific handlers.
    
    Args:
        ... (existing parameters)
        debug (bool, optional): Whether to collect debug information (default: False)
        debug_dir (str, optional): Directory to save debug info (default: './fam_debug')
        debug_interval (int, optional): Steps between debug dumps (default: 1000)
    """
    def __init__(self, params, lr=1e-3, alpha=0.9, beta=0.99, eps=1e-8,
                 weight_decay=0, n_bands=8, fam_start_step=100,
                 layer_boost=True, min_size=256, debug=False,
                 debug_dir='./fam_debug', debug_interval=1000):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps,
                       weight_decay=weight_decay, n_bands=n_bands,
                       fam_start_step=fam_start_step, 
                       layer_boost=layer_boost, min_size=min_size)
        self.debug = debug
        self.debug_info = {} if debug else None
        
        # Debug file settings
        self.debug_dir = debug_dir
        self.debug_interval = debug_interval
        self.last_dump_step = 0
        
        if debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            self.debug_file = os.path.join(
                debug_dir, 
                f"fam_debug_{datetime.now().strftime('%m%d_%H%M%S')}.json"
            )
            # Initialize the debug file with basic information
            with open(self.debug_file, 'w') as f:
                json.dump({
                    "optimizer": "FAMOptimizer",
                    "settings": {
                        "lr": lr,
                        "alpha": alpha,
                        "beta": beta,
                        "n_bands": n_bands,
                        "fam_start_step": fam_start_step,
                    },
                    "parameters": {},
                    "steps_recorded": []
                }, f, indent=2)
            print(f"FAM debug info will be saved to {self.debug_file}")
        
        # Register frequency handlers
        self.handlers = {
            "default": FrequencyHandler(),
            "conv": ConvFrequencyHandler(),
            "attention": AttentionFrequencyHandler(),
            "embedding": EmbeddingFrequencyHandler()
        }
        
        # Process param groups to add handlers
        param_groups = self._add_handlers_to_groups(params)
        super(FAMOptimizer, self).__init__(params=param_groups, defaults=defaults)
        
        print(f"FAM Optimizer initialized with parameter-specific handlers:") 
        print(f"  lr={lr}, alpha={alpha}, beta={beta}, n_bands={n_bands}")
        print(f"  fam_start_step={fam_start_step}, min_size={min_size}")
    
    def _add_handlers_to_groups(self, params):
        """Add appropriate handlers to parameter groups based on type"""
        if isinstance(params, list) and all(isinstance(pg, dict) for pg in params):
            # Already parameter groups, add handlers
            for pg in params:
                if 'handler' not in pg:
                    # Detect parameter type
                    if any('conv' in name.lower() for name in pg.get('names', [])):
                        pg['handler'] = 'conv'
                    elif any(name in name.lower() for name in pg.get('names', []) 
                             for name in ['attention', 'mha', 'self_attn']):
                        pg['handler'] = 'attention'
                    elif any(name in name.lower() for name in pg.get('names', [])
                             for name in ['embed', 'token']):
                        pg['handler'] = 'embedding'
                    else:
                        pg['handler'] = 'default'
            return params
        else:
            # Just parameters, wrap in default group
            return [{'params': params, 'handler': 'default'}]
    
    def get_handler(self, group):
        """Get the appropriate frequency handler for the parameter group"""
        handler_name = group.get('handler', 'default')
        return self.handlers[handler_name]
    
    def dump_debug_info(self, force=False):
        """Save the current debug information to file"""
        if not self.debug or not hasattr(self, 'debug_file'):
            return
        
        # Get current step - use the max step across all parameters
        current_step = max([self.state[p]['step'] for p in self.state], default=0)
        
        # Only dump if enough steps have passed or if forced
        if force or (current_step - self.last_dump_step >= self.debug_interval):
            try:
                # Load existing data
                with open(self.debug_file, 'r') as f:
                    debug_data = json.load(f)
                
                # Update with new data
                debug_data["steps_recorded"].append(current_step)
                
                for param_name, param_info in self.debug_info.items():
                    if param_name not in debug_data["parameters"]:
                        debug_data["parameters"][param_name] = {
                            "handler": param_info.get('handler', 'default'),
                            "steps": [],
                            "bands": [],
                            "alpha": []
                        }
                    
                    # Add new data points since last dump
                    last_recorded = len(debug_data["parameters"][param_name]["steps"])
                    if last_recorded < len(param_info['steps']):
                        debug_data["parameters"][param_name]["steps"].extend(param_info['steps'][last_recorded:])
                        debug_data["parameters"][param_name]["bands"].extend(param_info['bands'][last_recorded:])
                        debug_data["parameters"][param_name]["alpha"].extend(param_info['alpha'][last_recorded:])
                
                # Write updated data
                with open(self.debug_file, 'w') as f:
                    json.dump(debug_data, f)
                
                self.last_dump_step = current_step
                
                # Clear memory to prevent it from growing too large
                for param_info in self.debug_info.values():
                    param_info['steps'] = param_info['steps'][-10:]  # Keep only most recent entries
                    param_info['bands'] = param_info['bands'][-10:]
                    param_info['alpha'] = param_info['alpha'][-10:]
                    
            except Exception as e:
                print(f"Error dumping FAM debug info: {e}")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('FAMOptimizer does not support sparse gradients')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['freq_history'] = {}
                    state['param_name'] = f"param_{p_idx}"
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                beta = group['beta']
                lr = group['lr']
                n_bands = group['n_bands']
                
                # Get the appropriate handler for this parameter
                handler = self.get_handler(group)
                
                should_apply_fam = (
                    state['step'] > group['fam_start_step'] and
                    p.numel() > group['min_size']
                )
                
                if should_apply_fam:
                    try:
                        # Sample gradients for efficiency
                        if p.numel() > 10000:
                            if p.dim() > 1:
                                row_indices = torch.randperm(p.size(0))[:min(p.size(0), 64)]
                                col_indices = torch.randperm(p.size(1))[:min(p.size(1), 64)]
                                grad_sample = grad[row_indices][:, col_indices].flatten()
                            else:
                                sample_idx = torch.randperm(p.numel())[:1000]
                                grad_sample = grad.flatten()[sample_idx]
                        else:
                            grad_sample = grad.flatten()
                        
                        # Use parameter-specific frequency analysis
                        band_powers = handler.analyze(grad_sample, n_bands, group['eps'])
                        
                        # Add a few print statements to debug the first few iterations
                        if state['step'] <= 10 and p_idx == 0:
                            print(f"Step {state['step']}: Found {len(band_powers)} frequency bands")
                            print(f"Band powers: {[f'{v:.4f}' for v in band_powers]}")
                        
                        # Update frequency history
                        for i, power in enumerate(band_powers):
                            band_key = f'band_{i}'
                            if band_key not in state['freq_history']:
                                state['freq_history'][band_key] = power
                            else:
                                state['freq_history'][band_key] = (
                                    beta * state['freq_history'][band_key] +
                                    (1-beta) * power
                                )
                        
                        # Get current band values from history
                        band_values = [state['freq_history'].get(f'band_{i}', 0) 
                                      for i in range(n_bands)]
                        
                        # Use parameter-specific adaptive momentum
                        effective_alpha = handler.get_adaptive_momentum(band_values, alpha)
                        
                        if self.debug:
                            param_name = state['param_name']
                            if param_name not in self.debug_info:
                                self.debug_info[param_name] = {
                                    'steps': [], 
                                    'bands': [], 
                                    'handler': group.get('handler', 'default'),
                                    'alpha': []
                                }
                            
                            if state['step'] % 10 == 0:
                                self.debug_info[param_name]['steps'].append(state['step'])
                                self.debug_info[param_name]['bands'].append(band_values)
                                self.debug_info[param_name]['alpha'].append(effective_alpha)
                        
                        # Apply adaptive momentum update
                        exp_avg.mul_(effective_alpha).add_(grad, alpha=1-effective_alpha)
                    except Exception as e:
                        # Enhance the error reporting
                        import traceback
                        print(f"Error in FAM processing for parameter {p_idx}:")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Error message: {e}")
                        print(f"Parameter shape: {p.shape}, numel: {p.numel()}")
                        print(traceback.format_exc())
                        # Then fallback to standard momentum as you already do
                        exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                else:
                    # Standard momentum update
                    exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                
                # Apply update
                p.add_(exp_avg, alpha=-lr)
        
        if self.debug:
            # Check if we need to dump debug info
            self.dump_debug_info()
        
        return loss
    
    def __del__(self):
        """Clean up and final debug dump when optimizer is destroyed"""
        if self.debug:
            self.dump_debug_info(force=True)

def get_parameter_groups(model, lr=1e-3, weight_decay=0.0):
    """
    Create parameter groups for FAMOptimizer with appropriate handlers based on layer type
    """
    param_groups = []
    
    # Group parameters by layer type
    conv_params = []
    conv_names = []
    
    attn_params = []
    attn_names = []
    
    embed_params = []
    embed_names = []
    
    norm_params = []
    norm_names = []
    
    other_params = []
    other_names = []
    
    # Classify parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(x in name.lower() for x in ['conv', 'cnn']):
            conv_params.append(param)
            conv_names.append(name)
        elif any(x in name.lower() for x in ['attention', 'mha', 'self_attn']):
            attn_params.append(param)
            attn_names.append(name)
        elif any(x in name.lower() for x in ['embed', 'token']):
            embed_params.append(param)
            embed_names.append(name)
        elif any(x in name.lower() for x in ['norm', 'batch', 'layer']):
            norm_params.append(param)
            norm_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)
    
    # Create parameter groups with appropriate handlers
    if conv_params:
        param_groups.append({
            'params': conv_params,
            'names': conv_names,
            'lr': lr,
            'weight_decay': weight_decay,
            'alpha': 0.9,
            'handler': 'conv',
            'n_bands': 10
        })
    
    if attn_params:
        param_groups.append({
            'params': attn_params,
            'names': attn_names,
            'lr': lr,
            'weight_decay': weight_decay,
            'alpha': 0.92,
            'handler': 'attention',
            'n_bands': 12
        })
    
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'names': embed_names,
            'lr': lr * 0.8,  # Typically slower learning rate for embeddings
            'weight_decay': weight_decay * 1.5,  # More regularization for embeddings
            'alpha': 0.95,
            'handler': 'embedding',
            'n_bands': 8
        })
    
    if norm_params:
        param_groups.append({
            'params': norm_params,
            'names': norm_names,
            'lr': lr,
            'weight_decay': 0.0,  # No weight decay for normalization
            'alpha': 0.9,
            'handler': 'default',
            'n_bands': 4
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'names': other_names,
            'lr': lr,
            'weight_decay': weight_decay,
            'alpha': 0.9,
            'handler': 'default',
            'n_bands': 8
        })
    
    return param_groups
