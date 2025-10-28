import torch

class MaxFactor(torch.optim.Optimizer):
    __version__ = "1.0"

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
    
