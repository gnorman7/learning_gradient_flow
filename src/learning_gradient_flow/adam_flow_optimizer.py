# mypy: allow-untyped-defs
import warnings
from typing import Optional, Callable, Dict, Any, Union
from dataclasses import dataclass, asdict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT  # Use PyTorch's type hint
import learning_gradient_flow.sindy_tools as sindy_tools

# import pysindy as ps
import numpy as np
import learning_gradient_flow.gradient_flow_optimizer

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
    warnings.warn(
        "torchdiffeq library not found. GradientFlow optimizers using odeint will not work. "
        "Install with: pip install torchdiffeq"
    )

__all__ = ["VectorBasedOptimizer", "GradientFlow", "SINDyFlow", "SINDyFlowTrustRegion", "TrustRegionControl"]


# Only support a single parameter group.
class CustomAdam(learning_gradient_flow.gradient_flow_optimizer.VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.state = {'epoch': 0}
        self.state['exp_avg'] = None
        self.state['exp_avg_sq'] = None

    def _apply_adam_step(self, flat_params: Tensor, flat_grads: Tensor):
        # Apply the Adam update step
        beta1, beta2 = self.defaults['betas']
        lr = self.defaults['lr']
        eps = self.defaults['eps']

        # Initialize optimizer state (moment vectors) if this is the first step
        if self.state.get('exp_avg') is None:  # Use .get for safety, though we init to None
            self.state['exp_avg'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            self.state['exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)

        self.state['epoch'] += 1
        step_t = self.state['epoch']

        # Biased estimates
        exp_avg: Tensor = self.state['exp_avg']
        exp_avg_sq: Tensor = self.state['exp_avg_sq']

        # Biased estimates
        exp_avg.lerp_(flat_grads, weight=1 - beta1)
        exp_avg_sq.lerp_(torch.square(flat_grads), weight=1 - beta2)

        # Unbiased estimates
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t

        bias_correction2_sqrt = bias_correction2 ** 0.5

        lr_w_mhat_factor = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        # Update parameters: += lr_w_mhat_factor * exp_avg / denom
        flat_params.addcdiv_(exp_avg, denom, value=-lr_w_mhat_factor)
        self._set_params_from_flat(flat_params)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # get grads using VectorBasedOptimizer
        flat_params = self._gather_flat('params')
        flat_grads = self._gather_flat('grads')

        # Apply the Adam step
        self._apply_adam_step(flat_params, flat_grads)

        if closure is not None:
            return loss


class AdamFlow(CustomAdam):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_kwargs: Dict[str, Any] = {},
    ):
        if retrain_interval < history_size:
            raise ValueError(f"Retrain interval must be greater than or equal to history size: {retrain_interval}!")
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.defaults['history_size'] = history_size
        self.defaults['retrain_interval'] = retrain_interval
        self.defaults['sindy_kwargs'] = sindy_kwargs
        self.state['func_evals'] = 0

        self.state['history_params'] = []  # tracks the parameter history
        self.state['history_grads'] = []  # tracks the gradient history
        self.state['history_count'] = 0

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        history_size: int = self.defaults['history_size']
        retrain_interval: int = self.defaults['retrain_interval']
        sindy_kwargs: Dict[str, Any] = self.defaults['sindy_kwargs']

        # Check if we need to retrain the SINDy model
        # Crieria is just based on epoch number and retrain interval, but could be more complex
        needs_retrain = False
        if self.state['epoch'] % retrain_interval == 0:
            needs_retrain = True


        if needs_retrain:
            self.state['history_params'] = []
            self.state['history_grads'] = []
            self.state['history_count'] = 0
            self.dynamics = None

        if self.state['history_count'] < history_size:
            # --- History Collection Phase ---
            # Use true grads for Adam update
            loss = closure()
            self.state['func_evals'] += 1
            flat_params = self._gather_flat('params')
            flat_grads = self._gather_flat('grads')
            self.state['history_params'].append(flat_params.clone().detach())
            self.state['history_grads'].append(flat_grads.clone().detach())
            self.state['history_count'] += 1

            self._apply_adam_step(flat_params, flat_grads)
            return loss
        else:
            orig_loss = closure()
            # self.state['func_evals'] += 1 # we are evaluating the loss here, but not using any info from it.

            # build grad model if it doesn't already exist
            if self.dynamics is None:
                pred = self.build_grad_model(**sindy_kwargs)

            y0 = self._gather_flat('params')  # Already detached inside gather_flat
            # pass into the pred model to generate the grad prediction
            grad_pred = pred(y0)

            # We will then use an Adam update step.
            # This will update the parameters directly.
            # This also updates the epoch counter, momentum vectors.
            self._apply_adam_step(y0, grad_pred)

            # Update evaluation counter
            self.state['func_evals'] += 0  # No extra evaluations for SINDy model

            return orig_loss

    def build_grad_model(self, poly_order: int = 1,
                          include_bias: bool = True,
                          rcond: Optional[float] = 1e-7,
                          ) -> Callable[[Tensor], Tensor]:
        # first called when 'func_evals' == history_size
        # build the SINDy model using the history of parameters
        # each entry of self.state['history'] is a tensor of shape (num_params,)
        x = torch.stack(self.state['history_params'], dim=0)  # history_size, num_params
        x_dots = torch.stack(self.state['history_grads'], dim=0)

        d = x.shape[1]
        library = sindy_tools.create_sindy_library(input_dim=d,
                                                    poly_order=poly_order,
                                                    include_bias=include_bias)
        Theta = library(x)
        lhs_target = x_dots
        rhs_mat = Theta

        Xi = torch.linalg.lstsq(rhs_mat, lhs_target, rcond=rcond).solution

        pred = sindy_tools.create_predictor(Xi, library)
        return pred
