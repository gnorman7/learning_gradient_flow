# mypy: allow-untyped-defs
from typing import Callable, Optional, Union
import math
import warnings
from typing import Optional, Callable, Dict, Any, Union
from dataclasses import dataclass, asdict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT  # Use PyTorch's type hint

# import pysindy as ps
import numpy as np
import learning_gradient_flow.gradient_flow_optimizer
import learning_gradient_flow.sindy_tools as sindy_tools

from learning_gradient_flow.gradient_flow_optimizer import VectorBasedOptimizer, TrustRegionControl
from learning_gradient_flow.sindy_tools import SINDyParams

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
    warnings.warn(
        "torchdiffeq library not found. GradientFlow optimizers using odeint will not work. "
        "Install with: pip install torchdiffeq"
    )



# Only support a single parameter group.
class CustomAdam(VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.state = {'epoch': 0}
        self.state['exp_avg'] = None
        self.state['exp_avg_sq'] = None
        if amsgrad:
            self.state['max_exp_avg_sq'] = None


    def _apply_adam_step(self, flat_params: Tensor, flat_grads: Tensor):
        # Apply the Adam update step
        beta1, beta2 = self.defaults['betas']
        lr = self.defaults['lr']
        eps = self.defaults['eps']
        amsgrad = self.defaults['amsgrad']

        # Initialize optimizer state (moment vectors) if this is the first step
        if self.state.get('exp_avg') is None:  # Use .get for safety, though we init to None
            self.state['exp_avg'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            self.state['exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            if amsgrad:
                self.state['max_exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)

        self.state['epoch'] += 1
        step_t = self.state['epoch']

        # Biased estimates
        exp_avg: Tensor = self.state['exp_avg']
        exp_avg_sq: Tensor = self.state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq: Tensor = self.state['max_exp_avg_sq']

        # Biased estimates
        exp_avg.lerp_(flat_grads, weight=1 - beta1)
        exp_avg_sq.lerp_(torch.square(flat_grads), weight=1 - beta2)

        if amsgrad:
            max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)

        # Unbiased estimates
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t

        bias_correction2_sqrt = bias_correction2 ** 0.5
        if amsgrad:
            denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        lr_w_mhat_factor = lr / bias_correction1

        # Update parameters: += lr_w_mhat_factor * exp_avg / denom
        flat_params.addcdiv_(exp_avg, denom, value=-lr_w_mhat_factor)
        self._set_params_from_flat(flat_params)
        if amsgrad:
            self.state['max_exp_avg_sq'] = max_exp_avg_sq

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


class AdamODE(learning_gradient_flow.gradient_flow_optimizer.VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        timesteps_per_epoch: int = 10,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, timesteps_per_epoch=timesteps_per_epoch)
        super().__init__(params, defaults)
        self.state['epoch'] = 1
        flat_params = self._gather_flat('params')
        self.expanded_state = torch.zeros(len(flat_params), 3, dtype=flat_params.dtype, device=flat_params.device)
        self.expanded_state[:, 0] = flat_params

    def _get_grad(self, flat_params: Tensor) -> Tensor:
        """Computes the current gradient vector for given parameters.

        Args:
            flat_params: Current flat parameters
        """

        # Ensure gradients are enabled for loss computation and backward pass
        with torch.enable_grad():
            # Set model parameters temporarily to state 'y'
            self._set_params_from_flat(flat_params)

            # Closure should zero grads, compute loss, call backward()
            loss = self.closure()
            if not isinstance(loss, Tensor):
                warnings.warn("Closure did not return a Tensor.")

            # Track function evaluations
            self.evals_in_step[0] += 1

            flat_grad = self._gather_flat("grads")
            return flat_grad

    def expanded_state_derivative(self, t: float, expanded_state: Tensor, dt: float) -> Tensor:
        """Computes the time derivative of the expanded state.
        Args:
            t: Current time, e.g. epoch*dt
            expanded_state: Expanded state (n,3), params, momentum, variance
            dt: Time step size
        Returns:
            Tensor: Time derivative of the expanded state (n,3)
        """

        flat_params = expanded_state[:, 0]
        flat_momentum = expanded_state[:, 1]
        flat_variance = expanded_state[:, 2]

        # Compute the gradient
        flat_grad = self._get_grad(flat_params)
        beta1, beta2 = self.defaults['betas']
        eps = self.defaults['eps']

        # Mine
        # d_momentum_dt = ((1-beta1) / dt)*(flat_grad - flat_momentum)
        # d_variance_dt = ((1-beta2) / dt) * (flat_grad**2 - flat_variance)

        # beta1_t_term = beta1 ** ((t+dt) / dt)
        # beta2_t_term = beta2 ** ((t+dt) / dt)

        # flat_unbiased_momentum = flat_momentum / (1 - beta1_t_term)
        # flat_unbiased_variance = flat_variance / (1 - beta2_t_term)

        # d_params_dt = -flat_unbiased_momentum / (flat_unbiased_variance.sqrt() + eps)

        # silva_general_2020
        d_params_dt = -flat_momentum / (flat_variance.sqrt() + eps)
        # g_alpha1 = (1 - beta1) / (dt * (1 - beta1 ** (t/dt)))
        g_alpha1 = (1 - beta1) / (dt * (1 - beta1 ** ((t+dt)/dt)))
        d_momentum_dt = g_alpha1 * (flat_grad - flat_momentum)
        # g_alpha2 = (1 - beta2) / (dt * (1 - beta2 ** (t/dt)))
        g_alpha2 = (1 - beta2) / (dt * (1 - beta2 ** ((t+dt)/dt)))
        d_variance_dt = g_alpha2 * (flat_grad ** 2 - flat_variance)


        d_expanded_state_dt = torch.stack(
            [d_params_dt, d_momentum_dt, d_variance_dt], dim=1
        )
        return d_expanded_state_dt

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        timesteps_per_epoch = group['timesteps_per_epoch']
        self.evals_in_step = [0]  # Use list to pass by reference

        dt = lr/ timesteps_per_epoch

        self.closure = closure
        orig_loss = closure()

        if self.state['epoch'] == 1:
            flat_grad = self._gather_flat('grads')
            self.expanded_state[:, 1] = flat_grad.clone().detach()
            self.expanded_state[:, 2] = flat_grad.clone().detach() ** 2

        self.state['func_evals'] += 1

        t_span = torch.tensor([self.state['epoch'] * lr, (self.state['epoch'] + 1) * lr],
                              device=self._device)

        int_fn = lambda t, y: self.expanded_state_derivative(t, y, dt)

        expanded_state_result = torchdiffeq_odeint(
            int_fn,
            self.expanded_state,
            t_span,
            method='rk4',
            options={'step_size': dt},
        )

        self.state['func_evals'] += self.evals_in_step[0]

        # Get the final state after integration
        final_expanded_state = expanded_state_result[-1]
        self.expanded_state = final_expanded_state

        flat_params = final_expanded_state[:, 0]
        # Update the parameters in the optimizer
        self._set_params_from_flat(flat_params)

        self.state['epoch'] += 1

        return orig_loss


class AdamEuler(learning_gradient_flow.gradient_flow_optimizer.VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
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
        self.state['epoch'] = 0
        flat_params = self._gather_flat('params')
        self.expanded_state = torch.zeros(len(flat_params), 3, dtype=flat_params.dtype, device=flat_params.device)
        self.expanded_state[:, 0] = flat_params
        self.loss = None

    def _get_grad(self, flat_params: Tensor) -> Tensor:
        """Computes the current gradient vector for given parameters.

        Args:
            flat_params: Current flat parameters
        """

        # Ensure gradients are enabled for loss computation and backward pass
        with torch.enable_grad():
            # Set model parameters temporarily to state 'y'
            self._set_params_from_flat(flat_params)

            # Closure should zero grads, compute loss, call backward()
            loss = self.closure()
            if not isinstance(loss, Tensor):
                warnings.warn("Closure did not return a Tensor.")
            self.loss = loss

            # Track function evaluations
            self.evals_in_step[0] += 1

            flat_grad = self._gather_flat("grads")
            return flat_grad

    def step(self, closure: Callable[[], Tensor], same_time=False) -> Optional[Tensor]:
        """
        same_time: If True, update params, momentum, and variance at the same time, more resembling ODE, otherwise
            update params after momentum and variance, closer to true Adam update.
        """
        closure = torch.enable_grad()(closure)
        self.closure = closure

        group = self.param_groups[0]
        lr = group['lr']
        self.evals_in_step = [0]  # Use list to pass by reference

        dt = lr
        beta1, beta2 = self.defaults['betas']

        if self.state['epoch'] == 0:
            flat_params = self._gather_flat('params')
            flat_grad = self._get_grad(flat_params)

            self.expanded_state[:, 1] = (1 - beta1) * (flat_grad)
            self.expanded_state[:, 2] = (1 - beta2) * (flat_grad ** 2)
            self.state['epoch'] += 1

        # -------------------------------------------------------
        # Begin the evaluation of the time derivative function
        flat_params = self._gather_flat('params')
        flat_grad = self._get_grad(flat_params)
        eps = self.defaults['eps']
        self.state['epoch'] += 1
        step_t = self.state['epoch']

        # Apply the Euler update step, but just to the momentum and variance
        d_momentum_dt = ((1 - beta1) / dt) * (flat_grad - self.expanded_state[:, 1])
        d_variance_dt = ((1 - beta2) / dt) * (flat_grad ** 2 - self.expanded_state[:, 2])

        # Do update for these AT THE SAME TIME as params
        if same_time:
            momentum = self.expanded_state[:, 1]
            variance = self.expanded_state[:, 2]
        else:
            # Do update for these BEFORE using for params, like upwinding or Nesterov
            momentum = self.expanded_state[:, 1] + dt * d_momentum_dt
            variance = self.expanded_state[:, 2] + dt * d_variance_dt

        # epoch = t/dt
        # Note: even with same_time, we update from k+1, not from k (step_t = epoch + 1)
        unbiased_momentum = momentum / (1 - beta1 ** step_t)
        unbiased_variance = variance / (1 - beta2 ** step_t)

        # Now update the parameters using the updated momentum and variance
        d_params_dt = -unbiased_momentum / (unbiased_variance.sqrt() + eps)
        flat_params += d_params_dt * dt

        if same_time:
            momentum = self.expanded_state[:, 1] + dt * d_momentum_dt
            variance = self.expanded_state[:, 2] + dt * d_variance_dt

        # End the evaluation of the time derivative function
        #-------------------------------------------------------

        self.state['func_evals'] += self.evals_in_step[0]

        self.expanded_state = torch.stack(
            [flat_params, momentum, variance], dim=1
        )
        # Update the parameters in the optimizer
        self._set_params_from_flat(flat_params)

        return self.loss


class BaseAdamEulerLearned(VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        history_size: int = 100,
        sindy_params: Optional[SINDyParams] = None,
        subsample_factor: int = 1,
        points_added: int = 0,
        debug: bool = False,
        upgrade_momentum_with_model: bool = True
    ):
        if not lr > 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, history_size=history_size,
                        subsample_factor=subsample_factor)
        super().__init__(params, defaults)
        self.state['epoch'] = 0
        self.state['history'] = []
        self.state['grad_history'] = []
        self.state['history_count'] = 0
        self.state['sub_epoch'] = 0
        flat_params = self._gather_flat('params')
        self.expanded_state = torch.zeros(len(flat_params), 3, dtype=flat_params.dtype, device=flat_params.device)
        self.expanded_state[:, 0] = flat_params
        self.loss = None
        self.debug = debug
        self.points_added = points_added
        self.upgrade_momentum_with_model = upgrade_momentum_with_model

        if sindy_params is None:
            sindy_params = learning_gradient_flow.sindy_tools.SINDyParams()
            sindy_params.method = 'tracked'
        elif sindy_params.method != 'tracked':
            warnings.warn(f"SINDy method {sindy_params.method} is not 'tracked'. Using 'tracked' instead.")
            sindy_params.method = 'tracked'
        self.sindy_params = sindy_params

    def _get_grad(self, flat_params: Tensor) -> Tensor:
        """Computes the current gradient vector for given parameters, increments evals_in_step counter.

        Args:
            flat_params: Current flat parameters
        """

        # Ensure gradients are enabled for loss computation and backward pass
        with torch.enable_grad():
            # Set model parameters temporarily to state 'y'
            self._set_params_from_flat(flat_params)

            # Closure should zero grads, compute loss, call backward()
            loss = self.closure()
            if not isinstance(loss, Tensor):
                warnings.warn("Closure did not return a Tensor.")
            self.loss = loss

            # Track function evaluations
            self.evals_in_step[0] += 1

            flat_grad = self._gather_flat("grads")
            return flat_grad

    def _check_model_reset(self) -> bool:
        """Determines when the SINDy model should be reset. Should be True on the first iteration!"""
        raise NotImplementedError(
            "This method should be implemented in subclasses to determine WHEN to reset the model."
        )

    def _check_grad_alignment(self, flat_params: Tensor, flat_grad: Tensor, dt: float) -> Tensor:
        """Optionally checks if the gradient is aligned with the parameters."""
        return flat_grad

    def step(self, closure: Callable[[], Tensor], same_time=False) -> Optional[Tensor]:
        """
        same_time: If True, update params, momentum, and variance at the same time, more resembling ODE, otherwise
            update params after momentum and variance, closer to true Adam update.
        """
        closure = torch.enable_grad()(closure)
        self.closure = closure

        group = self.param_groups[0]
        lr = group['lr']
        eps = group['eps']
        history_size: int = group['history_size']
        self.evals_in_step = [0]  # Use list to pass by reference

        dt = lr
        beta1, beta2 = self.defaults['betas']

        if self.state['epoch'] == 0:
            flat_params = self._gather_flat('params')
            flat_grad = self._get_grad(flat_params)

            self.expanded_state[:, 1] = (1 - beta1) * (flat_grad)
            self.expanded_state[:, 2] = (1 - beta2) * (flat_grad ** 2)
            self.state['epoch'] += 1

        if self._check_model_reset():
            self.state['history'] = []
            self.state['grad_history'] = []
            self.state['history_count'] = 0
            self.dynamics = None
            if self.debug:
                print(f"[{self.__class__.__name__}] "
                      f"Epoch {self.state['epoch']}: Resetting SINDy model.")

        # -------------------------------------------------------
        # Begin the evaluation of the time derivative function
        flat_params = self._gather_flat('params')

        if self.state['history_count'] < history_size:
            flat_grad = self._get_grad(flat_params)
            self.state['history'].append(flat_params)
            self.state['grad_history'].append(flat_grad)
            self.state['history_count'] += 1
            model_is_active = False
        else:
            # build grad model if it doesn't already exist
            if self.dynamics is None:
                pred = self.build_sindy_model(sindy_params=self.sindy_params, points_added=self.points_added)
                self.dynamics = lambda y: pred(y)  # Don't need t dependence, as custom ODE solve

            # evaluate the model to get the grad
            flat_grad = self.dynamics(flat_params)
            flat_grad = self._check_grad_alignment(flat_params, flat_grad, dt)

            model_is_active = True

            # Compute loss, but it isn't used (or counted)
            self.loss = self.closure()

        if not model_is_active:
            self.state['epoch'] += 1
            step_t = self.state['epoch']

            # Apply the Euler update step, but just to the momentum and variance
            d_momentum_dt = ((1 - beta1) / dt) * (flat_grad - self.expanded_state[:, 1])
            d_variance_dt = ((1 - beta2) / dt) * (flat_grad ** 2 - self.expanded_state[:, 2])

            # Do update for these AT THE SAME TIME as params
            if same_time:
                momentum = self.expanded_state[:, 1]
                variance = self.expanded_state[:, 2]
            else:
                # Do update for these BEFORE using for params, like upwinding or Nesterov
                momentum = self.expanded_state[:, 1] + dt * d_momentum_dt
                variance = self.expanded_state[:, 2] + dt * d_variance_dt

            # epoch = t/dt
            unbiased_momentum = momentum / (1 - beta1 ** step_t)
            unbiased_variance = variance / (1 - beta2 ** step_t)

            # Now update the parameters using the updated momentum and variance
            d_params_dt = -unbiased_momentum / (unbiased_variance.sqrt() + eps)
            flat_params = flat_params + d_params_dt * dt

            if same_time:
                momentum = self.expanded_state[:, 1] + dt * d_momentum_dt
                variance = self.expanded_state[:, 2] + dt * d_variance_dt

            # End the evaluation of the time derivative function
            # -------------------------------------------------------

            self.state['func_evals'] += self.evals_in_step[0]
            self.expanded_state = torch.stack(
                [flat_params, momentum, variance], dim=1
            )
            self._set_params_from_flat(flat_params)

            return self.loss

        # Otherwise, model IS active, so we will use the subsampled grid.
        max_subepochs = self.defaults['subsample_factor']
        sub_dt = dt / max_subepochs
        for sub_epoch in range(max_subepochs):
            step_t = self.state['epoch'] + (sub_epoch+1) / max_subepochs
            # Apply the Euler update step, but just to the momentum and variance
            d_momentum_dt = ((1 - beta1) / sub_dt) * (flat_grad - self.expanded_state[:, 1])
            d_variance_dt = ((1 - beta2) / sub_dt) * (flat_grad ** 2 - self.expanded_state[:, 2])

            # Do update for these AT THE SAME TIME as params
            if same_time:
                momentum = self.expanded_state[:, 1]
                variance = self.expanded_state[:, 2]
            else:
                # Do update for these BEFORE using for params, like upwinding or Nesterov
                momentum = self.expanded_state[:, 1] + sub_dt * d_momentum_dt
                variance = self.expanded_state[:, 2] + sub_dt * d_variance_dt

            # epoch = t/dt
            unbiased_momentum = momentum / (1 - beta1 ** step_t)
            unbiased_variance = variance / (1 - beta2 ** step_t)

            # Now update the parameters using the updated momentum and variance
            d_params_dt = -unbiased_momentum / (unbiased_variance.sqrt() + eps)
            flat_params = flat_params + d_params_dt * sub_dt

            if same_time:
                momentum = self.expanded_state[:, 1] + sub_dt * d_momentum_dt
                variance = self.expanded_state[:, 2] + sub_dt * d_variance_dt

            if sub_epoch < max_subepochs - 1:
                flat_grad = self.dynamics(flat_params)
                # flat_grad = self._check_grad_alignment(flat_params, flat_grad, dt)

        self.state['func_evals'] += self.evals_in_step[0]
        if self.upgrade_momentum_with_model:
            self.expanded_state = torch.stack(
                [flat_params, momentum, variance], dim=1
            )
        self._set_params_from_flat(flat_params)

        self.state['epoch'] += 1
        return self.loss

    def interpolate(self, state: torch.Tensor, points_added: int = 0, order: int = 2) -> torch.Tensor:
        # state is history_size by num_params
        # Assume that the state is given equispaced. We will return equispaced too.
        # We will return something that is history_size * (1+points_added) by num_params
        history_size, num_params = state.shape
        if history_size < 3 and order == 2:
            raise ValueError("Cannot interpolate with order 2 on less than 3 points.")
        if points_added == 0:
            return state.clone()

        out = []
        for i in range(history_size - 1):
            out.append(state[i])
            if order == 1:
                for k in range(1, points_added+1):
                    alpha = k / (points_added + 1)
                    new_point = (1 - alpha) * state[i] + alpha * state[i + 1]
                    out.append(new_point)
            elif order == 2:
                if i == 0:
                    # Forward quadratic: points at x = 0, 1, 2
                    p0, p1, p2 = state[0], state[1], state[2]
                    for k in range(1, points_added + 1):
                        t = k / (points_added + 1)
                        L0 = (t - 1) * (t - 2) / 2.0
                        L1 = t * (2 - t)
                        L2 = t * (t - 1) / 2.0
                        out.append(p0 * L0 + p1 * L1 + p2 * L2)
                else:
                    # Centered quadratic: points at x = -1, 0, 1
                    pm, p0, pp = state[i - 1], state[i], state[i + 1]
                    for k in range(1, points_added + 1):
                        t = k / (points_added + 1)
                        Lm = t * (t - 1) / 2.0
                        L0 = 1.0 - t * t
                        Lp = t * (t + 1) / 2.0
                        out.append(pm * Lm + p0 * L0 + pp * Lp)

        out.append(state[-1])  # Always append the last point
        out = torch.stack(out, dim=0)

        return out


    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams,
                          points_added: int = 0) -> Callable[[Tensor], Tensor]:
        """Uses self.state['history'] to build a SINDy model.
        Args:
            poly_order: Polynomial order for the library.
            include_bias: Whether to include a constant term.
            truncation_rank: Rank for SVD truncation, None == no SVD.
            method: 'strong', 'weak', 'tracked' for system construction.
            test_func_params: For 'weak', param object to be passed to sindy_tools.assemble_weak_matrices
            solver_fn: Function to solve the linear system, takes rhs_mat, lhs_target, solver_params.
            solver_kwargs: kwargs for the solver function.
        Returns:
            A function that takes in a state and returns the state derivative.
        """

        # first called when 'func_evals' == history_size
        # build the SINDy model using the history of parameters
        # each entry of self.state['history'] is a tensor of shape (num_params,)

        x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
        n_hist = x.shape[0]
        x = self.interpolate(x, points_added=points_added, order=2)

        if sindy_params.truncation_rank is None:
            d = x.shape[1]
            library = sindy_tools.create_sindy_library(input_dim=d,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias,
                                                       use_ortho=sindy_params.use_ortho)
            Theta = library(x)
            dt_sindy = self.param_groups[0]['lr']
            # t_span = torch.arange(n_hist) * dt_sindy
            dt_intp = dt_sindy / (1 + points_added)
            t_span = torch.arange(x.shape[0]) * dt_intp
            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(x, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(x, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
            elif sindy_params.method == 'tracked':
                lhs_target = torch.stack(self.state['grad_history'], dim=0)
                lhs_target = self.interpolate(lhs_target, points_added=points_added, order=1)
                rhs_mat = Theta
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong', 'weak', or 'tracked'.")

            # General solution method for solving the linear system.
            # This should take the rhs_mat, lhs_target, and params object, returning the solution Xi
            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)

            pred = sindy_tools.create_predictor(Xi, library)
        else:
            truncation_rank = sindy_params.truncation_rank
            # Build SINDy model in SVD modes.
            # Note, this is different from MATLAB, as it returns Vh or V.T, not V
            U_full_svd, s_full_svd, Vh_full_svd = torch.linalg.svd(x.T, full_matrices=False)
            U_svd = U_full_svd[:, :truncation_rank]
            s_svd = s_full_svd[:truncation_rank]
            Vh_svd = Vh_full_svd[:truncation_rank, :]
            # change training data (x) to be in terms of mode coefficients
            mode_coeffs = torch.diag(s_svd) @ Vh_svd
            # now, we can build the library on the mode coefficients
            library = sindy_tools.create_sindy_library(input_dim=truncation_rank,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias,
                                                       use_ortho=sindy_params.use_ortho)
            Theta = library(mode_coeffs.T)
            dt_sindy = self.param_groups[0]['lr']
            t_span = torch.arange(x.shape[0]) * dt_sindy

            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(mode_coeffs.T, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(mode_coeffs.T, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
            elif sindy_params.method == 'tracked':
                grad_hist = torch.stack(self.state['grad_history'], dim=0)
                grad_hist = self.interpolate(grad_hist, points_added=points_added, order=1)
                # lhs_target = (U_svd.T @ grad_hist.T).T
                lhs_target = grad_hist @ U_svd
                rhs_mat = Theta
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong' or 'weak'.")

            # General solution method for solving the linear system.
            # This should take the rhs_mat, lhs_target, and params object, returning the solution Xi
            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)

            mode_pred = sindy_tools.create_predictor(Xi, library)
            # now, we need to create a predictor that takes in the full state and returns the state derivs,
            # not just on mode coeffs

            def pred(y: Tensor) -> Tensor:
                # y is d,1
                # compute the mode coefficients associated with y. We can do this through a matmul
                cur_mode_coeffs = U_svd.T @ y
                # generate predictions on the mode coefficients
                mode_pred_coeffs = mode_pred(cur_mode_coeffs)
                # now, we need to convert the mode predictions back to the full state space
                dydt = U_svd @ mode_pred_coeffs
                return dydt

        # dot(y) = g(y), this is g(y).
        # dot(y) = f(t, y) = g(y), f is dynamics
        return pred


class AdamEulerLearned(BaseAdamEulerLearned):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_params: Optional[SINDyParams] = None,
        subsample_factor: int = 1,
        points_added: int = 0,
        debug: bool = False,
        upgrade_momentum_with_model: bool = True
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, history_size=history_size,
                         sindy_params=sindy_params, subsample_factor=subsample_factor,
                         points_added=points_added, debug=debug,
                         upgrade_momentum_with_model=upgrade_momentum_with_model)
        if retrain_interval < history_size:
            raise ValueError(f"Retrain interval must be greater than or equal to history size: {retrain_interval}!")
        self.retrain_interval = retrain_interval

    def _check_model_reset(self) -> bool:
        """Check if the model should be reset based on the retrain interval."""
        return self.state['epoch'] % self.retrain_interval == 1


class AdamEulerLearnedTR(BaseAdamEulerLearned):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        history_size: int = 100,
        trust_region_control: TrustRegionControl = TrustRegionControl(),
        gamma_tr_radius: float = None,
        sindy_params: Optional[SINDyParams] = None,
        subsample_factor: int = 1,
        use_data_driven_thresh: bool = False,
        points_added: int = 0,
        debug: bool = False,
        upgrade_momentum_with_model: bool = True
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, history_size=history_size,
                         sindy_params=sindy_params, subsample_factor=subsample_factor,
                         points_added=points_added,
                         debug=debug, upgrade_momentum_with_model=upgrade_momentum_with_model)
        self.trust_region_control = trust_region_control
        if gamma_tr_radius is None:
            # If we set it just to the lr, we would check performance on the next step or two.
            # Give a little more extra default freedom to start.
            gamma_tr_radius = 10.0*lr
        self.gamma_tr_radius = gamma_tr_radius

        self.should_reset = True
        self.use_data_driven_thresh = use_data_driven_thresh

    def _check_model_reset(self) -> bool:
        """Check if the model should be reset based on the trust region control."""
        if self.should_reset:
            self.should_reset = False
            return True
        else:
            return False

    # TODO: use the grads from the history to get a reference range for acceptable angle differences
    # That is, compute the cosine similarity within self.state['history'] and use that to
    # inform thresholds, scaling trc.cosine_similarity_good_threshold and trc.cosine_similarity_bad_threshold
    def _check_grad_alignment(self,
                              flat_params: torch.Tensor,
                              flat_grad: torch.Tensor,
                              dt: float,
                              ) -> Tensor:
        y_diff = flat_params - self.state['history'][-1]
        y_diff_norm = torch.norm(y_diff, p=2)
        if y_diff_norm <= self.gamma_tr_radius:
            # We are within the trust region, so we can use the predicted gradient

            self._set_params_from_flat(flat_params)

            return flat_grad
        else:
            # We've exceeded the trust region
            # We'll compare against the true gradient, so return that regardless (after we compute it)
            # Still need to figure out how we change TR, and if we scrap the model.
            trc = self.trust_region_control

            if self.use_data_driven_thresh:
                # Use cosine similarity from history to get an idea of threshold adjustment
                number_of_cosines = min(3, len(self.state['grad_history']) - 1)
                cosines = []

                # get cosine similarities between the last few pairs
                # we could build some model, either with a distribution or with the change in this over time etc.
                # but for now, just use the last few pairs, assuming they're measuring the same thing
                for i in range(1, number_of_cosines + 1):
                    grad1 = self.state['grad_history'][-i]
                    grad2 = self.state['grad_history'][-(i + 1)]
                    cos_sim = torch.cosine_similarity(grad1, grad2, dim=0)
                    cosines.append(cos_sim.item())

                # Use the median cosine similarity as a threshold
                data_driven_thresh = torch.median(torch.tensor(cosines, dtype=flat_grad.dtype, device=flat_grad.device)).item()
                data_driven_thresh = max(data_driven_thresh, 0.05) # no negative allowed.
            else:
                data_driven_thresh = 1.0

            # _get_grad updates the loss evaluation count
            true_grad = self._get_grad(self._gather_flat('params'))
            if self.debug:
                print(f"[{self.__class__.__name__}] "
                      f"Epoch {self.state['epoch']}: Evaluated true gradient")

            # Case 0: both gradients are small.
            if torch.norm(true_grad) < trc.grad_tol * dt and torch.norm(flat_grad) < trc.grad_tol * dt:
                self.gamma_tr_radius *= 1.25
            else:
                cos_sim = torch.cosine_similarity(true_grad, flat_grad, dim=0)
                # Case 1: cos_sim is bad, so reduce the TR radius AND reset the model
                if cos_sim < trc.cosine_similarity_bad_threshold * data_driven_thresh:
                    self.gamma_tr_radius *= trc.radius_factor_bad
                    self.should_reset = True
                # Case 2: cos_sim is okay, so reduce the TR radius, but keep the model
                elif cos_sim < trc.cosine_similarity_good_threshold * data_driven_thresh:
                    # Alternatively for these two, we could interpolate the factor based on the cos sim
                    self.gamma_tr_radius *= trc.radius_factor_okay
                    self.should_reset = False
                # Case 3: cos_sim is good, so increase the TR radius, and keep the model
                else:
                    self.gamma_tr_radius *= trc.radius_factor_good
                    self.should_reset = False
            self.gamma_tr_radius = min(self.gamma_tr_radius, trc.max_radius)

            return true_grad


class AppendixAdam(VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.state = {'epoch': 0}
        self.state['exp_avg'] = None
        self.state['exp_avg_sq'] = None
        if amsgrad:
            self.state['max_exp_avg_sq'] = None
        self.state['func_evals'] = 0

    def step(self, closure: Optional[Callable[[], float]] = None):
        flat_params = self._gather_flat('params')

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grads = self._gather_flat('grads')
        self.state['func_evals'] += 1

        beta1, beta2 = self.defaults['betas']
        lr = self.defaults['lr']
        eps = self.defaults['eps']
        amsgrad = self.defaults['amsgrad']

        # Initialize optimizer state (moment vectors) if this is the first step
        if self.state.get('exp_avg') is None:  # Use .get for safety, though we init to None
            self.state['exp_avg'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            self.state['exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            if amsgrad:
                self.state['max_exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)

        # forward Euler step first
        # Biased estimates
        exp_avg: Tensor = self.state['exp_avg']
        exp_avg_sq: Tensor = self.state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq: Tensor = self.state['max_exp_avg_sq']

        dmdt_eta = (1 - beta1) * (grads - exp_avg)
        dvdt_eta = (1 - beta2) * (grads ** 2 - exp_avg_sq)

        # now parameter update, we already multiplied by dt.
        exp_avg = exp_avg + dmdt_eta
        exp_avg_sq = exp_avg_sq + dvdt_eta

        if amsgrad:
            max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)

        self.state['epoch'] += 1
        step_t = self.state['epoch']
        unbiased_exp_avg = exp_avg / (1 - beta1 ** step_t)
        if amsgrad:
            unbiased_exp_avg_sq = max_exp_avg_sq / (1 - beta2 ** step_t)
        else:
            unbiased_exp_avg_sq = exp_avg_sq / (1 - beta2 ** step_t)

        dparams_dt = -unbiased_exp_avg / (unbiased_exp_avg_sq.sqrt() + eps)
        flat_params = flat_params + lr * dparams_dt

        self._set_params_from_flat(flat_params)
        self.state['exp_avg'] = exp_avg
        self.state['exp_avg_sq'] = exp_avg_sq
        if amsgrad:
            self.state['max_exp_avg_sq'] = max_exp_avg_sq

        if closure is not None:
            return loss


class AppendixAdamLearned(VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        amsgrad: bool = False,
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_params: Optional[SINDyParams] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad,
                        history_size=history_size, retrain_interval=retrain_interval)
        super().__init__(params, defaults)
        self.state = {'epoch': 0}
        self.state['exp_avg'] = None
        self.state['exp_avg_sq'] = None
        if amsgrad:
            self.state['max_exp_avg_sq'] = None
        self.state['history'] = []
        self.state['grad_history'] = []
        self.state['func_evals'] = 0

        if sindy_params is None:
            sindy_params = learning_gradient_flow.sindy_tools.SINDyParams()
            sindy_params.method = 'tracked'
        elif sindy_params.method != 'tracked':
            warnings.warn(f"SINDy method {sindy_params.method} is not 'tracked'. Using 'tracked' instead.")
            sindy_params.method = 'tracked'
        self.sindy_params = sindy_params

    def _get_grad(self, flat_params: Tensor) -> Tensor:
        with torch.enable_grad():
            self._set_params_from_flat(flat_params)


    def step(self, closure: Optional[Callable[[], float]] = None):
        flat_params = self._gather_flat('params')

        # We do evaluate the loss on every step, but don't use any info in some cases.
        # In practice, we should group this step with the later if/else that uses the model or not.
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        beta1, beta2 = self.defaults['betas']
        lr = self.defaults['lr']
        eps = self.defaults['eps']
        amsgrad = self.defaults['amsgrad']
        history_size = self.defaults['history_size']
        retrain_interval = self.defaults['retrain_interval']

        # Check model reset (this is called on epoch 0)
        if self.state['epoch'] % retrain_interval == 0:
            self.state['history'] = []
            self.state['grad_history'] = []
            self.dynamics = None

        if len(self.state['history']) < history_size:
            grads = self._gather_flat('grads')
            self.state['history'].append(flat_params)
            self.state['grad_history'].append(grads)
            self.state['func_evals'] += 1
        else:
            if self.dynamics is None:
                pred = self.build_sindy_model(sindy_params=self.sindy_params)
                self.dynamics = lambda y: pred(y)
            grads = self.dynamics(flat_params)

        # Initialize optimizer state (moment vectors) if this is the first step
        if self.state.get('exp_avg') is None:  # Use .get for safety, though we init to None
            self.state['exp_avg'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            self.state['exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            if amsgrad:
                self.state['max_exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)

        # forward Euler step first
        # Biased estimates
        exp_avg: Tensor = self.state['exp_avg']
        exp_avg_sq: Tensor = self.state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq: Tensor = self.state['max_exp_avg_sq']

        dmdt_eta = (1 - beta1) * (grads - exp_avg)
        dvdt_eta = (1 - beta2) * (grads ** 2 - exp_avg_sq)

        # now parameter update, we already multiplied by dt.
        exp_avg = exp_avg + dmdt_eta
        exp_avg_sq = exp_avg_sq + dvdt_eta

        if amsgrad:
            max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)

        self.state['epoch'] += 1
        step_t = self.state['epoch']
        unbiased_exp_avg = exp_avg / (1 - beta1 ** step_t)
        if amsgrad:
            unbiased_exp_avg_sq = max_exp_avg_sq / (1 - beta2 ** step_t)
        else:
            unbiased_exp_avg_sq = exp_avg_sq / (1 - beta2 ** step_t)

        dparams_dt = -unbiased_exp_avg / (unbiased_exp_avg_sq.sqrt() + eps)

        flat_params = flat_params + lr * dparams_dt
        self._set_params_from_flat(flat_params)
        self.state['exp_avg'] = exp_avg
        self.state['exp_avg_sq'] = exp_avg_sq
        if amsgrad:
            self.state['max_exp_avg_sq'] = max_exp_avg_sq

        if closure is not None:
            return loss

    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams) -> Callable[[Tensor], Tensor]:
        x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
        dt_sindy = self.defaults['lr']
        t_span = torch.arange(x.shape[0]) * dt_sindy

        if sindy_params.truncation_rank is None:
            d = x.shape[1]
            library = sindy_tools.create_sindy_library(input_dim=d,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias,
                                                       use_ortho=sindy_params.use_ortho)
            Theta = library(x)
            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(x, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(x, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
            elif sindy_params.method == 'tracked':
                lhs_target = torch.stack(self.state['grad_history'], dim=0)
                rhs_mat = Theta
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong', 'weak', or 'tracked'.")

            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)
            pred = sindy_tools.create_predictor(Xi, library)
        else:
            truncation_rank = sindy_params.truncation_rank
            # Build SINDy model in SVD modes.
            # Note, this is different from MATLAB, as it returns Vh or V.T, not V
            U_full_svd, s_full_svd, Vh_full_svd = torch.linalg.svd(x.T, full_matrices=False)
            U_svd = U_full_svd[:, :truncation_rank]
            s_svd = s_full_svd[:truncation_rank]
            Vh_svd = Vh_full_svd[:truncation_rank, :]
            # change training data (x) to be in terms of mode coefficients
            mode_coeffs = torch.diag(s_svd) @ Vh_svd
            # now, we can build the library on the mode coefficients
            library = sindy_tools.create_sindy_library(input_dim=truncation_rank,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias,
                                                       use_ortho=sindy_params.use_ortho)
            Theta = library(mode_coeffs.T)

            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(mode_coeffs.T, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(mode_coeffs.T, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
            elif sindy_params.method == 'tracked':
                grad_hist = torch.stack(self.state['grad_history'], dim=0)
                # lhs_target = (U_svd.T @ grad_hist.T).T
                lhs_target = grad_hist @ U_svd
                rhs_mat = Theta
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong' or 'weak'.")

            # General solution method for solving the linear system.
            # This should take the rhs_mat, lhs_target, and params object, returning the solution Xi
            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)

            mode_pred = sindy_tools.create_predictor(Xi, library)
            # now, we need to create a predictor that takes in the full state and returns the state derivs,
            # not just on mode coeffs

            def pred(y: Tensor) -> Tensor:
                # y is d,1
                # compute the mode coefficients associated with y. We can do this through a matmul
                cur_mode_coeffs = U_svd.T @ y
                # generate predictions on the mode coefficients
                mode_pred_coeffs = mode_pred(cur_mode_coeffs)
                # now, we need to convert the mode predictions back to the full state space
                dydt = U_svd @ mode_pred_coeffs
                return dydt

        # dot(y) = g(y), this is g(y).
        # dot(y) = f(t, y) = g(y), f is dynamics
        return pred


# Assumes you already have this somewhere in the project
# from torchdiffeq import odeint as torchdiffeq_odeint
# and VectorBasedOptimizer provides:
#   _gather_flat("params"|"grads"), _set_params_from_flat(flat), self._device, etc.


class AppendixAdamContLearned(VectorBasedOptimizer):
    """
    Continuous-time (ODE-solved) version of the learned-gradient Adam:
      - Warm-up: run discrete Adam with true gradients and collect (a, grad) history
      - Learn: fit SINDy for grad ~= f_sindy(a)
      - Deploy: integrate the Adam ODE in (a, m, v) over [t_global, t_global + dt]
    """

    def __init__(
        self,
        params,
        *,
        # "eta" in the paper (nominal Adam step size used in bias correction + moment ODE scaling)
        lr: float = 1e-3,
        # Outer integration horizon per optimizer step
        dt: Optional[float] = None,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        history_size: int = 100,
        retrain_interval: int = 200,
        ode_solver_options: Optional[dict] = None,
        sindy_params=None,
        skip_unused_evals: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid eta: {lr}")
        if dt is None:
            dt = float(lr)
        if not 0.0 < dt:
            raise ValueError(f"Invalid dt: {dt}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=float(lr),
            dt=float(dt),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            amsgrad=bool(amsgrad),
            history_size=int(history_size),
            retrain_interval=int(retrain_interval),
            ode_solver_options=ode_solver_options or {},
        )
        super().__init__(params, defaults)

        self.state = {}
        self.state["epoch"] = 0                 # discrete outer-step count
        self.state["t_global"] = 0.0            # absolute time for bias correction
        self.state["func_evals"] = 0
        self.state["skip_unused_evals"] = skip_unused_evals

        # Adam states stored in flat form
        self.state["exp_avg"] = None
        self.state["exp_avg_sq"] = None
        self.state["max_exp_avg_sq"] = None     # if amsgrad

        # data buffers
        self.state["history"] = []
        self.state["grad_history"] = []

        # learned gradient surrogate
        self.dynamics = None

        # SINDy params
        if sindy_params is None:
            sindy_params = learning_gradient_flow.sindy_tools.SINDyParams()
            sindy_params.method = "tracked"
        elif sindy_params.method != "tracked":
            warnings.warn(f"SINDy method {sindy_params.method} is not 'tracked'. Using 'tracked' instead.")
            sindy_params.method = "tracked"
        self.sindy_params = sindy_params

    # ----------------------------
    # Helpers: pack/unpack ODE state
    # ----------------------------
    def _pack_state(self, a: Tensor, m: Tensor, v: Tensor, vmax: Optional[Tensor] = None) -> Tensor:
        if vmax is None:
            return torch.cat([a, m, v], dim=0)
        return torch.cat([a, m, v, vmax], dim=0)

    def _unpack_state(self, y: Tensor, d: int, amsgrad: bool):
        a = y[:d]
        m = y[d:2 * d]
        v = y[2 * d:3 * d]
        vmax = None
        if amsgrad:
            vmax = y[3 * d:4 * d]
        return a, m, v, vmax

    # ----------------------------
    # Main step
    # ----------------------------
    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        eta: float = group["lr"]
        dt: float = group["dt"]
        beta1, beta2 = group["betas"]
        eps: float = group["eps"]
        amsgrad: bool = group["amsgrad"]
        history_size: int = group["history_size"]
        retrain_interval: int = group["retrain_interval"]
        ode_options: dict = group["ode_solver_options"]

        # Periodic reset of data/surrogate (intended)
        if self.state["epoch"] % retrain_interval == 0:
            self.state["history"] = []
            self.state["grad_history"] = []
            self.dynamics = None

        skip_unused_evals = self.state["skip_unused_evals"]
        needs_true_eval = (len(self.state["history"]) < history_size) or (not skip_unused_evals)

        if needs_true_eval:
            loss = closure()
        else:
            loss = None

        # Pull flat params + grads; store safely
        a = self._gather_flat("params").detach()
        g = self._gather_flat("grads").detach()
        if needs_true_eval:
            self.state["func_evals"] += 1  # counts true gradient evals (if closure computed grads)

        # Initialize Adam moment states if needed
        if self.state["exp_avg"] is None:
            self.state["exp_avg"] = torch.zeros_like(a, memory_format=torch.preserve_format)
            self.state["exp_avg_sq"] = torch.zeros_like(a, memory_format=torch.preserve_format)
            if amsgrad:
                self.state["max_exp_avg_sq"] = torch.zeros_like(a, memory_format=torch.preserve_format)

        # ----------------------------
        # Warm-up / data phase: discrete Adam with true gradients
        # ----------------------------
        if len(self.state["history"]) < history_size:
            # Record snapshots defensively
            self.state["history"].append(a.clone().detach())
            self.state["grad_history"].append(g.clone().detach())

            # One discrete Adam step (same as your current AppendixAdamLearned ordering)
            m = self.state["exp_avg"]
            v = self.state["exp_avg_sq"]
            m = m + (1.0 - beta1) * (g - m)
            v = v + (1.0 - beta2) * (g * g - v)

            if amsgrad:
                vmax = self.state["max_exp_avg_sq"]
                vmax = torch.maximum(vmax, v)
            else:
                vmax = None

            # increment epoch/time (bias correction uses epoch count / absolute time)
            self.state["epoch"] += 1
            self.state["t_global"] += dt
            step_t = self.state["epoch"]

            mhat = m / (1.0 - (beta1 ** step_t))
            if amsgrad:
                vhat = vmax / (1.0 - (beta2 ** step_t))
            else:
                vhat = v / (1.0 - (beta2 ** step_t))

            dadt = -mhat / (vhat.sqrt() + eps)
            # Discrete update uses "eta" as the step size in the paper; keep consistent with your derivation
            a_next = a + eta * dadt
            self._set_params_from_flat(a_next)

            # Save updated moments
            self.state["exp_avg"] = m
            self.state["exp_avg_sq"] = v
            if amsgrad:
                self.state["max_exp_avg_sq"] = vmax

            return loss

        # ----------------------------
        # Learned / deployment phase: integrate Adam ODE in (a,m,v)
        # ----------------------------
        if self.dynamics is None:
            pred = self.build_sindy_model(sindy_params=self.sindy_params)
            # keep the name "dynamics" even though it predicts gradients
            self.dynamics = lambda y: pred(y)

        d = a.numel()
        m0 = self.state["exp_avg"].detach()
        v0 = self.state["exp_avg_sq"].detach()
        if amsgrad:
            vmax0 = self.state["max_exp_avg_sq"].detach()
        else:
            vmax0 = None

        y0 = self._pack_state(a, m0, v0, vmax0)

        # Build RHS for odeint; note we must use absolute time for bias correction
        logb1 = math.log(beta1)
        logb2 = math.log(beta2)

        def rhs(t: Tensor, y: Tensor) -> Tensor:
            a_t, m_t, v_t, vmax_t = self._unpack_state(y, d, amsgrad)

            # predicted gradient
            g_t = self.dynamics(a_t)

            # moment ODEs
            dm = (1.0 / eta) * (1.0 - beta1) * (g_t - m_t)
            dv = (1.0 / eta) * (1.0 - beta2) * (g_t * g_t - v_t)

            # bias correction factors as continuous-time analog
            # beta^{t/eta} = exp(log(beta) * t/eta)
            b1_pow = torch.exp((t / eta) * logb1)
            b2_pow = torch.exp((t / eta) * logb2)
            denom1 = (1.0 - b1_pow).clamp_min(1e-16)
            denom2 = (1.0 - b2_pow).clamp_min(1e-16)

            mhat = m_t / denom1

            if amsgrad:
                # Practical compromise: treat vmax as constant over the interval for the denominator,
                # then project it at the end of the outer step.
                vhat = vmax_t / denom2
            else:
                vhat = v_t / denom2

            da = -mhat / (vhat.sqrt() + eps)

            if amsgrad:
                # Keep vmax state unchanged during integration; project after solve.
                dvmax = torch.zeros_like(v_t)
                return self._pack_state(da, dm, dv, dvmax)
            else:
                return self._pack_state(da, dm, dv)

        # Integrate over absolute time interval
        t0 = torch.tensor(self.state["t_global"], device=a.device, dtype=a.dtype)
        t1 = torch.tensor(self.state["t_global"] + dt, device=a.device, dtype=a.dtype)
        t_span = torch.stack([t0, t1], dim=0)

        with torch.no_grad():
            y_traj = torchdiffeq_odeint(rhs, y0, t_span, **ode_options)
            y_final = y_traj[-1]

        a_f, m_f, v_f, vmax_f = self._unpack_state(y_final, d, amsgrad)

        # AMSGrad projection at boundary (monotone max) if enabled
        if amsgrad:
            vmax_proj = torch.maximum(self.state["max_exp_avg_sq"], v_f)
            self.state["max_exp_avg_sq"] = vmax_proj
        self.state["exp_avg"] = m_f
        self.state["exp_avg_sq"] = v_f

        # Apply final parameters
        self._set_params_from_flat(a_f)

        # Advance counters
        self.state["epoch"] += 1
        self.state["t_global"] += dt

        if skip_unused_evals:
            return torch.tensor(-1.0, device=a_f.device, dtype=a_f.dtype)
        return loss

    # Your existing build_sindy_model can be reused directly.
    # Two recommended changes for "continuous correctness":
    #   (i) when using strong/weak methods, ensure t_span uses the true sample spacing (dt or eta),
    #   (ii) ensure history/grad_history were stored with clone().detach() (done above).
    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams) -> Callable[[Tensor], Tensor]:
        x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
        dt_sindy = self.defaults['lr']
        t_span = torch.arange(x.shape[0]) * dt_sindy

        if sindy_params.truncation_rank is None:
            d = x.shape[1]
            library = sindy_tools.create_sindy_library(input_dim=d,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias,
                                                       use_ortho=sindy_params.use_ortho)
            Theta = library(x)
            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(x, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(x, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
            elif sindy_params.method == 'tracked':
                lhs_target = torch.stack(self.state['grad_history'], dim=0)
                rhs_mat = Theta
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong', 'weak', or 'tracked'.")

            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)
            pred = sindy_tools.create_predictor(Xi, library)
        else:
            truncation_rank = sindy_params.truncation_rank
            # Build SINDy model in SVD modes.
            # Note, this is different from MATLAB, as it returns Vh or V.T, not V
            U_full_svd, s_full_svd, Vh_full_svd = torch.linalg.svd(x.T, full_matrices=False)
            U_svd = U_full_svd[:, :truncation_rank]
            s_svd = s_full_svd[:truncation_rank]
            Vh_svd = Vh_full_svd[:truncation_rank, :]
            # change training data (x) to be in terms of mode coefficients
            mode_coeffs = torch.diag(s_svd) @ Vh_svd
            # now, we can build the library on the mode coefficients
            library = sindy_tools.create_sindy_library(input_dim=truncation_rank,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias,
                                                       use_ortho=sindy_params.use_ortho)
            Theta = library(mode_coeffs.T)

            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(mode_coeffs.T, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(mode_coeffs.T, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
            elif sindy_params.method == 'tracked':
                grad_hist = torch.stack(self.state['grad_history'], dim=0)
                # lhs_target = (U_svd.T @ grad_hist.T).T
                lhs_target = grad_hist @ U_svd
                rhs_mat = Theta
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong' or 'weak'.")

            # General solution method for solving the linear system.
            # This should take the rhs_mat, lhs_target, and params object, returning the solution Xi
            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)

            mode_pred = sindy_tools.create_predictor(Xi, library)
            # now, we need to create a predictor that takes in the full state and returns the state derivs,
            # not just on mode coeffs

            def pred(y: Tensor) -> Tensor:
                # y is d,1
                # compute the mode coefficients associated with y. We can do this through a matmul
                cur_mode_coeffs = U_svd.T @ y
                # generate predictions on the mode coefficients
                mode_pred_coeffs = mode_pred(cur_mode_coeffs)
                # now, we need to convert the mode predictions back to the full state space
                dydt = U_svd @ mode_pred_coeffs
                return dydt

        # dot(y) = g(y), this is g(y).
        # dot(y) = f(t, y) = g(y), f is dynamics
        return pred

class AppendixAdamContLearnedTrustRegion(AppendixAdamContLearned):
    """
    Trust-region version of AppendixAdamContLearned:
      - Warm-up: discrete Adam with true gradients, collect (a, grad) history
      - Learn: fit SINDy for grad ~= f_sindy(a)
      - Deploy: integrate Adam ODE, but enforce a trust region against the last true step
      - Retrain: only when trust-region checks indicate poor alignment
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        dt: Optional[float] = None,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        history_size: int = 100,
        gamma_tr_radius: Optional[float] = None,
        trust_region_control: Optional[TrustRegionControl] = None,
        ode_solver_options: Optional[dict] = None,
        sindy_params=None,
    ):
        super().__init__(
            params,
            lr=lr,
            dt=dt,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            history_size=history_size,
            retrain_interval=1,
            ode_solver_options=ode_solver_options,
            sindy_params=sindy_params,
        )

        if gamma_tr_radius is None:
            gamma_tr_radius = 2.0 * self.param_groups[0]["dt"]
        if not gamma_tr_radius > 0:
            raise ValueError(f"Trust-region radius gamma must be positive: {gamma_tr_radius}")
        if trust_region_control is None:
            trust_region_control = TrustRegionControl()

        self.state["gamma_tr_radius"] = float(gamma_tr_radius)
        self.param_groups[0]["trust_region_control"] = trust_region_control

        if "retrain_interval" in self.defaults:
            self.defaults.pop("retrain_interval", None)
        if "retrain_interval" in self.param_groups[0]:
            self.param_groups[0].pop("retrain_interval", None)

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        eta: float = group["lr"]
        dt: float = group["dt"]
        beta1, beta2 = group["betas"]
        eps: float = group["eps"]
        amsgrad: bool = group["amsgrad"]
        history_size: int = group["history_size"]
        ode_options: dict = group["ode_solver_options"]
        trc: TrustRegionControl = group["trust_region_control"]

        # Evaluate for performance reporting (required behavior)
        loss = closure()

        # Pull flat params + grads; store safely
        a = self._gather_flat("params").detach()
        g = self._gather_flat("grads").detach()

        # Initialize Adam moment states if needed
        if self.state["exp_avg"] is None:
            self.state["exp_avg"] = torch.zeros_like(a, memory_format=torch.preserve_format)
            self.state["exp_avg_sq"] = torch.zeros_like(a, memory_format=torch.preserve_format)
            if amsgrad:
                self.state["max_exp_avg_sq"] = torch.zeros_like(a, memory_format=torch.preserve_format)

        # ----------------------------
        # Warm-up / data phase
        # ----------------------------
        if len(self.state["history"]) < history_size:
            self.state["history"].append(a.clone().detach())
            self.state["grad_history"].append(g.clone().detach())
            self.state["func_evals"] += 1

            m = self.state["exp_avg"]
            v = self.state["exp_avg_sq"]
            m = m + (1.0 - beta1) * (g - m)
            v = v + (1.0 - beta2) * (g * g - v)

            if amsgrad:
                vmax = self.state["max_exp_avg_sq"]
                vmax = torch.maximum(vmax, v)
            else:
                vmax = None

            self.state["epoch"] += 1
            self.state["t_global"] += dt
            step_t = self.state["epoch"]

            mhat = m / (1.0 - (beta1 ** step_t))
            if amsgrad:
                vhat = vmax / (1.0 - (beta2 ** step_t))
            else:
                vhat = v / (1.0 - (beta2 ** step_t))

            dadt = -mhat / (vhat.sqrt() + eps)
            a_next = a + eta * dadt
            self._set_params_from_flat(a_next)

            self.state["exp_avg"] = m
            self.state["exp_avg_sq"] = v
            if amsgrad:
                self.state["max_exp_avg_sq"] = vmax

            self.state["y_last_closure"] = a_next.clone().detach()
            return loss

        # ----------------------------
        # Learned / deployment phase
        # ----------------------------
        if self.dynamics is None:
            pred = self.build_sindy_model(sindy_params=self.sindy_params)
            self.dynamics = lambda y: pred(y)

        d = a.numel()
        m0 = self.state["exp_avg"].detach()
        v0 = self.state["exp_avg_sq"].detach()
        if amsgrad:
            vmax0 = self.state["max_exp_avg_sq"].detach()
        else:
            vmax0 = None

        y0 = self._pack_state(a, m0, v0, vmax0)

        logb1 = math.log(beta1)
        logb2 = math.log(beta2)

        def rhs(t: Tensor, y: Tensor) -> Tensor:
            a_t, m_t, v_t, vmax_t = self._unpack_state(y, d, amsgrad)

            g_t = self.dynamics(a_t)

            dm = (1.0 / eta) * (1.0 - beta1) * (g_t - m_t)
            dv = (1.0 / eta) * (1.0 - beta2) * (g_t * g_t - v_t)

            b1_pow = torch.exp((t / eta) * logb1)
            b2_pow = torch.exp((t / eta) * logb2)
            denom1 = (1.0 - b1_pow).clamp_min(1e-16)
            denom2 = (1.0 - b2_pow).clamp_min(1e-16)

            mhat = m_t / denom1
            if amsgrad:
                vhat = vmax_t / denom2
            else:
                vhat = v_t / denom2

            da = -mhat / (vhat.sqrt() + eps)

            if amsgrad:
                dvmax = torch.zeros_like(v_t)
                return self._pack_state(da, dm, dv, dvmax)
            return self._pack_state(da, dm, dv)

        t0 = torch.tensor(self.state["t_global"], device=a.device, dtype=a.dtype)
        t1 = torch.tensor(self.state["t_global"] + dt, device=a.device, dtype=a.dtype)
        t_span = torch.stack([t0, t1], dim=0)

        with torch.no_grad():
            y_traj = torchdiffeq_odeint(rhs, y0, t_span, **ode_options)
            y_final = y_traj[-1]

        a_f, m_f, v_f, vmax_f = self._unpack_state(y_final, d, amsgrad)

        if "y_last_closure" not in self.state:
            self.state["y_last_closure"] = a.clone().detach()

        y_diff = a_f - self.state["y_last_closure"]
        y_diff_norm = torch.norm(y_diff, p=2)
        gamma_tr_radius = self.state["gamma_tr_radius"]

        if y_diff_norm <= gamma_tr_radius:
            if amsgrad:
                vmax_proj = torch.maximum(self.state["max_exp_avg_sq"], v_f)
                self.state["max_exp_avg_sq"] = vmax_proj
            self.state["exp_avg"] = m_f
            self.state["exp_avg_sq"] = v_f

            self._set_params_from_flat(a_f)

            self.state["epoch"] += 1
            self.state["t_global"] += dt
            return loss

        # ----------------------------
        # Trust region violated: take true discrete Adam step and compare updates
        # ----------------------------
        self.state["func_evals"] += 1
        y_pred_update = a_f - a

        m = self.state["exp_avg"]
        v = self.state["exp_avg_sq"]
        m = m + (1.0 - beta1) * (g - m)
        v = v + (1.0 - beta2) * (g * g - v)

        if amsgrad:
            vmax = self.state["max_exp_avg_sq"]
            vmax = torch.maximum(vmax, v)
        else:
            vmax = None

        self.state["epoch"] += 1
        self.state["t_global"] += dt
        step_t = self.state["epoch"]

        mhat = m / (1.0 - (beta1 ** step_t))
        if amsgrad:
            vhat = vmax / (1.0 - (beta2 ** step_t))
        else:
            vhat = v / (1.0 - (beta2 ** step_t))

        dadt = -mhat / (vhat.sqrt() + eps)
        a_after = a + eta * dadt
        self._set_params_from_flat(a_after)

        self.state["exp_avg"] = m
        self.state["exp_avg_sq"] = v
        if amsgrad:
            self.state["max_exp_avg_sq"] = vmax

        self.state["y_last_closure"] = a_after.clone().detach()

        y_update = a_after - a

        small_update_tol = trc.grad_tol * dt
        if torch.norm(y_update) < small_update_tol and torch.norm(y_pred_update) < small_update_tol:
            self.state["gamma_tr_radius"] = gamma_tr_radius * 1.25
        else:
            cos_sim = torch.cosine_similarity(y_update, y_pred_update, dim=0)
            if cos_sim < trc.cosine_similarity_bad_threshold:
                self.state["gamma_tr_radius"] = gamma_tr_radius * trc.radius_factor_bad
                self.state["history"] = [a_after.clone().detach()]
                self.state["grad_history"] = [g.clone().detach()]
                self.dynamics = None
            elif cos_sim < trc.cosine_similarity_good_threshold:
                self.state["gamma_tr_radius"] = gamma_tr_radius * trc.radius_factor_okay
            else:
                self.state["gamma_tr_radius"] = gamma_tr_radius * trc.radius_factor_good

        self.state["gamma_tr_radius"] = min(self.state["gamma_tr_radius"], trc.max_radius)
        return loss



