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
import learning_gradient_flow.sindy_tools

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


class AdamEulerLearned(learning_gradient_flow.gradient_flow_optimizer.VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: Union[float, Tensor] = 1e-8,
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_params: learning_gradient_flow.sindy_tools.SINDyParams = None
    ):
        if not lr > 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if retrain_interval < history_size:
            raise ValueError(f"Retrain interval must be greater than or equal to history size: {retrain_interval}!")
        if sindy_params is None:
            sindy_params = learning_gradient_flow.sindy_tools.SINDyParams()

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        history_size=history_size, retrain_interval=retrain_interval)
        super().__init__(params, defaults)
        self.state['epoch'] = 0
        self.state['history'] = []
        self.state['grad_history'] = []
        self.state['history_count'] = 0
        flat_params = self._gather_flat('params')
        self.expanded_state = torch.zeros(len(flat_params), 3, dtype=flat_params.dtype, device=flat_params.device)
        self.expanded_state[:, 0] = flat_params
        self.loss = None
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
        retrain_interval: int = group['retrain_interval']
        self.evals_in_step = [0]  # Use list to pass by reference

        dt = lr
        beta1, beta2 = self.defaults['betas']

        if self.state['epoch'] == 0:
            flat_params = self._gather_flat('params')
            flat_grad = self._get_grad(flat_params)

            self.expanded_state[:, 1] = (1 - beta1) * (flat_grad)
            self.expanded_state[:, 2] = (1 - beta2) * (flat_grad ** 2)
            self.state['epoch'] += 1


        if self.state['epoch'] % retrain_interval == 1:
            self.state['history'] = []
            self.state['grad_history'] = []
            self.state['history_count'] = 0
            self.dynamics = None

        # -------------------------------------------------------
        # Begin the evaluation of the time derivative function
        flat_params = self._gather_flat('params')

        if self.state['history_count'] < history_size:
            flat_grad = self._get_grad(flat_params)
            self.state['history'].append(flat_params)
            self.state['grad_history'].append(flat_grad)
            self.state['history_count'] += 1
        else:
            # build grad model if it doesn't already exist
            if self.dynamics is None:
                pred = self.build_sindy_model(sindy_params=self.sindy_params)
                self.dynamics = lambda y: pred(y) # Don't need t dependence, as custom ODE solve

            # evaluate the model to get the grad
            flat_grad = self.dynamics(flat_params)
            # Compute loss, but it isn't used (or counted)
            self.loss = self.closure()

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
        flat_params += d_params_dt * dt

        if same_time:
            momentum = self.expanded_state[:, 1] + dt * d_momentum_dt
            variance = self.expanded_state[:, 2] + dt * d_variance_dt

        # End the evaluation of the time derivative function
        # -------------------------------------------------------

        self.state['func_evals'] += self.evals_in_step[0]

        self.expanded_state = torch.stack(
            [flat_params, momentum, variance], dim=1
        )
        # Update the parameters in the optimizer
        self._set_params_from_flat(flat_params)

        return self.loss

    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams) -> Callable[[Tensor], Tensor]:
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
        if sindy_params.truncation_rank is None:
            d = x.shape[1]
            library = sindy_tools.create_sindy_library(input_dim=d,
                                                       poly_order=sindy_params.poly_order,
                                                       include_bias=sindy_params.include_bias)
            Theta = library(x)
            dt_sindy = self.param_groups[0]['lr']
            t_span = torch.arange(x.shape[0]) * dt_sindy
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
                                                       include_bias=sindy_params.include_bias)
            Theta = library(mode_coeffs.T)
            dt_sindy = self.backup_optimizer.param_groups[0]['lr']
            t_span = torch.arange(x.shape[0]) * dt_sindy

            if sindy_params.method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(mode_coeffs.T, Theta, t_span)
            elif sindy_params.method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(mode_coeffs.T, Theta, t_span,
                                                                         test_func_params=sindy_params.test_func_params)
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
