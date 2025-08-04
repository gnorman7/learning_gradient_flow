# mypy: allow-untyped-defs
import warnings
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, asdict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT  # Use PyTorch's type hint
import learning_gradient_flow.sindy_tools as sindy_tools

# import pysindy as ps
import numpy as np

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
    warnings.warn(
        "torchdiffeq library not found. GradientFlow optimizers using odeint will not work. "
        "Install with: pip install torchdiffeq"
    )

__all__ = ["VectorBasedOptimizer", "GradientFlow", "SINDyFlow", "SINDyFlowTrustRegion", "TrustRegionControl", "BatchedSINDyFlow"]


class VectorBasedOptimizer(Optimizer):
    """VectorBasedOptimizer, tools for reshaping and flattening parameters,
        assumes closure function, and track function evaluations.

        params: Iterable of parameters to optimize
    """

    def __init__(self, params: ParamsT, defaults: Optional[Dict[str, Any]] = {}) -> None:
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "Vector based optimizers do not support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        # Ensure all parameters are on the same device and real
        if len(self._params) > 0:
            self._device = self._params[0].device
            for p in self._params[1:]:
                if p.device != self._device:
                    raise ValueError("All parameters must be on the same device.")
                if torch.is_complex(p):
                    raise ValueError("Vector based optimizers do not support complex parameters.")
        else:
            self._device = None

        self._numel_cache = None
        self.state['func_evals'] = 0

    def _numel(self) -> int:
        """Computes the total number of elements for all real parameters."""
        if self._numel_cache is None:
            if not self._params:
                self._numel_cache = 0
            else:
                self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat(self, params_or_grads: str = "params") -> Tensor:
        """Gathers real parameters or gradients into a single flat tensor."""
        views = []
        source_list = self._params if params_or_grads == "params" else (p.grad for p in self._params)

        for p, source in zip(self._params, source_list):
            if params_or_grads == "grads":
                if source is None:
                    # Use zeros_like on the parameter data to get correct shape/device
                    view = torch.zeros_like(p.data).flatten()
                elif source.is_sparse:
                    view = source.to_dense().flatten()
                else:
                    # Use detach() on gradients to prevent graph issues if grads were reused?
                    # Or rely on zero_grad before backward. Let's assume fresh grads.
                    view = source.flatten()  # No detach needed if zero_grad is used before backward
            elif params_or_grads == "params":
                # Detach params when gathering to avoid feeding back into grad computation within ODE step
                view = p.data.detach().flatten()
            else:
                raise ValueError("params_or_grads must be 'params' or 'grads'")
            views.append(view)

        if not views:
            return torch.empty(0, device=self._device)
        return torch.cat(views, 0)

    def _set_params_from_flat(self, flat_params: Tensor):
        """Sets model's real parameters from a flat tensor."""
        offset = 0
        if flat_params.numel() != self._numel():
            raise ValueError(
                f"Size mismatch: flat tensor has {flat_params.numel()} elements, "
                f"but parameters have {self._numel()} elements ({self._numel_cache})."
            )

        for p in self._params:
            numel = p.numel()
            chunk = flat_params[offset: offset + numel].contiguous()
            # Use data to assign, avoiding autograd tracking here
            p.data.copy_(chunk.view_as(p.data))
            offset += numel

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        raise NotImplementedError


class GradientFlow(VectorBasedOptimizer):
    r"""
    Implements standard Gradient Flow using ODE integration via
    `torchdiffeq.odeint <https://github.com/rtqichen/torchdiffeq?tab=readme-ov-file#keyword-arguments-for-odeint_adjoint>`_.

    .. math::
        \frac{d(params)}{dt} = - \nabla L(params)
    """

    def __init__(
        self,
        params: ParamsT,
        dt: float = 1e-3,
        ode_solver_options: Dict[str, Any] = {},
    ):
        """
        Args:
            params: Iterable of parameters to optimize (must be real tensors).
            dt: Time interval for integration per step. (default: 1e-3)
            ode_solver_options: Options passed to `torchdiffeq.odeint`. (default: {})
        """

        if torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq library is required for GradientFlowBase. "
                "Install with: pip install torchdiffeq"
            )
        if not dt > 0:
            raise ValueError(f"Integration time dt must be positive: {dt}")

        super().__init__(
            params,
            defaults={"dt": dt, "ode_solver_options": ode_solver_options}
        )

    def _get_dynamics_vector(self, t: float, flat_params: Tensor) -> Tensor:
        """
        Standard gradient flow dynamics: d(params)/dt = -grad(L(params)).

        Args:
            t: Current time
            flat_params: Current flat parameters

        Returns:
            The negative flat gradient.
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
            return -flat_grad


    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        """
        Performs a single optimization step by integrating the dynamics ODE.

        Args:
            closure: Callable that reevaluates the model, runs backward, and returns loss.
                     This will be called multiple times by the ODE solver.

        Returns:
            The loss tensor computed *after* the integration step, or None if
            no parameters are present.
        """
        closure = torch.enable_grad()(closure)

        if not self._params:
            warnings.warn("Attempting to step GradientFlow optimizer with no parameters.")
            return None

        group = self.param_groups[0]
        dt = group["dt"]
        ode_options = group["ode_solver_options"]

        # Initial state (parameters) at t=0 for this step
        # Ensure y0 is detached from previous computation graphs
        y0 = self._gather_flat("params")  # Already detached inside gather_flat

        # --- Define the dynamics function for the ODE solver ---
        # This function must have the signature func(t, y)
        self.evals_in_step = [0]  # Use list to pass by reference

        # Store self reference for use inside ode_func
        self.closure = closure
        orig_loss = closure()
        self.state['func_evals'] += 1  # Count this initial evaluation?

        # Time span for integration for this step
        t_span = torch.tensor([0.0, dt], device=self._device)

        # --- Perform the ODE integration ---
        # odeint returns results at points specified in t_span
        y_result = torchdiffeq_odeint(
            self._get_dynamics_vector,
            y0,
            t_span,
            **ode_options
        )

        # Update evaluation counter
        self.state['func_evals'] += self.evals_in_step[0]

        # The result at the end of the interval dt
        y_final = y_result[-1]

        # Update the model parameters to the final state
        self._set_params_from_flat(y_final)

        return orig_loss


class SINDyFlow(VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        backup_optimizer: Optimizer,
        dt: float = 1e-3,
        ode_solver_options: Dict[str, Any] = {},
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_params: sindy_tools.SINDyParams = None
    ):
        if torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq library is required for GradientFlowBase. "
                "Install with: pip install torchdiffeq"
            )
        if not dt > 0:
            raise ValueError(f"Integration time dt must be positive: {dt}")
        if backup_optimizer.__class__.__name__ == "LBFGS":
            warnings.warn("Func Evals will likely be incorrect for LBFGS optimizer.")
        if retrain_interval < history_size:
            raise ValueError(f"Retrain interval must be greater than or equal to history size: {retrain_interval}!")
        if sindy_params is None:
            sindy_params = sindy_tools.SINDyParams()
        super().__init__(
            params,
            defaults={"dt": dt,
                      "ode_solver_options":ode_solver_options,
                      "history_size": history_size,
                      "retrain_interval": retrain_interval}
        )
        self.state['history'] = [] # tracks the parameter history
        self.state['history_count'] = 0
        self.state['epoch'] = 0
        self.backup_optimizer = backup_optimizer
        self.sindy_params = sindy_params

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        history_size = group["history_size"]
        dt = group["dt"]
        ode_options = group["ode_solver_options"]
        retrain_interval = group["retrain_interval"]

        # Check if we need to retrain the SINDy model
        # Crieria is just based on epoch number and retrain interval, but could be more complex
        if self.state['epoch'] % retrain_interval == 0:
            self.state['history'] = []
            self.state['history_count'] = 0
            self.dynamics = None

        if self.state['history_count'] < history_size:
            # Use backup optimizer for the first few steps
            loss = self.backup_optimizer.step(closure)
            self.state['func_evals'] += 1
            self.state['history'].append(self._gather_flat("params"))
            self.state['history_count'] += 1
            self.state['epoch'] += 1
            return loss
        else:
            orig_loss = closure()
            # self.state['func_evals'] += 1

            # build sindy model if it doesn't already exist
            if self.dynamics is None:
                pred = self.build_sindy_model(sindy_params=self.sindy_params)
                self.dynamics = lambda t, y: pred(y)

            # This will be (d,1) integration!
            y0 = self._gather_flat("params")  # Already detached inside gather_flat
            t_span = torch.tensor([0.0, dt], device=self._device)

            y_result = torchdiffeq_odeint(
                self.dynamics,
                y0.unsqueeze(1),
                t_span,
                **ode_options
            ).squeeze(-1)

            # Update evaluation counter
            self.state['func_evals'] += 0 # No extra evaluations for SINDy model

            # The result at the end of the interval dt
            y_final = y_result[-1]

            # Update the model parameters to the final state
            self._set_params_from_flat(y_final)

            self.state['epoch'] += 1
            return orig_loss

    def get_sindy_mats(self, sindy_params: sindy_tools.SINDyParams) -> tuple[Tensor, Tensor]:
        """Returns lhs_target and rhs_mat"""
        x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
        d = x.shape[1]
        library = sindy_tools.create_sindy_library(input_dim=d,
                                                   poly_order=sindy_params.poly_order,
                                                   include_bias=sindy_params.include_bias)
        Theta = library(x)
        dt_sindy = self.backup_optimizer.param_groups[0]['lr']
        t_span = torch.arange(x.shape[0]) * dt_sindy
        if sindy_params.method == 'strong':
            lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(x, Theta, t_span)
        elif sindy_params.method == 'weak':
            lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(x, Theta, t_span,
                                                                        test_func_params=sindy_params.test_func_params)
        else:
            raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong' or 'weak'.")
        return lhs_target, rhs_mat, library

    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams) -> Callable[[Tensor], Tensor]:
        """Uses self.state['history'] to build a SINDy model.
        Args:
            poly_order: Polynomial order for the library.
            include_bias: Whether to include a constant term.
            truncation_rank: Rank for SVD truncation, None == no SVD.
            method: 'strong' or 'weak' for system construction.
            test_func_params: For 'weak', param object to be passed to sindy_tools.assemble_weak_matrices
            solver_fn: Function to solve the linear system, takes rhs_mat, lhs_target, solver_params.
            solver_kwargs: kwargs for the solver function.
        Returns:
            A function that takes in a state and returns the state derivative.
        """


        # first called when 'func_evals' == history_size
        # build the SINDy model using the history of parameters
        # each entry of self.state['history'] is a tensor of shape (num_params,)

        if sindy_params.truncation_rank is None:
            rhs_mat, lhs_target, library = self.get_sindy_mats(sindy_params)

            # General solution method for solving the linear system.
            # This should take the rhs_mat, lhs_target, and params object, returning the solution Xi
            Xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)

            pred = sindy_tools.create_predictor(Xi, library)
        else:
            x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
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

@dataclass
class TrustRegionControl:
    grad_tol: float = 1e-8 # Tolerance for small updates, this won't use cosine_similarity
    cosine_similarity_good_threshold: float = 0.7  # Cosine similarity above this = good
    cosine_similarity_bad_threshold: float = 0.2  # Below this = very bad, retrain
    radius_factor_good: float=1.5
    radius_factor_okay: float=0.8
    radius_factor_bad: float=0.5
    max_radius: float = 0.5  # Maximum radius for the trust region

    def __str__(self) -> str:
        return "\n".join(f"{key}: {value}" for key, value in asdict(self).items())


class SINDyFlowTrustRegion(VectorBasedOptimizer):
    def __init__(
        self,
        params: ParamsT,
        backup_optimizer: Optimizer,
        dt: float = 1e-3,
        ode_solver_options: Dict[str, Any] = {},
        history_size: int = 100,
        sindy_kwargs: Dict[str, Any] = {},
        gamma_tr_radius: Optional[float] = None,
        trust_region_control: Optional[TrustRegionControl] = None,
        # comparison_frequency: int = 50, # we should use tr_radius instead to tell us when we compare!
    ):
        if torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq library is required for GradientFlowBase. "
                "Install with: pip install torchdiffeq"
            )
        if not dt > 0:
            raise ValueError(f"Integration time dt must be positive: {dt}")
        if backup_optimizer.__class__.__name__ == "LBFGS":
            warnings.warn("Func Evals will likely be incorrect for LBFGS optimizer.")
        if gamma_tr_radius is None:
            gamma_tr_radius = 2.0*dt
        if not gamma_tr_radius > 0:
            raise ValueError(f"Trust-region radius gamma must be positive: {gamma_tr_radius}")
        if trust_region_control is None:
            trust_region_control = TrustRegionControl()
        # if ode_solver_options {}
        if ode_solver_options == {}:
            ode_solver_options = {'rtol': 1e-5, 'atol': 1e-6}
        super().__init__(
            params,
            defaults={"dt": dt,
                      "ode_solver_options": ode_solver_options,
                      "history_size": history_size,
                      "sindy_kwargs": sindy_kwargs,
                      "trust_region_control": trust_region_control,
                      }
        )
        self.state['history'] = []  # tracks the parameter history
        self.state['history_count'] = 0
        self.state['epoch'] = 0
        self.state['gamma_tr_radius'] = gamma_tr_radius
        self.backup_optimizer = backup_optimizer
        self.sindy_kwargs = sindy_kwargs
        self.dynamics = None

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        history_size = group["history_size"]
        dt = group["dt"]
        ode_options = group["ode_solver_options"]

        if self.state['history_count'] < history_size:
            # Use backup optimizer for the first few steps
            loss = self.backup_optimizer.step(closure)
            self.state['func_evals'] += 1
            self.state['history'].append(self._gather_flat("params"))
            self.state['history_count'] += 1
            self.state['epoch'] += 1
            return loss
        else:
            # This is where we will measure the trust region from.
            # Maybe could use the last point in the history instead?
            if 'y_last_closure' not in self.state:
                self.state['y_last_closure'] = self._gather_flat("params")

            orig_loss = closure()
            # Don't count these, as we don't really use them, but we ARE evaluating it...
            # self.state['func_evals'] += 1

            # build sindy model if it doesn't already exist
            if self.dynamics is None:
                pred = self.build_sindy_model(**self.sindy_kwargs)
                self.dynamics = lambda t, y: pred(y)

            # This will be (d,1) integration!
            y0 = self._gather_flat("params")  # Already detached inside gather_flat
            t_span = torch.tensor([0.0, dt], device=self._device)

            y_result = torchdiffeq_odeint(
                self.dynamics,
                y0.unsqueeze(1),
                t_span,
                **ode_options
            ).squeeze(-1)

            # Update evaluation counter
            self.state['func_evals'] += 0  # No extra evaluations for SINDy model

            # The result at the end of the interval dt
            y_final = y_result[-1]

            # Check if the trust region is violated
            y_diff = y_final - self.state['y_last_closure']
            y_diff_norm = torch.norm(y_diff, p=2)
            gamma_tr_radius = self.state['gamma_tr_radius']
            if y_diff_norm <= gamma_tr_radius:
                # nothing special, carry with this iteration
                # Update the model parameters to the final state
                self._set_params_from_flat(y_final)
            else:
                # If the trust region is violated, we will evaluate the closure and see if we are still somewhat aligned
                # with the gradient. If not, we will use the backup optimizer to step instead (redo the history collection).
                # If successful, we can loosen the radius a bit.
                # We will base this on gradient alignment via cosine similarity.
                # Three options based on how well we do
                # 1. if very bad, redo training, and reduce radius a bit
                # 2. if not too bad, but reduce the radius a little less (no retraining)
                # 3. if good, increase the radius a bit (no retraining)
                # we will need to have special treatment for if the gradient is basically zero (good)
                trc: TrustRegionControl = group["trust_region_control"]

                # First, let's get the true next step at the current point through evaluating closure and collecting
                # we can use y0 from above for the "before", set those as the parameters, then take the step and record
                self._set_params_from_flat(y0)
                self.backup_optimizer.step(closure)
                self.state['func_evals'] += 1
                y_after = self._gather_flat("params")
                self.state['y_last_closure'] = y_after
                y_update = y_after - y0
                y_pred_update = y_final - y0

                # if both the grad and it's prediction are small (relative to the learning ratet / dt), that's "good"
                if torch.norm(y_update) < trc.grad_tol*dt and torch.norm(y_pred_update) < trc.grad_tol*dt:
                    # increase trust region radius a bit
                    self.state['gamma_tr_radius'] = gamma_tr_radius * 1.25
                else:
                    # use cosine similarity then
                    cos_sim = torch.cosine_similarity(y_update, y_pred_update, dim=0)
                    if cos_sim < trc.cosine_similarity_bad_threshold:
                        # very bad, redo training, and reduce radius a bit
                        self.state['gamma_tr_radius'] = gamma_tr_radius * trc.radius_factor_bad
                        self.state['history'] = [y_after]
                        self.state['history_count'] = 1
                        self.dynamics = None
                    elif cos_sim < trc.cosine_similarity_good_threshold:
                        # not too bad, but reduce the radius a little less (no retraining)
                        self.state['gamma_tr_radius'] = gamma_tr_radius * trc.radius_factor_okay
                    else:
                        # good, increase the radius a bit (no retraining)
                        self.state['gamma_tr_radius'] = gamma_tr_radius * trc.radius_factor_good
                self.state['gamma_tr_radius'] = min(self.state['gamma_tr_radius'], trc.max_radius)
            self.state['epoch'] += 1
            return orig_loss

    def build_sindy_model(self, poly_order: int = 1,
                          include_bias: bool = True,
                          rcond: Optional[float] = 1e-7,
                          truncation_rank: Optional[int] = None,
                          method: str = 'weak',
                          test_mat_kwargs: Dict[str, Any] = {},
                          ) -> Callable[[Tensor], Tensor]:
        # first called when 'func_evals' == history_size
        # build the SINDy model using the history of parameters
        # each entry of self.state['history'] is a tensor of shape (num_params,)
        x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
        if truncation_rank is None:
            d = x.shape[1]
            library = sindy_tools.create_sindy_library(input_dim=d,
                                                       poly_order=poly_order,
                                                       include_bias=include_bias)
            Theta = library(x)
            dt_sindy = self.backup_optimizer.param_groups[0]['lr']
            t_span = torch.arange(x.shape[0]) * dt_sindy
            if method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(x, Theta, t_span)
            elif method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(x, Theta, t_span,
                                                                         test_mat_kwargs=test_mat_kwargs)

            Xi = torch.linalg.lstsq(rhs_mat, lhs_target, rcond=rcond).solution

            pred = sindy_tools.create_predictor(Xi, library)
        else:
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
                                                    poly_order=poly_order,
                                                    include_bias=include_bias)
            Theta = library(mode_coeffs.T)
            dt_sindy = self.backup_optimizer.param_groups[0]['lr']
            t_span = torch.arange(x.shape[0]) * dt_sindy

            if method == 'strong':
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(mode_coeffs.T, Theta, t_span)
            elif method == 'weak':
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(mode_coeffs.T, Theta, t_span,
                                                                         test_mat_kwargs=test_mat_kwargs)

            Xi = torch.linalg.lstsq(rhs_mat, lhs_target, rcond=rcond).solution

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


class BatchedSINDyFlow:
    """
    Batch multiple optimizers with SindyFlow, but use a batched appraoch for the model building.
    """

    def __init__(
        self,
        # params_list: list[ParamsT],
        backup_optimizers: list[Optimizer],
        dt: float = 1e-3,
        ode_solver_options: Dict[str, Any] = {},
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_params: sindy_tools.SINDyParams = None,
        retain_all_states: bool = False,
    ):
        """
        Args:
            params: Iterable of parameters to optimize
            backup_optimizers: List of optimizers to use during history collection phase
            dt: Time interval for integration per step
            ode_solver_options: Options passed to torchdiffeq.odeint
            history_size: Number of steps to collect before building SINDy model
            retrain_interval: Number of steps between retraining the SINDy model
            sindy_params: Configuration for SINDy library and solver
            retain_all_states: Whether to use all observations (ever) for SINDy model, e.g. not just the recent ones.
        """
        if torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq library is required for BatchedSINDyFlow. "
                "Install with: pip install torchdiffeq"
            )

        if sindy_params is None:
            sindy_params = sindy_tools.SINDyParams()

        sindy_flow_optimizers = []
        for i, backup_optimizer in enumerate(backup_optimizers):
            sindy_flow_optimizer = SINDyFlow(backup_optimizer.param_groups[0]['params'],
                                             backup_optimizer=backup_optimizer,
                                             dt=dt,
                                             ode_solver_options=ode_solver_options,
                                             history_size=history_size*100,
                                             retrain_interval=retrain_interval*100,
                                             sindy_params=sindy_params)
            sindy_flow_optimizers.append(sindy_flow_optimizer)
        self.sindy_flow_optimizers = sindy_flow_optimizers

        self.state = {
            'history_count': 0,
            'epoch': 0,
            'func_evals': 0,
        }

        self.retrain_interval = retrain_interval
        self.history_size = history_size
        self.dynamics = None
        self.sindy_params = sindy_params
        self.dt = dt
        self.retain_all_states = retain_all_states
        self.lhs_targets = []
        self.rhs_mats = []


    def step(self, closures: list[Callable[[], Tensor]]) -> Optional[list[Tensor]]:
        if self.state['epoch'] % self.retrain_interval == 0:
            self.state['history_count'] = 0
            self.dynamics = None

        if self.state['history_count'] < self.history_size:
            losses = []
            func_eval_count = 0
            for closure, sindy_flow_optimizer in zip(closures, self.sindy_flow_optimizers):
                loss = sindy_flow_optimizer.step(closure)
                losses.append(loss)
                func_eval_count += sindy_flow_optimizer.state['func_evals']
            self.state['history_count'] += 1
            self.state['epoch'] += 1
            self.state['func_evals'] = func_eval_count
            return losses

        # In this case, we have enough history for sindy model!

        # We evaluate this loss, but don't do anything with it.
        orig_losses = []
        for closure, sindy_flow_optimizer in zip(closures, self.sindy_flow_optimizers):
            orig_loss = closure()
            orig_losses.append(orig_loss)
            # self.state['func_evals'] += 1 # don't count these, we don't use them.

        if self.dynamics is None:
            if not self.retain_all_states:
                # reset the matrices if we're not retaining all states
                self.lhs_targets = []
                self.rhs_mats = []
            for sindy_flow_optimizer in self.sindy_flow_optimizers:
                lhs_target, rhs_mat, library = sindy_flow_optimizer.get_sindy_mats(sindy_flow_optimizer.sindy_params)
                self.lhs_targets.append(lhs_target)
                self.rhs_mats.append(rhs_mat)

            big_rhs_mat = torch.cat(self.rhs_mats, dim=0)
            big_lhs_target = torch.cat(self.lhs_targets, dim=0)
            # print the sizes
            print(f"big_rhs_mat shape: {big_rhs_mat.shape}, big_lhs_target shape: {big_lhs_target.shape}")

            Xi = self.sindy_params.solver_fn(big_rhs_mat, big_lhs_target, self.sindy_params.solver_params)

            print(f'Xi: {Xi}')

            pred = sindy_tools.create_predictor(Xi, library)

            self.dynamics = lambda t, y: pred(y)

        for sindy_flow_optimizer in self.sindy_flow_optimizers:
            t_span = torch.tensor([0.0, sindy_flow_optimizer.param_groups[0]['dt']])
            y0 = sindy_flow_optimizer._gather_flat("params")
            y_result = torchdiffeq_odeint(
                self.dynamics,
                y0.unsqueeze(1),
                t_span,
                **sindy_flow_optimizer.param_groups[0]["ode_solver_options"]
            ).squeeze(-1)
            y_final = y_result[-1]
            sindy_flow_optimizer._set_params_from_flat(y_final)

        self.state['epoch'] += 1

        return orig_losses




















    # def build_sindy_model(self, poly_order: int = 1, include_bias: bool = True, rcond: Optional[float] = 1e-5):
    #     dt_sindy = self.backup_optimizer.param_groups[0]['lr']
    #     if poly_order >= 1:
    #         library_functions = [lambda x: x]
    #     if poly_order >= 2:
    #         library_functions.append(lambda x: x**2)
    #         library_functions.append(lambda x,y: x*y)
    #     if poly_order >= 3:
    #         library_functions.append(lambda x: x**3)
    #         library_functions.append(lambda x,y: x**2*y)
    #         library_functions.append(lambda x,y: x*y**2)
    #         library_functions.append(lambda x,y,z: x*y*z**2)
    #         library_functions.append(lambda x,y,z: x**2*y*z)
    #         library_functions.append(lambda x,y,z: x*y**2*z)
    #         library_functions.append(lambda x,y,z: x*y*z)


    #     t_train = np.arange(self.state['history_count']) * dt_sindy
    #     ode_lib = ps.WeakPDELibrary(
    #         library_functions=library_functions,
    #         include_bias=include_bias,
    #         spatiotemporal_grid=t_train,
    #         is_uniform=True
    #     )
    #     sindy_optimizer = ps.STLSQ(threshold=0.0, alpha=0.0)
    #     model = ps.SINDy(feature_library=ode_lib, optimizer=sindy_optimizer)
    #     # data to numpy
    #     x = torch.stack(self.state['history'], dim=0)  # history_size, num_params
    #     x = x.detach().cpu().numpy()
    #     model.fit(x)

    #     # create a method of predicting that we can return. This should be (d,1) tensor to (d,1) tensor
    #     def pred_fun(x: torch.Tensor) -> torch.Tensor:
    #         # x is d,1
    #         # library(x.T) is 1,P
    #         x = x.detach().cpu().numpy()
    #         dxdt_row = model.predict(x)
    #         return torch.from_numpy(dxdt_row).to(self._device).T

    #     return pred_fun

