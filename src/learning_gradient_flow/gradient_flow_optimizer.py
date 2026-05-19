# mypy: allow-untyped-defs
import warnings
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

import learning_gradient_flow.sindy_tools as sindy_tools

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
    warnings.warn(
        "torchdiffeq library not found. LGFGradientFlow will not work. "
        "Install with: pip install torchdiffeq"
    )


class VectorBasedOptimizer(Optimizer):
    """Optimizer base class for flattening and restoring a single parameter group."""

    def __init__(self, params: ParamsT, defaults: Optional[Dict[str, Any]] = None) -> None:
        if defaults is None:
            defaults = {}
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "Vector based optimizers do not support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
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
        self.state["func_evals"] = 0

    def _numel(self) -> int:
        if self._numel_cache is None:
            if not self._params:
                self._numel_cache = 0
            else:
                self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat(self, params_or_grads: str = "params") -> Tensor:
        views = []
        source_list = self._params if params_or_grads == "params" else (p.grad for p in self._params)

        for p, source in zip(self._params, source_list):
            if params_or_grads == "grads":
                if source is None:
                    view = torch.zeros_like(p.data).flatten()
                elif source.is_sparse:
                    view = source.to_dense().flatten()
                else:
                    view = source.flatten()
            elif params_or_grads == "params":
                view = p.data.detach().flatten()
            else:
                raise ValueError("params_or_grads must be 'params' or 'grads'")
            views.append(view)

        if not views:
            return torch.empty(0, device=self._device)
        return torch.cat(views, 0)

    def _set_params_from_flat(self, flat_params: Tensor):
        offset = 0
        if flat_params.numel() != self._numel():
            raise ValueError(
                f"Size mismatch: flat tensor has {flat_params.numel()} elements, "
                f"but parameters have {self._numel()} elements ({self._numel_cache})."
            )

        for p in self._params:
            numel = p.numel()
            chunk = flat_params[offset: offset + numel].contiguous()
            p.data.copy_(chunk.view_as(p.data))
            offset += numel

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        raise NotImplementedError


class LGFGradientFlow(VectorBasedOptimizer):
    """Learned Gradient Flow optimizer using a SINDy surrogate for the parameter dynamics."""

    def __init__(
        self,
        params: ParamsT,
        backup_optimizer: Optimizer,
        dt: float = 1e-3,
        ode_solver_options: Optional[Dict[str, Any]] = None,
        history_size: int = 100,
        retrain_interval: int = 200,
        sindy_params: Optional[sindy_tools.SINDyParams] = None,
        skip_unused_evals: bool = False,
    ):
        if torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq library is required for LGFGradientFlow. "
                "Install with: pip install torchdiffeq"
            )
        if not dt > 0:
            raise ValueError(f"Integration time dt must be positive: {dt}")
        if backup_optimizer.__class__.__name__ == "LBFGS":
            warnings.warn("Func Evals will likely be incorrect for LBFGS optimizer.")
        if retrain_interval < history_size:
            raise ValueError(
                f"Retrain interval must be greater than or equal to history size: {retrain_interval}!"
            )
        if ode_solver_options is None:
            ode_solver_options = {}
        if sindy_params is None:
            sindy_params = sindy_tools.SINDyParams()

        super().__init__(
            params,
            defaults={
                "dt": dt,
                "ode_solver_options": ode_solver_options,
                "history_size": history_size,
                "retrain_interval": retrain_interval,
            },
        )
        self.state["history"] = []
        self.state["history_count"] = 0
        self.state["epoch"] = 0
        self.state["skip_unused_evals"] = skip_unused_evals
        self.backup_optimizer = backup_optimizer
        self.sindy_params = sindy_params
        self.dynamics = None

    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        history_size = group["history_size"]
        dt = group["dt"]
        ode_options = group["ode_solver_options"]
        retrain_interval = group["retrain_interval"]

        if self.state["epoch"] % retrain_interval == 0:
            self.state["history"] = []
            self.state["history_count"] = 0
            self.dynamics = None

        if self.state["history_count"] < history_size:
            loss = self.backup_optimizer.step(closure)
            self.state["func_evals"] += 1
            self.state["history"].append(self._gather_flat("params"))
            self.state["history_count"] += 1
            self.state["epoch"] += 1
            return loss

        skip_unused_evals = self.state["skip_unused_evals"]
        if not skip_unused_evals:
            orig_loss = closure()

        if self.dynamics is None:
            pred = self.build_sindy_model(sindy_params=self.sindy_params)
            self.dynamics = lambda t, y: pred(y)

        y0 = self._gather_flat("params")
        t_span = torch.tensor([0.0, dt], device=self._device)

        y_result = torchdiffeq_odeint(
            self.dynamics,
            y0.unsqueeze(1),
            t_span,
            **ode_options,
        ).squeeze(-1)

        y_final = y_result[-1]
        self._set_params_from_flat(y_final)

        self.state["epoch"] += 1
        if skip_unused_evals:
            return torch.tensor(-1.0, device=self._device)
        return orig_loss

    def get_sindy_mats(
        self, sindy_params: sindy_tools.SINDyParams
    ) -> tuple[Tensor, Tensor, torch.nn.Module]:
        x = torch.stack(self.state["history"], dim=0)
        d = x.shape[1]
        library = sindy_tools.create_sindy_library(
            input_dim=d,
            poly_order=sindy_params.poly_order,
            include_bias=sindy_params.include_bias,
            use_ortho=sindy_params.use_ortho,
        )
        theta = library(x)
        dt_sindy = self.backup_optimizer.param_groups[0]["lr"]
        t_span = torch.arange(x.shape[0]) * dt_sindy

        if sindy_params.method == "strong":
            lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(
                x, theta, t_span, fd_order=sindy_params.fd_order
            )
        elif sindy_params.method == "weak":
            lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(
                x, theta, t_span, test_func_params=sindy_params.test_func_params
            )
        else:
            raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong' or 'weak'.")

        return lhs_target, rhs_mat, library

    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams) -> Callable[[Tensor], Tensor]:
        if sindy_params.truncation_rank is None:
            lhs_target, rhs_mat, library = self.get_sindy_mats(sindy_params)
            xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)
            pred = sindy_tools.create_predictor(xi, library)
        else:
            x = torch.stack(self.state["history"], dim=0)
            truncation_rank = sindy_params.truncation_rank

            u_full_svd, s_full_svd, vh_full_svd = torch.linalg.svd(x.T, full_matrices=False)
            u_svd = u_full_svd[:, :truncation_rank]
            s_svd = s_full_svd[:truncation_rank]
            vh_svd = vh_full_svd[:truncation_rank, :]
            mode_coeffs = torch.diag(s_svd) @ vh_svd

            library = sindy_tools.create_sindy_library(
                input_dim=truncation_rank,
                poly_order=sindy_params.poly_order,
                include_bias=sindy_params.include_bias,
                use_ortho=sindy_params.use_ortho,
            )
            theta = library(mode_coeffs.T)
            dt_sindy = self.backup_optimizer.param_groups[0]["lr"]
            t_span = torch.arange(x.shape[0]) * dt_sindy

            if sindy_params.method == "strong":
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(
                    mode_coeffs.T, theta, t_span, fd_order=sindy_params.fd_order
                )
            elif sindy_params.method == "weak":
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(
                    mode_coeffs.T,
                    theta,
                    t_span,
                    test_func_params=sindy_params.test_func_params,
                )
            else:
                raise ValueError(f"Method {sindy_params.method} not recognized. Use 'strong' or 'weak'.")

            xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)
            mode_pred = sindy_tools.create_predictor(xi, library)

            def pred(y: Tensor) -> Tensor:
                cur_mode_coeffs = u_svd.T @ y
                mode_pred_coeffs = mode_pred(cur_mode_coeffs)
                return u_svd @ mode_pred_coeffs

        return pred
