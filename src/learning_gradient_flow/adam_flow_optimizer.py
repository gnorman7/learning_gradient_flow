# mypy: allow-untyped-defs
import math
import warnings
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

import learning_gradient_flow.sindy_tools as sindy_tools
from learning_gradient_flow.gradient_flow_optimizer import VectorBasedOptimizer
from learning_gradient_flow.sindy_tools import SINDyParams

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
    warnings.warn(
        "torchdiffeq library not found. LGFAdam will not work. "
        "Install with: pip install torchdiffeq"
    )


class BaseAdam(VectorBasedOptimizer):
    """Baseline Adam update written in the optimizer-state form used by the paper."""

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
        self.state = {"epoch": 0}
        self.state["exp_avg"] = None
        self.state["exp_avg_sq"] = None
        if amsgrad:
            self.state["max_exp_avg_sq"] = None
        self.state["func_evals"] = 0

    def step(self, closure: Optional[Callable[[], Tensor]] = None):
        flat_params = self._gather_flat("params")

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grads = self._gather_flat("grads")
        self.state["func_evals"] += 1

        beta1, beta2 = self.defaults["betas"]
        lr = self.defaults["lr"]
        eps = self.defaults["eps"]
        amsgrad = self.defaults["amsgrad"]

        if self.state.get("exp_avg") is None:
            self.state["exp_avg"] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            self.state["exp_avg_sq"] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
            if amsgrad:
                self.state["max_exp_avg_sq"] = torch.zeros_like(
                    flat_params, memory_format=torch.preserve_format
                )

        exp_avg: Tensor = self.state["exp_avg"]
        exp_avg_sq: Tensor = self.state["exp_avg_sq"]
        if amsgrad:
            max_exp_avg_sq: Tensor = self.state["max_exp_avg_sq"]

        dmdt_eta = (1 - beta1) * (grads - exp_avg)
        dvdt_eta = (1 - beta2) * (grads**2 - exp_avg_sq)

        exp_avg = exp_avg + dmdt_eta
        exp_avg_sq = exp_avg_sq + dvdt_eta

        if amsgrad:
            max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)

        self.state["epoch"] += 1
        step_t = self.state["epoch"]
        unbiased_exp_avg = exp_avg / (1 - beta1**step_t)
        if amsgrad:
            unbiased_exp_avg_sq = max_exp_avg_sq / (1 - beta2**step_t)
        else:
            unbiased_exp_avg_sq = exp_avg_sq / (1 - beta2**step_t)

        dparams_dt = -unbiased_exp_avg / (unbiased_exp_avg_sq.sqrt() + eps)
        flat_params = flat_params + lr * dparams_dt

        self._set_params_from_flat(flat_params)
        self.state["exp_avg"] = exp_avg
        self.state["exp_avg_sq"] = exp_avg_sq
        if amsgrad:
            self.state["max_exp_avg_sq"] = max_exp_avg_sq

        if closure is not None:
            return loss


class LGFAdam(VectorBasedOptimizer):
    """
    Adam optimizer accelerated with a learned SINDy gradient surrogate.

    Warm-up steps use true gradients and collect parameter/gradient history.
    Later steps integrate the Adam ODE with the learned gradient model.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 1e-3,
        dt: Optional[float] = None,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        history_size: int = 100,
        retrain_interval: int = 200,
        ode_solver_options: Optional[dict] = None,
        sindy_params: Optional[SINDyParams] = None,
        skip_unused_evals: bool = False,
    ):
        if torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq library is required for LGFAdam. "
                "Install with: pip install torchdiffeq"
            )
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
        self.state["epoch"] = 0
        self.state["t_global"] = 0.0
        self.state["func_evals"] = 0
        self.state["skip_unused_evals"] = skip_unused_evals

        self.state["exp_avg"] = None
        self.state["exp_avg_sq"] = None
        self.state["max_exp_avg_sq"] = None

        self.state["history"] = []
        self.state["grad_history"] = []
        self.dynamics = None

        if sindy_params is None:
            sindy_params = sindy_tools.SINDyParams()
            sindy_params.method = "tracked"
        elif sindy_params.method != "tracked":
            warnings.warn(f"SINDy method {sindy_params.method} is not 'tracked'. Using 'tracked' instead.")
            sindy_params.method = "tracked"
        self.sindy_params = sindy_params

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

        a = self._gather_flat("params").detach()
        g = self._gather_flat("grads").detach()
        if needs_true_eval:
            self.state["func_evals"] += 1

        if self.state["exp_avg"] is None:
            self.state["exp_avg"] = torch.zeros_like(a, memory_format=torch.preserve_format)
            self.state["exp_avg_sq"] = torch.zeros_like(a, memory_format=torch.preserve_format)
            if amsgrad:
                self.state["max_exp_avg_sq"] = torch.zeros_like(a, memory_format=torch.preserve_format)

        if len(self.state["history"]) < history_size:
            self.state["history"].append(a.clone().detach())
            self.state["grad_history"].append(g.clone().detach())

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

            mhat = m / (1.0 - (beta1**step_t))
            if amsgrad:
                vhat = vmax / (1.0 - (beta2**step_t))
            else:
                vhat = v / (1.0 - (beta2**step_t))

            dadt = -mhat / (vhat.sqrt() + eps)
            a_next = a + eta * dadt
            self._set_params_from_flat(a_next)

            self.state["exp_avg"] = m
            self.state["exp_avg_sq"] = v
            if amsgrad:
                self.state["max_exp_avg_sq"] = vmax

            return loss

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

        if amsgrad:
            vmax_proj = torch.maximum(self.state["max_exp_avg_sq"], v_f)
            self.state["max_exp_avg_sq"] = vmax_proj
        self.state["exp_avg"] = m_f
        self.state["exp_avg_sq"] = v_f

        self._set_params_from_flat(a_f)

        self.state["epoch"] += 1
        self.state["t_global"] += dt

        if skip_unused_evals:
            return torch.tensor(-1.0, device=a_f.device, dtype=a_f.dtype)
        return loss

    def build_sindy_model(self, sindy_params: sindy_tools.SINDyParams) -> Callable[[Tensor], Tensor]:
        x = torch.stack(self.state["history"], dim=0)
        dt_sindy = self.defaults["lr"]
        t_span = torch.arange(x.shape[0]) * dt_sindy

        if sindy_params.truncation_rank is None:
            d = x.shape[1]
            library = sindy_tools.create_sindy_library(
                input_dim=d,
                poly_order=sindy_params.poly_order,
                include_bias=sindy_params.include_bias,
                use_ortho=sindy_params.use_ortho,
            )
            theta = library(x)
            if sindy_params.method == "strong":
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(x, theta, t_span)
            elif sindy_params.method == "weak":
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(
                    x, theta, t_span, test_func_params=sindy_params.test_func_params
                )
            elif sindy_params.method == "tracked":
                lhs_target = torch.stack(self.state["grad_history"], dim=0)
                rhs_mat = theta
            else:
                raise ValueError(
                    f"Method {sindy_params.method} not recognized. Use 'strong', 'weak', or 'tracked'."
                )

            xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)
            pred = sindy_tools.create_predictor(xi, library)
        else:
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

            if sindy_params.method == "strong":
                lhs_target, rhs_mat = sindy_tools.assemble_strong_matrices(
                    mode_coeffs.T, theta, t_span
                )
            elif sindy_params.method == "weak":
                lhs_target, rhs_mat = sindy_tools.assemble_weak_matrices(
                    mode_coeffs.T,
                    theta,
                    t_span,
                    test_func_params=sindy_params.test_func_params,
                )
            elif sindy_params.method == "tracked":
                grad_hist = torch.stack(self.state["grad_history"], dim=0)
                lhs_target = grad_hist @ u_svd
                rhs_mat = theta
            else:
                raise ValueError(
                    f"Method {sindy_params.method} not recognized. Use 'strong', 'weak', or 'tracked'."
                )

            xi = sindy_params.solver_fn(rhs_mat, lhs_target, sindy_params.solver_params)
            mode_pred = sindy_tools.create_predictor(xi, library)

            def pred(y: Tensor) -> Tensor:
                cur_mode_coeffs = u_svd.T @ y
                mode_pred_coeffs = mode_pred(cur_mode_coeffs)
                return u_svd @ mode_pred_coeffs

        return pred
