import os
import time
from pathlib import Path

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import gc
from torch.nn.utils import parameters_to_vector

from learning_gradient_flow import gradient_flow_optimizer, sindy_tools
from example_config import load_config

# Changes behavior from evaluating loss for demonstration purposes vs. skipping to benchmark time.
SKIP_UNUSED_EVALS = False

plt.close('all')
plt.style.use('default')

torch.set_default_dtype(torch.float32)

# number of shape functions in each direction
M = 15

# 2D integration grid
pts = 20 * M
x = torch.linspace(0, 1, pts + 1)
dx = x[1] - x[0]
dA = dx ** 2
x += dx / 2
x = x[:-1]
xx, yy = torch.meshgrid(x, x, indexing="ij")
X = torch.zeros((pts ** 2, 2))
X[:, 0] = torch.reshape(xx, (1, pts ** 2))
X[:, 1] = torch.reshape(yy, (1, pts ** 2))

p = 2


def build_basis(X_local, M_local):
    phi_local = torch.zeros((len(X_local), M_local ** 2))
    count = 0
    for i in range(1, M_local + 1):
        for j in range(1, M_local + 1):
            phi_local[:, count] = (
                torch.sin(i * torch.pi * X_local[:, 0])
                * torch.sin(j * torch.pi * X_local[:, 1])
            )
            count += 1
    return phi_local


def build_basis_grad(X_local, M_local):
    phi_x = torch.zeros((len(X_local), M_local ** 2))
    phi_y = torch.zeros((len(X_local), M_local ** 2))
    count = 0
    for i in range(1, M_local + 1):
        for j in range(1, M_local + 1):
            phi_x[:, count] = (
                i * torch.pi * torch.cos(i * torch.pi * X_local[:, 0])
                * torch.sin(j * torch.pi * X_local[:, 1])
            )
            phi_y[:, count] = (
                torch.sin(i * torch.pi * X_local[:, 0])
                * j * torch.pi * torch.cos(j * torch.pi * X_local[:, 1])
            )
            count += 1

    phi_grad_local = torch.zeros((len(X_local), M_local ** 2, 2))
    phi_grad_local[:, :, 0] = phi_x
    phi_grad_local[:, :, 1] = phi_y

    return phi_grad_local


def kappa(X_local):
    k0 = 1
    k1 = 20
    center = torch.tensor([0.5, 0.5])
    half_width = 0.25

    vals = k0 * torch.ones(len(X_local))
    mask = torch.all(torch.abs(X_local - center) <= half_width, dim=1)
    vals[mask] = k1
    return vals


def body(X_local):
    b0 = 1e7
    i = 4
    j = 3
    vals = b0 * X_local[:, 0] * torch.sin(i * torch.pi * X_local[:, 0]) * torch.sin(j * torch.pi * X_local[:, 1])
    return vals


sigma = 4

phi = build_basis(X, M)
phi_grad = build_basis_grad(X, M)
K = kappa(X)
B = body(X)

fext = dA * phi.T @ B


def system(a):
    u = phi @ a
    grad_u = phi_grad.transpose(1, 2) @ a
    dot_grad_u = grad_u.square().sum(dim=1)
    stiffness = torch.bmm(phi_grad, grad_u.unsqueeze(-1)).squeeze(-1)

    diffusion_obj = dA * torch.sum(K * dot_grad_u.pow(p))
    radiation_obj = dA * torch.sum(sigma * u.pow(5) / 5.0)
    force_obj = dA * torch.sum(B * u)
    obj = diffusion_obj + radiation_obj - force_obj

    coeff = 2.0 * K * p * dot_grad_u.pow(p - 1)
    diffusion = dA * (stiffness * coeff[:, None]).sum(dim=0)
    radiation = dA * (phi * (sigma * u.pow(4))[:, None]).sum(dim=0)
    sys = radiation - fext + diffusion

    coeff1 = 4.0 * K * p * (p - 1.0) * dot_grad_u.pow(p - 2)
    term1 = dA * (stiffness.T @ (stiffness * coeff1[:, None]))

    coeff2 = 2.0 * K * p * dot_grad_u.pow(p - 1)
    term2 = dA * torch.einsum('x,xil,xjl->ij', coeff2, phi_grad, phi_grad)

    diffusion_hess = term1 + term2
    w_rad = 4.0 * sigma * u.pow(3)
    radiation_hess = dA * (phi.T @ (phi * w_rad[:, None]))

    hess = radiation_hess + diffusion_hess
    return obj, sys, hess


def run_example(threshold: float, alpha: float, normalize_columns: bool = True, unbias: bool = True, output_dir: str | None = None):
    torch.manual_seed(0)
    np.random.seed(0)
    a0init = 3 + -6 * torch.rand(M ** 2, dtype=phi.dtype, device=phi.device)

    class parameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(a0init.clone(), requires_grad=True)

        def forward(self):
            return self.a

    optimizer_names = ['sgd', 'sindy_flow']

    optimizers = {}
    for opt_name in optimizer_names:
        optimizers[opt_name] = {
            'model': parameters(),
            'optimizer': None,
            'losses': [],
            'thetas': [],
            'param_history': [],
            'check': 1,
            'optimization_time_sec': 0.0,
        }

    epochs = 300
    lr = 0.15
    history_size = 15
    retrain_interval = 20
    thresh = 1e12

    solver_fn = sindy_tools.stlsq_sparse_solver
    solver_params = sindy_tools.STLSQParams(
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
        unbias=unbias,
        max_iter=3,
    )
    sindy_params = sindy_tools.SINDyParams(
        poly_order=2,
        include_bias=True,
        solver_fn=solver_fn,
        solver_params=solver_params,
    )

    optimizers['sgd']['optimizer'] = torch.optim.SGD(optimizers['sgd']['model'].parameters(), lr=lr)

    optimizers['sindy_flow']['optimizer'] = gradient_flow_optimizer.SINDyFlow(
        optimizers['sindy_flow']['model'].parameters(),
        backup_optimizer=torch.optim.SGD(
            optimizers['sindy_flow']['model'].parameters(),
            lr=lr,
        ),
        dt=lr,
        # ode_solver_options={"method": "dopri5", "options": {"atol": lr/10, "rtol": lr/10}},
        # ode_solver_options={"method": "dopri5", "atol": lr / 10, "rtol": lr / 10},
        # ode_solver_options={"method": "dopri5"},
        ode_solver_options={"method": "rk4", "options": {"step_size": lr}},
        history_size=history_size,
        retrain_interval=retrain_interval,
        sindy_params=sindy_params,
        skip_unused_evals=SKIP_UNUSED_EVALS,
    )

    for opt_name, opt_data in optimizers.items():
        opt_data['param_history'].append(opt_data['model'].a.detach().clone())

    def create_closure(model, optimizer):
        def closure():
            optimizer.zero_grad()
            params = parameters_to_vector(model.parameters())
            z_val, R, J = system(params)
            relax = 1.0
            gradient = relax * torch.linalg.solve(J, R)
            model.a.grad = gradient.clone()
            return z_val, R

        return closure

    def evaluate_model(model):
        params = parameters_to_vector(model.parameters())
        z_val, R, J = system(params)
        gradient = torch.linalg.solve(J, R)
        return z_val, R, gradient

    final_grad_sgd = None
    final_grad_sindy = None
    initial_sys_sgd = None
    initial_sys_sindy = None
    final_sys_sgd = None
    final_sys_sindy = None
    res0 = {}

    optimization_start_time = time.perf_counter()
    for epoch in range(epochs):
        for opt_name, opt_data in optimizers.items():
            model = opt_data['model']
            optimizer = opt_data['optimizer']
            closure_fn = create_closure(model, optimizer)
            step_start_time = time.perf_counter()
            step_result = optimizer.step(closure_fn)
            opt_data['optimization_time_sec'] += time.perf_counter() - step_start_time

            if isinstance(step_result, tuple):
                loss_val, R = step_result
            else:
                loss_val = step_result
                R = None

            if epoch == 0 and R is not None:
                if opt_name == 'sgd':
                    initial_sys_sgd = R.detach().clone()
                elif opt_name == 'sindy_flow':
                    initial_sys_sindy = R.detach().clone()

            if R is not None:
                res = torch.linalg.norm(R)
                if epoch == 0:
                    res0[opt_name] = res
                    print(res0[opt_name])

                if res0[opt_name] / res < thresh and opt_data['check'] == 1:
                    opt_data['losses'].append(res.item())
                    opt_data['param_history'].append(model.a.detach().clone())
                else:
                    opt_data['check'] = 0
            else:
                # In skip mode, SINDyFlow does not reevaluate the closure after the warmup phase,
                # so we keep tracking parameters without adding extra residual evaluations.
                opt_data['losses'].append(float('nan'))
                opt_data['param_history'].append(model.a.detach().clone())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}")
    optimization_elapsed = time.perf_counter() - optimization_start_time

    _, final_sys_sgd_eval, final_grad_sgd_eval = evaluate_model(optimizers['sgd']['model'])
    _, final_sys_sindy_eval, final_grad_sindy_eval = evaluate_model(optimizers['sindy_flow']['model'])

    if initial_sys_sgd is None or initial_sys_sindy is None:
        raise RuntimeError("Initial system residuals were not captured during optimization.")

    final_grad_sgd = final_grad_sgd_eval.detach().clone()
    final_grad_sindy = final_grad_sindy_eval.detach().clone()
    final_sys_sgd = final_sys_sgd_eval.detach().clone()
    final_sys_sindy = final_sys_sindy_eval.detach().clone()

    sgd_params = torch.stack(optimizers['sgd']['param_history'])
    sindy_params_hist = torch.stack(optimizers['sindy_flow']['param_history'])

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / "ex3_newton"
    else:
        output_dir = Path(output_dir)

    txt_dir = output_dir / "txts"
    fig_dir = output_dir / "figs"
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    poly_order = sindy_params.poly_order
    prefix = (
        f"unbias{int(unbias)}_norm{int(normalize_columns)}_thr{threshold}_"
        f"alpha{alpha}_p{poly_order}_M{retrain_interval}_"
    )

    np.savetxt(txt_dir / f"{prefix}sgd_params.txt", sgd_params.numpy(), fmt='%.6f')
    np.savetxt(txt_dir / f"{prefix}sindy_params.txt", sindy_params_hist.numpy(), fmt='%.6f')

    np.savetxt(txt_dir / f"{prefix}sgd_losses.txt", optimizers['sgd']['losses'])
    np.savetxt(txt_dir / f"{prefix}sindy_losses.txt", optimizers['sindy_flow']['losses'])

    sgd_params = torch.tensor(np.loadtxt(txt_dir / f"{prefix}sgd_params.txt"), dtype=phi.dtype)
    sindy_params_hist = torch.tensor(np.loadtxt(txt_dir / f"{prefix}sindy_params.txt"), dtype=phi.dtype)

    sgd_losses = torch.tensor(np.loadtxt(txt_dir / f"{prefix}sgd_losses.txt"), dtype=phi.dtype)
    sindy_losses = torch.tensor(np.loadtxt(txt_dir / f"{prefix}sindy_losses.txt"), dtype=phi.dtype)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    a_sgd = sgd_params[-1:, :].squeeze()
    a_sindy = sindy_params_hist[-1:, :].squeeze()

    u_newton = phi @ a_sgd
    u_sindy = phi @ a_sindy

    denominator = torch.sum(u_newton ** 2)
    numerator = torch.sum((u_sindy - u_newton) ** 2)
    dif = numerator / denominator

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.rcParams.update({'font.size': 16})

    plt.subplot(2, 2, 1)
    plt.semilogy(sgd_losses, color='red', label='Newton')
    plt.semilogy(sindy_losses, color='blue', linestyle='--', label='LGF')
    plt.xlabel('epoch')
    plt.ylabel('$| \\partial z / \\partial a |$')
    plt.title('Comparing optimizers')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(sgd_params[:, 0], color='red', label='Newton')
    plt.plot(sindy_params_hist[:, 0], color='blue', linestyle='--', label='LGF')
    plt.plot(sgd_params[:, 1:], color='red')
    plt.plot(sindy_params_hist[:, 1:], color='blue', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('$a$')
    plt.title('Parameter trajectories')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.contourf(xx, yy, torch.reshape((phi @ a_sgd), (pts, pts)), levels=30)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('True temperature field')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.contourf(xx, yy, torch.reshape((phi @ torch.abs(a_sindy - a_sgd)), (pts, pts)), levels=30)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Magnitude of error')
    plt.colorbar()

    plt.savefig(fig_dir / f"{prefix}comparison.png", dpi=300)
    plt.close('all')

    # also save the dif value in a text file
    dif_file = txt_dir / f"{prefix}dif.txt"
    with open(dif_file, 'w') as f:
        f.write(f"dif: {dif.item()}\n")

    initial_sys_norm_sgd = torch.linalg.norm(initial_sys_sgd).item()
    initial_sys_norm_sindy = torch.linalg.norm(initial_sys_sindy).item()
    final_sys_norm_sgd = torch.linalg.norm(final_sys_sgd).item()
    final_sys_norm_sindy = torch.linalg.norm(final_sys_sindy).item()

    final_grad_norm_sgd = torch.linalg.norm(final_grad_sgd).item()
    final_grad_norm_sindy = torch.linalg.norm(final_grad_sindy).item()
    grad_file = txt_dir / f"{prefix}grad_norms.txt"
    with open(grad_file, 'w') as f:
        f.write(f"initial_sys_norm_sgd: {initial_sys_norm_sgd}\n")
        f.write(f"initial_sys_norm_sindy: {initial_sys_norm_sindy}\n")
        f.write(f"final_sys_norm_sgd: {final_sys_norm_sgd}\n")
        f.write(f"final_sys_norm_sindy: {final_sys_norm_sindy}\n")
        f.write(f"final_grad_norm_sgd: {final_grad_norm_sgd}\n")
        f.write(f"final_grad_norm_sindy: {final_grad_norm_sindy}\n")

    timing_file = txt_dir / f"{prefix}timings.txt"
    with open(timing_file, 'w') as f:
        f.write(f"sgd optimization_time_sec: {optimizers['sgd']['optimization_time_sec']:.6f}\n")
        f.write(f"sindy_flow optimization_time_sec: {optimizers['sindy_flow']['optimization_time_sec']:.6f}\n")
        f.write(f"total_optimization_wall_clock_sec: {optimization_elapsed:.6f}\n")
        f.write(f"total epochs: {epochs}\n")
        f.write(f"skip_unused_evals: {int(SKIP_UNUSED_EVALS)}\n")


    del sgd_params, sindy_params_hist, sgd_losses, sindy_losses, a_sgd, a_sindy, u_newton, u_sindy
    gc.collect()

    return {
        "diff": dif.item(),
        "timings": {
            "sgd_optimization_time_sec": optimizers['sgd']['optimization_time_sec'],
            "sindy_flow_optimization_time_sec": optimizers['sindy_flow']['optimization_time_sec'],
            "total_optimization_wall_clock_sec": optimization_elapsed,
        },
    }


if __name__ == "__main__":
    config = load_config()
    run_example(threshold=config["threshold"], alpha=config["alpha"])
