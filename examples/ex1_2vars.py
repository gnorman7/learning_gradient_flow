import os
import time
from pathlib import Path

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector

from learning_gradient_flow import sindy_tools, gradient_flow_optimizer
from example_config import load_config

# Changes behavior from evaluating loss for demonstration purposes vs. skipping to benchmark time.
SKIP_UNUSED_EVALS = False
dtype = torch.float32

def run_example(threshold: float, alpha: float, normalize_columns: bool = True, unbias: bool = True, output_dir: str | None = None):
    torch.manual_seed(0)
    np.random.seed(0)
    plt.close('all')
    plt.style.use('default')
    torch.set_default_dtype(dtype)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    torch.set_default_dtype(dtype)

    # true material parameters
    Aa1 = 2
    Aa2 = 1
    A = torch.tensor([Aa1, Aa2], dtype=dtype)

    # number of basis functions
    N = 30

    # spatial integration grid
    pts = 250
    X = torch.linspace(0, 1, pts + 1)
    dX = X[1] - X[0]
    X += dX / 2
    X = X[:-1]

    # time of simulation
    T = 2

    # number of time steps
    t_steps = 500

    # time grid
    t_grid = torch.linspace(0, T, t_steps)
    dt = t_grid[1] - t_grid[0]

    # basis functions
    F = torch.stack([torch.sin((i + 1) * np.pi * X) for i in range(N)]).T
    dF = torch.stack([np.pi * (i + 1) * torch.cos((i + 1) * np.pi * X) for i in range(N)]).T

    # mass matrix
    M = dX * F.T @ F

    def stiffness(a):
        a1 = a[0]
        a2 = a[1]
        kappa = a1 * torch.ones(pts)
        kappa[X > 0.5] = a2
        mat = torch.einsum('x,xi,xj->ij', kappa, dF, dF)
        return mat

    # forcing function
    f0 = 1000
    force = torch.zeros((N, t_steps))
    force[0, :] = 0 * f0 * torch.sin(1 * np.pi * t_grid)
    force[1, :] = f0 * torch.sin(2 * np.pi * t_grid)

    def solve(a):
        K = stiffness(a)
        mat = torch.linalg.inv(M / dt + K)
        theta = torch.zeros((N, t_steps))
        for i in range(t_steps - 1):
            theta[:, i + 1] = mat @ (force[:, i + 1] + (M / dt) @ theta[:, i])
        return theta

    def loss(a):
        theta = solve(a)
        u = F @ theta
        val = dX * dt * torch.sum((u - v) ** 2)
        return val

    # generate data
    theta = solve(A)
    v = F @ theta

    # for contour plot of loss surface
    ptsa = 35
    a = torch.linspace(0.1, torch.max(A) + 2.25, ptsa)
    A1, A2 = torch.meshgrid(a, a, indexing="ij")
    z = torch.zeros((ptsa, ptsa))

    for i in range(ptsa):
        for j in range(ptsa):
            a = torch.tensor([A1[i, j], A2[i, j]], dtype=dtype)
            z[i, j] = loss(a)

    a0init = torch.tensor([0.5, 4.05], dtype=dtype)

    class parameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(a0init.clone(), requires_grad=True)

        def forward(self):
            return self.a

        def loss(self):
            a = self.forward()
            theta_local = solve(a)
            u = F @ theta_local
            val = torch.log(dX * dt * torch.sum((u - v) ** 2))
            return val

    # optimizer_names = ['sindy_flow']
    optimizer_names = ['sgd', 'sindy_flow']

    optimizers = {}
    for opt_name in optimizer_names:
        optimizers[opt_name] = {
            'model': parameters(),
            'optimizer': None,
            'losses': [],
            'param_history': [],
            'check': 1,
            'optimization_time_sec': 0.0,
        }

    epochs = 750
    lr = 1e-2
    history_size = 25
    retrain_interval = 50
    thresh = 5e-5

    fd_order = 2

    optimizers['sgd']['optimizer'] = torch.optim.SGD(optimizers['sgd']['model'].parameters(), lr=lr)

    solver_fn = sindy_tools.stlsq_sparse_solver
    solver_params = sindy_tools.STLSQParams(
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
        unbias=unbias,
        max_iter=3
    )

    sindy_params = sindy_tools.SINDyParams(
        poly_order=1,
        include_bias=False,
        method='strong',
        fd_order=fd_order,
        solver_fn=solver_fn,
        solver_params=solver_params,
    )

    ode_solver_options = {"method": "rk4", "options": {"step_size": lr}}
    # ode_solver_options = {"method": "dopri5", "atol": lr/10, "rtol": lr/10}
    # ode_solver_options = {}

    optimizers['sindy_flow']['optimizer'] = gradient_flow_optimizer.LGFGradientFlow(
        optimizers['sindy_flow']['model'].parameters(),
        backup_optimizer=torch.optim.SGD(
            optimizers['sindy_flow']['model'].parameters(),
            lr=lr,
        ),
        dt=lr,
        ode_solver_options=ode_solver_options,
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
            loss_val = model.loss()
            loss_val.backward()
            return loss_val

        return closure

    def get_grad(model):
        flat_grad = parameters_to_vector([p.grad for p in model.parameters()])
        return flat_grad

    optimization_start_time = time.perf_counter()
    for epoch in range(epochs):
        for opt_name, opt_data in optimizers.items():
            model = opt_data['model']
            optimizer = opt_data['optimizer']
            closure_fn = create_closure(model, optimizer)
            step_start_time = time.perf_counter()
            loss_val = optimizer.step(closure_fn)
            opt_data['optimization_time_sec'] += time.perf_counter() - step_start_time
            grad_norm = torch.linalg.norm(get_grad(model))

            has_true_loss = not (
                opt_name == 'sindy_flow'
                and SKIP_UNUSED_EVALS
                and optimizer.state['history_count'] >= history_size
                and optimizer.dynamics is not None
            )

            if has_true_loss:
                if torch.exp(loss_val).detach() > thresh and opt_data['check'] == 1:
                    opt_data['losses'].append(loss_val.item())
                    opt_data['param_history'].append(model.a.detach().clone())
                else:
                    opt_data['check'] = 0
            else:
                # In skip mode, learned LGF steps do not have a true loss value, so keep
                # recording the trajectory without letting the placeholder affect `check`.
                opt_data['losses'].append(float('nan'))
                opt_data['param_history'].append(model.a.detach().clone())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, grad_norm {grad_norm.item():.3e}")
    optimization_elapsed = time.perf_counter() - optimization_start_time

    asgd = torch.stack(optimizers['sgd']['param_history'])
    asindy = torch.stack(optimizers['sindy_flow']['param_history'])

    sgd = asgd[-1, :]
    sindy = asindy[-1, :]

    plt.rcParams.update({'font.size': 16})

    fig1 = plt.figure(figsize=(14, 8))
    cp = plt.contourf(A1.numpy(), A2.numpy(), np.log(z.numpy()), levels=30, cmap='viridis')
    plt.scatter(Aa1, Aa2, marker='x', s=250, color='green', label='true parameters')
    plt.plot(asgd[:, 0], asgd[:, 1], color='red', label='GD')
    plt.plot(asindy[:, 0], asindy[:, 1], '--', color='blue', label='LGF')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    plt.title('Comparing optimizers')
    plt.legend()
    plt.colorbar(cp, label='$\\log(z)$')

    fig2 = plt.figure(figsize=(10, 6))
    for opt_name, opt_data in optimizers.items():
        losses = opt_data['losses']
        if opt_name == 'sindy_flow':
            label = 'LGF'
        else:
            label = opt_name.upper()
        plt.plot(np.exp(losses), label=label)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss convergence')
    plt.legend()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / "ex1_2vars"
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    poly_order = sindy_params.poly_order
    prefix = (
        f"thr{threshold}_alpha{alpha}_norm{int(normalize_columns)}_"
        f"unbias{int(unbias)}_p{poly_order}_K{history_size}_M{retrain_interval}_fd{fd_order}_"
    )

    fig1.savefig(output_dir / f"{prefix}contour.png", dpi=300)
    fig2.savefig(output_dir / f"{prefix}loss.png", dpi=300)

    # write to file the final parameter values and final losses
    write_path = output_dir / f"{prefix}results.txt"
    with open(write_path, 'w') as f:
        f.write(f"SGD parameters: {sgd.numpy()}\n")
        f.write(f"SINDy Flow parameters: {sindy.numpy()}\n")
        for opt_name, opt_data in optimizers.items():
            final_loss = np.exp(opt_data['losses'][-1]) if opt_data['losses'] else float('inf')
            f.write(f"{opt_name} final loss: {final_loss}\n")
            f.write(f"{opt_name} optimization_time_sec: {opt_data['optimization_time_sec']:.6f}\n")
        f.write(f"total_optimization_wall_clock_sec: {optimization_elapsed:.6f}\n")
        f.write(f"total epochs: {epochs}\n")

    return {
        "sgd": sgd,
        "sindy": sindy,
        "losses": {k: v['losses'] for k, v in optimizers.items()},
        "timings": {
            "sgd_optimization_time_sec": optimizers['sgd']['optimization_time_sec'],
            "sindy_flow_optimization_time_sec": optimizers['sindy_flow']['optimization_time_sec'],
            "total_optimization_wall_clock_sec": optimization_elapsed,
        }
    }


if __name__ == "__main__":
    config = load_config()
    run_example(threshold=config["threshold"], alpha=config["alpha"])
