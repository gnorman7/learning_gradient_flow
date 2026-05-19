import os
from pathlib import Path

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from learning_gradient_flow import sindy_tools, adam_flow_optimizer
from torch.nn.utils import parameters_to_vector
from example_config import load_config


def run_example(threshold: float, alpha: float, normalize_columns: bool = True, unbias: bool = True, output_dir: str | None = None):
    torch.manual_seed(0)
    np.random.seed(0)
    plt.close('all')
    plt.style.use('default')

    # 1d grid
    pts = 50
    grid = torch.linspace(0, 1, pts)
    dx = grid[1] - grid[0]
    dV = dx ** 3
    grid += dx / 2
    grid = grid[:-1]

    x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")

    # neural network input (full grid for evaluation)
    X = torch.zeros((pts - 1) ** 3, 3)
    X[:, 0] = torch.reshape(x, (1, (pts - 1) ** 3))
    X[:, 1] = torch.reshape(y, (1, (pts - 1) ** 3))
    X[:, 2] = torch.reshape(z, (1, (pts - 1) ** 3))

    batch = 5000

    # constitutive relation
    E = 1
    v = 0.25
    lam = E * v / ((1 + v) * (1 - 2 * v))
    mu = E / (2 * (1 + v))

    C = torch.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] += 2 * mu
    for i in range(3, 6):
        C[i, i] = mu

    def b(X_local):
        choose = 2
        if choose == 0:
            source = torch.zeros((len(X_local), 3))
            source[:, 2] = 5 * torch.ones(len(X_local))
        elif choose == 1:
            source = torch.zeros((len(X_local), 3))
            source[:, 1] = 0.5 * torch.ones(len(X_local))
        elif choose == 2:
            source = torch.zeros((len(X_local), 3))
            source[:, 0] = -1 * (X_local[:, 1] - 0.5)
            source[:, 1] = 1 * (X_local[:, 0] - 0.5)
        elif choose == 3:
            source = torch.zeros((len(X_local), 3))
            source[:, 2] = -4.5 * torch.ones(len(X_local))
        elif choose == 4:
            source = torch.zeros((len(X_local), 3))
            source[:, 0] = -2.5 * (X_local[:, 0] - 0.5)
            source[:, 1] = -2.5 * (X_local[:, 1] - 0.5)
        elif choose == 5:
            source = torch.zeros((len(X_local), 3))
            source[:, 0] = 3 * (X_local[:, 0] - 0.5)
            source[:, 1] = 3 * (X_local[:, 1] - 0.5)
            source[:, 2] = -2.5 * torch.ones(len(X_local))
        return source

    b0 = 10

    which = 1

    class displacement(nn.Module):
        def __init__(self):
            super().__init__()
            n = 20
            self.layer_1 = nn.Linear(3, n)
            self.layer_2 = nn.Linear(n, n)
            self.output = nn.Linear(n, 3, bias=False)
            self.act = nn.Tanh()

        def forward(self, X_local):
            y_local = self.layer_1(X_local)
            y_local = self.act(y_local)
            y_local = self.layer_2(y_local)
            y_local = self.act(y_local)
            y_local = self.output(y_local)

            if which == 0:
                D = (
                    torch.sin(np.pi * X_local[:, 0])
                    * torch.sin(np.pi * X_local[:, 1])
                    * torch.sin(np.pi * X_local[:, 2])
                ).reshape(-1, 1)
            elif which == 1:
                D = X_local[:, 2].reshape(-1, 1)

            u_local = D * y_local
            return u_local

        def loss(self, X_local):
            X_local = X_local.detach().clone().requires_grad_(True)
            B_local = b0 * b(X_local)
            u_local = self.forward(X_local)

            grad_u1 = torch.autograd.grad(u_local[:, 0], X_local, grad_outputs=torch.ones_like(u_local[:, 0]), create_graph=True)[0]
            grad_u2 = torch.autograd.grad(u_local[:, 1], X_local, grad_outputs=torch.ones_like(u_local[:, 1]), create_graph=True)[0]
            grad_u3 = torch.autograd.grad(u_local[:, 2], X_local, grad_outputs=torch.ones_like(u_local[:, 2]), create_graph=True)[0]

            eps11 = grad_u1[:, 0]
            eps22 = grad_u2[:, 1]
            eps33 = grad_u3[:, 2]
            gamma12 = grad_u1[:, 1] + grad_u2[:, 0]
            gamma13 = grad_u1[:, 2] + grad_u3[:, 0]
            gamma23 = grad_u2[:, 2] + grad_u3[:, 1]

            eps = torch.zeros((len(X_local), 6))
            eps[:, 0] = eps11
            eps[:, 1] = eps22
            eps[:, 2] = eps33
            eps[:, 3] = gamma12
            eps[:, 4] = gamma13
            eps[:, 5] = gamma23

            sigma = torch.einsum('ij,xj->xi', C, eps)
            Psi = 0.5 * torch.einsum('xi,xi->', sigma, eps) / batch
            energy = Psi - torch.einsum('xi,xi->', B_local, u_local) / batch
            return energy

    optimizer_names = ['BaseAdam', 'LGFAdam']

    optimizers = {}
    for opt_name in optimizer_names:
        torch.manual_seed(123)
        optimizers[opt_name] = {
            'model': displacement(),
            'optimizer': None,
            'losses': [],
            'param_history': [],
        }

    epochs = 3000
    history_size = 35
    retrain_interval = 50

    betas = (0.9, 0.999)
    lr = 2.5e-3

    solver_fn = sindy_tools.stlsq_sparse_solver
    solver_params = sindy_tools.STLSQParams(
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
        unbias=unbias,
    )
    sindy_params = sindy_tools.SINDyParams(
        poly_order=1,
        include_bias=True,
        solver_fn=solver_fn,
        solver_params=solver_params,
    )

    optimizers['BaseAdam']['optimizer'] = adam_flow_optimizer.BaseAdam(
        optimizers['BaseAdam']['model'].parameters(), lr=lr, betas=betas
    )

    optimizers['LGFAdam']['optimizer'] = adam_flow_optimizer.LGFAdam(
        optimizers['LGFAdam']['model'].parameters(),
        lr=lr,
        betas=betas,
        history_size=history_size,
        retrain_interval=retrain_interval,
        sindy_params=sindy_params,
    )

    for opt_name, opt_data in optimizers.items():
        parameters = parameters_to_vector(opt_data['model'].parameters()).detach().clone()
        opt_data['param_history'].append(parameters)

    def create_closure(model, optimizer, X_batch):
        def closure():
            optimizer.zero_grad()
            loss_val = model.loss(X_batch)
            loss_val.backward()
            return loss_val

        return closure

    for epoch in range(epochs):
        X_batch = torch.rand((batch, 3))
        for opt_name, opt_data in optimizers.items():
            model = opt_data['model']
            optimizer = opt_data['optimizer']
            closure_fn = create_closure(model, optimizer, X_batch)
            loss_val = optimizer.step(closure_fn)
            opt_data['losses'].append(loss_val.item())
            parameters = parameters_to_vector(opt_data['model'].parameters()).detach().clone()
            opt_data['param_history'].append(parameters)

        if epoch % 100 == 0:
            print(f"epoch {epoch}")

    uadam = optimizers['BaseAdam']['model'].forward(X).detach()
    usindy = optimizers['LGFAdam']['model'].forward(X).detach()

    dif = uadam - usindy
    mag_dif = (dif[:, 0] ** 2 + dif[:, 1] ** 2 + dif[:, 2] ** 2)
    numerator = dV * torch.sum(mag_dif)

    mag_adam = (uadam[:, 0] ** 2 + uadam[:, 1] ** 2 + uadam[:, 2] ** 2)
    denominator = dV * torch.sum(mag_adam)

    error = numerator / denominator

    adam_params = torch.stack(optimizers['BaseAdam']['param_history'])
    sindy_params_hist = torch.stack(optimizers['LGFAdam']['param_history'])

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / "ex5_deep_random"
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"thr{threshold}_alpha{alpha}_norm{int(normalize_columns)}_unbias{int(unbias)}_M{retrain_interval}_batch{batch}_"

    np.savetxt(output_dir / f"{prefix}adam_params.txt", adam_params.numpy(), fmt='%.6f')
    np.savetxt(output_dir / f"{prefix}sindy_params.txt", sindy_params_hist.numpy(), fmt='%.6f')
    np.savetxt(output_dir / f"{prefix}adam_losses.txt", optimizers['BaseAdam']['losses'])
    np.savetxt(output_dir / f"{prefix}sindy_losses.txt", optimizers['LGFAdam']['losses'])

    adam_losses = torch.tensor(np.loadtxt(output_dir / f"{prefix}adam_losses.txt"), dtype=torch.float32)
    sindy_losses = torch.tensor(np.loadtxt(output_dir / f"{prefix}sindy_losses.txt"), dtype=torch.float32)

    k = 15
    indices = torch.randperm(adam_params.shape[1])[:k]
    adam_params_plot = adam_params[:, indices]
    sindy_params_plot = sindy_params_hist[:, indices]

    fig = plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.rcParams.update({'font.size': 16})

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(adam_losses, color='red', label='ADAM')
    ax1.plot(sindy_losses, color='blue', linestyle='--', label='LGF')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('$z$')
    ax1.set_title('Comparing optimizers')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(adam_params_plot[:, 0], color='red', label='ADAM')
    ax2.plot(sindy_params_plot[:, 0], color='blue', linestyle='--', label='LGF')
    ax2.plot(adam_params_plot[:, 1:], color='red')
    ax2.plot(sindy_params_plot[:, 1:], color='blue', linestyle='--')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('$a$')
    ax2.set_title('Parameter trajectories')
    ax2.legend()

    fig.savefig(output_dir / f"{prefix}dense.png", dpi=300)

    # save the error value in a text file
    error_file = output_dir / f"{prefix}error.txt"
    with open(error_file, 'w') as f:
        f.write(f"relative_error: {error.item()}\n")

    return {
        "error": error.item(),
        "adam_params": adam_params,
        "sindy_params": sindy_params_hist,
    }


if __name__ == "__main__":
    config = load_config()
    run_example(threshold=config["threshold"], alpha=config["alpha"])
