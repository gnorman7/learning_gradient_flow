import os
import time
from pathlib import Path

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from learning_gradient_flow import adam_flow_optimizer, sindy_tools
from example_config import load_config

import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv


plt.close('all')
plt.style.use('default')

plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

torch.set_default_dtype(torch.float32)

# Changes behavior from evaluating loss for demonstration purposes vs. skipping to benchmark time.
SKIP_UNUSED_EVALS = False


def run_example(
    threshold: float,
    alpha: float,
    normalize_columns: bool = True,
    unbias: bool = True,
    output_dir: str | None = None,
    show_plots: bool = False,
):
    torch.manual_seed(0)
    np.random.seed(0)
    # BUILD GEOMETRY AND MESH
    g = cfg.Geometry()

    g.point([0.5, 1])
    g.point([0, 1])
    g.point([0, 0])
    g.point([1, 0])
    g.point([1, 1])

    g.spline([0, 1], marker=2)
    g.spline([1, 2], marker=1)
    g.spline([2, 3], marker=1)
    g.spline([3, 4], marker=1)
    g.spline([4, 0], marker=2)

    g.surface([0, 1, 2, 3, 4])

    mesh = cfm.GmshMesh(g)
    mesh.el_type = 2
    mesh.dofs_per_node = 1
    mesh.el_size_factor = 0.09

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    n_el = edof.shape[0]
    nDofs = np.size(dofs)

    ex, ey = cfc.coordxtr(edof, coords, dofs)

    edof = torch.tensor(edof, dtype=torch.long)
    dofs = torch.tensor(dofs, dtype=torch.long)
    ex = torch.tensor(ex, dtype=torch.float32)
    ey = torch.tensor(ey, dtype=torch.float32)

    ec = torch.zeros(n_el, 2)
    ec[:, 0] = torch.mean(ex, axis=1)
    ec[:, 1] = torch.mean(ey, axis=1)

    bcdofs = torch.tensor(bdofs[1], dtype=torch.long)
    freedofs = torch.tensor(bdofs[2], dtype=torch.long)

    nr = nDofs - len(bcdofs)

    print(f'{n_el} elements, {nDofs} nodes, {nr} degrees of freedom')

    Meunit = (1 / 12) * torch.tensor(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=torch.float32
    )

    x1, x2, x3 = ex[:, 0], ex[:, 1], ex[:, 2]
    y1, y2, y3 = ey[:, 0], ey[:, 1], ey[:, 2]
    areas = 0.5 * torch.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    Me_base = areas[:, None, None] * Meunit

    C = torch.ones((n_el, 3, 3), dtype=torch.float32)
    C[:, :, 1] = ex
    C[:, :, 2] = ey
    C_inv = torch.linalg.inv(C)
    B_base = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    B = torch.matmul(B_base, C_inv)
    A = 0.5 * torch.linalg.det(C)
    Ke_all = A[:, None, None] * (B.transpose(1, 2) @ B)

    all_dofs = dofs.flatten() - 1
    bcdofs_0 = bcdofs - 1

    keep_mask = ~torch.isin(all_dofs, bcdofs_0)
    keep = all_dofs[keep_mask]

    T = 1
    t_steps = 300
    t_grid = torch.linspace(0, T, t_steps)
    dt = t_grid[1] - t_grid[0]

    def force(t):
        F0 = 10
        return F0 * torch.sin(4 * np.pi * t)

    F = torch.zeros([nr, t_steps])
    F[0, :] = force(t_grid)

    def m(ec_local):
        return 1 + 2 * ec_local[:, 1]

    m_true = m(ec)

    n_obs = nr // 1
    idx = torch.randperm(nr)[:n_obs]

    G = torch.zeros((n_obs, nr))
    G[torch.arange(n_obs), idx] = 1

    triangles = edof
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles - 1)

    fig = plt.figure(figsize=(14, 3.5))
    plt.rcParams.update({'font.size': 16})

    from matplotlib.gridspec import GridSpec

    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25, hspace=0.1, bottom=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    cfv.drawMesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofsPerNode,
        el_type=mesh.elType,
        filled=True,
        title='Geometry and BCs'
    )

    ax1.scatter(coords[:, 0][bcdofs - 1], coords[:, 1][bcdofs - 1], color='blue', label='fixed nodes')
    ax1.scatter(coords[:, 0][keep[idx]], coords[:, 1][keep[idx]], color='green', label='measurements')
    ax1.scatter(coords[:, 0][freedofs - 1], coords[:, 1][freedofs - 1], color='red', label='free nodes')
    ax1.scatter(coords[:, 0][0], coords[:, 1][0], color='pink', s=150, label='applied force')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_grid.numpy(), force(t_grid).numpy())
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$f(t)$')
    ax2.set_title('Forcing')

    ax3 = fig.add_subplot(gs[0, 2])
    pc = ax3.tripcolor(
        triang,
        facecolors=m_true,
        edgecolors='k',
        cmap='viridis'
    )
    fig.colorbar(pc, ax=ax3)
    ax3.set_title("Slowness field")
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    eltopo0 = edof - 1
    local_i, local_j = torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij")
    local_i = local_i.reshape(-1)
    local_j = local_j.reshape(-1)
    global_i = eltopo0[:, local_i].reshape(-1)
    global_j = eltopo0[:, local_j].reshape(-1)

    K = torch.zeros([nDofs, nDofs])
    K_vals = Ke_all.reshape(-1)
    K.index_put_((global_i, global_j), K_vals, accumulate=True)
    Kr = K[keep][:, keep]

    def solve(m_local):
        M = torch.zeros([nDofs, nDofs])
        M_vals = (m_local[:, None, None] * Me_base).reshape(-1)
        M.index_put_((global_i, global_j), M_vals, accumulate=True)

        Mr = M[keep][:, keep]
        Mrinv = torch.linalg.inv(Mr)

        u = torch.zeros((nr, t_steps))
        uplot = torch.zeros((nDofs, t_steps))

        for t in range(1, t_steps - 1):
            forcing = Mrinv @ (F[:, t] - Kr @ u[:, t])
            u[:, t + 1] = dt ** 2 * forcing + 2 * u[:, t] - u[:, t - 1]
            uplot[keep, t + 1] = u[:, t + 1]

        return u, uplot

    utrue, uplot = solve(m_true)

    v = G @ utrue

    u_max = uplot.max().item()

    print(f'max pressure value: {round(u_max, 2)}')

    class solver(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(1 * torch.ones(n_el) + 0 * torch.rand(n_el))

        def forward(self):
            u_local, _ = solve(torch.abs(self.a) + 1e-2)
            return u_local

        def loss(self):
            u_local = self.forward()
            error = 0.5 * dt * torch.sum((G @ u_local - v) ** 2)
            return error

    optimizer_names = ['AppendixAdam', 'AppendixAdamLearned']
    optimizers = {}
    for opt_name in optimizer_names:
        optimizers[opt_name] = {
            'model': solver(),
            'optimizer': None,
            'losses': [],
            'param_history': [],
            'optimization_time_sec': 0.0,
        }

    epochs = 2200
    history_size = 20
    retrain_interval = 40

    betas = (0.9, 0.999)
    lr = 2e-2

    solver_fn = sindy_tools.stlsq_sparse_solver
    solver_params = sindy_tools.STLSQParams(
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
        unbias=unbias,
        max_iter=3
    )

    ode_solver_options = {"method": "rk4", "options": {"step_size": lr}}
    sindy_params = sindy_tools.SINDyParams(
        poly_order=1,
        include_bias=True,
        solver_fn=solver_fn,
        solver_params=solver_params,
        use_ortho=False,
    )

    optimizers['AppendixAdam']['optimizer'] = torch.optim.Adam(
        optimizers['AppendixAdam']['model'].parameters(), lr=lr, betas=betas
    )

    optimizers['AppendixAdamLearned']['optimizer'] = adam_flow_optimizer.AppendixAdamContLearned(
        optimizers['AppendixAdamLearned']['model'].parameters(),
        lr=lr,
        betas=betas,
        history_size=history_size,
        retrain_interval=retrain_interval,
        sindy_params=sindy_params,
        skip_unused_evals=SKIP_UNUSED_EVALS,
        ode_solver_options=ode_solver_options,
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

    optimization_start_time = time.perf_counter()
    for epoch in range(epochs):
        for opt_name, opt_data in optimizers.items():
            model = opt_data['model']
            optimizer = opt_data['optimizer']
            closure_fn = create_closure(model, optimizer)
            step_start_time = time.perf_counter()
            loss_val = optimizer.step(closure_fn)
            opt_data['optimization_time_sec'] += time.perf_counter() - step_start_time

            if (
                opt_name == 'AppendixAdamLearned'
                and SKIP_UNUSED_EVALS
                and len(optimizer.state["history"]) >= history_size
                and optimizer.dynamics is not None
            ):
                opt_data['losses'].append(float('nan'))
            else:
                opt_data['losses'].append(loss_val.item())
            opt_data['param_history'].append(model.a.detach().clone())

        dif = torch.linalg.norm(torch.abs(model.a) - m_true)

        if epoch % 10 == 0:
            adam_loss = optimizers["AppendixAdam"]["losses"][-1]
            lgf_loss = optimizers["AppendixAdamLearned"]["losses"][-1]
            print(f'epoch {epoch}, parameter error {round(dif.item(), 2)}')
            print(f'Adam loss: {round(adam_loss, 4)}, LGF loss: {round(lgf_loss, 4)}')
    optimization_elapsed = time.perf_counter() - optimization_start_time

    adam_params = torch.stack(optimizers['AppendixAdam']['param_history'])
    sindy_params_hist = torch.stack(optimizers['AppendixAdamLearned']['param_history'])

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / "ex4_scattering_tests"
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    poly_order = sindy_params.poly_order
    prefix = (
        f"thr{threshold}_alpha{alpha}_norm{int(normalize_columns)}_"
        f"unbias{int(unbias)}_p{poly_order}_M{retrain_interval}_"
    )

    np.savetxt(output_dir / f'{prefix}adam_params.txt', adam_params.numpy(), fmt='%.6f')
    np.savetxt(output_dir / f'{prefix}sindy_params.txt', sindy_params_hist.numpy(), fmt='%.6f')
    np.savetxt(output_dir / f'{prefix}adam_losses.txt', optimizers['AppendixAdam']['losses'])
    np.savetxt(output_dir / f'{prefix}sindy_losses.txt', optimizers['AppendixAdamLearned']['losses'])

    adam_params = torch.abs(torch.tensor(np.loadtxt(output_dir / f'{prefix}adam_params.txt'), dtype=torch.float32))
    sindy_params_hist = torch.abs(torch.tensor(np.loadtxt(output_dir / f'{prefix}sindy_params.txt'), dtype=torch.float32))

    adam_losses = torch.tensor(np.loadtxt(output_dir / f'{prefix}adam_losses.txt'), dtype=torch.float32)
    sindy_losses = torch.tensor(np.loadtxt(output_dir / f'{prefix}sindy_losses.txt'), dtype=torch.float32)

    aadam = adam_params[-1:, :].squeeze()
    asindy = sindy_params_hist[-1:, :].squeeze()

    adam_error = torch.abs(aadam - m_true)
    sindy_error = torch.abs(asindy - m_true)

    norm = torch.sum(m_true ** 2)

    adam_sq = torch.sum((aadam - m_true) ** 2)
    sindy_sq = torch.sum((asindy - m_true) ** 2)

    mse_adam = adam_sq / norm
    mse_sindy = sindy_sq / norm


    k = 15
    indices = torch.randperm(adam_params.shape[1])[:k]
    adam_params_plot = torch.abs(adam_params[:, indices])
    sindy_params_plot = torch.abs(sindy_params_hist[:, indices])

    fig = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    plt.rcParams.update({'font.size': 16})
    plt.subplot(2, 2, 1)
    plt.semilogy(adam_losses, color='red', label='ADAM')
    plt.semilogy(sindy_losses, color='blue', linestyle='--', label='LGF')
    plt.xlabel('epoch')
    plt.ylabel('$z$')
    plt.title('Comparing optimizers')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(adam_params_plot[:, 0], color='red', label='ADAM')
    plt.plot(sindy_params_plot[:, 0], color='blue', linestyle='--', label='LGF')
    plt.plot(adam_params_plot[:, 1:], color='red')
    plt.plot(sindy_params_plot[:, 1:], color='blue', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('$a$')
    plt.title('Parameter trajectories')
    plt.legend()

    plt.subplot(2, 2, 3)
    pc1 = plt.tripcolor(
        triang,
        facecolors=adam_error,
        edgecolors='k',
        cmap='viridis',
        vmin=torch.min(adam_error),
        vmax=torch.max(adam_error)
    )
    plt.colorbar(pc1)
    plt.title("Error (ADAM)")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.subplot(2, 2, 4)
    pc2 = plt.tripcolor(
        triang,
        facecolors=sindy_error,
        edgecolors='k',
        cmap='viridis',
        vmin=torch.min(adam_error),
        vmax=torch.max(adam_error)
    )
    plt.colorbar(pc2)
    plt.title("Error (LGF)")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    fig.savefig(output_dir / f"{prefix}comparison.png", dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # directly save these mses in the same folder
    mse_file = output_dir / f"{prefix}mses.txt"
    with open(mse_file, 'w') as f:
        f.write(f"mse_adam: {mse_adam.item()}\n")
        f.write(f"mse_sindy: {mse_sindy.item()}\n")

    timing_file = output_dir / f"{prefix}timings.txt"
    with open(timing_file, 'w') as f:
        f.write(f"AppendixAdam optimization_time_sec: {optimizers['AppendixAdam']['optimization_time_sec']:.6f}\n")
        f.write(f"AppendixAdamLearned optimization_time_sec: {optimizers['AppendixAdamLearned']['optimization_time_sec']:.6f}\n")
        f.write(f"total_optimization_wall_clock_sec: {optimization_elapsed:.6f}\n")
        f.write(f"total epochs: {epochs}\n")
        f.write(f"skip_unused_evals: {int(SKIP_UNUSED_EVALS)}\n")

    return {
        "mse_adam": mse_adam.item(),
        "mse_sindy": mse_sindy.item(),
        "timings": {
            "AppendixAdam_optimization_time_sec": optimizers['AppendixAdam']['optimization_time_sec'],
            "AppendixAdamLearned_optimization_time_sec": optimizers['AppendixAdamLearned']['optimization_time_sec'],
            "total_optimization_wall_clock_sec": optimization_elapsed,
        },
    }


if __name__ == "__main__":
    config = load_config()

    run_example(threshold=1e-8, alpha=0.0, normalize_columns=True)
    # run_example(threshold=config["threshold"], alpha=config["alpha"])
