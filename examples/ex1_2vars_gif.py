import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import nn

from learning_gradient_flow import gradient_flow_optimizer, sindy_tools
from example_config import load_config


SKIP_UNUSED_EVALS = False
DTYPE = torch.float32


def run_example(
    threshold: float,
    alpha: float,
    normalize_columns: bool = True,
    unbias: bool = True,
    output_dir: str | None = None,
):
    torch.manual_seed(0)
    np.random.seed(0)
    plt.close("all")
    plt.style.use("default")
    torch.set_default_dtype(DTYPE)

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    # true material parameters
    Aa1 = 2.0
    Aa2 = 1.0
    A = torch.tensor([Aa1, Aa2], dtype=DTYPE)

    # number of basis functions
    N = 30

    # spatial integration grid
    pts = 250
    X = torch.linspace(0, 1, pts + 1)
    dX = X[1] - X[0]
    X += dX / 2
    X = X[:-1]

    # time of simulation and time grid
    T = 2.0
    t_steps = 500
    t_grid = torch.linspace(0, T, t_steps)
    dt = t_grid[1] - t_grid[0]

    # basis functions
    F = torch.stack([torch.sin((i + 1) * np.pi * X) for i in range(N)]).T
    dF = torch.stack(
        [np.pi * (i + 1) * torch.cos((i + 1) * np.pi * X) for i in range(N)]
    ).T

    # mass matrix
    M = dX * F.T @ F

    def stiffness(a):
        a1 = a[0]
        a2 = a[1]
        kappa = a1 * torch.ones(pts)
        kappa[X > 0.5] = a2
        return torch.einsum("x,xi,xj->ij", kappa, dF, dF)

    # forcing function
    f0 = 1000
    force = torch.zeros((N, t_steps))
    force[0, :] = 0 * f0 * torch.sin(np.pi * t_grid)
    force[1, :] = f0 * torch.sin(2 * np.pi * t_grid)

    def solve(a):
        K = stiffness(a)
        mat = torch.linalg.inv(M / dt + K)
        theta = torch.zeros((N, t_steps))
        for i in range(t_steps - 1):
            theta[:, i + 1] = mat @ (force[:, i + 1] + (M / dt) @ theta[:, i])
        return theta

    def raw_loss(a):
        theta = solve(a)
        u = F @ theta
        return dX * dt * torch.sum((u - v) ** 2)

    theta = solve(A)
    v = F @ theta

    # contour of the loss surface
    ptsa = 35
    a_grid = torch.linspace(0.1, torch.max(A) + 2.25, ptsa)
    A1, A2 = torch.meshgrid(a_grid, a_grid, indexing="ij")
    z = torch.zeros((ptsa, ptsa))
    for i in range(ptsa):
        for j in range(ptsa):
            z[i, j] = raw_loss(torch.tensor([A1[i, j], A2[i, j]], dtype=DTYPE))

    a0init = torch.tensor([0.5, 4.05], dtype=DTYPE)

    class Parameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(a0init.clone(), requires_grad=True)

        def forward(self):
            return self.a

        def loss(self):
            a = self.forward()
            theta_local = solve(a)
            u = F @ theta_local
            return torch.log(dX * dt * torch.sum((u - v) ** 2))

    epochs = 750
    lr = 1e-2
    history_size = 10
    retrain_interval = 30

    solver_fn = sindy_tools.stlsq_sparse_solver
    solver_params = sindy_tools.STLSQParams(
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
        unbias=unbias,
        max_iter=3,
    )

    sindy_params = sindy_tools.SINDyParams(
        poly_order=1,
        include_bias=False,
        method="strong",
        fd_order=2,
        solver_fn=solver_fn,
        solver_params=solver_params,
    )

    model = Parameters()
    optimizer = gradient_flow_optimizer.SINDyFlow(
        model.parameters(),
        backup_optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        dt=lr,
        ode_solver_options={"method": "rk4", "options": {"step_size": lr}},
        history_size=history_size,
        retrain_interval=retrain_interval,
        sindy_params=sindy_params,
        skip_unused_evals=SKIP_UNUSED_EVALS,
    )

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / "ex1_2vars_gif"
    else:
        output_dir = Path(output_dir)

    prefix = (
        f"thr{threshold}_alpha{alpha}_norm{int(normalize_columns)}_"
        f"unbias{int(unbias)}_p{sindy_params.poly_order}_"
        f"K{history_size}_M{retrain_interval}_fd{sindy_params.fd_order}"
    )
    frame_dir = output_dir / prefix / "gif_frames"
    os.makedirs(frame_dir, exist_ok=True)

    def create_closure():
        def closure():
            optimizer.zero_grad()
            loss_val = model.loss()
            loss_val.backward()
            return loss_val

        return closure

    states = [
        {
            "epoch": 0,
            "params": model.a.detach().clone(),
            "phase": "start",
            "status_text": "Initial guess",
            "loss": float(torch.exp(model.loss().detach()).item()),
        }
    ]

    for _ in range(epochs):
        epoch_before = optimizer.state["epoch"]
        history_before = optimizer.state["history_count"]
        dynamics_before = getattr(optimizer, "dynamics", None)

        is_rebuild_step = history_before >= history_size and dynamics_before is None
        if history_before < history_size:
            phase = "optimization"
            status_text = "Optimization iterations"
        else:
            phase = "learned"
            if is_rebuild_step:
                status_text = r"Build $\hat{f}$ and apply learned gradient flow"
            else:
                status_text = "Learned gradient flow"

        optimizer.step(create_closure())
        plotted_loss = float(torch.exp(model.loss().detach()).item())

        states.append(
            {
                "epoch": optimizer.state["epoch"],
                "params": model.a.detach().clone(),
                "phase": phase,
                "status_text": status_text,
                "loss": plotted_loss,
                "rebuild_epoch": epoch_before if is_rebuild_step else None,
            }
        )

    all_points = torch.stack([entry["params"] for entry in states]).numpy()
    rebuild_epochs = [
        entry["rebuild_epoch"] for entry in states if entry.get("rebuild_epoch") is not None
    ]

    x_min = min(np.min(all_points[:, 0]), Aa1) - 0.15
    x_max = max(np.max(all_points[:, 0]), Aa1) + 0.15
    y_min = min(np.min(all_points[:, 1]), Aa2) - 0.2
    y_max = max(np.max(all_points[:, 1]), Aa2) + 0.2

    optimization_color = "#D86A5E"
    learned_color = "#6FA34F"
    rebuild_color = "#3F3A34"
    current_color = "#222222"
    true_param_color = "#111111"

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color=true_param_color,
            markersize=10,
            markeredgewidth=2,
            label="true parameters",
        ),
        Line2D(
            [0],
            [0],
            color=optimization_color,
            linewidth=4,
            label="Optimization iterations",
        ),
        Line2D(
            [0],
            [0],
            color=learned_color,
            linewidth=4,
            linestyle="--",
            label="Learned gradient flow",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=rebuild_color,
            markeredgewidth=1.8,
            markersize=8,
            label=r"Build $\hat{f}$",
        ),
    ]

    def render_frame(frame_idx: int, pause_suffix: int = 0):
        fig, ax_contour = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)

        cp = ax_contour.contourf(
            A1.numpy(),
            A2.numpy(),
            np.log(z.numpy()),
            levels=24,
            cmap="Greys",
            alpha=0.82,
        )
        ax_contour.contour(
            A1.numpy(),
            A2.numpy(),
            np.log(z.numpy()),
            levels=8,
            colors="#B8B8B8",
            linewidths=0.45,
            alpha=0.8,
        )
        fig.colorbar(cp, ax=ax_contour, label=r"$\log(z)$")

        ax_contour.scatter(
            Aa1,
            Aa2,
            marker="x",
            s=160,
            color=true_param_color,
            linewidths=3,
        )

        for seg_idx in range(1, frame_idx + 1):
            xseg = all_points[seg_idx - 1 : seg_idx + 1, 0]
            yseg = all_points[seg_idx - 1 : seg_idx + 1, 1]
            color = (
                learned_color
                if states[seg_idx]["phase"] == "learned"
                else optimization_color
            )
            alpha = 0.98 if seg_idx == frame_idx else 0.8
            linewidth = 4.2 if seg_idx == frame_idx else 3.1
            linestyle = "--" if states[seg_idx]["phase"] == "learned" else "-"
            ax_contour.plot(
                xseg,
                yseg,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )

        visible_rebuilds = [epoch for epoch in rebuild_epochs if epoch <= frame_idx - 1]
        if visible_rebuilds:
            rebuild_points = np.array([all_points[epoch + 1] for epoch in visible_rebuilds])
            ax_contour.scatter(
                rebuild_points[:, 0],
                rebuild_points[:, 1],
                s=72,
                facecolors="white",
                edgecolors=rebuild_color,
                linewidths=1.5,
                zorder=6,
            )

        current_point = all_points[frame_idx]
        if frame_idx > 0:
            prev_point = all_points[frame_idx - 1]
            direction = current_point - prev_point
            arrow_start = current_point - 0.45 * direction
            ax_contour.annotate(
                "",
                xy=current_point,
                xytext=arrow_start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": current_color,
                    "linewidth": 2.0,
                    "mutation_scale": 15,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=7,
            )
        else:
            ax_contour.scatter(
                current_point[0],
                current_point[1],
                s=72,
                color=current_color,
                marker="D",
                edgecolors="white",
                linewidths=0.9,
                zorder=7,
            )

        ax_contour.set_xlim(x_min, x_max)
        ax_contour.set_ylim(y_min, y_max)
        ax_contour.set_xlabel(r"$a_1$")
        ax_contour.set_ylabel(r"$a_2$")
        ax_contour.set_title("LGF Trajectory")
        ax_contour.legend(handles=legend_handles, loc="lower left", framealpha=0.95)

        frame_name = frame_dir / f"frame_{frame_idx:04d}"
        if pause_suffix:
            frame_name = frame_dir / f"frame_{frame_idx:04d}_pause{pause_suffix:02d}"
        fig.savefig(frame_name.with_suffix(".png"), dpi=300)
        plt.close(fig)

    for frame_idx in range(len(states)):
        render_frame(frame_idx)
        if states[frame_idx].get("rebuild_epoch") is not None:
            for pause_idx in range(1, 5):
                render_frame(frame_idx, pause_suffix=pause_idx)

    manifest_path = frame_dir / "frame_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(f"Saved {len(list(frame_dir.glob('*.png')))} PNG frames at dpi=300.\n")
        handle.write(f"History size: {history_size}\n")
        handle.write(f"Retrain interval: {retrain_interval}\n")
        handle.write("Rebuild epochs: " + ", ".join(str(epoch) for epoch in rebuild_epochs) + "\n")
        handle.write("Phase colors:\n")
        handle.write(f"  optimization iterations: {optimization_color}\n")
        handle.write(f"  learned gradient flow: {learned_color}\n")
        handle.write(f"  rebuild markers: {rebuild_color}\n")

    return {
        "frame_dir": frame_dir,
        "frame_count": len(list(frame_dir.glob('*.png'))),
        "rebuild_epochs": rebuild_epochs,
    }


if __name__ == "__main__":
    config = load_config()
    run_example(threshold=config["threshold"], alpha=config["alpha"])
