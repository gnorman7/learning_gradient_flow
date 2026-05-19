import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils import vector_to_parameters
import matplotlib.pyplot as plt

from example_config import load_config


def run_plot(
    threshold: float,
    alpha: float,
    normalize_columns: bool = True,
    unbias: bool = True,
    two_pane: bool = False,
    use_random: bool = False,
    M: int = 50,
    output_dir: Optional[str] = None,
):
    torch.manual_seed(0)
    np.random.seed(0)

    retrain_interval = M

    # 1d grid
    pts = 50
    grid = torch.linspace(0, 1, pts)
    dx = grid[1] - grid[0]
    grid += dx / 2
    grid = grid[:-1]

    x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")

    # neural network input
    X = torch.zeros((pts - 1) ** 3, 3)
    X[:, 0] = torch.reshape(x, (1, (pts - 1) ** 3))
    X[:, 1] = torch.reshape(y, (1, (pts - 1) ** 3))
    X[:, 2] = torch.reshape(z, (1, (pts - 1) ** 3))

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

    if output_dir is None:
        example_dir = "ex5_deep_random" if use_random else "ex5_deep"
        output_dir = Path(__file__).resolve().parent / "outputs" / example_dir
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    batch_tag = "batch5000_" if use_random else ""
    prefix = (
        f"thr{threshold}_alpha{alpha}_norm{int(normalize_columns)}_"
        f"unbias{int(unbias)}_M{retrain_interval}_{batch_tag}"
    )

    adam_params = torch.tensor(
        np.loadtxt(output_dir / f"{prefix}adam_params.txt"), dtype=torch.float32
    )
    sindy_params = torch.tensor(
        np.loadtxt(output_dir / f"{prefix}sindy_params.txt"), dtype=torch.float32
    )
    adam_losses = torch.tensor(
        np.loadtxt(output_dir / f"{prefix}adam_losses.txt"), dtype=torch.float32
    )
    sindy_losses = torch.tensor(
        np.loadtxt(output_dir / f"{prefix}sindy_losses.txt"), dtype=torch.float32
    )

    plt.close("all")
    plt.style.use("default")

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # random subset of parameters
    k = 15
    torch.manual_seed(0)
    indices = torch.randperm(adam_params.shape[1])[:k]
    adam_params_plot = adam_params[:, indices]
    sindy_params_plot = sindy_params[:, indices]

    if two_pane:
        fig = plt.figure(figsize=(12, 5))
        plt.subplots_adjust(wspace=0.35, hspace=0.4)
        ax1 = fig.add_subplot(1, 2, 1)
    else:
        fig = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.5)
        ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(adam_losses, color="red", label="ADAM")
    ax1.plot(sindy_losses, color="blue", linestyle="--", label="LGF")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("$z$")
    ax1.set_title("Comparing optimizers")
    ax1.legend()

    if two_pane:
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(adam_params_plot[:, 0], color="red", label="ADAM")
    ax2.plot(sindy_params_plot[:, 0], color="blue", linestyle="--", label="LGF")
    ax2.plot(adam_params_plot[:, 1:], color="red")
    ax2.plot(sindy_params_plot[:, 1:], color="blue", linestyle="--")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("$a$")
    ax2.set_title("Parameter trajectories")
    ax2.legend()

    if not two_pane:
        # displacement magnitude (ADAM)
        net = displacement()
        vector_to_parameters(adam_params[-1].clone(), net.parameters())
        uadam = net(X).detach()
        mag_adam = (uadam[:, 0] ** 2 + uadam[:, 1] ** 2 + uadam[:, 2] ** 2) ** 0.5

        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        sc3 = ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=mag_adam, cmap="viridis")
        ax3.set_xlabel("$x_1$")
        ax3.set_ylabel("$x_2$")
        ax3.set_zlabel("$x_3$")
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.set_title("Displacement magnitude (ADAM)")
        fig.colorbar(sc3, ax=ax3, pad=0.25)

        # displacement magnitude (LGF)
        net = displacement()
        vector_to_parameters(sindy_params[-1].clone(), net.parameters())
        usindy = net(X).detach()
        mag_sindy = (usindy[:, 0] ** 2 + usindy[:, 1] ** 2 + usindy[:, 2] ** 2) ** 0.5

        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        sc4 = ax4.scatter(X[:, 0], X[:, 1], X[:, 2], c=mag_sindy, cmap="viridis")
        ax4.set_xlabel("$x_1$")
        ax4.set_ylabel("$x_2$")
        ax4.set_zlabel("$x_3$")
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_zticks([])
        ax4.set_title("Displacement magnitude (LGF)")
        fig.colorbar(sc4, ax=ax4, pad=0.25)

    # fig.suptitle(f"Solver: {prefix[:-1]}", fontsize=18)

    pane_tag = "2pane" if two_pane else "4pane"
    fig.savefig(output_dir / f"{prefix}dense_{pane_tag}.png", dpi=300)

    return {
        "adam_losses": adam_losses,
        "sindy_losses": sindy_losses,
    }


if __name__ == "__main__":
    config = load_config()
    run_plot(
        threshold=config["threshold"],
        alpha=config["alpha"],
        normalize_columns=True,
        unbias=True,
        two_pane=True,
        use_random=False,
        M=70
    )
