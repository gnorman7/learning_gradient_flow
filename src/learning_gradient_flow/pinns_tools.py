import torch
import torch.nn as nn
from typing import Optional, Callable
from collections import OrderedDict
import numpy as np


class DNN(nn.Module):
    def __init__(self, layers: list[int], activation: nn.Module = nn.Tanh()):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        activation = activation.lower() if isinstance(activation, str) else activation

        # set up layer order dict
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = activation

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = nn.Sequential(layerDict)

        # # Xavier Normal Initialization
        # for layer in self.layers:
        #     if (type(layer) == nn.modules.linear.Linear):
        #         # torch.nn.init.xavier_uniform_(layer.weight)
        #         nn.init.xavier_normal_(layer.weight)
        #         print('Xavier Normal Initialization!')
        # Can just use the default initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out

class SineOmega(nn.Module):
    """Sine activation function: sin(omega_0 * x)."""

    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class Siren(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int,
                 out_features: int, outermost_linear=True,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        """Siren implementation with linear layers exposed for LoRA hypernetwork."""
        super().__init__()

        layers_list = []
        current_dim = in_features

        # First layer
        first_linear = nn.Linear(current_dim, hidden_features, bias=True)
        self._initialize_weights(first_linear, is_first=True,
                                 omega_0=first_omega_0, in_features_for_init=current_dim)
        layers_list.append(first_linear)
        layers_list.append(SineOmega(omega_0=first_omega_0))
        current_dim = hidden_features

        # Hidden layers
        for _ in range(hidden_layers):
            hidden_linear = nn.Linear(current_dim, hidden_features, bias=True)
            self._initialize_weights(hidden_linear, is_first=False,
                                     omega_0=hidden_omega_0, in_features_for_init=current_dim)
            layers_list.append(hidden_linear)
            layers_list.append(SineOmega(omega_0=hidden_omega_0))
            # current_dim remains hidden_features for subsequent hidden layers

        # Output layer
        final_linear = nn.Linear(current_dim, out_features, bias=True)
        self._initialize_weights(final_linear, is_first=False,
                                 omega_0=hidden_omega_0, in_features_for_init=current_dim)
        layers_list.append(final_linear)
        if not outermost_linear:
            layers_list.append(SineOmega(omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers_list)

    def _initialize_weights(self, linear_layer: nn.Linear, is_first: bool,
                            omega_0: float, in_features_for_init: int):
        """Initializes weights for a linear layer in this SIREN network.

        Args:
            linear_layer: The nn.Linear module to initialize.
            is_first: True if this is the first linear layer in the SIREN.
            omega_0: sin(omega_0 * x) is the sine activation function.
            in_features_for_init: The number of input features, for scaling.
        """
        with torch.no_grad():
            if is_first:
                linear_layer.weight.uniform_(-1 / in_features_for_init,
                                             1 / in_features_for_init)
            else:
                linear_layer.weight.uniform_(-np.sqrt(6 / in_features_for_init) / omega_0,
                                             np.sqrt(6 / in_features_for_init) / omega_0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SIREN network."""
        output = self.net(coords)
        return output


def get_big_u_and_ut(model: Callable[[torch.Tensor], torch.Tensor],
                     xt_f: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        model: computes u at collocation points, using torch functions
        xt_f: tensor of collocation points, (N_f, 2)

    Returns:
        A tuple with two tensors:
            big_u: tensor of u, u_x, u_xx, etc at collocation points, (N_f, 2)
            u_t: tensor of u_t at collocation points, (N_f, 1), or u_tt if is_2nd_time

    Note, this is not part of the EqDiscoveryModel class, as it can be equally
    used by the true dynamics (PINN) by prescribing model as the true dynamics.
    """
    assert xt_f.requires_grad
    u = model(xt_f)
    u_xt = torch.autograd.grad(u, xt_f,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = u_xt[:, 0:1]
    u_t = u_xt[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_f,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    big_u = torch.cat([u, u_x, u_xx], dim=1)
    return big_u, u_t

class PINNsModel(nn.Module):
    def __init__(self, u_dnn: nn.Module, N_dnn: Callable[[torch.Tensor], torch.Tensor], ic_bc: bool=True):
        """
        Args:
            u_dnn: computes u(x,t,...)
            N_dnn: computes N(u, u_x, u_xx)
        """
        super(PINNsModel, self).__init__()
        self.u_net = u_dnn
        self.N_dnn = N_dnn
        if ic_bc:
            self.u_dnn = self.u_dnn_fn
        else:
            self.u_dnn = self.u_net


    def u_dnn_fn(self, xt: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet 0 BCs at -1, 1, and the IC u(x,0) = -sin(pi * x)"""
        decay_rate = torch.tensor(5.0)
        u_bc = lambda x: -torch.sin(np.pi * x)
        # This function is 1 at t=0 and 0 at t=1.
        ic_weight_fcn = lambda t: (torch.exp(-decay_rate * t) - torch.exp(-decay_rate)) / (1 - torch.exp(-decay_rate))
        ic_weight = ic_weight_fcn(xt[:, 1:2])
        u_bc_vals = u_bc(xt[:, 0:1])
        u = self.u_net(xt)
        out = ic_weight * u_bc_vals + (1 - ic_weight) * u * u_bc_vals
        return out

    def get_residual(self, xt_f: torch.Tensor):
        """Returns [N_f, 1] tensor of residuals at collocation points.

        Args:
            xt_f: Collocation points [x, t] of shape (N_f, 2) which are passed to u_dnn,
              and any more appended to N_dnn input
        Returns:
            residual: tensor of residuals at collocation points of shape (N_f, 1), u_t - N(...)
        """
        big_u, u_t = get_big_u_and_ut(self.u_dnn, xt_f)

        N_eval = self.N_dnn(big_u)
        assert N_eval.shape == u_t.shape
        residual = u_t - N_eval
        return residual

    def mse(self, xt_train: torch.Tensor, u_train: torch.Tensor):
        """u_train should be [N_u, 1]"""
        u_pred = self.u_dnn(xt_train)
        # make sure the shapes are correct
        assert u_pred.shape == u_train.shape
        return torch.mean((u_pred - u_train)**2)

    def u_on_meshgrid(self, T: torch.Tensor, X: torch.Tensor, Y: torch.Tensor = None):
        """Computes u on a meshgrid of T, X"""
        if Y is not None:
            xt = torch.stack((X.flatten(), Y.flatten(), T.flatten()), dim=1)
        else:
            xt = torch.stack((X.flatten(), T.flatten()), dim=1)
        u = self.u_dnn(xt)
        return u.reshape(T.shape)

    def forward(self, big_u: torch.Tensor):
        return self.N_dnn(big_u)


class saPINNsModel(PINNsModel):
    def __init__(self, u_dnn: nn.Module, N_dnn: Callable[[torch.Tensor], torch.Tensor],
                 N_f: int, ic_bc: bool = True):
        """
        Args:
            u_dnn: computes u(x,t,...)
            N_dnn: computes N(u, u_x, u_xx)
        """
        super().__init__(u_dnn, N_dnn, ic_bc)
        self.N_f = N_f
        self.collocation_weights = nn.Parameter(torch.ones(N_f, 1), requires_grad=True)

    def get_residual(self, xt_f: torch.Tensor):
        # call the parent method to get the residuals
        residual = super().get_residual(xt_f)
        # weight the residuals by the collocation weights
        weighted_residual = self.collocation_weights * residual
        return weighted_residual

    def invert_col_grads(self):
        # flips the sign of the grad components on the collocation weights
        self.collocation_weights.grad *= -1.0


