from typing import Tuple
import torch
import torch.nn as nn

def create_sindy_library(input_dim: int,
                       poly_order: int = 2,
                       include_bias: bool = True) -> nn.Module:
    """
    Constructs a SINDy-style library evaluator as a PyTorch module.

    Args:
        input_dim (int): Dimension of input a(t).
        poly_order (int): Maximum polynomial order for library.
        include_bias (bool): Whether to include a constant term.

    Returns:
        library (nn.Module): Module that maps x -> Theta(x) of shape (..., P).
    """
    class SINDyLibrary(nn.Module):
        def __init__(self, input_dim, poly_order, include_bias):
            super().__init__()
            self.input_dim = input_dim
            self.poly_order = poly_order
            self.include_bias = include_bias
            # compute number of features P for monomials up to poly_order
            # include_bias adds constant feature
            self.feature_count = 0
            if include_bias:
                self.feature_count += 1
            # Count monomials of degree 1..poly_order
            for deg in range(1, poly_order + 1):
                # combinations with repetition
                self.feature_count += int(
                    torch.combinations(torch.arange(input_dim), deg, with_replacement=True).shape[0]
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Evaluate the SINDy library on input x.
            x shape: (..., input_dim)
            returns: Theta shape (..., P)
            """
            batch_shape = x.shape[:-1]
            features = []
            if self.include_bias:
                features.append(torch.ones(*batch_shape, 1, device=x.device, dtype=x.dtype))
            # monomials
            for deg in range(1, self.poly_order + 1):
                # generate all combinations of indices for this degree
                idxs = torch.combinations(torch.arange(self.input_dim), deg, with_replacement=True)
                # compute product along each combination
                for idx in idxs:
                    # x[..., idx] has shape (..., deg)
                    term = torch.prod(x[..., idx], dim=-1, keepdim=True)
                    features.append(term)
            # concatenate along feature dimension
            return torch.cat(features, dim=-1)

    return SINDyLibrary(input_dim, poly_order, include_bias)


def make_polynomial_bump(t_grid: torch.Tensor,
                         a: float,
                         b: float,
                         p: int = 2,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate a polynomial bump test function phi(t) = C*(t-a)^p*(b-t)^p over a time grid,
    and its derivative dphi/dt(t). The function is 0 outside of (a, b).

    Args:
        t_grid   : Tensor of shape (N,) representing time points
        a, b     : float, start and end of support (a < b)
        normalize: if True, scale φ so that max|φ| = 1

    Returns:
        A tuple of tensors (phi, dphi) each of shape (N,)
    """
    t = t_grid

    # Mask for (a < t < b)
    mask = (t > a) & (t < b)

    # Compute unnormalized bump on support
    t_local = t[mask]
    tma = t_local - a
    bmt = b - t_local

    phi_local = tma**p * bmt**p
    dphi_local = (
        p * tma**(p - 1) * bmt**p -
        p * tma**p * bmt**(p - 1)
    )

    # Normalization constant (ensures max is 1)
    max_phi = torch.max(phi_local)
    phi_local = phi_local / max_phi
    dphi_local = dphi_local / max_phi

    # Fill full arrays
    phi = torch.zeros_like(t)
    dphi = torch.zeros_like(t)

    phi[mask] = phi_local
    dphi[mask] = dphi_local

    return phi, dphi

def create_test_mats(t_span: torch.Tensor,
                     width: int,
                     p: int = 5,
                     stride: int = 1,
                     include_endpoints: bool = True,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns bump test functions and their derivatives as (N,K) matrices, for use with weak form.

    Args:
        t_span: 1D tensor of time points (N,)
        width: half-width of bump (L), minimum 1, includes center point,
                so width = 3 gives 5 nonzero points (2*width - 1)
        stride: spacing between peaks of test functions set to 2*width for no overlap
        include_endpoints: whether or not the basis functions should be nonzero at t_span endpoints
    """

    center = len(t_span) - width - 1  # minimum width, maximum len(t_span) - width - 1
    a = center - width
    b = center + width

    L = 2 * width + 1
    dt = t_span[1] - t_span[0]
    if include_endpoints:
        t_span_ext = torch.cat((torch.tensor([t_span[0] - dt]), t_span, torch.tensor([t_span[-1] + dt])))
    else:
        t_span_ext = t_span

    N_ext = len(t_span_ext)

    # compute K using "valid" conv formula in index space
    # number of starting positions: floor((N_ext - L)/stride) + 1
    K = (N_ext - L) // stride + 1
    Phi = torch.zeros((len(t_span), K))
    dPhi = torch.zeros((len(t_span), K))

    for k in range(K):
        start = N_ext - L - k * stride      # start positions step left from the rightmost
        end = start + L                 # one-past the last index of the support
        a = t_span_ext[start]             # left time‐value of this bump
        b = t_span_ext[end - 1]           # right time‐value of this bump

        phi, dphi = make_polynomial_bump(t_span, a=a, b=b, p=p)
        Phi[:, k] = phi
        dPhi[:, k] = dphi

    # Phi shape: (N, K), dPhi shape: (N, K)
    return Phi, dPhi


def assemble_weak_matrices(X_data: torch.Tensor,
                           Theta: torch.Tensor,
                           t_span: torch.Tensor,
                           test_mat_kwargs: dict = {},
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        X_data: (n_t, d) matrix of states
        Theta: (n_t, P) matrix of library functions evaluated at X_data
        t_span: (n_t,) vector of time points
        test_mat_kwargs: kwargs for the basis functions created by `create_test_mats`

    Returns:
        (X_data_dPhi, Theta_Phi):
        A tuple of matrices to be used for solving for coefficients:
            X_data_dPhi: (K, d) target matrix, representing the time derivative
            Theta_Phi: (K, P) matrix multiplying the coefficients, representing library Theta
    """
    # get defaults for test_mat_kwargs
    if 'width' not in test_mat_kwargs:
        test_mat_kwargs['width'] = (len(t_span) // 25) // 2

    Phi, dPhi = create_test_mats(t_span, **test_mat_kwargs)
    dt = t_span[1] - t_span[0]

    Theta_Phi_3d = torch.bmm(Phi.unsqueeze(2), Theta.unsqueeze(1))
    X_data_dPhi_3d = torch.bmm(dPhi.unsqueeze(2), X_data.unsqueeze(1))

    Theta_Phi = -torch.trapezoid(Theta_Phi_3d, dx=dt, dim=0)  # now just K,n_states
    X_data_dPhi = torch.trapezoid(X_data_dPhi_3d, dx=dt, dim=0)  # now just K,P

    return X_data_dPhi, Theta_Phi

def assemble_strong_matrices(X_data: torch.Tensor,
                             Theta: torch.Tensor,
                             t_span: torch.Tensor):
    dt = t_span[1] - t_span[0]
    dxdt = get_derivative(dt, X_data)  # shape (n_t-2, d)
    Thetam2 = Theta[1:-1]  # shape (n_t-2, P) to match dxdt shape
    return dxdt, Thetam2


def get_derivative(dt, x: torch.Tensor):
    # assume equispaced t, with dt = 1
    # x: (..., N), return (..., N-2)
    dxdt = x[2:] - x[:-2]
    dxdt = dxdt / (2 * dt)  # central difference
    return dxdt

def create_predictor(Xi: torch.Tensor, library):
    """
    Construct a predictor function for the SINDy system.

    Xi: P,d coefficient matrix for system
    library: SINDy library function, maps (_,d) -> (_,P)

    """
    # x is d,1
    # library(x.T) is 1,P
    def pred_fun(x: torch.Tensor) -> torch.Tensor:
        Theta_row = library(x.T) # (1,P)
        dxdt_row = Theta_row @ Xi  # 1,P @ P,d = 1,d
        return dxdt_row.T  # (d,1)
    return pred_fun

if __name__ == "__main__":
    # Example usage:
    d = 10
    n_t = 100
    t_span = torch.linspace(0, 1, n_t)  # time points
    library = create_sindy_library(input_dim=d, poly_order=2)
    x = torch.randn(n_t, d) # transpose of data matrix?
    Theta = library(x)  # shape (n_t, P)

    lhs_target, rhs_mat = assemble_strong_matrices(x, Theta, t_span)
    lhs_weak_target, rhs_weak_mat = assemble_weak_matrices(x, Theta, t_span)
    # print and compare number of equations for the two methods
    print("Strong form equations:", lhs_target.shape[0])
    print("Weak form equations:", lhs_weak_target.shape[0])

    # now construct the system: (n_t-2, d) = (n_t-2, P) @ (P, d)
    # dxdt = Thetam2 @ Xi
    Xi = torch.linalg.lstsq(rhs_mat, lhs_target, rcond=1e-5).solution
    Xi_weak = torch.linalg.lstsq(rhs_weak_mat, lhs_weak_target, rcond=1e-5).solution
    print("Xi shape:", Xi.shape)  # shape (P, d)
    print("Xi_weak shape:", Xi_weak.shape)  # shape (P, d)

    pred = create_predictor(Xi, library)
    x_pred = pred(x[0:1].T) # input is (1,d) transposed to be (d,1)
    print(x_pred.shape)  # shape (d, 1)

    pred_weak = create_predictor(Xi_weak, library)
    x_pred_weak = pred_weak(x[0:1].T)
    print(x_pred_weak.shape)

    print("Xi_weak - Xi:", Xi_weak - Xi)  # should be small, but not zero
