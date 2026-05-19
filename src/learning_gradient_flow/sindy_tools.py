from __future__ import annotations
from typing import Optional, Tuple, Protocol
from dataclasses import dataclass
import warnings
import torch
import torch.nn as nn

#%% Libraries
def build_orthonormal_basis(
    Theta: torch.Tensor,
    rcond: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal basis for the columns of Theta.

    Returns:
        Theta_ortho: (n_samples, r) orthonormalized features
        transform: (n_features, r) matrix such that Theta_ortho = Theta @ transform
    """
    if Theta.ndim != 2:
        raise ValueError(f"Theta must be 2D (n_samples, n_features). Got shape {Theta.shape}.")
    n_samples, n_features = Theta.shape
    if n_samples == 0 or n_features == 0:
        return Theta.new_zeros((n_samples, 0)), Theta.new_zeros((n_features, 0))

    U, S, Vh = torch.linalg.svd(Theta, full_matrices=False)
    if S.numel() == 0:
        return Theta.new_zeros((n_samples, 0)), Theta.new_zeros((n_features, 0))

    tol = rcond * S.max()
    keep = S > tol
    if not torch.any(keep):
        warnings.warn("Orthonormalization removed all features; using identity transform instead.")
        transform = torch.eye(n_features, device=Theta.device, dtype=Theta.dtype)
        return Theta, transform

    if keep.sum() < n_features:
        warnings.warn(
            f"Orthonormalization reduced feature count from {n_features} to {keep.sum().item()}."
        )

    V = Vh.transpose(-2, -1)  # (n_features, r_full)
    transform = V[:, keep] * (1.0 / S[keep])
    Theta_ortho = Theta @ transform
    return Theta_ortho, transform


def create_sindy_library(input_dim: int,
                       poly_order: int = 2,
                       include_bias: bool = True,
                       use_ortho: bool = False,
                       ortho_rcond: float = 1e-12) -> nn.Module:
    """
    Constructs a SINDy-style library evaluator as a PyTorch module.

    Args:
        input_dim (int): Dimension of input a(t).
        poly_order (int): Maximum polynomial order for library.
        include_bias (bool): Whether to include a constant term.
        use_ortho (bool): If True, orthonormalize library features on first call and reuse.
        ortho_rcond (float): Relative cutoff for singular values in orthonormalization.

    Returns:
        library (nn.Module): Module that maps x -> Theta(x) of shape (..., P).
    """
    class SINDyLibrary(nn.Module):
        def __init__(self, input_dim, poly_order, include_bias, use_ortho, ortho_rcond):
            super().__init__()
            self.input_dim = input_dim
            self.poly_order = poly_order
            self.include_bias = include_bias
            self.use_ortho = use_ortho
            self.ortho_rcond = ortho_rcond
            self.register_buffer("_ortho_transform", torch.empty(0), persistent=False)
            self._ortho_ready = False
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

        def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
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

        def _apply_ortho(self, Theta: torch.Tensor) -> torch.Tensor:
            if not self._ortho_ready:
                Theta_flat = Theta.reshape(-1, Theta.shape[-1])
                Theta_ortho_flat, transform = build_orthonormal_basis(
                    Theta_flat, rcond=self.ortho_rcond
                )
                self._ortho_transform = transform
                self._ortho_ready = True
                self.feature_count = transform.shape[1]
                return Theta_ortho_flat.reshape(Theta.shape[:-1] + (transform.shape[1],))

            if self._ortho_transform.numel() == 0:
                return Theta.new_zeros(Theta.shape[:-1] + (0,))

            Theta_flat = Theta.reshape(-1, Theta.shape[-1])
            Theta_ortho_flat = Theta_flat @ self._ortho_transform
            return Theta_ortho_flat.reshape(
                Theta.shape[:-1] + (self._ortho_transform.shape[1],)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            Theta = self._compute_features(x)
            if not self.use_ortho:
                return Theta
            return self._apply_ortho(Theta)

    return SINDyLibrary(input_dim, poly_order, include_bias, use_ortho, ortho_rcond)

def create_predictor(Xi: torch.Tensor, library):
    """
    Construct a predictor function for the SINDy system.

    Xi: P,d coefficient matrix for system
    library: SINDy library function, maps (_,d) -> (_,P)

    """
    # x is d,1
    # library(x.T) is 1,P
    def pred_fun(x: torch.Tensor) -> torch.Tensor:
        Theta_row = library(x.T)  # (1,P)
        dxdt_row = Theta_row @ Xi  # 1,P @ P,d = 1,d
        return dxdt_row.T  # (d,1)
    return pred_fun

#%% Differentiation / Matrix Assembly
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

@dataclass
class TestFunctionParams:
    """Parameters for creating test functions."""
    width: Optional[int] = None
    p: int = 5
    stride: int = 1
    include_endpoints: bool = True

def assemble_weak_matrices(X_data: torch.Tensor,
                           Theta: torch.Tensor,
                           t_span: torch.Tensor,
                           test_func_params: Optional[TestFunctionParams] = None,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        X_data: (n_t, d) matrix of states
        Theta: (n_t, P) matrix of library functions evaluated at X_data
        t_span: (n_t,) vector of time points
        test_func_params: TestFunctionParams object for the basis functions created by `create_test_mats`

    Returns:
        (X_data_dPhi, Theta_Phi):
        A tuple of matrices to be used for solving for coefficients:
            X_data_dPhi: (K, d) target matrix, representing the time derivative
            Theta_Phi: (K, P) matrix multiplying the coefficients, representing library Theta
    """
    if test_func_params is None:
        test_func_params = TestFunctionParams()
    if test_func_params.width is None:
        width = max(1, (len(t_span) // 25) // 2)
        test_func_params.width = width

    Phi, dPhi = create_test_mats(t_span,
                                 width=test_func_params.width,
                                 p=test_func_params.p,
                                 stride=test_func_params.stride,
                                 include_endpoints=test_func_params.include_endpoints)
    dt = t_span[1] - t_span[0]

    Theta_Phi_3d = torch.bmm(Phi.unsqueeze(2), Theta.unsqueeze(1))
    X_data_dPhi_3d = torch.bmm(dPhi.unsqueeze(2), X_data.unsqueeze(1))

    Theta_Phi = -torch.trapezoid(Theta_Phi_3d, dx=dt, dim=0)  # now just K,n_states
    X_data_dPhi = torch.trapezoid(X_data_dPhi_3d, dx=dt, dim=0)  # now just K,P

    return X_data_dPhi, Theta_Phi

def assemble_strong_matrices(X_data: torch.Tensor,
                             Theta: torch.Tensor,
                             t_span: torch.Tensor,
                             fd_order: int = 2):
    dt = t_span[1] - t_span[0]
    # dxdt, Thetam2 = get_derivative(dt, X_data, Theta)  # shape (n_t-2, d)
    # assume equispaced t, with dt = 1
    # x: (..., N), return (..., N-2)
    if fd_order == 1:
        dxdt = X_data[1:] - X_data[:-1]
        dxdt = dxdt / dt  # forward difference
        Thetam2 = Theta[:-1]  # shape (n_t-1, P) to match dxdt shape
    elif fd_order == 2:
        dxdt = X_data[2:] - X_data[:-2]
        dxdt = dxdt / (2 * dt)  # central difference
        Thetam2 = Theta[1:-1]  # shape (n_t-2, P) to match dxdt shape
    elif fd_order == 4:
        # 4th order central difference
        dxdt = (-X_data[4:] + 8 * X_data[3:-1] - 8 * X_data[1:-3] + X_data[:-4]) / (12 * dt)
        Thetam2 = Theta[2:-2]  # shape (n_t-4, P) to match dxdt shape
    else:
        raise ValueError(f"fd_order {fd_order} not supported")
    return dxdt, Thetam2


#%% Linear system solving
@dataclass
class BaseSolverParams:
    """Base class for solver parameters, corresponds to BaseSolver"""
    pass

@dataclass
class DenseSolverParams(BaseSolverParams):
    """Collected parameters to be passed to pytorch.linalg.lstsq.

    Args:
        rcond: optional cutoff for the least-squares solver.
        driver: optional driver for the least-squares solver.
    """
    rcond: Optional[float] = None
    driver: Optional[str] = None

class BaseSolver(Protocol):
    """Protocol that defines the expected signature for solver functions."""
    def __call__(self, rhs_mat: torch.Tensor, lhs_target: torch.Tensor,
                 BaseSolverParams: BaseSolverParams) -> torch.Tensor:
        pass

def dense_solver(rhs_mat: torch.Tensor,
                 lhs_target: torch.Tensor,
                 dense_solver_params: DenseSolverParams = None
                 ) -> torch.Tensor:
    """Standard least-squares solve via PyTorch."""
    if dense_solver_params is None:
        dense_solver_params = DenseSolverParams()
    return torch.linalg.lstsq(rhs_mat, lhs_target,
                              rcond=dense_solver_params.rcond,
                              driver=dense_solver_params.driver).solution

#%% Overarching config
@dataclass
class SINDyParams:
    """Configuration for SINDy library, system construction, and linear solver."""
    poly_order: int = 1
    include_bias: bool = True
    truncation_rank: Optional[int] = None
    method: str = 'strong'
    test_func_params: Optional[TestFunctionParams] = None
    solver_fn: BaseSolver = dense_solver
    solver_params: Optional[BaseSolverParams] = None
    fd_order: Optional[int] = 2
    use_ortho: bool = False

# %% New STLSQ solver and params
@dataclass
class STLSQParams:
    """
    Minimal STLSQ params aligned with PySINDy defaults.

    threshold: hard-threshold applied to coefficients after each refit
    alpha: ridge regularization strength (alpha=0 reduces to plain least squares)
    max_iter: maximum STLSQ iterations
    normalize_columns: if True, normalize columns of rhs_mat before regression (and unscale at end)
    unbias: if True, do a final unregularized refit on the selected support
    """
    threshold: float = 0.1
    alpha: float = 0.05
    max_iter: int = 20
    normalize_columns: bool = False
    unbias: bool = True


def stlsq_sparse_solver(
    rhs_mat: torch.Tensor,                 # Theta / X: (n_samples, n_features)
    lhs_target: torch.Tensor,              # y:         (n_samples, n_targets) or (n_samples,)
    params: Optional[STLSQParams] = None
) -> torch.Tensor:
    r"""
    Solve lhs_target ≈ rhs_mat @ Xi with sparse Xi using sequentially-thresholded least squares (STLSQ).

    Iteration:
      1) Ridge regression on current support
      2) Hard threshold coefficients by magnitude
      3) Update support; stop when support does not change

    Returns:
      Xi of shape (n_features, n_targets)
    """
    if params is None:
        params = STLSQParams()

    if params.threshold < 0:
        raise ValueError("threshold cannot be negative")
    if params.alpha < 0:
        raise ValueError("alpha cannot be negative")
    if params.max_iter <= 0:
        raise ValueError("max_iter must be positive")

    X = rhs_mat
    y = lhs_target

    if y.ndim == 1:
        y = y.unsqueeze(1)
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError("rhs_mat must be 2D and lhs_target must be 1D or 2D")

    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise ValueError(f"lhs_target has {y.shape[0]} rows but rhs_mat has {n_samples}")

    device, dtype = X.device, X.dtype
    n_targets = y.shape[1]

    # Special case: no thresholding, no ridge -> basic least squares on original X
    if params.threshold == 0.0 and params.alpha == 0.0:
        return torch.linalg.lstsq(X, y).solution

    # ---- Optional column normalization ----
    if params.normalize_columns:
        col_norms = torch.linalg.norm(X, ord=2, dim=0, keepdim=True)  # (1, n_features)
        col_norms = torch.where(col_norms == 0, torch.ones_like(col_norms), col_norms)
        Xn = X / col_norms
    else:
        col_norms = torch.ones((1, n_features), device=device, dtype=dtype)
        Xn = X

    # ---- Small helpers ----
    def ridge_solve(Xa: torch.Tensor, ya: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Solve argmin_w ||Xa w - ya||_2^2 + alpha ||w||_2^2

        Xa: (n, p), ya: (n,) or (n, k)
        Returns w: (p,) or (p, k)
        """
        p = Xa.shape[1]
        if p == 0:
            return torch.zeros((0,) if ya.ndim == 1 else (0, ya.shape[1]), device=device, dtype=dtype)

        if alpha == 0.0:
            # Plain least squares
            return torch.linalg.lstsq(Xa, ya).solution

        n = Xa.shape[0]
        use_dual = p > n
        if use_dual:
            warnings.warn("Using dual ridge solve because more features than data.")
            A = Xa @ Xa.T
            A = A + alpha * torch.eye(n, device=device, dtype=dtype)
            z = torch.linalg.solve(A, ya)
            return Xa.T @ z

        A = Xa.T @ Xa
        A = A + alpha * torch.eye(p, device=device, dtype=dtype)
        b = Xa.T @ ya
        return torch.linalg.solve(A, b)

    def ls_solve(Xa: torch.Tensor, ya: torch.Tensor) -> torch.Tensor:
        """Unregularized least squares with a simple fallback if lstsq fails."""
        # try:
        return torch.linalg.lstsq(Xa, ya).solution
        # except RuntimeError:
        #     # Fallback: pseudoinverse
        #     return torch.linalg.pinv(Xa) @ ya

    # ---- STLSQ loop: support-stability stopping ----
    support = torch.ones((n_features, n_targets), device=device, dtype=torch.bool)
    Xi = torch.zeros((n_features, n_targets), device=device, dtype=dtype)

    for _ in range(params.max_iter):
        # Refit on current support (ridge)
        Xi_new = torch.zeros_like(Xi)
        for j in range(n_targets):
            active = support[:, j]
            if not torch.any(active):
                continue
            w_act = ridge_solve(Xn[:, active], y[:, j], params.alpha)  # (p,)
            Xi_new[active, j] = w_act

        # Threshold -> new support
        new_support = Xi_new.abs() >= params.threshold
        Xi_new = Xi_new * new_support.to(dtype=dtype)

        # Stop if support unchanged
        if torch.equal(new_support, support):
            Xi = Xi_new
            break

        support = new_support
        Xi = Xi_new

    # ---- Optional unbias: final unregularized refit on the selected support ----
    if params.unbias:
        Xi_unb = torch.zeros_like(Xi)
        for j in range(n_targets):
            active = support[:, j]
            if not torch.any(active):
                continue
            w_act = ls_solve(Xn[:, active], y[:, j])  # (p,)
            Xi_unb[active, j] = w_act
        Xi = Xi_unb

    # ---- Unnormalize coefficients ----
    if params.normalize_columns:
        Xi = Xi / col_norms.T  # (n_features, 1) broadcast over targets

    return Xi
