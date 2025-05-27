from typing import Optional, Any, Tuple, Optional, Protocol
from dataclasses import dataclass
import torch
import torch.nn as nn

#%% Libraries
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
        test_mat_kwargs: kwargs for the basis functions created by `create_test_mats`

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

@dataclass
class SparseSolverParams(BaseSolverParams):
    """Collected parameters for the sparse solver.

    Args:
        threshold: threshold for culling small coefficients on each iteration.
        max_iter: maximum number of iterations for the STLS algorithm.
        rcond: optional cutoff for the least-squares solver.
        driver: optional driver for the least-squares solver (see PyTorch docs).
        convergence_tol: tolerance change in Xi for convergence.
        normalize_columns: whether to normalize columns of rhs_mat before solving.
    """
    threshold: float = 0.01
    max_iter: int = 20
    rcond: Optional[float] = None
    driver: Optional[str] = None
    convergence_tol: float = 1e-7
    normalize_columns: bool = False

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

def stls_sparse_solver(rhs_mat: torch.Tensor,
                       lhs_target: torch.Tensor,
                       sparse_solver_params: SparseSolverParams = None
                       ) -> torch.Tensor:
    r"""

    The core idea of STLS is to iteratively:
    1. Solve the least-squares problem (Xi_dense = argmin ||lhs_mat - rhs_mat @ Xi||_2^2).
    2. Threshold small coefficients in Xi_dense to zero.
    3. Identify the set of non-zero ("active") coefficients.
    4. Re-solve the least-squares problem using only the columns of rhs_mat corresponding
       to these active coefficients.
    This process is repeated until the coefficient matrix Xi converges or max_iter is reached.

    Args:
        rhs_mat: The matrix multiplying the coefficients, e.g. library functions.
        lhs_target: The target matrix, e.g. time derivative of the state.
        SparseSolverParams: Parameters for the sparse solver, including e.g. threshold.
    Returns:
        The sparse coefficient matrix Xi such that lhs_mat \approx rhs_mat @ Xi.
    """
    if sparse_solver_params is None:
        sparse_solver_params = SparseSolverParams()

    n_features = rhs_mat.shape[1]
    n_targets = lhs_target.shape[1]
    device = rhs_mat.device
    dtype = rhs_mat.dtype

    # Optional column normalization for rhs_mat
    current_rhs_mat = rhs_mat
    if sparse_solver_params.normalize_columns:
        col_norms = torch.linalg.norm(rhs_mat, ord=2, dim=0, keepdim=True)  # Shape (1, n_features)
        # Avoid division by zero for zero columns (norm is 0).
        # If norm is zero, column is zero; division by 1 keeps it zero.
        col_norms[col_norms == 0] = 1.0
        current_rhs_mat = rhs_mat / col_norms
    else:
        # Create a dummy col_norms for consistent un-normalization step (division by 1)
        col_norms = torch.ones((1, n_features), device=device, dtype=dtype)

    # Initial guess for Xi using all features (dense solve)
    # Solves current_rhs_mat @ Xi = lhs_target

    xi = torch.linalg.lstsq(
        current_rhs_mat,
        lhs_target,
        rcond=sparse_solver_params.rcond,
        driver=sparse_solver_params.driver
    ).solution

    # STLS iteration
    for _iteration in range(sparse_solver_params.max_iter):
        xi_old = xi.clone()

        # Thresholding step: Identify coefficients smaller than threshold
        # This is applied to Xi, which are coefficients for current_rhs_mat (potentially normalized)
        small_indices_mask = torch.abs(xi) < sparse_solver_params.threshold
        xi[small_indices_mask] = 0.0

        # Re-solve for non-zero coefficients for each target variable
        for j in range(n_targets):
            # Identify non-zero ("active") coefficients for this target *after* current thresholding
            active_coeffs_mask_j = (xi[:, j] != 0.0)  # Boolean mask of shape (n_features,)

            if not torch.any(active_coeffs_mask_j):
                # All coefficients for this target are zero, ensure xi for this target is all zero
                xi[:, j] = 0.0
                continue

            # Select corresponding columns from current_rhs_mat (the matrix used in lstsq)
            rhs_mat_sparse_j = current_rhs_mat[:, active_coeffs_mask_j]

            target_vector_j = lhs_target[:, j]  # Shape (n_samples,)

            solution_active_j = torch.linalg.lstsq(
                rhs_mat_sparse_j,
                target_vector_j,
                rcond=sparse_solver_params.rcond,
                driver=sparse_solver_params.driver
            ).solution

            xi[active_coeffs_mask_j, j] = solution_active_j.squeeze()

        # Check for convergence using L-infinity norm of the change in Xi
        # Xi here is still potentially scaled if normalize_columns=True.
        # The comparison is consistent as both xi and xi_old are at the same scale.
        diff_norm = torch.linalg.norm(xi - xi_old, ord=float('inf'))
        if diff_norm < sparse_solver_params.convergence_tol:
            # print(f"STLS converged after {_iteration + 1} iterations.")
            break

    # else: # Executed if loop finishes without break
        # print(f"STLS reached max_iter ({max_iter}). Final diff_norm: {diff_norm:.2e}")

    # Un-normalize Xi if columns of rhs_mat were normalized
    # Xi_final = Xi_scaled / col_norms_transposed
    # col_norms has shape (1, n_features). col_norms.T has shape (n_features, 1).
    # Xi has shape (n_features, n_targets).
    # Broadcasting (n_features, n_targets) / (n_features, 1) divides each feature's coefficient
    # across all targets by that feature's norm.
    if sparse_solver_params.normalize_columns:  # Check again, as col_norms might be all ones if not normalized
        xi = xi / col_norms.T

    return xi

#%% Overarching config
@dataclass
class SINDyParams:
    """Configuration for SINDy library, system construction, and linear solver."""
    poly_order: int = 1
    include_bias: bool = True
    truncation_rank: Optional[int] = None
    method: str = 'weak'
    test_func_params: Optional[TestFunctionParams] = None
    solver_fn: BaseSolver = dense_solver
    solver_params: Optional[BaseSolverParams] = None
