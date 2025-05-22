import torch
import torchdiffeq
from learning_gradient_flow.sindy_tools import create_sindy_library, create_predictor, assemble_weak_matrices, stls_sparse_solver, dense_solver

def test_sindy_oscillator():
    """
    Test SINDy tools by discovering and simulating a damped oscillator system.

    This implementation uses column vectors (shape (d,1)) for the state
    representation, which better matches the mathematical formulation of
    dynamical systems.
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)

    noise_level = 0.1

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def lorentz_system(t, y):
        """Lorentz system dynamics. y is a column vector (3,1)."""
        # y shape: (3, 1)
        x, y_val, z = y.squeeze()  # Squeeze to use scalar operations
        dxdt = sigma * (y_val - x)
        dydt = x * (rho - z) - y_val
        dzdt = x * y_val - beta * z
        return torch.tensor([dxdt, dydt, dzdt]).view(-1, 1)  # Return as column vector (3,1)

    d = 3
    y0 = torch.tensor([[10.0], [10.0], [10.0]])  # Shape: (3, 1)
    t_span = torch.linspace(0, 25, 2500)  # Time span for integration

    # Solve the ODE
    solution = torchdiffeq.odeint(lorentz_system, y0, t_span)
    # solution shape: (n_times, d, 1)
    X_data = solution.squeeze(-1)  # Shape: (n_times, d)
    X_data += noise_level * torch.randn_like(X_data)  # Add noise

    # SINDy solve
    poly_order = 3
    library = create_sindy_library(input_dim=d, poly_order=poly_order, include_bias=True)

    Theta = library(X_data)
    test_mat_kwargs = {
        'width': 20,
        'p': 5,
        'stride': 10,
        'include_endpoints': True,
    }
    lhs_weak, rhs_weak = assemble_weak_matrices(X_data, Theta, t_span, test_mat_kwargs=test_mat_kwargs)

    threshold = 0.1
    normalize_columns = False
    Xi = stls_sparse_solver(rhs_weak, lhs_weak, threshold=threshold, normalize_columns=normalize_columns)

    P, d = Xi.shape
    Xi_true = torch.zeros((P, d))
    # dx/dt = -sigma*x + sigma*y
    Xi_true[1, 0] = -sigma  # x term for dx/dt
    Xi_true[2, 0] = sigma  # y term for dx/dt
    # dy/dt = rho*x - y - xz
    Xi_true[1, 1] = rho    # x term for dy/dt
    Xi_true[2, 1] = -1.0   # y term for dy/dt
    Xi_true[6, 1] = -1.0   # xz term for dy/dt
    # dz/dt = xy - beta*z
    Xi_true[5, 2] = 1.0    # xy term for dz/dt
    Xi_true[3, 2] = -beta  # z term for dz/dt

    # compare
    print("Discovered coefficients (Xi):")
    print(Xi)
    print("True coefficients (Xi_true):")
    print(Xi_true)
    assert torch.allclose(Xi, Xi_true, atol=1e-1), "Xi coefficients do not match expected values."

if __name__ == "__main__":
    test_sindy_oscillator()
