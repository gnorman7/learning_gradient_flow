import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from sindy_tools import create_sindy_library, get_derivative, create_predictor

def test_sindy_oscillator():
    """
    Test SINDy tools by discovering and simulating a damped oscillator system.

    This implementation uses column vectors (shape (d,1)) for the state
    representation, which better matches the mathematical formulation of
    dynamical systems.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    A_mat = torch.tensor([[0.0, 3.0], [-1.0, -0.2]])

    def damped_oscillator(t, state):
        """
        Damped oscillator system dynamics.

        Args:
            t: Time point (unused, but required by odeint)
            state: Tensor of shape [2, 1] with state variables [position, velocity]

        Returns:
            Tensor of shape [2, 1] with derivatives
        """
        # state shape: [2, 1]
        return A_mat @ state

    x0 = torch.tensor([[1.0], [0.0]])  # Shape: [2, 1]
    t_span = torch.linspace(0, 10, 200)

    # Generate the ground truth solution
    true_solution = odeint(damped_oscillator, x0.flatten(), t_span)  # Shape: [100, 2]

    library = create_sindy_library(input_dim=2, poly_order=2)
    Theta = library(true_solution)  # Shape: [100, P]

    dt = t_span[1] - t_span[0]  # Time step size
    dxdt = get_derivative(dt, true_solution)  # Shape: [98, 2]
    Theta_m2 = Theta[1:-1]  # Shape: [98, P] to match dxdt shape
    # Xi shape: [P, 2] - Each column represents coefficients for one state variable
    Xi = torch.linalg.lstsq(Theta_m2, dxdt, rcond=1e-5).solution

    print("Discovered coefficients (Xi):")
    print(Xi)

    # 4. Create predictor for the discovered system
    predictor = create_predictor(Xi, library)
    sindy_system = lambda t, state: predictor(state)  # SINDy system function

    # 5. Simulate the discovered system
    sindy_solution = odeint(sindy_system, x0, t_span).squeeze()  # Shape: [100, 2]

    # 6. Compare true and SINDy solutions
    mse = torch.mean((true_solution - sindy_solution) ** 2)
    print(f"Mean squared error: {mse.item():.6f}")

    # check that all close
    assert torch.allclose(true_solution, sindy_solution, atol=1e-2), "SINDy solution does not match true solution."

    # For a system with d=2, poly_order=2, include_bias=True:
    # Xi should have shape [P, 2] where P = 6 (1 bias + 2 linear + 3 quadratic)
    goal_Xi = torch.tensor([[0.0, 0.0], [A_mat[0, 0], A_mat[1,0]],
                            [A_mat[0, 1], A_mat[1,1]], [0.0, 0.0],
                            [0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(Xi, goal_Xi, atol=1e-1), "Xi coefficients do not match expected values."


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_span.numpy(), true_solution[:, 0].numpy(), 'b-', label=r'True $x_1$')
    plt.plot(t_span.numpy(), sindy_solution[:, 0].numpy(), 'r--', label=r'SINDy $x_1$')
    plt.xlabel('Time')
    plt.ylabel(r'Position ($x_1$)')
    plt.title('Position vs Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_span.numpy(), true_solution[:, 1].numpy(), 'b-', label=r'True $x_2$')
    plt.plot(t_span.numpy(), sindy_solution[:, 1].numpy(), 'r--', label=r'SINDy $x_2$')
    plt.xlabel('Time')
    plt.ylabel(r'Velocity ($x_2$)')
    plt.title('Velocity vs Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig('sindy_comparison.png')

    plt.figure(figsize=(6, 5))
    plt.plot(true_solution[:, 0].numpy(), true_solution[:, 1].numpy(), 'b-', label='True')
    plt.plot(sindy_solution[:, 0].numpy(), sindy_solution[:, 1].numpy(), 'r--', label='SINDy')
    plt.xlabel(r'Position ($x_1$)')
    plt.ylabel(r'Velocity ($x_1$)')
    plt.title('Phase Space')
    plt.legend()
    plt.savefig('sindy_phase_space.png')

    print("All tests passed!")
    return true_solution, sindy_solution, Xi

if __name__ == "__main__":
    test_sindy_oscillator()
