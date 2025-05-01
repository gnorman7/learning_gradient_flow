import torch
import torch.nn as nn
import pytest
from gradient_flow_optimizer import GradientFlowBase


class SimpleNet(nn.Module):
    """Simple neural network with two linear layers and tanh activation."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.act = nn.Tanh()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer2(self.act(self.layer1(x)))


class OptimizerTestClass(GradientFlowBase):
    """Test subclass of GradientFlowBase with exposed flattening methods."""

    def __init__(self, params):
        super().__init__(params, dt=1e-3)

    # Expose protected methods as public for testing
    def gather_flat(self, params_or_grads="params"):
        """Public wrapper for _gather_flat."""
        return self._gather_flat(params_or_grads)

    def set_params_from_flat(self, flat_params):
        """Public wrapper for _set_params_from_flat."""
        return self._set_params_from_flat(flat_params)

    def _get_dynamics_vector(self, t, flat_params, flat_grad):
        """Implementation required by abstract base class."""
        return -flat_grad  # Simple gradient descent


# List of devices to test - will be skipped if not available
@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    """Parametrized fixture providing device for testing."""
    device_name = request.param
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")
    return torch.device(device_name)


@pytest.fixture
def model(device):
    """Create a fresh model instance with reproducible parameters on specified device."""
    torch.manual_seed(0)
    return SimpleNet().to(device)


@pytest.fixture
def optimizer(model):
    """Create optimizer instance for testing flattening operations."""
    return OptimizerTestClass(model.parameters())


def test_flatten_unflatten_params(model, optimizer):
    """Test that parameters remain unchanged after flatten/unflatten cycle."""
    # Store original parameters for comparison
    original_params = {name: param.clone().detach()
                       for name, param in model.named_parameters()}

    # Flatten parameters
    flat_params = optimizer.gather_flat()

    # Unflatten parameters back into the model
    optimizer.set_params_from_flat(flat_params)

    # Validate parameters are unchanged
    for name, param in model.named_parameters():
        if not torch.allclose(param, original_params[name]):
            print(f"Original {name}:\n{original_params[name]}")
            print(f"After unflatten {name}:\n{param}")

        assert torch.allclose(param, original_params[name]), \
            f"Parameter {name} changed after flatten/unflatten operation"


def test_flatten_unflatten_with_modification(model, optimizer):
    """Test that modifications to flattened parameters propagate correctly."""
    # Flatten parameters
    flat_params = optimizer.gather_flat()

    # Modify the flattened parameters
    modified_params = flat_params + 1.0

    # Apply the modified parameters back to the model
    optimizer.set_params_from_flat(modified_params)

    # Check that all parameters have been increased by 1.0
    # Reflattening should match our modified_params
    new_params = optimizer.gather_flat()
    assert torch.allclose(new_params, modified_params), \
        "Parameter modifications were not correctly propagated"


def test_flatten_gradients(model, optimizer):
    """Test gradient flattening and handling of None gradients."""
    # Create a dummy input and compute gradients
    x = torch.randn(4, 2, device=model.layer1.weight.device)
    y = torch.randn(4, 1, device=model.layer1.weight.device)

    # Forward pass and compute loss
    output = model(x)
    loss = nn.MSELoss()(output, y)

    # Make sure gradients are None initially
    for param in model.parameters():
        param.grad = None

    # Test flattening with None gradients (should be zeros)
    flat_grads_before = optimizer.gather_flat("grads")
    assert torch.all(flat_grads_before == 0), "None gradients should be flattened to zeros"

    # Backward pass to compute gradients
    loss.backward()

    # Test flattening with actual gradients
    flat_grads_after = optimizer.gather_flat("grads")

    # Manual flattening for verification
    manual_flat_grads = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.allclose(flat_grads_after, manual_flat_grads), \
        "Gradient flattening doesn't match manual flattening"
