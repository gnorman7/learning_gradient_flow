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
    library = create_sindy_library(input_dim=d, poly_order=2)
    x = torch.randn(n_t, d) # transpose of data matrix?
    Theta = library(x)  # shape (n_t, P)
    Thetam2 = Theta[1:-1]  # shape (n_t-2, P)
    dxdt = get_derivative(x)  # shape (n_t-2, d)
    # now construct the system: (n_t-2, d) = (n_t-2, P) @ (P, d)
    # dxdt = Thetam2 @ Xi
    Xi = torch.linalg.lstsq(Thetam2, dxdt, rcond=1e-5).solution

    pred = create_predictor(Xi, library)
    x_pred = pred(x[0:1].T) # input is (1,d) transposed to be (d,1)
    print(x_pred.shape)  # shape (d, 1)
