import torch

from learning_gradient_flow import adam_flow_optimizer, gradient_flow_optimizer, sindy_tools


def _quadratic_model():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))

    def closure():
        if param.grad is not None:
            param.grad.zero_()
        loss = torch.sum(param**2)
        loss.backward()
        return loss

    return param, closure


def test_base_adam_smoke_step():
    param, closure = _quadratic_model()
    optimizer = adam_flow_optimizer.BaseAdam([param], lr=1e-2)

    loss = optimizer.step(closure)

    assert loss is not None
    assert optimizer.state["func_evals"] == 1
    assert torch.isfinite(param).all()


def test_lgf_adam_smoke_step():
    param, closure = _quadratic_model()
    optimizer = adam_flow_optimizer.LGFAdam(
        [param],
        lr=1e-2,
        history_size=2,
        sindy_params=sindy_tools.SINDyParams(method="tracked"),
    )

    loss = optimizer.step(closure)

    assert loss is not None
    assert optimizer.state["func_evals"] == 1
    assert len(optimizer.state["history"]) == 1
    assert torch.isfinite(param).all()


def test_lgf_gradient_flow_smoke_step():
    param, closure = _quadratic_model()
    backup_optimizer = torch.optim.SGD([param], lr=1e-2)
    optimizer = gradient_flow_optimizer.LGFGradientFlow(
        [param],
        backup_optimizer=backup_optimizer,
        dt=1e-2,
        history_size=2,
        retrain_interval=2,
    )

    loss = optimizer.step(closure)

    assert loss is not None
    assert optimizer.state["func_evals"] == 1
    assert optimizer.state["history_count"] == 1
    assert torch.isfinite(param).all()
