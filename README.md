# Learning Gradient Flow (LGF)

Code for "Learning Gradient Flow: Using Equation Discovery to Accelerate Engineering Optimization," to be published in *Computer Methods in Applied Mechanics and Engineering* (CMAME).

**CMAME:** [10.1016/j.cma.2026.119099](10.1016/j.cma.2026.119099)

**arXiv:** [arXiv:2602.13513](https://arxiv.org/abs/2602.13513)

## Overview

LGF uses data-driven equation discovery to learn continuous-time gradient flow dynamics from optimization trajectories. The discovered dynamics serve as a surrogate for the original optimization problem, accelerating convergence by avoiding expensive evaluations of the objective function and its gradient. LGF supports surrogate models of variable polynomial order in full- or reduced-dimensional spaces at user-defined intervals in the optimization process.

The paper demonstrates LGF on problems from engineering mechanics and scientific machine learning, including inverse problems, structural topology optimization, and PDE forward solves.

## Citation

```bibtex
@article{norman2026learning,
  title = {Learning gradient flow: Using equation discovery to accelerate engineering optimization},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {460},
  pages = {119099},
  year = {2026},
  issn = {0045-7825},
  doi = {https://doi.org/10.1016/j.cma.2026.119099},
  url = {https://www.sciencedirect.com/science/article/pii/S0045782526003725},
  author = {Grant Norman and Conor Rowan and Kurt Maute and Alireza Doostan},
  keywords = {Equation discovery, Data-driven modeling, Gradient flow, Surrogate models, Optimization}
}
```

## Installation

```bash
git clone https://github.com/gnorman7/learning_gradient_flow
cd learning_gradient_flow
pip install -e .
```


## Project Structure
- `src/`: Source Code
- `examples/`: Example scripts for the results in the paper
- `tests/`: Tests, using pytest
- `notebooks/`: Jupyter Notebooks
- `requirements.txt`: Python package requirements (TODO)


## Usage

For a quick check, start with the small two-variable problem:

```bash
python examples/ex1_2vars.py
```

Other paper examples include the Newton, deep-learning, and scattering examples:

```bash
python examples/ex3_newton.py
python examples/ex4_scattering.py
python examples/ex5_deep.py
```

The scattering example uses CALFEM for Python (`calfem-python`) and its
Gmsh-based meshing workflow, so it may need a little additional local setup:

```bash
pip install -e ".[examples]"
```


The main optimizer classes are:

- `learning_gradient_flow.gradient_flow_optimizer.LGFGradientFlow`: LGF optimizer
  for learning gradient-flow dynamics from an optimization trajectory.
- `learning_gradient_flow.adam_flow_optimizer.BaseAdam`: baseline Adam variant
  used for comparisons in the examples.
- `learning_gradient_flow.adam_flow_optimizer.LGFAdam`: Adam-style LGF optimizer
  that learns a SINDy surrogate for gradients.

SINDy library construction, weak/strong form matrix assembly, and sparse
regression utilities are in `learning_gradient_flow.sindy_tools`.

To run tests, install the optional development dependency first:

```bash
pip install -e ".[dev]"
python -m pytest
```
