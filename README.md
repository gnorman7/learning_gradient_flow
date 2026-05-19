# Learning Gradient Flow (LGF)

Code for "Learning Gradient Flow: Using Equation Discovery to Accelerate Engineering Optimization," accepted for publication in *Computer Methods in Applied Mechanics and Engineering* (CMAME).

**Paper:** [arXiv:2602.13513](https://arxiv.org/abs/2602.13513)

## Overview

LGF uses data-driven equation discovery to learn continuous-time gradient flow dynamics from optimization trajectories. The discovered dynamics serve as a surrogate for the original optimization problem, accelerating convergence by avoiding expensive evaluations of the objective function and its gradient. LGF supports surrogate models of variable polynomial order in full- or reduced-dimensional spaces at user-defined intervals in the optimization process.

The paper demonstrates LGF on problems from engineering mechanics and scientific machine learning, including inverse problems, structural topology optimization, and PDE forward solves.

## Citation

```bibtex
@article{norman2026learning,
  title={Learning Gradient Flow: Using Equation Discovery to Accelerate Engineering Optimization},
  author={Norman, Grant and Rowan, Conor and Maute, Kurt and Doostan, Alireza},
  journal={arXiv preprint arXiv:2602.13513},
  year={2026}
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

