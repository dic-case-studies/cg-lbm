# Color Gradient Lattice Boltzmann Simulator

This repo is a Color Gradient Lattice Boltzmann Method Simulator for immiscible fluids.

Based on `Subhedar, A. (2022). Color-gradient lattice Boltzmann model for immiscible fluids with density contrast. Physical review. E, 106 4-2, 045308 .`

## Installing

This project uses jax, installing jax is a prerequisite. Instruction to install jax [is available here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier)

### Installing library from Source

```
git clone https://github.com/dic-case-studies/cg-lbm.git
pip install -e .
```

### Installing from PyPI

Planned for the future

## Testing

First we need to install developer dependencies present in setup.py

From the root directory run

```
pytest

# Run with -s to capture the time from the performance benchmarks
pytest -s
```

## Examples

Examples of running the simulation are present in [notebooks folder](./notebooks/Color%20Gradient%20Lattice%20Boltzman%20Method.ipynb)

## Acknowledgements

CG-LBM was created from the collaboration between the Engineering for Research (e4r) group, Thoughtworks and [Dr. Amol Subhedar](https://www.che.iitb.ac.in/faculty/amol-subhedar), [Department of Chemical Engineering](https://www.che.iitb.ac.in/), Indian Institute of Technology, Bombay.

### Core Contributors

- [Nimalan M](https://github.com/mark1626/)
- [Ravinder Jajoria](https://github.com/ravinderj)
- [Maneesh Sutar](https://github.com/maneesh29s)
- [Malyadeep Bhattacharya](https://github.com/Malyadeep)

