# Computational Physics 2 - FYS4411/FYS9411 - Project2
Time dependent Hartree-Fock theory

## How to run our code

First, you need to install [Quantum Systems](https://github.com/Schoyen/quantum-systems). With Quantum systems being installed, running our code is very simple - all relevant files are in the folder "/code/Python/". The file "main.py" will generate all relevant plots; all hyperparameters of interest are to be changed in that file. The file "GHFSolver.py" contains the GHF-class, which contains relevant methods. The file "helper_functions.py" contains small functions used by the GHF-class that are, in a sense, more "general", such as numerical integration.

The content of "/code/Julia" contains a Julia-implementation of the same code, but is unfinished as of today.
