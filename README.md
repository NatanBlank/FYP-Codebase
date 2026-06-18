## Overview

This repository contains the code developed for my final-year project investigating the application of the Harrow-Hassidim-Lloyd (HHL) quantum linear systems algorithm to finite-difference discretisations of the Poisson equation.

The project evaluates the numerical behaviour, accuracy and computational characteristics of HHL in comparison with classical tridiagonal solvers. Both one-dimensional and two-dimensional Poisson problems are considered.

All quantum results were generated using noiseless statevector simulations within Qiskit.

## Repository Structure

### '1d example.py'

Representative one-dimensional Poisson solver.

This script solves

-u''(x)=f(x) on the unit interval subject to Dirichlet boundary conditions.

The implementation:

1. Constructs the finite-difference discretisation.
2. Solves the resulting tridiagonal system using the Thomas algorithm.
3. Solves the same system using HHL.
4. Compares both solutions against the analytical reference solution.

The representative example corresponds to the constant forcing case

f(x)=16, which was one of the benchmark problems studied throughout the project.

---

### '2d example.py'

Representative two-dimensional Poisson solver.

This script implements the hybrid quantum-classical methodology for extending HHL to two-dimensional finite-difference discretisations.

The problem solved is

-\nabla^2 u = f(x,y), on the unit square with homogeneous Dirichlet boundary conditions.

The representative example corresponds to

u(x,y)=\sin(\pi x)\sin(\pi y), with forcing f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y).

A line-Jacobi iteration is employed, with each row solve performed using HHL.

---

### 'Thomas.py'

Classical Thomas algorithm implementation used throughout the project as the benchmark tridiagonal solver.

---

### 'HHL'

Contains the HHL implementation and supporting classes used by the project.

The key files are:

* 'hhl.py'

  * Main HHL implementation.
* 'hhl_result.py'

  * Result container returned by the solver.
* 'linear_system_matrix.py'

  * Matrix interface classes.
* 'numpy_matrix.py'

  * NumPy-based matrix implementation.
* 'tridiagonal_toeplitz.py'

  * Structured matrix utilities used by the HHL framework.

These files were adapted from an existing educational HHL implementation and subsequently modified during the project.

---

## Running the Examples

### One-Dimensional Example

```bash
1d example.py
```

Outputs:

* Analytical solution
* Thomas solution
* HHL solution
* L2 error norm

---

### Two-Dimensional Example

```bash
2d example.py
```

Outputs:

* Numerical solution field
* Analytical solution field
* Iteration history
* Final error metrics

The script additionally writes CSV files containing the solution and convergence history.

---

## Parameters of Interest

If further experimentation is desired, the following parameters may be modified.

### 1D Solver

Within `1d example.py`:

```python
N = 17
epsilon = 1e-3
```

where:

* `N` controls mesh resolution (N-1 is the number of interior unknowns)
* `epsilon` controls HHL Hamiltonian simulation accuracy

Alternative forcing functions and boundary conditions may also be substituted.

---

### 2D Solver

Within `2d example.py`:

```python
N = 17
epsilon = 1e-3
max_iters = 1000
tol = 1e-8
```

where:

* `N` controls the spatial discretisation
* `epsilon` controls HHL accuracy
* `max_iters` is the maximum number of outer line-Jacobi iterations
* `tol` is the convergence tolerance

For compatibility with the current HHL implementation, the number of interior unknowns per row (`N-1`) should remain a power of two.

Examples:

```python
N = 5
N = 9
N = 17
N = 33
```

correspond to 4, 8, 16 and 32 interior unknowns respectively.

---

## Notes

The repository is intended as a representative demonstration of the methodology used throughout the project rather than a complete archive of every script used during data generation and post-processing.

The original project contained additional automation scripts for parameter sweeps, resource estimation, plotting, data collection and report generation. These have been omitted to keep the repository concise and focused on the core numerical methods.

All results reported in the dissertation were generated using variations and extensions of the methodologies implemented in the representative scripts provided here.
