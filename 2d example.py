import numpy as np
import pandas as pd

from Thomas import thomas
from HHL.hhl import HHL
from qiskit.primitives import StatevectorEstimator, StatevectorSampler


def u_true_fn(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def f_fn(X, Y):
    return 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)


def build_line_matrix(n):
    """
    Builds the tridiagonal matrix for each line-Jacobi row solve.

    For the 2D five-point finite-difference stencil:

        -u_{i-1,j} + 4u_{i,j} - u_{i+1,j}
        = h^2 f_{i,j} + u_{i,j-1} + u_{i,j+1}

    Hence the line matrix is tridiag(-1, 4, -1).
    """

    main_diag = 4 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)

    A = np.diag(main_diag)
    A += np.diag(off_diag, k=1)
    A += np.diag(off_diag, k=-1)

    return A, main_diag, off_diag


def compute_L2_error(U, U_true, h):
    return np.sqrt(h**2 * np.sum((U - U_true) ** 2))


def solve_line_hhl(rhs, A_line, hhl_solver):
    result = hhl_solver.solve(A_line, rhs)
    return result.solution


def run_2d_poisson_hhl():
    # --------------------------------------------------------
    # Discretisation
    # --------------------------------------------------------

    N = 17          # subintervals per direction
    n = N - 1       # interior points per direction = 16
    h = 1 / N

    epsilon = 1e-3
    max_iters = 1000
    tol = 1e-8

    if not np.log2(n).is_integer():
        raise ValueError("For this HHL implementation, n = N - 1 must be a power of 2.")

    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)

    X, Y = np.meshgrid(x, y, indexing="ij")

    F = f_fn(X, Y)
    U_true = u_true_fn(X, Y)

    # Homogeneous Dirichlet boundary conditions.
    U = np.zeros((N + 1, N + 1))

    # --------------------------------------------------------
    # Build row-solve matrix
    # --------------------------------------------------------

    A_line, main_diag, off_diag = build_line_matrix(n)

    # --------------------------------------------------------
    # Initialise HHL solver
    # --------------------------------------------------------

    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()

    hhl_solver = HHL(
        estimator=estimator,
        sampler=sampler,
        epsilon=epsilon,
    )

    history = []

    print("Starting representative 2D HHL line-Jacobi solve")
    print(f"N = {N}")
    print(f"Interior points per direction = {n}")
    print(f"epsilon = {epsilon}")

    # --------------------------------------------------------
    # Line-Jacobi iteration
    # --------------------------------------------------------

    for k in range(1, max_iters + 1):
        U_old = U.copy()
        U_new = U.copy()

        for j in range(1, N):
            rhs = np.zeros(n)

            for local_i, i in enumerate(range(1, N)):
                rhs[local_i] = (
                    h**2 * F[i, j]
                    + U_old[i, j - 1]
                    + U_old[i, j + 1]
                )

            # Left/right boundary terms.
            # These are zero here but retained for clarity.
            rhs[0] += U[0, j]
            rhs[-1] += U[N, j]

            row_solution = solve_line_hhl(rhs, A_line, hhl_solver)

            U_new[1:N, j] = row_solution

        U = U_new

        iteration_change = np.max(np.abs(U - U_old))
        L2_error = compute_L2_error(U, U_true, h)

        history.append([k, iteration_change, L2_error])

        print(
            f"Iteration {k:4d}: "
            f"change = {iteration_change:.6e}, "
            f"L2 error = {L2_error:.6e}"
        )

        if iteration_change < tol:
            print(f"\nConverged after {k} iterations.")
            break

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    pd.DataFrame(U).to_csv(
        "poisson_2d_hhl_solution.csv",
        index=False,
        header=False,
    )

    pd.DataFrame(U_true).to_csv(
        "poisson_2d_exact_solution.csv",
        index=False,
        header=False,
    )

    pd.DataFrame(
        history,
        columns=["iteration", "iteration_change", "L2_error"],
    ).to_csv(
        "poisson_2d_hhl_history.csv",
        index=False,
    )

    print("\nSaved:")
    print("  poisson_2d_hhl_solution.csv")
    print("  poisson_2d_exact_solution.csv")
    print("  poisson_2d_hhl_history.csv")

    return U, U_true, history


if __name__ == "__main__":
    U, U_true, history = run_2d_poisson_hhl()
