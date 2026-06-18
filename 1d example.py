import numpy as np

from Thomas import thomas
from HHL.hhl import HHL
from qiskit.primitives import StatevectorEstimator, StatevectorSampler


def build_A(N):
    n = N - 1

    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)

    A = np.diag(main_diag)
    A += np.diag(off_diag, k=1)
    A += np.diag(off_diag, k=-1)

    return A, main_diag, off_diag


# ---------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------

N = 17 #16 interior unknowns
epsilon = 1e-3

alpha = 0.0
beta = 0.0

h = 1 / N

x = np.linspace(0, 1, N + 1)
x_int = x[1:-1]


# f(x) = 16
f = lambda x: 16 + 0 * x

# analytical solution of -u'' = 16
u_true = 8 * x * (1 - x) + alpha * (1 - x) + beta * x


# ---------------------------------------------------------------------
# Build RHS and matrix
# ---------------------------------------------------------------------

b = h**2 * f(x_int)
b[0] += alpha
b[-1] += beta

A, main_diag, off_diag = build_A(N)


# ---------------------------------------------------------------------
# Thomas solve
# ---------------------------------------------------------------------

u_thomas_int = thomas(
    off_diag,
    main_diag,
    off_diag,
    b,
)

u_thomas = np.concatenate(
    ([alpha], u_thomas_int, [beta])
)


# ---------------------------------------------------------------------
# HHL solve
# ---------------------------------------------------------------------

estimator = StatevectorEstimator()
sampler = StatevectorSampler()

hhl = HHL(
    estimator=estimator,
    sampler=sampler,
    epsilon=epsilon,
)

result = hhl.solve(A, b)

u_hhl_int = result.solution

u_hhl = np.concatenate(
    ([alpha], u_hhl_int, [beta])
)


# ---------------------------------------------------------------------
# Error calculation
# ---------------------------------------------------------------------

L2_error = np.sqrt(
    h * np.sum((u_hhl - u_true) ** 2)
)

print(f"\nL2 Error (HHL vs Exact) = {L2_error:.6e}\n")


# ---------------------------------------------------------------------
# Output solution field
# ---------------------------------------------------------------------

print("x          Exact         Thomas        HHL")
print("-" * 50)

for xi, ue, ut, uh in zip(
    x,
    u_true,
    u_thomas,
    u_hhl,
):
    print(
        f"{xi:6.3f}   "
        f"{ue:10.6f}   "
        f"{ut:10.6f}   "
        f"{uh:10.6f}"
    )
