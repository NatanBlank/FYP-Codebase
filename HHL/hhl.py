from typing import Optional, Union, List
import numpy as np 
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation, Isometry
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal

from qiskit.circuit.library import Isometry
from numpy_matrix import NumPyMatrix
from tridiagonal_toeplitz import TridiagonalToeplitz

from qiskit.quantum_info import Statevector
try:
    # Newer Qiskit primitives (your environment)
    from qiskit.primitives import BaseEstimatorV2 as BaseEstimator
    from qiskit.primitives import BaseSamplerV2 as BaseSampler
except ImportError:
    # Older Qiskit primitives (fallback)
    from qiskit.primitives import BaseEstimatorV1 as BaseEstimator
    from qiskit.primitives import BaseSamplerV1 as BaseSampler
from .hhl_result import HHLResult

class HHL: 
    """
    HHL algorithm wrapper. 

    Responsibilities: 
        1) Construct the HHL circuit for Ax=b.
        2) Simulate the circuit
        3) Extract an approximate classical solution vector 
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        sampler: Optional[Union[BaseSampler, None]] = None,
        epsilon: float = 1e-4,
        ) -> None:
        """
        Args:
            estimator: Qiskit Estimator primative 
            sampler: Qiskit Sampler primative 
            epsilon: Target total error tolerance for the algorithm 
        """
        super().__init__()

        # --- Global error tolerance ---
        self.epsilon = float(epsilon)

        # --- Error budget split ---
        # These splits are intended to keep each major subroutine within a share of epsilon.
        self.eps_rotation   = self.epsilon / 3   # conditioned rotation / reciprocal stage
        self.eps_stateprep  = self.epsilon / 3   # state preparation stage
        self.eps_hamsim     = self.epsilon / 6   # Hamiltonian simulation stage

        # --- Scaling factor for the final solution ---
        # Updated later using eigenvalue bounds (if available).
        self.scaling = 1.0

        # --- Execution primitives (future-proofing) ---
        self.estimator = estimator
        self.sampler = sampler

        # Use exact reciprocal by default (ExactReciprocal circuit).
        self.use_exact_reciprocal = True

    def _get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float: 
        """
        Choose a scaling delta so that lambda_min is exactly represntable on an 
        n_l-qubit binary fraction grid (as used by QPE). 

        Intuition: 
            QPE encodes eigenvalues into a discretised set of phases. We pick delta to 
            align lambda_min with that discretisation. 

        Args:
            n_l: the number of qubits to represent the eigenvalues. 
            lambda_min: the smallest eigenvalue. 
            lambda_max: the largest eigenvalue. 
        
        Returns: 
            the value of delta (the scaling factor) 
        """
        # Scale lambda_min into the integer grid [0, 2^n_l - 1]
        scaled = abs(lambda_min) * (2**n_l - 1) / abs(lambda_max)

        # Guard against floating-point rounding issues 
        if abs(scaled - 1.0) < 1e-7:
            scaled = 1.0 
        
        # Convert integer part into binary fraction bits
        bits = format(int(scaled), "#0" + str(n_l + 2) + "b")[2:] # remove '0b'
        fraction = 0.0 
        for i, bit in enumerate(bits): 
            fraction += int(bit) / (2 ** (i+1))

        return fraction
    
    @staticmethod
    def _is_tridiagonal_toeplitz_hermitian(A: np.ndarray, atol: float = 1e-12):
        """
        Detect whether A is a real Hermitian tridiagonal Toeplitz matrix.

        Conditions (within atol):
        - A is square
        - A is Hermitian
        - A has no nonzeros outside the main and +/-1 diagonals
        - main diagonal is constant
        - off diagonals are constant and symmetric

        Returns:
        (is_toeplitz: bool, main_diag: float, off_diag: float)
        """
            
        A = np.asarray(A, dtype=complex)
        n = A.shape[0]

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return False, 0.0, 0.0

        if not np.allclose(A, A.conj().T, atol=atol):
            return False, 0.0, 0.0

        # Tridiagonal check: entries with |i-j|>1 must be ~0
        # Vectorised form:
        mask = np.ones_like(A, dtype=bool)
        idx = np.arange(n)
        mask[idx, idx] = False
        mask[idx[:-1], idx[1:]] = False
        mask[idx[1:], idx[:-1]] = False
        if np.any(np.abs(A[mask]) > atol):
            return False, 0.0, 0.0

        # Toeplitz constant diagonals
        main_diag = A[0, 0]
        if not np.allclose(np.diag(A), main_diag, atol=atol):
            return False, 0.0, 0.0

        if n >= 2:
            off_diag = A[0, 1]
            if not np.allclose(np.diag(A, 1), off_diag, atol=atol):
                return False, 0.0, 0.0
            if not np.allclose(np.diag(A, -1), np.conjugate(off_diag), atol=atol):
                return False, 0.0, 0.0
        else:
            off_diag = 0.0

        # This TridiagonalToeplitz implementation expects real floats
        if abs(main_diag.imag) > atol or abs(off_diag.imag) > atol:
            return False, 0.0, 0.0

        return True, float(main_diag.real), float(off_diag.real)

    def construct_circuit(self, matrix, vector, neg_vals: bool = True)-> QuantumCircuit: 
        """
        Construct the full HHL circuit for Ax=b. 

        Args: 
            matrix: either:
                - a QuantumCircuit implementing exp(iAt) 
                - a numpy array/list representing a Hermitian matrix A 
            vector: either: 
                - a QuantumCircuit that prepares |b> 
                - a numpy array/list representing b (will be normalised and loaded)
            neg_vals: if True, allocate extra sign support for potentially negative eigenvalues.
                      if False, computation is cheaper. 
        
        Returns: 
            QuantumCircuit: the assembled HHL circuit. 
        """
        # ------------------------------------------------------------
        # (1) STATE PREPARATION: build circuit that prepares |b>
        # ------------------------------------------------------------
        if isinstance(vector, QuantumCircuit): 
            b_circuit = vector
            nb = b_circuit.num_qubits       #number of qubits 
        else: 
            b = np.asarray(vector, dtype = complex).reshape(-1) 
            nb_float = np.log2(len(b))
            if nb_float % 1 != 0:
                raise ValueError("Length of b must be a power of 2.")
            nb = int(nb_float)
            b_circuit = QuantumCircuit(nb, name = "prepare_b")
            b_circuit.append(Isometry(b / np.linalg.norm(b), 0, 0), list(range(nb)))

        # Number of "flag" qubits: 1 success flag used by the conditioned rotation 
        nf = 1

        # ------------------------------------------------------------
        # (2) MATRIX / HAMILTONIAN SIMULATION: build object for exp(iAt)
        # ------------------------------------------------------------
        if isinstance(matrix, QuantumCircuit): 
            A_op = matrix
        else:
            A = np.asarray(matrix, dtype = complex) 
            
            #Validate HHL requirements for the NumPyMatrix path
            if A.shape[0] != A.shape[1]:
                raise ValueError("Input matrix must be square.") 
            if np.log2(A.shape[0]) % 1 != 0: 
                raise ValueError("Input matrix dimension must be 2^n.") 
            if not np.allclose(A, A.conj().T): 
                raise ValueError("Input matrix must be Hermitian.") 
            if A.shape[0] != 2**nb:
                raise ValueError("Matrix dimension must match vector dimension.") 
            
            # detect Toeplitz-tridiagonal structure and choose best implementation
            is_toeplitz, main_diag, off_diag = self._is_tridiagonal_toeplitz_hermitian(A, atol=1e-12)

            if is_toeplitz:
                # Structured simulation (avoids black-box UnitaryGate issues)
                A_op = TridiagonalToeplitz(
                    num_state_qubits=nb,
                    main_diag=main_diag,
                    off_diag=off_diag,
                    evolution_time=2 * np.pi,
                    # trotter_steps can be left default; it may be updated when evolution_time is set
                )
            else:
                # Default wrapper: dense expm-based simulation
                A_op = NumPyMatrix(A, evolution_time = 2 * np.pi) 

        # Apply Hamiltonian-simulation tolerance if supported 
        if hasattr(A_op, "tolerance"): 
            A_op.tolerance = self.eps_hamsim

        # ------------------------------------------------------------
        # (3) SET PHASE REGISTER SIZE: based on condition number estimate
        # ------------------------------------------------------------
        if hasattr(A_op, "condition_bounds") and A_op.condition_bounds() is not None:
            kappa = A_op.condition_bounds()[1]
        else:
            kappa = 1.0

        # Eigenvalue / phase register qubits 
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + int(neg_vals) 

        # ------------------------------------------------------------
        # (4) CALIBRATE QPE: eigenvalue bounds -> delta + evolution_time + scaling
        # ------------------------------------------------------------
        if hasattr(A_op, "eigs_bounds") and A_op.eigs_bounds() is not None: 
            lambda_min, lambda_max = A_op.eigs_bounds()

            # -neg_vals because one qubit may be used for sign handling 
            delta = self._get_delta(nl - int(neg_vals), lambda_min, lambda_max)

            # Update evolution time so that the QPE phases align properly 
            A_op.evolution_time = 2 * np.pi * delta / lambda_min / (2**int(neg_vals)) 

            # Store scaling so that we can recover the real magnitude later 
            self.scaling = lambda_min
        else: 
            # Fallback if eigenvalue bounds not available 
            delta = 1 / (2**nl)
            print("Warning: eigenvalue bounds not available; solution calculated up to scaling factor.") 
 
        # ------------------------------------------------------------
        # (5) RECIPROCAL / CONDITIONED ROTATION CIRCUIT
        # ------------------------------------------------------------
        if self.use_exact_reciprocal: 
            reciprocal = ExactReciprocal(nl, delta, neg_vals=neg_vals) 
            # Ancillas needed by A_op (if A_op defines them) 
            na = getattr(A_op, "num_ancillas", 0)
        else: 
            # Polynomial approximation approach 

            # Calculate breakpoints for the reciprocal approximation
            num_values = 2**nl 
            constant = delta 
            a = int(round(num_values ** (2/3)))
            
            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(abs(1 - (2 * constant / a) ** 2))
            degree = min(
                nb, 
                int(
                    np.log(
                        1 + (16.23 * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)
                           * kappa * (2 * kappa - self.eps_rotation)) / self.eps_rotation
                    )
                ),
            )
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            breakpoints = []
            for i in range(num_intervals): 
                # Add the breakpoint to the list
                breakpoints.append(a * (5**i))

                # Define the right breakpoint of the interval 
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)

            reciprocal = PiecewiseChebyshev(lambda x: np.arcsin(constant / x), degree, breakpoints, nl)
            na = max(getattr(A_op, "num_ancillas", 0), reciprocal.num_ancillas)

        # ------------------------------------------------------------
        # (6) REGISTERS: |b>/|x> register, phase register, ancillas, flag
        # ------------------------------------------------------------
        qb = QuantumRegister(nb, "state")       # RHS/solution register
        ql = QuantumRegister(nl, "phase")       # eigenvalue/phase register 
        qf = QuantumRegister(nf, "flag")        # success flag qubit 

        if na > 0:
            qa = AncillaRegister(na, "anc")     # ancilla qubits 
            qc = QuantumCircuit(qb, ql, qa, qf, name = "HHL") 
        else: 
            qc = QuantumCircuit(qb, ql, qf, name = "HHL") 

        # ------------------------------------------------------------
        # (7) ASSEMBLE HHL: prepare |b> -> QPE -> reciprocal rotation -> inverse QPE
        # ------------------------------------------------------------
        # State preparation 
        qc.append(b_circuit, qb[:]) 

        # QPE 
        phase_est = PhaseEstimation(nl, A_op)
        if na > 0: 
            qc.append(phase_est, ql[:] + qb[:] + qa[: getattr(A_op, "num_ancillas", 0)])
        else: 
            qc.append(phase_est, ql[:] + qb[:])

        # Conditioned rotation / reciprocal 
        if self.use_exact_reciprocal: 
            qc.append(reciprocal, ql[::-1] + [qf[0]])
        else: 
            qc.append(reciprocal.to_instruction(), ql[:] + [qf[0]] + qa[: reciprocal.num_ancillas])

        # Inverse QPE 
        if na > 0:
            qc.append(phase_est.inverse(), ql[:] + qb[:] + qa[: getattr(A_op, "num_ancillas", 0)])
        else:
            qc.append(phase_est.inverse(), ql[:] + qb[:])

        return qc

    @staticmethod 
    def get_solution_vector(solution, A, b) -> np.ndarray: 
        """
        Extract and nromalise the solution vector from the final HHL statevector. 

        Current assumption: 
            - the 'flag' qubit is the most significant qubit in the global statevector indexing 
            - the desired |x> amplitudes sit in the block where flag = |1> 

        Args: 
            solution: HHLResult-like object containing 'state' (statevector) array.
            A: numpy array (system matrix)
            b: numpy array (rhs vector) 

        Returns: 
            np.ndarray: extracted and rescaled solution vector x. 
        """
        full_state = Statevector(solution.state).data
        n_total = int(np.log2(len(full_state)))

        # Indices corresponding to flag=1 if the flag qubit is MSB: 
        start = 2 ** (n_total - 1) 
        end = start + len(b)

        x_block = full_state[start:end].real 

        # Rescale to best match Ax=b in Euclidean norm
        norm_factor = np.linalg.norm(A @ x_block) / np.linalg.norm(b) 
        return x_block / norm_factor
    
    @staticmethod
    def get_state_statevector(circuit: QuantumCircuit) -> np.ndarray: 
        """
        Simulate a circuit using statevector simulation and return the real part 
        """
        return np.real(Statevector(circuit).data) 
    
    def solve(self, matrix, vector) -> HHLResult: 
        """
        High-level solve method: 
            1) build HHL circuit 
            2) simulate it (Statevector) 
            3) extract classical solution vector

        Args:
            matrix: A matrix for the linear system
            vector: RHS vector of the linear system

        Returns:
            HHLResult: Dataclass with solution vector of the linear system 
        """
        solution = HHLResult

        solution.circuit = self.construct_circuit(matrix, vector) 
        solution.state = self.get_state_statevector(solution.circuit) 

        # Only valid if matrix/vector are numpy arrays here: 
        if isinstance(matrix, np.ndarray) and isinstance(vector, np.ndarray): 
            solution.solution = self.get_solution_vector(solution, matrix, vector) 
    
        return solution 
