"""
Canonical exact matchgate simulator (Heisenberg-picture sparse Pauli/Majorana
propagation, meet-in-the-middle scheduling) for arXiv:2302.02654.

This is the consolidated, optimized successor to ``simulator3.Simulator3``. It is
bit-identical to Simulator3 (worst |sim - dense| <= 1e-15 on the validate.py /
validate_cz.py suites) and 1.3-4x faster. Two behaviour-preserving optimizations
vs Simulator3:

  (1) get_orbit / get_cz_orbit use O(1) bit-twiddling instead of an O(N) base-4
      string reconstruction with a fresh np.zeros(2N) allocation per index.
  (2) The _R Majorana-transform matrices are memoized per (basis, gate), so the
      repeated gates of a Trotter circuit are transformed once, not every time.

Everything else (MITM scheduling, statevector init, expectation) matches
Simulator3 exactly.

Public API:
    Simulator(N)          -- the canonical exact simulator (use this)
    pauli_index(s)        -- 'ZII' -> int64 index used internally for Pauli strings
    pauli_vector(obs)     -- 'ZII' or {'ZII': 0.5, 'XXI': 0.5} -> typed-Dict vector
    support_bound(N, s)   -- worst-case Pauli-support size for s CZ/SWAP gates
    CircuitComplexityError -- raised when a circuit exceeds the max_terms guardrail
    SimulatorOpt          -- alias of Simulator, kept for the analysis/ harness

For the approximate (magnitude-thresholded) and rank-tracking variants, see
simulator4.Simulator4 and simulator5.Simulator5. For the one-sided Heisenberg-only
variant used in the Figure 6 rank-growth comparison, see simulator2.Simulator2.
"""
import itertools
import math
import numpy as np
from circuit_utils import (linear, quadratic, cubic, linear_cz, quadratic_cz, cubic_cz,
                           linear_basis, quadratic_basis, cubic_basis,
                           linear_basis_cz, quadratic_basis_cz, cubic_basis_cz)
from numba import njit
from numba.typed import Dict
from numba.core import types

# precomputed orbit-target arrays for the CZ branch (avoid rebuilding per call)
_CZ_L = (np.array([0, 0, 3, 3], np.int64), np.array([1, 2, 1, 2], np.int64))
_CZ_Q = (np.array([1, 1, 2, 2], np.int64), np.array([0, 3, 0, 3], np.int64))
_CZ_C = (np.array([1, 1, 2, 2], np.int64), np.array([1, 2, 1, 2], np.int64))


@njit(cache=True)
def dot(b, A):
    return np.dot(b, A.T)


@njit(cache=True)
def dot1(A, b):
    return np.dot(A, b)


@njit(cache=True)
def get_orbit(N, index, q):
    shift = 2 * (N - q - 2)
    subspace_group = (index >> shift) & np.int64(15)
    stem = index & ~(np.int64(15) << shift)
    mult = np.int64(1) << shift          # 4**(N-q-2)
    if subspace_group == 4 or subspace_group == 8 or subspace_group == 13 or subspace_group == 14:
        return 0, stem + mult * linear
    elif (subspace_group == 3 or subspace_group == 5 or subspace_group == 6
          or subspace_group == 9 or subspace_group == 10 or subspace_group == 12):
        return 1, stem + mult * quadratic
    elif subspace_group == 1 or subspace_group == 2 or subspace_group == 7 or subspace_group == 11:
        return 2, stem + mult * cubic
    else:
        return 3, np.array([index])


@njit(cache=True)
def get_cz_orbit(N, index, q1, q2, L0, L1, Q0, Q1, C0, C1):
    s1 = 2 * (N - q1 - 1)
    s2 = 2 * (N - q2 - 1)
    sg1 = (index >> s1) & np.int64(3)
    sg2 = (index >> s2) & np.int64(3)
    subspace_group = sg1 * 4 + sg2
    m1 = np.int64(1) << s1
    m2 = np.int64(1) << s2
    stem = index - (sg1 * m1) - (sg2 * m2)
    if subspace_group == 1 or subspace_group == 2 or subspace_group == 13 or subspace_group == 14:
        return 0, stem + m1 * L0 + m2 * L1
    elif subspace_group == 4 or subspace_group == 7 or subspace_group == 8 or subspace_group == 11:
        return 1, stem + m1 * Q0 + m2 * Q1
    elif subspace_group == 5 or subspace_group == 6 or subspace_group == 9 or subspace_group == 10:
        return 2, stem + m1 * C0 + m2 * C1
    else:
        return 3, np.array([index])


@njit(cache=True)
def main_loop(N, q, d, R0, R1, R2, flag):
    visited = set()
    for index in d.copy():
        if index not in visited:
            key, orbit = get_orbit(N, index, q)
            for x in orbit:
                visited.add(x)
            if key != 3:
                R = R0 if key == 0 else (R1 if key == 1 else R2)
                read_and_write(d, R, orbit, flag)


@njit(cache=True)
def main_loop_cz(N, q1, q2, d, R0, R1, R2, flag, L0, L1, Q0, Q1, C0, C1):
    visited = set()
    for index in d.copy():
        if index not in visited:
            key, orbit = get_cz_orbit(N, index, q1, q2, L0, L1, Q0, Q1, C0, C1)
            for x in orbit:
                visited.add(x)
            if key != 3:
                R = R0 if key == 0 else (R1 if key == 1 else R2)
                read_and_write(d, R, orbit, flag)


@njit(cache=True)
def read_and_write(d, current_R, orbit, flag):
    sub = np.zeros(len(current_R))
    for i, idx in enumerate(orbit):
        sub[i] = d.get(idx, 0.0)
    sub = dot(sub, current_R) if flag == 1 else dot1(current_R, sub)
    for i, idx in enumerate(orbit):
        if sub[i] != 0.0:
            d[idx] = sub[i]


@njit(cache=True)
def expectation(measurement_vector, rho_vector):
    acc = 0.0
    for key, value in measurement_vector.items():
        acc += rho_vector.get(key, 0.0) * value
    return acc


class CircuitComplexityError(Exception):
    """Raised when a circuit's worst-case cost exceeds the max_terms guardrail."""


#: Default cap on Pauli-vector terms (per side). 2**24 ~ 17M typed-dict entries
#: ~ a few GB of RAM. Pass max_terms=... to simulate() to raise it, or
#: max_terms=None to disable the guardrail entirely.
MAX_TERMS_DEFAULT = 2 ** 24


def support_bound(N, num_swaps):
    """Worst-case Pauli support per MITM side: chi = sum_i C(2N, 2i+2).

    Cost is exponential in the number of non-matchgate gates (CZ/CPhase/SWAP),
    not in qubit count. The meet-in-the-middle split absorbs about half of
    them on each side, hence m = ceil(num_swaps / 2). Capped at 4**N (the
    full Pauli basis), so small-N circuits are never rejected.
    """
    m = (num_swaps + 1) // 2
    chi = sum(math.comb(2 * N, 2 * i + 2) for i in range(m + 1))
    return min(chi, 4 ** N)


_PAULI_DIGIT = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


def pauli_index(pauli_string):
    """Map a Pauli string like 'ZII' (qubit 0 first) to its int64 vector index.

    Each qubit is a base-4 digit (I=0, X=1, Y=2, Z=3); qubit 0 is the most
    significant digit, so for N qubits: index = sum_q digit(q) * 4**(N-1-q).
    """
    idx = 0
    for ch in pauli_string.upper():
        idx = idx * 4 + _PAULI_DIGIT[ch]
    return np.int64(idx)


def pauli_vector(observable):
    """Build a typed-Dict Pauli vector from a friendly observable spec.

    Accepts a single Pauli string ('ZII'), a dict of real linear combinations
    ({'ZII': 0.5, 'XXI': 0.5}), or a dict keyed by precomputed integer indices.
    Identity coefficients are not supported (tr(rho)=1 terms cancel in <O>).
    """
    if isinstance(observable, str):
        observable = {observable: 1.0}
    vec = Dict.empty(key_type=types.int64, value_type=types.float64)
    for key, coeff in observable.items():
        idx = pauli_index(key) if isinstance(key, str) else np.int64(key)
        if idx == 0:
            raise ValueError("identity term not supported in a Pauli vector")
        vec[idx] = float(coeff)
    return vec


_BASIS_STACK = {}  # cache the (k,4,4) stacked basis arrays


def _R(U, basis):
    # R[i,j] = Re Tr(U^dag b_i U b_j)/4, fully vectorized (no Python loop).
    bid = id(basis)
    B = _BASIS_STACK.get(bid)
    if B is None:
        B = np.stack([np.asarray(b) for b in basis])
        _BASIS_STACK[bid] = B
    M = U.conj().T @ B @ U                      # (k,4,4)
    return np.real(np.einsum('iab,jba->ij', M, B)) / 4


class Simulator:
    """Exact meet-in-the-middle matchgate simulator (canonical)."""

    def __init__(self, N):
        self.N = N
        self.msm_lengths = []
        self.rho_lengths = []
        self._Rcache = {}

    def _get_R(self, U, basis, basis_tag):
        key = (basis_tag, U.tobytes())
        R = self._Rcache.get(key)
        if R is None:
            R = _R(U, basis)
            self._Rcache[key] = R
        return R

    def init_statevector(self):
        Z_indices = [3 * 4 ** i for i in range(self.N)]
        indices = []
        for i in range(1, self.num_swaps + 2):
            indices.extend([sum(j) for j in itertools.combinations(Z_indices, i)])
        sv = Dict.empty(key_type=types.int64, value_type=types.float64)
        for i in indices:
            sv[i] = 1.0
        return sv

    def simulate(self, circuit, measurement_vector=None, rho_vector=None, verbose=False,
                 max_terms=MAX_TERMS_DEFAULT):
        # plug-and-play: accept a Pauli string or {pauli_string: coeff} dict
        if measurement_vector is not None and not isinstance(measurement_vector, Dict):
            measurement_vector = pauli_vector(measurement_vector)
        if rho_vector is not None and not isinstance(rho_vector, Dict):
            rho_vector = pauli_vector(rho_vector)
        self.num_swaps = sum(1 for i in circuit if i[0] in ('SWAP', 'CZ'))
        if max_terms is not None:
            bound = support_bound(self.N, self.num_swaps)
            if bound > max_terms:
                raise CircuitComplexityError(
                    f"Circuit is too complex to simulate exactly: N={self.N} qubits with "
                    f"{self.num_swaps} non-matchgate gates (CZ/CPhase/SWAP) has a worst-case "
                    f"Pauli support of ~{bound:.2e} terms per side, above max_terms={max_terms:.2e}. "
                    f"Each CZ/SWAP multiplies the cost; matchgates ('MG') are nearly free. "
                    f"Options: reduce CZ/SWAP count, pass max_terms=<bigger int> if you have the "
                    f"RAM (~100 bytes/term), pass max_terms=None to disable this check, or use "
                    f"the approximate simulator4.Simulator4 with a magnitude threshold.")
        if rho_vector is None:
            rho_vector = self.init_statevector()
            self.rho_lengths.append(len(rho_vector))
        if measurement_vector is None:
            z = np.int64(2 ** (2 * self.N - 1) + 2 ** (2 * self.N - 2))
            measurement_vector = Dict.empty(key_type=types.int64, value_type=types.float64)
            measurement_vector[z] = 1.0
            self.msm_lengths.append(len(measurement_vector))
        xi = xj = 0
        self.circuit = circuit
        self.num_gates = len(circuit)
        while (xi + xj) <= self.num_gates - 1:
            if verbose:
                print('Progress:', np.round((xi + xj) / self.num_gates, 2) * 100, '%')
            if max_terms is not None and min(len(measurement_vector), len(rho_vector)) > max_terms:
                raise CircuitComplexityError(
                    f"Aborting mid-simulation: after {xi + xj}/{self.num_gates} gates both Pauli "
                    f"vectors exceed max_terms={max_terms:.2e} "
                    f"(measurement: {len(measurement_vector)}, rho: {len(rho_vector)} terms). "
                    f"Pass a larger max_terms (or max_terms=None) to push on, or use the "
                    f"approximate simulator4.Simulator4.")
            if len(measurement_vector) <= len(rho_vector):
                self.apply_gate(measurement_vector, -(xj + 1), flag=1)
                xj += 1
                self.msm_lengths.append(len(measurement_vector))
            else:
                self.apply_gate(rho_vector, xi, flag=0)
                xi += 1
                self.rho_lengths.append(len(rho_vector))
        return expectation(measurement_vector, rho_vector)

    def apply_gate(self, d, x, flag):
        gate_type, U, qubits = self.circuit[x]
        if gate_type == 'SWAP':
            self.apply_SWAP_gate(d, qubits)
        elif gate_type == 'CZ':
            if flag == 1:
                U = U.conj().T
            R0 = self._get_R(U, linear_basis_cz, 3)
            R1 = self._get_R(U, quadratic_basis_cz, 4)
            R2 = self._get_R(U, cubic_basis_cz, 5)
            main_loop_cz(self.N, qubits[0], qubits[1], d, R0, R1, R2, flag,
                         _CZ_L[0], _CZ_L[1], _CZ_Q[0], _CZ_Q[1], _CZ_C[0], _CZ_C[1])
        elif gate_type == 'MG':
            if flag == 1:
                U = U.conj().T
            R0 = self._get_R(U, linear_basis, 0)
            R1 = self._get_R(U, quadratic_basis, 1)
            R2 = self._get_R(U, cubic_basis, 2)
            main_loop(self.N, qubits[0], d, R0, R1, R2, flag)

    def apply_SWAP_gate(self, statevector, qubits):
        visited = set()
        for index in statevector.copy():
            if index not in visited:
                b = format(index, "0" + str(2 * self.N) + "b")
                s = (b[0:2 * qubits[0]] + b[2 * qubits[1]:2 * qubits[1] + 2]
                     + b[2 * qubits[0] + 2:2 * qubits[1]] + b[2 * qubits[0]:2 * qubits[0] + 2]
                     + b[2 * qubits[1] + 2:2 * self.N])
                new_index = int(s, 2)
                visited.add(new_index)
                statevector[new_index], statevector[index] = statevector[index], statevector.get(new_index, 0)
        return statevector


# Backward-compatible alias for the analysis/ harness (bench.py, compare.py, ...).
SimulatorOpt = Simulator
