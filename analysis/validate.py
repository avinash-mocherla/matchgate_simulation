"""
Correctness + smoke test for the matchgate simulators.

Goal: confirm the njit/typed-Dict path runs in this environment (py3.14 + numba 0.65)
and that simulator3's expectation matches a dense statevector ground truth.

Run:  python analysis/validate.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from circuit_utils import I, X, Y, Z, kron_product, tensor_identity
from circuit import random_type_circuit, A, G, random_MG


def dense_expectation(circuit, N):
    """Exact <0^N| U^dag Z_0 U |0^N> via a 2^N statevector. Qubit 0 = MSB."""
    dim = 2 ** N
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for gtype, U, qubits in circuit:
        Ufull = tensor_identity(N, U, qubits)
        psi = Ufull @ psi
    Z0 = tensor_identity(N, Z, 0)
    return np.real(np.conjugate(psi) @ (Z0 @ psi))


def run_sim(circuit, N):
    from simulator import Simulator
    sim = Simulator(N)
    return sim.simulate(circuit, verbose=False)


def main():
    rng = np.random.default_rng(0)
    print("=== smoke test: does the njit path run? ===")
    N = 4
    circuit = [('MG', random_MG(), (0, 1)), ('MG', random_MG(), (1, 2)),
               ('MG', random_MG(), (2, 3)), ('MG', random_MG(), (0, 1))]
    t0 = time.time()
    val = run_sim(circuit, N)
    print(f"Simulator ran. value={val:.6f}  (first call incl. JIT compile: {time.time()-t0:.2f}s)")

    print("\n=== correctness: MG-only circuits vs dense statevector ===")
    max_abs_err = 0.0
    for trial in range(8):
        N = rng.integers(3, 6)
        ngates = rng.integers(3, 12)
        pairs = [(i, i + 1) for i in range(N - 1)]
        circuit = []
        for _ in range(ngates):
            q = pairs[rng.integers(len(pairs))]
            circuit.append(('MG', random_MG(), q))
        exact = dense_expectation(circuit, N)
        got = run_sim(circuit, N)
        err = abs(exact - got)
        max_abs_err = max(max_abs_err, err)
        flag = "OK " if err < 1e-9 else "!! "
        print(f"  {flag} N={N} gates={ngates:2d}  dense={exact:+.6f}  sim={got:+.6f}  |err|={err:.2e}")
    print(f"\nmax |err| over MG-only trials: {max_abs_err:.2e}")


if __name__ == "__main__":
    main()
