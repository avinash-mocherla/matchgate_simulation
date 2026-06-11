"""
Correctness of the NON-matchgate path (CZ / CPhase / SWAP) vs dense statevector.
This is the paper's core contribution, so it matters most.

Run:  python analysis/validate_cz.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from circuit_utils import Z, tensor_identity
from circuit import random_MG, Cphase
from simulator import Simulator

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def dense_expectation(circuit, N):
    dim = 2 ** N
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for gtype, U, qubits in circuit:
        psi = tensor_identity(N, U, qubits) @ psi
    Z0 = tensor_identity(N, Z, 0)
    return np.real(np.conjugate(psi) @ (Z0 @ psi))


def main():
    rng = np.random.default_rng(1)
    print("=== correctness: MG + CZ/CPhase + SWAP vs dense statevector ===")
    worst = 0.0
    for trial in range(12):
        N = int(rng.integers(3, 6))
        pairs = [(i, i + 1) for i in range(N - 1)]
        n_mg = int(rng.integers(2, 8))
        n_cz = int(rng.integers(1, 4))
        n_sw = int(rng.integers(0, 2))
        circuit = []
        for _ in range(n_mg):
            circuit.append(('MG', random_MG(), pairs[rng.integers(len(pairs))]))
        for _ in range(n_cz):
            phi = float(rng.random() * 2 * np.pi)
            circuit.append(('CZ', Cphase(phi), pairs[rng.integers(len(pairs))]))
        for _ in range(n_sw):
            circuit.append(('SWAP', SWAP, pairs[rng.integers(len(pairs))]))
        rng.shuffle(circuit)
        exact = dense_expectation(circuit, N)
        sim = Simulator(N)
        got = sim.simulate(circuit, verbose=False)
        err = abs(exact - got)
        worst = max(worst, err)
        flag = "OK " if err < 1e-9 else "!! "
        print(f"  {flag} N={N} mg={n_mg} cz={n_cz} sw={n_sw}  dense={exact:+.6f}  sim={got:+.6f}  |err|={err:.2e}")
    print(f"\nworst |err|: {worst:.2e}")


if __name__ == "__main__":
    main()
