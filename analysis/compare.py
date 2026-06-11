"""
Compare SimulatorOpt vs Simulator3: correctness (vs dense + vs each other) and speed.

Run:  python analysis/compare.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from circuit import random_MG, Cphase, mod_trotter_circuit
from circuit_utils import Z, tensor_identity
from simulator3 import Simulator3
from analysis.simulator_opt import SimulatorOpt

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def dense_expectation(circuit, N):
    psi = np.zeros(2 ** N, dtype=complex); psi[0] = 1.0
    for gtype, U, qubits in circuit:
        psi = tensor_identity(N, U, qubits) @ psi
    Z0 = tensor_identity(N, Z, 0)
    return np.real(np.conjugate(psi) @ (Z0 @ psi))


def correctness():
    print("=== correctness: SimulatorOpt vs dense vs Simulator3 ===")
    rng = np.random.default_rng(7)
    worst_dense = worst_cross = 0.0
    for _ in range(12):
        N = int(rng.integers(3, 6))
        pairs = [(i, i + 1) for i in range(N - 1)]
        circuit = []
        for _ in range(int(rng.integers(2, 8))):
            circuit.append(('MG', random_MG(), pairs[rng.integers(len(pairs))]))
        for _ in range(int(rng.integers(1, 4))):
            circuit.append(('CZ', Cphase(float(rng.random() * 2 * np.pi)), pairs[rng.integers(len(pairs))]))
        for _ in range(int(rng.integers(0, 2))):
            circuit.append(('SWAP', SWAP, pairs[rng.integers(len(pairs))]))
        rng.shuffle(circuit)
        dense = dense_expectation(circuit, N)
        s3 = Simulator3(N).simulate(circuit)
        so = SimulatorOpt(N).simulate(circuit)
        worst_dense = max(worst_dense, abs(dense - so))
        worst_cross = max(worst_cross, abs(s3 - so))
    print(f"  worst |opt - dense|      = {worst_dense:.2e}")
    print(f"  worst |opt - simulator3| = {worst_cross:.2e}")


def speed():
    print("\n=== speed: Simulator3 vs SimulatorOpt ===")
    # warmup both JITs
    SimulatorOpt(3).simulate([('MG', random_MG(), (0, 1))])
    Simulator3(3).simulate([('MG', random_MG(), (0, 1))])

    def t(simcls, circuit, n, reps=3):
        best = np.inf
        for _ in range(reps):
            s = simcls(n)
            t0 = time.perf_counter()
            s.simulate(circuit)
            best = min(best, time.perf_counter() - t0)
        return best

    rng = np.random.default_rng(0)
    print("\n-- pure matchgate brickwall (depth 4), _R-recompute dominated --")
    print(f"{'n':>4} {'gates':>6} {'sim3(s)':>10} {'opt(s)':>10} {'speedup':>8}")
    for n in [8, 12, 16, 20, 26]:
        pairs = [(i, i + 1) for i in range(n - 1)]
        c = []
        for _ in range(4):
            for q in pairs:
                c.append(('MG', random_MG(), q))
        a, b = t(Simulator3, c, n), t(SimulatorOpt, c, n)
        print(f"{n:>4} {len(c):>6} {a:>10.4f} {b:>10.4f} {a/b:>7.1f}x")

    print("\n-- Fermi-Hubbard Trotter (n_sites=5 -> 10 qubits), repeated gates --")
    print(f"{'steps':>5} {'gates':>6} {'sim3(s)':>10} {'opt(s)':>10} {'speedup':>8}")
    for d in range(2, 7):
        c = mod_trotter_circuit(5, d, 1, 0.3, 0.9, 1.0)
        a, b = t(Simulator3, c, 10, reps=2), t(SimulatorOpt, c, 10, reps=2)
        print(f"{d:>5} {len(c):>6} {a:>10.4f} {b:>10.4f} {a/b:>7.1f}x")


if __name__ == "__main__":
    correctness()
    speed()
