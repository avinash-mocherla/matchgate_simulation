"""
Baseline benchmark + profiler for the matchgate simulator (simulator3, exact MITM).

- Scaling of wall-clock vs n_qubits for pure matchgate circuits (expect ~ n^2 rank).
- Scaling vs number of non-matchgates m (Trotter-like).
- cProfile of a representative run to locate Python-level hotspots.
- Microbenchmark: cost of recomputing the _R matrices every gate.

Run:  python analysis/bench.py
"""
import sys, os, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from circuit import random_MG, Cphase, brickwall, mod_trotter_circuit
from circuit_utils import linear_basis, quadratic_basis, cubic_basis
from simulator3 import Simulator3, _R

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def mg_circuit(n, depth, rng):
    pairs = [(i, i + 1) for i in range(n - 1)]
    c = []
    for _ in range(depth):
        for q in pairs:
            c.append(('MG', random_MG(), q))
    return c


def warmup():
    # trigger JIT compile so it doesn't pollute timings
    Simulator3(3).simulate([('MG', random_MG(), (0, 1))], verbose=False)


def time_call(circuit, n, repeats=3):
    best = np.inf
    for _ in range(repeats):
        sim = Simulator3(n)
        t = time.perf_counter()
        sim.simulate(circuit, verbose=False)
        best = min(best, time.perf_counter() - t)
    rank = max(sim.msm_lengths) if sim.msm_lengths else 0
    return best, rank


def scaling_pure_mg():
    print("\n=== pure matchgate: wall-clock vs n (depth=4 brickwall) ===")
    rng = np.random.default_rng(0)
    print(f"{'n':>4} {'gates':>7} {'maxrank':>9} {'time(s)':>10} {'us/gate':>9}")
    for n in [6, 8, 10, 12, 16, 20, 26]:
        c = mg_circuit(n, 4, rng)
        t, rank = time_call(c, n)
        print(f"{n:>4} {len(c):>7} {rank:>9} {t:>10.4f} {1e6*t/len(c):>9.1f}")


def scaling_trotter():
    print("\n=== Fermi-Hubbard Trotter: wall-clock vs steps (n_sites=5 -> 10 qubits) ===")
    print(f"{'steps':>6} {'gates':>7} {'maxrank':>9} {'time(s)':>10}")
    for d in range(0, 7):
        c = mod_trotter_circuit(5, d, 1, 0.3, 0.9, 1.0)
        t, rank = time_call(c, 10, repeats=2)
        print(f"{d:>6} {len(c):>7} {rank:>9} {t:>10.4f}")


def micro_R():
    print("\n=== microbenchmark: _R recomputation cost (per gate) ===")
    U = random_MG()
    N = 2000
    t = time.perf_counter()
    for _ in range(N):
        _R(U, linear_basis)
        _R(U, quadratic_basis)
        _R(U, cubic_basis)
    dt = (time.perf_counter() - t) / N
    print(f"  _R x3 (one gate's R-matrices): {1e6*dt:.1f} us/gate")
    print("  -> incurred fresh for EVERY gate occurrence, even repeated identical gates")


def profile_run():
    print("\n=== cProfile: Trotter n_sites=8 (16 qubits), 4 steps ===")
    c = mod_trotter_circuit(8, 4, 1, 0.3, 0.9, 1.0)
    sim = Simulator3(16)
    pr = cProfile.Profile()
    pr.enable()
    sim.simulate(c, verbose=False)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    warmup()
    scaling_pure_mg()
    scaling_trotter()
    micro_R()
    profile_run()
