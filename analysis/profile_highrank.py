"""Where does time go in the high-rank regime, and does measured rank match len_L?"""
import sys, os, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from circuit import mod_trotter_circuit, random_MG
from circuit_utils import len_L
from analysis.simulator_opt import SimulatorOpt

# warmup
SimulatorOpt(3).simulate([('MG', random_MG(), (0, 1))])

print("=== measured max rank vs paper's len_L bound (Trotter, n_sites=5, 10 qubits) ===")
print(f"{'steps':>5} {'#CZ(m)':>7} {'measured_maxrank':>17} {'len_L(10,m)':>14}")
for d in range(0, 7):
    c = mod_trotter_circuit(5, d, 1, 0.3, 0.9, 1.0)
    m = sum(1 for g in c if g[0] in ('CZ', 'SWAP'))
    s = SimulatorOpt(10)
    s.simulate(c)
    mr = max(s.msm_lengths + s.rho_lengths)
    print(f"{d:>5} {m:>7} {mr:>17} {len_L(10, m):>14.0f}")

print("\n=== cProfile: OPT at Trotter n_sites=8 (16 qubits), 4 steps (high rank) ===")
c = mod_trotter_circuit(8, 4, 1, 0.3, 0.9, 1.0)
s = SimulatorOpt(16)
pr = cProfile.Profile(); pr.enable()
s.simulate(c); pr.disable()
out = io.StringIO()
pstats.Stats(pr, stream=out).sort_stats('tottime').print_stats(10)
print(out.getvalue())
