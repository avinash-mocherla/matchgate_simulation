"""
Plug-and-play example for the matchgate simulator.

Run:  python example.py        (first run takes a few seconds extra for JIT compile)

Shows the three things you need:
  1. Build a circuit  -- a list of ('MG'|'CZ'|'SWAP', U, (q1, q2)) tuples.
  2. Run it           -- Simulator(N).simulate(circuit, observable).
  3. Read the result  -- a float: <0...0| C^dag O C |0...0>.
"""
import numpy as np
from simulator import Simulator
from circuit import G, A, K, Cphase, random_MG, trotter_circuit

# ---------------------------------------------------------------- 1. minimal
N = 4
circuit = [
    ('MG', random_MG(), (0, 1)),     # random matchgate on qubits 0,1
    ('MG', K(0.3), (1, 2)),          # XX-rotation matchgate on qubits 1,2
    ('MG', G(A(0.5), A(1.2)), (2, 3)),  # G(A,B) matchgate on qubits 2,3
]
sim = Simulator(N)
value = sim.simulate(circuit)        # default observable: Z on qubit 0
print(f"<Z_0>           = {value:+.6f}")

# ------------------------------------------------- 2. choose your observable
value = sim.simulate(circuit, 'IZII')                  # Z on qubit 1
print(f"<Z_1>           = {value:+.6f}")

value = sim.simulate(circuit, {'ZIII': 0.5, 'IZII': 0.5})   # (Z_0 + Z_1)/2
print(f"<(Z_0+Z_1)/2>   = {value:+.6f}")

# ------------------------------- 3. beyond matchgates: CZ / CPhase and SWAP
circuit = [
    ('MG', random_MG(), (0, 1)),
    ('CZ', Cphase(0.7), (1, 3)),     # controlled-phase, any qubit pair
    ('SWAP', np.eye(4), (2, 3)),     # SWAP, any qubit pair (matrix unused)
    ('MG', random_MG(), (2, 3)),
]
value = Simulator(N).simulate(circuit)
print(f"with CZ + SWAP  = {value:+.6f}")

# ----------------------------------- 4. a physical circuit: Fermi-Hubbard
# trotter_circuit(N_sites, N_steps, J, U, tau) simulates N_sites spinful
# sites on 2*N_sites qubits (spin-up block then spin-down block).
sites, steps = 3, 2
circuit = trotter_circuit(sites, steps, J=1.0, U=2.0, tau=0.1)
value = Simulator(2 * sites).simulate(circuit)
print(f"Fermi-Hubbard   = {value:+.6f}  ({sites} sites, {steps} Trotter steps)")
