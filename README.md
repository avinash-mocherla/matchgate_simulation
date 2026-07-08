# Matchgate Simulation

Exact classical simulator for **matchgate circuits with a bounded number of non-matchgate gates** (CZ / controlled-phase / SWAP), using Heisenberg-picture sparse Pauli propagation with meet-in-the-middle scheduling. Reference implementation for [arXiv:2302.02654](https://arxiv.org/abs/2302.02654).

Cost is **polynomial in qubit count** and **exponential only in the number of non-matchgate gates** — so a 50-qubit circuit with thousands of matchgates and a handful of CZs is cheap, while 20 CZs is not.

## Install and run

```bash
pip install -r requirements.txt   # numpy, scipy, numba, tqdm, matplotlib
python example.py                 # end-to-end demo (first run adds a few seconds of JIT compile)
python analysis/validate.py       # checks results against dense statevector (errors ~1e-16)
```

Requires Python ≥ 3.10. Tested on Python 3.14 / numpy 2.4 / numba 0.65.

## Quickstart

```python
from simulator import Simulator
from circuit import G, A, K, Cphase, random_MG

circuit = [
    ('MG', random_MG(), (0, 1)),   # matchgate on neighbouring qubits 0,1
    ('CZ', Cphase(0.7), (1, 3)),   # controlled-phase on any pair
    ('MG', K(0.3),      (2, 3)),
]
sim = Simulator(4)                          # 4 qubits
value = sim.simulate(circuit)               # <Z_0> by default
value = sim.simulate(circuit, 'IZII')       # <Z_1>
value = sim.simulate(circuit, {'ZIII': 0.5, 'IZII': 0.5})   # any real Pauli combo
```

The return value is the float `⟨0…0| C† O C |0…0⟩`.

---

## Usage contract (read this before writing code against the API)

This section states the exact rules. Everything here is load-bearing.

### The simulator

```python
from simulator import Simulator
sim = Simulator(N)                # N = number of qubits
value = sim.simulate(circuit, measurement_vector=None, rho_vector=None,
                     verbose=False, max_terms=2**24)
```

- **Initial state is always `|0…0⟩`** (as a Heisenberg-picture density-matrix expansion built internally). You do not prepare states; you prepend state-preparation gates to the circuit instead.
- **Returns** a Python float: the exact expectation value of the observable after the circuit.
- `Simulator` instances are reusable; `simulate` can be called repeatedly with different circuits/observables. `sim.msm_lengths` / `sim.rho_lengths` record Pauli-rank growth of the last run (used for the paper's figures).

### Circuit format

A circuit is a **Python list of 3-tuples** `(gate_type, U, (q1, q2))`, applied left-to-right (index 0 acts first on the state):

| `gate_type` | `U` | `(q1, q2)` constraint | meaning |
|---|---|---|---|
| `'MG'` | 4×4 complex matchgate `G(A, B)` with `det A == det B` | **must be nearest-neighbour**: `q2 == q1 + 1` | fermionic-linear-optics gate (free) |
| `'CZ'` | 4×4 controlled-phase-type gate, e.g. `Cphase(phi)` or the standard `CZ` | any pair `q1 < q2` | interaction gate (each one multiplies cost) |
| `'SWAP'` | ignored (pass `np.eye(4)`) | any pair `q1 < q2` | qubit relabelling (each one multiplies cost) |

Qubits are indexed `0 … N-1`. For a matchgate acting on a **non**-adjacent pair, do not pass it directly — route it with `circuit.convert_non_nn_ladder([...])`, which inserts SWAPs (and pays their cost).

Gate constructors live in `circuit.py`:

```python
from circuit import G, A, K, Cphase, random_MG, haar_random_MG
G(A_mat, B_mat)        # generic matchgate from two 2x2 blocks (use det A == det B)
A(theta)               # 2x2 real rotation block
K(theta)               # XX-rotation matchgate G(I, R_xx(theta)) — Trotter hopping term
Cphase(phi)            # diag(1, 1, 1, e^{i phi}); Cphase(pi) == CZ
random_MG()            # random matchgate
haar_random_MG()       # Haar-random matchgate
```

Circuit generators (also in `circuit.py`): `trotter_circuit(N_sites, N_steps, J, U, tau)` for spinful Fermi–Hubbard on `2*N_sites` qubits, `givens_circuit(N)`, `brickwall(N, depth)`, `random_type_circuit(...)`.

### Observables

The `measurement_vector` argument (2nd positional) accepts, in order of convenience:

1. `None` → defaults to `Z` on qubit 0.
2. A **Pauli string** of length N, qubit 0 leftmost: `'IZII'` = Z on qubit 1 of 4.
3. A **dict of real coefficients**: `{'ZIII': 0.5, 'XXII': 0.5}` = `0.5·Z₀ + 0.5·X₀X₁`. Identity strings (`'III…'`) are rejected — constant offsets must be added by the caller.
4. (advanced) a prebuilt numba typed `Dict[int64, float64]` from `pauli_vector(...)`; integer keys encode Pauli strings base-4, qubit 0 most significant, `I=0, X=1, Y=2, Z=3` (see `pauli_index`).

`rho_vector` (3rd positional) overrides the initial state's Pauli expansion — leave it `None` unless you know exactly what you are doing.

### Complexity guardrail

`simulate` estimates the worst-case Pauli support before running: `support_bound(N, s) = Σᵢ₌₀^⌈s/2⌉ C(2N, 2i+2)`, capped at `4^N`, where `s` = number of `'CZ'`+`'SWAP'` gates. If it exceeds `max_terms` (default `2**24` ≈ 1.7×10⁷ terms ≈ a few GB of RAM), it raises `CircuitComplexityError` immediately instead of hanging; a second runtime check aborts mid-run if both Pauli vectors actually outgrow `max_terms`.

- Pass `max_terms=<bigger int>` if you have the RAM (~100 bytes/term), or `max_terms=None` to disable.
- The bound is a worst case; real circuits with local structure (causal cones) often stay far below it. If the upfront check trips but you believe the circuit is benign, try `max_terms=None` with a small test first.
- Or switch to `simulator4.Simulator4`, which truncates small-magnitude Pauli terms (approximate, much cheaper).

### Things that will silently go wrong if ignored

- **`'MG'` on non-adjacent qubits** is not validated — it indexes the wrong bits and returns garbage. Keep matchgates nearest-neighbour or use `convert_non_nn_ladder`.
- **`U` for `'MG'` must actually be a matchgate** (`G(A,B)` block structure, `det A == det B`). Arbitrary 4×4 unitaries are not checked and give wrong answers.
- **`U` for `'CZ'` must be controlled-phase-type** (diagonal in the computational basis up to the matchgate-compatible form used in the paper, e.g. `Cphase(phi)`). A generic two-qubit gate is not supported.
- **Observables are real linear combinations of Pauli strings.** Complex coefficients are truncated to float.
- The first `simulate` call in a fresh process pays ~1–8 s of numba JIT compilation (cached afterwards in `__pycache__`).

## Repository map

| file | role |
|---|---|
| `simulator.py` | **the canonical exact simulator — use this** (`Simulator`, `pauli_index`, `pauli_vector`, `support_bound`, `CircuitComplexityError`) |
| `circuit.py` | gate constructors and circuit generators |
| `circuit_utils.py` | Pauli/orbit tables shared by all simulators |
| `example.py` | runnable quickstart covering the whole public API |
| `simulator2.py` | one-sided Heisenberg variant — only for the Fig. 6 rank-growth comparison |
| `simulator3.py` | previous-generation exact simulator (superseded by `simulator.py`, kept for reproducibility) |
| `simulator4.py` | approximate simulator with magnitude thresholding |
| `simulator5.py` | rank-tracking-only variant |
| `tests.py` | figure-generating scripts for the paper (long-running) |
| `analysis/` | validation (`validate.py`, `validate_cz.py`), benchmarks (`bench.py`, `compare.py`), notes (`ANALYSIS.md`) |
| `data/*.npy` | cached data behind the paper's figures |

## How it works (one paragraph)

Both the observable `O` and the initial state `ρ = |0…0⟩⟨0…0|` are stored as sparse real vectors over the N-qubit Pauli basis. Matchgates conjugate Paulis within closed orbits of at most 6 basis elements (the Majorana linear/quadratic/cubic sectors), so each `'MG'` gate is a few tiny dense matmuls on the sparse support — no growth. `'CZ'`/`'SWAP'` gates map single Paulis to sums, growing the support; the simulator therefore plays the circuit from **both ends** (gates pushed onto `O` from the back, onto `ρ` from the front), always advancing the currently-smaller side, and meets in the middle where `⟨O' ρ'⟩` is a sparse dot product. The hot loops are numba-`@njit`-compiled over typed dicts.

## Citation

```bibtex
@article{mocherla2023extending,
  title={Extending Matchgate Simulation Methods to Universal Quantum Circuits},
  author={Mocherla, Avinash and Lao, Lingling and Browne, Dan E.},
  journal={arXiv preprint arXiv:2302.02654},
  year={2023}
}
```
