# Matchgate Simulator — Analysis: Asymptotics, Performance, and Positioning

Analysis of the simulator in this repo (based on arXiv:2302.02654, *Extending Matchgate
Simulation Methods to Universal Quantum Circuits*, Mocherla–Lao–Browne). All performance
numbers below are measured on this machine (Python 3.14.3, numpy 2.4.3, numba 0.65.0) via
the scripts in `analysis/`.

## 0. What the simulator does (ground truth, verified)

It is a **Heisenberg-picture sparse Pauli/Majorana propagation** method:

- An operator on `n` qubits is a sparse map `int64 → float64` over the `4ⁿ` Pauli basis
  (base-4 digit per qubit, qubit 0 = MSB).
- A nearest-neighbour matchgate `G(A,B)` on qubits `(q,q+1)` acts **block-diagonally** on the
  16-dim 2-qubit Pauli space, split into the Majorana sectors the code calls
  `linear` (4), `quadratic` (6), `cubic` (4), with `{II, ZZ}` fixed. Matchgates **preserve
  Majorana degree** (free-fermion / Gaussian), so `Z₀` stays degree-2 and the support is `O(n²)`.
- Each **non-matchgate** (`CZ`/`CPhase`, or a non-adjacent `SWAP`) **raises the Majorana
  degree by ≤2**. After `m` of them, the support lives in the degree-≤2(m+1) sector, of size
  `χ = len_L(n,m) = Σ_{i=0}^{m} C(2n, 2i+2)`.
- **Meet-in-the-middle (MITM):** the measurement `Z₀` propagates backward, the input state
  forward; the loop always grows the *smaller* vector; the answer is their inner product.

**Correctness confirmed** to machine precision against a dense `2ⁿ` statevector, for both
pure-matchgate circuits and MG+CZ+SWAP circuits (`analysis/validate.py`,
`analysis/validate_cz.py`): worst `|sim − dense| = 4.4e-16`.

`simulator3.py` is the exact MITM simulator; `simulator4.py` adds magnitude thresholding
(approximate); `simulator5.py` only tracks rank growth; `simulator2.py` is a debug copy with
stray `print`s **inside the njit hot loop** (`get_cz_orbit`) — should be deleted/fixed.
`tests.py` does not run: it imports `from simulator import Simulator` and references
`Simulator2`, neither of which exists in the repo.

---

## 1. Asymptotics — is it improvable beyond `len_L`?

The runtime is `Θ(Σ_gates χ_t)` where `χ_t` is the live support size. The paper's headline
bounds follow from `χ ≤ len_L`:

- fixed `m`, `n→∞`:  `χ ~ C(2n,2m+2) ~ (2n)^{2m+2}/(2m+2)! = O((en/(m+1))^{2m+2})`
- `m = λn`:           `χ ~ 2^{2n·H((m+1)/n)}` (binomial entropy).

**Within the dense-Pauli representation, the rank `χ` is essentially worst-case tight:** a
*generic* CZ fills the whole degree-≤2(m+1) sector, so no bookkeeping trick beats `len_L` for
adversarial gates. But the *simulation cost* is improvable along five concrete axes:

**(a) The `len_L` bound is very loose in practice — locality/causal-cone.**
Measured max rank vs `len_L(10,m)` on Fermi–Hubbard Trotter (10 qubits, `analysis/profile_highrank.py`):

| #CZ `m` | measured max rank | `len_L(10,m)` | ratio |
|--------:|------------------:|--------------:|------:|
| 1 | 55 | 5 035 | 92× |
| 2 | 647 | 43 795 | 68× |
| 3 | 5 689 | 169 765 | 30× |
| 4 | 35 187 | 354 521 | 10× |
| 6 | 279 871 | 519 251 | 1.9× |

The degree increase is **localized to the modes the non-matchgates touch**. A bound based on
the union of the `m` gates' causal cones replaces `2n` with `O(#cone modes)`, which for
geometrically-local circuits is `≪ 2n`. This is both a tighter *theorem* and an exploitable
*pruning* (only sectors reachable from the cone are ever populated). The gap above is exactly
this effect (plus MITM), and it is largest precisely in the useful small-`m` regime.

**(b) Meet-in-the-middle gives a √ in the exponent — already implemented, not fully credited.**
Splitting the `m` non-matchgates across the cut means each side sees `~m/2`, so the dominant
term drops from `C(2n,2m+2)` to `~C(2n,m+2)` — the exponent roughly halves (`2m+2 → m+2`).
The code does grow the smaller side greedily, but the *stated* worst-case exponent (`2m+2`) is
the one-sided bound. Two cheap wins: (i) state the MITM-balanced bound; (ii) choose the cut by
**minimising over positions** `χ_left + χ_right` (balance non-matchgate count, not gate count)
— a precompute, helpful when the non-matchgates are clustered.

**(c) Representation change beats it asymptotically in the "magic-sparse" regime.**
Gaussian-rank / extent / branching methods keep the `n`-dependence at **fixed polynomial
degree** (covariance matrix, `O(n³)`) and put *all* the `m`-growth into a constant base `cᵐ`:

> `poly(n)·cᵐ`  vs  `(en/(m+1))^{2m+2}`.

For fixed `m` and `n→∞`, `poly(n)·cᵐ ≪ n^{2m+2}` — the dense-sector method is **asymptotically
dominated** when there are few non-matchgates on many qubits. The best verified constant is
`c = 4.5` per controlled-Z (Reardon-Smith–Oszmaniec–Korzekwa, arXiv:2307.12702). The
dense-sector bound becomes the *more honest / only meaningful* characterization when `m = Θ(n)`
(there `cᵐ` is itself `2^{Θ(n)}`, and `2^{2nH(λ)}` is the natural object).

**(d) Monte-Carlo over Pauli paths.** For an *additive-error* expectation value, importance-
sampling Pauli paths estimates the answer in `O(1/ε²)·(path cost)` instead of enumerating the
whole sector — far cheaper when `χ` is huge but only `⟨Z₀⟩ ± ε` is needed.

**(e) The `m=0` baseline is not optimal.** For pure nearest-neighbour matchgates a single Pauli
expectation is `O(n³)` (or `O(n²)` NN) via the covariance-matrix/Pfaffian formalism. The
sparse-Pauli method is competitive here but offers no asymptotic edge for a single observable;
its value is the *uniform* treatment of the `m>0` extension and Pauli-noise channels.

**Verdict.** `χ = len_L` is tight only for adversarial gates *in this representation*; the
*cost* is improvable (i) constant/polynomially via causal-cone pruning + optimal MITM cut
(easy, high-value, keeps exactness), and (ii) asymptotically via extent-branching or
path-sampling in the few-non-matchgate regime (different representation, loses worst-case
guarantee).

---

## 2. Performance — speedups (measured) and a benchmark harness

A reusable harness now exists: `validate.py`, `validate_cz.py` (correctness),
`bench.py` (scaling + cProfile), `compare.py` (head-to-head), `profile_highrank.py`.

**Profiling located two hotspots:**

1. **`_R(U, basis)` recomputed every gate, in numba *object mode* (`@jit(forceobj=True)`)**
   — ~250 µs per gate, and recomputed even for *identical* gates (Trotter reuses a handful of
   distinct gates). Dominates the low/medium-rank regime.
2. **`get_orbit` rebuilds a base-4 string in an `O(N)` Python loop and allocates a fresh
   `np.zeros(2N)` per index, per gate**; plus typed-dict hashing. Dominates the high-rank regime.

**`analysis/simulator_opt.py`** (`SimulatorOpt`) fixes both, behaviour-preserving
(bit-identical to `simulator3`, `|Δ| = 0`):

- `get_orbit`/`get_cz_orbit` → **O(1) bit-twiddling**:
  `subspace = (index >> shift) & 0xF`, `stem = index & ~(0xF << shift)`.
- `_R` → **vectorized einsum** (no Python loop) **+ memoized per unique gate**.

Measured speedups (`analysis/compare.py`):

| regime | speedup |
|---|---|
| pure matchgate brickwall (depth 4), n = 8…26 | **3.3–3.6×** |
| Fermi–Hubbard Trotter, 2…6 steps | **1.3–4.1×** |

**Remaining bottleneck (high rank):** after the fix, `_R` is negligible (cache: 45 calls)
and **96% of time is the njit `main_loop` doing typed-dict gather/scatter over `χ` indices.**
The next, larger lever (not yet implemented):

- **Drop the hash map.** Store coefficients in contiguous arrays under a canonical index
  layout so each orbit is a *strided slice* → vectorized gather/scatter.
- **Sector-batched BLAS.** All orbits of one type share the same `k×k` R-block; stack their
  sub-vectors into a `k×K` matrix and do **one** `R @ M` instead of `K` tiny matmuls.
- **`prange` parallelism.** Orbits within a gate touch disjoint indices → embarrassingly
  parallel (there is already a `multiprocessing_tut.py` hinting at this).
- Smaller: precompute `U.conj().T` once; snapshot keys to an array instead of `dict.copy()`;
  add `cache=True` to all njit fns; keep `simulator4`'s thresholding for approximate runs.

**Repo hygiene:** consolidate the four near-duplicate `simulator*.py` into one
(`Simulator(exact=True/threshold=ε)`); delete the debug `print`s in `simulator2.py`; repair or
replace `tests.py` so the figures are reproducible.

---

## 3. Where this sits in the literature

**Foundational (matchgate = free fermion), `m=0` baseline `O(n³)`:**
Valiant (SIAM J. Comput. 2002); Knill, quant-ph/0108033; Terhal–DiVincenzo, quant-ph/0108010;
Bravyi (Lagrangian/Grassmann), quant-ph/0404180; Jozsa–Miyake, arXiv:0804.4050.

**Matchgates + non-NN / universality:** Brod–Galvão, arXiv:1106.1863; Brod–Childs,
arXiv:1308.1463; Brod (generalized inputs/measurements), arXiv:1602.03539.

**Direct competitors — free fermions + a few non-Gaussian gates, `poly(n)·cᵐ`:**
- Hebenstreit–Jozsa–Kraus–Strelchuk–Yoganathan, arXiv:1905.08584 — fermionic magic states.
- Cudby–Strelchuk, arXiv:2307.12654 — Gaussian rank / Gaussian extent.
- Dias–Koenig, arXiv:2307.12912 (Quantum 2024) — phase-sensitive covariance method, poly(n)·extent.
- **Reardon-Smith–Oszmaniec–Korzekwa, arXiv:2307.12702 (Quantum 2024) — `O(4.5ᵏ)` for k CZ
  gates (≈2.12/gate at amplitude level), improving prior ~9/swap. The most direct competitor.**
- Dias–Bosse–Seddon, arXiv:2603.18869 (2026) — provably optimal extent decompositions.

**General framework (this method is a *structured special case*):**
Rall–Liang–Cook–Kretschmer (Pauli propagation), arXiv:1901.09070; Begušić–Gray–Chan,
arXiv:2306.16372 / 2308.05077; **Begušić–Chan, Sparse Pauli Dynamics, arXiv:2409.03097**;
Angrisani et al. (average-case poly-time), arXiv:2409.01706; Rudolph et al. (framework),
arXiv:2505.21606; Aharonov et al. (noisy RCS poly-time), arXiv:2211.03999.

**Closest sibling — same family, independent and ~contemporaneous:**
**Miller et al., "Simulation of Fermionic circuits using Majorana Propagation,"
arXiv:2503.18939 (2025)** — Heisenberg-picture Majorana-operator propagation with
**monomial-length truncation**, demonstrated to 52 fermionic modes, "orders of magnitude"
faster than tensor-network simulators on chemistry circuits.

**Positioning.** This method's distinguishing feature is that it tracks the *entire*
Majorana-degree-≤2(m+1) sector **densely and exactly**, with an a-priori worst-case bound `χ`
that needs no cancellation, sparsity, or favourable decomposition. The price: `poly(n)·cᵐ`
extent/branching methods asymptotically dominate it for **few non-matchgates on many qubits**,
and adaptive-truncation SPD / Majorana-propagation are typically far cheaper in practice on
structured circuits — but those give up the closed-form worst-case `χ`. The method's natural
niche is **exactness + worst-case guarantee + the `m = Θ(n)` regime**, and as a clean,
provably-bounded special case of Pauli/Majorana propagation. The highest-value research
direction is to **hybridize**: keep the exact block-diagonal matchgate action, but add
(i) causal-cone sector pruning and (ii) optional adaptive truncation (à la SPD / Miller et al.)
to push past the dense-sector wall while retaining the structure that makes this method clean.
