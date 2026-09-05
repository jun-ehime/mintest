# mintest

Minimum-size test design over a **Q-matrix** (an `items x skills` binary
matrix used in cognitive diagnosis / knowledge-component modeling).

`mintest` provides exact and approximate solvers for two related
combinatorial problems:

1. **Minimum r-cover**: the smallest subset of items such that every
   skill (knowledge component, KC) is exercised by at least `r`
   selected items (`r=1` is classical Set Cover).
2. **r-robust diagnostic separation**: the smallest subset of items
   such that every pair of knowledge states in the single-KC-deficit
   diagnostic model is distinguished by at least `r` selected items.

For each problem, three interchangeable solvers are provided:

| Solver     | Guarantee                              | Module              |
|------------|-----------------------------------------|----------------------|
| `greedy_*` | polynomial time, `1+ln(n_skills)`-approx (r=1) | `mintest.cover`, `mintest.separation` |
| `ilp_*`    | exact (PuLP / CBC)                      | `mintest.cover`      |
| `maxsat_*` | exact (PySAT / RC2 Partial MaxSAT)      | `mintest.cover`, `mintest.separation` (as `exact_separation`) |

The r-robust separation solver uses a pattern-multiplicity reduction
(cap each distinct item pattern's usable copies at `min(count, r)`)
that provably preserves the optimum while keeping the SAT encoding
small even when the item bank has many duplicate skill patterns; see
the docstring of `mintest.separation` for the proof sketch, and
`tests/test_separation.py::test_reduction_matches_bruteforce` for a
brute-force cross-check.

## Installation

```bash
pip install -e .            # from a checkout
# or, once published:
pip install mintest
```

Dependencies: `numpy`, `scipy`, `python-sat` (PySAT/RC2), `pulp`
(CBC by default, no external solver install required).

## Quick start (Python API)

```python
from mintest.qmatrix import synthetic_q_matrix, compute_stats
from mintest.cover import greedy_cover, ilp_cover, maxsat_cover
from mintest.separation import exact_separation, greedy_separation

Q = synthetic_q_matrix(n_items=200, n_skills=30, seed=0)
print(compute_stats(Q))

# Minimum KC cover
greedy = greedy_cover(Q, r=1)
exact = ilp_cover(Q, r=1)         # or maxsat_cover(Q, r=1)
print(greedy["n_selected"], exact["n_selected"])

# r-robust diagnostic separation
sep_r1 = exact_separation(Q, r=1)
sep_r2 = exact_separation(Q, r=2)
print(sep_r1["optimal"], sep_r2["optimal"])
```

## Command-line interface

```bash
mintest stats    --npz data/q_matrix_assistments.npz
mintest cover    --npz data/q_matrix_assistments.npz --method ilp --r 1
mintest separate --npz data/q_matrix_assistments.npz --method exact --r 2
```

## Reproducing the paper's ASSISTments results

The bundled `data/q_matrix_assistments.npz` is the Q-matrix built from
the public **ASSISTments 2009-2010 Skill Builder** dataset
(https://sites.google.com/site/assistmentsdata/, CC BY 4.0): 16,891
items x 110 skills, containing only the `(problem_id, skill_name)`
structure (no student response data).

```bash
python examples/assistments.py
```

Actual output from this repository's test run (2026-09-05, CBC 2.10 /
PySAT RC2, macOS, Python 3.9.6):

```
=== ASSISTments 2009-2010 Skill Builder ===
  items:            16,891
  skills:           110
  nonzero entries:  20,224
  sparsity:         0.010885
  uncovered skills: 0
  unique item patterns: 136
  skills/item  mean=1.20 median=1.0 min=1 max=4
  items/skill  mean=183.9 median=122.5 min=1 max=1040

--- Minimum KC cover (r=1) ---
greedy:  n_selected=88  coverage=1.0000
ilp:     n_selected=87  status=Optimal
maxsat:  n_selected=87  status=OPTIMAL

--- r-robust diagnostic separation ---
r=1 exact: optimal=97  status=OPTIMAL
r=2 exact: optimal=183  status=OPTIMAL

--- Comparison against expected paper values ---
  cover_ilp                expected=   87  got=   87  OK
  cover_maxsat             expected=   87  got=   87  OK
  cover_greedy             expected=   88  got=   88  OK
  separation_r1_exact      expected=   97  got=   97  OK
  separation_r2_exact      expected=  183  got=  183  OK

All reproduced: True
```

i.e. the minimum KC-covering test needs only **87** of the 16,891 bank
items (greedy finds 88), the minimum diagnostically-separating test
under the single-KC-deficit model needs **97** items at `r=1`, and
**183** items for 2-robust separation (`r=2`).

## What is *not* included

Per the data license of the other dataset used in the underlying
dissertation research (Eedi, CC BY-NC-ND 4.0, which forbids
redistribution of derivatives), this repository does **not** include
any Eedi-derived data, preprocessed Q-matrices, or result tables. Only
the ASSISTments-derived Q-matrix (CC BY 4.0) and synthetic data are
distributed here. Users with their own access to Eedi (or any other
Q-matrix source) can run every solver in this package directly on
their own `.npz`/dense-array Q-matrix.

## Testing

```bash
pip install -e ".[test]"
pytest -q
```

47 tests (as of `v0.1.0`) cover: greedy/ILP/MaxSAT agreement on the
minimum r-cover optimum across randomized synthetic instances (r=1,2,3);
edge cases (single item, uncoverable skills); the pattern-multiplicity
reduction and quotient-state-space reduction used by the exact
r-robust separation solver, cross-checked against exhaustive brute
force on small random instances; and basic Q-matrix construction/stats.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this software, please
cite the associated dissertation work (in preparation) and this
repository's Zenodo DOI (to be minted on first tagged release).

## License

MIT. See [`LICENSE`](LICENSE). The bundled ASSISTments-derived
Q-matrix (`data/q_matrix_assistments.npz`) retains the CC BY 4.0
license of its source dataset.
