"""
mintest.separation
===================

r-robust diagnostic separation test design.

Model
-----
Under the single-KC-deficit diagnostic model, a learner's knowledge
state is one of ``n_skills + 1`` states: ``s0`` (all skills mastered)
or ``s_k`` (every skill mastered except skill ``k``). A test item that
exercises skill set ``C(q)`` distinguishes:

- ``(s0, s_k)`` iff ``k in C(q)``;
- ``(s_i, s_j)`` iff exactly one of ``i, j`` is in ``C(q)`` (XOR).

A state pair is **r-robustly separated** by a set of selected items T
if at least ``r`` items in T distinguish it. Pairs that cannot be
separated by ``r`` items anywhere in the full bank are excluded from
the requirement (recorded as ``n_pairs_unsolvable_at_r``).

This module provides:

- :func:`build_pair_to_patterns` — pair -> separating-pattern-index map
  over a matrix of *unique* item patterns.
- :func:`build_discrimination_pairs` — same, but directly over item
  rows (no dedup), matching the pair definition above.
- :func:`quotient_reduce` — collapse KCs (columns) that are identical
  across the whole bank into equivalence classes. Two indistinguishable
  KCs are also indistinguishable as *states*, so this quotient does not
  change which pairs are separable nor the minimum r-robust test size
  (see the module-level theorem note below and
  ``tests/test_separation.py::test_quotient_preserves_optimum``).
- :func:`greedy_separation` — greedy r-robust separation (upper bound).
- :func:`exact_separation` — exact r-robust separation via Partial
  MaxSAT (PySAT RC2), using the pattern-multiplicity reduction below.

Reduction used by :func:`exact_separation`
-------------------------------------------
Encoding one Boolean variable per physical item does not scale well
when many items share the same skill pattern (a common situation in
real Q-matrices). Instead we dedupe items into unique patterns with
multiplicities (``counts``), and observe:

*Claim*: for a fixed ``r``, capping each pattern's usable copies at
``min(count, r)`` does not change the optimal r-robust separation size
nor which pairs are r-separable at all.

*Proof sketch*: whether an item separates a pair depends only on its
pattern. If a feasible solution uses more than ``r`` copies of some
pattern, trimming down to ``r`` copies keeps every pair's separator
count at ``>= r`` (a pair needing that pattern already gets `r` from
it) while only shrinking the solution — so a solution using at most
``min(count, r)`` copies per pattern is optimal. Likewise, a pair's
"total separators in the bank" ``>= r`` iff its "total separators in
the capped instance" ``>= r`` (capping only truncates counts that were
already ``< r``, or leaves ``count >= r`` cases at exactly ``r``).

This reduction, and the MaxSAT encoding built on top of it, are
verified against exhaustive brute force on random small instances in
``tests/test_separation.py``.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from typing import Optional

import numpy as np

from .qmatrix import to_dense_bool


def build_pair_to_patterns(uniq: np.ndarray) -> dict:
    """Given a matrix of unique item patterns ``uniq`` (n_patterns x
    n_skills, boolean), return ``{pair: [pattern_indices]}`` mapping
    each separable state pair to the patterns that separate it.

    Pair keys: ``(-1, k)`` for ``(s0, s_k)``; ``(i, j)`` with ``i < j``
    for ``(s_i, s_j)``. Pairs with zero separating patterns are omitted.
    """
    n_kc = uniq.shape[1]
    cols = [uniq[:, k] for k in range(n_kc)]
    pair_to_patterns = {}

    for k in range(n_kc):
        pats = np.where(cols[k])[0].tolist()
        pair_to_patterns[(-1, k)] = pats

    for i in range(n_kc):
        for j in range(i + 1, n_kc):
            pats = np.where(cols[i] ^ cols[j])[0].tolist()
            if pats:
                pair_to_patterns[(i, j)] = pats

    return pair_to_patterns


def build_discrimination_pairs(Q) -> dict:
    """Same as :func:`build_pair_to_patterns` but directly over item
    rows of ``Q`` (no dedup): ``{pair: [item_indices]}``.
    """
    Q_dense = to_dense_bool(Q)
    return build_pair_to_patterns(Q_dense)


def quotient_reduce(Q) -> dict:
    """Collapse KCs (columns) of ``Q`` that are identical across all
    items into equivalence classes, keeping one representative column
    per class.

    Returns a dict with:

    - ``Q_quotient``: the reduced boolean matrix (n_items x n_classes)
    - ``representative_columns``: original column index kept per class
    - ``class_of_column``: original column index -> class index
    - ``class_sizes``: size of each equivalence class
    - ``n_classes``, ``n_collapsed``: bookkeeping counts
    """
    Q_dense = to_dense_bool(Q)
    n_items, n_kc = Q_dense.shape
    col_key = [Q_dense[:, k].tobytes() for k in range(n_kc)]
    groups: dict = {}
    for k, key in enumerate(col_key):
        groups.setdefault(key, []).append(k)

    reps = sorted(g[0] for g in groups.values())
    key_to_rep = {}
    for key, members in groups.items():
        rep = min(members)
        key_to_rep[key] = rep
    rep_index = {rep: i for i, rep in enumerate(reps)}
    class_of_column = [rep_index[key_to_rep[col_key[k]]] for k in range(n_kc)]
    class_sizes = [len(groups[key]) for key in sorted(groups, key=lambda k: rep_index[key_to_rep[k]])]

    Q_quotient = Q_dense[:, reps]

    return {
        "Q_quotient": Q_quotient,
        "representative_columns": reps,
        "class_of_column": class_of_column,
        "class_sizes": class_sizes,
        "n_classes": len(reps),
        "n_collapsed": n_kc - len(reps),
    }


def greedy_separation(Q, r: int = 1, pair_to_items: Optional[dict] = None) -> dict:
    """Greedy r-robust separation (upper bound): repeatedly pick the
    item that reduces the largest number of still-unsatisfied pair
    requirements, until every separable pair has >= r separating items
    selected.
    """
    Q_dense = to_dense_bool(Q)
    n_items = Q_dense.shape[0]
    if pair_to_items is None:
        pair_to_items = build_discrimination_pairs(Q_dense)

    solvable = {pair: items for pair, items in pair_to_items.items() if len(items) >= r}
    unsolvable = {pair: items for pair, items in pair_to_items.items() if len(items) < r}

    remaining = {pair: r for pair in solvable}
    item_to_pairs: dict = {i: [] for i in range(n_items)}
    for pair, items in solvable.items():
        for i in items:
            item_to_pairs[i].append(pair)

    selected = []
    t0 = time.time()
    while any(v > 0 for v in remaining.values()):
        gains = np.zeros(n_items, dtype=np.int64)
        for i in range(n_items):
            gains[i] = sum(1 for p in item_to_pairs[i] if remaining.get(p, 0) > 0)
        if gains.max() == 0:
            break
        best = int(gains.argmax())
        selected.append(best)
        for p in item_to_pairs[best]:
            if remaining.get(p, 0) > 0:
                remaining[p] -= 1
    elapsed = time.time() - t0

    resolved = sum(1 for v in remaining.values() if v == 0)

    return {
        "algorithm": "greedy",
        "r": r,
        "n_selected": len(selected),
        "n_pairs_total": len(pair_to_items),
        "n_pairs_solvable": len(solvable),
        "n_pairs_unsolvable": len(unsolvable),
        "n_pairs_resolved": resolved,
        "elapsed_sec": elapsed,
        "selected": sorted(selected),
    }


def exact_separation(Q, r: int = 1, timeout_sec: int = 1200) -> dict:
    """Exact minimum r-robust separation via Partial MaxSAT (RC2), using
    the ``min(count, r)`` pattern-multiplicity reduction described in
    the module docstring.

    Returns a dict with ``status`` (``OPTIMAL``/``UNSAT``), ``optimal``
    (test size), pair bookkeeping, and
    ``selected_pattern_multiplicity`` (pattern index -> number of
    copies used; map back to original item indices via
    ``np.unique(..., return_inverse=True)`` if needed).
    """
    Q_dense = to_dense_bool(Q)
    uniq, counts = np.unique(Q_dense, axis=0, return_counts=True)
    return _solve_reduced(uniq, counts, r)


def _solve_reduced(uniq: np.ndarray, counts: np.ndarray, r: int) -> dict:
    try:
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF
        from pysat.card import CardEnc, EncType
    except ImportError as e:  # pragma: no cover
        raise ImportError("python-sat is required: pip install python-sat") from e

    t0 = time.time()
    pair_to_patterns = build_pair_to_patterns(uniq)

    caps = np.minimum(counts, r)
    pattern_vars = []
    vid = 0
    for p in range(len(counts)):
        vs = list(range(vid + 1, vid + 1 + int(caps[p])))
        pattern_vars.append(vs)
        vid += int(caps[p])
    n_vars = vid
    top_id = n_vars

    wcnf = WCNF()
    n_hard = 0

    # Symmetry breaking within a pattern's copies: x_1 >= x_2 >= ...
    for vs in pattern_vars:
        for a, b in zip(vs, vs[1:]):
            wcnf.append([a, -b])
            n_hard += 1

    n_unsolvable = 0
    n_enforced = 0
    for pair, pats in pair_to_patterns.items():
        total = int(counts[pats].sum()) if pats else 0
        if total < r:
            n_unsolvable += 1
            continue
        lits = [v for p in pats for v in pattern_vars[p]]
        if r == 1:
            wcnf.append(lits)
            n_hard += 1
        elif len(lits) == r:
            for v in lits:
                wcnf.append([v])
            n_hard += r
        else:
            atl = CardEnc.atleast(lits=lits, bound=r, top_id=top_id, encoding=EncType.seqcounter)
            for cl in atl.clauses:
                wcnf.append(cl)
            n_hard += len(atl.clauses)
            top_id = max(top_id, atl.nv)
        n_enforced += 1

    for v in range(1, n_vars + 1):
        wcnf.append([-v], weight=1)

    with RC2(wcnf) as rc2:
        model = rc2.compute()
        elapsed = time.time() - t0
        if model is None:
            return {"status": "UNSAT", "elapsed_sec": elapsed}
        selected = [v for v in model if 1 <= v <= n_vars]

    var_to_pattern = {}
    for p, vs in enumerate(pattern_vars):
        for v in vs:
            var_to_pattern[v] = p
    mult: dict = {}
    for v in selected:
        p = var_to_pattern[v]
        mult[p] = mult.get(p, 0) + 1

    return {
        "status": "OPTIMAL",
        "optimal": len(selected),
        "n_pairs_total": len(pair_to_patterns),
        "n_pairs_enforced": n_enforced,
        "n_pairs_unsolvable_at_r": n_unsolvable,
        "n_vars": n_vars,
        "n_hard_clauses": n_hard,
        "elapsed_sec": elapsed,
        "selected_pattern_multiplicity": {int(k): int(v) for k, v in sorted(mult.items())},
    }


def verify_solution_feasibility(uniq: np.ndarray, counts: np.ndarray, r: int, multiplicity: dict) -> bool:
    """Independently check (without relying on the SAT encoding) that a
    solution returned by :func:`exact_separation`/``_solve_reduced``
    satisfies every separable pair's ``>= r`` requirement.
    """
    caps = np.minimum(counts, r)
    for p, m in multiplicity.items():
        if not (0 < m <= caps[int(p)]):
            return False
    pair_to_patterns = build_pair_to_patterns(uniq)
    for pats in pair_to_patterns.values():
        if int(counts[pats].sum()) < r:
            continue
        if sum(multiplicity.get(p, 0) for p in pats) < r:
            return False
    return True


def brute_force_full(Q: np.ndarray, r: int) -> int:
    """Exhaustive optimum on the *full* (non-deduped) instance. Only
    for small test instances; used to validate the reduction claim.
    """
    n = Q.shape[0]
    pair_to_items = build_pair_to_patterns(Q)
    enforced = {pair: set(items) for pair, items in pair_to_items.items() if len(items) >= r}
    if not enforced:
        return 0
    for size in range(0, n + 1):
        for subset in itertools.combinations(range(n), size):
            ss = set(subset)
            if all(len(items & ss) >= r for items in enforced.values()):
                return size
    raise RuntimeError("infeasible (should not happen)")


def brute_force_reduced(uniq: np.ndarray, counts: np.ndarray, r: int) -> int:
    """Exhaustive optimum on the ``min(count, r)``-capped instance."""
    caps = np.minimum(counts, r).astype(int)
    rows = np.repeat(uniq, caps, axis=0)
    pair_to_patterns = build_pair_to_patterns(uniq)
    pat_of_row = np.repeat(np.arange(len(counts)), caps)
    enforced = []
    for pair, pats in pair_to_patterns.items():
        if int(counts[pats].sum()) >= r:
            pset = set(pats)
            qs = {i for i in range(len(rows)) if pat_of_row[i] in pset}
            enforced.append(qs)
    if not enforced:
        return 0
    n = len(rows)
    for size in range(0, n + 1):
        for subset in itertools.combinations(range(n), size):
            ss = set(subset)
            if all(len(qs & ss) >= r for qs in enforced):
                return size
    raise RuntimeError("infeasible (should not happen)")
