"""
mintest.cover
=============

Minimum (multi-)cover test design over a Q-matrix.

Given a Q-matrix ``Q`` of shape ``(n_items, n_skills)``, we want the
smallest subset of items (rows) such that every skill (column) is
covered by at least ``r`` selected items ("r-cover"; ``r=1`` is the
classical Set Cover problem).

Three interchangeable solvers are provided:

- :func:`greedy_cover` — the standard greedy approximation
  (``1 + ln(n_skills)``-approximate for ``r=1``).
- :func:`ilp_cover` — exact solution via integer linear programming
  (PuLP/CBC).
- :func:`maxsat_cover` — exact solution via Partial MaxSAT (PySAT/RC2).

On the same instance, ``ilp_cover`` and ``maxsat_cover`` always agree
on the optimal size (both are exact), and ``greedy_cover`` gives an
upper bound on it. See ``tests/test_cover.py``.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .qmatrix import to_dense_bool


def greedy_cover(Q, r: int = 1) -> dict:
    """Greedy r-cover: repeatedly pick the item covering the most
    still-undersatisfied skills, until every skill is covered ``r``
    times (or no progress is possible).

    Returns a result dict with ``selected`` (sorted list of row
    indices), ``n_selected``, ``coverage_rate``, and
    ``approx_ratio_upper`` (``1 + ln(n_skills)``, the classical greedy
    approximation guarantee for ``r=1``).
    """
    Q_dense = to_dense_bool(Q)
    n_items, n_skills = Q_dense.shape
    remaining = np.full(n_skills, r, dtype=np.int64)
    selected = []
    available = np.ones(n_items, dtype=bool)

    t0 = time.time()
    while remaining.max() > 0:
        needs = remaining > 0
        gains = (Q_dense & needs).sum(axis=1)
        gains = gains.astype(np.int64)
        gains[~available] = -1
        if gains.max() <= 0:
            break
        best = int(gains.argmax())
        selected.append(best)
        available[best] = False
        remaining = np.maximum(remaining - Q_dense[best].astype(np.int64), 0)
    elapsed = time.time() - t0

    cover_counts = Q_dense[selected].sum(axis=0) if selected else np.zeros(n_skills, dtype=int)
    n_covered = int((cover_counts >= r).sum())

    return {
        "algorithm": "greedy",
        "r": r,
        "n_items_total": int(n_items),
        "n_skills_total": int(n_skills),
        "n_selected": len(selected),
        "n_covered_skills": n_covered,
        "coverage_rate": n_covered / n_skills if n_skills else 1.0,
        "approx_ratio_upper": 1 + math.log(n_skills) if n_skills > 1 else 1.0,
        "elapsed_sec": elapsed,
        "selected": sorted(selected),
    }


def ilp_cover(Q, r: int = 1, timeout_sec: int = 300, verbose: bool = False) -> dict:
    """Exact minimum r-cover via ILP (PuLP, CBC solver by default).

    Constraint per skill k: ``sum_{i covering k} x_i >= r``.
    """
    try:
        import pulp
    except ImportError as e:  # pragma: no cover
        raise ImportError("PuLP is required: pip install pulp") from e

    Q_dense = to_dense_bool(Q)
    n_items, n_skills = Q_dense.shape

    uncoverable = [k for k in range(n_skills) if Q_dense[:, k].sum() < r]

    prob = pulp.LpProblem("MinRCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_items)]
    prob += pulp.lpSum(x)

    n_hard = 0
    for k in range(n_skills):
        covering = [i for i in range(n_items) if Q_dense[i, k]]
        if len(covering) < r:
            continue
        prob += pulp.lpSum(x[i] for i in covering) >= r, f"Cover_{k}"
        n_hard += 1

    solver = pulp.PULP_CBC_CMD(
        timeLimit=timeout_sec if timeout_sec > 0 else None,
        msg=1 if verbose else 0,
    )
    t0 = time.time()
    prob.solve(solver)
    elapsed = time.time() - t0
    status = pulp.LpStatus[prob.status]

    if prob.status != 1:
        return {
            "algorithm": "ilp",
            "r": r,
            "status": status,
            "n_selected": None,
            "elapsed_sec": elapsed,
        }

    selected = [i for i in range(n_items) if pulp.value(x[i]) > 0.5]
    cover_counts = Q_dense[selected].sum(axis=0) if selected else np.zeros(n_skills, dtype=int)
    n_covered = int((cover_counts >= r).sum())

    return {
        "algorithm": "ilp",
        "r": r,
        "status": status,
        "n_items_total": int(n_items),
        "n_skills_total": int(n_skills),
        "n_hard_constraints": n_hard,
        "n_uncoverable_skills": len(uncoverable),
        "n_selected": len(selected),
        "n_covered_skills": n_covered,
        "coverage_rate": n_covered / n_skills if n_skills else 1.0,
        "elapsed_sec": elapsed,
        "selected": sorted(selected),
    }


def maxsat_cover(Q, r: int = 1, timeout_sec: int = 300) -> dict:
    """Exact minimum r-cover via Partial MaxSAT (PySAT RC2).

    Hard clauses encode "each skill is covered >= r times" (a unit
    OR clause for ``r=1``, an AtLeast(r) cardinality encoding for
    ``r>=2``); soft (unit, weight 1) clauses penalize selecting each
    item, so RC2's optimum equals the minimum number of selected
    items.
    """
    try:
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF
        from pysat.card import CardEnc, EncType
    except ImportError as e:  # pragma: no cover
        raise ImportError("python-sat is required: pip install python-sat") from e

    Q_dense = to_dense_bool(Q)
    n_items, n_skills = Q_dense.shape

    wcnf = WCNF()
    top_id = n_items
    n_hard = 0
    uncoverable = []

    for k in range(n_skills):
        covering = [i + 1 for i in range(n_items) if Q_dense[i, k]]
        if len(covering) < r:
            uncoverable.append(k)
            continue
        if r == 1:
            wcnf.append(covering)
            n_hard += 1
        elif len(covering) == r:
            for v in covering:
                wcnf.append([v])
                n_hard += 1
        else:
            atl = CardEnc.atleast(lits=covering, bound=r, top_id=top_id, encoding=EncType.seqcounter)
            for cl in atl.clauses:
                wcnf.append(cl)
            n_hard += len(atl.clauses)
            top_id = max(top_id, atl.nv)

    for i in range(1, n_items + 1):
        wcnf.append([-i], weight=1)

    t0 = time.time()
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        elapsed = time.time() - t0
        if model is None:
            return {"algorithm": "maxsat", "r": r, "status": "UNSAT", "elapsed_sec": elapsed}
        selected = sorted(v - 1 for v in model if 1 <= v <= n_items)

    cover_counts = Q_dense[selected].sum(axis=0) if selected else np.zeros(n_skills, dtype=int)
    n_covered = int((cover_counts >= r).sum())

    return {
        "algorithm": "maxsat",
        "r": r,
        "status": "OPTIMAL",
        "n_items_total": int(n_items),
        "n_skills_total": int(n_skills),
        "n_hard_clauses": n_hard,
        "n_uncoverable_skills": len(uncoverable),
        "n_selected": len(selected),
        "n_covered_skills": n_covered,
        "coverage_rate": n_covered / n_skills if n_skills else 1.0,
        "elapsed_sec": elapsed,
        "selected": selected,
    }


def random_baseline(Q, n_select: int, r: int = 1, n_trials: int = 100, seed: int = 42) -> dict:
    """Coverage rate achieved by ``n_select`` uniformly random items,
    averaged over ``n_trials`` trials. Used as a naive baseline against
    :func:`greedy_cover`/:func:`ilp_cover`/:func:`maxsat_cover`.
    """
    import random as _random

    Q_dense = to_dense_bool(Q)
    n_items, n_skills = Q_dense.shape
    rng = _random.Random(seed)
    rates = []
    for _ in range(n_trials):
        sel = rng.sample(range(n_items), min(n_select, n_items))
        counts = Q_dense[sel].sum(axis=0)
        rates.append(float((counts >= r).sum()) / n_skills if n_skills else 1.0)
    rates = np.array(rates)
    return {
        "algorithm": "random",
        "r": r,
        "n_selected": n_select,
        "n_trials": n_trials,
        "mean_coverage_rate": float(rates.mean()),
        "std_coverage_rate": float(rates.std()),
        "min_coverage_rate": float(rates.min()),
        "max_coverage_rate": float(rates.max()),
    }
