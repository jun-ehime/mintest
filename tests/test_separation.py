import numpy as np
import pytest

from mintest.qmatrix import synthetic_q_matrix, to_dense_bool
from mintest.separation import (
    brute_force_full,
    brute_force_reduced,
    build_discrimination_pairs,
    build_pair_to_patterns,
    exact_separation,
    greedy_separation,
    quotient_reduce,
    verify_solution_feasibility,
)


def _random_instance(rng, n_kc_range=(3, 6), n_base_range=(3, 7), n_rows_range=(8, 13)):
    n_kc = int(rng.integers(*n_kc_range))
    n_base = int(rng.integers(*n_base_range))
    n_rows = int(rng.integers(*n_rows_range))
    base = rng.random((n_base, n_kc)) < 0.45
    empty_rows = base.sum(axis=1) == 0
    for i in np.where(empty_rows)[0]:
        base[i, rng.integers(n_kc)] = True
    idx = rng.integers(n_base, size=n_rows)
    return base[idx]


@pytest.mark.parametrize("trial", range(15))
def test_reduction_matches_bruteforce(trial):
    """min(count, r) reduction claim: full-instance brute force ==
    reduced-instance brute force == MaxSAT exact solve."""
    rng = np.random.default_rng(20260905 + trial)
    Q = _random_instance(rng)
    r = int(rng.integers(2, 4))

    uniq, counts = np.unique(Q, axis=0, return_counts=True)
    opt_full = brute_force_full(Q, r)
    opt_reduced = brute_force_reduced(uniq, counts, r)

    from mintest.separation import _solve_reduced

    res = _solve_reduced(uniq, counts, r)
    opt_maxsat = res.get("optimal")

    assert opt_full == opt_reduced == opt_maxsat


@pytest.mark.parametrize("trial", range(10))
def test_exact_solution_is_feasible(trial):
    rng = np.random.default_rng(999 + trial)
    Q = _random_instance(rng)
    r = int(rng.integers(1, 4))
    uniq, counts = np.unique(Q, axis=0, return_counts=True)

    from mintest.separation import _solve_reduced

    res = _solve_reduced(uniq, counts, r)
    if res["status"] == "OPTIMAL":
        assert verify_solution_feasibility(uniq, counts, r, res["selected_pattern_multiplicity"])


def test_greedy_is_upper_bound_of_exact():
    rng = np.random.default_rng(42)
    Q = _random_instance(rng, n_kc_range=(4, 7), n_rows_range=(15, 25))
    for r in (1, 2):
        g = greedy_separation(Q, r=r)
        e = exact_separation(Q, r=r)
        assert g["n_selected"] >= e["optimal"]


def test_quotient_preserves_optimum():
    """Duplicating a KC column (making two KCs Q-matrix-identical) must
    not change the r-robust separation optimum once reduced to the
    quotient state space, because the raw pair set and quotient pair
    set induce the same constraint families."""
    rng = np.random.default_rng(7)
    n_items, n_kc = 30, 6
    Q = (rng.random((n_items, n_kc)) < 0.3)
    for k in range(n_kc):
        if not Q[:, k].any():
            Q[rng.integers(n_items), k] = True

    # Duplicate column 0 as a new column -> creates a redundant KC that
    # is indistinguishable from column 0 by any item.
    Q_dup = np.concatenate([Q, Q[:, [0]]], axis=1)

    reduced = quotient_reduce(Q_dup)
    assert reduced["n_classes"] == n_kc
    assert reduced["n_collapsed"] == 1

    for r in (1, 2):
        exact_raw = exact_separation(Q_dup, r=r)
        exact_quotient = exact_separation(reduced["Q_quotient"], r=r)
        assert exact_raw["optimal"] == exact_quotient["optimal"]


def test_build_pair_to_patterns_matches_discrimination_pairs():
    rng = np.random.default_rng(3)
    Q = (rng.random((20, 5)) < 0.4)
    a = build_discrimination_pairs(Q)
    b = build_pair_to_patterns(to_dense_bool(Q))
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert sorted(a[k]) == sorted(b[k])
