import numpy as np
import pytest

from mintest.qmatrix import synthetic_q_matrix, to_dense_bool
from mintest.cover import greedy_cover, ilp_cover, maxsat_cover, random_baseline


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ilp_equals_maxsat_optimal_r1(seed):
    Q = synthetic_q_matrix(n_items=60, n_skills=12, seed=seed)
    ilp = ilp_cover(Q, r=1, timeout_sec=60)
    ms = maxsat_cover(Q, r=1, timeout_sec=60)
    assert ilp["status"] == "Optimal"
    assert ms["status"] == "OPTIMAL"
    assert ilp["n_selected"] == ms["n_selected"]
    assert ilp["coverage_rate"] == 1.0
    assert ms["coverage_rate"] == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_is_upper_bound_of_optimal_r1(seed):
    Q = synthetic_q_matrix(n_items=80, n_skills=15, seed=seed)
    greedy = greedy_cover(Q, r=1)
    ilp = ilp_cover(Q, r=1, timeout_sec=60)
    assert greedy["n_selected"] >= ilp["n_selected"]
    assert greedy["coverage_rate"] == 1.0


@pytest.mark.parametrize("r", [1, 2, 3])
def test_ilp_equals_maxsat_optimal_rcover(r):
    Q = synthetic_q_matrix(n_items=100, n_skills=10, seed=7, mean_skills_per_item=2.5)
    ilp = ilp_cover(Q, r=r, timeout_sec=60)
    ms = maxsat_cover(Q, r=r, timeout_sec=60)
    assert ilp["status"] == "Optimal"
    assert ms["status"] == "OPTIMAL"
    assert ilp["n_selected"] == ms["n_selected"]


def test_cover_solution_is_feasible():
    Q = synthetic_q_matrix(n_items=50, n_skills=10, seed=3)
    Q_dense = to_dense_bool(Q)
    result = ilp_cover(Q, r=1, timeout_sec=60)
    counts = Q_dense[result["selected"]].sum(axis=0)
    assert (counts >= 1).all()


def test_random_baseline_worse_than_greedy_on_average():
    Q = synthetic_q_matrix(n_items=200, n_skills=25, seed=1)
    greedy = greedy_cover(Q, r=1)
    rnd = random_baseline(Q, n_select=greedy["n_selected"], r=1, n_trials=50, seed=1)
    assert greedy["coverage_rate"] >= rnd["mean_coverage_rate"]


def test_empty_and_trivial_edge_cases():
    # single item, single skill: trivially coverable with 1 item
    Q = np.array([[1]], dtype=np.int8)
    g = greedy_cover(Q, r=1)
    assert g["n_selected"] == 1
    assert g["coverage_rate"] == 1.0

    ilp = ilp_cover(Q, r=1, timeout_sec=10)
    assert ilp["n_selected"] == 1


def test_uncoverable_skill_is_reported():
    # skill 2 (index) is never covered by any item
    Q = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int8,
    )
    ilp = ilp_cover(Q, r=1, timeout_sec=10)
    assert ilp["n_uncoverable_skills"] == 1
    assert ilp["coverage_rate"] < 1.0

    ms = maxsat_cover(Q, r=1, timeout_sec=10)
    assert ms["n_uncoverable_skills"] == 1
