import numpy as np

from mintest.qmatrix import build_q_matrix, compute_stats, synthetic_q_matrix


def test_build_q_matrix_from_pairs():
    pairs = [
        ("p1", "k1"),
        ("p1", "k2"),
        ("p2", "k2"),
        ("p2", "k2"),  # duplicate, should not double-count
        ("p3", "k3"),
    ]
    Q, item_ids, skill_names = build_q_matrix(pairs)
    assert item_ids == ["p1", "p2", "p3"]
    assert skill_names == ["k1", "k2", "k3"]
    assert Q.shape == (3, 3)
    assert Q.nnz == 4
    dense = Q.toarray()
    assert dense[0].tolist() == [1, 1, 0]
    assert dense[1].tolist() == [0, 1, 0]
    assert dense[2].tolist() == [0, 0, 1]


def test_compute_stats_basic():
    Q, _, _ = build_q_matrix([("p1", "k1"), ("p2", "k1"), ("p2", "k2")])
    stats = compute_stats(Q)
    assert stats["n_items"] == 2
    assert stats["n_skills"] == 2
    assert stats["uncovered_skills"] == 0


def test_compute_stats_uncovered_skill():
    Q = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int8)
    stats = compute_stats(Q)
    assert stats["uncovered_skills"] == 2


def test_synthetic_q_matrix_always_fully_coverable():
    Q = synthetic_q_matrix(n_items=100, n_skills=20, seed=123)
    stats = compute_stats(Q)
    assert stats["uncovered_skills"] == 0
    assert stats["n_items"] == 100
    assert stats["n_skills"] == 20
