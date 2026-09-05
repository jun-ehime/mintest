"""
mintest.qmatrix
================

Utilities to build, load, and describe a Q-matrix: a binary
``(n_items x n_skills)`` matrix that records which knowledge components
(KCs, a.k.a. skills) each test item exercises.

This module is deliberately generic (it is not tied to any single
dataset). It supports:

- building a Q-matrix from a table of ``(item_id, skill_name)`` pairs
  (:func:`build_q_matrix`),
- loading/saving a Q-matrix from/to ``scipy.sparse`` ``.npz`` files
  (:func:`load_q_matrix`, :func:`save_q_matrix`),
- computing descriptive statistics (:func:`compute_stats`), and
- generating synthetic Q-matrices for tests and demos
  (:func:`synthetic_q_matrix`).

No dataset-specific loader (e.g. for a particular CSV schema) lives
here; see ``examples/assistments.py`` for a worked example that reads
the public ASSISTments 2009-2010 Skill Builder export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz, spmatrix


def build_q_matrix(
    pairs: Iterable[Tuple[object, object]],
) -> Tuple[csr_matrix, list, list]:
    """Build a Q-matrix from an iterable of ``(item_id, skill_name)`` pairs.

    Duplicate pairs are ignored (a skill is either exercised by an item
    or not; multiplicities in the input do not change the matrix).

    Parameters
    ----------
    pairs:
        Iterable of ``(item_id, skill_name)`` tuples. ``item_id`` and
        ``skill_name`` may be any hashable, sortable type (e.g. ``int``
        or ``str``).

    Returns
    -------
    (Q, item_ids, skill_names):
        ``Q`` is a ``scipy.sparse.csr_matrix`` of shape
        ``(n_items, n_skills)`` with dtype ``int8``. ``item_ids`` and
        ``skill_names`` are sorted lists giving the row/column labels.
    """
    seen = set()
    items = set()
    skills = set()
    for item_id, skill_name in pairs:
        seen.add((item_id, skill_name))
        items.add(item_id)
        skills.add(skill_name)

    item_ids = sorted(items, key=_sort_key)
    skill_names = sorted(skills, key=_sort_key)
    item_idx = {v: i for i, v in enumerate(item_ids)}
    skill_idx = {v: i for i, v in enumerate(skill_names)}

    rows = np.empty(len(seen), dtype=np.int64)
    cols = np.empty(len(seen), dtype=np.int64)
    for i, (item_id, skill_name) in enumerate(seen):
        rows[i] = item_idx[item_id]
        cols[i] = skill_idx[skill_name]
    data = np.ones(len(seen), dtype=np.int8)

    Q = csr_matrix((data, (rows, cols)), shape=(len(item_ids), len(skill_names)))
    return Q, item_ids, skill_names


def _sort_key(v):
    # Allow sorting heterogeneous-but-comparable-within-type label sets
    # (all ints, or all strs) without crashing on mixed types.
    return (type(v).__name__, v)


def load_q_matrix(path) -> csr_matrix:
    """Load a Q-matrix previously saved with :func:`save_q_matrix`."""
    return load_npz(str(path))


def save_q_matrix(Q, path) -> None:
    """Save a (sparse or dense) Q-matrix to a ``.npz`` file."""
    if not isinstance(Q, spmatrix):
        Q = csr_matrix(Q)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(path), Q)


def to_dense_bool(Q) -> np.ndarray:
    """Return a dense boolean ``numpy`` array view of ``Q``."""
    if isinstance(Q, spmatrix):
        return Q.toarray().astype(bool)
    return np.asarray(Q).astype(bool)


def compute_stats(Q) -> dict:
    """Compute descriptive statistics of a Q-matrix.

    Parameters
    ----------
    Q:
        Q-matrix, sparse or dense, shape ``(n_items, n_skills)``.

    Returns
    -------
    dict with counts of items/skills, sparsity, per-item and per-skill
    coverage distributions, and the number of skills not covered by
    any item (``uncovered_skills``).
    """
    Q_dense = to_dense_bool(Q)
    n_items, n_skills = Q_dense.shape
    skills_per_item = Q_dense.sum(axis=1)
    items_per_skill = Q_dense.sum(axis=0)
    uncovered = int((items_per_skill == 0).sum())

    stats = {
        "n_items": int(n_items),
        "n_skills": int(n_skills),
        "n_nonzero": int(Q_dense.sum()),
        "sparsity": float(Q_dense.sum() / (n_items * n_skills)) if n_items and n_skills else 0.0,
        "skills_per_item_mean": float(skills_per_item.mean()) if n_items else 0.0,
        "skills_per_item_max": int(skills_per_item.max()) if n_items else 0,
        "skills_per_item_min": int(skills_per_item.min()) if n_items else 0,
        "skills_per_item_median": float(np.median(skills_per_item)) if n_items else 0.0,
        "items_per_skill_mean": float(items_per_skill.mean()) if n_skills else 0.0,
        "items_per_skill_max": int(items_per_skill.max()) if n_skills else 0,
        "items_per_skill_min": int(items_per_skill.min()) if n_skills else 0,
        "items_per_skill_median": float(np.median(items_per_skill)) if n_skills else 0.0,
        "uncovered_skills": uncovered,
        "n_unique_item_patterns": int(len(np.unique(Q_dense, axis=0))) if n_items else 0,
    }
    return stats


def synthetic_q_matrix(
    n_items: int = 200,
    n_skills: int = 30,
    seed: int = 42,
    mean_skills_per_item: float = 1.5,
    ensure_covered: bool = True,
) -> csr_matrix:
    """Generate a random synthetic Q-matrix, useful for tests and demos.

    Each item is assigned ``max(1, Poisson(mean_skills_per_item))``
    distinct skills chosen uniformly at random. If ``ensure_covered``,
    any skill left uncovered after the random draw is force-assigned
    to a random item so that a feasible full cover always exists.
    """
    rng = np.random.default_rng(seed)
    Q_dense = np.zeros((n_items, n_skills), dtype=np.int8)
    for i in range(n_items):
        k = max(1, min(n_skills, int(rng.poisson(mean_skills_per_item))))
        chosen = rng.choice(n_skills, size=k, replace=False)
        Q_dense[i, chosen] = 1

    if ensure_covered:
        for k in range(n_skills):
            if not Q_dense[:, k].any():
                Q_dense[rng.integers(n_items), k] = 1

    return csr_matrix(Q_dense)


def print_stats(stats: dict, label: str = "Q-matrix") -> None:
    """Pretty-print the dict returned by :func:`compute_stats`."""
    print(f"=== {label} ===")
    print(f"  items:            {stats['n_items']:,}")
    print(f"  skills:           {stats['n_skills']:,}")
    print(f"  nonzero entries:  {stats['n_nonzero']:,}")
    print(f"  sparsity:         {stats['sparsity']:.6f}")
    print(f"  uncovered skills: {stats['uncovered_skills']:,}")
    print(f"  unique item patterns: {stats['n_unique_item_patterns']:,}")
    print(
        f"  skills/item  mean={stats['skills_per_item_mean']:.2f} "
        f"median={stats['skills_per_item_median']:.1f} "
        f"min={stats['skills_per_item_min']} max={stats['skills_per_item_max']}"
    )
    print(
        f"  items/skill  mean={stats['items_per_skill_mean']:.1f} "
        f"median={stats['items_per_skill_median']:.1f} "
        f"min={stats['items_per_skill_min']} max={stats['items_per_skill_max']}"
    )
