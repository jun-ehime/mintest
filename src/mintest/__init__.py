"""mintest: minimum test design (cover / r-robust separation) over a Q-matrix."""

from .qmatrix import build_q_matrix, compute_stats, load_q_matrix, save_q_matrix, synthetic_q_matrix
from .cover import greedy_cover, ilp_cover, maxsat_cover, random_baseline
from .separation import (
    build_discrimination_pairs,
    build_pair_to_patterns,
    exact_separation,
    greedy_separation,
    quotient_reduce,
)

__version__ = "0.1.0"

__all__ = [
    "build_q_matrix",
    "compute_stats",
    "load_q_matrix",
    "save_q_matrix",
    "synthetic_q_matrix",
    "greedy_cover",
    "ilp_cover",
    "maxsat_cover",
    "random_baseline",
    "build_discrimination_pairs",
    "build_pair_to_patterns",
    "exact_separation",
    "greedy_separation",
    "quotient_reduce",
]
