"""
examples/assistments.py

Reproduce the headline numbers from the paper on the public ASSISTments
2009-2010 Skill Builder dataset (CC BY 4.0):

  - minimum KC-covering test (r=1 cover): 87 items (ILP/MaxSAT exact),
    88 items (greedy upper bound), out of 16,891 items / 110 skills.
  - r=1 diagnostic separation (single-KC-deficit model): 97 items exact.
  - r=2 robust diagnostic separation: 183 items exact.

Usage:
    python examples/assistments.py
    (or, from the package's dependency environment:)
    <path-to-venv>/bin/python examples/assistments.py

The bundled Q-matrix (data/q_matrix_assistments.npz) was derived from
the ASSISTments 2009-2010 Skill Builder dataset
(https://sites.google.com/site/assistmentsdata/, CC BY 4.0) by
build_q_matrix.py in the parent research repository; only the
(problem_id, skill_name) structure is retained, no student response
data.
"""

from __future__ import annotations

import json
from pathlib import Path

from mintest.cover import greedy_cover, ilp_cover, maxsat_cover
from mintest.qmatrix import compute_stats, load_q_matrix, print_stats
from mintest.separation import exact_separation, greedy_separation

DATA_PATH = Path(__file__).parent.parent / "data" / "q_matrix_assistments.npz"

EXPECTED = {
    "cover_ilp": 87,
    "cover_maxsat": 87,
    "cover_greedy": 88,
    "separation_r1_exact": 97,
    "separation_r2_exact": 183,
}


def main() -> None:
    Q = load_q_matrix(DATA_PATH)
    stats = compute_stats(Q)
    print_stats(stats, label="ASSISTments 2009-2010 Skill Builder")

    print("\n--- Minimum KC cover (r=1) ---")
    greedy = greedy_cover(Q, r=1)
    print(f"greedy:  n_selected={greedy['n_selected']}  coverage={greedy['coverage_rate']:.4f}")

    ilp = ilp_cover(Q, r=1, timeout_sec=300)
    print(f"ilp:     n_selected={ilp['n_selected']}  status={ilp['status']}")

    maxsat = maxsat_cover(Q, r=1, timeout_sec=300)
    print(f"maxsat:  n_selected={maxsat['n_selected']}  status={maxsat['status']}")

    print("\n--- r-robust diagnostic separation ---")
    sep_r1 = exact_separation(Q, r=1, timeout_sec=1200)
    print(f"r=1 exact: optimal={sep_r1['optimal']}  status={sep_r1['status']}")

    sep_r2 = exact_separation(Q, r=2, timeout_sec=1200)
    print(f"r=2 exact: optimal={sep_r2['optimal']}  status={sep_r2['status']}")

    results = {
        "cover_greedy": greedy["n_selected"],
        "cover_ilp": ilp["n_selected"],
        "cover_maxsat": maxsat["n_selected"],
        "separation_r1_exact": sep_r1["optimal"],
        "separation_r2_exact": sep_r2["optimal"],
    }

    print("\n--- Comparison against expected paper values ---")
    all_match = True
    for key, expected in EXPECTED.items():
        got = results.get(key)
        ok = got == expected
        all_match &= ok
        print(f"  {key:24s} expected={expected:>5}  got={got!s:>5}  {'OK' if ok else 'MISMATCH'}")

    print(f"\nAll reproduced: {all_match}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
