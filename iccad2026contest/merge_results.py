#!/usr/bin/env python3
"""
Merge per-case JSON files from results_json/ into test_optimizer_results.json.

Replaces test_results entries by test_id and recomputes total_score.

Examples
--------
  python merge_results.py
  python merge_results.py --base "test_optimizer_results copy.json"
  python merge_results.py --from-dir results_json/retry
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).parent
DEFAULT_BASE = SCRIPT_DIR / "test_optimizer_results copy.json"
DEFAULT_OUT = SCRIPT_DIR / "test_optimizer_results.json"
DEFAULT_FROM_DIR = SCRIPT_DIR / "results_json" / "retry"


def compute_total_score(costs: List[float], block_counts: List[int]) -> float:
    """Same weighting as iccad2026_evaluate_test.compute_total_score."""
    if not costs:
        return 0.0
    if not block_counts or all(n == 0 for n in block_counts):
        return sum(costs) / len(costs)
    max_n = max(block_counts)
    weights = [math.exp(n - max_n) for n in block_counts]
    total_weight = sum(weights)
    return sum(c * w for c, w in zip(costs, weights)) / total_weight


def load_case_entry(path: Path) -> Dict[str, Any]:
    """Load test_results[0] from a single-case evaluation JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("test_results", [])
    if not results:
        raise ValueError(f"No test_results in {path}")
    return results[0]


def discover_case_files(from_dir: Path) -> Dict[int, Path]:
    """Map test_id -> case_XXX.json under from_dir."""
    mapping: Dict[int, Path] = {}
    for path in sorted(from_dir.glob("case_*.json")):
        try:
            tid = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        mapping[tid] = path
    return mapping


def merge(
    base_path: Path,
    from_dir: Path,
    out_path: Path,
    test_ids: List[int] | None = None,
) -> None:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    test_results: List[Dict[str, Any]] = base.get("test_results", [])
    by_id = {int(r["test_id"]): r for r in test_results}

    case_files = discover_case_files(from_dir)
    if test_ids is not None:
        case_files = {tid: p for tid, p in case_files.items() if tid in test_ids}

    if not case_files:
        raise SystemExit(f"No case_*.json files found in {from_dir}")

    print(f"Base : {base_path} ({len(test_results)} cases, score={base.get('total_score')})")
    print(f"From : {from_dir}")
    print(f"Out  : {out_path}\n")

    replaced = 0
    for tid in sorted(case_files):
        path = case_files[tid]
        new_entry = load_case_entry(path)
        old = by_id.get(tid)
        old_cost = old.get("cost") if old else None
        old_feas = old.get("is_feasible") if old else None
        by_id[tid] = new_entry
        replaced += 1
        print(
            f"  case {tid:3d}  ← {path.name}  "
            f"feasible {old_feas}→{new_entry.get('is_feasible')}  "
            f"cost {old_cost}→{new_entry.get('cost')}"
        )

    merged = sorted(by_id.values(), key=lambda r: int(r["test_id"]))
    costs = [float(r.get("cost", 10.0)) for r in merged]
    blocks = [int(r.get("block_count", 0)) for r in merged]
    previous_score = base.get("total_score")
    new_score = compute_total_score(costs, blocks)

    base["test_results"] = merged
    base["total_score"] = new_score
    base["timestamp"] = datetime.now().isoformat()
    base["merge_note"] = {
        "merged_from": str(from_dir),
        "replaced_test_ids": sorted(case_files.keys()),
        "previous_total_score": previous_score,
    }

    out_path.write_text(json.dumps(base, indent=2, default=str), encoding="utf-8")

    n_feas = sum(1 for r in merged if r.get("is_feasible"))
    print(f"\nReplaced {replaced} case(s).")
    print(f"Total score: {new_score:.4f}  (was {previous_score})")
    print(f"Feasible   : {n_feas}/{len(merged)}")
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge retry case JSON into main results.")
    parser.add_argument(
        "--base",
        default=str(DEFAULT_BASE),
        help="Full results JSON to patch (default: test_optimizer_results copy.json)",
    )
    parser.add_argument(
        "--from-dir",
        default=str(DEFAULT_FROM_DIR),
        help="Directory with case_XXX.json files (default: results_json/retry)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output path (default: test_optimizer_results.json)",
    )
    parser.add_argument(
        "--test-ids",
        nargs="*",
        type=int,
        default=None,
        help="Only merge these test IDs (default: all case_*.json in from-dir)",
    )
    args = parser.parse_args()

    merge(
        Path(args.base),
        Path(args.from_dir),
        Path(args.out),
        args.test_ids,
    )


if __name__ == "__main__":
    main()
