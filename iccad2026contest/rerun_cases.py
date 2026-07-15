#!/usr/bin/env python3
"""
Re-run selected validation cases and save each result to its own JSON file.

Uses iccad2026_evaluate_test.py (same flow as a manual single-case run).

Default cases (from evaluate.log failures):
  43, 53, 62, 63, 71, 80, 91

Examples
--------
  python rerun_cases.py
  python rerun_cases.py --test-ids 43 63
  python rerun_cases.py --workers 3
  python rerun_cases.py --optimizer test_optimizer.py --out-dir results_json/retry
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
EVALUATOR = SCRIPT_DIR / "iccad2026_evaluate_test.py"
PYTHON = sys.executable

# Cases flagged in evaluate.log (overlap or recursion errors)
DEFAULT_TEST_IDS = [43, 53, 62, 63, 71, 80, 91]


def _run_one(args: Tuple[int, str, str]) -> Dict[str, Any]:
    """Run one case; save JSON to out_dir/case_XXX.json."""
    test_id, optimizer_path, out_dir = args
    out_path = Path(out_dir) / f"case_{test_id:03d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(EVALUATOR),
        "--evaluate",
        optimizer_path,
        "--test-id",
        str(test_id),
        "--output",
        str(out_path),
    ]

    result: Dict[str, Any] = {
        "test_id": test_id,
        "output_json": str(out_path),
        "feasible": False,
        "cost": 10.0,
        "runtime_s": 0.0,
        "error": None,
    }

    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR),
            timeout=3600,
        )
        elapsed = time.time() - t0
        stdout = proc.stdout + proc.stderr

        if "Feasible: 1" in stdout:
            result["feasible"] = True
        m = re.search(r"Avg Cost:\s+([\d.]+)", stdout)
        if m:
            result["cost"] = float(m.group(1))
        m = re.search(r"Avg Runtime:\s+([\d.]+)s", stdout)
        if m:
            result["runtime_s"] = float(m.group(1))
        elif elapsed > 0:
            result["runtime_s"] = elapsed

        if proc.returncode != 0 and not out_path.is_file():
            result["error"] = f"exit code {proc.returncode}"
        if "ERROR:" in stdout and not result["feasible"]:
            m = re.search(r"ERROR:\s*(.+)", stdout)
            if m:
                result["error"] = m.group(1).strip()

        if out_path.is_file():
            try:
                data = json.loads(out_path.read_text())
                tr = data.get("test_results", [{}])[0]
                result["feasible"] = bool(tr.get("is_feasible", result["feasible"]))
                result["cost"] = float(tr.get("cost", result["cost"]))
                result["runtime_s"] = float(
                    tr.get("runtime_seconds", result["runtime_s"])
                )
                if tr.get("error"):
                    result["error"] = tr["error"]
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        result["error"] = "TIMEOUT (>3600s)"
    except Exception as exc:
        result["error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run selected optimizer cases to separate JSON files."
    )
    parser.add_argument(
        "--optimizer",
        default="test_optimizer.py",
        help="Optimizer file (default: test_optimizer.py)",
    )
    parser.add_argument(
        "--test-ids",
        nargs="*",
        type=int,
        default=None,
        help=f"Case IDs to run (default: {DEFAULT_TEST_IDS})",
    )
    parser.add_argument(
        "--out-dir",
        default="results_json/retry",
        help="Directory for per-case JSON output (default: results_json/retry)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1 — one case at a time)",
    )
    args = parser.parse_args()

    test_ids: List[int] = args.test_ids if args.test_ids else DEFAULT_TEST_IDS
    optimizer_path = str((SCRIPT_DIR / args.optimizer).resolve())
    out_dir = str((SCRIPT_DIR / args.out_dir).resolve())
    workers = max(1, min(args.workers, len(test_ids)))

    print(f"Re-running {len(test_ids)} cases: {test_ids}")
    print(f"Optimizer : {optimizer_path}")
    print(f"Output dir: {out_dir}")
    print(f"Workers   : {workers}\n")

    tasks = [(tid, optimizer_path, out_dir) for tid in test_ids]
    results: List[Dict[str, Any]] = []
    wall_start = time.time()

    if workers == 1:
        for i, task in enumerate(tasks, 1):
            res = _run_one(task)
            results.append(res)
            status = "OK" if res["feasible"] else "FAIL"
            err = f"  ({res['error']})" if res.get("error") else ""
            print(
                f"  [{i}/{len(tasks)}] case {res['test_id']:3d}  {status}  "
                f"cost={res['cost']:.4f}  rt={res['runtime_s']:.1f}s  "
                f"→ {res['output_json']}{err}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                done += 1
                status = "OK" if res["feasible"] else "FAIL"
                err = f"  ({res['error']})" if res.get("error") else ""
                print(
                    f"  [{done}/{len(tasks)}] case {res['test_id']:3d}  {status}  "
                    f"cost={res['cost']:.4f}  rt={res['runtime_s']:.1f}s  "
                    f"→ {res['output_json']}{err}"
                )

    wall_elapsed = time.time() - wall_start
    results.sort(key=lambda r: r["test_id"])

    summary_path = Path(out_dir) / f"retry_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    summary_path.write_text(
        json.dumps(
            {
                "optimizer": args.optimizer,
                "test_ids": test_ids,
                "wall_time_s": round(wall_elapsed, 2),
                "n_feasible": sum(1 for r in results if r["feasible"]),
                "results": results,
            },
            indent=2,
        )
    )

    print(f"\nDone in {wall_elapsed:.1f}s wall time.")
    print(f"Feasible: {sum(1 for r in results if r['feasible'])}/{len(results)}")
    print(f"Summary  → {summary_path}")
    print("\nPer-case JSON files:")
    for r in results:
        print(f"  case_{r['test_id']:03d}.json  feasible={r['feasible']}  cost={r['cost']:.4f}")


if __name__ == "__main__":
    main()
