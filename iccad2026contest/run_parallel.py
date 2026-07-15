#!/usr/bin/env python3
"""
Parallel batch runner for the ICCAD 2026 floorplan optimizer.

Runs one or all validation cases using a pool of worker processes (one case
per worker).  Each case is evaluated in its own subprocess so that memory and
imports are fully isolated.  Results are written to individual JSON files under
results_json/ and a combined summary JSON is produced at the end.

Usage examples
--------------
  # All 100 cases, 10 workers, test_optimizer.py
  python run_parallel.py

  # Specific cases
  python run_parallel.py --test-ids 0 10 20 30 40

  # Different optimizer / worker count
  python run_parallel.py --optimizer my_optimizer.py --workers 8

  # Save combined summary to a custom path
  python run_parallel.py --summary results_json/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results_json"
EVALUATOR = SCRIPT_DIR / "iccad2026_evaluate_test.py"
PYTHON = sys.executable  # same venv

# ---------------------------------------------------------------------------
# Per-case worker
# ---------------------------------------------------------------------------

def _run_one_case(args: Tuple[int, str]) -> Dict[str, Any]:
    """
    Run `iccad2026_evaluate_test.py --evaluate <optimizer> --test-id <id>`
    in a subprocess and parse its stdout into a structured dict.

    Returns a dict with at minimum:
        test_id, feasible, cost, runtime_s,
        hard_ok, overlap_ok, area_ok, dimension_ok,
        soft_boundary, soft_grouping, soft_mib, v_rel,
        output_json, error (or None)
    """
    test_id, optimizer_path = args
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Each case writes its own JSON so names never collide
    out_json = RESULTS_DIR / f"case_{test_id:03d}.json"

    # The evaluator always writes to <submission_name>_results.json in cwd.
    # We redirect by setting a per-worker cwd that isolates that file.
    worker_cwd = RESULTS_DIR / f"_worker_{test_id:03d}"
    worker_cwd.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(EVALUATOR),
        "--evaluate", str(Path(optimizer_path).resolve()),
        "--test-id", str(test_id),
        "--verbose",
    ]

    result: Dict[str, Any] = {
        "test_id": test_id,
        "feasible": False,
        "cost": 10.0,
        "runtime_s": 0.0,
        "hard_ok": False,
        "overlap_ok": None,
        "area_ok": None,
        "dimension_ok": None,
        "soft_boundary": None,
        "soft_grouping": None,
        "soft_mib": None,
        "v_rel": None,
        "output_json": str(out_json),
        "error": None,
        "raw_stdout": "",
    }

    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(worker_cwd),
            timeout=600,
        )
        elapsed = time.time() - t0
        stdout = proc.stdout + proc.stderr
        result["raw_stdout"] = stdout

        # ---- parse feasibility & cost ----
        if "Feasible: 1" in stdout:
            result["feasible"] = True
        if "INFEASIBLE" not in stdout.split("EVALUATION RESULTS", 1)[-1] if "EVALUATION RESULTS" in stdout else True:
            m = re.search(r"Avg Cost:\s+([\d.]+)", stdout)
            if m:
                result["cost"] = float(m.group(1))
        m = re.search(r"Avg Runtime:\s+([\d.]+)s", stdout)
        if m:
            result["runtime_s"] = float(m.group(1))

        # ---- hard constraint checks ----
        result["hard_ok"] = "hard (combined)] OK" in stdout
        result["overlap_ok"] = "[overlap] OK" in stdout
        result["area_ok"] = "[area] OK" in stdout
        result["dimension_ok"] = "[dimension] OK" in stdout

        # ---- soft constraint summary ----
        m = re.search(
            r"Soft summary: boundary=(\d+), grouping=(\d+), MIB=(\d+), "
            r"total=\d+/\d+ \(V_rel=([\d.]+)\)",
            stdout,
        )
        if m:
            result["soft_boundary"] = int(m.group(1))
            result["soft_grouping"] = int(m.group(2))
            result["soft_mib"] = int(m.group(3))
            result["v_rel"] = float(m.group(4))

        # ---- copy the evaluator's JSON to our per-case file ----
        # Find which JSON was written (it ends with _results.json)
        for f in worker_cwd.glob("*_results.json"):
            try:
                data = json.loads(f.read_text())
                # Enrich with our parsed fields
                data["_parsed"] = {k: v for k, v in result.items()
                                   if k not in ("raw_stdout", "output_json", "error")}
                out_json.write_text(json.dumps(data, indent=2))
            except Exception:
                pass
            break  # only one file expected
        else:
            # write minimal JSON if evaluator didn't produce one
            out_json.write_text(json.dumps(result, indent=2, default=str))

    except subprocess.TimeoutExpired:
        result["error"] = "TIMEOUT (>600s)"
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Summary pretty-printer
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]], optimizer: str) -> None:
    total = len(results)
    feasible = sum(1 for r in results if r["feasible"])
    costs = [r["cost"] for r in results]
    runtimes = [r["runtime_s"] for r in results if r["runtime_s"] > 0]

    print("\n" + "=" * 70)
    print(f"PARALLEL EVALUATION  —  {optimizer}")
    print(f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)
    print(f"{'test_id':>7}  {'feas':>4}  {'cost':>8}  {'rt(s)':>7}  "
          f"{'hard':>4}  {'olap':>4}  {'area':>4}  {'dim':>4}  "
          f"{'bnd':>4}  {'grp':>4}  {'mib':>4}  {'vrel':>6}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["test_id"]):
        def yn(v: Optional[bool]) -> str:
            if v is None:
                return "  ? "
            return "  OK" if v else " ERR"
        def vi(v: Optional[int]) -> str:
            return f"{v:4d}" if v is not None else "   ?"
        def vf(v: Optional[float]) -> str:
            return f"{v:.4f}" if v is not None else "     ?"

        err = f"  {r['error'][:30]}" if r.get("error") else ""
        print(
            f"{r['test_id']:>7}  {'Y' if r['feasible'] else 'N':>4}  "
            f"{r['cost']:>8.4f}  {r['runtime_s']:>7.2f}  "
            f"{yn(r['hard_ok'])}  {yn(r['overlap_ok'])}  "
            f"{yn(r['area_ok'])}  {yn(r['dimension_ok'])}  "
            f"{vi(r['soft_boundary'])}  {vi(r['soft_grouping'])}  "
            f"{vi(r['soft_mib'])}  {vf(r['v_rel'])}"
            f"{err}"
        )
    print("-" * 70)
    avg_cost = sum(costs) / max(len(costs), 1)
    avg_rt = sum(runtimes) / max(len(runtimes), 1)
    print(f"  TOTAL  feasible={feasible}/{total}  avg_cost={avg_cost:.4f}  "
          f"avg_runtime={avg_rt:.2f}s")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run optimizer on multiple validation cases in parallel."
    )
    parser.add_argument(
        "--optimizer", default="test_optimizer.py",
        help="Optimizer file to evaluate (default: test_optimizer.py)",
    )
    parser.add_argument(
        "--test-ids", nargs="*", type=int, default=None,
        help="Case IDs to run. Default: all 100 (0-99).",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel worker processes (default: 10).",
    )
    parser.add_argument(
        "--summary", default=None,
        help="Path for combined summary JSON. "
             "Default: results_json/summary_<timestamp>.json",
    )
    args = parser.parse_args()

    test_ids: List[int] = args.test_ids if args.test_ids is not None else list(range(100))
    optimizer_path = str(Path(args.optimizer).resolve())
    n_workers = min(args.workers, len(test_ids))
    summary_path = (
        Path(args.summary) if args.summary
        else RESULTS_DIR / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(test_ids)} cases with {n_workers} workers …")
    print(f"Optimizer : {optimizer_path}")
    print(f"Output dir: {RESULTS_DIR}")
    print(f"Summary   : {summary_path}")
    print()

    tasks = [(tid, optimizer_path) for tid in test_ids]
    all_results: List[Dict[str, Any]] = []
    done = 0

    wall_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_case, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {
                    "test_id": tid, "feasible": False, "cost": 10.0,
                    "runtime_s": 0.0, "hard_ok": False,
                    "overlap_ok": None, "area_ok": None, "dimension_ok": None,
                    "soft_boundary": None, "soft_grouping": None, "soft_mib": None,
                    "v_rel": None, "output_json": "", "error": str(exc),
                }
            all_results.append(res)
            done += 1
            status = "OK " if res["feasible"] else "ERR"
            print(
                f"  [{done:3d}/{len(tasks)}] case {tid:3d}  "
                f"{status}  cost={res['cost']:.4f}  rt={res['runtime_s']:.1f}s"
                + (f"  !! {res['error']}" if res.get("error") else "")
            )

    wall_elapsed = time.time() - wall_start
    print(f"\nAll done in {wall_elapsed:.1f}s wall time.\n")

    all_results.sort(key=lambda r: r["test_id"])
    _print_summary(all_results, args.optimizer)

    # Write combined summary (without bulky raw_stdout)
    clean = [
        {k: v for k, v in r.items() if k != "raw_stdout"}
        for r in all_results
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "optimizer": args.optimizer,
        "timestamp": datetime.now().isoformat(),
        "wall_time_s": round(wall_elapsed, 2),
        "n_cases": len(all_results),
        "n_feasible": sum(1 for r in all_results if r["feasible"]),
        "avg_cost": sum(r["cost"] for r in all_results) / max(len(all_results), 1),
        "results": clean,
    }, indent=2, default=str))
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
