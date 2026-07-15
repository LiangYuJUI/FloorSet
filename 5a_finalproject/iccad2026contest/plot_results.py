#!/usr/bin/env python3
"""
Plot evaluation results from test_optimizer_results.json.

Generates:
  - cost_vs_blocks.png
  - runtime_vs_blocks.png
  - results_summary.png  (both plots stacked in one figure)

Run from the iccad2026contest directory (no arguments required):
  python plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from iccad2026_evaluate import compute_total_score
RESULTS_JSON = SCRIPT_DIR / "test_optimizer_results.json"
DPI = 150
FIGSIZE_COMBINED = (8, 10)
FIGSIZE_SINGLE = (8, 5)


def total_score_from_stored_costs(
    test_results: List[Dict[str, Any]],
) -> float:
    """Weighted total score from stored per-case costs in the JSON file."""
    costs = [float(entry["cost"]) for entry in test_results]
    blocks = [
        int(entry.get("block_count", entry.get("test_id", 0) + 21))
        for entry in test_results
    ]
    return compute_total_score(costs, blocks)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_results(path: Path) -> Dict[str, Any]:
    """Load and validate the evaluation JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "test_results" not in data:
        raise KeyError(f"Missing 'test_results' key in {path}")
    return data


def extract_series(
    test_results: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract block counts, costs, runtimes, and feasibility flags.

    Uses block_count from each entry; falls back to test_id + 21 only if missing.
    Returns sorted arrays (by block_count).
    """
    blocks: List[int] = []
    costs: List[float] = []
    runtimes: List[float] = []
    feasible: List[bool] = []
    test_ids: List[int] = []

    for entry in test_results:
        tid = int(entry.get("test_id", len(test_ids)))
        bc = entry.get("block_count")
        if bc is None:
            bc = tid + 21  # fallback (should not be needed)
        blocks.append(int(bc))
        costs.append(float(entry["cost"]))
        runtimes.append(float(entry.get("runtime_seconds", 0.0)))
        feasible.append(bool(entry.get("is_feasible", True)))
        test_ids.append(tid)

    # Sort by block count for a clean line plot
    order = np.argsort(blocks)
    return (
        np.array(blocks)[order],
        np.array(costs)[order],
        np.array(runtimes)[order],
        np.array(feasible)[order],
        np.array(test_ids)[order],
    )


def _plot_cost_axis(
    ax: plt.Axes,
    blocks: np.ndarray,
    costs: np.ndarray,
    feasible: np.ndarray,
    submission_name: str,
) -> None:
    """Draw cost vs block count on the given axes."""
    feas_mask = feasible.astype(bool)
    infeas_mask = ~feas_mask

    if feas_mask.any():
        ax.plot(
            blocks[feas_mask],
            costs[feas_mask],
            "-o",
            color="#2563eb",
            markersize=5,
            linewidth=1.2,
            label="Feasible",
            zorder=3,
        )
    if infeas_mask.any():
        ax.plot(
            blocks[infeas_mask],
            costs[infeas_mask],
            "x",
            color="#dc2626",
            markersize=8,
            linewidth=0,
            label="Infeasible",
            zorder=4,
        )

    ax.set_xlabel("Number of Blocks", fontsize=11)
    ax.set_ylabel("Cost", fontsize=11)
    ax.set_title(
        f"Cost vs. Block Count  ({submission_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(left=0)


def _plot_runtime_axis(
    ax: plt.Axes,
    blocks: np.ndarray,
    runtimes: np.ndarray,
    feasible: np.ndarray,
    submission_name: str,
) -> None:
    """Draw runtime vs block count on the given axes."""
    feas_mask = feasible.astype(bool)
    infeas_mask = ~feas_mask

    if feas_mask.any():
        ax.plot(
            blocks[feas_mask],
            runtimes[feas_mask],
            "-o",
            color="#059669",
            markersize=5,
            linewidth=1.2,
            label="Feasible",
            zorder=3,
        )
    if infeas_mask.any():
        ax.plot(
            blocks[infeas_mask],
            runtimes[infeas_mask],
            "x",
            color="#dc2626",
            markersize=8,
            linewidth=0,
            label="Infeasible",
            zorder=4,
        )

    ax.set_xlabel("Number of Blocks", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title(
        f"Runtime vs. Block Count  ({submission_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(left=0)


def save_single_plots(
    blocks: np.ndarray,
    costs: np.ndarray,
    runtimes: np.ndarray,
    feasible: np.ndarray,
    submission_name: str,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Save individual PNG files for cost and runtime."""
    cost_path = out_dir / "cost_vs_blocks.png"
    runtime_path = out_dir / "runtime_vs_blocks.png"

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _plot_cost_axis(ax, blocks, costs, feasible, submission_name)
    fig.tight_layout()
    fig.savefig(cost_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _plot_runtime_axis(ax, blocks, runtimes, feasible, submission_name)
    fig.tight_layout()
    fig.savefig(runtime_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return cost_path, runtime_path


def save_combined_plot(
    blocks: np.ndarray,
    costs: np.ndarray,
    runtimes: np.ndarray,
    feasible: np.ndarray,
    submission_name: str,
    timestamp: str,
    out_dir: Path,
) -> Path:
    """Save a vertically stacked figure with both subplots."""
    combined_path = out_dir / "results_summary.png"

    fig, (ax_cost, ax_rt) = plt.subplots(2, 1, figsize=FIGSIZE_COMBINED)
    fig.suptitle(
        f"{submission_name} — evaluation summary   |   {timestamp}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    _plot_cost_axis(ax_cost, blocks, costs, feasible, submission_name)
    _plot_runtime_axis(ax_rt, blocks, runtimes, feasible, submission_name)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(combined_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return combined_path


def print_summary(
    blocks: np.ndarray,
    costs: np.ndarray,
    runtimes: np.ndarray,
    feasible: np.ndarray,
    total_score: float,
    n_cases: int,
) -> None:
    """Print a short numeric summary to the terminal."""
    n_feas = int(feasible.sum())
    print(f"Loaded {n_cases} test cases.")
    print(f"  Feasible     : {n_feas} / {n_cases}")
    print(f"  Block range  : {blocks.min()} – {blocks.max()}")
    print(f"  Cost range   : {costs.min():.4f} – {costs.max():.4f}")
    print(f"  Runtime range: {runtimes.min():.2f}s – {runtimes.max():.2f}s")
    print(f"  Total score  : {total_score:.4f}")


def main() -> None:
    # 1. Load JSON
    data = load_results(RESULTS_JSON)

    submission_name = data.get("submission_name", "optimizer")
    timestamp = data.get("timestamp", "")
    test_results = data["test_results"]

    total_score = total_score_from_stored_costs(test_results)

    # 2. Extract arrays (stored costs from JSON)
    blocks, costs, runtimes, feasible, _ = extract_series(test_results)

    # 3. Print summary
    print_summary(blocks, costs, runtimes, feasible, total_score, len(test_results))

    # 4. Generate and save plots
    out_dir = SCRIPT_DIR
    cost_path, runtime_path = save_single_plots(
        blocks, costs, runtimes, feasible, submission_name, out_dir
    )
    combined_path = save_combined_plot(
        blocks, costs, runtimes, feasible, submission_name, timestamp, out_dir
    )

    print()
    print("Plots saved:")
    print(f"  {cost_path}")
    print(f"  {runtime_path}")
    print(f"  {combined_path}")


if __name__ == "__main__":
    main()
