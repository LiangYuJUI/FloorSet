#!/usr/bin/env python3
"""
Baseline constraint visualizations for validation cases.

Modes:
  preplaced — preplaced blocks + all fixed pin positions
  boundary  — blocks with boundary soft constraints (constraints[:, 4] != 0) + pins
  cluster   — blocks with grouping/cluster constraints (constraints[:, 3] != 0) + pins
  floorplan — entire layout (all blocks) + pins
  combine   — stitch preplaced, boundary, cluster, floorplan PNGs into one 2x2 image

Outputs (by default):
  images/baseline/preplaced/<prefix>_case_<N>.png
  images/baseline/boundary/<prefix>_case_<N>.png
  images/baseline/cluster/<prefix>_case_<N>.png
  images/baseline/floorplan/<prefix>_case_<N>.png
  images/baseline/combined/<prefix>_case_<N>.png

Examples:
  python visualize_baseline.py --mode preplaced --test-id 40
  python visualize_baseline.py --mode boundary --all
  python visualize_baseline.py --mode cluster --test-id 97
  python visualize_baseline.py --mode combine --test-id 62
  python visualize_baseline.py --mode combine --test-id 62 --generate-missing
  python visualize_baseline.py --mode preplaced --test-id 40 \\
      --results-json my_optimizer_results.json --name-prefix my_optimizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

Rect = Tuple[float, float, float, float]
Pin = Tuple[int, float, float]

BASELINE_ROOT = Path(current_dir) / "images" / "baseline"
DEFAULT_PREPLACED_DIR = BASELINE_ROOT / "preplaced"
DEFAULT_BOUNDARY_DIR = BASELINE_ROOT / "boundary"
DEFAULT_CLUSTER_DIR = BASELINE_ROOT / "cluster"
DEFAULT_FLOORPLAN_DIR = BASELINE_ROOT / "floorplan"
DEFAULT_COMBINED_DIR = BASELINE_ROOT / "combined"

# (panel title, subdirectory under images/baseline/, default filename prefix)
COMBINE_PANELS: Tuple[Tuple[str, Path, str], ...] = (
    ("Preplaced", DEFAULT_PREPLACED_DIR, "preplaced"),
    ("Boundary", DEFAULT_BOUNDARY_DIR, "boundary"),
    ("Cluster", DEFAULT_CLUSTER_DIR, "cluster"),
    ("Floorplan", DEFAULT_FLOORPLAN_DIR, "floorplan"),
)

# Bitmask: 1=left, 2=right, 4=top, 8=bottom (iccad2026_evaluate.py)
BOUNDARY_FACE_COLORS: Dict[int, str] = {
    1: "lightsteelblue",
    2: "yellowgreen",
    4: "deepskyblue",
    8: "goldenrod",
    5: "mediumpurple",
    6: "pink",
    9: "brown",
    10: "beige",
}


def _extract_gt_positions_from_polygons(
    polygons, block_count: int
) -> List[Rect]:
    gt_positions: List[Rect] = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            gt_positions.append(
                (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))
            )
        else:
            gt_positions.append((0.0, 0.0, 1.0, 1.0))
    return gt_positions


def _preplaced_indices(constraints, block_count: int) -> List[int]:
    if constraints is None:
        return []
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    if nc <= 1:
        return []
    return [i for i in range(block_count) if constraints[i, 1] != 0]


def _boundary_indices(constraints, block_count: int) -> List[int]:
    if constraints is None:
        return []
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    if nc <= 4:
        return []
    return [i for i in range(block_count) if constraints[i, 4] != 0]


def _cluster_indices(constraints, block_count: int) -> List[int]:
    if constraints is None:
        return []
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    if nc <= 3:
        return []
    return [i for i in range(block_count) if constraints[i, 3] != 0]


def _cluster_id(constraints, block_id: int) -> int:
    return int(constraints[block_id, 3].item())


def _cluster_color_map(cluster_ids: Sequence[int]) -> Dict[int, Tuple[float, float, float, float]]:
    """One distinct color per cluster id; same id always maps to the same color."""
    import matplotlib.pyplot as plt

    unique = sorted(set(cluster_ids))
    n = len(unique)
    cmap = plt.cm.tab20
    if n > 20:
        cmap = plt.cm.tab20b
    return {cid: cmap(i / max(n - 1, 1)) for i, cid in enumerate(unique)}


def _boundary_code(constraints, block_id: int) -> int:
    return int(constraints[block_id, 4].item())


def _boundary_label(code: int) -> str:
    named = {5: "TL", 6: "TR", 9: "BL", 10: "BR"}
    if code in named:
        return named[code]
    parts: List[str] = []
    if code & 1:
        parts.append("L")
    if code & 2:
        parts.append("R")
    if code & 4:
        parts.append("T")
    if code & 8:
        parts.append("B")
    return "+".join(parts) if parts else str(code)


def _boundary_face_color(code: int) -> str:
    if code in BOUNDARY_FACE_COLORS:
        return BOUNDARY_FACE_COLORS[code]
    return "coral"


def _load_solution_positions(
    results_json_path: Path, test_id: int, block_count: int
) -> List[Rect]:
    data = json.loads(results_json_path.read_text())
    for tr in data.get("test_results", []):
        if int(tr.get("test_id")) != int(test_id):
            continue
        positions = tr.get("positions")
        if not positions:
            raise ValueError(
                f"test_id={test_id} has no 'positions' in {results_json_path}"
            )
        return [tuple(map(float, p)) for p in positions][:block_count]
    raise ValueError(f"test_id={test_id} not found in {results_json_path}")


def _subset_rects(
    indices: Sequence[int],
    all_positions: Sequence[Rect],
) -> Tuple[List[int], List[Rect]]:
    order = list(indices)
    return order, [all_positions[i] for i in order]


def _valid_pins(pins_pos) -> List[Pin]:
    pins: List[Pin] = []
    if pins_pos is None or not hasattr(pins_pos, "shape") or pins_pos.numel() == 0:
        return pins
    for i in range(int(pins_pos.shape[0])):
        x = float(pins_pos[i, 0])
        y = float(pins_pos[i, 1])
        if x >= 0 and y >= 0:
            pins.append((i, x, y))
    return pins


def _canvas_limits(
    baseline_positions: Sequence[Rect],
    pins: Sequence[Pin],
    margin_frac: float = 0.05,
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for x, y, w, h in baseline_positions:
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    for _pin_id, px, py in pins:
        xs.append(px)
        ys.append(py)
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    mx = dx * margin_frac
    my = dy * margin_frac
    return x_min - mx, x_max + mx, y_min - my, y_max + my


def _solution_bbox(positions: Sequence[Rect]) -> Tuple[float, float, float, float]:
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    return x_min, y_min, x_max - x_min, y_max - y_min


def _plot_pins(ax, pins: Sequence[Pin], span: float) -> None:
    import matplotlib.patches as mpatches

    pin_radius = span * 0.008
    pin_font = max(6, min(8, int(7 * span / 200)))
    for pin_id, px, py in pins:
        ax.add_patch(
            mpatches.Circle(
                (px, py),
                radius=pin_radius,
                fill=True,
                facecolor="limegreen",
                edgecolor="darkgreen",
                linewidth=1.0,
                alpha=0.95,
                zorder=6,
            )
        )
        ax.text(
            px,
            py + pin_radius * 1.6,
            str(pin_id),
            ha="center",
            va="bottom",
            fontsize=pin_font,
            color="darkgreen",
            zorder=7,
        )


def _apply_canvas(
    ax,
    baseline_positions: Sequence[Rect],
    pins: Sequence[Pin],
) -> float:
    x_lo, x_hi, y_lo, y_hi = _canvas_limits(baseline_positions, pins)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return max(x_hi - x_lo, y_hi - y_lo, 1e-6)


def _plot_floorplan(
    ax,
    positions: Sequence[Rect],
    title: str,
    baseline_positions: Sequence[Rect],
    pins: Sequence[Pin],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    ax.set_title(title)
    span = _apply_canvas(ax, baseline_positions, pins)
    block_count = len(positions)
    denom = max(block_count - 1, 1)
    facecolors = plt.cm.tab20([i / denom for i in range(block_count)])
    label_font = max(5, min(8, int(90 / max(block_count, 1))))

    for i, (x, y, w, h) in enumerate(positions):
        ax.add_patch(
            mpatches.Rectangle(
                (x, y),
                w,
                h,
                fill=True,
                facecolor=facecolors[i % len(facecolors)],
                edgecolor="black",
                alpha=0.7,
                linewidth=0.8,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            str(i),
            ha="center",
            va="center",
            fontsize=label_font,
        )

    _plot_pins(ax, pins, span)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="steelblue",
                markeredgecolor="black",
                markersize=8,
                label=f"block ({block_count})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="limegreen",
                markeredgecolor="darkgreen",
                markersize=8,
                label=f"pin ({len(pins)})",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )


def _plot_preplaced_blocks(
    ax,
    block_indices: Sequence[int],
    positions: Sequence[Rect],
    title: str,
    baseline_positions: Sequence[Rect],
    pins: Sequence[Pin],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    ax.set_title(title)
    span = _apply_canvas(ax, baseline_positions, pins)

    n = len(block_indices)
    if n == 0:
        ax.text(
            0.5,
            0.5,
            "No preplaced blocks",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        denom = max(n - 1, 1)
        facecolors = plt.cm.tab20([i / denom for i in range(n)])
        for j, (block_id, (x, y, w, h)) in enumerate(zip(block_indices, positions)):
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=True,
                    facecolor=facecolors[j % len(facecolors)],
                    edgecolor="black",
                    alpha=0.85,
                    linewidth=1.2,
                )
            )
            ax.text(
                x + w / 2,
                y + h / 2,
                str(block_id),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    _plot_pins(ax, pins, span)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="steelblue",
                markeredgecolor="black",
                markersize=8,
                label="preplaced block",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="limegreen",
                markeredgecolor="darkgreen",
                markersize=8,
                label=f"pin ({len(pins)})",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )


def _plot_boundary_blocks(
    ax,
    block_indices: Sequence[int],
    positions: Sequence[Rect],
    boundary_codes: Sequence[int],
    title: str,
    baseline_positions: Sequence[Rect],
    pins: Sequence[Pin],
) -> None:
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    ax.set_title(title)
    span = _apply_canvas(ax, baseline_positions, pins)

    bx, by, bw, bh = _solution_bbox(baseline_positions)
    ax.add_patch(
        mpatches.Rectangle(
            (bx, by),
            bw,
            bh,
            fill=False,
            linestyle="--",
            edgecolor="dimgray",
            linewidth=1.5,
            zorder=1,
        )
    )
    ax.text(
        bx + bw * 0.02,
        by + bh * 0.98,
        "solution bbox",
        ha="left",
        va="top",
        fontsize=7,
        color="dimgray",
    )

    if not block_indices:
        ax.text(
            0.5,
            0.5,
            "No boundary-constrained blocks",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        for block_id, (x, y, w, h), code in zip(
            block_indices, positions, boundary_codes
        ):
            label = _boundary_label(code)
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=True,
                    facecolor=_boundary_face_color(code),
                    edgecolor="black",
                    alpha=0.85,
                    linewidth=1.2,
                    zorder=4,
                )
            )
            ax.text(
                x + w / 2,
                y + h / 2,
                f"{block_id}\n{label}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    _plot_pins(ax, pins, span)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="coral",
                markeredgecolor="black",
                markersize=8,
                label=f"boundary block ({len(block_indices)})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="limegreen",
                markeredgecolor="darkgreen",
                markersize=8,
                label=f"pin ({len(pins)})",
            ),
            Line2D(
                [0],
                [0],
                linestyle="--",
                color="dimgray",
                label="solution bbox",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )


def _plot_cluster_blocks(
    ax,
    block_indices: Sequence[int],
    positions: Sequence[Rect],
    cluster_ids: Sequence[int],
    title: str,
    baseline_positions: Sequence[Rect],
    pins: Sequence[Pin],
) -> None:
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    ax.set_title(title)
    span = _apply_canvas(ax, baseline_positions, pins)
    color_by_cluster = _cluster_color_map(cluster_ids)

    if not block_indices:
        ax.text(
            0.5,
            0.5,
            "No cluster blocks",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        for block_id, (x, y, w, h), cid in zip(
            block_indices, positions, cluster_ids
        ):
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=True,
                    facecolor=color_by_cluster[cid],
                    edgecolor="black",
                    alpha=0.85,
                    linewidth=1.2,
                    zorder=4,
                )
            )
            ax.text(
                x + w / 2,
                y + h / 2,
                f"{block_id}\nG{cid}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    _plot_pins(ax, pins, span)

    unique_clusters = sorted(set(cluster_ids))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color_by_cluster[cid],
            markeredgecolor="black",
            markersize=8,
            label=f"G{cid}",
        )
        for cid in unique_clusters
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="limegreen",
            markeredgecolor="darkgreen",
            markersize=8,
            label=f"pin ({len(pins)})",
        )
    )
    ncol = min(4, max(len(unique_clusters), 1))
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        ncol=ncol,
    )


def _load_case(test_id: int, data_path: str):
    from lite_dataset_test import FloorplanDatasetLiteTest  # type: ignore

    dataset = FloorplanDatasetLiteTest(data_path)
    sample = dataset[test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, _b2b, _p2b, pins_pos, constraints = inputs
    polygons, _metrics = labels
    block_count = int((area_target != -1).sum().item())
    gt_positions = _extract_gt_positions_from_polygons(polygons, block_count)
    preplaced = _preplaced_indices(constraints, block_count)
    boundary = _boundary_indices(constraints, block_count)
    pins = _valid_pins(pins_pos)
    cluster = _cluster_indices(constraints, block_count)
    return block_count, constraints, gt_positions, preplaced, boundary, cluster, pins


def _resolve_positions(
    test_id: int,
    block_count: int,
    gt_positions: List[Rect],
    results_json_path: Optional[Path],
) -> Tuple[List[Rect], str]:
    if results_json_path is not None:
        return (
            _load_solution_positions(results_json_path, test_id, block_count),
            f"solution ({results_json_path.name})",
        )
    return gt_positions, "reference layout (target)"


def visualize_preplaced_case(
    test_id: int,
    data_path: str = "../",
    results_json_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    name_prefix: str = "preplaced",
) -> Path:
    block_count, _constraints, gt_positions, preplaced, _boundary, _cluster, pins = (
        _load_case(test_id, data_path)
    )
    all_positions, source = _resolve_positions(
        test_id, block_count, gt_positions, results_json_path
    )
    block_indices, rects = _subset_rects(preplaced, all_positions)

    if out_path is None:
        DEFAULT_PREPLACED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_PREPLACED_DIR / f"{name_prefix}_case_{test_id}.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Please `pip install matplotlib`.") from e

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _plot_preplaced_blocks(
        ax,
        block_indices,
        rects,
        title=(
            f"Case {test_id} — preplaced ({len(block_indices)}/{block_count}) "
            f"+ pins ({len(pins)})\n{source}"
        ),
        baseline_positions=gt_positions,
        pins=pins,
    )
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"[preplaced] {out_path} "
        f"({len(block_indices)} block(s), {len(pins)} pin(s))"
    )
    return out_path


def visualize_boundary_case(
    test_id: int,
    data_path: str = "../",
    results_json_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    name_prefix: str = "boundary",
) -> Path:
    block_count, constraints, gt_positions, _preplaced, boundary, _cluster, pins = (
        _load_case(test_id, data_path)
    )
    all_positions, source = _resolve_positions(
        test_id, block_count, gt_positions, results_json_path
    )
    block_indices, rects = _subset_rects(boundary, all_positions)
    codes = [_boundary_code(constraints, i) for i in block_indices]

    if out_path is None:
        DEFAULT_BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_BOUNDARY_DIR / f"{name_prefix}_case_{test_id}.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Please `pip install matplotlib`.") from e

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _plot_boundary_blocks(
        ax,
        block_indices,
        rects,
        codes,
        title=(
            f"Case {test_id} — boundary ({len(block_indices)}/{block_count}) "
            f"+ pins ({len(pins)})\n{source}"
        ),
        baseline_positions=gt_positions,
        pins=pins,
    )
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"[boundary] {out_path} "
        f"({len(block_indices)} block(s), {len(pins)} pin(s))"
    )
    return out_path


def visualize_cluster_case(
    test_id: int,
    data_path: str = "../",
    results_json_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    name_prefix: str = "cluster",
) -> Path:
    block_count, constraints, gt_positions, _preplaced, _boundary, cluster, pins = (
        _load_case(test_id, data_path)
    )
    all_positions, source = _resolve_positions(
        test_id, block_count, gt_positions, results_json_path
    )
    block_indices, rects = _subset_rects(cluster, all_positions)
    cluster_ids = [_cluster_id(constraints, i) for i in block_indices]
    n_groups = len(set(cluster_ids))

    if out_path is None:
        DEFAULT_CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_CLUSTER_DIR / f"{name_prefix}_case_{test_id}.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Please `pip install matplotlib`.") from e

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _plot_cluster_blocks(
        ax,
        block_indices,
        rects,
        cluster_ids,
        title=(
            f"Case {test_id} — cluster ({len(block_indices)}/{block_count} blocks, "
            f"{n_groups} group(s)) + pins ({len(pins)})\n{source}"
        ),
        baseline_positions=gt_positions,
        pins=pins,
    )
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"[cluster] {out_path} "
        f"({len(block_indices)} block(s), {n_groups} cluster(s), {len(pins)} pin(s))"
    )
    return out_path


def visualize_floorplan_case(
    test_id: int,
    data_path: str = "../",
    results_json_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    name_prefix: str = "floorplan",
) -> Path:
    block_count, _constraints, gt_positions, _preplaced, _boundary, _cluster, pins = (
        _load_case(test_id, data_path)
    )
    all_positions, source = _resolve_positions(
        test_id, block_count, gt_positions, results_json_path
    )

    if out_path is None:
        DEFAULT_FLOORPLAN_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_FLOORPLAN_DIR / f"{name_prefix}_case_{test_id}.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Please `pip install matplotlib`.") from e

    fig_size = 8 if block_count <= 60 else 10
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    _plot_floorplan(
        ax,
        all_positions,
        title=(
            f"Case {test_id} — floorplan ({block_count} blocks, {len(pins)} pins)\n"
            f"{source}"
        ),
        baseline_positions=gt_positions,
        pins=pins,
    )
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"[floorplan] {out_path} "
        f"({block_count} block(s), {len(pins)} pin(s))"
    )
    return out_path


def _panel_image_path(
    panel_dir: Path,
    file_prefix: str,
    test_id: int,
) -> Path:
    return panel_dir / f"{file_prefix}_case_{test_id}.png"


def _resolve_combine_inputs(
    test_id: int,
    file_prefixes: Optional[Sequence[str]] = None,
) -> Tuple[List[Path], List[str]]:
    """Return the four panel PNG paths and titles."""
    paths: List[Path] = []
    titles: List[str] = []
    for i, (title, panel_dir, default_prefix) in enumerate(COMBINE_PANELS):
        prefix = (
            file_prefixes[i]
            if file_prefixes is not None
            else default_prefix
        )
        paths.append(_panel_image_path(panel_dir, prefix, test_id))
        titles.append(title)
    return paths, titles


def combine_baseline_images(
    test_id: int,
    out_path: Optional[Path] = None,
    file_prefixes: Optional[Sequence[str]] = None,
    generate_missing: bool = False,
    data_path: str = "../",
    results_json_path: Optional[Path] = None,
    name_prefix: str = "combined",
) -> Path:
    """
    Combine preplaced, boundary, cluster, and floorplan PNGs into one 2x2 image.

    Layout (left-to-right, top-to-bottom): preplaced | boundary
                                           cluster  | floorplan

    By default each panel uses `<mode>_case_<N>.png` from its mode directory.
    Pass file_prefixes=[p1,p2,p3,p4] or set --name-prefix to use one prefix for all four.
    """
    panel_paths, panel_titles = _resolve_combine_inputs(test_id, file_prefixes)

    if generate_missing:
        generators = (
            visualize_preplaced_case,
            visualize_boundary_case,
            visualize_cluster_case,
            visualize_floorplan_case,
        )
        prefixes = file_prefixes or [p[2] for p in COMBINE_PANELS]
        for gen, path, prefix in zip(generators, panel_paths, prefixes):
            if not path.exists():
                gen(
                    test_id,
                    data_path=data_path,
                    results_json_path=results_json_path,
                    out_path=path,
                    name_prefix=prefix,
                )

    missing = [p for p in panel_paths if not p.exists()]
    if missing:
        lines = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "Missing panel image(s). Generate them first, e.g.:\n"
            f"  python visualize_baseline.py --mode preplaced --test-id {test_id}\n"
            f"  python visualize_baseline.py --mode boundary --test-id {test_id}\n"
            f"  python visualize_baseline.py --mode cluster --test-id {test_id}\n"
            f"  python visualize_baseline.py --mode floorplan --test-id {test_id}\n"
            "Or rerun with --generate-missing\n"
            f"  {lines}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Please `pip install matplotlib`.") from e

    if out_path is None:
        DEFAULT_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_COMBINED_DIR / f"{name_prefix}_case_{test_id}.png"

    images = [plt.imread(str(p)) for p in panel_paths]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f"Case {test_id} — baseline overview", fontsize=14, y=0.98)

    for ax, image, title, src in zip(
        axes.flat, images, panel_titles, panel_paths
    ):
        ax.imshow(image, aspect="auto")
        ax.set_title(f"{title}\n({src.name})", fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[combine] {out_path}")
    return out_path


_MODE_CONFIG = {
    "preplaced": (visualize_preplaced_case, DEFAULT_PREPLACED_DIR),
    "boundary": (visualize_boundary_case, DEFAULT_BOUNDARY_DIR),
    "cluster": (visualize_cluster_case, DEFAULT_CLUSTER_DIR),
    "floorplan": (visualize_floorplan_case, DEFAULT_FLOORPLAN_DIR),
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline constraint PNGs: preplaced, boundary, cluster, floorplan, "
            "or combine four panels into one image."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(*tuple(_MODE_CONFIG.keys()), "combine"),
        default="preplaced",
        help="Visualization type (default: preplaced).",
    )
    parser.add_argument("--test-id", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-path", default="../")
    parser.add_argument("--results-json", default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG for a single --test-id.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for --all (mode-specific default under images/baseline/).",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Filename prefix (default: mode name).",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="For --mode combine: render any missing panel PNG before stitching.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.all and args.test_id is None:
        parser.error("Provide --test-id N or use --all")

    results_json = Path(args.results_json) if args.results_json else None
    if results_json is not None and not results_json.exists():
        raise SystemExit(f"Results JSON not found: {results_json}")

    test_ids = list(range(100)) if args.all else [int(args.test_id)]

    if args.mode == "combine":
        file_prefixes = (
            [args.name_prefix] * 4
            if args.name_prefix is not None
            else None
        )
        combine_prefix = args.name_prefix or "combined"
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_COMBINED_DIR
        for tid in test_ids:
            out = (
                out_dir / f"{combine_prefix}_case_{tid}.png"
                if args.all
                else (Path(args.out) if args.out else None)
            )
            combine_baseline_images(
                tid,
                out_path=out,
                file_prefixes=file_prefixes,
                generate_missing=args.generate_missing,
                data_path=args.data_path,
                results_json_path=results_json,
                name_prefix=combine_prefix,
            )
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()
        return

    name_prefix = args.name_prefix or args.mode
    visualize_fn, default_dir = _MODE_CONFIG[args.mode]

    if args.all:
        out_dir = Path(args.out_dir) if args.out_dir else default_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for tid in test_ids:
            visualize_fn(
                tid,
                data_path=args.data_path,
                results_json_path=results_json,
                out_path=out_dir / f"{name_prefix}_case_{tid}.png",
                name_prefix=name_prefix,
            )
    else:
        out = Path(args.out) if args.out else None
        visualize_fn(
            test_ids[0],
            data_path=args.data_path,
            results_json_path=results_json,
            out_path=out,
            name_prefix=name_prefix,
        )

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
