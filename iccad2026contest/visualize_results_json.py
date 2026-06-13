#!/usr/bin/env python3
"""
Visualize cases from an evaluation results JSON (e.g. my_optimizer_results.json).

Goal: match the same basic style as `python iccad2026_evaluate.py --visualize --test-id N`
(colored rectangles, block indices, equal aspect, autoscale).

Examples:
  python visualize_results_json.py --results-json my_optimizer_results.json --test-id 40
  python visualize_results_json.py --results-json my_optimizer_results.json --all
  python visualize_results_json.py --results-json my_optimizer_results.json --all --no-gt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DEFAULT_RESULTS_DIR = Path(current_dir) / "images" / "results"


def _extract_gt_positions_from_polygons(
    polygons, block_count: int
) -> List[Tuple[float, float, float, float]]:
    """Convert polygon vertex tensors into (x,y,w,h) rectangles via bbox."""
    gt_positions: List[Tuple[float, float, float, float]] = []
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


def _plot_positions(
    ax, positions: List[Tuple[float, float, float, float]], title: str
) -> None:
    import matplotlib.patches as mpatches

    block_count = len(positions)
    ax.set_title(title)

    denom = max(block_count - 1, 1)
    color_positions = [i / denom for i in range(block_count)]
    facecolors = __import__("matplotlib.pyplot").pyplot.cm.tab20(color_positions)

    for i, (x, y, w, h) in enumerate(positions):
        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            fill=True,
            facecolor=facecolors[i % len(facecolors)],
            edgecolor="black",
            alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=8)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def _load_results_data(results_json_path: Path) -> dict:
    return json.loads(results_json_path.read_text())


def _default_name_prefix(results_json_path: Path, data: Optional[dict] = None) -> str:
    if data is None:
        data = _load_results_data(results_json_path)
    submission = data.get("submission_name")
    if submission:
        return str(submission)
    stem = results_json_path.stem
    if stem.endswith("_results"):
        return stem[: -len("_results")]
    return stem


def _all_test_ids(data: dict) -> List[int]:
    ids = [int(tr["test_id"]) for tr in data.get("test_results", [])]
    return sorted(ids)


def _load_positions_from_results_json(
    data: dict, test_id: int, results_json_path: Path
) -> Tuple[List[Tuple[float, float, float, float]], Optional[int]]:
    for tr in data.get("test_results", []):
        if int(tr.get("test_id")) == int(test_id):
            positions = tr.get("positions")
            if not positions:
                raise ValueError(
                    f"test_id={test_id} has no 'positions' in {results_json_path}"
                )
            pos_tuples = [tuple(map(float, p)) for p in positions]
            block_count = tr.get("block_count")
            return pos_tuples, (
                int(block_count) if block_count is not None else None
            )
    raise ValueError(f"test_id={test_id} not found in {results_json_path}")


def visualize_results_case(
    results_json_path: Path,
    test_id: int,
    data_path: str = "../",
    out_path: Optional[Path] = None,
    show_gt: bool = True,
    name_prefix: Optional[str] = None,
    data: Optional[dict] = None,
) -> Path:
    """Save a PNG for one test_id from a results JSON file."""
    if data is None:
        data = _load_results_data(results_json_path)

    positions, block_count_hint = _load_positions_from_results_json(
        data, test_id, results_json_path
    )
    block_count = len(positions)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Please `pip install matplotlib`."
        ) from e

    gt_positions = None
    if show_gt:
        from lite_dataset_test import FloorplanDatasetLiteTest  # type: ignore

        dataset = FloorplanDatasetLiteTest(data_path)
        sample = dataset[test_id]
        inputs, labels = sample["input"], sample["label"]
        area_target, *_rest = inputs
        polygons, _metrics = labels
        inferred_block_count = int((area_target != -1).sum().item())

        if inferred_block_count != block_count:
            block_count = inferred_block_count
            positions = positions[:block_count]

        gt_positions = _extract_gt_positions_from_polygons(polygons, block_count)

    prefix = name_prefix or _default_name_prefix(results_json_path, data)
    if out_path is None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_RESULTS_DIR / f"{prefix}_case_{test_id}.png"

    if gt_positions is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        _plot_positions(
            axes[0],
            gt_positions,
            title=f"Validation Case {test_id} - Ground Truth ({block_count} blocks)",
        )
        _plot_positions(
            axes[1],
            positions[:block_count],
            title=f"Result JSON - Solution ({block_count} blocks)",
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        _plot_positions(
            ax,
            positions[:block_count],
            title=(
                f"Result JSON - Solution (test_id={test_id}, "
                f"{block_count} blocks)"
            ),
        )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize positions from an evaluation results JSON."
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Path to *_results.json from `iccad2026_evaluate.py --evaluate`",
    )
    parser.add_argument(
        "--test-id",
        type=int,
        default=None,
        help="Validation case index (0-99). Required unless --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export PNGs for every test_id in the results JSON.",
    )
    parser.add_argument(
        "--data-path",
        default="../",
        help="FloorSet data root (default: ../). Used for ground-truth panel.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG for a single --test-id (overrides default path).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Output directory for --all (default: {DEFAULT_RESULTS_DIR.relative_to(current_dir)}).",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Filename prefix (default: submission_name or stem of results JSON).",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Plot solution only (no ground-truth panel). Faster for --all.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window (single --test-id only).",
    )
    args = parser.parse_args()

    if not args.all and args.test_id is None:
        parser.error("Provide --test-id N or use --all")

    results_json_path = Path(args.results_json)
    if not results_json_path.exists():
        raise SystemExit(f"Results JSON not found: {results_json_path}")

    data = _load_results_data(results_json_path)
    name_prefix = args.name_prefix or _default_name_prefix(results_json_path, data)
    show_gt = not args.no_gt

    if args.all:
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        test_ids = _all_test_ids(data)
        print(
            f"Exporting {len(test_ids)} case(s) to {out_dir} "
            f"(prefix={name_prefix}, gt={'yes' if show_gt else 'no'})"
        )
        for tid in test_ids:
            visualize_results_case(
                results_json_path,
                tid,
                data_path=args.data_path,
                out_path=out_dir / f"{name_prefix}_case_{tid}.png",
                show_gt=show_gt,
                name_prefix=name_prefix,
                data=data,
            )
    else:
        out = Path(args.out) if args.out else None
        visualize_results_case(
            results_json_path,
            args.test_id,
            data_path=args.data_path,
            out_path=out,
            show_gt=show_gt,
            name_prefix=name_prefix,
            data=data,
        )
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
