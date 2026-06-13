#!/usr/bin/env python3
"""
Scan validation cases for bottom-left (BL) boundary blocks and their block type.

Boundary encoding (constraints[:, 4] bitmask):
  1=left, 2=right, 4=top, 8=bottom
  BL corner = left + bottom = 9

Block type (constraints columns 0–1, same as iccad2026_evaluate_test.py):
  - preplaced: constraints[i, 1] != 0
  - fixed-shape: constraints[i, 0] != 0 (and not preplaced)
  - soft: neither fixed nor preplaced (free block, area from target)

Usage:
  python check_bottom_left_blocks.py
  python check_bottom_left_blocks.py --data-path ../
  python check_bottom_left_blocks.py --test-id 40
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest

BOUNDARY_BOTTOM_LEFT = 1 + 8  # 9


@dataclass
class BLBlockInfo:
    block_id: int
    kind: str  # "preplaced", "fixed-shape", or "soft"


def classify_block(constraints: torch.Tensor, block_id: int) -> str:
    """Classify block as preplaced, fixed-shape, or soft."""
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc > 1 and float(constraints[block_id, 1].item()) != 0:
        return "preplaced"
    if nc > 0 and float(constraints[block_id, 0].item()) != 0:
        return "fixed-shape"
    return "soft"


def bottom_left_blocks(
    constraints: torch.Tensor, block_count: int
) -> List[BLBlockInfo]:
    """Return all BL corner blocks with type classification."""
    if constraints is None or block_count <= 0:
        return []
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 4:
        return []
    out: List[BLBlockInfo] = []
    for i in range(block_count):
        code = int(float(constraints[i, 4].item()))
        if code == BOUNDARY_BOTTOM_LEFT:
            out.append(BLBlockInfo(i, classify_block(constraints, i)))
    return out


def scan_case(
    dataset: FloorplanDatasetLiteTest, test_id: int
) -> Tuple[int, List[BLBlockInfo]]:
    """Return (block_count, list of BL blocks with types)."""
    sample = dataset[test_id]
    inputs = sample["input"]
    area_target, _b2b, _p2b, _pins, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    bl_blocks = bottom_left_blocks(constraints, block_count)
    return block_count, bl_blocks


def _format_bl_list(bl_blocks: List[BLBlockInfo]) -> str:
    if not bl_blocks:
        return "-"
    return ", ".join(f"{b.block_id}({b.kind})" for b in bl_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan BL boundary blocks and report soft vs preplaced (and fixed-shape)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../",
        help="FloorSet data root (default: ../)",
    )
    parser.add_argument(
        "--test-id",
        type=int,
        default=None,
        help="Scan a single case only (default: all cases in dataset)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Highlight cases with at least this many BL blocks (default: 2)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    print(f"Loading validation dataset from {data_path} ...")
    dataset = FloorplanDatasetLiteTest(str(data_path))
    n_cases = len(dataset)
    test_ids = [args.test_id] if args.test_id is not None else list(range(n_cases))

    case_counts = {0: 0, 1: 0, "other": 0}
    bl_kind_totals = {"soft": 0, "preplaced": 0, "fixed-shape": 0}
    multi: List[Tuple[int, int, List[BLBlockInfo]]] = []

    print(f"Scanning {len(test_ids)} case(s) ...\n")
    header = (
        f"{'case':>6}  {'blocks':>6}  {'#BL':>4}  "
        f"BL blocks: id(type)  [soft / preplaced / fixed-shape]"
    )
    print(header)
    print("-" * 72)

    for tid in test_ids:
        block_count, bl_blocks = scan_case(dataset, tid)
        n_bl = len(bl_blocks)
        for b in bl_blocks:
            bl_kind_totals[b.kind] = bl_kind_totals.get(b.kind, 0) + 1

        if n_bl == 0:
            case_counts[0] += 1
        elif n_bl == 1:
            case_counts[1] += 1
        else:
            case_counts["other"] += 1

        marker = " ***" if n_bl >= args.min_count else ""
        print(
            f"{tid:6d}  {block_count:6d}  {n_bl:4d}  "
            f"{_format_bl_list(bl_blocks)}{marker}"
        )

        if n_bl >= args.min_count:
            multi.append((tid, block_count, bl_blocks))

    total_bl = sum(bl_kind_totals.values())

    print("\n" + "=" * 72)
    print("SUMMARY — cases")
    print("=" * 72)
    print(f"  Cases scanned:              {len(test_ids)}")
    print(f"  With 0 BL blocks:           {case_counts[0]}")
    print(f"  With exactly 1 BL block:    {case_counts[1]}")
    print(f"  With 2+ BL blocks:          {case_counts['other']}")

    print("\n" + "=" * 72)
    print("SUMMARY — bottom-left boundary blocks by type")
    print("=" * 72)
    print(f"  Total BL blocks (all cases): {total_bl}")
    print(f"    soft:         {bl_kind_totals['soft']}")
    print(f"    preplaced:    {bl_kind_totals['preplaced']}")
    print(f"    fixed-shape:  {bl_kind_totals['fixed-shape']}")

    if total_bl > 0:
        print("\n  Fractions:")
        for kind in ("soft", "preplaced", "fixed-shape"):
            pct = 100.0 * bl_kind_totals[kind] / total_bl
            print(f"    {kind:12s} {pct:5.1f}%")

    print(f"\nCases with >={args.min_count} bottom-left boundary block(s):")
    if not multi:
        print("  (none)")
    else:
        for tid, bc, bl_blocks in multi:
            print(f"  test_id={tid}  blocks={bc}  {_format_bl_list(bl_blocks)}")


if __name__ == "__main__":
    main()
