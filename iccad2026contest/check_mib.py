#!/usr/bin/env python3
"""
Validate whether MIB groups contain only soft blocks.

FloorSet constraints[:, 2] = MIB group id:
  - 0 (or < 1): no MIB constraint (dataset convention)
  - 1, 2, ...: MIB group id (blocks must share identical w×h)

Block types (constraints[:, 0], constraints[:, 1]):
  - soft:         fixed=0, preplaced=0
  - fixed-shape:  fixed=1, preplaced=0
  - preplaced:    preplaced=1

Usage:
  python check_mib.py
  python check_mib.py --data-path ../
  python check_mib.py --test-id 40
  python check_mib.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest


@dataclass
class BlockTypeInfo:
    block_id: int
    kind: str  # "soft", "fixed-shape", "preplaced"
    mib_id: int
    target_wh: Tuple[float, float] = (-1.0, -1.0)


@dataclass
class MIBGroupReport:
    mib_id: int
    blocks: List[BlockTypeInfo] = field(default_factory=list)

    def kind_counts(self) -> Dict[str, int]:
        counts = {"soft": 0, "fixed-shape": 0, "preplaced": 0}
        for b in self.blocks:
            counts[b.kind] = counts.get(b.kind, 0) + 1
        return counts

    @property
    def is_clean(self) -> bool:
        c = self.kind_counts()
        return c["fixed-shape"] == 0 and c["preplaced"] == 0

    @property
    def non_soft(self) -> List[BlockTypeInfo]:
        return [b for b in self.blocks if b.kind != "soft"]


def classify_block(
    constraints: torch.Tensor,
    block_id: int,
    target_positions: Optional[torch.Tensor] = None,
) -> str:
    """
    Classify block type from constraints (authoritative).

    target_positions is only used to attach GT (w,h) for display, not typing.
    """
    del target_positions  # typing follows constraints only
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc > 1 and float(constraints[block_id, 1].item()) != 0:
        return "preplaced"
    if nc > 0 and float(constraints[block_id, 0].item()) != 0:
        return "fixed-shape"
    return "soft"


def mib_group_ids(constraints: torch.Tensor, block_count: int) -> List[int]:
    """
    Distinct MIB group ids with at least one member.

    Uses dataset convention: id >= 1 (0 means no MIB).
    """
    if constraints is None or block_count <= 0:
        return []
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 2:
        return []
    ids = set()
    for i in range(block_count):
        gid = int(float(constraints[i, 2].item()))
        if gid >= 1:
            ids.add(gid)
    return sorted(ids)


def analyse_mib_groups(
    constraints: torch.Tensor,
    block_count: int,
    target_positions: Optional[torch.Tensor] = None,
) -> List[MIBGroupReport]:
    """Build per-MIB-group reports with block types."""
    reports: List[MIBGroupReport] = []
    for gid in mib_group_ids(constraints, block_count):
        rep = MIBGroupReport(mib_id=gid)
        for i in range(block_count):
            if int(float(constraints[i, 2].item())) != gid:
                continue
            tw, th = (-1.0, -1.0)
            if target_positions is not None and i < len(target_positions):
                tw = float(target_positions[i, 2].item())
                th = float(target_positions[i, 3].item())
            rep.blocks.append(
                BlockTypeInfo(
                    block_id=i,
                    kind=classify_block(constraints, i, target_positions),
                    mib_id=gid,
                    target_wh=(tw, th),
                )
            )
        reports.append(rep)
    return reports


def build_target_positions_from_gt(
    polygons, block_count: int
) -> torch.Tensor:
    """
    Build optimizer-style target_positions from ground-truth polygons.

    All -1 by default; GT (x,y,w,h) filled from polygons for cross-check.
    """
    tp = torch.full((block_count, 4), -1.0)
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) == 0:
            continue
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        tp[i, 0] = float(x_min)
        tp[i, 1] = float(y_min)
        tp[i, 2] = float(x_max - x_min)
        tp[i, 3] = float(y_max - y_min)
    return tp


def scan_case(
    dataset: FloorplanDatasetLiteTest,
    test_id: int,
    use_gt_target: bool,
) -> Tuple[int, List[MIBGroupReport]]:
    sample = dataset[test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, _b2b, _p2b, _pins, constraints = inputs
    block_count = int((area_target != -1).sum().item())

    target_positions: Optional[torch.Tensor] = None
    if use_gt_target:
        polygons, _metrics = labels
        target_positions = build_target_positions_from_gt(polygons, block_count)
        # Mirror evaluator: mark fixed/preplaced dims on target_positions
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(block_count):
            is_fixed = nc > 0 and float(constraints[i, 0].item()) != 0
            is_preplaced = nc > 1 and float(constraints[i, 1].item()) != 0
            if is_preplaced:
                pass  # already has x,y,w,h from GT
            elif is_fixed:
                target_positions[i, 0] = -1
                target_positions[i, 1] = -1

    reports = analyse_mib_groups(constraints, block_count, target_positions)
    return block_count, reports


def _format_block(b: BlockTypeInfo) -> str:
    w, h = b.target_wh
    wh = f" w,h=({w:g},{h:g})" if w != -1 or h != -1 else ""
    return f"{b.block_id}:{b.kind}{wh}"


def _format_kind_counts(counts: Dict[str, int]) -> str:
    return (
        f"soft={counts['soft']}, fixed={counts['fixed-shape']}, "
        f"preplaced={counts['preplaced']}"
    )


def print_case_report(
    test_id: int,
    block_count: int,
    reports: List[MIBGroupReport],
    verbose: bool,
) -> Dict[str, int]:
    """Print one case; return {clean, mixed, total_groups}."""
    stats = {"total": len(reports), "clean": 0, "mixed": 0}
    if not reports:
        if verbose:
            print(f"  test_id={test_id}  blocks={block_count}  (no MIB groups)")
        return stats

    if verbose:
        print(f"\n--- test_id={test_id}  blocks={block_count}  MIB groups={len(reports)} ---")

    for rep in reports:
        counts = rep.kind_counts()
        status = "CLEAN" if rep.is_clean else "MIXED"
        if rep.is_clean:
            stats["clean"] += 1
        else:
            stats["mixed"] += 1

        block_str = ", ".join(_format_block(b) for b in rep.blocks)
        count_str = _format_kind_counts(counts)
        line = (
            f"  MIB group {rep.mib_id}: [{status}]  "
            f"n={len(rep.blocks)} ({count_str})  blocks=[{block_str}]"
        )
        if verbose:
            print(line)
        elif not rep.is_clean:
            offenders = ", ".join(_format_block(b) for b in rep.non_soft)
            print(
                f"  test_id={test_id}  MIB {rep.mib_id}  MIXED  "
                f"({count_str})  non-soft: [{offenders}]"
            )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether MIB groups contain only soft blocks."
    )
    parser.add_argument("--data-path", type=str, default="../")
    parser.add_argument("--test-id", type=int, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every MIB group for every case (default: only mixed groups)",
    )
    parser.add_argument(
        "--no-gt-target",
        action="store_true",
        help="Do not build target_positions from ground truth for cross-check",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    print(f"Loading validation dataset from {data_path} ...")
    dataset = FloorplanDatasetLiteTest(str(data_path))
    test_ids = [args.test_id] if args.test_id is not None else list(range(len(dataset)))

    total_groups = 0
    total_clean = 0
    total_mixed = 0
    cases_with_mib = 0
    mixed_cases: List[int] = []
    global_kind = {"soft": 0, "fixed-shape": 0, "preplaced": 0}

    print(f"Scanning {len(test_ids)} case(s) ...\n")

    for tid in test_ids:
        block_count, reports = scan_case(
            dataset, tid, use_gt_target=not args.no_gt_target
        )
        if reports:
            cases_with_mib += 1
        for rep in reports:
            for kind, n in rep.kind_counts().items():
                global_kind[kind] = global_kind.get(kind, 0) + n
        st = print_case_report(tid, block_count, reports, args.verbose)
        total_groups += st["total"]
        total_clean += st["clean"]
        total_mixed += st["mixed"]
        if st["mixed"] > 0:
            mixed_cases.append(tid)

    print("\n" + "=" * 72)
    print("OVERALL STATISTICS")
    print("=" * 72)
    print(f"  Cases scanned:                 {len(test_ids)}")
    print(f"  Cases with ≥1 MIB group:       {cases_with_mib}")
    print(f"  Total MIB groups:              {total_groups}")
    print(f"  Clean groups (all soft):       {total_clean}")
    print(f"  Mixed groups (has non-soft):   {total_mixed}")
    print("\n  MIB member blocks by type (across all groups):")
    print(f"    soft:         {global_kind['soft']}")
    print(f"    fixed-shape:  {global_kind['fixed-shape']}")
    print(f"    preplaced:    {global_kind['preplaced']}")

    if total_mixed == 0:
        print("\n  ✓ All MIB groups consist only of soft blocks.")
    else:
        print(f"\n  ✗ Mixed MIB groups found in {len(mixed_cases)} case(s): {mixed_cases}")
        print("  Re-run with --verbose for full per-group listings.")


if __name__ == "__main__":
    main()
