#!/usr/bin/env python3
"""
Inspect a FloorSet-Lite training shard file (layouts_*.th).

These `.th` files are PyTorch-serialized objects. They are not text files.
Use this script to print the structure (types) and tensor shapes.

Example:
  python inspect_training_shard.py --file ../floorset_lite/worker_0/layouts_0.th
  python inspect_training_shard.py --file ../floorset_lite/worker_0/layouts_0.th --layout-idx 0
"""

import argparse
from pathlib import Path
from typing import Any


def _describe(obj: Any, prefix: str = "", max_items: int = 10):
    """Recursively describe nested lists/tuples/dicts and tensors."""
    try:
        import torch
    except ImportError as e:
        raise SystemExit("This script requires torch. Activate your venv that has torch installed.") from e

    if isinstance(obj, torch.Tensor):
        print(f"{prefix}Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})")
        return

    if isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}(len={len(obj)})")
        for i, item in enumerate(obj[:max_items]):
            _describe(item, prefix=f"{prefix}  [{i}] ", max_items=max_items)
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj) - max_items} more items)")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{prefix}dict(len={len(keys)}) keys={keys[:max_items]}{'...' if len(keys) > max_items else ''}")
        for k in keys[:max_items]:
            _describe(obj[k], prefix=f"{prefix}  [{k!r}] ", max_items=max_items)
        return

    print(f"{prefix}{type(obj).__name__}: {repr(obj)[:120]}")


def main():
    ap = argparse.ArgumentParser(description="Inspect FloorSet-Lite training shard layouts_*.th")
    ap.add_argument("--file", required=True, help="Path to layouts_*.th")
    ap.add_argument("--layout-idx", type=int, default=None, help="If set, inspect a specific layout within the shard")
    ap.add_argument("--max-items", type=int, default=10, help="Max list/dict items to print per level")
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    import torch

    obj = torch.load(p, map_location="cpu")
    print(f"Loaded: {p}")
    _describe(obj, max_items=args.max_items)

    if args.layout_idx is not None:
        li = args.layout_idx
        print("\n" + "=" * 70)
        print(f"Inspecting layout_idx={li}")
        print("=" * 70)
        # Based on lite_dataset.py indexing:
        # obj[0][li] -> per-block matrix: [:,0]=area_target, [:,1:]=placement_constraints
        # obj[1][li] -> b2b_connectivity
        # obj[2][li] -> p2b_connectivity
        # obj[3][li] -> pins_pos
        # obj[4][li] -> tree_sol
        # obj[5][li] -> fp_sol
        # obj[6][li] -> metrics_sol
        names = [
            "block_features (area_target + constraints)",
            "b2b_connectivity",
            "p2b_connectivity",
            "pins_pos",
            "tree_sol",
            "fp_sol",
            "metrics_sol",
        ]
        for k, name in enumerate(names):
            try:
                item = obj[k][li]
            except Exception as e:
                print(f"[{k}] {name}: <unable to index> ({e})")
                continue
            print(f"[{k}] {name}:")
            _describe(item, prefix="  ", max_items=args.max_items)


if __name__ == "__main__":
    main()

