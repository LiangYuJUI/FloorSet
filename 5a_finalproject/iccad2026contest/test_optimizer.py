#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: B*-tree Simulated Annealing
  - GUARANTEES: Overlap-free, area constraints satisfied
  - NOT HANDLED: Fixed, preplaced, MIB, cluster, boundary constraints

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, MIB, cluster, boundary)
  - target_positions: [n, 4] target (x, y, w, h) per block.
      All -1 by default (free). For fixed-shape blocks, w and h are set.
      For preplaced blocks, all four (x, y, w, h) are set.

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
  - Floating-point coordinates allowed
  - Any aspect ratio (w/h) allowed

HARD CONSTRAINTS (violation = Cost 10.0):
  - NO OVERLAPS between blocks
  - AREA: w*h within 1% of area_targets[i]

RELAXED CONSTRAINTS:
  - Aspect ratio: Any w/h ratio is valid
  - Fixed outline: Removed (implicitly optimized via p2b HPWL and bbox area)
  - Coordinates: Floating-point allowed
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

# Set True to print pack() contour debug (once per process unless reset).
COUNT_DEBUG = False


def set_count_debug(enabled: bool) -> None:
    """Enable/disable contour debug prints from outside this module."""
    global COUNT_DEBUG
    COUNT_DEBUG = enabled

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

AREA_TOLERANCE = 0.01
_BOUNDARY_LEFT = 1
_BOUNDARY_RIGHT = 2
_BOUNDARY_TOP = 4
_BOUNDARY_BOTTOM = 8
_BOUNDARY_TOP_LEFT = 1 + 4  # 5
_BOUNDARY_BOTTOM_LEFT = 1 + 8  # 9
_BOUNDARY_BOTTOM_RIGHT = 2 + 8  # 10
_BOUNDARY_TOP_RIGHT = 2 + 4  # 6

# Simulated annealing schedule (edit here to control runtime vs quality).
SA_INITIAL_TEMP_PER_BLOCK = 100.0  # initial_temp = this × block_count
SA_FINAL_TEMP = 0.1
SA_COOLING_RATE = 0.6
SA_MOVES_PER_TEMP = 1000


def _is_preplaced(constraints: torch.Tensor, i: int) -> bool:
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    return nc > 1 and float(constraints[i, 1].item()) != 0


def _is_fixed_shape(constraints: torch.Tensor, i: int) -> bool:
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    return nc > 0 and float(constraints[i, 0].item()) != 0


def _is_soft(constraints: torch.Tensor, i: int) -> bool:
    return not _is_preplaced(constraints, i) and not _is_fixed_shape(constraints, i)


def _mib_group_id(constraints: torch.Tensor, i: int) -> Optional[int]:
    """Return MIB group id, or None if block has no MIB constraint."""
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 2:
        return None
    gid = int(float(constraints[i, 2].item()))
    if gid < 0:
        return None
    if gid == 0:
        return None
    return gid


def _find_bottom_left_block(constraints: torch.Tensor, block_count: int) -> int:
    if constraints is None or block_count == 0:
        return -1
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 4:
        return -1
    for i in range(block_count):
        if int(float(constraints[i, 4].item())) == _BOUNDARY_BOTTOM_LEFT:
            return i
    return -1


def _target_wh(
    i: int,
    area_targets: torch.Tensor,
    target_positions: Optional[torch.Tensor],
) -> Tuple[float, float]:
    if (target_positions is not None
            and float(target_positions[i, 2].item()) != -1
            and float(target_positions[i, 3].item()) != -1):
        return (
            float(target_positions[i, 2].item()),
            float(target_positions[i, 3].item()),
        )
    area = float(area_targets[i].item()) if area_targets[i].item() > 0 else 1.0
    side = math.sqrt(area)
    return side, side


def _analyse_mib_groups(
    constraints: torch.Tensor,
    block_count: int,
    target_positions: Optional[torch.Tensor],
    area_targets: torch.Tensor,
) -> Tuple[Dict[int, Tuple[float, float]], Set[int], Dict[int, int]]:
    """
    Returns (forced_dim_by_group, free_group_ids, block_to_mib_group).

    forced_dim: all soft members must use this (w, h).
    free_group: all-soft MIB group; shape chosen later (often by BL root).
    """
    forced: Dict[int, Tuple[float, float]] = {}
    free: Set[int] = set()
    block_to_mib: Dict[int, int] = {}

    groups: Dict[int, List[int]] = {}
    for i in range(block_count):
        gid = _mib_group_id(constraints, i)
        if gid is None:
            continue
        block_to_mib[i] = gid
        groups.setdefault(gid, []).append(i)

    for gid, members in groups.items():
        non_soft = [i for i in members if not _is_soft(constraints, i)]
        if non_soft:
            w, h = _target_wh(non_soft[0], area_targets, target_positions)
            forced[gid] = (w, h)
        else:
            free.add(gid)
    return forced, free, block_to_mib


def _w_cap_at_origin(h: float, preplaced_info: List[Tuple[float, float, float, float]], eps: float = 1e-6) -> float:
    """Max width for (0,0,w,h) without x-overlap with preplaced (y must overlap)."""
    cap = float("inf")
    for px, py, pw, ph in preplaced_info:
        if py + ph <= eps or py >= h - eps:
            continue
        if px > eps:
            cap = min(cap, px)
    return cap


def _h_cap_at_origin(w: float, preplaced_info: List[Tuple[float, float, float, float]], eps: float = 1e-6) -> float:
    cap = float("inf")
    for px, py, pw, ph in preplaced_info:
        if px + pw <= eps or px >= w - eps:
            continue
        if py > eps:
            cap = min(cap, py)
    return cap


def _area_in_tolerance(area: float, w: float, h: float) -> bool:
    actual = w * h
    return (1.0 - AREA_TOLERANCE) * area <= actual <= (1.0 + AREA_TOLERANCE) * area


def _clamp_dims_to_area(area: float, w: float, h: float) -> Tuple[float, float]:
    lo = (1.0 - AREA_TOLERANCE) * area
    hi = (1.0 + AREA_TOLERANCE) * area
    a = w * h
    if lo <= a <= hi:
        return w, h
    if a < lo:
        scale = math.sqrt(lo / max(a, 1e-12))
    else:
        scale = math.sqrt(hi / a)
    return w * scale, h * scale


def _overlaps_preplaced_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    preplaced_info: List[Tuple[float, float, float, float]],
    eps: float = 1e-6,
) -> bool:
    for px, py, pw, ph in preplaced_info:
        if (
            x + w > px + eps
            and x < px + pw - eps
            and y + h > py + eps
            and y < py + ph - eps
        ):
            return True
    return False


def _adjust_soft_root_at_origin(
    area: float,
    preplaced_info: List[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    """
    Case C: square root at (0,0), adjust w/h to avoid preplaced overlap within ±1% area.
    """
    w = h = math.sqrt(area)
    if not _overlaps_preplaced_rect(0.0, 0.0, w, h, preplaced_info):
        return w, h

    def aspect_score(wv: float, hv: float) -> float:
        return abs(wv / max(hv, 1e-12) - 1.0)

    candidates: List[Tuple[float, float]] = []

    w_cap = _w_cap_at_origin(h, preplaced_info)
    if w_cap < float("inf"):
        w1 = max(w_cap * (1.0 - 1e-6), 1e-6)
        h1 = area / w1
        w1, h1 = _clamp_dims_to_area(area, w1, h1)
        if _area_in_tolerance(area, w1, h1) and not _overlaps_preplaced_rect(0, 0, w1, h1, preplaced_info):
            candidates.append((w1, h1))

    h_cap = _h_cap_at_origin(w, preplaced_info)
    if h_cap < float("inf"):
        h2 = max(h_cap * (1.0 - 1e-6), 1e-6)
        w2 = area / h2
        w2, h2 = _clamp_dims_to_area(area, w2, h2)
        if _area_in_tolerance(area, w2, h2) and not _overlaps_preplaced_rect(0, 0, w2, h2, preplaced_info):
            candidates.append((w2, h2))

    for _ in range(10):
        if not candidates:
            w_try = max(w * 0.95, 1e-6)
            h_try = area / w_try
            w_try, h_try = _clamp_dims_to_area(area, w_try, h_try)
            if not _overlaps_preplaced_rect(0, 0, w_try, h_try, preplaced_info):
                candidates.append((w_try, h_try))
                break
            h_try = max(h * 0.95, 1e-6)
            w_try = area / h_try
            w_try, h_try = _clamp_dims_to_area(area, w_try, h_try)
            if not _overlaps_preplaced_rect(0, 0, w_try, h_try, preplaced_info):
                candidates.append((w_try, h_try))
                break
            w *= 0.9
            h *= 0.9
        else:
            break

    if candidates:
        return min(candidates, key=lambda wh: aspect_score(wh[0], wh[1]))
    w, h = _clamp_dims_to_area(area, w, h)
    return w, h


def _initialize_block_dimensions(
    block_count: int,
    constraints: torch.Tensor,
    area_targets: torch.Tensor,
    target_positions: Optional[torch.Tensor],
    preplaced_set: Set[int],
    preplaced_info: List[Tuple[float, float, float, float]],
    movable_indices: List[int],
) -> Tuple[List[float], List[float], int, Set[int]]:
    """
    Build movable width/height lists, B*-tree root (tree-local), and locked tree nodes.

    locked = fixed-shape + MIB members with forced or assigned shared dimensions.
    """
    forced_mib, free_mib, block_to_mib = _analyse_mib_groups(
        constraints, block_count, target_positions, area_targets
    )
    free_group_shape: Dict[int, Tuple[float, float]] = {}
    block_dims: Dict[int, Tuple[float, float]] = {}

    bl_global = _find_bottom_left_block(constraints, block_count)

    for i in range(block_count):
        if i in preplaced_set:
            block_dims[i] = _target_wh(i, area_targets, target_positions)
            continue
        if _is_fixed_shape(constraints, i):
            block_dims[i] = _target_wh(i, area_targets, target_positions)
            continue
        gid = block_to_mib.get(i)
        if gid is not None and gid in forced_mib:
            block_dims[i] = forced_mib[gid]
            continue
        if i == bl_global and _is_soft(constraints, i):
            area = float(area_targets[i].item()) if area_targets[i].item() > 0 else 1.0
            w, h = _adjust_soft_root_at_origin(area, preplaced_info)
            block_dims[i] = (w, h)
            if gid is not None and gid in free_mib:
                free_group_shape[gid] = (w, h)
            continue
        area = float(area_targets[i].item()) if area_targets[i].item() > 0 else 1.0
        block_dims[i] = (math.sqrt(area), math.sqrt(area))

    for i in range(block_count):
        if i in preplaced_set or not _is_soft(constraints, i):
            continue
        gid = block_to_mib.get(i)
        if gid is None:
            continue
        if gid in forced_mib:
            block_dims[i] = forced_mib[gid]
        elif gid in free_group_shape:
            block_dims[i] = free_group_shape[gid]

    mov_widths: List[float] = []
    mov_heights: List[float] = []
    mib_locked_global: Set[int] = set()
    for i in movable_indices:
        w, h = block_dims[i]
        mov_widths.append(w)
        mov_heights.append(h)
        gid = block_to_mib.get(i)
        if gid is not None:
            mib_locked_global.add(i)

    tree_root = -1
    if bl_global >= 0 and bl_global in movable_indices:
        tree_root = movable_indices.index(bl_global)

    locked_tree: Set[int] = set()
    for tree_i, orig_i in enumerate(movable_indices):
        if _is_fixed_shape(constraints, orig_i) or orig_i in mib_locked_global:
            locked_tree.add(tree_i)

    return mov_widths, mov_heights, tree_root, locked_tree


def _preplaced_blocks(
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
    block_count: int,
) -> Tuple[List[int], List[Tuple[float, float, float, float]]]:
    """Return (indices, (x, y, w, h)) for preplaced blocks."""
    indices: List[int] = []
    info: List[Tuple[float, float, float, float]] = []
    if constraints is None or block_count == 0:
        return indices, info
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 1 or target_positions is None:
        return indices, info
    for i in range(block_count):
        if float(constraints[i, 1].item()) == 0:
            continue
        x = float(target_positions[i, 0].item())
        y = float(target_positions[i, 1].item())
        w = float(target_positions[i, 2].item())
        h = float(target_positions[i, 3].item())
        indices.append(i)
        info.append((x, y, w, h))
    return indices, info


def _fixed_shape_tree_nodes(
    constraints: torch.Tensor,
    movable_indices: List[int],
    mib_locked_tree_nodes: Optional[Set[int]] = None,
) -> Set[int]:
    """Tree-local indices that must not rotate (fixed-shape + MIB-locked)."""
    fixed: Set[int] = set(mib_locked_tree_nodes or ())
    if constraints is None or not movable_indices:
        return fixed
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 0:
        return fixed
    for tree_i, orig_i in enumerate(movable_indices):
        if float(constraints[orig_i, 0].item()) != 0:
            fixed.add(tree_i)
    return fixed


def _tree_boundary_codes(
    constraints: Optional[torch.Tensor],
    movable_indices: List[int],
) -> Optional[List[int]]:
    """Tree-local boundary codes (constraints[:, 4]) for movable blocks."""
    if constraints is None or not movable_indices:
        return None
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 4:
        return None
    return [int(float(constraints[gi, 4].item())) for gi in movable_indices]


class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(
        self,
        n_blocks: int,
        widths: List[float],
        heights: List[float],
        preplaced_info: Optional[List[Tuple[float, float, float, float]]] = None,
        preplaced_indices: Optional[List[int]] = None,
        movable_indices: Optional[List[int]] = None,
        total_blocks: int = 0,
        fixed_tree_nodes: Optional[Set[int]] = None,
        preferred_root: int = -1,
        permanent_root: int = -1,
        boundary_codes: Optional[List[int]] = None,
    ):
        """
        B*-tree over movable blocks only.

        Tree node i maps to original block movable_indices[i].
        Preplaced blocks are fixed obstacles during pack().

        permanent_root: tree-local index that must always remain root and
        cannot be moved or resized by SA (e.g. bottom-left boundary block).
        """
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self.total_blocks = total_blocks if total_blocks > 0 else n_blocks
        self.movable_indices = (
            list(movable_indices) if movable_indices is not None else list(range(n_blocks))
        )
        self.preplaced_indices = list(preplaced_indices or [])
        self.preplaced_info = list(preplaced_info or [])
        self.fixed_tree_nodes: Set[int] = set(fixed_tree_nodes or ())
        self.preferred_root = preferred_root
        self.permanent_root = permanent_root
        self.boundary_codes = list(boundary_codes) if boundary_codes is not None else None
        self.bottom_chain_nodes: Set[int] = set()
        self.left_chain_nodes: Set[int] = set()
        self._build_tree()
    
    def _build_tree(self) -> None:
        if self.n == 0:
            return
        if (
            self.boundary_codes is not None
            and len(self.boundary_codes) == self.n
            and 0 <= self.permanent_root < self.n
        ):
            self._build_constrained_tree()
        else:
            self._build_random_tree()

    def _rebuild_chain_membership(self) -> None:
        """Walk left/right chains from root; populate membership sets."""
        self.bottom_chain_nodes = set()
        self.left_chain_nodes = set()
        if self.n == 0 or self.root < 0:
            return
        self.bottom_chain_nodes.add(self.root)
        node = self.left[self.root]
        while node != -1:
            self.bottom_chain_nodes.add(node)
            node = self.left[node]
        self.left_chain_nodes.add(self.root)
        node = self.right[self.root]
        while node != -1:
            self.left_chain_nodes.add(node)
            node = self.right[node]

    def _attach_child(self, parent: int, block: int, as_left: bool) -> None:
        if as_left:
            self.left[parent] = block
        else:
            self.right[parent] = block
        self.parent[block] = parent

    def _build_constrained_tree(self) -> None:
        """
        Build bottom (left-child) and left (right-child) boundary chains from root.

        Bottom chain: root.left -> ... -> BR (10) -> right-only (2).
        Left chain: root.right -> ... -> TL (5) -> top-only (4).
        Top-right (6) and other unconstrained blocks are inserted as free nodes.
        """
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        self.root = self.permanent_root
        codes = self.boundary_codes or [0] * self.n

        bottom_middle: List[int] = []
        left_middle: List[int] = []
        right_only: List[int] = []
        top_only: List[int] = []
        br_node = -1
        tl_node = -1

        for tree_i in range(self.n):
            if tree_i == self.root:
                continue
            code = codes[tree_i]
            if code == _BOUNDARY_BOTTOM_RIGHT:
                br_node = tree_i
            elif code == _BOUNDARY_TOP_LEFT:
                tl_node = tree_i
            elif code == _BOUNDARY_TOP_RIGHT:
                continue  # free block — boundary penalty guides placement
            elif code == _BOUNDARY_RIGHT:
                right_only.append(tree_i)
            elif code == _BOUNDARY_TOP:
                top_only.append(tree_i)
            elif code & _BOUNDARY_BOTTOM:
                bottom_middle.append(tree_i)
            elif code & _BOUNDARY_LEFT:
                left_middle.append(tree_i)

        random.shuffle(bottom_middle)
        random.shuffle(left_middle)
        random.shuffle(right_only)
        random.shuffle(top_only)

        bottom_chain_order = list(bottom_middle)
        if br_node >= 0:
            bottom_chain_order.append(br_node)
        bottom_chain_order.extend(right_only)

        if bottom_chain_order:
            self._attach_child(self.root, bottom_chain_order[0], as_left=True)
            for a, b in zip(bottom_chain_order, bottom_chain_order[1:]):
                self._attach_child(a, b, as_left=True)

        left_chain_order = list(left_middle)
        if tl_node >= 0:
            left_chain_order.append(tl_node)
        left_chain_order.extend(top_only)

        if left_chain_order:
            self._attach_child(self.root, left_chain_order[0], as_left=False)
            for a, b in zip(left_chain_order, left_chain_order[1:]):
                self._attach_child(a, b, as_left=False)

        self._rebuild_chain_membership()

        chain_nodes = self.bottom_chain_nodes | self.left_chain_nodes
        free_blocks = [i for i in range(self.n) if i not in chain_nodes]
        random.shuffle(free_blocks)

        for block in free_blocks:
            self._insert_free_block(block)

    def _insert_free_block(self, block: int) -> None:
        """Attach a free block without breaking bottom/left chains."""
        slots: List[Tuple[int, bool, bool]] = []
        for parent in range(self.n):
            if parent == block:
                continue
            if self.left[parent] == -1 and parent not in self.bottom_chain_nodes:
                preferred = parent in self.left_chain_nodes
                slots.append((parent, True, preferred))
            if self.right[parent] == -1 and parent not in self.left_chain_nodes:
                preferred = parent in self.bottom_chain_nodes
                slots.append((parent, False, preferred))

        if slots:
            preferred_slots = [s for s in slots if s[2]]
            parent, as_left, _ = random.choice(
                preferred_slots if preferred_slots else slots
            )
            self._attach_child(parent, block, as_left)
            return

        parent = random.randint(0, self.n - 1)
        while parent == block:
            parent = random.randint(0, self.n - 1)
        self._insert_at_leaf_constrained(block, parent)

    def _insert_at_leaf_constrained(self, block: int, start: int) -> None:
        """Leaf insert respecting bottom/left chain sides."""
        current = start
        while True:
            go_left = random.random() < 0.5
            if go_left:
                if self.left[current] == -1 and current not in self.bottom_chain_nodes:
                    self._attach_child(current, block, as_left=True)
                    return
                if self.left[current] != -1:
                    current = self.left[current]
                    continue
            if self.right[current] == -1 and current not in self.left_chain_nodes:
                self._attach_child(current, block, as_left=False)
                return
            if self.right[current] != -1:
                current = self.right[current]
            elif self.left[current] != -1:
                current = self.left[current]
            else:
                for p in range(self.n):
                    if p == block:
                        continue
                    if self.left[p] == -1 and p not in self.bottom_chain_nodes:
                        self._attach_child(p, block, as_left=True)
                        return
                return

    def _chain_swap_allowed(self, b1: int, b2: int) -> bool:
        if b1 == self.permanent_root or b2 == self.permanent_root:
            return False
        b1_chain = b1 in self.bottom_chain_nodes or b1 in self.left_chain_nodes
        b2_chain = b2 in self.bottom_chain_nodes or b2 in self.left_chain_nodes
        if not b1_chain or not b2_chain:
            return False
        if b1 in self.bottom_chain_nodes and b2 in self.bottom_chain_nodes:
            return True
        if b1 in self.left_chain_nodes and b2 in self.left_chain_nodes:
            return True
        return False

    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n

        order = list(range(self.n))
        random.shuffle(order)
        forced_root = -1
        if 0 <= self.permanent_root < self.n:
            forced_root = self.permanent_root
        elif 0 <= self.preferred_root < self.n:
            forced_root = self.preferred_root
        if forced_root >= 0:
            order.remove(forced_root)
            order.insert(0, forced_root)
        self.root = order[0]

        # Attach remaining blocks as children; forced_root stays root (parent -1).
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)

        if 0 <= self.permanent_root < self.n:
            self.root = self.permanent_root
        self._rebuild_chain_membership()
    
    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]
    
    @staticmethod
    def _merge_contour(contour: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        merged: List[Tuple[float, float]] = []
        for x_end, y_top in contour:
            if merged and merged[-1][1] == y_top:
                merged[-1] = (x_end, y_top)
            else:
                merged.append((x_end, y_top))
        return merged

    @staticmethod
    def _raise_contour(
        contour: List[Tuple[float, float]], x_start: float, x_end: float, y_top: float
    ) -> List[Tuple[float, float]]:
        """
        Raise contour height to at least y_top on [x_start, x_end).

        Contour is a sorted list of (x_end, height): segment i spans
        (contour[i-1].x_end, contour[i].x_end] with the given height.
        """
        new_contour: List[Tuple[float, float]] = []
        for i, (cx_end, cy_top) in enumerate(contour):
            cx_start = contour[i - 1][0] if i > 0 else 0.0
            if cx_end <= x_start:
                new_contour.append((cx_end, cy_top))
            elif cx_start >= x_end:
                new_contour.append((cx_end, cy_top))
            else:
                if cx_start < x_start:
                    new_contour.append((x_start, cy_top))
                if cx_end > x_end:
                    new_contour.append((cx_end, cy_top))

        # Extend the active skyline height from the last breakpoint to x_start.
        # Without this, e.g. [(0, 0)] + obstacle [95, 104) becomes [(0, 0), (104, 134)]
        # and get_contour_y treats [0, 104) as height 134 instead of [0, 95) at 0.
        if new_contour:
            last_x, last_y = new_contour[-1]
            if last_x < x_start:
                new_contour.append((x_start, last_y))
        else:
            new_contour.append((x_start, 0.0))

        insert_pos = 0
        for i, (cx_end, _) in enumerate(new_contour):
            if cx_end <= x_start:
                insert_pos = i + 1
        new_contour.insert(insert_pos, (x_end, y_top))
        new_contour.sort(key=lambda seg: seg[0])
        merged = BStarTree._merge_contour(new_contour)
        return merged if merged else [(x_end, 0.0)]

    @staticmethod
    def _overlaps_preplaced(
        x: float,
        y: float,
        w: float,
        h: float,
        preplaced_info: List[Tuple[float, float, float, float]],
        eps: float = 1e-6,
    ) -> bool:
        return _overlaps_preplaced_rect(x, y, w, h, preplaced_info, eps)

    @staticmethod
    def _placement_y(
        x: float,
        w: float,
        h: float,
        skyline_y: float,
        preplaced_info: List[Tuple[float, float, float, float]],
        eps: float = 1e-6,
    ) -> float:
        """
        Placement height from movable skyline, then bump y only on 2D overlap.

        Preplaced blocks are not merged into the 1D skyline (that would mark
        the whole column up to y_top). Gaps below a preplaced block stay usable.
        """
        y = max(skyline_y, 0.0)
        changed = True
        while changed:
            changed = False
            for px, py, pw, ph in preplaced_info:
                if x + w <= px + eps or x >= px + pw - eps:
                    continue
                if y + h <= py + eps or y >= py + ph - eps:
                    continue
                y = py + ph
                changed = True
        return y

    def _compact_horizontally(
        self,
        positions: List[Tuple[float, float, float, float]],
        max_iterations: int = 20,
        eps: float = 1e-6,
    ) -> None:
        """
        Shift movable blocks left as far as possible without changing y/w/h.

        Preplaced blocks are fixed obstacles. Iterates until no moves or
        max_iterations (left-to-right order may require multiple passes).
        """
        if self.n == 0:
            return

        movable_global = list(self.movable_indices)

        for _ in range(max_iterations):
            movable_global.sort(key=lambda idx: positions[idx][0])
            changed = False

            for i_idx in movable_global:
                x_i, y_i, w_i, h_i = positions[i_idx]
                candidate_x = 0.0

                for j_idx in range(self.total_blocks):
                    if j_idx == i_idx:
                        continue
                    x_j, y_j, w_j, h_j = positions[j_idx]
                    if y_i + eps < y_j + h_j and y_i + h_i > y_j + eps:
                        candidate_x = max(candidate_x, x_j + w_j)

                if candidate_x < x_i - eps:
                    positions[i_idx] = (candidate_x, y_i, w_i, h_i)
                    changed = True

            if not changed:
                break

    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Pack movable blocks via B*-tree DFS; preplaced blocks stay fixed.

        Skyline contour tracks movable blocks only. Preplaced rectangles are
        enforced with 2D overlap checks so space below/above them stays usable.
        """
        positions: List[Tuple[float, float, float, float]] = [
            (0.0, 0.0, 1.0, 1.0) for _ in range(self.total_blocks)
        ]
        for idx, rect in zip(self.preplaced_indices, self.preplaced_info):
            positions[idx] = rect

        if self.n == 0:
            return positions

        contour: List[Tuple[float, float]] = [(0.0, 0.0)]

        def get_contour_y(x_start: float, x_end: float) -> float:
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i - 1][0] if i > 0 else 0.0
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y

        def update_contour(x_start: float, x_end: float, y_top: float) -> None:
            nonlocal contour
            contour = BStarTree._raise_contour(contour, x_start, x_end, y_top)

        def dfs(node: int, parent_right_edge: float) -> None:
            if node == -1:
                return

            w, h = self.widths[node], self.heights[node]
            if node == self.root:
                x = 0.0
                if self.permanent_root != -1:
                    y = 0.0
                else:
                    skyline_y = get_contour_y(0.0, w)
                    y = BStarTree._placement_y(
                        x, w, h, skyline_y, self.preplaced_info
                    )
            else:
                x = parent_right_edge
                skyline_y = get_contour_y(x, x + w)
                y = BStarTree._placement_y(
                    x, w, h, skyline_y, self.preplaced_info
                )

            orig = self.movable_indices[node]
            positions[orig] = (x, y, w, h)
            update_contour(x, x + w, y + h)

            dfs(self.left[node], x + w)
            dfs(self.right[node], x)

        dfs(self.root, 0.0)
        self._compact_horizontally(positions)
        return positions
    
    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        new.total_blocks = self.total_blocks
        new.movable_indices = self.movable_indices.copy()
        new.preplaced_indices = self.preplaced_indices.copy()
        new.preplaced_info = list(self.preplaced_info)
        new.fixed_tree_nodes = set(self.fixed_tree_nodes)
        new.preferred_root = self.preferred_root
        new.permanent_root = self.permanent_root
        new.boundary_codes = list(self.boundary_codes) if self.boundary_codes else None
        new.bottom_chain_nodes = set(self.bottom_chain_nodes)
        new.left_chain_nodes = set(self.left_chain_nodes)
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        if block == self.permanent_root:
            return
        if block in self.fixed_tree_nodes:
            return
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        if not self._chain_swap_allowed(b1, b2):
            return
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if block == self.permanent_root:
            return
        if block in self.bottom_chain_nodes or block in self.left_chain_nodes:
            return
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h
    
    def _delete_node(self, node: int):
        if node == self.permanent_root:
            return
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        
        if replacement != -1:
            self.parent[replacement] = parent
        
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1
    
    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# Boundary soft-constraint bit mask (constraints[:, 4]) — see module-level _BOUNDARY_* constants.


def _boundary_penalty_squared_sum(
    positions: List[Tuple[float, float, float, float]],
    constraints: torch.Tensor,
    block_count: int,
) -> float:
    """
    Sum of squared edge distances for blocks with boundary constraints.

    Per required bit: distance is 0 on the correct bbox edge; penalty uses d².
    Corner codes sum distances for each active bit (e.g. 9 = left + bottom).
    """
    if constraints is None or block_count <= 0 or not positions:
        return 0.0
    ncols = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if ncols <= 4:
        return 0.0

    n = min(block_count, len(positions))
    x_min = min(positions[i][0] for i in range(n))
    y_min = min(positions[i][1] for i in range(n))
    x_max = max(positions[i][0] + positions[i][2] for i in range(n))
    y_max = max(positions[i][1] + positions[i][3] for i in range(n))

    penalty = 0.0
    for i in range(n):
        code = int(float(constraints[i, 4].item()))
        if code == 0:
            continue
        bx, by, bw, bh = positions[i]
        block_penalty = 0.0
        if code & _BOUNDARY_LEFT:
            d = bx - x_min
            block_penalty += d * d
        if code & _BOUNDARY_RIGHT:
            d = x_max - (bx + bw)
            block_penalty += d * d
        if code & _BOUNDARY_TOP:
            d = y_max - (by + bh)
            block_penalty += d * d
        if code & _BOUNDARY_BOTTOM:
            d = by - y_min
            block_penalty += d * d
        penalty += block_penalty
    return penalty


def _blocks_share_edge(
    x1: float,
    y1: float,
    w1: float,
    h1: float,
    x2: float,
    y2: float,
    w2: float,
    h2: float,
    eps: float = 1e-6,
) -> bool:
    """True if two axis-aligned rectangles share a full edge (not corner-only)."""
    vertical_touch = (
        abs(x1 + w1 - x2) < eps or abs(x2 + w2 - x1) < eps
    )
    if vertical_touch and y1 < y2 + h2 - eps and y1 + h1 > y2 + eps:
        return True
    horizontal_touch = (
        abs(y1 + h1 - y2) < eps or abs(y2 + h2 - y1) < eps
    )
    if horizontal_touch and x1 < x2 + w2 - eps and x1 + w1 > x2 + eps:
        return True
    return False


def _grouping_penalty(
    positions: List[Tuple[float, float, float, float]],
    constraints: Optional[torch.Tensor],
    block_count: int,
    eps: float = 1e-6,
) -> float:
    """
    Squared grouping violation: sum_g (c_g - 1)^2 over cluster groups.

    Blocks with constraints[:, 3] < 0 are ungrouped. Edge-adjacency uses
    shared full edges only (corner contact does not connect).
    """
    if constraints is None or block_count <= 0 or not positions:
        return 0.0
    ncols = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if ncols <= 3:
        return 0.0

    n = min(block_count, len(positions))
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        gid = int(float(constraints[i, 3].item()))
        if gid < 0:
            continue
        groups.setdefault(gid, []).append(i)

    penalty = 0.0
    for members in groups.values():
        k = len(members)
        if k <= 1:
            continue

        parent = list(range(k))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a in range(k):
            ia = members[a]
            x1, y1, w1, h1 = positions[ia]
            for b in range(a + 1, k):
                ib = members[b]
                x2, y2, w2, h2 = positions[ib]
                if _blocks_share_edge(x1, y1, w1, h1, x2, y2, w2, h2, eps):
                    union(a, b)

        components = len({find(i) for i in range(k)})
        violation = components - 1
        if violation > 0:
            penalty += violation * violation

    return penalty


# =============================================================================
# OPTIMIZER CLASS - Replace this with your algorithm
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    B*-tree Simulated Annealing baseline.
    
    REPLACE THIS CLASS WITH YOUR ALGORITHM.
    Keep the solve() signature the same.
    """

    # Weights for _cost(): hpwl_b2b, hpwl_p2b, bounding-box area.
    COST_WEIGHT_HPWL_B2B = 1.0
    COST_WEIGHT_HPWL_P2B = 1.0
    COST_WEIGHT_BBOX_AREA = 0.01
    # Multiplier for boundary penalty after per-instance calibration in solve().
    COST_WEIGHT_BOUNDARY = 1.0
    # Target ratio: boundary_penalty_weight * B_pen ≈ BOUNDARY_HPWL_RATIO * HPWL at init.
    BOUNDARY_HPWL_RATIO = 0.1
    COST_WEIGHT_GROUPING = 1.0
    # Target ratio: grouping_weight * G_pen ≈ GROUPING_HPWL_RATIO * HPWL at init.
    GROUPING_HPWL_RATIO = 0.05

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.final_temp = SA_FINAL_TEMP
        self.cooling_rate = SA_COOLING_RATE
        self.moves_per_temp = SA_MOVES_PER_TEMP
        self._boundary_weight: float = 1.0
        self._grouping_weight: float = 1.0
    
    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        """
        B*-tree SA optimization.
        
        REPLACE THIS METHOD with your algorithm.
        Must return List[(x, y, w, h)] with exactly block_count entries.
        """
        preplaced_indices, preplaced_info = _preplaced_blocks(
            constraints, target_positions, block_count
        )
        preplaced_set: Set[int] = set(preplaced_indices)

        movable_indices = [
            i for i in range(block_count) if i not in preplaced_set
        ]
        mov_widths, mov_heights, tree_root, locked_tree = _initialize_block_dimensions(
            block_count,
            constraints,
            area_targets,
            target_positions,
            preplaced_set,
            preplaced_info,
            movable_indices,
        )

        fixed_tree_nodes = _fixed_shape_tree_nodes(
            constraints, movable_indices, locked_tree
        )
        rotatable_nodes = [
            i for i in range(len(movable_indices))
            if i not in fixed_tree_nodes and i != tree_root
        ]

        tree_boundary_codes = _tree_boundary_codes(constraints, movable_indices)

        tree = BStarTree(
            len(movable_indices),
            mov_widths,
            mov_heights,
            preplaced_info=preplaced_info,
            preplaced_indices=preplaced_indices,
            movable_indices=movable_indices,
            total_blocks=block_count,
            fixed_tree_nodes=fixed_tree_nodes,
            preferred_root=tree_root,
            permanent_root=tree_root,
            boundary_codes=tree_boundary_codes,
        )
        current_positions = tree.pack()
        self._calibrate_boundary_weight(
            current_positions,
            constraints,
            block_count,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
        )
        current_cost = self._cost(
            current_positions,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            block_count,
        )

        best_tree = tree.copy()
        best_positions = current_positions
        best_cost = current_cost

        # Simulated Annealing (see SA_* constants at top of file)
        self.initial_temp = SA_INITIAL_TEMP_PER_BLOCK * block_count
        temp = self.initial_temp
        while temp > self.final_temp:
            for _ in range(self.moves_per_temp):
                old_tree = tree.copy()
                
                # Random move (only rotate and delete-insert to preserve area)
                if tree.n <= 0:
                    break
                can_rotate = len(rotatable_nodes) > 0
                can_reinsert = tree.n > 1
                if not can_rotate and not can_reinsert:
                    break
                if can_rotate and (not can_reinsert or random.randint(0, 1) == 0):
                    tree.move_rotate(random.choice(rotatable_nodes))
                else:
                    tree.move_delete_insert(random.randint(0, tree.n - 1))
                
                new_positions = tree.pack()
                new_cost = self._cost(
                    new_positions,
                    b2b_connectivity,
                    p2b_connectivity,
                    pins_pos,
                    constraints,
                    block_count,
                )
                
                # Accept/reject
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_positions = new_positions
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = new_positions
                        best_tree = tree.copy()
                else:
                    tree = old_tree
            
            temp *= self.cooling_rate
        
        return best_positions
    
    def _calibrate_penalty_weights(
        self,
        positions: List[Tuple[float, float, float, float]],
        constraints: torch.Tensor,
        block_count: int,
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> None:
        """
        Scale boundary and grouping penalties to ~target fractions of initial HPWL.

        Uses the first packed solution so weights follow case scale (HPWL, bbox).
        """
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        hpwl_scale = max(
            self.COST_WEIGHT_HPWL_B2B * hpwl_b2b
            + self.COST_WEIGHT_HPWL_P2B * hpwl_p2b,
            1.0,
        )
        boundary_pen = _boundary_penalty_squared_sum(
            positions, constraints, block_count
        )
        target_boundary = self.BOUNDARY_HPWL_RATIO * hpwl_scale
        if boundary_pen > 1e-9:
            boundary_scale = target_boundary / boundary_pen
        else:
            area = calculate_bbox_area(positions)
            boundary_scale = target_boundary / max(area * 0.01, 1.0)
        self._boundary_weight = self.COST_WEIGHT_BOUNDARY * boundary_scale

        grouping_pen = _grouping_penalty(positions, constraints, block_count)
        target_grouping = self.GROUPING_HPWL_RATIO * hpwl_scale
        if grouping_pen > 1e-9:
            grouping_scale = target_grouping / grouping_pen
        else:
            grouping_scale = target_grouping / max(hpwl_scale * 0.01, 1.0)
        self._grouping_weight = self.COST_WEIGHT_GROUPING * grouping_scale

        if self.verbose:
            print(
                f"[cost] boundary_weight={self._boundary_weight:.4g} "
                f"(init boundary_pen={boundary_pen:.4g}, hpwl_scale={hpwl_scale:.4g}); "
                f"grouping_weight={self._grouping_weight:.4g} "
                f"(init grouping_pen={grouping_pen:.4g})"
            )

    def _calibrate_boundary_weight(
        self,
        positions: List[Tuple[float, float, float, float]],
        constraints: torch.Tensor,
        block_count: int,
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> None:
        """Backward-compatible alias for penalty weight calibration."""
        self._calibrate_penalty_weights(
            positions, constraints, block_count, b2b_conn, p2b_conn, pins_pos
        )

    def _cost(
        self,
        positions: List[Tuple[float, float, float, float]],
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
        block_count: Optional[int] = None,
    ) -> float:
        """Evaluate solution quality (lower is better)."""
        n_blocks = block_count if block_count is not None else len(positions)
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        cost = (
            self.COST_WEIGHT_HPWL_B2B * hpwl_b2b
            + self.COST_WEIGHT_HPWL_P2B * hpwl_p2b
            + self.COST_WEIGHT_BBOX_AREA * area
        )
        if constraints is not None:
            boundary_pen = _boundary_penalty_squared_sum(
                positions, constraints, n_blocks
            )
            cost += self._boundary_weight * boundary_pen
            grouping_pen = _grouping_penalty(positions, constraints, n_blocks)
            cost += self._grouping_weight * grouping_pen
        return cost