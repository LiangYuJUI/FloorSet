#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: Bottom-left packing + compaction
  - GUARANTEES: Overlap-free pack, area on soft blocks,
    exact fixed (w,h) and preplaced (x,y,w,h) when target_positions is set
  - Tightens bbox via greedy BL placement and down/left squeeze
  - NOT HANDLED: MIB, cluster, boundary soft constraints

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
from typing import List, Optional, Set, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


def _constraint_masks(
    constraints: torch.Tensor, block_count: int
) -> Tuple[Set[int], Set[int]]:
    """Return (preplaced_indices, fixed_only_indices)."""
    pre: Set[int] = set()
    fixed_only: Set[int] = set()
    if constraints is None or block_count == 0:
        return pre, fixed_only
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    for i in range(block_count):
        if nc > 1 and float(constraints[i, 1].item()) != 0:
            pre.add(i)
        elif nc > 0 and float(constraints[i, 0].item()) != 0:
            fixed_only.add(i)
    return pre, fixed_only


def _outline_from_pins(pins_pos: torch.Tensor, scale: float = 3) -> Tuple[float, float]:
    """Return an abundant outline (W,H) based on max valid pin coords."""
    if pins_pos is None or pins_pos.numel() == 0:
        return 100.0, 100.0
    valid = (pins_pos[:, 0] >= 0) & (pins_pos[:, 1] >= 0)
    if not bool(valid.any().item()):
        return 100.0, 100.0
    max_x = float(pins_pos[valid, 0].max().item())
    max_y = float(pins_pos[valid, 1].max().item())
    W = max(1.0, max_x * scale)
    H = max(1.0, max_y * scale)
    return W, H


def _corner_boundary_blocks(
    constraints: torch.Tensor, block_count: int
) -> Tuple[Set[int], dict]:
    """
    Return (corner_indices, corner_code_by_index) for boundary-corner blocks only.

    Boundary encoding (bitmask): 1=left, 2=right, 4=top, 8=bottom.
    Corners are sums: TL=5, TR=6, BL=9, BR=10.
    """
    corner: Set[int] = set()
    codes: dict = {}
    if constraints is None or block_count == 0:
        return corner, codes
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 4:
        return corner, codes
    for i in range(block_count):
        code = int(float(constraints[i, 4].item()))
        if code in (5, 6, 9, 10):
            corner.add(i)
            codes[i] = code
    return corner, codes


def _block_dimensions(
    i: int,
    area_targets: torch.Tensor,
    target_positions: Optional[torch.Tensor],
) -> Tuple[float, float]:
    if (
        target_positions is not None
        and float(target_positions[i, 2].item()) != -1
        and float(target_positions[i, 3].item()) != -1
    ):
        return (
            float(target_positions[i, 2].item()),
            float(target_positions[i, 3].item()),
        )
    area = float(area_targets[i].item()) if area_targets[i].item() > 0 else 1.0
    s = math.sqrt(area)
    return s, s


def _rects_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    eps: float = 1e-6,
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)
    return overlap_x > eps and overlap_y > eps


def _overlaps_any(
    rect: Tuple[float, float, float, float],
    positions: List[Tuple[float, float, float, float]],
    ignore_idx: int,
    block_count: int,
) -> bool:
    for j in range(block_count):
        if j == ignore_idx:
            continue
        if _rects_overlap(rect, positions[j]):
            return True
    return False


def _shift_pack_to_origin(
    positions: List[Tuple[float, float, float, float]],
    pack_indices: List[int],
    obstacles: Optional[Set[int]] = None,
) -> None:
    """Translate pack_indices so their bbox starts at (0,0), avoiding obstacles."""
    if not pack_indices:
        return
    min_x = min(positions[i][0] for i in pack_indices)
    min_y = min(positions[i][1] for i in pack_indices)
    dx, dy = -min_x, -min_y
    if dx != 0.0 or dy != 0.0:
        for i in pack_indices:
            x, y, w, h = positions[i]
            positions[i] = (x + dx, y + dy, w, h)

    if not obstacles:
        return

    eps = 1e-6
    for _ in range(256):
        if not any(
            _rects_overlap(positions[i], positions[j])
            for i in pack_indices
            for j in obstacles
        ):
            return
        shift_x = 0.0
        lift = 0.0
        for i in pack_indices:
            xi, yi, wi, hi = positions[i]
            for j in obstacles:
                xj, yj, wj, hj = positions[j]
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                if ox > eps and oy > eps:
                    # Prefer sliding right before lifting up.
                    shift_x = max(shift_x, xj + wj - xi + eps)
                    lift = max(lift, yj + hj - yi + eps)
        if shift_x > eps:
            for i in pack_indices:
                x, y, w, h = positions[i]
                positions[i] = (x + shift_x, y, w, h)
            continue
        if lift > eps:
            for i in pack_indices:
                x, y, w, h = positions[i]
                positions[i] = (x, y + lift, w, h)
            continue
        return

def _squeeze_layout(
    positions: List[Tuple[float, float, float, float]],
    movable: List[int],
    block_count: int,
    step: float = 1.0,
    passes: int = 12,
) -> None:
    """Push movable blocks down, then left, without overlaps."""
    eps = 1e-6
    for _ in range(passes):
        order = sorted(
            movable,
            key=lambda i: (-positions[i][1], positions[i][0]),
        )
        for i in order:
            x, y, w, h = positions[i]
            while y > eps:
                ny = max(0.0, y - step)
                if not _overlaps_any((x, ny, w, h), positions, i, block_count):
                    y = ny
                else:
                    break
            while x > eps:
                nx = max(0.0, x - step)
                if not _overlaps_any((nx, y, w, h), positions, i, block_count):
                    x = nx
                else:
                    break
            positions[i] = (x, y, w, h)


def _layout_from_tree_pack(
    block_count: int,
    tree_indices: List[int],
    tree_positions: List[Tuple[float, float, float, float]],
    preplaced: Set[int],
    fixed_only: Set[int],
    corner_blocks: Set[int],
    corner_codes: dict,
    corner_dims: dict,
    outline_wh: Tuple[float, float],
    target_positions: Optional[torch.Tensor],
    squeeze_passes: int = 16,
) -> List[Tuple[float, float, float, float]]:
    """Merge B*-tree pack, squeeze down/left, then shift pack to (0,0)."""
    full: List[Tuple[float, float, float, float]] = [
        (0.0, 0.0, 1.0, 1.0)
    ] * block_count
    if target_positions is not None:
        for i in preplaced:
            full[i] = tuple(
                float(target_positions[i, j].item()) for j in range(4)
            )
    for j, gi in enumerate(tree_indices):
        full[gi] = tree_positions[j]
    if target_positions is not None:
        for i in fixed_only:
            x, y, _, _ = full[i]
            tw = float(target_positions[i, 2].item())
            th = float(target_positions[i, 3].item())
            full[i] = (x, y, tw, th)
    # Place ONLY the bottom-left corner block (BL=9) at init; other corners use B*-tree.
    # Preplaced blocks are handled separately and always remain fixed.
    W, H = outline_wh
    for i in corner_blocks:
        if i in preplaced:
            continue
        if int(corner_codes.get(i, 0)) != 9:
            continue
        if (
            target_positions is not None
            and float(target_positions[i, 2].item()) != -1
            and float(target_positions[i, 3].item()) != -1
        ):
            w = float(target_positions[i, 2].item())
            h = float(target_positions[i, 3].item())
        else:
            w, h = corner_dims.get(i, (full[i][2], full[i][3]))
        full[i] = (0.0, 0.0, w, h)
    if squeeze_passes > 0:
        _squeeze_layout(full, tree_indices, block_count, step=0.5, passes=squeeze_passes)
    _shift_pack_to_origin(full, tree_indices, preplaced | corner_blocks)
    if target_positions is not None:
        for i in preplaced:
            full[i] = tuple(
                float(target_positions[i, j].item()) for j in range(4)
            )
        for i in fixed_only:
            x, y, _, _ = full[i]
            full[i] = (
                x,
                y,
                float(target_positions[i, 2].item()),
                float(target_positions[i, 3].item()),
            )
        # Re-apply bottom-left corner anchor after any numerical drift.
        for i in corner_blocks:
            if i in preplaced or int(corner_codes.get(i, 0)) != 9:
                continue
            if (
                float(target_positions[i, 2].item()) != -1
                and float(target_positions[i, 3].item()) != -1
            ):
                w = float(target_positions[i, 2].item())
                h = float(target_positions[i, 3].item())
            else:
                w, h = corner_dims.get(i, (full[i][2], full[i][3]))
            full[i] = (0.0, 0.0, w, h)
    return full


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()
    
    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        
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
    
    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Compute (x, y, w, h) from tree structure.
        
        Uses proper contour tracking to ensure overlap-free placement.
        B*-tree rules:
        - Left child: placed to the RIGHT of parent
        - Right child: placed ABOVE parent (same x as parent)
        """
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        
        # Contour: sorted list of (x_end, y_top) representing skyline
        # At any x, the contour height is the y_top of the rightmost segment with x_end > x
        contour = [(0.0, 0.0)]  # Start with ground level
        
        def get_contour_y(x_start: float, x_end: float) -> float:
            """Find max y in contour for range [x_start, x_end]."""
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                # Get x_start of this segment
                cx_start = contour[i-1][0] if i > 0 else 0.0
                # Check if segments overlap
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y
        
        def update_contour(x_start: float, x_end: float, y_top: float):
            """Add a new block to the contour."""
            nonlocal contour
            new_contour = []
            
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                
                # Before the new block
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                # After the new block
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                # Overlapping - need to split
                else:
                    # Part before new block
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    # Part after new block
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            
            # Add the new block segment
            # Find where to insert
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            
            # Sort by x_end and merge adjacent segments with same y
            new_contour.sort(key=lambda x: x[0])
            
            # Merge adjacent segments with same height
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)  # Extend previous
                else:
                    merged.append((x_end, y_top))
            
            contour = merged if merged else [(x_end, 0.0)]
        
        # DFS traversal to place blocks
        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return
            
            w, h = self.widths[node], self.heights[node]
            
            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            
            # Left child: to the RIGHT of this node
            dfs(self.left[node], x + w)
            # Right child: ABOVE this node (same x, will stack due to contour)
            dfs(self.right[node], x)
        
        dfs(self.root, 0.0)
        
        # Verify no overlaps (should never happen with correct contour)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    # Fix by pushing j up
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)
        
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
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
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


# =============================================================================
# OPTIMIZER CLASS - Replace this with your algorithm
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    B*-tree SA with down/left compaction after each pack.

    REPLACE THIS CLASS WITH YOUR ALGORITHM.
    Keep the solve() signature the same.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.initial_temp = 100.0
        self.final_temp = 1.0
        self.cooling_rate = 0.9
        self.moves_per_temp = 20
        self.squeeze_passes = 20

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
        if block_count <= 0:
            return []

        preplaced, fixed_only = _constraint_masks(constraints, block_count)
        all_corners, all_corner_codes = _corner_boundary_blocks(constraints, block_count)
        # Preplaced blocks must remain fixed coordinates.
        # Init: only bottom-left corner (BL=9) is anchored; TL/TR/BR pack via B*-tree.
        bl_candidates = sorted(
            i
            for i in all_corners
            if i not in preplaced and int(all_corner_codes.get(i, 0)) == 9
        )
        corner_blocks = {bl_candidates[0]} if bl_candidates else set()
        corner_codes = {i: 9 for i in corner_blocks}
        corner_dims = {
            i: _block_dimensions(i, area_targets, target_positions) for i in corner_blocks
        }
        tree_indices = [
            i
            for i in range(block_count)
            if i not in preplaced and i not in corner_blocks
        ]
        outline_wh = _outline_from_pins(pins_pos, scale=1.5)

        if not tree_indices:
            if target_positions is None:
                return [(0.0, 0.0, 1.0, 1.0)] * block_count
            return [
                tuple(float(target_positions[i, j].item()) for j in range(4))
                for i in range(block_count)
            ]

        widths, heights = [], []
        for i in tree_indices:
            w, h = _block_dimensions(i, area_targets, target_positions)
            widths.append(w)
            heights.append(h)

        no_rotate_local = {
            j for j, gi in enumerate(tree_indices) if gi in fixed_only
        }
        rotate_locals = [
            j for j in range(len(tree_indices)) if j not in no_rotate_local
        ]

        def layout(tree: BStarTree, squeeze: bool = False) -> List[Tuple[float, float, float, float]]:
            return _layout_from_tree_pack(
                block_count,
                tree_indices,
                tree.pack(),
                preplaced,
                fixed_only,
                corner_blocks,
                corner_codes,
                corner_dims,
                outline_wh,
                target_positions,
                squeeze_passes=self.squeeze_passes if squeeze else 0,
            )

        tree = BStarTree(len(tree_indices), widths, heights)
        current_positions = layout(tree)
        current_cost = self._cost(
            current_positions, b2b_connectivity, p2b_connectivity, pins_pos
        )

        best_tree = tree.copy()
        best_cost = current_cost

        temp = self.initial_temp
        while temp > self.final_temp:
            for _ in range(self.moves_per_temp):
                old_tree = tree.copy()
                if random.randint(0, 1) == 0:
                    if not rotate_locals:
                        tree = old_tree
                        continue
                    tree.move_rotate(random.choice(rotate_locals))
                else:
                    tree.move_delete_insert(random.randint(0, len(tree_indices) - 1))

                new_positions = layout(tree)
                if check_overlap(new_positions) > 0:
                    tree = old_tree
                    continue

                new_cost = self._cost(
                    new_positions, b2b_connectivity, p2b_connectivity, pins_pos
                )
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_positions = new_positions
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tree = tree.copy()
                else:
                    tree = old_tree

            temp *= self.cooling_rate

        final_pos = layout(best_tree, squeeze=True)
        # _squeeze_layout(
        #     final_pos, tree_indices, block_count, step=0.25, passes=self.squeeze_passes
        # )
        # _shift_pack_to_origin(final_pos, tree_indices, preplaced | corner_blocks)
        # if target_positions is not None:
        #     for i in preplaced:
        #         final_pos[i] = tuple(
        #             float(target_positions[i, j].item()) for j in range(4)
        #         )
        #     for i in fixed_only:
        #         x, y, _, _ = final_pos[i]
        #         final_pos[i] = (
        #             x,
        #             y,
        #             float(target_positions[i, 2].item()),
        #             float(target_positions[i, 3].item()),
        #         )
        return final_pos
    
    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.08
