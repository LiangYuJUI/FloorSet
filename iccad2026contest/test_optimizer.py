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
from typing import List, Optional, Set, Tuple

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
) -> Set[int]:
    """Tree-local indices of movable blocks with fixed (w, h)."""
    fixed: Set[int] = set()
    if constraints is None or not movable_indices:
        return fixed
    nc = int(constraints.shape[1]) if constraints.dim() > 1 else 0
    if nc <= 0:
        return fixed
    for tree_i, orig_i in enumerate(movable_indices):
        if float(constraints[orig_i, 0].item()) != 0:
            fixed.add(tree_i)
    return fixed


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
    ):
        """
        B*-tree over movable blocks only.

        Tree node i maps to original block movable_indices[i].
        Preplaced blocks are fixed obstacles during pack().
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
        """True if (x, y, w, h) intersects any preplaced rectangle."""
        for px, py, pw, ph in preplaced_info:
            if (
                x + w > px + eps
                and x < px + pw - eps
                and y + h > py + eps
                and y < py + ph - eps
            ):
                return True
        return False

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
                skyline_y = get_contour_y(0.0, w)
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
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        if block in self.fixed_tree_nodes:
            return
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


# Boundary soft-constraint bit mask (constraints[:, 4])
_BOUNDARY_LEFT = 1
_BOUNDARY_RIGHT = 2
_BOUNDARY_TOP = 4
_BOUNDARY_BOTTOM = 8


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

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.final_temp = 0.1
        self.cooling_rate = 0.95
        self.moves_per_temp = 100
        self._boundary_weight: float = 1.0
    
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

        def block_dimensions(i: int) -> Tuple[float, float]:
            if (target_positions is not None and
                    target_positions[i, 2] != -1 and target_positions[i, 3] != -1):
                return (
                    float(target_positions[i, 2]),
                    float(target_positions[i, 3]),
                )
            area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            side = math.sqrt(area * 0.991)
            return side, side

        movable_indices = [
            i for i in range(block_count) if i not in preplaced_set
        ]
        mov_widths, mov_heights = [], []
        for i in movable_indices:
            w, h = block_dimensions(i)
            mov_widths.append(w)
            mov_heights.append(h)

        fixed_tree_nodes = _fixed_shape_tree_nodes(constraints, movable_indices)
        rotatable_nodes = [
            i for i in range(len(movable_indices)) if i not in fixed_tree_nodes
        ]

        tree = BStarTree(
            len(movable_indices),
            mov_widths,
            mov_heights,
            preplaced_info=preplaced_info,
            preplaced_indices=preplaced_indices,
            movable_indices=movable_indices,
            total_blocks=block_count,
            fixed_tree_nodes=fixed_tree_nodes,
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

        # Simulated Annealing (initial temperature scales with instance size)
        self.initial_temp = 100.0 * block_count
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
    
    def _calibrate_boundary_weight(
        self,
        positions: List[Tuple[float, float, float, float]],
        constraints: torch.Tensor,
        block_count: int,
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> None:
        """
        Scale boundary penalty to ~BOUNDARY_HPWL_RATIO × initial HPWL magnitude.

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
            scale = target_boundary / boundary_pen
        else:
            area = calculate_bbox_area(positions)
            # Fallback when no boundary blocks or already satisfied: tie to bbox².
            scale = target_boundary / max(area * 0.01, 1.0)
        self._boundary_weight = self.COST_WEIGHT_BOUNDARY * scale
        if self.verbose:
            print(
                f"[cost] boundary_weight={self._boundary_weight:.4g} "
                f"(init boundary_pen={boundary_pen:.4g}, hpwl_scale={hpwl_scale:.4g})"
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
        return cost