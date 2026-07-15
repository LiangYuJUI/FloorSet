# `test_optimizer.py` — B*-Tree Simulated Annealing Optimizer

This document describes the floorplanning optimizer in  
[`iccad2026contest/test_optimizer.py`](../iccad2026contest/test_optimizer.py).

The optimizer uses a **movable-only B*-tree** representation, **constraint-aware initialization**, and **simulated annealing (SA)** to produce overlap-free placements that respect hard constraints (area, fixed/preplaced dimensions) while improving soft constraints (boundary, grouping) and wirelength.

---

## High-level flow (`MyOptimizer.solve`)

```
solve()
  │
  ├─ INITIALIZATION
  │    ├─ _preplaced_blocks              → fixed obstacles
  │    ├─ _initialize_block_dimensions   → (w, h) per movable block
  │    ├─ _fixed_shape_tree_nodes        → nodes that cannot rotate
  │    ├─ _tree_boundary_codes           → boundary codes for tree nodes
  │    └─ BStarTree(...)                 → constrained tree topology
  │
  ├─ FIRST PACK + CALIBRATION
  │    ├─ BStarTree.pack()               → initial (x, y, w, h)
  │    └─ _calibrate_penalty_weights()    → boundary / grouping weights
  │
  └─ SIMULATED ANNEALING LOOP
       ├─ move_rotate / move_delete_insert
       ├─ pack()
       ├─ _cost()
       └─ accept / reject → return best_positions
```

---

## Input / output

### `solve()` inputs

| Argument | Shape | Meaning |
|----------|-------|---------|
| `block_count` | int | Number of blocks |
| `area_targets` | `[n]` | Target area per block |
| `b2b_connectivity` | `[E, 3]` | Block-to-block nets `(i, j, weight)` |
| `p2b_connectivity` | `[E, 3]` | Pin-to-block nets `(pin, block, weight)` |
| `pins_pos` | `[n_pins, 2]` | Pin coordinates |
| `constraints` | `[n, 5]` | `(fixed, preplaced, MIB, cluster, boundary)` |
| `target_positions` | `[n, 4]` | Target `(x, y, w, h)`; `-1` where unknown |

### `constraints` columns

| Col | Name | Meaning |
|-----|------|---------|
| 0 | fixed | `1` = fixed-shape (immutable `w, h`) |
| 1 | preplaced | `1` = fixed `(x, y, w, h)` |
| 2 | MIB | Multi-instantiation group id (`≥1`; `0`/`-1` = none) |
| 3 | cluster | Grouping id (`≥0`; `-1` = none) |
| 4 | boundary | Bitmask: `1=left, 2=right, 4=top, 8=bottom` |

Corner codes: `5=TL`, `6=TR`, `9=BL` (permanent root), `10=BR`.

### Output

`List[(x, y, w, h)]` with exactly `block_count` tuples.

---

## Tunable constants (top of file)

```python
AREA_TOLERANCE = 0.01          # ±1% area for soft blocks

SA_INITIAL_TEMP_PER_BLOCK = 100.0
SA_FINAL_TEMP = 0.1
SA_COOLING_RATE = 0.6
SA_MOVES_PER_TEMP = 1000
```

**SA runtime estimate:**

```
initial_temp = SA_INITIAL_TEMP_PER_BLOCK × block_count
temp_steps   ≈ ceil( log(SA_FINAL_TEMP / initial_temp) / log(SA_COOLING_RATE) )
total_moves  ≈ SA_MOVES_PER_TEMP × temp_steps
```

Each move calls `pack()` + `_cost()` (dominant cost).

---

# Part 1 — Initialization step

Functions called before the SA loop starts.

## 1.1 Block classification helpers

### `_is_preplaced(constraints, i) -> bool`

Returns `True` if block `i` has `constraints[i, 1] != 0`.  
Preplaced blocks are **excluded from the B*-tree** and written directly into `positions` during `pack()`.

### `_is_fixed_shape(constraints, i) -> bool`

Returns `True` if `constraints[i, 0] != 0`. Fixed-shape blocks stay in the tree but their `(w, h)` cannot change.

### `_is_soft(constraints, i) -> bool`

Returns `True` if the block is neither preplaced nor fixed-shape. Only soft blocks may change aspect ratio during SA.

### `_mib_group_id(constraints, i) -> Optional[int]`

Returns the MIB group id from `constraints[i, 2]`, or `None` if the block has no MIB constraint (`< 0` or `0`).

---

## 1.2 Preplaced extraction

### `_preplaced_blocks(constraints, target_positions, block_count)`

**Returns:** `(preplaced_indices, preplaced_info)`

- Scans all blocks with `constraints[:, 1] != 0`.
- Reads exact `(x, y, w, h)` from `target_positions`.
- Used in `solve()` to build `preplaced_set` and pass obstacle rectangles to `BStarTree`.

---

## 1.3 MIB analysis

### `_analyse_mib_groups(constraints, block_count, target_positions, area_targets)`

**Returns:** `(forced_dim, free_mib, block_to_mib)`

| Output | Meaning |
|--------|---------|
| `forced_dim` | `group_id → (w, h)` when a group contains a fixed/preplaced member |
| `free_mib` | Group ids where **all** members are soft (shape chosen later) |
| `block_to_mib` | `block_index → group_id` |

**Rule:** If any non-soft member exists in a MIB group, all soft members in that group must share that member's `(w, h)`.

### `_target_wh(i, area_targets, target_positions) -> (w, h)`

Reads `(w, h)` from `target_positions` when set; otherwise returns `√area` square.

---

## 1.4 Bottom-left root dimension adjustment

### `_find_bottom_left_block(constraints, block_count) -> int`

Finds the unique block with boundary code `9` (bottom-left). Returns global index, or `-1`.

### `_adjust_soft_root_at_origin(area, preplaced_info) -> (w, h)`

Used when the BL block is **soft** and not in a forced MIB group.

1. Start with square `w = h = √area`.
2. If it overlaps any preplaced rectangle at `(0, 0)`, try shrinking `w` or `h` to clear obstacles while keeping area within ±1% (`AREA_TOLERANCE`).
3. Picks the candidate closest to square aspect ratio.

**Helpers used:**

| Function | Role |
|----------|------|
| `_overlaps_preplaced_rect` | 2D overlap test vs preplaced obstacles |
| `_w_cap_at_origin` | Max width at `y=0` before hitting a preplaced block |
| `_h_cap_at_origin` | Max height at `x=0` before hitting a preplaced block |
| `_area_in_tolerance` | Checks `0.99×area ≤ w×h ≤ 1.01×area` |
| `_clamp_dims_to_area` | Scales `(w, h)` into the legal area band |

---

## 1.5 Main dimension initializer

### `_initialize_block_dimensions(...)`

**Returns:** `(mov_widths, mov_heights, tree_root, locked_tree)`

Central initialization routine. For every block:

| Block type | `(w, h)` assignment |
|------------|---------------------|
| Preplaced | From `target_positions` (not in movable lists) |
| Fixed-shape | From `target_positions` |
| Soft + forced MIB | Forced group `(w, h)` |
| Soft BL root (code 9) | `_adjust_soft_root_at_origin` |
| Soft + free MIB | Same shape as BL root if BL is in that group; else `√area` square |
| Other soft | `√area` square |

Also sets:

- **`tree_root`** — tree-local index of BL block (permanent root), or `-1` if BL is preplaced.
- **`locked_tree`** — tree-local indices that cannot rotate (fixed-shape + all MIB members).

---

## 1.6 Tree node locking

### `_fixed_shape_tree_nodes(constraints, movable_indices, mib_locked_tree_nodes)`

Merges MIB-locked tree nodes with fixed-shape nodes into `fixed_tree_nodes` passed to `BStarTree`. These nodes are excluded from `move_rotate`.

### `_tree_boundary_codes(constraints, movable_indices)`

Builds a per-tree-node list of `constraints[:, 4]` values for movable blocks. Used by `_build_constrained_tree`.

---

## 1.7 B*-tree construction (`BStarTree.__init__` → `_build_tree`)

The tree is built over **movable blocks only**. Tree node `i` maps to global block `movable_indices[i]`.

### `_build_tree()`

Dispatches to:

- **`_build_constrained_tree()`** — when `boundary_codes` and `permanent_root` are set.
- **`_build_random_tree()`** — fallback.

### `_build_constrained_tree()`

Builds two boundary chains from the permanent root (BL block, code 9):

**Bottom chain (left-child spine)** — blocks at `y = 0`:

```
root → [bottom_middle…] → BR(10) → [right_only(2)…]
```

**Left chain (right-child spine)** — blocks at `x = 0`:

```
root → [left_middle…] → TL(5) → [top_only(4)…]
```

**Top-right (code 6)** is **not** chained; inserted as a free block (boundary penalty guides SA).

**Free blocks** (no boundary bits, including TR) are attached via `_insert_free_block` without breaking chain spines.

| Helper | Role |
|--------|------|
| `_attach_child(parent, block, as_left)` | Sets `left` or `right` pointer + `parent` |
| `_rebuild_chain_membership()` | Fills `bottom_chain_nodes` / `left_chain_nodes` by walking spines |
| `_insert_free_block(block)` | Attaches free block to a valid empty slot |
| `_insert_at_leaf_constrained(block, start)` | Leaf fallback respecting chain sides |

### Chain protection sets

After construction, `_rebuild_chain_membership()` populates:

- `bottom_chain_nodes` — all nodes on `root.left.left…` spine
- `left_chain_nodes` — all nodes on `root.right.right…` spine

The permanent root is in **both** sets.

---

## 1.8 First placement (`BStarTree.pack`)

Called once before SA and after every SA move.

### `pack() -> List[(x, y, w, h)]`

1. Write preplaced blocks into `positions` from `preplaced_info`.
2. **DFS** from `root`:
   - **Root:** `x = 0`; `y = 0` if `permanent_root` is set, else skyline + `_placement_y`.
   - **Left child:** placed to the **right** of parent (`x = parent_right_edge`).
   - **Right child:** placed at same `x`, above parent (higher `y` via skyline).
3. **`_placement_y`** — bumps `y` upward only when a 2D overlap with a preplaced block would occur (preplaced blocks are not merged into the 1D skyline).
4. **`_compact_horizontally(positions)`** — post-process: shift movable blocks left as far as possible without changing `y/w/h` or creating overlaps.

**Packing helpers:**

| Function | Role |
|----------|------|
| `_merge_contour` | Merges adjacent skyline segments at equal height |
| `_raise_contour` | Updates 1D skyline after placing a block |
| `_overlaps_preplaced` | Wrapper for preplaced overlap test |
| `_compact_horizontally` | Left-compaction post-processing |

---

## 1.9 Penalty weight calibration (before SA)

### `_calibrate_penalty_weights(...)`  
(alias: `_calibrate_boundary_weight`)

Runs on the **first packed solution** to scale soft-constraint weights relative to initial HPWL:

| Weight | Target |
|--------|--------|
| `_boundary_weight` | `≈ 10%` of HPWL scale (`BOUNDARY_HPWL_RATIO = 0.1`) |
| `_grouping_weight` | `≈ 5%` of HPWL scale (`GROUPING_HPWL_RATIO = 0.05`) |

---

# Part 2 — Simulated annealing step

## 2.1 SA schedule

Configured via `SA_*` constants (see above). In `solve()`:

```python
temp = SA_INITIAL_TEMP_PER_BLOCK * block_count
while temp > SA_FINAL_TEMP:
    for _ in range(SA_MOVES_PER_TEMP):
        ...  # one SA move
    temp *= SA_COOLING_RATE
```

## 2.2 SA moves (`BStarTree` methods)

Each iteration picks **one** of two moves (50/50 when both are available):

### `move_rotate(block)`

Swaps `widths[block]` and `heights[block]` (90° rotation, area preserved).

**Blocked for:**
- `permanent_root`
- `fixed_tree_nodes` (fixed-shape + MIB-locked)

In `solve()`, `rotatable_nodes` excludes fixed/MIB/permanent-root nodes.

### `move_delete_insert(block)`

Deletes `block` from the tree and reinserts it as a child of a random node.

**Blocked for:**
- `permanent_root`
- Any node in `bottom_chain_nodes` or `left_chain_nodes`

**Helpers:**

| Function | Role |
|----------|------|
| `_delete_node(node)` | Removes node; promotes child if needed |
| `_insert_node(node, target, as_left)` | Attaches node under `target` |

### `move_swap(b1, b2)` *(implemented, not used in current SA loop)*

Swaps `(w, h)` between two blocks. Only allowed when `_chain_swap_allowed` returns `True` (both in the same bottom or left chain, neither is permanent root).

### `copy()`

Deep-copies tree topology, dimensions, and chain membership. Used for SA rollback and best-solution tracking.

---

## 2.3 Cost evaluation (`_cost`)

Called after every `pack()`. **Lower is better** for SA acceptance.

```
cost = w_b2b × HPWL_b2b
     + w_p2b × HPWL_p2b
     + w_area × bbox_area
     + boundary_weight × boundary_penalty
     + grouping_weight × grouping_penalty
```

### Primary terms (from `iccad2026_evaluate`)

| Function | Measures |
|----------|----------|
| `calculate_hpwl_b2b` | Weighted block-to-block HPWL |
| `calculate_hpwl_p2b` | Weighted pin-to-block HPWL |
| `calculate_bbox_area` | Bounding-box area of all blocks |

### Soft-constraint penalties

#### `_boundary_penalty_squared_sum(positions, constraints, block_count)`

For each block with a boundary code, sums **squared distances** from the block edge to the required layout bounding-box edge (left/right/top/bottom per bitmask).

#### `_grouping_penalty(positions, constraints, block_count)`

For each cluster group (`constraints[:, 3] ≥ 0`):

1. Build adjacency graph where blocks share a **full edge** (`_blocks_share_edge`).
2. Count connected components `c` via union-find.
3. Add `(c - 1)²` to the penalty.

Corner-only contact does **not** count as connected.

---

## 2.4 Accept / reject

```python
delta = new_cost - current_cost
if delta < 0 or random() < exp(-delta / temp):
    accept  # update current + maybe best
else:
    tree = old_tree  # rollback
```

Returns `best_positions` after cooling completes.

---

# Part 3 — Function index

## Initialization only

| Function | Purpose |
|----------|---------|
| `_is_preplaced` | Detect preplaced blocks |
| `_is_fixed_shape` | Detect fixed-shape blocks |
| `_is_soft` | Detect soft blocks |
| `_mib_group_id` | Read MIB group id |
| `_preplaced_blocks` | Extract preplaced indices + rectangles |
| `_analyse_mib_groups` | MIB forced/free group analysis |
| `_target_wh` | Read or default `(w, h)` |
| `_find_bottom_left_block` | Find BL corner block (code 9) |
| `_adjust_soft_root_at_origin` | BL soft-block shape at `(0,0)` |
| `_overlaps_preplaced_rect` | 2D overlap vs preplaced |
| `_w_cap_at_origin` / `_h_cap_at_origin` | Clearance caps for root shaping |
| `_area_in_tolerance` / `_clamp_dims_to_area` | Area ±1% helpers |
| `_initialize_block_dimensions` | **Main dimension + root initializer** |
| `_fixed_shape_tree_nodes` | Nodes forbidden from rotation |
| `_tree_boundary_codes` | Per-tree-node boundary codes |
| `BStarTree._build_tree` | Tree topology dispatcher |
| `BStarTree._build_constrained_tree` | Boundary chain construction |
| `BStarTree._build_random_tree` | Random fallback tree |
| `BStarTree._rebuild_chain_membership` | Fill chain node sets |
| `BStarTree._attach_child` | Link parent → child |
| `BStarTree._insert_free_block` | Insert unconstrained blocks |
| `BStarTree._insert_at_leaf_constrained` | Constrained leaf insert |
| `BStarTree.pack` | **Initial placement** |
| `BStarTree._placement_y` | 2D preplaced bump |
| `BStarTree._compact_horizontally` | Left compaction |
| `_calibrate_penalty_weights` | Calibrate soft penalty weights |

## SA loop only

| Function | Purpose |
|----------|---------|
| `BStarTree.move_rotate` | SA move: rotate block |
| `BStarTree.move_delete_insert` | SA move: restructure tree |
| `BStarTree.move_swap` | SA move: swap dims (not in loop) |
| `BStarTree._delete_node` | Delete helper for reinsert |
| `BStarTree._insert_node` | Insert helper for reinsert |
| `BStarTree._chain_swap_allowed` | Chain-aware swap guard |
| `BStarTree.copy` | Snapshot for rollback / best |
| `BStarTree.pack` | Re-pack after each move |
| `_cost` | **SA objective function** |
| `_boundary_penalty_squared_sum` | Boundary soft penalty |
| `_grouping_penalty` | Grouping soft penalty |
| `_blocks_share_edge` | Edge-adjacency for grouping |

---

# Running evaluation

```bash
cd iccad2026contest
source ../.venv/bin/activate

# Single case
python iccad2026_evaluate_test.py --evaluate test_optimizer.py --test-id 0

# All cases in parallel (10 workers)
python run_parallel.py --optimizer test_optimizer.py --workers 10

# Visualize one result
python visualize_results_json.py \
  --results-json results_json/case_000.json --test-id 0 --no-gt
```

Per-case JSON output from `run_parallel.py` is saved under `iccad2026contest/results_json/`.
