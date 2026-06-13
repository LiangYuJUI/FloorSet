# Contest helper scripts

Run all commands from `iccad2026contest/` with the project venv active:

```bash
source ../../../.venv/bin/activate   # repo root .venv
cd third_party/FloorSet/iccad2026contest
```

## `visualize_baseline.py`

Baseline constraint PNGs for validation cases (`0`–`99`). Use `--mode` to choose the visualization.

| Mode | Shows | Default output directory |
|------|--------|---------------------------|
| `preplaced` | Blocks with `constraints[i, 1] != 0` + all pins | `images/baseline/preplaced/` |
| `boundary` | Blocks with `constraints[i, 4] != 0` (boundary soft constraint) + pins + dashed solution bbox | `images/baseline/boundary/` |
| `cluster` | Blocks with `constraints[i, 3] != 0` (grouping/cluster) + pins; **one color per cluster id** | `images/baseline/cluster/` |
| `floorplan` | **All blocks** in the layout + pins (full reference or optimizer solution) | `images/baseline/floorplan/` |
| `combine` | **2×2 mosaic** of the four panels above (one PNG per case) | `images/baseline/combined/` |

**Pins** are green circles from `pins_pos` (`x, y >= 0`). The viewport is the baseline block bounding box (expanded to include pins).

**Boundary labels** use the contest bitmask: `L`, `R`, `T`, `B`, and corners `TL`, `TR`, `BL`, `BR` (codes `5`, `6`, `9`, `10`).

### Preplaced — single case

```bash
python visualize_baseline.py --mode preplaced --test-id 40
```

Writes `images/baseline/preplaced/preplaced_case_40.png`.

### Preplaced — optimizer positions

```bash
python visualize_baseline.py --mode preplaced --test-id 40 \
  --results-json my_optimizer_results.json \
  --name-prefix my_optimizer \
  --out images/baseline/preplaced/my_optimizer_case_40.png
```

### Boundary — single case

```bash
python visualize_baseline.py --mode boundary --test-id 99
```

Writes `images/baseline/boundary/boundary_case_99.png`.

### Cluster — single case

Blocks sharing the same `cluster_id` (`constraints[i, 3]`) use the same fill color. Labels show `block_id` and `G<cluster_id>`.

```bash
python visualize_baseline.py --mode cluster --test-id 97
```

Writes `images/baseline/cluster/cluster_case_97.png`.

### Floorplan — entire layout

```bash
python visualize_baseline.py --mode floorplan --test-id 62
```

Writes `images/baseline/floorplan/floorplan_case_62.png`. With optimizer results:

```bash
python visualize_baseline.py --mode floorplan --test-id 62 \
  --results-json my_optimizer_results.json \
  --name-prefix my_optimizer
```

### Combine — four panels in one PNG

Requires the four panel files (`preplaced_case_N.png`, `boundary_case_N.png`, etc.) to exist, or use `--generate-missing` to create them first.

```bash
python visualize_baseline.py --mode combine --test-id 62
python visualize_baseline.py --mode combine --test-id 62 --generate-missing
python visualize_baseline.py --mode combine --all --generate-missing
```

Writes `images/baseline/combined/combined_case_62.png` (layout: preplaced | boundary / cluster | floorplan).

### All validation cases

```bash
python visualize_baseline.py --mode preplaced --all
python visualize_baseline.py --mode boundary --all
python visualize_baseline.py --mode cluster --all
python visualize_baseline.py --mode floorplan --all
python visualize_baseline.py --mode combine --all
```

### Options

| Flag | Description |
|------|-------------|
| `--mode {preplaced,boundary,cluster,floorplan,combine}` | Visualization type (default: `preplaced`). |
| `--generate-missing` | With `combine`: render missing panel PNGs before stitching. |
| `--test-id N` | Case index (`0`–`99`). Required unless `--all`. |
| `--all` | Export all 100 cases. |
| `--data-path PATH` | FloorSet data root (default: `../`). |
| `--results-json PATH` | Optional `*_results.json`; use optimizer block positions. |
| `--out PATH` | Output PNG for one `--test-id`. |
| `--out-dir PATH` | Output directory for `--all` (mode-specific default). |
| `--name-prefix PREFIX` | Filename prefix (default: mode name). |
| `--show` | Open matplotlib window after saving. |

## `visualize_results_json.py`

Side-by-side **ground truth vs optimizer solution** from a `*_results.json` file (or solution-only with `--no-gt`).

Default output directory: `images/results/`. Filename prefix defaults to `submission_name` in the JSON (e.g. `my_optimizer_case_40.png`).

### All cases from `my_optimizer_results.json`

```bash
python visualize_results_json.py --results-json my_optimizer_results.json --all
```

Writes `images/results/my_optimizer_case_0.png` … `my_optimizer_case_99.png`.

Faster (solution only, no dataset load):

```bash
python visualize_results_json.py --results-json my_optimizer_results.json --all --no-gt
```

### Single case

```bash
python visualize_results_json.py --results-json my_optimizer_results.json --test-id 40
```

### Options

| Flag | Description |
|------|-------------|
| `--results-json PATH` | Required. Evaluation output JSON. |
| `--test-id N` | One case (`0`–`99`). Required unless `--all`. |
| `--all` | Export every `test_id` in the JSON. |
| `--out-dir PATH` | Directory for `--all` (default: `images/results`). |
| `--out PATH` | Output file for a single `--test-id`. |
| `--name-prefix PREFIX` | Override filename prefix (default: `submission_name`). |
| `--no-gt` | Skip ground-truth panel. |
| `--data-path PATH` | FloorSet data root (default: `../`). |
| `--show` | Open matplotlib window (single case). |

### Related commands

Ground truth (evaluator):

```bash
python iccad2026_evaluate.py --visualize --test-id 0
```
