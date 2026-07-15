#!/usr/bin/env bash
# Run full validation evaluation or re-score saved results.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/iccad2026contest"

if [[ $# -ge 1 && "$1" == "--score" ]]; then
  python iccad2026_evaluate.py --score results/test_optimizer_results.json --data-path "$ROOT"
  exit 0
fi

if [[ $# -ge 1 && "$1" == "--test-id" ]]; then
  python iccad2026_evaluate.py --evaluate test_optimizer.py --test-id "$2" --data-path "$ROOT"
  exit 0
fi

python iccad2026_evaluate.py --evaluate test_optimizer.py --data-path "$ROOT" --output results/test_optimizer_results.json
