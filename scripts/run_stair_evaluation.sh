#!/usr/bin/env bash
# Run the assessment's four-step-staircase play test at every speed tier and
# collect the results.
#
# Usage:
#   scripts/run_stair_evaluation.sh <CHECKPOINT> [OUT_DIR]
#
# <CHECKPOINT> may be a run directory or a specific model_*.pt.
# Produces OUT_DIR/results_0.4.csv, results_0.6.csv, results_0.8.csv and prints a
# combined summary.  The terrain, collision setup and success criteria are
# identical across tiers; only the constant command speed differs, so the tiers
# stay comparable (assessment 4.3).
set -uo pipefail

CKPT="${1:?usage: run_stair_evaluation.sh <CHECKPOINT> [OUT_DIR]}"
OUT_DIR="${2:-results}"
TRIALS="${TRIALS:-5}"
SEED="${SEED:-0}"

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$OUT_DIR"

echo "checkpoint : $CKPT"
echo "trials     : $TRIALS"
echo "seed base  : $SEED"
echo "out dir    : $OUT_DIR"
echo

declare -A TASKS=(
  ["0.4"]="Quadruped-3320-Stairs"
  ["0.6"]="Quadruped-3320-Stairs-0.6"
  ["0.8"]="Quadruped-3320-Stairs-0.8"
)

fail=0
for speed in 0.4 0.6 0.8; do
  task="${TASKS[$speed]}"
  csv="$OUT_DIR/results_${speed}.csv"
  echo "=============================================================="
  echo "tier $speed m/s  ->  $task"
  echo "=============================================================="
  python scripts/eval_stairs.py "$task" \
    --checkpoint "$CKPT" \
    --trials "$TRIALS" \
    --out "$csv" \
    --seed "$SEED"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[warn] tier $speed did not meet the pass criterion (exit $rc)"
    fail=1
  fi
  echo
done

echo "=============================================================="
echo "summary"
echo "=============================================================="
python - "$OUT_DIR" "$TRIALS" <<'PY'
import csv, sys
from pathlib import Path

out_dir, trials = Path(sys.argv[1]), int(sys.argv[2])
print(f"{'speed':>6} {'trials':>7} {'success':>8} {'pass(>=4/5)':>12}   file")
for speed in ("0.4", "0.6", "0.8"):
    p = out_dir / f"results_{speed}.csv"
    if not p.exists():
        print(f"{speed:>6} {'-':>7} {'-':>8} {'-':>12}   (missing)")
        continue
    rows = list(csv.DictReader(p.open()))
    ok = sum(1 for r in rows if str(r.get("success", "0")).strip() in ("1", "True", "true"))
    verdict = "YES" if ok >= 4 else "no"
    print(f"{speed:>6} {len(rows):>7} {ok:>8} {verdict:>12}   {p}")
PY

if [ "$fail" -ne 0 ]; then
  echo
  echo "NOTE: at least one tier failed the >=4/5 criterion.  Keep the failing"
  echo "      results.csv files -- assessment 4.3 requires failed runs to be kept."
fi
exit 0
