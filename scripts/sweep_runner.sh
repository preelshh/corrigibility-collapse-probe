#!/bin/bash
set -e
cd "/Users/priyalsharma/Library/Application Support/Claude/scratch-workspaces/a18956ca-2181-4d8c-9f5e-534c8b5f3953/3e84b6ac-35a5-4775-8d78-39b3a8bc0027/scratch-2026-11-08-613fca"
MODELS=(gemma-4-26b-a4b-it gemma-4-31b-it qwen3.8-27b qwen3.5-397b)
for m in "${MODELS[@]}"; do
  echo "=== $m :: MAIN + C1_BASELINE, sigma sweep ==="
  python3.12 run_plain.py --model "$m" --sigmas 0.1 0.3 0.5 0.7 0.9 \
    --conditions MAIN C1_BASELINE --reps 10 \
    --run-workers 2 --agent-workers 6 --out results.jsonl
  echo "=== $m :: C2/C3/C4 interventions ==="
  python3.12 run_plain.py --model "$m" --sigmas 0.5 0.7 \
    --conditions C2_COSTLESS C3_ROLLBACK C4_QUORUM --reps 10 \
    --run-workers 2 --agent-workers 6 --out results.jsonl
  echo "=== $m :: statement variants (bare, coherence) at sigma=0.5 ==="
  python3.12 run_plain.py --model "$m" --sigmas 0.5 --conditions MAIN \
    --variant bare --reps 10 --run-workers 2 --agent-workers 6 --out results.jsonl
  python3.12 run_plain.py --model "$m" --sigmas 0.5 --conditions MAIN \
    --variant coherence --reps 10 --run-workers 2 --agent-workers 6 --out results.jsonl
  echo "=== $m DONE ==="
done
echo "=== FULL SWEEP COMPLETE ==="
