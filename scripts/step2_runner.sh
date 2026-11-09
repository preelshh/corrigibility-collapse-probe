#!/bin/bash
# Waits for the currently-running overnight_runner.sh (sigma-grid + Part B +
# analysis + samples + summary) to finish, then runs the newly-approved
# Step 2 (apply_risk dose sweep) and regenerates the final reports so they
# include it. Same budget/stop conditions as the overnight run: n=30/cell,
# only floor models, per-cell exception handling, everything logged.
set -u
cd "$(dirname "$0")"
export ADAPTION_API_KEY=<SET-VIA-ENV-OR-KEYCHAIN-NOT-HARDCODED>
MODELS="gemma-4-31b-it qwen3.8-27b glm-5.2 deepseek-v4-flash"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a results/overnight.log
}

while kill -0 21634 2>/dev/null; do sleep 5; done
log "=== overnight_runner.sh (PID 21634) finished — starting Step 2 ==="

log "STEP 2: running apply_risk dose sweep (confirmed text, floor models only)"
python3.12 probe.py --models $MODELS --step2 --calibrate-n 30 --agent-workers 4 \
    --out probe_results.jsonl >> results/step2_log.txt 2>&1
log "Step 2 exited with code $? — see results/step2_log.txt"

log "Regenerating summary + morning summary to include Step 2"
python3.12 write_summary.py >> results/summary_log.txt 2>&1
python3.12 write_morning_summary.py >> results/summary_log.txt 2>&1
log "Step 2 pass complete."
