#!/bin/bash
# Overnight unattended runner for the probe experiment.
# Health-checks the Adaption API with backoff until it recovers (cap: 6h),
# then resumes sigma-grid + Part B (resumable), then runs the analysis.
# Every step is logged to results/overnight.log with a MORNING SUMMARY
# written/updated at the top once the priority list is done or time runs out.
set -u
cd "$(dirname "$0")"
KEY_A=<SET-VIA-ENV-OR-KEYCHAIN-NOT-HARDCODED>
KEY_B=<SET-VIA-ENV-OR-KEYCHAIN-NOT-HARDCODED>
mkdir -p results

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a results/overnight.log
}

log "=== overnight run starting ==="

health_check() {
    ADAPTION_API_KEY="$1" python3.12 -c "
from adaption import Adaption
import sys
client = Adaption(max_retries=0, timeout=20)
try:
    resp = client.responses.create(model='gemma-4-31b-it', instructions='',
        input=[{'role':'user','content':'Say OK.'}], temperature=0.7,
        max_output_tokens=32, tools=[], store=False)
    sys.exit(0 if resp.output_text else 1)
except Exception:
    sys.exit(1)
" > /dev/null 2>&1
}

# --- health check loop: wait for the Adaption backend to stop 504ing.
# Tries both keys each round in case the outage turns out to be per-account
# rather than platform-wide (a bare 504 from the gateway suggests the latter,
# but it's cheap to check).
DEADLINE=$(( $(date +%s) + 6*3600 ))
BACKOFF=30
HEALTHY=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if health_check "$KEY_A"; then
        export ADAPTION_API_KEY="$KEY_A"
        HEALTHY=1
        log "API health check passed on the primary key — resuming data collection"
        break
    fi
    if health_check "$KEY_B"; then
        export ADAPTION_API_KEY="$KEY_B"
        HEALTHY=1
        log "Primary key still 504ing, but the second key worked — using it. " \
            "This means the outage was per-account, not platform-wide."
        break
    fi
    log "API still unhealthy on both keys (504/outage) — backing off ${BACKOFF}s before next check"
    sleep "$BACKOFF"
    BACKOFF=$(( BACKOFF < 300 ? BACKOFF * 2 : 300 ))
done

if [ "$HEALTHY" -ne 1 ]; then
    log "API never recovered within the 6h window on either key — stopping. Needs attention in the morning."
    exit 1
fi

MODELS="gemma-4-31b-it qwen3.8-27b glm-5.2 deepseek-v4-flash"

log "PRIORITY 1a: resuming sigma-grid (Part A)"
python3.12 probe.py --models $MODELS --sigma-grid --calibrate-n 30 --agent-workers 4 \
    --out probe_results.jsonl >> results/sigmagrid_log.txt 2>&1
log "sigma-grid resume exited with code $? — see results/sigmagrid_log.txt"

log "PRIORITY 1b: resuming motivation-control (Part B)"
python3.12 probe.py --models $MODELS --motivation-control --calibrate-n 30 --agent-workers 4 \
    --out probe_results.jsonl >> results/partb_log.txt 2>&1
log "Part B resume exited with code $? — see results/partb_log.txt"

log "PRIORITY 1c: running analyze_probe.py (tables + figure)"
python3.12 analyze_probe.py >> results/analyze_log.txt 2>&1
log "analyze_probe.py exited with code $? — see results/analyze_log.txt"

log "PRIORITY 2: apply_risk (Step 2) dose text is still DRAFT/unconfirmed from the prior session " \
    "(scenario.py's _APPLY_RISK is a single placeholder, not the 3-level mild/moderate/severe dose " \
    "the audit spec calls for, and I don't have the user's verbatim text for it). Per the decision " \
    "rule 'do not invent new conditions or change existing prompts,' I am NOT fabricating that text " \
    "and NOT running Step 2. Logged as NEEDS APPROVAL."

log "PRIORITY 3: pulling sample replies"
python3.12 pull_sample_replies.py >> results/sample_replies_log.txt 2>&1
log "pull_sample_replies.py exited with code $? — see results/sample_replies_log.txt"

log "PRIORITY 4: drafting results summary"
python3.12 write_summary.py >> results/summary_log.txt 2>&1
log "write_summary.py exited with code $? — see results/summary_log.txt"

log "=== overnight run finished — writing MORNING SUMMARY ==="
python3.12 write_morning_summary.py >> results/summary_log.txt 2>&1
log "done."
