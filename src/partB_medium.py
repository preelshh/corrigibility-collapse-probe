"""Fill the missing motivation=medium cells in Part B (motivation control).

probe_results.partB.jsonl has motivation={none,low} x sigma{0.1,0.9} x
stmt{inevitability,bare}, Rhea present. The main sigma grid runs at
motivation=medium throughout, but Part B never ran a medium row, so there
was no cell holding Rhea/sigma/stmt fixed while sweeping motivation
none->low->medium. This adds exactly that: motivation=medium,
sigma in {0.1, 0.9}, stmt="inevitability" only (per the user's scoped
request), same 4 models, n=30 each, appended to probe_results.partB.jsonl
with the same schema and cell-naming convention as the existing rows
("motivation=medium sigma=S stmt=inevitability").

Built the same way probe.py's motivation_control()/sigma_grid() build their
cells: scn.Condition(committed_present=True, inconsistency_costly=True,
mismatch_dose="off", motivation=...) + isolated_pwait(). Reuses probe.py's
isolated_pwait/wilson_ci and run_plain's model backends untouched.

Usage:
    python3.12 partB_medium.py
"""

from __future__ import annotations

import itertools
import json
import pathlib
import time

import probe as pb
import run_plain as rp
import scenario as scn

MODELS = ["glm-5.2", "deepseek-v4-flash", "qwen3.8-27b", "gemma-4-31b-it"]
N = 30
AGENT_WORKERS = 4
TEMPERATURE = 0.7
MAX_TOKENS = 3072
SIGMAS = [0.1, 0.9]
STMT = "inevitability"
MOTIVATION = "medium"
OUT = pathlib.Path("probe_results.partB.jsonl")


def main():
    done = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                rec = json.loads(line)
                done.add((rec["model"], rec["cell"]))
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"\nPart B — filling motivation=medium, Rhea present, stmt={STMT}. "
          f"n={N} per cell.\n", flush=True)
    print(f"{'model':<20}{'cell':<40}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}",
          flush=True)
    print("-" * 99, flush=True)

    t_start = time.time()
    fh = OUT.open("a")
    for model_label in MODELS:
        limiter = rp.RateLimiter(None)
        model = rp.AdaptionModel(model_label, TEMPERATURE, MAX_TOKENS, limiter)
        for sigma in SIGMAS:
            cell = f"motivation={MOTIVATION} sigma={sigma} stmt={STMT}"
            if (model_label, cell) in done:
                print(f"(skipping {model_label} / {cell} — already in {OUT.name})",
                      flush=True)
                continue
            t0 = time.time()
            cond = scn.Condition(
                f"PARTB_{model_label}_{MOTIVATION}_s{sigma}_{STMT}",
                committed_present=True, inconsistency_costly=True,
                quorum_gate=False, motivation=MOTIVATION, mismatch_dose="off")
            line = scn.statement_for_round(STMT, 0)
            r = pb.isolated_pwait(model, model_label, cond, sigma, N,
                                  AGENT_WORKERS, line, STMT, "adaption")
            r.update(cell=cell, sigma=sigma, motivation=MOTIVATION, stmt=STMT)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"{model_label:<20}{cell:<40}{r['n']:>5}{r['waits']:>6}"
                  f"{r['p_wait']:>10.3f}{ci_str:>18}  ({time.time()-t0:.0f}s)",
                  flush=True)
    fh.close()
    print(f"\nWrote {OUT.resolve()}  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
