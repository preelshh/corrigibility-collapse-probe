"""Collect raw reply text for results/sample_replies.md:
  - 5 replies from every interior cell (0.1 < P(WAIT) < 0.9) across all
    jsonl result files (no new API calls — these are already stored).
  - 5 replies from a gemma floor cell and 5 from gemma's costless sigma=0.1
    cell. These two specific gemma cells are freshly re-queried at n=5
    (tiny, well under the 30-cell cap) because the OLD isolated_pwait only
    stored sample text for interior cells, so no raw text exists yet for
    gemma's floor cells in the already-written jsonl files.
"""
import json
import pathlib

import scenario as scn
import run_plain as rp
from probe import isolated_pwait

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def main():
    lines = ["# Sample replies\n"]

    lines.append("## Interior cells (0.1 < P(WAIT) < 0.9), from stored data\n")
    all_rows = (load("probe_results.sigmagrid.jsonl") + load("probe_results.partB.jsonl")
               + load("probe_results.passA.jsonl") + load("probe_results.step1.jsonl"))
    seen = set()
    any_interior = False
    for r in all_rows:
        key = (r.get("model"), r.get("cell"))
        if key in seen:
            continue
        p = r.get("p_wait")
        if p is None or not (0.1 < p < 0.9):
            continue
        texts = r.get("sample_texts") or [{"choice": "?", "text": t}
                                          for t in r.get("sample_reasoning", [])]
        if not texts:
            continue
        seen.add(key)
        any_interior = True
        lines.append(f"\n### {r['model']} / {r['cell']}  (P(WAIT)={p:.3f}, n={r.get('n')})\n")
        for t in texts[:5]:
            lines.append(f"- **[{t['choice']}]** {t['text']}")
    if not any_interior:
        lines.append("\n(none found)\n")

    lines.append("\n\n## gemma-4-31b-it — fresh n=5 pulls for qualitative illustration\n")
    import run_plain as rp
    limiter = rp.RateLimiter(None)
    model = rp.AdaptionModel("gemma-4-31b-it", 0.7, 3072, limiter)

    def pull(label, cond, sigma, committed_line, tag):
        r = isolated_pwait(model, "gemma-4-31b-it", cond, sigma, 5, 4,
                           committed_line, tag, "adaption")
        lines.append(f"\n### {label}  (P(WAIT)={r['p_wait']:.3f}, n={r['n']})\n")
        for t in r["sample_texts"]:
            lines.append(f"- **[{t['choice']}]** {t['text']}")
        return r

    floor_cond = scn.Condition("SAMPLE_GEMMA_FLOOR", committed_present=True,
                               inconsistency_costly=True, quorum_gate=False,
                               motivation="medium", mismatch_dose="off")
    pull("Floor cell: sigma=0.1, premise=costly, stmt=inevitability", floor_cond, 0.1,
         scn.statement_for_round("inevitability", 0), "inevitability")

    costless_cond = scn.Condition("SAMPLE_GEMMA_COSTLESS", committed_present=True,
                                  inconsistency_costly=False, quorum_gate=False,
                                  motivation="medium", mismatch_dose="off")
    pull("Costless cell: sigma=0.1, premise=costless, stmt=inevitability", costless_cond, 0.1,
         scn.statement_for_round("inevitability", 0), "inevitability")

    out = RESULTS / "sample_replies.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
