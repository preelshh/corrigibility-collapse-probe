"""Prepend a MORNING SUMMARY block to results/overnight.log: done /
skipped-needs-approval / broken, with the running tally of API errors
logged during the night. Run last, after everything else.
"""
import json
import pathlib

LOG = pathlib.Path("results/overnight.log")


def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def cell_count(path):
    return len(load(path))


def main():
    sigmagrid = load("probe_results.sigmagrid.jsonl")
    partb = load("probe_results.partb.jsonl") or load("probe_results.partB.jsonl")
    present = [r for r in sigmagrid if r.get("cell") != "rhea_absent"]
    expected_sigma_cells = 5 * 3 * 4  # 5 sigmas x 3 (premise,stmt) combos x 4 models
    expected_partb_cells = 2 * 2 * 2 * 4  # 2 motivations x 2 sigmas x 2 stmts x 4 models

    skipped = []
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            if "skipped" in line.lower() or "needs attention" in line.lower() or "NEEDS APPROVAL" in line:
                skipped.append(line)

    done = []
    done.append(f"Sigma grid (Part A): {len(present)}/{expected_sigma_cells} cells written "
               f"to probe_results.sigmagrid.jsonl")
    done.append(f"Motivation control (Part B): {len(partb)}/{expected_partb_cells} cells "
               f"written to probe_results.partB.jsonl")
    tables = pathlib.Path("results/tables.txt")
    done.append(f"Analysis tables: {'written to results/tables.txt' if tables.exists() else 'MISSING'}")
    fig = pathlib.Path("results/figures/sigma_grid.png")
    done.append(f"Figure: {'written to results/figures/sigma_grid.png' if fig.exists() else 'MISSING'}")
    samples = pathlib.Path("results/sample_replies.md")
    done.append(f"Sample replies: {'written to results/sample_replies.md' if samples.exists() else 'MISSING'}")
    summ = pathlib.Path("results/summary.md")
    done.append(f"Summary: {'written to results/summary.md' if summ.exists() else 'MISSING'}")

    step2 = load("probe_results.step2.jsonl")
    needs_approval = []
    if step2:
        floor = sorted({r["model"] for r in step2})
        done.append(f"Step 2 (apply_risk dose, CONFIRMED text): {len(step2)} cells for "
                   f"floor model(s) {floor}, written to probe_results.step2.jsonl")
    else:
        needs_approval.append(
            "Step 2 (apply_risk 3-dose sweep) — approved and the text is confirmed in "
            "scenario.py (APPLY_RISK_TEXT), but probe_results.step2.jsonl is still empty. "
            "Either no model was at the floor, or it hasn't run/finished yet.")

    broken = []
    for r in present:
        if r.get("parse_ok", 1.0) < 0.8:
            broken.append(f"{r['model']} / {r['cell']}: parse_ok={r['parse_ok']:.2f}")
        if r.get("n_api_errors", 0) > 0:
            broken.append(f"{r['model']} / {r['cell']}: {r['n_api_errors']} API error(s) "
                          f"during collection — {r.get('api_errors', [])}")

    block = ["=" * 70, "MORNING SUMMARY", "=" * 70, ""]
    block.append("DONE:")
    for d in done:
        block.append(f"  - {d}")
    block.append("")
    block.append("NEEDS APPROVAL:")
    for n in needs_approval:
        block.append(f"  - {n}")
    block.append("")
    block.append("BROKEN / FLAGGED CELLS (parse_ok<0.8 or API errors during collection):")
    if broken:
        for b in broken:
            block.append(f"  - {b}")
    else:
        block.append("  (none)")
    block.append("")
    block.append(f"Progress-log lines mentioning a skip/stop during the night: {len(skipped)}")
    for s in skipped[:15]:
        block.append(f"  {s}")
    block.append("=" * 70)
    block.append("")

    existing = LOG.read_text() if LOG.exists() else ""
    LOG.write_text("\n".join(block) + "\n" + existing)
    print("Prepended MORNING SUMMARY to results/overnight.log")


if __name__ == "__main__":
    main()
