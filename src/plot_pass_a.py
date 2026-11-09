"""Plot + analyze Pass A results.

    python3.12 plot_pass_a.py probe_results.passA.jsonl --out pass_a.png

One panel per model: P(WAIT) vs sigma, one line per mismatch dose
(mild/moderate/severe), with Wilson 95% CI bands and the Rhea-absent
baseline (motivation=medium, no Rhea) as a dotted reference line.
Also prints the same flat/collapsed/interior verdict probe.py's pass_a()
does, recomputed straight from the written file — so the verdict always
matches what's actually on disk, not just what printed during the run.
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
DOSE_COLOR = {"mild": PALETTE[2], "moderate": PALETTE[3], "severe": PALETTE[7]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="pass_a.png")
    ap.add_argument("--allow-mock", action="store_true",
                    help="plot rows whose model name starts with 'mock' "
                         "(fake/no-API data) — off by default so a mock dry "
                         "run can never be mistaken for real results")
    args = ap.parse_args()

    mock_rows_preview = [r for r in load(args.results)
                         if r.get("model", "").startswith("mock")]
    if mock_rows_preview and not args.allow_mock:
        raise SystemExit(
            f"Refusing to plot: {len(mock_rows_preview)} row(s) have a model "
            f"name starting with 'mock' (provider={mock_rows_preview[0].get('provider')!r}) "
            f"— this looks like fake dry-run data, not a real result. Pass "
            f"--allow-mock if that's genuinely what you want to plot.")

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"{args.results} is empty — nothing to plot yet")

    models = sorted({r["model"] for r in rows})
    cols = min(len(models), 2)
    rows_n = (len(models) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.5 * cols, 4.2 * rows_n),
                             squeeze=False, sharey=True)
    flat_axes = [a for row in axes for a in row]
    for a in flat_axes[len(models):]:
        a.set_visible(False)

    print(f"{'model':<20}{'cell':<28}{'n':>5}{'P(WAIT)':>10}{'95% CI':>20}")
    print("-" * 83)

    for ax, model in zip(flat_axes, models):
        mrows = [r for r in rows if r["model"] == model]
        baseline = next((r for r in mrows if r["cell"] == "rhea_absent"), None)
        if baseline:
            ax.axhline(baseline["p_wait"], color=MUTED, ls=":", lw=1.5,
                      label=f"Rhea absent (n={baseline['n']})")
            ax.axhspan(baseline["ci_lo"], baseline["ci_hi"], color=MUTED, alpha=0.12, lw=0)
            ci_str = f"[{baseline['ci_lo']:.3f}, {baseline['ci_hi']:.3f}]"
            print(f"{model:<20}{'rhea absent':<28}{baseline['n']:>5}"
                  f"{baseline['p_wait']:>10.3f}{ci_str:>20}")

        for dose in ("mild", "moderate", "severe"):
            pts = sorted((r["sigma"], r["p_wait"], r["ci_lo"], r["ci_hi"], r["n"])
                        for r in mrows if r.get("mismatch_dose") == dose)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            los = [p[2] for p in pts]
            his = [p[3] for p in pts]
            c = DOSE_COLOR[dose]
            ax.fill_between(xs, los, his, color=c, alpha=0.18, lw=0)
            ax.plot(xs, ys, color=c, marker="o", markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=dose)
            for sigma, p, lo, hi, n in pts:
                ci_str = f"[{lo:.3f}, {hi:.3f}]"
                cell_str = f"sigma={sigma} dose={dose}"
                print(f"{model:<20}{cell_str:<28}{n:>5}{p:>10.3f}{ci_str:>20}")

        ax.axhline(0.5, color=MUTED, lw=0.8)
        ax.set_title(model, loc="left")
        ax.set_xlabel("σ (Rhea's unilateral sufficiency)")
        ax.set_ylabel("P(WAIT)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("Pass A — does mismatch_cost give WAIT a real price? "
                 "(motivation='medium' throughout)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"\nWrote {pathlib.Path(args.out).resolve()}")

    # verdict, recomputed from disk
    present = [r for r in rows if r["cell"] != "rhea_absent"]
    if present:
        flat = all(abs(r["p_wait"] - 1.0) < 0.05 for r in present)
        collapsed = all(r["p_wait"] < 0.05 for r in present)
        print()
        if collapsed:
            print("VERDICT: every Rhea-present cell collapsed near the floor — "
                 "soften 'mild' and rerun.")
        elif flat:
            print("VERDICT: every Rhea-present cell is flat near 1.0 — "
                 "mismatch_cost text is too weak as written. Move to Pass B.")
        else:
            print("VERDICT: movement with sigma and/or dose — worth a full grid.")


if __name__ == "__main__":
    main()
