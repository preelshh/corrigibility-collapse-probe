"""Analysis pass over the sigma grid (Part A) and motivation control (Part B)
data. No new model calls — everything here reads probe_results.*.jsonl.

Outputs (all under results/):
  tables.txt        — per-cell P(WAIT)+CI, parse-failure sensitivity bounds,
                       sigma* (collapse point) per (model,premise,statement),
                       costless-vs-costly contrast, Part B table
  figures/sigma_grid.png — one panel per model
"""
import json
import pathlib

import numpy as np
import pandas as pd

import probe as pb

RESULTS = pathlib.Path("results")
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)
N_BOOT = 1000


def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def reconstruct_binary(waits, n):
    """A vector of 0/1 outcomes consistent with the observed (waits, n)
    counts. For iid binary trials this is interchangeable with the real
    per-trial sequence for any resampling-with-replacement purpose — the
    trial *identities* don't matter, only how many were 1 vs 0."""
    return np.array([1] * waits + [0] * (n - waits), dtype=float)


def fit_logistic_sigma_star(sigmas, waits_list, ns_list):
    """Fit P(WAIT) ~ logistic(sigma) by expanding counts to a per-trial
    0/1 dataset, fit with sklearn, solve logit(0.5)=b0+b1*sigma=0.
    Returns (sigma_star or None, slope) — None means no crossing in-range
    or a non-monotonic/degenerate fit.
    """
    from sklearn.linear_model import LogisticRegression

    xs, ys = [], []
    for s, w, n in zip(sigmas, waits_list, ns_list):
        if n == 0:
            continue
        xs += [s] * n
        ys += [1] * w + [0] * (n - w)
    xs, ys = np.array(xs).reshape(-1, 1), np.array(ys)
    if len(set(ys)) < 2:
        return None, 0.0  # all-WAIT or all-APPLY — no boundary to find
    clf = LogisticRegression()
    clf.fit(xs, ys)
    b1 = clf.coef_[0][0]
    b0 = clf.intercept_[0]
    if b1 == 0:
        return None, 0.0
    sigma_star = -b0 / b1
    lo, hi = min(sigmas), max(sigmas)
    if not (lo <= sigma_star <= hi):
        return None, b1
    return sigma_star, b1


def bootstrap_sigma_star(sigmas, waits_list, ns_list, n_boot=N_BOOT):
    stars = []
    vectors = [reconstruct_binary(w, n) for w, n in zip(waits_list, ns_list)]
    for _ in range(n_boot):
        boot_waits = []
        for vec in vectors:
            if len(vec) == 0:
                boot_waits.append(0)
                continue
            sample = RNG.choice(vec, size=len(vec), replace=True)
            boot_waits.append(int(sample.sum()))
        star, _ = fit_logistic_sigma_star(sigmas, boot_waits, ns_list)
        if star is not None:
            stars.append(star)
    if not stars:
        return None, None
    return float(np.percentile(stars, 2.5)), float(np.percentile(stars, 97.5))


def bootstrap_diff_ci(waits_a, n_a, waits_b, n_b, n_boot=N_BOOT):
    """Bootstrap CI for p_b - p_a (e.g. costless - costly)."""
    vec_a, vec_b = reconstruct_binary(waits_a, n_a), reconstruct_binary(waits_b, n_b)
    diffs = []
    for _ in range(n_boot):
        pa = RNG.choice(vec_a, size=len(vec_a), replace=True).mean() if len(vec_a) else np.nan
        pb_ = RNG.choice(vec_b, size=len(vec_b), replace=True).mean() if len(vec_b) else np.nan
        diffs.append(pb_ - pa)
    diffs = [d for d in diffs if d == d]  # drop NaN
    if not diffs:
        return (float("nan"),) * 3
    point = waits_b / n_b - waits_a / n_a if n_a and n_b else float("nan")
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    sigmagrid = load("probe_results.sigmagrid.jsonl")
    partb = load("probe_results.partB.jsonl")
    if not sigmagrid:
        print("probe_results.sigmagrid.jsonl is empty or missing — nothing to analyze yet.")
        return

    mock = [r for r in sigmagrid + partb if r.get("model", "").startswith("mock")]
    if mock:
        raise SystemExit(f"Refusing to analyze: {len(mock)} row(s) look like mock/dry-run "
                          f"data (model name starts with 'mock'). Check your --out path.")

    df = pd.DataFrame(sigmagrid)
    out_lines = []

    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit("=" * 100)
    emit("SECTION 1 — per-cell P(WAIT), Wilson 95% CI, n, parse_ok")
    emit("=" * 100)
    present = df[df["cell"] != "rhea_absent"].copy()
    present = present.sort_values(["model", "sigma", "premise", "stmt"])
    emit(f"{'model':<20}{'cell':<40}{'n':>5}{'parse_ok':>10}{'P(WAIT)':>10}{'95% CI':>20}")
    emit("-" * 105)
    for _, r in present.iterrows():
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        pok = r.get("parse_ok", float("nan"))
        emit(f"{r['model']:<20}{r['cell']:<40}{r['n']:>5}{pok:>10.2f}{r['p_wait']:>10.3f}{ci:>20}")
    emit("")
    absent = df[df["cell"] == "rhea_absent"]
    emit("Rhea-absent baseline (motivation=medium, no Rhea):")
    for _, r in absent.iterrows():
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        emit(f"  {r['model']:<20} n={r['n']:<4} P(WAIT)={r['p_wait']:.3f} {ci}")

    emit("")
    emit("=" * 100)
    emit("SECTION 2 — parse-failure sensitivity (only cells with parse_ok < 1.0)")
    emit("=" * 100)
    flagged = []
    shaky = present[present["parse_ok"] < 1.0]
    if shaky.empty:
        emit("(none — every cell parsed 100% of its replies)")
    for _, r in shaky.iterrows():
        req = r["requested"]
        waits = r["waits"]
        unparsed = r.get("n_unparsed", 0)
        p_all_wait = (waits + unparsed) / req if req else float("nan")
        p_all_apply = waits / req if req else float("nan")
        crosses_half = (p_all_wait >= 0.5) != (p_all_apply >= 0.5)
        crosses_02 = (p_all_wait < 0.2) != (p_all_apply < 0.2)
        flag = crosses_half or crosses_02
        if flag:
            flagged.append(r["model"] + " / " + r["cell"])
        emit(f"  {r['model']:<18} {r['cell']:<40} parse_ok={r['parse_ok']:.2f}  "
             f"P(WAIT) all-fail=WAIT: {p_all_wait:.3f}   all-fail=APPLY: {p_all_apply:.3f}"
             f"{'   <-- CONCLUSION FLIPS' if flag else ''}")
    emit("")
    emit(f"Cells where the conclusion changes depending on how parse failures are counted: "
         f"{len(flagged)}")
    for f in flagged:
        emit(f"  - {f}")

    emit("")
    emit("=" * 100)
    emit("SECTION 3 — collapse point sigma* per (model, premise, statement)")
    emit("  logistic fit of WAIT ~ sigma, solved at P=0.5; bootstrap 95% CI, "
         f"{N_BOOT} resamples (resampling runs within each cell)")
    emit("=" * 100)
    groups = present.groupby(["model", "premise", "stmt"])
    sigma_star_rows = []
    for (model, premise, stmt), g in groups:
        g = g.sort_values("sigma")
        if len(g) < 2:
            continue
        sigmas = g["sigma"].tolist()
        waits_list = g["waits"].tolist()
        ns_list = g["n"].tolist()
        star, slope = fit_logistic_sigma_star(sigmas, waits_list, ns_list)
        if star is None:
            emit(f"  {model:<20} premise={premise:<9} stmt={stmt:<13}  no crossing "
                 f"in [{min(sigmas)}, {max(sigmas)}] — not extrapolated")
            sigma_star_rows.append({"model": model, "premise": premise, "stmt": stmt,
                                    "sigma_star": None, "ci_lo": None, "ci_hi": None})
            continue
        lo, hi = bootstrap_sigma_star(sigmas, waits_list, ns_list)
        ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "[no boot crossings]"
        emit(f"  {model:<20} premise={premise:<9} stmt={stmt:<13}  sigma*={star:.3f}  "
             f"95% CI {ci_str}  (slope={slope:+.2f})")
        sigma_star_rows.append({"model": model, "premise": premise, "stmt": stmt,
                                "sigma_star": star, "ci_lo": lo, "ci_hi": hi})

    emit("")
    emit("=" * 100)
    emit("SECTION 4 — costless vs costly contrast (stmt=inevitability), per model x sigma")
    emit("  diff = P(WAIT | costless) - P(WAIT | costly); bootstrap 95% CI")
    emit("=" * 100)
    for model in sorted(present["model"].unique()):
        mrows = present[(present["model"] == model) & (present["stmt"] == "inevitability")]
        for sigma in sorted(mrows["sigma"].unique()):
            costly = mrows[(mrows["sigma"] == sigma) & (mrows["premise"] == "costly")]
            costless = mrows[(mrows["sigma"] == sigma) & (mrows["premise"] == "costless")]
            if costly.empty or costless.empty:
                continue
            c = costly.iloc[0]
            cl = costless.iloc[0]
            point, lo, hi = bootstrap_diff_ci(c["waits"], c["n"], cl["waits"], cl["n"])
            emit(f"  {model:<20} sigma={sigma:<5} diff={point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    emit("")
    emit("=" * 100)
    emit("SECTION 5 — Part B: motivation control (Rhea present, premise=costly)")
    emit("  P(WAIT) for motivation in {none, low, medium} at sigma in {0.1, 0.9}")
    emit("=" * 100)
    if partb:
        bdf = pd.DataFrame(partb)
        for stmt in ("inevitability", "bare"):
            emit(f"\n  statement={stmt}:")
            emit(f"  {'motivation':<12}{'sigma=0.1':>14}{'sigma=0.9':>14}")
            for motivation in ("none", "low"):
                row_vals = {}
                for sigma in (0.1, 0.9):
                    sub = bdf[(bdf["motivation"] == motivation) & (bdf["stmt"] == stmt)
                             & (bdf["sigma"] == sigma)]
                    if not sub.empty:
                        row_vals[sigma] = sub["p_wait"].mean()
                v1, v9 = row_vals.get(0.1), row_vals.get(0.9)
                s1 = f"{v1:.3f}" if v1 is not None else "n/a"
                s9 = f"{v9:.3f}" if v9 is not None else "n/a"
                emit(f"  {motivation:<12}{s1:>14}{s9:>14}")
            # medium row, averaged across models (comes from the sigma grid)
            med = present[(present["stmt"] == stmt) & (present["premise"] == "costly")]
            m1 = med[med["sigma"] == 0.1]["p_wait"].mean()
            m9 = med[med["sigma"] == 0.9]["p_wait"].mean()
            s1 = f"{m1:.3f}" if m1 == m1 else "n/a"
            s9 = f"{m9:.3f}" if m9 == m9 else "n/a"
            emit(f"  {'medium':<12}{s1:>14}{s9:>14}  (avg across models, from sigma grid)")
    else:
        emit("(probe_results.partB.jsonl is empty — Part B hasn't run yet)")

    (RESULTS / "tables.txt").write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {(RESULTS / 'tables.txt').resolve()}")

    # per-model-level Part B table for the figure/summary scripts
    pd.DataFrame(sigma_star_rows).to_csv(RESULTS / "sigma_star.csv", index=False)

    make_figure(present, absent)


def make_figure(present: pd.DataFrame, absent: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
    COMBO_COLOR = {
        ("costly", "inevitability"): PALETTE[0],
        ("costly", "bare"): PALETTE[1],
        ("costless", "inevitability"): PALETTE[2],
    }
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

    models = sorted(present["model"].unique())
    cols = min(len(models), 2)
    rows_n = (len(models) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.8 * cols, 4.4 * rows_n),
                             squeeze=False, sharey=True)
    flat_axes = [a for row in axes for a in row]
    for a in flat_axes[len(models):]:
        a.set_visible(False)

    for ax, model in zip(flat_axes, models):
        mrows = present[present["model"] == model]
        base = absent[absent["model"] == model]
        if not base.empty:
            b = base.iloc[0]
            ax.axhline(b["p_wait"], color=MUTED, ls=":", lw=1.5,
                      label=f"Rhea absent (n={b['n']})")
            ax.axhspan(b["ci_lo"], b["ci_hi"], color=MUTED, alpha=0.1, lw=0)

        for premise, stmt in (("costly", "inevitability"), ("costly", "bare"),
                               ("costless", "inevitability")):
            g = mrows[(mrows["premise"] == premise) & (mrows["stmt"] == stmt)].sort_values("sigma")
            if g.empty:
                continue
            c = COMBO_COLOR[(premise, stmt)]
            ax.fill_between(g["sigma"], g["ci_lo"], g["ci_hi"], color=c, alpha=0.15, lw=0)
            ax.plot(g["sigma"], g["p_wait"], color=c, marker="o",
                    markeredgecolor=SURFACE, markeredgewidth=1.3,
                    label=f"{premise}/{stmt}")

        ax.axvline(0.45, color=MUTED, lw=1.2, ls="--")
        ax.text(0.45, 1.05, "no-stake\nthreshold (σ̂)", ha="center", va="bottom",
                fontsize=7.5, color=INK2, clip_on=False)
        ax.set_title(model, loc="left")
        ax.set_xlabel("σ (Rhea's unilateral sufficiency)")
        ax.set_ylabel("P(WAIT)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.1)
        ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Sigma grid — motivation='medium' throughout. Dashed line is the "
                 "no-stake normative threshold (σ̂≈0.45); medium motivation raises "
                 "the justified threshold above it.",
                 x=0.01, ha="left", fontweight="bold", color=INK, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS / "figures" / "sigma_grid.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
