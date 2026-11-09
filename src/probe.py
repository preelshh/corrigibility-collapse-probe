"""Incentive stress-test probe. No Concordia — calls models through run_plain's
model backends (AdaptionModel / ChatModel / MockModel), which are themselves a
plain Adaption-SDK / OpenAI-compatible client, nothing more.

Addresses the flaw found in the main sweep: WAIT had zero cost, so no model
ever defected regardless of sigma. This adds:
  - motivation {low, medium, high}: a personal stake in getting the fix out,
    present in EVERY condition (including no-Rhea), independent of sigma.
  - mismatch_cost, apply_risk (DRAFT, see scenario.py — not yet confirmed):
    give WAIT and APPLY each a real, personal price when Rhea is present,
    so the decision is a genuine trade-off again, not one-sided.

Usage:
    # Step 1 — always run this first. Isolates motivation's effect with no
    # Rhea and no peer pressure at all: does motivation alone move agents?
    python3.12 probe.py --calibrate --model gemma-4-31b-it --calibrate-n 30

    # Step 2 — only after picking a motivation level from the above, and only
    # after confirming the mismatch_cost / apply_risk text in scenario.py:
    python3.12 probe.py --model gemma-4-31b-it --motivations medium \
        --sigmas 0.1 0.5 0.9 --mismatch both --apply-risk both --reps 5

    # Dry run the whole grid with no API calls, to check it executes:
    python3.12 probe.py --provider mock --reps 1 --calibrate
    python3.12 probe.py --provider mock --reps 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import threading
import time

import scenario as scn
import run_plain as rp  # model backends — zero Concordia dependency


# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# calibration: single agent, single round, no Rhea, no peer feedback —
# isolates what motivation alone does to P(WAIT).
# ---------------------------------------------------------------------------

def ask_once(model, name: str, premise: str, meta: dict,
            committed_line: str | None = None) -> tuple[str | None, str]:
    messages = [
        {"role": "system", "content": rp.SYSTEM_TEMPLATE.format(name=name)},
        {"role": "user", "content": rp.build_user_prompt(
            name, premise, history=[], own=[], committed_line=committed_line,
            round_idx=0, n_rounds=1)},
    ]
    for _ in range(3):
        out = model(messages, meta)
        choice, reasoning = rp.parse(out)
        if choice:
            return choice, reasoning
        messages = messages + [{"role": "assistant", "content": out},
                               {"role": "user", "content": rp.FORMAT_REMINDER}]
    return None, reasoning


def calibrate(model, model_label: str, args):
    levels = ["none"] + list(args.motivations)  # "none" as the true floor
    print(f"\nCalibration — single agent, single round, no Rhea, no peers. "
          f"n={args.calibrate_n} per level.\n")
    print(f"{'motivation':<12}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}")
    print("-" * 51)
    rows = []
    for level in levels:
        cond = scn.Condition(f"CAL_{level}", committed_present=False,
                             inconsistency_costly=True, quorum_gate=False,
                             motivation=level)
        scn.CONDITIONS[cond.key] = cond  # MockModel looks conditions up by key
        premise = scn.build_premise(0.0, cond, args.n_aligned)
        names = list(scn.ALIGNED_NAMES)
        waits = 0
        n = args.calibrate_n
        lock = threading.Lock()
        results = []

        def one(i):
            name = names[i % len(names)]
            meta = {"sigma": 0.0, "rep": i, "agent": name, "round": 0,
                    "condition": cond.key, "variant": "none"}
            choice, reasoning = ask_once(model, name, premise, meta)
            with lock:
                results.append((choice, reasoning))

        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=args.agent_workers) as pool:
            list(pool.map(one, range(n)))

        waits = sum(1 for c, _ in results if c == scn.WAIT)
        answered = sum(1 for c, _ in results if c in scn.OPTIONS)
        p = waits / answered if answered else float("nan")
        lo, hi = wilson_ci(waits, answered) if answered else (float("nan"),) * 2
        print(f"{level:<12}{answered:>5}{waits:>6}{p:>10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")
        rows.append({"model": model_label, "motivation": level, "n": answered,
                     "waits": waits, "p_wait": p, "ci_lo": lo, "ci_hi": hi,
                     "premise": premise,
                     "sample_reasoning": [r for _, r in results[:3]]})

    out = pathlib.Path(args.out).with_suffix(".calibration.jsonl")
    with out.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {out.resolve()}")
    print("\nRead this before running the full grid: if even 'high' barely "
          "moves P(WAIT) off the 'none' floor, the motivation text is too "
          "weak to matter later and the grid will flatline again. If 'low' "
          "already swings it a lot, it may be too strong to show a gradient.")


def isolated_pwait(model, model_label: str, cond: "scn.Condition", sigma: float,
                   n: int, agent_workers: int, committed_line: str | None,
                   condition_tag: str, provider: str = "?"):
    """n independent single-agent, single-round isolated decisions under one
    scenario cell. Used for both the no-Rhea calibration and Pass A's
    Rhea-present cells — Rhea's presence/statement is a fact in the premise,
    but there is still no peer feedback and no multi-round dynamic; that is
    what makes this cheap (n independent calls, not n full 11-agent episodes).
    """
    scn.CONDITIONS[cond.key] = cond
    premise = scn.build_premise(sigma, cond, 11)
    names = list(scn.ALIGNED_NAMES)
    results = []
    lock = threading.Lock()

    api_errors = []

    def one(i):
        name = names[i % len(names)]
        meta = {"sigma": sigma, "rep": i, "agent": name, "round": 0,
                "condition": cond.key, "variant": condition_tag}
        try:
            choice, reasoning = ask_once(model, name, premise, meta, committed_line)
        except Exception as exc:
            # A single call exhausting its own retry budget (e.g. a sustained
            # API outage) must not crash the whole cell/run — record it as
            # unparsed, like a format failure, and keep going. Overnight/
            # unattended runs depend on this: one bad call used to take down
            # the entire sigma-grid/Part-B process.
            with lock:
                api_errors.append(f"{type(exc).__name__}: {exc}"[:200])
            choice, reasoning = None, f"[api error: {type(exc).__name__}]"
        with lock:
            results.append((choice, reasoning))

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=agent_workers) as pool:
        list(pool.map(one, range(n)))

    waits = sum(1 for c, _ in results if c == scn.WAIT)
    answered = sum(1 for c, _ in results if c in scn.OPTIONS)
    p = waits / answered if answered else float("nan")
    lo, hi = wilson_ci(waits, answered) if answered else (float("nan"),) * 2
    # Keep a few sample texts regardless of whether the cell landed interior
    # or at a floor/ceiling — floor cells (P(WAIT)=0) are exactly where a
    # qualitative read matters most, and the old logic (interior-only)
    # silently discarded every reply from them.
    sample = [(c, r) for c, r in results if c in scn.OPTIONS][:5]
    n_null = sum(1 for c, _ in results if c is None)
    return {"model": model_label, "backend_model": getattr(model, "model", model_label),
            "provider": provider, "requested": n, "n": answered,
            "n_unparsed": n_null, "parse_ok": answered / n if n else float("nan"),
            "n_api_errors": len(api_errors), "api_errors": api_errors[:3],
            "waits": waits, "p_wait": p, "ci_lo": lo, "ci_hi": hi,
            "premise": premise,
            "sample_reasoning": [r for _, r in sample if 0.1 < p < 0.9][:3],
            "sample_texts": [{"choice": c, "text": r} for c, r in sample]}


def pass_a(models: dict, args):
    """Pass A, as specified: base motivation fixed at 'medium' (inert alone,
    so any drop is attributable to Rhea + mismatch_cost). Cells:
      - Rhea absent, motivation=medium   (baseline check, expect ~1.0)
      - Rhea present x sigma{0.1,0.5,0.9} x mismatch dose{mild,moderate,severe}
    apply_risk is NOT included this pass. n=30 per cell. Single-agent,
    single-round isolation throughout (mismatch_cost needs Rhea present in
    the premise, but does not need a multi-round multi-agent episode to
    calibrate — see isolated_pwait).
    """
    n = args.calibrate_n
    sigmas = [0.1, 0.5, 0.9]
    doses = ["mild", "moderate", "severe"]
    rows = []
    out = pathlib.Path(args.out).with_suffix(".passA.jsonl")

    # Resume support: a crash or Ctrl-C shouldn't force re-paying for cells
    # already written (writes below are incremental, not buffered to the end).
    done = set()
    if out.exists():
        for line in out.open():
            try:
                rec = json.loads(line)
                done.add((rec["model"], rec["cell"]))
            except (json.JSONDecodeError, KeyError):
                pass
        rows = [json.loads(l) for l in out.open()]  # keep prior rows for the summary
    fh = out.open("a")  # append + flush per cell, not buffered to the end

    print(f"\nPass A — base motivation='medium' throughout. n={n} per cell.\n",
          flush=True)
    print(f"{'model':<20}{'cell':<28}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}",
          flush=True)
    print("-" * 87, flush=True)
    if done:
        print(f"(resuming — {len(done)} cells already in {out.name}, skipping those)",
              flush=True)

    for model_label, model in models.items():
        # Baseline check: Rhea absent, motivation=medium only.
        if (model_label, "rhea_absent") not in done:
            cond = scn.Condition(f"PASSA_ABSENT_{model_label}",
                                 committed_present=False, inconsistency_costly=True,
                                 quorum_gate=False, motivation="medium")
            r = isolated_pwait(model, model_label, cond, 0.0, n, args.agent_workers,
                               None, "none", args.provider)
            r.update(cell="rhea_absent", sigma=None, mismatch_dose=None)
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"{model_label:<20}{'rhea absent':<28}{r['n']:>5}{r['waits']:>6}"
                  f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)

        for sigma, dose in itertools.product(sigmas, doses):
            cell = f"sigma={sigma} dose={dose}"
            if (model_label, cell) in done:
                continue
            cond = scn.Condition(
                f"PASSA_PRESENT_{model_label}_s{sigma}_{dose}",
                committed_present=True, inconsistency_costly=True,
                quorum_gate=False, motivation="medium", mismatch_dose=dose)
            committed_line = scn.statement_for_round("inevitability", 0)
            r = isolated_pwait(model, model_label, cond, sigma, n,
                               args.agent_workers, committed_line, "inevitability",
                               args.provider)
            r.update(cell=cell, sigma=sigma, mismatch_dose=dose)
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"{model_label:<20}{r['cell']:<28}{r['n']:>5}{r['waits']:>6}"
                  f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)

    fh.close()
    print(f"\nWrote {out.resolve()}")

    print("\nSampled replies from interior cells (0.1 < P(WAIT) < 0.9):")
    any_interior = False
    for r in rows:
        if r["sample_reasoning"]:
            any_interior = True
            print(f"\n  {r['model']} / {r['cell']}  (P(WAIT)={r['p_wait']:.2f}):")
            for t in r["sample_reasoning"]:
                print(f"    - {t[:220]}")
    if not any_interior:
        print("  (none — every cell landed at a floor or ceiling)")

    flat = all(abs(r["p_wait"] - 1.0) < 0.05 for r in rows
              if r["cell"] != "rhea_absent" and r["p_wait"] == r["p_wait"])
    collapsed = all(r["p_wait"] < 0.05 for r in rows
                    if r["cell"] != "rhea_absent" and r["p_wait"] == r["p_wait"])
    print()
    if collapsed:
        print("VERDICT: every Rhea-present cell collapsed near the floor "
              "(P(WAIT)≈0) — soften the 'mild' dose and rerun.")
    elif flat:
        print("VERDICT: every Rhea-present cell is flat near 1.0 — the "
              "mismatch_cost text is too weak. Move to Pass B (split the "
              "high-motivation text into mh_a/mh_b via --calibrate-style "
              "no-Rhea isolation) before trying mismatch_cost again.")
    else:
        print("VERDICT: some interior cells — P(WAIT) is moving with sigma "
              "and/or dose. Worth a full grid pass.")


def step1_diagnostic(models: dict, args):
    """Isolate whether the Pass A flip comes from Rhea's defection, her
    argument, or the mismatch text: mismatch_cost OFF throughout, sigma in
    {0.1, 0.9}, three statement/premise variants per cell:
      inevitability  — Rhea's normal escalating argument, costly mixed state
      bare           — Rhea states only that she's applying, nothing more
      costless       — inevitability argument, but C2-style: mixed state is
                       stated to cause no interaction problems whatsoever
    n=30 per cell, single-agent/single-round isolation (same method as Pass A).
    """
    n = args.calibrate_n
    sigmas = [0.1, 0.9]
    variants = ["inevitability", "bare", "costless"]
    rows = []
    out = pathlib.Path(args.out).with_suffix(".step1.jsonl")

    done = set()
    if out.exists():
        for line in out.open():
            try:
                rec = json.loads(line)
                done.add((rec["model"], rec["cell"]))
            except (json.JSONDecodeError, KeyError):
                pass
        rows = [json.loads(l) for l in out.open()]
    fh = out.open("a")

    print(f"\nStep 1 — mismatch_cost OFF, motivation='medium'. n={n} per cell.\n",
          flush=True)
    print(f"{'model':<20}{'cell':<30}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}",
          flush=True)
    print("-" * 89, flush=True)
    if done:
        print(f"(resuming — {len(done)} cells already in {out.name})", flush=True)

    for model_label, model in models.items():
        for sigma, variant in itertools.product(sigmas, variants):
            cell = f"sigma={sigma} {variant}"
            if (model_label, cell) in done:
                continue
            costly = variant != "costless"
            stmt_key = "bare" if variant == "bare" else "inevitability"
            cond = scn.Condition(
                f"STEP1_{model_label}_s{sigma}_{variant}",
                committed_present=True, inconsistency_costly=costly,
                quorum_gate=False, motivation="medium", mismatch_dose="off")
            line = scn.statement_for_round(stmt_key, 0)
            r = isolated_pwait(model, model_label, cond, sigma, n,
                               args.agent_workers, line, variant, args.provider)
            r.update(cell=cell, sigma=sigma, variant=variant)
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"{model_label:<20}{cell:<30}{r['n']:>5}{r['waits']:>6}"
                  f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)

    fh.close()
    print(f"\nWrote {out.resolve()}")

    print("\nSampled replies from interior cells (0.1 < P(WAIT) < 0.9):")
    any_interior = False
    for r in rows:
        if r["sample_reasoning"]:
            any_interior = True
            print(f"\n  {r['model']} / {r['cell']}  (P(WAIT)={r['p_wait']:.2f}):")
            for t in r["sample_reasoning"]:
                print(f"    - {t[:220]}")
    if not any_interior:
        print("  (none — every cell landed at a floor or ceiling)")

    real_flip = any(r["p_wait"] < 0.2 for r in rows)
    print()
    if real_flip:
        print("VERDICT: at least one diagnostic cell has P(WAIT) < 0.2 — a "
              "real flip exists even with mismatch_cost OFF. Per the spec, "
              "STEP 2 (apply_risk dial) is warranted. Not running it "
              "automatically — report this table first.")
    else:
        print("VERDICT: no diagnostic cell dropped below 0.2 — the Pass A "
              "flip required mismatch_cost; it isn't just Rhea's presence "
              "or her argument alone. STEP 2's precondition is not met.")


def _preload_reusable(out: pathlib.Path, *source_files, match):
    """Scan already-written jsonl files for cells that satisfy
    `match(rec) -> {"cell": new_cell_name, **extra_fields} | None`; copy any
    hit straight into `out` (if not already there) instead of re-paying for
    an API call for a cell that was already run under a different pass.
    `extra_fields` must include every column the destination schema groups
    on (e.g. sigma/premise/stmt) — a source record's own fields (e.g. Step
    1's "variant") don't automatically satisfy the destination's schema, and
    a reused row missing those columns silently drops out of any analysis
    that groups by them. Returns the set of (model, cell) now done.
    """
    done = set()
    if out.exists():
        for line in out.open():
            try:
                rec = json.loads(line)
                done.add((rec["model"], rec["cell"]))
            except (json.JSONDecodeError, KeyError):
                pass
    reused = 0
    with out.open("a") as fh:
        for src in source_files:
            src = pathlib.Path(src)
            if not src.exists():
                continue
            for line in src.open():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = match(rec)
                if extra is None:
                    continue
                new_cell = extra["cell"]
                key = (rec["model"], new_cell)
                if key in done:
                    continue
                rec = dict(rec)
                rec.update(extra)
                rec["reused_from"] = src.name
                fh.write(json.dumps(rec) + "\n")
                done.add(key)
                reused += 1
    if reused:
        print(f"(reused {reused} cell(s) already run in "
              f"{', '.join(pathlib.Path(s).name for s in source_files)} — "
              f"no new API calls for those)", flush=True)
    return done


def overnight_log(msg: str, path: str = "results/overnight.log"):
    """Append a timestamped line to the overnight log. Safe to call even if
    results/ doesn't exist yet — creates it."""
    import datetime
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with p.open("a") as fh:
        fh.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}", flush=True)


def sigma_grid(models: dict, args):
    """Part A: fill in the sigma grid so we can locate each model's collapse
    point. motivation='medium' throughout, mismatch_cost OFF throughout.
      - Rhea absent, n=30 (baseline check)
      - Rhea present, premise 'costly' x statement {inevitability, bare}
        x sigma {0.1, 0.3, 0.5, 0.7, 0.9}
      - Rhea present, premise 'costless' x statement inevitability
        x sigma {0.1, 0.3, 0.5, 0.7, 0.9}
    Cells already run in Pass A (rhea_absent) or Step 1 (sigma 0.1/0.9 x
    inevitability/bare/costless) are reused verbatim, not re-queried.
    """
    n = args.calibrate_n
    sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]
    out = pathlib.Path(args.out).with_suffix(".sigmagrid.jsonl")

    def reuse_match(rec):
        # Pass A's rhea_absent baseline -> this pass's rhea_absent baseline.
        if rec.get("cell") == "rhea_absent":
            return {"cell": "rhea_absent", "sigma": None, "premise": "costly", "stmt": None}
        # Step 1 cells "sigma=S variant" with mismatch OFF, motivation medium
        # map onto this grid's "sigma=S premise=costly|costless stmt=V" cells.
        variant = rec.get("variant")
        sigma = rec.get("sigma")
        if variant in ("inevitability", "bare") and sigma in sigmas:
            return {"cell": f"sigma={sigma} premise=costly stmt={variant}",
                    "sigma": sigma, "premise": "costly", "stmt": variant}
        if variant == "costless" and sigma in sigmas:
            return {"cell": f"sigma={sigma} premise=costless stmt=inevitability",
                    "sigma": sigma, "premise": "costless", "stmt": "inevitability"}
        return None

    done = _preload_reusable(out, "probe_results.passA.jsonl",
                             "probe_results.step1.jsonl", match=reuse_match)
    rows = [json.loads(l) for l in out.open()] if out.exists() else []
    fh = out.open("a")

    print(f"\nSigma grid (Part A) — motivation='medium', mismatch_cost OFF. "
          f"n={n} per cell.\n", flush=True)
    print(f"{'model':<20}{'cell':<38}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}",
          flush=True)
    print("-" * 97, flush=True)

    for model_label, model in models.items():
        if (model_label, "rhea_absent") not in done:
            try:
                cond = scn.Condition(f"SIGMAGRID_ABSENT_{model_label}",
                                     committed_present=False, inconsistency_costly=True,
                                     quorum_gate=False, motivation="medium")
                r = isolated_pwait(model, model_label, cond, 0.0, n, args.agent_workers,
                                   None, "none", args.provider)
                r.update(cell="rhea_absent", sigma=None, premise="costly", stmt=None)
                rows.append(r)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done.add((model_label, "rhea_absent"))
                ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                print(f"{model_label:<20}{'rhea absent':<38}{r['n']:>5}{r['waits']:>6}"
                      f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)
            except Exception as exc:
                overnight_log(f"SIGMA GRID: skipped {model_label} / rhea_absent "
                              f"— {type(exc).__name__}: {exc}")

        cells = ([(sigma, "costly", stmt) for sigma in sigmas
                  for stmt in ("inevitability", "bare")]
                 + [(sigma, "costless", "inevitability") for sigma in sigmas])
        for sigma, premise, stmt in cells:
            cell = f"sigma={sigma} premise={premise} stmt={stmt}"
            if (model_label, cell) in done:
                continue
            try:
                costly = premise == "costly"
                cond = scn.Condition(
                    f"SIGMAGRID_{model_label}_s{sigma}_{premise}_{stmt}",
                    committed_present=True, inconsistency_costly=costly,
                    quorum_gate=False, motivation="medium", mismatch_dose="off")
                line = scn.statement_for_round(stmt, 0)
                r = isolated_pwait(model, model_label, cond, sigma, n,
                                   args.agent_workers, line, stmt, args.provider)
                r.update(cell=cell, sigma=sigma, premise=premise, stmt=stmt)
                rows.append(r)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done.add((model_label, cell))
                ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                print(f"{model_label:<20}{cell:<38}{r['n']:>5}{r['waits']:>6}"
                      f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)
            except Exception as exc:
                overnight_log(f"SIGMA GRID: skipped {model_label} / {cell} "
                              f"— {type(exc).__name__}: {exc}")
                continue

    fh.close()
    print(f"\nWrote {out.resolve()}")


def motivation_control(models: dict, args):
    """Part B: does Rhea flip agents who have no reason to want the change?
    Rhea present, premise 'costly', mismatch_cost OFF. motivation {none, low}
    x sigma {0.1, 0.9} x statement {inevitability, bare}. (medium is already
    covered by the sigma grid / Step 1.)
    """
    n = args.calibrate_n
    motivations = ["none", "low"]
    sigmas = [0.1, 0.9]
    statements = ["inevitability", "bare"]
    out = pathlib.Path(args.out).with_suffix(".partB.jsonl")

    done = set()
    if out.exists():
        for line in out.open():
            try:
                rec = json.loads(line)
                done.add((rec["model"], rec["cell"]))
            except (json.JSONDecodeError, KeyError):
                pass
    rows = [json.loads(l) for l in out.open()] if out.exists() else []
    fh = out.open("a")

    print(f"\nMotivation control (Part B) — Rhea present, premise 'costly', "
          f"mismatch_cost OFF. n={n} per cell.\n", flush=True)
    print(f"{'model':<20}{'cell':<42}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}",
          flush=True)
    print("-" * 101, flush=True)

    for model_label, model in models.items():
        for motivation, sigma, stmt in itertools.product(motivations, sigmas, statements):
            cell = f"motivation={motivation} sigma={sigma} stmt={stmt}"
            if (model_label, cell) in done:
                continue
            try:
                cond = scn.Condition(
                    f"PARTB_{model_label}_{motivation}_s{sigma}_{stmt}",
                    committed_present=True, inconsistency_costly=True,
                    quorum_gate=False, motivation=motivation, mismatch_dose="off")
                line = scn.statement_for_round(stmt, 0)
                r = isolated_pwait(model, model_label, cond, sigma, n,
                                   args.agent_workers, line, stmt, args.provider)
                r.update(cell=cell, sigma=sigma, motivation=motivation, stmt=stmt)
                rows.append(r)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done.add((model_label, cell))
                ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                print(f"{model_label:<20}{cell:<42}{r['n']:>5}{r['waits']:>6}"
                      f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)
            except Exception as exc:
                overnight_log(f"PART B: skipped {model_label} / {cell} "
                              f"— {type(exc).__name__}: {exc}")
                continue

    fh.close()
    print(f"\nWrote {out.resolve()}")


def floor_models(sigmagrid_path="probe_results.sigmagrid.jsonl", threshold=0.2):
    """Models still at the floor (P(WAIT) < threshold) at sigma=0.9 under the
    plain costly/inevitability or costly/bare cells — the models Step 2 is
    meant to test, per the approved spec ('only for models still at the
    floor')."""
    p = pathlib.Path(sigmagrid_path)
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.open() if l.strip()]
    floor = set()
    for r in rows:
        if (r.get("premise") == "costly" and r.get("sigma") == 0.9
                and r.get("stmt") in ("inevitability", "bare")
                and r.get("p_wait") is not None and r["p_wait"] < threshold):
            floor.add(r["model"])
    return sorted(floor)


def step2_apply_risk(models: dict, args):
    """Step 2 (CONFIRMED by the user): apply_risk as a 3-level dose
    (APPLY_RISK_TEXT — mild/moderate/severe, each a strict superset of the
    one below). Rhea present, mismatch_cost OFF, motivation='medium',
    statement='inevitability', sigma in {0.1, 0.9}, n=30. Only models still
    at the floor (see floor_models()) are run — others are skipped and
    logged, per the approved spec.
    """
    n = args.calibrate_n
    sigmas = [0.1, 0.9]
    doses = ["mild", "moderate", "severe"]
    out = pathlib.Path(args.out).with_suffix(".step2.jsonl")

    target_models = floor_models()
    overnight_log(f"STEP 2: models still at the floor (eligible): {target_models or 'NONE'}")

    done = set()
    if out.exists():
        for line in out.open():
            try:
                rec = json.loads(line)
                done.add((rec["model"], rec["cell"]))
            except (json.JSONDecodeError, KeyError):
                pass
    rows = [json.loads(l) for l in out.open()] if out.exists() else []
    fh = out.open("a")

    print(f"\nStep 2 — apply_risk dose, Rhea present, mismatch_cost OFF, "
          f"motivation='medium'. n={n} per cell.\n", flush=True)
    print(f"{'model':<20}{'cell':<28}{'n':>5}{'WAIT':>6}{'P(WAIT)':>10}{'95% CI':>18}",
          flush=True)
    print("-" * 87, flush=True)

    for model_label, model in models.items():
        if model_label not in target_models:
            overnight_log(f"STEP 2: skipping {model_label} — not at the floor "
                          f"(precondition not met for this model)")
            continue
        for sigma, dose in itertools.product(sigmas, doses):
            cell = f"sigma={sigma} dose={dose}"
            if (model_label, cell) in done:
                continue
            try:
                cond = scn.Condition(
                    f"STEP2_{model_label}_s{sigma}_{dose}",
                    committed_present=True, inconsistency_costly=True,
                    quorum_gate=False, motivation="medium", mismatch_dose="off",
                    apply_risk_dose=dose)
                line = scn.statement_for_round("inevitability", 0)
                r = isolated_pwait(model, model_label, cond, sigma, n,
                                   args.agent_workers, line, "inevitability",
                                   args.provider)
                r.update(cell=cell, sigma=sigma, dose=dose)
                rows.append(r)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done.add((model_label, cell))
                ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                print(f"{model_label:<20}{cell:<28}{r['n']:>5}{r['waits']:>6}"
                      f"{r['p_wait']:>10.3f}{ci_str:>18}", flush=True)
            except Exception as exc:
                overnight_log(f"STEP 2: skipped {model_label} / {cell} "
                              f"— {type(exc).__name__}: {exc}")
                continue

    fh.close()
    print(f"\nWrote {out.resolve()}")

    if not target_models:
        print("\nNo model was at the floor — Step 2 had nothing to test.")
        return

    print("\nDoes deference climb back? (comparing to the sigma-grid floor at the same sigma)")
    grid_rows = {(r["model"], r.get("sigma")): r["p_wait"]
                for r in (json.loads(l) for l in open("probe_results.sigmagrid.jsonl"))
                if r.get("premise") == "costly" and r.get("stmt") == "inevitability"}
    for model_label in target_models:
        for sigma in sigmas:
            baseline = grid_rows.get((model_label, sigma))
            cells = [r for r in rows if r["model"] == model_label and r["sigma"] == sigma]
            if not cells or baseline is None:
                continue
            best = max(cells, key=lambda r: r["p_wait"])
            climbed = best["p_wait"] > baseline + 0.05
            print(f"  {model_label} sigma={sigma}: baseline P(WAIT)={baseline:.3f} -> "
                 f"best dose ({best['dose']}) P(WAIT)={best['p_wait']:.3f}"
                 f"{'  <-- CLIMBS BACK' if climbed else '  (no real recovery)'}")


# ---------------------------------------------------------------------------
# full grid: motivation x rhea{absent,present} x sigma x mismatch_cost x
# apply_risk (the last two only when rhea is present, per the spec).
# ---------------------------------------------------------------------------

def grid_cells(args):
    cells = []
    # rhea absent: motivation only, sigma is unused by the no-Rhea premise.
    for motivation in args.motivations:
        key = f"PROBE_ABSENT_{motivation}"
        cond = scn.Condition(key, committed_present=False,
                             inconsistency_costly=True, quorum_gate=False,
                             motivation=motivation)
        cells.append((cond, 0.0))
    # rhea present: full cross. --mismatch is a dose now (off/mild/moderate/
    # severe/all), kept as a CLI string rather than hardcoded here, since
    # Pass A is what determines which dose is worth using in this grid.
    mismatch_opts = (["off", "mild", "moderate", "severe"] if args.mismatch == "all"
                     else [args.mismatch])
    risk_opts = [False, True] if args.apply_risk == "both" else [args.apply_risk == "on"]
    for motivation, sigma, dose, risk in itertools.product(
            args.motivations, args.sigmas, mismatch_opts, risk_opts):
        key = f"PROBE_PRESENT_{motivation}_s{sigma}_m{dose}_r{int(risk)}"
        cond = scn.Condition(key, committed_present=True,
                             inconsistency_costly=True, quorum_gate=False,
                             motivation=motivation, mismatch_dose=dose,
                             apply_risk=risk)
        cells.append((cond, sigma))
    return cells


def run_grid(model, model_label: str, args):
    out_path = pathlib.Path(args.out)
    skip = rp.done_cells(out_path)
    cells = grid_cells(args)
    jobs = []
    for cond, sigma in cells:
        scn.CONDITIONS[cond.key] = cond  # register so run_plain can look it up
        for rep in range(args.reps):
            if (model_label, cond.key, "inevitability", sigma, rep) in skip:
                continue
            jobs.append((cond, sigma, rep))

    print(f"{len(cells)} cells x up to {args.reps} reps = {len(jobs)} "
          f"episodes to run ({len(skip)} already in {out_path})")
    lock = threading.Lock()
    done = 0

    def one(job):
        nonlocal done
        cond, sigma, rep = job
        t0 = time.time()
        try:
            ep = rp.run_episode(
                model, sigma=sigma, cond_key=cond.key, variant="inevitability",
                rep=rep, n_aligned=args.n_aligned, n_rounds=args.rounds,
                sticky=True, agent_workers=args.agent_workers)
            error = None
        except Exception as exc:
            ep, error = {"declarations": []}, f"{type(exc).__name__}: {exc}"
        record = {
            "model": model_label, "provider": args.provider,
            "condition": cond.key, "variant": "inevitability",
            "sigma": sigma, "rep": rep, "n_aligned": args.n_aligned,
            "rounds": args.rounds, "motivation": cond.motivation,
            "mismatch_dose": cond.mismatch_dose, "apply_risk": cond.apply_risk,
            "rhea_present": cond.committed_present,
            "sigma_hat": scn.normative_threshold(args.n_aligned),
            "error": error, "seconds": round(time.time() - t0, 1), **ep,
        }
        final = [d for d in record["declarations"] if d["round"] == args.rounds - 1]
        n_wait = sum(d["choice"] == scn.WAIT for d in final)
        with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            done += 1
            print(f"[{done}/{len(jobs)}] {cond.key:<42} "
                  f"deference={n_wait}/{len(final)} ({record['seconds']}s)"
                  + (f"  ERROR {error}" if error else ""))

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=args.run_workers) as pool:
        list(pool.map(one, jobs))
    print(f"\nWrote to {out_path.resolve()}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="adaption",
                    choices=list(rp.PROVIDERS) + ["adaption", "mock"])
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--run-workers", type=int, default=1)
    ap.add_argument("--agent-workers", type=int, default=4)
    ap.add_argument("--n-aligned", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="probe_results.jsonl")

    ap.add_argument("--calibrate", action="store_true",
                    help="single-agent, single-round, no-Rhea sweep over "
                         "motivation levels only — run this before the grid")
    ap.add_argument("--pass-a", action="store_true",
                    help="run Pass A: motivation=medium fixed, rhea "
                         "absent/present x sigma x mismatch dose, isolated "
                         "single-shot decisions, across --models")
    ap.add_argument("--step1", action="store_true",
                    help="run Step 1 diagnostic cells (mismatch_cost OFF; "
                         "inevitability/bare/costless x sigma{0.1,0.9})")
    ap.add_argument("--sigma-grid", action="store_true",
                    help="Part A: sigma in {0.1,0.3,0.5,0.7,0.9} x "
                         "premise{costly,costless} x statement — fills in "
                         "the collapse-point curve; reuses Pass A / Step 1 "
                         "cells where they match")
    ap.add_argument("--motivation-control", action="store_true",
                    help="Part B: motivation{none,low} x sigma{0.1,0.9} x "
                         "statement{inevitability,bare}, Rhea present")
    ap.add_argument("--step2", action="store_true",
                    help="Step 2: apply_risk 3-level dose (CONFIRMED text), "
                         "sigma{0.1,0.9}, only for models still at the floor "
                         "in probe_results.sigmagrid.jsonl")
    ap.add_argument("--models", nargs="+", default=None,
                    help="model list for --pass-a/--step1 (one shared --provider)")
    ap.add_argument("--calibrate-n", type=int, default=30)

    ap.add_argument("--motivations", nargs="+", default=["low", "medium", "high"],
                    choices=["low", "medium", "high"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--mismatch", choices=["off", "mild", "moderate", "severe", "all"], default="all")
    ap.add_argument("--apply-risk", choices=["off", "on", "both"], default="both")
    args = ap.parse_args()

    def build_model(name: str):
        limiter = rp.RateLimiter(args.rpm)  # separate limiter per model
        if args.provider == "mock":
            return rp.MockModel(name)
        if args.provider == "adaption":
            return rp.AdaptionModel(name, args.temperature, args.max_tokens, limiter)
        return rp.ChatModel(args.provider, name, args.base_url,
                            args.temperature, args.max_tokens, limiter)

    if args.pass_a:
        model_names = args.models or [args.model]
        models = {name: build_model(name) for name in model_names}
        pass_a(models, args)
    elif args.step1:
        model_names = args.models or [args.model]
        models = {name: build_model(name) for name in model_names}
        step1_diagnostic(models, args)
    elif args.sigma_grid:
        model_names = args.models or [args.model]
        models = {name: build_model(name) for name in model_names}
        sigma_grid(models, args)
    elif args.motivation_control:
        model_names = args.models or [args.model]
        models = {name: build_model(name) for name in model_names}
        motivation_control(models, args)
    elif args.step2:
        model_names = args.models or [args.model]
        models = {name: build_model(name) for name in model_names}
        step2_apply_risk(models, args)
    elif args.calibrate:
        model = build_model(args.model)
        calibrate(model, model.model, args)
    else:
        model = build_model(args.model)
        run_grid(model, model.model, args)


if __name__ == "__main__":
    main()
