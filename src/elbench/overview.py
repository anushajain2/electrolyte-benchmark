"""Aggregate-only overview page (docs/index.html) and the README figure.

Everything here is computed from aggregate result tables (summary, bootstrap,
paper counts, label-effect tables). No CALiSol measurement row is written, so
the page can be hosted publicly.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RunManifest
from .labels import model_label, task_label

TRACK = "molal_w"
MODELS = ["extra_trees", "ridge", "arrhenius_local"]
SERIES_VAR = {"extra_trees": "--s1", "ridge": "--s2", "arrhenius_local": "--s3"}
LIGHT = {"extra_trees": "#2a78d6", "ridge": "#eb6834", "arrhenius_local": "#1baf7a"}
SETTINGS = ["interp_formulation", "source_transfer", "author_transfer", "conc_high", "temp_high", "temp_low", "component_transfer"]
AXIS_MAX = 0.6
REPORT_LINK = "../reports/v0.5/report.html"
REPO = "https://github.com/anushajain2/electrolyte-benchmark"


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _factor(v: float) -> str:
    return f"about a factor of {10 ** v:.2f}" if v >= 0.18 else f"about {100 * (10 ** v - 1):.0f}%"


def _settings_table(res: Path) -> pd.DataFrame:
    """One row per (setting, model): group-mean error with a group-bootstrap interval."""
    s = pd.read_csv(res / "tables" / "summary.csv")
    b = pd.read_csv(res / "tables" / "bootstrap.csv")
    s = s[(s.track_id == TRACK) & (s.subset == "all") & s.model.isin(MODELS) & s.complete]
    b = b[(b.track_id == TRACK) & (b.subset == "all") & b.model.isin(MODELS)]
    b = b.sort_values("comparison", key=lambda c: c != "with_local").drop_duplicates(["task_id", "model"])
    out = s.merge(b[["task_id", "model", "ci_low", "ci_high"]], on=["task_id", "model"], how="left")
    out = out[out.task_id.isin(SETTINGS)]
    out["order"] = out.task_id.map(SETTINGS.index)
    return out.sort_values(["order", "model"])[["task_id", "model", "group_mean_mae_log10", "ci_low", "ci_high", "n_groups", "n_rows", "reporting_level"]]


def _funnel(rep: Path) -> pd.DataFrame:
    p = rep / "paper_counts.csv"  # written by `report` when versions are compared
    if not p.exists():
        return pd.DataFrame(columns=["stage", "count", "cause"])
    pc = pd.read_csv(p)
    keep = [("(all)", "papers in the backbone collection", "Papers in the CALiSol collection"),
            ("(all)", "papers with >=1 included row", "Papers with at least one usable row"),
            (TRACK, "papers in this track", "Papers in the primary track (molality recipes)"),
            (TRACK, "source groups", "Source groups after provenance linking"),
            (TRACK, "source groups testable in source transfer", "Groups testable when held out")]
    rows = []
    for tr, step, lab in keep:
        r = pc[(pc.track_id == tr) & (pc.step == step)].iloc[0]
        rows.append({"stage": lab, "count": int(r["count"]), "cause": r.cause})
    return pd.DataFrame(rows)


def _cleaning(inv: Path) -> pd.DataFrame:
    """Error change from evaluating an easier case set vs from cleaner training labels (Extra Trees, floor 0.2 mS/cm)."""
    ev = pd.read_csv(inv / "label_evaluation_filtering.csv")
    tr = pd.read_csv(inv / "label_training_cleaning.csv")
    rows = []
    for task, tier in [("source_transfer", "verified"), ("source_transfer", "verified_and_inferred"), ("temp_low", "verified_and_inferred"),
                       ("interp_formulation", "verified_and_inferred"), ("conc_high", "verified_and_inferred")]:
        e = ev[(ev.definition == f"{tier}@0.2") & (ev.task_id == task) & (ev.model == "extra_trees")]
        t = tr[(tr.tier == tier) & (tr.task_id == task) & (tr.model == "extra_trees")]
        if e.empty or t.empty:
            continue
        rows.append({"task_id": task, "tier": "verified source" if tier == "verified" else "verified + inferred sources",
                     "evaluation_filtering": float(e.change.iat[0]), "training_cleaning": float(t.training_effect.iat[0]),
                     "coverage": float(e.coverage.iat[0]), "train_rows_removed": int(t.train_rows_removed.sum())})
    return pd.DataFrame(rows, columns=["task_id", "tier", "evaluation_filtering", "training_cleaning", "coverage", "train_rows_removed"])


def _ablation(inv: Path) -> pd.DataFrame:
    p = inv / "temperature_ablation.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------- charts (HTML/CSS, responsive)

def _x(v: float) -> float:
    return 100 * min(v, AXIS_MAX) / AXIS_MAX


def _settings_chart(t: pd.DataFrame) -> str:
    ticks = np.arange(0, AXIS_MAX + 1e-9, 0.1)
    grid = "".join(f"<span class='gl' style='left:{_x(v):.2f}%'></span>" for v in ticks)
    f2 = f"<span class='f2' style='left:{_x(np.log10(2)):.2f}%'></span>"
    rows = []
    for task in SETTINGS:
        tt = t[t.task_id == task]
        if tt.empty:
            continue
        n = tt.iloc[0]
        sub = f"{int(n.n_groups)} groups · {int(n.n_rows):,} cases" + (" · case study" if n.reporting_level == "case_study" else "")
        marks = []
        for j, m in enumerate(MODELS):
            r = tt[tt.model == m]
            if r.empty:
                continue
            r = r.iloc[0]
            v, lo, hi = r.group_mean_mae_log10, r.ci_low, r.ci_high
            off = v > AXIS_MAX
            tip = (f"<b>{_e(model_label(m))}</b><br>{_e(task_label(task))}<br>Mean error {v:.3f} log10 ({_factor(v)})"
                   f"<br>95% group-bootstrap interval {lo:.3f} to {hi:.3f}<br>{int(r.n_groups)} source groups")
            ci = (f"<span class='ci' style='left:{_x(lo):.2f}%;width:{max(_x(hi) - _x(lo), 0.3):.2f}%;background:var({SERIES_VAR[m]})'></span>")
            dot = (f"<button class='dot{' off' if off else ''}' style='left:{_x(v):.2f}%;--c:var({SERIES_VAR[m]})' data-tip='{_e(tip)}' "
                   f"aria-label='{_e(model_label(m))}: {v:.3f}'></button>")
            lab = ""
            if task == "temp_low" or off:
                text = f"{model_label(m)} {v:.2f}" if not off else f"{model_label(m)} → {v:.1f}"
                side = "r" if _x(v) < 70 else "l"
                lab = f"<span class='dl {side}' style='left:{_x(v):.2f}%'>{_e(text)}</span>"
            marks.append(f"<div class='lane' style='top:{8 + j * 14}px'>{ci}{dot}{lab}</div>")
        rows.append(f"<div class='row'><div class='lab'>{_e(task_label(task))}<span>{_e(sub)}</span></div>"
                    f"<div class='track'>{grid}{f2}{''.join(marks)}</div></div>")
    axis = "".join(f"<span class='tk' style='left:{_x(v):.2f}%'>{v:.1f}</span>" for v in ticks)
    legend = "".join(f"<span class='lg'><i style='background:var({SERIES_VAR[m]})'></i>{_e(model_label(m))}</span>" for m in MODELS)
    return (f"<div class='legend'>{legend}<span class='lg'><i class='f2key'></i>factor-of-2 error</span></div>"
            f"<div class='dotplot'>{''.join(rows)}<div class='row axis'><div class='lab'></div><div class='track'>{axis}</div></div></div>"
            "<p class='axcap'>Mean error per source group, log10(conductivity); lower is better. Lines show 95% intervals from resampling source groups. "
            "Per-formulation Arrhenius is defined only for temperature extrapolation.</p>")


def _funnel_chart(f: pd.DataFrame) -> str:
    top = f["count"].max()
    rows = []
    for r in f.itertuples():
        tip = f"<b>{r.count}</b> {_e(r.stage.lower())}<br>Reduced by: {_e(r.cause)}"
        rows.append(f"<div class='frow'><div class='flab'>{_e(r.stage)}</div><div class='fbar-wrap'>"
                    f"<button class='fbar' style='width:{100 * r.count / top:.1f}%' data-tip='{_e(tip)}' aria-label='{_e(r.stage)}: {r.count}'></button>"
                    f"<span class='fval'>{r.count}</span></div></div>")
    return f"<div class='funnel'>{''.join(rows)}</div>"


def _cleaning_chart(c: pd.DataFrame) -> str:
    lo = min(-0.03, c[["evaluation_filtering", "training_cleaning"]].min().min() * 1.15) - 0.006  # room for the value label
    hi = max(0.005, c[["evaluation_filtering", "training_cleaning"]].max().max() * 1.3)
    pos = lambda v: 100 * (v - lo) / (hi - lo)  # noqa: E731
    zero = pos(0)
    ticks = [v for v in np.arange(-0.03, hi + 1e-9, 0.01) if v >= lo]
    grid = "".join(f"<span class='gl' style='left:{pos(v):.2f}%'></span>" for v in ticks)
    rows = []
    for r in c.itertuples():
        bars = []
        for j, (col, var, name) in enumerate([("evaluation_filtering", "--s4", "Easier test set"), ("training_cleaning", "--s5", "Cleaner training labels")]):
            v = getattr(r, col)
            left, width = (pos(v), zero - pos(v)) if v < 0 else (zero, pos(v) - zero)
            tip = (f"<b>{name}</b><br>{_e(task_label(r.task_id))}, {_e(r.tier)}<br>Change in mean error {v:+.3f} log10"
                   f"<br>Test cases kept: {100 * r.coverage:.0f}%" + (f"<br>Training rows removed: {r.train_rows_removed}" if col == "training_cleaning" else ""))
            bars.append(f"<button class='cbar' style='top:{6 + j * 14}px;left:{left:.2f}%;width:{max(width, 0.4):.2f}%;background:var({var})' "
                        f"data-tip='{_e(tip)}' aria-label='{name}: {v:+.3f}'></button>"
                        f"<span class='cval' style='top:{3 + j * 14}px;{'right' if v < 0 else 'left'}:{(100 - left) if v < 0 else (left + width):.2f}%'>{v:+.3f}</span>")
        rows.append(f"<div class='row'><div class='lab'>{_e(task_label(r.task_id))}<span>floor on {_e(r.tier)} · {100 * r.coverage:.0f}% of cases kept</span></div>"
                    f"<div class='track'>{grid}<span class='zero' style='left:{zero:.2f}%'></span>{''.join(bars)}</div></div>")
    axis = "".join(f"<span class='tk' style='left:{pos(v):.2f}%'>{v:+.2f}</span>" for v in ticks)
    legend = ("<span class='lg'><i style='background:var(--s4)'></i>Easier test set (same predictions, rescored on kept cases)</span>"
              "<span class='lg'><i style='background:var(--s5)'></i>Cleaner training labels (refit, scored on identical cases)</span>")
    return (f"<div class='legend'>{legend}</div><div class='dotplot cleaning'>{''.join(rows)}"
            f"<div class='row axis'><div class='lab'></div><div class='track'>{axis}</div></div></div>"
            "<p class='axcap'>Change in Extra Trees mean error (log10) when labels below 0.2 mS/cm from linear-axis sources are treated as unreliable. Negative = lower error.</p>")


def _table(df: pd.DataFrame, caption: str) -> str:
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{_e(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<details class='tv'><summary>{_e(caption)}</summary><div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>"


# ---------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#a8a69e;--s5:#6250d6;--fun:#86b6ef;--accent:#2a78d6;--tipbg:#0b0b0b;--tipink:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#6b6a64;--s5:#9085e9;--fun:#256abf;--accent:#3987e5;--tipbg:#f0efec;--tipink:#0b0b0b}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#6b6a64;--s5:#9085e9;--fun:#256abf;--accent:#3987e5;--tipbg:#f0efec;--tipink:#0b0b0b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 6px}
h3{font-size:15px;margin:0 0 6px}
p{margin:6px 0 10px;color:var(--ink2)} .lede{font-size:17px;color:var(--ink);max-width:760px}
a{color:var(--accent)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:18px 18px 14px;margin:14px 0}
.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:14px}
.choice{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px}
.choice dl{margin:8px 0 0;font-size:13.5px}.choice dt{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-top:8px}
.choice dd{margin:2px 0 0;color:var(--ink)} .num{font-variant-numeric:tabular-nums}
.links{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
.links a{display:inline-block;padding:8px 14px;border:1px solid var(--ring);border-radius:999px;background:var(--surface);text-decoration:none;color:var(--ink);font-size:14px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin:4px 0 10px}
.lg{display:inline-flex;align-items:center;gap:6px}.lg i{width:10px;height:10px;border-radius:50%;display:inline-block}
.lg i.f2key{width:0;height:12px;border-left:1.5px dashed var(--muted);border-radius:0}
.dotplot .row{display:grid;grid-template-columns:minmax(150px,240px) 1fr;gap:12px;align-items:stretch;border-top:1px solid var(--grid)}
.dotplot .row.axis{border-top:1px solid var(--axis)}
.lab{font-size:13.5px;padding:7px 0;color:var(--ink)}.lab span{display:block;font-size:12px;color:var(--muted)}
.track{position:relative;min-height:52px;margin-right:56px}
.cleaning .track{min-height:40px}
.axis .track{min-height:22px}
.gl{position:absolute;top:0;bottom:0;border-left:1px solid var(--grid)}
.f2{position:absolute;top:0;bottom:0;border-left:1.5px dashed var(--muted)}
.zero{position:absolute;top:0;bottom:0;border-left:1.5px solid var(--axis)}
.lane{position:absolute;left:0;right:0;height:12px}
.ci{position:absolute;top:5px;height:2px;opacity:.45;border-radius:1px}
.dot{position:absolute;top:0;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:var(--c);border:2px solid var(--surface);padding:0;cursor:pointer}
.dot::after{content:"";position:absolute;inset:-8px}
.dot.off{border-radius:2px;transform:rotate(45deg)}
.dot:focus-visible,.fbar:focus-visible,.cbar:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.dl{position:absolute;top:-3px;font-size:11.5px;line-height:16px;color:var(--ink2);white-space:nowrap;background:var(--surface);padding:0 4px;border-radius:3px}
.dl.r{margin-left:10px}.dl.l{transform:translateX(-100%);margin-left:-10px}
.tk{position:absolute;top:3px;transform:translateX(-50%);font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.funnel .frow{display:grid;grid-template-columns:minmax(150px,280px) 1fr;gap:12px;align-items:center;margin:6px 0}
.flab{font-size:13.5px}.fbar-wrap{display:flex;align-items:center;gap:8px}
.fbar{height:22px;border:0;border-radius:0 4px 4px 0;background:var(--fun);padding:0;cursor:pointer;min-width:4px}
.fval{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}
.cbar{position:absolute;height:11px;border:0;border-radius:2px;padding:0;cursor:pointer}
.cval{position:absolute;font-size:11.5px;color:var(--ink2);padding:0 5px;font-variant-numeric:tabular-nums;white-space:nowrap}
.axcap{font-size:12.5px;color:var(--muted);margin-top:8px}
.tv{margin-top:8px;font-size:13px}.tv summary{cursor:pointer;color:var(--ink2)}
.tw{overflow-x:auto}table{border-collapse:collapse;margin-top:6px;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:4px 10px;text-align:left;white-space:nowrap}th{color:var(--ink2);font-weight:600}
#tip{position:fixed;pointer-events:none;background:var(--tipbg);color:var(--tipink);font-size:12.5px;line-height:1.45;padding:8px 10px;border-radius:8px;max-width:280px;opacity:0;transition:opacity .08s;z-index:10}
.note{font-size:13px;color:var(--muted)}
footer{margin-top:40px;font-size:12.5px;color:var(--muted)}
@media (max-width:640px){.dotplot .row,.funnel .frow{grid-template-columns:1fr;gap:0}.lab{padding-bottom:0}.track{margin-right:40px}h1{font-size:23px}}
"""

JS = """
const tip=document.getElementById('tip');
function show(e){const t=e.currentTarget;tip.innerHTML=t.dataset.tip;tip.style.opacity=1;
 const r=t.getBoundingClientRect();let x=r.left+r.width/2+12,y=r.top-8;
 const w=tip.offsetWidth,h=tip.offsetHeight;if(x+w>innerWidth-8)x=r.left-w-12;if(x<8)x=8;if(y+h>innerHeight-8)y=innerHeight-h-8;if(y<8)y=r.bottom+8;
 tip.style.left=x+'px';tip.style.top=y+'px';}
function hide(){tip.style.opacity=0}
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('mouseenter',show);el.addEventListener('focus',show);el.addEventListener('mouseleave',hide);el.addEventListener('blur',hide);});
"""


def _choices(t: pd.DataFrame, c: pd.DataFrame) -> str:
    def v(task, m):
        r = t[(t.task_id == task) & (t.model == m)]
        return (float(r.group_mean_mae_log10.iat[0]), int(r.n_groups.iat[0])) if len(r) else (float("nan"), 0)

    et_i, g_i = v("interp_formulation", "extra_trees")
    ar_c, g_c = v("temp_low", "arrhenius_local")
    ri_c, _ = v("temp_low", "ridge")
    et_c, _ = v("temp_low", "extra_trees")
    et_s, g_s = v("source_transfer", "extra_trees")
    cards = [
        ("New formulations in familiar chemistry",
         "Your new mixture uses the same salt and solvents as existing data, at measured temperatures and concentrations.",
         "Training data in the same chemistry systems.",
         "Extra Trees, Ridge, k-NN, gradient boosting, training median.",
         f"Extra Trees: {et_i:.3f} log10 ({_factor(et_i)}), lowest in all {g_i} source groups.",
         "Groups come from 3 author groups; this is interpolation, not discovery."),
        ("Warm measurements → colder estimates",
         "You measured a formulation at room temperature and above and need it below the coldest measurement.",
         "At least three warm-side temperatures of the same formulation (for the Arrhenius fit).",
         "Per-formulation Arrhenius, Ridge (log target), Extra Trees.",
         f"Arrhenius {ar_c:.3f} and Ridge {ri_c:.3f} vs Extra Trees {et_c:.3f} log10; lower in every one of {g_c} groups.",
         "Tested to 253 K and below. Near the glass transition or freezing, Arrhenius bends (VTF behaviour), so error grows with distance."),
        ("Transfer to a new source",
         "You want a model trained on other papers to predict a lab or paper it has never seen.",
         "Test conditions inside the training data's common support.",
         "Extra Trees, Ridge, k-NN, gradient boosting, training median.",
         f"Extra Trees: {et_s:.3f} log10 ({_factor(et_s)}) on {g_s} testable source groups.",
         f"Only {g_s} of 7 groups are testable; about half of the improvement seen after label cleaning comes from an easier test set."),
    ]
    out = []
    for title, when, req, base, res, bound in cards:
        out.append(f"<div class='choice'><h3>{_e(title)}</h3><p class='note'>{_e(when)}</p><dl>"
                   f"<dt>Data requirement</dt><dd>{_e(req)}</dd><dt>Baselines scored</dt><dd>{_e(base)}</dd>"
                   f"<dt>Strongest result</dt><dd class='num'>{_e(res)}</dd><dt>Evidence boundary</dt><dd>{_e(bound)}</dd></dl></div>")
    return "<div class='choices'>" + "".join(out) + "</div>"


def readme_figure(t: pd.DataFrame, path: Path) -> Path:
    """Static PNG of the settings chart for the README (light theme, direct labels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [s for s in SETTINGS if s in set(t.task_id)]
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(tasks) + 1.3), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for j, m in enumerate(MODELS):
        for i, task in enumerate(tasks):
            r = t[(t.task_id == task) & (t.model == m)]
            if r.empty:
                continue
            v, lo, hi = r.group_mean_mae_log10.iat[0], r.ci_low.iat[0], r.ci_high.iat[0]
            y = i + (j - 1) * 0.22
            ax.plot([lo, min(hi, AXIS_MAX)], [y, y], color=LIGHT[m], alpha=0.45, lw=1.6, solid_capstyle="round")
            if v > AXIS_MAX:
                ax.scatter(AXIS_MAX, y, marker="D", s=40, color=LIGHT[m], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
                ax.text(AXIS_MAX - 0.015, y, f"{model_label(m)} → {v:.1f}", ha="right", va="center", fontsize=7.5, color="#52514e",
                        bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=0.6), zorder=4)
            else:
                ax.scatter(v, y, s=46, color=LIGHT[m], edgecolor="#fcfcfb", linewidth=1.5, zorder=3, label=model_label(m) if i == tasks.index("temp_low") else None)
                if task == "temp_low":
                    right = v < 0.45
                    ax.text(v + (0.012 if right else -0.012), y, f"{model_label(m)} {v:.2f}", va="center", ha="left" if right else "right", fontsize=7.5,
                            color="#52514e", bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=0.6), zorder=4)
    ax.axvline(np.log10(2), color="#898781", lw=1, ls="--", zorder=1)
    ax.text(np.log10(2), -0.75, " factor-of-2 error", fontsize=7.5, color="#52514e", va="bottom")
    ax.set_yticks(range(len(tasks)), [task_label(s) for s in tasks], fontsize=8.5, color="#0b0b0b")
    ax.invert_yaxis()
    ax.set_ylim(len(tasks) - 0.5, -0.8)
    ax.set_xlim(0, AXIS_MAX)
    ax.set_xlabel("Mean error per source group, log10(conductivity)  (lower is better; lines: 95% group-bootstrap interval)", fontsize=8, color="#52514e")
    ax.tick_params(axis="x", labelsize=7.5, colors="#898781")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e1e0d9", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.4, -0.18), frameon=False)
    ax.set_title("Which baseline is best depends on the setting", fontsize=10.5, loc="left", color="#0b0b0b", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def overview(cfg: Config, out_dir: Path | None = None) -> Path:
    man = RunManifest("overview", cfg)
    res, rep = cfg.p("results"), cfg.p("reports")
    inv = res / "investigation"
    out_dir = Path(out_dir) if out_dir else cfg.root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    t, f, c, ab = _settings_table(res), _funnel(rep), _cleaning(inv), _ablation(inv)

    tt = t.assign(Setting=t.task_id.map(task_label), Model=t.model.map(model_label)).round(3)
    tt = tt[["Setting", "Model", "group_mean_mae_log10", "ci_low", "ci_high", "n_groups", "n_rows"]].rename(
        columns={"group_mean_mae_log10": "Mean error (log10)", "ci_low": "95% low", "ci_high": "95% high", "n_groups": "Source groups", "n_rows": "Test cases"})
    ct = c.assign(Setting=c.task_id.map(task_label)).round(3)[["Setting", "tier", "evaluation_filtering", "training_cleaning", "coverage", "train_rows_removed"]].rename(
        columns={"tier": "Floor applied to", "evaluation_filtering": "Easier test set", "training_cleaning": "Cleaner training labels", "coverage": "Share of cases kept", "train_rows_removed": "Training rows removed"})
    abl = ""
    if not ab.empty and {"extra_trees", "ridge"} <= set(ab[ab.task_id == "temp_low"].model):
        a = ab[ab.task_id == "temp_low"].pivot_table(index="model", columns="variant", values="group_mean_mae_log10")
        a = a[[x for x in ("all features", "T only (no 1/T)", "1/T only (no T)", "no temperature") if x in a]].round(3)
        a.index = a.index.map(model_label)
        abl = ("<h2>Is it the model or the temperature feature?</h2><p>Refitting without 1/T or without T shows the cold-extrapolation result is mainly a model-family effect: "
               f"Ridge with T alone ({a.loc[model_label('ridge'), 'T only (no 1/T)']:.3f}) still beats Extra Trees with every feature "
               f"({a.loc[model_label('extra_trees'), 'all features']:.3f}). Trees predict a constant beyond the training range, whichever form of temperature they see.</p>"
               + _table(a.reset_index().rename(columns={"model": "Model"}), "Cold extrapolation, mean error (log10) by temperature representation"))

    meta = json.loads((res / "investigation" / "README.json").read_text()) if (res / "investigation" / "README.json").exists() else {}
    body = f"""
<main>
<p class='note'>Electrolyte generalization benchmark · v{_e(cfg['benchmark_version'])} · aggregate results only</p>
<h1>Which conductivity model should you trust for your next electrolyte?</h1>
<p class='lede'>It depends on what you already measured. On public CALiSol data for liquid Li-salt electrolytes, the model that wins on new formulations of familiar chemistry
loses badly when asked for colder temperatures than it has seen, and fewer than a third of the source papers can be tested as genuinely new sources.</p>
<div class='links'><a href='{REPORT_LINK}'>Full report</a><a href='{REPO}#readme'>Code and README</a><a href='{REPO}/blob/main/METHODS.md'>What the models see</a><a href='{REPO}#run-it-yourself'>Run the workbench</a></div>

<h2>Three situations, three answers</h2>
{_choices(t, c)}

<h2>Model performance across settings</h2>
<p>Each dot is a source-group-weighted mean error. A gap of 0.30 log10 is a factor of 2 in conductivity.</p>
<div class='card'>{_settings_chart(t)}{_table(tt, 'Show as table')}</div>

<h2>From 27 papers to 4 testable source groups</h2>
<p>Papers that reuse each other's measurements or come from the same campaign are grouped before any split, so a held-out source is not secretly in training.
Grouping, not paper count, sets how many independent tests exist.</p>
{f"<div class='card'>{_funnel_chart(f)}{_table(f.rename(columns={'stage': 'Stage', 'count': 'Count', 'cause': 'Reduced by'}), 'Show as table')}</div>" if len(f) else "<p class='note'>Paper counts are not available for this configuration.</p>"}

<h2>Better labels, or an easier test?</h2>
<p>Low conductivities read from linear-axis plots are unreliable. Removing them lowers error, but two different things happen: the test set gets easier, and the training labels get cleaner.
Holding fold membership fixed separates the two. For cold extrapolation the whole improvement is an easier test set.</p>
{f"<div class='card'>{_cleaning_chart(c)}{_table(ct, 'Show as table')}</div>" if len(c) else "<p class='note'>Label-effect comparisons are not available for this configuration.</p>"}

{abl}

<h2>What this does and does not show</h2>
<p>Seven source groups drawn from three author groups are not independent laboratories; intervals resample groups and are conditional on that grouping.
Results come from versions made after earlier test results had been seen, so they are evidence for further validation, not a leaderboard.
Ten CALiSol rows with wrong temperatures were confirmed against the original paper and are excluded from the reference; the upstream report has not yet been submitted.
No CALiSol measurement is shown on this page.</p>
<footer>Generated by <code>elbench overview</code> from aggregate result tables{(' · investigation run ' + _e(meta.get('run_id', ''))) if meta.get('run_id') else ''}.
Data: CALiSol-23 (De Blasio et al., Sci. Data 2024). Code: MIT licence.</footer>
</main>
<div id='tip' role='tooltip'></div>"""
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Electrolyte Generalization Benchmark</title><meta name='description' content='Which ionic-conductivity baseline to trust depends on the setting: aggregate results from the electrolyte generalization benchmark.'>"
            f"<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>")
    path = out_dir / "index.html"
    path.write_text(page)
    readme_figure(t, out_dir / "settings_overview.png")
    man.add_output(path)
    man.add_output(out_dir / "settings_overview.png")
    man.write(rep / "runs")
    return path
