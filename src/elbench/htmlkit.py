"""Minimal, dependency-free HTML report helpers (deterministic output)."""
from __future__ import annotations

import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#5f6368;--line:#dcdcd8;--accent:#2f5d8a;--chip:#eef2f6;--warn:#8a5a00;--warnbg:#fff4dc}
@media (prefers-color-scheme: dark){:root{--bg:#161718;--fg:#e8e8e6;--muted:#a0a3a7;--line:#33363a;--accent:#8fb6dc;--chip:#23272b;--warn:#f0c46a;--warnbg:#2c2414}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:1.7em;margin:.2em 0}h2{font-size:1.25em;margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.3em}h3{font-size:1.05em;margin-top:1.6em}
p,li{max-width:80ch}.muted{color:var(--muted)}code{background:var(--chip);padding:1px 4px;border-radius:3px;font-size:.92em}
.tbl{overflow-x:auto;margin:.6em 0 1.2em}table{border-collapse:collapse;font-size:.86em;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th{text-align:left;position:sticky;top:0;background:var(--bg)}
td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto;background:#fff;border-radius:4px}
.note{background:var(--warnbg);color:var(--warn);padding:8px 12px;border-radius:4px;max-width:80ch}
.chip{display:inline-block;background:var(--chip);border-radius:10px;padding:0 8px;margin-right:4px;font-size:.85em}
nav a{margin-right:12px;color:var(--accent)}a{color:var(--accent)}
"""


def esc(x) -> str:
    return html.escape(str(x))


def table(df: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int | None = None, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>(empty)</p>"
    d = df.reset_index() if index else df
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    head = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    rows = []
    for r in d.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("<td></td>" if pd.isna(v) else f"<td>{floatfmt.format(v)}</td>")
            else:
                cells.append(f"<td>{esc(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else ''}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = f"<p class='muted'>showing {max_rows} of {len(df)} rows</p>" if max_rows is not None and len(df) > max_rows else ""
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{more}"


def fig_img(fig, alt: str = "") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return f"<img alt='{esc(alt)}' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'/>"


def page(title: str, body: str, subtitle: str = "") -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{esc(title)}</h1><p class='muted'>{esc(subtitle)}</p>{body}</main></body></html>"
    )
