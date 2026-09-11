"""Charts for the delegation experiment."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Reference palette, categorical slot 1; each panel shows a single series.
BAR_COLOR = {"light": "#2a78d6", "dark": "#3987e5"}


def _nice_step(span: float, bins: int) -> float:
    raw = span / bins
    for step in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if step >= raw:
            return step
    return raw


def plot_payoff_hists(
    totals_by_agent: dict[str, list[float]],
    theme: str | None = "light",
    bins: int = 24,
    columns: int = 3,
) -> alt.FacetChart:
    """Small multiples: one payout histogram per agent on a shared x axis.

    Each panel's title names the agent and its average payout; bars show the
    share of simulated totals in each payout range, with a hover tooltip.
    """
    all_totals = np.concatenate([np.asarray(t, dtype=float) for t in totals_by_agent.values()])
    lo, hi = float(all_totals.min()), float(all_totals.max())
    step = _nice_step(max(hi - lo, 0.5), bins)
    edges = np.arange(np.floor(lo / step) * step, hi + step, step)
    if len(edges) < 2:
        edges = np.array([edges[0], edges[0] + step])

    rows, panels = [], []
    for name, totals in totals_by_agent.items():
        counts, _ = np.histogram(totals, bins=edges)
        panel = f"{name} (average ${np.mean(totals):.2f})"
        panels.append(panel)
        for count, start, end in zip(counts, edges[:-1], edges[1:]):
            rows.append({
                "panel": panel,
                "agent": name,
                "start": start,
                "end": end,
                "share": count / len(totals),
                "range": f"${start:.2f} to ${end:.2f}",
            })

    chart = (
        alt.Chart(pd.DataFrame(rows))
        # No corner radius: Vega-Lite drops binned bars that have one.
        .mark_bar(color=BAR_COLOR.get(theme or "light", BAR_COLOR["light"]), binSpacing=2)
        .encode(
            x=alt.X(
                "start:Q",
                bin="binned",
                title="Total payout on these questions",
                axis=alt.Axis(format="$,.0f"),
            ),
            x2="end:Q",
            y=alt.Y("share:Q", title="Share of simulations", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("agent:N", title="Agent"),
                alt.Tooltip("range:N", title="Payout"),
                alt.Tooltip("share:Q", title="Share of simulations", format=".1%"),
            ],
        )
        .properties(width=220, height=140)
    )
    return chart.facet(
        facet=alt.Facet("panel:N", title=None, sort=panels),
        columns=columns,
    )
