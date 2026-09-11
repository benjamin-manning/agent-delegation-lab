"""Visualizations for the delegation experiment."""

from __future__ import annotations

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from decisions import DECISIONS

# ---- style ----
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "safe": "#457B9D",
    "risky": "#E63946",
    "positive": "#2A9D8F",
    "negative": "#E76F51",
    "neutral": "#264653",
    "grid": "#E0E0E0",
    "bg_light": "#F8F9FA",
}

CATEGORY_COLORS = {
    "Risk": "#2E86AB",
    "Loss": "#E76F51",
    "Insurance": "#457B9D",
    "Ambiguity": "#A23B72",
    "Investment": "#2A9D8F",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def plot_total_payoff_distribution(simulation: dict, agent_name: str, budget_kept: float) -> plt.Figure:
    """Histogram of total payoff across simulations, with budget line."""
    totals = simulation["total_payoffs"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(totals, bins=30, color=COLORS["primary"], edgecolor="white",
            alpha=0.85, density=False)

    mean_total = np.mean(totals)
    median_total = np.median(totals)
    p5 = np.percentile(totals, 5)
    p95 = np.percentile(totals, 95)

    ax.axvline(mean_total, color=COLORS["secondary"], linestyle="--",
               linewidth=2.5, label=f"Mean: ${mean_total:.2f}")
    ax.axvline(median_total, color=COLORS["neutral"], linestyle=":",
               linewidth=2, label=f"Median: ${median_total:.2f}")

    # Shade the 5th-95th percentile range
    ax.axvspan(p5, p95, alpha=0.08, color=COLORS["primary"],
               label=f"90% range: ${p5:.2f} - ${p95:.2f}")

    ax.set_xlabel("Total Payoff from 8 Decisions ($)")
    ax.set_ylabel("Frequency (out of 1,000 simulations)")
    ax.set_title(f"Payoff Distribution: {agent_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    # Net payout annotation
    net_mean = budget_kept + mean_total
    ax.text(
        0.02, 0.95,
        f"Budget kept: ${budget_kept:.2f}\n"
        f"Mean game earnings: ${mean_total:.2f}\n"
        f"Expected total payout: ${net_mean:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["bg_light"],
                  edgecolor=COLORS["grid"], alpha=0.9),
    )

    fig.tight_layout()
    return fig


def plot_problem_breakdown(simulation: dict, agent_name: str) -> plt.Figure:
    """Bar chart of expected value per problem, colored by category."""
    per_problem = simulation["per_problem"]

    fig, ax = plt.subplots(figsize=(12, 5))

    titles = [p["title"] for p in per_problem]
    evs = [p["ev"] for p in per_problem]
    categories = [p["category"] for p in per_problem]
    colors = [CATEGORY_COLORS.get(c, COLORS["primary"]) for c in categories]

    bars = ax.bar(range(len(titles)), evs, color=colors, edgecolor="white", width=0.6)

    for i, (bar, ev, choice) in enumerate(zip(bars, evs, per_problem)):
        # EV label
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"${ev:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Short choice label below bar
        chosen = choice["chosen"]
        # Truncate for display
        short = chosen[:20] + "..." if len(chosen) > 23 else chosen
        ax.text(bar.get_x() + bar.get_width() / 2, -0.8,
                short, ha="center", va="top", fontsize=7, rotation=0,
                color=COLORS["neutral"])

    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Expected Value of Chosen Option ($)")
    ax.set_title(f"Choices by Problem: {agent_name}", fontsize=14, fontweight="bold")

    # Legend for categories
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in CATEGORY_COLORS if c in set(categories)]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.set_ylim(bottom=-2)
    fig.tight_layout()
    return fig


def plot_comparison(simulations: list[dict], names: list[str], budgets_kept: list[float]) -> plt.Figure:
    """Compare payoff distributions across multiple agents."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["positive"],
                   COLORS["risky"], COLORS["neutral"]]

    for i, (sim, name, bk) in enumerate(zip(simulations, names, budgets_kept)):
        totals = sim["total_payoffs"]
        net = [t + bk for t in totals]
        color = colors_list[i % len(colors_list)]
        ax.hist(net, bins=25, alpha=0.45, color=color, edgecolor="white",
                label=f"{name} (mean: ${np.mean(net):.2f})")

    ax.set_xlabel("Total Payout (budget kept + game earnings) ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Agent Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
