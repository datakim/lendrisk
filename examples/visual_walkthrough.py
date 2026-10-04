"""Reproduce the README figures with real package outputs and synthetic data.

Run from the repository root after installing matplotlib:
    python examples/visual_walkthrough.py
Use --output-dir to save figures somewhere other than docs/assets.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

from lendrisk import OptimalBinning, RevenueAdvance, RevenueShock, make_merchant_cashflows

INK = "#18363c"
MUTED = "#526b70"
TEAL = "#147d78"
GOLD = "#b57b18"
RED = "#bc4c36"
PAPER = "#fcfaf5"
GRID = "#dce5e3"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.parse_math": False,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "path",
        }
    )


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=160)
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def financing(output: Path) -> dict:
    data = make_merchant_cashflows(days=540, seed=42)
    path = data.iloc[180:]
    advance = RevenueAdvance(principal=30_000, factor_rate=1.12, holdback_rate=0.10)
    variants = [
        ("Baseline", path, TEAL),
        ("Sales -20%", RevenueShock("sales_down_20pct", 0.8).apply(path), GOLD),
        ("Sales -40%", RevenueShock("sales_down_40pct", 0.6).apply(path), RED),
    ]
    results = [
        (name, advance.simulate(frame, opening_cash=5_000), color)
        for name, frame, color in variants
    ]
    summaries = {name: result.summary() for name, result, _ in results}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.24, top=0.57, wspace=0.30)
    fig.text(0.065, 0.93, "What happens when sales fall?", size=23, weight="bold")
    fig.text(0.065, 0.87, "$30,000 advance · $33,600 receivable · 10% revenue share", color=MUTED)
    metrics = [
        ("BASELINE PAYOFF", f"Day {summaries['Baseline']['payoff_days']}", TEAL),
        ("-40% SALES: UNPAID", f"${summaries['Sales -40%']['remaining_balance']:,.0f}", RED),
        (
            "-40% SALES: LOWEST CASH",
            f"-${abs(summaries['Sales -40%']['minimum_cash_balance']):,.0f}",
            RED,
        ),
    ]
    for xpos, (label, value, color) in zip([0.065, 0.385, 0.705], metrics, strict=True):
        fig.text(xpos, 0.775, label, size=9, color=MUTED, weight="bold")
        fig.text(xpos, 0.705, value, size=23, color=color, weight="bold")
    for name, result, color in results:
        days = np.arange(len(result.schedule) + 1)
        balance = np.r_[result.target_repayment, result.schedule["remaining_balance"]]
        cash = np.r_[35_000, result.schedule["cash_balance"]]
        axes[0].plot(days, balance, label=name, color=color, lw=2.4)
        axes[1].plot(days, cash, label=name, color=color, lw=2.4)
    axes[0].set_title("Receivable still unpaid", loc="left", size=13, weight="bold", pad=13)
    axes[1].set_title("Merchant cash after financing", loc="left", size=13, weight="bold", pad=13)
    axes[0].set_ylim(-1_000, 36_000)
    axes[1].axhline(0, color=RED, lw=1, linestyle="--")
    axes[1].fill_between([0, 360], -10_000, 0, color=RED, alpha=0.07)
    axes[1].set_ylim(-10_000, 130_000)
    money = FuncFormatter(
        lambda value, _: f"-${abs(value) / 1000:g}k" if value < 0 else f"${value / 1000:g}k"
    )
    for ax in axes:
        ax.yaxis.set_major_formatter(money)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("Days after funding", labelpad=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.065),
        frameon=False,
        ncol=3,
        fontsize=10,
    )
    fig.text(
        0.065,
        0.035,
        "Synthetic seed 42 · 360-day horizon · costs held fixed under shocks · "
        "scheduled payments assumed collected",
        color=MUTED,
        size=8,
    )
    save(fig, output, "financing-scenarios")
    printable = {
        name: {
            key: summary[key]
            for key in [
                "repaid",
                "payoff_days",
                "remaining_balance",
                "minimum_cash_balance",
                "negative_cash_days",
            ]
        }
        for name, summary in summaries.items()
    }
    (output / "scenario-summary.json").write_text(json.dumps(printable, indent=2) + "\n")
    hero(output, results, summaries)
    return printable


def hero(output: Path, results, summaries) -> None:
    """An original SVG masthead with miniature, actual remaining-balance curves."""
    paths = []
    for _, result, color in results:
        values = np.r_[result.target_repayment, result.schedule["remaining_balance"]]
        sampled = np.unique(np.r_[np.arange(0, len(values), 6), len(values) - 1])
        coords = [
            (805 + day / 360 * 300, 120 + (1 - values[day] / 33_600) * 140) for day in sampled
        ]
        command = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        light = {TEAL: "#86d7bb", GOLD: "#edbd62", RED: "#f48d70"}[color]
        paths.append(f'<path d="{command}" fill="none" stroke="{light}" stroke-width="3"/>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360"
  viewBox="0 0 1200 360" role="img" aria-labelledby="title desc">
  <title id="title">lendrisk — Understand credit. Follow the cash.</title>
  <desc id="desc">An installable Python toolkit for binning, scorecards and revenue financing.
  Three actual synthetic repayment paths: baseline and sales declines of 20 and 40 percent.</desc>
  <rect width="1200" height="360" rx="12" fill="#112c31"/>
  <path d="M54 75 H70 V61 H86 V47 H102" stroke="#86d7bb" stroke-width="4" fill="none"/>
  <text x="120" y="76" fill="#fcfaf5"
    font-family="Verdana,sans-serif" font-size="36" font-weight="bold">
    lendrisk</text>
  <text x="54" y="155" fill="#fcfaf5"
    font-family="Georgia,serif" font-size="46">
    Understand credit.</text>
  <text x="54" y="213" fill="#86d7bb"
    font-family="Georgia,serif" font-size="46">
    Follow the cash.</text>
  <text x="56" y="266" fill="#c0d1d1"
    font-family="Verdana,sans-serif" font-size="16">
    Native binning · Explainable scores · Revenue financing</text>
  <text x="56" y="319" fill="#c0d1d1"
    font-family="monospace" font-size="13">
    PYTHON 3.10+  /  APACHE 2.0  /  ALPHA</text>
  <line x1="750" y1="48" x2="750" y2="312" stroke="#35565b"/>
  <text x="805" y="74" fill="#c0d1d1"
    font-family="monospace" font-size="13">
    ONE DEAL. THREE REVENUE PATHS.</text>
  <text x="805" y="105" fill="#c0d1d1"
    font-family="Verdana,sans-serif" font-size="12">
    Unpaid receivable · synthetic demo</text>
  <path d="M805 120 V260 H1105" stroke="#47656a" fill="none"/>
  {"".join(paths)}
  <text x="805" y="287" fill="#c0d1d1" font-family="monospace" font-size="11">FUNDING</text>
  <text x="1053" y="287" fill="#c0d1d1" font-family="monospace" font-size="11">DAY 360</text>
  <text x="805" y="319" fill="#86d7bb"
    font-family="Verdana,sans-serif" font-size="13">
    Baseline paid off on day {summaries["Baseline"]["payoff_days"]}</text>
</svg>"""
    (output / "banner.svg").write_text(svg + "\n")


def binning(output: Path) -> None:
    rng = np.random.default_rng(42)
    ratio = rng.uniform(0.05, 0.95, 1_000)
    default = rng.binomial(1, 0.02 + 0.42 * ratio**2)
    model = OptimalBinning(max_n_bins=5, monotonic_trend="ascending").fit(ratio, default)
    table = model.table()
    regular = table.loc[~table["bin"].isin(["Missing", "Special"])].copy()
    prebin = np.searchsorted(model.prebin_splits_, ratio, side="right")
    counts = np.bincount(prebin)
    rates = np.bincount(prebin, weights=default) / counts
    edges = np.r_[0.05, model.prebin_splits_, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.23, top=0.72, wspace=0.3)
    fig.text(0.065, 0.93, "From noisy values to readable risk groups", size=22, weight="bold")
    fig.text(
        0.065,
        0.86,
        f"1,000 synthetic observations · 20 candidate prebins → "
        f"{len(regular)} monotonic bins · solver: {model.status_}",
        color=MUTED,
    )
    axes[0].scatter(
        (edges[:-1] + edges[1:]) / 2,
        rates,
        color=MUTED,
        alpha=0.55,
        s=32,
        label="Candidate prebin rate",
    )
    final_edges = np.r_[0.05, model.splits_, 0.95]
    axes[0].stairs(
        regular["event_rate"],
        final_edges,
        baseline=None,
        color=TEAL,
        lw=3,
        label="Optimized bin rate",
    )
    axes[0].set_xlabel("Synthetic debt ratio")
    axes[0].set_ylabel("Observed default share")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, max(rates.max() + 0.07, 0.4))
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    xpos = np.arange(len(regular))
    bars = axes[1].bar(
        xpos, regular["woe"], color=[TEAL if v >= 0 else RED for v in regular["woe"]], width=0.6
    )
    for bar, count in zip(bars, regular["count"], strict=True):
        value = bar.get_height()
        axes[1].annotate(
            f"n={count:.0f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6 if value >= 0 else -13),
            textcoords="offset points",
            ha="center",
            size=9,
        )
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].set_xticks(xpos, [f"Bin {i + 1}" for i in xpos])
    axes[1].set_ylabel("Weight of evidence (WoE)")
    axes[1].set_xlabel("Increasing debt ratio →")
    axes[1].set_ylim(regular["woe"].min() - 0.4, regular["woe"].max() + 0.4)
    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.text(
        0.065,
        0.075,
        "Positive WoE: more non-defaults relative to defaults. "
        "Negative WoE: the reverse. Empty Missing/Special bins omitted.",
        size=9,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.035,
        "Training-data illustration only. "
        "Validate bins and model performance on a separate cohort.",
        size=9,
        color=MUTED,
    )
    save(fig, output, "native-binning")
    table.to_csv(output / "binning-table.csv", index=False, float_format="%.6f")


def score_scale(output: Path) -> None:
    pdo, base_score, base_odds = 20, 600, 50
    points = np.arange(480, 681)
    odds = base_odds * 2.0 ** ((points - base_score) / pdo)
    pd_hat = 1 / (1 + odds)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.23, top=0.72)
    fig.text(0.09, 0.92, "What does a credit score mean?", size=22, weight="bold")
    fig.text(0.09, 0.83, "With PDO = 20, adding 20 points doubles good:bad odds.", color=MUTED)
    ax.plot(points, pd_hat, color=TEAL, lw=3)
    for score, label in [
        (580, "580 points\n25:1 odds · 3.85% PD"),
        (600, "600 points\n50:1 odds · 1.96% PD"),
        (620, "620 points\n100:1 odds · 0.99% PD"),
    ]:
        risk = 1 / (1 + base_odds * 2 ** ((score - base_score) / pdo))
        ax.scatter(score, risk, s=45, color=TEAL, zorder=5)
        ax.annotate(
            label,
            (score, risk),
            xytext=(-16, 42 if score == 580 else 55),
            textcoords="offset points",
            size=9,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED},
        )
    ax.set_xlim(545, 660)
    ax.set_ylim(0, 0.125)
    ax.set_xlabel("Score points (higher → lower modeled risk)", labelpad=10)
    ax.set_ylabel("Model probability of default (PD)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.09,
        0.045,
        "A mathematical score-to-PD mapping, not observed default rates "
        "or evidence of model calibration.",
        size=9,
        color=MUTED,
    )
    save(fig, output, "score-scale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    style()
    summary = financing(output)
    binning(output)
    score_scale(output)
    print(json.dumps(summary, indent=2))
    print(f"Figures saved to {output.resolve()}")


if __name__ == "__main__":
    main()
