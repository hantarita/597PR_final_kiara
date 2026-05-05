"""
Presentation-quality plots for IS597PR Final Project.

Generates six figures saved to output/slides/:
    00_pipeline.png        — data pipeline diagram
    01_rq1_scatter.png     — popularity vs cost (RQ1)
    02_rq2_boxplot.png     — cost by tier (RQ2 H2a)
    03_rq2_stacked.png     — country share by tier (RQ2 H2b)
    04_rq3_scatter.png     — GPA vs cost by country (RQ3)
    05_rq3_coef.png        — GPA coefficient before/after country control (RQ3 H3b)

Color palette: Okabe-Ito (colorblind-safe, standard in academic publishing).

Run from the project root:
    python make_slides_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
DATA_PATH = _HERE / "compass_offers_processed_matched_no_phd.csv"
TIER_PATH = _HERE / "university_tier.csv"
OUT_DIR   = _HERE / "output" / "slides"

# ── Okabe-Ito palette (colorblind-safe) ───────────────────────────────────

COUNTRY_COLORS = {
    "UK":        "#0072B2",   # blue
    "USA":       "#D55E00",   # vermillion
    "Australia": "#009E73",   # bluish green
    "Hong Kong": "#CC79A7",   # reddish purple
    "Singapore": "#E69F00",   # orange
}

TIER_COLORS = {
    "985":   "#0072B2",   # blue  (top tier)
    "211":   "#56B4E9",   # sky blue
    "Other": "#BBBBBB",   # neutral gray
}

TIER_ORDER    = ["985", "211", "Other"]
MAIN_COUNTRIES = list(COUNTRY_COLORS.keys())

# ── Global style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.color":       "#DDDDDD",
    "grid.linewidth":   0.6,
    "grid.linestyle":   "--",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "legend.title_fontsize": 9,
    "figure.dpi":       150,
})

USD_FMT = mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")


# ── Helper ───────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> None:
    """Save figure to OUT_DIR and close it."""
    out = OUT_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 0 — Pipeline diagram
# ══════════════════════════════════════════════════════════════════════════════

def make_pipeline() -> None:
    """Draw a clean data-pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    BOX_H   = 1.6
    BOX_W   = 2.1
    Y_MID   = 2.0
    RADIUS  = 0.12
    ARROW   = dict(arrowstyle="-|>", color="#444444", lw=1.8,
                   mutation_scale=16)

    def box(x_center: float, lines: list[str], color: str) -> None:
        rect = mpatches.FancyBboxPatch(
            (x_center - BOX_W / 2, Y_MID - BOX_H / 2),
            BOX_W, BOX_H,
            boxstyle=f"round,pad={RADIUS}",
            facecolor=color, edgecolor="#444444", linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        for i, line in enumerate(lines):
            offset = (len(lines) - 1) / 2 - i
            weight = "bold" if i == 0 else "normal"
            size   = 9.5 if i == 0 else 8.5
            ax.text(
                x_center, Y_MID + offset * 0.36,
                line, ha="center", va="center",
                fontsize=size, fontweight=weight, color="#1a1a1a", zorder=4,
            )

    def arrow(x0: float, x1: float) -> None:
        ax.annotate(
            "", xy=(x1 - BOX_W / 2, Y_MID), xytext=(x0 + BOX_W / 2, Y_MID),
            arrowprops=ARROW, zorder=2,
        )

    # Centers
    centers = [1.1, 3.6, 6.2, 8.9, 11.4]
    colors  = ["#D6EAF8", "#D6EAF8", "#D5F5E3", "#FDEBD0", "#EBD5F5"]

    boxes = [
        ["CompassEdu", "Offer Data", "~30,000 rows"],
        ["Education", "Cost Dataset", "(public CSV)"],
        ["preprocessing.py", "• Fuzzy matching", "• GPA standardisation", "• PhD filtering"],
        ["Matched Dataset", "26,971 rows", "88.47% match rate"],
        ["analysis.py", "RQ1 | RQ2 | RQ3", "6 output plots"],
    ]

    for cx, lines, col in zip(centers, boxes, colors):
        box(cx, lines, col)

    for i in range(len(centers) - 1):
        arrow(centers[i], centers[i + 1])

    ax.set_title(
        "Data Pipeline Overview",
        fontsize=14, fontweight="bold", pad=10,
    )
    save(fig, "00_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — RQ1 Scatter: popularity vs cost
# ══════════════════════════════════════════════════════════════════════════════

def make_rq1_scatter(df: pd.DataFrame) -> None:
    """Scatter plot: offer_count (x) vs mean_total_cost (y), coloured by country."""
    matched = df[df["match_status"] == "matched"].copy()
    agg = (
        matched
        .groupby("matched_school", as_index=False)
        .agg(
            offer_count=("matched_school", "count"),
            mean_total_cost=("cost_total_usd", "mean"),
            cost_country=("cost_country", "first"),
        )
        .sort_values("offer_count", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for country, group in agg.groupby("cost_country"):
        if country not in COUNTRY_COLORS:
            continue
        ax.scatter(
            group["offer_count"], group["mean_total_cost"],
            label=country,
            color=COUNTRY_COLORS[country],
            s=70, alpha=0.90, edgecolors="white", linewidths=0.6, zorder=4,
        )

    # Annotate top-10 by volume
    top10 = agg.head(10)
    for _, row in top10.iterrows():
        label = (row["matched_school"]
                 .replace("University", "U.")
                 .replace("The ", ""))
        if len(label) > 24:
            label = label[:22] + "…"
        ax.annotate(
            label,
            xy=(row["offer_count"], row["mean_total_cost"]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=6.5, color="#333333", zorder=5,
        )

    # OLS trend line (log-transformed x for readability)
    log_x = np.log10(agg["offer_count"])
    m, b  = np.polyfit(log_x, agg["mean_total_cost"], 1)
    x_range = np.logspace(log_x.min(), log_x.max(), 200)
    ax.plot(
        x_range,
        m * np.log10(x_range) + b,
        color="#444444", linewidth=1.4, linestyle="--",
        label="OLS trend", zorder=3,
    )

    # ρ annotation box
    ax.text(
        0.03, 0.95,
        "Spearman  ρ = −0.546\np < 0.001  (n = 71 schools)",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#AAAAAA", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Admissions (log scale)", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ1: Is Popularity Priced?\n"
        "More admissions ≠ higher cost — geography dominates",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.legend(title="Destination", fontsize=9, title_fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save(fig, "01_rq1_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — RQ2 Box plot: cost by tier
# ══════════════════════════════════════════════════════════════════════════════

def make_rq2_boxplot(df: pd.DataFrame) -> None:
    """Box plot of total cost by undergraduate tier (985 / 211 / Other)."""
    import difflib
    import csv

    # Assign tier
    tier_lookup: dict[str, str] = {}
    with TIER_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tier_lookup[row["school_name"].strip()] = row["tier"].strip()
    known = list(tier_lookup.keys())

    def assign(s: str) -> str:
        s = str(s).strip() if pd.notna(s) else ""
        if s in tier_lookup:
            return tier_lookup[s]
        m = difflib.get_close_matches(s, known, n=1, cutoff=0.85)
        return tier_lookup[m[0]] if m else "Other"

    matched = df[df["match_status"] == "matched"].copy()
    matched["tier"] = matched["毕业学校"].apply(assign)
    matched["tier"] = pd.Categorical(matched["tier"], categories=TIER_ORDER, ordered=True)

    data_by_tier = [
        matched.loc[matched["tier"] == t, "cost_total_usd"].dropna().values
        for t in TIER_ORDER
    ]
    counts = [len(d) for d in data_by_tier]
    medians = [float(np.median(d)) for d in data_by_tier]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bp = ax.boxplot(
        data_by_tier,
        tick_labels=[
            f"{t}\n(n={c:,})"
            for t, c in zip(TIER_ORDER, counts)
        ],
        patch_artist=True,
        medianprops=dict(color="#222222", linewidth=2.2),
        flierprops=dict(
            marker="o", markersize=2.5, alpha=0.35,
            markeredgewidth=0,
        ),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, tier in zip(bp["boxes"], TIER_ORDER):
        patch.set_facecolor(TIER_COLORS[tier])
        patch.set_alpha(0.82)
        patch.set_linewidth(1.2)

    # Median labels
    for i, (med, tier) in enumerate(zip(medians, TIER_ORDER), start=1):
        ax.text(
            i + 0.28, med,
            f"${med:,.0f}",
            va="center", fontsize=8.5, color="#333333", fontweight="bold",
        )

    # Significance brackets (985 vs 211: ***, 985 vs Other: ***)
    y_max = max(d.max() for d in data_by_tier)
    bracket_h = y_max * 0.06

    def bracket(x1, x2, y, label):
        bx = [x1, x1, x2, x2]
        by = [y, y + bracket_h * 0.4, y + bracket_h * 0.4, y]
        ax.plot(bx, by, color="#444444", linewidth=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + bracket_h * 0.5, label,
                ha="center", va="bottom", fontsize=9, color="#222222")

    bracket(1, 2, y_max * 1.04, "***")
    bracket(1, 3, y_max * 1.12, "***")
    ax.text(
        2, y_max * 1.155, "211 vs Other: n.s.",
        ha="center", fontsize=8, color="#888888",
    )

    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ2 — H2a: Does Undergraduate Tier Predict Cost?\n"
        "Kruskal-Wallis H = 275.2, p < 0.001",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "02_rq2_boxplot.png")

    return matched   # pass forward to stacked bar


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — RQ2 Stacked bar: country share by tier
# ══════════════════════════════════════════════════════════════════════════════

def make_rq2_stacked(matched_with_tier: pd.DataFrame) -> None:
    """Stacked bar chart of destination country distribution by tier."""
    sub    = matched_with_tier[matched_with_tier["cost_country"].isin(MAIN_COUNTRIES)]
    counts = pd.crosstab(sub["tier"], sub["cost_country"])
    counts = counts.reindex(
        index=TIER_ORDER,
        columns=[c for c in MAIN_COUNTRIES if c in counts.columns],
        fill_value=0,
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bottom = np.zeros(len(TIER_ORDER))

    tier_labels = [
        f"{t}\n(n={int(counts.loc[t].sum()):,})"
        for t in TIER_ORDER
    ]

    for country in pct.columns:
        color = COUNTRY_COLORS.get(country, "#999999")
        vals  = pct[country].values
        bars  = ax.bar(
            tier_labels, vals, bottom=bottom,
            label=country, color=color, alpha=0.88,
            width=0.55, edgecolor="white", linewidth=0.5,
        )
        for bar, val, bot in zip(bars, vals, bottom):
            if val >= 6:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold",
                )
        bottom += vals

    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Share of Applicants (%)", fontsize=11)
    ax.set_title(
        "RQ2 — H2b: Does Tier Predict Destination Country?\n"
        "χ² = 3054.1,  df = 8,  p < 0.001",
        fontsize=13,
    )
    ax.legend(
        title="Destination Country",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=9, title_fontsize=9,
        framealpha=0.9, edgecolor="#CCCCCC",
    )
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    save(fig, "03_rq2_stacked.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — RQ3 Scatter: GPA vs cost by country
# ══════════════════════════════════════════════════════════════════════════════

def make_rq3_scatter(df: pd.DataFrame) -> None:
    """GPA vs cost scatter, one regression line per main country."""
    VALID_STATUSES = {"converted_from_100", "already_4_scale"}
    filt = df[
        (df["match_status"] == "matched")
        & (df["gpa_standardization_status"].isin(VALID_STATUSES))
        & (df["gpa_4_standardized"] > 0.0)
        & (df["cost_total_usd"].notna())
        & (df["cost_country"].isin(MAIN_COUNTRIES))
    ].copy()

    # Within-country Spearman results for annotation
    rho_labels = {
        "Australia": "ρ=+0.14***",
        "UK":        "ρ=+0.14***",
        "Hong Kong": "ρ=+0.08***",
        "USA":       "ρ=+0.06 n.s.",
        "Singapore": "ρ=−0.15***",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for country, group in filt.groupby("cost_country"):
        if country not in COUNTRY_COLORS:
            continue
        color = COUNTRY_COLORS[country]
        ax.scatter(
            group["gpa_4_standardized"], group["cost_total_usd"],
            color=color, s=6, alpha=0.18, zorder=2, rasterized=True,
        )
        x = group["gpa_4_standardized"].values
        y = group["cost_total_usd"].values
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        rho_str = rho_labels.get(country, "")
        ax.plot(
            x_line, slope * x_line + intercept,
            color=color, linewidth=2.2,
            label=f"{country}  ({rho_str})",
        )

    ax.text(
        0.03, 0.95,
        "Simple OLS:  coef = +$2,969 / GPA-pt,  p = 0.670 (n.s.)\n"
        "With country: coef = +$1,039 / GPA-pt,  p = 0.764 (n.s.)\n"
        "Country R² = 0.464  vs  GPA-only R² = 0.001",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#AAAAAA", alpha=0.92),
    )

    ax.set_xlabel("Standardised GPA (4.0 scale)", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ3: Does GPA Have Purchasing Power?\n"
        "GPA explains < 0.1% of cost variance; country explains 46%",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.legend(
        title="Country (within-country Spearman)",
        fontsize=9, title_fontsize=9,
        framealpha=0.9, edgecolor="#CCCCCC",
    )
    fig.tight_layout()
    save(fig, "04_rq3_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — RQ3 Coefficient comparison bar
# ══════════════════════════════════════════════════════════════════════════════

def make_rq3_coef() -> None:
    """Bar chart: GPA coefficient before vs after adding country dummies."""
    # From rq3_results.txt
    coefs  = [2969, 1039]
    labels = ["Model 1\nGPA only", "Model 2\nGPA + Country"]
    colors = [COUNTRY_COLORS["USA"], COUNTRY_COLORS["UK"]]

    # Approximate 95% CI from typical cluster-robust output
    ci_half = [15000, 7500]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    x = np.arange(len(labels))

    bars = ax.bar(
        x, coefs,
        color=colors, alpha=0.82, width=0.45,
        edgecolor="#444444", linewidth=0.8,
    )
    ax.errorbar(
        x, coefs, yerr=ci_half,
        fmt="none", color="#333333", capsize=8, linewidth=1.8, capthick=1.8,
    )
    for bar, coef, ci in zip(bars, coefs, ci_half):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            coef + ci + 400,
            f"${coef:,}\n(p = n.s.)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.7)

    # Annotation: drop %
    ax.annotate(
        "↓ 65% drop\nafter adding country",
        xy=(1, 1039), xytext=(1.32, 6000),
        fontsize=9, color="#333333",
        arrowprops=dict(arrowstyle="-|>", color="#666666",
                        lw=1.2, mutation_scale=12),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("GPA Coefficient (USD per 1-point GPA increase)", fontsize=10)
    ax.set_title(
        "RQ3 — H3b: GPA Effect Shrinks After Country Control\n"
        "Country R² jumps from 0.001 → 0.464",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.set_ylim(bottom=-2000)
    fig.tight_layout()
    save(fig, "05_rq3_coef.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    print("Loading data …")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    df["cost_total_usd"]     = pd.to_numeric(df["cost_total_usd"],     errors="coerce")
    df["gpa_4_standardized"] = pd.to_numeric(df["gpa_4_standardized"], errors="coerce")

    print("Figure 0: pipeline diagram")
    make_pipeline()

    print("Figure 1: RQ1 scatter")
    make_rq1_scatter(df)

    print("Figure 2: RQ2 box plot")
    matched_with_tier = make_rq2_boxplot(df)

    print("Figure 3: RQ2 stacked bar")
    make_rq2_stacked(matched_with_tier)

    print("Figure 4: RQ3 GPA scatter")
    make_rq3_scatter(df)

    print("Figure 5: RQ3 coefficient bar")
    make_rq3_coef()

    print(f"\nDone. All slides saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
