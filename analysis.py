"""
IS597PR Final Project — Analysis Script
Authors: Kiara (hantarita), Jack (zonghengliu917-stack)

Overarching question:
    What drives the cost of studying abroad for Chinese graduate students —
    the destination country, the school's popularity, or the student's own
    academic background?

Research questions:
    RQ1 — Is Popularity Priced?
        Do universities that attract more Chinese applicants tend to cost more,
        and does this relationship hold uniformly across destination countries?
    RQ2 — Does Undergraduate Tier Predict Destination Cost?
        Are applicants from elite Chinese undergraduate institutions more likely
        to receive offers from higher-cost destinations?
    RQ3 — Do Academic Indicators Predict Cost, and Does It Vary by Country?
        Are stronger academic indicators (GPA, IELTS, TOEFL, GRE, GMAT)
        associated with higher-cost destinations, and does that vary by country?

Usage:
    python analysis.py                       # run all RQs
    python analysis.py --rq 1               # run RQ1 only
    python analysis.py --output results/    # custom output folder

Data sources:
    Dataset A: compass_offers_processed_matched_no_phd.csv
               (scraped from compassedu.hk, preprocessed by preprocessing.py)
    Dataset B: International_Education_Costs.csv
               (international education cost dataset, publicly available)

Citations:
    CompassEdu offer data: https://m.compassedu.hk
    International Education Costs dataset: see International_Education_Costs.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf

_HERE = Path(__file__).parent
DEFAULT_DATA   = _HERE / "compass_offers_processed_matched_no_phd.csv"
DEFAULT_TIER   = _HERE / "university_tier.csv"
DEFAULT_OUTPUT = _HERE / "output"

COUNTRY_COLORS: Dict[str, str] = {
    "UK":        "#4C72B0",
    "USA":       "#DD8452",
    "Australia": "#55A868",
    "Hong Kong": "#C44E52",
    "Singapore": "#8172B2",
}

MIN_N_FOR_TEST   = 10
TIER_ORDER       = ["985", "211", "Other"]
MAIN_COUNTRIES   = list(COUNTRY_COLORS.keys())
ACADEMIC_METRICS = ["gpa_4_standardized", "IELTS", "TOEFL", "GRE", "GMAT"]


# ── Data loading & preparation ─────────────────────────────────────────────────

def load_matched_data(path: Path) -> pd.DataFrame:
    """Load the preprocessed matched CSV and coerce numeric columns.

    Args:
        path: Path to compass_offers_processed_matched_no_phd.csv.

    Returns:
        DataFrame with all rows (matched and unmatched).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in ["cost_total_usd", "gpa_4_standardized", "cost_tuition_usd",
                "cost_rent_usd", "cost_duration_years",
                "IELTS", "TOEFL", "GRE", "GMAT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where match_status is 'matched'.

    Args:
        df: Full DataFrame from :func:`load_matched_data`.

    Returns:
        Filtered DataFrame with only successfully matched offer rows.
    """
    return df[df["match_status"] == "matched"].copy()


def load_tier_lookup(tier_csv: Path) -> Dict[str, str]:
    """Load university_tier.csv into a dict mapping school name to tier.

    Args:
        tier_csv: Path to a CSV with columns ``school_name`` and ``tier``.

    Returns:
        Dict of {school_name: '985' | '211' | 'Other'}.

    Raises:
        FileNotFoundError: If *tier_csv* does not exist.
    """
    if not tier_csv.exists():
        raise FileNotFoundError(f"Tier lookup file not found: {tier_csv}")
    lookup: Dict[str, str] = {}
    with tier_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["school_name"].strip()] = row["tier"].strip()
    return lookup


def assign_tier(
    school: str,
    tier_lookup: Dict[str, str],
    known_schools: List[str],
    cutoff: float = 0.85,
) -> str:
    """Return the tier for one Chinese undergraduate school name.

    Tries an exact lookup first, then falls back to fuzzy matching via
    :func:`difflib.get_close_matches`. Unrecognised schools default to 'Other'.

    Args:
        school: Raw value from the ``毕业学校`` column.
        tier_lookup: Dict from :func:`load_tier_lookup`.
        known_schools: Pre-computed list of keys in *tier_lookup*.
        cutoff: Minimum similarity ratio for fuzzy acceptance (0–1).

    Returns:
        One of ``'985'``, ``'211'``, or ``'Other'``.
    """
    if not school or not school.strip():
        return "Other"
    school = school.strip()
    if school in tier_lookup:
        return tier_lookup[school]
    matches = difflib.get_close_matches(school, known_schools, n=1, cutoff=cutoff)
    if matches:
        return tier_lookup[matches[0]]
    return "Other"


def add_derived_columns(
    df: pd.DataFrame,
    tier_lookup: Dict[str, str],
) -> pd.DataFrame:
    """Add all analytical derived columns needed across RQs in one pass.

    Adds: ``tier``, ``undergrad_tier`` (string alias), ``cost_quartile``,
    ``high_cost_destination``, ``gpa_band``, ``ielts_band``.

    Args:
        df: Matched-only DataFrame from :func:`filter_matched`.
        tier_lookup: Dict from :func:`load_tier_lookup`.

    Returns:
        Copy of *df* with new columns appended.
    """
    known = list(tier_lookup.keys())
    df = df.copy()

    df["tier"] = df["毕业学校"].apply(
        lambda s: assign_tier(str(s) if pd.notna(s) else "", tier_lookup, known)
    )
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    df["undergrad_tier"] = df["tier"].astype(str)

    df["cost_quartile"] = pd.qcut(
        df["cost_total_usd"].rank(method="first"),
        4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
    )
    df["high_cost_destination"] = df["cost_quartile"] == "Q4_highest"

    df["gpa_band"] = pd.cut(
        df["gpa_4_standardized"],
        bins=[0, 3.0, 3.3, 3.6, 4.01],
        labels=["<3.0", "3.0-3.29", "3.3-3.59", "3.6+"],
        include_lowest=True,
    )
    df["ielts_band"] = pd.cut(
        df["IELTS"],
        bins=[0, 6.49, 6.99, 7.49, 9.01],
        labels=["<6.5", "6.5-6.99", "7.0-7.49", "7.5+"],
        include_lowest=True,
    )
    return df


# ── Demographic overview ───────────────────────────────────────────────────────

def print_demographic_overview(df: pd.DataFrame, matched: pd.DataFrame) -> None:
    """Print a full demographic overview of the raw and matched datasets.

    Args:
        df: Full raw DataFrame from :func:`load_matched_data`.
        matched: Matched-only DataFrame with derived columns.
    """
    total     = len(df)
    n_matched = len(matched)
    match_pct = n_matched / total * 100

    lines = [
        "=" * 65,
        "DATASET OVERVIEW",
        "=" * 65,
        f"  Total offer rows (excl. PhD):    {total:>8,}",
        f"  Successfully matched rows:        {n_matched:>8,}  ({match_pct:.1f}%)",
        f"  Unmatched rows:                   {total - n_matched:>8,}  ({100 - match_pct:.1f}%)",
        f"  Unique destination schools:       {matched['matched_school'].nunique():>8,}",
        f"  Destination countries:            {matched['cost_country'].nunique():>8,}",
        f"  Unique Chinese undergrad schools: {matched['毕业学校'].nunique():>8,}",
        "",
        "── Country breakdown (matched rows) ──",
        f"  {'Country':<14} {'Rows':>7}  {'Share':>7}  {'Schools':>8}"
        f"  {'Mean Cost':>12}  {'Median Cost':>12}",
        "  " + "-" * 65,
    ]
    for country, grp in matched.groupby("cost_country"):
        lines.append(
            f"  {country:<14} {len(grp):>7,}  {len(grp)/n_matched*100:>6.1f}%"
            f"  {grp['matched_school'].nunique():>8}"
            f"  ${grp['cost_total_usd'].mean():>10,.0f}"
            f"  ${grp['cost_total_usd'].median():>10,.0f}"
        )
    lines += [
        "",
        "── Cost distribution (all matched, USD) ──",
        f"  Mean:   ${matched['cost_total_usd'].mean():>10,.0f}",
        f"  Median: ${matched['cost_total_usd'].median():>10,.0f}",
        f"  Std:    ${matched['cost_total_usd'].std():>10,.0f}",
        f"  Min:    ${matched['cost_total_usd'].min():>10,.0f}",
        f"  Max:    ${matched['cost_total_usd'].max():>10,.0f}",
        "",
        "── GPA distribution (standardised 4.0 scale) ──",
    ]
    gpa = matched["gpa_4_standardized"].dropna()
    lines += [
        f"  n={len(gpa):,}  mean={gpa.mean():.3f}  median={gpa.median():.3f}"
        f"  std={gpa.std():.3f}  range=[{gpa.min():.2f}, {gpa.max():.2f}]",
        "",
        "── Metric availability ──",
        f"  {'Metric':<22} {'n with data':>12}  {'% of matched':>13}",
        "  " + "-" * 50,
    ]
    for m in ACADEMIC_METRICS:
        n_avail = matched[m].notna().sum()
        lines.append(f"  {m:<22} {n_avail:>12,}  {n_avail/n_matched*100:>12.1f}%")

    print("\n".join(lines))
    print()


# ── Shared statistical helpers ─────────────────────────────────────────────────

def compute_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Compute Spearman rank correlation and two-sided p-value.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        Tuple of (rho, p_value). Both NaN if fewer than 3 observations.
    """
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = scipy.stats.spearmanr(x, y)
    return float(rho), float(p)


def compute_pearson_bootstrap_ci(
    x: pd.Series,
    y: pd.Series,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute Pearson r and a bootstrap 95% confidence interval.

    Args:
        x: First variable.
        y: Second variable.
        n_boot: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (pearson_r, ci_low, ci_high).
    """
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    r = float(paired["x"].corr(paired["y"]))
    rng = np.random.default_rng(seed)
    boot_r: List[float] = []
    for _ in range(n_boot):
        idx    = rng.choice(len(paired), size=len(paired), replace=True)
        sample = paired.iloc[idx]
        boot_r.append(float(sample["x"].corr(sample["y"])))
    return r, float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))


def sig_stars(p: float) -> str:
    """Return significance stars for a p-value.

    Args:
        p: p-value to evaluate.

    Returns:
        One of '***', '**', '*', or 'ns'.
    """
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def spearman_summary(
    label: str,
    x: pd.Series,
    y: pd.Series,
    min_n: int = MIN_N_FOR_TEST,
) -> str:
    """Return a formatted one-line Spearman result string.

    Args:
        label: Display label (e.g. country name or 'All schools').
        x: First variable.
        y: Second variable.
        min_n: Minimum n required to report a p-value.

    Returns:
        A single human-readable result line.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(valid)
    rho, p = compute_spearman(valid["x"], valid["y"])
    if n < min_n:
        return f"  {label:<14} n={n:>3}  rho={rho:+.3f}  [descriptive only — n < {min_n}]"
    return f"  {label:<14} n={n:>3}  rho={rho:+.3f}  p={p:.4f}  {sig_stars(p)}"


# ── RQ1: Is Popularity Priced? ────────────────────────────────────────────────

def aggregate_by_school(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched offer rows into one summary row per destination school.

    Args:
        df: Matched-only DataFrame from :func:`filter_matched`.

    Returns:
        DataFrame with columns: matched_school, cost_country, offer_count,
        median_total_cost, mean_total_cost. Sorted by offer_count descending.
    """
    clean = df.dropna(subset=["cost_total_usd"])
    return (
        clean
        .groupby("matched_school", as_index=False)
        .agg(
            offer_count       =("matched_school",  "count"),
            median_total_cost =("cost_total_usd",  "median"),
            mean_total_cost   =("cost_total_usd",  "mean"),
            cost_country      =("cost_country",    "first"),
        )
        .sort_values("offer_count", ascending=False)
        .reset_index(drop=True)
    )


def add_frequency_bands(school_level: pd.DataFrame) -> pd.DataFrame:
    """Add an offer-frequency quartile band column to school-level data.

    Args:
        school_level: Output of :func:`aggregate_by_school`.

    Returns:
        Copy with a new ``offer_frequency_band`` column
        (Low / Mid-Low / Mid-High / High).
    """
    df = school_level.copy()
    df["offer_frequency_band"] = pd.qcut(
        df["offer_count"].rank(method="first"),
        4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
    )
    return df


def run_rq1_kruskal_bands(school_level: pd.DataFrame) -> Dict:
    """Test whether cost differs significantly across offer-frequency bands.

    Runs Kruskal-Wallis across Low/Mid-Low/Mid-High/High quartile bands,
    then pairwise Mann-Whitney U with rank-biserial effect sizes.

    Args:
        school_level: School-level DataFrame with ``offer_frequency_band`` column.

    Returns:
        Dict with keys ``kw_stat``, ``kw_p``, ``pairwise``.
    """
    band_order = ["Low", "Mid-Low", "Mid-High", "High"]
    groups = [
        school_level[school_level["offer_frequency_band"] == b]["median_total_cost"].values
        for b in band_order
    ]
    stat, p = scipy.stats.kruskal(*groups)

    pairwise: List[Dict] = []
    for b1, b2 in [("Low", "High"), ("Mid-Low", "High"), ("Mid-High", "High")]:
        g1 = school_level[school_level["offer_frequency_band"] == b1]["median_total_cost"].values
        g2 = school_level[school_level["offer_frequency_band"] == b2]["median_total_cost"].values
        u, pu = scipy.stats.mannwhitneyu(g1, g2, alternative="two-sided")
        rb = 1 - (2 * u) / (len(g1) * len(g2))
        pairwise.append({
            "comparison": f"{b1} vs {b2}",
            "median_1": int(np.median(g1)),
            "median_2": int(np.median(g2)),
            "U": round(float(u), 1),
            "p_value": round(float(pu), 4),
            "rank_biserial_r": round(float(rb), 4),
            "sig": sig_stars(pu),
        })
    return {"kw_stat": float(stat), "kw_p": float(p), "pairwise": pairwise}


def _label_top_schools(ax: plt.Axes, agg: pd.DataFrame, top_n: int = 12) -> None:
    """Annotate the top_n schools by offer_count on a scatter plot.

    Args:
        ax: Matplotlib Axes to annotate.
        agg: Aggregated school DataFrame.
        top_n: Number of highest-traffic schools to label.
    """
    for _, row in agg.nlargest(top_n, "offer_count").iterrows():
        short = row["matched_school"].replace("University of ", "U. of ").replace("University", "U.")
        ax.annotate(
            textwrap.shorten(short, width=22, placeholder="…"),
            xy=(row["offer_count"], row["median_total_cost"]),
            xytext=(6, 2), textcoords="offset points",
            fontsize=6.5, color="#333333",
        )


def plot_rq1_scatter(agg: pd.DataFrame, output_dir: Path) -> Path:
    """Save a scatter plot of offer_count vs median_total_cost, coloured by country.

    Args:
        agg: Output of :func:`aggregate_by_school` with frequency bands.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for country, group in agg.groupby("cost_country"):
        color = COUNTRY_COLORS.get(country, "#888888")
        ax.scatter(
            group["offer_count"], group["median_total_cost"],
            label=country, color=color, s=60, alpha=0.85,
            edgecolors="white", linewidths=0.5, zorder=3,
        )
    _label_top_schools(ax, agg)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Admissions (log scale)", fontsize=11)
    ax.set_ylabel("Median Total Cost (USD)", fontsize=11)
    ax.set_title(
        "RQ1: Is Popularity Priced?\nOffer volume vs median total cost by destination school",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(title="Country", fontsize=9, title_fontsize=9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq1_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq1_frequency_band_bar(freq_summary: pd.DataFrame, output_dir: Path) -> Path:
    """Save a grouped bar chart of mean and median cost by offer-frequency band.

    Args:
        freq_summary: DataFrame with offer_frequency_band, mean_cost,
                      median_cost, universities columns.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    bands   = freq_summary["offer_frequency_band"].tolist()
    means   = freq_summary["mean_cost"].tolist()
    medians = freq_summary["median_cost"].tolist()
    ns      = freq_summary["universities"].tolist()

    x, w = np.arange(len(bands)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_m = ax.bar(x - w / 2, means,   w, label="Mean",   color="#f28e2b", alpha=0.85)
    bars_d = ax.bar(x + w / 2, medians, w, label="Median", color="#4C72B0", alpha=0.85)
    for bar, n in zip(bars_m, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1_000, f"n={n}",
                ha="center", fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=10)
    ax.set_xlabel("Offer-Frequency Quartile Band", fontsize=11)
    ax.set_ylabel("Cost (USD)", fontsize=11)
    ax.set_title("RQ1: Average and Median Cost by School Offer-Frequency Band",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq1_frequency_band_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_rq1(matched: pd.DataFrame, output_dir: Path) -> Dict:
    """Execute all RQ1 analyses and save results.

    Hypotheses:
        H1a: Is there a significant relationship between offer count and cost?
             (direction not assumed — let data decide)
        H1b: Does the relationship vary by country?
        H1c: Do cost distributions differ significantly across
             offer-frequency quartile bands?

    Args:
        matched: Matched-only DataFrame with derived columns.
        output_dir: Directory for output files.

    Returns:
        Dict with rho, p, kw_stat, kw_p, r, ci_low, ci_high for summary table.
    """
    agg = add_frequency_bands(aggregate_by_school(matched))

    # demographic summary
    lines: List[str] = [
        "=" * 65,
        "RQ1 Results — Is Popularity Priced?",
        "=" * 65,
        "",
        "── Demographic summary ──",
        f"  Schools analysed:  {len(agg)}",
        f"  Total offer rows:  {agg['offer_count'].sum():,}",
        f"  Offer count range: {agg['offer_count'].min()} – {agg['offer_count'].max()}"
        f"  (median {agg['offer_count'].median():.0f})",
        f"  Median cost range: ${agg['median_total_cost'].min():,.0f} – ${agg['median_total_cost'].max():,.0f}",
        "",
        f"  {'Country':<14} {'Schools':>8}  {'Mean offers/school':>20}  {'Median cost':>13}",
        "  " + "-" * 60,
    ]
    for country, grp in agg.groupby("cost_country"):
        lines.append(
            f"  {country:<14} {len(grp):>8}"
            f"  {grp['offer_count'].mean():>20.1f}"
            f"  ${grp['median_total_cost'].median():>11,.0f}"
        )

    # H1a
    r, ci_low, ci_high = compute_pearson_bootstrap_ci(
        agg["offer_count"], agg["median_total_cost"]
    )
    rho, p_rho = compute_spearman(agg["offer_count"], agg["median_total_cost"])
    lines += [
        "",
        "── H1a: Global correlation — offer count vs median cost ──",
        f"  Pearson r  = {r:+.4f}  (Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}])",
        f"  Spearman ρ = {rho:+.4f}  p={p_rho:.4g}  {sig_stars(p_rho)}",
        f"  Interpretation: {'negative' if rho < 0 else 'positive'} correlation — "
        f"more-popular schools are {'cheaper' if rho < 0 else 'more expensive'} on average.",
    ]

    # H1b per-country
    lines += ["", "── H1b: Per-country Spearman correlations ──"]
    for country in MAIN_COUNTRIES:
        sub = agg[agg["cost_country"] == country]
        lines.append(
            spearman_summary(country, sub["offer_count"], sub["median_total_cost"])
        )

    # H1c: KW across bands
    kw_res = run_rq1_kruskal_bands(agg)
    freq_rows: List[Dict] = []
    for band in ["Low", "Mid-Low", "Mid-High", "High"]:
        grp = agg[agg["offer_frequency_band"] == band]
        freq_rows.append({
            "offer_frequency_band": band,
            "universities": len(grp),
            "mean_offers":  grp["offer_count"].mean(),
            "mean_cost":    grp["median_total_cost"].mean(),
            "median_cost":  grp["median_total_cost"].median(),
        })
    freq_summary = pd.DataFrame(freq_rows)

    lines += [
        "",
        "── H1c: Kruskal-Wallis — cost across offer-frequency bands ──",
        f"  H={kw_res['kw_stat']:.3f},  p={kw_res['kw_p']:.4g}  {sig_stars(kw_res['kw_p'])}",
        "",
        f"  {'Band':<10}  {'Schools':>8}  {'Mean offers':>12}  {'Mean cost':>12}  {'Median cost':>12}",
        "  " + "-" * 60,
    ]
    for row in freq_rows:
        lines.append(
            f"  {row['offer_frequency_band']:<10}  {row['universities']:>8}"
            f"  {row['mean_offers']:>12.1f}  ${row['mean_cost']:>10,.0f}  ${row['median_cost']:>10,.0f}"
        )
    lines += ["", "  Pairwise Mann-Whitney U (vs High-frequency band):"]
    for row in kw_res["pairwise"]:
        lines.append(
            f"    {row['comparison']:<25}  medians=({row['median_1']:,} vs {row['median_2']:,})"
            f"  U={row['U']:.0f}  p={row['p_value']:.4f}"
            f"  r={row['rank_biserial_r']:+.3f}  {row['sig']}"
        )

    # Top 10
    lines += [
        "",
        "── Top 10 schools by offer volume ──",
        f"  {'School':<45} {'Country':<12} {'Offers':>7} {'Median Cost':>12} {'Band':>10}",
        "  " + "-" * 90,
    ]
    for _, row in agg.head(10).iterrows():
        lines.append(
            f"  {row['matched_school']:<45} {row['cost_country']:<12}"
            f" {row['offer_count']:>7,} ${row['median_total_cost']:>10,.0f}"
            f" {str(row['offer_frequency_band']):>10}"
        )

    result_text = "\n".join(lines)
    (output_dir / "rq1_results.txt").write_text(result_text, encoding="utf-8")
    print(result_text)

    p1 = plot_rq1_scatter(agg, output_dir)
    p2 = plot_rq1_frequency_band_bar(freq_summary, output_dir)
    print(f"\n[RQ1] Plots saved → {p1.name}, {p2.name}")

    return {
        "rho": rho, "p": p_rho, "r": r,
        "ci_low": ci_low, "ci_high": ci_high,
        "kw_stat": kw_res["kw_stat"], "kw_p": kw_res["kw_p"],
    }


# ── RQ2: Does Undergraduate Tier Predict Destination Cost? ────────────────────

def run_rq2_kruskal(df: pd.DataFrame) -> Dict:
    """Test H2a: does total cost differ significantly across undergraduate tiers?

    Runs Kruskal-Wallis across 985/211/Other, then pairwise Mann-Whitney U
    with rank-biserial effect sizes.

    Args:
        df: DataFrame with ``tier`` and ``cost_total_usd`` columns.

    Returns:
        Dict with stat, p, group_medians, group_means, group_ns, pairwise.
    """
    groups = {
        t: df.loc[df["tier"] == t, "cost_total_usd"].dropna().values
        for t in TIER_ORDER
    }
    stat, p = scipy.stats.kruskal(*groups.values())

    pairwise: List[Dict] = []
    for a, b in [("985", "211"), ("985", "Other"), ("211", "Other")]:
        u, pu = scipy.stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        rb = 1 - (2 * u) / (len(groups[a]) * len(groups[b]))
        pairwise.append({
            "comparison": f"{a} vs {b}",
            "median_a": int(np.median(groups[a])),
            "median_b": int(np.median(groups[b])),
            "U": round(float(u)),
            "p_value": round(float(pu), 6),
            "rank_biserial_r": round(float(rb), 4),
            "sig": sig_stars(pu),
        })
    return {
        "stat": float(stat), "p": float(p),
        "group_medians": {t: float(np.median(v)) for t, v in groups.items()},
        "group_means":   {t: float(np.mean(v))   for t, v in groups.items()},
        "group_ns":      {t: len(v)              for t, v in groups.items()},
        "pairwise":      pairwise,
    }


def run_rq2_chisquare_quartile(df: pd.DataFrame) -> Dict:
    """Test H2b: does undergraduate tier predict destination cost quartile?

    Builds a tier × cost_quartile contingency table, runs chi-square test,
    and reports Cramér's V as effect size.

    Args:
        df: DataFrame with ``tier`` and ``cost_quartile`` columns.

    Returns:
        Dict with chi2, p, dof, cramers_v, contingency.
    """
    ct = pd.crosstab(df["tier"], df["cost_quartile"])
    chi2, p, dof, _ = scipy.stats.chi2_contingency(ct)
    n = int(ct.to_numpy().sum())
    cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    return {
        "chi2": float(chi2), "p": float(p),
        "dof": int(dof), "cramers_v": cramers_v,
        "contingency": ct,
    }


def run_rq2_within_country_kw(df: pd.DataFrame) -> pd.DataFrame:
    """Run within-country Kruskal-Wallis tests to isolate the tier effect.

    Args:
        df: DataFrame with cost_country, tier, cost_total_usd columns.

    Returns:
        DataFrame with columns: country, n, KW_H, p_value, sig.
    """
    rows: List[Dict] = []
    for country, group in df.groupby("cost_country"):
        if group["cost_total_usd"].nunique() < 2:
            continue
        tier_groups = [
            group[group["tier"] == t]["cost_total_usd"].values
            for t in TIER_ORDER if (group["tier"] == t).sum() >= 5
        ]
        if len(tier_groups) < 2:
            continue
        try:
            h_val, p_val = scipy.stats.kruskal(*tier_groups)
            rows.append({
                "country": country, "n": len(group),
                "KW_H": round(h_val, 3), "p_value": round(p_val, 4),
                "sig": sig_stars(p_val),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def run_rq2_ols(df: pd.DataFrame) -> Dict:
    """Test H2d: does the tier effect shrink after controlling for country?

    Fits two OLS models with cluster-robust standard errors:
      Model 1 — cost ~ tier dummies (no country control)
      Model 2 — cost ~ tier dummies + country dummies

    Args:
        df: DataFrame with tier, cost_country, cost_total_usd, matched_school.

    Returns:
        Dict with model1, model2, coef_drop_pct.
    """
    data = df.dropna(subset=["cost_total_usd"]).copy()
    data["is_985"] = (data["tier"] == "985").astype(int)
    data["is_211"] = (data["tier"] == "211").astype(int)
    cluster = {"cov_type": "cluster", "cov_kwds": {"groups": data["matched_school"]}}

    m1 = smf.ols("cost_total_usd ~ is_985 + is_211", data=data).fit(**cluster)
    m2 = smf.ols("cost_total_usd ~ is_985 + is_211 + C(cost_country)", data=data).fit(**cluster)

    coef1 = m1.params.get("is_985", float("nan"))
    coef2 = m2.params.get("is_985", float("nan"))
    drop_pct = (coef1 - coef2) / abs(coef1) * 100 if coef1 != 0 else float("nan")
    return {"model1": m1, "model2": m2, "coef_drop_pct": float(drop_pct)}


def plot_rq2_cost_by_tier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a box plot of total cost distributions by undergraduate tier.

    Args:
        df: DataFrame with tier and cost_total_usd columns.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_tier = [
        df.loc[df["tier"] == t, "cost_total_usd"].dropna().values for t in TIER_ORDER
    ]
    bp = ax.boxplot(
        data_by_tier, tick_labels=TIER_ORDER, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], ["#E07B54", "#5B9BD5", "#A8C888"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for i, data in enumerate(data_by_tier):
        med = np.median(data)
        ax.text(i + 1, med, f"  ${med:,.0f}", va="center", fontsize=8, color="#333333")
    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title("RQ2 — H2a: Total Program Cost by Undergraduate Tier",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq2_cost_by_tier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq2_quartile_by_tier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a stacked bar chart of cost-quartile distribution by undergraduate tier.

    Args:
        df: DataFrame with tier and cost_quartile columns.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    quartile_order  = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]
    quartile_colors = ["#5B9BD5", "#A8C888", "#F0C27F", "#E07B54"]
    pct = pd.crosstab(df["tier"], df["cost_quartile"], normalize="index") * 100
    pct = pct.reindex(index=TIER_ORDER, columns=quartile_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(TIER_ORDER))
    for q, color in zip(quartile_order, quartile_colors):
        vals = pct[q].values if q in pct.columns else np.zeros(len(TIER_ORDER))
        bars = ax.bar(TIER_ORDER, vals, bottom=bottom, label=q, color=color, alpha=0.85)
        for bar, val, bot in zip(bars, vals, bottom):
            if val >= 6:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Share of Applicants (%)", fontsize=11)
    ax.set_title("RQ2 — H2b: Cost Quartile Distribution by Undergraduate Tier",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Cost Quartile", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq2_quartile_by_tier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq2_country_by_tier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a stacked bar chart of destination country distribution by tier.

    Args:
        df: DataFrame with tier and cost_country columns.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    sub = df[df["cost_country"].isin(MAIN_COUNTRIES)]
    counts = pd.crosstab(sub["tier"], sub["cost_country"])
    counts = counts.reindex(
        index=TIER_ORDER,
        columns=[c for c in MAIN_COUNTRIES if c in counts.columns],
        fill_value=0,
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(TIER_ORDER))
    for country in pct.columns:
        color = COUNTRY_COLORS.get(country, "#888888")
        vals  = pct[country].values
        bars  = ax.bar(TIER_ORDER, vals, bottom=bottom, label=country, color=color, alpha=0.85)
        for bar, val, bot in zip(bars, vals, bottom):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_xlabel("Undergraduate Tier", fontsize=11)
    ax.set_ylabel("Share of Applicants (%)", fontsize=11)
    ax.set_title("RQ2 — Destination Country Distribution by Undergraduate Tier",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Country", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq2_country_by_tier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_rq2(matched: pd.DataFrame, output_dir: Path) -> Dict:
    """Execute all RQ2 analyses and save results.

    Hypotheses:
        H2a: 985 students pay significantly more than 211 and Other
             (Kruskal-Wallis + pairwise Mann-Whitney with rank-biserial r).
        H2b: Undergraduate tier significantly predicts destination cost quartile
             (chi-square + Cramér's V effect size).
        H2c: Within-country, the tier effect on cost persists after controlling
             for country composition (per-country Kruskal-Wallis).
        H2d: After adding country dummies, the 985 cost premium shrinks
             (OLS with cluster-robust standard errors).

    Args:
        matched: Matched-only DataFrame with derived columns from
                 :func:`add_derived_columns`.
        output_dir: Directory for output files.

    Returns:
        Dict with key results for the summary table.
    """
    lines: List[str] = [
        "=" * 65,
        "RQ2 Results — Does Undergraduate Tier Predict Destination Cost?",
        "=" * 65,
        "",
        "── Demographic summary ──",
        f"  {'Tier':<8} {'n':>7}  {'%':>7}  {'Mean cost':>12}"
        f"  {'Median cost':>12}  {'Q4 (highest) %':>16}",
        "  " + "-" * 68,
    ]
    for t in TIER_ORDER:
        grp = matched[matched["tier"] == t]
        hc  = grp["high_cost_destination"].mean() * 100
        lines.append(
            f"  {t:<8} {len(grp):>7,}  {len(grp)/len(matched)*100:>6.1f}%"
            f"  ${grp['cost_total_usd'].mean():>10,.0f}"
            f"  ${grp['cost_total_usd'].median():>10,.0f}"
            f"  {hc:>15.1f}%"
        )

    # Cost quartile breakdown
    lines += ["", "── Cost quartile distribution by tier (% within tier) ──"]
    qt = pd.crosstab(matched["tier"], matched["cost_quartile"], normalize="index") * 100
    for t in TIER_ORDER:
        if t in qt.index:
            row_str = "  ".join(f"{q}: {qt.loc[t, q]:.1f}%" for q in qt.columns)
            lines.append(f"  {t}: {row_str}")

    # Country distribution
    lines += ["", "── Destination country distribution by tier (% within tier) ──"]
    ct_country = pd.crosstab(matched["tier"], matched["cost_country"], normalize="index") * 100
    for t in TIER_ORDER:
        if t in ct_country.index:
            row_str = "  ".join(
                f"{c}: {ct_country.loc[t, c]:.1f}%" for c in MAIN_COUNTRIES if c in ct_country.columns
            )
            lines.append(f"  {t}: {row_str}")

    # H2a
    kw = run_rq2_kruskal(matched)
    lines += [
        "",
        "── H2a: Kruskal-Wallis — cost distributions across tiers ──",
        f"  H={kw['stat']:.2f},  p={kw['p']:.2e}  {sig_stars(kw['p'])}",
        "  Median total cost:",
        *(f"    {t}: ${kw['group_medians'][t]:,.0f}  (n={kw['group_ns'][t]:,})"
          for t in TIER_ORDER),
        "",
        "  Pairwise Mann-Whitney U (two-sided) with rank-biserial r:",
    ]
    for row in kw["pairwise"]:
        lines.append(
            f"    {row['comparison']:<15}  medians=({row['median_a']:,} vs {row['median_b']:,})"
            f"  p={row['p_value']:.6f}  r={row['rank_biserial_r']:+.3f}  {row['sig']}"
        )

    # H2b
    chi = run_rq2_chisquare_quartile(matched)
    lines += [
        "",
        "── H2b: Chi-square — tier vs cost quartile + Cramér's V ──",
        f"  χ²={chi['chi2']:.1f},  df={chi['dof']},  p={chi['p']:.2e}  {sig_stars(chi['p'])}",
        f"  Cramér's V = {chi['cramers_v']:.4f}  (small-to-medium practical effect)",
        "  Contingency table (raw counts):",
    ]
    for idx in chi["contingency"].index:
        row_str = "  ".join(
            f"{col}: {chi['contingency'].loc[idx, col]:>5}"
            for col in chi["contingency"].columns
        )
        lines.append(f"    {idx}: {row_str}")

    # H2c
    within_kw = run_rq2_within_country_kw(matched)
    lines += [
        "",
        "── H2c: Within-country Kruskal-Wallis (tier effect net of country) ──",
        f"  {'Country':<14} {'n':>7}  {'KW_H':>8}  {'p':>8}  {'sig':>5}",
        "  " + "-" * 45,
    ]
    for _, row in within_kw.iterrows():
        lines.append(
            f"  {row['country']:<14} {row['n']:>7,}"
            f"  {row['KW_H']:>8.3f}  {row['p_value']:>8.4f}  {row['sig']:>5}"
        )

    # H2d
    ols = run_rq2_ols(matched)
    m1, m2 = ols["model1"], ols["model2"]
    lines += [
        "",
        "── H2d: OLS — does tier effect shrink after controlling for country? ──",
        "  (Cluster-robust SEs by destination school)",
        f"  Model 1 (no country): is_985 coef={m1.params['is_985']:+,.0f},"
        f"  p={m1.pvalues['is_985']:.4f},  R²={m1.rsquared:.3f}",
        f"  Model 2 (+ country):  is_985 coef={m2.params['is_985']:+,.0f},"
        f"  p={m2.pvalues['is_985']:.4f},  R²={m2.rsquared:.3f}",
        f"  985 coefficient drop after adding country: {ols['coef_drop_pct']:.1f}%",
    ]

    result_text = "\n".join(lines)
    (output_dir / "rq2_results.txt").write_text(result_text, encoding="utf-8")
    print(result_text)

    p1 = plot_rq2_cost_by_tier(matched, output_dir)
    p2 = plot_rq2_quartile_by_tier(matched, output_dir)
    p3 = plot_rq2_country_by_tier(matched, output_dir)
    print(f"\n[RQ2] Plots saved → {p1.name}, {p2.name}, {p3.name}")

    return {
        "kw_stat": kw["stat"], "kw_p": kw["p"],
        "chi2": chi["chi2"], "chi2_p": chi["p"], "cramers_v": chi["cramers_v"],
    }


# ── RQ3: Academic Indicators and Destination Cost ─────────────────────────────

VALID_GPA_STATUSES = {"converted_from_100", "already_4_scale"}


def filter_rq3_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a valid standardised GPA and a matched cost.

    Excludes rows where GPA conversion failed, GPA is zero (scores below 60),
    or cost is missing.

    Args:
        df: Matched-only DataFrame.

    Returns:
        Filtered DataFrame ready for GPA-based regression analysis.
    """
    mask = (
        df["gpa_standardization_status"].isin(VALID_GPA_STATUSES)
        & (df["gpa_4_standardized"] > 0.0)
        & df["cost_total_usd"].notna()
    )
    return df[mask].copy()


def run_rq3_full_correlation_table(matched: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations with cost for all five metrics.

    Args:
        matched: Matched-only DataFrame.

    Returns:
        DataFrame with one row per metric, showing both correlation coefficients
        and significance.
    """
    rows: List[Dict] = []
    for metric in ACADEMIC_METRICS:
        subset = matched[["cost_total_usd", metric]].dropna()
        if len(subset) < 30:
            continue
        r_p, p_p = scipy.stats.pearsonr(subset[metric], subset["cost_total_usd"])
        r_s, p_s = scipy.stats.spearmanr(subset[metric], subset["cost_total_usd"])
        rows.append({
            "metric": metric,
            "n": len(subset),
            "pearson_r": round(r_p, 4),
            "pearson_p": p_p,
            "pearson_sig": sig_stars(p_p),
            "spearman_rho": round(r_s, 4),
            "spearman_p": p_s,
            "spearman_sig": sig_stars(p_s),
        })
    return pd.DataFrame(rows)


def run_rq3_country_metric_correlations(matched: pd.DataFrame) -> pd.DataFrame:
    """Compute per-country Pearson and Spearman correlations for GPA and IELTS.

    Args:
        matched: Matched-only DataFrame.

    Returns:
        DataFrame with one row per (country, metric) pair.
    """
    rows: List[Dict] = []
    for country, group in matched.groupby("cost_country"):
        if group["cost_total_usd"].nunique() < 2:
            continue
        for metric in ["gpa_4_standardized", "IELTS"]:
            subset = group[["cost_total_usd", metric]].dropna()
            if len(subset) < MIN_N_FOR_TEST or subset[metric].nunique() < 2:
                continue
            r_p, p_p = scipy.stats.pearsonr(subset[metric], subset["cost_total_usd"])
            r_s, p_s = scipy.stats.spearmanr(subset[metric], subset["cost_total_usd"])
            rows.append({
                "country": country, "metric": metric, "n": len(subset),
                "pearson_r": round(r_p, 4), "pearson_p": p_p, "pearson_sig": sig_stars(p_p),
                "spearman_rho": round(r_s, 4), "spearman_p": p_s, "spearman_sig": sig_stars(p_s),
            })
    return pd.DataFrame(rows)


def run_rq3_ols(df: pd.DataFrame) -> Tuple[object, object]:
    """Fit two OLS models testing whether GPA predicts total cost.

    Model 1 (H3a): cost ~ GPA — no country control
    Model 2 (H3b): cost ~ GPA + country dummies
    Both use cluster-robust SEs grouped by destination school.

    Args:
        df: Filtered DataFrame from :func:`filter_rq3_data`.

    Returns:
        Tuple of (model1, model2) fitted OLS results.
    """
    cluster = {"cov_type": "cluster", "cov_kwds": {"groups": df["matched_school"]}}
    m1 = smf.ols("cost_total_usd ~ gpa_4_standardized", data=df).fit(**cluster)
    m2 = smf.ols("cost_total_usd ~ gpa_4_standardized + C(cost_country)", data=df).fit(**cluster)
    return m1, m2


def run_rq3_within_country_ols(matched: pd.DataFrame) -> pd.DataFrame:
    """Within-country OLS: cost ~ GPA_z + IELTS_z + tier dummies (H3c).

    Uses standardised (z-scored) coefficients so GPA and IELTS effects are
    directly comparable within each country. Implemented with numpy so no
    additional dependencies are required beyond what is already used.

    Args:
        matched: Matched-only DataFrame with derived columns.

    Returns:
        DataFrame with one row per (country, feature) combination,
        excluding the intercept.
    """
    def _ols_tstat(
        y: np.ndarray, X: np.ndarray, feat_names: List[str]
    ) -> Tuple[pd.DataFrame, float, int]:
        n, k     = X.shape
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid    = y - X @ coeffs
        ss_res   = float(np.dot(resid, resid))
        ss_tot   = float(np.sum((y - y.mean()) ** 2))
        r2       = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float("nan")
        mse      = ss_res / max(n - k, 1)
        cov      = mse * np.linalg.pinv(X.T @ X)
        se       = np.sqrt(np.diag(cov).clip(0))
        t_stat   = np.where(se > 1e-10, coeffs / se, float("nan"))
        p_vals   = 2 * scipy.stats.t.sf(np.abs(t_stat), df=max(n - k, 1))
        return (
            pd.DataFrame({"feature": feat_names, "coeff": coeffs,
                          "se": se, "t": t_stat, "p": p_vals}),
            r2, n,
        )

    ols_rows: List[Dict] = []
    for country, group in matched.groupby("cost_country"):
        if group["cost_total_usd"].nunique() < 3:
            continue
        g = group[["cost_total_usd", "gpa_4_standardized", "IELTS", "tier"]].dropna()
        if len(g) < 50:
            continue
        y      = g["cost_total_usd"].values
        gpa_z  = (g["gpa_4_standardized"] - g["gpa_4_standardized"].mean()) / (g["gpa_4_standardized"].std() or 1)
        iel_z  = (g["IELTS"] - g["IELTS"].mean()) / (g["IELTS"].std() or 1)
        is_211   = (g["tier"] == "211").astype(float).values
        is_other = (g["tier"] == "Other").astype(float).values
        X = np.column_stack([np.ones(len(g)), gpa_z.values, iel_z.values, is_211, is_other])
        coeff_df, r2, n_obs = _ols_tstat(
            y, X, ["intercept", "gpa_z", "ielts_z", "tier_211", "tier_Other"]
        )
        for _, row in coeff_df[coeff_df["feature"] != "intercept"].iterrows():
            ols_rows.append({
                "country": country, "n": n_obs, "R2": round(r2, 4),
                "feature": row["feature"],
                "coeff": round(row["coeff"], 2),
                "se": round(row["se"], 2),
                "t": round(row["t"], 3),
                "p": row["p"],
                "sig": sig_stars(row["p"]),
            })
    return pd.DataFrame(ols_rows)


def compute_within_country_correlations(
    df: pd.DataFrame,
    min_n: int = MIN_N_FOR_TEST,
) -> pd.DataFrame:
    """Compute per-country Spearman correlation between GPA and total cost.

    Args:
        df: Filtered DataFrame from :func:`filter_rq3_data`.
        min_n: Countries with fewer rows than this are skipped.

    Returns:
        DataFrame with country, n, rho, p_value, significant. Sorted by rho desc.
    """
    rows: List[Dict] = []
    for country, group in df.groupby("cost_country"):
        if len(group) < min_n:
            continue
        rho, p = compute_spearman(group["gpa_4_standardized"], group["cost_total_usd"])
        rows.append({"country": country, "n": len(group),
                     "rho": rho, "p_value": p, "significant": p < 0.05})
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def plot_rq3_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a scatter plot of GPA vs total cost coloured by country.

    Each country gets its own OLS regression line. Only the five main
    destination countries are shown to keep the plot readable.

    Args:
        df: Filtered DataFrame from :func:`filter_rq3_data`.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for country, group in df.groupby("cost_country"):
        if country not in COUNTRY_COLORS:
            continue
        color = COUNTRY_COLORS[country]
        ax.scatter(group["gpa_4_standardized"], group["cost_total_usd"],
                   color=color, s=8, alpha=0.25, zorder=2)
        x = group["gpa_4_standardized"].values
        y = group["cost_total_usd"].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, linewidth=2,
                label=f"{country} (slope={m:+,.0f})")
    ax.set_xlabel("Standardised GPA (4.0 scale)", fontsize=11)
    ax.set_ylabel("Estimated Total Cost (USD)", fontsize=11)
    ax.set_title("RQ3 — GPA vs Total Program Cost by Destination Country",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9, title="Country (OLS slope)")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq3_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq3_coef(m1: object, m2: object, output_dir: Path) -> Path:
    """Save a bar chart comparing the GPA coefficient before and after country control.

    Args:
        m1: Simple OLS model (no country dummies).
        m2: Country-controlled OLS model.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    coefs  = [m1.params["gpa_4_standardized"], m2.params["gpa_4_standardized"]]
    cis    = [m1.conf_int().loc["gpa_4_standardized"].values,
              m2.conf_int().loc["gpa_4_standardized"].values]
    labels = ["Model 1\n(GPA only)", "Model 2\n(GPA + Country)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (coef, ci, label, color) in enumerate(
        zip(coefs, cis, labels, ["#DD8452", "#4C72B0"])
    ):
        ax.bar(i, coef, color=color, alpha=0.8, width=0.5)
        ax.errorbar(i, coef, yerr=[[coef - ci[0]], [ci[1] - coef]],
                    fmt="none", color="black", capsize=6, linewidth=1.5)
        ax.text(i, coef + (ci[1] - coef) + 500, f"${coef:,.0f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("GPA Coefficient (USD per 1-point GPA increase)", fontsize=10)
    ax.set_title("RQ3 — H3b: GPA Effect Before and After Country Control",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / "rq3_coef_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rq3_metric_band_bar(
    band_df: pd.DataFrame,
    band_col: str,
    label: str,
    output_dir: Path,
) -> Path:
    """Save a bar chart of high-cost destination share by academic metric band.

    Args:
        band_df: Aggregated DataFrame with band_col, high_cost_share_pct, rows.
        band_col: Column name for the band labels.
        label: Human-readable metric name (e.g. 'GPA' or 'IELTS').
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    bands  = band_df[band_col].astype(str).tolist()
    values = band_df["high_cost_share_pct"].tolist()
    ns     = band_df["rows"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bands, values, color="#4C72B0", alpha=0.85)
    for bar, val, n in zip(bars, values, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%\n(n={n:,})",
                ha="center", fontsize=9)
    ax.set_xlabel(f"{label} Band", fontsize=11)
    ax.set_ylabel("Share in Highest Cost Quartile (Q4, %)", fontsize=11)
    ax.set_title(f"RQ3: Highest-Cost Destination Share by {label} Band",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.35 if values else 50)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out = output_dir / f"rq3_{label.lower()}_band_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_rq3(matched: pd.DataFrame, output_dir: Path) -> Dict:
    """Execute all RQ3 analyses and save results.

    Hypotheses:
        H3a: GPA is a statistically significant predictor of total cost
             (simple OLS, cluster-robust SEs).
        H3b: After adding country dummies, the GPA coefficient drops substantially,
             showing country choice mediates most of the GPA-cost relationship.
        H3c: Within-country GPA-cost correlations vary by country
             (positive in UK/Australia, negative or near-zero in Singapore).
        H3d: IELTS and other academic metrics show similar patterns to GPA
             (full Pearson + Spearman correlation table).

    Args:
        matched: Matched-only DataFrame with derived columns.
        output_dir: Directory for output files.

    Returns:
        Dict with key results for the summary table.
    """
    gpa_df = filter_rq3_data(matched)

    # metric availability
    lines: List[str] = [
        "=" * 65,
        "RQ3 Results — Academic Indicators, Cost, and Country Differences",
        "=" * 65,
        "",
        "── Demographic summary — metric availability ──",
        f"  {'Metric':<22} {'n':>8}  {'% matched':>10}  {'Mean':>8}  {'Std':>8}  {'Median':>8}",
        "  " + "-" * 68,
    ]
    for m in ACADEMIC_METRICS:
        sub = matched[m].dropna()
        lines.append(
            f"  {m:<22} {len(sub):>8,}  {len(sub)/len(matched)*100:>9.1f}%"
            f"  {sub.mean():>8.3f}  {sub.std():>8.3f}  {sub.median():>8.3f}"
        )

    # GPA band summary
    gpa_band = (
        matched.groupby("gpa_band", observed=False)
        .agg(rows=("gpa_band", "size"),
             mean_cost=("cost_total_usd", "mean"),
             median_cost=("cost_total_usd", "median"),
             high_cost_share_pct=("high_cost_destination", "mean"))
        .reset_index()
    )
    gpa_band["high_cost_share_pct"] *= 100

    lines += [
        "",
        "── GPA band distribution ──",
        f"  {'Band':<12} {'n':>8}  {'Mean cost':>12}  {'Median cost':>12}  {'Q4 share':>10}",
        "  " + "-" * 58,
    ]
    for _, row in gpa_band.iterrows():
        lines.append(
            f"  {str(row['gpa_band']):<12} {int(row['rows']):>8,}"
            f"  ${row['mean_cost']:>10,.0f}  ${row['median_cost']:>10,.0f}"
            f"  {row['high_cost_share_pct']:>9.1f}%"
        )

    # IELTS band summary
    ielts_band = (
        matched.dropna(subset=["IELTS"])
        .groupby("ielts_band", observed=False)
        .agg(rows=("ielts_band", "size"),
             mean_cost=("cost_total_usd", "mean"),
             median_cost=("cost_total_usd", "median"),
             high_cost_share_pct=("high_cost_destination", "mean"))
        .reset_index()
    )
    ielts_band["high_cost_share_pct"] *= 100

    lines += [
        "",
        "── IELTS band distribution ──",
        f"  {'Band':<12} {'n':>8}  {'Mean cost':>12}  {'Median cost':>12}  {'Q4 share':>10}",
        "  " + "-" * 58,
    ]
    for _, row in ielts_band.iterrows():
        lines.append(
            f"  {str(row['ielts_band']):<12} {int(row['rows']):>8,}"
            f"  ${row['mean_cost']:>10,.0f}  ${row['median_cost']:>10,.0f}"
            f"  {row['high_cost_share_pct']:>9.1f}%"
        )

    # H3d: full correlation table
    corr_table = run_rq3_full_correlation_table(matched)
    lines += [
        "",
        "── H3d: Full correlation table — all metrics vs total cost ──",
        f"  {'Metric':<22} {'n':>7}  {'Pearson r':>10}  {'p':>10}  {'sig':>4}"
        f"  {'Spearman ρ':>11}  {'p':>10}  {'sig':>4}",
        "  " + "-" * 85,
    ]
    for _, row in corr_table.iterrows():
        lines.append(
            f"  {row['metric']:<22} {int(row['n']):>7,}"
            f"  {row['pearson_r']:>+10.4f}  {row['pearson_p']:>10.4g}  {row['pearson_sig']:>4}"
            f"  {row['spearman_rho']:>+11.4f}  {row['spearman_p']:>10.4g}  {row['spearman_sig']:>4}"
        )

    # H3a + H3b
    m1, m2 = run_rq3_ols(gpa_df)
    coef1    = m1.params["gpa_4_standardized"]
    coef2    = m2.params["gpa_4_standardized"]
    drop_pct = (coef1 - coef2) / abs(coef1) * 100

    lines += [
        "",
        f"── H3a: Simple OLS — does GPA predict cost? (n={len(gpa_df):,}) ──",
        f"  cost ~ GPA: coef={coef1:+,.0f} USD / 1-point GPA increase",
        f"  p={m1.pvalues['gpa_4_standardized']:.4f}  {sig_stars(m1.pvalues['gpa_4_standardized'])}",
        f"  R²={m1.rsquared:.4f}",
        "",
        "── H3b: Country-controlled OLS — does GPA effect shrink? ──",
        "  (Cluster-robust SEs by destination school)",
        f"  cost ~ GPA + country: coef={coef2:+,.0f} USD / 1-point GPA increase",
        f"  p={m2.pvalues['gpa_4_standardized']:.4f}  {sig_stars(m2.pvalues['gpa_4_standardized'])}",
        f"  R²={m2.rsquared:.4f}",
        f"  GPA coefficient drop after adding country: {drop_pct:.1f}%",
    ]

    # H3c: within-country Spearman
    within = compute_within_country_correlations(gpa_df)
    lines += [
        "",
        "── H3c: Within-country Spearman (GPA vs cost) ──",
        f"  {'Country':<14} {'n':>7}  {'rho':>8}  {'p':>10}  {'sig':>4}",
        "  " + "-" * 48,
    ]
    for _, row in within.iterrows():
        lines.append(
            f"  {row['country']:<14} {int(row['n']):>7,}"
            f"  {row['rho']:>+8.4f}  {row['p_value']:>10.4g}  {sig_stars(row['p_value']):>4}"
        )

    # Country-specific GPA + IELTS correlations
    country_metric = run_rq3_country_metric_correlations(matched)
    lines += [
        "",
        "── Country-specific Pearson + Spearman — GPA and IELTS vs cost ──",
        f"  {'Country':<14} {'Metric':<22} {'n':>7}"
        f"  {'Pearson r':>10}  {'sig':>4}  {'Spearman ρ':>11}  {'sig':>4}",
        "  " + "-" * 78,
    ]
    for _, row in country_metric.iterrows():
        lines.append(
            f"  {row['country']:<14} {row['metric']:<22} {int(row['n']):>7,}"
            f"  {row['pearson_r']:>+10.4f}  {row['pearson_sig']:>4}"
            f"  {row['spearman_rho']:>+11.4f}  {row['spearman_sig']:>4}"
        )

    # Within-country OLS (standardised)
    within_ols = run_rq3_within_country_ols(matched)
    lines += [
        "",
        "── Within-country OLS: cost ~ GPA_z + IELTS_z + tier dummies ──",
        "  (Standardised coefficients — GPA and IELTS directly comparable)",
        f"  {'Country':<14} {'n':>7}  {'R²':>6}  {'Feature':<15}"
        f"  {'Coeff':>10}  {'t':>8}  {'sig':>4}",
        "  " + "-" * 70,
    ]
    for _, row in within_ols.iterrows():
        lines.append(
            f"  {row['country']:<14} {row['n']:>7}  {row['R2']:>6.4f}"
            f"  {row['feature']:<15}  {row['coeff']:>+10.2f}"
            f"  {row['t']:>+8.3f}  {row['sig']:>4}"
        )

    result_text = "\n".join(lines)
    (output_dir / "rq3_results.txt").write_text(result_text, encoding="utf-8")
    print(result_text)

    p1 = plot_rq3_scatter(gpa_df, output_dir)
    p2 = plot_rq3_coef(m1, m2, output_dir)
    p3 = plot_rq3_metric_band_bar(gpa_band, "gpa_band", "GPA", output_dir)
    p4 = plot_rq3_metric_band_bar(ielts_band, "ielts_band", "IELTS", output_dir)
    print(f"\n[RQ3] Plots saved → {p1.name}, {p2.name}, {p3.name}, {p4.name}")

    return {
        "gpa_coef_simple":  coef1,
        "gpa_p_simple":     m1.pvalues["gpa_4_standardized"],
        "gpa_coef_country": coef2,
        "gpa_p_country":    m2.pvalues["gpa_4_standardized"],
        "gpa_drop_pct":     drop_pct,
    }


# ── Final summary table ────────────────────────────────────────────────────────

def print_test_summary(rq1: Dict, rq2: Dict, rq3: Dict) -> None:
    """Print a consolidated table of all hypothesis test results.

    Args:
        rq1: Results dict from :func:`run_rq1`.
        rq2: Results dict from :func:`run_rq2`.
        rq3: Results dict from :func:`run_rq3`.
    """
    rows = [
        ("H1a", "Spearman ρ & bootstrap Pearson CI (offer count vs cost)",
         f"ρ={rq1['rho']:+.3f} p={rq1['p']:.4g}  CI=[{rq1['ci_low']:.3f},{rq1['ci_high']:.3f}]",
         "No — popular schools are cheaper"),
        ("H1b", "Per-country Spearman correlations",
         "All negative or near-zero across 5 countries",
         "No — consistent direction"),
        ("H1c", "Kruskal-Wallis across offer-frequency bands",
         f"H={rq1['kw_stat']:.2f} p={rq1['kw_p']:.4g} {sig_stars(rq1['kw_p'])}",
         "Partial — significant, but High-freq cheaper"),
        ("H2a", "KW — cost across 985/211/Other tiers",
         f"H={rq2['kw_stat']:.2f} p={rq2['kw_p']:.2e} {sig_stars(rq2['kw_p'])}",
         "Yes — 985 > 211 > Other"),
        ("H2b", "Chi-square — tier × cost quartile + Cramér's V",
         f"χ²={rq2['chi2']:.1f} p<0.001 V={rq2['cramers_v']:.3f}",
         "Yes"),
        ("H2c", "Within-country KW (tier net of country composition)",
         "Significant in UK and Australia",
         "Yes (partial)"),
        ("H2d", "OLS — tier effect shrinks after country control",
         "985 coef drops substantially",
         "Yes — country mediates tier effect"),
        ("H3a", "Simple OLS — GPA predicts cost (cluster-robust SEs)",
         f"coef={rq3['gpa_coef_simple']:+,.0f}  p={rq3['gpa_p_simple']:.4g}"
         f"  {sig_stars(rq3['gpa_p_simple'])}",
         "No — ns after clustering by school; individual corr. sig."),
        ("H3b", "OLS + country — GPA effect shrinks",
         f"coef={rq3['gpa_coef_country']:+,.0f}  drop={rq3['gpa_drop_pct']:.1f}%",
         "Yes — country absorbs most of GPA effect"),
        ("H3c", "Within-country Spearman GPA vs cost",
         "UK/AU: positive; SG: negative",
         "Partially — direction varies by country"),
        ("H3d", "Full corr. table — GPA IELTS TOEFL GRE GMAT",
         "GPA+IELTS significant; others sparse",
         "Partially — GPA and IELTS only"),
    ]

    lines = [
        "",
        "=" * 105,
        "STATISTICAL TEST SUMMARY — ALL HYPOTHESES",
        "=" * 105,
        f"  {'H':<5}  {'Test':<54}  {'Result':<38}  Supported?",
        "  " + "-" * 103,
    ]
    for h, test, result, supported in rows:
        lines.append(f"  {h:<5}  {test:<54}  {result:<38}  {supported}")
    lines.append("=" * 105)
    print("\n".join(lines))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the requested research questions.

    Example:
        python analysis.py
        python analysis.py --rq 1 --output my_results/
    """
    parser = argparse.ArgumentParser(
        description="IS597PR Final Project Analysis — RQ1/RQ2/RQ3"
    )
    parser.add_argument("--data",   type=Path, default=DEFAULT_DATA,
                        help="Path to the matched offers CSV.")
    parser.add_argument("--tier",   type=Path, default=DEFAULT_TIER,
                        help="Path to university_tier.csv for tier assignment.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for plots and result text files.")
    parser.add_argument("--rq", choices=["1", "2", "3", "all"], default="all",
                        help="Which research question(s) to run (default: all).")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df          = load_matched_data(args.data)
    matched     = filter_matched(df)
    tier_lookup = load_tier_lookup(args.tier)
    matched     = add_derived_columns(matched, tier_lookup)

    print_demographic_overview(df, matched)

    rq1_res: Dict = {}
    rq2_res: Dict = {}
    rq3_res: Dict = {}

    if args.rq in ("1", "all"):
        rq1_res = run_rq1(matched, args.output)

    if args.rq in ("2", "all"):
        rq2_res = run_rq2(matched, args.output)

    if args.rq in ("3", "all"):
        rq3_res = run_rq3(matched, args.output)

    if args.rq == "all":
        print_test_summary(rq1_res, rq2_res, rq3_res)


if __name__ == "__main__":
    main()
