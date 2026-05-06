"""Export RQ3 tables and improved scatter as PNG files for presentation."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (
    load_matched_data, filter_matched, load_tier_lookup, add_derived_columns,
    filter_rq3_data, run_rq3_full_correlation_table,
    run_rq3_country_metric_correlations, run_rq3_within_country_ols,
    run_rq3_ols, sig_stars,
)

OUT = Path("output/slides")
OUT.mkdir(parents=True, exist_ok=True)

df      = load_matched_data(Path("compass_offers_processed_matched_no_phd.csv"))
matched = add_derived_columns(filter_matched(df), load_tier_lookup(Path("university_tier.csv")))
gpa_df  = filter_rq3_data(matched)

COUNTRY_COLOR = {
    "UK": "#4e79a7", "USA": "#f28e2b", "Australia": "#59a14f",
    "Hong Kong": "#e15759", "Singapore": "#af7aa1",
}

# ── helper: save dataframe as table image ─────────────────────────────────────
def save_table(df_in: pd.DataFrame, title: str, fname: str,
               col_widths=None, fontsize=11):
    df_in = df_in.reset_index(drop=True)
    nrows, ncols = df_in.shape
    fig_h = max(1.2 + nrows * 0.42, 2.4)
    fig_w = max(ncols * 1.6, 7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=10)
    tbl = ax.table(
        cellText=df_in.values,
        colLabels=df_in.columns.tolist(),
        cellLoc="center", loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#cccccc")
    fig.tight_layout()
    path = OUT / fname
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")

# ── Table 1: Five-metric correlation table (H3d) ─────────────────────────────
print("Exporting H3d correlation table...")
ct = run_rq3_full_correlation_table(matched)
ct_clean = ct[["metric", "n", "pearson_r", "pearson_sig", "spearman_rho", "spearman_sig"]].copy()
ct_clean.columns = ["Metric", "n", "Pearson r", "Pearson sig", "Spearman ρ", "Spearman sig"]
ct_clean["Metric"] = ct_clean["Metric"].replace("gpa_4_standardized", "GPA")
ct_clean["Pearson r"]   = ct_clean["Pearson r"].round(3)
ct_clean["Spearman ρ"]  = ct_clean["Spearman ρ"].round(3)
save_table(ct_clean, "H3d — Five Academic Metrics vs Total Cost", "rq3_h3d_corr_table.png")

# ── Table 2: Country × GPA band pivot ────────────────────────────────────────
print("Exporting Country × GPA pivot...")
countries = ["UK", "Australia", "Hong Kong", "Singapore", "USA"]
piv = (matched[matched["cost_country"].isin(countries)]
       .groupby(["cost_country", "gpa_band"], observed=False)
       .agg(mean=("cost_total_usd", "mean")).reset_index()
       .pivot(index="cost_country", columns="gpa_band", values="mean")
       .round(0).astype(int))
piv.columns.name = None
piv = piv.reset_index().rename(columns={"cost_country": "Country"})
for c in piv.columns[1:]:
    piv[c] = piv[c].apply(lambda v: f"${v:,}")
save_table(piv, "Mean Cost (USD) by Country × GPA Band  [Singapore reversal]",
           "rq3_gpa_country_pivot.png", col_widths=[0.18, 0.18, 0.18, 0.18, 0.18])

# ── Table 3: Within-country OLS (H3c) ────────────────────────────────────────
print("Exporting within-country OLS table...")
ols_df = run_rq3_within_country_ols(matched)
ols_show = ols_df[ols_df["feature"] != "intercept"].copy()
ols_show["coeff"] = ols_show["coeff"].apply(lambda v: f"${v:+,.0f}")
ols_show["p"]     = ols_show["p"].apply(lambda v: f"{v:.3f}")
ols_show["R²"]    = ols_show["R2"].round(3)
ols_show = ols_show[["country", "feature", "coeff", "p", "sig", "R²"]].copy()
ols_show.columns = ["Country", "Feature", "Coef (USD)", "p-value", "Sig", "R²"]
ols_show["Feature"] = ols_show["Feature"].replace(
    {"gpa_z": "GPA_z", "ielts_z": "IELTS_z",
     "tier_211": "Tier: 211", "tier_Other": "Tier: Other"})
save_table(ols_show, "H3c — Within-Country OLS: cost ~ GPA_z + IELTS_z + Tier",
           "rq3_h3c_ols_table.png", col_widths=[0.16, 0.16, 0.18, 0.14, 0.10, 0.10])

# ── Chart: Improved scatter — GPA band mean cost per country (line chart) ────
print("Exporting improved RQ3 line chart...")
band_order = ["<3.0", "3.0-3.29", "3.3-3.59", "3.6+"]
plot_countries = ["UK", "Australia", "Singapore", "Hong Kong"]

agg = (matched[matched["cost_country"].isin(plot_countries)]
       .groupby(["cost_country", "gpa_band"], observed=False)
       .agg(mean_cost=("cost_total_usd", "mean"), n=("cost_total_usd", "size"))
       .reset_index())
agg["gpa_band"] = pd.Categorical(agg["gpa_band"], categories=band_order, ordered=True)
agg = agg.sort_values("gpa_band")

fig, ax = plt.subplots(figsize=(9, 5.5))
for country in plot_countries:
    sub = agg[agg["cost_country"] == country].sort_values("gpa_band")
    costs = sub["mean_cost"].values
    ax.plot(band_order, costs, marker="o", linewidth=2.5, markersize=8,
            color=COUNTRY_COLOR[country], label=country)
    for band, cost in zip(band_order, costs):
        ax.annotate(f"${cost/1000:.0f}k", xy=(band, cost),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5,
                    color=COUNTRY_COLOR[country], fontweight="bold")

ax.set_xlabel("GPA Band", fontsize=11)
ax.set_ylabel("Mean Total Cost (USD)", fontsize=11)
ax.set_title("RQ3 — Mean Study Cost by GPA Band and Country\n"
             "Singapore reverses while UK/Australia/HK increase",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
ax.legend(fontsize=10, framealpha=0.85)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

# highlight Singapore
ax.annotate("Singapore: higher GPA\n→ cheaper schools ↓",
            xy=("3.6+", agg[(agg["cost_country"]=="Singapore") &
                             (agg["gpa_band"]=="3.6+")]["mean_cost"].values[0]),
            xytext=(-120, -45), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#af7aa1", lw=1.5),
            fontsize=9, color="#af7aa1", fontweight="bold")

fig.tight_layout()
path = OUT / "rq3_gpa_line_chart.png"
fig.savefig(path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {path}")

print("\nDone. All files in output/slides/")
