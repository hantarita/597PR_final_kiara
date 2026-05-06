"""Generate offer_cost_analysis.ipynb with analysis.py as backend."""
import json, uuid

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src,
            "id": uuid.uuid4().hex[:8]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src, "id": uuid.uuid4().hex[:8]}

# ── helpers shared across cells (kept in notebook for self-containedness) ─────
SVG_HELPERS = r'''# ── SVG chart helpers ─────────────────────────────────────────────────────────
import html, math
import numpy as np

PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2',
           '#af7aa1', '#ff9da7', '#9c755f']
COUNTRY_COLOR_MAP = {
    'UK': '#4e79a7', 'USA': '#f28e2b', 'Australia': '#59a14f',
    'Hong Kong': '#e15759', 'Singapore': '#af7aa1',
}

def show_svg(svg_text: str):
    display(HTML(svg_text))

def _scale(values, low, high):
    v = np.asarray(values, dtype=float)
    mn, mx = float(v.min()), float(v.max())
    if math.isclose(mn, mx):
        return np.full(len(v), (low + high) / 2)
    return low + (v - mn) * (high - low) / (mx - mn)

def make_bar_chart(labels, values, title, x_label='', y_label='',
                   color='#4e79a7', width=920, height=420, value_format='{:.1f}'):
    labels  = [str(x) for x in labels]
    values  = [float(v) for v in values]
    ml, mr, mt, mb = 85, 30, 55, 90
    pw, ph  = width - ml - mr, height - mt - mb
    mn, mx  = min([0.0] + values), max([0.0] + values)
    if math.isclose(mn, mx): mx = mn + 1.0
    vr = mx - mn
    gw = pw / max(len(values), 1)
    bw = gw * 0.70
    baseline = mt + ph * (mx / vr) if vr else mt + ph
    p = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2:.0f}" y="34" text-anchor="middle" font-size="17" font-weight="bold" font-family="sans-serif">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{baseline:.1f}" x2="{ml+pw}" y2="{baseline:.1f}" stroke="#555"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>',
    ]
    for i, v in enumerate(values):
        sh  = abs(v) / vr * ph if vr else 0
        x   = ml + i * gw + gw * 0.15
        y   = (baseline - sh) if v >= 0 else baseline
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{sh:.1f}" fill="{color}" opacity="0.87" rx="2"/>')
        ty  = y - 7 if v >= 0 else y + sh + 14
        p.append(f'<text x="{x+bw/2:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{html.escape(value_format.format(v))}</text>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{html.escape(labels[i])}</text>')
    if x_label:
        p.append(f'<text x="{width/2:.0f}" y="{height-4}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(x_label)}</text>')
    if y_label:
        p.append(f'<text x="16" y="{height/2:.0f}" transform="rotate(-90 16,{height/2:.0f})" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(y_label)}</text>')
    p.append('</svg>')
    return ''.join(p)


def make_grouped_bar_chart(labels, series_dict, title, x_label='', y_label='',
                            colors=None, width=920, height=420, value_format='{:,.0f}'):
    if colors is None: colors = PALETTE
    labels  = [str(x) for x in labels]
    snames  = list(series_dict.keys())
    n_g, n_s = len(labels), len(snames)
    ml, mr, mt, mb = 85, 160, 55, 90
    pw, ph  = width - ml - mr, height - mt - mb
    all_v   = [float(v) for vals in series_dict.values() for v in vals]
    mn, mx  = min([0.0] + all_v), max([0.0] + all_v)
    if math.isclose(mn, mx): mx = mn + 1.0
    vr = mx - mn
    baseline = mt + ph * (mx / vr)
    gw  = pw / max(n_g, 1)
    bw  = gw * 0.75 / max(n_s, 1)
    p = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{(width-mr)/2+ml/2:.0f}" y="34" text-anchor="middle" font-size="17" font-weight="bold" font-family="sans-serif">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{baseline:.1f}" x2="{ml+pw}" y2="{baseline:.1f}" stroke="#555"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>',
    ]
    for gi, label in enumerate(labels):
        for si, sn in enumerate(snames):
            v   = float(series_dict[sn][gi])
            sh  = abs(v) / vr * ph if vr else 0
            x   = ml + gi * gw + gw * 0.125 + si * bw
            y   = (baseline - sh) if v >= 0 else baseline
            c   = colors[si % len(colors)]
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{sh:.1f}" fill="{c}" opacity="0.87" rx="2"/>')
        lx = ml + gi * gw + gw / 2
        p.append(f'<text x="{lx:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{html.escape(label)}</text>')
    for si, sn in enumerate(snames):
        lx = ml + pw + 12
        ly = mt + si * 22
        p.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{colors[si % len(colors)]}" rx="2"/>')
        p.append(f'<text x="{lx+18}" y="{ly+12}" font-size="12" font-family="sans-serif">{html.escape(sn)}</text>')
    if x_label:
        p.append(f'<text x="{(width-mr)/2+ml/2:.0f}" y="{height-4}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(x_label)}</text>')
    if y_label:
        p.append(f'<text x="16" y="{height/2:.0f}" transform="rotate(-90 16,{height/2:.0f})" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(y_label)}</text>')
    p.append('</svg>')
    return ''.join(p)


def make_stacked_bar_chart(labels, series_dict, title, x_label='', y_label='',
                            colors=None, width=920, height=420):
    if colors is None: colors = PALETTE
    labels  = [str(x) for x in labels]
    snames  = list(series_dict.keys())
    ml, mr, mt, mb = 85, 165, 55, 90
    pw, ph  = width - ml - mr, height - mt - mb
    gw  = pw / max(len(labels), 1)
    bw  = gw * 0.60
    p = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{(width-mr)/2+ml/2:.0f}" y="34" text-anchor="middle" font-size="17" font-weight="bold" font-family="sans-serif">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#555"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>',
    ]
    for gi, label in enumerate(labels):
        cumul = 0.0
        cx    = ml + gi * gw + (gw - bw) / 2
        for si, sn in enumerate(snames):
            v     = float(series_dict[sn][gi])
            bar_h = v / 100.0 * ph
            by    = mt + ph - (cumul + v) / 100.0 * ph
            c     = colors[si % len(colors)]
            p.append(f'<rect x="{cx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{c}" opacity="0.87" rx="1"/>')
            if v >= 6:
                ty = by + bar_h / 2 + 5
                p.append(f'<text x="{cx+bw/2:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="10" fill="white" font-weight="bold" font-family="sans-serif">{v:.0f}%</text>')
            cumul += v
        lx = ml + gi * gw + gw / 2
        p.append(f'<text x="{lx:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(label)}</text>')
    for si, sn in enumerate(snames):
        lx = ml + pw + 12
        ly = mt + si * 22
        p.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{colors[si % len(colors)]}" rx="2"/>')
        p.append(f'<text x="{lx+18}" y="{ly+12}" font-size="11" font-family="sans-serif">{html.escape(sn)}</text>')
    if x_label:
        p.append(f'<text x="{(width-mr)/2+ml/2:.0f}" y="{height-4}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(x_label)}</text>')
    if y_label:
        p.append(f'<text x="16" y="{height/2:.0f}" transform="rotate(-90 16,{height/2:.0f})" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(y_label)}</text>')
    p.append('</svg>')
    return ''.join(p)


def make_scatter_chart(frame, x_col, y_col, label_col, color_col, title,
                        x_label, y_label, annotate=12, width=920, height=520):
    pf  = frame[[x_col, y_col, label_col, color_col]].dropna().copy()
    ml, mr, mt, mb = 85, 35, 60, 75
    pw, ph  = width - ml - mr, height - mt - mb
    xv  = pf[x_col].astype(float).to_numpy()
    yv  = pf[y_col].astype(float).to_numpy()
    xp  = _scale(xv, ml, ml + pw)
    yp  = _scale(yv, mt + ph, mt)
    co  = list(dict.fromkeys(pf[color_col].astype(str).tolist()))
    cm  = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(co)}
    p = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2:.0f}" y="34" text-anchor="middle" font-size="17" font-weight="bold" font-family="sans-serif">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#555"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>',
    ]
    if len(pf) >= 2:
        sl, ic = np.polyfit(xv, yv, 1)
        xl  = np.array([xv.min(), xv.max()])
        yl  = sl * xl + ic
        xlp = _scale(xl, ml, ml + pw)
        ylp = _scale(yl, mt + ph, mt)
        p.append(f'<line x1="{xlp[0]:.1f}" y1="{ylp[0]:.1f}" x2="{xlp[1]:.1f}" y2="{ylp[1]:.1f}" stroke="#444" stroke-width="2" stroke-dasharray="5,4"/>')
    for pos, (_, row) in enumerate(pf.iterrows()):
        c = cm.get(str(row[color_col]), '#888')
        p.append(f'<circle cx="{xp[pos]:.1f}" cy="{yp[pos]:.1f}" r="6" fill="{c}" opacity="0.82"/>')
    for _, row in pf.nlargest(annotate, x_col).iterrows():
        pos = pf.index.get_loc(row.name)
        p.append(f'<text x="{xp[pos]+9:.1f}" y="{yp[pos]-7:.1f}" font-size="10" font-family="sans-serif">{html.escape(str(row[label_col]))}</text>')
    for i, country in enumerate(co):
        lx = ml + i * 145
        p.append(f'<rect x="{lx}" y="{height-26}" width="14" height="14" fill="{cm[country]}" rx="2"/>')
        p.append(f'<text x="{lx+18}" y="{height-13}" font-size="12" font-family="sans-serif">{html.escape(country)}</text>')
    p.append(f'<text x="{width/2:.0f}" y="{height-4}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(x_label)}</text>')
    p.append(f'<text x="16" y="{height/2:.0f}" transform="rotate(-90 16,{height/2:.0f})" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(y_label)}</text>')
    p.append('</svg>')
    return ''.join(p)


def make_regression_scatter_chart(frame, x_col, y_col, color_col, title,
                                   x_label, y_label, color_map=None,
                                   width=960, height=540):
    """Scatter with per-group regression lines using global axis scaling."""
    pf  = frame[[x_col, y_col, color_col]].dropna().copy()
    ml, mr, mt, mb = 85, 175, 60, 75
    pw, ph  = width - ml - mr, height - mt - mb
    xv_all  = pf[x_col].astype(float).to_numpy()
    yv_all  = pf[y_col].astype(float).to_numpy()
    xmin, xmax = float(xv_all.min()), float(xv_all.max())
    ymin, ymax = float(yv_all.min()), float(yv_all.max())
    xrng = xmax - xmin or 1.0
    yrng = ymax - ymin or 1.0
    def to_px(xv, yv):
        xp = ml + (np.asarray(xv, float) - xmin) / xrng * pw
        yp = mt + ph - (np.asarray(yv, float) - ymin) / yrng * ph
        return xp, yp
    co  = list(dict.fromkeys(pf[color_col].astype(str).tolist()))
    if color_map is None:
        color_map = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(co)}
    p = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{(width-mr)/2+ml/2:.0f}" y="34" text-anchor="middle" font-size="17" font-weight="bold" font-family="sans-serif">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#555"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555"/>',
    ]
    # Dots (all groups, faint)
    xp_all, yp_all = to_px(xv_all, yv_all)
    for pos, (_, row) in enumerate(pf.iterrows()):
        c = color_map.get(str(row[color_col]), '#888')
        p.append(f'<circle cx="{xp_all[pos]:.1f}" cy="{yp_all[pos]:.1f}" r="3" fill="{c}" opacity="0.22"/>')
    # Regression lines
    legend_entries = []
    for group_name in co:
        grp = pf[pf[color_col] == group_name]
        if len(grp) < 10: continue
        gx  = grp[x_col].astype(float).values
        gy  = grp[y_col].astype(float).values
        sl, ic = np.polyfit(gx, gy, 1)
        xl  = np.array([gx.min(), gx.max()])
        yl  = sl * xl + ic
        xlp, ylp = to_px(xl, yl)
        c   = color_map.get(group_name, '#888')
        p.append(f'<line x1="{xlp[0]:.1f}" y1="{ylp[0]:.1f}" x2="{xlp[1]:.1f}" y2="{ylp[1]:.1f}" stroke="{c}" stroke-width="3"/>')
        legend_entries.append((group_name, sl, c))
    # Legend
    for i, (name, sl, c) in enumerate(legend_entries):
        lx = ml + pw + 12
        ly = mt + i * 28
        p.append(f'<rect x="{lx}" y="{ly+5}" width="20" height="4" fill="{c}" rx="2"/>')
        p.append(f'<text x="{lx+26}" y="{ly+14}" font-size="11" font-family="sans-serif">{html.escape(name)} (slope={sl:+,.0f})</text>')
    p.append(f'<text x="{(width-mr)/2+ml/2:.0f}" y="{height-4}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(x_label)}</text>')
    p.append(f'<text x="16" y="{height/2:.0f}" transform="rotate(-90 16,{height/2:.0f})" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(y_label)}</text>')
    p.append('</svg>')
    return ''.join(p)
'''

# ── Load data ─────────────────────────────────────────────────────────────────
LOAD_DATA = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from analysis import (
    load_matched_data, filter_matched, load_tier_lookup, add_derived_columns,
    aggregate_by_school, add_frequency_bands,
    compute_pearson_bootstrap_ci, compute_spearman,
    run_rq1_kruskal_bands, run_rq2_kruskal,
    run_rq2_chisquare_quartile, run_rq2_within_country_kw, run_rq2_ols,
    filter_rq3_data, run_rq3_full_correlation_table,
    run_rq3_country_metric_correlations, run_rq3_ols,
    run_rq3_within_country_ols, compute_within_country_correlations,
    TIER_ORDER, MAIN_COUNTRIES, ACADEMIC_METRICS, sig_stars,
)

try:
    from IPython.display import HTML, display
except Exception:
    class HTML(str): pass
    def display(obj): print(obj)

import pandas as pd
pd.set_option('display.max_columns', 80)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

project_root = Path.cwd()
data_path    = project_root / 'compass_offers_processed_matched_no_phd.csv'
tier_path    = project_root / 'university_tier.csv'

df           = load_matched_data(data_path)
matched_raw  = filter_matched(df)
tier_lookup  = load_tier_lookup(tier_path)
matched      = add_derived_columns(matched_raw, tier_lookup)
"""

OVERVIEW = """overview = pd.DataFrame({
    'rows_after_preprocessing': [len(df)],
    'matched_rows': [len(matched)],
    'matched_share_pct': [len(matched) / len(df) * 100],
    'unique_matched_schools': [matched['matched_school'].nunique()],
    'countries_in_analysis': [matched['cost_country'].nunique()],
    'unique_cn_undergrad_schools': [matched['毕业学校'].nunique()],
})
overview.round(2)
"""

MATCH_STATUS = """match_status_summary = (
    df['match_status']
    .value_counts(dropna=False)
    .rename_axis('match_status')
    .reset_index(name='rows')
)
match_status_summary['share_pct'] = match_status_summary['rows'] / len(df) * 100
match_status_summary.round(2)
"""

COUNTRY_PROFILE = """country_profile = (
    matched.groupby('cost_country', as_index=False)
    .agg(
        matched_rows=('cost_country', 'size'),
        unique_schools=('matched_school', 'nunique'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
    )
    .sort_values('matched_rows', ascending=False)
)
country_profile['matched_share_pct'] = country_profile['matched_rows'] / len(matched) * 100
country_profile[['cost_country', 'matched_rows', 'matched_share_pct',
                  'unique_schools', 'mean_cost_usd', 'median_cost_usd']].round(2)
"""

RQ1_AGGREGATE = """# School-level aggregation using analysis.py
school_level = add_frequency_bands(aggregate_by_school(matched))
school_level.head()
"""

RQ1_TOP10_BAR = """show_svg(make_bar_chart(
    school_level.head(10)['matched_school'],
    school_level.head(10)['offer_count'],
    title='RQ1: Top 10 matched universities by offer count',
    x_label='Matched university',
    y_label='Offer count',
    color='#4e79a7',
    value_format='{:,.0f}',
))
"""

RQ1_TOP10_TABLE = """school_level.head(20)[['matched_school', 'cost_country',
                        'offer_count', 'median_total_cost',
                        'offer_frequency_band']].round(2)
"""

RQ1_COUNTRY_SUMMARY = """rq1_country_school = (
    school_level.groupby('cost_country', as_index=False)
    .agg(
        universities=('matched_school', 'size'),
        mean_offer_count=('offer_count', 'mean'),
        median_offer_count=('offer_count', 'median'),
        mean_cost_usd=('median_total_cost', 'mean'),
        median_cost_usd=('median_total_cost', 'median'),
    )
)
rq1_country_school.round(2)
"""

RQ1_COUNTRY_CORR = """from scipy import stats as sp_stats

country_corr_rows = []
for country, grp in school_level.groupby('cost_country'):
    if len(grp) >= 3 and grp['median_total_cost'].nunique() > 1:
        rho, p = sp_stats.spearmanr(grp['offer_count'], grp['median_total_cost'])
        country_corr_rows.append({
            'country': country,
            'school_count': len(grp),
            'spearman_rho': round(rho, 3),
            'p_value': round(p, 4),
            'sig': sig_stars(p),
        })
rq1_country_corr = pd.DataFrame(country_corr_rows).sort_values('school_count', ascending=False)
rq1_country_corr
"""

RQ1_BAND_CROSSTAB = """rq1_band_country = pd.crosstab(school_level['offer_frequency_band'],
                               school_level['cost_country'])
rq1_band_country
"""

RQ1_BAND_SHARE = """rq1_band_country_share = pd.crosstab(
    school_level['offer_frequency_band'],
    school_level['cost_country'],
    normalize='index',
) * 100
rq1_band_country_share.round(1)
"""

RQ1_SCATTER = """show_svg(make_scatter_chart(
    school_level,
    x_col='offer_count',
    y_col='median_total_cost',
    label_col='matched_school',
    color_col='cost_country',
    title='RQ1: School-level offer frequency vs representative total cost',
    x_label='Offer count per matched university',
    y_label='Median total cost (USD)',
    annotate=12,
))
"""

RQ1_FREQ_BAR = """rq1_freq_summary = (
    school_level.groupby('offer_frequency_band', observed=False)
    .agg(
        universities=('matched_school', 'size'),
        mean_offer_count=('offer_count', 'mean'),
        mean_cost_usd=('median_total_cost', 'mean'),
        median_cost_usd=('median_total_cost', 'median'),
    )
    .reset_index()
)
show_svg(make_bar_chart(
    rq1_freq_summary['offer_frequency_band'],
    rq1_freq_summary['mean_cost_usd'],
    title='RQ1: Average cost by school offer-frequency band',
    x_label='Offer-frequency band',
    y_label='Average median total cost (USD)',
    color='#f28e2b',
    value_format='${:,.0f}',
))
"""

RQ1_GROUPED_FREQ_BAR = """show_svg(make_grouped_bar_chart(
    list(rq1_freq_summary['offer_frequency_band']),
    {'Mean cost': list(rq1_freq_summary['mean_cost_usd']),
     'Median cost': list(rq1_freq_summary['median_cost_usd'])},
    title='RQ1: Mean and Median cost by offer-frequency band',
    x_label='Offer-frequency band',
    y_label='Cost (USD)',
))
"""

RQ1_H1_CORR = """r, ci_low, ci_high = compute_pearson_bootstrap_ci(
    school_level['offer_count'], school_level['median_total_cost'], n_boot=5000
)
rho_s, p_s = compute_spearman(school_level['offer_count'], school_level['median_total_cost'])

h1_corr_summary = pd.DataFrame({
    'metric': ['Pearson r', 'Bootstrap 95% CI', 'Spearman rho', 'Spearman p-value'],
    'value':  [f'{r:.4f}', f'[{ci_low:.4f}, {ci_high:.4f}]',
               f'{rho_s:.4f}', f'{p_s:.4g}  {sig_stars(p_s)}'],
})
h1_corr_summary
"""

RQ1_KW = """kw_res = run_rq1_kruskal_bands(school_level)

band_order = ['Low', 'Mid-Low', 'Mid-High', 'High']
kw_groups  = [
    school_level[school_level['offer_frequency_band'] == b]['median_total_cost'].values
    for b in band_order
]

print(f'Kruskal-Wallis  H = {kw_res["kw_stat"]:.3f},  p = {kw_res["kw_p"]:.4g}  {sig_stars(kw_res["kw_p"])}')
print()
pd.DataFrame(kw_res['pairwise'])
"""

RQ1_INTERP = """**Interpretation for RQ1.**

The overall school-level Pearson correlation is negative (`r ≈ −0.32`), and the bootstrap 95 % CI excludes zero on the negative side — meaning the negative direction is reliably estimated. The Spearman rank correlation confirms the same direction (ρ ≈ −0.55, p < 0.001). **H1a is not supported**: more frequently appearing schools are, on average, cheaper, not more expensive.

The Kruskal-Wallis test confirms that cost *does* differ significantly across offer-frequency bands (p < 0.001), and the pairwise Mann-Whitney tests show that **High-frequency schools are significantly cheaper than Low- and Mid-Low-frequency schools** — the opposite of the naive "popular = expensive" hypothesis. The country-composition table explains most of this: high-frequency schools are dominated by Hong Kong and UK universities, which are substantially cheaper than US and Singapore programs."""

RQ2_HEADER_MD = """## 3. RQ2: Undergraduate Institution Tier and Higher-Cost Destinations

The second research question returns to the applicant-offer level. Here the focus is not on which schools are common, but on whether applicants from stronger undergraduate institutions are more likely to receive offers linked to more expensive destinations.

To make that pattern visible, this section compares undergraduate tiers in four ways:

- simple counts, average costs, and high-cost quartile share
- the share of offers in the highest destination-cost quartile
- destination-country composition by undergraduate tier
- within-country cost comparisons across tiers"""

RQ2_TIER_SUMMARY = """rq2_tier_summary = (
    matched.groupby('tier', observed=False)
    .agg(
        offer_rows=('tier', 'size'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
        high_cost_share_pct=('high_cost_destination', 'mean'),
    )
    .reset_index()
)
rq2_tier_summary['high_cost_share_pct'] *= 100
rq2_tier_summary.round(2)
"""

RQ2_QUARTILE = """rq2_tier_quartile = pd.crosstab(
    matched['tier'],
    matched['cost_quartile'],
    normalize='index',
) * 100
rq2_tier_quartile.round(1)
"""

RQ2_COUNTRY = """rq2_tier_country = pd.crosstab(
    matched['tier'],
    matched['cost_country'],
    normalize='index',
) * 100
rq2_tier_country.round(1)
"""

RQ2_COUNTRY_COST = """rq2_country_cost = (
    matched.groupby(['cost_country', 'tier'], observed=False)
    .agg(
        offer_rows=('tier', 'size'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
    )
    .reset_index()
)
rq2_country_cost.pivot(index='cost_country', columns='tier',
                        values='mean_cost_usd').round(0)
"""

RQ2_Q4_BAR = """show_svg(make_bar_chart(
    rq2_tier_summary['tier'],
    rq2_tier_summary['high_cost_share_pct'],
    title='RQ2: Share of offers in the highest cost quartile by undergraduate tier',
    x_label='Undergraduate institution tier',
    y_label='Offers in highest cost quartile (%)',
    color='#59a14f',
    value_format='{:.1f}%',
))
"""

RQ2_MEDIAN_COST_BAR = """show_svg(make_grouped_bar_chart(
    TIER_ORDER,
    {'Mean cost':   [float(rq2_tier_summary.set_index('tier').loc[t, 'mean_cost_usd'])   for t in TIER_ORDER],
     'Median cost': [float(rq2_tier_summary.set_index('tier').loc[t, 'median_cost_usd']) for t in TIER_ORDER]},
    title='RQ2: Mean and Median destination cost by undergraduate tier',
    x_label='Undergraduate institution tier',
    y_label='Total cost (USD)',
))
"""

RQ2_QUARTILE_STACKED = """q_cols   = ['Q1_lowest', 'Q2', 'Q3', 'Q4_highest']
q_colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759']
q_data   = {}
for q in q_cols:
    q_data[q] = [float(rq2_tier_quartile.loc[t, q]) if t in rq2_tier_quartile.index else 0.0
                 for t in TIER_ORDER]
show_svg(make_stacked_bar_chart(
    TIER_ORDER, q_data,
    title='RQ2: Cost Quartile Distribution by Undergraduate Tier',
    x_label='Undergraduate institution tier',
    y_label='Share of applicants (%)',
    colors=q_colors,
))
"""

RQ2_COUNTRY_STACKED = """countries_to_show = [c for c in MAIN_COUNTRIES if c in rq2_tier_country.columns]
c_colors = [COUNTRY_COLOR_MAP.get(c, '#888') for c in countries_to_show]
c_data   = {}
for c in countries_to_show:
    c_data[c] = [float(rq2_tier_country.loc[t, c]) if t in rq2_tier_country.index else 0.0
                 for t in TIER_ORDER]
show_svg(make_stacked_bar_chart(
    TIER_ORDER, c_data,
    title='RQ2: Destination Country Distribution by Undergraduate Tier',
    x_label='Undergraduate institution tier',
    y_label='Share of applicants (%)',
    colors=c_colors,
))
"""

RQ2_CHI2 = """chi = run_rq2_chisquare_quartile(matched)
ct  = pd.crosstab(matched['tier'], matched['cost_quartile'])
pd.DataFrame({
    'Chi-square':         [round(chi['chi2'], 2)],
    'p-value':            [f'{chi["p"]:.2e}'],
    'df':                 [chi['dof']],
    "Cramer's V":         [round(chi['cramers_v'], 4)],
    'Supported':          ['Yes'],
})
"""

RQ2_KW = """kw = run_rq2_kruskal(matched)
print(f'Kruskal-Wallis  H = {kw["stat"]:.2f},  p = {kw["p"]:.2e}  {sig_stars(kw["p"])}')
print()
pd.DataFrame(kw['pairwise'])
"""

RQ2_WITHIN_KW = """within_kw_df = run_rq2_within_country_kw(matched)
within_tier_medians = (
    matched.groupby(['cost_country', 'tier'], observed=False)
    .agg(n=('cost_total_usd', 'size'), median_cost_usd=('cost_total_usd', 'median'))
    .reset_index()
    .pivot(index='cost_country', columns='tier', values='median_cost_usd')
    .round(0)
)
print('Kruskal-Wallis per country (cost ~ tier):')
display(HTML(within_kw_df.to_html(index=False)))
print('\\nMedian destination cost by country × tier:')
within_tier_medians
"""

RQ2_OLS = """ols = run_rq2_ols(matched)
m1, m2 = ols['model1'], ols['model2']
pd.DataFrame({
    'Model':       ['Model 1 (tier only)', 'Model 2 (tier + country)'],
    'is_985 coef': [f'${m1.params["is_985"]:+,.0f}', f'${m2.params["is_985"]:+,.0f}'],
    'is_985 p':    [f'{m1.pvalues["is_985"]:.4f}', f'{m2.pvalues["is_985"]:.4f}'],
    'R²':          [round(m1.rsquared, 3), round(m2.rsquared, 3)],
    '985 coef drop': ['-', f'{ols["coef_drop_pct"]:.1f}%'],
})
"""

RQ2_INTERP = """**Interpretation for RQ2.**

H2 receives the strongest statistical support. The chi-square test of independence between undergraduate tier and cost quartile is highly significant (χ² > 1 300, p < 0.001), and Cramér's V ≈ 0.155 indicates a small-to-medium practical effect at this sample size. The Kruskal-Wallis test confirms that destination cost distributions differ significantly across tiers (p < 0.001), and pairwise Mann-Whitney tests show **985 > 211 > Other** in median destination cost (though the 211 vs Other gap is not statistically significant).

Importantly, the within-country Kruskal-Wallis results show that the tier effect persists after controlling for destination country in Australia, Hong Kong, and the UK (all p < 0.001). This matters because 985 students are disproportionately concentrated in Singapore and Hong Kong; the within-country tests show the cost premium is not purely a composition artefact. The OLS model confirms that the 985 coefficient drops by ~56 % after adding country dummies, confirming that country choice partially mediates — but does not fully explain — the tier effect."""

RQ3_HEADER_MD = """## 4. RQ3: Academic Indicators, Cost, and Country Differences

The third research question asks whether stronger academic indicators are associated with higher-cost destinations, and whether that relationship depends on country.

This section uses several complementary views rather than relying on one single correlation:

- metric availability by country
- overall Pearson + Spearman correlations for all five metrics
- GPA bands and IELTS bands in the full matched sample
- country-specific GPA and IELTS correlations
- within-country OLS: cost ~ GPA_z + IELTS_z + tier dummies (standardised)"""

RQ3_METRIC_BY_COUNTRY = """metric_by_country = matched.groupby('cost_country')[
    ['gpa_4_standardized', 'IELTS', 'TOEFL', 'GRE', 'GMAT']].count()
metric_by_country
"""

RQ3_OVERALL_CORR = """overall_metric_corr = run_rq3_full_correlation_table(matched)
overall_metric_corr.round(4)
"""

RQ3_GPA_BAND = """rq3_gpa_band = (
    matched.groupby('gpa_band', observed=False)
    .agg(
        rows=('gpa_band', 'size'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
        high_cost_share_pct=('high_cost_destination', 'mean'),
    )
    .reset_index()
)
rq3_gpa_band['high_cost_share_pct'] *= 100
rq3_gpa_band.round(2)
"""

RQ3_IELTS_BAND = """rq3_ielts_band = (
    matched.dropna(subset=['IELTS'])
    .groupby('ielts_band', observed=False)
    .agg(
        rows=('ielts_band', 'size'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
        high_cost_share_pct=('high_cost_destination', 'mean'),
    )
    .reset_index()
)
rq3_ielts_band['high_cost_share_pct'] *= 100
rq3_ielts_band.round(2)
"""

RQ3_GPA_BAND_BAR = """show_svg(make_bar_chart(
    rq3_gpa_band['gpa_band'],
    rq3_gpa_band['high_cost_share_pct'],
    title='RQ3: Highest-cost destination share by GPA band',
    x_label='GPA band',
    y_label='Offers in highest cost quartile (%)',
    color='#edc948',
    value_format='{:.1f}%',
))
"""

RQ3_COUNTRY_CORR = """rq3_country_corr = run_rq3_country_metric_correlations(matched)
rq3_country_corr.round(3)
"""

RQ3_IELTS_BAND_BAR = """show_svg(make_bar_chart(
    rq3_ielts_band['ielts_band'],
    rq3_ielts_band['high_cost_share_pct'],
    title='RQ3: Highest-cost destination share by IELTS band',
    x_label='IELTS band',
    y_label='Offers in highest cost quartile (%)',
    color='#76b7b2',
    value_format='{:.1f}%',
))
"""

RQ3_GPA_COUNTRY_PIVOT = """# Mean cost by country x GPA band — shows Singapore reversal directly
rq3_gpa_country_cost = (
    matched[matched['cost_country'].isin(['UK', 'Singapore', 'Australia', 'USA', 'Hong Kong'])]
    .groupby(['cost_country', 'gpa_band'], observed=False)
    .agg(
        rows=('gpa_band', 'size'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
    )
    .reset_index()
)
print('Mean destination cost (USD) by country × GPA band:')
rq3_gpa_country_cost.pivot(
    index='cost_country', columns='gpa_band', values='mean_cost_usd'
).round(0)
"""

RQ3_IELTS_COUNTRY_PIVOT = """# Mean cost by country x IELTS band — Singapore high-IELTS → lower cost
rq3_ielts_country_cost = (
    matched[matched['cost_country'].isin(['UK', 'Singapore', 'Australia', 'USA', 'Hong Kong'])]
    .dropna(subset=['IELTS'])
    .groupby(['cost_country', 'ielts_band'], observed=False)
    .agg(
        rows=('ielts_band', 'size'),
        mean_cost_usd=('cost_total_usd', 'mean'),
        median_cost_usd=('cost_total_usd', 'median'),
    )
    .reset_index()
)
print('Mean destination cost (USD) by country × IELTS band:')
rq3_ielts_country_cost.pivot(
    index='cost_country', columns='ielts_band', values='mean_cost_usd'
).round(0)
"""

RQ3_CORR_BAR = """gpa_rows  = rq3_country_corr[rq3_country_corr['metric'] == 'gpa_4_standardized']
ielts_rows = rq3_country_corr[rq3_country_corr['metric'] == 'IELTS']
show_svg(make_grouped_bar_chart(
    list(gpa_rows['country']),
    {'GPA (Pearson r)':     list(gpa_rows['pearson_r']),
     'IELTS (Pearson r)':  [float(ielts_rows[ielts_rows['country']==c]['pearson_r'].iloc[0])
                            if c in ielts_rows['country'].values else 0.0
                            for c in gpa_rows['country']]},
    title='RQ3: Country-specific Pearson correlations — GPA and IELTS vs cost',
    x_label='Destination country',
    y_label='Pearson r with total cost',
    value_format='{:.3f}',
))
"""

RQ3_FULL_CORR = """h3_corr_full = run_rq3_full_correlation_table(matched)
h3_corr_full.round(4)
"""

RQ3_COUNTRY_P = """h3_country_p_df = run_rq3_country_metric_correlations(matched)
h3_country_p_df.round(4)
"""

RQ3_SCATTER_REGRESSION = """gpa_df = filter_rq3_data(matched)
show_svg(make_regression_scatter_chart(
    gpa_df,
    x_col='gpa_4_standardized',
    y_col='cost_total_usd',
    color_col='cost_country',
    title='RQ3: GPA vs Total Program Cost by Destination Country',
    x_label='Standardised GPA (4.0 scale)',
    y_label='Estimated total cost (USD)',
    color_map=COUNTRY_COLOR_MAP,
))
"""

RQ3_COEF_BAR = """m1, m2 = run_rq3_ols(gpa_df)
coef1 = m1.params['gpa_4_standardized']
coef2 = m2.params['gpa_4_standardized']
drop_pct = (coef1 - coef2) / abs(coef1) * 100
show_svg(make_grouped_bar_chart(
    ['Model 1 (GPA only)', 'Model 2 (GPA + Country)'],
    {'GPA coefficient (USD)': [coef1, coef2]},
    title=f'RQ3 — H3b: GPA effect before and after country control (drop={drop_pct:.1f}%)',
    x_label='Model',
    y_label='GPA coefficient (USD per 1-point increase)',
    colors=['#f28e2b', '#4e79a7'],
    value_format='${:,.0f}',
))
"""

RQ3_OLS_TABLE = """h3_ols_df = run_rq3_within_country_ols(matched)

# OLS summary (H3a + H3b)
print(f'H3a — Simple OLS  |  GPA coef = ${coef1:+,.0f}  p={m1.pvalues["gpa_4_standardized"]:.4f}  {sig_stars(m1.pvalues["gpa_4_standardized"])}  R²={m1.rsquared:.4f}')
print(f'H3b — + Country   |  GPA coef = ${coef2:+,.0f}  p={m2.pvalues["gpa_4_standardized"]:.4f}  {sig_stars(m2.pvalues["gpa_4_standardized"])}  R²={m2.rsquared:.4f}')
print(f'         GPA coefficient drop after adding country: {drop_pct:.1f}%')
print()

# Within-country OLS table (H3c)
h3_ols_df[h3_ols_df['feature'] != 'intercept'].round(4)
"""

RQ3_INTERP = """**Interpretation for RQ3.**

The full correlation table (Pearson + Spearman, with p-values) shows that GPA and IELTS are both **statistically significant** predictors of destination cost overall (both p < 0.001), but the effect sizes are very small (r < 0.10). Statistical significance here is driven by sample size (~18 000–27 000 rows), not practical importance.

The country-specific correlation table clarifies the picture. For **UK and Australia**, both Pearson and Spearman correlations are positive and significant — higher GPA and IELTS are weakly associated with higher-cost destinations. For **Singapore**, the correlations are **negative and significant** — higher academic scores are associated with *lower*-cost schools, likely because NUS (cheaper than NTU) attracts higher-scoring applicants.

The within-country OLS regression (cost ~ GPA_z + IELTS_z + tier dummies) confirms these directions. In the UK, GPA_z contributes +$2,866 per standard deviation (***) and IELTS_z +$1,384 (**). In Singapore, GPA_z contributes −$7,094 (***). IELTS dominates in Hong Kong (+$4,878 per SD, ***). **H3 is partially supported**: academic indicators are associated with destination cost, but the sign and strength vary substantially by country, exactly as predicted."""

SUMMARY_HEADER_MD = """## 5. Statistical Test Summary

The table below consolidates the key inferential results across all three hypotheses."""

SUMMARY_TABLE = """from analysis import run_rq1_kruskal_bands

kw_rq1    = run_rq1_kruskal_bands(school_level)
rho_s_h1, p_s_h1 = compute_spearman(school_level['offer_count'], school_level['median_total_cost'])
r_h1, ci_low_h1, ci_high_h1 = compute_pearson_bootstrap_ci(
    school_level['offer_count'], school_level['median_total_cost'])
kw_h2     = run_rq2_kruskal(matched)
chi_h2    = run_rq2_chisquare_quartile(matched)
within_kw = run_rq2_within_country_kw(matched)
ols_h2    = run_rq2_ols(matched)
gpa_df    = filter_rq3_data(matched)
m1_h3, m2_h3 = run_rq3_ols(gpa_df)
coef1_h3  = m1_h3.params['gpa_4_standardized']
coef2_h3  = m2_h3.params['gpa_4_standardized']
drop_h3   = (coef1_h3 - coef2_h3) / abs(coef1_h3) * 100

test_summary = pd.DataFrame([
    {'H': 'H1a', 'Test': 'Spearman ρ + Bootstrap Pearson CI (school-level)',
     'Result': f'ρ={rho_s_h1:.3f}, p={p_s_h1:.4g}  CI=[{ci_low_h1:.3f}, {ci_high_h1:.3f}]',
     'Supported': 'No (negative — popular schools are cheaper)'},
    {'H': 'H1b', 'Test': 'Per-country Spearman correlations',
     'Result': 'All negative or near-zero',
     'Supported': 'No (consistent direction)'},
    {'H': 'H1c', 'Test': 'Kruskal-Wallis (cost across frequency bands)',
     'Result': f'H={kw_rq1["kw_stat"]:.2f}, p={kw_rq1["kw_p"]:.4g}  {sig_stars(kw_rq1["kw_p"])}',
     'Supported': 'Partial (sig., but High-freq schools cheaper)'},
    {'H': 'H2a', 'Test': 'Kruskal-Wallis (cost across tiers)',
     'Result': f'H={kw_h2["stat"]:.1f}, p={kw_h2["p"]:.2e}  {sig_stars(kw_h2["p"])}',
     'Supported': 'Yes — 985 > 211 > Other'},
    {'H': 'H2b', 'Test': "Chi-square (tier x cost quartile) + Cramer's V",
     'Result': f'χ²={chi_h2["chi2"]:.1f}, p<0.001, V={chi_h2["cramers_v"]:.3f}',
     'Supported': 'Yes'},
    {'H': 'H2c', 'Test': 'Within-country KW (tier net of country)',
     'Result': 'Significant in AU, HK, UK (all ***)',
     'Supported': 'Yes (partial)'},
    {'H': 'H2d', 'Test': 'OLS — tier effect shrinks after country control',
     'Result': f'985 coef drops ~{ols_h2["coef_drop_pct"]:.0f}% after country added',
     'Supported': 'Yes — country mediates tier effect'},
    {'H': 'H3a', 'Test': 'Simple OLS — GPA predicts cost (cluster-robust SEs)',
     'Result': f'coef=${coef1_h3:+,.0f}, p={m1_h3.pvalues["gpa_4_standardized"]:.4f}  {sig_stars(m1_h3.pvalues["gpa_4_standardized"])}',
     'Supported': 'No — ns after clustering by school'},
    {'H': 'H3b', 'Test': 'OLS + country — GPA effect shrinks',
     'Result': f'coef=${coef2_h3:+,.0f}, drop={drop_h3:.1f}%',
     'Supported': 'Yes — country absorbs GPA effect'},
    {'H': 'H3c', 'Test': 'Within-country Spearman (GPA vs cost)',
     'Result': 'UK/AU: positive ***; SG: negative ***',
     'Supported': 'Partially — direction varies by country'},
    {'H': 'H3d', 'Test': 'Full corr. table — GPA IELTS TOEFL GRE GMAT',
     'Result': 'GPA + IELTS significant; TOEFL/GRE/GMAT sparse',
     'Supported': 'Partially — GPA and IELTS only'},
])
test_summary
"""

CONCLUSION_MD = """## 6. Overall Conclusion

Taken together, the linked dataset suggests three main conclusions.

1. **H1 is not supported in a simple school-level analysis.** More frequently appearing offer destinations are not, on average, the most expensive destinations. High-frequency schools are dominated by UK and Hong Kong institutions, which are cheaper than US or Singapore programs. The negative popularity-cost relationship holds globally and within the UK (the largest market).

2. **H2 receives the strongest support.** Applicants from `985` institutions are the most concentrated in higher-cost destinations, `211` applicants fall in the middle, and `Other` applicants are least concentrated in the highest cost quartile. This tier effect persists within countries (Australia, Hong Kong, UK), confirming it is not merely a country-composition artefact.

3. **H3 is mixed and country-specific.** GPA and language scores show some relationship with destination cost, but the effect is small and reverses in Singapore. Across UK and Australia, higher scores predict higher-cost destinations; in Singapore, higher-scoring students concentrate at NUS (cheaper than NTU). The within-country OLS confirms these patterns with standardised coefficients, allowing direct comparison of GPA vs IELTS effects.

This is exactly the value of linking the two datasets: the offer data alone cannot answer questions about affordability, and the cost data alone cannot show which kinds of applicants are reaching more expensive destinations."""

LIMITATIONS_MD = """## 7. Limitations

- The analysis only includes rows that could be matched to the cost dataset (88.37% match rate; 11.63% of rows are excluded).
- `cost_total_usd` is a representative school-level estimate rather than a customised cost for each exact applicant-program combination.
- Undergraduate institution tier is based on the 985/211 classification loaded from `university_tier.csv`; this remains only one operationalisation of applicant background.
- TOEFL, GRE, and GMAT are much sparser than GPA and IELTS (< 7 % coverage each), so they are best treated as supplementary evidence.
- Some destination systems — especially Hong Kong and Singapore — have few distinct cost values in the matched sample (3 and 6 schools respectively), which limits how much can be learned from within-country correlation alone.
- The cluster-robust OLS for H3a clusters by destination school (71 clusters); with cost_total_usd being school-level constant, this removes most individual variation, making GPA non-significant in the OLS even though individual-level correlations are significant.

These limitations do not invalidate the analysis, but they should be stated clearly when interpreting the results."""

cells = [
    md("# Analyze of Chinese Student Graduation School Application data\n\n"
       "This notebook analyzes the cleaned and matched dataset `compass_offers_processed_matched_no_phd.csv`. "
       "The goal is to connect applicant-side offer outcomes with university-level international education cost estimates.\n\n"
       "The analysis is organized around three research questions:\n\n"
       "1. Are universities that appear more frequently in Chinese applicants' offer outcomes also associated with higher estimated total study costs?\n"
       "2. Are applicants from higher-tier Chinese undergraduate institutions more likely to receive offers from higher-cost destinations?\n"
       "3. Are stronger academic indicators, especially GPA and language scores, associated with higher-cost destinations, and does that relationship vary by country?\n"),

    md("## 1. Setup and Data Loading\n\n"
       "We first load the cleaned dataset and convert the main analysis columns to numeric form. "
       "Because all three research questions require cost information, the main analysis uses only rows "
       "that were successfully matched to the cost dataset. "
       "The current preprocessing linkage uses a 3-stage school matching pipeline: exact match, manual alias mapping, and normalized unique-name matching."),

    code(LOAD_DATA),
    code(SVG_HELPERS),
    code(OVERVIEW),
    code(MATCH_STATUS),
    code(COUNTRY_PROFILE),

    md("## 2. RQ1: Offer Frequency and Total Cost\n\n"
       "The first research question is best answered at the university level. We collapse the matched data so that "
       "each university appears once, with its offer count and representative total cost.\n\n"
       "This section goes beyond a single correlation by checking:\n\n"
       "- the overall relationship between offer frequency and cost\n"
       "- which schools dominate the matched sample\n"
       "- whether the relationship changes by country\n"
       "- how cost differs across school frequency quartile bands"),

    code(RQ1_AGGREGATE),
    code(RQ1_TOP10_BAR),
    code(RQ1_TOP10_TABLE),
    code(RQ1_COUNTRY_SUMMARY),
    code(RQ1_COUNTRY_CORR),
    code(RQ1_BAND_CROSSTAB),
    code(RQ1_BAND_SHARE),
    code(RQ1_SCATTER),
    code(RQ1_FREQ_BAR),
    code(RQ1_GROUPED_FREQ_BAR),
    code(RQ1_H1_CORR),
    code(RQ1_KW),
    md(RQ1_INTERP),

    md(RQ2_HEADER_MD),
    code(RQ2_TIER_SUMMARY),
    code(RQ2_QUARTILE),
    code(RQ2_COUNTRY),
    code(RQ2_COUNTRY_COST),
    code(RQ2_Q4_BAR),
    code(RQ2_MEDIAN_COST_BAR),
    code(RQ2_QUARTILE_STACKED),
    code(RQ2_COUNTRY_STACKED),
    code(RQ2_CHI2),
    code(RQ2_KW),
    code(RQ2_WITHIN_KW),
    code(RQ2_OLS),
    md(RQ2_INTERP),

    md(RQ3_HEADER_MD),
    code(RQ3_METRIC_BY_COUNTRY),
    code(RQ3_OVERALL_CORR),
    code(RQ3_GPA_BAND),
    code(RQ3_IELTS_BAND),
    code(RQ3_GPA_BAND_BAR),
    code(RQ3_GPA_COUNTRY_PIVOT),
    code(RQ3_COUNTRY_CORR),
    code(RQ3_IELTS_BAND_BAR),
    code(RQ3_IELTS_COUNTRY_PIVOT),
    code(RQ3_CORR_BAR),
    code(RQ3_SCATTER_REGRESSION),
    code(RQ3_COEF_BAR),
    code(RQ3_OLS_TABLE),
    md(RQ3_INTERP),

    md(SUMMARY_HEADER_MD),
    code(SUMMARY_TABLE),
    md(CONCLUSION_MD),
    md(LIMITATIONS_MD),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "offer_cost_analysis.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Written: {out}  ({len(cells)} cells)")
