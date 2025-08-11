# app.py
# Streamlit app for Google Play analysis (EDA, Simpson's paradox, OLS regressions)

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Page config (call once, at the very top)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Play Store Explorer", layout="wide")
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")


# -----------------------------------------------------------------------------
# Robust relative path to a default CSV (in the repo)
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "apps_clean.csv.gz"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def col_first(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first existing column name from candidates, else raise KeyError."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the columns exist: {candidates}")


def parse_installs(x) -> float:
    """Parse strings like '1,000,000+' -> 1000000 (low bound)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s.endswith("+"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_price(x) -> float:
    """Parse price strings -> numeric USD. '$2.99' -> 2.99, '0'->0.0"""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("$", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def parse_size_mb(x) -> float:
    """Parse size strings -> MB. '14M'->14, '850k'->0.85, 'Varies with device'->NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if "varies" in s:
        return np.nan
    try:
        if s.endswith("m"):
            return float(s[:-1])
        if s.endswith("mb"):
            return float(s[:-2])
        if s.endswith("k"):
            return float(s[:-1]) / 1000.0
        # raw number
        return float(s)
    except Exception:
        return np.nan


def year_from_date(x) -> float:
    """Extract year from ISO-like date strings."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    # Try split by '-', ' ' etc.
    try:
        y = int(s[:4])
        return float(y)
    except Exception:
        return np.nan


def ym_from_date(x) -> float:
    """Encode YYYY-MM (or YYYY-MM-DD) into months since 0 (YYYY*12 + MM)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        parts = s.split("-")
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        return float(y * 12 + m)
    except Exception:
        return np.nan


def safe_rotate_xlabels(ax, deg: float = 45.0, ha: str = "right"):
    """Rotate x tick labels safely (no tick_params kwargs that differ by version)."""
    for lab in ax.get_xticklabels():
        lab.set_rotation(deg)
        lab.set_ha(ha)


# -----------------------------------------------------------------------------
# Column normalization
# -----------------------------------------------------------------------------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns across multiple dataset variants and build features."""

    # ---- Category text ----
    try:
        cat_col = col_first(df, "Category (categorical)", "Category_text", "Category")
    except KeyError:
        # create from numeric codes if nothing else exists
        if "Category" in df.columns:
            df["Category_text"] = df["Category"].astype(str)
            cat_col = "Category_text"
        else:
            raise
    df["Category_text"] = (
        df[cat_col].astype(str).str.replace("_", " ", regex=False).str.title()
    )

    # ---- Rating ----
    rat_col = "Rating" if "Rating" in df.columns else col_first(
        df, "rating", "rating_num", "Rating_num"
    )
    df["Rating_num"] = pd.to_numeric(df[rat_col], errors="coerce")

    # ---- Reviews ----
    rev_col = "Reviews" if "Reviews" in df.columns else col_first(
        df, "reviews", "Reviews_num"
    )
    df["Reviews_num"] = pd.to_numeric(df[rev_col], errors="coerce")

    # ---- Installs ----
    if "Installs_num" in df.columns:
        df["Installs_num"] = pd.to_numeric(df["Installs_num"], errors="coerce")
    else:
        inst_col = "Installs" if "Installs" in df.columns else col_first(
            df, "installs"
        )
        df["Installs_num"] = df[inst_col].apply(parse_installs)

    # ---- Price / Type -> Is_Paid ----
    if "Price_num" in df.columns:
        df["Price_num"] = pd.to_numeric(df["Price_num"], errors="coerce").fillna(0.0)
    else:
        if "Price" in df.columns:
            df["Price_num"] = df["Price"].apply(parse_price)
        else:
            df["Price_num"] = 0.0

    def norm_type(v):
        s = str(v).strip().lower()
        if s in {"paid", "p", "1", "true", "yes", "$", "paid app"}:
            return "Paid"
        return "Free"

    if "Type" in df.columns:
        df["Type"] = df["Type"].apply(norm_type)
    else:
        df["Type"] = np.where(df["Price_num"] > 0, "Paid", "Free")

    df["Is_Paid"] = (df["Type"].str.lower() == "paid").astype(int)

    # ---- Size (MB) ----
    if "Size_MB" in df.columns:
        df["Size_MB"] = pd.to_numeric(df["Size_MB"], errors="coerce")
    else:
        size_col = "Size" if "Size" in df.columns else col_first(df, "size")
        df["Size_MB"] = df[size_col].apply(parse_size_mb)

    # ---- Last updated & time features ----
    if "Last Updated" in df.columns:
        df["LastUpdated"] = df["Last Updated"]
        df["LastUpdated_Year"] = df["LastUpdated"].apply(year_from_date)
        df["LastUpdated_YM"] = df["LastUpdated"].apply(ym_from_date)
    else:
        # cleaned variants
        y = None
        m = None
        for cand in ["Last Updated (Year)", "LastUpdated (Year)", "LastUpdated_Year"]:
            if cand in df.columns:
                y = cand
                break
        for cand in [
            "Last Updated (Year/Month)",
            "LastUpdated (Year/Month)",
            "LastUpdated_YM",
        ]:
            if cand in df.columns:
                m = cand
                break
        df["LastUpdated_Year"] = pd.to_numeric(df.get(y, np.nan), errors="coerce")
        df["LastUpdated_YM"] = pd.to_numeric(df.get(m, np.nan), errors="coerce")
        df["LastUpdated"] = pd.NaT

    # ---- Clean ranges & logs ----
    df = df[(df["Rating_num"] >= 0) & (df["Rating_num"] <= 5)]
    df["log10_installs"] = np.log10(df["Installs_num"].replace(0, np.nan))
    df["log10_reviews"] = np.log10(df["Reviews_num"].replace(0, np.nan))

    # recency in years relative to max month available (only if YM exists)
    ym = pd.to_numeric(df["LastUpdated_YM"], errors="coerce")
    max_ym = ym.dropna().max()
    if pd.notna(max_ym):
        df["Recency_years"] = (max_ym - ym) / 12.0
    else:
        df["Recency_years"] = np.nan

    # Keep only necessary columns for the app
    keep = [
        "App",
        "Category_text",
        "Rating_num",
        "Reviews_num",
        "Installs_num",
        "Size_MB",
        "Type",
        "Price_num",
        "Is_Paid",
        "LastUpdated",
        "LastUpdated_Year",
        "LastUpdated_YM",
        "log10_installs",
        "log10_reviews",
        "Recency_years",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan

    return df


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_data_default() -> pd.DataFrame:
    # avoid dtype changes that can bloat memory
    return pd.read_csv(DATA_PATH, low_memory=False)


@st.cache_data(show_spinner=True, ttl=3600)
def load_data_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), low_memory=False)


@st.cache_data(show_spinner=True, ttl=3600)
def load_data_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url, low_memory=False)


# -----------------------------------------------------------------------------
# Sidebar: data source + filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")

    url = st.text_input("Optional: CSV URL (public)")
    up = st.file_uploader("Or upload a CSV", type=["csv"])

    if up is not None:
        df_raw = load_data_from_bytes(up.getvalue())
    elif url.strip():
        df_raw = load_data_from_url(url.strip())
    else:
        df_raw = load_data_default()

    # Normalize columns
    df = ensure_columns(df_raw.copy())

    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    st.header("Filters")
    cats = sorted([c for c in df["Category_text"].dropna().unique()])
    sel_cats = st.multiselect("Categories", options=cats, default=cats)

    type_choice = st.radio("Type", options=["All", "Free", "Paid"], horizontal=True)

    min_reviews = int(np.nanmin(df["Reviews_num"])) if df["Reviews_num"].notna().any() else 0
    max_reviews = int(np.nanmax(df["Reviews_num"])) if df["Reviews_num"].notna().any() else 1000
    min_rev = st.slider(
        "Minimum reviews", min_value=min_reviews, max_value=max_reviews, value=min_reviews
    )

    # Year range
    yr_min = int(np.nanmin(df["LastUpdated_Year"])) if df["LastUpdated_Year"].notna().any() else 2010
    yr_max = int(np.nanmax(df["LastUpdated_Year"])) if df["LastUpdated_Year"].notna().any() else 2018
    yr_lo, yr_hi = st.slider(
        "Updated year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max)
    )

# Apply filters
mask = (
    df["Category_text"].isin(sel_cats)
    & (df["Reviews_num"] >= min_rev)
    & (df["LastUpdated_Year"].between(yr_lo, yr_hi, inclusive="both"))
)
if type_choice != "All":
    mask &= df["Type"].str.lower().eq(type_choice.lower())
df_f = df.loc[mask].copy()

st.caption(f"Active filter ≈ {df_f.shape[0]:,} apps")


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_explore, tab_simpson, tab_reg, tab_fe, tab_about = st.tabs(
    ["Overview", "Explore", "Simpson", "Regression", "Free vs Paid (FE)", "About"]
)


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Apps", f"{df_f.shape[0]:,}")
    with c2:
        med_rating = np.nanmedian(df_f["Rating_num"]) if len(df_f) else np.nan
        st.metric("Median rating", f"{med_rating:.2f}" if pd.notna(med_rating) else "–")
    with c3:
        share_paid = (
            df_f["Is_Paid"].mean() if len(df_f) else np.nan
        )
        st.metric("Paid share", f"{100*share_paid:.1f}%" if pd.notna(share_paid) else "–")
    with c4:
        med_inst = np.nanmedian(df_f["Installs_num"]) if len(df_f) else np.nan
        st.metric("Median installs", f"{int(med_inst):,}" if pd.notna(med_inst) else "–")

    st.markdown("**Sample (first 10):**")
    show_cols = [
        "App",
        "Category_text",
        "Rating_num",
        "Reviews_num",
        "Size_MB",
        "Installs_num",
        "Type",
        "Price_num",
        "LastUpdated",
        "LastUpdated_Year",
        "LastUpdated_YM",
    ]
    st.dataframe(df_f[show_cols].head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# Explore
# -----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Quick EDA")

    cc1, cc2 = st.columns(2)

    with cc1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df_f["Rating_num"].dropna(),
            bins=np.arange(1.0, 5.05, 0.1),
            ax=ax,
        )
        ax.set_title("Rating distribution")
        ax.set_xlabel("Rating (1–5)")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with cc2:
        top = (
            df_f["Category_text"]
            .value_counts()
            .sort_values(ascending=False)
            .head(10)
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index, top.values)
        ax.set_title("Top categories by app count")
        ax.set_xlabel("Apps")
        safe_rotate_xlabels(ax, 0)  # horizontal bars, no rotation
        st.pyplot(fig, clear_figure=True)

    # Scatter (log–log, sampled)
    st.subheader("Installs vs Reviews (log10–log10)")
    dsc = df_f[["log10_reviews", "log10_installs"]].dropna()
    if len(dsc) >= 10:
        sample = dsc.sample(n=min(3000, len(dsc)), random_state=1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sample["log10_reviews"], sample["log10_installs"], s=8, alpha=0.4)
        ax.set_xlabel("log10(Reviews)")
        ax.set_ylabel("log10(Installs)")
        ax.set_title("log–log scatter")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Not enough data after filters.")


# -----------------------------------------------------------------------------
# Simpson
# -----------------------------------------------------------------------------
with tab_simpson:
    st.subheader("Weighted vs unweighted mean rating (Simpson-like flip)")

    cats_avail = sorted(df_f["Category_text"].dropna().unique())
    if not cats_avail:
        st.info("No categories available after filters.")
    else:
        selected = st.selectbox("Pick a category", options=cats_avail, index=0)

        def wmean(series, weights):
            series = pd.Series(series)
            weights = pd.Series(weights).fillna(0.0)
            s = np.sum(weights)
            if s <= 0:
                return np.nanmean(series)
            return float(np.sum(series * weights) / s)

        overall_unw = np.nanmean(df_f["Rating_num"])
        overall_w = wmean(df_f["Rating_num"], df_f["Reviews_num"])

        sub = df_f[df_f["Category_text"] == selected]
        cat_unw = np.nanmean(sub["Rating_num"])
        cat_w = wmean(sub["Rating_num"], sub["Reviews_num"])

        # Build a tidy table for plotting: two groups × two bars (two colors only)
        t = pd.DataFrame(
            {
                "Group": ["Overall", "Overall", selected, selected],
                "Kind": ["Unweighted", "Weighted", "Unweighted", "Weighted"],
                "Value": [overall_unw, overall_w, cat_unw, cat_w],
            }
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        # Two colors, one for Unweighted, one for Weighted
        palette = {"Unweighted": sns.color_palette()[0], "Weighted": sns.color_palette()[1]}
        for i, (grp, gdf) in enumerate(t.groupby("Group", sort=False)):
            xs = np.array([i - 0.15, i + 0.15])
            ys = gdf["Value"].values
            kinds = gdf["Kind"].values
            ax.bar(xs, ys, width=0.28, color=[palette[k] for k in kinds], label=None)
            for x, y in zip(xs, ys):
                if pd.notna(y):
                    ax.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(t["Group"].unique())))
        ax.set_xticklabels(t["Group"].unique())
        ax.set_ylabel("Average rating")
        ax.set_title("Unweighted vs Weighted mean rating")
        # Legend for the two bar colors
        handles = [
            plt.Line2D([0], [0], color=palette["Unweighted"], lw=10, label="Unweighted"),
            plt.Line2D([0], [0], color=palette["Weighted"], lw=10, label="Weighted"),
        ]
        ax.legend(handles=handles, loc="upper right")
        st.pyplot(fig, clear_figure=True)


# -----------------------------------------------------------------------------
# Regression: log10(Installs) ~ log10(Reviews)
# -----------------------------------------------------------------------------
with tab_reg:
    st.subheader("Installs vs Reviews — OLS (log10–log10) with 95% CI & prediction band")

    d = df_f[["log10_installs", "log10_reviews"]].dropna()
    if len(d) < 30 or d["log10_reviews"].nunique() < 5:
        st.info("Not enough data after filters for a stable regression.")
    else:
        d = d.copy()
        d["const"] = 1.0
        model = sm.OLS(d["log10_installs"], d[["const", "log10_reviews"]]).fit(
            cov_type="HC1"
        )

        # Prediction grid
        x_grid = np.linspace(d["log10_reviews"].min(), d["log10_reviews"].max(), 120)
        Xp = pd.DataFrame({"const": 1.0, "log10_reviews": x_grid})
        pred = model.get_prediction(Xp)
        summ = pred.summary_frame(alpha=0.05)

        fig, ax = plt.subplots(figsize=(8, 5))
        sample = d.sample(n=min(4000, len(d)), random_state=1)
        ax.scatter(sample["log10_reviews"], sample["log10_installs"], s=8, alpha=0.35, label="Apps")

        # Mean fit and intervals
        ax.plot(x_grid, summ["mean"], lw=2, label="OLS fit")
        ax.fill_between(x_grid, summ["mean_ci_lower"], summ["mean_ci_upper"], alpha=0.2, label="95% CI (mean)")
        ax.fill_between(x_grid, summ["obs_ci_lower"], summ["obs_ci_upper"], alpha=0.15, label="95% prediction interval")

        ax.set_xlabel("log10(Reviews)")
        ax.set_ylabel("log10(Installs)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        st.caption(
            f"Model quality: n = {len(d):,}, R-squared (robust se) ≈ {model.rsquared:.3f} • "
            f"Slope (elasticity) = {model.params['log10_reviews']:.3f}"
        )


# -----------------------------------------------------------------------------
# Free vs Paid (Fixed effects: Category dummies)
# -----------------------------------------------------------------------------
with tab_fe:
    st.subheader("Free vs Paid — OLS with Category dummies (robust SE)")

    dfm = df_f[
        ["log10_installs", "log10_reviews", "Rating_num", "Size_MB", "Recency_years", "Is_Paid", "Category_text"]
    ].dropna()
    # Need both types present and enough categories
    if len(dfm) < 50 or dfm["Is_Paid"].nunique() < 2 or dfm["Category_text"].nunique() < 2:
        st.info("Not enough data after filters for a stable FE regression.")
    else:
        # Build a clean modeling frame
        model_df = dfm.copy()
        # C(Category_text) builds category fixed effects
        formula = "log10_installs ~ Is_Paid + log10_reviews + Rating_num + Size_MB + Recency_years + C(Category_text)"
        fe = smf.ols(formula, data=model_df).fit(cov_type="HC1")

        # Tidy coef table
        ci = fe.conf_int(alpha=0.05)
        ci.columns = ["ci_low", "ci_high"]
        coef_tbl = pd.DataFrame(
            {
                "coef": fe.params,
                "std_err": fe.bse,
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "p_value": fe.pvalues,
            }
        ).loc[["Is_Paid", "log10_reviews", "Rating_num", "Size_MB", "Recency_years"]]

        st.dataframe(coef_tbl.style.format(
            {"coef": "{:.3f}", "std_err": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "p_value": "{:.3g}"}
        ), use_container_width=True)

        # Partial effect plot for Is_Paid (point + CI)
        beta = fe.params["Is_Paid"]
        se = fe.bse["Is_Paid"]
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se

        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.errorbar([beta], [0], xerr=[[beta - lo], [hi - beta]], fmt="o", capsize=4)
        ax.axvline(0.0, ls="--", lw=1, c="gray")
        ax.set_yticks([])
        ax.set_xlabel("Effect on log10(installs)")
        ax.set_title("Paid effect with 95% CI")
        st.pyplot(fig, clear_figure=True)

        st.caption(
            "Interpretation: a one-unit increase in the Paid indicator (Paid vs Free) shifts "
            f"log10(installs) by {beta:.3f}. Values < 0 imply a penalty relative to Free."
        )


# -----------------------------------------------------------------------------
# About
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### About this app

**What we analyzed**  
We explored Google Play apps to understand how ratings, review volumes, categories, and monetization (Free vs Paid) relate to demand (installs).  
We were particularly interested in:
- How averages can flip once we weight by reliability (review volume) — a Simpson-like effect.
- The elasticity between reviews and installs on a log–log scale.
- The effect of being Paid vs Free after controlling for app mix (category fixed effects).

**What the app provides**
- **Overview:** headline KPIs and a sample of the filtered data.  
- **Explore:** rating distribution, top categories, and a log–log scatter (Installs vs Reviews).  
- **Simpson:** compare unweighted vs weighted mean rating — one pair for the whole dataset and one for a selected category.  
- **Regression:** OLS of log10(Installs) on log10(Reviews) with robust SEs, 95% CI, and a prediction band.  
- **Free vs Paid (FE):** OLS with category dummies and robust SEs to quantify the Paid effect.

**Notes**
- All regressions use robust (HC1) standard errors.  
- Tick label rotation is handled safely for Matplotlib compatibility.
        """
    )
