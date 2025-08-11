# app.py
# Streamlit app for Google Play analysis (EDA, Simpson's paradox, OLS regressions)
# All UI text in English (per course standard)

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -------------------------
# Page & style
# -------------------------
st.set_page_config(page_title="Play Store Explorer", layout="wide")
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")

# -------------------------
# Helpers: robust utils
# -------------------------
DATA_PATH = Path(__file__).parent / "data" / "apps_clean.csv.gz"

def rotate_xticks(ax, deg: int = 45, ha: str = "right"):
    """Safe tick rotation + alignment for new Matplotlib versions."""
    ax.tick_params(axis="x", which="both", labelrotation=deg)
    plt.setp(ax.get_xticklabels(), ha=ha)

def parse_int(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = s.replace(",", "").replace("+", "")
    if s.isdigit():
        return int(s)
    return np.nan

def parse_price(s):
    if pd.isna(s):
        return 0.0
    s = str(s).strip().replace("$", "").replace("₪", "")
    try:
        return float(s)
    except Exception:
        return 0.0

def parse_size_mb(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if "varies" in s:
        return np.nan
    try:
        if s.endswith("m"):
            return float(s[:-1])
        if s.endswith("mb"):
            return float(s[:-2])
        if s.endswith("k") or s.endswith("kb"):
            # Convert KB to MB
            num = s[:-1] if s.endswith("k") else s[:-2]
            return float(num) / 1024.0
        # pure number
        return float(s)
    except Exception:
        return np.nan

# -------------------------
# Data loading (cached)
# -------------------------
@st.cache_data(show_spinner=False, ttl=86400)  # cache for 24h
def load_default() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, low_memory=False)

def load_from_upload_or_url() -> pd.DataFrame | None:
    """Allow the user to upload a CSV or paste a public URL to CSV (Dropbox/GitHub raw)."""
    left = st.sidebar
    url = left.text_input("Optional: CSV URL (public)", placeholder="https://...csv")
    upload = left.file_uploader("Or upload a CSV", type=["csv"])
    df = None

    if upload is not None:
        try:
            df = pd.read_csv(upload, low_memory=False)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    if df is None and url:
        try:
            df = pd.read_csv(url, low_memory=False)
        except Exception as e:
            st.error(f"Could not fetch URL: {e}")

    return df

def clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create robust numeric features used by all views."""
    df = df_raw.copy()

    # Column name normalization (work with common Kaggle schema variants)
    cols = {c.lower().strip(): c for c in df.columns}
    # Required-ish names with fallbacks:
    col_name   = cols.get("app", cols.get("app name", cols.get("name", None)))
    col_cat    = cols.get("category", cols.get("categories", None))
    col_type   = cols.get("type", None)
    col_price  = cols.get("price", None)
    col_size   = cols.get("size", None)
    col_rate   = cols.get("rating", cols.get("ratings", None))
    col_rev    = cols.get("reviews", None)
    col_inst   = cols.get("installs", None)
    col_upd    = cols.get("last updated", cols.get("last_update", None))

    # Create harmonized columns if exist
    if col_rate and col_rate in df:
        df["Rating_num"] = pd.to_numeric(df[col_rate], errors="coerce")
    else:
        df["Rating_num"] = np.nan

    if col_rev and col_rev in df:
        df["Reviews_num"] = df[col_rev].apply(parse_int)
    else:
        df["Reviews_num"] = np.nan

    if col_inst and col_inst in df:
        df["Installs_num"] = df[col_inst].apply(parse_int)
    else:
        df["Installs_num"] = np.nan

    if col_type and col_type in df:
        df["Type_text"] = df[col_type].astype(str)
    else:
        df["Type_text"] = np.where(df.get("Price", pd.Series(np.nan)).notna(), "Paid", "Free")

    if col_price and col_price in df:
        df["Price_num"] = df[col_price].apply(parse_price)
    else:
        df["Price_num"] = 0.0

    if col_cat and col_cat in df:
        df["Category_text"] = df[col_cat].astype(str).str.replace("_", " ").str.title()
    else:
        df["Category_text"] = "Unknown"

    if col_size and col_size in df:
        df["Size_MB"] = df[col_size].apply(parse_size_mb)
    else:
        df["Size_MB"] = np.nan

    if col_upd and col_upd in df:
        df["LastUpdated"] = pd.to_datetime(df[col_upd], errors="coerce")
    else:
        df["LastUpdated"] = pd.NaT

    # Additional constructed features
    df["Is_Paid_num"] = np.where((df["Price_num"] > 0) | (df["Type_text"].str.lower() == "paid"), 1.0, 0.0)

    # Recency in years (relative to the max valid date)
    if df["LastUpdated"].notna().any():
        latest = df["LastUpdated"].max()
        df["Recency_years"] = (latest - df["LastUpdated"]).dt.days / 365.25
    else:
        df["Recency_years"] = np.nan

    # Log transforms (safe)
    df["log10_installs"] = np.log10(np.clip(df["Installs_num"].astype(float), 1, None))
    df["log10_reviews"] = np.log10(np.clip(df["Reviews_num"].astype(float), 1, None))

    # Keep only rows that have a category, rating in range, and numeric installs/reviews
    df = df[df["Category_text"].notna()].copy()
    df = df[(df["Rating_num"].between(1, 5)) | df["Rating_num"].isna()]
    return df

# -------------------------
# Load data with fallbacks
# -------------------------
user_df = load_from_upload_or_url()
if user_df is None:
    with st.spinner("Loading data..."):
        user_df = load_default()

df = clean_df(user_df)
st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# -------------------------
# Sidebar filters
# -------------------------
cats = sorted(df["Category_text"].dropna().unique().tolist())
sel_cats = st.sidebar.multiselect("Categories", cats, default=[])
type_choice = st.sidebar.radio("Type", ["All", "Free", "Paid"], horizontal=True)
min_reviews = int(st.sidebar.number_input("Minimum reviews", value=0, min_value=0, step=100))
year_min = int((df["LastUpdated"].dt.year.min() if df["LastUpdated"].notna().any() else 2010))
year_max = int((df["LastUpdated"].dt.year.max() if df["LastUpdated"].notna().any() else 2018))
yr_range = st.sidebar.slider("Updated year range", min_value=year_min, max_value=year_max,
                             value=(year_min, year_max))

flt = df.copy()
if sel_cats:
    flt = flt[flt["Category_text"].isin(sel_cats)]
if type_choice != "All":
    if type_choice == "Free":
        flt = flt[flt["Is_Paid_num"] == 0]
    else:
        flt = flt[flt["Is_Paid_num"] == 1]
if min_reviews > 0:
    flt = flt[flt["Reviews_num"] >= min_reviews]
if flt["LastUpdated"].notna().any():
    flt = flt[flt["LastUpdated"].dt.year.between(yr_range[0], yr_range[1])]

st.caption(f"Active filter ≈ {flt.shape[0]:,} apps")

# -------------------------
# Tabs
# -------------------------
tab_over, tab_explore, tab_simp, tab_reg, tab_fe, tab_about = st.tabs(
    ["Overview", "Explore", "Simpson", "Regression", "Free vs Paid (FE)", "About"]
)

# -------------------------
# Overview
# -------------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Apps", f"{flt.shape[0]:,}")
    c2.metric("Median rating", f"{flt['Rating_num'].median():.2f}")
    c3.metric("Paid share", f"{100*flt['Is_Paid_num'].mean():.1f}%")
    c4.metric("Median installs", f"{flt['Installs_num'].median():,}")

    st.markdown("**Sample (first 10):**")
    st.dataframe(flt.head(10), use_container_width=True, height=280)

# -------------------------
# Explore
# -------------------------
with tab_explore:
    st.subheader("Quick EDA")

    # Rating distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(flt["Rating_num"].dropna(), bins=30, ax=ax)
    ax.set_xlabel("Rating (1–5)")
    ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)

    # Top categories by count
    top = flt["Category_text"].value_counts().head(12).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(x=top.index, y=top.values, ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Apps")
    rotate_xticks(ax, 45, "right")
    st.pyplot(fig, use_container_width=True)

    # Scatter (log-log) Installs vs Reviews
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(flt["log10_reviews"], flt["log10_installs"], alpha=0.15, s=10)
    ax.set_xlabel("log10(Reviews)")
    ax.set_ylabel("log10(Installs)")
    st.pyplot(fig, use_container_width=True)

# -------------------------
# Simpson’s paradox demo
# -------------------------
with tab_simp:
    st.subheader("Weighted vs unweighted means (Simpson-like reversal)")

    # pick a category to compare with overall
    cat_pick = st.selectbox("Pick a category", cats, index=(cats.index("Events") if "Events" in cats else 0))

    sub = flt.copy()
    sub_cat = sub[sub["Category_text"] == cat_pick].copy()

    def wavg(series, weights):
        w = np.clip(weights.fillna(0).astype(float), 0.0, None)
        x = series.astype(float)
        if w.sum() <= 0:
            return np.nan
        return np.average(x, weights=w)

    overall_unw = sub["Rating_num"].mean()
    overall_w   = wavg(sub["Rating_num"], sub["Reviews_num"])
    cat_unw     = sub_cat["Rating_num"].mean()
    cat_w       = wavg(sub_cat["Rating_num"], sub_cat["Reviews_num"])

    plot_df = pd.DataFrame({
        "Group": ["Overall (Unweighted)", "Overall (Weighted)",
                  f"{cat_pick} (Unweighted)", f"{cat_pick} (Weighted)"],
        "Mean":  [overall_unw, overall_w, cat_unw, cat_w]
    })

    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.barplot(data=plot_df, x="Group", y="Mean", ax=ax, palette=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.set_ylabel("Average rating")
    ax.set_xlabel("")
    rotate_xticks(ax, 20, "right")
    for p in ax.patches:
        y = p.get_height()
        ax.text(p.get_x() + p.get_width()/2, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(3.6, 4.7)
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        "What we show: two means (unweighted vs weighted-by-reviews) for *Overall* and for the chosen category. "
        "Sometimes the ordering flips when we weight by volume — a classic Simpson-like effect."
    )

# -------------------------
# Regression (log–log): Installs ~ Reviews
# -------------------------
with tab_reg:
    st.subheader("Installs vs Reviews — OLS (log10–log10) with 95% CI & prediction band")

    reg = flt.dropna(subset=["log10_installs", "log10_reviews"]).copy()
    if reg.shape[0] < 50:
        st.info("Not enough data after filters for a stable regression.")
    else:
        X = sm.add_constant(reg["log10_reviews"].values)
        y = reg["log10_installs"].values
        model = sm.OLS(y, X).fit(cov_type="HC1")

        # Grid for predictions
        xg = np.linspace(reg["log10_reviews"].quantile(0.02), reg["log10_reviews"].quantile(0.98), 100)
        Xg = sm.add_constant(xg)
        pred = model.get_prediction(Xg)
        frame = pred.summary_frame(alpha=0.05)  # mean_ci_lower/upper & obs_ci_lower/upper

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(reg["log10_reviews"], reg["log10_installs"], alpha=0.12, s=10, label="Apps")
        ax.plot(xg, frame["mean"], lw=2, label="OLS fit")
        ax.fill_between(xg, frame["mean_ci_lower"], frame["mean_ci_upper"], alpha=0.25, label="95% CI (mean)")
        ax.fill_between(xg, frame["obs_ci_lower"], frame["obs_ci_upper"], alpha=0.12, label="95% prediction interval")
        ax.set_xlabel("log10(Reviews)  • +1 tick = ×10 reviews")
        ax.set_ylabel("log10(Installs) • +1 tick = ×10 installs")
        ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=True)

        beta = model.params[1]
        ci_lo, ci_hi = model.conf_int(alpha=0.05)[1]
        st.caption(
            f"Model quality: n = {reg.shape[0]:,},  R² (robust se) = {model.rsquared:.3f}  \n"
            f"Slope (elasticity) = {beta:.3f} (95% CI: {ci_lo:.3f}, {ci_hi:.3f}). "
            f"Interpretation: +1% reviews ≈ +{(10**beta - 1)*100:.1f}% installs (on average)."
        )

# -------------------------
# Free vs Paid — OLS with Category FE
# -------------------------
with tab_fe:
    st.subheader("Free vs Paid — OLS with Category dummies (robust SE)")

    # Build modeling frame (drop rows missing core predictors)
    fe = flt.dropna(subset=["Rating_num", "log10_installs"]).copy()
    if fe.shape[0] < 200:
        st.info("Not enough data after filters for a stable FE regression.")
    else:
        # Feature set
        main_cols = ["Is_Paid_num", "Rating_num"]
        if "Size_MB" in fe.columns:
            main_cols.append("Size_MB")
        if "Recency_years" in fe.columns:
            main_cols.append("Recency_years")

        X_base = fe[main_cols].astype(float)
        # Category FE
        dums = pd.get_dummies(fe["Category_text"], prefix="cat", drop_first=True)
        X = pd.concat([X_base, dums], axis=1)
        X = sm.add_constant(X)
        y = fe["log10_installs"].astype(float).values

        model = sm.OLS(y, X).fit(cov_type="HC1")

        # Coeff table (main effects only)
        keep = ["const", "Is_Paid_num", "Rating_num", "Size_MB", "Recency_years"]
        keep = [k for k in keep if k in model.params.index]
        ci = model.conf_int().loc[keep]
        coef_tbl = pd.DataFrame({
            "coef": model.params.loc[keep],
            "std_err": model.bse.loc[keep],
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": model.pvalues.loc[keep]
        })
        st.dataframe(coef_tbl.style.format(precision=6), use_container_width=True)

        # Interpret paid effect on original scale (approx, log10 → multiplier)
        if "Is_Paid_num" in model.params.index:
            paid_beta = model.params["Is_Paid_num"]
            paid_ci = model.conf_int().loc["Is_Paid_num"]
            mult = 10**paid_beta
            mult_lo, mult_hi = 10**paid_ci[0], 10**paid_ci[1]
            st.caption(
                f"Interpretation: being Paid multiplies installs by ≈ **{mult:.2f}×** "
                f"(95% CI: {mult_lo:.2f}–{mult_hi:.2f}). Values < 1 imply a penalty vs Free."
            )

        # 2-point plot (expected log10(installs): Free vs Paid), others at their means (baseline category)
        means = X.drop(columns=[c for c in X.columns if c.startswith("cat_")]).mean()
        for is_paid in [0.0, 1.0]:
            means[f"Is_Paid_num"] = is_paid
            if "const" not in means.index:  # safety
                means = pd.concat([pd.Series({"const": 1.0}), means])
            if "const" in X.columns and means.get("const", 0) != 1.0:
                means["const"] = 1.0

        def yhat_and_ci(row: pd.Series):
            exog = row.reindex(model.params.index).fillna(0.0).values.reshape(1, -1)
            pr = model.get_prediction(exog=exog)
            sf = pr.summary_frame(alpha=0.05)
            return float(sf["mean"]), float(sf["mean_ci_lower"]), float(sf["mean_ci_upper"])

        row_free = X.drop(columns=[c for c in X.columns if c.startswith("cat_")]).mean()
        row_paid = row_free.copy()
        row_free["Is_Paid_num"] = 0.0
        row_paid["Is_Paid_num"] = 1.0
        for nm in ["const"]:
            if nm in model.params.index and nm not in row_free.index:
                row_free.loc[nm] = 1.0
                row_paid.loc[nm] = 1.0

        y0, lo0, hi0 = yhat_and_ci(row_free)
        y1, lo1, hi1 = yhat_and_ci(row_paid)

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        xs = ["Free", "Paid"]
        ys = [y0, y1]
        los = [lo0, lo1]
        his = [hi0, hi1]
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                    fmt="o", capsize=6, lw=2)
        ax.set_ylabel("Expected log10(installs)")
        ax.set_title("Expected log10(installs): Free vs Paid (95% CI), controls at mean")
        st.pyplot(fig, use_container_width=True)

# -------------------------
# About
# -------------------------
with tab_about:
    st.markdown(
        """
        ### About this app
        - **Overview**: headline KPIs and a sample.
        - **Explore**: basic EDA (distribution, top categories, scatter).
        - **Simpson**: compare unweighted vs weighted rating means; a reversal can emerge.
        - **Regression**: log–log OLS of Installs ~ Reviews with 95% CI & prediction interval.
        - **Free vs Paid (FE)**: OLS with Category fixed effects and robust SE; shows paid effect size.
        
        **Data loading**  
        - You can upload your CSV or paste a public CSV URL (Dropbox/GitHub raw).  
        - If none provided, the app uses `data/apps_clean.csv.gz` from the repo.

        **Notes**  
        - New Matplotlib versions changed `tick_params` behavior; we rotate ticks safely via a helper.
        - All regressions use robust (HC1) standard errors.
        """
    )
