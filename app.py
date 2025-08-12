import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------- Page config -----------------------------------
st.set_page_config(page_title="Play Store Explorer", layout="wide")

# ----------------------------- Helpers ---------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "apps_clean.csv.gz"

def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a column with $ signs, commas, plus signs, 'Varies with device', etc. to numeric."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)  # keep digits/dot/minus/scientific
    out = pd.to_numeric(s, errors="coerce")
    return out

def parse_installs(s: pd.Series) -> pd.Series:
    """Convert values like '1,000,000+' to the lower bound 1000000."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    low = s.astype(str).str.replace(r"[^\d]", "", regex=True)
    return pd.to_numeric(low, errors="coerce")

def parse_size_mb(s: pd.Series) -> pd.Series:
    """Convert '25M', '800k', 'Varies with device' to MB (approx)."""
    s = s.astype(str).str.strip().str.lower()
    varies = s.str.contains("varies")
    mb = pd.to_numeric(s.str.replace("m", "", regex=False), errors="coerce")
    kb = pd.to_numeric(s.str.replace("k", "", regex=False), errors="coerce")
    out = mb
    out = out.combine_first(kb / 1024.0)
    out[varies] = np.nan
    return out

def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def safe_rotate_xticks(ax, rotation=45, ha="right"):
    """Rotate x tick labels without using unsupported tick_params kwargs."""
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rotation)
        lbl.set_horizontalalignment(ha)

def top_n_categories(cat_series: pd.Series, n=40) -> pd.Series:
    """Keep top-n frequent categories; others -> 'Other'."""
    counts = cat_series.value_counts(dropna=False)
    keep = set(counts.head(n).index.tolist())
    return cat_series.where(cat_series.isin(keep), "Other")

def confint_95(result, param):
    ci = result.conf_int(alpha=0.05)
    if isinstance(ci, pd.DataFrame):
        if param in ci.index:
            return float(ci.loc[param, 0]), float(ci.loc[param, 1])
        return (np.nan, np.nan)
    if isinstance(ci, np.ndarray) and ci.shape[0] == len(result.params):
        idx = list(result.params.index).index(param)
        return float(ci[idx, 0]), float(ci[idx, 1])
    return (np.nan, np.nan)

# ----------------------------- Loading & cleaning -----------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_raw_csv(url: str | None) -> pd.DataFrame:
    if url:
        return pd.read_csv(url, low_memory=False)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, low_memory=False)
    st.stop()  # No data available

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names: spaces->_, dashes->_, unify casing
    rename = {c: c.strip().replace(" ", "_").replace("-", "_") for c in df.columns}
    df.rename(columns=rename, inplace=True)

    # Make sure required columns exist, creating fallbacks when possible
    # Category text
    if "Category" in df.columns:
        df["Category_text"] = df["Category"].astype(str)
    elif "Category_text" not in df.columns:
        df["Category_text"] = "Unknown"

    # Ratings numeric
    rating_col = "Rating" if "Rating" in df.columns else ("rating" if "rating" in df.columns else None)
    if rating_col:
        df["Rating_num"] = pd.to_numeric(df[rating_col], errors="coerce")
    elif "Rating_num" not in df.columns:
        df["Rating_num"] = np.nan

    # Reviews numeric
    reviews_col = "Reviews" if "Reviews" in df.columns else ("reviews" if "reviews" in df.columns else None)
    if reviews_col:
        df["Reviews_num"] = _to_numeric_series(df[reviews_col])
    elif "Reviews_num" not in df.columns:
        df["Reviews_num"] = np.nan

    # Installs numeric
    installs_col = "Installs" if "Installs" in df.columns else ("installs" if "installs" in df.columns else None)
    if installs_col:
        df["Installs_num"] = parse_installs(df[installs_col])
    elif "Installs_num" not in df.columns:
        df["Installs_num"] = np.nan

    # Size -> MB
    size_col = "Size" if "Size" in df.columns else ("size" if "size" in df.columns else None)
    if size_col:
        df["Size_MB"] = parse_size_mb(df[size_col])
    elif "Size_MB" not in df.columns:
        df["Size_MB"] = np.nan

    # Last Updated
    for c in ["Last_Updated", "LastUpdated", "last_updated", "lastUpdate"]:
        if c in df.columns:
            df["LastUpdated"] = coerce_datetime(df[c])
            break
    if "LastUpdated" not in df.columns:
        df["LastUpdated"] = pd.NaT

    # Price (used for Is_Paid fallback)
    price_col = None
    for c in ["Price", "price"]:
        if c in df.columns:
            price_col = c
            break
    price_num = _to_numeric_series(df[price_col]) if price_col else pd.Series(np.nan, index=df.index)

    # Type / Is_Paid robust derivation
    # 1) If Type exists and is string -> paid if equals 'paid'
    # 2) Else, infer from Price > 0
    # 3) If Type is numeric {0,1}, align it to Price mapping (flip if needed)
    if "Type" in df.columns:
        typ = df["Type"]
        if typ.dtype == object:
            t = typ.astype(str).str.strip().str.lower()
            df["Is_Paid"] = t.eq("paid").astype(int)
        else:
            # Numeric Type (unknown encoding) -> align to price
            if price_col is not None:
                paid_by_price = (price_num.fillna(0) > 0).astype(int)
                type_num = pd.to_numeric(typ, errors="coerce").fillna(0).astype(int)
                agree = (type_num == paid_by_price).mean()
                agree_flipped = ((1 - type_num) == paid_by_price).mean()
                df["Is_Paid"] = (type_num if agree >= agree_flipped else (1 - type_num)).astype(int)
            else:
                df["Is_Paid"] = pd.to_numeric(typ, errors="coerce").fillna(0).astype(int)
    else:
        df["Is_Paid"] = (price_num.fillna(0) > 0).astype(int)

    # Clean obvious bad rows
    # Keep rows with at least Reviews or Installs available
    keep = (~df["Reviews_num"].isna()) | (~df["Installs_num"].isna())
    return df.loc[keep].reset_index(drop=True)

# ----------------------------- UI / Filters -----------------------------------
with st.sidebar:
    st.header("Filters")
    url = st.text_input("Optional: CSV URL (public)", value="")
    st.caption("You can leave this empty to use the sample in the repo.")

# Load + clean
df_raw = load_raw_csv(url if url.strip() else None)
df = ensure_columns(df_raw)

with st.sidebar:
    # Category filter
    all_cats = sorted(df["Category_text"].dropna().astype(str).unique().tolist())
    default_cats = all_cats  # select all by default
    cats = st.multiselect("Categories", all_cats, default=default_cats)

    # Type filter
    type_choice = st.radio("Type", options=["All", "Free", "Paid"], horizontal=True)

    # Minimum reviews
    min_reviews = st.slider("Minimum reviews", 0, int(df["Reviews_num"].fillna(0).max()), 0, step=1)

    # Updated years range
    years = df["LastUpdated"].dropna().dt.year
    if years.empty:
        yr_min, yr_max = 2010, 2018
    else:
        yr_min, yr_max = int(years.min()), int(years.max())
    year_from, year_to = st.slider("Updated year range", yr_min, yr_max, (yr_min, yr_max))

# Apply filters
mask = df["Category_text"].astype(str).isin(cats) if len(cats) > 0 else True
mask &= df["Reviews_num"].fillna(0) >= min_reviews
if not df["LastUpdated"].isna().all():
    y = df["LastUpdated"].dt.year
    mask &= (y >= year_from) & (y <= year_to)

if type_choice != "All":
    mask &= (df["Is_Paid"] == (1 if type_choice == "Paid" else 0))

dff = df.loc[mask].copy()
st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
st.caption(f"Active filter ≈ {len(dff):,} apps")

# ----------------------------- KPI row ----------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Apps", f"{len(dff):,}")
with col2:
    med_rating = dff["Rating_num"].median()
    st.metric("Median rating", f"{med_rating:,.2f}" if pd.notna(med_rating) else "—")
with col3:
    if len(dff) > 0:
        paid_share = dff["Is_Paid"].mean() * 100
    else:
        paid_share = np.nan
    st.metric("Paid share", f"{paid_share:,.1f}%" if pd.notna(paid_share) else "—")
with col4:
    med_inst = dff["Installs_num"].median()
    st.metric("Median installs", f"{int(med_inst):,}" if pd.notna(med_inst) else "—")

# ----------------------------- Tabs -------------------------------------------
tab_overview, tab_explore, tab_simpson, tab_reg, tab_fe, tab_about = st.tabs(
    ["Overview", "Explore", "Simpson", "Regression", "Free vs Paid (FE)", "About"]
)

# ----------------------------- Overview ---------------------------------------
with tab_overview:
    st.subheader("Sample (first 10)")
    st.dataframe(
        dff[["App"]+[c for c in ["Category_text", "Rating_num", "Reviews_num", "Size_MB", "Installs_num", "Type"] if c in dff.columns]].head(10),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------- Explore ----------------------------------------
with tab_explore:
    st.subheader("Quick EDA")

    left, right = st.columns(2)
    with left:
        st.caption("Rating distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dff["Rating_num"].dropna(), bins=30)
        ax.set_xlabel("Rating (1–5)")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.caption("Top categories by count")
        top = dff["Category_text"].value_counts().head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index, top.values)
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        st.pyplot(fig, clear_figure=True)

# ----------------------------- Simpson ----------------------------------------
with tab_simpson:
    st.subheader("Weighted vs unweighted rating means (Simpson-like check)")
    if len(dff) == 0:
        st.info("No data after filters.")
    else:
        # Overall
        overall_unw = dff["Rating_num"].mean()
        weights = dff["Reviews_num"].clip(lower=0).fillna(0)
        overall_w = np.average(dff["Rating_num"].fillna(overall_unw), weights=weights.replace(0, 1))

        # Pick a prominent category (by count)
        cat_counts = dff["Category_text"].value_counts()
        if len(cat_counts) > 0:
            top_cat = st.selectbox("Category to highlight", cat_counts.index.tolist(), index=0)
            cat_df = dff[dff["Category_text"] == top_cat]
            cat_unw = cat_df["Rating_num"].mean()
            w_cat = cat_df["Reviews_num"].clip(lower=0).fillna(0).replace(0, 1)
            cat_w = np.average(cat_df["Rating_num"].fillna(cat_unw), weights=w_cat)

            labels = ["Overall (Unweighted)", "Overall (Weighted)",
                      f"{top_cat} (Unweighted)", f"{top_cat} (Weighted)"]
            values = [overall_unw, overall_w, cat_unw, cat_w]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Average rating")
            # annotate
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No categories available after filters.")

# ----------------------------- Regression (log-log) ---------------------------
with tab_reg:
    st.subheader("Installs vs Reviews — OLS (log10–log10) with 95% CI & prediction band")
    reg_df = dff.copy()
    reg_df = reg_df[(reg_df["Installs_num"] > 0) & (reg_df["Reviews_num"] > 0)]
    if len(reg_df) < 100:
        st.info("Not enough data after filters for a stable regression.")
    else:
        reg_df["logI"] = np.log10(reg_df["Installs_num"])
        reg_df["logR"] = np.log10(reg_df["Reviews_num"])
        X = sm.add_constant(reg_df["logR"])
        y = reg_df["logI"]
        model = sm.OLS(y, X).fit(cov_type="HC1")

        slope = model.params["logR"]
        ci_lo, ci_hi = confint_95(model, "logR")

        # Plot
        xg = np.linspace(reg_df["logR"].min(), reg_df["logR"].max(), 100)
        Xg = sm.add_constant(pd.Series(xg, name="logR"))
        yg = model.predict(Xg)

        # 95% CI for mean prediction
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        _, ci_low, ci_high = wls_prediction_std(model, Xg, alpha=0.05)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(reg_df["logR"], reg_df["logI"], s=6, alpha=0.25)
        ax.plot(xg, yg)
        ax.fill_between(xg, ci_low, ci_high, alpha=0.2)
        ax.set_xlabel("log10(Reviews)")
        ax.set_ylabel("log10(Installs)")
        # Text box
        txt = (f"$R^2$ = {model.rsquared:,.3f}\n"
               f"Slope (elasticity) = {slope:,.3f}\n"
               f"95% CI = [{ci_lo:,.3f}, {ci_hi:,.3f}]")
        ax.text(0.98, 0.02, txt, ha="right", va="bottom", transform=ax.transAxes, fontsize=10)
        st.pyplot(fig, clear_figure=True)

# ----------------------------- Free vs Paid (FE) ------------------------------
with tab_fe:
    st.subheader("Free vs Paid — OLS with Category dummies (robust SE)")
    fe_df = dff.copy()
    fe_df = fe_df[(fe_df["Installs_num"] > 0) & (~fe_df["Is_Paid"].isna())]

    # Require both classes
    if fe_df["Is_Paid"].nunique() < 2 or len(fe_df) < 200:
        st.info("Not enough data after filters for a stable FE regression.")
    else:
        # Response on log scale
        fe_df["logI"] = np.log10(fe_df["Installs_num"].clip(lower=1))

        # Keep top-40 categories to avoid a huge dummy matrix
        fe_df["Category_FE"] = top_n_categories(fe_df["Category_text"].astype(str), n=40)

        # Controls
        X = pd.DataFrame({
            "Is_Paid": fe_df["Is_Paid"].astype(int),
            "Rating_num": fe_df["Rating_num"].fillna(fe_df["Rating_num"].median()),
            "Size_MB": fe_df["Size_MB"].fillna(fe_df["Size_MB"].median()),
        })
        X = pd.get_dummies(pd.concat([X, fe_df[["Category_FE"]]], axis=1), columns=["Category_FE"], drop_first=True)
        X = sm.add_constant(X)
        y = fe_df["logI"]

        fe_model = sm.OLS(y, X).fit(cov_type="HC1")
        coef = fe_model.params.get("Is_Paid", np.nan)
        lo, hi = confint_95(fe_model, "Is_Paid")

        # Convert log effect to % multiplier on installs (approx)
        pct_mult = (10 ** coef)  # multiplicative on installs
        st.write("**Model quality:**")
        st.code(
            f"n = {len(fe_df):,}\n"
            f"R-squared (robust se): {fe_model.rsquared:,.3f}"
        )

        # Show a small coef table
        tbl = pd.DataFrame({
            "coef": [coef],
            "ci_low": [lo],
            "ci_high": [hi],
            "multiplier_on_installs": [pct_mult]
        }, index=["Is_Paid"])
        st.dataframe(tbl, use_container_width=True)

        # Simple two-point expected plot (mean controls)
        x0 = X.copy()
        x0.loc[:, "Is_Paid"] = 0
        xp = X.copy()
        xp.loc[:, "Is_Paid"] = 1
        y0 = fe_model.predict(x0).median()
        yp = fe_model.predict(xp).median()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Free", "Paid"], [y0, yp])
        for i, v in enumerate([y0, yp]):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylabel("Expected log10(Installs)")
        st.pyplot(fig, clear_figure=True)

# ----------------------------- About ------------------------------------------
with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
**What we explored:** We analyzed Google Play apps to understand how installs relate to
user reviews and how paid vs free apps differ after controlling for category and other factors.

**Highlights:**
- The log–log regression shows a strong elasticity: as reviews grow, installs tend to grow
  with a stable slope on the log scale.
- Simpson-like reversals can appear: overall averages may flip once we weight by review
  volume or segment by category.
- After controlling for category (fixed effects) and basic features, the `Is_Paid` coefficient
  quantifies the penalty or premium for paid apps in expected installs.

**How to use:**
- Load your CSV or paste a public URL (Dropbox/GitHub raw). If omitted, the app uses
  `data/apps_clean.csv.gz` from the repo.
- Use the sidebar to filter by category, type (Free/Paid), minimum reviews, and update year range.
- Tabs:
  - *Overview*: KPIs and a sample of the filtered data.
  - *Explore*: quick distributions and top categories.
  - *Simpson*: compare unweighted vs weighted means to reveal potential reversals.
  - *Regression*: log–log OLS of Installs vs Reviews with 95% CI and prediction interval.
  - *Free vs Paid (FE)*: OLS with Category fixed effects (robust SE) showing the Paid effect.
"""
    )
