import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------- Page config -----------------------------------
st.set_page_config(page_title="Play Store Explorer", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "apps_clean.csv.gz"

# ----------------------------- Small helpers ---------------------------------
def _to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def parse_installs(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s.astype(float)
    low = s.astype(str).str.replace(r"[^\d]", "", regex=True)
    return pd.to_numeric(low, errors="coerce")

def parse_size_mb(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    varies = s.str.contains("varies")
    mb = pd.to_numeric(s.str.replace("m", "", regex=False), errors="coerce")
    kb = pd.to_numeric(s.str.replace("k", "", regex=False), errors="coerce")
    out = mb.combine_first(kb / 1024.0)
    out[varies] = np.nan
    return out

def safe_rotate_xticks(ax, rotation=45, ha="right"):
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rotation)
        lbl.set_horizontalalignment(ha)

def top_n_categories(cat: pd.Series, n=40) -> pd.Series:
    counts = cat.value_counts(dropna=False)
    keep = set(counts.head(n).index)
    return cat.where(cat.isin(keep), "Other")

# ----------------------------- Load & clean ----------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_raw_csv(url: Optional[str]) -> pd.DataFrame:
    if url:
        return pd.read_csv(url, low_memory=False)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, low_memory=False)
    st.stop()

def pick_category_text(df: pd.DataFrame) -> pd.Series:
    # Prefer an existing text/label categorical column
    lower = {c.lower(): c for c in df.columns}
    for key in lower:
        if "category" in key and ("categorical" in key or "label" in key or "text" in key or "name" in key):
            return df[lower[key]].astype(str)

    if "Category" in df.columns:
        if df["Category"].dtype == object:
            return df["Category"].astype(str)
        # numeric fallback – still cast to str so filters show labels not floats
        return df["Category"].astype(str)

    # Last resort
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize names
    rename = {c: c.strip().replace(" ", "_").replace("-", "_") for c in df.columns}
    df.rename(columns=rename, inplace=True)

    # Category text
    df["Category_text"] = pick_category_text(df)

    # Rating
    if "Rating" in df.columns:
        df["Rating_num"] = pd.to_numeric(df["Rating"], errors="coerce")
    elif "rating" in df.columns:
        df["Rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["Rating_num"] = np.nan

    # Reviews
    for c in ["Reviews", "reviews"]:
        if c in df.columns:
            df["Reviews_num"] = _to_numeric_series(df[c])
            break
    else:
        df["Reviews_num"] = np.nan

    # Installs
    for c in ["Installs", "installs"]:
        if c in df.columns:
            df["Installs_num"] = parse_installs(df[c])
            break
    else:
        df["Installs_num"] = np.nan

    # Size
    for c in ["Size", "size"]:
        if c in df.columns:
            df["Size_MB"] = parse_size_mb(df[c])
            break
    else:
        df["Size_MB"] = np.nan

    # Last updated
    for c in ["Last_Updated", "LastUpdated", "last_updated", "lastUpdate"]:
        if c in df.columns:
            df["LastUpdated"] = pd.to_datetime(df[c], errors="coerce")
            break
    else:
        df["LastUpdated"] = pd.NaT

    # Price (for Is_Paid fallback)
    price_num = None
    for c in ["Price", "price"]:
        if c in df.columns:
            price_num = _to_numeric_series(df[c])
            break
    if price_num is None:
        price_num = pd.Series(np.nan, index=df.index)

    # Robust Is_Paid
    if "Type" in df.columns:
        typ = df["Type"]
        if typ.dtype == object:
            t = typ.astype(str).str.strip().str.lower()
            df["Is_Paid"] = t.eq("paid").astype(int)
        else:
            # numeric; align with price if available
            tn = pd.to_numeric(typ, errors="coerce").fillna(0).astype(int)
            if price_num.notna().any():
                paid_by_price = (price_num.fillna(0) > 0).astype(int)
                agree = (tn == paid_by_price).mean()
                agree_flip = ((1 - tn) == paid_by_price).mean()
                df["Is_Paid"] = (tn if agree >= agree_flip else (1 - tn)).astype(int)
            else:
                df["Is_Paid"] = tn.astype(int)
    else:
        df["Is_Paid"] = (price_num.fillna(0) > 0).astype(int)

    keep = (~df["Reviews_num"].isna()) | (~df["Installs_num"].isna())
    return df.loc[keep].reset_index(drop=True)

# ----------------------------- Sidebar ---------------------------------------
with st.sidebar:
    st.header("Filters")
    url = st.text_input("Optional: CSV URL (public)", value="")
    st.caption("Leave empty to use the sample in the repo.")

df_raw = load_raw_csv(url if url.strip() else None)
df = ensure_columns(df_raw)

with st.sidebar:
    all_cats = sorted(df["Category_text"].dropna().astype(str).unique().tolist())
    cats = st.multiselect("Categories", all_cats, default=all_cats)
    type_choice = st.radio("Type", ["All", "Free", "Paid"], horizontal=True)

    min_reviews = st.slider("Minimum reviews", 0, int(df["Reviews_num"].fillna(0).max()), 0, step=1)
    years = df["LastUpdated"].dropna().dt.year
    yr_min, yr_max = (2010, 2018) if years.empty else (int(years.min()), int(years.max()))
    year_from, year_to = st.slider("Updated year range", yr_min, yr_max, (yr_min, yr_max))

# Apply filters
mask = df["Category_text"].astype(str).isin(cats) if cats else True
mask &= df["Reviews_num"].fillna(0) >= min_reviews
if not df["LastUpdated"].isna().all():
    y = df["LastUpdated"].dt.year
    mask &= (y >= year_from) & (y <= year_to)
if type_choice != "All":
    mask &= (df["Is_Paid"] == (1 if type_choice == "Paid" else 0))
dff = df.loc[mask].copy()

st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
st.caption(f"Active filter ≈ {len(dff):,} apps")

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Apps", f"{len(dff):,}")
with c2:
    med_rating = dff["Rating_num"].median()
    st.metric("Median rating", f"{med_rating:,.2f}" if pd.notna(med_rating) else "—")
with c3:
    paid_share = dff["Is_Paid"].mean() * 100 if len(dff) else np.nan
    st.metric("Paid share", f"{paid_share:,.1f}%" if pd.notna(paid_share) else "—")
with c4:
    med_inst = dff["Installs_num"].median()
    st.metric("Median installs", f"{int(med_inst):,}" if pd.notna(med_inst) else "—")

# Tabs
tab_overview, tab_explore, tab_simpson, tab_reg, tab_fe, tab_about = st.tabs(
    ["Overview", "Explore", "Simpson", "Regression", "Free vs Paid (FE)", "About"]
)

# Overview
with tab_overview:
    st.subheader("Sample (first 10)")
    cols = ["App", "Category_text", "Rating_num", "Reviews_num", "Size_MB", "Installs_num", "Type"]
    cols = [c for c in cols if c in dff.columns]
    st.dataframe(dff[cols].head(10), use_container_width=True, hide_index=True)

# Explore
with tab_explore:
    st.subheader("Quick EDA")
    l, r = st.columns(2)
    with l:
        st.caption("Rating distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dff["Rating_num"].dropna(), bins=30)
        ax.set_xlabel("Rating (1–5)")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)
    with r:
        st.caption("Top categories by count")
        top = dff["Category_text"].value_counts().head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index, top.values)
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        st.pyplot(fig, clear_figure=True)

# Simpson
with tab_simpson:
    st.subheader("Weighted vs unweighted rating means")
    if len(dff) == 0:
        st.info("No data after filters.")
    else:
        overall_unw = dff["Rating_num"].mean()
        w = dff["Reviews_num"].clip(lower=0).fillna(0).replace(0, 1)
        overall_w = np.average(dff["Rating_num"].fillna(overall_unw), weights=w)

        cat_counts = dff["Category_text"].value_counts()
        if len(cat_counts) == 0:
            st.info("No categories available after filters.")
        else:
            top_cat = st.selectbox("Category to highlight", cat_counts.index.tolist(), index=0)
            cat_df = dff[dff["Category_text"] == top_cat]
            cu = cat_df["Rating_num"].mean()
            cw = np.average(cat_df["Rating_num"].fillna(cu), weights=cat_df["Reviews_num"].clip(lower=0).fillna(0).replace(0, 1))

            labels = ["Overall (Unweighted)", "Overall (Weighted)", f"{top_cat} (Unweighted)", f"{top_cat} (Weighted)"]
            vals = [overall_unw, overall_w, cu, cw]
            fig, ax = plt.subplots(figsize=(8, 4))
            overall_color = "#1f77b4"  # כחול לכלליים
            cat_color     = "#ff7f0e"  # כתום לקטגוריה
            colors = [overall_color, overall_color, cat_color, cat_color]
            ax.bar(range(len(vals)), vals, color=colors)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Average rating")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig, clear_figure=True)

# Regression (log-log)
with tab_reg:
    st.subheader("Installs vs Reviews — OLS (log10–log10) with 95% CI & prediction band")
    reg = dff[(dff["Installs_num"] > 0) & (dff["Reviews_num"] > 0)].copy()
    if len(reg) < 100:
        st.info("Not enough data after filters for a stable regression.")
    else:
        reg["logI"] = np.log10(reg["Installs_num"])
        reg["logR"] = np.log10(reg["Reviews_num"])
        X = sm.add_constant(reg["logR"].astype(float))
        y = reg["logI"].astype(float)
        model = sm.OLS(y, X).fit(cov_type="HC1")

        xg = np.linspace(reg["logR"].min(), reg["logR"].max(), 100)
        Xg = sm.add_constant(pd.Series(xg, name="logR"))
        yg = model.predict(Xg)

        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        _, ci_low, ci_high = wls_prediction_std(model, Xg, alpha=0.05)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(reg["logR"], reg["logI"], s=6, alpha=0.25)
        ax.plot(xg, yg)
        ax.fill_between(xg, ci_low, ci_high, alpha=0.2)
        ax.set_xlabel("log10(Reviews)")
        ax.set_ylabel("log10(Installs)")
        txt = (f"$R^2$ = {model.rsquared:,.3f}\n"
               f"Slope = {model.params['logR']:,.3f}")
        ax.text(0.98, 0.02, txt, ha="right", va="bottom", transform=ax.transAxes, fontsize=10)
        st.pyplot(fig, clear_figure=True)

# Free vs Paid (FE)
with tab_fe:
    st.subheader("Free vs Paid — OLS with Category dummies (robust SE)")
    fe = dff[(dff["Installs_num"] > 0) & (~dff["Is_Paid"].isna())].copy()
    if fe["Is_Paid"].nunique() < 2 or len(fe) < 200:
        st.info("Not enough data after filters for a stable FE regression.")
    else:
        fe["logI"] = np.log10(fe["Installs_num"].clip(lower=1))
        fe["Category_FE"] = top_n_categories(fe["Category_text"].astype(str), n=40)

        base = pd.DataFrame({
            "Is_Paid": fe["Is_Paid"].astype(int),
            "Rating_num": pd.to_numeric(fe["Rating_num"], errors="coerce"),
            "Size_MB": pd.to_numeric(fe["Size_MB"], errors="coerce"),
        })
        base["Rating_num"].fillna(base["Rating_num"].median(), inplace=True)
        base["Size_MB"].fillna(base["Size_MB"].median(), inplace=True)

        dummies = pd.get_dummies(fe["Category_FE"].astype(str), prefix="cat", drop_first=True, dtype=float)
        X = pd.concat([base, dummies], axis=1)

        # Clean X/y: numeric only, no NaN/Inf, add constant
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        y = pd.to_numeric(fe["logI"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = np.isfinite(y.to_numpy()) & np.isfinite(X.to_numpy()).all(axis=1)
        X, y = X.loc[mask], y.loc[mask]
        # Drop any all-zero columns to avoid singularities
        X = X.loc[:, (X != 0).any(axis=0)]
        X = sm.add_constant(X, has_constant="add")

        fe_model = sm.OLS(y, X).fit(cov_type="HC1")

        coef = fe_model.params.get("Is_Paid", np.nan)
        ci = fe_model.conf_int().loc["Is_Paid"].tolist() if "Is_Paid" in fe_model.params.index else [np.nan, np.nan]
        mult = 10 ** coef if pd.notna(coef) else np.nan

        st.write("**Model summary (key fields):**")
        st.code(f"n = {int(y.shape[0]):,}\nR-squared = {fe_model.rsquared:,.3f}")
        st.dataframe(
            pd.DataFrame(
                {"coef": [coef], "ci_low": [ci[0]], "ci_high": [ci[1]], "multiplier_on_installs": [mult]},
                index=["Is_Paid"]
            ),
            use_container_width=True
        )

        # Visualize expected log10(installs) for Free vs Paid at mean controls
        X0 = X.copy(); X0["Is_Paid"] = 0.0
        X1 = X.copy(); X1["Is_Paid"] = 1.0
        y0 = fe_model.predict(X0).median()
        y1 = fe_model.predict(X1).median()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Free", "Paid"], [y0, y1], color=["#1f77b4", "#ff7f0e"])
        for i, v in enumerate([y0, y1]):
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
