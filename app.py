import streamlit as st                   # Streamlit UI
import pandas as pd                      # DataFrame handling

st.set_page_config(page_title="Play Store Explorer", layout="wide")  # Nice layout

@st.cache_data(show_spinner=False, ttl=86400)  # Cache for 24h so reloads are instant
def load_data():
    return pd.read_csv("data/apps_clean.csv.gz")  # Load local file from repo

with st.spinner("Loading data..."):
    df = load_data()

st.success(f"Loaded {df.shape[0]:,} rows")


# app.py
# Streamlit app for Google Play analysis (EDA, Simpson's paradox, OLS regressions)
# All comments and UI text in English, per course standard.

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Google Play – Data Science Midterm",
    page_icon="📱",
    layout="wide"
)
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")

# ----------------------------
# Helpers: parsing & cleaning
# ----------------------------
def parse_installs(x):
    # Parse strings like "1,000,000+" into float low-bound
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("+", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_size_mb(x):
    # Parse "14M", "800k", "Varies with device" into MB (float)
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().lower()
    if s in {"varies with device", "nan", ""}:
        return np.nan
    try:
        if s.endswith("mb"):
            return float(s[:-2])
        if s.endswith("m"):
            return float(s[:-1])
        if s.endswith("kb"):
            return float(s[:-2]) / 1024.0
        if s.endswith("k"):
            return float(s[:-1]) / 1024.0
        return float(s)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def load_data_from_url(url: str) -> pd.DataFrame:
    # Load CSV from a public URL (Dropbox/GitHub raw/Kaggle direct)
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV from URL: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data_from_upload(file) -> pd.DataFrame:
    # Load CSV from user upload
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return pd.DataFrame()

def choose_first_present(df: pd.DataFrame, candidates):
    # Return first existing column name from candidates
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def clean_make_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Build a modeling-ready frame with engineered features.
    df = df_raw.copy()

    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # Core: App, Category, Rating, Reviews, Size, Installs, Type, Price, Last Updated
    col_app      = choose_first_present(df, ["App", "app"])
    col_cat      = choose_first_present(df, ["Category", "category", "Category (categorical)"])
    col_rating   = choose_first_present(df, ["Rating", "rating", "Rating_num"])
    col_reviews  = choose_first_present(df, ["Reviews", "reviews", "Reviews_num"])
    col_size     = choose_first_present(df, ["Size", "size", "Size_MB"])
    col_installs = choose_first_present(df, ["Installs", "installs", "Installs_num"])
    col_type     = choose_first_present(df, ["Type", "type", "Type_clean"])
    col_price    = choose_first_present(df, ["Price", "price", "Price_USD"])
    col_updated  = choose_first_present(df, ["Last Updated", "last updated", "LastUpdated"])

    # Rename a safe subset so later code is stable
    rename_map = {}
    if col_app:      rename_map[col_app]      = "App"
    if col_cat:      rename_map[col_cat]      = "Category_raw"
    if col_rating:   rename_map[col_rating]   = "Rating_raw"
    if col_reviews:  rename_map[col_reviews]  = "Reviews_raw"
    if col_size:     rename_map[col_size]     = "Size_raw"
    if col_installs: rename_map[col_installs] = "Installs_raw"
    if col_type:     rename_map[col_type]     = "Type_raw"
    if col_price:    rename_map[col_price]    = "Price_raw"
    if col_updated:  rename_map[col_updated]  = "Updated_raw"
    df = df.rename(columns=rename_map)

    # Build numeric fields
    df["Installs_num"] = df.get("Installs_raw", np.nan).apply(parse_installs)
    df["Reviews_num"]  = pd.to_numeric(df.get("Reviews_raw", np.nan), errors="coerce")
    df["Rating_num"]   = pd.to_numeric(df.get("Rating_raw", np.nan),  errors="coerce")
    df["Size_MB"]      = df.get("Size_raw", np.nan).apply(parse_size_mb)

    # Type / Paid
    if "Type_raw" in df.columns:
        t = df["Type_raw"].astype(str).str.strip().str.lower()
        df["Type_clean"] = np.where(t.eq("paid"), "paid", "free")
    else:
        # fallback via price
        price_num = pd.to_numeric(df.get("Price_raw", 0), errors="coerce").fillna(0.0)
        df["Type_clean"] = np.where(price_num > 0, "paid", "free")
    df["Is_Paid"]     = df["Type_clean"].eq("paid")
    df["Is_Paid_num"] = df["Is_Paid"].astype(int)

    # Category text
    if "Category_raw" in df.columns:
        df["Category_text"] = (df["Category_raw"].astype(str)
                               .str.replace("_", " ", regex=False)
                               .str.title())
    else:
        df["Category_text"] = "Unknown"

    # Updated date
    df["LastUpdated"] = pd.to_datetime(df.get("Updated_raw", np.nan), errors="coerce")

    # Recency, log features
    df["Recency_years"]   = ((pd.Timestamp.today(tz=None) - df["LastUpdated"]).dt.days / 365.25).astype(float)
    df["log10_installs"]  = np.log10(df["Installs_num"].replace(0, np.nan))
    df["log10_reviews"]   = np.log10(df["Reviews_num"].replace(0, np.nan))

    # Basic pruning
    df = df.dropna(subset=["Installs_num", "Reviews_num"])

    return df

# ----------------------------
# Sidebar I/O
# ----------------------------
st.sidebar.header("Data")
default_url = st.sidebar.text_input(
    "Optional: CSV URL (public)",
    value="",
    help="Paste a public CSV link (Dropbox/GitHub raw). Leave empty to use an uploaded file."
)
uploaded = st.sidebar.file_uploader("Or upload a CSV", type=["csv"])

if default_url:
    df_raw = load_data_from_url(default_url)
elif uploaded is not None:
    df_raw = load_data_from_upload(uploaded)
else:
    st.sidebar.info("Load data via URL or upload a CSV to begin.")
    st.stop()

if df_raw.empty:
    st.error("No data loaded. Please provide a valid CSV.")
    st.stop()

df = clean_make_features(df_raw)
if df.empty:
    st.error("Data cleaning failed—please check your file.")
    st.stop()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

cats = sorted(df["Category_text"].dropna().unique().tolist())
pick_cats = st.sidebar.multiselect("Category", options=cats, default=[])
type_opts = ["Free", "Paid"]
pick_type = st.sidebar.multiselect("Type", options=type_opts, default=type_opts)

min_reviews = st.sidebar.slider("Minimum reviews", min_value=0, max_value=int(df["Reviews_num"].max()), value=0, step=50)
max_recency = st.sidebar.slider("Max years since last update", min_value=0.0, max_value=float(np.nan_to_num(df["Recency_years"].quantile(0.99), nan=10.0)),
                                value=float(np.nan_to_num(df["Recency_years"].quantile(0.95), nan=5.0)), step=0.25)

mask = (df["Reviews_num"] >= min_reviews) & (df["Recency_years"].fillna(max_recency) <= max_recency)
if pick_cats:
    mask &= df["Category_text"].isin(pick_cats)
if len(pick_type) < 2:
    want_paid = "Paid" in pick_type
    mask &= df["Is_Paid"].eq(want_paid)

dfv = df.loc[mask].copy()

# ----------------------------
# KPIs
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Apps", f"{len(dfv):,}")
c2.metric("Median rating", f"{dfv['Rating_num'].median(skipna=True):.2f}")
c3.metric("Paid share", f"{(dfv['Is_Paid'].mean()*100):.1f}%")
c4.metric("Median installs (approx)", f"{np.floor(dfv['Installs_num'].median()):,.0f}")

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Explore", "Simpson", "Regression (log–log)", "Free vs Paid (FE)", "About"])

# --- Overview
with tab1:
    st.subheader("Data overview")
    st.write("Sample rows after cleaning/feature engineering:")
    st.dataframe(dfv.head(20), use_container_width=True)

    st.write("Columns available:")
    st.code(", ".join(dfv.columns))

# --- Explore
with tab2:
    st.subheader("Exploration")

    colA, colB = st.columns(2, gap="large")
    with colA:
        st.caption("Rating distribution")
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        sns.histplot(dfv["Rating_num"].dropna(), bins=20, ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    with colB:
        st.caption("Top 12 categories by median installs")
        top = (dfv.groupby("Category_text", observed=False)["Installs_num"]
                 .median().sort_values(ascending=False).head(12))
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        sns.barplot(x=top.values, y=top.index, ax=ax, orient="h")
        ax.set_xlabel("Median installs (approx)")
        ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)

    st.caption("log10(Installs) vs log10(Reviews) — raw scatter")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(dfv["log10_reviews"], dfv["log10_installs"], s=8, alpha=0.2)
    ax.set_xlabel("log10(Reviews) • +1 = ×10")
    ax.set_ylabel("log10(Installs) • +1 = ×10")
    st.pyplot(fig, use_container_width=True)

# --- Simpson
with tab3:
    st.subheader("Simpson’s paradox — unweighted vs weighted ratings")

    cat_for_simpson = st.selectbox("Pick a category", options=sorted(dfv["Category_text"].dropna().unique().tolist()))
    sub = dfv[dfv["Category_text"] == cat_for_simpson].copy()

    # Unweighted + weighted mean rating
    overall_unw = dfv["Rating_num"].mean(skipna=True)
    m_overall = dfv["Rating_num"].notna() & dfv["Reviews_num"].notna()
    overall_w = np.average(dfv.loc[m_overall, "Rating_num"], weights=dfv.loc[m_overall, "Reviews_num"])

    cat_unw = sub["Rating_num"].mean(skipna=True)
    m_cat = sub["Rating_num"].notna() & sub["Reviews_num"].notna()
    cat_w = np.average(sub.loc[m_cat, "Rating_num"], weights=sub.loc[m_cat, "Reviews_num"]) if m_cat.any() else np.nan

    plotdf = pd.DataFrame({
        "Group": ["Overall","Overall",cat_for_simpson,cat_for_simpson],
        "Mean":  ["Unweighted","Weighted","Unweighted","Weighted"],
        "Value": [overall_unw, overall_w, cat_unw, cat_w]
    })

    colors = ["#4C78A8","#4C78A8","#72B7B2","#72B7B2"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(np.arange(4), plotdf["Value"].values, color=colors, width=0.6)
    ax.set_xticks(np.arange(4), [f"Overall\n({m})" if g=="Overall" else f"{g}\n({m})" for g,m in zip(plotdf["Group"], plotdf["Mean"])])
    ax.set_ylabel("Average rating")
    ax.set_title(f"Simpson-like reversal: Overall vs {cat_for_simpson}")
    ax.set_ylim(3.8, 4.7)
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, y+0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig, use_container_width=True)

# --- Regression (log–log)
with tab4:
    st.subheader("OLS: log10(Installs) ~ log10(Reviews) with 95% CI & prediction interval")

    # Prepare data
    dd = dfv.dropna(subset=["log10_installs","log10_reviews"]).copy()
    if len(dd) < 30:
        st.warning("Not enough data after filtering for a stable regression. Relax filters.")
    else:
        X = sm.add_constant(dd[["log10_reviews"]])
        y = dd["log10_installs"]
        ols = sm.OLS(y, X).fit(cov_type="HC1")

        st.write(f"**n = {int(ols.nobs)} | R-squared = {ols.rsquared:.3f}**")

        # Coef table
        ci = pd.DataFrame(ols.conf_int(0.05), columns=["ci_low","ci_high"], index=ols.params.index)
        coef_tbl = pd.concat([ols.params.rename("coef"), ols.bse.rename("std_err"), ci, ols.pvalues.rename("p_value")], axis=1)
        st.dataframe(coef_tbl.style.format(precision=4), use_container_width=True)

        # Grid predictions
        xg = np.linspace(dd["log10_reviews"].min(), dd["log10_reviews"].max(), 250)
        Xg = sm.add_constant(pd.DataFrame({"log10_reviews": xg}))
        pr = ols.get_prediction(Xg).summary_frame(alpha=0.05)

        # Hexbin layer + bands
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        hb = ax.hexbin(dd["log10_reviews"], dd["log10_installs"], gridsize=45, mincnt=1, cmap="Blues", alpha=0.85)
        ax.plot(xg, pr["mean"], lw=2, label="OLS fit")
        ax.fill_between(xg, pr["mean_ci_lower"], pr["mean_ci_upper"], alpha=0.22, label="95% CI (mean)")
        ax.fill_between(xg, pr["obs_ci_lower"], pr["obs_ci_upper"], alpha=0.12, label="95% prediction interval")
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Point density")
        ax.set_xlabel("log10(Reviews) • +1 = ×10")
        ax.set_ylabel("log10(Installs) • +1 = ×10")
        ax.set_title("Installs vs Reviews — OLS with 95% CI & prediction band")
        ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=True)

        beta = float(ols.params["log10_reviews"])
        st.info(f"Elasticity (slope): **{beta:.3f}** → +1% reviews ≈ +{beta:.1f}% installs (on average).")

# --- Free vs Paid (with FE)
with tab5:
    st.subheader("Free vs Paid — adjusted effect with controls & category FE")

    # Prepare data
    dff = dfv.dropna(subset=["log10_installs","Rating_num","Size_MB","Recency_years","Category_text"]).copy()
    if len(dff) < 100:
        st.warning("Not enough data after filtering for a robust FE model. Relax filters.")
    else:
        dff["Is_Paid_num"] = dff["Is_Paid"].astype(int)
        formula = "log10_installs ~ Is_Paid_num + Rating_num + Size_MB + Recency_years + C(Category_text)"
        fe = smf.ols(formula, data=dff).fit(cov_type="HC1")

        st.write(f"**n = {int(fe.nobs)} | R-squared = {fe.rsquared:.3f}**")

        ci = fe.conf_int(0.05)
        ci.columns = ["ci_low","ci_high"]
        keep = ["Is_Paid_num","Rating_num","Size_MB","Recency_years"]
        coef_tbl = pd.concat([fe.params.rename("coef"),
                              fe.bse.rename("std_err"),
                              ci,
                              fe.pvalues.rename("p_value")], axis=1).loc[keep]
        st.dataframe(coef_tbl.style.format(precision=4), use_container_width=True)

        # Marginal means (Free vs Paid) keeping sample mix
        X = pd.DataFrame(fe.model.exog, columns=fe.model.exog_names)
        cov = fe.cov_params()
        g_free = X.copy(); g_free["Is_Paid_num"] = 0
        g_paid = X.copy(); g_paid["Is_Paid_num"] = 1
        gf = g_free.mean(axis=0).values
        gp = g_paid.mean(axis=0).values
        mean_free = float(gf @ fe.params.values)
        mean_paid = float(gp @ fe.params.values)
        se_free = float(np.sqrt(gf @ cov.values @ gf))
        se_paid = float(np.sqrt(gp @ cov.values @ gp))
        z = 1.96
        means = pd.DataFrame({
            "group": ["Free","Paid"],
            "mean":  [mean_free, mean_paid],
            "ci_lo": [mean_free - z*se_free, mean_paid - z*se_paid],
            "ci_hi": [mean_free + z*se_free, mean_paid + z*se_paid],
        })

        # Plot bars
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        xpos = np.arange(2)
        bars = ax.bar(xpos, means["mean"], width=0.6, color=["#4C78A8","#F58518"], edgecolor="black", linewidth=0.5)
        yerr = np.vstack([means["mean"] - means["ci_lo"], means["ci_hi"] - means["mean"]])
        ax.errorbar(xpos, means["mean"], yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=5, capthick=1.2)
        ax.set_xticks(xpos, ["Free","Paid"])
        ax.set_ylabel("Expected log10(installs)")
        ax.set_title("Expected log10(installs): Free vs Paid (95% CI), with controls")
        for i,b in enumerate(bars):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{means['mean'][i]:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(min(means["ci_lo"])-0.1, max(means["ci_hi"])+0.2)
        st.pyplot(fig, use_container_width=True)

        paid_mult = 10 ** fe.params["Is_Paid_num"]
        st.info(f"Is_Paid effect → installs × **{paid_mult:.2f}** (values < 1 imply a penalty vs Free).")

# --- About
with tab6:
    st.subheader("About this app")
    st.markdown("""
This Streamlit app implements the core analyses from my midterm:
- Data cleaning + feature engineering consistent with the course notebook.
- Exploration (rating distribution, category medians, scatter log–log).
- Simpson’s paradox demo (unweighted vs weighted ratings).
- OLS regressions with robust SE: log–log Installs~Reviews (95% CI & prediction band) and Free vs Paid with controls & category fixed effects.
Notes:
- Install counts on Google Play are bucketed; hexbin/jitter are used for visualization only.
- All regressions are **associational**, not causal.
    """)
