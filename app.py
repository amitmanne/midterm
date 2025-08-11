# app.py — Google Play Explorer (EDA, Simpson, OLS regressions)
# Runs fast on Streamlit Cloud. All plots use safe tick-helpers for Matplotlib>=3.8.

import streamlit as st
import numpy as np
import pandas as pd
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Play Store Explorer", layout="wide")

# ----------------------------
# Small helpers
# ----------------------------
def safe_rotate_x(ax, rot=45, align="right"):
    """Matplotlib 3.8+: avoid ha/va kwargs on tick_params."""
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rot)
        lbl.set_ha(align)

def col_first(df: pd.DataFrame, *cands) -> str:
    """Pick the first column name that exists in df."""
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"None of {cands} found")

def parse_installs(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return np.nan if x <= 0 else float(x)
    s = str(x).strip().replace("+", "").replace(",", "")
    if s.isdigit():
        return float(s)
    return np.nan

def parse_price(x):
    if pd.isna(x):
        return 0.0
    s = str(x).replace("$", "").strip()
    try:
        return float(s)
    except:
        return 0.0

def parse_size_mb(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    try:
        if s.endswith("M"):
            return float(s.replace("M", ""))
        if s.endswith("K"):
            return float(s.replace("K", "")) / 1024.0
        # some datasets already numeric
        return float(s)
    except:
        return np.nan

def year_from_date(x):
    try:
        return pd.to_datetime(x, errors="coerce").year
    except:
        return np.nan

def months_from_date(x):
    try:
        dt = pd.to_datetime(x, errors="coerce")
        return 12 * dt.year + dt.month
    except:
        return np.nan

# ----------------------------
# Data load (local fallback + user CSV/URL)
# ----------------------------
DATA_PATH = Path(__file__).parent / "data" / "apps_clean.csv.gz"

@st.cache_data(show_spinner=False, ttl=86400)
def load_default() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, low_memory=False)

def load_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url, low_memory=False)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and create clean feature set the app expects.
    Works with:
      - Kaggle raw CSV
      - our cleaned CSV (apps_clean.csv.gz)
    """
    # Try to locate category text column
    cat_col = None
    try:
        cat_col = col_first(
            df,
            "Category (categorical)",  # our cleaned
            "Category_text",           # some notebooks
            "Category"                 # Kaggle raw
        )
    except KeyError:
        # as a last resort create text from numeric codes
        if "Category" in df.columns:
            df["Category_text"] = df["Category"].astype(str)
            cat_col = "Category_text"
        else:
            raise

    # Base columns present in both versions (some may be named a bit differently)
    # Create canonical names
    # Rating
    rat_col = "Rating" if "Rating" in df.columns else col_first(df, "rating", "rating_num", "Rating_num")
    df["Rating_num"] = pd.to_numeric(df[rat_col], errors="coerce")

    # Reviews
    rev_col = "Reviews" if "Reviews" in df.columns else col_first(df, "reviews", "Reviews_num")
    df["Reviews_num"] = pd.to_numeric(df[rev_col], errors="coerce")

    # Installs (may be like '1,000,000+')
    inst_col = "Installs" if "Installs" in df.columns else col_first(df, "installs", "Installs_num")
    df["Installs_num"] = df[inst_col].apply(parse_installs)

    # Type / Price
    if "Type" in df.columns:
        df["Type"] = df["Type"].fillna("Free")
    else:
        df["Type"] = np.where((df.get("Price", 0).fillna(0) > 0) | (df.get("Price_num", 0).fillna(0) > 0), "Paid", "Free")
    price_col = "Price" if "Price" in df.columns else df.get("Price_num", pd.Series(dtype=float))
    df["Price_num"] = df[price_col].apply(parse_price) if isinstance(price_col, str) else df["Price_num"].fillna(0)

    # Size
    size_col = "Size" if "Size" in df.columns else col_first(df, "Size_MB", "size")
    df["Size_MB"] = df[size_col].apply(parse_size_mb)

    # Last Updated (as text in Kaggle)
    last_upd = None
    if "Last Updated" in df.columns:
        last_upd = "Last Updated"
    elif "Last Updated (Year/Month)" in df.columns or "Last Updated (Year)" in df.columns:
        last_upd = None  # already split below
    else:
        # try lowercase
        last_upd = col_first(df, "last_updated", "LastUpdated")

    if last_upd:
        df["LastUpdated"] = df[last_upd]
        df["LastUpdated_Year"] = df["LastUpdated"].apply(year_from_date)
        df["LastUpdated_YM"] = df["LastUpdated"].apply(months_from_date)
    else:
        # cleaned versions may already have Year/Month
        y = col_first(df, "Last Updated (Year)", "LastUpdated (Year)", "LastUpdated_Year")
        m = col_first(df, "Last Updated (Year/Month)", "LastUpdated (Year/Month)", "LastUpdated_YM")
        df["LastUpdated_Year"] = pd.to_numeric(df[y], errors="coerce")
        df["LastUpdated_YM"] = pd.to_numeric(df[m], errors="coerce")
        df["LastUpdated"] = pd.NaT

    # Category text (final)
    df["Category_text"] = df[cat_col].astype(str).str.replace("_", " ").str.title()

    # Convenience flags
    df["Is_Paid"] = (df["Type"].str.lower() == "paid").astype(int)

    # Drop rows with missing rating out of range
    df = df[(df["Rating_num"] >= 0) & (df["Rating_num"] <= 5)]

    # Add logs (avoid -inf)
    df["log10_installs"] = np.log10(df["Installs_num"].replace(0, np.nan))
    df["log10_reviews"] = np.log10(df["Reviews_num"].replace(0, np.nan))

    # Recency in years (relative scale using Year/Month)
    max_ym = pd.to_numeric(df["LastUpdated_YM"], errors="coerce").dropna().max()
    df["Recency_years"] = np.where(
        np.isfinite(df["LastUpdated_YM"]),
        (max_ym - df["LastUpdated_YM"]) / 12.0,
        np.nan,
    )

    return df

# ----------------------------
# Load data (URL / file / fallback)
# ----------------------------
with st.sidebar:
    st.markdown("### Data")
    url = st.text_input("Optional: CSV URL (public)", value="")
    upl = st.file_uploader("Or upload a CSV", type=["csv"])

if url.strip():
    df_raw = load_from_url(url.strip())
elif upl is not None:
    df_raw = pd.read_csv(upl, low_memory=False)
else:
    df_raw = load_default()

df = ensure_columns(df_raw.copy())

st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.markdown("### Filters")
    cats = sorted(df["Category_text"].dropna().unique().tolist())
    sel_cats = st.multiselect("Categories", cats, default=[])

    type_choice = st.radio("Type", ["All", "Free", "Paid"], horizontal=True)
    min_reviews = st.slider("Minimum reviews", 0, int(np.nanmax(df["Reviews_num"])) if len(df) else 1000, 0, step=100)

    year_min = int(pd.to_numeric(df["LastUpdated_Year"], errors="coerce").dropna().min())
    year_max = int(pd.to_numeric(df["LastUpdated_Year"], errors="coerce").dropna().max())
    yr_rng = st.slider("Updated year range", year_min, year_max, (year_min, year_max))

# Apply filters
mask = np.ones(len(df), dtype=bool)
if sel_cats:
    mask &= df["Category_text"].isin(sel_cats)
if type_choice == "Free":
    mask &= (df["Is_Paid"] == 0)
elif type_choice == "Paid":
    mask &= (df["Is_Paid"] == 1)
mask &= (df["Reviews_num"].fillna(0) >= min_reviews)
mask &= df["LastUpdated_Year"].between(yr_rng[0], yr_rng[1], inclusive="both")
dff = df.loc[mask].copy()
st.caption(f"Active filter ≈ {len(dff):,} apps")

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["Overview", "Explore", "Simpson", "Regression", "Free vs Paid (FE)", "About"])

# ---------- Overview ----------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Apps", f"{len(dff):,}")
    with c2:
        st.metric("Median rating", f"{np.nanmedian(dff['Rating_num']):.2f}")
    with c3:
        paid_share = np.mean(dff["Is_Paid"]) if len(dff) else np.nan
        st.metric("Paid share", f"{100*paid_share:.1f}%")
    with c4:
        med_inst = np.nanmedian(dff["Installs_num"]) if len(dff) else np.nan
        st.metric("Median installs", f"{med_inst:,.0f}" if np.isfinite(med_inst) else "—")

    st.markdown("#### Sample (first 10):")
    show_cols = ["App", "Category_text", "Rating_num", "Reviews_num", "Size_MB",
                 "Installs_num", "Type", "Price_num", "Content Rating", "Genres",
                 "LastUpdated", "Current Ver", "Android Ver", "LastUpdated_Year",
                 "LastUpdated_YM"]
    show_cols = [c for c in show_cols if c in dff.columns]
    st.dataframe(dff[show_cols].head(10), use_container_width=True, height=280)

# ---------- Explore ----------
with tabs[1]:
    st.markdown("### Quick EDA")
    gr1, gr2 = st.columns(2)
    with gr1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(dff["Rating_num"].dropna(), bins=25, ax=ax)
        ax.set_xlabel("Rating (1–5)"); ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    with gr2:
        # Top categories by volume
        top_cat = dff["Category_text"].value_counts().head(10).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        top_cat.plot(kind="barh", ax=ax)
        ax.set_xlabel("Apps"); ax.set_ylabel("Category")
        st.pyplot(fig, use_container_width=True)

    # Optional scatter: show only if enough data
    enough = dff["log10_installs"].notna().sum() > 300
    if enough:
        st.markdown("#### Reviews vs Installs (log–log)")
        fig, ax = plt.subplots(figsize=(7, 5))
        x = dff["log10_reviews"].values
        y = dff["log10_installs"].values
        # jitter to avoid “lines”
        xj = x + np.random.normal(0, 0.01, size=len(x))
        yj = y + np.random.normal(0, 0.01, size=len(y))
        ax.scatter(xj, yj, s=8, alpha=0.25)
        ax.set_xlabel("log10(Reviews)"); ax.set_ylabel("log10(Installs)")
        st.pyplot(fig, use_container_width=True)

# ---------- Simpson ----------
with tabs[2]:
    st.markdown("### Simpson-like reversal: Overall vs chosen category")
    # Choose category to highlight (default: the one with most apps)
    cat_counts = dff["Category_text"].value_counts()
    if len(cat_counts) == 0:
        st.info("No data after filters.")
    else:
        cat_pick = st.selectbox("Pick a category", cat_counts.index.tolist(), index=0)

        # Overall means
        overall_uw = np.nanmean(dff["Rating_num"])
        overall_w = np.average(dff["Rating_num"], weights=dff["Reviews_num"].clip(lower=1))

        # Category means
        dcat = dff[dff["Category_text"] == cat_pick]
        cat_uw = np.nanmean(dcat["Rating_num"])
        cat_w = np.average(dcat["Rating_num"], weights=dcat["Reviews_num"].clip(lower=1)) if len(dcat) else np.nan

        tbl = pd.DataFrame({
            "Group": ["Overall (Unweighted)", "Overall (Weighted)", f"{cat_pick} (Unweighted)", f"{cat_pick} (Weighted)"],
            "Mean": [overall_uw, overall_w, cat_uw, cat_w]
        })

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(tbl["Group"], tbl["Mean"], color=["#4e79a7","#f28e2b","#59a14f","#e15759"])
        ax.set_ylabel("Average rating"); ax.set_ylim(3.6, 4.6)
        safe_rotate_x(ax, 25, "right")
        for b, v in zip(bars, tbl["Mean"]):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig, use_container_width=True)

# ---------- Regression: log-log OLS ----------
with tabs[3]:
    st.markdown("### Installs vs Reviews — OLS (log10–log10) with 95% CI & prediction band")
    reg = dff[["log10_installs", "log10_reviews"]].dropna()
    if len(reg) < 300:
        st.info("Not enough data after filters for a stable regression.")
    else:
        X = sm.add_constant(reg["log10_reviews"])
        ols = sm.OLS(reg["log10_installs"], X).fit(cov_type="HC1")

        # grid for line/intervals
        xs = np.linspace(reg["log10_reviews"].min(), reg["log10_reviews"].max(), 100)
        Xs = sm.add_constant(xs)
        pred = ols.get_prediction(Xs)
        summ = pred.summary_frame(alpha=0.05)

        fig, ax = plt.subplots(figsize=(7, 5))
        # points (light)
        ax.scatter(reg["log10_reviews"], reg["log10_installs"], s=8, alpha=0.12, label="Apps")
        # mean CI
        ax.plot(xs, summ["mean"], lw=2, label="OLS fit")
        ax.fill_between(xs, summ["mean_ci_lower"], summ["mean_ci_upper"], alpha=0.2, label="95% CI (mean)")
        # prediction interval
        ax.fill_between(xs, summ["obs_ci_lower"], summ["obs_ci_upper"], alpha=0.12, label="95% prediction interval")

        ax.set_xlabel("log10(Reviews)"); ax.set_ylabel("log10(Installs)")
        ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=True)

        # Coefficients table
        ci = ols.conf_int(alpha=0.05)
        coef_tbl = pd.DataFrame({
            "coef": ols.params,
            "std_err": ols.bse,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": ols.pvalues
        })
        st.dataframe(coef_tbl.style.format(precision=4), use_container_width=True)

# ---------- Free vs Paid with Category FE ----------
with tabs[4]:
    st.markdown("### Free vs Paid — OLS with Category dummies (robust SE)")
    fe = dff[["log10_installs", "Rating_num", "Size_MB", "Recency_years", "Is_Paid", "Category_text"]].dropna()
    if len(fe) < 300 or fe["Category_text"].nunique() < 3:
        st.info("Not enough data after filters for a stable FE regression.")
    else:
        # Build formula with category FE
        fe["Size_MB"] = fe["Size_MB"].clip(lower=0)
        formula = "log10_installs ~ Is_Paid + Rating_num + Size_MB + Recency_years + C(Category_text)"
        m = smf.ols(formula, data=fe).fit(cov_type="HC1")

        st.write("**Model quality:**  n =", len(fe), " |  R-squared (robust se):", f"{m.rsquared:.3f}")
        ci = m.conf_int(alpha=0.05)
        coef_tbl = pd.DataFrame({
            "coef": m.params, "std_err": m.bse,
            "ci_low": ci[0], "ci_high": ci[1], "p_value": m.pvalues
        })
        st.dataframe(coef_tbl.loc[["Intercept", "Is_Paid", "Rating_num", "Size_MB", "Recency_years"]].style.format(precision=4),
                    use_container_width=True)

        # Visualize paid effect with CI
        beta = m.params["Is_Paid"]; lo, hi = ci.loc["Is_Paid", 0], ci.loc["Is_Paid", 1]
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.hlines(0, lo, hi, lw=6, color="#4e79a7")
        ax.plot([beta, beta], [ -0.1, 0.1 ], lw=2, color="#e15759")
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks([]); ax.set_xlabel("Paid effect on log10(installs) (95% CI)")
        st.pyplot(fig, use_container_width=True)

# ---------- About ----------
with tabs[5]:
    st.markdown("## About this project")
    st.markdown("""
**What we studied:**  
- האם דירוגים גבוהים באמת “מנצחים” כששוקלים נפח סקירות (reviews)?  
- מה האלסטיות בין Reviews לבין Installs?  
- האם “Paid” פוגע בביצועים גם אחרי שליטה בקטגוריה ובמאפיינים?  

**Highlights:**  
- דוגמת **Simpson**: סדר הקטגוריות לפי ממוצע לא-משוקלל יכול להתהפך כששוקלים לפי נפח ביקורות—קטגוריה קטנה עם ממוצע גבוה נחלשת מול גדולה עם אמינות גבוהה.  
- ברגרסיית **log–log**, האלסטיות של Installs ביחס ל-Reviews מתקרבת ל-1 (קו לינארי על סולם לוג).  
- במודל **Free vs Paid (עם Fixed Effects לקטגוריה)** מתקבל אפקט שלילי מובהק ל-Paid (לאחר שליטה ב-Rating/Size/Recency).  

**How to use this app:**  
- העלה קובץ CSV (או הדבק URL ציבורי). אם לא הועלה קובץ—נטען את הדאטה שב־repo.  
- בצד ימין קבע סינונים (קטגוריות, Free/Paid, מינימום ביקורות, טווח שנים).  
- עבור בין הלשוניות: Overview, Explore, Simpson, Regression, Free vs Paid (FE).  
- כל המודלים משתמשים ב-robust (HC1) SE.  
""")
