# app.py
# Google Play Explorer — Streamlit
# Tabs: Overview, Explore (EDA), Simpson, Regression (log-log), Free vs Paid (FE), About

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------
# Page + style
# -----------------------------
st.set_page_config(page_title="Play Store Explorer", layout="wide")
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")

# -----------------------------
# Data sources
# -----------------------------
DATA_PATH = Path(__file__).parent / "data" / "apps_clean.csv.gz"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), low_memory=False)

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _read_csv_path(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False)

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _read_csv_url(u: str) -> pd.DataFrame:
    return pd.read_csv(u, low_memory=False)

# -----------------------------
# Cleaning helpers (robust to raw/clean CSV)
# -----------------------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def parse_installs(x):
    # works for raw Kaggle "100,000+" etc. or clean numeric
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("+", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Category_text
    if "Category_text" not in d.columns:
        if "Category" in d.columns:
            d["Category_text"] = d["Category"].astype(str)
        elif "category" in d.columns:
            d["Category_text"] = d["category"].astype(str)
        else:
            d["Category_text"] = "Unknown"

    # Type_label
    if "Type_label" not in d.columns:
        base = d.get("Type", d.get("type", pd.Series(index=d.index, dtype=str))).astype(str)
        d["Type_label"] = base

    # Rating_num
    if "Rating_num" not in d.columns and "Rating" in d.columns:
        d["Rating_num"] = _to_num(d["Rating"])
    elif "Rating_num" in d.columns:
        d["Rating_num"] = _to_num(d["Rating_num"])

    # Reviews
    if "Reviews" in d.columns:
        d["Reviews"] = _to_num(d["Reviews"])
    elif "reviews" in d.columns:
        d["Reviews"] = _to_num(d["reviews"])

    # Installs_num
    if "Installs_num" not in d.columns:
        col = "Installs" if "Installs" in d.columns else ("installs" if "installs" in d.columns else None)
        if col is not None:
            d["Installs_num"] = d[col].map(parse_installs)
    else:
        d["Installs_num"] = _to_num(d["Installs_num"])

    # Size_MB (best-effort)
    if "Size_MB" in d.columns:
        d["Size_MB"] = _to_num(d["Size_MB"])
    elif "Size" in d.columns:
        # raw may contain 'Varies with device' or '12M'
        def _size_to_mb(s):
            if pd.isna(s): return np.nan
            s = str(s).strip()
            if "Varies" in s: return np.nan
            if s.endswith("k") or s.endswith("K"):
                return float(s[:-1]) / 1024.0
            if s.endswith("M") or s.endswith("m"):
                return float(s[:-1])
            try: return float(s)
            except: return np.nan
        d["Size_MB"] = d["Size"].map(_size_to_mb)

    # LastUpdated (datetime)
    if "LastUpdated" in d.columns:
        d["LastUpdated"] = pd.to_datetime(d["LastUpdated"], errors="coerce")
    elif "Last Updated" in d.columns:
        d["LastUpdated"] = pd.to_datetime(d["Last Updated"], errors="coerce")

    # Year for filter
    if "LastUpdated" in d.columns:
        d["LastUpdated_year"] = d["LastUpdated"].dt.year

    return d

# -----------------------------
# Sidebar: load + filters
# -----------------------------
st.sidebar.header("Data")
csv_url = st.sidebar.text_input("Optional: CSV URL (public)")
upload = st.sidebar.file_uploader("Or upload a CSV", type=["csv"])

with st.spinner("Loading data..."):
    try:
        if upload is not None:
            df_raw = _read_csv_bytes(upload.read())
        elif csv_url.strip():
            df_raw = _read_csv_url(csv_url.strip())
        else:
            df_raw = _read_csv_path(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

df = normalize_df(df_raw)
st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Filters
st.sidebar.header("Filters")

cats = sorted(df["Category_text"].dropna().unique().tolist())
sel_cats = st.sidebar.multiselect("Categories", cats, default=[])

type_opt = st.sidebar.radio("Type", ["All", "Free", "Paid"], horizontal=True)

max_reviews = int(_to_num(df["Reviews"]).fillna(0).max())
min_reviews = st.sidebar.slider("Minimum reviews", 0, max(1000, max_reviews), 0, step=100)

if "LastUpdated_year" in df.columns:
    yr_min, yr_max = int(df["LastUpdated_year"].min()), int(df["LastUpdated_year"].max())
    yr_rng = st.sidebar.slider("Updated year range", yr_min, yr_max, (yr_min, yr_max))
else:
    yr_rng = None

df_f = df.copy()
if sel_cats:
    df_f = df_f[df_f["Category_text"].isin(sel_cats)]
if type_opt != "All":
    df_f = df_f[df_f["Type_label"].astype(str).str.contains(type_opt, case=False, na=False)]
df_f = df_f[_to_num(df_f["Reviews"]).fillna(0) >= min_reviews]
if yr_rng and "LastUpdated_year" in df_f.columns:
    df_f = df_f[df_f["LastUpdated_year"].between(yr_rng[0], yr_rng[1])]

st.caption(f"**Active filter →** {len(df_f):,} apps")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_explore, tab_simpson, tab_reg, tab_fe, tab_about = st.tabs(
    ["🔎 Overview", "🧭 Explore", "♟️ Simpson", "📈 Regression", "💲 Free vs Paid (FE)", "ℹ️ About"]
)

# -----------------------------
# Tab: Overview
# -----------------------------
with tab_overview:
    st.write("### Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Apps", f"{len(df_f):,}")
    med_rating = _to_num(df_f.get("Rating_num")).median()
    c2.metric("Median rating", f"{med_rating:.2f}" if not np.isnan(med_rating) else "—")
    paid_share = df_f["Type_label"].astype(str).str.contains("Paid", case=False, na=False).mean() if len(df_f) else 0.0
    c3.metric("Paid share", f"{100*paid_share:.1f}%")
    med_installs = _to_num(df_f.get("Installs_num")).median()
    c4.metric("Median installs", f"{med_installs:,.0f}" if not np.isnan(med_installs) else "—")

    st.divider()
    st.write("Sample (first 10):")
    st.dataframe(df_f.head(10))

# -----------------------------
# Tab: Explore (EDA)
# -----------------------------
with tab_explore:
    st.write("### Quick EDA")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        if "Rating_num" in df_f.columns:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            sns.histplot(_to_num(df_f["Rating_num"]).dropna(), bins=30, ax=ax)
            ax.set_title("Rating distribution")
            ax.set_xlabel("Rating (1–5)")
            st.pyplot(fig)

        # Top categories
        top = df_f["Category_text"].value_counts().head(12).reset_index()
        top.columns = ["Category", "n"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.barplot(data=top, x="Category", y="n", ax=ax)
        ax.set_title("Top categories (by count)")
        ax.set_xlabel("Category")
        ax.set_ylabel("Apps")
        ax.tick_params(axis="x", rotation=45, ha="right")
        st.pyplot(fig)

    with col2:
        # Scatter log-log Installs vs Reviews (sample for speed)
        d = df_f[["Installs_num", "Reviews"]].copy()
        d["Installs_num"] = _to_num(d["Installs_num"])
        d["Reviews"] = _to_num(d["Reviews"])
        d = d.dropna()
        d = d[(d["Installs_num"] > 0) & (d["Reviews"] > 0)]
        if len(d) > 0:
            n_show = min(8000, len(d))
            d = d.sample(n_show, random_state=42) if len(d) > n_show else d
            d["logI"] = np.log10(d["Installs_num"])
            d["logR"] = np.log10(d["Reviews"])
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.scatter(d["logR"], d["logI"], s=10, alpha=0.25)
            ax.set_xlabel("log10(Reviews)")
            ax.set_ylabel("log10(Installs)")
            ax.set_title("Installs vs Reviews (log–log)")
            st.pyplot(fig)

# -----------------------------
# Tab: Simpson
# -----------------------------
with tab_simpson:
    st.write("### Simpson-like reversal (weighted vs unweighted means)")

    cats_all = sorted(df["Category_text"].dropna().unique().tolist())
    defaults = [c for c in ["Events", "Medical"] if c in cats_all][:2]
    pair = st.multiselect("Pick two categories", cats_all, default=defaults, max_selections=2)

    if len(pair) != 2:
        st.info("Pick exactly two categories.")
    else:
        a, b = pair
        sub = df[df["Category_text"].isin([a, b])].dropna(subset=["Rating_num", "Reviews"]).copy()
        if sub.empty:
            st.warning("No data for the selected categories.")
        else:
            unweighted = sub.groupby("Category_text")["Rating_num"].mean()
            wmean = sub.groupby("Category_text").apply(
                lambda g: (_to_num(g["Rating_num"]) * _to_num(g["Reviews"])).sum() / _to_num(g["Reviews"]).sum()
            )

            plotdf = pd.DataFrame({
                "Category": [a, a, b, b],
                "Mean type": ["Unweighted", "Weighted", "Unweighted", "Weighted"],
                "Rating": [unweighted.get(a, np.nan), wmean.get(a, np.nan),
                           unweighted.get(b, np.nan), wmean.get(b, np.nan)],
            })

            colors = {"Unweighted": "#4C78A8", "Weighted": "#F58518"}
            fig, ax = plt.subplots(figsize=(6.6, 3.6))
            for mt in ["Unweighted", "Weighted"]:
                dd = plotdf[plotdf["Mean type"] == mt]
                ax.bar(dd["Category"], dd["Rating"], width=0.4, label=mt, color=colors[mt])
                for x, v in zip(dd["Category"], dd["Rating"]):
                    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(3.8, 4.7)
            ax.set_ylabel("Average rating")
            ax.set_title(f"Overall vs Category — {a} & {b}")
            ax.legend(title="Mean type")
            st.pyplot(fig)

            st.markdown(
                """
**What I show:** Two bars per category – *unweighted* mean and *weighted-by-reviews* mean.  
**Why it matters:** Small groups with high raw averages can lose to large groups once we weight by reliability (review counts).  
**Takeaway for PMs:** Always check weighted metrics and consider fixed effects (category) before prioritizing “top-rated” items.
                """
            )

# -----------------------------
# Tab: Regression (log–log)
# -----------------------------
with tab_reg:
    st.write("### OLS regression: log10(Installs) ~ log10(Reviews)")
    dat = df_f[["Installs_num", "Reviews"]].copy()
    dat["Installs_num"] = _to_num(dat["Installs_num"])
    dat["Reviews"] = _to_num(dat["Reviews"])
    dat = dat.dropna()
    dat = dat[(dat["Installs_num"] > 0) & (dat["Reviews"] > 0)]
    if dat.empty:
        st.warning("No data after filters.")
    else:
        dat["log10_installs"] = np.log10(dat["Installs_num"])
        dat["log10_reviews"] = np.log10(dat["Reviews"])

        X = sm.add_constant(dat["log10_reviews"])
        m = sm.OLS(dat["log10_installs"], X).fit(cov_type="HC1")

        # Coef table
        ci = pd.DataFrame(m.conf_int(alpha=0.05), columns=["ci_low", "ci_high"])
        coef_tbl = pd.DataFrame({"coef": m.params, "std_err": m.bse, "p_value": m.pvalues}).join(ci)

        st.code(f"n = {len(dat):,}\nR-squared (robust se): {m.rsquared:.3f}")
        st.dataframe(coef_tbl.style.format({"coef":"{:.6f}","std_err":"{:.6f}","ci_low":"{:.6f}","ci_high":"{:.6f}","p_value":"{:.2e}"}))

        # Prediction grid + bands
        grid = np.linspace(dat["log10_reviews"].min(), dat["log10_reviews"].max(), 120)
        pred = m.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)

        # light jitter for points so they don't band visually
        rng = np.random.default_rng(42)
        xj = dat["log10_reviews"] + rng.normal(0, 0.01, size=len(dat))

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(xj, dat["log10_installs"], s=10, alpha=0.25, label=f"Apps (n={len(dat):,})")
        ax.fill_between(grid, pred["obs_ci_lower"], pred["obs_ci_upper"], alpha=0.12, label="95% prediction interval")
        ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], alpha=0.20, label="95% CI (mean)")
        ax.plot(grid, pred["mean"], lw=2.2, label="OLS fit")
        ax.set_xlabel("log10(Reviews)")
        ax.set_ylabel("log10(Installs)")
        ax.set_title("Installs vs Reviews — OLS (log10–log10) with 95% CI & prediction band")

        beta = m.params.get("log10_reviews", np.nan)
        ax.text(0.98, 0.18,
                f"$R^2$ = {m.rsquared:.3f}\n"
                f"Slope (elasticity) = {beta:.3f}\n"
                "Interpretation: +1% reviews ≈ +1.0% installs.",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=10)
        ax.legend(loc="lower right")
        st.pyplot(fig)

# -----------------------------
# Tab: Free vs Paid (FE with controls)
# -----------------------------
with tab_fe:
    st.write("### Free vs Paid — OLS with controls and Category Fixed Effects")
    need = ["Installs_num", "Rating_num", "Size_MB", "Type_label", "Category_text"]
    if "LastUpdated" in df_f.columns:
        need += ["LastUpdated"]
    if not all(c in df.columns for c in need):
        st.warning("Required columns missing for FE model.")
    else:
        d = df_f[need].copy()
        d["Installs_num"] = _to_num(d["Installs_num"])
        d["Rating_num"]   = _to_num(d["Rating_num"])
        d["Size_MB"]      = _to_num(d["Size_MB"])
        d["Is_Paid_num"]  = d["Type_label"].astype(str).str.contains("Paid", case=False, na=False).astype(float)

        if "LastUpdated" in d.columns:
            d["Recency_years"] = (pd.Timestamp.today() - pd.to_datetime(d["LastUpdated"], errors="coerce")).dt.days / 365.25
        else:
            d["Recency_years"] = np.nan

        d = d.dropna(subset=["Installs_num","Rating_num","Size_MB","Is_Paid_num"])
        d = d[d["Installs_num"] > 0]
        if len(d) < 100:
            st.warning("Not enough rows for FE model after filters.")
        else:
            d["log10_installs"] = np.log10(d["Installs_num"])
            # Base covariates
            X_base = d[["Is_Paid_num","Rating_num","Size_MB","Recency_years"]].fillna(0.0)
            # Category fixed effects via dummies
            X_cat  = pd.get_dummies(d["Category_text"], prefix="cat", drop_first=True)
            X      = pd.concat([X_base, X_cat], axis=1)
            X      = sm.add_constant(X)
            y      = d["log10_installs"]

            fe = sm.OLS(y, X).fit(cov_type="HC1")

            # Coef table (main effects only for clarity)
            main = ["Is_Paid_num","Rating_num","Size_MB","Recency_years"]
            ci = pd.DataFrame(fe.conf_int(alpha=0.05), columns=["ci_low","ci_high"])
            coef_tbl = pd.DataFrame({"coef": fe.params, "std_err": fe.bse, "p_value": fe.pvalues}).join(ci)
            st.code(f"n = {len(d):,}\nR-squared (robust se): {fe.rsquared:.3f}")
            st.dataframe(coef_tbl.loc[["const"]+main].style.format(
                {"coef":"{:.6f}","std_err":"{:.6f}","ci_low":"{:.6f}","ci_high":"{:.6f}","p_value":"{:.2e}"}))

            # Predicted means for Free vs Paid at the *average covariates* (delta method CI)
            cov = fe.cov_params()
            xbar = X.mean(axis=0)
            # Free
            v_free = xbar.copy()
            v_free["Is_Paid_num"] = 0.0
            mu_free = float(np.dot(v_free, fe.params))
            se_free = float(np.sqrt(np.dot(v_free @ cov, v_free)))
            # Paid
            v_paid = xbar.copy()
            v_paid["Is_Paid_num"] = 1.0
            mu_paid = float(np.dot(v_paid, fe.params))
            se_paid = float(np.sqrt(np.dot(v_paid @ cov, v_paid)))

            z = 1.96
            means = pd.DataFrame({
                "group": ["Free","Paid"],
                "mean":  [mu_free, mu_paid],
                "lo":    [mu_free - z*se_free, mu_paid - z*se_paid],
                "hi":    [mu_free + z*se_free, mu_paid + z*se_paid],
            })

            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            xpos = np.arange(2)
            ax.errorbar(xpos, means["mean"], yerr=[means["mean"]-means["lo"], means["hi"]-means["mean"]],
                        fmt="o", capsize=5, lw=2, label="95% CI (mean)")
            ax.set_xticks(xpos, means["group"])
            ax.set_ylabel("Expected log10(installs)")
            ax.set_title("Expected log10(Installs): Free vs Paid (95% CI), with controls")
            st.pyplot(fig)

            # Quick interpretation line
            coef_paid = fe.params.get("Is_Paid_num", np.nan)
            ci_paid   = coef_tbl.loc["Is_Paid_num", ["ci_low","ci_high"]].values
            st.caption(
                f"Paid coefficient = {coef_paid:.3f} (95% CI {ci_paid[0]:.3f}, {ci_paid[1]:.3f}). "
                "Negative implies a penalty vs Free, after controlling for rating, size, recency, and category FE."
            )

# -----------------------------
# Tab: About
# -----------------------------
with tab_about:
    st.write("### About this app")
    st.markdown(
        """
**What’s inside**

- **Sidebar filters:** Category, Free/Paid, minimum reviews, and (when available) updated-year range.  
- **Overview:** quick KPIs and sample rows.  
- **Explore:** rating distribution, top categories, and log–log scatter of installs vs reviews.  
- **Simpson:** shows how weighting by review counts can flip the ordering between categories.  
- **Regression:** OLS of log10(installs) on log10(reviews) with a 95% confidence band and prediction interval.  
- **Free vs Paid (FE):** OLS with controls (rating, size, recency) and **category fixed effects**, with 95% CIs.

**Notes**

- The app accepts either your own CSV (upload/URL) or the repository file `data/apps_clean.csv.gz`.  
- Caching keeps first load under ~2 minutes; later runs are instant.  
- Regressions run on the **filtered subset**.
        """
    )

st.markdown("---")
st.caption("Tip: if the dataset is huge, use filters or turn to a smaller sample in the Explore tab for faster plotting.")
