# ────────────────────────────────────────────────
# smartwatch_dashboard.py
# Streamlit dashboard for Smartwatch & Livestream-shopping survey
# Handles Google-Forms ranking grid (two layouts) + Age×Gender rank stacks
# ────────────────────────────────────────────────
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1 ── Page config
st.set_page_config(page_title="Smartwatch Survey Dashboard", layout="wide")
st.title("🔎 Smartwatch & Livestream-Shopping Survey Dashboard")

# 2 ── Upload file
uploaded_file = st.file_uploader(
    "📂 Upload the Google-Forms responses Excel file", type=["xlsx", "xls"]
)
if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# 3 ── Load & clean
@st.cache_data(show_spinner=False)
def load_clean(xl_file) -> pd.DataFrame:
    df = pd.read_excel(xl_file)

    # 3-a tidy headers
    df.columns = (
        df.columns.str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("’", "'", regex=False)
    )

    # 3-b long Q → short key
    exact = {
        "How much do you think a new premium smartwatch should cost?": "Price",
        "What is the primary reason you upgrade/buy a smartwatch?": "Reason",
        "Where do you plan to purchase your next smartwatch?": "BuyChannel",
        "Which marketing channels are most effective in reaching you?": "MktChannels",
        "Which social media do you use the most?": "TopSocial",
        "How do you prefer to research a product or service before making a purchase?": "ResearchMethod",
        "Are livestreams important in learning about the product that you’re interested in buying?": "LiveDemoImportant",
        "Are limited-time vouchers given during livestreams important in helping you purchase the product that you’re interested in buying?": "LiveVoucherImportant",
        "How important is the after-sales service of a smartwatch brand to you?": "AfterSalesImportance",
        "What would most encourage you to stay loyal to a smartwatch brand?": "LoyaltyDriver",
    }
    df = df.rename(columns={k: v for k, v in exact.items() if k in df.columns})

    # 3-c fuzzy Age/Gender
    for col in df.columns:
        l = col.lower()
        if "age" in l and "age" not in df.columns:
            df = df.rename(columns={col: "Age"})
        if "gender" in l and "gender" not in df.columns:
            df = df.rename(columns={col: "Gender"})

    # 3-d convert ranking grid (two layouts)
    feature_names = [
        "Slim and Stylish Design",
        "Large Display",
        "Long Battery Life",
        "Strong Durability",
        "Health and activity tracking",
        "Compatibility with your iPhone/Android",
        "NFC Support",
    ]
    pat = re.compile(r"\[(\d)\]$")
    cols_pat   = [c for c in df.columns if pat.search(c)]
    cols_digit = [c for c in df.columns if c.strip().isdigit() and 1 <= int(c) <= 7]
    rank_cols  = cols_pat if cols_pat else cols_digit

    if rank_cols:
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = pd.NA
        for col in rank_cols:
            rank_num = int(pat.search(col).group(1)) if cols_pat else int(col.strip())
            for idx, feat in df[col].dropna().items():
                if feat in feature_names:
                    df.at[idx, feat] = rank_num
        df = df.drop(columns=rank_cols)

    return df


df = load_clean(uploaded_file)
if not {"Age", "Gender"}.issubset(df.columns):
    st.error("⚠️ Age and/or Gender column missing – check spreadsheet.")
    st.stop()

# 4 ── Filters
st.sidebar.header("🔍 Filters")
ages_all = sorted(df["Age"].dropna().unique())
genders_all = sorted(df["Gender"].dropna().unique())
age_sel = st.sidebar.multiselect("Age groups", ages_all, default=ages_all)
gen_sel = st.sidebar.multiselect("Gender", genders_all, default=genders_all)
df = df[df["Age"].isin(age_sel) & df["Gender"].isin(gen_sel)]

# 5 ── Helpers
def pct_tbl(series, total):
    return series.value_counts().div(total).mul(100)

def bar_pct(series, ax, horiz=False):
    if horiz:
        series.plot.barh(ax=ax)
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.1f}%", (p.get_width(), p.get_y()+p.get_height()/2),
                        ha="left", va="center", fontsize=8)
        ax.set_xlabel("Percentage (%)"); ax.set_ylabel("")
    else:
        series.plot.bar(ax=ax)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x()+p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Percentage (%)"); ax.set_xlabel("")

icons = {
    "Slim and Stylish Design": "✨",
    "Large Display":           "📱",
    "Long Battery Life":       "🔋",
    "Strong Durability":       "🛡️",
    "Health and activity tracking": "❤️",
    "Compatibility with your iPhone/Android": "📲",
    "NFC Support":             "📳",
}
feature_cols = [c for c in icons if c in df.columns]
group = df.groupby(["Age", "Gender"])

# 6 ── Tabs
tab_purch, tab_mkt, tab_post, tab_feat = st.tabs(
    ["🛒 Purchasing", "📣 Marketing", "🔄 Post-purchase", "🔧 Features"]
)

# ──────────────────────────────────────────────────────────
# (A) Purchasing
# ──────────────────────────────────────────────────────────
with tab_purch:
    st.header("Purchasing Preferences")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution (%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_pct(df["Age"].value_counts(normalize=True).mul(100).sort_index(), ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Gender Split (%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Gender"].value_counts(normalize=True).mul(100).plot.pie(
            autopct="%1.1f%%", ax=ax, textprops={"fontsize": 8}
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    if "Price" in df.columns:
        st.markdown("---")
        st.subheader("💰 Preferred Price (Mode by Age × Gender)")
        st.dataframe(
            df.groupby(["Age", "Gender"])["Price"]
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
              .unstack().sort_index()
        )

    if feature_cols or "Reason" in df.columns:
        st.markdown("---"); st.markdown("### Breakdown by Age × Gender (%)")

    for (age, gen), sub in group:
        st.markdown(f"**Age {age} • {gen}**")
        cols = st.columns(2 if "Reason" in df.columns else 1)

        if "Reason" in df.columns:
            cols[0].markdown("*Top-3 Reasons*")
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_pct(pct_tbl(sub["Reason"], len(sub)).head(3), ax, horiz=True)
            cols[0].pyplot(fig)

        if feature_cols:
            cols[-1].markdown("*Top-5 Features*")
            ranked = sub.melt(value_vars=feature_cols, var_name="Feature", value_name="Rank")
            ranked = ranked[pd.to_numeric(ranked["Rank"], errors="coerce") <= 5]
            feat_pct = ranked["Feature"].value_counts().div(len(sub)).mul(100).head(5)
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_pct(feat_pct.sort_values(), ax, horiz=True)
            ax.set_yticklabels([f"{icons.get(i,'')} {i}" for i in feat_pct.index])
            cols[-1].pyplot(fig)

# ──────────────────────────────────────────────────────────
# (B) Marketing
# ──────────────────────────────────────────────────────────
with tab_mkt:
    st.header("Marketing Preferences")
    for (age, gen), sub in group:
        st.markdown(f"**Age {age} • {gen}**")
        c1, c2, c3 = st.columns(3)

        if "MktChannels" in df.columns:
            c1.markdown("*Effective Channels*")
            ch = (
                sub["MktChannels"].dropna()
                  .str.split(",", expand=False).explode().str.strip()
            )
            if not ch.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                bar_pct(pct_tbl(ch, len(sub)), ax, horiz=True)
                c1.pyplot(fig)

        if "TopSocial" in df.columns:
            c2.markdown("*Top Social Platform*")
            if not sub["TopSocial"].dropna().empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                bar_pct(pct_tbl(sub["TopSocial"], len(sub)), ax)
                c2.pyplot(fig)

        if "ResearchMethod" in df.columns:
            c3.markdown("*Preferred Research Method*")
            if not sub["ResearchMethod"].dropna().empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                bar_pct(pct_tbl(sub["ResearchMethod"], len(sub)), ax, horiz=True)
                c3.pyplot(fig)

# ──────────────────────────────────────────────────────────
# (C) Post-purchase
# ──────────────────────────────────────────────────────────
with tab_post:
    st.header("Post-purchase Preferences")
    for (age, gen), sub in group:
        st.markdown(f"**Age {age} • {gen}**")
        c1, c2, c3, c4 = st.columns(4)

        if "AfterSalesImportance" in df.columns:
            c1.markdown("*After-sales Importance*")
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_pct(pct_tbl(sub["AfterSalesImportance"], len(sub)), ax)
            c1.pyplot(fig)

        if "LoyaltyDriver" in df.columns:
            c2.markdown("*Key Loyalty Drivers*")
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_pct(pct_tbl(sub["LoyaltyDriver"], len(sub)), ax, horiz=True)
            c2.pyplot(fig)

        if "LiveDemoImportant" in df.columns:
            c3.markdown("*Livestream Demo Importance*")
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_pct(pct_tbl(sub["LiveDemoImportant"], len(sub)), ax)
            c3.pyplot(fig)

        if "LiveVoucherImportant" in df.columns:
            c4.markdown("*Voucher Incentive Importance*")
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_pct(pct_tbl(sub["LiveVoucherImportant"], len(sub)), ax)
            c4.pyplot(fig)

# ──────────────────────────────────────────────────────────
# (D) Features
# ──────────────────────────────────────────────────────────
with tab_feat:
    st.header("Smartwatch-Feature Preferences")

    if not feature_cols:
        st.info("No feature-ranking columns detected.")
        st.stop()

    # 1. Overall Top-5
    st.subheader("🏆 Overall Top-5 Features (%)")
    overall = (
        df[feature_cols].apply(pd.to_numeric, errors="coerce")
          .le(5).sum().div(df.shape[0]).mul(100)
          .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_pct(overall.head(5).sort_values(), ax, horiz=True)
    ax.set_yticklabels([f"{icons.get(i,'')} {i}" for i in overall.head(5).index])
    st.pyplot(fig)

    # 2. Global rank distribution (numeric x-positions)
    st.markdown("---")
    st.subheader("📊 Rank Distribution (All Respondents)")
    rank_long = df.melt(value_vars=feature_cols, var_name="Feature", value_name="Rank")
    rank_long["Rank"] = pd.to_numeric(rank_long["Rank"], errors="coerce")
    rank_dist = (
        rank_long.dropna()
                 .groupby(["Feature", "Rank"]).size()
                 .groupby(level=0).apply(lambda s: s / s.sum() * 100)
                 .unstack(fill_value=0).sort_index()
    ).sort_values(by=1, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rank_dist))
    bottom = np.zeros(len(rank_dist))
    for r in sorted(rank_dist.columns):
        ax.bar(x, rank_dist[r].values, bottom=bottom, label=f"Rank {r}")
        bottom += rank_dist[r].values
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{icons.get(i,'')} {i}" for i in rank_dist.index],
        rotation=45, ha="right"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    st.pyplot(fig)

    # 3. Slice-level stacks inside expanders (numeric x)
    st.markdown("---")
    st.subheader("🔍 Rank Distribution by Age × Gender")
    for (age, gen), sub in group:
        with st.expander(f"Age {age} • {gen}", expanded=False):
            if sub[feature_cols].notna().sum().sum() == 0:
                st.write("No ranking data for this slice.")
                continue
            sub_long = sub.melt(value_vars=feature_cols,
                                var_name="Feature", value_name="Rank")
            sub_long["Rank"] = pd.to_numeric(sub_long["Rank"], errors="coerce")
            dist = (
                sub_long.dropna()
                        .groupby(["Feature", "Rank"]).size()
                        .groupby(level=0).apply(lambda s: s / s.sum() * 100)
                        .unstack(fill_value=0).sort_index()
            ).sort_values(by=1, ascending=False)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            x = np.arange(len(dist))
            bottom = np.zeros(len(dist))
            for r in sorted(dist.columns):
                ax.bar(x, dist[r].values, bottom=bottom, label=f"R{r}")
                bottom += dist[r].values
            ax.set_ylabel("%")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{icons.get(i,'')} {i}" for i in dist.index],
                rotation=45, ha="right", fontsize=8
            )
            ax.legend(title="Rank", fontsize=7)
            st.pyplot(fig)

    # 4. Top feature table
    st.markdown("---")
    st.subheader("🏅 Top Feature per Age × Gender")
    grid = []
    for (age, gen), sub in group:
        ranked = sub.melt(value_vars=feature_cols, var_name="Feature", value_name="Rank")
        ranked = ranked[pd.to_numeric(ranked["Rank"], errors="coerce") <= 5]
        top_feat = ranked["Feature"].value_counts().idxmax() if not ranked.empty else "—"
        grid.append({
            "Age": age,
            "Gender": gen,
            "Top Feature": f"{icons.get(top_feat,'')} {top_feat}",
        })
    st.dataframe(pd.DataFrame(grid).sort_values(["Age", "Gender"]))
