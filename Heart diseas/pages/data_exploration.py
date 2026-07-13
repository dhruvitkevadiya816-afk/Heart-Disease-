"""
pages/data_exploration.py
==========================
Interactive EDA: summary stats, missing values, correlation heatmap,
histograms, boxplots, and feature importance.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, DISPLAY_NAMES, get_feature_descriptions
from utils.preprocessor import get_preprocessing_report


# ── Colour constants ───────────────────────────────────────────────────────────
CARD_BG   = "rgba(0,0,0,0)"
PLOT_FONT = dict(color="white", family="Inter, sans-serif")
AXIS_STYLE = dict(gridcolor="#2a2d3e", zeroline=False, linecolor="#444")
LEGEND_BG  = dict(bgcolor="rgba(30,33,48,0.85)", bordercolor="#444", borderwidth=1)


def _layout(title: str = "", height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="white")),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=PLOT_FONT, height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
        legend=LEGEND_BG,
    )


def render_data_exploration():
    """Render the Data Exploration page."""

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <h2>Dataset Exploration</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset…"):
        try:
            df = load_data()
        except Exception as e:
            st.error(f"❌ Could not load dataset: {e}")
            return

    # ── Rename for display ─────────────────────────────────────────────────────
    df_display = df.rename(columns=DISPLAY_NAMES)
    target_col = DISPLAY_NAMES.get("target", "Heart Disease")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Overview", "🔍 Missing Values", "📈 Distributions",
        "📦 Boxplots", "🌡️ Correlations", "🎯 Feature Insights"
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 — Overview
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        report = get_preprocessing_report(df)

        # Top metric row
        m1, m2, m3, m4 = st.columns(4)
        for col, (icon, val, lbl, color) in zip(
            [m1, m2, m3, m4],
            [
                ("🗃️", report["total_rows"],    "Total Rows",         "#6C63FF"),
                ("🔬", report["total_cols"],    "Total Columns",      "#00D4AA"),
                ("🔴", report["missing_total"], "Missing Values",     "#FF4B6E"),
                ("🔁", report["duplicates"],    "Duplicate Rows",     "#FFB347"),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="metric-value" style="color:{color}; font-size:1.6rem;">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dataset preview
        st.markdown("#### 📄 Dataset Preview")
        st.dataframe(
            df_display.head(20).style.format(precision=2),
            use_container_width=True, height=340
        )

        # Statistical summary
        st.markdown("#### 📊 Statistical Summary")
        summary = df_display.describe().round(3)
        st.dataframe(summary.style.background_gradient(cmap="coolwarm", axis=None),
                     use_container_width=True)

        # Feature descriptions
        st.markdown("#### 📖 Feature Descriptions")
        desc = get_feature_descriptions()
        desc_df = pd.DataFrame(
            [(DISPLAY_NAMES.get(k, k), v) for k, v in desc.items()],
            columns=["Feature", "Description"]
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 — Missing Values
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        missing = df.isnull().sum()
        missing_pct = (df.isnull().mean() * 100).round(2)
        missing_df = pd.DataFrame({
            "Feature":       [DISPLAY_NAMES.get(c, c) for c in df.columns],
            "Missing Count": missing.values,
            "Missing %":     missing_pct.values,
        }).sort_values("Missing %", ascending=False)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 🔍 Missing Value Summary")
            st.dataframe(
                missing_df.style.background_gradient(subset=["Missing %"],
                                                      cmap="Reds"),
                use_container_width=True, hide_index=True
            )
        with c2:
            fig = px.bar(
                missing_df[missing_df["Missing Count"] > 0],
                x="Feature", y="Missing %",
                color="Missing %",
                color_continuous_scale="Reds",
                title="Missing Value Percentage by Feature",
                labels={"Missing %": "Missing (%)"},
            )
            if missing_df["Missing Count"].sum() == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="✅ No missing values found!",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=18, color="#00D4AA"),
                    xref="paper", yref="paper",
                )
            fig.update_layout(**_layout("Missing Value Analysis"))
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3 — Distributions (Histograms)
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("#### 📈 Feature Distributions by Class")
        numeric_cols = [c for c in df.columns if c != "target" and df[c].dtype != object]
        display_numeric = [DISPLAY_NAMES.get(c, c) for c in numeric_cols]

        sel_feature = st.selectbox("Select Feature", display_numeric, key="hist_feat")
        raw_feat = {v: k for k, v in DISPLAY_NAMES.items()}.get(sel_feature, sel_feature)

        fig = go.Figure()
        for target_val, label, color in [(0, "Healthy", "#00D4AA"), (1, "Heart Disease", "#FF4B6E")]:
            subset = df[df["target"] == target_val][raw_feat].dropna()
            fig.add_trace(go.Histogram(
                x=subset, name=label, opacity=0.75,
                marker_color=color, nbinsx=25,
            ))

        fig.update_layout(
            barmode="overlay",
            **_layout(f"Distribution of {sel_feature} by Heart Disease Status", 400)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grid of all features
        st.markdown("#### 🗂️ All Feature Distributions")
        n_cols = 3
        chunks = [numeric_cols[i:i+n_cols] for i in range(0, len(numeric_cols), n_cols)]
        for chunk in chunks:
            cols = st.columns(n_cols)
            for col_ui, feat in zip(cols, chunk):
                with col_ui:
                    fig_mini = px.histogram(
                        df, x=feat, color="target",
                        color_discrete_map={0: "#00D4AA", 1: "#FF4B6E"},
                        barmode="overlay", opacity=0.75,
                        title=DISPLAY_NAMES.get(feat, feat),
                        labels={"target": "Class"},
                        nbins=20,
                    )
                    fig_mini.update_layout(
                        height=260, showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=PLOT_FONT,
                        margin=dict(l=10, r=10, t=35, b=10),
                        xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4 — Boxplots
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("#### 📦 Feature Distribution by Class (Boxplots)")
        numeric_cols = [c for c in df.columns if c != "target" and df[c].dtype != object]
        display_numeric_bx = [DISPLAY_NAMES.get(c, c) for c in numeric_cols]
        df_melt = df.melt(id_vars=["target"], value_vars=numeric_cols,
                          var_name="feature", value_name="value")
        df_melt["feature"] = df_melt["feature"].map(
            lambda x: DISPLAY_NAMES.get(x, x)
        )
        df_melt["Class"] = df_melt["target"].map({0: "Healthy", 1: "Heart Disease"})

        sel_box = st.selectbox("Select Feature", display_numeric_bx, key="box_feat")
        raw_box = {v: k for k, v in DISPLAY_NAMES.items()}.get(sel_box, sel_box)

        fig_box = go.Figure()
        for target_val, label, color in [(0, "Healthy", "#00D4AA"), (1, "Heart Disease", "#FF4B6E")]:
            subset = df[df["target"] == target_val][raw_box].dropna()
            fig_box.add_trace(go.Box(
                y=subset, name=label,
                marker_color=color,
                boxmean="sd",
                jitter=0.3, pointpos=-1.8,
                marker=dict(opacity=0.4, size=4),
            ))

        fig_box.update_layout(**_layout(f"Boxplot: {sel_box}", 420))
        st.plotly_chart(fig_box, use_container_width=True)

        # All boxplots in a grid
        st.markdown("#### 🗂️ All Features — Boxplot Grid")
        n_cols_bx = 3
        chunks_bx = [numeric_cols[i:i+n_cols_bx] for i in range(0, len(numeric_cols), n_cols_bx)]
        for chunk in chunks_bx:
            cols = st.columns(n_cols_bx)
            for col_ui, feat in zip(cols, chunk):
                with col_ui:
                    fig_bx = go.Figure()
                    for tv, lb, clr in [(0, "Healthy", "#00D4AA"), (1, "Disease", "#FF4B6E")]:
                        fig_bx.add_trace(go.Box(
                            y=df[df["target"] == tv][feat].dropna(),
                            name=lb, marker_color=clr, boxmean=True,
                        ))
                    fig_bx.update_layout(
                        title=DISPLAY_NAMES.get(feat, feat),
                        height=270, showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=PLOT_FONT,
                        margin=dict(l=10, r=10, t=35, b=10),
                        xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                    )
                    st.plotly_chart(fig_bx, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 5 — Correlations
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("#### 🌡️ Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr().round(3)
        labels = [DISPLAY_NAMES.get(c, c) for c in corr.columns]

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=labels, y=labels,
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(
            **_layout("Pearson Correlation Matrix", 600),
            xaxis=dict(tickangle=-40, **AXIS_STYLE),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Correlation with target
        st.markdown("#### 🎯 Feature Correlation with Heart Disease")
        target_corr = numeric_df.corr()["target"].drop("target").sort_values()
        labels_tc   = [DISPLAY_NAMES.get(c, c) for c in target_corr.index]
        colors_tc   = ["#FF4B6E" if v > 0 else "#00D4AA" for v in target_corr.values]

        fig_tc = go.Figure(go.Bar(
            x=target_corr.values, y=labels_tc,
            orientation="h", marker_color=colors_tc,
            text=[f"{v:.3f}" for v in target_corr.values],
            textposition="outside",
        ))
        fig_tc.update_layout(**_layout("Correlation with Heart Disease Target", 420))
        st.plotly_chart(fig_tc, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 6 — Feature Insights
    # ════════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("#### 🎯 Feature-wise Insights")

        # Age distribution
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if "age" in df.columns:
                fig_age = px.violin(
                    df, y="age", x="target", color="target",
                    color_discrete_map={0: "#00D4AA", 1: "#FF4B6E"},
                    box=True, points="outliers",
                    labels={"target": "Class", "age": "Age"},
                    title="Age Distribution by Class",
                )
                fig_age.update_layout(
                    **_layout("Age by Heart Disease Status", 380),
                    showlegend=False,
                )
                st.plotly_chart(fig_age, use_container_width=True)

        with col_f2:
            if "sex" in df.columns:
                sex_disease = df.groupby(["sex", "target"]).size().reset_index(name="count")
                sex_disease["Sex"]   = sex_disease["sex"].map({0: "Female", 1: "Male"})
                sex_disease["Class"] = sex_disease["target"].map({0: "Healthy", 1: "Disease"})
                fig_sex = px.bar(
                    sex_disease, x="Sex", y="count", color="Class",
                    color_discrete_map={"Healthy": "#00D4AA", "Disease": "#FF4B6E"},
                    barmode="group", title="Heart Disease by Sex",
                )
                fig_sex.update_layout(**_layout("Heart Disease by Sex", 380))
                st.plotly_chart(fig_sex, use_container_width=True)

        # Chest pain type
        if "cp" in df.columns:
            cp_labels = {0: "Typical Angina", 1: "Atypical Angina",
                         2: "Non-Anginal Pain", 3: "Asymptomatic"}
            df_cp = df.copy()
            df_cp["Chest Pain"] = df_cp["cp"].map(cp_labels)
            df_cp["Class"] = df_cp["target"].map({0: "Healthy", 1: "Disease"})
            cp_counts = df_cp.groupby(["Chest Pain", "Class"]).size().reset_index(name="Count")

            fig_cp = px.bar(
                cp_counts, x="Chest Pain", y="Count", color="Class",
                color_discrete_map={"Healthy": "#00D4AA", "Disease": "#FF4B6E"},
                barmode="group", title="Chest Pain Type vs Heart Disease",
            )
            fig_cp.update_layout(**_layout("", 380))
            st.plotly_chart(fig_cp, use_container_width=True)
