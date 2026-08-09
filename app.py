"""
Sales Forecasting Dashboard
----------------------------
A gamified Streamlit dashboard for exploring simulated_financial_forecasting_data.csv,
training a Random Forest model, and generating live predictions.

Honest-by-design note: `sales` is nearly collinear with `target_sales` (r ~ 0.98).
The five macro/market features are weak predictors on their own. This app surfaces
that finding explicitly rather than hiding it behind a shiny accuracy number.

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------
# 1. PAGE SETUP & THEMING
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecast Arena",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a "gamified" look: gradient header, card-style metrics, hover effects
CUSTOM_CSS = """
<style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35);
    }
    .hero-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.3rem;
        font-size: 1.05rem;
    }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 1rem 1rem 0.6rem 1rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 24px rgba(139, 92, 246, 0.35);
    }

    /* Badge / XP bar container */
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }

    .predict-result {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.6rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.4);
        margin-top: 1rem;
    }
    .predict-result h2 {
        color: white;
        font-size: 2.4rem;
        margin: 0;
    }
    .predict-result p {
        color: rgba(255,255,255,0.9);
        margin: 0.2rem 0 0 0;
    }

    /* Section subheaders */
    .section-title {
        border-left: 5px solid #8b5cf6;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.6rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

FEATURE_COLS = ["sales", "market_indicator_1", "market_indicator_2",
                 "gdp_growth", "unemployment_rate", "inflation_rate"]
MACRO_COLS = ["market_indicator_1", "market_indicator_2",
              "gdp_growth", "unemployment_rate", "inflation_rate"]
TARGET_COL = "target_sales"
SEED = 42

# --------------------------------------------------------------------------------------
# 2. DATA LOADING
# --------------------------------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """Load the CSV either from an uploaded file or the bundled local copy."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("simulated_financial_forecasting_data.csv")
    return df


@st.cache_resource
def train_model(df, use_sales: bool):
    """Train a Random Forest on either all features or macro-only features.
    Cached by (row-count fingerprint via df, use_sales) so retraining only happens
    when the underlying data or feature set actually changes.
    """
    cols = FEATURE_COLS if use_sales else MACRO_COLS
    X = df[cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "mae": mean_absolute_error(y_test, preds),
    }

    return model, cols, X_test, y_test, preds, metrics


# --------------------------------------------------------------------------------------
# HERO BANNER
# --------------------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>🎯 Sales Forecast Arena</h1>
        <p>Explore the data, train the model, and put your forecasting powers to the test.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# SIDEBAR: DATA UPLOAD, FILTERS, MODEL CONTROLS
# --------------------------------------------------------------------------------------

st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your own CSV (optional)", type=["csv"],
    help="Must contain the same 7 columns as simulated_financial_forecasting_data.csv"
)

df_raw = load_data(uploaded_file)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter the data")

sales_min, sales_max = float(df_raw["sales"].min()), float(df_raw["sales"].max())
sales_range = st.sidebar.slider(
    "Sales range", min_value=sales_min, max_value=sales_max,
    value=(sales_min, sales_max)
)

gdp_min, gdp_max = float(df_raw["gdp_growth"].min()), float(df_raw["gdp_growth"].max())
gdp_range = st.sidebar.slider(
    "GDP growth range", min_value=gdp_min, max_value=gdp_max,
    value=(gdp_min, gdp_max)
)

df = df_raw[
    (df_raw["sales"].between(*sales_range)) &
    (df_raw["gdp_growth"].between(*gdp_range))
].reset_index(drop=True)

st.sidebar.caption(f"Showing **{len(df)}** of {len(df_raw)} rows after filtering.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model mode")
use_sales_toggle = st.sidebar.radio(
    "Train the model using:",
    options=["All features (incl. sales)", "Macro indicators only (no sales)"],
    index=0,
)
use_sales = use_sales_toggle.startswith("All")

if len(df) < 50:
    st.sidebar.warning("Fewer than 50 rows after filtering — model quality may be unstable. "
                        "Widen the filters for a more reliable model.")
    df_for_model = df_raw  # fall back to full data so the model still trains sensibly
else:
    df_for_model = df

model, model_cols, X_test, y_test, preds, metrics = train_model(df_for_model, use_sales)

# --------------------------------------------------------------------------------------
# TOP-LEVEL KPI CARDS
# --------------------------------------------------------------------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("R² Score", f"{metrics['r2']:.3f}")
kpi2.metric("RMSE", f"{metrics['rmse']:.2f}")
kpi3.metric("MAE", f"{metrics['mae']:.2f}")

accuracy_pct = max(0.0, min(1.0, metrics["r2"])) * 100
kpi4.markdown(
    f'<div class="badge">🏆 Model Accuracy: {accuracy_pct:.0f}%</div>',
    unsafe_allow_html=True,
)

st.progress(max(0.0, min(1.0, metrics["r2"])))

if use_sales:
    st.info("💪 **sales** is in the mix — it's doing almost all the work. "
            "Accuracy here reflects near-collinearity, not genuine macro forecasting.")
else:
    st.warning("📉 Macro-only mode: no `sales` feature. This is the honest test — "
               "watch the R² drop. If it's near zero, the macro indicators aren't predictive.")

# --------------------------------------------------------------------------------------
# 3. TABS: Overview / Visualizations / Model & Prediction / Insights
# --------------------------------------------------------------------------------------

tab_overview, tab_viz, tab_model, tab_insights = st.tabs(
    ["📋 Overview", "📊 Visualizations", "🎮 Model & Prediction", "💡 Insights"]
)

# ---- TAB 1: OVERVIEW ----
with tab_overview:
    st.markdown('<h3 class="section-title">Dataset Snapshot</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing values", f"{int(df.isnull().sum().sum())}")

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<h3 class="section-title">Summary Statistics</h3>', unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)

# ---- TAB 2: VISUALIZATIONS ----
with tab_viz:
    st.markdown('<h3 class="section-title">Correlation Heatmap</h3>', unsafe_allow_html=True)
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    fig_heatmap = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
        zmin=-1, zmax=1,
    )
    fig_heatmap.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<h4 class="section-title">Distribution Explorer</h4>', unsafe_allow_html=True)
        selected_var = st.selectbox("Choose a variable", FEATURE_COLS + [TARGET_COL], index=0)
        fig_hist = px.histogram(
            df, x=selected_var, nbins=40, marginal="box",
            color_discrete_sequence=["#8b5cf6"],
        )
        fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown('<h4 class="section-title">sales vs target_sales</h4>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df, x="sales", y="target_sales", trendline="ols",
            color_discrete_sequence=["#ec4899"], opacity=0.55,
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<h4 class="section-title">Macro Indicator Radar (filtered avg)</h4>', unsafe_allow_html=True)
        # Normalize each macro column to 0-1 (using full dataset range) so the radar is readable
        radar_vals = []
        for c in MACRO_COLS:
            lo, hi = df_raw[c].min(), df_raw[c].max()
            val = df[c].mean() if len(df) else df_raw[c].mean()
            norm = (val - lo) / (hi - lo) if hi > lo else 0.5
            radar_vals.append(norm)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=MACRO_COLS + [MACRO_COLS[0]],
            fill="toself",
            line_color="#f59e0b",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False, margin=dict(l=30, r=30, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_d:
        st.markdown('<h4 class="section-title">Model Confidence Gauge</h4>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy_pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8b5cf6"},
                "steps": [
                    {"range": [0, 40], "color": "#450a0a"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100], "color": "#064e3b"},
                ],
            },
            title={"text": "R² as Accuracy Score"},
        ))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<h4 class="section-title">Feature Importance</h4>', unsafe_allow_html=True)
    importances = pd.Series(model.feature_importances_, index=model_cols).sort_values(ascending=True)
    fig_importance = px.bar(
        importances, orientation="h",
        color=importances.values, color_continuous_scale="Purples",
        labels={"value": "Importance", "index": "Feature"},
    )
    fig_importance.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_importance, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown('<h4 class="section-title">Predicted vs Actual</h4>', unsafe_allow_html=True)
        fig_pva = px.scatter(
            x=y_test, y=preds, opacity=0.55, color_discrete_sequence=["#10b981"],
            labels={"x": "Actual target_sales", "y": "Predicted target_sales"},
        )
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        fig_pva.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                      line=dict(color="red", dash="dash"), name="Perfect fit"))
        fig_pva.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pva, use_container_width=True)

    with col_f:
        st.markdown('<h4 class="section-title">Residuals</h4>', unsafe_allow_html=True)
        residuals = y_test - preds
        fig_resid = px.scatter(
            x=preds, y=residuals, opacity=0.55, color_discrete_sequence=["#f43f5e"],
            labels={"x": "Predicted target_sales", "y": "Residual"},
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        fig_resid.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_resid, use_container_width=True)

    st.markdown('<h4 class="section-title">Sales Trend (by row index)</h4>', unsafe_allow_html=True)
    fig_line = px.line(
        df.reset_index(), x="index", y=["sales", "target_sales"],
        color_discrete_sequence=["#6366f1", "#ec4899"],
    )
    fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_line, use_container_width=True)

# ---- TAB 3: MODEL & PREDICTION ----
with tab_model:
    st.markdown('<h3 class="section-title">🔮 Try a Prediction</h3>', unsafe_allow_html=True)
    st.caption(
        "Set feature values below, then hit **Predict**. "
        f"Currently training on: {'all features' if use_sales else 'macro indicators only'}."
    )

    input_cols = st.columns(3)
    input_values = {}
    for i, col in enumerate(model_cols):
        lo, hi = float(df_raw[col].min()), float(df_raw[col].max())
        default = float(df_raw[col].mean())
        with input_cols[i % 3]:
            input_values[col] = st.slider(
                col.replace("_", " ").title(), min_value=lo, max_value=hi, value=default
            )

    predict_clicked = st.button("🔮 Predict Sales", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([input_values])[model_cols]
        prediction = model.predict(input_df)[0]

        st.balloons()

        st.markdown(
            f"""
            <div class="predict-result">
                <p>Predicted target_sales</p>
                <h2>{prediction:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence gauge for this specific prediction run (reuses overall model R²)
        g1, g2 = st.columns(2)
        with g1:
            fig_pred_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy_pct,
                number={"suffix": "%"},
                title={"text": "Model Confidence (R²-based)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#10b981"}},
            ))
            fig_pred_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pred_gauge, use_container_width=True)

        with g2:
            fig_pred_hist = px.histogram(
                df_raw, x="target_sales", nbins=40, opacity=0.7,
                color_discrete_sequence=["#8b5cf6"],
            )
            fig_pred_hist.add_vline(
                x=prediction, line_color="#ec4899", line_width=3,
                annotation_text="Your prediction", annotation_position="top",
            )
            fig_pred_hist.update_layout(
                title="Where your prediction lands vs. historical target_sales",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pred_hist, use_container_width=True)

        if use_sales:
            st.toast("This feature is doing all the work! 💪", icon="💪")
        else:
            st.toast("Macro indicators alone are a tough sell — accuracy will drop 📉", icon="📉")

    st.markdown('<h4 class="section-title">Live Model Metrics</h4>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['r2']:.3f}")
    m2.metric("RMSE", f"{metrics['rmse']:.2f}")
    m3.metric("MAE", f"{metrics['mae']:.2f}")

# ---- TAB 4: INSIGHTS ----
with tab_insights:
    st.markdown('<h3 class="section-title">💡 What\'s really driving these predictions?</h3>', unsafe_allow_html=True)

    with st.expander("📌 Why does accuracy jump so much when `sales` is included?", expanded=True):
        st.write(
            """
            `sales` and `target_sales` are nearly the same signal — their correlation is
            about **0.98**. That means a model that includes `sales` doesn't need to
            understand economic conditions at all; it's essentially learning a
            near-identity mapping. Flip the sidebar toggle to **"Macro indicators only"**
            and watch the R² and accuracy badge fall — that's the honest picture of what
            the five macro/market features can actually explain on their own.
            """
        )

    with st.expander("📌 So are the macro indicators useless?"):
        st.write(
            """
            Based on this dataset: largely yes, for predicting `target_sales`. Their
            individual correlations with `target_sales` are all below 0.03 in magnitude,
            and a regression of the *leftover* variance (after accounting for `sales`)
            on the macro features explains almost none of it. That doesn't mean
            `gdp_growth`, `unemployment_rate`, etc. are meaningless in the real world —
            it means *in this particular simulated dataset*, they carry no measurable
            signal for this target.
            """
        )

    with st.expander("📌 What would make this a genuinely useful forecasting tool?"):
        st.write(
            """
            - A **time index** so `target_sales` represents a genuine future value,
              evaluated with a chronological (not random) split
            - **Segment identifiers** (region, product line) to capture cross-sectional
              variation instead of one dominant numeric feature
            - **Actionable levers** like pricing or promotional spend, which a business
              could actually change — unlike macro indicators, which are typically
              outside a company's control anyway
            """
        )

    st.markdown('<h4 class="section-title">Current Session Stats</h4>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    s1.metric("Mode", "All features" if use_sales else "Macro only")
    s2.metric("Rows used to train", f"{len(df_for_model):,}")
