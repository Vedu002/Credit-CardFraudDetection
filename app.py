"""
AI-Driven Credit Card Fraud Detection
Hybrid Deep Stacking Ensemble: 1D-CNN + Bi-LSTM + Transformer + XGBoost Meta-Learner

Based on: Ileberi & Sun (2024), IEEE Access.
"""
import json
import os
import time

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from fraud_detection import PositionEmbedding  # Custom layer for Transformer loading

# ----------------------------------------------------------------------------
# Page Config & Branding
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Driven Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "results/models"
RESULTS_CSV = "results/results.csv"
CURVES_PNG = "results/curves.png"
SAMPLE_CSV = "sample_real.csv"

# ----------------------------------------------------------------------------
# Custom CSS Design System (Cyber-Fintech Dark Glassmorphism)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #070B14;
    --card-bg: rgba(15, 23, 42, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --primary-cyan: #06B6D4;
    --primary-blue: #3B82F6;
    --accent-indigo: #6366F1;
    --accent-purple: #8B5CF6;
    --success-emerald: #10B981;
    --warning-amber: #F59E0B;
    --danger-rose: #EF4444;
    --text-muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Background & Glow */
.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 45%),
                #070B14;
    color: #F8FAFC;
}

/* Glass Card */
.sentinel-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 22px 24px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
    margin-bottom: 20px;
    transition: all 0.25s ease;
}

.sentinel-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.7);
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(8, 47, 73, 0.7) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.25);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-top: 6px;
    font-weight: 400;
}

/* Telemetry Pills */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    border: 1px solid transparent;
}

.badge-online {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border-color: rgba(16, 185, 129, 0.3);
}

.badge-neural {
    background: rgba(99, 102, 241, 0.18);
    color: #A5B4FC;
    border-color: rgba(99, 102, 241, 0.35);
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #67E8F9;
    border-color: rgba(6, 182, 212, 0.3);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.2);
    color: #FCA5A5;
    border-color: rgba(239, 68, 68, 0.4);
}

/* Pulsing Indicator */
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    box-shadow: 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse 1.8s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Metric Display Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #06B6D4);
}

.kpi-card.danger::before {
    background: linear-gradient(90deg, #EF4444, #F59E0B);
}

.kpi-card.success::before {
    background: linear-gradient(90deg, #10B981, #06B6D4);
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 8px 0 4px 0;
    font-family: 'JetBrains Mono', monospace;
}

.kpi-subtext {
    font-size: 0.78rem;
    color: #64748B;
    font-weight: 500;
}

/* Verdict Banners */
.verdict-fraud {
    background: linear-gradient(135deg, rgba(153, 27, 27, 0.4) 0%, rgba(26, 12, 19, 0.8) 100%);
    border: 1px solid rgba(239, 68, 68, 0.5);
    box-shadow: 0 0 35px -5px rgba(239, 68, 68, 0.35);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
}

.verdict-safe {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(8, 24, 25, 0.8) 100%);
    border: 1px solid rgba(16, 185, 129, 0.5);
    box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
}

.model-pill-score {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0A0F1D;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Custom Tabs */
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
    padding: 10px 18px;
    border-radius: 8px;
    transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0A0E1A;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Artifact Loader & Inference Logic
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts(models_dir):
    """Loads all saved models, scalers, and metadata."""
    required = [
        "cnn.keras", "lstm.keras", "transformer.keras",
        "xgb_base.json", "meta_xgb.json", "scaler.pkl", "feature_names.json"
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        return None

    custom = {"PositionEmbedding": PositionEmbedding}
    cnn = tf.keras.models.load_model(os.path.join(models_dir, "cnn.keras"), custom_objects=custom)
    lstm = tf.keras.models.load_model(os.path.join(models_dir, "lstm.keras"), custom_objects=custom)
    transformer = tf.keras.models.load_model(os.path.join(models_dir, "transformer.keras"), custom_objects=custom)

    xgb_base = XGBClassifier()
    xgb_base.load_model(os.path.join(models_dir, "xgb_base.json"))
    meta = XGBClassifier()
    meta.load_model(os.path.join(models_dir, "meta_xgb.json"))

    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    # Initialize fast SHAP TreeExplainer for feature attribution
    shap_explainer = shap.TreeExplainer(xgb_base)

    return dict(
        cnn=cnn, lstm=lstm, transformer=transformer,
        xgb_base=xgb_base, meta=meta, scaler=scaler,
        feature_names=feature_names,
        shap_explainer=shap_explainer
    )



def predict(art, X_raw):
    """Generates predictions across base learners and stacking meta-ensemble."""
    X = art["scaler"].transform(X_raw[art["feature_names"]].values.astype("float32"))
    p_cnn = art["cnn"].predict(X, verbose=0).ravel()
    p_lstm = art["lstm"].predict(X, verbose=0).ravel()
    p_tr = art["transformer"].predict(X, verbose=0).ravel()
    p_xgb = art["xgb_base"].predict_proba(X)[:, 1]
    Z = np.column_stack([p_cnn, p_lstm, p_tr, p_xgb])
    p_ensemble = art["meta"].predict_proba(Z)[:, 1]
    return pd.DataFrame({
        "1D-CNN": p_cnn,
        "Bi-LSTM": p_lstm,
        "Transformer": p_tr,
        "XGBoost (Raw)": p_xgb,
        "Ensemble": p_ensemble,
    })


# ----------------------------------------------------------------------------
# System Health Check
# ----------------------------------------------------------------------------
art = load_artifacts(MODELS_DIR)
if art is None:
    st.error(
        f"⚠️ **Model Artifacts Missing in `{MODELS_DIR}/`**\n\n"
        "Please generate the model weights first:\n"
        "```bash\npython fraud_detection.py --data creditcard.csv --epochs 15\n```\n"
        "Then restart this application."
    )
    st.stop()


# ----------------------------------------------------------------------------
# Sidebar Control Deck
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 1.8rem;">💳</span>
            <div>
                <div style="font-size: 1.08rem; font-weight: 800; letter-spacing: -0.01em; color: #FFFFFF; line-height: 1.2;">AI-Driven Fraud Detection</div>
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Credit Card Intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0 16px 0;'>", unsafe_allow_html=True)

    # Telemetry Status Box
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px 14px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; font-weight: 600; color: #94A3B8;">NETWORK STATUS</span>
                <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; color: #34D399; font-weight: 700;">
                    <span class="pulse-dot"></span> ONLINE
                </span>
            </div>
            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.6;">
                <div>• 1D-CNN (Spatial Conv) <span style="color:#10B981; float:right;">Active</span></div>
                <div>• Bi-LSTM (Temporal Seq) <span style="color:#10B981; float:right;">Active</span></div>
                <div>• Transformer (Attention) <span style="color:#10B981; float:right;">Active</span></div>
                <div>• XGBoost (Gradient Tree) <span style="color:#10B981; float:right;">Active</span></div>
                <div>• Meta-Stacking Ensemble <span style="color:#38BDF8; float:right;">Synchronized</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Decision Policy")
    threshold = st.slider(
        "Alert Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Transactions with Ensemble score >= Threshold trigger immediate fraud escalation.",
    )

    # Preset Modes
    st.caption("Quick Threshold Presets:")
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("Strict", help="Threshold = 0.30 (Max fraud catch rate)"):
        st.session_state["preset_threshold"] = 0.30
        threshold = 0.30
        st.rerun()
    if col_p2.button("Balance", help="Threshold = 0.50 (Recommended baseline)"):
        st.session_state["preset_threshold"] = 0.50
        threshold = 0.50
        st.rerun()
    if col_p3.button("Lenient", help="Threshold = 0.75 (Frictionless commerce)"):
        st.session_state["preset_threshold"] = 0.75
        threshold = 0.75
        st.rerun()

    # Dynamic explanation based on threshold
    if threshold < 0.35:
        mode_badge = "<span style='color: #F59E0B;'>⚠️ HIGH SENSITIVITY</span>"
        mode_desc = "Catches nearly all fraudulent attempts, but review queue volume increases."
    elif threshold > 0.65:
        mode_badge = "<span style='color: #38BDF8;'>🕊️ HIGH SPECIFICITY</span>"
        mode_desc = "Minimizes false alarms and checkout friction. Conservative flagging."
    else:
        mode_badge = "<span style='color: #10B981;'>⚖️ OPTIMAL EQUILIBRIUM</span>"
        mode_desc = "Balanced production trade-off between recall and low customer friction."

    st.markdown(f"""
        <div style="font-size: 0.76rem; background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 10px; margin-top: 10px; border-left: 3px solid #38BDF8;">
            <b>Policy Mode:</b> {mode_badge}<br>
            <span style="color: #94A3B8;">{mode_desc}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size: 0.74rem; color: #64748B; line-height: 1.5;">
            <b>Reference Literature:</b><br>
            Ileberi & Sun (2024), <i>IEEE Access</i>.<br>
            "A Hybrid Deep Learning Ensemble Model for Credit Card Fraud Detection"
        </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Hero Header
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span class="badge-pill badge-online"><span class="pulse-dot"></span> Production Engine Active</span>
            <span class="badge-pill badge-neural">Hybrid Neural Stacking</span>
            <span class="badge-pill badge-cyan">IEEE Access 2024</span>
        </div>
        <h1 class="hero-title">💳 AI-Driven Credit Card Fraud Detection</h1>
        <div class="hero-subtitle">
            Autonomous multi-model surveillance combining <b>1D-CNN</b> (Spatial), <b>Bi-LSTM</b> (Temporal), <b>Transformer</b> (Self-Attention), and <b>XGBoost Stacking</b>.
        </div>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
        <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 10px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Active Threshold</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">{threshold:.2f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Main Navigation Tabs
# ----------------------------------------------------------------------------
tab_batch, tab_single, tab_arch, tab_policy = st.tabs([
    "📊 Batch Audit & Stream Scanner",
    "⚡ Real-Time Single Transaction Inspector",
    "🔬 Model Architecture & Benchmarks",
    "🛡️ Decision Rules & Financial Impact",
])


# ============================================================================
# TAB 1: Batch Scoring & Quarantine Stream
# ============================================================================
with tab_batch:
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Transaction Batch Audit</h3>
                <div style="font-size: 0.85rem; color: #94A3B8;">Analyze high-volume transaction payloads through the multi-learner ensemble.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_src1, col_src2 = st.columns([1.2, 2])
    with col_src1:
        data_source = st.radio(
            "Select Data Payload:",
            ["⚡ Built-in Audit Sample (Real Fraud Cases)", "📤 Upload Custom CSV"],
            horizontal=False,
        )

    df_to_score = None

    if "Built-in" in data_source:
        if os.path.exists(SAMPLE_CSV):
            df_to_score = pd.read_csv(SAMPLE_CSV)
            st.success(f"✓ Loaded `{SAMPLE_CSV}` with **{len(df_to_score)}** transactions ready for inference.")
        else:
            st.error(f"Sample file `{SAMPLE_CSV}` not found in root folder.")
    else:
        uploaded_file = st.file_uploader("Upload CSV formatted with V1-V28, Time, Amount", type=["csv"])
        if uploaded_file is not None:
            df_to_score = pd.read_csv(uploaded_file)

    if df_to_score is not None:
        missing_cols = [c for c in art["feature_names"] if c not in df_to_score.columns]
        if missing_cols:
            st.error(f"❌ Input CSV is missing required model features: `{missing_cols}`")
        else:
            # Score
            t_start = time.time()
            with st.spinner("Executing neural-tree ensemble inference..."):
                predictions_df = predict(art, df_to_score)
            infer_duration = (time.time() - t_start) * 1000

            scored_df = pd.concat([df_to_score.reset_index(drop=True), predictions_df], axis=1)
            scored_df["Flagged"] = scored_df["Ensemble"] >= threshold
            scored_df["Risk Tier"] = np.where(
                scored_df["Ensemble"] >= threshold, "🚨 CRITICAL",
                np.where(scored_df["Ensemble"] >= 0.30, "⚠️ SUSPICIOUS", "✅ SAFE")
            )

            n_total = len(scored_df)
            n_flagged = int(scored_df["Flagged"].sum())
            flag_rate = n_flagged / n_total if n_total > 0 else 0.0

            total_volume = float(scored_df["Amount"].sum()) if "Amount" in scored_df.columns else 0.0
            flagged_volume = float(scored_df.loc[scored_df["Flagged"], "Amount"].sum()) if "Amount" in scored_df.columns else 0.0

            # Consensus rate between base learners and ensemble
            base_cols = ["1D-CNN", "Bi-LSTM", "Transformer", "XGBoost (Raw)"]
            base_votes = (scored_df[base_cols] >= threshold).sum(axis=1)
            consensus_agreed = ((base_votes >= 2) == scored_df["Flagged"]).mean()

            # KPI Cards
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-card">
                        <div class="kpi-label">💳 Audited Payload</div>
                        <div class="kpi-value">{n_total:,}</div>
                        <div class="kpi-subtext">Latency: {infer_duration:.1f}ms (~{infer_duration/n_total:.2f}ms / row)</div>
                    </div>
                    <div class="kpi-card danger">
                        <div class="kpi-label">🚨 Fraud Quarantine</div>
                        <div class="kpi-value" style="color: #F87171;">{n_flagged:,} <span style="font-size: 1rem; color: #94A3B8;">({flag_rate:.1%})</span></div>
                        <div class="kpi-subtext">Transactions >= {threshold:.2f} probability</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">💰 Protected Capital</div>
                        <div class="kpi-value" style="color: #38BDF8;">${flagged_volume:,.2f}</div>
                        <div class="kpi-subtext">Total volume screened: ${total_volume:,.2f}</div>
                    </div>
                    <div class="kpi-card success">
                        <div class="kpi-label">🤝 Multi-Model Consensus</div>
                        <div class="kpi-value" style="color: #34D399;">{consensus_agreed:.1%}</div>
                        <div class="kpi-subtext">Concordance across 4 base learners</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Interactive Visualizations
            st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; margin-top: 10px;'>Risk Distribution & Anomaly Clustering</h4>", unsafe_allow_html=True)
            vis_c1, vis_c2 = st.columns(2)

            with vis_c1:
                # Probability Distribution Histogram
                hist_data = scored_df[["Ensemble", "Risk Tier"]].copy()
                hist_chart = alt.Chart(hist_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Ensemble:Q", bin=alt.Bin(maxbins=25), title="Ensemble Fraud Probability Score"),
                    y=alt.Y("count()", title="Transaction Volume"),
                    color=alt.Color("Risk Tier:N", scale=alt.Scale(
                        domain=["✅ SAFE", "⚠️ SUSPICIOUS", "🚨 CRITICAL"],
                        range=["#10B981", "#F59E0B", "#EF4444"]
                    )),
                    tooltip=["count()", alt.Tooltip("Risk Tier:N")]
                ).properties(
                    title="Risk Score Probability Spectrum",
                    height=280
                ).configure_view(
                    strokeWidth=0
                )
                st.altair_chart(hist_chart, use_container_width=True)

            with vis_c2:
                # Amount vs Probability Scatter Plot
                if "Amount" in scored_df.columns:
                    scatter_data = scored_df[["Amount", "Ensemble", "Risk Tier", "1D-CNN", "Bi-LSTM", "Transformer", "XGBoost (Raw)"]].copy()
                    scatter_chart = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.8).encode(
                        x=alt.X("Amount:Q", title="Transaction Amount ($)", scale=alt.Scale(type="log")),
                        y=alt.Y("Ensemble:Q", title="Ensemble Fraud Risk", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Risk Tier:N", scale=alt.Scale(
                            domain=["✅ SAFE", "⚠️ SUSPICIOUS", "🚨 CRITICAL"],
                            range=["#10B981", "#F59E0B", "#EF4444"]
                        )),
                        tooltip=["Amount:Q", "Ensemble:Q", "1D-CNN:Q", "Transformer:Q", "Risk Tier:N"]
                    ).properties(
                        title="Exposure vs Risk Score (Log Scale)",
                        height=280
                    ).configure_view(
                        strokeWidth=0
                    )
                    st.altair_chart(scatter_chart, use_container_width=True)

            # Quarantine Table & Filtering
            st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; margin-top: 20px;'>Quarantine Stream & Risk Inspector</h4>", unsafe_allow_html=True)
            filter_tier = st.selectbox(
                "Filter stream view:",
                ["All Transactions", "🚨 Critical Fraud Only (Score >= Threshold)", "⚠️ Suspicious Review (Score >= 0.30)", "✅ Verified Safe (< 0.30)"],
                index=0
            )

            filtered_df = scored_df.copy()
            if "Critical" in filter_tier:
                filtered_df = filtered_df[filtered_df["Risk Tier"] == "🚨 CRITICAL"]
            elif "Suspicious" in filter_tier:
                filtered_df = filtered_df[filtered_df["Risk Tier"].isin(["⚠️ SUSPICIOUS", "🚨 CRITICAL"])]
            elif "Safe" in filter_tier:
                filtered_df = filtered_df[filtered_df["Risk Tier"] == "✅ SAFE"]

            display_cols = ["Risk Tier", "Ensemble", "1D-CNN", "Bi-LSTM", "Transformer", "XGBoost (Raw)", "Amount"] + [c for c in ["Class", "Time"] if c in filtered_df.columns]
            st.dataframe(
                filtered_df[display_cols].sort_values("Ensemble", ascending=False).style.format({
                    "Ensemble": "{:.2%}",
                    "1D-CNN": "{:.2%}",
                    "Bi-LSTM": "{:.2%}",
                    "Transformer": "{:.2%}",
                    "XGBoost (Raw)": "{:.2%}",
                    "Amount": "${:,.2f}",
                }).background_gradient(subset=["Ensemble"], cmap="Reds", vmin=0, vmax=1),
                use_container_width=True,
                height=320,
            )

            # Ground Truth Validation (if 'Class' column exists)
            if "Class" in df_to_score.columns:
                from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

                y_true = df_to_score["Class"].astype(int).values
                y_pred = scored_df["Flagged"].astype(int).values

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                auc_roc = roc_auc_score(y_true, scored_df["Ensemble"])
                auc_pr = average_precision_score(y_true, scored_df["Ensemble"])
                f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

                st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;'>", unsafe_allow_html=True)
                st.markdown("#### 🎯 Ground Truth Audit (Class Label Present)")

                gt1, gt2, gt3, gt4, gt5 = st.columns(5)
                gt1.metric("Sensitivity (Recall)", f"{recall:.2%}", help="Fraction of actual fraud successfully detected")
                gt2.metric("Specificity", f"{specificity:.2%}", help="Fraction of legitimate transactions approved")
                gt3.metric("F1-Score", f"{f1_score:.3f}", help="Harmonic mean of precision and recall")
                gt4.metric("AUC-ROC", f"{auc_roc:.3f}", help="Area under Receiver Operating Characteristic curve")
                gt5.metric("AUC-PR", f"{auc_pr:.3f}", help="Area under Precision-Recall curve")

                # Confusion Matrix visual pills
                st.markdown(f"""
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px;">
                        <div class="model-pill-score" style="border-left: 3px solid #10B981;">
                            <div style="font-size: 0.72rem; color: #94A3B8;">TRUE POSITIVES (CAUGHT)</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #10B981;">{tp}</div>
                        </div>
                        <div class="model-pill-score" style="border-left: 3px solid #F59E0B;">
                            <div style="font-size: 0.72rem; color: #94A3B8;">FALSE POSITIVES (FALSE ALARM)</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #F59E0B;">{fp}</div>
                        </div>
                        <div class="model-pill-score" style="border-left: 3px solid #EF4444;">
                            <div style="font-size: 0.72rem; color: #94A3B8;">FALSE NEGATIVES (MISSED)</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #EF4444;">{fn}</div>
                        </div>
                        <div class="model-pill-score" style="border-left: 3px solid #38BDF8;">
                            <div style="font-size: 0.72rem; color: #94A3B8;">TRUE NEGATIVES (CLEAN)</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8;">{tn}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Export Button
            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Scored Audit Report (CSV)",
                data=scored_df.to_csv(index=False),
                file_name="sentinel_scored_audit.csv",
                mime="text/csv",
                help="Exports the full dataset appended with model predictions and risk flags."
            )


# ============================================================================
# TAB 2: Real-Time Single Transaction Inspector & Simulator
# ============================================================================
with tab_single:
    st.markdown("""
        <div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Real-Time Transaction Inspector</h3>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 14px;">
                Inspect individual transactions in real-time or test pre-configured fraud / legitimate profiles.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Scenarios / Presets
    PRESETS = {
        "🚨 Card Skimming Attack (Confirmed Fraud)": {
            "Time": 406.0, "Amount": 0.0,
            "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.998, "V5": -0.522, "V6": -1.427,
            "V7": -2.537, "V8": 1.392, "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.899,
            "V13": -0.595, "V14": -4.289, "V15": 0.390, "V16": -1.141, "V17": -2.830, "V18": -0.017,
            "V19": 0.417, "V20": 0.127, "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
            "V25": 0.045, "V26": 0.178, "V27": 0.261, "V28": -0.143
        },
        "🚨 High-Value Midnight Theft (Confirmed Fraud)": {
            "Time": 4462.0, "Amount": 239.93,
            "V1": -2.303, "V2": 1.759, "V3": -0.360, "V4": 2.330, "V5": -0.822, "V6": -0.076,
            "V7": 0.562, "V8": -0.399, "V9": -0.238, "V10": -1.525, "V11": 2.033, "V12": -6.560,
            "V13": 0.023, "V14": -1.470, "V15": -0.699, "V16": -2.282, "V17": -4.782, "V18": -2.616,
            "V19": -1.334, "V20": -0.430, "V21": -0.294, "V22": -0.932, "V23": 0.173, "V24": -0.087,
            "V25": -0.156, "V26": -0.543, "V27": 0.040, "V28": -0.153
        },
        "✅ Verified Grocery Purchase (Legitimate)": {
            "Time": 78340.0, "Amount": 1.98,
            "V1": -0.815, "V2": 1.319, "V3": 1.329, "V4": 0.027, "V5": -0.285, "V6": -0.654,
            "V7": 0.322, "V8": 0.436, "V9": -0.704, "V10": -0.601, "V11": 0.097, "V12": 0.710,
            "V13": 0.779, "V14": 0.354, "V15": 0.953, "V16": -0.104, "V17": 0.129, "V18": -0.964,
            "V19": -0.477, "V20": -0.009, "V21": -0.129, "V22": -0.369, "V23": 0.091, "V24": 0.401,
            "V25": -0.261, "V26": 0.081, "V27": 0.162, "V28": 0.059
        },
        "✅ High-End Electronics Purchase (Legitimate)": {
            "Time": 73506.0, "Amount": 690.23,
            "V1": 0.144, "V2": -2.652, "V3": -1.021, "V4": -0.086, "V5": -1.189, "V6": -0.350,
            "V7": 0.655, "V8": -0.411, "V9": -0.795, "V10": 0.312, "V11": -1.508, "V12": -0.374,
            "V13": 0.245, "V14": 0.159, "V15": 0.122, "V16": -1.455, "V17": 0.040, "V18": 1.029,
            "V19": -0.365, "V20": 0.907, "V21": -0.092, "V22": -1.215, "V23": -0.709, "V24": -0.378,
            "V25": 0.288, "V26": 1.072, "V27": -0.200, "V28": 0.108
        }
    }
    # Active preset in session state or default
    preset_names = list(PRESETS.keys())
    active_profile = st.session_state.get("active_preset", preset_names[0])
    defaults = PRESETS[active_profile]

    # Initialize session_state defaults if not present
    for f in art["feature_names"]:
        if f"inp_{f}" not in st.session_state:
            st.session_state[f"inp_{f}"] = float(defaults.get(f, 0.0))

    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94A3B8; margin-bottom: 8px;'>QUICK SCENARIO PRESETS</div>", unsafe_allow_html=True)
    p_cols = st.columns(4)
    preset_names = list(PRESETS.keys())
    for idx, p_name in enumerate(preset_names):
        with p_cols[idx]:
            if st.button(p_name, key=f"btn_p_{idx}", use_container_width=True):
                st.session_state["active_preset"] = p_name
                for k, v in PRESETS[p_name].items():
                    st.session_state[f"inp_{k}"] = float(v)
                st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'>", unsafe_allow_html=True)

    # Primary Input Controls
    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        st.markdown("<div class='sentinel-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-bottom: 12px;'>💳 Transaction Context</div>", unsafe_allow_html=True)

        val_amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=25000.0,
            step=1.0,
            format="%.2f",
            key="inp_Amount"
        )

        val_time = st.number_input(
            "Timestamp Offset (seconds)",
            min_value=0.0,
            max_value=200000.0,
            step=100.0,
            key="inp_Time"
        )

        st.markdown("<div style='margin-top: 16px; font-size: 0.8rem; color: #94A3B8; font-weight: 600;'>PRIMARY ANOMALY INDICATORS</div>", unsafe_allow_html=True)
        st.caption("PCA components with highest statistical correlation to credit card fraud anomalies (V14, V10, V12, V17, V4, V11).")

        key_anom = ["V14", "V10", "V12", "V17", "V4", "V11"]
        for feat in key_anom:
            st.slider(
                f"{feat} Component",
                min_value=-25.0,
                max_value=25.0,
                step=0.1,
                key=f"inp_{feat}"
            )

        # Collapsible Secondary Features
        with st.expander("🛠️ Secondary PCA Features (V1-V9, V13-V28)", expanded=False):
            sec_cols = st.columns(3)
            remaining_feats = [f for f in art["feature_names"] if f not in ["Time", "Amount"] + key_anom]
            for i, feat in enumerate(remaining_feats):
                with sec_cols[i % 3]:
                    st.number_input(
                        feat,
                        min_value=-25.0,
                        max_value=25.0,
                        step=0.1,
                        key=f"inp_{feat}"
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    # Live Inference & Dramatic Verdict Display
    with col_in2:
        # Build vector directly from widget state
        row_dict = {}
        for f in art["feature_names"]:
            row_dict[f] = float(st.session_state.get(f"inp_{f}", defaults.get(f, 0.0)))

        row_df = pd.DataFrame([row_dict])
        single_probs = predict(art, row_df).iloc[0]
        ens_prob = float(single_probs["Ensemble"])
        is_fraud = ens_prob >= threshold

        # Dramatic Verdict Card
        if is_fraud:
            verdict_html = f"""
                <div class="verdict-fraud">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="badge-pill badge-danger"><span class="pulse-dot" style="background:#EF4444;"></span> HIGH RISK CRITICAL</span>
                            <h2 style="color: #F87171; margin: 10px 0 4px 0; font-size: 1.8rem; font-weight: 800;">🚨 TRANSACTION FLAGGED AS FRAUD</h2>
                            <div style="color: #FCA5A5; font-size: 0.92rem;">Autonomous Action: <b>CARD ACCESS FROZEN • 2FA STEP-UP CHALLENGE</b></div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Confidence Score</div>
                            <div style="font-size: 2.3rem; font-weight: 900; color: #EF4444; font-family: 'JetBrains Mono', monospace;">{ens_prob:.1%}</div>
                        </div>
                    </div>
                </div>
            """
        else:
            verdict_html = f"""
                <div class="verdict-safe">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="badge-pill badge-online"><span class="pulse-dot"></span> CLEAN TRANSACTION</span>
                            <h2 style="color: #34D399; margin: 10px 0 4px 0; font-size: 1.8rem; font-weight: 800;">✅ TRANSACTION AUTHORIZED</h2>
                            <div style="color: #A7F3D0; font-size: 0.92rem;">Autonomous Action: <b>INSTANT ZERO-FRICTION COMMERCE APPROVAL</b></div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Fraud Likelihood</div>
                            <div style="font-size: 2.3rem; font-weight: 900; color: #10B981; font-family: 'JetBrains Mono', monospace;">{ens_prob:.1%}</div>
                        </div>
                    </div>
                </div>
            """

        st.markdown(verdict_html, unsafe_allow_html=True)

        # Multi-Model Voting Breakdown Cards
        st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #94A3B8; margin-bottom: 8px;'>ENSEMBLE BASE-LEARNER DISCORDANCE</div>", unsafe_allow_html=True)

        m_cols = st.columns(4)
        models_meta = [
            ("1D-CNN", single_probs["1D-CNN"], "Spatial Convolution", "#38BDF8"),
            ("Bi-LSTM", single_probs["Bi-LSTM"], "Temporal Recurrence", "#818CF8"),
            ("Transformer", single_probs["Transformer"], "Self-Attention", "#C084FC"),
            ("XGBoost (Raw)", single_probs["XGBoost (Raw)"], "Boosted Trees", "#F472B6"),
        ]

        for m_idx, (m_name, m_prob, m_role, m_color) in enumerate(models_meta):
            with m_cols[m_idx]:
                st.markdown(f"""
                    <div class="model-pill-score" style="border-top: 3px solid {m_color};">
                        <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600;">{m_name}</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{m_prob:.1%}</div>
                        <div style="font-size: 0.68rem; color: #64748B;">{m_role}</div>
                    </div>
                """, unsafe_allow_html=True)

        # Interactive Comparative Bar Chart
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Architecture": ["1D-CNN", "Bi-LSTM", "Transformer", "XGBoost (Raw)", "Ensemble Meta"],
            "Fraud Probability": [
                single_probs["1D-CNN"],
                single_probs["Bi-LSTM"],
                single_probs["Transformer"],
                single_probs["XGBoost (Raw)"],
                single_probs["Ensemble"]
            ],
            "Type": ["Base Learner", "Base Learner", "Base Learner", "Base Learner", "Stacking Meta"]
        })

        bar_chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Architecture:N", title=None, sort=None),
            y=alt.Y("Fraud Probability:Q", title="Probability Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Architecture:N", scale=alt.Scale(
                domain=["1D-CNN", "Bi-LSTM", "Transformer", "XGBoost (Raw)", "Ensemble Meta"],
                range=["#06B6D4", "#6366F1", "#A855F7", "#EC4899", "#EF4444" if is_fraud else "#10B981"]
            ), legend=None),
            tooltip=["Architecture:N", alt.Tooltip("Fraud Probability:Q", format=".2%")]
        ).properties(
            title="Individual Model Decision Breakdown",
            height=240
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # --------------------------------------------------------------------
        # Explainable AI: SHAP Feature Attribution
        # --------------------------------------------------------------------
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 26px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h4 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">
                        🔍 Explainable AI (XAI): SHAP Feature Attribution
                    </h4>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 3px;">
                        Cooperative Game Theory Shapley values showing why this specific transaction was classified as fraud or legitimate.
                    </div>
                </div>
                <span class="badge-pill" style="background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3);">
                    ⚡ TreeExplainer Active
                </span>
            </div>
        """, unsafe_allow_html=True)

        # Compute SHAP explanation for this transaction
        X_scaled_single = art["scaler"].transform(row_df[art["feature_names"]].values.astype("float32"))
        X_scaled_single_df = pd.DataFrame(X_scaled_single, columns=art["feature_names"])
        shap_explanation = art["shap_explainer"](X_scaled_single_df)

        vals = shap_explanation[0].values
        feat_names = np.array(art["feature_names"])
        top_risk_idx = np.argsort(vals)[::-1]
        top_mitigate_idx = np.argsort(vals)

        # Top Drivers KPI Cards
        c_risk, c_mit = st.columns(2)
        with c_risk:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #F87171; margin-bottom: 8px;'>🚨 TOP FRAUD RISK DRIVERS (+SHAP)</div>", unsafe_allow_html=True)
            top_pos = [i for i in top_risk_idx if vals[i] > 0.001][:3]
            if top_pos:
                for idx in top_pos:
                    raw_val = row_dict[feat_names[idx]]
                    st.markdown(f"""
                        <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; font-size: 0.82rem;">
                            <span><b>{feat_names[idx]}</b> = <span style="font-family: 'JetBrains Mono', monospace; color:#CBD5E1;">{raw_val:.3f}</span></span>
                            <span style="color: #F87171; font-weight: 700; font-family: 'JetBrains Mono', monospace;">+{vals[idx]:.3f} SHAP</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No positive fraud risk drivers for this transaction.")

        with c_mit:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #34D399; margin-bottom: 8px;'>🛡️ TOP CLEARANCE FACTORS (-SHAP)</div>", unsafe_allow_html=True)
            top_neg = [i for i in top_mitigate_idx if vals[i] < -0.001][:3]
            if top_neg:
                for idx in top_neg:
                    raw_val = row_dict[feat_names[idx]]
                    st.markdown(f"""
                        <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; font-size: 0.82rem;">
                            <span><b>{feat_names[idx]}</b> = <span style="font-family: 'JetBrains Mono', monospace; color:#CBD5E1;">{raw_val:.3f}</span></span>
                            <span style="color: #34D399; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{vals[idx]:.3f} SHAP</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No significant mitigating clearance factors for this transaction.")

        # Matplotlib Dark Glassmorphism SHAP Waterfall Chart
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")

        with plt.rc_context({
            "text.color": "#F1F5F9",
            "axes.labelcolor": "#94A3B8",
            "xtick.color": "#94A3B8",
            "ytick.color": "#E2E8F0",
            "font.family": "sans-serif",
            "font.size": 8.5
        }):
            shap.plots.waterfall(shap_explanation[0], max_display=9, show=False)
            plt.title("Transaction Decision Waterfall (f(x) vs E[f(X)])", color="#F8FAFC", fontsize=11, fontweight="bold", pad=12)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)



# ============================================================================
# TAB 3: Model Architecture & Benchmarks
# ============================================================================
with tab_arch:
    st.markdown("""
        <div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Deep Learning & Stacking Architecture</h3>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 18px;">
                Architecture details and empirical performance metrics on the Kaggle European Cardholders dataset.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Architecture Pipeline Diagram
    st.markdown("""
        <div class="sentinel-card">
            <div style="font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-bottom: 14px;">🧠 Hybrid 2-Stage Stacking Topology</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; text-align: center;">
                <div style="background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">STAGE 1A</div>
                    <div style="font-size: 1rem; font-weight: 800; color: #38BDF8; margin: 4px 0;">1D-CNN</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Spatial Conv filters (32, 64, 128) + MaxPool + Dropout</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">STAGE 1B</div>
                    <div style="font-size: 1rem; font-weight: 800; color: #818CF8; margin: 4px 0;">Bi-LSTM</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Stacked Recurrent Units (50 & 100) for sequential relations</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">STAGE 1C</div>
                    <div style="font-size: 1rem; font-weight: 800; color: #C084FC; margin: 4px 0;">Transformer</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Position Embedding + Multi-Head Self-Attention (4 heads)</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">STAGE 1D</div>
                    <div style="font-size: 1rem; font-weight: 800; color: #F472B6; margin: 4px 0;">XGBoost Base</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Gradient Boosted Decision Trees trained on raw feature vector</div>
                </div>
                <div style="background: rgba(14, 116, 144, 0.25); padding: 14px; border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.4);">
                    <div style="font-size: 0.72rem; color: #67E8F9; font-weight: 700;">STAGE 2 (META)</div>
                    <div style="font-size: 1rem; font-weight: 800; color: #22D3EE; margin: 4px 0;">XGBoost Meta</div>
                    <div style="font-size: 0.75rem; color: #E0F2FE;">Trained on out-of-fold base learner probabilities</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Benchmark Results CSV display
    if os.path.exists(RESULTS_CSV):
        res_df = pd.read_csv(RESULTS_CSV, index_col=0)
        st.markdown("#### 🏆 Independent Test-Set Benchmark Table")
        st.caption("Stratified 60/20/20 train/validation/test split evaluated on 56,962 untouched holdout transactions.")

        st.dataframe(
            res_df.style.format({
                "Sensitivity": "{:.2%}",
                "Specificity": "{:.2%}",
                "Precision": "{:.2%}",
                "F1": "{:.3f}",
                "AUC-ROC": "{:.4f}",
                "AUC-PR": "{:.4f}",
                "TP": "{:,}", "FP": "{:,}", "FN": "{:,}", "TN": "{:,}"
            }).highlight_max(subset=["Sensitivity", "Specificity", "F1", "AUC-ROC", "AUC-PR"], color="#065F46"),
            use_container_width=True
        )

        # Comparative AUC-ROC & Sensitivity Chart
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        chart_res = res_df.reset_index().rename(columns={"index": "Model"})
        melted_res = chart_res.melt(id_vars=["Model"], value_vars=["Sensitivity", "Specificity", "AUC-ROC"], var_name="Metric", value_name="Score")

        perf_chart = alt.Chart(melted_res).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-15)),
            y=alt.Y("Score:Q", title="Metric Score (0-1)", scale=alt.Scale(domain=[0.8, 1.0])),
            color=alt.Color("Metric:N", scale=alt.Scale(
                domain=["Sensitivity", "Specificity", "AUC-ROC"],
                range=["#EC4899", "#10B981", "#38BDF8"]
            )),
            xOffset="Metric:N",
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".2%")]
        ).properties(
            title="Model Performance Comparison (Sensitivity vs Specificity vs AUC-ROC)",
            height=300
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(perf_chart, use_container_width=True)

    # Curves Display
    if os.path.exists(CURVES_PNG):
        st.markdown("#### 📈 Empirical ROC and Precision-Recall Curves")
        st.image(CURVES_PNG, caption="Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves generated by fraud_detection.py", use_container_width=True)


# ============================================================================
# TAB 4: Decision Rules & Policy Simulation
# ============================================================================
with tab_policy:
    st.markdown("""
        <div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Decision Rules & Financial Impact Simulator</h3>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 18px;">
                Simulate business policy tiers, chargeback recovery, and cost savings from AI-driven fraud mitigation.
            </div>
        </div>
    """, unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns(2)

    with sim_c1:
        st.markdown("<div class='sentinel-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-bottom: 12px;'>🧮 Enterprise Volume Parameters</div>", unsafe_allow_html=True)

        monthly_volume = st.number_input("Monthly Transactions Audited", min_value=1000, max_value=50000000, value=250000, step=10000)
        avg_ticket = st.number_input("Average Transaction Amount ($)", min_value=5.0, max_value=5000.0, value=95.0, step=5.0)
        baseline_fraud_rate = st.slider("Baseline Fraud Rate (%)", min_value=0.01, max_value=2.00, value=0.17, step=0.01) / 100
        chargeback_fee = st.number_input("Merchant Bank Chargeback Fee ($ / incident)", min_value=10.0, max_value=100.0, value=25.0, step=5.0)

        st.markdown("</div>", unsafe_allow_html=True)

    with sim_c2:
        # Calculate impact
        est_fraud_incidents = monthly_volume * baseline_fraud_rate
        est_fraud_dollars = est_fraud_incidents * avg_ticket
        est_chargeback_fines = est_fraud_incidents * chargeback_fee
        total_baseline_loss = est_fraud_dollars + est_chargeback_fines

        # Ensemble detection assumptions from test metrics (86.9% recall)
        ai_prevented_fraud_dollars = est_fraud_dollars * 0.869
        ai_prevented_fines = est_chargeback_fines * 0.869
        total_savings = ai_prevented_fraud_dollars + ai_prevented_fines
        remaining_loss = total_baseline_loss - total_savings

        st.markdown(f"""
            <div class="sentinel-card" style="border-color: rgba(16, 185, 129, 0.4);">
                <div style="font-size: 0.95rem; font-weight: 700; color: #10B981; margin-bottom: 12px;'>💼 Estimated Monthly Financial Protection</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: #34D399; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;">
                    ${total_savings:,.2f}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 16px;">Net capital saved per month at 86.9% ensemble recall rate</div>

                <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.8;">
                    <div>• Estimated Unmitigated Fraud Loss: <b style="color:#F87171; float:right;">${total_baseline_loss:,.2f}</b></div>
                    <div>• Direct Stolen Capital Recovered: <b style="color:#38BDF8; float:right;">+${ai_prevented_fraud_dollars:,.2f}</b></div>
                    <div>• Avoided Bank Chargeback Penalties: <b style="color:#A7F3D0; float:right;">+${ai_prevented_fines:,.2f}</b></div>
                    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 8px 0;">
                    <div>• Residual Unprevented Exposure: <b style="color:#94A3B8; float:right;">${remaining_loss:,.2f}</b></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Automated Policy Tier Matrix
    st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; margin-top: 10px;'>Automated Rule Enforcement Matrix</h4>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
            <div class="sentinel-card" style="border-top: 4px solid #10B981;">
                <span class="badge-pill badge-online">Tier 1 // Low Risk (&lt; 0.30)</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 4px 0;">Frictionless Clearance</h4>
                <div style="font-size: 0.8rem; color:#94A3B8; line-height: 1.5;">
                    Immediate settlement. Zero customer authentication challenge. Sub-5ms automated clearance.
                </div>
            </div>
            <div class="sentinel-card" style="border-top: 4px solid #F59E0B;">
                <span class="badge-pill badge-cyan">Tier 2 // Ambiguous (0.30 - {threshold:.2f})</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 4px 0;">Step-Up Verification</h4>
                <div style="font-size: 0.8rem; color:#94A3B8; line-height: 1.5;">
                    Prompt 3D-Secure 2.0 biometric OTP or push notification approval before payment confirmation.
                </div>
            </div>
            <div class="sentinel-card" style="border-top: 4px solid #EF4444;">
                <span class="badge-pill badge-danger">Tier 3 // Escalated (&gt;= {threshold:.2f})</span>
                <h4 style="color:#F8FAFC; margin: 10px 0 4px 0;">Auto-Block & Quarantine</h4>
                <div style="font-size: 0.8rem; color:#94A3B8; line-height: 1.5;">
                    Decline authorization code. Temporary card freeze. Flag transaction to Security Operations Center.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)