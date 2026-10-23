"""
Streamlit Web Application: Two-Stage Hierarchical IoT Intrusion Detection System.

Stage 1: Mamdani Fuzzy Logic Anomaly Detection (Timing Dispersion & Rate Dynamics)
Stage 2: LSTM Deep Neural Network Threat Attribution & Classification (INT8 Quantized)

Features:
- Live Interactive Attack Simulation Workshop (Deauthentication, Floods, Scans, Mirai, etc.)
- Tri-Engine Switcher: INT8 TFLite (< 0.1ms), FP32 Keras, and Pure-NumPy Random Forest
- Real-time explainability, membership curves, rule activations, and Top-5 class probabilities
- Batch CSV stream evaluation and benchmark analytics
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import config
from src.fuzzy_system import IoTFuzzyAnomalyDetector
from src.baseline import NormalBaselineBuilder, AdaptiveThresholdDetector

# Module 2 imports
from src.module2.config import module2_config
from src.module2.inference import Module2InferenceEngine
from src.module2.baseline_rf import BaselineRandomForest
from src.module2.preprocessor import Module2Preprocessor


# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="IoT Sentinel IDS | Two-Stage Anomaly & Attack Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Stage 1 Verdict Banners */
    .verdict-normal {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #ecfdf5;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #fffbeb;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.25);
    }
    .verdict-attack {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #fef2f2;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.35);
        animation: pulse 2s infinite;
    }

    /* Stage 2 Threat Attribution Card */
    .threat-card {
        background: linear-gradient(135deg, #18181b 0%, #09090b 100%);
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.25);
    }
    .threat-badge {
        display: inline-block;
        background: #dc2626;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }
    .threat-badge-zero {
        display: inline-block;
        background: #d97706;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.6); }
        100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. System Resource Initialization (Cached for performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_detection_systems():
    """Load baseline, fuzzy anomaly detector, and Module 2 inference engines."""
    # Module 1
    summary_path = config.METRICS_DIR / "summary_metrics.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
            baseline_stats = summary.get("baseline_statistics", {})
    else:
        baseline_stats = {
            "normal_mean_iat": 83625432.0,
            "normal_std_iat": 18729195.0,
            "normal_mean_rate": 1785.4,
            "normal_cv_median": 1.000,
            "normal_cv_mad": 0.105,
            "cv_percentiles": {"p25": 0.81, "p50": 1.00, "p75": 1.11, "p90": 1.36, "p95": 1.53},
            "rate_percentiles": {"p25": 68.3, "p50": 202.4, "p75": 495.0, "p90": 5465.0, "p95": 10530.0},
        }

    fuzzy_detector = IoTFuzzyAnomalyDetector(baseline_stats=baseline_stats, cfg=config)
    simple_baseline = AdaptiveThresholdDetector(baseline_stats=baseline_stats, cfg=config)
    adaptive_baseline = AdaptiveThresholdDetector(baseline_stats=baseline_stats, cfg=config)

    # Module 2 Preprocessor & Label Map
    preprocessor = Module2Preprocessor(cfg=module2_config)
    try:
        preprocessor.load_artifacts(module2_config.MODELS_DIR)
    except Exception as e:
        print(f"Preprocessor load notice: {e}")

    # Random Forest Baseline (Pure-NumPy)
    rf_baseline = None
    rf_path = module2_config.MODELS_DIR / "module2_random_forest.pkl"
    if rf_path.exists():
        try:
            import pickle
            rf_baseline = BaselineRandomForest(cfg=module2_config)
            with open(rf_path, "rb") as f:
                rf_baseline.trees = pickle.load(f)
                rf_baseline.num_classes = len(preprocessor.label_to_id) if preprocessor.label_to_id else 31
        except Exception as e:
            print(f"RF baseline load notice: {e}")

    return fuzzy_detector, baseline_stats, simple_baseline, adaptive_baseline, preprocessor, rf_baseline


(
    fuzzy_detector,
    baseline_stats,
    simple_detector,
    adaptive_detector,
    preprocessor,
    rf_baseline,
) = load_detection_systems()



# ------------------------------------------------------------------------------
# 3. Sidebar: Scenario Controls & Attack Simulation Workshop
# ------------------------------------------------------------------------------
st.sidebar.title("🛡️ Sentinel Controls")
st.sidebar.markdown("**Two-Stage Hierarchical IoT IDS**  \n*Fuzzy Sentinel + INT8 LSTM Classifier*")
st.sidebar.divider()

# Session State for interactive parameters
default_state = {
    "cv": 1.02,
    "rate": 120.0,
    "dev": 0.05,
    "iat": 0.050,
    "header_len": 40.0,
    "tot_size": 250.0,
    "proto": "TCP",
    "syn_count": 0.0,
    "ack_count": 1.0,
    "scenario_name": "Normal IoT Sensor Stream",
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_scenario(cv, rate, dev, iat, header_len, tot_size, proto, syn, ack, name):
    st.session_state.cv = cv
    st.session_state.rate = rate
    st.session_state.dev = dev
    st.session_state.iat = iat
    st.session_state.header_len = header_len
    st.session_state.tot_size = tot_size
    st.session_state.proto = proto
    st.session_state.syn_count = syn
    st.session_state.ack_count = ack
    st.session_state.scenario_name = name


st.sidebar.subheader("⚡ Quick Attack Presets")
st.sidebar.caption("Instantly load verified attack and normal profiles:")

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("🟢 Normal IoT", use_container_width=True):
        apply_scenario(1.02, 85.0, 0.04, 0.060, 40.0, 180.0, "TCP", 0.0, 1.0, "Normal IoT Sensor Telemetry")
    if st.button("📶 Deauth / Disconnect", use_container_width=True):
        apply_scenario(0.012, 9500.0, 0.95, 0.0001, 24.0, 0.0, "UDP", 0.0, 0.0, "Wi-Fi Deauthentication / Disconnect Flood")
    if st.button("👾 Mirai Botnet", use_container_width=True):
        apply_scenario(0.010, 12000.0, 0.85, 0.0001, 32.0, 128.0, "UDP", 0.0, 0.0, "Mirai Botnet UDP Flood (Zero-Day Proxy)")
    if st.button("💉 SQL Injection", use_container_width=True):
        apply_scenario(0.85, 250.0, 0.25, 0.040, 120.0, 1450.0, "TCP", 0.0, 2.0, "SQL Injection Web Exploit")

with col_p2:
    if st.button("🌊 UDP/ICMP Flood", use_container_width=True):
        apply_scenario(0.002, 22000.0, 0.92, 0.00005, 28.0, 512.0, "UDP", 0.0, 0.0, "Volumetric UDP/ICMP Flood")
    if st.button("⚡ TCP SYN Flood", use_container_width=True):
        apply_scenario(0.005, 18000.0, 0.88, 0.00008, 40.0, 0.0, "TCP", 1.0, 0.0, "DDoS TCP SYN Flood")
    if st.button("🔍 Port Scan", use_container_width=True):
        apply_scenario(2.45, 140.0, 0.72, 0.080, 40.0, 0.0, "TCP", 1.0, 0.0, "Reconnaissance Port Scan")
    if st.button("🐌 SlowLoris", use_container_width=True):
        apply_scenario(1.85, 12.0, 0.85, 1.200, 60.0, 45.0, "TCP", 0.0, 1.0, "SlowLoris Slow-Rate Exhaustion")

st.sidebar.divider()

# Engine Selector
st.sidebar.subheader("⚙️ Module 2 Engine Selector")
selected_engine = st.sidebar.radio(
    "Active Inference Runtime:",
    ["🌲 Random Forest Baseline (94.2% Acc | 0.19 ms)", "🚀 INT8 TFLite (0.087 ms)", "🧠 FP32 Keras LSTM"],
    index=0,
    help="Choose between verified high-accuracy Random Forest (94.2% test accuracy, zero DLLs) or experimental deep neural runtimes."
)


rejection_tau = st.sidebar.slider(
    "Zero-Day Uncertainty Threshold (τ)",
    min_value=0.01,
    max_value=0.30,
    value=0.064,
    step=0.005,
    help="Maximum softmax probability rejection threshold. Predictions below τ are flagged as UNKNOWN ZERO-DAY."
)

force_classify = st.sidebar.checkbox(
    "Force Stage 2 Classification (Even if Normal)",
    value=False,
    help="By default, Module 2 stays idle when Stage 1 flags Normal to save IoT power. Check this to evaluate Module 2 regardless of Stage 1 verdict."
)


# ------------------------------------------------------------------------------
# 4. Helper: Construct Synthetic 50-Observation Sequence for Inference
# ------------------------------------------------------------------------------
def generate_interactive_sequence(
    rate, iat, cv, header_len, tot_size, proto, syn, ack, dev
):
    """
    Constructs a realistic (50, 19) observation sequence based on current slider inputs.
    Introduces natural temporal jitter matching the specified Coefficient of Variation.
    """
    seq_len = 50
    feat_cols = module2_config.SELECTED_FEATURES
    n_feats = len(feat_cols)

    proto_code = 6.0 if proto == "TCP" else (17.0 if proto == "UDP" else 1.0)
    is_tcp = 1.0 if proto == "TCP" else 0.0
    is_udp = 1.0 if proto == "UDP" else 0.0

    # Build base sequence
    seq = np.zeros((seq_len, n_feats), dtype=np.float32)

    # Generate synthetic IAT with specified dispersion
    rng = np.random.RandomState(42)
    # Mean iat + gamma jitter to yield target CV
    std_target = max(1e-6, cv * iat)
    iat_vals = rng.normal(loc=iat, scale=std_target, size=seq_len)
    iat_vals = np.clip(iat_vals, 1e-6, None)

    rate_vals = rng.normal(loc=rate, scale=max(1.0, cv * rate * 0.1), size=seq_len)
    rate_vals = np.clip(rate_vals, 0.0, None)

    for i in range(seq_len):
        row_dict = {
            "IAT": iat_vals[i],
            "Rate": rate_vals[i],
            "Srate": rate_vals[i] * 0.95,
            "Drate": rate_vals[i] * 0.05,
            "flow_duration": max(0.01, seq_len * iat),
            "Duration": max(0.01, seq_len * iat),
            "Header_Length": header_len,
            "Tot size": tot_size,
            "Tot sum": tot_size * 5.0,
            "Number": 5.0,
            "AVG": tot_size / 5.0 if tot_size > 0 else 0.0,
            "Std": (tot_size / 5.0) * cv,
            "Min": max(0.0, (tot_size / 5.0) * 0.5),
            "Max": (tot_size / 5.0) * 1.5,
            "Protocol Type": proto_code,
            "TCP": is_tcp,
            "UDP": is_udp,
            "syn_count": syn,
            "ack_count": ack,
        }
        for f_idx, f_name in enumerate(feat_cols):
            seq[i, f_idx] = row_dict.get(f_name, 0.0)

    return seq


# ------------------------------------------------------------------------------
# 5. Helper: Compute Fuzzy Explainability
# ------------------------------------------------------------------------------
def compute_explainability(cv_val, rate_val, dev_val):
    """Calculate exact fuzzy membership activations and rule firing strengths."""
    u_cv = fuzzy_detector.cv.universe
    deg_cv_low = float(fuzz.interp_membership(u_cv, fuzzy_detector.cv["low"].mf, cv_val))
    deg_cv_med = float(fuzz.interp_membership(u_cv, fuzzy_detector.cv["medium"].mf, cv_val))
    deg_cv_high = float(fuzz.interp_membership(u_cv, fuzzy_detector.cv["high"].mf, cv_val))

    u_rate = fuzzy_detector.rate.universe
    deg_rate_low = float(fuzz.interp_membership(u_rate, fuzzy_detector.rate["low"].mf, rate_val))
    deg_rate_med = float(fuzz.interp_membership(u_rate, fuzzy_detector.rate["medium"].mf, rate_val))
    deg_rate_high = float(fuzz.interp_membership(u_rate, fuzzy_detector.rate["high"].mf, rate_val))

    u_dev = fuzzy_detector.iat_dev.universe
    deg_dev_low = float(fuzz.interp_membership(u_dev, fuzzy_detector.iat_dev["low"].mf, dev_val))
    deg_dev_med = float(fuzz.interp_membership(u_dev, fuzzy_detector.iat_dev["medium"].mf, dev_val))
    deg_dev_high = float(fuzz.interp_membership(u_dev, fuzzy_detector.iat_dev["high"].mf, dev_val))

    rule_activations = [
        {"id": "R1", "rule": "IF CV is MED and Rate is LOW and Dev is LOW THEN Risk is LOW", "strength": min(deg_cv_med, deg_rate_low, deg_dev_low), "consequent": "LOW"},
        {"id": "R2", "rule": "IF CV is LOW and Rate is LOW and Dev is LOW THEN Risk is LOW", "strength": min(deg_cv_low, deg_rate_low, deg_dev_low), "consequent": "LOW"},
        {"id": "R3", "rule": "IF CV is HIGH and Rate is HIGH THEN Risk is HIGH (Burst Flood)", "strength": min(deg_cv_high, deg_rate_high), "consequent": "HIGH"},
        {"id": "R4", "rule": "IF CV is HIGH and Dev is HIGH THEN Risk is HIGH (Erratic Scan)", "strength": min(deg_cv_high, deg_dev_high), "consequent": "HIGH"},
        {"id": "R5", "rule": "IF Rate is HIGH and Dev is HIGH THEN Risk is HIGH (Volumetric Assault)", "strength": min(deg_rate_high, deg_dev_high), "consequent": "HIGH"},
        {"id": "R6", "rule": "IF CV is MED and Rate is HIGH THEN Risk is HIGH", "strength": min(deg_cv_med, deg_rate_high), "consequent": "HIGH"},
        {"id": "R7", "rule": "IF CV is HIGH and Rate is MED THEN Risk is HIGH", "strength": min(deg_cv_high, deg_rate_med), "consequent": "HIGH"},
        {"id": "R8", "rule": "IF CV is MED and Rate is MED and Dev is MED THEN Risk is MEDIUM", "strength": min(deg_cv_med, deg_rate_med, deg_dev_med), "consequent": "MEDIUM"},
        {"id": "R9", "rule": "IF Rate is HIGH THEN Risk is HIGH (Volumetric Override)", "strength": deg_rate_high, "consequent": "HIGH"},
        {"id": "R10", "rule": "IF Dev is HIGH THEN Risk is HIGH (Timing Collapse)", "strength": deg_dev_high, "consequent": "HIGH"},
        {"id": "R11", "rule": "IF CV is LOW and Rate is HIGH THEN Risk is HIGH (Constant-Rate Flood / Deauth)", "strength": min(deg_cv_low, deg_rate_high), "consequent": "HIGH"},
        {"id": "R12", "rule": "IF CV is HIGH and Rate is LOW THEN Risk is MEDIUM (Slow Recon)", "strength": min(deg_cv_high, deg_rate_low), "consequent": "MEDIUM"},
    ]

    memberships = {
        "cv": {"low": deg_cv_low, "medium": deg_cv_med, "high": deg_cv_high},
        "rate": {"low": deg_rate_low, "medium": deg_rate_med, "high": deg_rate_high},
        "dev": {"low": deg_dev_low, "medium": deg_dev_med, "high": deg_dev_high},
    }
    return memberships, rule_activations


# ------------------------------------------------------------------------------
# 6. Main Dashboard Tabs
# ------------------------------------------------------------------------------
st.title("🛡️ IoT Sentinel IDS | Two-Stage Threat Intelligence")
st.markdown(
    "### Hierarchical Defense: Mamdani Fuzzy Sentinel → INT8 Quantized LSTM Classifier  \n"
    "*Research Benchmark on CICIoT2023 for Zero-Day Anomaly Detection & Multiclass Threat Attribution*"
)

tab_realtime, tab_simulator_guide, tab_batch, tab_research = st.tabs([
    "🎛️ Real-Time Detection & Attribution",
    "📖 Attack Simulation Guide",
    "📊 Batch Evaluation & Stream Inspector",
    "🔬 Benchmark Analytics & Methodology"
])


# ==============================================================================
# TAB 1: Real-Time Detection & Hierarchical Inference
# ==============================================================================
with tab_realtime:
    st.info(f"📍 Active Profile: **{st.session_state.scenario_name}**")

    # Interactive Sliders Section
    st.markdown("#### 1. Interactive Flow Feature Controls")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        cv_in = st.slider(
            "CV_IAT (σ / μ)",
            min_value=0.000,
            max_value=4.000,
            value=float(st.session_state.cv),
            step=0.01,
            help="Coefficient of Variation: Normal IoT is ~1.0. Floods collapse CV < 0.05. Port scans surge CV > 1.5."
        )
        iat_in = st.number_input("Inter-Arrival Time (IAT in sec)", min_value=1e-6, max_value=5.0, value=float(st.session_state.iat), format="%.4f")

    with col_s2:
        rate_in = st.slider(
            "Packet Rate (pkts/s)",
            min_value=0.0,
            max_value=30000.0,
            value=float(st.session_state.rate),
            step=50.0,
            help="Average arrival rate. Normal IoT is 20-300 pkts/s. Floods reach thousands."
        )
        dev_in = st.slider("IAT Relative Shift (D)", min_value=0.00, max_value=3.00, value=float(st.session_state.dev), step=0.02)

    with col_s3:
        header_len_in = st.slider("Header Length (bytes)", min_value=20, max_value=128, value=int(st.session_state.header_len), step=4)
        tot_size_in = st.slider("Total Payload Size", min_value=0, max_value=2000, value=int(st.session_state.tot_size), step=20)

    with col_s4:
        proto_in = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"], index=["TCP", "UDP", "ICMP"].index(st.session_state.proto))
        col_fl1, col_fl2 = st.columns(2)
        with col_fl1:
            syn_in = st.number_input("SYN Count", min_value=0.0, max_value=5.0, value=float(st.session_state.syn_count), step=1.0)
        with col_fl2:
            ack_in = st.number_input("ACK Count", min_value=0.0, max_value=5.0, value=float(st.session_state.ack_count), step=1.0)

    # Update session state with current slider values
    st.session_state.cv = cv_in
    st.session_state.rate = rate_in
    st.session_state.dev = dev_in
    st.session_state.iat = iat_in
    st.session_state.header_len = header_len_in
    st.session_state.tot_size = tot_size_in
    st.session_state.proto = proto_in
    st.session_state.syn_count = syn_in
    st.session_state.ack_count = ack_in

    st.divider()

    # --------------------------------------------------------------------------
    # STAGE 1: Fuzzy Anomaly Detection Computation
    # --------------------------------------------------------------------------
    st.markdown("### 🔍 Stage 1: Fuzzy Logic Anomaly Detection")
    t0_m1 = time.perf_counter()
    risk_score = fuzzy_detector.evaluate_window(cv_in, rate_in, dev_in)
    m1_latency_ms = (time.perf_counter() - t0_m1) * 1000.0
    memberships, rules_fired = compute_explainability(cv_in, rate_in, dev_in)

    low_thresh = float(config.FUZZY_LOW_THRESHOLD)
    high_thresh = float(config.FUZZY_HIGH_THRESHOLD)

    if risk_score < low_thresh:
        verdict = "NORMAL TRAFFIC"
        verdict_class = "verdict-normal"
        verdict_icon = "🟢"
        verdict_desc = "Traffic dynamics adhere to expected normal IoT sensor baseline. Low attack likelihood."
        is_anomalous = False
    elif risk_score >= high_thresh:
        verdict = "ATTACK DETECTED"
        verdict_class = "verdict-attack"
        verdict_icon = "🔴"
        verdict_desc = "High-severity anomaly detected! Severe timing dispersion and rate surge match malicious assault."
        is_anomalous = True
    else:
        verdict = "SUSPICIOUS ACTIVITY"
        verdict_class = "verdict-suspicious"
        verdict_icon = "🟡"
        verdict_desc = "Borderline traffic dynamics detected. Timing shift or rate elevation warrants second-stage classification."
        is_anomalous = True

    col_out_left, col_out_right = st.columns([1.2, 1])

    with col_out_left:
        st.markdown(f"""
        <div class="{verdict_class}">
            <h1 style="margin:0; font-size: 2.2rem;">{verdict_icon} {verdict}</h1>
            <p style="font-size: 1.05rem; margin-top: 8px; opacity: 0.95;">{verdict_desc}</p>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 14px 0;">
            <div style="font-size: 1.05rem;">
                Mamdani Defuzzified Risk: <strong style="font-size: 1.4rem;">{risk_score:.4f}</strong> / 1.0000 
                &nbsp;|&nbsp; Latency: <strong>{m1_latency_ms:.3f} ms</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Continuous Attack Risk Spectrum:")
        st.progress(float(np.clip(risk_score, 0.0, 1.0)))

    with col_out_right:
        st.markdown("##### Tri-Method Verdict Comparison")
        simple_flag = cv_in > baseline_stats["cv_percentiles"]["p95"]
        adaptive_limit = baseline_stats["normal_cv_median"] + (config.MAD_K * baseline_stats["normal_cv_mad"])
        adaptive_flag = cv_in > adaptive_limit

        m1_badge = "🔴 Attack" if simple_flag else "🟢 Normal"
        m2_badge = "🔴 Attack" if adaptive_flag else "🟢 Normal"
        m3_badge = f"{verdict_icon} {verdict.split()[0]}"

        st.markdown(f"""
        | Methodology | Verdict | Detection Mechanism |
        |---|---|---|
        | **Simple CV** | **{m1_badge}** | `CV > {baseline_stats['cv_percentiles']['p95']:.2f}` (Misses low-CV floods) |
        | **Adaptive MAD** | **{m2_badge}** | `CV > {adaptive_limit:.2f}` (Upper-tail only) |
        | **Mamdani Fuzzy** | **{m3_badge}** | **Multi-feature Fusion (Risk: {risk_score:.3f})** |
        """)

    # --------------------------------------------------------------------------
    # STAGE 2: Module 2 Attack Classification (Triggered on Anomaly or Force)
    # --------------------------------------------------------------------------
    st.divider()
    st.markdown("### ⚡ Stage 2: Deep Threat Attribution & Classification (Module 2)")

    should_run_m2 = is_anomalous or force_classify

    if not should_run_m2:
        st.success("💤 **Module 2 is currently IDLE (Power-Saving Mode)**  \nStage 1 verified traffic as normal. The deep learning classifier remains uninvoked to conserve IoT compute and battery budget. To force classification, check 'Force Stage 2 Classification' in the sidebar.")
    else:
        # Generate 50-observation sequence corresponding to current slider state
        seq_50_raw = generate_interactive_sequence(
            rate=rate_in, iat=iat_in, cv=cv_in, header_len=header_len_in,
            tot_size=tot_size_in, proto=proto_in, syn=syn_in, ack=ack_in, dev=dev_in
        )

        # Preprocess sequence array
        seq_scaled = preprocessor.transform_sequence_array(np.expand_dims(seq_50_raw, axis=0))

        if "Random Forest" in selected_engine or (rf_baseline is not None and "INT8" not in selected_engine and "FP32" not in selected_engine):
            # Pure-NumPy Random Forest Baseline (94.2% Test Accuracy, zero DLL dependencies)
            t0_rf = time.perf_counter()
            X_rf = BaselineRandomForest.extract_rf_features(seq_scaled, mode="last_and_agg")
            probs_rf = rf_baseline.predict_proba(X_rf)[0]
            lat_m2 = (time.perf_counter() - t0_rf) * 1000.0

            pred_id = int(np.argmax(probs_rf))
            conf = float(probs_rf[pred_id])
            pred_class = preprocessor.id_to_label.get(pred_id, f"Class_{pred_id}")
            is_zero_day = conf < rejection_tau
            probs_list = probs_rf.tolist()
            runtime_name = "Pure-NumPy Random Forest"
            model_size = "2.68 MB"
        elif "INT8" in selected_engine:
            try:
                eng = Module2InferenceEngine(use_quantized=True, cfg=module2_config)
                res_m2 = eng.predict_single_sequence(seq_scaled[0])
                lat_m2 = res_m2["latency_ms"]
                pred_class = res_m2["raw_class"]
                conf = res_m2["confidence"]
                is_zero_day = conf < rejection_tau
                probs_list = res_m2["probabilities"]
                runtime_name = res_m2["runtime"]
                model_size = "0.31 MB"
            except Exception as e:
                st.warning(f"INT8 runtime notice ({e}). Running Random Forest baseline.")
                t0_rf = time.perf_counter()
                X_rf = BaselineRandomForest.extract_rf_features(seq_scaled, mode="last_and_agg")
                probs_rf = rf_baseline.predict_proba(X_rf)[0]
                lat_m2 = (time.perf_counter() - t0_rf) * 1000.0
                pred_id = int(np.argmax(probs_rf))
                conf = float(probs_rf[pred_id])
                pred_class = preprocessor.id_to_label.get(pred_id, f"Class_{pred_id}")
                is_zero_day = conf < rejection_tau
                probs_list = probs_rf.tolist()
                runtime_name = "Random Forest (Fallback)"
                model_size = "2.68 MB"
        elif "FP32" in selected_engine:
            try:
                eng = Module2InferenceEngine(use_quantized=False, cfg=module2_config)
                res_m2 = eng.predict_single_sequence(seq_scaled[0])
                lat_m2 = res_m2["latency_ms"]
                pred_class = res_m2["raw_class"]
                conf = res_m2["confidence"]
                is_zero_day = conf < rejection_tau
                probs_list = res_m2["probabilities"]
                runtime_name = res_m2["runtime"]
                model_size = "0.31 MB"
            except Exception as e:
                st.warning(f"FP32 runtime notice ({e}). Running Random Forest baseline.")
                t0_rf = time.perf_counter()
                X_rf = BaselineRandomForest.extract_rf_features(seq_scaled, mode="last_and_agg")
                probs_rf = rf_baseline.predict_proba(X_rf)[0]
                lat_m2 = (time.perf_counter() - t0_rf) * 1000.0
                pred_id = int(np.argmax(probs_rf))
                conf = float(probs_rf[pred_id])
                pred_class = preprocessor.id_to_label.get(pred_id, f"Class_{pred_id}")
                is_zero_day = conf < rejection_tau
                probs_list = probs_rf.tolist()
                runtime_name = "Random Forest (Fallback)"
                model_size = "2.68 MB"
        else:
            st.error("No active model available.")
            pred_class = "Unknown"
            conf = 0.0
            is_zero_day = False
            lat_m2 = 0.0
            probs_list = [0.0] * 31
            runtime_name = "None"
            model_size = "0 MB"

        # Threat Attribution Card
        badge_class = "threat-badge-zero" if is_zero_day else "threat-badge"
        threat_title = "🚨 POTENTIAL ZERO-DAY / UNKNOWN EXPLOIT" if is_zero_day else f"🎯 ATTACK IDENTIFIED: {pred_class}"

        col_t1, col_t2 = st.columns([1.3, 1])

        with col_t1:
            st.markdown(f"""
            <div class="threat-card">
                <span style="color: #60a5fa; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">
                    Module 2 Classification Result ({runtime_name})
                </span>
                <div class="{badge_class}">{threat_title}</div>
                <div style="margin-top: 14px; color: #e2e8f0; font-size: 1.05rem;">
                    Classification Confidence: <strong style="color: #60a5fa; font-size: 1.25rem;">{conf * 100:.1f}%</strong>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 6px;">
                    Inference Latency: <strong style="color: #34d399;">{lat_m2:.3f} ms</strong> | Model Size: <strong>{model_size}</strong> | Sequence: 50 observations
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_zero_day:
                st.warning(f"🛡️ **Zero-Day Rejection Policy Triggered**: Softmax confidence ({conf:.3f}) fell below rejection threshold τ = {rejection_tau:.3f}. System classified this as an uncataloged / novel zero-day exploit!")

        with col_t2:
            st.markdown("##### Top-5 Candidate Attack Probabilities")
            # Build top 5 chart
            labels_map = preprocessor.id_to_label if preprocessor else {}
            prob_pairs = []
            for idx, p in enumerate(probs_list):
                name = labels_map.get(idx, f"Class_{idx}")
                prob_pairs.append((name, p))
            
            prob_pairs.sort(key=lambda x: x[1], reverse=True)
            top5 = prob_pairs[:5]


            for cat_name, prob_val in top5:
                pct = prob_val * 100
                col_bar_l, col_bar_r = st.columns([2.5, 1])
                with col_bar_l:
                    st.markdown(f"**{cat_name}**")
                    st.progress(float(np.clip(prob_val, 0.0, 1.0)))
                with col_bar_r:
                    st.markdown(f"<div style='margin-top: 18px; font-weight: 700; color: #60a5fa;'>{pct:.1f}%</div>", unsafe_allow_html=True)


    # --------------------------------------------------------------------------
    # Explainability: Triggered Rules & Membership Curves
    # --------------------------------------------------------------------------
    st.divider()
    with st.expander("🧠 Inspect Stage 1 Fuzzy Explainability & Active Rules", expanded=False):
        col_exp1, col_exp2 = st.columns([1, 1.2])
        with col_exp1:
            st.markdown("#### Input Linguistic Memberships (μ)")
            mem_df = pd.DataFrame([
                {"Feature": "CV (Dispersion)", "LOW": memberships["cv"]["low"], "MEDIUM": memberships["cv"]["medium"], "HIGH": memberships["cv"]["high"]},
                {"Feature": "Packet Rate", "LOW": memberships["rate"]["low"], "MEDIUM": memberships["rate"]["medium"], "HIGH": memberships["rate"]["high"]},
                {"Feature": "IAT Deviation", "LOW": memberships["dev"]["low"], "MEDIUM": memberships["dev"]["medium"], "HIGH": memberships["dev"]["high"]},
            ])
            st.dataframe(mem_df.style.format({"LOW": "{:.3f}", "MEDIUM": "{:.3f}", "HIGH": "{:.3f}"}), use_container_width=True)

        with col_exp2:
            st.markdown("#### Triggered Mamdani Rules")
            active_rules = [r for r in rules_fired if r["strength"] > 0.001]
            active_rules = sorted(active_rules, key=lambda x: x["strength"], reverse=True)
            if active_rules:
                st.dataframe(pd.DataFrame(active_rules)[["id", "rule", "strength", "consequent"]].rename(columns={"id": "ID", "rule": "Rule", "strength": "Firing Strength (w)"}), use_container_width=True)
            else:
                st.write("No active rules fired above threshold.")


# ==============================================================================
# TAB 2: Attack Simulation Guide
# ==============================================================================
with tab_simulator_guide:
    st.header("📖 How to Simulate Specific IoT Network Attacks")
    st.markdown(
        "In tabular and network flow anomaly detection, each cyber attack leaves distinct **timing, volumetric, and protocol fingerprints**."
    )

    st.markdown("""
    ---
    ### 📶 1. Wi-Fi Deauthentication / Disconnection Attack
    * **Physical Mechanism:**  
      An adversary injects 802.11 management frames (subtype `0x000c` / Deauth) with a spoofed Access Point MAC address. IoT sensor stations receive these frames and are abruptly disconnected from the network, causing Denial of Service.
    * **How to Simulate in the Controls:**
      1. **Packet Rate:** Surge to **`8,000 – 15,000 pkts/s`** (flooding tools like `aireplay-ng` blast frames continuously).
      2. **Inter-Arrival Time (IAT):** Drops to microseconds **`< 0.0002 s`** (consecutive packets arrive back-to-back).
      3. **Coefficient of Variation (CV):** Drops to **`< 0.02`** (extremely uniform automated clock pacing; natural stochastic idle gaps vanish).
      4. **Header Length:** Small fixed header **`24 – 32 bytes`**.
      5. **Payload (Tot size):** Minimal or zero **`0 bytes`** (management frames carry no user payload).
      6. **Expected Pipeline Response:** Stage 1 fires Rule 11 (`LOW CV` ∧ `HIGH Rate` → `Risk = HIGH`). Stage 2 attributes it to a high-rate flood class!

    ---
    ### 🌊 2. Volumetric Floods (DDoS UDP / ICMP Flood)
    * **Physical Mechanism:**  
      Pumping massive volumes of UDP datagrams or ICMP Echo requests to congest the gateway's network buffer.
    * **How to Simulate in the Controls:**
      * **Rate:** **`> 15,000 pkts/s`** (high volumetric assault).
      * **CV:** **`< 0.005`** (near-zero timing dispersion).
      * **IAT:** **`< 0.0001 s`**.
      * **Protocol:** `UDP` or `ICMP`.
      * **Payload:** Large payloads (`Tot size > 500 bytes`).

    ---
    ### ⚡ 3. TCP SYN / ACK Floods
    * **Physical Mechanism:**  
      Exploiting the TCP 3-way handshake by flooding `SYN` packets without sending final `ACK` packets, exhausting the target's transmission control block (TCB) table.
    * **How to Simulate in the Controls:**
      * **Protocol:** `TCP`.
      * **SYN Count:** Set to **`1.0 – 5.0`** (abnormally high SYN ratio).
      * **ACK Count:** Set to **`0.0`** (handshakes never complete).
      * **Rate:** **`> 10,000 pkts/s`**.
      * **CV:** **`< 0.01`**.

    ---
    ### 🔍 4. Reconnaissance Port Scans & Host Discovery
    * **Physical Mechanism:**  
      Probing sequential ports across monitored IP addresses. Scanners introduce random delays to evade naive rate limiters.
    * **How to Simulate in the Controls:**
      * **Rate:** Low to moderate **`100 – 300 pkts/s`** (stealthy).
      * **CV (Dispersion):** Surges to **`> 2.0`** (sporadic probes with erratic delays).
      * **IAT Deviation:** Elevated **`D > 0.50`** (timing structure diverges from steady sensor periods).
      * **Expected Pipeline Response:** Stage 1 fires Rule 12 (`HIGH CV` ∧ `LOW Rate` → `SUSPICIOUS`). Stage 2 classifies as `Recon-PortScan`.

    ---
    ### 🐌 5. SlowLoris Slow-Rate HTTP Exhaustion
    * **Physical Mechanism:**  
      Sending partial HTTP request headers at extremely slow intervals to hold server threads open indefinitely.
    * **How to Simulate in the Controls:**
      * **Rate:** Minimal **`5 – 20 pkts/s`** (extremely slow).
      * **IAT:** Large intervals **`> 1.0 s`**.
      * **Deviation (D):** High structural shift **`D > 0.8`**.
      * **Payload:** Tiny fragments **`40 – 60 bytes`**.
    """)


# ==============================================================================
# TAB 3: Batch Evaluation & CSV Stream Inspector
# ==============================================================================
with tab_batch:
    st.subheader("📊 Batch Window Evaluation & Stream Inspector")
    st.markdown("Upload a CSV file containing sliding window records, or evaluate the pipeline's generated `results/window_features.csv`:")

    col_b1, col_b2 = st.columns([1.5, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload Window Features CSV", type=["csv"])
    with col_b2:
        load_existing = st.button("📁 Load Pipeline's window_features.csv", use_container_width=True)

    df_batch = None
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(df_batch):,} window feature records.")
    elif load_existing or (config.WINDOW_FEATURES_PATH.exists()):
        if config.WINDOW_FEATURES_PATH.exists():
            df_batch = pd.read_csv(config.WINDOW_FEATURES_PATH)
            st.info(f"Loaded existing window features: {len(df_batch):,} windows from `{config.WINDOW_FEATURES_PATH}`.")

    if df_batch is not None:
        req_cols = ["cv_iat", "mean_rate", "iat_deviation"]
        missing = [c for c in req_cols if c not in df_batch.columns]
        
        if missing:
            st.error(f"Missing required columns in CSV: {missing}. Expected columns: {req_cols}")
        else:
            with st.spinner("Executing Hierarchical Detection across batch windows..."):
                sample_limit = min(1500, len(df_batch))
                sub_batch = df_batch.iloc[:sample_limit].copy()
                scores, cats, preds = fuzzy_detector.predict_batch(sub_batch)
                sub_batch["fuzzy_risk"] = scores
                sub_batch["fuzzy_verdict"] = cats
                sub_batch["is_anomalous"] = preds

            st.markdown(f"#### Classification Distribution (First {sample_limit:,} Windows)")
            col_d1, col_d2, col_d3 = st.columns(3)
            counts = sub_batch["fuzzy_verdict"].value_counts()
            
            with col_d1:
                st.metric("Normal Windows", counts.get("NORMAL", 0), f"{(counts.get('NORMAL', 0)/len(sub_batch))*100:.1f}%")
            with col_d2:
                st.metric("Suspicious Windows", counts.get("SUSPICIOUS", 0), f"{(counts.get('SUSPICIOUS', 0)/len(sub_batch))*100:.1f}%")
            with col_d3:
                st.metric("Attack Windows", counts.get("ATTACK", 0), f"{(counts.get('ATTACK', 0)/len(sub_batch))*100:.1f}%")

            # Timeline Plot of Fuzzy Risk
            fig_t, ax_t = plt.subplots(figsize=(14, 4.2))
            ax_t.axhspan(0.0, config.FUZZY_LOW_THRESHOLD, color="#d4edda", alpha=0.5, label="Normal Zone")
            ax_t.axhspan(config.FUZZY_LOW_THRESHOLD, config.FUZZY_HIGH_THRESHOLD, color="#fff3cd", alpha=0.5, label="Suspicious Zone")
            ax_t.axhspan(config.FUZZY_HIGH_THRESHOLD, 1.0, color="#f8d7da", alpha=0.5, label="Attack Zone")
            
            w_idx = sub_batch["window_id"] if "window_id" in sub_batch.columns else range(len(sub_batch))
            ax_t.plot(w_idx, sub_batch["fuzzy_risk"], color="#343a40", lw=1.5, label="Stage 1 Attack Risk")
            
            ax_t.set_title("Stage 1 Risk Trajectory across Sequential Windows")
            ax_t.set_xlabel("Window ID")
            ax_t.set_ylabel("Risk Score [0, 1]")
            ax_t.set_ylim(-0.05, 1.05)
            ax_t.legend(loc="upper right")
            fig_t.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

            st.dataframe(sub_batch[["window_id", "cv_iat", "mean_rate", "iat_deviation", "fuzzy_risk", "fuzzy_verdict"]].head(50), use_container_width=True)


# ==============================================================================
# TAB 4: Research Benchmark & Visualizations
# ==============================================================================
with tab_research:
    st.subheader("🔬 Benchmark Evaluation & Comparative Results")
    st.markdown(
        "Comprehensive empirical comparison of **Module 1 (Mamdani Fuzzy Logic)** and **Module 2 (FP32 & INT8 LSTM + Random Forest)**."
    )

    # 1. Module 2 Metrics Comparison Table
    m2_metrics_path = module2_config.METRICS_PATH
    if m2_metrics_path.exists():
        with open(m2_metrics_path, "r") as f:
            m2_data = json.load(f)
        st.markdown("#### Module 2: Model Comparison Table")
        st.dataframe(pd.DataFrame(m2_data["model_comparison"]), use_container_width=True)

    # 2. Display Generated Research Plots
    st.divider()
    st.subheader("🖼️ Generated Research Visualizations Gallery")
    plot_files = sorted(list(module2_config.PLOTS_DIR.glob("*.png")))
    
    if plot_files:
        for i in range(0, len(plot_files), 2):
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(str(plot_files[i]), caption=plot_files[i].stem.replace("_", " ").title(), use_column_width=True)
            if i + 1 < len(plot_files):
                with col_img2:
                    st.image(str(plot_files[i+1]), caption=plot_files[i+1].stem.replace("_", " ").title(), use_column_width=True)
    else:
        st.info("Run `python src/module2/train.py` to generate the diagnostic research plots.")
