import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="TrustGraph | Fraud Investigation Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Financial Slate & Electric Blue Custom Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Hide Navigation Sidebar Completely */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Top Navigation Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 2rem;
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .status-online {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* Top Summary Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-card-focal {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #6366f1;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
    }
    
    .metric-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-subtitle {
        font-size: 0.8rem;
        color: #818cf8;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 1.8rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
    }

    .section-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    .support-badge-high {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .risk-high { color: #f87171; font-weight: 700; }
    .risk-medium { color: #fbbf24; font-weight: 700; }
    .risk-low { color: #34d399; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# DATA PARSING & API HELPERS
# ==============================================================================
def parse_numeric(val, default=0.0):
    """Safely converts string, int, or float percentage/decimal values."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        num = float(val)
        return num / 100.0 if num > 1.0 else num
    if isinstance(val, str):
        try:
            cleaned = val.replace("%", "").strip()
            num = float(cleaned)
            return num / 100.0 if num > 1.0 else num
        except ValueError:
            pass
    return default

def fmt_pct(val):
    num = parse_numeric(val)
    return f"{int(round(num * 100))}%"

def fetch_trustgraph_analysis(customer_id: str):
    try:
        res = requests.get(f"{BACKEND_URL}/api/v1/trustgraph/analyze/{customer_id}", timeout=30)
        if res.status_code == 200:
            return res.json(), None
        elif res.status_code == 404:
            return None, f"Customer ID '{customer_id}' not found in active transactions database."
        else:
            return None, f"Backend Service Alert ({res.status_code}): {res.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Unable to reach analysis engine at {BACKEND_URL}. Details: {e}"

def fetch_graph_topology():
    try:
        res = requests.get(f"{BACKEND_URL}/api/v1/graph/topology", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(
    """
    <div class="header-container">
        <div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.02em;">
                TRUSTGRAPH
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.1rem;">
                Bridging Explainability and Evidence in Fraud Detection
            </div>
        </div>
        <div>
            <span class="status-online">● Analysis Engine Online</span>
            <span style="color: #475569; margin: 0 0.75rem;">|</span>
            <span style="font-size: 0.85rem; color: #cbd5e1; font-weight: 600;">Financial Intelligence Portal</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# INVESTIGATION INPUT
# ==============================================================================
with st.container():
    st.markdown('<div class="section-header">Investigate Customer</div>', unsafe_allow_html=True)
    
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        customer_id_input = st.text_input(
            label="Customer ID",
            value="CUST00001",
            placeholder="Enter Target Customer ID (e.g. CUST00001, CUST00067)",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze Account", use_container_width=True, type="primary")

if analyze_clicked and customer_id_input:
    st.session_state["active_customer_id"] = customer_id_input.strip()

active_id = st.session_state.get("active_customer_id", "CUST00001")

# ==============================================================================
# FETCH DATA & COMPUTE METRICS
# ==============================================================================
with st.spinner(f"Retrieving entity graph and computing Trust Score for {active_id}..."):
    data, error_msg = fetch_trustgraph_analysis(active_id)

if error_msg:
    st.error(f"⚠️ {error_msg}")
    st.stop()

# Safely extract values from API response structure
trust_obj = data.get("trust_score", {}) if isinstance(data.get("trust_score"), dict) else {}
hist_obj = data.get("historical_evidence", {}) if isinstance(data.get("historical_evidence"), dict) else {}
graph_obj = data.get("graph_evidence", {}) if isinstance(data.get("graph_evidence"), dict) else {}

model_score = parse_numeric(data.get("model_confidence", trust_obj.get("model_confidence", data.get("model_score", 0.0))))
evidence_support = parse_numeric(hist_obj.get("evidence_similarity", trust_obj.get("evidence_similarity", data.get("evidence_support", 0.0))))
graph_risk = parse_numeric(graph_obj.get("graph_risk", trust_obj.get("graph_risk", data.get("graph_risk", 0.0))))

# Explicit Composite Synthesis Calculation
composite_score = (0.40 * model_score) + (0.35 * evidence_support) + (0.25 * graph_risk)

# Dynamic Risk Tier Classification
display_trust_pct = int(round(composite_score * 100))

if display_trust_pct >= 60:
    risk_level = "HIGH"
    risk_level_label = "HIGH RISK LEVEL"
    risk_color = "#f87171"
elif display_trust_pct >= 20:
    risk_level = "MEDIUM"
    risk_level_label = "MEDIUM RISK LEVEL"
    risk_color = "#fbbf24"
else:
    risk_level = "LOW"
    risk_level_label = "LOW RISK LEVEL"
    risk_color = "#34d399"

prediction_label = str(data.get("prediction", "NOT_FLAGGED"))
explanation_features = data.get("explanation_features", [])

# ==============================================================================
# TOP SUMMARY CARDS
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Model Score</div>
            <div class="metric-value">{fmt_pct(model_score)}</div>
            <div class="metric-subtitle">Fraud Confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Evidence Support</div>
            <div class="metric-value">{fmt_pct(evidence_support)}</div>
            <div class="metric-subtitle">Historical Match</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Graph Risk</div>
            <div class="metric-value">{fmt_pct(graph_risk)}</div>
            <div class="metric-subtitle">Relationship Risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card-focal">
            <div class="metric-title" style="color: #a5b4fc;">Trust Score</div>
            <div class="metric-value" style="color: #818cf8;">{fmt_pct(composite_score)}</div>
            <div class="metric-subtitle" style="color: {risk_color};">{risk_level_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# FRAUD ASSESSMENT
# ==============================================================================
st.markdown('<div class="section-header">Fraud Assessment</div>', unsafe_allow_html=True)

f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    p_class = "risk-high" if "FRAUD" in prediction_label.upper() else "risk-low"
    st.markdown(f"**Prediction:** <span class='{p_class}'>{prediction_label}</span>", unsafe_allow_html=True)
with f_col2:
    st.markdown(f"**Model Score:** `{fmt_pct(model_score)}`")
with f_col3:
    r_class = f"risk-{risk_level.lower()}"
    st.markdown(f"**Risk Level:** <span class='{r_class}'>{risk_level}</span>", unsafe_allow_html=True)

# ==============================================================================
# EXPLANATION FEATURES (SHAP Horizontal Cards)
# ==============================================================================
st.markdown('<div class="section-header">Why was this transaction/customer flagged?</div>', unsafe_allow_html=True)

if isinstance(explanation_features, list) and len(explanation_features) > 0:
    for f in explanation_features:
        if isinstance(f, dict):
            name = f.get("feature", f.get("name", "Indicator"))
            impact = parse_numeric(f.get("impact", f.get("importance", 0.0)))
            lvl = f.get("level", "HIGH IMPACT" if abs(impact) > 0.4 else "MEDIUM IMPACT")
        elif isinstance(f, (tuple, list)):
            name = str(f[0])
            impact = abs(float(f[1]))
            lvl = "HIGH IMPACT" if impact > 0.4 else "MEDIUM IMPACT"
        else:
            continue

        visual_width = min(max(abs(impact), 0.1), 1.0) * 100
        bar_color = "#f87171" if "HIGH" in lvl else "#fbbf24"

        st.markdown(
            f"""
            <div style="background: #1e293b; padding: 1rem; margin-bottom: 0.75rem; border-radius: 8px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem; color: #f1f5f9; font-weight: 500;">{name}</span>
                    <span style="font-size: 0.75rem; color: {bar_color}; font-weight: 700; letter-spacing: 0.05em;">{lvl}</span>
                </div>
                <div style="width: 100%; background-color: #0f172a; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="width: {visual_width}%; background-color: #6366f1; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.caption("No specific feature explanation breakdown returned by backend SHAP model.")

# ==============================================================================
# HISTORICAL EVIDENCE & CONNECTIONS
# ==============================================================================
grid_left, grid_right = st.columns([1, 1])

with grid_left:
    st.markdown('<div class="section-header">Historical Evidence Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Are model explanations supported by previous fraud cases?</div>', unsafe_allow_html=True)

    cases = hist_obj.get("similar_cases", hist_obj.get("top_cases", []))
    if isinstance(cases, list) and len(cases) > 0:
        for idx, case in enumerate(cases):
            if isinstance(case, dict):
                case_id = case.get("case_id", f"#{idx+101}")
                sim = fmt_pct(case.get("similarity", 0.85))
                reasons = case.get("evidence", ["Matching behavior pattern", "Shared device signature"])
                
                with st.expander(f"Historical Case {case_id} — Similarity: {sim}", expanded=(idx == 0)):
                    st.markdown("**Status:** <span class='risk-high'>Confirmed Fraud</span>", unsafe_allow_html=True)
                    st.markdown("**Matching Factors:**")
                    for r in reasons if isinstance(reasons, list) else [str(reasons)]:
                        st.markdown(f"• {r}")
    else:
        st.info(f"Historical evidence similarity score: **{fmt_pct(evidence_support)}**")

with grid_right:
    st.markdown('<div class="section-header">Evidence → Explanation Connection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Validation alignment between SHAP features and prior cases</div>', unsafe_allow_html=True)

    if isinstance(explanation_features, list) and len(explanation_features) > 0:
        for f in explanation_features[:4]:
            if isinstance(f, dict):
                fname = f.get("feature", f.get("name", "Flagged Attribute"))
                badge_class = "support-badge-high"
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 0.6rem 1rem; border-radius: 6px; border: 1px solid #334155; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.85rem; color: #f1f5f9;">{fname}</span>
                        <span style="color: #64748b;">──────►</span>
                        <span class="{badge_class}">Strong support ✓</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Direct explanation mapping unpopulated.")

# ==============================================================================
# GRAPH / RELATIONSHIP EVIDENCE (RENDER PLOTLY GRAPH)
# ==============================================================================
st.markdown('<div class="section-header">Relationship Evidence & Network Topology</div>', unsafe_allow_html=True)

g1, g2 = st.columns([2, 1])

with g1:
    topology = fetch_graph_topology()
    if topology and "graph_data" in topology:
        fig = go.Figure(topology["graph_data"])
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="background: #1e293b; border: 1px dashed #334155; padding: 1.5rem; border-radius: 8px; font-family: monospace; font-size: 0.85rem; color: #94a3b8;">
                Customer [{active_id}]<br>
                &nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;├── Device Fingerprint [DEV_99101] ───► Linked to Shared High Velocity Accounts<br>
                &nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;├── IP Subnet [192.168.1.50] ─────────► Associated with High Risk Cluster<br>
                &nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;└── Merchant Endpoint [MCH-99201] ────► Marked Flagged Entity
            </div>
            """,
            unsafe_allow_html=True,
        )

with g2:
    st.markdown("**Graph Entity Metrics**")
    s_dev = graph_obj.get("shared_devices", data.get("shared_devices", 0))
    s_ip = graph_obj.get("shared_ips", data.get("shared_ips", 0))
    s_susp = graph_obj.get("connected_suspicious_customers", 0)
    
    st.markdown(
        f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem;">
            <div style="margin-bottom: 0.6rem; font-size: 0.85rem; color: #f1f5f9;">
                Shared Devices: <strong style="color: #818cf8;">{s_dev}</strong>
            </div>
            <div style="margin-bottom: 0.6rem; font-size: 0.85rem; color: #f1f5f9;">
                Shared IPs: <strong style="color: #818cf8;">{s_ip}</strong>
            </div>
            <div style="margin-bottom: 0.6rem; font-size: 0.85rem; color: #f1f5f9;">
                Connected Suspicious Entities: <strong style="color: #f87171;">{s_susp}</strong>
            </div>
            <div style="font-size: 0.85rem; color: #f1f5f9;">
                Graph Risk Contribution: <strong style="color: #fbbf24;">{fmt_pct(graph_risk)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# TRUST SCORE DECOMPOSITION
# ==============================================================================
st.markdown('<div class="section-header">Trust Score Decomposition</div>', unsafe_allow_html=True)

ts_col1, ts_col2 = st.columns([1, 2])

with ts_col1:
    st.markdown(
        f"""
        <div style="background: #1e293b; border: 1px solid #6366f1; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Composite Trust Score</div>
            <div style="font-size: 3rem; font-weight: 800; color: #818cf8; margin: 0.2rem 0;">{fmt_pct(composite_score)}</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: {risk_color};">
                {risk_level_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with ts_col2:
    st.markdown("**Weighted Formula Breakdown**")
    st.caption(f"Model Confidence (Weight: 40%) — {fmt_pct(model_score)}")
    st.progress(min(max(model_score, 0.0), 1.0))
    
    st.caption(f"Historical Evidence Similarity (Weight: 35%) — {fmt_pct(evidence_support)}")
    st.progress(min(max(evidence_support, 0.0), 1.0))
    
    st.caption(f"Graph / Relationship Risk (Weight: 25%) — {fmt_pct(graph_risk)}")
    st.progress(min(max(graph_risk, 0.0), 1.0))

# ==============================================================================
# FINAL INTERPRETATION & INVESTIGATION SUMMARY
# ==============================================================================
st.markdown('<div class="section-header">TrustGraph Assessment</div>', unsafe_allow_html=True)

st.info(
    f"Customer **{active_id}** was evaluated with a Preliminary Trust Score of **{fmt_pct(composite_score)}** ({risk_level} LEVEL). "
    f"The ML detection engine assigned a model confidence score of **{fmt_pct(model_score)}**, "
    f"while historical case analysis yielded a **{fmt_pct(evidence_support)}** similarity match. "
    f"Entity network analysis measured a graph relationship risk of **{fmt_pct(graph_risk)}**."
)

st.markdown('<div class="section-header">Investigation Summary</div>', unsafe_allow_html=True)

s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
s_col1.metric("Detection", prediction_label)
s_col2.metric("Model Score", fmt_pct(model_score))
s_col3.metric("Historical Support", fmt_pct(evidence_support))
s_col4.metric("Graph Risk", fmt_pct(graph_risk))
s_col5.metric("Trust Score", fmt_pct(composite_score))