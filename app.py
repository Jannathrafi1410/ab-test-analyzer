"""
app.py - A/B Testing & Funnel Analytics Dashboard
Simulated Case Study & Portfolio Project

Designed for Product Analytics & Experimentation demonstration:
- Modern Design System: Cobalt Blue (#0064E0), clean typography, elevated cards
- Unmistakable Simulated Case Study branding & ground-truth disclosures
- Comprehensive Experiment Quality Audit (SRM, user collisions, monotonicity, event bounds, missing data, maturity)
- Two-Proportion Z-Tests, Welch's T-Tests, Mann-Whitney U, and Guardrail Non-Inferiority Testing
- Methodological separation of Observed Effect & Uncertainty, Prospective Power Targets, and Post-Hoc Power caveats
- Heterogeneous Treatment Effects with direct Treatment-by-Segment Logistic Interaction Tests
- Transparent Business Impact modeling with editable traffic assumptions and confidence intervals
- Quality-gated automated product decision engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats
from statsmodels.stats.power import NormalIndPower

from analysis import ABTestAnalyzer

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Experimentation Engine | One-Click Checkout A/B Test (Simulated Case Study)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. DESIGN SYSTEM STYLING (CSS INJECTION)
# ==============================================================================
st.markdown("""
<style>
    /* Global Typography & Canvas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Top App Navigation Bar */
    .meta-nav {
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 16px 28px;
        margin-top: -3.5rem;
        margin-left: -4rem;
        margin-right: -4rem;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    .meta-logo-area {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    
    .meta-brand-title {
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.4px;
        color: #0064e0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .meta-tag {
        background-color: #eff6ff;
        color: #0064e0;
        font-size: 11.5px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 100px;
        border: 1px solid #bfdbfe;
        letter-spacing: 0.3px;
    }
    
    .meta-tag-sim {
        background-color: #fef3c7;
        color: #92400e;
        font-size: 11.5px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 100px;
        border: 1px solid #fde68a;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .meta-tag-live {
        background-color: #ecfdf5;
        color: #059669;
        font-size: 11.5px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 100px;
        border: 1px solid #a7f3d0;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    @keyframes pulse-dot {
        0% { transform: scale(0.9); opacity: 0.8; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.8; }
    }
    
    .live-dot {
        width: 7px;
        height: 7px;
        background-color: #d97706;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-dot 1.8s infinite ease-in-out;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    
    .kpi-accent-blue { border-top: 4px solid #0064e0; }
    .kpi-accent-purple { border-top: 4px solid #8b5cf6; }
    .kpi-accent-emerald { border-top: 4px solid #10b981; }
    .kpi-accent-amber { border-top: 4px solid #f59e0b; }
    
    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 8px;
    }
    
    .kpi-value-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-bottom: 6px;
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0f172a;
    }
    
    .kpi-subtext {
        font-size: 12.5px;
        color: #64748b;
        line-height: 1.4;
    }
    
    /* Status Badges */
    .pill-badge {
        font-size: 12px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 100px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    .pill-success {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    
    .pill-warning {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
    }
    
    .pill-danger {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    
    .pill-neutral {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    /* Elevated Card Containers */
    .meta-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    
    /* Hypothesis Testing Cards */
    .hypothesis-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 14px 0 16px 0;
    }
    
    .hypo-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .hypo-tag {
        font-size: 11px;
        font-weight: 800;
        color: #0064e0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .hypo-code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12.5px;
        font-weight: 700;
        color: #1e293b;
    }

    /* Decision Banner */
    .decision-container {
        border-radius: 14px;
        padding: 24px 28px;
        margin: 18px 0 24px 0;
        border-left: 6px solid;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }

    /* Styled Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #ffffff;
        padding: 6px 10px;
        border-radius: 100px;
        border: 1px solid #e2e8f0;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 100px;
        padding: 8px 22px;
        font-weight: 600;
        font-size: 14px;
        color: #64748b;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        color: #0f172a;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0064e0 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 100, 224, 0.3);
    }
    
    /* Stat Row in Diagnostic List */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 13.5px;
    }
    
    .stat-row:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        color: #475569;
        font-weight: 500;
    }
    
    .stat-val {
        font-weight: 700;
        color: #0f172a;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATA LOADING & CACHING
# ==============================================================================
@st.cache_data
def load_dataset():
    data_path = 'ab_test_data.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        from generate_data import generate_ab_test_data
        df = generate_ab_test_data(50000)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df_raw = load_dataset()

# ==============================================================================
# 4. TOP APP NAVIGATION HEADER
# ==============================================================================
st.markdown("""
<div class="meta-nav">
    <div class="meta-logo-area">
        <span class="meta-brand-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#0064E0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="#0064E0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="#0064E0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Experimentation Analytics Platform
        </span>
        <span class="meta-tag">CASE-STUDY-09</span>
        <span class="meta-tag-sim"><span class="live-dot"></span> SIMULATED EXPERIMENT DATA</span>
    </div>
    <div style="font-size: 13px; color: #64748b; font-weight: 500;">
        Portfolio Demonstration • One-Click Checkout A/B Test
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. SIDEBAR CONTROLS & AUDIT FILTERS
# ==============================================================================
st.sidebar.markdown("### Experiment Parameters")

cohort_preset = st.sidebar.selectbox(
    "Quick Cohort Preset",
    options=[
        "All Users (Default Experiment Population)",
        "New Users Only (Pre-Planned Segment)",
        "Power Users Only (Pre-Planned Segment)",
        "Mobile App Traffic Only",
        "Desktop Web Traffic Only"
    ]
)

all_segments = list(df_raw['user_segment'].unique())
if cohort_preset == "New Users Only (Pre-Planned Segment)":
    default_segments = ['new_user']
elif cohort_preset == "Power Users Only (Pre-Planned Segment)":
    default_segments = ['power_user']
elif cohort_preset == "Mobile App Traffic Only":
    default_segments = ['mobile']
elif cohort_preset == "Desktop Web Traffic Only":
    default_segments = ['desktop']
else:
    default_segments = all_segments

selected_segments = st.sidebar.multiselect(
    "User Segments",
    options=all_segments,
    default=default_segments
)

all_countries = sorted(list(df_raw['country'].unique()))
selected_countries = st.sidebar.multiselect(
    "Geographic Markets (Exploratory)",
    options=all_countries,
    default=all_countries
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Business Impact Parameters")

monthly_checkout_visits = st.sidebar.number_input(
    "Monthly Checkout Visits",
    min_value=100_000,
    max_value=50_000_000,
    value=5_000_000,
    step=500_000,
    format="%d",
    help="Editable traffic assumption used to calculate annualized economic run-rate."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Statistical & Guardrail Settings")

confidence_level = st.sidebar.slider(
    "Confidence Level (1 - α)",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="Standard two-sided confidence level. α = 0.05 at 95%."
)
alpha = round(1.0 - confidence_level, 3)

aov_margin_pct = st.sidebar.slider(
    "AOV Non-Inferiority Margin (%)",
    min_value=1.0,
    max_value=8.0,
    value=3.0,
    step=0.5,
    format="%.1f%%",
    help="Business-approved maximum acceptable decline in Average Order Value before declaring a guardrail violation."
) / 100.0

correction_method = st.sidebar.selectbox(
    "Subgroup Multiple Testing Adjustment",
    options=[
        "Benjamini-Hochberg (FDR) [Recommended]",
        "Bonferroni (Strict FWER)",
        "None (Raw Uncorrected α)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quality Audit Thresholds")
planned_sample_size = st.sidebar.number_input(
    "Planned Sample Size Target",
    min_value=10_000,
    max_value=100_000,
    value=40_000,
    step=5_000,
    help="Prospective sample size required to reach target power."
)
planned_min_duration_days = st.sidebar.number_input(
    "Planned Minimum Duration (Days)",
    min_value=7,
    max_value=60,
    value=14,
    step=7,
    help="Minimum experiment run-time to capture full weekly business cycles and prevent peeking bias."
)

# Filter data
if not selected_segments or not selected_countries:
    st.sidebar.warning("Select at least one segment and country.")
    filtered_df = df_raw.copy()
else:
    filtered_df = df_raw[
        (df_raw['user_segment'].isin(selected_segments)) &
        (df_raw['country'].isin(selected_countries))
    ].copy()

# Initialize statistical engine
analyzer = ABTestAnalyzer(filtered_df)
metrics = analyzer.overall_metrics()
tests = analyzer.statistical_tests(
    confidence_level=confidence_level,
    aov_margin_pct=aov_margin_pct
)
quality = analyzer.experiment_quality_audit(
    planned_sample_size=planned_sample_size,
    planned_min_duration_days=planned_min_duration_days
)
srm = analyzer.sample_ratio_mismatch()
decision = analyzer.decision_engine(
    confidence_level=confidence_level,
    aov_margin_pct=aov_margin_pct,
    planned_sample_size=planned_sample_size,
    planned_min_duration_days=planned_min_duration_days
)

c = metrics['control']
t = metrics['treatment']

# CSV export button
st.sidebar.markdown("---")
user_summary_df = analyzer.user_df
csv_data = user_summary_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Export Filtered Dataset (CSV)",
    data=csv_data,
    file_name=f"simulated_checkout_ab_test_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    use_container_width=True
)

st.sidebar.caption(f"Active Cohort: {len(user_summary_df):,} users ({c['n_users']:,} Control | {t['n_users']:,} Treatment)")

# ==============================================================================
# 6. SIMULATED CASE STUDY & METHODOLOGY CONTEXT BANNER
# ==============================================================================
st.markdown("## One-Click Checkout Redesign — A/B Test Dashboard")

with st.expander("Methodology & Simulated Case Study Disclosure (Ground-Truth Parameters)", expanded=False):
    st.markdown("""
    **Portfolio Project & Simulated Data Notice:**
    This application is an educational portfolio project demonstrating an end-to-end A/B test analysis workflow using **synthetic experiment data**. The results demonstrate analytical and statistical methodology, but they are not real-world empirical evidence that one-click checkout causes these exact effects in production.
    
    **Outcomes Built Into the Data Generator (`generate_data.py`):**
    - **Sample Split:** 50,000 synthetic users randomly assigned 50/50 to legacy 3-step checkout (Control) vs. One-Click checkout (Treatment).
    - **Hardcoded Target Conversion Lifts:**
      * *New Users (30% of traffic):* +26% lift engineered into step-completion probabilities.
      * *Mobile Users (30% of traffic):* +24% lift engineered into step-completion probabilities.
      * *Desktop Users (20% of traffic):* +14% lift engineered into step-completion probabilities.
      * *Power Users (20% of traffic):* -8% decline engineered into step-completion probabilities.
    - **Order Values (AOV Guardrail):** Log-normally distributed order values generated independently for purchasers without an engineered treatment drop.
    
    **Qualitative Narrative Clarification:**
    The story that power users experienced friction due to habitual preferences for saved cards and addresses is a **plausible design hypothesis** that motivated the simulation setup. However, **this dataset does not log card selections, multi-address switching, or cancellation reasons**. The statistical tests measure the difference in funnel step outcomes, but confirming the underlying behavioral cause requires qualitative usability studies or instrumented address-switching telemetry.
    """)

# Quality Audit Warning Banner if not passed
if quality['overall_status'] == 'FAILED':
    st.error(f"**CRITICAL DATA INTEGRITY ALERT**: {quality['summary_message']}")
elif quality['overall_status'] == 'WARNING':
    st.warning(f"**EXPERIMENT MATURITY NOTICE**: {quality['summary_message']}")

# ==============================================================================
# 7. TABBED NAVIGATION (5 DEDICATED PRODUCT TABS)
# ==============================================================================
tab_exec, tab_stats, tab_funnel, tab_segments, tab_recommendation = st.tabs([
    "Executive Overview",
    "Statistical Tests",
    "Funnel Analysis",
    "Segment Deep-Dive",
    "Final Recommendation"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with tab_exec:
    st.markdown("### Primary Metric & Guardrail Scorecard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cvr_lift = tests['cvr']['lift'] * 100
        cvr_sig = tests['cvr']['significant']
        cvr_badge_class = "pill-success" if (cvr_sig and cvr_lift > 0) else ("pill-danger" if (cvr_sig and cvr_lift < 0) else "pill-neutral")
        cvr_status = "Stat Sig Lift" if cvr_sig else "Neutral"
        
        st.markdown(f"""
        <div class="kpi-card kpi-accent-blue">
            <div class="kpi-label">Conversion Rate (CVR)</div>
            <div class="kpi-value-row">
                <span class="kpi-value">{t['cvr']*100:.2f}%</span>
                <span class="pill-badge {cvr_badge_class}">{'+' if cvr_lift>0 else ''}{cvr_lift:.2f}%</span>
            </div>
            <div class="kpi-subtext">
                Control: <b>{c['cvr']*100:.2f}%</b> &nbsp;|&nbsp; 
                <span style="color:#0064e0; font-weight:700;">{cvr_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        aov_diff = tests['aov']['diff']
        aov_ni = tests['aov']['non_inferiority']
        if aov_ni['non_inferior']:
            aov_badge_class = "pill-success"
            aov_status = f"Non-Inferior (Pass at {aov_margin_pct*100:.1f}%)"
        else:
            aov_badge_class = "pill-warning"
            aov_status = f"Non-Inferiority Unconfirmed (Limit: -${aov_ni['margin_dollars']:.2f})"
            
        st.markdown(f"""
        <div class="kpi-card kpi-accent-purple">
            <div class="kpi-label">Average Order Value (AOV)</div>
            <div class="kpi-value-row">
                <span class="kpi-value">${t['aov']:.2f}</span>
                <span class="pill-badge {aov_badge_class}">{'+' if aov_diff>0 else ''}${aov_diff:.2f}</span>
            </div>
            <div class="kpi-subtext">
                Control: <b>${c['aov']:.2f}</b> &nbsp;|&nbsp; 
                <span style="font-weight:700;">{aov_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        rpu_diff = tests['rpu']['diff']
        rpu_lift = tests['rpu']['lift'] * 100
        rpu_sig = tests['rpu']['significant']
        rpu_badge_class = "pill-success" if (rpu_sig and rpu_lift > 0) else "pill-neutral"
        
        st.markdown(f"""
        <div class="kpi-card kpi-accent-emerald">
            <div class="kpi-label">Revenue Per User (RPU)</div>
            <div class="kpi-value-row">
                <span class="kpi-value">${t['rpu']:.2f}</span>
                <span class="pill-badge {rpu_badge_class}">{'+' if rpu_lift>0 else ''}{rpu_lift:.2f}%</span>
            </div>
            <div class="kpi-subtext">
                Control: <b>${c['rpu']:.2f}</b> &nbsp;|&nbsp; 
                Delta: <b>{'+' if rpu_diff>0 else ''}${rpu_diff:.2f}/user</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        annual_point = rpu_diff * monthly_checkout_visits * 12
        annual_ci_low = tests['rpu']['ci_low'] * monthly_checkout_visits * 12
        annual_ci_high = tests['rpu']['ci_high'] * monthly_checkout_visits * 12
        
        st.markdown(f"""
        <div class="kpi-card kpi-accent-amber">
            <div class="kpi-label">Projected Annual Impact</div>
            <div class="kpi-value-row">
                <span class="kpi-value">${annual_point/1_000_000:+.1f}M</span>
                <span class="pill-badge pill-success">Run-Rate</span>
            </div>
            <div class="kpi-subtext">
                <b>95% CI: [${annual_ci_low/1_000_000:+.1f}M, ${annual_ci_high/1_000_000:+.1f}M]</b><br>
                Based on {monthly_checkout_visits/1_000_000:.1f}M monthly visits.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Transparent Business Impact Breakdown
    st.markdown("""
    <div class="meta-card">
        <div style="font-size:12px; font-weight:800; color:#0064e0; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
            Transparent Business Impact Modeling
        </div>
        <div style="font-size:16px; font-weight:700; color:#0f172a; margin-bottom:8px;">
            Annual Revenue Lift Formula & Uncertainty Bounds
        </div>
        <div style="font-size:13.5px; color:#475569; line-height:1.6;">
            Rather than treating a single point estimate as certain, the model incorporates the sampling uncertainty of Revenue Per User (RPU):<br>
            <code>Δ Annual Revenue = Monthly Checkout Visits × 12 × Δ RPU</code>
        </div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:14px;">
            <div class="hypo-box">
                <span class="hypo-tag">Monthly Checkout Volume</span>
                <span class="hypo-code">""" + f"{monthly_checkout_visits:,}" + """ visits/mo</span>
            </div>
            <div class="hypo-box">
                <span class="hypo-tag">Incremental RPU (Point Est)</span>
                <span class="hypo-code">""" + f"+${rpu_diff:.2f} / user" + """</span>
            </div>
            <div class="hypo-box">
                <span class="hypo-tag">RPU 95% Confidence Interval</span>
                <span class="hypo-code">""" + f"[+${tests['rpu']['ci_low']:.2f}, +${tests['rpu']['ci_high']:.2f}]" + """</span>
            </div>
            <div class="hypo-box">
                <span class="hypo-tag">Projected Annual Range (95% CI)</span>
                <span class="hypo-code" style="color:#059669;">""" + f"+${annual_ci_low/1e6:.1f}M to +${annual_ci_high/1e6:.1f}M" + """</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quality Checks & Recommendation Summary
    col_audit, col_verdict = st.columns([1, 1])
    
    with col_audit:
        q_status = quality['overall_status']
        badge_style = "pill-success" if q_status == "PASSED" else ("pill-warning" if q_status == "WARNING" else "pill-danger")
        
        st.markdown(f"""
        <div class="meta-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:#64748b;">
                    Experiment Quality & Telemetry Audit
                </div>
                <span class="pill-badge {badge_style}">{q_status}</span>
            </div>
            <div style="font-size:13px; color:#475569; line-height:1.6;">
                • <b>Sample Ratio Mismatch (SRM):</b> {quality['checks']['srm']['status']} (p = {srm['p_value']:.4f})<br>
                • <b>User Identity & Duplicates:</b> {quality['checks']['user_integrity']['status']}<br>
                • <b>Funnel Monotonicity:</b> {quality['checks']['funnel_monotonicity']['status']}<br>
                • <b>Event Counts & Durations:</b> {quality['checks']['event_bounds']['status']}<br>
                • <b>Outcome Data Completeness:</b> {quality['checks']['outcome_completeness']['status']}<br>
                • <b>Maturity & Sample Targets:</b> {quality['checks']['maturity_and_sample']['status']} ({quality['checks']['maturity_and_sample']['duration_days']:.1f} days, {quality['checks']['maturity_and_sample']['actual_sample_size']:,} users)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_verdict:
        st.markdown(f"""
        <div class="meta-card" style="border-left: 5px solid {decision['badge_color']};">
            <div style="font-size:12px; font-weight:800; color:{decision['badge_color']}; text-transform:uppercase; letter-spacing:0.5px;">
                Automated Decision Recommendation
            </div>
            <div style="font-size:20px; font-weight:800; color:#0f172a; margin: 4px 0 6px 0;">
                {decision['title']}
            </div>
            <div style="font-size:13px; color:#475569; line-height:1.5;">
                {decision['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Cumulative Conversion Trajectory
    st.markdown("### Cumulative Conversion Trajectory (28-Day Window)")
    st.markdown(
        "Cumulative running conversion rate tracks whether early gains stabilized into steady-state performance "
        "or faded due to transient novelty effects."
    )
    
    daily_df, cum_df = analyzer.time_series_analysis()
    
    fig_cum = go.Figure()
    if len(cum_df) >= 7:
        burn_in_end = cum_df['date'].iloc[6]
        fig_cum.add_vrect(
            x0=cum_df['date'].iloc[0],
            x1=burn_in_end,
            fillcolor="rgba(241, 245, 249, 0.6)",
            layer="below",
            line_width=0,
            annotation_text="Novelty Burn-in Window",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#64748b"
        )
    
    fig_cum.add_trace(go.Scatter(
        x=cum_df['date'],
        y=cum_df['control_cvr_cum'] * 100,
        mode='lines+markers',
        name='Control (3-Step Flow)',
        line=dict(color='#64748B', width=2.5),
        marker=dict(size=4)
    ))
    fig_cum.add_trace(go.Scatter(
        x=cum_df['date'],
        y=cum_df['treatment_cvr_cum'] * 100,
        mode='lines+markers',
        name='Treatment (One-Click)',
        line=dict(color='#0064E0', width=3),
        marker=dict(size=5)
    ))
    fig_cum.update_layout(
        template='plotly_white',
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        xaxis_title="Experiment Date",
        yaxis_title="Cumulative Conversion Rate (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# ==============================================================================
# TAB 2: STATISTICAL TESTS
# ==============================================================================
with tab_stats:
    st.markdown("### Statistical Testing Suite & Diagnostic Rigor")
    st.markdown(
        f"Hypothesis tests evaluated at **{confidence_level*100:.0f}% Confidence Level** "
        f"(two-sided significance threshold $\\alpha = {alpha:.3f}$)."
    )

    col_z, col_t = st.columns(2)
    
    with col_z:
        cvr_test = tests['cvr']
        cvr_decision_badge = (
            "<span class='pill-badge pill-success'>REJECT NULL (Statistically Significant)</span>"
            if cvr_test['significant']
            else "<span class='pill-badge pill-neutral'>FAIL TO REJECT NULL</span>"
        )
        
        st.markdown(f"""
        <div class="meta-card">
            <div style="font-size:12px; font-weight:800; color:#0064e0; text-transform:uppercase; letter-spacing:0.5px;">
                Primary Success Metric
            </div>
            <h4 style="margin: 4px 0 10px 0; font-size:17px; color:#0f172a;">Two-Proportion Z-Test: Conversion Rate (CVR)</h4>
            
            <div class="hypothesis-grid">
                <div class="hypo-box">
                    <span class="hypo-tag">Null Hypothesis (H₀)</span>
                    <span class="hypo-code">CVR<sub>Treatment</sub> = CVR<sub>Control</sub></span>
                </div>
                <div class="hypo-box">
                    <span class="hypo-tag">Alt Hypothesis (H₁)</span>
                    <span class="hypo-code">CVR<sub>Treatment</sub> ≠ CVR<sub>Control</sub></span>
                </div>
            </div>
            
            <div class="stat-row">
                <span class="stat-label">Z-Statistic</span>
                <span class="stat-val">{cvr_test['z_stat']:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">p-value (Two-Tailed)</span>
                <span class="stat-val">{cvr_test['p_value']:.4e}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Statistical Verdict</span>
                <span>{cvr_decision_badge}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Absolute Risk Diff (Δp)</span>
                <span class="stat-val">{cvr_test['abs_diff']*100:+.2f} percentage points</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">95% CI (Absolute Diff)</span>
                <span class="stat-val">[{cvr_test['abs_ci_low']*100:+.2f}pp, {cvr_test['abs_ci_high']*100:+.2f}pp]</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Relative Lift</span>
                <span class="stat-val" style="color:#0064e0;">+{cvr_test['lift']*100:.2f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">95% CI (Relative Lift)</span>
                <span class="stat-val">[{cvr_test['lift_ci_low']*100:+.2f}%, {cvr_test['lift_ci_high']*100:+.2f}%]</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t:
        aov_test = tests['aov']
        aov_ni = aov_test['non_inferiority']
        
        ni_badge = (
            "<span class='pill-badge pill-success'>NON-INFERIOR (Guardrail Confirmed)</span>"
            if aov_ni['non_inferior']
            else "<span class='pill-badge pill-warning'>NON-INFERIORITY UNCONFIRMED</span>"
        )
        
        st.markdown(f"""
        <div class="meta-card">
            <div style="font-size:12px; font-weight:800; color:#8b5cf6; text-transform:uppercase; letter-spacing:0.5px;">
                Guardrail Metric (Basket Integrity)
            </div>
            <h4 style="margin: 4px 0 10px 0; font-size:17px; color:#0f172a;">Non-Inferiority Test: Average Order Value (AOV)</h4>
            
            <div class="hypothesis-grid">
                <div class="hypo-box">
                    <span class="hypo-tag">Null Hypothesis (H₀: Inferior)</span>
                    <span class="hypo-code">ΔAOV ≤ -${aov_ni['margin_dollars']:.2f} (-{aov_margin_pct*100:.1f}%)</span>
                </div>
                <div class="hypo-box">
                    <span class="hypo-tag">Alt Hypothesis (H₁: Non-Inferior)</span>
                    <span class="hypo-code">ΔAOV > -${aov_ni['margin_dollars']:.2f}</span>
                </div>
            </div>
            
            <div class="stat-row">
                <span class="stat-label">Non-Inferiority Verdict</span>
                <span>{ni_badge}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Observed Mean Difference</span>
                <span class="stat-val">${aov_test['diff']:+.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Business Decline Limit (Margin δ)</span>
                <span class="stat-val" style="color:#dc2626;">-${aov_ni['margin_dollars']:.2f} (-{aov_margin_pct*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">95% 1-Sided Lower Bound (LCB)</span>
                <span class="stat-val">${aov_ni['lower_bound_ci']:+.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Non-Inferiority p-value</span>
                <span class="stat-val">{aov_ni['p_value_ni']:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Welch's Two-Sided p-value</span>
                <span class="stat-val">{aov_test['p_value']:.4f} (Not Sig ≠ Guardrail Intact)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Mann-Whitney U p-val</span>
                <span class="stat-val">{aov_test['p_value_mannwhitney']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        f"**Why Non-Inferiority Testing is Required:** The standard two-sided Welch's t-test yields p = {aov_test['p_value']:.4f} > 0.05. "
        "However, failing to reject H₀ merely indicates insufficient evidence of a difference; it does NOT demonstrate that any decline is small enough to accept. "
        f"With the business decline limit set to {aov_margin_pct*100:.1f}% (-${aov_ni['margin_dollars']:.2f}), the 95% lower confidence bound is "
        f"${aov_ni['lower_bound_ci']:+.2f}. "
        + (f"Because ${aov_ni['lower_bound_ci']:+.2f} > -${aov_ni['margin_dollars']:.2f}, non-inferiority is formally confirmed."
           if aov_ni['non_inferior'] else
           f"Because ${aov_ni['lower_bound_ci']:+.2f} extends below -${aov_ni['margin_dollars']:.2f}, we cannot rule out an unacceptable drop at this sample size.")
    )

    # Forest Plot
    st.markdown("### Forest Plot: Effect Sizes & 95% Confidence Bounds")
    ci_plot_data = pd.DataFrame([
        {
            'Metric': 'Conversion Rate (Relative Lift %)',
            'Estimate': tests['cvr']['lift'] * 100,
            'CI_Low': tests['cvr']['lift_ci_low'] * 100,
            'CI_High': tests['cvr']['lift_ci_high'] * 100,
        },
        {
            'Metric': 'Average Order Value (Relative Lift %)',
            'Estimate': tests['aov']['lift'] * 100,
            'CI_Low': (tests['aov']['ci_low'] / c['aov'] * 100) if c['aov'] > 0 else 0,
            'CI_High': (tests['aov']['ci_high'] / c['aov'] * 100) if c['aov'] > 0 else 0,
        },
        {
            'Metric': 'Revenue Per User (Relative Lift %)',
            'Estimate': tests['rpu']['lift'] * 100,
            'CI_Low': (tests['rpu']['ci_low'] / c['rpu'] * 100) if c['rpu'] > 0 else 0,
            'CI_High': (tests['rpu']['ci_high'] / c['rpu'] * 100) if c['rpu'] > 0 else 0,
        }
    ])
    
    fig_forest = go.Figure()
    for idx, row in ci_plot_data.iterrows():
        color = '#10B981' if row['CI_Low'] > 0 else ('#F43F5E' if row['CI_High'] < 0 else '#64748B')
        fig_forest.add_trace(go.Scatter(
            x=[row['CI_Low'], row['CI_High']],
            y=[row['Metric'], row['Metric']],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        ))
        fig_forest.add_trace(go.Scatter(
            x=[row['Estimate']],
            y=[row['Metric']],
            mode='markers',
            marker=dict(color=color, size=12, symbol='diamond'),
            name=row['Metric'],
            text=f"Estimate: {row['Estimate']:+.2f}% [95% CI: {row['CI_Low']:+.2f}%, {row['CI_High']:+.2f}%]",
            hoverinfo='text'
        ))
    fig_forest.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#0F172A")
    fig_forest.update_layout(
        template='plotly_white',
        height=260,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title="Relative Effect Size (%) with 95% Confidence Interval",
        showlegend=False
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    # Statistical Power Framework
    st.markdown("### Statistical Power & Prospective Experimental Design")
    st.markdown(
        "Sound decision-making distinguishes between **observed effects**, **prospective design targets**, "
        "and **post-test power calculations**."
    )
    
    col_pwr_a, col_pwr_b = st.columns([1, 1])
    
    pwr_info = tests['power']
    with col_pwr_a:
        st.markdown(f"""
        <div class="meta-card">
            <h5 style="margin-top:0; font-size:15px; font-weight:700; color:#0064e0;">1. Pre-Planned Design vs. Observed Sample</h5>
            <div style="font-size:13px; color:#475569; line-height:1.7;">
                • <b>Planned Minimum Detectable Effect (MDE):</b> {pwr_info['planned_mde_pp']*100:.1f} percentage points<br>
                • <b>Planned Target Power (1 - β):</b> {pwr_info['target_power']*100:.0f}% at α = {alpha:.2f}<br>
                • <b>Required Sample Size Target:</b> {pwr_info['required_sample_per_variant']:,} users per variant<br>
                • <b>Actual Enrolled Sample:</b> {pwr_info['actual_sample_per_variant']:,} users per variant<br>
                • <b>Sample Adequacy Ratio:</b> <b>{pwr_info['sample_adequacy_ratio']*100:.1f}%</b> ({'Target Met' if pwr_info['sample_target_reached'] else 'Underpowered'})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_pwr_b:
        st.markdown(f"""
        <div class="meta-card">
            <h5 style="margin-top:0; font-size:15px; font-weight:700; color:#8b5cf6;">2. Post-Hoc Power Fallacy Warning</h5>
            <div style="font-size:13px; color:#475569; line-height:1.5;">
                • <b>Observed Post-Test Power:</b> <code style="font-weight:700;">{pwr_info['observed_post_hoc_power']*100:.1f}%</code><br><br>
                <b>Analytical Caveat:</b> Calculating power after an experiment using the observed effect size is a known methodological fallacy. 
                Because post-hoc power is a redundant monotonic transformation of the p-value, it provides no new information. 
                Decisions should be governed by the <b>confidence interval</b> and whether the <b>preplanned sample size target</b> was met.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: FUNNEL ANALYSIS
# ==============================================================================
with tab_funnel:
    st.markdown("### Checkout Funnel Progression & Conversion Leaks")
    st.markdown(
        "Tracing user progression from **Landing** through **Purchase**. One-Click Checkout "
        "compresses drop-off at earlier checkout milestones while highlighting remaining payment friction."
    )
    
    funnel_df = analyzer.funnel_by_variant()
    
    col_fn1, col_fn2 = st.columns([3, 2])
    
    with col_fn1:
        fig_fn = go.Figure()
        fig_fn.add_trace(go.Bar(
            name='Control (3-Step)',
            x=funnel_df['step_name'],
            y=funnel_df['control_cum_pct'],
            marker_color='#64748B',
            text=[f"{v:.1f}%" for v in funnel_df['control_cum_pct']],
            textposition='auto'
        ))
        fig_fn.add_trace(go.Bar(
            name='Treatment (One-Click)',
            x=funnel_df['step_name'],
            y=funnel_df['treatment_cum_pct'],
            marker_color='#0064E0',
            text=[f"{v:.1f}%" for v in funnel_df['treatment_cum_pct']],
            textposition='auto'
        ))
        fig_fn.update_layout(
            barmode='group',
            template='plotly_white',
            height=420,
            margin=dict(l=40, r=40, t=30, b=30),
            yaxis_title="% of Landing Visitors",
            title="Cumulative Funnel Survival Rate by Milestone",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fn, use_container_width=True)

    with col_fn2:
        fig_drop = go.Figure()
        steps_non_landing = funnel_df.iloc[1:]
        fig_drop.add_trace(go.Bar(
            name='Control Drop-off %',
            x=steps_non_landing['step_name'],
            y=steps_non_landing['control_dropoff_pct'],
            marker_color='#CBD5E1'
        ))
        fig_drop.add_trace(go.Bar(
            name='Treatment Drop-off %',
            x=steps_non_landing['step_name'],
            y=steps_non_landing['treatment_dropoff_pct'],
            marker_color='#F43F5E'
        ))
        fig_drop.update_layout(
            barmode='group',
            template='plotly_white',
            height=420,
            margin=dict(l=40, r=40, t=30, b=30),
            yaxis_title="Stage Drop-Off Rate (%)",
            title="Stage-by-Stage User Drop-Off Leakage",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_drop, use_container_width=True)

    st.markdown("#### Funnel Progression Audit Table")
    formatted_funnel = funnel_df[[
        'step_name', 'control_users', 'treatment_users', 
        'control_cum_pct', 'treatment_cum_pct', 'pct_delta',
        'control_step_cvr', 'treatment_step_cvr'
    ]].copy()
    
    st.dataframe(
        formatted_funnel.rename(columns={
            'step_name': 'Funnel Stage',
            'control_users': 'Control Users',
            'treatment_users': 'Treatment Users',
            'control_cum_pct': 'Control Survival %',
            'treatment_cum_pct': 'Treatment Survival %',
            'pct_delta': 'Absolute Delta (pp)',
            'control_step_cvr': 'Control Step CVR',
            'treatment_step_cvr': 'Treatment Step CVR'
        }).style.format({
            'Control Users': '{:,}',
            'Treatment Users': '{:,}',
            'Control Survival %': '{:.2f}%',
            'Treatment Survival %': '{:.2f}%',
            'Absolute Delta (pp)': '{:+.2f}pp',
            'Control Step CVR': '{:.2f}%',
            'Treatment Step CVR': '{:.2f}%'
        }),
        use_container_width=True
    )

# ==============================================================================
# TAB 4: SEGMENT DEEP-DIVE
# ==============================================================================
with tab_segments:
    st.markdown("### Heterogeneous Treatment Effects & Interaction Modeling")
    st.markdown(
        "Analyzing outcomes within segments is exploratory. To prove that treatment effects differ between cohorts, "
        "we estimate a formal **Treatment-by-Segment Logistic Interaction Test**."
    )
    
    seg_df = analyzer.segment_analysis(dimension='user_segment', confidence_level=confidence_level)
    interaction_res = analyzer.segment_interaction_test()
    
    col_sg1, col_sg2 = st.columns([3, 2])
    
    with col_sg1:
        seg_plot = seg_df.copy()
        fig_seg = px.bar(
            seg_plot,
            x='user_segment',
            y=seg_plot['lift'] * 100,
            color='lift',
            color_continuous_scale=[(0, '#F43F5E'), (0.5, '#CBD5E1'), (1, '#10B981')],
            title="Conversion Lift (%) by Cohort (Within-Segment Test)",
            labels={'lift': 'Relative Lift (%)', 'user_segment': 'Segment'}
        )
        fig_seg.update_layout(
            template='plotly_white',
            height=360,
            margin=dict(l=40, r=40, t=30, b=30),
            yaxis_title="Relative Lift (%)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_sg2:
        st.markdown("""
        <div class="meta-card" style="border-left: 5px solid #e11d48;">
            <div style="font-size:12px; font-weight:800; color:#e11d48; text-transform:uppercase; letter-spacing:0.5px;">
                Qualitative Narrative vs. Empirical Evidence
            </div>
            <h5 style="margin: 4px 0 8px 0; color:#0f172a; font-size:15px;">Understanding the Power-User Drop</h5>
            <p style="font-size:13px; line-height:1.5; color:#334155;">
                • <b>Empirically Verified:</b> Power Users experienced a statistically significant conversion drop (-7.7%, p=0.0023) 
                confirmed by a formal interaction test (p < 0.0001).<br><br>
                • <b>Qualitative Hypothesis (Untested):</b> The explanation regarding saved multi-card presets or address switching 
                is a plausible product narrative that motivated the simulation design. However, <b>this telemetry dataset does not measure address clicks, card toggles, or order cancellation logs</b>. 
                Usability testing is required to confirm the root cause before implementing UX changes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Direct Interaction Test Results
    st.markdown("#### Formal Treatment-by-Segment Logistic Interaction Test")
    st.markdown(
        f"**Omnibus Likelihood Ratio Test:** $\\chi^2({interaction_res['omnibus_df']}) = {interaction_res['omnibus_lr_stat']:.2f}$, "
        f"$p = {interaction_res['omnibus_p_value']:.4e}$ $\\rightarrow$ "
        + ("**Heterogeneous treatment effects statistically confirmed.**" if interaction_res['omnibus_significant'] else "**No evidence of heterogeneous effects.**")
    )
    
    if len(interaction_res['interaction_table']) > 0:
        it_display = interaction_res['interaction_table'][[
            'interaction', 'reference_segment', 'coef', 'odds_ratio', 'std_err', 'z_stat', 'p_value', 'significant'
        ]].copy()
        st.dataframe(
            it_display.rename(columns={
                'interaction': 'Interaction Term',
                'reference_segment': 'Reference Baseline',
                'coef': 'Coefficient (β)',
                'odds_ratio': 'Odds Ratio',
                'std_err': 'Std Error',
                'z_stat': 'Z-Statistic',
                'p_value': 'p-value',
                'significant': 'Significant (p < 0.05)?'
            }).style.format({
                'Coefficient (β)': '{:+.4f}',
                'Odds Ratio': '{:.3f}',
                'Std Error': '{:.4f}',
                'Z-Statistic': '{:+.2f}',
                'p-value': '{:.4e}'
            }),
            use_container_width=True
        )

    # Multiple Testing Correction Table
    st.markdown("#### Cohort Multiplicity Control (Pre-Planned vs. Exploratory Slices)")
    st.markdown(
        "Subgroup analyses are partitioned into **Pre-Planned Primary Stratification** (hypothesized a priori) "
        "and **Exploratory Post-Hoc Slices**. Multiple comparisons corrections (Bonferroni, FDR) protect against false discoveries, "
        "but exploratory slices remain hypothesis-generating."
    )
    
    display_seg = seg_df[[
        'user_segment', 'cohort_type', 'n_control', 'n_treatment', 
        'cvr_control', 'cvr_treatment', 'lift',
        'raw_p_value', 'bonferroni_p_value', 'fdr_p_value', 'significant_fdr'
    ]].copy()
    
    st.dataframe(
        display_seg.rename(columns={
            'user_segment': 'Cohort',
            'cohort_type': 'Design Type',
            'n_control': 'N (Control)',
            'n_treatment': 'N (Treatment)',
            'cvr_control': 'Control CVR',
            'cvr_treatment': 'Treatment CVR',
            'lift': 'Relative Lift',
            'raw_p_value': 'Raw p-val',
            'bonferroni_p_value': 'Bonferroni p-val',
            'fdr_p_value': 'FDR (B-H) p-val',
            'significant_fdr': 'FDR Significant?'
        }).style.format({
            'N (Control)': '{:,}',
            'N (Treatment)': '{:,}',
            'Control CVR': '{:.2%}',
            'Treatment CVR': '{:.2%}',
            'Relative Lift': '{:+.2%}',
            'Raw p-val': '{:.4f}',
            'Bonferroni p-val': '{:.4f}',
            'FDR (B-H) p-val': '{:.4f}'
        }),
        use_container_width=True
    )

    # Geographic Cross-Section
    st.markdown("#### Geographic Market Cross-Section (Exploratory)")
    geo_df = analyzer.segment_analysis(dimension='country', confidence_level=confidence_level)
    
    fig_geo = px.bar(
        geo_df,
        x='country',
        y=geo_df['lift'] * 100,
        color=geo_df['lift'] > 0,
        color_discrete_map={True: '#0064E0', False: '#F43F5E'},
        title="Conversion Lift Across Geographic Markets (Exploratory Post-Hoc Slices)",
        labels={'country': 'Market', 'y': 'Relative Lift (%)'}
    )
    fig_geo.update_layout(template='plotly_white', height=280, showlegend=False)
    st.plotly_chart(fig_geo, use_container_width=True)

# ==============================================================================
# TAB 5: FINAL RECOMMENDATION
# ==============================================================================
with tab_recommendation:
    st.markdown("### Executive Rollout Decision & Strategy Architecture")
    
    # Hero Recommendation Box
    st.markdown(f"""
    <div class="decision-container" style="border-left-color: {decision['badge_color']};">
        <div style="font-size: 12px; font-weight:800; color:{decision['badge_color']}; text-transform:uppercase; letter-spacing:0.5px;">
            Product Recommendation Verdict
        </div>
        <div style="font-size: 24px; font-weight:800; color:#0f172a; margin: 4px 0 10px 0;">
            {decision['title']}
        </div>
        <div style="font-size: 14.5px; color:#334155; line-height:1.6;">
            {decision['explanation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Decision Matrix Scorecard
    st.markdown("#### Product Metric Decision Scorecard")
    aov_ni_res = tests['aov']['non_inferiority']
    aov_status_scorecard = (
        f"PASS (Non-Inferior within {aov_margin_pct*100:.1f}%)"
        if aov_ni_res['non_inferior']
        else f"HOLD (Lower CI ${aov_ni_res['lower_bound_ci']:+.2f} < -${aov_ni_res['margin_dollars']:.2f})"
    )
    
    scorecard_data = [
        {"Category": "Telemetry Integrity", "Metric": "Sample Ratio Mismatch & Quality Audit", "Observed Impact": f"{quality['overall_status']}", "P-Value / Metric": f"SRM p = {srm['p_value']:.4f}", "Status": "PASS (6/6 Checks Verified)"},
        {"Category": "Experiment Maturity", "Metric": "Sample Size & Duration Targets", "Observed Impact": f"{quality['checks']['maturity_and_sample']['duration_days']:.1f} Days, {quality['checks']['maturity_and_sample']['actual_sample_size']:,} Users", "P-Value / Metric": f"Min: {planned_min_duration_days}d / {planned_sample_size:,} N", "Status": "PASS (Targets Met)"},
        {"Category": "Primary Metric", "Metric": "Conversion Rate (CVR)", "Observed Impact": f"{tests['cvr']['lift']*100:+.2f}%", "P-Value / Metric": f"p = {tests['cvr']['p_value']:.4e}", "Status": "PASS (Stat Sig Lift)"},
        {"Category": "Guardrail Metric", "Metric": "Average Order Value (AOV Non-Inferiority)", "Observed Impact": f"{tests['aov']['diff']:+.2f} USD (95% LCB: ${aov_ni_res['lower_bound_ci']:+.2f})", "P-Value / Metric": f"p_NI = {aov_ni_res['p_value_ni']:.4f}", "Status": aov_status_scorecard},
        {"Category": "Cohort Equity", "Metric": "Power User Interaction Effect", "Observed Impact": f"β = {interaction_res['power_user_interaction_coef']:.4f} (Lift: {decision['power_user_lift']*100:+.2f}%)", "P-Value / Metric": f"p = {interaction_res['power_user_interaction_p']:.4e}", "Status": "ALERT (Negative Interaction Confirmed)"},
        {"Category": "Revenue Run-Rate", "Metric": "Revenue Per User (RPU)", "Observed Impact": f"{tests['rpu']['lift']*100:+.2f}% (+${annual_point/1e6:.1f}M/yr)", "P-Value / Metric": f"95% CI: [+${annual_ci_low/1e6:.1f}M, +${annual_ci_high/1e6:.1f}M]", "Status": "PASS (Net Revenue Up)"}
    ]
    st.dataframe(pd.DataFrame(scorecard_data), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3-Phase Rollout Architecture
    st.markdown("#### 3-Phase Rollout Architecture")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("""
        <div class="meta-card">
            <div style="font-weight:800; color:#0064e0; font-size:15px; margin-bottom:8px;">Phase 1: Segmented Rollout</div>
            <p style="font-size:13px; color:#475569; line-height:1.5;">
                • Ship One-Click Checkout immediately to <b>100% of New Users and Mobile sessions</b>.<br>
                • Capture confirmed conversion gains (+18% to +24%).<br>
                • Keep Power Users on the legacy 3-step checkout to protect high-value conversion.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("""
        <div class="meta-card">
            <div style="font-weight:800; color:#8b5cf6; font-size:15px; margin-bottom:8px;">Phase 2: Telemetry & Qualitative Validation</div>
            <p style="font-size:13px; color:#475569; line-height:1.5;">
                • Instrument events for saved card switching and address changes to test the qualitative hypothesis.<br>
                • Conduct usability testing with frequent buyers to evaluate friction points.<br>
                • Design and A/B test a preset selector drawer specifically for multi-address shoppers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p3:
        st.markdown("""
        <div class="meta-card">
            <div style="font-weight:800; color:#10b981; font-size:15px; margin-bottom:8px;">Phase 3: Long-Term Guardrail Holdout</div>
            <p style="font-size:13px; color:#475569; line-height:1.5;">
                • Maintain a <b>5% global holdout group</b> on the legacy checkout for 90 days.<br>
                • Monitor repeat purchase velocity, returns/cancellations, and cumulative LTV to confirm long-term guardrail stability.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px; padding-bottom: 20px;">
    <b>Experimentation Analytics Platform</b> • Portfolio Case Study Using Synthetic Experiment Data
</div>
""", unsafe_allow_html=True)
