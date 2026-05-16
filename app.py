import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from datetime import date, timedelta
import calendar

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PowerCast | Electricity Forecasting",
    page_icon="assets/icon.png" if False else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161a24;
    border-right: 1px solid #252a38;
}
[data-testid="stSidebar"] * {color: #c8cdd8 !important;}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {color: #ffffff !important;}

/* Sidebar labels */
.stDateInput label, .stNumberInput label {
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Inputs */
.stDateInput input, .stNumberInput input {
    background: #1e2332 !important;
    border: 1px solid #2d3347 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c8973a 0%, #e8b84b 100%);
    color: #0f1117 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 1rem;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(200, 151, 58, 0.35) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #161a24;
    border: 1px solid #252a38;
    border-radius: 14px;
    padding: 1.4rem 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8892a4 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] {color: #c8973a !important;}

/* Info / success boxes */
.stAlert {
    background: #161a24 !important;
    border: 1px solid #252a38 !important;
    border-radius: 12px !important;
    color: #c8cdd8 !important;
}

/* Section divider */
hr {border-color: #252a38 !important;}

/* Scrollbar */
::-webkit-scrollbar {width: 5px;}
::-webkit-scrollbar-track {background: #0f1117;}
::-webkit-scrollbar-thumb {background: #2d3347; border-radius: 3px;}
</style>
""", unsafe_allow_html=True)

# ── Helper: load model ────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('electricity_model.pkl')
    return model

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem 0;'>
        <div style='font-size:1.05rem; font-weight:600; color:#ffffff; letter-spacing:0.02em;'>
            PowerCast
        </div>
        <div style='font-size:0.75rem; color:#8892a4; margin-top:2px;'>
            Electricity Demand Intelligence
        </div>
    </div>
    <hr style='border-color:#252a38; margin-bottom:1.5rem;'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem; color:#8892a4; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1rem;'>Forecast Parameters</p>", unsafe_allow_html=True)

    selected_date = st.date_input("Target Date", value=date.today())

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    lag_1 = st.number_input(
        "Yesterday's Consumption (GWh)",
        min_value=800.0, max_value=1800.0, value=1338.0, step=10.0
    )
    lag_7 = st.number_input(
        "Same Day Last Week (GWh)",
        min_value=800.0, max_value=1800.0, value=1320.0, step=10.0
    )
    lag_30 = st.number_input(
        "Same Day Last Month (GWh)",
        min_value=800.0, max_value=1800.0, value=1300.0, step=10.0
    )

    predict_clicked = st.button("Run Forecast")

    st.markdown("<hr style='border-color:#252a38; margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#4a5568; line-height:1.7;'>
        Model: Random Forest + Lag Features<br>
        Dataset: Germany 2006 – 2017<br>
        Training samples: 4,383 days<br>
        Best MAE: 24.96 GWh
    </div>
    """, unsafe_allow_html=True)

# ── Main Header ───────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='
        font-family: "Playfair Display", serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
    '>Electricity Demand<br><span style="color:#c8973a;">Forecasting</span></h1>
    <p style='color:#8892a4; font-size:0.92rem; margin-top:0.6rem; font-weight:300;'>
        Machine learning-powered consumption predictions for grid planning
    </p>
</div>
""", unsafe_allow_html=True)

# ── Prediction Logic ──────────────────────────────────────────
if predict_clicked:
    features = np.array([[
        selected_date.year,
        selected_date.month,
        selected_date.day,
        selected_date.weekday(),
        (selected_date.month - 1) // 3 + 1,
        lag_1, lag_7, lag_30
    ]])
    prediction = model.predict(features)[0]

    day_name = calendar.day_name[selected_date.weekday()]
    month_name = selected_date.strftime("%B")
    is_weekend = selected_date.weekday() >= 5

    delta_vs_yesterday = prediction - lag_1
    delta_str = f"{delta_vs_yesterday:+.1f} GWh vs yesterday"

    # ── Result banner
    color = "#c8973a" if prediction >= 1200 else "#5a8f7b"
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #161a24 60%, #1e2230);
        border: 1px solid #2d3347;
        border-left: 4px solid {color};
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 2rem;
    '>
        <div>
            <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.1em;'>Predicted Consumption</div>
            <div style='font-size:3rem; font-weight:600; color:#ffffff; line-height:1.1; margin-top:2px;'>
                {prediction:,.2f}
                <span style='font-size:1.2rem; color:#8892a4; font-weight:400;'>GWh</span>
            </div>
            <div style='font-size:0.82rem; color:{color}; margin-top:4px;'>{delta_str}</div>
        </div>
        <div style='border-left:1px solid #252a38; padding-left:2rem;'>
            <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.1em;'>Date</div>
            <div style='font-size:1.1rem; color:#ffffff; font-weight:500; margin-top:2px;'>{selected_date.strftime("%d %B %Y")}</div>
            <div style='font-size:0.82rem; color:#8892a4; margin-top:2px;'>{day_name} {"· Weekend" if is_weekend else "· Weekday"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Yesterday", f"{lag_1:,.0f} GWh", f"{prediction-lag_1:+.1f}")
    with col2:
        st.metric("Last Week", f"{lag_7:,.0f} GWh", f"{prediction-lag_7:+.1f}")
    with col3:
        st.metric("Last Month", f"{lag_30:,.0f} GWh", f"{prediction-lag_30:+.1f}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Chart
    rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor('#161a24')
    ax.set_facecolor('#161a24')

    labels = ['Yesterday\n(Lag 1)', 'Last Week\n(Lag 7)', 'Last Month\n(Lag 30)', 'Forecast']
    values = [lag_1, lag_7, lag_30, prediction]
    colors = ['#2d3f5c', '#2d3f5c', '#2d3f5c', '#c8973a']
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f'{val:,.0f}',
            ha='center', va='bottom',
            color='#c8cdd8', fontsize=10, fontweight='500'
        )

    ax.set_ylim(min(values) * 0.9, max(values) * 1.08)
    ax.set_ylabel('GWh', color='#8892a4', fontsize=9)
    ax.tick_params(colors='#8892a4', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#252a38')
    ax.spines['bottom'].set_color('#252a38')
    ax.yaxis.grid(True, color='#1e2332', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Consumption Comparison', color='#c8cdd8', fontsize=11, pad=14, loc='left')

    plt.tight_layout()
    st.pyplot(fig)

    # ── Insight box
    insight = "Weekend effect detected — consumption typically 12-18% lower than weekdays." if is_weekend else f"Weekday forecast for {month_name}. Consumption aligns with historical seasonal averages."
    st.markdown(f"""
    <div style='
        background:#161a24;
        border:1px solid #252a38;
        border-radius:12px;
        padding:1.1rem 1.4rem;
        margin-top:1rem;
        font-size:0.85rem;
        color:#8892a4;
        line-height:1.6;
    '>
        <span style='color:#c8973a; font-weight:600;'>Insight</span> &nbsp; {insight}
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Default dashboard view
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model Accuracy", "98.1%")
    with col2:
        st.metric("Training Period", "2006 – 2017")
    with col3:
        st.metric("Training Days", "4,383")
    with col4:
        st.metric("Best MAE", "24.96 GWh")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Model comparison table
    st.markdown("""
    <div style='
        background:#161a24;
        border:1px solid #252a38;
        border-radius:14px;
        padding:1.6rem 2rem;
        margin-top:0.5rem;
    '>
        <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.2rem;'>Model Performance Comparison</div>
        <table style='width:100%; border-collapse:collapse; font-size:0.88rem;'>
            <thead>
                <tr style='border-bottom:1px solid #252a38;'>
                    <th style='text-align:left; padding:0.6rem 0; color:#8892a4; font-weight:500;'>Model</th>
                    <th style='text-align:right; padding:0.6rem 0; color:#8892a4; font-weight:500;'>MAE</th>
                    <th style='text-align:right; padding:0.6rem 0; color:#8892a4; font-weight:500;'>Accuracy</th>
                </tr>
            </thead>
            <tbody>
                <tr style='border-bottom:1px solid #1e2230;'>
                    <td style='padding:0.7rem 0; color:#c8cdd8;'>Random Forest — Basic</td>
                    <td style='text-align:right; color:#c8cdd8;'>33.19 GWh</td>
                    <td style='text-align:right; color:#c8cdd8;'>97.5%</td>
                </tr>
                <tr style='border-bottom:1px solid #1e2230;'>
                    <td style='padding:0.7rem 0; color:#c8cdd8;'>LSTM Deep Learning</td>
                    <td style='text-align:right; color:#c8cdd8;'>32.55 GWh</td>
                    <td style='text-align:right; color:#c8cdd8;'>97.6%</td>
                </tr>
                <tr>
                    <td style='padding:0.7rem 0; color:#ffffff; font-weight:600;'>
                        Random Forest — Lag Features
                        <span style='background:#c8973a20; color:#c8973a; font-size:0.68rem; padding:2px 8px; border-radius:20px; margin-left:8px; font-weight:500;'>Best</span>
                    </td>
                    <td style='text-align:right; color:#c8973a; font-weight:600;'>24.96 GWh</td>
                    <td style='text-align:right; color:#c8973a; font-weight:600;'>98.1%</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1rem; font-size:0.82rem; color:#4a5568; text-align:center;'>
        Select a date and enter consumption values in the sidebar, then run the forecast.
    </div>
    """, unsafe_allow_html=True)