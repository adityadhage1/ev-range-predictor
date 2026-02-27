# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Pulse EV Predictor",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL ECO-GREEN CSS (Custom Injection)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com');

:root {
    --bg-primary:   #0a0a0f;
    --bg-card:      #111118;
    --accent:       #00FF41; /* ELECTRIC GREEN */
    --accent-glow:  rgba(0, 255, 65, 0.4);
    --text-primary: #f0f0f5;
    --border:       rgba(0, 255, 65, 0.2);
}

.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #0d0d15 !important;
    border-right: 1px solid var(--accent);
}

/* Glowing Green Headings */
h1, h2, h3 { 
    font-family: 'Orbitron', sans-serif !important; 
    color: var(--accent) !important;
    text-shadow: 0 0 15px var(--accent-glow);
}

/* Electric Green Sliders */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
    box-shadow: 0 0 10px var(--accent-glow);
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] + div > div > div {
    background: linear-gradient(90deg, var(--accent), #00d4ff) !important;
}

/* Metric Card Styling */
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 10px var(--accent-glow);
}

/* Scrollbar */
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PHYSICS ENGINE (Energy Drain)
# ─────────────────────────────────────────────
def simulate_energy_drain(speed_kmh, temp_celsius, ac_on, battery_health, base_capacity_kwh=75.0):
    k_drag = 0.0042
    drag_energy = k_drag * speed_kmh**2
    rolling_energy = 1.5 + 0.012 * speed_kmh
    base_electronics = 20.0
    
    temp_penalty = 1 + 0.008 * (20 - temp_celsius) if temp_celsius < 20 else 1 + 0.003 * (temp_celsius - 20)
    ac_penalty = (1500 / max(speed_kmh, 10)) if ac_on else 0.0
    
    consumption = ((drag_energy + rolling_energy + base_electronics) * temp_penalty) + ac_penalty
    usable_kwh = base_capacity_kwh * (battery_health / 100.0) * (1 / temp_penalty) * 0.95
    predicted_range = (usable_kwh * 1000) / consumption

    return {
        "consumption": round(consumption, 1),
        "usable_kwh": round(usable_kwh, 2),
        "range": round(predicted_range, 1),
        "penalty": round((temp_penalty - 1) * 100, 1)
    }

# ─────────────────────────────────────────────
#  XGBOOST MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def get_model():
    np.random.seed(42)
    N = 15000
    speeds = np.random.uniform(20, 160, N)
    temps = np.random.uniform(-20, 45, N)
    acs = np.random.randint(0, 2, N)
    health = np.random.uniform(60, 100, N)
    
    y = [simulate_energy_drain(s, t, bool(a), h)["range"] for s, t, a, h in zip(speeds, temps, acs, health)]
    X = pd.DataFrame({"speed": speeds, "temp": temps, "ac": acs, "health": health})
    
    model = xgb.XGBRegressor(n_estimators=100).fit(X, y)
    return model

model = get_model()

# ─────────────────────────────────────────────
#  SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ DRIVE PARAMETERS")
    speed = st.slider("Speed (km/h)", 20, 160, 100)
    temp = st.slider("Temperature (°C)", -20, 45, 25)
    health = st.slider("Battery Health (%)", 50, 100, 95)
    ac = st.toggle("Air Conditioning", value=True)

# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
st.title("🍃 ECO-PULSE AI")
st.subheader("High-Fidelity EV Range Predictor")

res = simulate_energy_drain(speed, temp, ac, health)

# Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Range", f"{res['range']} km")
col2.metric("Consumption", f"{res['consumption']} Wh/km")
col3.metric("Usable Battery", f"{res['usable_kwh']} kWh")
col4.metric("Temp Penalty", f"{res['penalty']}%")

# 🟢 THE GREEN GAUGE
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=res['range'],
    number={'suffix': " km", 'font': {'color': '#00FF41', 'family': 'Orbitron'}},
    gauge={
        'axis': {'range': [0, 600], 'tickcolor': "#00FF41"},
        'bar': {'color': "#00FF41"},
        'bgcolor': "rgba(0,0,0,0)",
        'steps': [{'range': [0, 600], 'color': "rgba(0, 255, 65, 0.1)"}],
    }
))
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
st.plotly_chart(fig, use_container_width=True)

