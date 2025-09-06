import time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Synthetic Demo)", layout="wide")

# ---- SETTINGS (match your file name exactly) ----
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"
REPLAY_SPEED = 600   # higher = faster (600 → ~0.1s per row)
WINDOW_MINUTES = 120

# ---- SESSION STATE ----
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=[
        "timestamp","region","lat","lon","risk_score","activity_index","supply_pressure","morale_index"
    ])
if "running" not in st.session_state:
    st.session_state.running = False

# ---- SIDEBAR CONTROLS ----
st.sidebar.title("Controls")
speed = st.sidebar.select_slider("Replay speed", [1, 10, 60, 120, 300, 600, 1200], value=REPLAY_SPEED)
window = st.sidebar.select_slider("Window (minutes)", [30,60,120,240,480,720], value=WINDOW_MINUTES)
risk_w = st.sidebar.slider("Weight: Risk Incidents", 0.0, 1.0, 0.45, 0.05)
infra_w = st.sidebar.slider("Weight: Infra Stress", 0.0, 1.0, 0.25, 0.05)
supply_w = st.sidebar.slider("Weight: Supply Relief", 0.0, 1.0, 0.20, 0.05)
env_w   = st.sidebar.slider("Weight: Env Risk", 0.0, 1.0, 0.10, 0.05)

# ---- HELPERS ----
def compute_composite(df):
    if df.empty:
        return df.assign(composite=np.nan)
    rii = df["risk_score"]
    infra = 100 - df["supply_pressure"]*0.6
    supply_relief = df["supply_pressure"]
    env = np.clip(100 - df["morale_index"], 0, 100)
    comp = (risk_w*rii + infra_w*infra + supply_w*supply_relief + env_w*env)
    return df.assign(composite=np.clip(comp, 0, 100))

def replay_loop(speed_val):
    """Read CSV and append rows quickly; write progress to sidebar + logs."""
    try:
        # robust read (handles BOM & common CSV quirks)
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        st.sidebar.success(f"Loaded {le
