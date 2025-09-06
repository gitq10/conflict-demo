import os, time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Synthetic Demo)", layout="wide")

# ---- SETTINGS ----
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"  # <-- must be in repo root
REPLAY_SPEED = 1200   # faster = more rows per second (1200 ≈ 0.05s/row)
WINDOW_MINUTES = 120

# ---- SESSION STATE ----
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=[
        "timestamp","region","lat","lon","risk_score","activity_index","supply_pressure","morale_index"
    ])
if "running" not in st.session_state:
    st.session_state.running = False

# ---- UTILITIES ----
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV once (handles common quirks)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Basic column sanity
    expected = {"timestamp","region","lat","lon","risk_score","activity_index","supply_pressure","morale_index"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df

def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(composite=np.nan)
    rii = df["risk_score"]
    infra = 100 - df["supply_pressure"]*0.6
    supply_relief = df["supply_pressure"]
    env = np.clip(100 - df["morale_index"], 0, 100)
    comp = (0.45*rii + 0.25*infra + 0.20*supply_relief + 0.10*env)
    return df.assign(composite=np.clip(comp, 0, 100))

def replay_loop(speed_val: int):
    """Append rows quickly; write progress to sidebar + logs."""
    try:
        df = load_csv(DATA_FILE)
        st.sidebar.success(f"Loaded {len(df):,} rows from {DATA_FILE}")
        print(f"[replay] loaded {len(df)} rows from {DATA_FILE}")
    except Exception as e:
        st.sidebar.error(f"Read error: {e}")
        print(f"[replay] ERROR reading file: {e}")
        st.session_state.running = False
        return

    for i, ev in df.iterrows():
        if not st.session_state.running:
            print("[replay] stopped")
            break
        st.session_state.buffer = pd.concat(
            [st.session_state.buffer, pd.DataFrame([ev])],
            ignore_index=True
        )
        if i % 1000 == 0:
            st.sidebar.write(f"Replayed {i:,} rows…")
            print(f"[replay] {i} rows appended")
        time.sleep(1/float(speed_val))

# ---- SIDEBAR ----
st.sidebar.title("Controls")
speed = st.sidebar.select_slider("Replay speed", [1, 10, 60, 120, 300, 600, 1200], value=REPLAY_SPEED)
window = st.sidebar.select_slider("Window (minutes)", [30,60,120,240,480,720], value=WINDOW_MINUTES)

# Quick debug panel so you can SEE what Streamlit sees
with st.sidebar.expander("File Debug (click to open)", expanded=False):
    st.write("**Working directory:**", os.getcwd())
    try:
        files = os.listdir(".")
        st.write("**Files here:**", files[:50])
        if os.path.exists(DATA_FILE):
            st.success(f"{DATA_FILE} exists ✓  size: {os.path.getsize(DATA_FILE):,} bytes")
        else:
            st.error(f"{DATA_FILE} NOT FOUND in repo root")
    except Exception as e:
        st.error(f"Listing error: {e}")

st.sidebar.markdown("---")
cA, cB, cC = st.sidebar.columns(3)
start = cA.button("▶ Start Replay")
stop = cB.button("⏹ Stop")
snap = cC.button("⚡ Load Snapshot Now")

if start and not st.session_state.running:
    st.session_state.running = True
    import threading
    threading.Thread(target=replay_loop, args=(speed,), daemon=True).start()

if stop:
    st.session_state.running = False

# Snapshot load (IMMEDIATE data without replay/thread)
if snap:
    try:
        df_all_
