import time, json, threading
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Synthetic Demo)", layout="wide")

DATA_JSONL = "synthetic_conflict_stream.jsonl"
REPLAY_SPEED = 60
WINDOW_MINUTES = 120

if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=[
        "timestamp","region","lat","lon","risk_score","activity_index","supply_pressure","morale_index"
    ])
if "running" not in st.session_state:
    st.session_state.running = False

st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Replay Synthetic", "Upload CSV (append)"])
speed = st.sidebar.select_slider("Replay speed", [1, 10, 30, 60, 120], value=REPLAY_SPEED)
window = st.sidebar.select_slider("Window (minutes)", [30,60,120,240,480], value=WINDOW_MINUTES)
risk_w = st.sidebar.slider("Weight: Risk Incidents", 0.0, 1.0, 0.45, 0.05)
infra_w = st.sidebar.slider("Weight: Infra Stress", 0.0, 1.0, 0.25, 0.05)
supply_w = st.sidebar.slider("Weight: Supply Relief", 0.0, 1.0, 0.20, 0.05)
env_w   = st.sidebar.slider("Weight: Env Risk", 0.0, 1.0, 0.10, 0.05)

if mode == "Upload CSV (append)":
    up = st.sidebar.file_uploader("Upload events CSV", type=["csv"])
    if up:
        df_new = pd.read_csv(up)
        st.session_state.buffer = pd.concat([st.session_state.buffer, df_new], ignore_index=True)
        st.sidebar.success(f"Appended {len(df_new)} rows.")

colA, colB = st.columns([1,3])

def compute_composite(df):
    if df.empty: 
        return df.assign(composite=np.nan)
    rii = df["risk_score"]
    infra = 100 - df["supply_pressure"]*0.6
    supply_relief = df["supply_pressure"]
    env = np.clip(100 - df["morale_index"], 0, 100)
    comp = (risk_w*rii + infra_w*infra + supply_w*supply_relief + env_w*env)
    return df.assign(composite=np.clip(comp, 0, 100))

def replay_loop(path_str, speed_val):
    path = Path(path_str)
    if not path.exists(): 
        st.session_state.running = False
        return
    while st.session_state.running:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not st.session_state.running: break
                try:
                    ev = json.loads(line)
                    row = pd.DataFrame([ev])
                    st.session_state.buffer = pd.concat([st.session_state.buffer, row], ignore_index=True)
                except Exception:
                    pass
                time.sleep(1/float(speed_val))

with colA:
    st.markdown("### Live Controls")
    c1, c2 = st.columns(2)
    if c1.button("▶ Start Replay"):
        if not st.session_state.running:
            st.session_state.running = True
            threading.Thread(target=replay_loop, args=(DATA_JSONL, speed), daemon=True).start()
    if c2.button("⏹ Stop"):
        st.session_state.running = False

if not st.session_state.buffer.empty:
    if st.session_state.buffer["timestamp"].dtype == object:
        st.session_state.buffer["timestamp"] = pd.to_datetime(st.session_state.buffer["timestamp"], errors="coerce", utc=True)
    buf = st.session_state.buffer.dropna(subset=["timestamp"]).copy()
    buf.sort_values("timestamp", inplace=True)
    if len(buf) > 0:
        t_max = buf["timestamp"].max()
        buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=window)]
    buf = compute_composite(buf)

    alerts = buf[ (buf["composite"] > 75) | (buf["risk_score"] > 80) ].tail(10)

    with colA:
        st.metric("Events in window", len(buf))
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} — {a['region']}** • Composite {a['composite']:.1f} • Risk {a['risk_score']:.1f}")

    center_lat = float(buf["lat"].mean()) if len(buf) else 48.5
    center_lon = float(buf["lon"].mean()) if len(buf) else 36.5
    layer = pdk.Layer(
        "HeatmapLayer",
        data=buf,
        get_position='[lon, lat]',
        get_weight='composite',
        radiusPixels=40,
    )
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6),
        layers=[layer],
        tooltip={"text": "{region}\\ncomp={composite}\\nrisk={risk_score}\\nsupply={supply_pressure}\\nmorale={morale_index}"}
    )
    with colB:
        st.pydeck_chart(deck)
        st.dataframe(buf.tail(200))

else:
    with colB:
        st.info("No data yet. Start replay or upload CSV.")
