import time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Synthetic Demo)", layout="wide")

# ---- SETTINGS ----
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"  # your CSV name
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
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        st.sidebar.success(f"Loaded {len(df):,} rows from {DATA_FILE}")
        print(f"[replay] loaded {len(df)} rows from {DATA_FILE}")
    except Exception as e:
        st.error(f"Could not read {DATA_FILE}: {e}")
        print(f"[replay] ERROR reading file: {e}")
        st.session_state.running = False
        return

    expected = {"timestamp","region","lat","lon","risk_score","activity_index","supply_pressure","morale_index"}
    missing = expected - set(df.columns)
    if missing:
        st.error(f"CSV missing columns: {missing}")
        st.session_state.running = False
        return

    for i, ev in df.iterrows():
        if not st.session_state.running:
            print("[replay] stopped")
            break
        st.session_state.buffer = pd.concat([st.session_state.buffer, pd.DataFrame([ev])], ignore_index=True)
        if i % 500 == 0:
            st.sidebar.write(f"Replayed {i:,} rows…")
            print(f"[replay] {i} rows appended")
        time.sleep(1/float(speed_val))

# ---- LAYOUT ----
colA, colB = st.columns([1,3])

with colA:
    st.markdown("### Live Controls")
    c1, c2 = st.columns(2)
    if c1.button("▶ Start Replay"):
        if not st.session_state.running:
            st.session_state.running = True
            import threading
            threading.Thread(target=replay_loop, args=(speed,), daemon=True).start()
    if c2.button("⏹ Stop"):
        st.session_state.running = False

# ---- MAIN VIEW ----
if not st.session_state.buffer.empty:
    buf = st.session_state.buffer.copy()

    if buf["timestamp"].dtype == object:
        buf["timestamp"] = pd.to_datetime(buf["timestamp"], errors="coerce", utc=True)
    buf.sort_values("timestamp", inplace=True, na_position="last")

    if len(buf) > 0 and buf["timestamp"].notna().any():
        t_max = buf["timestamp"].max()
        buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=window)]

    buf = compute_composite(buf)

    alerts = buf[(buf["composite"] > 75) | (buf["risk_score"] > 80)].tail(10)

    with colA:
        st.metric("Events in window", len(buf))
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} — {a['region']}** "
                         f"• Composite {a['composite']:.1f} • Risk {a['risk_score']:.1f}")

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
        tooltip={"text": "{region}\\nComposite={composite:.1f}\\nRisk={risk_score:.1f}\\nSupply={supply_pressure:.1f}\\nMorale={morale_index:.1f}"}
    )
    with colB:
        st.pydeck_chart(deck)
        st.dataframe(buf.tail(300), use_container_width=True)
else:
    with colB:
        st.info("No data yet. Click ▶ Start Replay. (If stuck: ⋮ → Clear cache → Rerun)")
