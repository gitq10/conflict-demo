import os, time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Synthetic Demo)", layout="wide")

# ===== SETTINGS =====
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"  # must be in the SAME folder as app.py
WINDOW_MINUTES = 120
REPLAY_SPEED = 1200  # only used if you click Start Replay; 1200 ≈ 0.05s/row

# ===== SESSION =====
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(columns=[
        "timestamp","region","lat","lon",
        "risk_score","activity_index","supply_pressure","morale_index"
    ])
if "running" not in st.session_state:
    st.session_state.running = False

# ===== HELPERS =====
def safe_read_csv(path: str) -> pd.DataFrame:
    # Try common read modes to survive weird encodings/delimiters
    tries = [
        dict(encoding="utf-8-sig"),
        dict(encoding="utf-8", engine="python"),
        dict(encoding="latin1", engine="python"),
    ]
    last_err = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            return df
        except Exception as e:
            last_err = e
    raise last_err

def ensure_columns(df: pd.DataFrame):
    expected = {"timestamp","region","lat","lon",
                "risk_score","activity_index","supply_pressure","morale_index"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.assign(composite=np.nan)
    rii = df["risk_score"]
    infra = 100 - df["supply_pressure"]*0.6
    supply_relief = df["supply_pressure"]
    env = np.clip(100 - df["morale_index"], 0, 100)
    comp = (0.45*rii + 0.25*infra + 0.20*supply_relief + 0.10*env)
    return df.assign(composite=np.clip(comp, 0, 100))

def replay_loop(speed_val: int):
    try:
        df = safe_read_csv(DATA_FILE)
        ensure_columns(df)
        st.sidebar.success(f"Loaded {len(df):,} rows from {DATA_FILE}")
    except Exception as e:
        st.sidebar.error(f"Read error: {e}")
        st.session_state.running = False
        return

    for i, ev in df.iterrows():
        if not st.session_state.running:
            break
        st.session_state.buffer = pd.concat(
            [st.session_state.buffer, pd.DataFrame([ev])],
            ignore_index=True
        )
        if i % 1000 == 0:
            st.sidebar.write(f"Replayed {i:,} rows…")
        time.sleep(1/float(speed_val))

# ===== SIDEBAR =====
st.sidebar.title("Controls")
window = st.sidebar.select_slider("Window (minutes)",
                                  [30,60,120,240,480,720], value=WINDOW_MINUTES)

with st.sidebar.expander("File Debug", expanded=False):
    st.write("**Working directory:**", os.getcwd())
    try:
        files = os.listdir(".")
        st.write("**Files here:**", files[:100])
        if os.path.exists(DATA_FILE):
            st.success(f"{DATA_FILE} exists ✓  size: {os.path.getsize(DATA_FILE):,} bytes")
        else:
            st.error(f"{DATA_FILE} NOT FOUND next to app.py")
    except Exception as e:
        st.error(f"Listing error: {e}")

st.sidebar.markdown("---")
cA, cB = st.sidebar.columns(2)
start = cA.button("▶ Start Replay")
stop  = cB.button("⏹ Stop")
if start and not st.session_state.running:
    st.session_state.running = True
    import threading
    threading.Thread(target=replay_loop, args=(REPLAY_SPEED,), daemon=True).start()
if stop:
    st.session_state.running = False

# ===== AUTO-LOAD ON STARTUP (zero clicks) =====
if st.session_state.buffer.empty and os.path.exists(DATA_FILE):
    try:
        df_all = safe_read_csv(DATA_FILE)
        ensure_columns(df_all)
        # Parse timestamp once
        if df_all["timestamp"].dtype == object:
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce", utc=True)
        st.session_state.buffer = df_all.copy()
        st.sidebar.success(f"Auto-loaded {len(df_all):,} rows from {DATA_FILE}")
    except Exception as e:
        st.sidebar.error(f"Auto-load error: {e}")

# ===== MAIN VIEW =====
colA, colB = st.columns([1,3])

if not st.session_state.buffer.empty:
    buf = st.session_state.buffer.copy()
    # Make sure timestamps are datetime
    if buf["timestamp"].dtype == object:
        buf["timestamp"] = pd.to_datetime(buf["timestamp"], errors="coerce", utc=True)
    buf.sort_values("timestamp", inplace=True, na_position="last")

    # Filter to last N minutes (if timestamps exist)
    if len(buf) > 0 and buf["timestamp"].notna().any():
        t_max = buf["timestamp"].max()
        if pd.notna(t_max):
            buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=window)]

    buf = compute_composite(buf)

    # Alerts
    alerts = buf[(buf["composite"] > 75) | (buf["risk_score"] > 80)] \
               .tail(12)[["timestamp","region","composite","risk_score"]]

    with colA:
        st.markdown("### Status")
        st.metric("Events in window", len(buf))
        st.write("Replay running:", "✅" if st.session_state.running else "⏸️")
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} – {a['region']}** • "
                         f"Composite {a['composite']:.1f} • Risk {a['risk_score']:.1f}")

    # Map
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
        st.warning("I can’t find or load your data yet.\n\n"
                   "• Make sure **synthetic_conflict_risk_90d_15min.csv** is in the **same folder** as app.py\n"
                   "• Then click the **Rerun** button (⋮ in the top-right)\n"
                   "• Open **File Debug** in the sidebar to confirm the file exists")
