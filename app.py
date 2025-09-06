import os, time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Demo)", layout="wide")

# ===== SETTINGS =====
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"   # must be in the same folder as app.py
WINDOW_MINUTES = 120
BATCH_ROWS = 100       # rows added each tick
TICK_SECONDS = 1.0     # 1-second demo cadence

# ===== SESSION =====
if "df_all" not in st.session_state:
    st.session_state.df_all = None
if "replay_index" not in st.session_state:
    st.session_state.replay_index = 0
if "running" not in st.session_state:
    st.session_state.running = True  # auto-start ON
if "preloaded" not in st.session_state:
    st.session_state.preloaded = False

# ===== HELPERS =====
def read_csv_safely(path: str) -> pd.DataFrame:
    for kw in ({"encoding":"utf-8-sig"},
               {"encoding":"utf-8", "engine":"python"},
               {"encoding":"latin1", "engine":"python"}):
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            last_err = kw
    raise RuntimeError(f"Could not read {path} with common encodings")

def ensure_columns(df: pd.DataFrame):
    expected = {"timestamp","region","lat","lon",
                "risk_score","activity_index","supply_pressure","morale_index"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df.assign(composite=np.nan)
    rii = df["risk_score"]
    infra = 100 - df["supply_pressure"]*0.6
    supply_relief = df["supply_pressure"]
    env = np.clip(100 - df["morale_index"], 0, 100)
    comp = (0.45*rii + 0.25*infra + 0.20*supply_relief + 0.10*env)
    return df.assign(composite=np.clip(comp, 0, 100))

def load_data_once():
    if st.session_state.df_all is not None:
        return
    if not os.path.exists(DATA_FILE):
        st.sidebar.error(f"{DATA_FILE} NOT FOUND next to app.py")
        return
    df = read_csv_safely(DATA_FILE)
    ensure_columns(df)
    if df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.sort_values("timestamp", inplace=True, na_position="last")
    st.session_state.df_all = df.reset_index(drop=True)
    st.sidebar.success(f"Loaded {len(df):,} rows from {DATA_FILE}")

# ===== LOAD DATA =====
load_data_once()
df_all = st.session_state.df_all

# ===== SIDEBAR =====
st.sidebar.title("Controls")

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

col_btn = st.sidebar.columns(3)
if col_btn[0].button("▶ Start"):
    st.session_state.running = True
if col_btn[1].button("⏸ Stop"):
    st.session_state.running = False
if col_btn[2].button("↺ Reset"):
    st.session_state.running = False
    st.session_state.replay_index = 0
    st.session_state.preloaded = False

window = st.sidebar.select_slider(
    "Window (minutes)",
    [30, 60, 120, 240, 480, 720],
    value=WINDOW_MINUTES
)

# ===== PRELOAD SOME ROWS SO IT'S INSTANTLY ALIVE =====
if df_all is not None and not st.session_state.preloaded:
    preload = min(20 * BATCH_ROWS, len(df_all))  # ~20 seconds of data
    st.session_state.replay_index = preload
    st.session_state.preloaded = True

# ===== BUILD CURRENT VIEW (from 0..replay_index) =====
colA, colB = st.columns([1,3])

if df_all is not None and st.session_state.replay_index > 0:
    i = st.session_state.replay_index
    buf = df_all.iloc[:i].copy()

    # Apply time window
    if buf["timestamp"].notna().any():
        t_max = buf["timestamp"].max()
        if pd.notna(t_max):
            buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=window)]

    buf = compute_composite(buf)

    with colA:
        st.markdown("### Status")
        st.metric("Replay index", f"{i:,} / {len(df_all):,}")
        st.write("Playing:", "✅" if st.session_state.running else "⏸️")
        st.progress(i / max(len(df_all), 1))

        alerts = buf[(buf["composite"] > 75) | (buf["risk_score"] > 80)] \
                   .tail(8)[["timestamp","region","composite","risk_score"]]
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} – {a['region']}** "
                         f"• Composite {a['composite']:.1f} • Risk {a['risk_score']:.1f}")

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
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6),
        layers=[layer],
        tooltip={"text": "{region}\\nComposite={composite:.1f}\\nRisk={risk_score:.1f}\\nSupply={supply_pressure:.1f}\\nMorale={morale_index:.1f}"}
    )
    with colB:
        st.pydeck_chart(deck)
        st.dataframe(buf.tail(25), use_container_width=True)
else:
    with colB:
        if df_all is None:
            st.warning("I can’t find or load your data yet. Put "
                       "`synthetic_conflict_risk_90d_15min.csv` next to `app.py`, then rerun.")
        else:
            st.info("Click ▶ Start to begin the live replay.")

# ===== AUTO-ADVANCE + AUTO-RERUN =====
if st.session_state.running and df_all is not None:
    # advance pointer (bounded)
    st.session_state.replay_index = min(
        st.session_state.replay_index + BATCH_ROWS,
        len(df_all)
    )
    # wait a second so humans can see it, then rerun
    time.sleep(TICK_SECONDS)
    st.experimental_rerun()
