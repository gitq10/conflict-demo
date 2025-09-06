import os
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Demo)", layout="wide")

# ===== SETTINGS =====
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"
WINDOW_MINUTES = 120
BATCH_ROWS = 100       # how many rows to add each tick
REFRESH_MS = 1000      # refresh every 1 second

# ===== SESSION STATE =====
if "df_all" not in st.session_state:
    st.session_state.df_all = None
if "replay_index" not in st.session_state:
    st.session_state.replay_index = 0
if "running" not in st.session_state:
    st.session_state.running = True  # auto-start ON by default

# ===== HELPERS =====
def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

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

def load_data_once():
    if st.session_state.df_all is not None:
        return
    if not os.path.exists(DATA_FILE):
        st.sidebar.error(f"{DATA_FILE} NOT FOUND")
        return
    df = safe_read_csv(DATA_FILE)
    ensure_columns(df)
    if df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.sort_values("timestamp", inplace=True, na_position="last")
    st.session_state.df_all = df.reset_index(drop=True)
    st.sidebar.success(f"Loaded {len(df):,} rows")

# ===== LOAD DATA =====
load_data_once()
df_all = st.session_state.df_all

# ===== AUTO-REFRESH WHILE RUNNING =====
if st.session_state.running and df_all is not None:
    st.session_state.replay_index = min(
        st.session_state.replay_index + BATCH_ROWS,
        len(df_all)
    )
    st.autorefresh(interval=REFRESH_MS, key="auto_refresh")

# ===== MAIN VIEW =====
colA, colB = st.columns([1,3])

if df_all is not None and st.session_state.replay_index > 0:
    i = st.session_state.replay_index
    buf = df_all.iloc[:i].copy()

    # Window filter
    if buf["timestamp"].notna().any():
        t_max = buf["timestamp"].max()
        buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=WINDOW_MINUTES)]

    buf = compute_composite(buf)

    # Status panel
    with colA:
        st.markdown("### Status")
        st.metric("Replay index", f"{i:,} / {len(df_all):,}")
        st.progress(i / len(df_all))

        alerts = buf[(buf["composite"] > 75) | (buf["risk_score"] > 80)] \
                   .tail(5)[["timestamp","region","composite","risk_score"]]
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} – {a['region']}** "
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
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6),
        layers=[layer],
        tooltip={"text": "{region}\\nComposite={composite:.1f}\\nRisk={risk_score:.1f}"}
    )
    with colB:
        st.pydeck_chart(deck)
        st.dataframe(buf.tail(20), use_container_width=True)
else:
    with colB:
        st.warning("No data yet. Check CSV file.")
