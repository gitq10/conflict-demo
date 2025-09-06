import os, time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Demo)", layout="wide")

# ========= SETTINGS =========
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"  # must sit next to app.py
TICK_SECONDS = 1.0                                   # refresh cadence for demo
DEFAULT_WINDOW_MIN = 120

# ========= SESSION STATE ========
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None       # 15-min source
if "df_dense" not in st.session_state:
    st.session_state.df_dense = None     # 1-min interpolated
if "use_dense" not in st.session_state:
    st.session_state.use_dense = True    # default to "wow" mode
if "replay_index" not in st.session_state:
    st.session_state.replay_index = 0
if "running" not in st.session_state:
    st.session_state.running = True      # auto-play ON

# ========= HELPERS =========
def read_csv_safely(path: str) -> pd.DataFrame:
    for kw in ({"encoding": "utf-8-sig"},
               {"encoding": "utf-8", "engine": "python"},
               {"encoding": "latin1", "engine": "python"}):
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

@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    df = read_csv_safely(path)
    ensure_columns(df)
    if df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.sort_values(["timestamp","region"], inplace=True, na_position="last")
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=True)
def make_dense_1min(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Interpolate to 1-minute per region (lat/lon carried forward)
    num_cols = ["risk_score","activity_index","supply_pressure","morale_index"]
    keep_cols = ["lat","lon"]
    out = []
    for region, g in df_raw.groupby("region", sort=False):
        g = g.set_index("timestamp").sort_index()
        dense = g[keep_cols].ffill().bfill().resample("1T").ffill()
        interpolated = g[num_cols].interpolate(method="time").resample("1T").interpolate("time")
        dense[num_cols] = interpolated[num_cols]
        dense["region"] = region
        out.append(dense.reset_index())
    df_dense = pd.concat(out, ignore_index=True)
    df_dense = df_dense[["timestamp","region","lat","lon"] + num_cols].sort_values(["timestamp","region"])
    return df_dense.reset_index(drop=True)

# ========= LOAD DATA =========
if st.session_state.df_raw is None:
    if not os.path.exists(DATA_FILE):
        st.stop()
    st.session_state.df_raw = load_raw(DATA_FILE)

if st.session_state.df_dense is None:
    st.session_state.df_dense = make_dense_1min(st.session_state.df_raw)

# Which table to use
df_raw = st.session_state.df_raw
df_dense = st.session_state.df_dense
df_all = df_dense if st.session_state.use_dense else df_raw

regions_count = df_all["region"].nunique()
total_rows = len(df_all)

# ========= SIDEBAR =========
st.sidebar.title("Controls")

with st.sidebar.expander("File Debug", expanded=False):
    st.write("**Working directory:**", os.getcwd())
    st.write("**Files here:**", os.listdir(".")[:50])
    if os.path.exists(DATA_FILE):
        st.success(f"{DATA_FILE} exists ✓ size: {os.path.getsize(DATA_FILE):,} bytes")
    else:
        st.error(f"{DATA_FILE} NOT FOUND next to app.py")

gran = st.sidebar.selectbox(
    "Granularity",
    ["1-min (interpolated)", "15-min (raw)"],
    index=0 if st.session_state.use_dense else 1
)
st.session_state.use_dense = gran.startswith("1-min")
df_all = df_dense if st.session_state.use_dense else df_raw
regions_count = df_all["region"].nunique()
total_rows = len(df_all)

minutes_per_tick = st.sidebar.select_slider(
    "Minutes per tick (demo speed)",
    options=[1,2,5,10,15,30,60],
    value=1
)
rows_per_tick = max(minutes_per_tick * regions_count, 1)

window = st.sidebar.select_slider(
    "Window (minutes)",
    options=[15,30,60,120,240,480,720],
    value=DEFAULT_WINDOW_MIN
)

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Start"): st.session_state.running = True
if c2.button("⏸ Stop"):  st.session_state.running = False
if c3.button("↺ Reset"):
    st.session_state.running = False
    st.session_state.replay_index = 0

# ========= PRELOAD FIRST SCREEN =========
if st.session_state.replay_index == 0:
    preload_minutes = 10
    st.session_state.replay_index = min(preload_minutes * regions_count, total_rows)

# ========= ADVANCE POINTER =========
if st.session_state.running:
    st.session_state.replay_index = min(st.session_state.replay_index + rows_per_tick, total_rows)

# ========= BUILD CURRENT VIEW =========
i = st.session_state.replay_index
colA, colB = st.columns([1,3])

if i > 0:
    buf = df_all.iloc[:i].copy()
    if buf["timestamp"].notna().any():
        t_max = buf["timestamp"].max()
        if pd.notna(t_max):
            buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=window)]
    buf = compute_composite(buf)

    with colA:
        st.markdown("### Status")
        st.metric("Replay index", f"{i:,} / {total_rows:,}")
        st.write("Playing:", "✅" if st.session_state.running else "⏸️")
        st.write("Granularity:", "1-min" if st.session_state.use_dense else "15-min")
        st.write("Minutes/tick:", minutes_per_tick)
        st.progress(i / max(total_rows, 1))

        alerts = buf[(buf["composite"] > 75) | (buf["risk_score"] > 80)] \
                   .tail(10)[["timestamp","region","composite","risk_score"]]
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} – {a['region']}** "
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
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6),
        layers=[layer],
        tooltip={"text": "{region}\\nComposite={composite:.1f}\\nRisk={risk_score:.1f}\\nSupply={supply_pressure:.1f}\\nMorale={morale_index:.1f}"}
    )
    with colB:
        st.pydeck_chart(deck)
        st.dataframe(buf.tail(30), use_container_width=True)
else:
    with colB:
        st.info("Click ▶ Start to play. If nothing moves, verify the CSV exists in the repo root.")

# ====== AUTO-RERUN (1s), only while playing ======
if st.session_state.running:
    time.sleep(TICK_SECONDS)
    st.rerun()   # <— use st.rerun(), not experimental_rerun
