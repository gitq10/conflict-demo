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
        # carry positions; interpolate indices
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
        st.stop()  # Streamlit will show the warning block below
    st.session_state.df_raw = load_raw(DATA_FILE)

if st.session_state.df_dense is None:
    # Build dense table once (about 15x more rows than raw)
    st.session_state.df_dense = make_dense_1min(st.session_state.df_raw)

# Choose which table to use
df_raw = st.session_state.df_raw
df_dense = st.session_state.df_dense
df_all = df_dense if st.session_state.use_dense else df_raw

# Some counts
regions_count = df_all["r]()
