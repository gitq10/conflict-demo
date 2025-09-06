import os, io, time, json
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Conflict Early-Warning (Demo Suite)", layout="wide")

# ========= SETTINGS =========
DATA_FILE = "synthetic_conflict_risk_90d_15min.csv"   # must sit next to app.py
TICK_SECONDS = 1.0                                    # refresh cadence for demo
DEFAULT_WINDOW_MIN = 60                               # smaller = more visible motion
DEFAULT_GRAN = "1-min (interpolated)"                 # wow-mode by default

# ========= SESSION =========
ss = st.session_state
def _init():
    if "df_raw" not in ss:       ss.df_raw = None          # 15-min source
    if "df_dense" not in ss:     ss.df_dense = None        # 1-min interpolated
    if "df_work" not in ss:      ss.df_work = None         # active, mutable copy
    if "use_dense" not in ss:    ss.use_dense = True
    if "gran_label" not in ss:   ss.gran_label = DEFAULT_GRAN
    if "replay_index" not in ss: ss.replay_index = 0
    if "running" not in ss:      ss.running = True         # auto-play ON
    if "last_mode" not in ss:    ss.last_mode = None
_init()

# ========= HELPERS =========
def read_csv_safely(path: str) -> pd.DataFrame:
    for kw in ({"encoding": "utf-8-sig"},
               {"encoding": "utf-8", "engine": "python"},
               {"encoding": "latin1", "engine": "python"}):
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            pass
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
    df_dense = df_dense[["timestamp","region","lat","lon"] + num_cols] \
                     .sort_values(["timestamp","region"])
    return df_dense.reset_index(drop=True)

def set_active_mode(use_dense: bool):
    # Prepare a fresh, mutable working copy for this mode
    base = ss.df_dense if use_dense else ss.df_raw
    ss.df_work = base.copy()
    ss.use_dense = use_dense
    ss.last_mode = "dense" if use_dense else "raw"
    ss.replay_index = 0

def rows_per_minute(regions_count: int) -> int:
    # 1 row per region per minute (dense). Raw is every 15 minutes.
    return regions_count if ss.use_dense else max(regions_count // 15, 1)

def minutes_to_rows(mins: int, regions_count: int) -> int:
    if ss.use_dense:
        return mins * regions_count
    # raw 15-min steps
    steps = (mins // 15) * regions_count
    return max(steps, 0)

def now_timestamp():
    if ss.replay_index == 0: 
        return ss.df_work["timestamp"].min()
    i = min(ss.replay_index-1, len(ss.df_work)-1)
    return ss.df_work.iloc[i]["timestamp"]

# ========= LOAD DATA =========
if ss.df_raw is None:
    if not os.path.exists(DATA_FILE):
        st.stop()
    ss.df_raw = load_raw(DATA_FILE)
if ss.df_dense is None:
    ss.df_dense = make_dense_1min(ss.df_raw)

# initialize active mode on first load
if ss.df_work is None:
    set_active_mode(use_dense=True)

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
    index=0 if ss.use_dense else 1
)
if gran != ss.gran_label:
    set_active_mode(use_dense = gran.startswith("1-min"))
    ss.gran_label = gran

regions_count = ss.df_work["region"].nunique()
total_rows    = len(ss.df_work)

minutes_per_tick = st.sidebar.select_slider(
    "Minutes per tick (demo speed)",
    options=[1,2,5,10,15,30,60],
    value=1
)
rows_per_tick = max(minutes_to_rows(minutes_per_tick, regions_count), 1)

window = st.sidebar.select_slider(
    "Window (minutes)",
    options=[15,30,60,120,240,480,720],
    value=DEFAULT_WINDOW_MIN
)

# --- Client CSV upload (merge live) ---
up = st.sidebar.file_uploader("Append CSV (same columns)", type=["csv"])
if up is not None:
    try:
        new = pd.read_csv(up)
        ensure_columns(new)
        if new["timestamp"].dtype == object:
            new["timestamp"] = pd.to_datetime(new["timestamp"], errors="coerce", utc=True)
        ss.df_work = pd.concat([ss.df_work, new], ignore_index=True) \
                       .sort_values(["timestamp","region"]).reset_index(drop=True)
        total_rows = len(ss.df_work)
        st.sidebar.success(f"Merged {len(new):,} rows.")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

# --- Hotspot injector (sales button) ---
st.sidebar.markdown("### Inject Hotspot")
col_h1, col_h2 = st.sidebar.columns(2)
hot_region = col_h1.selectbox("Region", sorted(ss.df_work["region"].unique()))
hot_mins   = col_h2.select_slider("Duration (min)", [5,10,15,20,30,45,60], value=20)
hot_boost  = st.sidebar.slider("Magnitude (+risk)", 5, 50, 25, step=5)
if st.sidebar.button("🔥 Inject"):
    t0 = now_timestamp()
    if pd.isna(t0):
        st.sidebar.warning("Start the replay first.")
    else:
        t1 = t0 + pd.Timedelta(minutes=hot_mins)
        mask = (ss.df_work["region"]==hot_region) & \
               (ss.df_work["timestamp"]>=t0) & (ss.df_work["timestamp"]<=t1)
        ss.df_work.loc[mask, "risk_score"] = np.clip(ss.df_work.loc[mask, "risk_score"] + hot_boost, 0, 100)
        # simple coupled effects
        ss.df_work.loc[mask, "supply_pressure"] = np.clip(ss.df_work.loc[mask, "supply_pressure"] - hot_boost*0.4, 0, 100)
        ss.df_work.loc[mask, "morale_index"]    = np.clip(ss.df_work.loc[mask, "morale_index"] - hot_boost*0.2, 0, 100)
        st.sidebar.success(f"Injected hotspot in {hot_region} for {hot_mins} minutes.")

# --- Fast Forward ---
st.sidebar.markdown("### Fast-Forward")
ff1, ff2, ff3 = st.sidebar.columns(3)
def fast_forward(mins: int):
    inc = minutes_to_rows(mins, regions_count)
    ss.replay_index = min(ss.replay_index + inc, total_rows)
if ff1.button("+5m"):   fast_forward(5)
if ff2.button("+30m"):  fast_forward(30)
if ff3.button("+2h"):   fast_forward(120)

# --- Playback controls ---
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Start"): ss.running = True
if c2.button("⏸ Stop"):  ss.running = False
if c3.button("↺ Reset"):
    ss.running = False
    ss.replay_index = 0

# ========= PRELOAD FIRST VIEW =========
if ss.replay_index == 0:
    preload_m = 10
    ss.replay_index = min(minutes_to_rows(preload_m, regions_count), total_rows)

# ========= ADVANCE POINTER =========
if ss.running:
    ss.replay_index = min(ss.replay_index + rows_per_tick, total_rows)

# ========= BUILD CURRENT VIEW =========
i = ss.replay_index
colA, colB = st.columns([1,3])

if i > 0:
    buf = ss.df_work.iloc[:i].copy()
    if buf["timestamp"].notna().any():
        t_max = buf["timestamp"].max()
        if pd.notna(t_max):
            buf = buf[buf["timestamp"] >= t_max - pd.Timedelta(minutes=window)]
    buf = compute_composite(buf)

    # --- Recommendations (simple rules over current window) ---
    recs = []
    if not buf.empty:
        grp = buf.groupby("region").agg(
            comp=("composite","mean"),
            risk=("risk_score","mean"),
            supply=("supply_pressure","mean"),
            morale=("morale_index","mean"),
            events=("region","count")
        ).reset_index()
        # Rule 1: high comp & low supply
        for _, r in grp.iterrows():
            if r["comp"]>80 and r["supply"]<40:
                recs.append({"region": r["region"], "priority": 1,
                             "action": "Open supply corridor; push 2 logistics teams",
                             "why": f"Composite {r['comp']:.0f} with low supply {r['supply']:.0f}"})
            # Rule 2: high risk surge
            if r["risk"]>85:
                recs.append({"region": r["region"], "priority": 2,
                             "action": "Deploy QRF & ISR; jam EW in sector",
                             "why": f"Risk {r['risk']:.0f} sustained"})
            # Rule 3: morale collapse
            if r["morale"]<35 and r["comp"]>70:
                recs.append({"region": r["region"], "priority": 3,
                             "action": "Rotate units & psyops messaging",
                             "why": f"Morale {r['morale']:.0f} with Composite {r['comp']:.0f}"})
        recs = sorted(recs, key=lambda x: x["priority"])[:6]

    # --- Status / Alerts / Recommendations ---
    with colA:
        st.markdown("### Status")
        st.metric("Replay index", f"{i:,} / {total_rows:,}")
        st.write("Playing:", "✅" if ss.running else "⏸️")
        st.write("Granularity:", "1-min" if ss.use_dense else "15-min")
        st.write("Minutes/tick:", minutes_per_tick)
        st.progress(i / max(total_rows, 1))

        alerts = buf[(buf["composite"] > 80) | (buf["risk_score"] > 85)] \
                   .tail(10)[["timestamp","region","composite","risk_score"]]
        if not alerts.empty:
            st.markdown("#### Alerts")
            for _, a in alerts.iterrows():
                st.write(f"**{a['timestamp']} – {a['region']}** • "
                         f"Composite {a['composite']:.1f} • Risk {a['risk_score']:.1f}")

        if recs:
            st.markdown("#### Recommendations")
            for r in recs:
                st.write(f"**{r['region']}** — {r['action']}  \n*Why:* {r['why']}")

            # Downloads
            rec_df = pd.DataFrame(recs)
            csv_bytes = rec_df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(recs, indent=2).encode("utf-8")
            st.download_button("📥 Download actions (CSV)", data=csv_bytes, file_name="actions.csv", mime="text/csv")
            st.download_button("📥 Download actions (JSON)", data=json_bytes, file_name="actions.json", mime="application/json")

    # --- Map + Table ---
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

# ====== AUTO-RERUN (1s) while playing ======
if ss.running:
    time.sleep(TICK_SECONDS)
    st.rerun()
