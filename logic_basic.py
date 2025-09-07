import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= Utilities =================

def _ensure_tz_series(series, tz="Asia/Shanghai"):
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        s = s.dt.tz_localize(tz)
    else:
        s = s.dt.tz_convert(tz)
    return s

def _infer_tick_size_from_prices(prices, max_unique=6000):
    s = pd.Series(prices, dtype=float).dropna()
    if s.nunique() < 2: return np.nan
    vals = np.sort(s.unique())
    if len(vals) > max_unique:
        vals = np.sort(np.random.choice(vals, size=max_unique, replace=False))
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0: return np.nan
    tick = np.quantile(diffs, 0.1)
    return float(pd.Series([tick]).round(6).iloc[0])

def _zscore(x, window, min_periods=None):
    if min_periods is None:
        min_periods = max(30, window // 6)
    s = pd.Series(x, dtype=float)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mu) / (sd.replace(0, np.nan))
    return z.values

def _label_bucket_cn_a(ts):
    t = ts.timetz()
    tz = "Asia/Shanghai"
    def tt(s): return pd.to_datetime(s).tz_localize(tz).timetz()
    cuts = { "0915": tt("09:15"), "0930": tt("09:30"), "0935": tt("09:35"),
             "1125": tt("11:25"), "1130": tt("11:30"), "1300": tt("13:00"),
             "1305": tt("13:05"), "1445": tt("14:45"), "1457": tt("14:57"),
             "1500": tt("15:00") }
    if t >= cuts["0915"] and t < cuts["0930"]: return "auction_open"
    if t >= cuts["0930"] and t < cuts["0935"]: return "open"
    if t >= cuts["0935"] and t < cuts["1125"]: return "morning"
    if t >= cuts["1125"] and t < cuts["1130"]: return "pre_lunch_end"
    if t >= cuts["1130"] and t < cuts["1300"]: return "lunch_break"
    if t >= cuts["1300"] and t < cuts["1305"]: return "mini_open"
    if t >= cuts["1305"] and t < cuts["1445"]: return "afternoon"
    if t >= cuts["1445"] and t < cuts["1457"]: return "last15"
    if t >= cuts["1457"] and t <= cuts["1500"]: return "close_auction"
    return "off"

BUCKET_CONF = {
    "auction_open": {"ban": True}, "lunch_break": {"ban": True},
    "close_auction": {"ban": True}, "off": {"ban": True},
    "open": {"S_mul": 1.10, "alpha_add": 0.0002, "T_wait": 3},
    "mini_open": {"S_mul": 1.05, "alpha_add": 0.0002, "T_wait": 3},
    "morning": {"S_mul": 1.00, "alpha_add": 0.0, "T_wait": 5},
    "pre_lunch_end": {"S_mul": 1.15, "alpha_add": 0.0002, "T_wait": 3},
    "afternoon": {"S_mul": 1.00, "alpha_add": 0.0, "T_wait": 5},
    "last15": {"S_mul": 1.10, "alpha_add": 0.0002, "T_wait": 3},
}

# ================= Signal generator (vol-dominant) =================

def generate_signals_vol_dominant(l1_df: pd.DataFrame, trade_df: pd.DataFrame, params: dict=None):
    """
    Inputs:
      - l1_df: columns [time, bidPrice, bidSize, askPrice, askSize]
      - trade_df: columns [time, price, vol]
    Output:
      DataFrame with precursor features, vol-dominant cost gate, and signals.
    """
    P = {
        "tz": "Asia/Shanghai",
        "W_box": 60, "W_touch": 10, "W_fast": 3,
        "W_rv": 30, "W_norm": 1800, "W_z": 180,
        "alpha_base": 0.0006,
        "S_threshold": 0.68,
        # vol-dominant cost-gate via rolling quantiles of rel_spread
        "p_low": 0.60, "p_high": 0.97,
        "gate_floor_mult": 0.80, "gate_cap_mult": 1.05,
    }
    if params: P.update(params)

    q = l1_df.copy()
    q = q.rename(columns={
        'time':'time',
        'bidprice':'bidPrice','bidPrice':'bidPrice',
        'bidsize':'bidSize','bidSize':'bidSize',
        'askprice':'askPrice','askPrice':'askPrice',
        'asksize':'askSize','askSize':'askSize'
    })
    need = ["time","bidPrice","bidSize","askPrice","askSize"]
    miss = [c for c in need if c not in q.columns]
    if miss: raise ValueError(f"L1 missing: {miss}")

    q["time"] = _ensure_tz_series(q["time"], P["tz"])
    q = q.sort_values("time").reset_index(drop=True)
    q["date"] = q["time"].dt.tz_convert(P["tz"]).dt.date

    # Basic
    for c in ["bidPrice","askPrice","bidSize","askSize"]:
        q[c] = q[c].astype(float)
    q["mid"] = (q["bidPrice"] + q["askPrice"]) / 2.0
    q["spread"] = (q["askPrice"] - q["bidPrice"])
    q["rel_spread"] = q["spread"] / q["mid"].replace(0, np.nan)

    # Tick per day
    ticks=[]
    for d,g in q.groupby("date", sort=False):
        tick = _infer_tick_size_from_prices(pd.concat([g["bidPrice"], g["askPrice"]]).values)
        ticks.append(pd.Series(tick, index=g.index))
    q["tick_size"] = pd.concat(ticks).sort_index()
    q["tick_rel"] = q["tick_size"] / q["mid"].replace(0, np.nan)

    # Trade resample → value per second (vps)
    t = trade_df.copy()
    t = t.rename(columns={'time':'time','price':'price','vol':'vol'})
    if not {'time','price','vol'}.issubset(t.columns):
        raise ValueError("trade_df must have [time, price, vol]")
    t["time"] = _ensure_tz_series(t["time"], P["tz"])
    t = t.sort_values("time")
    t["value"] = t["price"].astype(float) * t["vol"].astype(float)
    # 1-second resample, 5-second smoother
    vps = (t.set_index("time")["value"]
             .resample("1s").sum().rolling(5, min_periods=1).sum()
             .rename("vps").reset_index())
    # as-of merge
    q = pd.merge_asof(q.sort_values("time"), vps.sort_values("time"),
                      on="time", direction="backward", tolerance=pd.Timedelta("1s"))
    q["vps"] = q["vps"].fillna(0.0)
    q["vps_z"] = _zscore(q["vps"].values, P["W_z"])

    # Queue/tempo features
    q["q_imb"] = (q["bidSize"] - q["askSize"]) / (q["bidSize"] + q["askSize"]).replace(0, np.nan)
    q["dq"] = q["q_imb"] - q["q_imb"].shift(1)
    d_asize = q["askSize"].diff()
    q["ask_thin_freq"] = (d_asize < 0).astype(int).rolling(P["W_fast"], min_periods=1).mean()
    q["bid_step"] = (q["bidPrice"] > q["bidPrice"].shift(1)).astype(int)\
                    .rolling(P["W_fast"], min_periods=1).sum()

    # Box/touch
    roll_max = q["mid"].rolling(P["W_box"], min_periods=1).max()
    roll_min = q["mid"].rolling(P["W_box"], min_periods=1).min()
    q["range_width"] = (roll_max - roll_min) / q["mid"].replace(0, np.nan)

    tol = 0.5 * (q["tick_size"] / q["mid"])
    q["touch_upper"] = ((roll_max - q["mid"]) / q["mid"] <= tol).astype(int)
    q["touch_density"] = q["touch_upper"].rolling(P["W_touch"], min_periods=1).mean()
    recent_min = q["mid"].rolling(P["W_touch"], min_periods=1).min()
    q["pullback_depth"] = (roll_max - recent_min) / q["mid"].replace(0, np.nan)

    # Compression regime
    ret = q["mid"].pct_change()
    q["rv_short"] = ret.rolling(P["W_rv"], min_periods=5).std()
    def _roll_pct(arr, win, mp):
        s = pd.Series(arr)
        return s.rolling(win, min_periods=mp).apply(
            lambda v: (pd.Series(v).rank(pct=True).iloc[-1]), raw=False)
    q["range_pct"] = _roll_pct(q["range_width"].values, P["W_norm"], 120).values
    q["rv_pct"] = _roll_pct(q["rv_short"].values, P["W_norm"], 120).values
    q["compression"] = ((q["range_pct"] <= 0.25) & (q["rv_pct"] >= 0.35)).astype(int)

    # Z-features
    q["z_dq"] = _zscore(q["dq"].values, P["W_z"])
    q["z_askthin"] = _zscore(q["ask_thin_freq"].values, P["W_z"])
    q["z_bidstep"] = _zscore(q["bid_step"].values, P["W_z"])
    q["z_touch"] = _zscore(q["touch_density"].values, P["W_z"])
    q["z_pullback"] = _zscore(q["pullback_depth"].values, P["W_z"])
    q["z_spread"] = _zscore(q["rel_spread"].values, P["W_z"])
    # Resiliency: slow refill after thinning = higher
    q["z_resil"] = _zscore((-d_asize.fillna(0)).values, P["W_z"])

    # Composite S
    q["S"] = (
        0.30*q["z_dq"].fillna(0) +
        0.20*q["z_askthin"].fillna(0) +
        0.15*q["z_bidstep"].fillna(0) +
        0.15*q["z_touch"].fillna(0) +
        0.10*(-q["z_pullback"].fillna(0)) +
        0.05*(-q["z_spread"].fillna(0)) +
        0.05*q["z_resil"].fillna(0)
    )

    # Vol-dominant cost gate via rolling rel_spread quantile p(vps_z)
    zc = np.clip(q["vps_z"].values, -2, 3)
    p = P["p_low"] + (P["p_high"] - P["p_low"]) * (1.0 / (1.0 + np.exp(-zc)))
    p = np.clip(p, P["p_low"], P["p_high"])

    rel = pd.Series(q["rel_spread"].values)
    win = P["W_norm"]
    q50 = rel.rolling(win, min_periods=120).quantile(0.50).values
    q80 = rel.rolling(win, min_periods=120).quantile(0.80).values
    q95 = rel.rolling(win, min_periods=120).quantile(0.95).values
    gate_low = q50 * (1 - np.clip((p-0.60)/0.20, 0, 1)) + q80 * np.clip((p-0.60)/0.20, 0, 1)
    gate_high = q80 * (1 - np.clip((p-0.80)/0.15, 0, 1)) + q95 * np.clip((p-0.80)/0.15, 0, 1)
    cost_gate = np.where(p < 0.80, gate_low, gate_high)
    med = rel.rolling(win, min_periods=120).median().values
    mx = rel.rolling(win, min_periods=120).max().values
    cost_gate = np.maximum(cost_gate, P["gate_floor_mult"]*med)
    cost_gate = np.minimum(cost_gate, P["gate_cap_mult"]*mx)
    q["cost_gate"] = cost_gate

    # Adaptive micro-break
    q["alpha_adp"] = np.maximum(P["alpha_base"], 0.5 * q["tick_rel"])

    # Buckets and per-bucket tuning
    q["bucket"] = q["time"].apply(_label_bucket_cn_a)
    S_thr = np.full(len(q), P["S_threshold"], dtype=float)
    T_wait = np.full(len(q), 5, dtype=int)
    alpha_add = np.zeros(len(q))
    ban = np.zeros(len(q), dtype=bool)
    for i,b in enumerate(q["bucket"].values):
        conf = BUCKET_CONF.get(b, {})
        if conf.get("ban", False): ban[i] = True
        S_thr[i] = P["S_threshold"] * conf.get("S_mul", 1.0)
        alpha_add[i] = conf.get("alpha_add", 0.0)
        T_wait[i] = conf.get("T_wait", 5)
    q["S_threshold_adj"] = S_thr
    q["alpha_adp"] = q["alpha_adp"] + alpha_add
    q["T_wait_sec"] = T_wait
    q["ban"] = ban

    # Gates
    prebreak_ok = (q["mid"] >= (roll_max * (1 + q["alpha_adp"]))) | (q["touch_density"] >= 0.2)
    spread_ok = (q["rel_spread"] <= q["cost_gate"])
    compression_ok = (q["compression"] == 1)

    q["front_run_enter"] = (
        (q["S"] >= q["S_threshold_adj"]) &
        prebreak_ok & spread_ok & compression_ok & (~q["ban"])
    ).astype(int)

    # Light confirm using true trades: ofi_z + vol_z
    # Resample trades to 1s bars
    trade_sec = (t.set_index("time")[["price","vol"]]
                   .resample("1s").agg({"price":"last","vol":"sum"})
                   .fillna({"vol":0}))
    trade_sec["ret"] = trade_sec["price"].pct_change().fillna(0)
    trade_sec["val"] = (trade_sec["price"].fillna(method="ffill").fillna(0) * trade_sec["vol"])
    trade_sec["vol_z"] = _zscore(trade_sec["vol"].values, 180)
    trade_sec["val_z"] = _zscore(trade_sec["val"].values, 180)
    # Approx OFI at 1s: price up → +vol, down → -vol
    trade_sec["ofi"] = np.sign(trade_sec["ret"]) * trade_sec["vol"]
    trade_sec["ofi_z"] = _zscore(trade_sec["ofi"].values, 180)
    trade_sec = trade_sec.reset_index()

    # Merge 1s trade stats back to tick rows (as-of)
    q = pd.merge_asof(q.sort_values("time"), trade_sec[["time","vol_z","val_z","ofi_z"]].sort_values("time"),
                      on="time", direction="backward", tolerance=pd.Timedelta("1s"))
    q[["vol_z","val_z","ofi_z"]] = q[["vol_z","val_z","ofi_z"]].fillna(0)

    q["light_confirm"] = (
        (q["ofi_z"] >= 0.2) |
        (q["vol_z"] >= 0.7) |
        ((q["touch_density"] >= 0.3) & spread_ok)
    )

    return q

# ================= Backtest (60s hold) =================

def backtest_hold60(signals: pd.DataFrame, mode="enter_then_scale_if_confirm", hold_sec=60, fill="ba"):
    df = signals.copy().reset_index(drop=True)
    need = ["time","front_run_enter","light_confirm","T_wait_sec",
            "bidPrice","askPrice","mid"]
    for c in need:
        if c not in df.columns: raise ValueError(f"signals missing {c}")

    times = pd.to_datetime(df["time"]).values
    bid = df["bidPrice"].values.astype(float)
    ask = df["askPrice"].values.astype(float)
    mid = df["mid"].values.astype(float)
    enter_flag = df["front_run_enter"].values.astype(int)
    light = df["light_confirm"].values.astype(bool)
    Twait = df["T_wait_sec"].values.astype(int)

    def _idx_at_or_after(t):
        i = np.searchsorted(times, t, side="left")
        return i if i < len(times) else None
    def _px(i, side):
        if i is None: return np.nan
        return mid[i] if fill=="mid" else (ask[i] if side=="buy" else bid[i])

    trades=[]
    enter_idx = np.flatnonzero(enter_flag==1)
    for i0 in enter_idx:
        t0 = times[i0]
        t_dead = t0 + pd.Timedelta(seconds=int(Twait[i0]))
        mask = (times > t0) & (times <= t_dead) & light
        found = np.flatnonzero(mask)
        i_confirm = int(found[0]) if len(found) else None

        if mode=="enter_only":
            ie=i0; ix=_idx_at_or_after(t0+pd.Timedelta(seconds=hold_sec))
            if ix is None: continue
            p_in=_px(ie,"buy"); p_out=_px(ix,"sell")
            if np.isnan(p_in) or np.isnan(p_out): continue
            trades.append({"entry_time":times[ie],"exit_time":times[ix],
                           "entry_idx":ie,"exit_idx":ix,"weight":1.0,
                           "ret":(p_out-p_in)/p_in,
                           "date": pd.Timestamp(times[ie]).tz_convert("Asia/Shanghai").date()})
        elif mode=="enter_then_scale_if_confirm":
            # leg 1
            ix=_idx_at_or_after(t0+pd.Timedelta(seconds=hold_sec))
            if ix is not None:
                p_in=_px(i0,"buy"); p_out=_px(ix,"sell")
                if not (np.isnan(p_in) or np.isnan(p_out)):
                    trades.append({"entry_time":times[i0],"exit_time":times[ix],
                                   "entry_idx":i0,"exit_idx":ix,"weight":0.35,
                                   "ret":(p_out-p_in)/p_in,
                                   "date": pd.Timestamp(times[i0]).tz_convert("Asia/Shanghai").date()})
            # leg 2
            if i_confirm is not None:
                t1 = times[i_confirm]
                ix=_idx_at_or_after(t1+pd.Timedelta(seconds=hold_sec))
                if ix is not None:
                    p_in=_px(i_confirm,"buy"); p_out=_px(ix,"sell")
                    if not (np.isnan(p_in) or np.isnan(p_out)):
                        trades.append({"entry_time":times[i_confirm],"exit_time":times[ix],
                                       "entry_idx":i_confirm,"exit_idx":ix,"weight":0.65,
                                       "ret":(p_out-p_in)/p_in,
                                       "date": pd.Timestamp(times[i_confirm]).tz_convert("Asia/Shanghai").date()})
        else: # confirm_only
            if i_confirm is None: continue
            ie=i_confirm; ix=_idx_at_or_after(times[ie]+pd.Timedelta(seconds=hold_sec))
            if ix is None: continue
            p_in=_px(ie,"buy"); p_out=_px(ix,"sell")
            if np.isnan(p_in) or np.isnan(p_out): continue
            trades.append({"entry_time":times[ie],"exit_time":times[ix],
                           "entry_idx":ie,"exit_idx":ix,"weight":1.0,
                           "ret":(p_out-p_in)/p_in,
                           "date": pd.Timestamp(times[ie]).tz_convert("Asia/Shanghai").date()})

    trades = pd.DataFrame(trades)
    if trades.empty:
        return trades, pd.DataFrame(), pd.Series({"n_legs":0,"n_trades":0,"win_rate":np.nan,"avg_bp":np.nan})
    trades["ret_bp"] = trades["ret"]*1e4
    trades = trades.sort_values(["entry_time","exit_time"]).reset_index(drop=True)
    trades["trade_id"] = trades["entry_idx"].rank(method="dense").astype(int)

    trade_sum = trades.groupby("trade_id").apply(
        lambda g: pd.Series({
            "entry_time": g["entry_time"].min(),
            "exit_time": g["exit_time"].max(),
            "date": g["date"].iloc[0],
            "ret": np.sum(g["ret"]*g["weight"]),
            "legs": len(g)
        })
    ).reset_index(drop=True)
    trade_sum["ret_bp"] = trade_sum["ret"]*1e4

    summary = pd.Series({
        "n_legs": len(trades),
        "n_trades": len(trade_sum),
        "avg_bp": trade_sum["ret_bp"].mean(),
        "median_bp": trade_sum["ret_bp"].median(),
        "std_bp": trade_sum["ret_bp"].std(ddof=0),
        "win_rate": (trade_sum["ret"]>0).mean()
    })
    return trades, trade_sum, summary

# ================= Plotting =================

def plot_results(trade_sum: pd.DataFrame, signals: pd.DataFrame, title="Backtest"):
    if trade_sum.empty:
        print("No trades to plot."); return
    ts = trade_sum.sort_values("exit_time").copy()
    ts["cum_bp"] = ts["ret_bp"].cumsum()

    fig = plt.figure(figsize=(12,9))
    gs = fig.add_gridspec(3,2, height_ratios=[1,1,1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts["exit_time"], ts["cum_bp"], color="#2a9d8f")
    ax1.set_title(f"{title} - Cumulative PnL (bp)")
    ax1.set_ylabel("bp"); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    daily = ts.groupby("date")["ret_bp"].sum()
    ax2.bar(daily.index.astype(str), daily.values, color="#457b9d")
    ax2.set_title("Daily PnL (bp)"); ax2.tick_params(axis='x', rotation=45); ax2.grid(True, axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    cnt = signals.groupby("bucket")["front_run_enter"].sum().sort_values(ascending=False)
    ax3.bar(cnt.index, cnt.values, color="#8d99ae")
    ax3.set_title("Signal count by bucket"); ax3.tick_params(axis='x', rotation=45); ax3.grid(True, axis="y", alpha=0.3)

    rnd = ts.sample(1, random_state=42).iloc[0]
    t0 = rnd["entry_time"] - pd.Timedelta(seconds=30)
    t1 = rnd["exit_time"] + pd.Timedelta(seconds=30)
    seg = signals[(signals["time"]>=t0)&(signals["time"]<=t1)].copy()
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(seg["time"], seg["mid"], label="mid", color="#264653")
    ax4.fill_between(seg["time"], seg["bidPrice"], seg["askPrice"], color="#e9c46a", alpha=0.25, label="bid-ask")
    ax4.scatter([rnd["entry_time"]],[seg.loc[seg["time"]==rnd["entry_time"],"mid"].iloc[0]], color="green", label="entry")
    ax4.scatter([rnd["exit_time"]],[seg.loc[seg["time"]==rnd["exit_time"],"mid"].iloc[0]], color="red", label="exit")
    ax4.set_title("Example trade context"); ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

# ================= Example run =================

if __name__ == "__main__":
    # Replace the next two lines with your actual data reading.
    # l1_df = pd.read_csv("l1_multi_day.csv")
    # trade_df = pd.read_csv("trades_multi_day.csv")
    
    sig = generate_signals_vol_dominant(l1_df, trade_df, params=dict(
    alpha_base=0.0006,
    S_threshold=0.68,
    W_norm=1800,   # 30-minute history for quantiles
    p_high=0.97
    ))
    legs, trades, summary = backtest_hold60(sig, mode="enter_then_scale_if_confirm", hold_sec=60, fill="ba")
    print(summary)
    plot_results(trades, sig, title="Vol-dominant precursor (with trades), 60s hold")