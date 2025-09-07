# -*- coding: utf-8 -*-
"""
A股低价粗价阶（tick=0.01）友好的前置信号 + 回测 + 绘图（以 600200.SS 场景为例）
- 成本门槛（点差）完全关闭，但保留 cost_gate 供参考
- 回测采用“固定持有时长（默认60秒）”，支持 mid/BA 两种填价
- 绘图包含：价格与进出场、S与阈值、门槛通过率、累计PnL与分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ============ 基础工具函数 ============

def _ensure_tz_series(series, tz="Asia/Shanghai"):
    """确保时间列带时区；若无则本地化为上海时区。"""
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz)
    else:
        s = s.dt.tz_convert(tz)
    return s

def _infer_tick_size_from_prices(prices, max_unique=6000):
    """从价位集合估计价阶；低价股建议直接 force_tick_size=0.01。"""
    s = pd.Series(prices, dtype=float).dropna()
    if s.nunique() < 2:
        return np.nan
    vals = np.sort(s.unique())
    if len(vals) > max_unique:
        vals = np.sort(np.random.choice(vals, size=max_unique, replace=False))
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    tick = np.quantile(diffs, 0.1)
    return float(pd.Series([tick]).round(6).iloc[0])

def _zscore(x, window, min_periods=None):
    """滚动标准分；低采样/粗tick推荐稍短窗口以更快适应。"""
    if min_periods is None:
        min_periods = max(30, window // 6)
    s = pd.Series(x, dtype=float)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mu) / (sd.replace(0, np.nan))
    return z.values

def _label_bucket_cn_a(ts):
    """上交所/深交所常见时段标签（上海时区），用于禁做与阈值微调。"""
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
    "open": {"S_mul": 1.08, "T_wait": 3},
    "mini_open": {"S_mul": 1.05, "T_wait": 3},
    "morning": {"S_mul": 1.00, "T_wait": 5},
    "pre_lunch_end": {"S_mul": 1.10, "T_wait": 3},
    "afternoon": {"S_mul": 1.00, "T_wait": 5},
    "last15": {"S_mul": 1.08, "T_wait": 3},
}

# ============ 信号生成（含诊断；成本关闭） ============

def generate_signals_vol_dominant_cn(l1_df: pd.DataFrame,
                                     trade_df: pd.DataFrame=None,
                                     params: dict=None) -> pd.DataFrame:
    """
    输入:
      - l1_df: 必填，列包含 [time, bidPrice, bidSize, askPrice, askSize]
      - trade_df: 可选，列包含 [time, price, vol]；若提供，可用于更可靠的轻确认(light_confirm)
    输出:
      - 含特征、门槛布尔值、诊断列、最终信号(front_run_enter, light_confirm)的 DataFrame

    重要说明:
      - 成本门槛已关闭：spread_ok 恒为 True；仍计算 cost_gate 仅供参考
      - 针对低价(≈1元)、tick=0.01、相对点差≈1%的股票，默认阈值偏宽，先让信号跑起来
    """

    # ---- 默认参数（针对 600200.SS 场景） ----
    P = {
        "tz": "Asia/Shanghai",
        "force_tick_size": 0.01,   # 已知 A股价阶，低价股强烈建议固定为 0.01
        "W_box": 60,               # 近高/近低窗口(秒)
        "W_touch": 15,             # 触顶密度窗口；粗tick建议略长
        "W_fast": 3,               # 快速节奏窗口
        "W_rv": 30,                # 短波动率窗口
        "W_norm": 1800,            # 百分位归一窗口（30分钟）
        "W_z": 120,                # zscore窗口（低价股用较短窗口以加快自适应）
        # S 入场阈值（会按时段微调）；starter 阶段稍宽松
        "S_threshold": 0.63,
        # 微突破阈值：alpha_adp = max(alpha_base, alpha_scale * tick_rel) 并截顶
        "alpha_base": 0.0003,      # 3 bp
        "alpha_scale": 0.25,       # 0.25 * 100bp = 25bp
        "alpha_cap": 0.0060,       # 上限 60 bp（小于一个tick=100bp）
        # 压缩判定：默认使用“宽松版”，同时输出严格版供对比
        "compression_mode": "loose",  # loose / strict
        "compression_loose": {"range_pct_max": 0.40, "rv_pct_min": 0.20},
        "compression_strict": {"range_pct_max": 0.25, "rv_pct_min": 0.35},
        # 触顶替代路径：触顶密度达阈值亦可视为预突破成立
        "touch_prebreak_thr": 0.12,
        # 轻确认阈值（若有成交数据）
        "confirm_ofi_thr": 0.10,
        "confirm_vol_thr": 0.55,
        "touch_confirm_thr": 0.28,
        # S 的滚动百分位线（诊断用）
        "W_S_pct": 1800,
        # 成本门槛计算但不启用（spread_ok恒为True）
        "p_low": 0.60, "p_high": 0.97,
        "gate_floor_mult": 1.00, "gate_cap_mult": 1.10,
        # Starter 阶段限制 S 阈值不超过 cap，避免初期过紧
        "S_thresh_cap": 0.63,
    }
    if params:
        P.update(params)

    # ---- 标准化输入 ----
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
    if miss:
        raise ValueError(f"L1 缺少列: {miss}")

    q["time"] = _ensure_tz_series(q["time"], P["tz"])
    q = q.sort_values("time").reset_index(drop=True)
    q["date"] = q["time"].dt.tz_convert(P["tz"]).dt.date

    # ---- 基础字段 ----
    for c in ["bidPrice","askPrice","bidSize","askSize"]:
        q[c] = q[c].astype(float)
    q["mid"] = (q["bidPrice"] + q["askPrice"]) / 2.0
    q["spread"] = (q["askPrice"] - q["bidPrice"])
    q["rel_spread"] = q["spread"] / q["mid"].replace(0, np.nan)

    # ---- 价阶（tick） ----
    if P.get("force_tick_size") is not None:
        q["tick_size"] = float(P["force_tick_size"])
    else:
        ticks=[]
        for d, g in q.groupby("date", sort=False):
            tick = _infer_tick_size_from_prices(pd.concat([g["bidPrice"], g["askPrice"]]).values)
            ticks.append(pd.Series(tick, index=g.index))
        q["tick_size"] = pd.concat(ticks).sort_index()

    q["tick_rel"] = q["tick_size"] / q["mid"].replace(0, np.nan)  # 低价股常见：~0.01(=100bp)

    # ---- L1节奏特征 ----
    q["q_imb"] = (q["bidSize"] - q["askSize"]) / (q["bidSize"] + q["askSize"]).replace(0, np.nan)
    q["dq"] = q["q_imb"] - q["q_imb"].shift(1)
    d_asize = q["askSize"].diff()
    q["ask_thin_freq"] = (d_asize < 0).astype(int).rolling(P["W_fast"], min_periods=1).mean()
    q["bid_step"] = (q["bidPrice"] > q["bidPrice"].shift(1)).astype(int)\
                    .rolling(P["W_fast"], min_periods=1).sum()

    # ---- 近高/近低 与 触顶密度 ----
    roll_max = q["mid"].rolling(P["W_box"], min_periods=1).max()
    roll_min = q["mid"].rolling(P["W_box"], min_periods=1).min()
    q["range_width"] = (roll_max - roll_min) / q["mid"].replace(0, np.nan)

    # 粗tick：允许“接近近高”即算触顶。用半个tick作为容差。
    tol = 0.5 * (q["tick_size"] / q["mid"])
    q["touch_upper"] = ((roll_max - q["mid"]) / q["mid"] <= tol).astype(int)
    q["touch_density"] = q["touch_upper"].rolling(P["W_touch"], min_periods=1).mean()

    recent_min = q["mid"].rolling(P["W_touch"], min_periods=1).min()
    q["pullback_depth"] = (roll_max - recent_min) / q["mid"].replace(0, np.nan)

    # ---- 压缩(compression) ----
    ret = q["mid"].pct_change()
    q["rv_short"] = ret.rolling(P["W_rv"], min_periods=5).std()

    def _roll_pct(arr, win, mp):
        s = pd.Series(arr)
        return s.rolling(win, min_periods=mp).apply(
            lambda v: (pd.Series(v).rank(pct=True).iloc[-1]), raw=False)
    q["range_pct"] = _roll_pct(q["range_width"].values, P["W_norm"], 120).values
    q["rv_pct"]    = _roll_pct(q["rv_short"].values,  P["W_norm"], 120).values

    q["compression_strict"] = (
        (q["range_pct"] <= P["compression_strict"]["range_pct_max"]) &
        (q["rv_pct"]    >= P["compression_strict"]["rv_pct_min"])
    ).astype(int)
    q["compression_loose"] = (
        (q["range_pct"] <= P["compression_loose"]["range_pct_max"]) &
        (q["rv_pct"]    >= P["compression_loose"]["rv_pct_min"])
    ).astype(int)
    q["compression"] = q["compression_loose"] if P["compression_mode"]=="loose" else q["compression_strict"]

    # ---- Z分特征 与 组合分数 S ----
    q["z_dq"]       = _zscore(q["dq"].values, P["W_z"])
    q["z_askthin"]  = _zscore(q["ask_thin_freq"].values, P["W_z"])
    q["z_bidstep"]  = _zscore(q["bid_step"].values, P["W_z"])
    q["z_touch"]    = _zscore(q["touch_density"].values, P["W_z"])
    q["z_pullback"] = _zscore(q["pullback_depth"].values, P["W_z"])
    q["z_spread"]   = _zscore(q["rel_spread"].values, P["W_z"])
    q["z_resil"]    = _zscore((-d_asize.fillna(0)).values, P["W_z"])  # 回补越慢越“韧性高”

    q["S"] = (
        0.30*q["z_dq"].fillna(0) +
        0.20*q["z_askthin"].fillna(0) +
        0.15*q["z_bidstep"].fillna(0) +
        0.15*q["z_touch"].fillna(0) +
        0.10*(-q["z_pullback"].fillna(0)) +
        0.05*(-q["z_spread"].fillna(0)) +
        0.05*q["z_resil"].fillna(0)
    )

    # ---- 合并逐笔成交（可选）增强确认 ----
    if trade_df is not None and {'time','price','vol'}.issubset(set(trade_df.columns)):
        t = trade_df.copy()
        t["time"] = _ensure_tz_series(t["time"], P["tz"])
        t = t.sort_values("time")
        trade_sec = (t.set_index("time")[["price","vol"]]
                       .resample("1s").agg({"price":"last","vol":"sum"})
                       .fillna({"vol":0}))
        trade_sec["ret"] = trade_sec["price"].pct_change().fillna(0)
        trade_sec["val"] = (trade_sec["price"].fillna(method="ffill").fillna(0) * trade_sec["vol"])
        trade_sec["vol_z"] = _zscore(trade_sec["vol"].values, 180)
        trade_sec["ofi"] = np.sign(trade_sec["ret"]) * trade_sec["vol"]
        trade_sec["ofi_z"] = _zscore(trade_sec["ofi"].values, 180)
        trade_sec = trade_sec.reset_index()

        q = pd.merge_asof(q.sort_values("time"),
                          trade_sec[["time","vol_z","ofi_z"]].sort_values("time"),
                          on="time", direction="backward", tolerance=pd.Timedelta("1s"))
        q[["vol_z","ofi_z"]] = q[["vol_z","ofi_z"]].fillna(0)
    else:
        q["vol_z"] = 0.0
        q["ofi_z"] = 0.0

    # ---- 时段微调 与 ban ----
    q["bucket"] = q["time"].apply(_label_bucket_cn_a)
    S_thr = np.full(len(q), P["S_threshold"], dtype=float)
    T_wait = np.full(len(q), 5, dtype=int)
    ban = np.zeros(len(q), dtype=bool)
    for i, b in enumerate(q["bucket"].values):
        conf = BUCKET_CONF.get(b, {})
        if conf.get("ban", False): ban[i] = True
        S_thr[i] = P["S_threshold"] * conf.get("S_mul", 1.0)
        T_wait[i] = conf.get("T_wait", 5)
    q["S_threshold_adj"] = np.minimum(S_thr, P["S_thresh_cap"])  # Starter 阶段限制上限
    q["T_wait_sec"] = T_wait
    q["ban"] = ban

    # ---- alpha_adp（微突破阈值） ----
    q["alpha_adp"] = np.maximum(P["alpha_base"], P["alpha_scale"] * q["tick_rel"])
    q["alpha_adp"] = np.clip(q["alpha_adp"], P["alpha_base"], P["alpha_cap"])
    q["alpha_bp"]  = q["alpha_adp"] * 1e4  # bp单位，便于直观判断量级

    # ---- 成本门槛：仅计算，不参与判定 ----
    rel = pd.Series(q["rel_spread"].values)
    win = P["W_norm"]
    med = rel.rolling(win, min_periods=120).median().values
    mx  = rel.rolling(win, min_periods=120).max().values
    q80 = rel.rolling(win, min_periods=120).quantile(0.80).values
    cost_gate = np.maximum(q80, P["gate_floor_mult"]*med)
    cost_gate = np.minimum(cost_gate, P["gate_cap_mult"]*mx)
    q["cost_gate"] = cost_gate

    # ---- 入场门槛（成本关闭版） ----
    roll_max_60 = q["mid"].rolling(60, min_periods=1).max()
    prebreak_ok = (q["mid"] >= roll_max_60 * (1.0 + q["alpha_adp"])) | (q["touch_density"] >= P["touch_prebreak_thr"])
    compression_ok = (q["compression"] == 1)
    S_ok = (q["S"] >= q["S_threshold_adj"])
    spread_ok = np.ones(len(q), dtype=bool)   # 成本门槛关闭
    not_ban = ~q["ban"].astype(bool)

    q["S_ok"] = S_ok.astype(int)
    q["prebreak_ok"] = prebreak_ok.astype(int)
    q["compression_ok"] = compression_ok.astype(int)
    q["not_ban"] = not_ban.astype(int)
    q["spread_ok"] = 1  # 恒为1

    q["front_run_enter"] = (S_ok & prebreak_ok & compression_ok & not_ban).astype(int)

    # ---- 轻确认（不依赖成本） ----
    q["light_confirm"] = (
        (q["ofi_z"] >= P["confirm_ofi_thr"]) |
        (q["vol_z"] >= P["confirm_vol_thr"]) |
        (q["touch_density"] >= P["touch_confirm_thr"]) |
        (q["z_bidstep"] >= 0.10)
    )

    # ---- S 的滚动百分位线（诊断/画图友好） ----
    W_S = P["W_S_pct"]
    S_series = q["S"].astype(float)
    q["S_p60"] = S_series.rolling(W_S, min_periods=120).quantile(0.60)
    q["S_p80"] = S_series.rolling(W_S, min_periods=120).quantile(0.80)
    q["S_p90"] = S_series.rolling(W_S, min_periods=120).quantile(0.90)

    # ---- 每日快速摘要（简易诊断） ----
    def _summ(g):
        return pd.Series({
            "rows": len(g),
            "enter_%": g["front_run_enter"].mean()*100 if len(g) else 0,
            "S_ok_%": g["S_ok"].mean()*100 if len(g) else 0,
            "prebreak_%": g["prebreak_ok"].mean()*100 if len(g) else 0,
            "compress_%": g["compression_ok"].mean()*100 if len(g) else 0,
            "not_ban_%": g["not_ban"].mean()*100 if len(g) else 0,
            "S_med": g["S"].median(),
            "S_p90": g["S"].quantile(0.90),
            "alpha_bp_med": (g["alpha_adp"].median()*1e4) if len(g) else np.nan,
            "touch_p90": g["touch_density"].quantile(0.90) if len(g) else np.nan,
            "rel_spread_bp_med": (g["rel_spread"].median()*1e4) if len(g) else np.nan,
        })
    try:
        daily_diag = q.groupby("date", sort=False).apply(_summ)
        map_diag = {idx: daily_diag.loc[idx].to_dict() for idx in daily_diag.index}
        q["diag_summary"] = q["date"].map(lambda d: str(map_diag.get(d, {})))
    except Exception:
        q["diag_summary"] = ""

    return q

# ============ 回测（固定持有时长 60s；mid/BA 填价；稳健边界处理） ============
def backtest_hold60_cn(signals: pd.DataFrame,
                       mode="enter_then_scale_if_confirm",
                       hold_sec=60,
                       fill="mid"):
    """
    简洁稳健版 60s 回测（已修复 searchsorted 时间类型报错）
    - 关键修复：把分组后的时间列统一转为 numpy datetime64[ns]，查询时间同样转为此类型再 searchsorted
    """
    df = signals.copy()
    need = ["time","front_run_enter","light_confirm","T_wait_sec",
            "bidPrice","askPrice","mid","bucket","date"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"signals 缺少列: {miss}")

    # 去掉无效时间
    df = df[pd.notna(df["time"])].copy()

    all_legs = []

    # 小工具：把一列时间统一成 numpy datetime64[ns]（仅用于搜索，绘图仍可用原始带时区时间）
    def _to_np_dt64(series):
        ts = pd.to_datetime(series)
        # 若带时区，先转 UTC 再去 tz，保证是统一的 naive datetime64[ns]
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        return ts.values.astype("datetime64[ns]")

    # 把单个时间也转成 datetime64[ns]
    def _coerce_t(t):
        tt = pd.to_datetime(t)
        if getattr(getattr(tt, "tz", None), "zone", None) or getattr(tt, "tzinfo", None):
            tt = tt.tz_convert("UTC").tz_localize(None)
        return np.datetime64(tt.to_datetime64())  # datetime64[ns]

    for d, g in df.groupby("date", sort=False):
        g = g.sort_values("time").reset_index(drop=True)

        # 统一好的时间数组（用于 searchsorted）
        times_np = _to_np_dt64(g["time"])

        bid = g["bidPrice"].astype(float).values
        ask = g["askPrice"].astype(float).values
        mid = g["mid"].astype(float).values
        enter_flag = g["front_run_enter"].astype(int).values
        light = g["light_confirm"].astype(bool).values
        Twait = g["T_wait_sec"].astype(int).values

        if len(g) == 0:
            continue

        def _idx_at_or_after(t):
            i = np.searchsorted(times_np, _coerce_t(t), side="left")
            return int(i) if i < len(times_np) else None

        def _px(i, side):
            if i is None: return np.nan
            if fill == "mid": return mid[i]
            return ask[i] if side=="buy" else bid[i]

        # 会话允许的最后索引（避开午休/集合竞价）
        allowed = ~(g["bucket"].isin(["auction_open","lunch_break","close_auction","off"]))
        last_allowed_idx = int(np.where(allowed.values)[0][-1]) if allowed.any() else len(g)-1

        # 第一腿：enter_only 或 enter_then_scale_if_confirm
        if mode != "confirm_only":
            enter_idx = np.flatnonzero(enter_flag==1)
            for i0 in enter_idx:
                t0 = g["time"].iloc[i0]  # 用原始 pandas 时间做加法
                t_exit_1 = t0 + pd.Timedelta(seconds=hold_sec)
                ix1 = _idx_at_or_after(t_exit_1)
                if ix1 is None or ix1 > last_allowed_idx:
                    ix1 = last_allowed_idx
                p_in1 = _px(i0, "buy")
                p_out1 = _px(ix1, "sell")
                if not (np.isnan(p_in1) or np.isnan(p_out1)):
                    w1 = 0.35 if mode=="enter_then_scale_if_confirm" else 1.0
                    all_legs.append({
                        "date": d,
                        "entry_time": g["time"].iloc[i0],
                        "exit_time": g["time"].iloc[ix1],
                        "entry_idx": int(g.index[i0]),
                        "exit_idx": int(g.index[ix1]),
                        "leg": 1,
                        "weight": w1,
                        "ret": (p_out1 - p_in1) / p_in1
                    })

                # 第二腿：T_wait 内出现轻确认
                if mode == "enter_then_scale_if_confirm":
                    t_dead = t0 + pd.Timedelta(seconds=int(Twait[i0]))
                    # 用 pandas 布尔索引找确认时间，再用 searchsorted 取出场
                    mask = (g["time"] > t0) & (g["time"] <= t_dead) & (g["light_confirm"]==1)
                    if mask.any():
                        i_confirm = int(np.flatnonzero(mask.values)[0])
                        t_exit_2 = g["time"].iloc[i_confirm] + pd.Timedelta(seconds=hold_sec)
                        ix2 = _idx_at_or_after(t_exit_2)
                        if ix2 is None or ix2 > last_allowed_idx:
                            ix2 = last_allowed_idx
                        p_in2 = _px(i_confirm, "buy")
                        p_out2 = _px(ix2, "sell")
                        if not (np.isnan(p_in2) or np.isnan(p_out2)):
                            all_legs.append({
                                "date": d,
                                "entry_time": g["time"].iloc[i_confirm],
                                "exit_time": g["time"].iloc[ix2],
                                "entry_idx": int(g.index[i_confirm]),
                                "exit_idx": int(g.index[ix2]),
                                "leg": 2,
                                "weight": 0.65,
                                "ret": (p_out2 - p_in2) / p_in2
                            })

        # confirm_only 模式：仅确认建仓
        else:
            confirm_idx = np.flatnonzero(light==1)
            for ic in confirm_idx:
                t_c = g["time"].iloc[ic]
                t_exit_c = t_c + pd.Timedelta(seconds=hold_sec)
                ixc = _idx_at_or_after(t_exit_c)
                if ixc is None or ixc > last_allowed_idx:
                    ixc = last_allowed_idx
                p_in = _px(ic, "buy")
                p_out = _px(ixc, "sell")
                if not (np.isnan(p_in) or np.isnan(p_out)):
                    all_legs.append({
                        "date": d,
                        "entry_time": g["time"].iloc[ic],
                        "exit_time": g["time"].iloc[ixc],
                        "entry_idx": int(g.index[ic]),
                        "exit_idx": int(g.index[ixc]),
                        "leg": 1,
                        "weight": 1.0,
                        "ret": (p_out - p_in) / p_in
                    })

    legs = pd.DataFrame(all_legs)
    if legs.empty:
        trades = pd.DataFrame()
        summary = pd.Series({"n_legs":0,"n_trades":0,"avg_bp":np.nan,"median_bp":np.nan,
                             "std_bp":np.nan,"win_rate":np.nan,"cum_bp":0.0,"sharpe":np.nan})
        return legs, trades, summary

    legs["ret_bp"] = legs["ret"]*1e4
    legs = legs.sort_values(["entry_time","exit_time"]).reset_index(drop=True)

    # 合并为 trade
    legs["trade_id"] = legs["entry_time"].rank(method="dense").astype(int)
    trades = legs.groupby("trade_id").apply(
        lambda g: pd.Series({
            "date": g["date"].iloc[0],
            "entry_time": g["entry_time"].min(),
            "exit_time": g["exit_time"].max(),
            "legs": len(g),
            "ret": np.sum(g["ret"]*g["weight"]),
        })
    ).reset_index(drop=True)
    trades["ret_bp"] = trades["ret"]*1e4

    summary = pd.Series({
        "n_legs": len(legs),
        "n_trades": len(trades),
        "avg_bp": trades["ret_bp"].mean(),
        "median_bp": trades["ret_bp"].median(),
        "std_bp": trades["ret_bp"].std(ddof=0),
        "win_rate": (trades["ret"]>0).mean(),
        "cum_bp": trades["ret_bp"].sum(),
        "sharpe": np.nan if trades["ret"].std(ddof=0)==0 else (trades["ret"].mean()/trades["ret"].std(ddof=0))*np.sqrt(252)
    })
    return legs, trades, summary

# ============ 绘图 ============

def plot_results_cn(sig: pd.DataFrame, legs: pd.DataFrame, trades: pd.DataFrame, title="600200.SS 示例（成本关闭）"):
    """
    绘制：
      - 价格与进出场（左上）
      - S 与阈值/百分位线（左中）
      - 门槛通过率滚动均值（左下）
      - 累计PnL曲线与每笔分布（右列上下）
    """
    if sig.empty:
        print("signals 为空，无法绘图。")
        return

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 2, width_ratios=[2.2, 1.0], height_ratios=[1.2, 1.0, 1.0], figure=fig, wspace=0.25, hspace=0.28)

    # --- 左上：价格与进出场 ---
    ax0 = fig.add_subplot(gs[0,0])
    t = pd.to_datetime(sig["time"]).dt.tz_convert("Asia/Shanghai")
    ax0.plot(t, sig["mid"], color="#2b8cbe", lw=1.2, label="中间价")
    ax0.fill_between(t, sig["bidPrice"], sig["askPrice"], color="#a6bddb", alpha=0.15, label="买卖价区间")

    if not legs.empty:
        ax0.scatter(pd.to_datetime(legs["entry_time"]), 
                    legs["leg"].map({1:"#1b9e77", 2:"#d95f02"}), # 仅用于颜色映射，位置用价格
                    alpha=0)  # 占位避免图例报错
        # 在图上画进出场点（价格取回测时刻的 mid/BA，不再重算）
        for _, r in legs.iterrows():
            ax0.axvline(pd.to_datetime(r["entry_time"]), color="#1b9e77", alpha=0.25, lw=0.8)
            ax0.axvline(pd.to_datetime(r["exit_time"]),  color="#d95f02", alpha=0.25, lw=0.8)

    ax0.set_title(f"{title} - 价格与进出场", fontsize=12, weight="bold")
    ax0.set_ylabel("价格")
    ax0.legend(loc="upper left")

    # --- 左中：S 与阈值 ---
    ax1 = fig.add_subplot(gs[1,0], sharex=ax0)
    ax1.plot(t, sig["S"], color="#4daf4a", lw=1.0, label="S")
    if "S_threshold_adj" in sig.columns:
        ax1.plot(t, sig["S_threshold_adj"], color="#e41a1c", lw=1.0, ls="--", label="S阈值(调整)")
    if "S_p80" in sig.columns:
        ax1.plot(t, sig["S_p80"], color="#377eb8", lw=0.8, ls=":", label="S_p80(诊断)")
    if "S_p90" in sig.columns:
        ax1.plot(t, sig["S_p90"], color="#984ea3", lw=0.8, ls=":", label="S_p90(诊断)")
    # 高亮通过 S_ok 的区域
    if "S_ok" in sig.columns:
        m = sig["S_ok"]==1
        ax1.fill_between(t, sig["S"].min(), sig["S"].max(), where=m, color="#4daf4a", alpha=0.05, transform=ax1.get_xaxis_transform())

    ax1.set_title("S 与阈值/百分位线", fontsize=12, weight="bold")
    ax1.set_ylabel("S 值")
    ax1.legend(loc="upper left")

    # --- 左下：门槛通过率滚动均值 ---
    ax2 = fig.add_subplot(gs[2,0], sharex=ax0)
    win = 120  # 2分钟滚动
    def roll_mean(col):
        return col.rolling(win, min_periods=max(10, win//4)).mean()*100
    cols = [c for c in ["S_ok","prebreak_ok","compression_ok","not_ban"] if c in sig.columns]
    for c in cols:
        ax2.plot(t, roll_mean(sig[c].astype(float)), lw=1.0, label=f"{c}通过率(%)")
    ax2.set_title("门槛通过率（滚动）", fontsize=12, weight="bold")
    ax2.set_ylabel("%")
    ax2.set_xlabel("时间")
    ax2.legend(loc="upper left", ncol=2)

    # --- 右上：累计PnL（按 legs 与 trades） ---
    ax3 = fig.add_subplot(gs[0,1])
    if not trades.empty:
        # 累计PnL按交易先后
        cum_trade = trades.sort_values("entry_time")["ret_bp"].cumsum()
        ax3.plot(range(1, len(cum_trade)+1), cum_trade.values, color="#1f78b4", lw=1.4, label="Trades累积(bp)")
    if not legs.empty:
        cum_leg = legs.sort_values("entry_time")["ret_bp"].cumsum()
        ax3.plot(range(1, len(cum_leg)+1), cum_leg.values, color="#33a02c", lw=1.0, alpha=0.8, label="Legs累积(bp)")
    ax3.set_title("累计PnL", fontsize=12, weight="bold")
    ax3.set_xlabel("序号")
    ax3.set_ylabel("累积(bp)")
    ax3.legend(loc="best")

    # --- 右下：每笔分布 与 文本摘要 ---
    ax4 = fig.add_subplot(gs[1:,1])
    ax4.set_title("每笔收益分布 与 摘要", fontsize=12, weight="bold")
    if not trades.empty:
        sns.histplot(trades["ret_bp"], bins=30, kde=True, color="#6a3d9a", ax=ax4, alpha=0.75)
        ax4.axvline(0, color="k", lw=1.0, alpha=0.6)
        # 文本摘要
        avg = trades["ret_bp"].mean()
        med = trades["ret_bp"].median()
        std = trades["ret_bp"].std(ddof=0)
        win = (trades["ret"]>0).mean()*100
        cum = trades["ret_bp"].sum()
        txt = f"n_trades={len(trades)}\navg={avg:.2f} bp\nmed={med:.2f} bp\nstd={std:.2f} bp\nwin={win:.1f}%\ncum={cum:.1f} bp"
        ax4.text(0.98, 0.95, txt, ha="right", va="top", transform=ax4.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#444", alpha=0.9), fontsize=10)
        ax4.set_xlabel("单笔收益(bp)")
        ax4.set_ylabel("频数")
    else:
        ax4.text(0.5, 0.5, "无交易可显示", ha="center", va="center", fontsize=12)

    plt.suptitle(title, fontsize=14, weight="bold")
    plt.show()

# ============ 快速检查通过率（可选） ============

def print_gate_pass_rates(sig: pd.DataFrame):
    cols = ["S_ok","prebreak_ok","compression_ok","not_ban","spread_ok","front_run_enter","light_confirm"]
    cols = [c for c in cols if c in sig.columns]
    if not cols:
        print("无可用门槛列。"); return
    stat = sig[cols].mean(numeric_only=True)*100
    print((stat.round(2).astype(str) + "%").to_dict())

# ============ 使用示例（按需替换你的数据源） ============

if __name__ == "__main__":
    # 假设 600200.SS 在 2025-09-05 的 L1 数据加载为 l1_df：
    # 需要列：time, bidPrice, bidSize, askPrice, askSize
    # 例如：
    # l1_df = pd.read_csv("l1_600200_20250905.csv")
    # 可选逐笔成交：
    # trade_df = pd.read_csv("trades_600200_20250905.csv")

    # 已准备好 l1_df，则这样运行：
    # sig = generate_signals_vol_dominant_cn(l1_df, trade_df=None)
    # print_gate_pass_rates(sig)

    # legs, trades, summary = backtest_hold60_cn(sig, mode="enter_then_scale_if_confirm", hold_sec=60, fill="mid")
    # print("回测摘要：")
    # print(summary)

    # 绘图（标题可自定义）
    # plot_results_cn(sig, legs, trades, title="600200.SS 2025-09-05（成本关闭，hold=60s，mid填价）")
    pass