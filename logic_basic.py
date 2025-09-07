import numpy as np
import pandas as pd

# ===== 基礎工具 =====

def _infer_tick_size_from_prices(prices, max_unique=5000):
    s = pd.Series(prices).dropna().astype(float)
    if s.empty or s.nunique() < 2:
        return np.nan
    vals = np.sort(s.unique())
    if len(vals) > max_unique:
        vals = np.sort(np.random.choice(vals, size=max_unique, replace=False))
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    tick = np.quantile(diffs, 0.1)  # 取小分位以對抗雜質
    # 合理小數位
    if tick < 1e-6: return round(tick, 6)
    if tick < 1e-4: return round(tick, 5)
    if tick < 1e-3: return round(tick, 4)
    return round(tick, 3)

def _rolling_percentile(x, window, min_periods=None):
    if min_periods is None:
        min_periods = max(10, window // 5)
    s = pd.Series(x, dtype=float)
    out = s.rolling(window=window, min_periods=min_periods)\
           .apply(lambda v: (v.rank(pct=True).iloc[-1]) if len(v)>0 else np.nan, raw=False)
    return out.values

def _zscore(x, window, min_periods=None):
    if min_periods is None:
        min_periods = max(10, window // 5)
    s = pd.Series(x, dtype=float)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mu) / (sd.replace(0, np.nan))
    return z.values

def _ensure_tz_series(series, tz="Asia/Shanghai"):
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        s = s.dt.tz_localize(tz)
    else:
        s = s.dt.tz_convert(tz)
    return s

# ===== A股時段分組 =====

def _label_bucket_cn_a(ts):
    # ts 為帶時區的 Timestamp（Asia/Shanghai）
    t = ts.timetz()  # 包含時區資訊的 time
    t0930 = pd.to_datetime("09:30").tz_localize("Asia/Shanghai").timetz()
    t0935 = pd.to_datetime("09:35").tz_localize("Asia/Shanghai").timetz()
    t1125 = pd.to_datetime("11:25").tz_localize("Asia/Shanghai").timetz()
    t1130 = pd.to_datetime("11:30").tz_localize("Asia/Shanghai").timetz()
    t1300 = pd.to_datetime("13:00").tz_localize("Asia/Shanghai").timetz()
    t1305 = pd.to_datetime("13:05").tz_localize("Asia/Shanghai").timetz()
    t1445 = pd.to_datetime("14:45").tz_localize("Asia/Shanghai").timetz()
    t1457 = pd.to_datetime("14:57").tz_localize("Asia/Shanghai").timetz()
    t1500 = pd.to_datetime("15:00").tz_localize("Asia/Shanghai").timetz()
    t0915 = pd.to_datetime("09:15").tz_localize("Asia/Shanghai").timetz()

    # 禁做：集合競價 09:15–09:30
    if t >= t0915 and t < t0930:
        return "auction_open"
    # 開盤5分鐘
    if t >= t0930 and t < t0935:
        return "open"
    # 上午常態
    if t >= t0935 and t < t1125:
        return "morning"
    # 午前尾段（更嚴）
    if t >= t1125 and t < t1130:
        return "pre_lunch_end"
    # 午休
    if t >= t1130 and t < t1300:
        return "lunch_break"
    # 午後mini-open
    if t >= t1300 and t < t1305:
        return "mini_open"
    # 午後常態
    if t >= t1305 and t < t1445:
        return "afternoon"
    # 尾盤15分鐘
    if t >= t1445 and t < t1457:
        return "last15"
    # 收盤集合競價（禁做）
    if t >= t1457 and t <= t1500:
        return "close_auction"
    # 其他（盤外/異常）
    return "off"

# 每個分組的門檻微調（你可依標的流動性調整）
BUCKET_CONF = {
    "auction_open": {"ban": True},
    "lunch_break":  {"ban": True},
    "close_auction":{"ban": True},
    "off":          {"ban": True},

    "open":        {"S_mul": 1.15, "alpha_add": 0.0003, "cost_mul": 1.3, "T_wait": 3},
    "mini_open":   {"S_mul": 1.08, "alpha_add": 0.0002, "cost_mul": 1.2, "T_wait": 3},
    "morning":     {"S_mul": 1.00, "alpha_add": 0.0000, "cost_mul": 1.0, "T_wait": 5},
    "pre_lunch_end":{"S_mul": 1.20, "alpha_add": 0.0003, "cost_mul": 0.8, "T_wait": 3},
    "afternoon":   {"S_mul": 1.00, "alpha_add": 0.0000, "cost_mul": 1.0, "T_wait": 5},
    "last15":      {"S_mul": 1.10, "alpha_add": 0.0002, "cost_mul": 0.9, "T_wait": 3},
}

# ===== 主函式：含分組版 =====

def generate_front_run_signals_ashare(intraday_tick: pd.DataFrame,
                                      intraday_trade: pd.DataFrame = None,
                                      params: dict = None) -> pd.DataFrame:
    """
    A股分組版：低價股友善的「先手預判→輕確認→加倉/退出」。
    會根據盤中分組自動調整 S 門檻、突破 alpha、成本上限 cost_gate、T_wait。
    """

    # 參數（可外部覆蓋）
    P = {
        'W_box': 60, 'W_touch': 10, 'W_fast': 3,
        'W_rv': 30, 'W_norm': 1200, 'W_z': 180,
        'alpha_base': 0.0007,
        'max_rel_spread': 0.0012,
        'S_threshold': 0.70,
        'T_wait': 5,
        'confirm_ofi_z': 0.2,
        'confirm_vol_z': 0.6,
        'fail_retrace_ticks': 0.5,
        'w_dq': 0.30, 'w_askthin': 0.20, 'w_bidstep': 0.15,
        'w_touch': 0.15, 'w_pullback': 0.10, 'w_spread': 0.10, 'w_resil': 0.10,
        'tz': "Asia/Shanghai",
    }
    if params:
        P.update(params)

    # ---------- 數據準備 ----------
    q = intraday_tick.copy()
    # 標準欄位
    rename_map = {}
    for c in q.columns:
        lc = c.lower()
        if lc == 'time': rename_map[c] = 'time'
        elif lc == 'bidprice': rename_map[c] = 'bidPrice'
        elif lc == 'bidsize': rename_map[c] = 'bidSize'
        elif lc == 'askprice': rename_map[c] = 'askPrice'
        elif lc == 'asksize': rename_map[c] = 'askSize'
    q = q.rename(columns=rename_map)

    need_cols = {'time','bidPrice','bidSize','askPrice','askSize'}
    miss = need_cols - set(q.columns)
    if miss:
        raise ValueError(f"intraday_tick 缺少欄位: {miss}")

    # 時區統一
    q['time'] = _ensure_tz_series(q['time'], tz=P['tz'])
    q = q.sort_values('time').reset_index(drop=True)

    # 基本量價
    q['mid'] = (q['bidPrice'].astype(float) + q['askPrice'].astype(float)) / 2.0
    q['spread'] = (q['askPrice'] - q['bidPrice']).astype(float)
    q['rel_spread'] = q['spread'] / q['mid'].replace(0, np.nan)

    # 推斷 tick_size / tick_rel
    tick_guess = _infer_tick_size_from_prices(pd.concat([q['bidPrice'], q['askPrice'], q['mid']], axis=0).values)
    q['tick_size'] = tick_guess
    q['tick_rel'] = q['tick_size'] / q['mid'].replace(0, np.nan)

    # L1 隊列不平衡/變化
    q['q_imb'] = (q['bidSize'] - q['askSize']) / (q['bidSize'] + q['askSize']).replace(0, np.nan)
    q['dq'] = q['q_imb'] - q['q_imb'].shift(1)

    # ask 薄化頻率（W_fast 內下降比例）
    d_asize = q['askSize'].diff()
    ask_down = (d_asize < 0).astype(int)
    q['ask_thin_freq'] = ask_down.rolling(P['W_fast'], min_periods=1).mean()

    # bid 上跳步頻（W_fast 內 bidPrice 上移次數）
    bid_step_raw = (q['bidPrice'] > q['bidPrice'].shift(1)).astype(int)
    q['bid_step'] = bid_step_raw.rolling(P['W_fast'], min_periods=1).sum()

    # 箱體/觸碰/回撤
    roll_max = q['mid'].rolling(P['W_box'], min_periods=1).max()
    roll_min = q['mid'].rolling(P['W_box'], min_periods=1).min()
    q['range_width'] = (roll_max - roll_min) / q['mid'].replace(0, np.nan)

    tol = 0.5 * q['tick_rel']
    q['touch_upper'] = ((roll_max - q['mid'])/q['mid'] <= tol).astype(int)
    q['touch_density'] = q['touch_upper'].rolling(P['W_touch'], min_periods=1).mean()

    recent_min = q['mid'].rolling(P['W_touch'], min_periods=1).min()
    q['pullback_depth'] = (roll_max - recent_min) / q['mid'].replace(0, np.nan)

    # 壓縮準備：range vs rv 的分位
    ret = q['mid'].pct_change()
    q['rv_short'] = ret.rolling(P['W_rv'], min_periods=5).std()
    q['range_pct'] = _rolling_percentile(q['range_width'].values, P['W_norm'])
    q['rv_pct'] = _rolling_percentile(q['rv_short'].values, P['W_norm'])
    q['compression'] = ((q['range_pct'] <= 0.2) & (q['rv_pct'] >= 0.4)).astype(int)

    # z 分供評分
    q['z_dq'] = _zscore(q['dq'].values, P['W_z'])
    q['z_askthin'] = _zscore(q['ask_thin_freq'].values, P['W_z'])
    q['z_bidstep'] = _zscore(q['bid_step'].values, P['W_z'])
    q['z_touch'] = _zscore(q['touch_density'].values, P['W_z'])
    q['z_pullback'] = _zscore(q['pullback_depth'].values, P['W_z'])
    q['z_spread'] = _zscore(q['rel_spread'].values, P['W_z'])
    q['z_resil'] = _zscore((-d_asize.fillna(0)).values, P['W_z'])  # 回補慢=好

    # 前導評分 S
    q['S'] = (
        P['w_dq'] * q['z_dq'].fillna(0) +
        P['w_askthin'] * q['z_askthin'].fillna(0) +
        P['w_bidstep'] * q['z_bidstep'].fillna(0) +
        P['w_touch'] * q['z_touch'].fillna(0) +
        P['w_pullback'] * (-q['z_pullback'].fillna(0)) +
        P['w_spread'] * (-q['z_spread'].fillna(0)) +
        P['w_resil'] * q['z_resil'].fillna(0)
    )

    # 自適應突破與成本
    q['alpha_adp_base'] = np.maximum(P['alpha_base'], 0.5 * q['tick_rel'])
    q['cost_gate_base'] = np.minimum(P['max_rel_spread'], 0.8 * q['tick_rel'])

    # 盤中分組
    q['bucket'] = q['time'].apply(_label_bucket_cn_a)

    # 將分組微調應用到每筆
    ban_mask = np.zeros(len(q), dtype=bool)
    S_adj = np.full(len(q), P['S_threshold'], dtype=float)
    alpha_adj = q['alpha_adp_base'].values.copy()
    cost_adj = q['cost_gate_base'].values.copy()
    T_wait_arr = np.full(len(q), P['T_wait'], dtype=int)

    conf_map = BUCKET_CONF
    for i, b in enumerate(q['bucket'].values):
        conf = conf_map.get(b, {})
        if conf.get("ban", False):
            ban_mask[i] = True
        S_adj[i] = P['S_threshold'] * conf.get("S_mul", 1.0)
        alpha_adj[i] = alpha_adj[i] + conf.get("alpha_add", 0.0)
        cost_adj[i] = cost_adj[i] * conf.get("cost_mul", 1.0)
        if "T_wait" in conf:
            T_wait_arr[i] = int(conf["T_wait"])

    q['S_threshold_adj'] = S_adj
    q['alpha_adp'] = alpha_adj
    q['cost_gate'] = cost_adj
    q['ban'] = ban_mask
    q['T_wait_sec'] = T_wait_arr

    # 微突破/允許條件
    prebreak_ok = (q['mid'] >= (roll_max * (1 + q['alpha_adp']))) | (q['touch_density'] >= 0.2)
    spread_ok = (q['rel_spread'] <= q['cost_gate'])
    compression_ok = (q['compression'] == 1)

    # 先手入場（小倉）
    q['front_run_enter'] = (
        (q['S'] >= q['S_threshold_adj']) &
        prebreak_ok & spread_ok & compression_ok &
        (~q['ban'])
    ).astype(int)

    # 交易資料（可選）→ vol_z
    if intraday_trade is not None and 'time' in intraday_trade.columns:
        t = intraday_trade.copy()
        t = t.rename(columns={c: ('time' if c.lower()=='time' else c) for c in t.columns})
        t['time'] = _ensure_tz_series(t['time'], tz=P['tz'])
        t = t.sort_values('time')
        q = pd.merge_asof(q.sort_values('time'),
                          t[['time','volume','price']].sort_values('time'),
                          on='time', direction='backward', tolerance=pd.Timedelta('1s'))
        vol = q['volume'].fillna(0).astype(float)
        q['vol_z'] = _zscore(vol.values, P['W_z'])
    else:
        q['vol_z'] = np.nan

    # OFI 近似（L1 proxy）
    ofi_proxy = (q['bidPrice'].diff().fillna(0) > 0).astype(int) - (q['askPrice'].diff().fillna(0) < 0).astype(int)
    q['ofi_z'] = _zscore(ofi_proxy.values, P['W_z'])

    # 輕確認條件
    q['light_confirm'] = (
        (q['ofi_z'] >= P['confirm_ofi_z']) |
        (q['vol_z'] >= P['confirm_vol_z']) |
        ((q['touch_density'] >= 0.3) & spread_ok)
    )

    # ---------- 事件流程：用「秒」為窗口（依分組 T_wait） ----------
    q['confirm_add'] = 0
    q['fail_exit'] = 0
    q['time_exit'] = 0
    q['pos_suggest'] = 0.0
    q.loc[q['front_run_enter'] == 1, 'pos_suggest'] = 0.35

    # 逐筆處理先手點：以秒為窗口（精確，不受不規則tick影響）
    # 註：為簡潔採用迴圈，日內量通常可接受；若極高速資料再做向量化優化
    times = q['time'].values
    mids = q['mid'].values
    tick_rel = q['tick_rel'].values
    light = q['light_confirm'].values.astype(bool)

    enter_idx = np.flatnonzero(q['front_run_enter'].values == 1)
    for idx in enter_idx:
        t0 = times[idx]
        wait_s = int(q['T_wait_sec'].iloc[idx])
        t1 = t0 + pd.Timedelta(seconds=wait_s)
        # 見未來窗口索引
        mask = (times > t0) & (times <= t1)
        if not mask.any():
            q.at[idx, 'time_exit'] = 1
            q.at[idx, 'pos_suggest'] = 0.0
            continue
        # 輕確認
        if light[mask].any():
            q.at[idx, 'confirm_add'] = 1
            q.at[idx, 'pos_suggest'] = 1.0
            continue
        # 回撤判斷（相對 0.5 tick）
        min_mid = np.nanmin(mids[mask])
        retrace_thr_abs = P['fail_retrace_ticks'] * tick_rel[idx] * mids[idx]
        if (mids[idx] - min_mid) >= retrace_thr_abs:
            q.at[idx, 'fail_exit'] = 1
            q.at[idx, 'pos_suggest'] = 0.0
        else:
            q.at[idx, 'time_exit'] = 1
            q.at[idx, 'pos_suggest'] = 0.0

    # ---------- 輸出 ----------
    out_cols = [
        'time','bucket','mid','bidPrice','askPrice','bidSize','askSize',
        'spread','rel_spread','tick_size','tick_rel',
        'q_imb','dq','ask_thin_freq','bid_step','touch_density','pullback_depth',
        'range_width','range_pct','rv_short','rv_pct','compression',
        'S','S_threshold_adj','alpha_adp','cost_gate','vol_z','ofi_z',
        'front_run_enter','confirm_add','fail_exit','time_exit','pos_suggest','T_wait_sec'
    ]
    return q[out_cols].copy()