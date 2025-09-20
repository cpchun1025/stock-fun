# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import re

# -------------------------
# Config
# -------------------------
base_dir = Path("data")
date_str = "20250920"  # change or compute dynamically
data_dir = base_dir / date_str
overwrite = False  # True to force regenerate even if outputs exist

# -------------------------
# Fonts (optional Chinese)
# -------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------
# Plot helpers (your fixed versions)
# -------------------------
def plot_intraday_fixed(df_day: pd.DataFrame, ev_day: dict, title: str, save_path, *,
                        use_chinese=False):
    if use_chinese:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
        plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2.5, 1]})

    price_line, = ax[0].plot(df_day.index, df_day["close"], label="Close",
                             color="#1f77b4", lw=1.4, zorder=2)
    vwap_line = None
    if "vwap" in df_day.columns:
        vwap_line, = ax[0].plot(df_day.index, df_day["vwap"], label="VWAP",
                                color="#ff7f0e", lw=1.2, ls="--", zorder=1)

    def scatter_if_any(times, y_series, label, color, marker):
        if times is None or len(times) == 0:
            return None
        y = y_series.loc[times]
        return ax[0].scatter(times, y, label=label, color=color, marker=marker,
                             s=55, zorder=4, edgecolor="black", linewidths=0.5)

    h_bv  = scatter_if_any(ev_day.get("big_volume", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "Big Vol (Z≥3)", "#2ca02c", "o")
    h_br  = scatter_if_any(ev_day.get("big_return", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "Big Move (Z≥3)", "#d62728", "^")
    h_cb  = scatter_if_any(ev_day.get("combo_spike", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "Combo", "#9467bd", "s")
    h_bup = scatter_if_any(ev_day.get("breakout_up", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "Breakout Up", "#17becf", "P")
    h_bdn = scatter_if_any(ev_day.get("breakout_dn", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "Breakout Down", "#bcbd22", "X")
    h_vxu = scatter_if_any(ev_day.get("vwap_cross_up", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "VWAP Cross Up", "#ff9896", "v")
    h_vxd = scatter_if_any(ev_day.get("vwap_cross_dn", pd.DataFrame()).get("timestamp", []),
                           df_day["close"], "VWAP Cross Down", "#aec7e8", "1")

    ax[0].set_title(title)
    ax[0].set_ylabel("Price")
    ax[0].grid(True, ls="--", alpha=0.3)

    handles = [h for h in [price_line, vwap_line, h_bv, h_br, h_cb, h_bup, h_bdn, h_vxu, h_vxd] if h is not None]
    labels  = [h.get_label() for h in handles]
    if handles:
        ax[0].legend(handles, labels, ncol=4, fontsize=9)

    dt = mdates.date2num(df_day.index.to_pydatetime())
    if len(dt) >= 2:
        step = np.median(np.diff(dt))
        bar_width = step * 0.8
    else:
        bar_width = 0.0005

    ax[1].bar(df_day.index, df_day["volume"], width=bar_width, color="#8c564b", align="center")
    ax[1].set_ylabel("Volume")
    ax[1].grid(True, ls="--", alpha=0.3)

    is_single_day = df_day.index.normalize().nunique() == 1
    if is_single_day:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax[1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    else:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax[1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def volz_heatmap(df: pd.DataFrame, save_path,
                 date_fmt="%Y-%m-%d",
                 x_tick_every=6,
                 z_clip=3.0,
                 latest_on_top=True,
                 figsize_scale=0.28):
    tmp = df.copy()
    tmp["time_str"] = tmp.index.strftime("%H:%M")
    piv = tmp.pivot_table(index="date", columns="time_str", values="vol_z", aggfunc="mean")
    piv.index = pd.to_datetime(piv.index).strftime(date_fmt)
    if latest_on_top:
        piv = piv.iloc[::-1]
    fig_h = max(4, len(piv) * figsize_scale)
    plt.figure(figsize=(14, fig_h))
    hm_kwargs = dict(cmap="coolwarm", center=0, cbar_kws={"label": "Vol Z", "shrink": 0.9})
    if z_clip is not None:
        hm_kwargs.update(vmin=-float(z_clip), vmax=float(z_clip))
    ax = sns.heatmap(piv, **hm_kwargs)
    ax.set_title("成交量Z分数热力图")
    ax.set_ylabel("Date"); ax.set_xlabel("Time")
    cols = piv.columns.to_list()
    ticks = np.arange(0, len(cols), max(1, int(x_tick_every)))
    ax.set_xticks(ticks); ax.set_xticklabels([cols[i] for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

# -------------------------
# Data preparation + events
# -------------------------
def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"day": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    num_cols = ["open","high","low","close","volume"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("timestamp").set_index("timestamp")
    df["date"] = pd.to_datetime(df.index.date)
    df["tod"]  = df.index.time
    df["ret"] = np.log(df["close"]).diff()
    pv = df["close"] * df["volume"]
    df["cum_pv"]  = pv.groupby(df["date"]).cumsum()
    df["cum_vol"] = df["volume"].groupby(df["date"]).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]
    df["dist_to_vwap_bps"] = (df["close"] / df["vwap"] - 1.0) * 1e4

    def zscore_day(s: pd.Series):
        m, sd = s.mean(), s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - m) / sd

    df["vol_z"] = df.groupby("date")["volume"].transform(zscore_day)
    df["ret_z"] = df.groupby("date")["ret"].transform(zscore_day)
    return df

def in_session(df: pd.DataFrame):
    am = (df["tod"] >= time(9,31)) & (df["tod"] <= time(11,30))
    pm = (df["tod"] >= time(13,1)) & (df["tod"] <= time(15,0))
    return df[am | pm]

def detect_events(df: pd.DataFrame,
                  vol_z_thr=3.0, ret_z_thr=3.0,
                  breakout_window=30, breakout_vol_z=2.0,
                  streak_k=6, gap_open_thr=0.01):
    out = {}
    big_vol = df[df["vol_z"] >= vol_z_thr]
    big_ret = df[np.abs(df["ret_z"]) >= ret_z_thr]
    combo   = df[(df["vol_z"] >= vol_z_thr) & (np.abs(df["ret_z"]) >= ret_z_thr)]
    out["big_volume"] = big_vol.reset_index()
    out["big_return"] = big_ret.reset_index()
    out["combo_spike"] = combo.reset_index()

    def rolling_breakout(g):
        g = g.sort_index().copy()
        g["hh"] = g["high"].rolling(breakout_window, min_periods=breakout_window).max().shift(1)
        g["ll"] = g["low"].rolling(breakout_window,  min_periods=breakout_window).min().shift(1)
        g["breakout_up"] = (g["high"] > g["hh"]) & (g["vol_z"] >= breakout_vol_z)
        g["breakout_dn"] = (g["low"]  < g["ll"]) & (g["vol_z"] >= breakout_vol_z)
        return g

    rb = df.groupby("date", group_keys=False).apply(rolling_breakout)
    out["breakout_up"] = rb[rb["breakout_up"]].reset_index()
    out["breakout_dn"] = rb[rb["breakout_dn"]].reset_index()

    def streaks(g):
        g = g.sort_index().copy()
        g["chg"] = np.sign(g["close"].diff())
        nz = g["chg"].replace(0, np.nan).ffill().fillna(0)
        group_id = (nz != nz.shift()).cumsum()
        g["streak_len"] = g.groupby(group_id)["chg"].cumcount() + 1
        g["streak_dir"] = nz
        def last_per_run(x):
            rid = (x["streak_len"].diff().fillna(1) != 1).cumsum()
            return x.groupby(rid).tail(1)
        up_last = last_per_run(g[(g["streak_dir"] > 0) & (g["streak_len"] >= streak_k)].copy())
        dn_last = last_per_run(g[(g["streak_dir"] < 0) & (g["streak_len"] >= streak_k)].copy())
        return up_last, dn_last

    ups, dns = [], []
    for d, g in df.groupby("date"):
        u, v = streaks(g)
        if not u.empty: ups.append(u)
        if not v.empty: dns.append(v)
    out["streak_up"] = pd.concat(ups).reset_index() if ups else pd.DataFrame()
    out["streak_dn"] = pd.concat(dns).reset_index() if dns else pd.DataFrame()

    oc = df.loc[df["tod"] == time(9,31)][["date","open"]].copy()
    prev_close = df.loc[df["tod"] == time(15,0)][["date","close"]].copy()
    prev_close["date"] = prev_close["date"] + pd.Timedelta(days=1)
    gap = oc.merge(prev_close.rename(columns={"close":"prev_close"}), on="date", how="left")
    gap["gap_ret"] = gap["open"]/gap["prev_close"] - 1.0
    out["gap_open"] = gap[np.abs(gap["gap_ret"]) >= gap_open_thr].copy()

    close_auct = df[(df["tod"] == time(15,0)) & (df["vol_z"] >= 2.0)].copy().reset_index()
    out["close_auction_spike"] = close_auct

    def vwap_cross(g):
        g = g.sort_index().copy()
        sign = np.sign(g["close"] - g["vwap"])
        cross = (sign != sign.shift()) & sign.notna() & sign.shift().notna()
        g["vwap_cross_up"] = cross & (sign > 0) & (g["vol_z"] >= 2.0)
        g["vwap_cross_dn"] = cross & (sign < 0) & (g["vol_z"] >= 2.0)
        return g

    vc = df.groupby("date", group_keys=False).apply(vwap_cross)
    out["vwap_cross_up"] = vc[vc["vwap_cross_up"]].reset_index()
    out["vwap_cross_dn"] = vc[vc["vwap_cross_dn"]].reset_index()
    return out, rb, vc

def export_events(ev: dict, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in ev.items():
        (folder / f"{k}.csv").write_text(v.to_csv(index=False, encoding="utf-8-sig"))
    # Using write_text above to ensure encoding; if large, use v.to_csv directly:
    for k, v in ev.items():
        v.to_csv(folder / f"{k}.csv", index=False, encoding="utf-8-sig")

# -------------------------
# Runner
# -------------------------
def run_for_symbol_file(csv_path: Path):
    # symbol from file name: stock_zh_a_minute_{SYMBOL}.csv
    m = re.match(r"stock_zh_a_minute_(.+)\.csv$", csv_path.name)
    if not m:
        print(f"[SKIP] Not a stock minute file: {csv_path.name}")
        return
    symbol = m.group(1)
    out_dir = csv_path.parent / f"{symbol}_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If outputs already exist and not overwriting, skip heavy work
    png_any = any((out_dir / f).exists() for f in [
        f"{symbol}_volz_heatmap.png"
    ])
    if png_any and not overwrite:
        print(f"[SKIP] Outputs exist for {symbol}, folder {out_dir.name}")
        return

    print(f"[LOAD] {symbol}: {csv_path.name}")
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[SKIP] Empty CSV for {symbol}")
        return

    bars = prepare_bars(df)
    bars = in_session(bars)
    events, rb_full, vc_full = detect_events(bars)

    # Plot per day intraday
    for d, g in bars.groupby("date"):
        ev_day = {k: v[v["date"] == d].copy() if "date" in v.columns else v
                  for k, v in events.items()}
        plot_intraday_fixed(
            g,
            ev_day,
            title=f"{symbol} {str(d.date())} 分钟图（事件标记）",
            save_path=out_dir / f"{symbol}_{str(d.date())}_intraday.png",
            use_chinese=True
        )

    # Heatmap
    volz_heatmap(bars, out_dir / f"{symbol}_volz_heatmap.png")

    # Export event CSVs
    for k, v in events.items():
        v.to_csv(out_dir / f"{k}.csv", index=False, encoding="utf-8-sig")

    print(f"[DONE] {symbol} -> {out_dir}")

def main():
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

    files = sorted(p for p in data_dir.glob("stock_zh_a_minute_*.csv"))
    if not files:
        print(f"[INFO] No minute CSVs under {data_dir}")
        return

    for csv_path in files:
        run_for_symbol_file(csv_path)

    print(f"[ALL DONE] Outputs under {data_dir}")

if __name__ == "__main__":
    main()