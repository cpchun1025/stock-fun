# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import time
import random

import akshare as ak
import pandas as pd
from pandas.tseries.offsets import BDay

# ============ Config ============
# Previous business day in yyyymmdd / yyyymm
today_str = (datetime.today() - BDay(1)).strftime("%Y%m%d")
today_mth = (datetime.today() - BDay(1)).strftime("%Y%m")

# Output folder: data/<yyyymmdd>
out_dir = Path("data") / today_str
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[INIT] Output folder: {out_dir.resolve()}")

# Symbols (can be a single str or a list)
# Example list (duplicates allowed; we will deduplicate while preserving order)
stocks = ['sh000001', 'sh000300', 'sh601816', 'sh601658', 'sh600221', 'sh600029', 'sh600519',
          'sz399001', 'sz002456', 'sz002456', 'sz300316', 'sz003816', 'sz000617', 'sz002498']

# If you sometimes only set `stock = "sh600519"`, normalize to a list:
# try:
#     stock  # type: ignore # noqa
#     if isinstance(stock, str):
#         stocks = [stock]
# except NameError:
#     pass

# ============ Helpers ============
def save_df(df: pd.DataFrame, path: Path, overwrite: bool = False) -> bool:
    """
    Save df to path. Returns True if saved, False if skipped.
    """
    if not overwrite and path.exists():
        print(f"[SKIP] Exists: {path.name}")
        return False
    if df is None:
        print(f"[SKIP] None:  {path.name}")
        return False
    if hasattr(df, "empty") and df.empty:
        print(f"[SKIP] Empty: {path.name}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {path.name}: {len(df):,} rows")
    return True

def fetch_market_data(folder: Path, overwrite: bool = False):
    """
    Fetch market-wise data and save, but skip if the CSV already exists.
    Extend the 'tasks' dict as you need.
    """
    tasks = {
        f"stock_sse_summary.csv":            lambda: ak.stock_sse_summary(),
        f"stock_szse_summary_{today_str}.csv": lambda: ak.stock_szse_summary(date=today_str),
        f"stock_sse_deal_daily_{today_str}.csv": lambda: ak.stock_sse_deal_daily(date=today_str),
        # Add monthly stuff if needed, e.g. saved under daily folder:
        # f"stock_szse_sector_summary_{today_mth}.csv": lambda: ak.stock_szse_sector_summary(),
    }

    for fname, fn in tasks.items():
        path = folder / fname
        if path.exists() and not overwrite:
            print(f"[SKIP] Market file exists: {fname}")
            continue
        try:
            df = fn()
        except Exception as e:
            print(f"[ERR ] Fetch market {fname}: {e}")
            continue
        save_df(df, path, overwrite=overwrite)

def fetch_stock_minutes(symbols, folder: Path, period: str = "1", adjust: str = "qfq",
                        delay_range=(10, 20), overwrite: bool = False):
    """
    Fetch minute data for each stock. Skips existing files.
    Sleeps 10–20s between iterations by default.
    """
    # Deduplicate while preserving order
    seen = set()
    uniq_symbols = []
    for s in symbols:
        if s not in seen:
            uniq_symbols.append(s)
            seen.add(s)

    for i, sym in enumerate(uniq_symbols, 1):
        fname = f"stock_zh_a_minute_{sym}.csv"
        path = folder / fname
        if path.exists() and not overwrite:
            print(f"[SKIP] {sym} exists: {fname}")
        else:
            try:
                print(f"[CALL] {i}/{len(uniq_symbols)} ak.stock_zh_a_minute({sym}, period={period}, adjust={adjust})")
                df = ak.stock_zh_a_minute(symbol=sym, period=period, adjust=adjust)
                save_df(df, path, overwrite=overwrite)
            except Exception as e:
                print(f"[ERR ] {sym}: {e}")

        # Sleep between calls to be gentle to the API
        lo, hi = delay_range
        wait = random.randint(int(lo), int(hi))
        print(f"[SLEEP] {wait}s before next call...")
        time.sleep(wait)

# ============ Run ============
if __name__ == "__main__":
    # 1) Market-wise (skip if exists)
    fetch_market_data(out_dir, overwrite=False)

    # 2) Stock-wise (skip if exists; sleep 10–20s per symbol)
    fetch_stock_minutes(stocks, out_dir, period="1", adjust="qfq",
                        delay_range=(10, 20), overwrite=False)

    print("[DONE] All tasks completed.")