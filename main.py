# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import argparse
import logging
from datetime import datetime, date
from typing import Iterable, List

import akshare as ak
import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import execute_values


# =========================
# Config (env-overridable)
# =========================
DEFAULT_DB_URL = "postgresql+psycopg2://user:pass@postgres:5432/market"

DB_URL = os.getenv("DB_URL", DEFAULT_DB_URL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
TZ_NAME = os.getenv("TZ", "Asia/Shanghai")

# API fetch behavior
DEFAULT_PERIOD = os.getenv("PERIOD", "1")   # minute period, "1" means 1-minute bars
DEFAULT_ADJUST = os.getenv("ADJUST", "qfq") # "qfq", "hfq", or None per akshare docs
DELAY_MIN = int(os.getenv("DELAY_MIN", "10"))
DELAY_MAX = int(os.getenv("DELAY_MAX", "20"))
PAGE_SIZE = int(os.getenv("UPSERT_PAGE_SIZE", "1000"))

# Symbols: override with env CSV if desired
DEFAULT_SYMBOLS = [
    "sh000001", "sh000300", "sh601816", "sh601658", "sh600221", "sh600029", "sh600519",
    "sz399001", "sz002456", "sz300316", "sz003816", "sz000617", "sz002498"
]
ENV_SYMBOLS = os.getenv("SYMBOLS")  # comma-separated
SYMBOLS = [s.strip() for s in ENV_SYMBOLS.split(",")] if ENV_SYMBOLS else DEFAULT_SYMBOLS


# =========================
# Logging
# =========================
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# =========================
# Database DDL and helpers
# =========================
DDL = """
CREATE TABLE IF NOT EXISTS quotes_minute (
  symbol text NOT NULL,
  ts timestamptz NOT NULL,
  open numeric,
  high numeric,
  low numeric,
  close numeric,
  volume numeric,
  PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS quotes_minute_ts_idx ON quotes_minute (ts);

CREATE TABLE IF NOT EXISTS fetch_log (
  id bigserial PRIMARY KEY,
  symbol text NOT NULL,
  date date NOT NULL,
  granularity text NOT NULL DEFAULT '1m',
  status text NOT NULL CHECK (status IN ('success','empty','error','partial')),
  rows_ingested integer NOT NULL DEFAULT 0,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  error text
);

CREATE UNIQUE INDEX IF NOT EXISTS fetch_log_unique ON fetch_log(symbol, date, granularity);
"""

UPSERT_SQL = """
INSERT INTO quotes_minute (symbol, ts, open, high, low, close, volume)
VALUES %s
ON CONFLICT (symbol, ts) DO UPDATE SET
  open = EXCLUDED.open,
  high = EXCLUDED.high,
  low = EXCLUDED.low,
  close = EXCLUDED.close,
  volume = EXCLUDED.volume;
"""


def get_engine() -> Engine:
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    return engine


def init_db(engine: Engine) -> None:
    logger.info("Initializing database schema (if needed)")
    with engine.begin() as conn:
        # Split on semicolons to execute each statement
        for stmt in [s.strip() for s in DDL.strip().split(";")]:
            if stmt:
                conn.execute(text(stmt))


def already_fetched(engine: Engine, symbol: str, d: date, granularity: str = "1m") -> bool:
    q = text("""
        SELECT 1 FROM fetch_log
        WHERE symbol = :symbol AND date = :d AND granularity = :g AND status = 'success'
        LIMIT 1
    """)
    with engine.begin() as conn:
        r = conn.execute(q, {"symbol": symbol, "d": d, "g": granularity}).first()
    return r is not None


def mark_fetch(
    engine: Engine,
    symbol: str,
    d: date,
    granularity: str,
    status: str,
    rows: int,
    error: str | None = None,
) -> None:
    upsert = text("""
        INSERT INTO fetch_log (symbol, date, granularity, status, rows_ingested, finished_at, error)
        VALUES (:symbol, :d, :g, :status, :rows, now(), :error)
        ON CONFLICT (symbol, date, granularity)
        DO UPDATE SET
            status = EXCLUDED.status,
            rows_ingested = EXCLUDED.rows_ingested,
            finished_at = EXCLUDED.finished_at,
            error = EXCLUDED.error
    """)
    with engine.begin() as conn:
        conn.execute(upsert, {
            "symbol": symbol,
            "d": d,
            "g": granularity,
            "status": status,
            "rows": rows,
            "error": error
        })


def dedupe_preserve_order(symbols: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in symbols:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# =========================
# Ingestion
# =========================
def to_tz_aware(ts_series: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(ts_series, errors="coerce")
    # If already tz-aware, convert; else localize
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz)
    else:
        s = s.dt.tz_convert(tz)
    return s


def ingest_minute_df(engine: Engine, symbol: str, df: pd.DataFrame, page_size: int = PAGE_SIZE) -> int:
    if df is None or df.empty:
        return 0

    # Normalize columns from akshare: expect 'day', 'open', 'high', 'low', 'close', 'volume'
    rename_map = {"day": "ts", "vol": "volume", "amount": "volume"}
    df = df.rename(columns=rename_map)

    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Timestamps -> Asia/Shanghai, then store as UTC in DB
    df["ts"] = to_tz_aware(df["ts"], TZ_NAME).dt.tz_convert("UTC")

    # Ensure numeric dtypes
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NA timestamps
    df = df.dropna(subset=["ts"])

    # Attach symbol
    df["symbol"] = symbol

    # Records for upsert
    # psycopg2 expects native Python types; convert Timestamp to python datetime (UTC)
    records = list(
        zip(
            df["symbol"].astype(str),
            df["ts"].dt.tz_localize(None).dt.to_pydatetime(),
            df["open"].astype(float),
            df["high"].astype(float),
            df["low"].astype(float),
            df["close"].astype(float),
            df["volume"].astype(float),
        )
    )
    if not records:
        return 0

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, UPSERT_SQL, records, page_size=page_size)
        raw_conn.commit()
    finally:
        raw_conn.close()

    return len(records)


def filter_df_to_date_shanghai(df: pd.DataFrame, d: date) -> pd.DataFrame:
    # df['day'] may be naive; localize to Asia/Shanghai for correct filtering
    if "day" not in df.columns:
        return df
    day_series = pd.to_datetime(df["day"], errors="coerce")
    if getattr(day_series.dt, "tz", None) is None:
        day_series = day_series.dt.tz_localize(TZ_NAME)
    else:
        day_series = day_series.dt.tz_convert(TZ_NAME)
    start = pd.Timestamp(d, tz=TZ_NAME)
    end = start + pd.Timedelta(days=1)
    mask = (day_series >= start) & (day_series < end)
    return df.loc[mask].copy()


def fetch_and_store_symbol_day(
    engine: Engine,
    symbol: str,
    d: date,
    period: str = DEFAULT_PERIOD,
    adjust: str = DEFAULT_ADJUST,
    delay_range: tuple[int, int] = (DELAY_MIN, DELAY_MAX),
) -> None:
    granularity = "1m"

    if already_fetched(engine, symbol, d, granularity=granularity):
        logger.info(f"[SKIP] already fetched {symbol} {d}")
        return

    try:
        logger.info(f"[CALL] ak.stock_zh_a_minute(symbol={symbol}, period={period}, adjust={adjust})")
        df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust=adjust)

        if df is None or df.empty:
            mark_fetch(engine, symbol, d, granularity, status="empty", rows=0)
            logger.warning(f"[EMPTY] {symbol} {d}")
        else:
            # Try to keep only rows for the requested date in local market time
            df_day = filter_df_to_date_shanghai(df, d)
            target_df = df_day if not df_day.empty else df

            rows = ingest_minute_df(engine, symbol, target_df)
            mark_fetch(engine, symbol, d, granularity, status="success", rows=rows)
            logger.info(f"[INGEST] {symbol} {d} rows={rows}")

    except Exception as e:
        logger.exception(f"[ERR] fetch {symbol} {d}: {e}")
        mark_fetch(engine, symbol, d, granularity, status="error", rows=0, error=str(e))

    # Rate limiting sleep between symbols
    lo, hi = delay_range
    if hi < lo:
        hi = lo
    wait = random.randint(int(lo), int(hi))
    logger.info(f"[SLEEP] {wait}s before next call")
    time.sleep(wait)


def run_daily(engine: Engine, symbols: List[str], d: date, delay_range=(DELAY_MIN, DELAY_MAX)) -> None:
    symbols = dedupe_preserve_order(symbols)
    for sym in symbols:
        fetch_and_store_symbol_day(engine, sym, d, delay_range=delay_range)


def business_days_between(end_date: date, months: int = 9) -> List[date]:
    # Start date N calendar months before end_date, inclusive
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(months=months)).date()
    # Business day range in local calendar (Mon-Fri); adjust if need exchange-specific holidays
    bdays = pd.bdate_range(start=start_date, end=end_date, freq="C")
    return [d.date() for d in bdays]


def run_backfill(
    engine: Engine,
    symbols: List[str],
    end_date: date,
    months: int = 9,
    delay_range=(5, 10),
) -> None:
    days = business_days_between(end_date=end_date, months=months)
    logger.info(f"[BACKFILL] {len(days)} business days from {(pd.Timestamp(end_date) - pd.DateOffset(months=months)).date()} to {end_date}")
    for d in days:
        logger.info(f"[DAY] {d}")
        run_daily(engine, symbols, d, delay_range=delay_range)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A-share minute ETL to PostgreSQL")
    p.add_argument("--mode", choices=["daily", "backfill"], default="daily",
                   help="Run mode: daily for prev business day, backfill for last N months")
    p.add_argument("--months", type=int, default=9,
                   help="Backfill lookback in calendar months (used in backfill mode)")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols; overrides env and defaults")
    p.add_argument("--delay-min", type=int, default=DELAY_MIN, help="Min delay between API calls (s)")
    p.add_argument("--delay-max", type=int, default=DELAY_MAX, help="Max delay between API calls (s)")
    p.add_argument("--period", type=str, default=DEFAULT_PERIOD, help="akshare minute period, e.g., '1'")
    p.add_argument("--adjust", type=str, default=DEFAULT_ADJUST, help="akshare adjust param, e.g., 'qfq'")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Symbols override
    symbols = SYMBOLS
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Apply runtime delay config
    global DEFAULT_PERIOD, DEFAULT_ADJUST
    DEFAULT_PERIOD = args.period
    DEFAULT_ADJUST = args.adjust
    delay_range = (max(0, args.delay_min), max(0, args.delay_max))

    # DB
    engine = get_engine()
    init_db(engine)

    # Determine dates
    prev_bday = (datetime.today() - BDay(1)).date()

    if args.mode == "backfill":
        run_backfill(engine, symbols, end_date=prev_bday, months=args.months, delay_range=delay_range)
        logger.info("[DONE] backfill")
    else:
        run_daily(engine, symbols, prev_bday, delay_range=delay_range)
        logger.info("[DONE] daily")


if __name__ == "__main__":
    main()