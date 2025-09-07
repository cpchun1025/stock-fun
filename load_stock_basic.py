#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from datetime import datetime, date

# === Config ===
CSV_PATH = r"D:\development\data\tushare_stock_basic_20250907015312.csv"
PG_HOST = "cpwork.myds.me"
PG_PORT = int(os.environ.get("PGPORT", "32769"))
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "b8cf9n7A")
PG_DATABASE = os.environ.get("PGDATABASE", "postgres")  # change if needed
TABLE = "tushare.stock_basic"

# Columns as per Tushare stock_basic (doc_id=25) and your DDL
ALL_COLS = [
    "ts_code", "symbol", "name", "area", "industry",
    "fullname", "enname", "cnspell", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
    "is_hs", "act_name", "act_ent_type"
]

DATE_COLS = ["list_date", "delist_date"]

CHUNKSIZE = 50000
BATCHSIZE = 10000


def parse_date_value(x):
    """
    Convert various representations to Python date or None.
    Accepts:
      - '' / '0' / 'NaT' / 'None' / 'nan' -> None
      - 'YYYYMMDD' -> date
      - 'YYYY-MM-DD' -> date
    """
    if x is None:
        return None
    # If it's already a datetime/date
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()

    s = str(x).strip()
    if s == "" or s in {"0", "NaT", "NAT", "None", "none", "NaN", "nan"}:
        return None

    # Try YYYYMMDD first
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            pass
    # Try ISO format YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        # Last resort: pandas to_datetime with coerce
        try:
            t = pd.to_datetime(s, errors="coerce")
            return None if pd.isna(t) else t.date()
        except Exception:
            return None


def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected columns exist
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = None

    # Trim strings to avoid stray spaces
    for c in df.columns:
        if c not in DATE_COLS:
            df[c] = df[c].astype(object).map(
                lambda v: v.strip() if isinstance(v, str) else v
            )

    # Parse date columns to Python date or None
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = df[c].map(parse_date_value)

    # Keep only known columns and in required order
    df = df[ALL_COLS]

    # Normalize empty strings to None everywhere
    df = df.where(~df.isin(["", " "]), None)

    # Important: ensure pandas NaN/NaT become None for psycopg2
    df = df.astype(object).where(pd.notna(df), None)
    return df


def connect():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DATABASE
    )


def upsert_chunk(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    df = sanitize_frame(df)

    cols = ALL_COLS
    columns_sql = ",".join([f'"{c}"' for c in cols])
    update_cols = [c for c in cols if c != "ts_code"]
    update_sql = ",".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])

    insert_sql = f"""
        INSERT INTO {TABLE} ({columns_sql})
        VALUES %s
        ON CONFLICT ("ts_code") DO UPDATE SET
        {update_sql};
    """

    records = list(df.itertuples(index=False, name=None))

    with conn.cursor() as cur:
        extras.execute_values(cur, insert_sql, records, page_size=BATCHSIZE)
    return len(records)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    total = 0
    conn = connect()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # keep schema handy
            cur.execute("SET search_path TO public, tushare;")

        print(f"Loading: {csv_path}")
        # Read as string to avoid implicit float/int coercion of codes
        for i, chunk in enumerate(pd.read_csv(
            csv_path,
            chunksize=CHUNKSIZE,
            dtype=str,
            encoding="utf-8-sig",
            engine="python"
        ), start=1):
            inserted = upsert_chunk(conn, chunk)
            conn.commit()
            total += inserted
            print(f"  chunk {i}: upserted {inserted} rows, total {total}")

        print(f"DONE. Total upserted rows: {total}")
    except Exception as e:
        conn.rollback()
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()


if __name__ == "__main__":
    main()