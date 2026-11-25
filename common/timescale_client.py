"""
TimescaleDB (PostgreSQL) Client and Data Access Layer for Virtual Sensor WHB #1.
Supports writing soft sensor predictions and daily aggregations into TimescaleDB hypertables
with idempotent batch upserts and snake_case column normalization.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# Load .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DEFAULT_HOST = os.getenv("TIMESCALE_HOST", os.getenv("INFLUX_HOST", "localhost"))
DEFAULT_PORT = int(os.getenv("TIMESCALE_PORT", "5432"))
DEFAULT_USER = os.getenv("TIMESCALE_USER", "postgres")
DEFAULT_PASSWORD = os.getenv("TIMESCALE_PASSWORD", "")
DEFAULT_DATABASE = os.getenv("TIMESCALE_DATABASE", "whb_medco")
DEFAULT_SCHEMA = os.getenv("TIMESCALE_SCHEMA", "public")


def normalize_column_name(name: str) -> str:
    """
    Normalizes a DataFrame column name into a clean SQL-friendly snake_case identifier.
    Example: 'diff_%' -> 'diff_pct', 'measured_103A1' -> 'measured_103a1',
             '46-FIC-103A.Q_opening' -> 'tag_46_fic_103a_q_opening'.
    """
    s = str(name).strip()
    s = s.replace("%", "_pct")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"tag_{s}"
    return s


def _pandas_dtype_to_pg(dtype: Any) -> str:
    """Maps a pandas/numpy dtype to a PostgreSQL column type."""
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMPTZ"
    return "TEXT"


def get_connection(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    database: str = DEFAULT_DATABASE,
) -> psycopg2.extensions.connection:
    """Returns an authenticated psycopg2 connection to TimescaleDB."""
    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=database,
        connect_timeout=30,
    )


def ensure_database_exists(
    database: str = DEFAULT_DATABASE,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
) -> None:
    """Ensures the target PostgreSQL/TimescaleDB database exists if permissions allow."""
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname="postgres",
        connect_timeout=30,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
                print(f"[TimescaleDB] Database '{database}' created successfully.")
    except psycopg2.Error as exc:
        # Non-fatal if database already exists or user lacks CREATEDB privilege
        print(f"[TimescaleDB] Skipping database creation check for '{database}': {exc}")
    finally:
        conn.close()


def ensure_hypertable(
    conn: psycopg2.extensions.connection,
    table_name: str,
    columns_with_types: List[Tuple[str, str]],
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """
    Ensures the target schema, table (with 'time TIMESTAMPTZ PRIMARY KEY'),
    missing columns, and TimescaleDB hypertable exist.
    """
    full_table_name = f"{schema}.{table_name}"
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        except psycopg2.Error:
            conn.rollback()

        col_defs = [sql.SQL("{} TIMESTAMPTZ NOT NULL PRIMARY KEY").format(sql.Identifier("time"))]
        for col_name, pg_type in columns_with_types:
            col_defs.append(sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(pg_type)))

        create_stmt = sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
            sql.Identifier(schema),
            sql.Identifier(table_name),
            sql.SQL(", ").join(col_defs),
        )
        cur.execute(create_stmt)

        # Ensure any newly added columns exist on pre-existing tables
        for col_name, pg_type in columns_with_types:
            alter_stmt = sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} {}").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.Identifier(col_name),
                sql.SQL(pg_type),
            )
            cur.execute(alter_stmt)

        cur.execute(
            "SELECT create_hypertable(%s, %s, if_not_exists => TRUE, migrate_data => TRUE)",
            (full_table_name, "time"),
        )
    conn.commit()


def _sanitize_value(val: Any) -> Any:
    """Converts pandas/numpy scalars and NaN/Inf values into native Python/SQL types."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return str(val)


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    database: str = DEFAULT_DATABASE,
    schema: str = DEFAULT_SCHEMA,
    time_column: str = "Timestamp",
    batch_size: int = 5000,
    normalize_columns: bool = True,
) -> bool:
    """
    Writes a pandas DataFrame into a TimescaleDB hypertable using idempotent batch upserts.
    Normalizes column names to snake_case by default.
    """
    if df.empty:
        return True

    time_col = time_column if time_column in df.columns else "time"
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame columns: {df.columns.tolist()}")

    df_work = df.copy()
    ts_series = pd.to_datetime(df_work[time_col])
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")
    else:
        ts_series = ts_series.dt.tz_convert("UTC")

    feature_cols = [c for c in df_work.columns if c != time_col]
    if normalize_columns:
        col_mapping: Dict[str, str] = {c: normalize_column_name(c) for c in feature_cols}
    else:
        col_mapping = {c: c for c in feature_cols}

    pg_cols = [col_mapping[c] for c in feature_cols]
    columns_with_types = [(col_mapping[c], _pandas_dtype_to_pg(df_work[c].dtype)) for c in feature_cols]

    # Deduplicate by timestamp within the batch so ON CONFLICT doesn't hit the same key twice in one INSERT
    df_work["_ts_utc"] = ts_series
    df_work = df_work.drop_duplicates(subset=["_ts_utc"], keep="last").reset_index(drop=True)

    rows = []
    ts_py = df_work["_ts_utc"].dt.to_pydatetime()
    col_arrays = [df_work[c].to_numpy() for c in feature_cols]
    for i in range(len(df_work)):
        row = [ts_py[i]] + [_sanitize_value(arr[i]) for arr in col_arrays]
        rows.append(tuple(row))

    conn = get_connection(database=database)
    try:
        ensure_hypertable(conn, table_name=table_name, columns_with_types=columns_with_types, schema=schema)

        all_cols = ["time"] + pg_cols
        insert_cols = sql.SQL(", ").join(sql.Identifier(c) for c in all_cols)
        if pg_cols:
            update_assignments = sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
                for c in pg_cols
            )
            conflict_clause = sql.SQL("ON CONFLICT ({}) DO UPDATE SET {}").format(
                sql.Identifier("time"),
                update_assignments,
            )
        else:
            conflict_clause = sql.SQL("ON CONFLICT ({}) DO NOTHING").format(sql.Identifier("time"))

        upsert_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s {}").format(
            sql.Identifier(schema),
            sql.Identifier(table_name),
            insert_cols,
            conflict_clause,
        )

        with conn.cursor() as cur:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                execute_values(cur, upsert_query.as_string(conn), batch, page_size=batch_size)
        conn.commit()
        return True
    finally:
        conn.close()


def query_dataframe(
    query_str: str,
    params: Optional[Tuple[Any, ...]] = None,
    database: str = DEFAULT_DATABASE,
) -> pd.DataFrame:
    """
    Executes a SQL query against TimescaleDB and returns a pandas DataFrame.
    Normalizes 'time' column into a tz-naive 'Timestamp' column.
    """
    conn = get_connection(database=database)
    try:
        with conn.cursor() as cur:
            cur.execute(query_str, params)
            if cur.description is None:
                return pd.DataFrame()
            col_names = [desc[0] for desc in cur.description]
            records = cur.fetchall()

        if not records:
            return pd.DataFrame(columns=col_names)

        df = pd.DataFrame(records, columns=col_names)
        if "time" in df.columns:
            df = df.rename(columns={"time": "Timestamp"})

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            if df["Timestamp"].dt.tz is not None:
                df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        return df
    finally:
        conn.close()
