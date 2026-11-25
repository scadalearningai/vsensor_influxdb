"""
InfluxDB 1.x Client and Data Access Layer for Virtual Sensor WHB #1.
Supports querying time-series process tags into pandas DataFrame and writing soft sensor predictions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from influxdb import DataFrameClient, InfluxDBClient

# Load .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DEFAULT_HOST = os.getenv("INFLUX_HOST", "192.168.5.216")
DEFAULT_PORT = int(os.getenv("INFLUX_PORT", "8086"))
DEFAULT_USER = os.getenv("INFLUX_USER", "spcuser")
DEFAULT_PASSWORD = os.getenv("INFLUX_PASSWORD", "Kembar12345")
DEFAULT_DATABASE = os.getenv("INFLUX_DATABASE", "whb_medco")
DEFAULT_HISTORIAN_MEASUREMENT = "historian_clean"


def get_client(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    username: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    database: str = DEFAULT_DATABASE,
) -> InfluxDBClient:
    """Returns an authenticated InfluxDBClient (v1.x)."""
    return InfluxDBClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        timeout=60,
    )


def get_dataframe_client(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    username: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    database: str = DEFAULT_DATABASE,
) -> DataFrameClient:
    """Returns an authenticated DataFrameClient (v1.x)."""
    return DataFrameClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        timeout=60,
    )


def ensure_database_exists(database: str = DEFAULT_DATABASE) -> None:
    """Ensures that the target database exists in InfluxDB."""
    client = get_client(database="")
    existing_dbs = [d["name"] for d in client.get_list_database()]
    if database not in existing_dbs:
        client.create_database(database)
        print(f"[InfluxDB] Database '{database}' created successfully.")
    client.close()


def query_dataframe(
    query_str: str,
    database: str = DEFAULT_DATABASE,
) -> pd.DataFrame:
    """
    Executes an InfluxQL query and returns a pandas DataFrame.
    Normalizes 'time' column into a standard 'Timestamp' column without timezone.
    """
    client = get_client(database=database)
    try:
        res = client.query(query_str)
        points = list(res.get_points())
        if not points:
            return pd.DataFrame()

        df = pd.DataFrame(points)
        if "time" in df.columns:
            df = df.rename(columns={"time": "Timestamp"})

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            if df["Timestamp"].dt.tz is not None:
                df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)

        return df
    finally:
        client.close()


def query_interval_tags(
    tags: Iterable[str],
    start_dt: Union[str, pd.Timestamp],
    end_dt: Union[str, pd.Timestamp],
    measurement: str = DEFAULT_HISTORIAN_MEASUREMENT,
    database: str = DEFAULT_DATABASE,
) -> pd.DataFrame:
    """
    Queries specific process tags between start_dt and end_dt.
    Handles quoting of tag names with special characters (e.g. '46-FIC-103A1.A').
    """
    t_start = pd.to_datetime(start_dt).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_end = pd.to_datetime(end_dt).strftime("%Y-%m-%dT%H:%M:%SZ")

    quoted_tags = [f'"{t}"' for t in tags if t != "Timestamp"]
    fields_clause = ", ".join(quoted_tags)

    query_str = (
        f'SELECT {fields_clause} FROM "{measurement}" '
        f"WHERE time >= '{t_start}' AND time <= '{t_end}'"
    )

    df = query_dataframe(query_str, database=database)
    if df.empty:
        return df

    # Ensure chronological order
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


def query_multiple_intervals(
    tags: Iterable[str],
    intervals_df: pd.DataFrame,
    start_col: str = "start_dt",
    end_col: str = "end_dt",
    measurement: str = DEFAULT_HISTORIAN_MEASUREMENT,
    database: str = DEFAULT_DATABASE,
) -> pd.DataFrame:
    """
    Loads data across multiple intervals and concatenates them in chronological order.
    """
    frames = []
    for _, row in intervals_df.iterrows():
        df_int = query_interval_tags(
            tags=tags,
            start_dt=row[start_col],
            end_dt=row[end_col],
            measurement=measurement,
            database=database,
        )
        if not df_int.empty:
            frames.append(df_int)

    if not frames:
        return pd.DataFrame(columns=["Timestamp"] + list(tags))

    df_full = pd.concat(frames, ignore_index=True)
    df_full = df_full.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df_full


def escape_tag_value(val: Any) -> str:
    """Escapes tag values for Line Protocol."""
    return str(val).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def escape_field_key(key: str) -> str:
    """Escapes field keys for Line Protocol."""
    return str(key).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def write_dataframe(
    df: pd.DataFrame,
    measurement: str,
    database: str = DEFAULT_DATABASE,
    tag_columns: Optional[List[str]] = None,
    time_column: str = "Timestamp",
    batch_size: int = 5000,
) -> bool:
    """
    Writes a pandas DataFrame into InfluxDB 1.x using fast line protocol batches.
    Handles exact nanosecond timestamp conversion.
    """
    if df.empty:
        return True

    ensure_database_exists(database)
    tag_cols = set(tag_columns or [])
    time_col = time_column if time_column in df.columns else "time"
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame columns: {df.columns.tolist()}")

    field_cols = [c for c in df.columns if c not in tag_cols and c != time_col]

    client = get_client(database=database)
    try:
        lines = []
        # Convert timestamp to nanosecond integers
        timestamps_ns = pd.to_datetime(df[time_col]).astype("datetime64[ns]").astype("int64").values

        # Extract values
        tag_dict = {col: [escape_tag_value(v) for v in df[col].values] for col in tag_cols}

        total_rows = len(df)
        for i in range(total_rows):
            # 1. Measurement and tags
            line_parts = [measurement]
            if tag_dict:
                for col, vals in tag_dict.items():
                    line_parts.append(f"{escape_tag_value(col)}={vals[i]}")
            prefix = ",".join(line_parts)

            # 2. Fields
            fields = []
            for col in field_cols:
                val = df.iat[i, df.columns.get_loc(col)]
                if pd.isna(val) or val is None:
                    continue
                if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                    fields.append(f"{escape_field_key(col)}={val}i")
                elif isinstance(val, (float, np.floating)):
                    if np.isneginf(val) or np.isposinf(val) or np.isnan(val):
                        continue
                    fields.append(f"{escape_field_key(col)}={val}")
                elif isinstance(val, (bool, np.bool_)):
                    fields.append(f"{escape_field_key(col)}={str(val).lower()}")
                else:
                    escaped_str = str(val).replace('"', '\\"')
                    fields.append(f'{escape_field_key(col)}="{escaped_str}"')

            if not fields:
                continue

            fields_str = ",".join(fields)
            ts = timestamps_ns[i]
            lines.append(f"{prefix} {fields_str} {ts}")

            if len(lines) >= batch_size:
                client.write_points(lines, protocol="line", database=database)
                lines.clear()

        if lines:
            client.write_points(lines, protocol="line", database=database)
            lines.clear()

        return True
    finally:
        client.close()
