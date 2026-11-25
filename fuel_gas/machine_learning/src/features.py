"""
Feature Preparation & Physics Gating module for Fuel Gas Soft Sensor (46-FIC-103A1.A).

Uses strictly the 7 verified raw operational process tags without derived features
(zero target/setpoint leakage):
- 46-TIC-101.A (Incinerator flue gas temperature)
- 46-TI-102.A (Feedwater temperature)
- 46-PIC-109.A (Steam drum pressure PV)
- 46-TIC-106 (Process temperature)
- 46-FIC-101_PV (Incinerator acid/feed flow PV)
- 46-AI-101.A (Excess O2 analyzer)
- 46-FIC-103A1.Q (Fuel gas control valve opening)
"""

from typing import List
import pandas as pd

from src.config import (
    MAX_STACK_O2,
    MIN_FLUE_TEMP,
    MIN_VALVE_OPENING,
    SELECTED_FEATURE_TAGS,
    VALVE_OPENING_TAG,
)


def build_features(df_raw: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Prepares raw process features and deterministic operational physics gating flags.

    - Ensures chronological ordering when Timestamp is present
    - Derives gating indicators (whb1_on, valve_open, fuel_gas_active) for shutdown safety
    """
    df = df_raw.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

    for tag in SELECTED_FEATURE_TAGS:
        if tag in df.columns:
            df[tag] = pd.to_numeric(df[tag], errors="coerce")

    q = df[VALVE_OPENING_TAG].clip(lower=0.0, upper=100.0)
    stack_o2 = df["46-AI-101.A"]
    flue_temp = df["46-TIC-101.A"]

    df["whb1_on"] = (
        (flue_temp > MIN_FLUE_TEMP)
        & ((stack_o2 < MAX_STACK_O2) | stack_o2.isna())
    ).astype(int)
    df["valve_open"] = (q > MIN_VALVE_OPENING).astype(int)
    df["fuel_gas_active"] = ((df["whb1_on"] == 1) & (df["valve_open"] == 1)).astype(int)

    return df


def get_feature_columns(feature_set: str = "selected") -> List[str]:
    """Returns the 5 verified raw process feature columns."""
    if feature_set in ("selected", "full"):
        return list(SELECTED_FEATURE_TAGS)
    raise ValueError(f"Unknown feature_set: {feature_set}")

