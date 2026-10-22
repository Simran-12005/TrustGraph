"""
EDA Tool
--------
Automated exploratory profiling of a (filtered) transaction slice.
Invoked by the planner only for broad/exploratory queries -- skipped for
single-entity or narrowly-targeted queries to keep latency low.
"""
import pandas as pd
import numpy as np

from backend.config import aml_config
from backend.tools._numeric_utils import safe_float


def run_eda(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"row_count": 0, "message": "No transactions matched the filters."}

    ctr_threshold = aml_config.get().ctr_threshold
    structuring_low = ctr_threshold * 0.85
    amount = df["amount"]
    by_type = df["txn_type"].value_counts().to_dict()
    by_channel = df["channel"].value_counts().to_dict()
    by_segment = df["segment"].value_counts().to_dict() if "segment" in df else {}

    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
    daily_volume = (
        df_ts.set_index("timestamp").resample("D")["amount"]
        .agg(["count", "sum"]).reset_index()
    )
    daily_volume["timestamp"] = daily_volume["timestamp"].dt.strftime("%Y-%m-%d")

    near_threshold = df[(df.amount >= structuring_low) & (df.amount < ctr_threshold)]

    return {
        "row_count": int(len(df)),
        "unique_customers": int(df["customer_id"].nunique()),
        "date_range": [str(df_ts.timestamp.min()), str(df_ts.timestamp.max())],
        "amount_stats": {
            "mean": round(float(amount.mean()), 2),
            "median": round(float(amount.median()), 2),
            "std": round(safe_float(amount.std()), 2),
            "p95": round(float(amount.quantile(0.95)), 2),
            "max": round(float(amount.max()), 2),
        },
        "by_txn_type": by_type,
        "by_channel": by_channel,
        "by_segment": by_segment,
        "near_ctr_threshold_count": int(len(near_threshold)),
        "near_ctr_threshold_pct": round(len(near_threshold) / len(df) * 100, 2),
        "daily_volume_series": daily_volume.to_dict(orient="records"),
    }
