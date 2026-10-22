"""
Feature Engineering Tool
-------------------------
Builds AML-relevant, model-ready / rule-ready features aggregated per
customer_id from a (filtered) transaction slice. Feature set is designed
around known typologies rather than generic stats:

  txn_count                 - raw activity level
  total_amount / avg_amount - volume
  amount_std                - behavioural volatility
  near_threshold_count/pct  - structuring signal (txns in $8.5k-$10k band)
  max_daily_txn_count       - velocity spike signal
  rapid_cycle_count         - layering signal (TRANSFER_IN followed by
                               TRANSFER_OUT of similar amount within 3h)
  distinct_counterparty_countries
  high_risk_country_txn_pct
  night_txn_pct             - off-hours activity (23:00-05:00)
"""
import pandas as pd
import numpy as np

from backend.config import aml_config
from backend.tools._numeric_utils import safe_float

# Fallback constants only. Live values are read from aml_config at call
# time in build_features() so POST /api/v1/rules/config actually changes
# detection behaviour instead of being cosmetic.
STRUCTURING_LOW_DEFAULT, STRUCTURING_HIGH_DEFAULT = 8500, 10000
RAPID_CYCLE_AMOUNT_TOLERANCE = 0.15  # 15% symmetric band around the inbound amount


def _rapid_cycle_count(g: pd.DataFrame, window_min: int) -> int:
    g = g.sort_values("timestamp")
    ins = g[g.txn_type == "TRANSFER_IN"]
    outs = g[g.txn_type == "TRANSFER_OUT"]
    count = 0
    for _, i in ins.iterrows():
        window_end = i.timestamp + pd.Timedelta(minutes=window_min)
        matches = outs[
            (outs.timestamp > i.timestamp) & (outs.timestamp <= window_end) &
            (outs.amount >= i.amount * (1 - RAPID_CYCLE_AMOUNT_TOLERANCE)) &
            (outs.amount <= i.amount * (1 + RAPID_CYCLE_AMOUNT_TOLERANCE))
        ]
        if len(matches) > 0:
            count += 1
    return count


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cfg = aml_config.get()
    ctr_threshold = cfg.ctr_threshold
    structuring_low = ctr_threshold * (STRUCTURING_LOW_DEFAULT / STRUCTURING_HIGH_DEFAULT)
    structuring_high = ctr_threshold
    rapid_cycle_window_min = cfg.rapid_cashout_mins

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    rows = []
    for cust_id, g in df.groupby("customer_id"):
        near_thresh = g[(g.amount >= structuring_low) & (g.amount < structuring_high)]
        daily_counts = g.groupby("date").size()
        night_txns = g[(g.hour >= 23) | (g.hour < 5)]

        rows.append({
            "customer_id": cust_id,
            "segment": g["segment"].iloc[0] if "segment" in g else None,
            "kyc_risk_rating": g["kyc_risk_rating"].iloc[0] if "kyc_risk_rating" in g else None,
            "txn_count": int(len(g)),
            "total_amount": round(float(g.amount.sum()), 2),
            "avg_amount": round(float(g.amount.mean()), 2),
            "amount_std": round(safe_float(g.amount.std()), 2),
            "near_threshold_count": int(len(near_thresh)),
            "near_threshold_pct": round(len(near_thresh) / len(g) * 100, 2),
            "max_daily_txn_count": int(daily_counts.max()) if len(daily_counts) else 0,
            "active_days": int(g["date"].nunique()),
            "rapid_cycle_count": _rapid_cycle_count(g, rapid_cycle_window_min),
            "distinct_counterparty_countries": int(g["counterparty_country"].nunique()),
            "high_risk_country_txn_pct": round(
                safe_float(g.get("is_high_risk_country", pd.Series([0])).mean()) * 100, 2
            ),
            "night_txn_pct": round(len(night_txns) / len(g) * 100, 2),
        })
    return pd.DataFrame(rows)
