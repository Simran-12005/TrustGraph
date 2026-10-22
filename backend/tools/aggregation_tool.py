"""
Aggregation Tool
------------------
Handles pure threshold-rule queries like "which customers made 10+
transactions under $10,000?" directly via groupby/aggregation -- no ML
model needed. Returns a feature-table-shaped DataFrame so downstream
risk_classification / explanation tools can consume it uniformly.
"""
import pandas as pd

from backend.tools._numeric_utils import safe_float


def run_aggregation(df: pd.DataFrame, min_txn_count: int, max_amount: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    under_threshold = df[df.amount < max_amount]
    counts = under_threshold.groupby("customer_id").size().rename("matching_txn_count")
    qualifying = counts[counts >= min_txn_count]

    if qualifying.empty:
        return pd.DataFrame()

    rows = []
    for cust_id, cnt in qualifying.items():
        g = df[df.customer_id == cust_id]
        g_match = under_threshold[under_threshold.customer_id == cust_id]
        rows.append({
            "customer_id": cust_id,
            "segment": g["segment"].iloc[0] if "segment" in g else None,
            "kyc_risk_rating": g["kyc_risk_rating"].iloc[0] if "kyc_risk_rating" in g else None,
            "txn_count": int(len(g)),
            "matching_txn_count": int(cnt),
            "total_amount": round(float(g.amount.sum()), 2),
            "avg_amount": round(float(g.amount.mean()), 2),
            "amount_std": round(safe_float(g.amount.std()), 2),
            "near_threshold_count": int(cnt),  # reuse downstream field names
            "near_threshold_pct": round(cnt / len(g) * 100, 2),
            "max_daily_txn_count": int(g.groupby(pd.to_datetime(g.timestamp).dt.date).size().max()),
            "rapid_cycle_count": 0,
            "distinct_counterparty_countries": int(g["counterparty_country"].nunique()),
            "high_risk_country_txn_pct": round(safe_float(g.get("is_high_risk_country", pd.Series([0])).mean()) * 100, 2),
            "night_txn_pct": 0.0,
            "rule_score": min(1.0, 0.15 * cnt),
            "ml_score": 0.0,
            "hybrid_score": min(1.0, 0.15 * cnt),
            "trigger_reasons": [
                f"{int(cnt)} transactions under ${max_amount:,.0f} "
                f"(threshold rule: {min_txn_count}+ required)"
            ],
            "shap_top_features": [],
        })
    return pd.DataFrame(rows).sort_values("matching_txn_count", ascending=False).reset_index(drop=True)
