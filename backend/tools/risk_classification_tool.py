"""
Risk Classification Tool
--------------------------
Converts hybrid anomaly scores into business-facing risk tiers and a
recommended escalation action. Thresholds are context-adjustable: a
customer already flagged HIGH by KYC starts from a stricter bar, matching
real compliance practice (risk-based approach under FATF Recommendation 1).
"""
import pandas as pd

from backend.config import aml_config

# Fallback defaults only -- live values come from aml_config.get() at call
# time in _classify_row() so POST /api/v1/rules/config actually changes
# classification instead of being cosmetic.
BASE_THRESHOLDS_DEFAULT = {"MEDIUM": 0.35, "HIGH": 0.65}
KYC_HIGH_RISK_ADJUSTMENT = -0.10  # HIGH KYC customers get flagged at a lower score


def _classify_row(hybrid_score: float, kyc_risk_rating: str) -> tuple[str, float]:
    cfg = aml_config.get()
    base_high = cfg.high_risk_threshold
    base_medium = cfg.medium_risk_threshold
    adj = KYC_HIGH_RISK_ADJUSTMENT if kyc_risk_rating == "HIGH" else 0.0
    effective_high = max(0.3, base_high + adj)
    effective_medium = max(0.15, base_medium + adj)

    if hybrid_score >= effective_high:
        return "HIGH", effective_high
    elif hybrid_score >= effective_medium:
        return "MEDIUM", effective_medium
    return "LOW", effective_medium


def _recommend_action(risk_level: str) -> str:
    return {"HIGH": "REPORT", "MEDIUM": "REVIEW", "LOW": "MONITOR"}[risk_level]


def classify(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    df = scored_df.copy()
    classified = df.apply(
        lambda r: _classify_row(r["hybrid_score"], r.get("kyc_risk_rating", "LOW")), axis=1
    )
    df["risk_level"] = classified.apply(lambda x: x[0])
    df["threshold_applied"] = classified.apply(lambda x: x[1])
    df["recommended_action"] = df["risk_level"].apply(_recommend_action)
    return df
