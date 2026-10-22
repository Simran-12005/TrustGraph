"""
Anomaly Detection Tool
------------------------
Hybrid approach, deliberately not pure black-box ML:

  1. RULE SCORE   - deterministic, regulator-explainable triggers per
                     known typology (structuring, layering, velocity).
                     Weighted sum in [0, 1].
  2. ML SCORE     - unsupervised IsolationForest over the engineered
                     feature matrix, catching NOVEL patterns the rules
                     don't cover. Normalized to [0, 1] via percentile rank
                     so it's comparable across differently-sized query
                     result sets.
  3. HYBRID SCORE = 0.6 * rule_score + 0.4 * ml_score

Why hybrid: pure rules miss novel typologies; pure ML is a black box that
a bank's compliance/legal team cannot defend to a regulator. Blending both
gives coverage + defensibility, and the rule component alone is enough to
generate a human-readable explanation even when the ML model drives the
final score up.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.config import aml_config

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

RULE_WEIGHT = 0.6
ML_WEIGHT = 0.4

ML_FEATURES = [
    "txn_count", "total_amount", "avg_amount", "amount_std",
    "near_threshold_pct", "max_daily_txn_count", "rapid_cycle_count",
    "distinct_counterparty_countries", "high_risk_country_txn_pct", "night_txn_pct",
]


def _rule_score(row: pd.Series) -> tuple[float, list[str]]:
    score = 0.0
    triggers = []

    if row.near_threshold_count >= 3:
        contrib = min(0.4, 0.1 * row.near_threshold_count)
        score += contrib
        triggers.append(
            f"{int(row.near_threshold_count)} transactions between "
            f"${8500:,} and ${9999:,} (just under the $10,000 CTR threshold)"
        )
    if row.rapid_cycle_count >= 1:
        contrib = min(0.35, 0.15 * row.rapid_cycle_count)
        score += contrib
        triggers.append(
            f"{int(row.rapid_cycle_count)} rapid in-out fund cycle(s) within "
            f"{180} minutes, consistent with layering"
        )
    velocity_threshold = aml_config.get().velocity_threshold_count
    if row.max_daily_txn_count >= velocity_threshold:
        contrib = min(0.25, 0.02 * row.max_daily_txn_count)
        score += contrib
        triggers.append(
            f"peak of {int(row.max_daily_txn_count)} transactions in a single day, "
            f"an unusual velocity spike"
        )
    if row.high_risk_country_txn_pct >= 30:
        contrib = 0.1
        score += contrib
        triggers.append(
            f"{row.high_risk_country_txn_pct:.0f}% of counterparties in higher-risk jurisdictions"
        )
    if row.night_txn_pct >= 40:
        contrib = 0.05
        score += contrib
        triggers.append(f"{row.night_txn_pct:.0f}% of activity occurred during off-hours (23:00-05:00)")

    return min(score, 1.0), triggers


def _shap_top_features(iso: IsolationForest, X: np.ndarray, feature_names: list, top_k: int = 3) -> list:
    """
    Returns, per row, the top_k features driving the ML anomaly score, with sign
    flipped so positive = pushes toward MORE anomalous (empirically verified:
    shap's raw TreeExplainer output on IsolationForest correlates at -0.996
    with our -score_samples anomaly convention).
    """
    if not _SHAP_AVAILABLE or len(X) < 8:
        return [[] for _ in range(len(X))]
    try:
        explainer = shap.TreeExplainer(iso)
        sv = explainer.shap_values(X)
        contributions = -sv  # flip sign: positive now means "toward anomalous"
        results = []
        for row in contributions:
            idx = np.argsort(-np.abs(row))[:top_k]
            results.append([
                {"feature": feature_names[i], "contribution": round(float(row[i]), 4)}
                for i in idx if abs(row[i]) > 1e-6
            ])
        return results
    except Exception as e:
        print(f"[anomaly_detection_tool] SHAP explanation failed, skipping: {e}")
        return [[] for _ in range(len(X))]

def run_anomaly_detection(features_df: pd.DataFrame, contamination: float = 0.06) -> pd.DataFrame:
    """
    Takes the per-customer feature table (from feature_engineering_tool)
    and returns it enriched with rule_score, ml_score, hybrid_score, and
    trigger_reasons.
    """
    if features_df.empty:
        return features_df

    df = features_df.copy()

    rule_results = df.apply(_rule_score, axis=1)
    df["rule_score"] = rule_results.apply(lambda x: x[0])
    df["trigger_reasons"] = rule_results.apply(lambda x: x[1])

    X = df[ML_FEATURES].fillna(0).values
    if len(df) >= 8:  # IsolationForest needs a reasonable sample to be meaningful
        iso = IsolationForest(
            n_estimators=200, contamination=min(contamination, 0.3), random_state=42
        )
        iso.fit(X)
        raw_scores = -iso.score_samples(X)  # higher = more anomalous
        ranks = pd.Series(raw_scores).rank(pct=True)  # normalize to [0,1]
        df["ml_score"] = ranks.values
        df["shap_top_features"] = _shap_top_features(iso, X, ML_FEATURES)
    else:
        df["ml_score"] = 0.0
        df["shap_top_features"] = [[] for _ in range(len(df))]

    df["hybrid_score"] = round(RULE_WEIGHT * df["rule_score"] + ML_WEIGHT * df["ml_score"], 4)

    # TrustGraph 50% milestone: model_confidence is the trust_score_tool-facing
    # name for the same hybrid_score. Kept as a plain alias column (not a
    # rename) so nothing that reads hybrid_score elsewhere in the pipeline
    # breaks -- see Section 4A / Section 13 rule 2 ("preserve existing APIs").
    df["model_confidence"] = df["hybrid_score"]

    return df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
