"""
Explanation Tool
------------------
Turns rule trigger_reasons + risk tier + the original query intent into a
concise natural-language explanation an investigator can drop straight
into a SAR (Suspicious Activity Report) narrative. Deliberately template-
driven (not LLM-generated at this stage) so every explanation is
deterministic, auditable, and traceable to a specific numeric trigger --
critical for regulatory defensibility.
"""
import pandas as pd

PATTERN_LABELS = {
    "STRUCTURING": "structuring / smurfing",
    "LAYERING": "layering",
    "VELOCITY_SPIKE": "velocity spike",
    "MULTIPLE": "multiple typologies",
}


def _dominant_pattern(row: pd.Series) -> str:
    patterns = []
    if row.near_threshold_count >= 3:
        patterns.append("STRUCTURING")
    if row.rapid_cycle_count >= 1:
        patterns.append("LAYERING")
    if row.max_daily_txn_count >= 10:
        patterns.append("VELOCITY_SPIKE")
    if len(patterns) > 1:
        return "MULTIPLE"
    return patterns[0] if patterns else ("ANOMALOUS_PROFILE" if row.ml_score > 0.7 else "NONE")


def _explanation_features(row: pd.Series, max_items: int = 5) -> list:
    """
    TrustGraph 50% milestone: structured, machine-readable version of the
    same signal `_build()` turns into prose below -- rule triggers and/or
    SHAP top features -- so evidence_tool / trust_score_tool can consume it
    without parsing sentences. Impact is a rough [0,1] "how much did this
    matter" score, not a calibrated probability.
    """
    feats = []
    shap_feats = row.get("shap_top_features", []) or []
    for f in shap_feats:
        feats.append({
            "feature": f["feature"],
            "impact": round(min(abs(f["contribution"]) * 2, 1.0), 3),
        })
    if row.trigger_reasons:
        # Rule triggers don't carry a numeric per-feature weight the way SHAP
        # does, so give each a flat impact ordered by the sequence they fired
        # in (first trigger = strongest, matching _rule_score's contribution
        # ordering).
        n = len(row.trigger_reasons)
        for i, reason in enumerate(row.trigger_reasons):
            feats.append({
                "feature": reason,
                "impact": round(max(1.0 - i / max(n, 1), 0.2), 3),
            })
    feats.sort(key=lambda f: -f["impact"])
    return feats[:max_items]


def explain(classified_df: pd.DataFrame, query_intent: dict) -> pd.DataFrame:
    if classified_df.empty:
        return classified_df

    df = classified_df.copy()
    df["pattern_type"] = df.apply(_dominant_pattern, axis=1)
    df["explanation_features"] = df.apply(_explanation_features, axis=1)

    def _build(row):
        pattern_label = PATTERN_LABELS.get(row.pattern_type, "an anomalous transaction profile")
        if row.trigger_reasons:
            reasons_text = "; ".join(row.trigger_reasons)
        elif row.ml_score > 0.7:
            shap_feats = row.get("shap_top_features", []) or []
            if shap_feats:
                feat_text = "; ".join(
                    f"{f['feature'].replace('_', ' ')} "
                    f"({'+' if f['contribution'] > 0 else ''}{f['contribution']:.3f} impact)"
                    for f in shap_feats
                )
                reasons_text = (
                    f"no rule-based trigger matched, but the ML model (IsolationForest, "
                    f"top {100 - row.ml_score * 100:.0f}th percentile anomaly rank) flagged this "
                    f"profile primarily due to: {feat_text}"
                )
            else:
                reasons_text = (
                    f"unsupervised model ranked this customer in the top "
                    f"{100 - row.ml_score * 100:.0f}th percentile of anomalous behaviour "
                    f"across the queried population, without matching a known rule pattern"
                )
        else:
            reasons_text = "no strong individual triggers; flagged only by baseline scoring"

        target = query_intent.get("pattern_type") if isinstance(query_intent, dict) else None
        query_note = (
            f" (query specifically targeted {target} patterns)"
            if target and target.upper() != "ANY" else ""
        )

        return (
            f"Customer {row.customer_id} flagged {row.risk_level} risk for {pattern_label}{query_note}. "
            f"Hybrid anomaly score: {row.hybrid_score:.2f} (rule-based: {row.rule_score:.2f}, "
            f"ML/IsolationForest: {row.ml_score:.2f}). Key triggers: {reasons_text}."
        )

    df["explanation"] = df.apply(_build, axis=1)
    return df
