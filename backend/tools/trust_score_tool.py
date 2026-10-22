"""
TrustGraph Score Tool
------------------------
New component (does not exist in the original SocGen AML project).

Combines three independent signals into one Preliminary Trust Score:
  - model_confidence     : does the model think this is suspicious? (anomaly_detection_tool)
  - evidence_similarity   : do we have similar CONFIRMED fraud cases? (evidence_tool)
  - graph_risk             : is this customer linked to suspicious others via
                              shared device/IP? (graph_engine)

Weights (0.40 / 0.35 / 0.25) are a reasonable prototype starting point for
the 50% milestone, NOT scientifically calibrated -- see Section 7. They
should be revisited in Phase 2 (e.g. via labelled backtesting), not treated
as final.

This module deliberately sits ALONGSIDE the existing agent pipeline
(backend/agent/orchestrator.py) rather than being wired into it, per
Section 10 of the TrustGraph spec ("we will integrate the agent and planner
into TrustGraph after the 50% presentation"). run_trustgraph_analysis()
below is a separate, self-contained entry point.
"""
import pandas as pd

from backend.db import db_utils
from backend.tools.feature_engineering_tool import build_features
from backend.tools.anomaly_detection_tool import run_anomaly_detection
from backend.tools.risk_classification_tool import classify
from backend.tools.explanation_tool import explain
from backend.tools.evidence_tool import build_historical_case_library, find_similar_historical_cases
from backend.graph_engine import compute_relationship_risk

MODEL_CONFIDENCE_WEIGHT = 0.40
EVIDENCE_SIMILARITY_WEIGHT = 0.35
GRAPH_RISK_WEIGHT = 0.25


def compute_trust_score(model_confidence, evidence_similarity, graph_risk):
    """
    Computes a composite Trust Score from three normalized sub-signals (0.0 to 1.0).
    Returns composite score, percentage string, and risk tier badge.
    """
    # Normalize inputs if passed as percentages (>1.0)
    m = model_confidence / 100.0 if model_confidence > 1.0 else model_confidence
    e = evidence_similarity / 100.0 if evidence_similarity > 1.0 else evidence_similarity
    g = graph_risk / 100.0 if graph_risk > 1.0 else graph_risk

    # Formula synthesis: 40% Model + 35% Evidence + 25% Graph
    composite = (0.40 * m) + (0.35 * e) + (0.25 * g)
    score_pct = int(round(composite * 100))

    if score_pct >= 60:
        level = "HIGH"
        label = "HIGH RISK LEVEL"
    elif score_pct >= 25:
        level = "MEDIUM"
        label = "MEDIUM RISK LEVEL"
    else:
        level = "LOW"
        label = "LOW RISK LEVEL"

    return {
        "trust_score": composite,
        "trust_score_pct": f"{score_pct}%",
        "risk_level": level,
        "risk_level_label": label,
        "model_confidence": m,
        "evidence_similarity": e,
        "graph_risk": g
    }


def run_trustgraph_analysis(customer_id: str) -> dict:
    """
    End-to-end TrustGraph pipeline for one customer -- Section 12 acceptance
    criteria:
      ML detection -> SHAP explanation -> historical evidence search ->
      graph analysis -> preliminary trust score.

    Kept customer-level (not per-transaction) to match the existing
    codebase's unit of analysis (see anomaly_detection_tool /
    feature_engineering_tool, both of which score per-customer).
    """
    all_txns = db_utils.load_transactions()
    if all_txns.empty or customer_id not in set(all_txns.customer_id):
        return {"error": f"No transactions found for customer_id={customer_id}"}

    customer_txns = all_txns[all_txns.customer_id == customer_id]

    # 1-2. ML detection + SHAP
    features = build_features(customer_txns)
    scored = run_anomaly_detection(features)
    if scored.empty:
        return {"error": "Feature/anomaly scoring produced no rows."}
    scored = classify(scored)
    scored = explain(scored, query_intent={})
    row = scored.iloc[0]

    # 3. Historical evidence search
    history_lib = build_historical_case_library(exclude_customer_id=customer_id)
    evidence_result = find_similar_historical_cases(row, history_lib, all_txns, top_k=3)

    # 4. Graph / relationship evidence
    gt = db_utils.get_ground_truth()
    confirmed_txn_ids = set(gt.loc[gt.is_suspicious == 1, "transaction_id"])
    suspicious_customer_ids = set(
        all_txns.loc[all_txns.transaction_id.isin(confirmed_txn_ids), "customer_id"].unique()
    )
    graph_result = compute_relationship_risk(customer_id, all_txns, suspicious_customer_ids)

    # 5. Preliminary Trust Score
    trust_result = compute_trust_score(
        model_confidence=row.get("model_confidence", row.get("hybrid_score", 0.0)),
        evidence_similarity=evidence_result["evidence_similarity"],
        graph_risk=graph_result["graph_risk"],
    )

    return {
        "customer_id": customer_id,
        "prediction": "FRAUD" if row.get("risk_level") in ("HIGH", "MEDIUM") else "NOT_FLAGGED",
        "risk_level": row.get("risk_level"),
        "model_confidence": trust_result["model_confidence"],
        "explanation": row.get("explanation"),
        "explanation_features": row.get("explanation_features", []),
        "historical_evidence": evidence_result,
        "graph_evidence": graph_result,
        "trust_score": trust_result,
    }


if __name__ == "__main__":
    import json
    all_txns = db_utils.load_transactions()
    sample_customer = all_txns.customer_id.iloc[0]
    print(json.dumps(run_trustgraph_analysis(sample_customer), indent=2, default=str))
