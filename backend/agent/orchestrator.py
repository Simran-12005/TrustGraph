"""
Orchestrator
--------------
Ties intent parsing -> planning -> tool execution -> structured output.
This is the single entry point the FastAPI layer calls.
"""
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.agent.intent_parser import parse_intent
from backend.agent.planner import build_plan
from backend.db import db_utils
from backend.tools.eda_tool import run_eda
from backend.tools.feature_engineering_tool import build_features
from backend.tools.anomaly_detection_tool import run_anomaly_detection
from backend.tools.risk_classification_tool import classify
from backend.tools.explanation_tool import explain
from backend.tools.aggregation_tool import run_aggregation
from backend.tools.evidence_tool import gather_evidence

RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def run_agent(user_query: str, top_n: int = 15) -> dict:
    t0 = time.time()
    run_id = str(uuid.uuid4())[:8]

    # 1. INTENT
    intent = parse_intent(user_query)

    # 2. PLAN
    plan = build_plan(intent)
    filters = plan["filters"]

    result = {
        "run_id": run_id,
        "user_query": user_query,
        "parsed_intent": intent,
        "execution_plan": {"query_type": plan["query_type"], "steps": plan["steps"],
                            "filters_applied": filters, "reasoning": plan["reasoning"]},
        "eda": None,
        "flagged_items": [],
        "total_candidates_scanned": 0,
        "latency_ms": None,
    }

    # 3. LOAD DATA (always first)
    steps = plan["steps"]

    # aggregation_tool applies max_amount itself as the threshold rule over
    # each customer's FULL activity (to compute true txn_count/total_amount
    # alongside the under-threshold count) -- pre-filtering the load by
    # max_amount here would corrupt that. Only pass it through as a data
    # filter for non-aggregation query types.
    load_max_amount = filters.get("max_amount") if "aggregation_rule" not in steps else None

    df = db_utils.load_transactions(
        customer_id=filters.get("customer_id"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
        country=filters.get("country"),
        segment=filters.get("segment"),
        txn_type=filters.get("txn_type"),
        max_amount=load_max_amount,
    )
    result["total_candidates_scanned"] = int(len(df))

    if df.empty:
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        result["message"] = "No transactions matched the parsed filters."
        return result

    scored = None

    # 4. EDA (only if planned)
    if "eda" in steps:
        result["eda"] = run_eda(df)

    # 5. FEATURE ENGINEERING + DETECTION, or AGGREGATION shortcut
    if "aggregation_rule" in steps:
        scored = run_aggregation(df, intent.get("min_txn_count", 5), intent.get("max_amount", 10000))
    elif "feature_engineering" in steps:
        features = build_features(df)
        if "anomaly_detection" in steps and not features.empty:
            scored = run_anomaly_detection(features)
        else:
            scored = features
            if not scored.empty:
                scored["rule_score"] = 0.0
                scored["ml_score"] = 0.0
                scored["hybrid_score"] = 0.0
                scored["trigger_reasons"] = [[] for _ in range(len(scored))]
                scored["shap_top_features"] = [[] for _ in range(len(scored))]

    # 6. RISK CLASSIFICATION
    if scored is not None and not scored.empty and "risk_classification" in steps:
        scored = classify(scored)

    # 7. EXPLANATION
    if scored is not None and not scored.empty and "explanation" in steps:
        scored = explain(scored, intent)

    # 8. ASSEMBLE + PERSIST FLAGS
    #
    # "flagged_items" should mean genuinely suspicious entities, not just
    # "whatever ranked highest among the query pool" -- previously this
    # always returned up to top_n rows regardless of risk_level, so a query
    # over an entirely clean population still came back with LOW/MONITOR
    # items that read as suspicious activity in the UI. Now: prefer
    # MEDIUM/HIGH; if none qualify, fall back to the highest-scoring rows
    # but say so explicitly.
    flags = []
    no_material_risk = False
    if scored is not None and not scored.empty:
        scored = scored.sort_values(
            by="hybrid_score" if "hybrid_score" in scored.columns else "matching_txn_count",
            ascending=False,
        )
        if "risk_level" in scored.columns:
            material = scored[scored["risk_level"].isin(["HIGH", "MEDIUM"])]
            if material.empty:
                no_material_risk = True
                top = scored.head(min(top_n, 3))
            else:
                top = material.head(top_n)
        else:
            top = scored.head(top_n)
        for _, row in top.iterrows():
            evidence = gather_evidence(df, row["customer_id"])
            flags.append({
                "flag_id": str(uuid.uuid4())[:8],
                "entity_type": "CUSTOMER",
                "entity_id": row["customer_id"],
                "risk_score": round(float(row.get("hybrid_score", 0)), 4),
                "risk_level": row.get("risk_level", "LOW"),
                "pattern_type": row.get("pattern_type", intent.get("pattern_type", "UNKNOWN")),
                "explanation": row.get("explanation", "No explanation generated."),
                "recommended_action": row.get("recommended_action", "MONITOR"),
                "supporting_metrics": {
                    "txn_count": int(row.get("txn_count", 0)),
                    "total_amount": float(row.get("total_amount", 0)),
                    "near_threshold_count": int(row.get("near_threshold_count", 0)),
                    "rapid_cycle_count": int(row.get("rapid_cycle_count", 0)),
                    "max_daily_txn_count": int(row.get("max_daily_txn_count", 0)),
                },
                "shap_top_features": row.get("shap_top_features", []) or [],
                "evidence": evidence,
            })
        flags.sort(key=lambda f: RISK_ORDER.get(f["risk_level"], 3))

    result["flagged_items"] = flags
    if no_material_risk:
        result["message"] = (
            "No entities crossed the MEDIUM/HIGH risk threshold for this query. "
            "Showing the highest-scoring (still LOW-risk) candidates for reference only."
        )
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)

    # persist for audit trail
    try:
        db_utils.save_agent_run(
            run_id, datetime.now(timezone.utc).isoformat(), user_query, intent,
            result["execution_plan"], {"n_flags": len(flags), "top_risk": flags[0]["risk_level"] if flags else None},
        )
        if flags:
            db_utils.save_flags(run_id, flags)
    except Exception as e:
        print(f"[orchestrator] Failed to persist run (non-fatal): {e}")

    return result


if __name__ == "__main__":
    import json
    for q in [
        "Find structuring patterns in the last 30 days",
        "Which customers made 10 or more transactions under $10,000?",
        "Is customer ID 00042 suspicious?",
        "Analyse this dataset for suspicious activity",
    ]:
        print("=" * 80)
        print("QUERY:", q)
        out = run_agent(q, top_n=5)
        print(json.dumps({k: v for k, v in out.items() if k != "eda"}, indent=2, default=str))
