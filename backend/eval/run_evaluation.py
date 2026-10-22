"""
Offline Evaluation
--------------------
Runs the full (non-agentic) detection pipeline -- feature engineering ->
hybrid anomaly detection -> risk classification -- over the ENTIRE
dataset, then compares HIGH/MEDIUM-flagged customers against the
injected ground-truth labels to report precision/recall/F1.

This is a pipeline-level sanity check (not the agent's dynamic-plan path,
which operates on query-filtered slices). It's what should be quoted in
the pitch deck / README as the model's offline accuracy.

Run: python -m backend.eval.run_evaluation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from backend.db import db_utils
from backend.tools.feature_engineering_tool import build_features
from backend.tools.anomaly_detection_tool import run_anomaly_detection
from backend.tools.risk_classification_tool import classify


def evaluate():
    txns = db_utils.load_transactions()
    gt = db_utils.get_ground_truth()

    features = build_features(txns)
    scored = run_anomaly_detection(features)
    classified = classify(scored)

    # ground truth at customer level: a customer is a true positive if ANY
    # of their transactions were labeled suspicious
    gt_by_customer = (
        txns[["transaction_id", "customer_id"]]
        .merge(gt, on="transaction_id")
        .groupby("customer_id")["is_suspicious"].max()
    )

    classified = classified.set_index("customer_id")
    classified["actual_suspicious"] = gt_by_customer.reindex(classified.index).fillna(0).astype(int)
    classified["predicted_flag"] = classified["risk_level"].isin(["HIGH", "MEDIUM"]).astype(int)
    classified["predicted_high_only"] = (classified["risk_level"] == "HIGH").astype(int)

    def _prf(pred_col: str) -> dict:
        tp = int(((classified[pred_col] == 1) & (classified.actual_suspicious == 1)).sum())
        fp = int(((classified[pred_col] == 1) & (classified.actual_suspicious == 0)).sum())
        fn = int(((classified[pred_col] == 0) & (classified.actual_suspicious == 1)).sum())
        tn = int(((classified[pred_col] == 0) & (classified.actual_suspicious == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    high_only = _prf("predicted_high_only")
    high_plus_medium = _prf("predicted_flag")

    print("=" * 60)
    print("OFFLINE EVALUATION (customer-level)")
    print("=" * 60)
    print(f"Customers scored:       {len(classified)}")
    print(f"Actual suspicious:      {int(classified.actual_suspicious.sum())}")
    print()
    print("-- HIGH only (-> REPORT tier) --")
    print(high_only)
    print()
    print("-- HIGH + MEDIUM (-> REPORT + REVIEW queue) --")
    print(high_plus_medium)
    print()
    print("Risk level breakdown:")
    print(classified.risk_level.value_counts())
    print()
    print("Suspicious customers missed entirely (false negatives at HIGH+MEDIUM), sample:")
    fn_rows = classified[(classified.predicted_flag == 0) & (classified.actual_suspicious == 1)]
    print(fn_rows[["txn_count", "hybrid_score", "risk_level"]].head(10))

    return {
        "customers_scored": int(len(classified)),
        "actual_suspicious": int(classified.actual_suspicious.sum()),
        "risk_level_breakdown": classified.risk_level.value_counts().to_dict(),
        "high_only_report_tier": high_only,
        "high_plus_medium_review_queue": high_plus_medium,
    }


if __name__ == "__main__":
    evaluate()
