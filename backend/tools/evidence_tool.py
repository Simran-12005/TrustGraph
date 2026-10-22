"""
Evidence Tool
---------------
Feature engineering and anomaly detection operate at the CUSTOMER level
(a hackathon-spec requirement, since typologies like structuring only
make sense as a pattern across multiple transactions). But a reviewer
acting on a flag needs to see the actual transactions -- this tool
reconstructs the specific evidence behind each trigger so a flag is never
just a customer_id and a number.

Returns, per customer:
  near_threshold_transactions : the actual structuring-band transactions
  rapid_cycle_pairs           : the actual TRANSFER_IN/TRANSFER_OUT pairs
                                 that triggered a layering signal
  peak_day                    : the single-day burst behind a velocity flag
"""
import numpy as np
import pandas as pd

from backend.config import aml_config
from backend.db import db_utils

STRUCTURING_LOW_DEFAULT, STRUCTURING_HIGH_DEFAULT = 8500, 10000
RAPID_CYCLE_AMOUNT_TOLERANCE = 0.15  # 15% symmetric band, matches feature_engineering_tool.py

TXN_COLS = ["transaction_id", "timestamp", "amount", "txn_type", "channel", "counterparty_country"]


def _txn_records(df: pd.DataFrame) -> list:
    cols = [c for c in TXN_COLS if c in df.columns]
    out = df[cols].copy()
    out["timestamp"] = out["timestamp"].astype(str)
    return out.to_dict(orient="records")


def _find_rapid_cycle_pairs(g: pd.DataFrame, window_min: int) -> list:
    g = g.sort_values("timestamp")
    ins = g[g.txn_type == "TRANSFER_IN"]
    outs = g[g.txn_type == "TRANSFER_OUT"]
    pairs = []
    for _, i in ins.iterrows():
        window_end = i.timestamp + pd.Timedelta(minutes=window_min)
        matches = outs[
            (outs.timestamp > i.timestamp) & (outs.timestamp <= window_end) &
            (outs.amount >= i.amount * (1 - RAPID_CYCLE_AMOUNT_TOLERANCE)) &
            (outs.amount <= i.amount * (1 + RAPID_CYCLE_AMOUNT_TOLERANCE))
        ]
        for _, o in matches.iterrows():
            pairs.append({
                "in_transaction": _txn_records(pd.DataFrame([i]))[0],
                "out_transaction": _txn_records(pd.DataFrame([o]))[0],
                "minutes_between": round((o.timestamp - i.timestamp).total_seconds() / 60, 1),
            })
    return pairs


def gather_evidence(df: pd.DataFrame, customer_id: str, max_items: int = 12) -> dict:
    """df must be the raw (unaggregated) transaction slice for this customer's query scope."""
    g = df[df.customer_id == customer_id].copy()
    if g.empty:
        return {"near_threshold_transactions": [], "rapid_cycle_pairs": [], "peak_day": None}

    cfg = aml_config.get()
    ctr_threshold = cfg.ctr_threshold
    structuring_low = ctr_threshold * (STRUCTURING_LOW_DEFAULT / STRUCTURING_HIGH_DEFAULT)
    structuring_high = ctr_threshold

    g["timestamp"] = pd.to_datetime(g["timestamp"])
    g["date"] = g["timestamp"].dt.date

    near_thresh = g[(g.amount >= structuring_low) & (g.amount < structuring_high)].sort_values("timestamp")
    rapid_pairs = _find_rapid_cycle_pairs(g, cfg.rapid_cashout_mins)

    daily_counts = g.groupby("date").size()
    peak_day_evidence = None
    if len(daily_counts) and daily_counts.max() >= 5:
        peak_date = daily_counts.idxmax()
        peak_txns = g[g.date == peak_date].sort_values("timestamp")
        peak_day_evidence = {
            "date": str(peak_date),
            "transaction_count": int(len(peak_txns)),
            "transactions": _txn_records(peak_txns.head(max_items)),
        }

    return {
        "near_threshold_transactions": _txn_records(near_thresh.head(max_items)),
        "near_threshold_total_count": int(len(near_thresh)),
        "rapid_cycle_pairs": rapid_pairs[:max_items],
        "peak_day": peak_day_evidence,
    }


# ============================================================================
# TrustGraph 50% milestone additions (Section 4C / Section 5)
# ----------------------------------------------------------------------------
# gather_evidence() above answers "what did THIS customer do". The functions
# below answer a different question: "have we seen this PATTERN before, in
# cases we already know were fraud". This is new functionality, not a
# modification of the existing rule-evidence path -- both coexist.
# ============================================================================

# Same feature set anomaly_detection_tool scores on, so similarity is
# computed in the same space the model's own decision was made in.
from backend.tools.anomaly_detection_tool import ML_FEATURES  # noqa: E402

MAX_HISTORICAL_CASES = 300  # cap for the 50% prototype; avoids scanning the full 1500-customer base every query


def build_historical_case_library(exclude_customer_id: str = None) -> pd.DataFrame:
    """
    Builds a feature-vector library of customers with at least one
    ground-truth-confirmed suspicious transaction, for similarity search.

    Prototype-level (Section 5): pulls confirmed cases, re-runs the existing
    feature_engineering_tool over their full history. Not FAISS/embeddings --
    a lightweight, explicit approach that's easy to demo and reason about,
    per "use whichever approach is easiest and most compatible".
    """
    from backend.tools.feature_engineering_tool import build_features

    gt = db_utils.get_ground_truth()
    confirmed_txn_ids = gt.loc[gt.is_suspicious == 1, "transaction_id"]
    if confirmed_txn_ids.empty:
        return pd.DataFrame()

    all_txns = db_utils.load_transactions()
    confirmed_customers = all_txns.loc[
        all_txns.transaction_id.isin(confirmed_txn_ids), "customer_id"
    ].unique()
    if exclude_customer_id is not None:
        confirmed_customers = [c for c in confirmed_customers if c != exclude_customer_id]
    confirmed_customers = confirmed_customers[:MAX_HISTORICAL_CASES]

    history = all_txns[all_txns.customer_id.isin(confirmed_customers)]
    if history.empty:
        return pd.DataFrame()

    features = build_features(history)
    features["case_id"] = "F" + features["customer_id"].str.replace("CUST", "", regex=False)
    features["status"] = "CONFIRMED_FRAUD"
    return features


# Not every ML_FEATURES entry is equally informative about FRAUD LIKENESS.
# Volume features (txn_count, total_amount, avg_amount, amount_std) mostly
# reflect customer segment/size, not suspicious behavior, and were
# dominating the plain cosine similarity -- every customer looked ~0.9+
# similar to every fraud case regardless of actual risk. Down-weight those,
# up-weight the typology-specific signals the rule engine itself uses.
SIMILARITY_FEATURE_WEIGHTS = {
    "txn_count": 0.3,
    "total_amount": 0.3,
    "avg_amount": 0.3,
    "amount_std": 0.3,
    "near_threshold_pct": 1.5,
    "max_daily_txn_count": 1.2,
    "rapid_cycle_count": 1.5,
    "distinct_counterparty_countries": 0.7,
    "high_risk_country_txn_pct": 1.2,
    "night_txn_pct": 1.0,
}


def find_similar_historical_cases(
    current_features: pd.Series,
    historical_library: pd.DataFrame,
    raw_txn_df: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    """
    Cosine similarity between the current customer's feature vector and the
    historical fraud-case library, in ML_FEATURES space.

    Uses z-score standardization (against the historical library's own
    mean/std) rather than min-max, and applies SIMILARITY_FEATURE_WEIGHTS so
    generic volume features don't swamp the typology-specific ones. Without
    this, similarity scores clustered around 0.9+ for nearly every customer
    (fraud or not), which made the "historical evidence" signal
    non-discriminative -- fixed after noticing clean customers scoring as
    "similar" to confirmed fraud as actual repeat offenders did.

    Returns the top_k matches WITH the actual supporting evidence
    (gather_evidence per case) -- per Section 4C, similarity alone
    ("evidence_similarity = 0.91") is not an acceptable output on its own.
    """
    if historical_library is None or historical_library.empty:
        return {
            "current_transaction_id": None,
            "similar_cases": [],
            "evidence_similarity": 0.0,
            "support_level": "NONE",
        }

    weights = np.array([SIMILARITY_FEATURE_WEIGHTS.get(f, 1.0) for f in ML_FEATURES])

    hist_matrix = historical_library[ML_FEATURES].fillna(0)
    current_row = pd.DataFrame([current_features[ML_FEATURES]]).fillna(0)

    means = hist_matrix.mean()
    stds = hist_matrix.std().replace(0, 1)  # avoid div-by-zero on constant columns

    hist_z = ((hist_matrix - means) / stds).values * weights
    current_z = (((current_row - means) / stds).values * weights)[0]

    current_norm = np.linalg.norm(current_z)
    hist_norms = np.linalg.norm(hist_z, axis=1)
    denom = np.where((hist_norms * current_norm) == 0, 1, hist_norms * current_norm)
    similarities = (hist_z @ current_z) / denom
    # Cosine similarity on z-scored data can go negative (opposite-direction
    # profiles); clip to [0, 1] since a negative "similarity" to fraud isn't
    # a meaningful evidence signal for this use case.
    similarities = np.clip(similarities, 0, 1)

    top_idx = np.argsort(-similarities)[:top_k]

    similar_cases = []
    for i in top_idx:
        case_row = historical_library.iloc[i]
        sim = round(float(similarities[i]), 4)
        case_evidence = gather_evidence(raw_txn_df, case_row.customer_id, max_items=3)
        similar_cases.append({
            "case_id": case_row.case_id,
            "customer_id": case_row.customer_id,
            "similarity": sim,
            "evidence": {
                "txn_count": int(case_row.txn_count),
                "total_amount": float(case_row.total_amount),
                "near_threshold_count": int(case_row.near_threshold_count),
                "rapid_cycle_count": int(case_row.rapid_cycle_count),
                "night_txn_pct": float(case_row.night_txn_pct),
                "supporting_transactions": case_evidence,
                "status": "CONFIRMED_FRAUD",
            },
        })

    top_similarity = similar_cases[0]["similarity"] if similar_cases else 0.0
    # Thresholds re-tuned against the z-score+weighted similarity distribution:
    # clean customers cluster ~0.60-0.92 (mean ~0.79), confirmed fraud cases
    # cluster ~0.85-1.0 (mean ~0.95). 0.93/0.80 separates them without
    # mislabeling an ordinary customer as "HIGH support".
    if top_similarity >= 0.93:
        support_level = "HIGH"
    elif top_similarity >= 0.80:
        support_level = "MEDIUM"
    else:
        support_level = "LOW"

    return {
        "current_transaction_id": current_features.get("customer_id"),
        "similar_cases": similar_cases,
        "evidence_similarity": top_similarity,
        "support_level": support_level,
        "note": "Historical evidence supports the model's explanation; it does not by itself prove fraud.",
    }
