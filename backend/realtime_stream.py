"""
Real-Time Stream Simulation & Ingestion Engine.
"""

import datetime
import random
import sqlite3
import pandas as pd
from backend.dataset_manager import dataset_manager
from backend.db import db_utils

REQUIRED_TXN_COLS = ["transaction_id", "customer_id", "timestamp", "amount",
                     "txn_type", "channel", "counterparty_country"]


def process_single_transaction(tx_dict: dict) -> dict:
    """
    Ingests 1 new transaction: writes it into the real aml.db `transactions`
    table (the one backend.db.db_utils / the agent actually queries) so a
    simulated live transaction is genuinely visible to /agent/query, not just
    an in-memory side channel that nothing downstream ever reads. Also
    refreshes the in-memory dataset_manager view for the graph/EDA widgets.
    """
    row = {k: tx_dict.get(k) for k in REQUIRED_TXN_COLS}

    conn = sqlite3.connect(db_utils.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO transactions (transaction_id, customer_id, timestamp, amount, "
            "txn_type, channel, counterparty_country) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row["transaction_id"], row["customer_id"], row["timestamp"], row["amount"],
             row["txn_type"], row["channel"], row["counterparty_country"]),
        )
        conn.commit()
    finally:
        conn.close()

    # Invalidate the in-memory dataset_manager cache so the next graph/EDA
    # read picks up the newly-ingested row instead of a stale snapshot.
    dataset_manager.active_df = None
    updated_df = dataset_manager.get_active_data()
    dataset_manager.metadata["total_rows"] = len(updated_df)

    return {
        "status": "success",
        "ingested_transaction": tx_dict,
        "total_transactions_now": len(updated_df),
    }


def generate_random_transaction() -> dict:
    """
    Generates a synthetic live transaction payload for an EXISTING customer,
    so it can actually be picked up and scored by the real agent pipeline
    (previously the random customer_id format, e.g. "CUST_00500", never
    matched the real "CUST00500"-style IDs, so simulated transactions could
    never join to a real customer record).
    """
    conn = sqlite3.connect(db_utils.DB_PATH)
    try:
        row = conn.execute(
            "SELECT customer_id FROM customers ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    customer_id = row[0] if row else "CUST00001"

    return {
        "transaction_id": f"TXNLIVE{random.randint(100000, 999999)}",
        "customer_id": customer_id,
        "amount": round(random.uniform(500.0, 45000.0), 2),
        "timestamp": datetime.datetime.now().isoformat(),
        "txn_type": random.choice(["DEPOSIT", "WITHDRAWAL", "TRANSFER_IN", "TRANSFER_OUT"]),
        "channel": random.choice(["ACH", "WIRE", "CASH_DEPOSIT", "CARD"]),
        "counterparty_country": random.choice(["US", "FR", "GB", "AE", "SG", "NG", "DE"]),
    }