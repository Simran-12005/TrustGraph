"""
ETL + data-access layer.

Loads customers.csv / transactions.csv into SQLite following schema.sql,
and exposes typed helper functions the agent's tools use to pull filtered
slices of data (so tools never write raw SQL inline -- keeps the SQL
injection surface at zero given the agent may construct filter values
from natural language).
"""
import sqlite3
import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = os.environ.get("AML_DB_PATH", str(Path(__file__).resolve().parents[2] / "aml.db"))
CORE_SCHEMA_PATH = Path(__file__).resolve().parent / "core_schema.sql"
AUDIT_SCHEMA_PATH = Path(__file__).resolve().parent / "audit_schema.sql"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(customers_csv: Optional[str] = None, transactions_csv: Optional[str] = None) -> None:
    """(Re)build the SQLite database from CSV source files. Idempotent.

    Only the ETL-managed tables (customers/transactions/ground_truth_eval)
    are dropped and recreated. The audit tables (agent_runs/flags) are
    created only if missing and are NEVER dropped here -- reloading a
    dataset must not erase the record of past agent decisions.
    """
    customers_csv = customers_csv or str(DATA_DIR / "customers.csv")
    transactions_csv = transactions_csv or str(DATA_DIR / "transactions.csv")

    conn = get_conn()
    with open(CORE_SCHEMA_PATH) as f:
        conn.executescript(f.read())
    with open(AUDIT_SCHEMA_PATH) as f:
        conn.executescript(f.read())

    customers = pd.read_csv(customers_csv)
    txns = pd.read_csv(transactions_csv)

    ground_truth = txns[["transaction_id", "is_suspicious", "pattern_type"]].copy()
    ground_truth["is_suspicious"] = ground_truth["is_suspicious"].astype(int)
    txns_clean = txns.drop(columns=["is_suspicious", "pattern_type"])

    customers.to_sql("customers", conn, if_exists="append", index=False)
    txns_clean.to_sql("transactions", conn, if_exists="append", index=False)
    ground_truth.to_sql("ground_truth_eval", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"Loaded {len(customers)} customers, {len(txns_clean)} transactions -> {DB_PATH}")


def load_transactions(
    customer_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    country: Optional[str] = None,
    segment: Optional[str] = None,
    txn_type: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Parameterized filtered pull of transactions joined with customer segment info."""
    query = """
        SELECT t.*, c.segment, c.country AS customer_country,
               c.is_high_risk_country, c.kyc_risk_rating
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE 1=1
    """
    params: list = []
    if customer_id:
        query += " AND t.customer_id = ?"
        params.append(customer_id)
    if start_date:
        query += " AND t.timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND t.timestamp <= ?"
        params.append(end_date)
    if country:
        query += " AND c.country = ?"
        params.append(country)
    if segment:
        query += " AND c.segment = ?"
        params.append(segment)
    if txn_type:
        query += " AND t.txn_type = ?"
        params.append(txn_type)
    if min_amount is not None:
        query += " AND t.amount >= ?"
        params.append(min_amount)
    if max_amount is not None:
        query += " AND t.amount <= ?"
        params.append(max_amount)
    query += " ORDER BY t.timestamp"
    if limit:
        query += f" LIMIT {int(limit)}"

    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_ground_truth(transaction_ids: Optional[list] = None) -> pd.DataFrame:
    """Used only by the offline evaluation script -- never by the live agent."""
    conn = get_conn()
    if transaction_ids:
        placeholders = ",".join("?" * len(transaction_ids))
        df = pd.read_sql_query(
            f"SELECT * FROM ground_truth_eval WHERE transaction_id IN ({placeholders})",
            conn, params=transaction_ids,
        )
    else:
        df = pd.read_sql_query("SELECT * FROM ground_truth_eval", conn)
    conn.close()
    return df


def save_agent_run(run_id: str, created_at: str, user_query: str, parsed_intent: dict,
                    execution_plan: dict, result_summary: dict) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO agent_runs (run_id, created_at, user_query, parsed_intent, execution_plan, result_summary) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, created_at, user_query, json.dumps(parsed_intent),
         json.dumps(execution_plan), json.dumps(result_summary)),
    )
    conn.commit()
    conn.close()


def save_flags(run_id: str, flags: list) -> None:
    conn = get_conn()
    for f in flags:
        conn.execute(
            "INSERT INTO flags (flag_id, run_id, entity_type, entity_id, risk_score, risk_level, "
            "pattern_type, explanation, recommended_action) VALUES (?,?,?,?,?,?,?,?,?)",
            (f["flag_id"], run_id, f["entity_type"], f["entity_id"], f["risk_score"],
             f["risk_level"], f["pattern_type"], f["explanation"], f["recommended_action"]),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
