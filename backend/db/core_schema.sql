-- AML Investigation Platform -- core data schema (ETL-managed).
-- These tables are rebuilt from CSV on every init_db() call, so they are
-- safe to DROP + CREATE. Audit tables (agent_runs, flags) are handled
-- separately in audit_schema.sql and are NEVER dropped here -- a dataset
-- reload must not destroy the compliance audit trail.

DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id         TEXT PRIMARY KEY,
    segment              TEXT NOT NULL,
    country              TEXT NOT NULL,
    is_high_risk_country  INTEGER NOT NULL,
    onboarding_date      TEXT NOT NULL,
    kyc_risk_rating       TEXT NOT NULL
);

DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    transaction_id        TEXT PRIMARY KEY,
    customer_id           TEXT NOT NULL REFERENCES customers(customer_id),
    timestamp              TEXT NOT NULL,
    amount                 REAL NOT NULL,
    txn_type               TEXT NOT NULL,
    channel                 TEXT NOT NULL,
    counterparty_country   TEXT NOT NULL,
    device_id               TEXT,
    ip_address               TEXT
);
CREATE INDEX idx_txn_customer ON transactions(customer_id);
CREATE INDEX idx_txn_timestamp ON transactions(timestamp);
CREATE INDEX idx_txn_amount ON transactions(amount);

DROP TABLE IF EXISTS ground_truth_eval;
CREATE TABLE ground_truth_eval (
    transaction_id  TEXT PRIMARY KEY REFERENCES transactions(transaction_id),
    is_suspicious    INTEGER NOT NULL,
    pattern_type     TEXT NOT NULL
);
