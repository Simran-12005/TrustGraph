-- Audit trail tables. Deliberately uses CREATE TABLE IF NOT EXISTS (never
-- DROP) so that reloading the transaction dataset -- which is a routine
-- operation -- can never wipe the record of past agent decisions. A real
-- bank compliance tool must be able to reconstruct any automated decision
-- for regulatory review months later, independent of dataset refreshes.

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id           TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    user_query        TEXT NOT NULL,
    parsed_intent      TEXT NOT NULL,     -- JSON
    execution_plan     TEXT NOT NULL,     -- JSON
    result_summary     TEXT NOT NULL      -- JSON
);

CREATE TABLE IF NOT EXISTS flags (
    flag_id          TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES agent_runs(run_id),
    entity_type        TEXT NOT NULL,     -- CUSTOMER | TRANSACTION
    entity_id          TEXT NOT NULL,
    risk_score         REAL NOT NULL,
    risk_level         TEXT NOT NULL,     -- LOW | MEDIUM | HIGH
    pattern_type       TEXT NOT NULL,
    explanation         TEXT NOT NULL,
    recommended_action  TEXT NOT NULL     -- MONITOR | REVIEW | REPORT
);
