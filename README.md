# AML Investigation Agent


An agent-driven AML investigation platform that parses a natural-language investigator
query, dynamically decides which analytical tools it needs (not a fixed pipeline),
runs hybrid rule-based + ML anomaly detection over transaction data, and returns
risk-scored, explainable, audit-logged results.

---

## 1. Why this architecture

Traditional AML tooling has two failure modes: **rule engines** that are explainable
but drown compliance teams in false positives, and **black-box ML** that catches novel
patterns but can't be defended to a regulator. This project doesn't pick a side — it
architects around the tension:

- **Hybrid detection** (`anomaly_detection_tool.py`): `hybrid_score = 0.6·rule_score + 0.4·ml_score`.
  Rules give deterministic, regulator-explainable triggers (structuring, layering,
  velocity). An `IsolationForest` catches anomalies the rules don't anticipate. The
  rule component alone is always enough to generate a defensible explanation, even
  when the ML component is what pushed a score over threshold. **For the cases where
  ML alone drives the flag** (no rule matched), `shap_top_features` uses
  `shap.TreeExplainer` on the IsolationForest to surface the top 3 features actually
  driving that specific score — so no flag is ever backed by "the model said so" alone.
- **Transaction-level evidence, not just aggregates** (`evidence_tool.py`): feature
  engineering and scoring operate at the customer level (structuring is a pattern
  *across* transactions), but every flag also carries the actual supporting
  transaction records — the specific near-threshold deposits, the specific
  TRANSFER_IN/TRANSFER_OUT pair behind a layering flag, the specific burst-day
  transaction list. A reviewer can act on a flag without re-querying the database.
- **Agentic, not pipeline** (`planner.py`): the agent parses intent and skips stages
  that don't apply to the query. A single-customer lookup never runs full-dataset EDA.
  A pure threshold question ("10+ transactions under $10k") never touches the ML model.
  This is the difference between an agent and a script — see §3 for the mapping to the
  three worked examples in the hackathon spec.
- **Audit trail by default** (`schema.sql`, `agent_runs` / `flags` tables): every agent
  run — query, parsed intent, execution plan, and resulting flags — is persisted. In a
  real bank, an automated compliance decision that can't be reconstructed six months
  later is not shippable.
- **Graceful degradation** (`intent_parser.py`): if `ANTHROPIC_API_KEY` isn't set or the
  API call fails, the agent falls back to a deterministic regex parser rather than
  crashing. A production banking tool should never go down because an LLM call timed out.

---

## 2. Architecture diagram

```
                     ┌─────────────────────────┐
   Investigator  ──▶ │   FastAPI  /agent/query  │
   query             └────────────┬─────────────┘
                                   ▼
                      ┌────────────────────────┐
                      │   Intent Parser         │  Claude tool-use (haiku) → structured
                      │   (LLM + regex fallback)│  intent: query_type, filters, pattern
                      └────────────┬────────────┘
                                   ▼
                      ┌────────────────────────┐
                      │   Planner                │  Decides WHICH tools, WHAT ORDER,
                      │   (dynamic execution plan)│  WHAT DATA SUBSET — not fixed
                      └────────────┬────────────┘
                                   ▼
        ┌───────────────┬─────────────────┬───────────────────┬───────────────────┐
        ▼                ▼                 ▼                    ▼
  ┌───────────┐  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐
  │ EDA Tool   │  │ Feature Eng.   │  │ Anomaly Detection│  │ Aggregation Tool  │
  │ (optional) │  │ Tool           │  │ Tool (hybrid)     │  │ (threshold path)  │
  └───────────┘  └───────────────┘  └─────────────────┘  └──────────────────┘
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │  Risk Classification    │  score → LOW / MEDIUM / HIGH
                      │  Tool                    │  + KYC-aware threshold adjustment
                      └────────────┬────────────┘
                                   ▼
                      ┌────────────────────────┐
                      │  Explanation Tool        │  template-driven, tied to query
                      └────────────┬────────────┘  intent + numeric triggers
                                   ▼
                      ┌────────────────────────┐
                      │  SQLite audit log        │  agent_runs, flags
                      └────────────┬────────────┘
                                   ▼
                        Structured JSON response ──▶ Streamlit demo UI
```

---

## 3. Adaptive behaviour — mapped to the spec's worked examples

| User Query | Detected `query_type` | Tools invoked | Tools skipped |
|---|---|---|---|
| "Find structuring patterns in the last 30 days" | `TARGETED_PATTERN` | feature_engineering → anomaly_detection → risk_classification → explanation | EDA |
| "Which customers made 10+ transactions under $10,000?" | `AGGREGATION_THRESHOLD` | aggregation_rule → risk_classification → explanation | EDA, ML anomaly detection |
| "Is customer ID 4521 suspicious?" | `SINGLE_ENTITY` | feature_engineering (scoped to 1 customer) → anomaly_detection → risk_classification → explanation | EDA |
| "Analyse this dataset for suspicious activity" | `BROAD_EDA` | eda → feature_engineering → anomaly_detection → risk_classification → explanation | (none) |

This logic lives entirely in `backend/agent/planner.py` — a judge can open that one
file and see the plan branch on `query_type`, with the reasoning string returned to
the caller for full transparency. Verified by `tests/test_agent.py`.

---

## 4. Repo structure
````text
aml-agent/
...
aml-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                # Automated Pytest workflow on push/PR
├── backend/
│   ├── agent/                    # Agentic Core (LLM + Dynamic Planner)
│   │   ├── __init__.py
│   │   ├── intent_parser.py      # NL -> Structured Intent (OpenAI + Regex Fallback)
│   │   ├── planner.py            # Dynamic Execution Plan Builder
│   │   └── orchestrator.py       # Plan Execution, Tool Routing & Response Assembly
│   ├── tools/                    # Modular Pipeline Execution Tools
│   │   ├── __init__.py
│   │   ├── eda_tool.py           # Exploratory Data Analysis tool
│   │   ├── feature_engineering_tool.py # Behavioral & Threshold metrics generator
│   │   ├── anomaly_detection_tool.py    # Hybrid rules + IsolationForest + SHAP
│   │   ├── risk_classification_tool.py  # Risk scoring (LOW/MEDIUM/HIGH) + Escalation
│   │   ├── explanation_tool.py   # Query-tied natural language explanations
│   │   ├── aggregation_tool.py   # Non-ML threshold-rule path
│   │   └── evidence_tool.py      # Transaction-level evidence per flagged item
│   ├── db/                       # Persistence & Database Access Layer
│   │   ├── __init__.py
│   │   ├── schema.sql            # SQLite/PostgreSQL schema for audit trails & cases
│   │   └── db_utils.py           # Dataset ETL & parameterized query helpers
│   ├── eval/                     # Model & System Offline Evaluation
│   │   ├── __init__.py
│   │   └── run_evaluation.py     # Precision/Recall/F1 vs. injected ground truth
│   ├── dataset_manager.py        # Active dataset context manager (In-Memory + File)
│   ├── graph_engine.py           # NetworkX + Plotly topology graph renderer
│   ├── realtime_stream.py        # Live transaction stream simulation & ingestion
│   └── main.py                   # FastAPI Service endpoints (/agent/query, /eval, etc.)
├── data/
│   ├── transactions.csv          # Default active transaction dataset
│   └── generate_synthetic_data.py # Synthetic dataset generator with typologies
├── frontend/
│   └── app.py                    # Streamlit Enterprise UI Dashboard
├── tests/
│   ├── __init__.py
│   └── test_agent.py             # Unit & Integration test suite (15 tests)
├── .env.example                  # Environment variables template (Keys kept hidden)
├── .gitignore                    # Excludes venv/, .env, __pycache__/, *.sqlite3
├── Dockerfile                    # Container configuration for FastAPI Backend
├── Dockerfile.frontend           # Container configuration for Streamlit Frontend
├── docker-compose.yml            # Multi-container orchestrator (Backend + Frontend)
├── requirements.txt              # Python package dependencies
└── README.md                     # Developer, Tester & Architecture Documentation
---

## 5. Setup

```bash
git clone <repo_url> && cd aml-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # optional — add ANTHROPIC_API_KEY for LLM intent parsing

# 1. Generate synthetic data
python data/generate_synthetic_data.py --n_customers 1500 --n_days 180

# 2. Load into SQLite
python -m backend.db.db_utils

# 3. Run tests
pytest tests/ -v

# 4. Start the API
uvicorn backend.main:app --reload --port 8000

# 5. (new terminal) Start the demo UI
streamlit run frontend/app.py
```

Try it without any API key first — the regex fallback handles all four demo queries
correctly. Add `ANTHROPIC_API_KEY` to `.env` to switch on full LLM-based intent parsing
for arbitrary phrasing.

### Or, one-command spin-up with Docker

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # optional
docker compose up --build
# backend:  http://localhost:8000
# frontend: http://localhost:8501
```

The backend image generates the synthetic dataset and builds the SQLite DB at
build time, so it's immediately queryable on first run. *(Note: the Docker path is
written and reviewed but not build-tested in this delivery environment, which has no
Docker daemon — verify `docker compose up --build` once before relying on it for a
live demo.)*

---

## 6. Offline evaluation (against injected ground truth)

`python -m backend.eval.run_evaluation` scores the full pipeline against the synthetic
dataset's known-injected structuring/layering/velocity patterns:

| Tier | Precision | Recall |
|---|---|---|
| HIGH only (→ REPORT) | ~0.34 | ~0.32 |
| HIGH + MEDIUM (→ REPORT + REVIEW queue) | ~0.17 | **~0.98** |

**Why recall is prioritized over precision at the MEDIUM tier**: in AML, a missed
suspicious-activity report (false negative) carries regulatory and reputational risk
far exceeding the cost of an analyst spending a few extra minutes clearing a false
positive. The two-tier design (`REPORT` for high-confidence, `REVIEW` for a wider net)
mirrors how real compliance teams triage — it deliberately doesn't try to make one
threshold do both jobs. Thresholds are single constants in
`risk_classification_tool.py` and are the first thing to tune with real labelled data.

---

## 7. Design decisions 
- **Why SQLite, not Postgres?** Zero-ops for a hackathon judge to run locally in one
  command. `db_utils.py` isolates all SQL behind parameterized functions, so swapping
  to Postgres/Snowflake is a connection-string change, not a rewrite.
- **Why IsolationForest, not a supervised model?** Real AML data has few confirmed
  labels and adversaries adapt — supervised models trained on last year's known
  typologies miss next year's. Unsupervised anomaly detection doesn't require labels
  and generalizes to novel behaviour, at the cost of noisier precision (mitigated by
  the hybrid rule blend and tiered risk output).
- **Why template-based explanations, not LLM-generated?** Determinism. A SAR narrative
  needs to say the same thing every time given the same inputs, and every sentence
  needs to trace back to a specific numeric trigger a regulator can audit — an LLM
  free-generating prose over the same data isn't guaranteed to do that. The one case
  where a flag has no rule trigger (pure ML catch) is the weakest point for
  explainability — closed with SHAP (see below), not with more prose.
- **How is the "ML said so" black-box problem actually solved, not just claimed?**
  `shap.TreeExplainer` runs directly on the fitted `IsolationForest` for any flag with
  no rule trigger, returning the top 3 features actually driving that score (verified
  the sign convention empirically — SHAP's raw output on `IsolationForest` correlates
  at -0.996 with our anomaly-score direction, so it's flipped before display). This
  turns "the model ranked you in the 95th percentile" into "flagged primarily due to:
  max_daily_txn_count (+0.31 impact), night_txn_pct (+0.18 impact)" — an investigator
  can act on that.
- **Why does the planner literally skip tool calls instead of always running everything
  and filtering after?** Two reasons: (1) it's a direct requirement of the spec ("must
  not follow a fixed sequential pipeline"), and (2) it's real latency savings — running
  IsolationForest over a full dataset for a single-entity lookup is wasted compute at
  production scale.
- **Known limitation:** the regex intent-parser fallback is a demo-grade convenience,
  not the primary path — full deployment assumes `ANTHROPIC_API_KEY` is always set;
  precision at the HIGH tier (~0.34) would be the first thing tuned against real
  labelled SAR outcomes before production use.

#   T r u s t G r a p h  
 