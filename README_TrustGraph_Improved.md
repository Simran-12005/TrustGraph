# AML Investigation Agent

**An agent-driven anti-money laundering (AML) investigation platform** that turns natural-language investigator queries into adaptive analysis plans, detects suspicious transaction patterns using hybrid rules and machine learning, and returns risk-scored results with explainable evidence and an audit trail.

> Built to demonstrate how agentic planning, explainable anomaly detection, and transaction-level evidence can work together in an AML investigation workflow.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Adaptive Agent Planning](#adaptive-agent-planning)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Run with Docker](#run-with-docker)
- [API and Demo UI](#api-and-demo-ui)
- [Offline Evaluation](#offline-evaluation)
- [Design Decisions](#design-decisions)
- [Limitations and Future Work](#limitations-and-future-work)

## Overview

Traditional AML systems often face a trade-off:

- **Rule-based systems** are transparent but can generate many false positives.
- **Black-box ML systems** can detect less obvious patterns but are harder to explain and audit.

This project combines both approaches. An agent interprets the investigator's question, builds a query-specific execution plan, invokes only the tools needed, and returns risk classifications with supporting transaction evidence.

The system is designed as a **demonstration and evaluation project**, not a production-ready banking compliance product.

## Key Features

### 🧠 Adaptive, agent-driven planning
The planner chooses which tools to run, in what order, and over which data subset. It does not execute one fixed pipeline for every query.

### 🔎 Hybrid suspicious-activity detection
The anomaly detector combines deterministic rules with an `IsolationForest` model:

`hybrid_score = 0.6 × rule_score + 0.4 × ml_score`

Rules cover patterns such as structuring, layering, and transaction velocity. The unsupervised model can surface unusual behavior not covered by the configured rules.

### 📌 Transaction-level evidence
Flags include supporting transaction records—for example, deposits near a threshold, transfer-in/transfer-out pairs, or transactions from a burst-activity day—so investigators can inspect the evidence without immediately rerunning a database query.

### 💬 Explainable results
Rule-triggered flags can be explained through their numeric triggers. For flags driven by ML without a matching rule, the system uses SHAP-based feature attribution to identify the top contributing features.

### 🧾 Audit logging
Agent queries, parsed intent, execution plans, and resulting flags are persisted in the database for later review.

### 🛟 Graceful fallback
If the Anthropic API key is missing or an API call fails, the intent parser falls back to deterministic regex-based parsing rather than stopping the workflow.

### 📈 Evaluation and visualization
The project includes offline evaluation against synthetic ground-truth patterns, a graph renderer, a simulated real-time transaction stream, a FastAPI backend, and a Streamlit demo UI.

## Architecture

```text
┌────────────────────────────┐
│ Investigator natural query │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ FastAPI: /agent/query      │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ Intent Parser              │
│ LLM tool-use + regex       │
│ fallback                   │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ Dynamic Planner            │
│ Selects tools, order, and  │
│ data scope                 │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ Analysis Tools              │
│ • EDA (optional)            │
│ • Feature engineering       │
│ • Hybrid anomaly detection  │
│ • Threshold aggregation     │
│ • Risk classification      │
│ • Evidence and explanation │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ SQLite audit log            │
│ agent_runs, flags            │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ Structured JSON response   │
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ Streamlit demo dashboard   │
└────────────────────────────┘
```

## Adaptive Agent Planning

The planner adapts the execution path to the query type.

| Example investigator query | Detected query type | Tools invoked | Tools skipped |
|---|---|---|---|
| “Find structuring patterns in the last 30 days” | `TARGETED_PATTERN` | Feature engineering → anomaly detection → risk classification → explanation | EDA |
| “Which customers made 10+ transactions under $10,000?” | `AGGREGATION_THRESHOLD` | Aggregation rule → risk classification → explanation | EDA, ML anomaly detection |
| “Is customer ID 4521 suspicious?” | `SINGLE_ENTITY` | Feature engineering scoped to one customer → anomaly detection → risk classification → explanation | EDA |
| “Analyse this dataset for suspicious activity” | `BROAD_EDA` | EDA → feature engineering → anomaly detection → risk classification → explanation | None |

The query-specific routing logic lives in `backend/agent/planner.py`. The planner returns its reasoning with the execution plan to support transparent review. The supplied project notes identify `tests/test_agent.py` as the test suite for this behavior.

## Technology Stack

| Area | Technologies |
|---|---|
| API backend | Python, FastAPI, Uvicorn |
| Agent orchestration | LLM-based intent parsing, regex fallback, dynamic planner |
| LLM integration | Anthropic API (optional) |
| Anomaly detection | Rule-based scoring, scikit-learn `IsolationForest`, SHAP |
| Data processing | Pandas and Python-based feature engineering |
| Database and audit trail | SQLite; schema designed with PostgreSQL compatibility in mind |
| Graph visualization | NetworkX, Plotly |
| Demo interface | Streamlit |
| Testing | Pytest |
| Deployment | Docker, Docker Compose, GitHub Actions CI |

## Repository Structure

```text
aml-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── backend/
│   ├── agent/
│   │   ├── intent_parser.py
│   │   ├── planner.py
│   │   └── orchestrator.py
│   ├── tools/
│   │   ├── eda_tool.py
│   │   ├── feature_engineering_tool.py
│   │   ├── anomaly_detection_tool.py
│   │   ├── risk_classification_tool.py
│   │   ├── explanation_tool.py
│   │   ├── aggregation_tool.py
│   │   └── evidence_tool.py
│   ├── db/
│   │   ├── schema.sql
│   │   └── db_utils.py
│   ├── eval/
│   │   └── run_evaluation.py
│   ├── dataset_manager.py
│   ├── graph_engine.py
│   ├── realtime_stream.py
│   └── main.py
├── data/
│   ├── transactions.csv
│   └── generate_synthetic_data.py
├── frontend/
│   └── app.py
├── tests/
│   └── test_agent.py
├── .env.example
├── .gitignore
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python installed
- `pip`
- Git, if cloning the repository
- An Anthropic API key only if you want to use LLM-based intent parsing; the regex fallback can be used without it
- Docker and Docker Compose only if using the containerized setup

### 1. Clone the repository

Replace `<YOUR-USERNAME>` with your GitHub username:

```bash
git clone https://github.com/<YOUR-USERNAME>/aml-agent.git
cd aml-agent
```

If you already have the project locally, open a terminal in the project directory and continue with the next step.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Add your `ANTHROPIC_API_KEY` to `.env` if you want LLM-based intent parsing. Keep real secrets out of Git and never commit `.env`.

### 5. Generate synthetic data

```bash
python data/generate_synthetic_data.py --n_customers 1500 --n_days 180
```

### 6. Initialize the database

```bash
python -m backend.db.db_utils
```

### 7. Run the tests

```bash
pytest tests/ -v
```

### 8. Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

### 9. Start the Streamlit interface

Open a **second terminal**, activate the virtual environment again if needed, then run:

```bash
streamlit run frontend/app.py
```

Try the demo without an API key first. The regex fallback is intended to handle the four example query types above. Add a valid `ANTHROPIC_API_KEY` to enable LLM-based parsing for more varied phrasing.

## Run with Docker

If Docker and Docker Compose are installed, you can start the services with:

```bash
docker compose up --build
```

To provide an optional API key, set `ANTHROPIC_API_KEY` in your environment or configure it through your local environment setup before starting the containers.

Expected local endpoints:

- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

> **Note:** The supplied project notes say the Docker workflow had not been build-tested in the delivery environment. Run and verify the Docker setup locally before relying on it for a demo.

## API and Demo UI

- **FastAPI backend:** `http://localhost:8000`
- **Streamlit demo UI:** `http://localhost:8501`
- **Main agent endpoint:** `/agent/query`

For interactive API documentation, FastAPI commonly exposes Swagger UI at `http://localhost:8000/docs` when enabled by the application.

## Offline Evaluation

Run the evaluation module:

```bash
python -m backend.eval.run_evaluation
```

The evaluation uses synthetic data with injected structuring, layering, and velocity patterns.

| Reporting tier | Precision (approx.) | Recall (approx.) |
|---|---:|---:|
| HIGH only → report | 0.34 | 0.32 |
| HIGH + MEDIUM → report and review queue | 0.17 | **0.98** |

### Interpreting these results

The current setup emphasizes recall when HIGH and MEDIUM flags are both considered, helping reduce the chance of missing suspicious patterns at the cost of more false positives. These results are from the project's synthetic evaluation setup; they should not be interpreted as performance guarantees on real financial transaction data.

Risk thresholds are defined in `backend/tools/risk_classification_tool.py` and should be calibrated against representative, labelled data before any real-world use.

## Design Decisions

### Why combine rules and machine learning?
Rules provide deterministic triggers that are straightforward to inspect. `IsolationForest` can identify unusual behavior beyond the configured rules. The hybrid score combines the two signals while retaining rule-based explanations where available.

### Why use IsolationForest?
Confirmed AML labels can be scarce, and suspicious behavior can evolve. An unsupervised anomaly detector does not require labelled examples, although its results may be noisy and require careful evaluation.

### Why use template-based explanations?
Template-based explanations make it easier to trace statements back to specific numeric triggers. For ML-driven flags without a matching rule, SHAP-based attribution is used to surface influential features rather than relying only on a generic model score.

### Why does the planner skip tools?
Query-specific tool selection avoids unnecessary work—for example, running full-dataset exploratory analysis for a single-customer query. It also demonstrates that the system does not follow a fixed sequential pipeline.

### Why SQLite?
SQLite keeps local setup simple for a demo. The database access layer is intended to isolate SQL operations so that a different database backend can be introduced with limited changes, subject to implementation and compatibility testing.

## Limitations and Future Work

- The project uses synthetic transaction data; results may not generalize to real financial activity.
- The regex intent-parser fallback is intended as a simple fallback, not a replacement for evaluating the LLM path on varied investigator queries.
- The reported precision and recall indicate that thresholds and detection logic need further calibration.
- Docker setup should be verified locally before use in a live demo.
- Before any real-world deployment, the system would need rigorous validation, security review, access controls, privacy safeguards, monitoring, and compliance-team oversight.

---

**Disclaimer:** This repository is an educational/demo project. It does not provide legal, regulatory, or financial-crime compliance advice and should not be used as the sole basis for real AML decisions.
