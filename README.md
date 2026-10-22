# AML Investigation Agent

An agent-driven Anti-Money Laundering (AML) investigation platform that interprets natural-language investigator queries, builds a dynamic analysis plan, detects suspicious transaction patterns using rules and machine learning, and returns risk-scored results with supporting evidence and audit logs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Example Queries](#example-queries)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Run with Docker](#run-with-docker)
- [Evaluation](#evaluation)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

## Overview

Traditional AML systems face a trade-off:

- **Rule-based systems** are explainable but may generate many false positives.
- **Machine-learning systems** can detect unusual behavior but may be harder to explain.

This project combines both approaches in an agent-driven workflow. The agent interprets an investigator's question, chooses the relevant tools, analyzes the appropriate transaction data, and returns risk classifications with evidence for review.

## Features

- **Natural-language intent parsing:** Converts investigator questions into structured intent using an LLM, with a regex-based fallback.
- **Dynamic planning:** Selects which analysis tools to run based on the query rather than using a fixed pipeline.
- **Hybrid anomaly detection:** Combines rule-based scores with an `IsolationForest` anomaly score.
- **Risk classification:** Categorizes flagged activity into risk levels such as `LOW`, `MEDIUM`, and `HIGH`.
- **Explainable results:** Uses numeric rule triggers and SHAP-based feature attribution for selected ML-driven flags.
- **Transaction-level evidence:** Includes supporting transaction records for review.
- **Audit logging:** Persists agent runs, execution plans, and resulting flags.
- **Graph visualization:** Provides transaction-network visualization using NetworkX and Plotly.
- **Simulated real-time stream:** Includes a simulated transaction stream for demonstration.
- **Evaluation tools:** Includes an offline evaluation module and tests.
- **Demo interface:** Provides a Streamlit UI backed by a FastAPI service.

## Architecture

```text
Investigator Query
        |
        v
+------------------------+
| FastAPI API             |
| /agent/query            |
+------------------------+
        |
        v
+------------------------+
| Intent Parser           |
| LLM + regex fallback    |
+------------------------+
        |
        v
+------------------------+
| Dynamic Planner         |
| Selects tools and scope |
+------------------------+
        |
        v
+-------------------------------+
| Analysis Tools                |
| - EDA (optional)              |
| - Feature engineering         |
| - Hybrid anomaly detection    |
| - Threshold aggregation       |
| - Risk classification         |
| - Evidence and explanation    |
+-------------------------------+
        |
        v
+------------------------+
| SQLite Audit Log        |
| Agent runs and flags    |
+------------------------+
        |
        v
+------------------------+
| Structured JSON Result  |
+------------------------+
        |
        v
+------------------------+
| Streamlit Dashboard     |
+------------------------+
```

## Example Queries

The planner adapts its execution path to the query type.

| Investigator query | Query type | Example execution path |
|---|---|---|
| “Find structuring patterns in the last 30 days” | `TARGETED_PATTERN` | Feature engineering → anomaly detection → risk classification → explanation |
| “Which customers made 10+ transactions under $10,000?” | `AGGREGATION_THRESHOLD` | Aggregation → risk classification → explanation |
| “Is customer ID 4521 suspicious?” | `SINGLE_ENTITY` | Customer-scoped analysis → risk classification → explanation |
| “Analyze this dataset for suspicious activity” | `BROAD_EDA` | EDA → feature engineering → anomaly detection → risk classification → explanation |

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | Python, FastAPI, Uvicorn |
| Agent planning | Python, LLM intent parsing, regex fallback |
| LLM integration | Anthropic API (optional) |
| Anomaly detection | Rule-based scoring, scikit-learn `IsolationForest` |
| Explainability | SHAP |
| Data processing | Pandas |
| Database | SQLite |
| Graph visualization | NetworkX, Plotly |
| Frontend | Streamlit |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| CI | GitHub Actions |

## Project Structure

```text
TrustGraph/
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

> Update this tree if your actual folder or file names differ.

## Getting Started

### Prerequisites

- Python
- `pip`
- Git
- Docker and Docker Compose (optional)
- An Anthropic API key (optional; the regex fallback can be used without it)

### 1. Clone the repository

Replace `<YOUR-USERNAME>` with your GitHub username.

```bash
git clone https://github.com/<YOUR-USERNAME>/TrustGraph.git
cd TrustGraph
```

If the project is already on your computer, open a terminal in the project directory and continue below.

### 2. Create a virtual environment

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

Copy the example environment file.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Add your `ANTHROPIC_API_KEY` to `.env` if you want LLM-based intent parsing. Never commit `.env` or expose API keys in your repository.

### 5. Generate synthetic data

```bash
python data/generate_synthetic_data.py --n_customers 1500 --n_days 180
```

### 6. Initialize the database

```bash
python -m backend.db.db_utils
```

### 7. Run tests

```bash
pytest tests/ -v
```

### 8. Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

### 9. Start the Streamlit UI

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run frontend/app.py
```

The API is expected at `http://localhost:8000` and the Streamlit interface at `http://localhost:8501`.

## Run with Docker

With Docker and Docker Compose installed, run:

```bash
docker compose up --build
```

Configure the optional `ANTHROPIC_API_KEY` through your local environment or Docker configuration before starting the services.

- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

Verify the Docker workflow on your machine before using it for a live demo.

## Evaluation

Run the offline evaluation module:

```bash
python -m backend.eval.run_evaluation
```

The supplied evaluation figures are based on synthetic transaction data with injected suspicious patterns.

| Evaluation tier | Precision (approx.) | Recall (approx.) |
|---|---:|---:|
| HIGH only | 0.34 | 0.32 |
| HIGH + MEDIUM | 0.17 | **0.98** |

Considering both HIGH and MEDIUM flags increases recall while also increasing false positives. These figures are specific to the project's synthetic evaluation and are not a guarantee of performance on real transaction data.

## Design Decisions

### Why combine rules and machine learning?
Rules provide transparent, deterministic triggers. `IsolationForest` can identify unusual behavior beyond the configured rules. The hybrid score is described as:

```text
hybrid_score = 0.6 * rule_score + 0.4 * ml_score
```

### Why use IsolationForest?
Unsupervised anomaly detection can be useful when labelled suspicious-activity examples are limited. However, it can produce false positives and needs evaluation and calibration.

### Why use template-based explanations?
Template-based explanations can tie findings to measurable triggers. SHAP-based feature attribution is used for selected flags without a matching rule trigger.

### Why use a dynamic planner?
The planner avoids running every tool for every query. For example, a single-customer question does not need a full-dataset EDA pass.

### Why SQLite?
SQLite keeps local setup simple for a demo. The database layer can be adapted if a different database is required, subject to implementation and testing.

## Limitations

- The project uses synthetic data; results may not generalize to real-world financial activity.
- Detection thresholds require calibration against representative labelled data.
- Regex-based intent parsing is a fallback and may not handle every possible query phrasing.
- Docker setup should be verified in the target environment.
- Real-world deployment would require security controls, privacy safeguards, monitoring, validation, and human compliance oversight.

## Disclaimer

This project is intended for educational and demonstration purposes. It is not a production-ready AML compliance system and should not be used as the sole basis for real financial-crime or regulatory decisions.
