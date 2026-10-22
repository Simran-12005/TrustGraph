# TrustGraph: Bridging Explainability and Evidence in Fraud Detection

TrustGraph is a fraud-detection decision-support prototype that brings together **machine-learning predictions, explainable AI, historical-case evidence, and relationship-graph analysis**. Instead of presenting an investigator with a model score alone, it combines these signals into a decomposable **Trust Score** and records the investigator's decision in an audit log.

The project was developed as **BITE497J – Project I** at the Vellore Institute of Technology (VIT), Vellore.

## Table of Contents

- [Problem](#problem)
- [What TrustGraph Does](#what-trustgraph-does)
- [System Workflow](#system-workflow)
- [Trust Score](#trust-score)
- [Demonstration Results](#demonstration-results)
- [Dataset and Model Evaluation](#dataset-and-model-evaluation)
- [Technology Stack](#technology-stack)
- [Core Components](#core-components)
- [Setup](#setup)
- [Limitations](#limitations)
- [Team](#team)
- [Disclaimer](#disclaimer)

## Problem

Fraud-detection models can flag suspicious customers, but a prediction score alone does not explain why an account was flagged or whether the explanation is supported by historical cases. In addition, fraud may involve shared devices and IP addresses that connect otherwise unrelated accounts.

TrustGraph addresses this gap by combining:

1. **Prediction and explainability** — a supervised classifier with SHAP feature explanations and DiCE counterfactual recourse.
2. **Historical evidence validation** — vector-similarity retrieval of similar cases from a PostgreSQL/pgvector case repository.
3. **Relationship analysis** — a graph of customers, devices, and IP addresses.
4. **Auditable investigation** — a dashboard that presents the signals together and stores investigator decisions.

## What TrustGraph Does

- Scores customer records with a supervised fraud classifier.
- Generates local explanations using **SHAP**.
- Generates counterfactual examples using **DiCE** to show changes that could alter a model decision.
- Retrieves similar historical cases using sentence embeddings and **PostgreSQL with pgvector**.
- Analyses shared devices and IP addresses using **NetworkX** and visualises relationships with **PyVis**.
- Combines four signals into a weighted Trust Score from 0 to 100.
- Presents results in a **Streamlit** investigator dashboard.
- Records investigator decisions through a **FastAPI** endpoint into a PostgreSQL audit table.

## System Workflow

```text
                  Transaction / Customer Data
                             |
                             v
                   Data Preprocessing
                             |
                             v
                  Feature Engineering
                             |
                             v
                  Fraud Classification
                             |
               +-------------+-------------+
               |             |             |
               v             v             v
          SHAP Drivers    DiCE Recourse  Model Score
               |             |             |
               +-------------+-------------+
                             |
             +---------------+---------------+
             |                               |
             v                               v
   Historical Evidence Retrieval       Entity Graph
   PostgreSQL + pgvector               NetworkX + PyVis
             |                               |
             v                               v
     Evidence Similarity                 Graph Risk
             |                               |
             +---------------+---------------+
                             |
                             v
                  Explanation-Evidence
                      Consistency
                             |
                             v
                       Trust Score
                             |
                             v
                 Streamlit Investigator UI
                             |
                             v
                   Investigator Decision
                             |
                             v
                  FastAPI Audit Endpoint
                             |
                             v
                  PostgreSQL audit_logs
```

## Trust Score

The Trust Score combines four components, each represented on a 0–100 scale:

| Component | Symbol | Weight | Meaning |
|---|---:|---:|---|
| Model Score | M | 0.35 | Classifier confidence |
| Evidence Similarity | E | 0.25 | Similarity to the closest historical case |
| Graph Risk | G | 0.25 | Risk derived from relationships in the entity graph |
| Explanation-Evidence Consistency | K | 0.15 | Agreement between the explanation and retrieved case |

The formula is:

```text
Trust Score = 0.35 × M + 0.25 × E + 0.25 × G + 0.15 × K
```

The result is a **risk-evidence score**, not a calibrated probability of fraud. A higher score indicates that more supporting signals contribute to a high-risk assessment. The component values are shown separately so investigators can understand how the score was formed.

## Demonstration Results

The report demonstrates the dashboard with three illustrative customer cases.

| Customer | Model Score | Evidence | Graph Risk | Consistency | Trust Score | Level |
|---|---:|---:|---:|---:|---:|---|
| `CUST00009` | 90% | 77% | 100% | 80% | 87% | HIGH |
| `CUST00002` | 53% | 35% | 100% | 46% | 59% | MEDIUM |
| `CUST00003` | 5% | 0% | 0% | 5% | 2% | LOW |

These cases illustrate how model confidence, historical evidence, graph risk, and consistency contribute to the final score. They are examples from the project demonstration, not a statistical estimate of overall model performance.

## Dataset and Model Evaluation

The project report describes a hybrid dataset containing **69,280 records** across an 11-column schema:

- **68,568 benign records** (98.97%)
- **712 suspicious records** (1.03%)

The reported baseline classifier results are:

| Metric | Reported result |
|---|---:|
| Accuracy | 98.19% |
| Recall | 2.82% |
| AUC-ROC | 0.5695 |

The high accuracy alongside low recall highlights the effect of severe class imbalance: the baseline misses many suspicious records. Therefore, the classifier should be treated as a limited baseline, and the reported metrics should be considered when interpreting dashboard results.

The report identifies further model comparison, systematic ablation studies, and improved evaluation as future work.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Classification | XGBoost |
| Explainable AI | SHAP, DiCE |
| Evidence retrieval | Sentence embeddings, cosine similarity |
| Vector storage | PostgreSQL, pgvector |
| Graph analytics | NetworkX |
| Graph visualisation | PyVis |
| Backend / audit API | FastAPI |
| Dashboard | Streamlit |
| Data processing | Python data-processing pipeline |
| Database audit trail | PostgreSQL `audit_logs` table |

## Core Components

### 1. Fraud Classification
Processes structured customer/transaction data and produces a fraud prediction with a model score.

### 2. Explainability
Uses SHAP to identify influential features and DiCE to generate counterfactual examples for an individual prediction.

### 3. Evidence Validation
Embeds a customer's behavioural description and retrieves the most similar historical case using vector similarity search in PostgreSQL/pgvector.

### 4. Relationship Graph
Connects customers with shared devices and IP addresses. NetworkX supports graph analysis, while PyVis provides interactive visualisation.

### 5. Trust Score Fusion
Combines model confidence, evidence similarity, graph risk, and explanation-evidence consistency into a single score with visible component contributions.

### 6. Investigator Dashboard and Audit Logging
The Streamlit dashboard presents the prediction, explanation, historical evidence, graph, and Trust Score. Investigator decisions are submitted to a FastAPI endpoint and persisted in PostgreSQL.

## Setup

The report identifies the main technologies and system components, but the exact repository file layout, dependency versions, and environment-variable names should be confirmed against the code in this GitHub repository before following setup commands.

A typical local setup sequence for a Python project is:

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`, if present.
4. Configure database connection settings and any required environment variables.
5. Start the FastAPI backend and Streamlit dashboard using the commands defined by the project.
6. Ensure PostgreSQL and the pgvector extension are configured before running evidence retrieval and audit logging.

**Before publishing:** replace this section with the exact, tested commands and configuration required by your current codebase.

## Limitations

- The reported baseline classifier has low recall (2.82%) despite high accuracy (98.19%).
- The dataset is highly imbalanced, with suspicious records accounting for 1.03% of the reported data.
- The three dashboard cases are illustrative demonstrations, not a representative statistical sample.
- The implemented graph focuses on shared customer, device, and IP relationships; the report notes that financial-flow edges are not modelled.
- Real-time, production-scale ingestion, integration with a live banking core system, regulatory certification, and multi-institution federated sharing are outside the current scope.
- The Trust Score is a decision-support signal, not a calibrated probability and not a substitute for investigator review.

## Team

| Team member | Primary area |
|---|---|
| Simran Gupta | Machine learning, feature selection, class balancing, SHAP and DiCE |
| Uzma Habeeba Shaik | System architecture, FastAPI backend, Trust Score fusion, PostgreSQL/pgvector, audit logging |
| Akshaya Ashok Nair | Streamlit dashboard, NetworkX graph, PyVis visualisation, JSON export |

**Project:** BITE497J – Project I  
**Institution:** School of Computer Science Engineering and Information Systems, Vellore Institute of Technology (VIT), Vellore  
**Project guide:** Dr. Ranichandra C, Associate Professor, SCORE

## Disclaimer

TrustGraph is an academic project prototype for fraud-detection decision support. It is not a production-ready financial compliance system and should not be used as the sole basis for real-world fraud, account, or regulatory decisions.
