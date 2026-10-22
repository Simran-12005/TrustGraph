"""
Audit Logs page -- Streamlit multipage.

Lets a reviewer look up any past agent run by run_id (or browse recent
runs) and see the full persisted record: the original query, the parsed
intent, the execution plan the planner chose, and every flag that run
produced. This is the read side of the audit trail described in the
README ("every agent run ... is persisted").
"""
import os
import sqlite3
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("AML_API_URL", "http://127.0.0.1:8000")
DB_PATH = os.environ.get("AML_DB_PATH", "aml.db")

st.set_page_config(page_title="Audit Logs - AML Platform", layout="wide")
st.title("Audit Trail")
st.caption("Every agent run is persisted to SQLite (agent_runs / flags tables) independent of dataset reloads.")


def _list_recent_runs(limit: int = 25) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT run_id, created_at, user_query FROM agent_runs "
            "ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["run_id", "created_at", "user_query"])


st.markdown("### Recent Runs")
recent = _list_recent_runs()
if recent.empty:
    st.info("No agent runs logged yet. Run a query on the main page first.")
else:
    st.dataframe(recent, hide_index=True, use_container_width=True)

st.divider()
st.markdown("### Look up a specific run")
run_id_input = st.text_input("Run ID", placeholder="e.g. ee5d86f7")

if st.button("Fetch Run", type="primary"):
    if not run_id_input.strip():
        st.warning("Enter a run_id first.")
    else:
        try:
            resp = requests.get(f"{API_URL}/agent/runs/{run_id_input.strip()}", timeout=10)
            if resp.status_code == 404:
                st.error("No run found with that run_id.")
            else:
                resp.raise_for_status()
                run = resp.json()
                c1, c2 = st.columns(2)
                c1.metric("Run ID", run["run_id"])
                c2.metric("Created At", run["created_at"][:19].replace("T", " "))
                st.markdown("**Original Query**")
                st.code(run["user_query"], language="text")

                t1, t2, t3 = st.tabs(["Parsed Intent", "Execution Plan", "Flags"])
                with t1:
                    st.json(run["parsed_intent"])
                with t2:
                    st.json(run["execution_plan"])
                with t3:
                    if run["flags"]:
                        st.dataframe(pd.DataFrame(run["flags"]), hide_index=True, use_container_width=True)
                    else:
                        st.info("No flags were raised on this run.")
        except Exception as e:
            st.error(f"Could not reach backend at {API_URL}: {e}")
