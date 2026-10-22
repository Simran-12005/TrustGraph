"""
System Health page -- Streamlit multipage.

Surfaces GET /api/v1/system/health (backend/db status, memory, uptime,
active dataset) plus the offline evaluation metrics, for an operator to
sanity-check the platform without needing curl/Postman.
"""
import os
import time
import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("AML_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="System Health - AML Platform", layout="wide")
st.title("System Health")

auto_refresh = st.toggle("Auto-refresh every 5s", value=False)

placeholder = st.empty()


def _render():
    with placeholder.container():
        try:
            health = requests.get(f"{API_URL}/api/v1/system/health", timeout=10).json()
        except Exception as e:
            st.error(f"Backend unreachable at {API_URL}: {e}")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Backend", health.get("backend_status", "?"))
        c2.metric("Database", health.get("database_status", "?"))
        c3.metric("API", health.get("api_status", "?"))
        c4.metric("Model Loaded", "Yes" if health.get("model_loaded") else "No")

        c5, c6, c7 = st.columns(3)
        c5.metric("Transactions", f"{health.get('number_of_transactions', 0):,}")
        c6.metric("Memory (MB)", health.get("memory_usage_mb", 0))
        c7.metric("Uptime (s)", health.get("agent_uptime_seconds", 0))

        st.caption(f"Active dataset: {health.get('current_dataset', 'unknown')}")

        st.divider()
        st.markdown("### Offline Model Evaluation")
        if st.button("Run Evaluation Now"):
            try:
                ev = requests.get(f"{API_URL}/eval", timeout=60).json()
                st.caption(
                    f"{ev['customers_scored']} customers scored, "
                    f"{ev['actual_suspicious']} actual ground-truth entities"
                )
                tier_df = pd.DataFrame([
                    {"Tier": "HIGH only (REPORT)", **ev["high_only_report_tier"]},
                    {"Tier": "HIGH + MEDIUM (REVIEW)", **ev["high_plus_medium_review_queue"]},
                ])[["Tier", "precision", "recall", "f1", "tp", "fp", "fn"]]
                st.dataframe(tier_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


_render()

if auto_refresh:
    time.sleep(5)
    st.rerun()
