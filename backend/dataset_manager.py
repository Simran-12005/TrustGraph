"""
Dataset Manager for Custom CSV Uploads and Active Dataset Context.
"""

import io
import sqlite3
import pandas as pd
from typing import Dict, Any, Tuple

REQUIRED_TRANSACTION_COLS = {'transaction_id', 'customer_id', 'amount', 'timestamp'}
REQUIRED_CUSTOMER_COLS = {'customer_id', 'risk_score'}
REQUIRED_CUSTOMER_COLS_ALT = {'customer_id', 'kyc_risk_rating'}


class DatasetManager:
    _instance = None

    def __new__(cls, db_path: str = "aml_platform.db"):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance.active_df = None
            cls._instance.metadata = {
                "filename": "synthetic_dataset.csv",
                "is_custom": False,
                "total_rows": 0,
                "total_customers": 0,
                "dataset_type": "transaction"
            }
        return cls._instance

    def validate_csv(self, df: pd.DataFrame) -> Tuple[bool, str, str]:
        df.columns = df.columns.str.lower().str.strip()
        cols = set(df.columns)
        
        if REQUIRED_TRANSACTION_COLS.issubset(cols):
            return True, "transaction", "Valid transaction-level dataset."
        elif REQUIRED_CUSTOMER_COLS.issubset(cols) or REQUIRED_CUSTOMER_COLS_ALT.issubset(cols):
            return True, "customer", "Valid customer-level dataset."
        else:
            missing_tx = REQUIRED_TRANSACTION_COLS - cols
            missing_cust = REQUIRED_CUSTOMER_COLS - cols
            return False, "unknown", f"Missing required columns. Transaction missing: {missing_tx} OR Customer missing: {missing_cust}"

    def load_custom_csv(self, file_contents: bytes, filename: str) -> Dict[str, Any]:
        df = pd.read_csv(io.BytesIO(file_contents))
        valid, dataset_type, msg = self.validate_csv(df)
        if not valid:
            raise ValueError(msg)

        total_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0

        self.active_df = df
        self.metadata = {
            "filename": filename,
            "is_custom": True,
            "total_rows": len(df),
            "total_customers": int(total_customers),
            "dataset_type": dataset_type
        }
        
        conn = sqlite3.connect(self.db_path)
        df.to_sql("active_transactions", conn, if_exists="replace", index=False)
        conn.close()

        return self.metadata

    def get_active_data(self) -> pd.DataFrame:
        if self.active_df is not None:
            return self.active_df

        # 1) A custom CSV uploaded via /api/v1/dataset/upload, if any.
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT * FROM active_transactions", conn)
            conn.close()
            if not df.empty:
                self.active_df = df
                return df
        except Exception:
            pass

        # 2) Fall back to the real transaction dataset the agent actually
        # queries (backend.db.db_utils / aml.db), so the graph and other
        # dataset-manager-backed features reflect real data by default
        # rather than an always-empty separate database.
        try:
            from backend.db import db_utils
            df = db_utils.load_transactions()
        except Exception:
            df = pd.DataFrame(columns=['transaction_id', 'customer_id', 'amount', 'timestamp'])

        self.active_df = df
        return df


dataset_manager = DatasetManager()