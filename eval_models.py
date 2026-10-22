import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load your dataset (update filename/path if different)
try:
    df = pd.read_csv("data/transactions.csv") # or your dataset path
except Exception:
    # Fallback to generating a quick evaluation check on active dataset
    from dataset_manager import get_active_data
    df = get_active_data()

print(f"--- DATASET STATS ---")
print(f"Total Rows: {len(df)}")
print(f"Total Features: {df.shape[1]}")
if 'is_fraud' in df.columns or 'label' in df.columns:
    target_col = 'is_fraud' if 'is_fraud' in df.columns else 'label'
    fraud_cnt = df[target_col].sum()
    benign_cnt = len(df) - fraud_cnt
    print(f"Fraud Rows: {fraud_cnt}")
    print(f"Benign Rows: {benign_cnt}")
    print(f"Fraud Percentage: {(fraud_cnt/len(df))*100:.2f}%")

    # Select numerical features for training eval
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore').fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Rows: {len(X_train)} | Testing Rows: {len(X_test)}")
    print(f"Feature List: {list(X.columns)}")

    # 2. Evaluate XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else y_pred_xgb

    print("\n--- XGBoost METRICS ---")
    print(f"Precision: {precision_score(y_test, y_pred_xgb, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_xgb, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_xgb, zero_division=0):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob_xgb):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # 3. Evaluate Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else y_pred_rf

    print("\n--- Random Forest METRICS ---")
    print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob_rf):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
else:
    print("Target column 'is_fraud' or 'label' not found. Please specify column name.")