import sqlite3
import pandas as pd

def seed_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Clear old schemas and caches
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP TABLE IF EXISTS active_transactions")
    cursor.execute("DROP TABLE IF EXISTS ground_truth")

    schema_tx = """
    CREATE TABLE transactions (
        transaction_id TEXT PRIMARY KEY,
        customer_id TEXT,
        timestamp TEXT,
        amount REAL,
        txn_type TEXT,
        channel TEXT,
        counterparty_country TEXT,
        device_id TEXT,
        ip_address TEXT,
        is_suspicious INTEGER
    )
    """
    cursor.execute(schema_tx)
    cursor.execute(schema_tx.replace("transactions", "active_transactions"))

    cursor.execute("""
    CREATE TABLE ground_truth (
        transaction_id TEXT PRIMARY KEY,
        is_suspicious INTEGER
    )
    """)

    df_cust = pd.read_csv("customers.csv")
    rows = []
    gt_rows = []

    # Explicit Customer Tiers for Pitch & Demo Consistency
    HIGH_RISK_IDS = {"CUST00001", "CUST00009", "CUST00039", "CUST00067"}
    MEDIUM_RISK_IDS = {"CUST00002", "CUST00004", "CUST00010"}

    for idx, row in df_cust.iterrows():
        c_id = str(row['customer_id'])
        kyc = str(row['kyc_risk_rating']).upper()
        
        if c_id in HIGH_RISK_IDS or kyc == 'HIGH':
            # HIGH RISK: 5 rapid wire transfers ($48,500) + Shared Fraud Ring + Ground Truth Flag
            # Triggers: High Model Confidence (~80%), Evidence Match (~80%), Graph Risk (~80%) -> Trust Score > 80%
            for i in range(1, 6):
                t_id = f"TXN_{c_id}_{i}"
                rows.append((
                    t_id, c_id, f"2026-03-01T10:0{i}:00", 48500.00,
                    "TRANSFER_OUT", "WIRE", "AE",
                    "DEV_FRAUD_RING_HIGH", "192.168.100.99", 1
                ))
                gt_rows.append((t_id, 1))

        elif c_id in MEDIUM_RISK_IDS or kyc == 'MEDIUM':
            # MEDIUM RISK: 3 transfers ($14,500) + Shared Medium Device (No Ground Truth Flag)
            # Triggers: Moderate Model Score (~40%), Moderate Graph Risk (~40%), Evidence (0%) -> Trust Score ~35%-45%
            for i in range(1, 4):
                t_id = f"TXN_{c_id}_{i}"
                rows.append((
                    t_id, c_id, f"2026-03-01T11:0{i}:00", 14500.00,
                    "TRANSFER_OUT", "WIRE", row['country'],
                    "DEV_SHARED_MED_HUB", "10.20.100.50", 0
                ))
                gt_rows.append((t_id, 0))

        else:
            # LOW RISK: Standard single $850 deposit + Unique Device & IP
            # Triggers: Model (0%), Evidence (0%), Graph Risk (~5%) -> Trust Score < 10%
            t_id = f"TXN_{c_id}_1"
            rows.append((
                t_id, c_id, "2026-03-01T12:00:00", 850.00,
                "DEPOSIT", "ACH", row['country'],
                f"DEV_CLEAN_{c_id}", f"172.16.{idx // 255}.{idx % 250 + 1}", 0
            ))
            gt_rows.append((t_id, 0))

    cursor.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    cursor.executemany("INSERT INTO active_transactions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    cursor.executemany("INSERT INTO ground_truth VALUES (?,?)", gt_rows)

    conn.commit()
    conn.close()

# Seed both local databases to prevent service mismatches
seed_database("aml.db")
seed_database("aml_platform.db")
print("✅ Databases successfully re-seeded with explicit Low, Medium, and High risk tiers!")