"""
Synthetic AML dataset generator.

Produces two CSVs:
  - customers.csv   : customer master data (segment, country, risk profile)
  - transactions.csv: transaction ledger with injected laundering typologies

Ground-truth columns (is_suspicious, pattern_type) are included ONLY for
offline evaluation / demo scoring. They are NEVER shown to the agent at
inference time (backend/db/db_utils.py loads them into a separate table).

Patterns injected:
  1. Structuring / smurfing   - many transactions just under the $10,000
                                 CTR reporting threshold, in short bursts.
  2. Rapid layering           - fast in-out cycling of funds across a
                                 small ring of accounts within hours.
  3. Velocity spikes          - a normally quiet customer suddenly firing
                                 many transactions in a single day.

Run: python generate_synthetic_data.py --n_customers 1500 --n_days 180
"""
import argparse
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

SEGMENTS = ["Retail", "SME", "Corporate", "Private Banking"]
COUNTRIES = ["FR", "US", "GB", "AE", "SG", "NG", "PA", "CH", "DE", "IN"]
HIGH_RISK_COUNTRIES = {"AE", "NG", "PA"}  # illustrative, not a real FATF list
CHANNELS = ["WIRE", "ACH", "CASH_DEPOSIT", "CARD", "INTERNAL_TRANSFER"]
TXN_TYPES = ["DEPOSIT", "WITHDRAWAL", "TRANSFER_IN", "TRANSFER_OUT"]


def gen_customers(n_customers: int) -> pd.DataFrame:
    rows = []
    for i in range(n_customers):
        cid = f"CUST{i+1:05d}"
        segment = RNG.choice(SEGMENTS, p=[0.55, 0.25, 0.12, 0.08])
        country = RNG.choice(COUNTRIES, p=[0.25, 0.2, 0.15, 0.08, 0.07, 0.07, 0.05, 0.05, 0.04, 0.04])
        onboarding_days_ago = int(RNG.integers(30, 3000))
        rows.append({
            "customer_id": cid,
            "segment": segment,
            "country": country,
            "is_high_risk_country": country in HIGH_RISK_COUNTRIES,
            "onboarding_date": (datetime.today() - timedelta(days=onboarding_days_ago)).date().isoformat(),
            "kyc_risk_rating": RNG.choice(["LOW", "MEDIUM", "HIGH"], p=[0.7, 0.22, 0.08]),
        })
    return pd.DataFrame(rows)


def _base_amount(segment: str) -> float:
    scale = {"Retail": 400, "SME": 3500, "Corporate": 20000, "Private Banking": 15000}[segment]
    return float(max(5, RNG.gamma(shape=2.0, scale=scale / 2)))


def gen_normal_transactions(customers: pd.DataFrame, n_days: int) -> pd.DataFrame:
    rows = []
    start = datetime.today() - timedelta(days=n_days)
    for _, cust in customers.iterrows():
        n_txns = int(RNG.poisson(lam={"Retail": 25, "SME": 60, "Corporate": 120, "Private Banking": 40}[cust.segment]))
        for _ in range(n_txns):
            ts = start + timedelta(
                days=int(RNG.integers(0, n_days)),
                hours=int(RNG.integers(6, 22)),
                minutes=int(RNG.integers(0, 60)),
            )
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust.customer_id,
                "timestamp": ts.isoformat(),
                "amount": round(_base_amount(cust.segment), 2),
                "txn_type": RNG.choice(TXN_TYPES),
                "channel": RNG.choice(CHANNELS, p=[0.2, 0.25, 0.15, 0.3, 0.1]),
                "counterparty_country": RNG.choice(COUNTRIES),
                "is_suspicious": False,
                "pattern_type": "NONE",
            })
    return pd.DataFrame(rows)


def inject_structuring(customers: pd.DataFrame, n_days: int, n_rings: int = 40) -> pd.DataFrame:
    """Bursts of cash deposits just under the $10,000 CTR threshold."""
    rows = []
    start = datetime.today() - timedelta(days=n_days)
    targets = customers.sample(n=n_rings, random_state=1)
    for _, cust in targets.iterrows():
        burst_day = int(RNG.integers(0, n_days))
        burst_start = start + timedelta(days=burst_day)
        n_txns = int(RNG.integers(4, 9))
        for k in range(n_txns):
            ts = burst_start + timedelta(hours=int(RNG.integers(0, 30)))
            amount = round(float(RNG.uniform(8800, 9970)), 2)  # just under $10k
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust.customer_id,
                "timestamp": ts.isoformat(),
                "amount": amount,
                "txn_type": "DEPOSIT",
                "channel": "CASH_DEPOSIT",
                "counterparty_country": cust.country,
                "is_suspicious": True,
                "pattern_type": "STRUCTURING",
            })
    return pd.DataFrame(rows)


def inject_layering(customers: pd.DataFrame, n_days: int, n_rings: int = 20) -> pd.DataFrame:
    """Fast in->out cycling through an account within hours (layering)."""
    rows = []
    start = datetime.today() - timedelta(days=n_days)
    targets = customers.sample(n=n_rings, random_state=2)
    for _, cust in targets.iterrows():
        day = int(RNG.integers(0, n_days))
        base = start + timedelta(days=day, hours=int(RNG.integers(0, 12)))
        amount = round(float(RNG.uniform(15000, 60000)), 2)
        rows.append({
            "transaction_id": str(uuid.uuid4()), "customer_id": cust.customer_id,
            "timestamp": base.isoformat(), "amount": amount, "txn_type": "TRANSFER_IN",
            "channel": "WIRE", "counterparty_country": RNG.choice(list(HIGH_RISK_COUNTRIES)),
            "is_suspicious": True, "pattern_type": "LAYERING",
        })
        out_ts = base + timedelta(minutes=int(RNG.integers(20, 180)))
        rows.append({
            "transaction_id": str(uuid.uuid4()), "customer_id": cust.customer_id,
            "timestamp": out_ts.isoformat(), "amount": round(amount * RNG.uniform(0.9, 0.99), 2),
            "txn_type": "TRANSFER_OUT", "channel": "WIRE",
            "counterparty_country": RNG.choice(list(HIGH_RISK_COUNTRIES)),
            "is_suspicious": True, "pattern_type": "LAYERING",
        })
    return pd.DataFrame(rows)


def inject_velocity_spike(customers: pd.DataFrame, n_days: int, n_targets: int = 25) -> pd.DataFrame:
    """A normally quiet customer fires an abnormal number of transactions in one day."""
    rows = []
    start = datetime.today() - timedelta(days=n_days)
    targets = customers.sample(n=n_targets, random_state=3)
    for _, cust in targets.iterrows():
        day = int(RNG.integers(0, n_days))
        base = start + timedelta(days=day)
        for _ in range(int(RNG.integers(12, 25))):
            ts = base + timedelta(minutes=int(RNG.integers(0, 1440)))
            rows.append({
                "transaction_id": str(uuid.uuid4()), "customer_id": cust.customer_id,
                "timestamp": ts.isoformat(), "amount": round(_base_amount(cust.segment) * RNG.uniform(1, 3), 2),
                "txn_type": RNG.choice(["TRANSFER_OUT", "WITHDRAWAL"]), "channel": RNG.choice(CHANNELS),
                "counterparty_country": RNG.choice(COUNTRIES),
                "is_suspicious": True, "pattern_type": "VELOCITY_SPIKE",
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_customers", type=int, default=1500)
    ap.add_argument("--n_days", type=int, default=180)
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    customers = gen_customers(args.n_customers)
    normal = gen_normal_transactions(customers, args.n_days)
    structuring = inject_structuring(customers, args.n_days)
    layering = inject_layering(customers, args.n_days)
    velocity = inject_velocity_spike(customers, args.n_days)

    txns = pd.concat([normal, structuring, layering, velocity], ignore_index=True)
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    txns["transaction_id"] = [f"TXN{i+1:07d}" for i in range(len(txns))]

    customers.to_csv(f"{args.out_dir}/customers.csv", index=False)
    txns.to_csv(f"{args.out_dir}/transactions.csv", index=False)

    print(f"Customers: {len(customers)}")
    print(f"Transactions: {len(txns)}  (suspicious: {int(txns.is_suspicious.sum())}, "
          f"{txns.is_suspicious.mean()*100:.2f}%)")
    print(txns.pattern_type.value_counts())


if __name__ == "__main__":
    main()
