"""
One-off script: adds device_id / ip_address columns to data/transactions.csv.

TrustGraph milestone note (added by Claude, not part of the original
SocGen hackathon submission): the original dataset has no device/IP
fields, but TrustGraph's relationship-evidence layer needs *something*
to build a shared-entity graph on. This script synthesizes plausible
values rather than fabricating them as if real:

  - Each customer gets 1-2 "home" devices and IPs drawn from a pool
    sized smaller than the customer count, so overlap happens naturally
    (like two people using the same shared home router / kiosk device).
  - A deliberate fraud-ring cluster: a subset of customers already
    flagged is_suspicious=True in the ground truth are biased to share
    devices/IPs with each other at a higher rate than the general
    population, so the graph-risk signal has something real to find.
  - Deterministic (fixed seed) so re-running doesn't change the graph.

Run once: python data/synthesize_device_ip.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent
TXN_PATH = DATA_DIR / "transactions.csv"

DEVICE_POOL_SIZE = 4000  # >> customer count -> incidental overlap is rare, not the norm
IP_POOL_SIZE = 3500
RING_SIZE = 40            # suspicious customers pulled into a shared-device/IP ring
RING_DEVICE_POOL = 6      # small pool -> heavy overlap within the ring
RING_IP_POOL = 5


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(TXN_PATH)

    if "device_id" in df.columns and "ip_address" in df.columns:
        print("device_id/ip_address already present, skipping.")
        return

    customers = df["customer_id"].unique()
    n_cust = len(customers)

    device_pool = [f"DEV{ i:05d}" for i in range(DEVICE_POOL_SIZE)]
    ip_pool = [f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{i % 255}" for i in range(IP_POOL_SIZE)]

    # Pick a fraud-ring cluster from customers with any confirmed-suspicious txns
    suspicious_customers = df.loc[df["is_suspicious"] == True, "customer_id"].unique()
    rng.shuffle(suspicious_customers)
    ring_customers = set(suspicious_customers[:RING_SIZE])

    ring_devices = [f"DEV-RING-{i:03d}" for i in range(RING_DEVICE_POOL)]
    ring_ips = [f"192.168.{50+i}.{1+i}" for i in range(RING_IP_POOL)]

    # Assign each customer a primary + optional secondary device/IP
    cust_devices = {}
    cust_ips = {}
    for c in customers:
        if c in ring_customers:
            primary_dev = rng.choice(ring_devices)
            primary_ip = rng.choice(ring_ips)
            secondary_dev = rng.choice(device_pool) if rng.random() < 0.3 else None
            secondary_ip = rng.choice(ip_pool) if rng.random() < 0.3 else None
        else:
            primary_dev = rng.choice(device_pool)
            primary_ip = rng.choice(ip_pool)
            secondary_dev = rng.choice(device_pool) if rng.random() < 0.15 else None
            secondary_ip = rng.choice(ip_pool) if rng.random() < 0.15 else None
        cust_devices[c] = (primary_dev, secondary_dev)
        cust_ips[c] = (primary_ip, secondary_ip)

    def pick_device(c):
        primary, secondary = cust_devices[c]
        return secondary if (secondary is not None and rng.random() < 0.2) else primary

    def pick_ip(c):
        primary, secondary = cust_ips[c]
        return secondary if (secondary is not None and rng.random() < 0.2) else primary

    df["device_id"] = df["customer_id"].map(lambda c: pick_device(c))
    df["ip_address"] = df["customer_id"].map(lambda c: pick_ip(c))

    df.to_csv(TXN_PATH, index=False)
    print(f"Wrote device_id/ip_address for {len(df)} rows, {n_cust} customers "
          f"({len(ring_customers)} in synthetic fraud-ring cluster).")


if __name__ == "__main__":
    main()
