# reset_device_ip.py
import pandas as pd

df = pd.read_csv('data/transactions.csv')
if 'device_id' in df.columns:
    df = df.drop(columns=['device_id', 'ip_address'])
    df.to_csv('data/transactions.csv', index=False)
    print("Dropped device_id/ip_address, ready to re-synthesize.")
else:
    print("No device_id/ip_address columns found — nothing to reset.")