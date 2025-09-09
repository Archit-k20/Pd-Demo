"""
generate_synthetic.py
Generates three synthetic CSV files for PdM simulation:
- synthetic_normal.csv
- synthetic_slow_degradation.csv
- synthetic_sudden_fault.csv
Also optionally writes combined_dataset.csv with a `label` column and `scenario` tag.

Usage:
    python data/generate_synthetic.py --outdir ./data --n 10000 --seed 42 --combined
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse

def make_timestamps(n, start=None, freq_seconds=1):
    """Return list of n datetimes spaced by freq_seconds starting at start (datetime)."""
    if start is None:
        start = datetime.now()
    return [start + timedelta(seconds=i*freq_seconds) for i in range(n)]

def generate_normal(n=10000, start=None, freq_seconds=1, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    ts = make_timestamps(n, start=start, freq_seconds=freq_seconds)
    temp = 60 + 0.2 * rng.randn(n)
    vibration = 2 + 0.05 * rng.randn(n)
    pressure = 30 + 0.5 * rng.randn(n)
    rpm = 1500 + 5 * rng.randn(n)
    fault = np.zeros(n, dtype=int)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temp,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "fault": fault
    })
    return df

def generate_slow_degradation(n=10000, fault_start=8000, start=None, freq_seconds=1, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed + 1)
    else:
        rng = np.random
    ts = make_timestamps(n, start=start, freq_seconds=freq_seconds)
    # baseline noise + slowly increasing/decreasing trends after fault_start
    temp = 60 + 0.2 * rng.randn(n)
    temp += np.linspace(0, 10, n) * (np.arange(n) >= fault_start)
    vibration = 2 + 0.05 * rng.randn(n)
    vibration += np.linspace(0, 3, n) * (np.arange(n) >= fault_start)
    pressure = 30 + 0.5 * rng.randn(n)
    pressure -= np.linspace(0, 2, n) * (np.arange(n) >= fault_start)
    rpm = 1500 + 5 * rng.randn(n)
    rpm -= np.linspace(0, 100, n) * (np.arange(n) >= fault_start)
    fault = np.zeros(n, dtype=int)
    fault[fault_start:] = 1
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temp,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "fault": fault
    })
    return df

def generate_sudden_fault(n=10000, fault_time=5000, start=None, freq_seconds=1, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed + 2)
    else:
        rng = np.random
    ts = make_timestamps(n, start=start, freq_seconds=freq_seconds)
    temp = 60 + 0.2 * rng.randn(n)
    vibration = 2 + 0.05 * rng.randn(n)
    pressure = 30 + 0.5 * rng.randn(n)
    rpm = 1500 + 5 * rng.randn(n)

    # inject a sudden spike over 50 samples
    end = min(n, fault_time + 50)
    span = end - fault_time
    if span > 0:
        temp[fault_time:end] += np.linspace(10, 30, span)
        vibration[fault_time:end] += np.linspace(3, 10, span)
        pressure[fault_time:end] -= np.linspace(2, 6, span)
        rpm[fault_time:end] -= np.linspace(100, 400, span)

    fault = np.zeros(n, dtype=int)
    fault[fault_time:] = 1
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temp,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "fault": fault
    })
    return df

def main(outdir, n=10000, seed=42, combined=False, freq_seconds=1):
    os.makedirs(outdir, exist_ok=True)
    # choose spaced start times so each file has distinct timestamps
    now = datetime.now()
    start_normal = now
    start_slow = now + timedelta(seconds=n + 10)   # offset by n seconds
    start_sudden = now + timedelta(seconds=2*(n + 10))

    normal = generate_normal(n=n, start=start_normal, freq_seconds=freq_seconds, seed=seed)
    slow_deg = generate_slow_degradation(n=n, fault_start=int(n*0.8),
                                        start=start_slow, freq_seconds=freq_seconds, seed=seed)
    sudden = generate_sudden_fault(n=n, fault_time=int(n*0.5),
                                   start=start_sudden, freq_seconds=freq_seconds, seed=seed)

    # Save individual files (keep original 'fault' column for clarity)
    normal_path = os.path.join(outdir, "synthetic_normal.csv")
    slow_path = os.path.join(outdir, "synthetic_slow_degradation.csv")
    sudden_path = os.path.join(outdir, "synthetic_sudden_fault.csv")

    normal.to_csv(normal_path, index=False)
    slow_deg.to_csv(slow_path, index=False)
    sudden.to_csv(sudden_path, index=False)
    print(f"Saved: {normal_path}, {slow_path}, {sudden_path}")

    # Optionally create combined dataset with `label` and `scenario` columns
    if combined:
        # add scenario tags and label (0 normal, 1 fault)
        normal2 = normal.copy()
        normal2['scenario'] = 'normal'
        normal2['label'] = 0

        slow2 = slow_deg.copy()
        slow2['scenario'] = 'slow_degradation'
        slow2['label'] = slow2['fault'].astype(int)  # 0 before fault_start, 1 after

        sudden2 = sudden.copy()
        sudden2['scenario'] = 'sudden_fault'
        sudden2['label'] = sudden2['fault'].astype(int)

        combined = pd.concat([normal2, slow2, sudden2], ignore_index=True)
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle rows
        combined_path = os.path.join(outdir, "combined_dataset.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined dataset with label and scenario: {combined_path}")
    else:
        print("Combined dataset generation skipped (use --combined to enable)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PdM CSVs")
    parser.add_argument("--outdir", type=str, default="./data", help="output directory for CSVs")
    parser.add_argument("--n", type=int, default=10000, help="rows per file")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--combined", action="store_true", help="also write combined_dataset.csv")
    parser.add_argument("--freq", type=int, default=1, help="timestamp frequency in seconds")
    args = parser.parse_args()
    main(args.outdir, n=args.n, seed=args.seed, combined=args.combined, freq_seconds=args.freq)
