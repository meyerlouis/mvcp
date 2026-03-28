"""
run_house.py — One SLURM task = one seed × one model.

Usage:  python run_house.py <seed> <model_name>
        python run_house.py 0 Ridge
        python run_house.py 7 GBR

Output: results_house/seed_0000_Ridge.json
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from realdata_experiments.run_experiment import run_single

# ═══════════════════════════════════════════════════════════
#  DATASET CONFIG
# ═══════════════════════════════════════════════════════════

DATASET_NAME = 'blog_data'
RESULTS_DIR  = f'results/{DATASET_NAME}_4d_'
DATA_DIR     = '../data'

def load_dataset():
    df = pd.read_csv(Path(DATA_DIR) / 'feldman' / 'blog_data.csv', header = None)
    targets = [60, 61, 279, 280]
    X = df[df.columns.difference(targets)].values.astype(np.float64)
    Y = df[targets].values.astype(np.float64)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    return X[mask], Y[mask]

# ═══════════════════════════════════════════════════════════
#  RUN CONFIG
# ═══════════════════════════════════════════════════════════

ALPHAS = [0.01, 0.02, 0.05, 0.1]
SPLIT  = (0.5, 0.25, 0.25)

# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    seed = int(sys.argv[1])
    model_name = sys.argv[2]

    print(f"{'='*60}", flush=True)
    print(f"  Dataset: {DATASET_NAME}", flush=True)
    print(f"  Seed:    {seed}", flush=True)
    print(f"  Model:   {model_name}", flush=True)
    print(f"  Alphas:  {ALPHAS}", flush=True)
    print(f"  Node:    {os.environ.get('SLURMD_NODENAME', 'local')}", flush=True)
    print(f"{'='*60}", flush=True)

    X, Y = load_dataset()
    print(f"  Loaded: X={X.shape}, Y={Y.shape}", flush=True)

    t0 = time.time()
    result = run_single(X, Y, seed=seed, model_name=model_name,
                        split=SPLIT, alphas=ALPHAS, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    payload = {
        'dataset': DATASET_NAME,
        'seed': seed,
        'model': model_name,
        'split': list(SPLIT),
        'alphas': ALPHAS,
        'result': make_serializable(result),
        'elapsed_s': elapsed,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, f'seed_{seed:04d}_{model_name}_4d.json')
    with open(outfile, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved {outfile}", flush=True)

if __name__ == '__main__':
    main()
