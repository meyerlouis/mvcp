"""
run_one_seed.py — One SLURM array task = one seed, all alphas and lengthscales.
Usage:  python run_one_seed.py <seed>
Output: results/seed_<SEED>.json
"""

import sys, os, json, time
import numpy as np

# Single-threaded BLAS — no contention, no surprises
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ── Adds project root to path so core modules (scorers.py, experiment.py, etc.) are importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scorers import MahalanobisScorer, KernelScorer, BonferroniScorer, DensityScorer
from experiment import run_single_seed
from data_generators import generate_dgp

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


# ═══════════════════════════════════════════════════════════
#  CONFIG — edit these
# ═══════════════════════════════════════════════════════════

N              = 20_000
ALPHAS         = [0.1, 0.02, 0.05, 0.01]
LENGTHSCALES   = [0.1, 0.3, 0.5, 0.8, 1, 2, 5]  # lengthscale parameters to try
VOLUME_SAMPLES = 100_000
RESULTS_DIR    = 'results_lengthscale_v3_final'

def create_scorer_factories():
    """Create scorer factories that loop over lengthscales for Kernel and Density scorers."""
    factories = {}

    # Add Mahal and Bonferroni without lengthscale (or with placeholder)
    # factories['Mahal'] = lambda: MahalanobisScorer()
    # factories['Bonferroni'] = lambda: BonferroniScorer()

    # Create factories for each lengthscale for Kernel scorer
    for ls in LENGTHSCALES:
        factories[f'Kernel_ls{ls}'] = lambda ls=ls: KernelScorer(auto_parameters=False, lengthscale=ls)

    # Create factories for each lengthscale for Density scorer (KDE)
    for ls in LENGTHSCALES:
        factories[f'Density_ls{ls}'] = lambda ls=ls: DensityScorer(auto_parameters=False, lengthscale=ls)

    return factories

def fresh_models():
    """Return fresh (unfitted) model dict. Called per seed."""
    return {
        'Linear': LinearRegression(),
        'NN': MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=0),
    }


# ═══════════════════════════════════════════════════════════

def make_serializable(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    seed = int(sys.argv[1])
    print(f"Seed {seed} | n={N} | alphas={ALPHAS} | lengthscales={LENGTHSCALES}", flush=True)
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'local')}", flush=True)

    t0 = time.time()

    out = run_single_seed(
        dgp_fn=generate_dgp,
        models=fresh_models(),
        scorer_factories=create_scorer_factories(),
        alphas=ALPHAS,
        n=N,
        seed=seed,
        volume_n_samples=VOLUME_SAMPLES,
    )

    elapsed = time.time() - t0
    print(f"Seed {seed} done in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # Add configuration info to payload
    payload = {
        'seed': seed,
        'n': N,
        'alphas': ALPHAS,
        'lengthscales': LENGTHSCALES,
        'auto_params': make_serializable(out['auto_params']),
        'results': make_serializable(out['results']),
        'elapsed_s': elapsed,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, f'seed_{seed:04d}.json')
    with open(outfile, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {outfile}", flush=True)


if __name__ == '__main__':
    main()
