"""
Core experiment pipeline for real-data conformal prediction.

Atomic unit: run_single(X, Y, seed, model_name)
    -> one seed, one model, all scorers, all alphas.

Scorer fitting (the expensive part) is alpha-independent, so alphas
loop cheaply inside run_single. Parallelise over (seed, model_name).

Usage (sequential):
    from realdata_experiments.run_experiment import run_single, summarize, print_summary

    r = run_single(X, Y, seed=0, model_name='Ridge')
    # r['Kernel'][0.1]['coverage'] -> float

Usage (parallel):
    from itertools import product
    from joblib import Parallel, delayed
    from realdata_experiments.run_experiment import run_single, summarize, print_summary

    jobs = list(product(range(50), ['Ridge', 'MLP']))
    all_results = Parallel(n_jobs=-1)(
        delayed(run_single)(X, Y, seed, mn)
        for seed, mn in jobs
    )
    summary = summarize(all_results)
    print_summary(summary)
"""

import numpy as np
import time
import warnings

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from scorers import (
    KernelScorer, MahalanobisScorer, BonferroniScorer, DensityScorer,
    conformal_quantile,
)
from volume_estimator import estimate_volume, ellipsoid_volume
from metrics import compute_wsc


# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_SPLIT = (0.50, 0.25, 0.25)
DEFAULT_ALPHAS = [0.01, 0.02, 0.05, 0.1]

VOLUME_N_SAMPLES = 1_000_000

WSC_N_SLABS = 200


MODEL_REGISTRY = {
    'Ridge': lambda: MultiOutputRegressor(Ridge(alpha=1.0)),
    'GBR': lambda: MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=0,
        ),
        n_jobs=1,
    ),
    'MLP': lambda: MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        random_state=0,
    ),
}

SCORER_FACTORIES = {
    'Mahal':      lambda: MahalanobisScorer(),
    'Bonferroni': lambda: BonferroniScorer(),
    'Density':    lambda: DensityScorer(auto_parameters=True),
    'Kernel':     lambda: KernelScorer(auto_parameters=True),
}


def _split_and_scale(X, Y, seed, split):
    """Split into train/cal/test and standardize."""
    n = len(X)
    perm = np.random.RandomState(seed).permutation(n)

    n_train = int(n * split[0])
    n_cal = int(n * split[1])

    X_tr, Y_tr = X[perm[:n_train]], Y[perm[:n_train]]
    X_cal, Y_cal = X[perm[n_train:n_train + n_cal]], Y[perm[n_train:n_train + n_cal]]
    X_te, Y_te = X[perm[n_train + n_cal:]], Y[perm[n_train + n_cal:]]

    scaler_x = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(Y_tr)

    return {
        'X_train': scaler_x.transform(X_tr),
        'Y_train': scaler_y.transform(Y_tr),
        'X_cal':   scaler_x.transform(X_cal),
        'Y_cal':   scaler_y.transform(Y_cal),
        'X_test':  scaler_x.transform(X_te),
        'Y_test':  scaler_y.transform(Y_te),
    }


def run_single(X, Y, seed, model_name, split=None, alphas=None, verbose=True):
    """
    One seed, one model, all scorers, all alphas.

    :param X: (n, p) raw features
    :param Y: (n, d) raw targets
    :param seed: int, controls train/cal/test split
    :param model_name: str, key in MODEL_REGISTRY ('Ridge', 'GBR', 'MLP')
    :param split: tuple (train, cal, test), default (0.5, 0.25, 0.25)
    :param alphas: list of floats, default [0.01, 0.05, 0.1]
    :param verbose: bool

    :return: dict with structure:
        {
            'seed': int,
            'model': str,
            'model_rmse': float,
            '<scorer_name>': {
                <alpha>: {
                    'coverage': float,
                    'volume': float,
                    'wsc': float,
                    'threshold': float,
                },
                ...
                'params': {'lengthscale': float, 'gamma': float, ...},
            },
            ...
            'meta': {'n_train': int, 'n_cal': int, 'n_test': int, 'd': int, 'p': int, 'elapsed': float},
        }
    """
    t0 = time.time()
    split = split or DEFAULT_SPLIT
    alphas = alphas or DEFAULT_ALPHAS

    # ── Split and standardize ─────────────────────────────────────
    data = _split_and_scale(X, Y, seed, split)
    d = data['Y_train'].shape[1]
    p = data['X_train'].shape[1]

    # ── Fit model ─────────────────────────────────────────────────
    print(f'Fitting {model_name}...')
    model = MODEL_REGISTRY[model_name]()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(data['X_train'], data['Y_train'])
    print(f'Getting residuals...')
    pred_cal = model.predict(data['X_cal'])
    pred_test = model.predict(data['X_test'])
    if pred_cal.ndim == 1:
        pred_cal = pred_cal[:, None]
    if pred_test.ndim == 1:
        pred_test = pred_test[:, None]

    rc = data['Y_cal'] - pred_cal
    rt = data['Y_test'] - pred_test
    model_rmse = float(np.sqrt(np.mean((data['Y_test'] - pred_test) ** 2)))

    # ── Fit all scorers (expensive, alpha-independent) ────────────
    fitted_scorers = {}
    scores_cal = {}
    scores_test = {}

    for scorer_name, factory in SCORER_FACTORIES.items():
        print(f'Fitting {scorer_name}...')
        scorer = factory()
        scorer.fit(rc)
        print(f'Scorer {scorer_name} fitted')
        fitted_scorers[scorer_name] = scorer
        scores_cal[scorer_name] = scorer.score(rc)
        scores_test[scorer_name] = scorer.score(rt)
        print(f'Scorer {scorer_name} scored')
    # ── Evaluate at each alpha (cheap) ────────────────────────────
    # Pre-compute Mahalanobis thresholds for volume estimation
    mahal_scorer = fitted_scorers['Mahal']
    mahal_thresholds = {
        alpha: conformal_quantile(scores_cal['Mahal'], alpha)
        for alpha in alphas
    }

    results = {
        'seed': seed,
        'model': model_name,
        'model_rmse': model_rmse,
        'meta': {
            'n_train': data['X_train'].shape[0],
            'n_cal': data['X_cal'].shape[0],
            'n_test': data['X_test'].shape[0],
            'd': d,
            'p': p,
        },
    }

    for scorer_name, scorer in fitted_scorers.items():
        results[scorer_name] = {}

        # Log auto params
        params = {}
        if hasattr(scorer, 'lengthscale') and scorer.lengthscale is not None:
            params['lengthscale'] = float(scorer.lengthscale)
        if hasattr(scorer, 'gamma') and scorer.gamma is not None:
            params['gamma'] = float(scorer.gamma)
        results[scorer_name]['params'] = params

        for alpha in alphas:
            print(f'Estimating results for scorer {scorer_name} and alpha {alpha}...')
            threshold = conformal_quantile(scores_cal[scorer_name], alpha)
            covered = scores_test[scorer_name] <= threshold
            coverage = float(covered.mean())
            wsc = compute_wsc(data['X_test'], covered, n_slabs=WSC_N_SLABS)

            # Volume
            if isinstance(scorer, MahalanobisScorer) and scorer.cov is not None:
                volume = ellipsoid_volume(scorer.cov, threshold, d)
            else:
                volume = estimate_volume(
                    scorer, threshold, rc,
                    mahal_scorer=mahal_scorer,
                    mahal_threshold=mahal_thresholds[alpha],
                    n_samples=VOLUME_N_SAMPLES,
                    seed=0,
                )

            results[scorer_name][alpha] = {
                'coverage': coverage,
                'volume': float(volume),
                'wsc': wsc,
                'threshold': float(threshold),
            }

    elapsed = time.time() - t0
    results['meta']['elapsed'] = elapsed

    if verbose:
        print(f'  seed={seed} model={model_name} rmse={model_rmse:.4f} [{elapsed:.1f}s]')

    return results


# ══════════════════════════════════════════════════════════════════
#  Aggregation
# ══════════════════════════════════════════════════════════════════

def summarize(all_results):
    """
    Aggregate list of run_single outputs across seeds.

    :param all_results: list of dicts from run_single()
    :return: dict {model: {scorer: {alpha: {metric: {mean, std, se}}}}}
    """
    # Group by model
    by_model = {}
    for r in all_results:
        mn = r['model']
        if mn not in by_model:
            by_model[mn] = []
        by_model[mn].append(r)

    scorer_names = list(SCORER_FACTORIES.keys())
    alphas = all_results[0][scorer_names[0]].keys()
    alphas = [a for a in alphas if isinstance(a, float)]

    summary = {}
    for mn, runs in by_model.items():
        summary[mn] = {}

        # RMSE
        rmses = [r['model_rmse'] for r in runs]
        n_v = len(rmses)
        summary[mn]['model_rmse'] = {
            'mean': float(np.mean(rmses)),
            'std': float(np.std(rmses, ddof=1)) if n_v > 1 else 0.0,
            'se': float(np.std(rmses, ddof=1) / np.sqrt(n_v)) if n_v > 1 else 0.0,
        }

        for sn in scorer_names:
            summary[mn][sn] = {}
            for alpha in alphas:
                summary[mn][sn][alpha] = {}
                for metric in ['coverage', 'volume', 'wsc']:
                    vals = [r[sn][alpha][metric] for r in runs]
                    n_v = len(vals)
                    summary[mn][sn][alpha][metric] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals, ddof=1)) if n_v > 1 else 0.0,
                        'se': float(np.std(vals, ddof=1) / np.sqrt(n_v)) if n_v > 1 else 0.0,
                    }

    return summary


def print_summary(summary):
    """Pretty-print aggregated results."""
    model_names = list(summary.keys())
    scorer_names = [k for k in summary[model_names[0]]
                    if k not in ('model_rmse',)]

    alphas = sorted(summary[model_names[0]][scorer_names[0]].keys())
    fmt = lambda s: f"{s['mean']:.4f}±{s['se']:.4f}"

    for alpha in alphas:
        print(f"\n{'='*78}")
        print(f"  α = {alpha}  (target = {1-alpha:.2f})")
        print(f"{'='*78}")
        print(f"  {'Model':<8s} {'Scorer':<14s} {'Coverage':>16s} "
              f"{'Volume':>16s} {'WSC':>16s}")
        print(f"  {'-'*72}")

        for mn in model_names:
            for si, sn in enumerate(scorer_names):
                m = summary[mn][sn][alpha]
                label = mn if si == 0 else ''
                print(f"  {label:<8s} {sn:<14s} "
                      f"{fmt(m['coverage']):>16s} "
                      f"{fmt(m['volume']):>16s} "
                      f"{fmt(m['wsc']):>16s}")
            print()
