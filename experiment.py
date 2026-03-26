"""
Experiment runner for kernel conformal prediction.

Core design: everything is generic over scorers, models, alphas, seeds.
Adding a method = adding one entry to the scorer_factories dict.

Main entry points:
    run_single_seed  - One seed, all models × scorers × alphas
    run_experiment   - Multi-seed with aggregation
    print_summary    - Pretty-print results
"""

import numpy as np
import time

from scorers import conformal_quantile, MahalanobisScorer
from volume_estimator import mc_volume, ellipsoid_volume, estimate_volume
from metrics import compute_coverage, compute_wsc
from pipeline import generate_and_fit


def run_single_seed(dgp_fn, models, scorer_factories, alphas,
                    n=20_000, split=(0.5, 0.25, 0.25), seed=21,
                    volume_n_samples=100_000):
    """
    Run one seed: generate data, fit models, fit all scorers, evaluate at all alphas.

    :param dgp_fn: callable, dgp_fn(n, seed) -> X, Y, f_true
    :param models: dict {name: unfitted sklearn model}
    :param scorer_factories: dict {name: callable() -> scorer instance}
    :param alphas: list of float, miscoverage levels
    :param n: int, total sample size
    :param split: tuple (train, cal, test) proportions
    :param seed: int
    :param volume_n_samples: int, MC samples for volume

    :return: dict with structure:
        {
            'seed': int,
            'fitted_scorers': {model: {scorer: fitted_instance}},
            'auto_params': {model: {scorer: {param: value}}},
            'results': {model: {scorer: {alpha: {cov, vol, wsc, threshold}}}},
            'data': data dict (for plotting)
        }
    """
    # Stage 1: Generate data and fit models
    data = generate_and_fit(dgp_fn, models, n=n, split=split, seed=seed)

    fitted_scorers = {}   # model -> scorer_name -> fitted scorer
    auto_params = {}      # model -> scorer_name -> {lengthscale, gamma, ...}
    results = {}          # model -> scorer_name -> alpha -> metrics

    for model_name in data['residuals_cal']:
        rc = data['residuals_cal'][model_name]
        rt = data['residuals_test'][model_name]
        X_test = data['X_test']
        d = rc.shape[1] if rc.ndim > 1 else 1

        fitted_scorers[model_name] = {}
        auto_params[model_name] = {}
        results[model_name] = {}

        # Stage 2: Fit all scorers on calibration residuals
        for scorer_name, factory in scorer_factories.items():
            scorer = factory()
            # print(f'Seed n°{seed} — Fitting Scorer {scorer_name}.....', flush=True)
            scorer.fit(rc)
            # print(f'Seed n°{seed} — Fitted Scorer {scorer_name}', flush=True)
            fitted_scorers[model_name][scorer_name] = scorer

            # Log auto params if available
            params = {}
            if hasattr(scorer, 'lengthscale') and scorer.lengthscale is not None:
                params['lengthscale'] = float(scorer.lengthscale)
            if hasattr(scorer, 'gamma') and scorer.gamma is not None:
                params['gamma'] = float(scorer.gamma)
            auto_params[model_name][scorer_name] = params

            # Score calibration and test
            scores_cal = scorer.score(rc)
            # print(f'Seed n°{seed} — Scored Calibration for Scorer {scorer_name}', flush=True)
            scores_test = scorer.score(rt)
            # print(f'Seed n°{seed} — Scored Test for Scorer {scorer_name}', flush=True)
            # Stage 3: Evaluate at each alpha
            results[model_name][scorer_name] = {}

            for alpha in alphas:
                threshold = conformal_quantile(scores_cal, alpha)
                covered = scores_test <= threshold
                cov = float(covered.mean())
                wsc = compute_wsc(X_test, covered)

                # Volume estimation
                # Use exact formula for Mahalanobis, MC/importance for others
                if isinstance(scorer, MahalanobisScorer) and scorer.cov is not None:
                    vol = ellipsoid_volume(scorer.cov, threshold, d)
                else:
                    # For d>=4, need mahal as proposal — find it if available
                    mahal = fitted_scorers[model_name].get('Mahal')
                    mahal_thresh = None
                    if mahal is not None and 'Mahal' in results[model_name]:
                        mahal_thresh = results[model_name]['Mahal'].get(alpha, {}).get('threshold')
                    # print(f'Seed n°{seed} — Estimating volume for Scorer {scorer_name} and alpha = {alpha}', flush=True)
                    vol = estimate_volume(
                        scorer, threshold, rc,
                        mahal_scorer=mahal,
                        mahal_threshold=mahal_thresh,
                        n_samples=volume_n_samples,
                        seed=0,  # fixed for fair ratios
                    )

                results[model_name][scorer_name][alpha] = {
                    'coverage': cov,
                    'volume': vol,
                    'wsc': wsc,
                    'threshold': threshold,
                }

    return {
        'seed': seed,
        'fitted_scorers': fitted_scorers,
        'auto_params': auto_params,
        'results': results,
        'data': data,
    }


def run_experiment(dgp_fn, models, scorer_factories, alphas,
                   n_seeds=50, n=20_000, split=(0.5, 0.25, 0.25),
                   base_seed=21, volume_n_samples=100_000,
                   verbose=True):
    """
    Multi-seed experiment with aggregation.

    :param dgp_fn: callable
    :param models: dict {name: model}
    :param scorer_factories: dict {name: callable() -> scorer}
    :param alphas: list of float
    :param n_seeds: int
    :param n: int, sample size per seed
    :param split: tuple
    :param base_seed: int
    :param volume_n_samples: int
    :param verbose: bool

    :return: (all_runs, summary)
        all_runs: list of run_single_seed outputs
        summary: dict {model: {scorer: {alpha: {metric: {mean, std, se}}}}}
    """
    all_runs = []
    t0 = time.time()

    for i in range(n_seeds):
        seed = base_seed + i
        if verbose:
            elapsed = time.time() - t0
            # print(f"  Seed {i+1}/{n_seeds} (seed={seed}, elapsed={elapsed:.0f}s)", flush=True)

        run = run_single_seed(
            dgp_fn, models, scorer_factories, alphas,
            n=n, split=split, seed=seed,
            volume_n_samples=volume_n_samples,
        )
        all_runs.append(run)

    # Aggregate
    summary = _aggregate(all_runs, models, scorer_factories, alphas)

    if verbose:
        elapsed = time.time() - t0
        # print(f"  Done: {n_seeds} seeds in {elapsed:.0f}s")

    return all_runs, summary


def _aggregate(all_runs, models, scorer_factories, alphas):
    """
    Aggregate results across seeds: mean, std, se for each metric.

    :return: dict {model: {scorer: {alpha: {metric: {mean, std, se}}}}}
    """
    model_names = list(models.keys())
    scorer_names = list(scorer_factories.keys())
    metrics = ['coverage', 'volume', 'wsc']

    summary = {}

    for mn in model_names:
        summary[mn] = {}
        for sn in scorer_names:
            summary[mn][sn] = {}
            for alpha in alphas:
                summary[mn][sn][alpha] = {}
                for metric in metrics:
                    vals = [
                        run['results'][mn][sn][alpha][metric]
                        for run in all_runs
                    ]
                    n_vals = len(vals)
                    summary[mn][sn][alpha][metric] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals, ddof=1)) if n_vals > 1 else 0.0,
                        'se': float(np.std(vals, ddof=1) / np.sqrt(n_vals)) if n_vals > 1 else 0.0,
                    }

    # Also aggregate auto params
    for mn in model_names:
        for sn in scorer_names:
            param_keys = all_runs[0]['auto_params'][mn][sn].keys()
            for pk in param_keys:
                vals = [run['auto_params'][mn][sn][pk] for run in all_runs]
                summary[mn][sn][f'param_{pk}'] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }

    return summary


# ── Pretty Print ─────────────────────────────────────────────

def print_summary(summary, alphas=None):
    """
    Print multi-seed results as a table.

    :param summary: output of run_experiment
    :param alphas: list of alphas to display (default: all)
    """
    model_names = list(summary.keys())
    scorer_names = list(summary[model_names[0]].keys())
    # Filter out param_ keys from scorer_names
    scorer_names = [s for s in scorer_names if not s.startswith('param_')]

    if alphas is None:
        alphas = sorted(summary[model_names[0]][scorer_names[0]].keys())
        alphas = [a for a in alphas if isinstance(a, float)]

    fmt = lambda s: f"{s['mean']:.3f}±{s['se']:.3f}"

    for alpha in alphas:
        print(f"\n{'='*78}")
        print(f"  α = {alpha}")
        print(f"{'='*78}")
        print(f"  {'Model':<10s} {'Scorer':<14s} {'Coverage':>14s} "
              f"{'Volume':>14s} {'WSC':>14s}")
        print(f"  {'-'*66}")

        for mn in model_names:
            for si, sn in enumerate(scorer_names):
                m = summary[mn][sn][alpha]
                model_label = mn if si == 0 else ''
                print(f"  {model_label:<10s} {sn:<14s} "
                      f"{fmt(m['coverage']):>14s} "
                      f"{fmt(m['volume']):>14s} "
                      f"{fmt(m['wsc']):>14s}")
            print()


def print_volume_ratios(summary, reference='Mahal', alphas=None):
    """
    Print volume ratios relative to a reference scorer.

    :param summary: output of run_experiment
    :param reference: str, scorer name to use as denominator
    :param alphas: list of alphas to display
    """
    model_names = list(summary.keys())
    scorer_names = list(summary[model_names[0]].keys())
    scorer_names = [s for s in scorer_names if not s.startswith('param_')]

    if alphas is None:
        alphas = sorted(summary[model_names[0]][scorer_names[0]].keys())
        alphas = [a for a in alphas if isinstance(a, float)]

    others = [s for s in scorer_names if s != reference]

    print(f"\n{'='*60}")
    print(f"  Volume ratios (vs {reference})")
    print(f"{'='*60}")

    header = f"  {'Model':<10s} {'Scorer':<14s}"
    for alpha in alphas:
        header += f" {'α='+str(alpha):>10s}"
    print(header)
    print(f"  {'-'*56}")

    for mn in model_names:
        for sn in others:
            row = f"  {mn:<10s} {sn:<14s}"
            for alpha in alphas:
                ref_vol = summary[mn][reference][alpha]['volume']['mean']
                this_vol = summary[mn][sn][alpha]['volume']['mean']
                ratio = this_vol / ref_vol if ref_vol > 0 else float('inf')
                row += f" {ratio:>10.3f}"
            print(row)
        print()
