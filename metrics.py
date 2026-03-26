"""
Evaluation metrics for conformal prediction.

Pure functions, no dependencies on scorers or pipeline.
"""

import numpy as np


def compute_coverage(test_scores, threshold):
    """
    Marginal coverage: fraction of test points inside the prediction region.

    :param test_scores: (n,) array of nonconformity scores on test set
    :param threshold: float, conformal threshold
    :return: float
    """
    return float((test_scores <= threshold).mean())


# def compute_wsc(X_test, covered, n_slabs=200, min_slab_size=30, seed=42):
#     """
#     Worst-slab coverage.
#
#     Searches random linear projections of X for the slab with worst
#     conditional coverage. Higher (closer to 1-α) is better.
#
#     :param X_test: (n, d) array of test features
#     :param covered: (n,) boolean array, True if Y_test in prediction region
#     :param n_slabs: int, number of random directions to search
#     :param min_slab_size: int, minimum points in a slab to count
#     :param seed: int
#     :return: float, worst coverage found across all slabs
#     """
#     if X_test.ndim == 1:
#         X_test = X_test[:, None]
#
#     n, d = X_test.shape
#     rng = np.random.default_rng(seed)
#     worst = 1.0
#
#     for _ in range(n_slabs):
#         # Random projection direction
#         w = rng.standard_normal(d)
#         w /= np.linalg.norm(w)
#         proj = X_test @ w
#
#         # Search slabs of width ~20% of the range
#         lo, hi = proj.min(), proj.max()
#         width = (hi - lo) * 0.2
#
#         for q in np.linspace(0.0, 0.8, 20):
#             cut_lo = np.quantile(proj, q)
#             cut_hi = cut_lo + width
#             mask = (proj >= cut_lo) & (proj <= cut_hi)
#
#             if mask.sum() >= min_slab_size:
#                 worst = min(worst, covered[mask].mean())
#
#     return float(worst)


def compute_wsc(X_test, covered, n_slabs=200, min_slab_size=30, seed=42):
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    n, d = X_test.shape
    rng = np.random.default_rng(seed)
    worst = 1.0

    for _ in range(n_slabs):
        w = rng.standard_normal(d)
        w /= np.linalg.norm(w)
        proj = X_test @ w

        for q in np.linspace(0.0, 0.8, 20):
            mask = (proj >= np.quantile(proj, q)) & (proj <= np.quantile(proj, q + 0.2))

            if mask.sum() >= min_slab_size:
                worst = min(worst, covered[mask].mean())

    return float(worst)