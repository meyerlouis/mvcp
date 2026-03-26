"""
Volume estimation for prediction regions.

Strategies:
    mc_volume          - Bounding-box MC, reliable for d ≤ 3
    ellipsoid_volume   - Exact closed-form for Mahalanobis ellipsoids
    importance_volume  - Uniform sampling from expanded ellipsoid, d ≥ 4
    estimate_volume    - Dispatcher
"""

import numpy as np
from scipy.special import gammaln
import warnings


def mc_volume(scorer, threshold, reference_points, n_samples=100_000, pad=0.01, seed=0):
    """
    Estimate volume of {ε : scorer.score(ε) ≤ threshold} via bounding-box MC.

    Fixed seed ensures same MC samples across scorers → low-noise volume ratios.
    Only reliable for d ≤ 3.

    :param scorer: object with .score(eps) -> (n,) array
    :param threshold: float, conformal threshold
    :param reference_points: (T, p) array (for bounding box)
    :param n_samples: int
    :param pad: float, fraction of range to pad
    :param seed: int
    :return: float
    """
    ref = np.asarray(reference_points)
    if ref.ndim == 1:
        ref = ref[:, None]

    lo = ref.min(axis=0)
    hi = ref.max(axis=0)
    margin = (hi - lo) * pad
    lo -= margin
    hi += margin

    rng = np.random.default_rng(seed)
    samples = rng.uniform(lo, hi, size=(n_samples, ref.shape[1]))

    # Batch scoring to avoid OOM (e.g. 500k × 15k RBF matrix = 56GB)
    BATCH = 10_000
    n_inside = 0
    for i in range(0, n_samples, BATCH):
        batch = samples[i:i+BATCH]
        n_inside += int(np.sum(scorer.score(batch) <= threshold))

    bbox_volume = np.prod(hi - lo)
    return float(bbox_volume * n_inside / n_samples)


def ellipsoid_volume(cov, threshold, d):
    """
    Exact volume of {ε : (ε-μ)'Σ⁻¹(ε-μ) ≤ threshold}.

    :param cov: (d, d) covariance matrix Σ
    :param threshold: float
    :param d: int, dimension
    :return: float
    """
    log_ball = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1) + (d / 2) * np.log(threshold)
    sign, log_det = np.linalg.slogdet(cov)
    return float(np.exp(log_ball + 0.5 * log_det))


def _sample_uniform_ellipsoid(mu, cov, threshold, n_samples, rng):
    """
    Sample uniformly from {ε : (ε-μ)'Σ⁻¹(ε-μ) ≤ threshold}.

    :param mu: (d,) center
    :param cov: (d, d) covariance matrix
    :param threshold: float
    :param n_samples: int
    :param rng: numpy Generator
    :return: (n_samples, d) array
    """
    d = len(mu)
    L = np.linalg.cholesky(cov)

    # Uniform direction on unit sphere
    z = rng.standard_normal((n_samples, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    # Uniform radius in d-ball: r ~ U[0,1]^{1/d}
    r = rng.uniform(0, 1, size=(n_samples, 1)) ** (1.0 / d)

    return mu + np.sqrt(threshold) * (r * z) @ L.T


def importance_volume(scorer, threshold, mahal_scorer, mahal_threshold,
                      n_samples=200_000, expansion=1.5, seed=0):
    """
    Estimate volume of kernel region by uniform sampling from an
    expanded Mahalanobis ellipsoid.

    Volume = V_expanded × (fraction of uniform samples inside kernel region)

    :param scorer: object with .score() method
    :param threshold: float, kernel conformal threshold
    :param mahal_scorer: fitted MahalanobisScorer (provides μ, Σ)
    :param mahal_threshold: float, Mahalanobis conformal threshold
    :param n_samples: int
    :param expansion: float, factor to expand proposal (>1)
    :param seed: int
    :return: float
    """
    rng = np.random.default_rng(seed)

    mu = mahal_scorer.mu
    cov = mahal_scorer.cov
    d = len(mu)

    expanded_cov = expansion**2 * cov
    expanded_threshold = mahal_threshold * expansion**2

    samples = _sample_uniform_ellipsoid(mu, expanded_cov, expanded_threshold, n_samples, rng)

    # Batch scoring to avoid OOM
    BATCH = 10_000
    n_in_kernel = 0
    for i in range(0, n_samples, BATCH):
        batch = samples[i:i+BATCH]
        n_in_kernel += int(np.sum(scorer.score(batch) <= threshold))

    rate = n_in_kernel / n_samples
    if rate < 1e-4:
        warnings.warn(
            f"importance_volume: acceptance rate {rate:.2e} very low. "
            f"Consider increasing expansion (currently {expansion})."
        )

    proposal_volume = ellipsoid_volume(expanded_cov, expanded_threshold, d)
    return float(proposal_volume * rate)


def estimate_volume(scorer, threshold, reference_points,
                    mahal_scorer=None, mahal_threshold=None,
                    n_samples=200_000, seed=0):
    """
    Dispatch: mc_volume for d ≤ 3, importance_volume for d ≥ 4.

    For MahalanobisScorer at any d, prefer ellipsoid_volume() directly.

    :param scorer: fitted scorer
    :param threshold: float
    :param reference_points: (T, p) array
    :param mahal_scorer: fitted MahalanobisScorer (required for d ≥ 4)
    :param mahal_threshold: float (required for d ≥ 4)
    :param n_samples: int
    :param seed: int
    :return: float
    """
    ref = np.asarray(reference_points)
    if ref.ndim == 1:
        ref = ref[:, None]
    d = ref.shape[1]

    if d <= 4:
        return mc_volume(scorer, threshold, ref, n_samples=n_samples, seed=seed)
    else:
        if mahal_scorer is None or mahal_threshold is None:
            raise ValueError("Need fitted MahalanobisScorer for importance sampling at d ≥ 4")
        return importance_volume(scorer, threshold, mahal_scorer, mahal_threshold,
                                n_samples=n_samples, seed=seed)
