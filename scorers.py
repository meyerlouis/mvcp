"""
Core nonconformity scorers for kernel conformal prediction.
"""

import numpy as np
import logging
from scipy.spatial.distance import pdist


def _ensure_2d(x):
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def rbf_matrix(X, Y, lengthscale):
    """
    RBF (Gaussian) kernel Gram matrix.

    K_ij = exp(-||X_i - Y_j||² / (2ℓ²))

    :param X: (n, p) array
    :param Y: (m, p) array
    :param lengthscale: float, kernel bandwidth ℓ
    :return: (n, m) array
    """
    X, Y = _ensure_2d(X), _ensure_2d(Y)
    sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ Y.T + np.sum(Y**2, axis=1)
    return np.exp(-np.maximum(sq_dists, 0) / (2 * lengthscale**2))


# def median_lengthscale(eps, subsample=5000):
#     """
#     Median heuristic for RBF lengthscale: ℓ = median pairwise distance / 2.
#
#     The division by 2 is standard practice (Gretton et al., 2012) to ensure
#     the kernel resolves local structure rather than global spread.
#
#     :param eps: (T, p) array of calibration residuals
#     :param subsample: max points for pdist (statistical, not structural)
#     :return: float
#     """
#     eps = _ensure_2d(eps)
#     T = eps.shape[0]
#     if T > subsample:
#         idx = np.random.default_rng(42).choice(T, subsample, replace=False)
#         eps = eps[idx]
#     return float(np.median(pdist(eps, 'euclidean'))) / 2.0


def median_lengthscale(eps, subsample=5000):
    eps = _ensure_2d(eps)
    d = eps.shape[1]
    T = eps.shape[0]
    if T > subsample:
        idx = np.random.default_rng(24).choice(T, subsample, replace=False)
        eps = eps[idx]
    return float(np.median(pdist(eps, 'euclidean'))) * d / 2.0


def auto_gamma(Kc):
    """
    Automatic γ selection: 90th percentile of the eigenvalue spectrum
    of the centered Gram matrix.

    This sets γ so that ~10% of KPCA directions are retained at full
    weight (λ_j >> γ) while the remaining 90% are regularized.

    :param Kc: (T, T) centered Gram matrix
    :return: float
    """
    eigvals = np.linalg.eigvalsh(Kc)[::-1]
    tol = eigvals[0] * 1e-10
    active = eigvals[eigvals > tol]
    return float(np.percentile(active, 90))


def conformal_quantile(scores, alpha):
    """
    Conformal quantile: ⌈(1-α)(T+1)⌉ / T quantile of calibration scores.

    When this exceeds 1.0 (very small T or α), returns max(scores),
    giving infinite coverage (never reject). Correct finite-sample behavior.

    :param scores: (T,) array of calibration scores
    :param alpha: float in [0, 1]
    :return: float
    """
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    T = len(scores)
    level = min(np.ceil((1 - alpha) * (T + 1)) / T, 1.0)
    return float(np.quantile(scores, level))

class BonferroniScorer:
    """
    Coordinate-wise Bonferroni baseline.

    Score = max_j |ε_j - μ_j| / σ_j (standardized max-coordinate).
    Prediction regions are hyper-rectangles at level α/d per dimension.
    Uses standard deviation (not MAD) for consistency with literature.
    """

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, calibration_set):
        eps = _ensure_2d(calibration_set)
        self.mu = eps.mean(axis=0)
        sigma = eps.std(axis=0, ddof=1)
        self.sigma = np.where(sigma > 0, sigma, 1e-10)
        return self

    def score(self, eps_new):
        eps_new = _ensure_2d(eps_new)
        return np.max(np.abs(eps_new - self.mu) / self.sigma, axis=1)

    def __repr__(self):
        return "BonferroniScorer()"


class MahalanobisScorer:
    """
    Standard Mahalanobis distance scorer.

    Score = (ε - μ)' Σ⁻¹ (ε - μ).
    Prediction regions are ellipsoids. Equivalent to Xu et al. (2024)
    and ECM (Johnstone et al., 2021) with global covariance.
    """

    def __init__(self, regularization=1e-8):
        self.regularization = regularization
        self.mu = None
        self.cov = None
        self.cov_inv = None

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)
        p = eps_cal.shape[1]
        self.mu = eps_cal.mean(axis=0)
        self.cov = np.cov(eps_cal.T).reshape(p, p) + self.regularization * np.eye(p)
        self.cov_inv = np.linalg.inv(self.cov)
        return self

    def score(self, eps):
        eps = _ensure_2d(eps)
        d = eps - self.mu
        return np.sum(d @ self.cov_inv * d, axis=1)

    def __repr__(self):
        return f"MahalanobisScorer(reg={self.regularization})"


class KDEScorer:
    """
    Kernel Density Estimation scorer.

    Score = -log p̂(ε), where p̂ is a Gaussian KDE fitted on
    calibration residuals. Higher score = lower density = less conforming.

    Included as a baseline to compare against our kernel score.
    Our score contains a density-like term (MMD²) PLUS a KPCA correction.
    KDE captures only the density part. Dies at d≥4 due to curse of
    dimensionality, so only used for d=2 datasets.
    """

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.kde = None

    def fit(self, calibration_set):
        from scipy.stats import gaussian_kde

        eps_cal = _ensure_2d(calibration_set)

        if self.bandwidth is not None:
            self.kde = gaussian_kde(eps_cal.T, bw_method=self.bandwidth)
        else:
            self.kde = gaussian_kde(eps_cal.T)  # Scott's rule

        return self

    def score(self, eps_new):
        eps_new = _ensure_2d(eps_new)
        log_density = np.log(np.maximum(self.kde(eps_new.T), 1e-300))
        return -log_density

    def __repr__(self):
        bw = self.kde.factor if self.kde is not None else self.bandwidth
        return f"KDEScorer(bw={bw})"


class DensityScorer:
    """
    Kernel density (MMD²) scorer — our score's γ → ∞ limit.

    Score = k(ε,ε) - 2·mean_i[k(ε, ε_i)] + mean_{i,j}[k(ε_i, ε_j)]

    Equivalent to HPD-split with RBF kernel density estimation.
    Same lengthscale as KernelScorer for fair comparison: the ONLY
    difference is the absence of the KPCA correction term.
    """

    def __init__(self, lengthscale=None, auto_parameters=True):
        self.lengthscale = lengthscale
        self.auto_parameters = auto_parameters
        self.eps_cal = None
        self.grand_mean = None

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)

        if self.auto_parameters:
            self.lengthscale = median_lengthscale(eps_cal)

        K = rbf_matrix(eps_cal, eps_cal, self.lengthscale)
        self.grand_mean = K.mean()
        self.eps_cal = eps_cal
        return self

    def score(self, eps_new):
        eps_new = _ensure_2d(eps_new)
        Kv = rbf_matrix(eps_new, self.eps_cal, self.lengthscale)
        p_hat = Kv.mean(axis=1)
        return 1.0 - 2 * p_hat + self.grand_mean

    def __repr__(self):
        return f"DensityScorer(ℓ={self.lengthscale}, auto={self.auto_parameters})"


class KernelScorer:
    """
    Kernel nonconformity scorer (GP posterior variance).

    Score = k̃(ε,ε) - k̃*(ε)' (K̃ + γI)⁻¹ k̃*(ε)

    Equals the GP posterior variance with centered kernel k̃ and
    noise level γ (Proposition 5). Decomposes as MMD² minus KPCA
    correction (Proposition 3). Recovers regularized Mahalanobis
    distance for linear kernel (Proposition 1).
    """

    def __init__(self, lengthscale=None, gamma=None, auto_parameters=True):
        self.auto_parameters = auto_parameters

        if not auto_parameters:
            if lengthscale is None:
                raise ValueError(
                    "When auto_parameters=False, lengthscale must be provided."
                )

        self.lengthscale = lengthscale
        self.gamma = gamma

        self.eps_cal = None
        self.col_mean = None
        self.grand_mean = None
        self.A = None

    def fit(self, calibration_set):
        eps_cal = _ensure_2d(calibration_set)
        T = eps_cal.shape[0]

        if self.auto_parameters:
            self.lengthscale = median_lengthscale(eps_cal)

        K = rbf_matrix(eps_cal, eps_cal, self.lengthscale)

        self.col_mean = K.mean(axis=0)
        self.grand_mean = K.mean()

        Kc = K - self.col_mean[None, :] - self.col_mean[:, None] + self.grand_mean

        if self.gamma == None:
            self.gamma = auto_gamma(Kc)

        self.A = np.linalg.solve(Kc + self.gamma * np.eye(T), np.eye(T))
        self.eps_cal = eps_cal

        return self

    def score(self, eps):
        eps = _ensure_2d(eps)

        Kv = rbf_matrix(eps, self.eps_cal, self.lengthscale)
        kcm = Kv.mean(axis=1)

        Kv_c = Kv - kcm[:, None] - self.col_mean[None, :] + self.grand_mean

        # Centered self-kernel: k̃(ε, ε) = k(ε,ε) - 2E[k(ε,·)] + E[k(·,·)]
        # For RBF: k(ε, ε) = 1
        self_c = 1.0 - 2 * kcm + self.grand_mean

        # GP posterior variance = self_c - k̃*' (K̃ + γI)⁻¹ k̃*
        return self_c - np.sum(Kv_c @ self.A * Kv_c, axis=1)

    def __repr__(self):
        return f"KernelScorer(ℓ={self.lengthscale}, γ={self.gamma}, auto={self.auto_parameters})"
