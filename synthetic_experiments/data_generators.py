"""
Data generating processes for synthetic experiments.
"""

import numpy as np
import logging


def generate_multimodal_1d(n, means, sigmas, probs=None, center=True, seed=None):
    """
    1D mixture of Gaussians.

    :param n: int, sample size
    :param means: array of component means
    :param sigmas: array of component std devs
    :param probs: array of mixture probabilities (default: uniform)
    :param center: bool, subtract sample mean
    :param seed: int

    :return samples: (n,) array
    :return modes: (n,) array of mode assignments
    """
    logging.info('Generating multimodal data')

    rng = np.random.default_rng(seed)
    mus, sigmas = np.asarray(means), np.asarray(sigmas)

    if probs is None:
        probs = np.ones(len(mus)) / len(mus)
    else:
        probs = np.asarray(probs, dtype=float)
        probs /= probs.sum()

    modes = rng.choice(len(mus), size=n, p=probs)
    samples = mus[modes] + sigmas[modes] * rng.standard_normal(n)

    if center:
        samples -= samples.mean()

    return samples, modes


def generate_dgp(n, seed=42):
    """
    2D response with nonlinear mean function and heteroscedastic noise.

    :param n: int, sample size
    :param seed: int

    :return X: (n,) array of predictors
    :return Y: (n, 2) array of responses
    :return f_true: (n, 2) array of true conditional means
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, n)

    f1 = 3 * X**2 - 1.5 * X + np.sin(4 * np.pi * X)
    f2 = 2 * X**3 - X + 0.5 * np.cos(3 * np.pi * X)
    f_true = np.column_stack([f1, f2])

    z1 = rng.normal(0, 0.3, n)
    z2 = rng.normal(0, 0.1, n)
    eps = np.column_stack([z1, z2 + 0.5 * z1**2 - 0.5 * 0.3**2])
    eps -= eps.mean(axis=0)

    Y = f_true + eps
    return X, Y, f_true
