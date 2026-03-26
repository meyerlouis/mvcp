"""
Data generation and model fitting pipeline.

One function: generate_and_fit. Takes a DGP and models, returns
data splits, fitted models, residuals, predictions.
"""

import numpy as np
from sklearn.base import clone


def generate_and_fit(dgp_fn, models, n=20_000, split=(0.4, 0.3, 0.3), seed=42):
    """
    Generate data from a DGP, split, fit models, compute residuals.

    :param dgp_fn: callable, dgp_fn(n, seed) -> X (n, d_x), Y (n, d_y), f_true (n, d_y)
    :param models: dict {name: unfitted sklearn-compatible model}
    :param n: int, total sample size
    :param split: tuple of 3 floats (train, cal, test) proportions
    :param seed: int
    :return: dict with all data splits, fitted models, residuals, predictions
    """
    X, Y, f_true = dgp_fn(n, seed=seed)

    # Ensure 2D
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    if f_true.ndim == 1: f_true = f_true[:, None]

    # Split
    props = np.array(split, dtype=float)
    props /= props.sum()
    cuts = (np.cumsum(props) * n).astype(int)

    idx = np.random.RandomState(seed).permutation(n)
    i_tr  = idx[:cuts[0]]
    i_cal = idx[cuts[0]:cuts[1]]
    i_te  = idx[cuts[1]:]

    X_tr,  Y_tr              = X[i_tr],  Y[i_tr]
    X_cal, Y_cal, f_cal      = X[i_cal], Y[i_cal], f_true[i_cal]
    X_te,  Y_te,  f_te       = X[i_te],  Y[i_te],  f_true[i_te]

    # Fit models and compute residuals
    fitted = {}
    resid_cal, resid_test = {}, {}
    pred_cal, pred_test = {}, {}

    for name, model in models.items():
        m = clone(model)
        m.fit(X_tr, Y_tr.ravel() if Y_tr.shape[1] == 1 else Y_tr)
        fitted[name] = m

        pc = m.predict(X_cal)
        pt = m.predict(X_te)
        if pc.ndim == 1: pc = pc[:, None]
        if pt.ndim == 1: pt = pt[:, None]

        pred_cal[name]  = pc
        pred_test[name] = pt
        resid_cal[name]  = Y_cal - pc
        resid_test[name] = Y_te  - pt

    return {
        'X_train': X_tr, 'Y_train': Y_tr,
        'X_cal': X_cal, 'Y_cal': Y_cal,
        'X_test': X_te, 'Y_test': Y_te,
        'f_true_cal': f_cal, 'f_true_test': f_te,
        'fitted_models': fitted,
        'residuals_cal': resid_cal, 'residuals_test': resid_test,
        'predictions_cal': pred_cal, 'predictions_test': pred_test,
    }