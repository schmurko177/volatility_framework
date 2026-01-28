"""QLIKE loss for volatility forecast evaluation."""

from __future__ import annotations

import numpy as np


def qlike(
    realized_var: np.ndarray | list[float],
    forecast_var: np.ndarray | list[float],
) -> np.ndarray:
    """Compute the QLIKE (quasi-likelihood) loss between realized and forecast variances.

    The QLIKE loss is a robust alternative to MSE for variance forecasts and
    is consistent for the conditional variance under the log-normal specification.
    For each pair (r, f) of realized and forecast variance, the loss is:

    .. math::

        \\text{QLIKE}(r, f) = \\frac{r}{f} - \\log\\left(\\frac{r}{f}\\right) - 1

    where :math:`r` is realized variance and :math:`f` is forecast variance.
    Both inputs must have the same shape; values should be positive.

    Parameters
    ----------
    realized_var :
        Array-like of realized variances (e.g. squared returns or realized
        variance estimates).
    forecast_var :
        Array-like of forecast variances, same shape as ``realized_var``.

    Returns
    -------
    numpy.ndarray
        Element-wise QLIKE loss values, same shape as the inputs.

    Raises
    ------
    ValueError
        If ``realized_var`` and ``forecast_var`` have incompatible shapes.
    """
    r = np.asarray(realized_var, dtype=float)
    f = np.asarray(forecast_var, dtype=float)

    if r.shape != f.shape:
        raise ValueError(
            f"realized_var and forecast_var must have the same shape; "
            f"got {r.shape} and {f.shape}."
        )

    ratio = r / f
    return ratio - np.log(ratio) - 1.0
