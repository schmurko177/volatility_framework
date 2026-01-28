"""Exponentially weighted moving average (EWMA) volatility model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseVolatilityModel


class EWMAVolatilityModel(BaseVolatilityModel):
    """EWMA-based volatility forecasting model.

    The model assumes that the conditional variance follows an
    exponentially weighted moving average recursion of the form

    .. math::

        \\sigma_t^2 = \\lambda \\sigma_{t-1}^2 + (1 - \\lambda) r_t^2,

    where :math:`r_t` denotes the return at time :math:`t` and
    :math:`\\lambda \\in (0, 1)` is the decay parameter controlling the
    memory of the process. Volatility forecasts are obtained as the
    square root of the recursively estimated variance.
    """

    def __init__(self, lambda_: float = 0.94) -> None:
        """Create a new EWMA volatility model.

        Parameters
        ----------
        lambda_:
            Decay parameter :math:`\\lambda` in the EWMA recursion. Values
            close to 1 give more weight to older observations.
        """
        if not 0.0 < lambda_ < 1.0:
            raise ValueError("lambda_ must be in the open interval (0, 1).")

        self.lambda_ = float(lambda_)
        super().__init__(name="EWMA", params={"lambda": self.lambda_})

        # Array of fitted volatilities (standard deviations).
        self.volatility_: np.ndarray | None = None

    def fit(self, returns: Sequence[float]) -> "EWMAVolatilityModel":
        """Fit the EWMA model to a sequence of returns.

        Parameters
        ----------
        returns:
            One-dimensional array-like of returns (e.g. log-returns).

        Returns
        -------
        EWMAVolatilityModel
            The fitted model instance.
        """
        r = np.asarray(returns, dtype=float).ravel()
        if r.size == 0:
            raise ValueError("returns must contain at least one observation.")

        lam = self.lambda_
        one_minus_lam = 1.0 - lam

        var = np.empty_like(r)
        # Initialize variance with the first squared return.
        var[0] = r[0] ** 2
        for t in range(1, r.size):
            var[t] = lam * var[t - 1] + one_minus_lam * r[t] ** 2

        self.volatility_ = np.sqrt(var)
        return self

    def predict(self, h: int = 1) -> np.ndarray:
        """Forecast future volatility over a given horizon.

        The EWMA model implies a flat volatility term structure, so each
        of the next ``h`` periods shares the same forecast equal to the
        last fitted volatility.

        Parameters
        ----------
        h:
            Forecast horizon (number of periods to predict). Must be a
            positive integer.

        Returns
        -------
        numpy.ndarray
            Array of length ``h`` with the volatility forecast for each
            future period.

        Raises
        ------
        ValueError
            If ``h`` is not a positive integer or if the model has not
            been fitted prior to calling this method.
        """
        if h <= 0:
            raise ValueError("h must be a positive integer.")

        if self.volatility_ is None or self.volatility_.size == 0:
            raise ValueError("Model must be fitted before calling predict().")

        last_vol = float(self.volatility_[-1])
        return np.full(shape=int(h), fill_value=last_vol, dtype=float)

