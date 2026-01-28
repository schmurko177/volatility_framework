"""Base abstractions for volatility forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence


class BaseVolatilityModel(ABC):
    """Abstract base class for volatility forecasting models.

    Subclasses are expected to implement model-specific fitting and
    forecasting logic while adhering to this minimal interface. The
    typical workflow is::

        model = ConcreteVolatilityModel(params={...})
        model.fit(returns)
        forecasts = model.predict(h=10)

    Implementations should treat ``returns`` as a sequence of numerical
    returns (e.g. daily log-returns) and ``predict`` as returning a
    sequence of volatility forecasts for the next ``h`` periods.
    """

    #: Human-readable model identifier (e.g. "GARCH(1,1)").
    name: str

    #: Dictionary of model hyperparameters and configuration.
    params: Dict[str, Any]

    def __init__(self, name: str, params: Dict[str, Any] | None = None) -> None:
        """Initialize a volatility model.

        Parameters
        ----------
        name:
            Human-readable name for the model instance.
        params:
            Optional dictionary of model parameters or configuration
            values. Implementations may further validate or extend this
            mapping during ``fit``.
        """
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit(self, returns: Sequence[float]) -> "BaseVolatilityModel":
        """Fit the model on a sequence of returns.

        Parameters
        ----------
        returns:
            Historical return series used to estimate model parameters.

        Returns
        -------
        BaseVolatilityModel
            The fitted model instance (``self``), to support fluent-style
            usage such as ``model.fit(returns).predict(h=1)``.
        """

    @abstractmethod
    def predict(self, h: int = 1) -> Sequence[float]:
        """Generate volatility forecasts for the next ``h`` periods.

        Parameters
        ----------
        h:
            Forecast horizon, i.e. number of future periods to predict.

        Returns
        -------
        Sequence[float]
            Volatility forecasts for each of the next ``h`` periods.
        """

