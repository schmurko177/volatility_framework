Volatility Framework
====================

Volatility Framework is a research-oriented Python framework for building,
evaluating, and comparing volatility forecasting models. It is designed to
support systematic experimentation, from raw data ingestion through model
development and rigorous performance analysis.

Planned components:
- Data processing pipeline for raw and cleaned datasets
- Model implementations for volatility forecasting
- Evaluation metrics for forecast quality and risk assessment
- Jupyter notebooks for exploratory analysis and experiment reports

Experiments and research workflows are demonstrated primarily through
Jupyter notebooks stored in the `notebooks` directory.

## Volatility Framework (volaframe)

This framework provides baseline volatility forecasting models implemented
with a unified, object-oriented interface for research and comparison.

### Implemented models
- EWMA volatility (RiskMetrics-style)
- Rolling-window variance baseline

### Model architecture

Volatility models are implemented as Python classes. Each model encapsulates
its own estimation logic and forecasting behavior while adhering to a common
interface. New volatility models can be added by defining a new class that
inherits from the base volatility model.

### Creating volatility model instances

A concrete volatility model is created by instantiating the corresponding
class and passing model-specific hyperparameters to its constructor.

For example, an EWMA volatility model is created as:

"
from volaframe.models import EWMAVolatilityModel
ewma = EWMAVolatilityModel(lambda_=0.94)
"

Here, EWMAVolatilityModel is the model class and lambda_ is a keyword
argument controlling the exponential decay factor of the EWMA process.

A rolling-window variance model is created analogously:

"
from volaframe.models import RollingVarianceModel
rolling = RollingVarianceModel(window=21)
"

In this case, window is a keyword argument specifying the length of the
rolling estimation window.

Each instantiated object represents a specific volatility model with fixed
hyperparameters.

Interface
All volatility model classes follow a common interface:

fit(returns) — estimate model parameters from a sequence of returns

predict(h) — generate volatility forecasts for a given horizon

volatility_ — in-sample volatility estimates after fitting

Example usage:

"
import numpy as np
from volaframe.models import EWMAVolatilityModel, RollingVarianceModel
from volaframe.metrics import qlike

# Example return series
returns = np.random.normal(0, 0.01, size=500)

# EWMA model
ewma = EWMAVolatilityModel(lambda_=0.94)
ewma.fit(returns)

# Rolling variance model
rolling = RollingVarianceModel(window=21)
rolling.fit(returns)

# Forecasts
ewma_forecast = ewma.predict(h=5)
rolling_forecast = rolling.predict(h=5)

# QLIKE evaluation (in-sample variance)
realized_var = returns**2
qlike_ewma = qlike(realized_var[-len(ewma.volatility_):], ewma.volatility_**2).mean()
qlike_rolling = qlike(realized_var[-len(rolling.volatility_):], rolling.volatility_**2).mean()
Evaluation
QLIKE loss for variance forecast evaluation

Intended use
The framework is designed to be imported into Jupyter notebooks and used for
empirical experiments on financial time series (e.g., AAPL, MSFT, S&P 500),
enabling transparent comparison of different volatility forecasting models.