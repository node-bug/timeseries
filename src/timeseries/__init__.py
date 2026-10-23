"""Time series pattern matching and conditional forecasting.

Works on any Yahoo ticker, at 1-minute or daily resolution.  The two are not
interchangeable and are never mixed: :mod:`timeseries.timeframes` owns every constant
that depends on the bar interval, and a query is only ever scored against bars of its
own resolution.

1-minute is the default, so a caller that never mentions a resolution gets the
behaviour this package has always had.
"""

# Preload the threadsafe OpenMP runtime before any submodule can trigger a Numba
# parallel launch.  Numba resolves its threading layer lazily, on the first
# ``parallel=True`` call, so this has to happen early -- and it is best-effort: if the
# library is missing or will not load, this is a no-op and Numba picks its own layer.
# See :mod:`timeseries._threading` for why this is needed at all.
from . import _threading  # noqa: F401

__version__ = "0.2.0"