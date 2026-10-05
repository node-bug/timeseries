"""Time series pattern matching and conditional forecasting.

Works on any Yahoo ticker, at 1-minute or daily resolution.  The two are not
interchangeable and are never mixed: :mod:`timeseries.timeframes` owns every constant
that depends on the bar interval, and a query is only ever scored against bars of its
own resolution.

1-minute is the default, so a caller that never mentions a resolution gets the
behaviour this package has always had.
"""

__version__ = "0.2.0"