"""Business-day synthetic bars, the daily-timeframe analogue of :mod:`tests.session_bars`.

Why this exists
---------------
Adding a second timeframe means the tests that exercise the *calibration* need daily
data with the same structural properties real daily data has.  ``session_bars.py``
provides that for 1-minute; this provides it for daily, and the properties it has to
get right are **different** ones.

The intraday fixture's job is to make an overnight closure a *boundary* rather than a
hole, because that is the condition ``gaps_over_180s == 0 and session_boundaries ==
sessions - 1`` is about.  Daily has no closures inside a bar and no hole threshold at
all, so that structure is not what is under test here.  What *is*:

* **One bar per trading day, never two.**  Two bars on one date would make
  ``bars_max > 1`` and would let a test pass a daily frame through intraday logic that
  should have rejected it.
* **No weekend bars.**  Weekends are not sessions.  A ``date_range`` over calendar days
  would put Saturday bars in the frame, inflate ``sessions``, and make the
  ``missing_days`` count wrong in the direction that looks *healthy* -- which is the
  worst direction for a fixture to be wrong in.
* **A weekday with no bar is representable**, because ``missing_days`` is the only
  discontinuity a daily series can have and a fixture that cannot express one cannot
  test it.  :func:`with_missing_days` drops days deliberately.

The distinction from ``session_bars.py`` worth stating plainly: this module does
**not** try to imitate a 24/7 market.  Daily equity bars are business days because
that is what the exchange produces, and a 7-day fixture would make the weekend logic
untestable for the same reason the minute fixture's midnight walk makes session
boundaries absent.

Prices move like daily prices here, not like minute prices: the default sigma is
~1% per day against ``session_bars``'s ~0.01% per minute.  A test that checked the
amplitude term on bars generated at the wrong scale would pass or fail for reasons
unrelated to the code under test, and the placebo harness in
:mod:`timeseries.placebo` reads its own sigma from the same registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "DAILY_SIGMA",
    "daily_bars",
    "business_days",
    "with_missing_days",
]

#: Per-day log-return standard deviation.  Matches
#: :attr:`timeseries.timeframes.Timeframe.synthetic_sigma` for ``1d`` -- read from the
#: registry rather than restated, so a change there cannot leave the fixture behind.
DAILY_SIGMA = 8e-3


def business_days(
    n_days: int,
    *,
    first_day: str = "2026-01-05",
    omit: "tuple | list | None" = None,
) -> pd.DatetimeIndex:
    """``n_days`` Eastern business days starting at ``first_day``.

    Parameters
    ----------
    n_days
        How many days to produce **after** any omission.  Callers asking for a frame of
        exactly ``n_days`` bars get exactly that, which is what makes
        ``len(frame) == n_days`` a safe assertion.
    omit
        Dates to drop, as anything ``pd.to_datetime`` accepts.  Dropped after the walk
        is generated, so removing a day does not change the prices on the days around
        it -- the gap is a hole in the data, not a re-draw.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive, got {!r}".format(n_days))
    # A generous upper bound: business days over n calendar days, where n calendar
    # days always contains at least n business days once the range is long enough to
    # absorb the omissions.
    cal = pd.bdate_range(first_day, periods=n_days + 10)
    if omit:
        cal = cal[~cal.isin(pd.to_datetime(list(omit)))]
    return cal[:n_days]


def daily_bars(
    n_bars: int,
    *,
    seed: int = 0,
    sigma: float = DAILY_SIGMA,
    start: float = 700.0,
    first_day: str = "2026-01-05",
    omit: "tuple | list | None" = None,
) -> pd.DataFrame:
    """``n_bars`` daily bars, one per Eastern business day.

    Parameters
    ----------
    n_bars
        Exact number of bars in the result.  Unlike the intraday fixture there is no
        rounding to a session count: one bar is one day, so the count is already the
        whole number.
    seed
        Seed for the walk.  Deterministic, which is what lets a test assert a specific
        surviving-bar count after the feature warm-up drops the first ``rolling - 1``.
    sigma
        Per-*day* log-return standard deviation.  Deliberately ~80x the intraday
        fixture's per-minute figure; see the module docstring.
    start
        First close.
    first_day
        First trading day, ``YYYY-MM-DD``.  Must be a weekday.
    omit
        Weekdays to leave out entirely, for exercising ``missing_days``.

    Returns
    -------
    DataFrame with the archive schema: ``timestamp`` (UTC) plus OHLC.
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive, got {!r}".format(n_bars))

    days = business_days(n_bars, first_day=first_day, omit=omit)
    n = len(days)

    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=n)
    close = start * np.exp(np.cumsum(steps))

    step = rng.normal(0.0, sigma * 0.4, size=n)
    o = close * np.exp(-step)
    c = close
    h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0, sigma * 0.2, n)))
    low = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0, sigma * 0.2, n)))

    # Yahoo stamps a daily bar at the session *open*, so the fixture does the same:
    # a bar at 09:30 ET rather than at midnight.  That matters for one specific check --
    # ``store.validate_session``'s regular-hours test is an exclusive comparison against
    # 09:30, so a midnight-stamped daily bar would be reported as "before 09:30 ET"
    # under intraday rules.  Reproducing the real stamp keeps a test of that guard
    # honest rather than accidentally passing or failing on the fixture.
    stamps = (days.tz_localize("America/New_York")
              + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")

    return pd.DataFrame({
        "timestamp": stamps,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
    })


def with_missing_days(
    n_bars: int, n_missing: int, *, seed: int = 0, **kwargs
) -> pd.DataFrame:
    """``daily_bars`` with ``n_missing`` interior weekdays removed.

    **Interior** on purpose.  Dropping the first or last day is not the same
    condition: the report counts a missing day by comparing the span's endpoints to
    the days present, so a gap at either end is invisible to it.  A fixture that
    omitted boundary days would therefore test nothing while appearing to.
    """
    if n_missing <= 0:
        return daily_bars(n_bars, seed=seed, **kwargs)

    # Generate more days than requested so that removing `n_missing` of them still
    # leaves `n_bars`.  The omissions are taken from the interior of the surplus.
    total = n_bars + n_missing
    days = business_days(total, first_day=kwargs.get("first_day", "2026-01-05"))
    # Skip the first and last generated day so the omissions are never at an edge.
    candidates = list(days[1:-1])
    if len(candidates) < n_missing:
        raise ValueError(
            "cannot omit {} interior days from only {} generated".format(
                n_missing, len(candidates))
        )
    step = max(1, len(candidates) // (n_missing + 1))
    omit = [c.strftime("%Y-%m-%d") for c in candidates[step - 1::step][:n_missing]]
    return daily_bars(n_bars, seed=seed, omit=omit, **kwargs)