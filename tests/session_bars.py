"""Session-structured synthetic bars, shared by the tests that used the QQQ archive.

Why this exists
---------------
Several tests used ``data/qqq_1min_20260831_20260930.csv`` as their fixture.  That
worked, but it coupled the test suite to a 892 KB binary blob: the tests silently
skipped when it was absent, the bar count was hard-coded from it (``N = 8049``), and
``data/`` had to be fetched before the suite would run at all.  This module produces
the same *shape* of data — real trading sessions with real overnight closures — from a
seed, so the suite is self-contained and every bar count is a parameter rather than a
number copied out of a file.

The distinction that matters is the session structure, not the price path.  §BF's
guard asserts

    gaps_over_180s == 0        and        session_boundaries == sessions - 1

i.e. that an overnight closure is counted as a *boundary* and never as a *hole*.  A
naive ``date_range(freq="1min")`` walk has no closures at all, so it cannot exercise
that guard: it would pass a broken filter just as happily as a correct one.  This
generator therefore emits each session as 390 consecutive minutes and then jumps to
the next session's open, which is exactly the condition the guard is about.

Timestamps are anchored to real US/Eastern session times (09:30–16:00 ET) because
:func:`timeseries.features.quality_report` buckets sessions in Eastern time; a
generator working in UTC would misclassify every boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "BARS_PER_SESSION",
    "SESSION_OPEN_ET",
    "SESSION_CLOSE_ET",
    "session_bars",
    "session_count_for",
]

#: Bars in one regular US equity session (09:30–16:00 ET, inclusive of the 16:00 bar).
BARS_PER_SESSION = 390

#: Session window in US/Eastern local time.  ``Timedelta`` requires hh:mm:ss, so the
#: offset is spelled out rather than read as "09:30".
SESSION_OPEN_ET = "09:30:00"
SESSION_CLOSE_ET = "16:00:00"

#: ``timeseries.store.validate_session`` also uses 390; the two are independent
#: constants that happen to agree, and a change to either should be a deliberate act.


def _session_starts(n_sessions: int, first_day: str = "2026-01-05") -> np.ndarray:
    """UTC timestamps of the opening bar of each of ``n_sessions`` trading days.

    Weekends are skipped, matching a real calendar, so a generator asked for 20
    sessions spans 4 weeks rather than 20 consecutive days.  The day arithmetic is in
    Eastern time because that is where "a trading day" is defined.
    """
    day = pd.Timestamp(first_day, tz="America/New_York")
    starts = []
    while len(starts) < n_sessions:
        # 5 == Saturday, 6 == Sunday.
        if day.dayofweek < 5:
            starts.append(day.normalize() + pd.Timedelta(SESSION_OPEN_ET))
        day += pd.Timedelta(days=1)
    return np.array([pd.Timestamp(s).tz_convert("UTC") for s in starts], dtype="object")


def session_bars(
    n_bars: int,
    *,
    seed: int = 0,
    sigma: float = 1e-4,
    start: float = 700.0,
    first_day: str = "2026-01-05",
) -> pd.DataFrame:
    """Bars laid out as ``ceil(n_bars / 390)`` real trading sessions.

    Parameters
    ----------
    n_bars
        Target number of bars.  The result has at least this many; it is rounded up to
        a whole session count so no session is ever truncated (a truncated one would
        show up as a short day in the quality report and break ``bars_min == bars_max``
        style assumptions in callers).
    seed
        Seed for the walk.  The same seed always yields the same bars,
        which is what lets a test assert a specific ``n_bars`` after feature cleaning.
    sigma
        Per-bar log-return standard deviation.
    start
        First close.
    first_day
        First trading day, as ``YYYY-MM-DD`` in Eastern time.  Must be a weekday for
        the session count to line up exactly with the number of sessions produced.

    Returns
    -------
    DataFrame with the same schema as a real archive: ``timestamp`` (UTC) plus OHLC.
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive, got {!r}".format(n_bars))

    n_sessions = -(-int(n_bars) // BARS_PER_SESSION)  # ceil
    opens = _session_starts(n_sessions, first_day)

    # One continuous log-return series is drawn across the *whole* span, not per
    # session.  Resampling at each session boundary would make the return series
    # artificially jumpy there and give the overnight closure a visible signature in
    # the features, which is not what real data looks like and would let a
    # shape-matching test pass for the wrong reason.
    total = n_sessions * BARS_PER_SESSION
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=total)
    close = start * np.exp(np.cumsum(steps))

    # Timestamps: 390 consecutive minutes per session, then a jump to the next open.
    # ``date_range`` per session rather than arithmetic, so the tz round-trip cannot
    # drift and a session can never be silently one bar short.  The frames are
    # concatenated as an *index* rather than through ``np.concatenate``, which would
    # strip the timezone and leave naive timestamps downstream.
    stamps = pd.DatetimeIndex(
        [ts for op in opens
         for ts in pd.date_range(pd.Timestamp(op), periods=BARS_PER_SESSION, freq="min")]
    ).tz_convert("UTC")

    # Sub-bar wiggle, so open/high/low are not exact copies of close.
    step = rng.normal(0.0, sigma * 0.4, size=total)
    o = close * np.exp(-step)
    h = np.maximum(o, close) * (1.0 + np.abs(rng.normal(0, sigma * 0.2, total)))
    low = np.minimum(o, close) * (1.0 - np.abs(rng.normal(0, sigma * 0.2, total)))

    return pd.DataFrame(
        {
            "timestamp": stamps,
            "open": o,
            "high": h,
            "low": low,
            "close": close,
        }
    )


def session_count_for(n_bars: int) -> int:
    """How many sessions ``session_bars(n_bars)`` will produce."""
    return -(-int(n_bars) // BARS_PER_SESSION)
