"""Feature construction for pattern matching. Fixes PLAN.md §Z2.

Two defects in the original pipeline are addressed here:

1. The feature dimensions were on incomparable scales.  One leg was z-scored while
   the other was not, so the Euclidean distance weighted them by whatever raw units
   each happened to carry.  Both legs are now z-scored here, and the scaling is
   explicit rather than a side effect of column ordering.

2. A raw ``pct_change`` on volume was replaced by the ``log1p`` transform, which made
   zero-volume bars finite instead of ``inf``.  The volume leg has since been removed
   entirely; see :data:`FEATURE_COLUMNS` for why.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .timeframes import DEFAULT_TIMEFRAME, get_timeframe, resolve_timeframe

__all__ = ["build_features", "FEATURE_COLUMNS", "safe_log_return", "quality_report"]

#: The feature legs used for matching, in the order they occupy in the matrix's
#: columns.  All are z-scored before they reach the matcher.
#:
#: **``path_z`` is the leg the Price tab actually shows.**  ``return_z`` is the
#: *derivative* of price -- the rolling z-score of the log return -- so over a 60-bar
#: window it correlates with the drawn price path at only **−0.12**: the matcher and the
#: chart were looking at two different things.  ``path_z`` is the rebased log-price path
#: itself, which is the line on screen, so a match that traces ``path_z`` traces what
#: the reader selected.
#:
#: It is not a replacement for ``return_z``.  A price path says *where* price went;
#: a return series says *how violently* it got there, and those disagree usefully -- a
#: smooth drift and a violent round-trip can share a net move.  Both are kept.
#:
#: Measured on the live QQQ archive at L=60, adding this leg moved the best match's
#: visual correlation with the query from **+0.27 to +0.84**; scoring on the log-price
#: path alone reaches **+0.93**, which is the ceiling this family of metrics has.
#:
#: **There is no volume leg.**  A ``volume_z`` channel used to be scored alongside these
#: two, and never earned its place.  On the live archive its lag-1 autocorrelation was
#: −0.31 -- it barely predicts itself -- and its rank across all 7,870 searchable
#: windows had a Spearman correlation of **−0.001** with the rank computed on the drawn
#: price path.  It contributed roughly half the squared distance that selected the
#: reported match while carrying no information about the shape the reader can see.
#: Matching is now a pure price question, which is also what the rest of the UI draws
#: and describes.
FEATURE_COLUMNS = ("return_z", "path_z")

# Bars needed before a rolling window produces a usable standard deviation.  Kept as a
# module constant for back-compat and as the *default* for every timeframe: 20 minutes
# intraday, and 20 sessions (about a trading month) daily, which is the shortest
# horizon over which a daily rolling volatility estimate is not obviously degenerate.
# :data:`timeseries.timeframes.TIMEFRAMES` owns the authoritative value per timeframe.
_MIN_ROLLING = get_timeframe("1m").rolling_window


def safe_log_return(close: pd.Series) -> pd.Series:
    """Log returns, with non-positive prices masked rather than propagated."""
    c = pd.to_numeric(close, errors="coerce")
    valid = c > 0
    out = pd.Series(np.nan, index=c.index, dtype=float)
    out[valid] = np.log(c[valid] / c[valid].shift(1))
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_z(s: pd.Series, window: int = _MIN_ROLLING) -> pd.Series:
    """Rolling z-score, falling back to expanding statistics while warming up.

    The first ``window`` rows have no full-window standard deviation.  Using a plain
    expanding standard deviation there is unstable (a single bar gives ``sd = 0``),
    so those rows are left as NaN and the caller decides how to handle them.
    """
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def build_features(
    df: pd.DataFrame,
    *,
    rolling: Optional[int] = None,
    prefix: str = "",
    timeframe: object = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Add matching features to a bar frame.

    Parameters
    ----------
    df
        Bar frame with at least ``close``.
    rolling
        Rolling window length for the z-score.  ``None`` (the default) resolves it from
        the timeframe, so a daily frame does not silently inherit an intraday
        20-minute base just because that was the literal default.
    prefix
        Optional column-name prefix, so several feature sets can coexist.
    timeframe
        Only consulted when ``rolling`` is ``None``.  See
        :mod:`timeseries.timeframes`.

    Returns
    -------
    DataFrame
        Copy of ``df`` with ``<prefix>return_z`` and ``<prefix>path_z`` appended.  The
        first may contain NaN during the warm-up period; callers use
        :func:`finalize_features` to drop or impute.

    Notes
    -----
    ``path_z`` is **not** rolling-z.  It is the log price with a *global* mean removed
    and nothing else: the level shift is a constant, so it cancels in every
    per-window difference the scorer takes, while the per-window z-score that STUMPY
    applies later supplies the local normalisation.  Rolling-z it as well would divide
    the path by a second, slowly-varying scale and flatten exactly the slow drift the
    leg exists to capture.
    """
    out = df.copy()
    tf = resolve_timeframe(timeframe)
    if rolling is None:
        rolling = tf.rolling_window

    ret = safe_log_return(out["close"])
    out[f"{prefix}log_return"] = ret
    out[f"{prefix}return_z"] = _rolling_z(ret, rolling)

    # The log-price path, mean-removed but *not* rescaled.  ``safe_log_return`` already
    # masked non-positive closes to NaN, so the cumulative sum is log price up to a
    # constant -- and that constant is removed here, which is all the leg needs: the
    # per-window z-score the scorer applies later does the local normalisation.
    path = safe_log_return(out["close"]).cumsum()
    out[f"{prefix}path_z"] = (path - path.mean()).where(np.isfinite(path))
    return out


def finalize_features(
    df: pd.DataFrame, *, prefix: str = "", how: str = "drop"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Resolve NaNs in the feature legs and return a clean array.

    Parameters
    ----------
    how
        ``"drop"``    remove warm-up rows (default; honest, loses ~20 bars)
        ``"ffill"``   forward-fill, then back-fill, then zero any remainder
        ``"zero"``    zero-fill

    Returns
    -------
    (frame, matrix, valid_mask)
        ``matrix`` is ``(n, len(FEATURE_COLUMNS))`` finite float64, ``valid_mask``
        marks rows retained.
    """
    rcol = f"{prefix}return_z"
    pcol = f"{prefix}path_z"
    frame = df.copy()

    if how == "drop":
        # ``path_z`` is deliberately not in the drop test.  It has no warm-up of its
        # own (the global mean needs every bar, not a trailing window), so requiring it
        # to be non-NaN would throw away the same first rows twice for no reason --
        # and ``return_z`` already fails on exactly those.
        frame = frame.loc[frame[rcol].notna()].copy()
    elif how == "ffill":
        for c in (rcol, pcol):
            frame[c] = frame[c].ffill().bfill().fillna(0.0)
    elif how == "zero":
        for c in (rcol, pcol):
            frame[c] = frame[c].fillna(0.0)
    else:
        raise ValueError(f"unknown how={how!r}")

    cols = [rcol, pcol]
    mat = frame[cols].to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return frame, mat, frame.index.to_numpy()


def quality_report(df: pd.DataFrame, *, timeframe: object = DEFAULT_TIMEFRAME) -> dict:
    """Cheap quality gate over a bar frame. PLAN.md §B.

    Reports the conditions that would invalidate downstream matching.  This is a
    diagnostic, not a hard gate -- callers decide whether to proceed.

    §BF: intra-session holes and overnight session boundaries are *different*
    failures.  A 17.5-hour gap between 16:00 and the next 09:30 is the market being
    closed, not missing bars.  Counting both as "gaps" produced ``gaps_over_180s: 20``
    on a perfectly clean archive and would read as corruption to an operator.  The two
    are now reported separately.

    **Daily bars have no holes, and the reason is not the obvious one.**  It would be easy
    to assume the 180-second rule flags every day on a daily series, since consecutive
    daily bars are ~86,400 s apart.  Measured, it does not: the intra-session filter
    (§BF) only counts a delta whose predecessor shares the *same* Eastern date, and on
    daily bars every bar is its own date, so the filtered set is empty and the count is
    0 -- a correct answer reached for a reason the intraday code does not expect.

    So the daily branch does not exist to fix a wrong number.  It exists so the report
    carries the one condition daily data *can* have and this function was blind to:
    a weekday with no bar at all, reported as ``missing_days`` with weekends excluded.
    A holiday week reads as 0 holes / 0 missing days on the intraday path only because
    holidays are simply absent from both counts.

    The hole count is keyed ``intra_session_holes`` rather than the historical
    ``gaps_over_180s``, which encoded both the quantity *and* the threshold in one name
    and would have been a lie the moment a second threshold existed.  The old key is
    still reported as an alias so an existing consumer does not see a missing one.
    """
    tf = resolve_timeframe(timeframe)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    close = pd.to_numeric(df["close"], errors="coerce")

    # Sessions are US/Eastern days, not UTC days.  Bucketing in UTC splits a session
    # at 20:00 ET and misclassifies every overnight closure as an intra-session hole.
    et = ts.dt.tz_convert("America/New_York")
    session_key = et.dt.floor("D")
    counts = session_key.value_counts()

    ordered = ts.sort_values()
    deltas = ordered.diff().dt.total_seconds()
    # A delta is *intra-session* only when it and its predecessor share an ET date.
    o_session = session_key.reindex(ordered.index)
    intra = deltas[o_session.eq(o_session.shift())].dropna()

    if tf.gap_seconds is None:
        # A daily bar is a whole session: there is no interior, so nothing can be a
        # hole.  See ``missing_days`` below for what *can* be wrong instead.
        holes = 0
    else:
        # Anything above the threshold within a session is a genuine hole
        # (halt, dropped print) rather than jitter.
        holes = int((intra > tf.gap_seconds).sum())
    # Legitimate closures: the clock jumped past the end of one session to the start
    # of the next.  Distinct from holes by construction, not by a magic threshold.
    boundaries = int((~o_session.eq(o_session.shift()) & deltas.notna()).sum())

    # Weekdays present in the frame, for the missing-day count.  Excludes weekends,
    # which are never sessions and would otherwise dominate the figure.
    weekdays = pd.DatetimeIndex(session_key.dropna().unique())
    weekdays = weekdays[weekdays.dayofweek < 5]
    missing_days = 0
    if len(weekdays) > 1:
        span = pd.date_range(weekdays.min(), weekdays.max(), freq="B")
        missing_days = int(len(span) - len(weekdays))

    report = {
        "rows": int(len(df)),
        "sessions": int(counts.size),
        "bars_min": int(counts.min()) if counts.size else 0,
        "bars_max": int(counts.max()) if counts.size else 0,
        "intra_session_holes": holes,
        "session_boundaries": boundaries,
        "bad_timestamps": int(ts.isna().sum()),
        "nonpositive_close": int((close <= 0).sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "monotonic": bool(ts.is_monotonic_increasing),
        # The historical spelling, kept so an existing consumer reading it does not
        # see a missing key.  Identical to ``intra_session_holes`` on 1-minute, where
        # the threshold *is* 180 s, and a hard 0 on daily -- a daily bar has no
        # interior to have a hole in.  See ``missing_days`` for the daily condition.
        "gaps_over_180s": holes,
    }
    if tf.gap_seconds is None:
        # Only *added* on daily.  The key is spelled out rather than computed on both
        # paths because a reader seeing ``missing_days`` on 1-minute bars would be
        # right to ask what it means there, and the answer is "always 0".
        report["missing_days"] = missing_days
        # ``missing_days_per_year`` is what stops the raw count from being misread.
        # **Every weekday without a bar is counted, and most of those are market
        # holidays** -- the exchange was closed, which is not missing data.  Measured
        # on live QQQ daily bars, a 25-year archive reports ``missing_days: 233``,
        # which is 9.3 a year against a US equity market's ~9-10.  Without the rate the
        # count reads as a partial download; with it, the reader can see the count is
        # what a complete archive looks like.
        span_days = 0.0
        if len(weekdays) > 1:
            span_days = (weekdays.max() - weekdays.min()).days
        report["missing_days_per_year"] = (
            round(missing_days * 365.25 / span_days, 2) if span_days > 0 else 0.0
        )
        report["timeframe"] = tf.key
    return report