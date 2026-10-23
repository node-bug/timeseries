"""Progression paths: what followed *each* lookback length, plotted together.

The **Progression** tab asks one question at many scales at once.  For every lookback
length ``L`` -- 10 bars, 20 bars, 30 bars, and so on up to the archive's own limit --
it finds the historical windows most similar to the archive's *last* ``L`` bars, and
then projects what happened next.  All of those projections are drawn on a single
chart, simultaneously.

**Why the sweep is the point.**  A single forecast at one length answers "what
usually followed this shape?".  The sweep answers a question no single forecast can:
*does the answer depend on how much history you look back?*  When the 10-bar
projection and the 500-bar projection disagree, the disagreement is informative --
it says the recent tape is saying something the longer history does not confirm.
Reading one curve would hide that entirely.

The module is deliberately free of any presentation concern.  It produces numbers and
the paths they describe; deciding how to colour them, how many to label and what the
caption says is the UI's job.

Two rules govern everything here, and both come from the matching engine rather than
from this module:

1. **The query is always the archive's own tail.**  Every length asks about the
   *most recent* ``L`` bars, so the sweep is a comparison of lookback lengths over one
   fixed moment in time, not a comparison of unrelated windows.  Changing the ticker
   or the archive changes all of them together.

2. **A length that cannot be answered is reported, not hidden.**  A 500-bar query
   against a 600-bar archive leaves too little history for the matcher to say
   anything useful; a 300-bar forward horizon on 1-minute data censors a large share
   of the candidate pool (§BX).  Both cases appear in the result as an explicit
   shortfall rather than as a silently missing line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .forecast import ForecastPath, forecast_paths
from .matching import DEFAULT_AMPLITUDE_WEIGHT

__all__ = [
    "ProgressionPath",
    "ProgressionSweep",
    "progression_paths",
    "default_lengths",
    "MAX_PROGRESSION_LENGTH",
]


#: The longest window the sweep will ask about, in bars.
#:
#: 500 is the app's own stated ceiling for a reference window on the **Progression**
#: tab, and it is deliberately *not* ``Timeframe.max_query_bars`` (390 on 1-minute,
#: 252 on daily).  That constant is the widest span the app reads as a *shape* on the
#: Price tab, where a window sits beside its own price tape and has to stay
#: recognisable.  Here the window is only ever a z-scored shape feeding a matcher --
#: it is never drawn as a price series -- so the limit is the one the user asked for.
MAX_PROGRESSION_LENGTH = 500


def default_lengths(
    n_bars: int,
    *,
    step: int = 10,
    max_length: int = MAX_PROGRESSION_LENGTH,
    min_length: int = 10,
) -> List[int]:
    """The lookback lengths to sweep, from ``min_length`` upward in ``step``-bar steps.

    The list is truncated at ``min(max_length, n_bars)`` rather than at some fraction
    of the archive.  A sweep needs the *whole* range the user asked for -- 10 through
    500 -- because the point of the sweep is to see the answer change with the
    lookback, and truncating it to the archive's length would quietly answer a
    different question on a short archive than on a long one.

    What the archive *can* support is a separate matter, decided per length by the
    matcher and reported as a shortfall (:attr:`ProgressionSweep.shortfall`).
    """
    step = max(1, int(step))
    min_length = max(1, int(min_length))
    top = int(max_length)
    if top < min_length:
        return []
    lengths = list(range(min_length, top + 1, step))
    # A trailing length that the step skipped is still worth asking about: it is the
    # longest lookback in range, and the sweep's last point is the one that most
    # often differs from the one before it.
    if lengths and lengths[-1] != top:
        lengths.append(top)
    return lengths


@dataclass(frozen=True)
class ProgressionPath:
    """The projection for **one** lookback length.

    Attributes
    ----------
    length
        The lookback ``L``: how many of the archive's most recent bars were used as
        the query.
    path
        The aggregated forward path for this length -- a
        :class:`timeseries.forecast.ForecastPath`, holding the median and the
        interquartile band over the matched windows.  ``None`` when this length could
        not be answered at all.
    best
        The single closest match's *own* forward path, in percent from its own anchor
        close.  ``None`` when there was no match, or when it had no complete forward
        history.

        This is the literal "best matched" reading of the question, and it is kept
        beside the aggregate rather than instead of it: a median over many windows
        is the claim about what *usually* happened, and one concrete window is the
        evidence behind it.  Drawing only the aggregate would hide how much the
        matches disagreed; drawing only the best would present one anecdote as an
        expectation.
    n_matches
        How many windows contributed to ``path``.
    n_requested
        How many windows were asked of the matcher -- the caller's ``k``.  This is
        the denominator the shortfall is measured against, and it is deliberately
        *not* :attr:`timeseries.forecast.ForecastPath.n_requested`, which counts the
        matches handed to the aggregator rather than the ones originally asked for.
    n_found
        How many windows the matcher actually returned.  Two things can reduce this
        below ``n_requested``: non-max suppression (§M) can leave no admissible
        candidate for some of the requested slots, and §BX's mask can remove part of
        the pool before the search starts.
    distance
        The best match's distance, or ``nan`` when there was none.
    percentile
        The best match's §E percentile rank against the candidate pool.
    n_candidates
        The size of the pool the best match was ranked against.
    n_masked
        Candidates withheld by §BX's forward-horizon mask.
    reason
        Why this length produced no path, or ``""`` when it did.  One of
        ``"archive too short"``, ``"no match"``, ``"no forward path"``, or ``""``.
        Stated rather than inferred, so the UI can say what happened instead of
        drawing a gap.
    """

    length: int
    path: Optional[ForecastPath] = None
    best: Optional[np.ndarray] = None
    n_matches: int = 0
    n_requested: int = 0
    n_found: int = 0
    distance: float = float("nan")
    percentile: float = float("nan")
    n_candidates: int = 0
    n_masked: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether this length produced a drawable projection."""
        return self.path is not None


@dataclass
class ProgressionSweep:
    """Every length's answer, and the summaries the chart needs.

    Attributes
    ----------
    paths
        One :class:`ProgressionPath` per requested length, in the order asked for.
        Lengths that could not be answered are present with ``ok`` false, so the
        reader can see *which* lookbacks failed rather than only how many.
    lengths
        The lengths that were requested, echoed for convenience.
    horizon
        The forward horizon every path projects over, in bars.
    n_bars
        The archive's size, in bars.
    matrix
        ``(len(paths), horizon + 1)`` of the per-length median curves, with ``nan``
        where a length could not be answered.  This is the array the chart's
        "median across lengths" line is drawn from, and it is stored rather than
        recomputed so the chart and the caption cannot disagree about it.
    """

    paths: List[ProgressionPath] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)
    horizon: int = 0
    n_bars: int = 0

    # -- summaries ---------------------------------------------------------- #

    @property
    def answered(self) -> List[ProgressionPath]:
        """The lengths that produced a path, in the order asked for."""
        return [p for p in self.paths if p.ok]

    @property
    def failed(self) -> List[ProgressionPath]:
        """The lengths that could not be answered."""
        return [p for p in self.paths if not p.ok]

    @property
    def shortfall(self) -> List[ProgressionPath]:
        """Lengths that answered, but from fewer windows than were asked for.

        Separate from :attr:`failed` because the two are different problems.  A
        failed length is an absence; a short length is a *thinner* answer than
        intended, and a median computed from seven windows instead of thirty is
        materially less trustworthy than one computed from thirty.  Both are reported;
        they are not merged.
        """
        return [p for p in self.paths if p.ok and p.n_matches < p.n_requested]

    @property
    def sparse(self) -> List[ProgressionPath]:
        """Lengths where the matcher itself returned fewer windows than requested.

        Distinct from :attr:`shortfall`: here the *ranking* was made over a thinner
        pool, because §BX's mask or non-max suppression left fewer candidates than
        were asked for.  In :attr:`shortfall` the matcher found its windows and the
        archive ran out afterwards.
        """
        return [p for p in self.paths if p.ok and p.n_found < p.n_requested]

    @property
    def matrix(self) -> np.ndarray:
        """``(len(paths), horizon + 1)`` of median curves, ``nan`` where unanswered."""
        if not self.paths:
            return np.empty((0, self.horizon + 1), dtype=float)
        out = np.full((len(self.paths), self.horizon + 1), np.nan, dtype=float)
        for i, p in enumerate(self.paths):
            if p.path is not None:
                out[i] = p.path.median
        return out

    @property
    def consensus(self) -> np.ndarray:
        """The median *across* the per-length medians.

        A reader looking at fifty overlapping traces cannot read the centre of the
        cloud by eye, so the middle of it is computed and drawn: what the lengths
        agree on.  Where they disagree this line is where the disagreement sits.

        ``nan`` for any offset at which fewer than half the answered lengths have a
        value, so the line never claims a consensus where there was one.
        """
        mat = self.matrix
        if mat.size == 0:
            return np.empty(0, dtype=float)
        with np.errstate(invalid="ignore"):
            counts = np.sum(np.isfinite(mat), axis=0)
            out = np.nanmedian(mat, axis=0)
        out[counts < max(1, mat.shape[0] // 2)] = np.nan
        return out

    def as_dict(self) -> dict:
        """A plain-data view, for callers that want to hand this to a chart builder."""
        return {
            "lengths": list(self.lengths),
            "horizon": int(self.horizon),
            "n_bars": int(self.n_bars),
            "n_answered": len(self.answered),
            "n_failed": len(self.failed),
        }


def _best_forward_series(
    close: np.ndarray,
    start: int,
    length: int,
    horizon: int,
) -> Optional[np.ndarray]:
    """The best match's own forward path, in percent from its anchor close.

    Rebuilt here rather than taken from :class:`timeseries.forecast.ForecastPath`,
    because that object stores the *aggregate* -- the median and the quartiles -- and
    the individual windows' series with it.  The reconstruction is exact: the same
    slice and the same rebasing :func:`timeseries.forecast.forecast_paths` applies,
    measured from the anchor bar (the last bar *inside* the window, per §Z1).

    ``None`` when the window has no complete forward history, or when its anchor
    close is not a usable positive number.  Both are "this window cannot say what
    happened next", and both are reported rather than papered over with a flat line.
    """
    close = np.asarray(close, dtype=float)
    anchor = int(start) + int(length) - 1
    end = anchor + int(horizon) + 1
    if anchor < 0 or end > close.size:
        return None
    base = close[anchor]
    if not np.isfinite(base) or base <= 0:
        return None
    window = close[anchor:end]
    if not np.isfinite(window).all():
        return None
    return (window / base - 1.0) * 100.0


def progression_paths(
    pipe,
    *,
    lengths: Sequence[int],
    horizon: int,
    k: int = 30,
    amplitude_weight: float = DEFAULT_AMPLITUDE_WEIGHT,
) -> ProgressionSweep:
    """Sweep the lookback lengths and project each one.

    For each length ``L`` in ``lengths``:

    1. the query is the archive's **most recent** ``L`` bars -- one fixed moment,
       read at many scales;
    2. :meth:`timeseries.pipeline.Pipeline.match` finds the closest historical
       windows, with §BX's forward-horizon mask applied so a candidate whose
       projection would run across a session closure cannot win on an inflated
       overnight gap;
    3. :func:`timeseries.forecast.forecast_paths` aggregates the matched windows
       into a median path and an interquartile band;
    4. the single closest match's own forward path is reconstructed alongside it.

    Parameters
    ----------
    pipe
        The :class:`timeseries.pipeline.Pipeline` to search.
    lengths
        The lookback lengths to sweep, in bars.
    horizon
        How many bars forward every path projects.
    k
        How many matched windows to request per length.
    amplitude_weight
        The matcher's amplitude penalty weight; the same dial every other tab uses.

    Returns
    -------
    ProgressionSweep
        One entry per requested length, in order, including the ones that failed.

    Notes
    -----
    **Every length is answered against the same archive and the same instant.**  The
    query for length ``L`` is the tail of the archive's last ``L`` bars, so the sweep
    compares lookbacks rather than comparing windows -- a different design, and the
    one that makes the resulting traces comparable to each other.

    **The length itself is the query's exclusion zone.**  §M's exclusion margin
    defaults to one query length, so the shorter lookbacks are automatically unable
    to match the bars immediately before them, and the longer ones are excluded from
    a wider neighbourhood.  This is inherited from the matcher and is not overridden
    here: an overlapping candidate is not independent evidence, and a 10-bar window
    that matched the 9 bars it shares with the query would be matching the query.

    **Failures are per length and are reported per length.**  A 500-bar lookback
    against a 600-bar archive has almost no history to compare against; a 300-bar
    horizon on 1-minute data censors a large share of the pool.  Neither is an
    error, and neither is allowed to vanish: the sweep records why each length
    produced nothing, and the UI states it.
    """
    lengths = [int(v) for v in lengths]
    horizon = int(horizon)
    k = int(k)
    n_bars = int(getattr(pipe, "n_bars", 0) or 0)
    close = np.asarray(getattr(pipe, "close", np.empty(0)), dtype=float)

    out: List[ProgressionPath] = []
    for length in lengths:
        if length <= 0:
            out.append(ProgressionPath(length=length, reason="archive too short"))
            continue

        # The archive must hold the window *and* enough history behind it for the
        # matcher to have anything to compare against.  ``pipe.match`` requires a
        # query span inside the series; a length past the archive's own size has
        # no such span.
        if n_bars <= length:
            out.append(ProgressionPath(length=length, reason="archive too short"))
            continue

        start = n_bars - length
        try:
            query = pipe.query_span(start, n_bars)
            result = pipe.match(query, k=k,
                                amplitude_weight=float(amplitude_weight),
                                max_horizon=horizon)
        except Exception:  # noqa: BLE001 - one bad length must not lose the sweep
            out.append(ProgressionPath(length=length, reason="no match"))
            continue

        matches = list(getattr(result, "matches", []) or [])
        if not matches:
            out.append(ProgressionPath(
                length=length,
                n_requested=k,
                n_found=0,
                n_candidates=int(getattr(result, "n_candidates", 0) or 0),
                n_masked=int(getattr(result, "n_masked", 0) or 0),
                reason="no match",
            ))
            continue

        starts = np.asarray([m.start for m in matches], dtype=np.int64)
        distances = np.asarray([m.distance for m in matches], dtype=float)
        path = forecast_paths(close, starts, length, horizon, distances=distances)

        best_match = matches[0]
        best_series = _best_forward_series(close, best_match.start, length, horizon)

        out.append(ProgressionPath(
            length=length,
            path=path,
            best=best_series,
            n_matches=int(path.n_matches) if path is not None else 0,
            n_requested=k,
            n_found=len(matches),
            distance=float(best_match.distance),
            percentile=float(best_match.percentile),
            n_candidates=int(getattr(result, "n_candidates", 0) or 0),
            n_masked=int(getattr(result, "n_masked", 0) or 0),
            reason="" if path is not None else "no forward path",
        ))

    return ProgressionSweep(
        paths=out,
        lengths=lengths,
        horizon=horizon,
        n_bars=n_bars,
    )
