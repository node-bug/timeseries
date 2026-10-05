"""
Pattern matching engine for QQQ 1-minute bars.

Implements the three-stage funnel from PLAN.md §C, with STUMPY as the *only*
scoring engine:

  Stage 1  coarse scan over every candidate window, via STUMPY's ``mass``
  Stage 2  the same distance, ranked -- STUMPY computes the whole profile in one
           JIT-compiled pass, so there is no separate cheap/expensive split
  Stage 3  hygiene: exclusion zone, non-maximum suppression

**Why there is exactly one path.**  A matrix profile is defined by sliding a query
over *one contiguous series*; ``stumpy.mass`` computes precisely that and nothing
else.  It cannot score a pre-materialised library of overlapping windows, because
such a library is not a contiguous series -- treating it as one silently compares
the wrong things.  The library route this module used to expose therefore could
never have been STUMPY-backed, which is why it is removed rather than kept as a
fallback: it was a *different metric over a different candidate set*, not a slower
path to the same answer.

Design notes that correspond to specific plan sections:

* §M  The query window lives INSIDE the search space.  Every candidate whose span
       overlaps the query is rejected before scoring, otherwise the query is
       guaranteed to return itself at distance ~0.
* §M  Non-maximum suppression keeps temporally separated matches, because a stride-1
       window library contains thousands of near-duplicates of the same moment.
* §E  Nothing here reports a p-value; callers get a score plus a percentile rank so
       the UI can report "top 0.4% of 8000 windows" rather than a bare distance.
* §BC Query and candidate are normalised identically, and STUMPY guarantees it:
       ``normalize=True`` z-scores the query and every subsequence together, in one
       place, so the asymmetry cannot be reintroduced by a caller.
* §BD A metric that cannot be ranked is not offered.  The old DTW path reported a
       percentile computed against a *Euclidean* distribution; rather than keep a
       second metric whose percentile is honestly ``NaN``, the second metric is gone.

§M's exclusion and suppression are applied here rather than delegated to STUMPY, so
the rules that decide what counts as evidence live in one place and cannot drift away
from the scorer's own defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "Query",
    "Match",
    "MatchResult",
    "zscore",
    "find_matches",
    "non_max_suppression",
    "exclusion_mask",
    "forward_returns",
    "forward_horizon_valid_mask",
    "DEFAULT_AMPLITUDE_WEIGHT",
    "SCORER",
]

# The single scoring engine, recorded on every result so a number's provenance
# travels with it.
SCORER = "stumpy"

#: How strongly a window's **size of move** counts, relative to its normalised shape.
#:
#: STUMPY's distance is scale-free: every window is z-scored before comparison, so the
#: query is exactly as far from a dead-calm window as from a violent one.  On the price
#: chart those are visually opposite, so a pure shape match can be a window the reader
#: would never have chosen.  See the *Why there is an amplitude term* note on
#: :func:`find_matches` for the measurements that motivated this.
#:
#: **The default is 1.0, and it is deliberately strong.**  Measured on the live QQQ
#: archive at L=60, the pure-shape best match had a visual correlation of only **+0.27**
#: with the query's own price path, while the best match scored on the drawn price path
#: reached **+0.93** -- and the two searches picked windows **3,258 bars apart**.  A
#: default that leaves the reported match uncorrelated with the thing the reader selected
#: is not a defensible default, so amplitude is on by default rather than opt-in.
#:
#: At 1.0 a candidate must move as far as the query, in the same direction, to compete
#: on shape alone -- which is what "when did price do *this*?" means.  Set it to 0 to get
#: the previous shape-only behaviour; the sidebar exposes the dial so the choice is the
#: reader's, and every reported percentile is computed against whichever rule is in
#: force, so the number always describes the search that produced it.
DEFAULT_AMPLITUDE_WEIGHT = 1.0


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def zscore(x: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Z-normalise a 1-D array.

    §C: matching must happen on returns rather than price levels, and each window is
    z-scored independently so shape is compared independently of level and scale.

    Returns an all-zero array for constant input rather than NaN/inf -- a flat window
    is a legitimate (if uninteresting) pattern and must not poison the search.

    The scorer does not call this: ``stumpy.mass(normalize=True)`` performs the same
    transform internally, on both sides of every comparison.  It stays public because
    callers that *draw* a query must reproduce the scorer's normalisation exactly, and
    because the flat-window convention is a real case worth stating explicitly.
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std(ddof=ddof)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Query:
    """A pattern to search for.

    §R: both the live-window path and the manual-selection path produce this, so the
    engine has one interface regardless of where the query came from.

    Attributes
    ----------
    vector
        The normalised feature vector compared against the library.
    start, stop
        Half-open bar-index span ``[start, stop)`` this query occupies in its own
        series.  Required for the exclusion zone (§M); a query detached from its
        source series cannot exclude itself and will match itself.
    label
        Human-readable provenance, e.g. "2026-09-30 15:32" -- surfaced in the UI.
    """

    vector: np.ndarray
    start: int
    stop: int
    label: str = ""

    @property
    def length(self) -> int:
        return int(self.stop - self.start)

    @classmethod
    def from_span(
        cls,
        features: np.ndarray,
        start: int,
        stop: int,
        label: str = "",
        per_window: bool = True,
    ) -> "Query":
        """Build a query from ``features[start:stop]``.

        Parameters
        ----------
        features
            ``(n,)`` single series or ``(n, d)`` multi-channel features.
        start, stop
            Half-open bar span.
        per_window
            Re-normalise the slice along its first axis so the query carries no
            residual level from the larger series.  For 2-D input this z-scores each
            channel independently, which keeps the distance from being dominated by
            whichever channel happens to have larger raw units.
        """
        if not 0 <= start < stop <= len(features):
            raise ValueError(
                f"invalid span [{start}, {stop}) for array of length {len(features)}"
            )
        vec = np.asarray(features[start:stop], dtype=float)
        if per_window:
            if vec.ndim == 1:
                vec = zscore(vec)
            elif vec.ndim == 2:
                vec = np.stack([zscore(vec[:, c]) for c in range(vec.shape[1])], axis=-1)
            else:
                raise ValueError(f"unsupported query ndim {vec.ndim}")
        return cls(vector=vec, start=start, stop=stop, label=label)


# --------------------------------------------------------------------------- #
# Stage 3 helpers: exclusion and suppression
# --------------------------------------------------------------------------- #
def exclusion_mask(
    candidate_starts: np.ndarray,
    query: Query,
    margin: Optional[int] = None,
) -> np.ndarray:
    """Mark candidates overlapping the query window, or within ``margin`` of it.

    §M: a candidate starting at ``s`` occupies ``[s, s + L)``.  It overlaps the query
    span ``[q0, q1)`` iff ``s < q1`` and ``s + L > q0``.

    ``margin`` defaults to one query length.  The wider margin is deliberate: the
    usual matrix-profile convention is ~m/4, which suppresses only the query itself
    and still returns the immediately adjacent windows -- which share almost all of
    their bars with the query and are not independent evidence either.

    The default is read from :func:`timeseries.matrix_profile.excl_zone_for`, so the
    margin policy has exactly one definition.  An explicit ``margin`` still wins,
    which is what lets the tests contrast the strict margin against the library's.
    """
    from .matrix_profile import excl_zone_for

    L = query.length
    if margin is None:
        margin = excl_zone_for(L, strict=True)
    lo = query.start - margin - L
    hi = query.stop + margin
    return (candidate_starts > lo) & (candidate_starts < hi)


def non_max_suppression(
    starts: np.ndarray,
    scores: np.ndarray,
    min_separation: int,
    k: int,
) -> np.ndarray:
    """Greedily take the best ``k`` starts, rejecting any within ``min_separation``.

    Smaller ``scores`` is better.  Returns indices into ``starts`` **best-first**.

    §M/§C: without this the top-5 is usually the same moment five times, because
    stride-1 windows overlap by length-1 bars.

    The returned order is load-bearing, not cosmetic.  Callers label these "match #1",
    "#2", ... and the Price tab draws index 0 as *the* match found, so returning them
    sorted by position rather than by score silently substituted an arbitrary early
    window for the best one.  It also made the visible match depend on the ``k``
    slider: NMS is greedy, so a larger ``k`` can accept a window that then displaces
    the true best from index 0.  Ties keep their (stable) discovery order.
    """
    if len(starts) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(scores, kind="stable")
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= k:
            break
        s = int(starts[i])
        if all(abs(s - int(starts[c])) >= min_separation for c in chosen):
            chosen.append(int(i))
    # `order` is ascending in score, so appending preserves best-first order.
    return np.array(chosen, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    """One accepted analogue."""

    start: int
    stop: int
    distance: float
    percentile: float
    timestamps: tuple = ()
    forward_returns: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    matches: list
    n_candidates: int
    n_excluded: int
    n_after_nms: int
    query: "Query"
    method: str
    #: Candidates removed because ``valid_mask`` marked them inadmissible (PLAN.md §BX).
    #: Tracked separately from ``n_excluded`` so the UI can report a data-quality
    #: suppression as something distinct from the §M self-match guard.
    n_masked: int = 0

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def match_percentiles(self) -> np.ndarray:
        """Each accepted match's percentile -- its position in the distance ranking.

        PLAN.md §BW: the Forecast tab's random-window baseline has to be drawn from
        the same *region of the distance ranking* as the matches, and this is what
        makes that possible.  Forward return is correlated with distance rank (on a
        random walk, corr = **-0.85** between the distance decile and the mean forward
        return), so a baseline drawn uniformly from the whole archive compares a match
        set drawn from the extreme low-distance tail against a population it can never
        resemble, and reports the gap as significant ~50% of the time.
        """
        return np.array([m.percentile for m in self.matches], dtype=float)


# --------------------------------------------------------------------------- #
# THE Z1 FIX
# --------------------------------------------------------------------------- #
def forward_returns(
    close: np.ndarray,
    match_starts: np.ndarray,
    window_length: int,
    horizons,
) -> dict:
    """Return the forward log-return from the END of each matched window.

    THIS FUNCTION IS THE FIX FOR PLAN.md §Z1.

    The previous implementation anchored the forward return at the window *start*::

        log(close[start + h] / close[start])

    With ``window_length=60`` and ``horizon=5`` that measures bars 0-5 *of the pattern
    that already matched* -- a tautology that produces a confident number regardless
    of what the archive contains.  §E explicitly requires that forward returns not
    overlap the query window.

    The corrected anchor is the bar immediately after the window::

        log(close[start + L - 1 + h] / close[start + L - 1])

    Parameters
    ----------
    close
        1-D close series.
    match_starts
        Start index of each matched window.
    window_length
        Length ``L`` of the matched window.
    horizons
        Iterable of bar counts to look forward.

    Returns
    -------
    dict
        ``{horizon: array_of_returns}`` with NaN where the horizon runs past the end
        of the data, so callers can count valid observations rather than silently
        treating a missing tail as zero.
    """
    close = np.asarray(close, dtype=float)
    out: dict = {}
    for h in horizons:
        h = int(h)
        anchor = match_starts + window_length - 1  # last bar INSIDE the window
        end = anchor + h  # first bar AFTER the horizon
        valid = (anchor >= 0) & (end < len(close))
        rets = np.full(len(match_starts), np.nan, dtype=float)
        if valid.any():
            a = anchor[valid]
            b = end[valid]
            rets[valid] = np.log(close[b] / close[a])
        out[h] = rets
    return out


def forward_horizon_valid_mask(boundary: np.ndarray, window_length: int,
                               max_horizon: int) -> np.ndarray:
    """Per window-start: does the forward horizon avoid every bar-crossing in ``boundary``?

    PLAN.md §BX.  ``boundary`` marks bars that *open* a new session; a forward return is
    corrupt exactly when one of those falls inside the horizon.

    The anchor is the window's last bar (§Z1), so the horizon covers bars
    ``(anchor, anchor + max_horizon]``.  A boundary *at* ``anchor`` is deliberately not
    counted: that bar is itself the first bar of a new session, so its forward return
    starts inside one and never leaves it.

    Written against window *starts*, so it can be handed straight to
    :func:`find_matches`' ``valid_mask`` and be applied before the §E percentile
    population is formed -- a window that could never be returned must not be part of
    the distribution the returned windows are ranked against.

    Parameters
    ----------
    boundary
        Boolean over bars, ``True`` where the bar starts a new session.
    window_length, max_horizon
        ``L`` and ``h``; the horizon is ``L - 1 + h`` bars from the window start.

    Returns
    -------
    np.ndarray
        Boolean over the ``n - L + 1`` window starts.  Starts whose horizon would run
        past the end of the data are ``False``, matching :func:`forward_returns`, which
        returns ``nan`` for them.
    """
    b = np.asarray(boundary, dtype=bool)
    n = b.size
    starts = np.arange(max(0, n - int(window_length) + 1), dtype=np.int64)
    out = np.zeros(starts.size, dtype=bool)
    if starts.size == 0:
        return out

    anchor = starts + int(window_length) - 1
    end = anchor + int(max_horizon)
    fits = (anchor >= 0) & (end < n)
    if not fits.any():
        return out

    # Prefix sums turn "is any boundary in (anchor, end]" into O(1) per window, so the
    # whole mask is O(n) rather than O(n * h) -- worth having because it runs on every
    # interactive match, not just a nightly batch.
    cs = np.concatenate([[0], np.cumsum(b, dtype=np.int64)])
    a = anchor[fits]
    e = end[fits]
    out[fits] = (cs[e + 1] - cs[a + 1]) == 0
    return out


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def find_matches(
    series: np.ndarray,
    query: "Query",
    *,
    k: int = 10,
    exclusion_margin: Optional[int] = None,
    nms_separation: Optional[int] = None,
    amplitude_weight: float = DEFAULT_AMPLITUDE_WEIGHT,
    amplitude_series: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> MatchResult:
    """Find the windows of ``series`` most similar to ``query``, scored by STUMPY.

    Parameters
    ----------
    series
        ``(n,)`` or ``(n, d)`` feature matrix -- the *contiguous* feature series, not
        a pre-materialised window library and not pre-normalised.  ``query.start`` and
        ``query.stop`` index into it.
    query
        The pattern to find.
    k
        Maximum matches returned, after suppression.
    exclusion_margin
        Bars of margin around the query.  Default: one query length (§M).
    nms_separation
        Minimum bar gap between accepted matches.  Default: one query length.
    amplitude_weight
        How strongly a window's **size of move** counts, relative to its normalised
        shape.  ``0`` recovers pure shape-matching; ``1`` (the default) demands a
        candidate that moved about as far as the query did, in the same direction.
    amplitude_series
        The series the realised move is measured from -- **raw log returns, not
        ``return_z``**.  This is load-bearing: :func:`amplitude_profile` works by
        summing a window, and summing a rolling-z rescales each bar by its *local*
        volatility, so a 3-sigma move in a quiet hour and a 1-sigma move in a frantic
        one both sum to about the same thing.  That inverts the signal -- the penalty
        measured on ``return_z`` rated a window that moved 3× further than the query
        (0.327% vs 0.017%) as *closer* than one that matched its move.  The default
        derives it from the price leg as a caller-independent fallback; the pipeline
        passes the true one.
    valid_mask
        Optional boolean over *window starts*, ``True`` where a candidate is admissible.
        Removed from the pool before NMS and before the percentile population, exactly
        like :func:`exclusion_mask` (§M/§BN2: the rules that decide what counts as
        evidence live in this codebase, never delegated to STUMPY).  Defaults to all
        admissible.

        PLAN.md §BX uses this to drop windows whose **forward horizon** would run across
        a session closure -- measured at a 20x forward-return inflation on the live
        archive.  It is deliberately *not* used to drop windows that merely *straddle* a
        boundary: those distort the ranking but not the reported return, and excluding
        them costs 31% of the pool at L=60.  See
        :func:`timeseries.pipeline.session_boundary_mask` for why the two are not the
        same question.

    Returns
    -------
    MatchResult
        Matches best-first, each carrying a percentile rank against the full candidate
        distribution (§E -- report a rank, not a bare distance).

    Notes
    -----
    **The query is re-read from the series, not taken from ``query.vector``.**  STUMPY
    normalises the query and every candidate together (``normalize=True``), so handing
    it an already-z-scored ``query.vector`` would scale only one side of every
    comparison.  Passing the raw span is what makes §BC's normalisation symmetry a
    property of the library instead of a rule every caller has to remember.

    **Why there is an amplitude term.**  STUMPY's normalised distance is *scale-free by
    construction*: it z-scores both windows, so the flattest and the most violent window
    in the archive are the same distance from any query.  Measured on the live QQQ
    archive over 60-bar windows, a window that moved **+0.013%** and one that moved
    **+0.829%** both ended up with a per-window feature standard deviation of ≈0.95 --
    and the feature-space distance between them was 16.4, essentially the maximum
    possible.  So the two were not confused *with each other*; the point is that the
    query cannot be told apart from a window of a completely different size, because
    size is not part of what the distance measures.

    On a price chart a 0.013% window and a 0.829% window are visually opposite: one is
    a flat line, the other a swing.  A reader who brushes a swing and is shown a flat
    line has been given a "match" they would never have picked, and the percentile will
    still call it rare.  Adding ``amplitude_weight * (m_i - m_q) / scale`` to the
    squared distance restores the one dimension normalisation removes, using the same
    units the chart is drawn in.

    The scale is the standard deviation of every window's realised move, so the term is
    measured in *window-standard-deviations of move* and stays comparable as the
    archive grows.  It is added to the **squared** distance so it is a genuine part of
    the metric rather than a tie-breaker applied after ranking -- which matters, because
    a tie-break after ranking would let a near-tiny in shape distance be overridden by
    amplitude, and then report a percentile computed against a distribution that never
    included that rule.

    **The candidate population is every bar position**, including the partial windows
    at the tail, exactly as a matrix profile is defined.  ``n_candidates`` reports that
    population, so the percentile means the same thing everywhere else.

    Exclusion (§M) is applied here rather than via STUMPY's ``excl_zone``: its default
    is ``m/4``, which suppresses only the trivial match and still returns the query's
    immediate neighbours -- windows sharing ``m-1`` bars with it, which are not
    independent evidence.  :func:`exclusion_mask` encodes the stricter full-window
    margin.
    """
    from .matrix_profile import amplitude_profile, distance_profile

    series = np.asarray(series, dtype=float)
    if series.ndim not in (1, 2):
        raise ValueError(f"series must be 1-D or 2-D, got shape {series.shape}")
    if not 0 <= query.start < query.stop <= len(series):
        raise ValueError(
            f"query span [{query.start}, {query.stop}) is outside series of length {len(series)}"
        )

    # Score the query's own slice of the series, so STUMPY sees raw data on both sides
    # and does its own z-normalisation.
    q_raw = series[query.start : query.stop]
    all_dists = distance_profile(q_raw, series, query_idx=query.start)

    # PLAN.md §BX: caller-supplied admissibility, normalised **before** it is used
    # anywhere.  It is read by the amplitude reference below as well as by the
    # candidate filter, so a definition sitting next to the filter -- after the
    # amplitude block has already run -- would be a use-before-assignment bug waiting
    # for the first caller to pass a mask.  Every early return below also reports
    # `n_masked`, so a fully censored search is never indistinguishable from an
    # unfiltered one.
    n_candidates = len(all_dists)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != (n_candidates,):
            raise ValueError(
                f"valid_mask must be over window starts: expected {(n_candidates,)}, "
                f"got {vm.shape}"
            )
    else:
        vm = np.ones(n_candidates, dtype=bool)
    n_masked = int((~vm).sum())

    # Amplitude: the component the normalised distance above cannot see.  Computed
    # from channel 0, which for this package is a rolling-z of the log return, so its
    # sum over a window is that window's real log move.
    #
    # The penalty is expressed in units of a *typical* shape distance, not in units of
    # the move's own standard deviation.  That scaling is load-bearing.  The two terms
    # go into a sum of squares, so their magnitudes have to be commensurable or the
    # larger one simply wins every comparison: at L=60 the shape distance of a random
    # window is ~11.3, so its square is ~128, while an amplitude penalty of one move
    # standard deviation squares to 1.  Added unscaled, the amplitude term changed the
    # reported best match by 0.1% -- numerically present, functionally absent.
    if amplitude_weight:
        amp_src = series if amplitude_series is None else amplitude_series
        moves = amplitude_profile(amp_src, query.length)
        q_move = moves[query.start]
        finite = np.isfinite(moves)
        if np.isfinite(q_move) and finite.any():
            # Median over the *searchable* windows, not the median of every window:
            # the excluded ones sit near the query by construction and would drag the
            # reference toward a scale the reader never sees.  §BX also drops the
            # censored windows here, so `ref` and `scale` describe exactly the
            # population the match is scored within.
            eligible = finite & ~exclusion_mask(np.arange(len(moves), dtype=np.int64),
                                                query, exclusion_margin)
            if vm.size == moves.size:
                eligible &= vm
            pool = moves[eligible] if eligible.any() else moves[finite]
            scale = float(np.std(pool))
            # A zero scale means every window moved by the same amount, so there is no
            # amplitude signal to add and dividing by it would manufacture one.
            if scale > 1e-12:
                ref = float(np.median(all_dists[finite])) if finite.any() else 0.0
                if ref > 1e-12:
                    penalty = amplitude_weight * ref * (moves - q_move) / scale
                    all_dists = np.sqrt(
                        all_dists**2 + np.where(finite, penalty, 0.0)**2
                    )

    starts = np.arange(len(all_dists), dtype=np.int64)
    n_total = len(starts)

    if n_total == 0:
        return MatchResult([], 0, 0, 0, query, SCORER, n_masked=n_masked)

    # §M: drop the query and its neighbours, then suppress the rest by separation.
    # `n_excluded` is the **total** removed from the pool, §M's rule and the mask
    # together, and it must agree with the size of the population the §E percentile
    # was computed over.  `n_masked` is the mask's own count and is *not* the marginal
    # number it contributed: the two rules overlap on windows near the query.  The UI
    # shows them side by side for that reason -- they must never be summed.
    excl = ~exclusion_mask(starts, query, exclusion_margin)
    keep = excl & vm
    n_excluded = n_total - int(keep.sum())
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return MatchResult([], n_total, n_excluded, 0, query, SCORER,
                           n_masked=n_masked)

    sep = query.length if nms_separation is None else nms_separation
    chosen = non_max_suppression(starts[idx], all_dists[idx], sep, k)

    # §E: the rank is taken over every *searchable* window, not just the ones that
    # survived -- so widening `k` cannot move it.  Crucially it is NOT taken over the
    # excluded windows: those overlap the query and so sit at distance ~0 by
    # construction, and counting them would make every match look rarer than it is.
    # That is precisely the inflation §E exists to prevent.
    surv_dists = all_dists[idx]
    finite = np.isfinite(surv_dists)
    population = surv_dists[finite]

    matches: list = []
    for i in chosen:
        s = int(starts[idx][i])
        d = float(all_dists[idx][i])
        pct = (
            float((population <= d).mean())
            if population.size and np.isfinite(d)
            else float("nan")
        )
        matches.append(Match(start=s, stop=s + query.length, distance=d, percentile=pct))

    return MatchResult(
        matches=matches,
        n_candidates=n_total,
        n_excluded=n_excluded,
        n_after_nms=len(matches),
        query=query,
        method=SCORER,
        n_masked=n_masked,
    )

