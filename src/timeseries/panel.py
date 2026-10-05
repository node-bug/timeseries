"""Cross-sectional pattern search over a multi-ticker panel.  PLAN.md §C/§M/§E.

The difference from :mod:`timeseries.matching`
----------------------------------------------
:mod:`timeseries.matching` answers *"when has this ticker looked like this before?"*
-- the query and the archive are the same contiguous series, so the candidate
population is the ticker's own history.

This module answers a different question: *"has anything in the S&P 500 looked like
this?"*  Three things follow from the candidate population being many tickers rather
than one, and each is handled explicitly:

1.  **A window may not span two tickers.**  A matrix profile is defined by sliding a
    query over *one contiguous series* (§C/§V).  If the panel were concatenated into
    a single array, a window at a ticker boundary would compare a tail of AAPL to a
    head of JPM, which is not a pattern that ever occurred.  Each ticker is therefore
    scored as its own series via ``stumpy.mass``, and the distances are concatenated
    with the coordinates needed to locate each one.  That is what
    :func:`find_panel_matches` does, and it is why this is not a call to
    :func:`timeseries.matching.find_matches`.

2.  **The candidate population is now heterogeneous.**  A percentile computed over
    all windows of one ticker is comparable to itself; over 500 tickers it is
    dominated by whichever names happened to be quiet that day, and a liquid mega-cap
    is a much harder bar to resemble than a thinly-traded small cap.  §E's honesty
    feature therefore reports three things side by side -- the global percentile, the
    **same-ticker** percentile (is this unusual for *this* stock?), and the
    **sector** percentile.  A window can be unremarkable globally and remarkable
    within its own ticker; those are different claims and only one of them is usually
    what the user meant.

3.  **Ties across tickers are the interesting case, not an error.**  When 400 names
    move together, the best match is not one company -- it is the factor.  So a
    *ticker cap* is applied on top of non-maximum suppression: no single ticker may
    occupy more than a few of the top slots.  Without it, a sector-wide move returns
    50 windows of the same three tickers and looks like 50 independent pieces of
    evidence when it is one.

Nothing here computes a forecast.  Forward returns and baselines stay in
:mod:`timeseries.forecast`, and a panel match is turned into a forecast by
:meth:`PanelSearch.run`, which delegates to the same
:func:`timeseries.forecast.conditional_forecast` the single-ticker path uses -- so
the baseline comparison cannot be skipped by using this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import matrix_profile as MP
from .features import build_features, finalize_features
from .forecast import MIN_MATCHES, conditional_forecast
from .matching import DEFAULT_AMPLITUDE_WEIGHT, forward_horizon_valid_mask
from .store import PanelStore, session_et

__all__ = [
    "PanelMatch",
    "PanelQuery",
    "PanelResult",
    "PanelSearch",
    "build_panel",
    "find_panel_matches",
]


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelQuery:
    """A pattern to look for anywhere in the panel, plus where it came from.

    ``ticker``/``start``/``stop`` are the query's own coordinates in its home series.
    They are *not* used to build the search vector, because the search vector is
    already normalised (see :func:`PanelSearch.prepare`).  They are used to stop a
    match from being the query itself: within the home ticker, candidates overlapping
    the query span are excluded, exactly as §M requires for a single series.
    """

    vector: np.ndarray
    ticker: str
    start: int
    stop: int
    label: str = ""

    @property
    def length(self) -> int:
        return int(self.stop - self.start)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class PanelMatch:
    """One accepted analogue from anywhere in the panel."""

    ticker: str
    start: int                 # bar index within that ticker's series
    stop: int
    session: str               # Eastern date the window starts in
    timestamp: pd.Timestamp
    distance: float
    percentile: float          # rank among ALL panel windows (§E)
    percentile_same_ticker: float
    percentile_same_sector: float
    n_windows_in_ticker: int
    n_windows_in_sector: int

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker, "session": self.session,
            "timestamp": self.timestamp, "start": self.start, "stop": self.stop,
            "distance": self.distance, "percentile": self.percentile,
            "percentile_same_ticker": self.percentile_same_ticker,
            "percentile_same_sector": self.percentile_same_sector,
        }


@dataclass
class PanelResult:
    """Matches plus the candidate population the ranks were computed against."""

    matches: list
    query: PanelQuery
    n_candidates: int
    n_tickers: int
    n_excluded: int
    method: str
    #: Candidates dropped because their forward horizon would cross a session
    #: closure (PLAN.md §BX).  Reported apart from ``n_excluded`` so a data-quality
    #: suppression stays distinguishable from the §M self-match guard.
    n_masked: int = 0
    tickers_represented: list = field(default_factory=list)
    sectors_represented: list = field(default_factory=list)

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_distinct_tickers(self) -> int:
        return len({m.ticker for m in self.matches})


# --------------------------------------------------------------------------- #
# Feature preparation
# --------------------------------------------------------------------------- #
def build_panel(
    bars: dict[str, pd.DataFrame],
    *,
    rolling: int = 20,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the per-ticker feature matrices a panel search scores against.

    Parameters
    ----------
    bars
        ``{ticker: frame}``, each frame sorted by timestamp with OHLC columns.
    rolling
        Rolling window for the z-score, matching
        :func:`timeseries.features.build_features`.

    Returns
    -------
    dict
        ``{ticker: (matrix, aligned_frame)}`` where ``matrix`` is ``(n, 2)`` finite
        float64 in the same order as ``aligned_frame``'s rows.  The alignment is the
        point: a match's bar index means nothing unless the timestamps at that index
        are carried with it, which is what makes a match reportable as a *date*.

    Tickers with too few clean bars to form one window are dropped rather than
    padded, so a short name cannot contribute matches built on three bars.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, frame in bars.items():
        if frame is None or len(frame) == 0:
            continue
        src = frame if "session" in frame.columns else _with_session(frame)
        feats = build_features(src, rolling=rolling)
        aligned, matrix, _ = finalize_features(feats, how="drop")
        if matrix.shape[0] < 2:
            continue
        out[sym] = (matrix, aligned)
    return out


def _with_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the Eastern session date when the caller did not supply it.

    A match is only useful if it can be reported as a *date*, and the session column
    is what makes that possible.  Deriving it here means :func:`find_panel_matches`
    can always label a match, rather than emitting a blank session for frames that
    arrived from somewhere other than :class:`~timeseries.store.PanelStore`.
    """
    out = frame.copy()
    out["session"] = session_et(out["timestamp"]).dt.strftime("%Y-%m-%d")
    return out


def _horizon_admissible(frame: pd.DataFrame, m: int,
                        max_horizon: Optional[int]) -> np.ndarray:
    """PLAN.md §BX: per window-start admissibility for one ticker's aligned frame.

    Built here rather than passed in because the panel frames already carry the
    timestamps their matrix rows correspond to, and a mask can only be trusted when
    whoever builds it is the same thing that aligned the rows.  Two sources of
    session label are accepted, in order of preference:

    * a ``session`` column, which :class:`~timeseries.store.PanelStore` supplies; and
    * timestamps bucketed into US/Eastern days, which covers a frame that arrived
      from :func:`build_features` via another route.

    Returns all-``True`` when ``max_horizon`` is ``None`` (correction off) or the
    frame carries neither column, so a frame with no date information is treated as
    uncensored rather than silently discarded.

    **Daily must not be censored this way at all.**  §BX exists to stop a *window*
    from spanning two trading sessions, because a 1-minute window that crosses the
    close contains an overnight gap and its forward return is meaningless.  But a
    daily bar **is** a whole session: consecutive bars are consecutive sessions by
    definition, so "does this window span two sessions?" is true for every window
    of length > 1.

    Measured, and this is the whole bug: AAPL has 11,543 session boundaries in
    11,544 bars on daily — one per bar — against 20 in 7,920 bars intraday, one per
    390.  Applying §BX unchanged on daily admitted **0 of 11,515 windows at every
    horizon**, so a daily search reported `ok=True` with 1,284,802 candidates
    scored and **zero** matches, at any ``k``, silently, for every ticker.

    The fix is to recognise that the correction is vacuous rather than to tune it.
    Censoring is skipped when a window can only ever span multiple sessions, which
    is detected by asking whether *any* run of ``m`` consecutive bars stays inside
    one session.  If none does, the rule has nothing to say and every window is
    admissible.  That is honest: on daily there is no in-session window to protect.

    ``forward_horizon_valid_mask`` is imported **into this function's scope**, not
    only where :func:`find_panel_matches` happens to import it.  A function-local
    import in a *different* function does not put the name in module scope, so
    calling it from here raised ``NameError`` at runtime -- a failure that only
    appeared once a frame actually needed censoring, and never on the uncensored
    path that returns early.
    """
    from .matching import forward_horizon_valid_mask

    n_windows = max(0, len(frame) - int(m) + 1) if frame is not None else 0
    if max_horizon is None or frame is None or len(frame) == 0:
        return np.ones(n_windows, dtype=bool)
    if "session" in frame.columns:
        keys = frame["session"].astype("string")
    elif "timestamp" in frame.columns:
        keys = session_et(pd.to_datetime(frame["timestamp"], utc=True,
                                         errors="coerce")).astype("string")
    else:
        return np.ones(n_windows, dtype=bool)

    boundary = (keys.to_numpy()[:-1] != keys.to_numpy()[1:]).astype(bool)
    if not _censoring_can_bind(boundary, int(m)):
        # Every window spans >= 1 boundary, so §BX would reject everything.
        return np.ones(n_windows, dtype=bool)
    return forward_horizon_valid_mask(
        np.concatenate([[False], boundary]), int(m), int(max_horizon)
    )


def _censoring_can_bind(boundary: np.ndarray, m: int) -> bool:
    """Is there any run of ``m`` consecutive bars that stays inside one session?

    ``True`` when at least one window would survive the censoring, ``False`` when
    the rule rejects all of them.  The daily case is the second: one boundary per
    bar, so no run of two or more bars is ever intra-session.

    Indices follow :func:`~timeseries.matching.forward_horizon_valid_mask`: window
    start ``s`` spans bars ``[s, s + m - 1]``, and ``boundary[i]`` marks the bar at
    ``i`` as opening a session.  So the window is intra-session exactly when no
    boundary lies in ``(s, s + m - 1]`` -- the boundary at ``s`` is the window's
    own first bar and does not make it span anything.

    Measured rather than inferred, because the two cases differ by a factor of 390
    in boundary density and the distinction is invisible in the code: both have a
    ``session`` column, both have boundaries, and only the *density* says whether
    the rule can do anything.
    """
    m = int(m)
    if m <= 1:
        return True
    n = int(boundary.size)
    if n == 0:
        return True
    n_starts = max(0, n - m + 1)
    if n_starts == 0:
        return False
    # ``crossed[k]`` counts boundaries at indices < k.
    crossed = np.concatenate([[0], np.cumsum(boundary)])
    starts = np.arange(n_starts, dtype=np.int64)
    last = starts + m - 1                      # inclusive window end
    if last.max() >= n:                       # cannot happen, but do not guess
        return True
    inside = (crossed[last + 1] - crossed[starts + 1]) == 0
    return bool(inside.any())


def _log_returns(frame: pd.DataFrame) -> Optional[np.ndarray]:
    """Raw per-bar log returns for a feature-aligned panel frame.

    Mirrors ``Pipeline.log_returns``: prefers the ``log_return`` column
    ``build_features`` already computed, and falls back to differencing the log close.
    A non-positive close yields ``nan`` rather than ``inf``, so one bad print cannot
    turn an entire ticker's move profile into NaN.
    """
    if frame is None or len(frame) == 0:
        return None
    if "log_return" in frame.columns:
        r = pd.to_numeric(frame["log_return"], errors="coerce").to_numpy(dtype=float)
    else:
        c = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(c), prepend=np.nan)
    return np.where(np.isfinite(r), r, np.nan)


def _exclude_window_by_time(frame: pd.DataFrame, m: int,
                           span: tuple[pd.Timestamp, pd.Timestamp],
                           margin: int) -> tuple[Optional[np.ndarray], int]:
    """``(mask, count)`` over one ticker's window starts: those overlapping ``span``.

    The index-space counterpart of :func:`timeseries.matching.exclusion_mask`, for the
    case where the caller's query does not index this frame.  A window starting at
    ``i`` occupies bars ``[i, i + m)``, so it overlaps ``[t0, t1)`` iff its first bar
    is before ``t1`` and its last bar is at or after ``t0``.

    ``margin`` widens the span by one window length on each side before testing, which
    is the same policy :func:`exclusion_mask` applies by default: the matrix-profile
    convention of ~m/4 suppresses only the query itself and still returns its
    immediate neighbours, which share ``m - 1`` bars with it and are not independent
    evidence either.

    **The mask is returned as well as the count, and the caller must apply the mask
    rather than the count.**  A count alone tempts a blanket "this ticker is out",
    which is not what §M means: it removes the query's *neighbourhood* and leaves the
    rest of that ticker's history in the pool.

    Timestamps are compared as UTC nanoseconds.  Casting a tz-aware ``Timestamp`` to
    ``np.datetime64`` silently drops the offset and the comparison then raises rather
    than falling back -- so the naive-looking version fails on exactly the UTC
    timestamps the archive stores.  Nanoseconds are also what the comparison actually
    is, so this is the direct expression of it.

    Returns ``(None, 0)`` when the frame carries no usable ``timestamp`` column, which
    means "exclude nothing".  That is the safe direction: inventing an index-based
    guard here would reintroduce exactly the mis-addressing this function exists to
    avoid.  A NaT bar lands at the int64 minimum and fails both inequalities, so an
    unparsable stamp is never excluded and never wins a match.
    """
    n_starts = max(0, len(frame) - int(m) + 1)
    if n_starts == 0 or "timestamp" not in frame.columns:
        return None, 0
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None, 0

    t0, t1 = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")

    # Compared as **int64 nanoseconds since epoch**, not as datetimes.  Casting a
    # tz-aware ``Timestamp`` to ``np.datetime64`` silently drops the offset, and the
    # comparison against a tz-aware scalar then raises rather than falling back --
    # so the naive-looking version of this line fails on exactly the UTC timestamps
    # the archive stores.  Nanoseconds are also what the comparison actually is, so
    # this is the direct expression of it.
    pad_ns = int(margin) * 60 * 1_000_000_000
    lo_ns, hi_ns = t0.value - pad_ns, t1.value + pad_ns

    # NaT becomes the int64 minimum, so an unparsable bar fails both inequalities and
    # is never excluded.
    as_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")

    # `end` is the timestamp of each window's LAST bar, so the overlap test is a
    # plain two-sided comparison on (start <= hi) & (end >= lo).
    start_at = as_ns[:n_starts]
    end_at = as_ns[int(m) - 1:int(m) - 1 + n_starts]
    mask = (start_at <= hi_ns) & (end_at >= lo_ns)
    return mask, int(mask.sum())


def _slice(valid_arr: np.ndarray, sym: str,
           dists: dict[str, np.ndarray]) -> np.ndarray:
    """Recover one ticker's boolean slice from the pooled candidate array.

    The pool is built by concatenating tickers in ``dists`` order, so a per-ticker view
    is a contiguous slice of the same width as that ticker's profile.  Reconstructing it
    from the widths (rather than from ``owner_arr == sym``) keeps this correct even
    though the *labels* are object arrays and not used for indexing.
    """
    start = 0
    for other, d in dists.items():
        if other == sym:
            break
        start += d.size
    return valid_arr[start:start + dists[sym].size]


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def find_panel_matches(
    query: np.ndarray,
    panel: dict[str, tuple[np.ndarray, pd.DataFrame]],
    *,
    sectors: Optional[dict[str, str]] = None,
    home_ticker: Optional[str] = None,
    query_span: Optional[tuple[int, int]] = None,
    k: int = 20,
    max_per_ticker: int = 3,
    nms_separation: Optional[int] = None,
    top_n: int = 400,
    max_horizon: Optional[int] = None,
    amplitude_weight: float = DEFAULT_AMPLITUDE_WEIGHT,
    moves: Optional[dict[str, np.ndarray]] = None,
    q_move: Optional[float] = None,
    exclude_times: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    exclude_ticker: Optional[str] = None,
) -> PanelResult:
    """Score one query against every ticker in the panel, best-first.

    Each ticker's contiguous feature matrix gets its own ``stumpy.mass`` pass (§C),
    so a window never straddles a ticker boundary.  The passes are then pooled into
    one candidate population, and §E's rank is computed over that whole population --
    not over a per-ticker shortlist, because a shortlist would make every rank look
    good by construction, which is the multiple-comparisons trap §E exists to avoid.

    The three percentiles
    ---------------------
    * ``percentile``            -- rank among every window in the panel.
    * ``percentile_same_ticker`` -- rank within the query's own ticker, i.e. "is this
      unusual *for this stock*".  Computed from a full distance profile over that
      ticker, not from the retrieved matches, so it is available even when the panel
      cap suppressed most of that ticker's candidates.
    * ``percentile_same_sector`` -- rank within the query's GICS sector.

    Selection then applies, in order: the PLAN.md §BX forward-horizon mask, the
    home-ticker exclusion zone (§M), the top-``top_n`` prefilter, non-maximum suppression
    within a ticker, and finally the per-ticker cap so a factor move cannot fill the
    whole result.

    ``max_horizon`` is the largest forward horizon the caller will report.  Windows
    whose horizon would cross a session closure are dropped before ranking, so the
    §E percentiles describe exactly the population the matches were drawn from.  It
    defaults to ``None`` -- correction **off** -- because :meth:`PanelIndex.forecast`
    and :meth:`PanelIndex.search` are separate calls and only the former knows the
    horizon; set it from there, or pass it explicitly.

    ``amplitude_weight`` / ``moves``
        The same **size-of-move** term :func:`timeseries.matching.find_matches`
        applies, for the same reason: STUMPY's distance is scale-free, so a panel
        search on pure shape would return a dead-calm micro-cap swing as readily as
        a violent one, and on a chart the reader is shown those are visually opposite.

        **This defaults to ``DEFAULT_AMPLITUDE_WEIGHT`` -- off-by-default is not
        available here, deliberately.**  The single-ticker path defaults to 1.0 for
        the measured reason documented on that constant, and this module scores the
        same two features with the same scorer, so leaving the term off would make
        one ``amplitude_weight`` mean two different metrics depending on which tab
        produced the number.  The app's own contract is that the dial is "the same
        for every search in the app"; a cross-sectional search that ignored it would
        break that in the one direction that is hardest for a reader to notice,
        because the returned matches would still look plausible.

        ``moves`` carries the per-ticker realised-move profiles, already computed by
        the caller via :func:`timeseries.matrix_profile.amplitude_profile`.  They are
        an *input* rather than something computed here on purpose: a panel score has
        to be identical across reruns, and deriving the move profile from the panel
        feature matrix on every call would both cost a second pass per ticker and
        invite a silent drift from the scale the caller measured.  ``None`` with a
        non-zero weight scores pure shape, exactly as ``amplitude_weight=0`` does.

        The scale is the pooled standard deviation of realised moves across the
        **whole panel**, not per ticker, and that is a real difference from the
        single-ticker path rather than an oversight.  A per-ticker scale would
        measure each name in units of its own volatility and make a routine 0.3%
        wobble in a quiet name score as a better size-match than a 0.3% move in a
        liquid one -- which is the opposite of what "did price move like this"
        means, and would systematically bias the panel toward thin names.  One
        pooled scale asks the same question of every candidate.

    ``exclude_times`` / ``exclude_ticker``
        A **timestamp-based** self-match guard, for callers whose query lives in a
        different series than the panel.

        ``query_span`` is an index range, so it is only meaningful against the home
        ticker's own feature matrix.  When a caller builds its query from a *live
        fetch* while the panel's bars come from the last archive sync, the two index
        spaces do not correspond: passing ``query_span`` would hand
        ``exclusion_mask`` a range that addresses unrelated bars and suppress a
        random region of that ticker, silently.  Leaving ``home_ticker`` empty avoids
        that but re-opens §M in its original form -- **measured on the live archive,
        the panel then returns the query's own window at distance 0.000 as the top
        match whenever the query's ticker is in the panel**, which is precisely the
        result the single-ticker path exists to prevent.

        ``exclude_times`` closes it properly, because the *clock* is common to both
        series even when their indices are not.  Every candidate window overlapping
        ``[t0, t1)`` in the given ticker is dropped, so the query cannot match
        itself no matter how the two frames are indexed.  The margin is one window
        length on each side, matching :func:`exclusion_mask`'s default, so a window
        that shares almost all its bars with the query is rejected too.

        This is applied *before* the percentile pool is built, so the reported rank
        describes a population from which the query was genuinely removed -- counting
        the query's own window would make every remaining match look rarer than it
        is, which is the inflation §E exists to prevent.
    """
    from .matching import non_max_suppression, exclusion_mask

    q = np.asarray(query, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError(f"query must be 1-D or 2-D with at least one bar, got shape {q.shape}")
    m = q.shape[0]

    sectors = sectors or {}
    sep = m if nms_separation is None else int(nms_separation)
    scorer = "stumpy"

    # ---- Stage 1: one distance profile per ticker ------------------------- #
    dists: dict[str, np.ndarray] = {}
    for sym, (matrix, _frame) in panel.items():
        if matrix.shape[0] < m:
            continue
        if matrix.shape[1] != q.shape[1]:
            raise ValueError(
                f"ticker {sym} has {matrix.shape[1]} feature channels, query has {q.shape[1]}"
            )
        d = MP.distance_profile(np.ascontiguousarray(q), np.ascontiguousarray(matrix))
        if d.size:
            dists[sym] = d

    if not dists:
        return PanelResult([], PanelQuery(q, home_ticker or "", 0, m), 0, 0, 0, scorer)

    # ---- Stage 1b: timestamp-based self-exclusion (§M) --------------------- #
    # Built here, before the candidate pool, so the rank cannot be computed over a
    # population that still contains the query itself.  See ``exclude_times``.
    time_mask: Optional[np.ndarray] = None
    time_excluded = 0
    if exclude_times is not None and exclude_ticker and exclude_ticker in dists:
        time_mask, time_excluded = _exclude_window_by_time(
            panel[exclude_ticker][1], m, exclude_times, sep
        )

    # ---- The candidate population ----------------------------------------- #
    # Stacking every ticker's profile into one array is what makes the global rank
    # possible.  `owner` / `offset` carry each window's coordinates back out.
    all_d: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    offset: list[np.ndarray] = []
    admissible: list[np.ndarray] = []
    for sym, d in dists.items():
        n = len(d)
        all_d.append(d)
        owner.append(np.full(n, sym, dtype=object))
        offset.append(np.arange(n, dtype=np.int64))
        # PLAN.md §BX.  Computed per ticker from that ticker's own aligned frame, so
        # no caller has to supply anything: the panel frames are already aligned to
        # their matrix rows, which is the only thing that makes an index-level mask
        # sound in the first place.
        admissible.append(_horizon_admissible(panel[sym][1], m, max_horizon))
    pooled = np.concatenate(all_d)
    owner_arr = np.concatenate(owner)
    offset_arr = np.concatenate(offset)
    valid_arr = np.concatenate(admissible)
    n_candidates = int(pooled.size)
    n_excluded_time = 0

    if time_mask is not None and time_excluded:
        # Folded into ``valid_arr`` rather than kept separate, so the §E rank pool,
        # the per-ticker pools and the survivor selection all see the same
        # population.  ``n_candidates`` deliberately stays the *pre-exclusion* count:
        # it is the size of the archive's window population, which is what the UI
        # reports as "windows scored", and the count the reader can verify against
        # the archive.
        #
        # ``&=`` rather than ``= False``.  Assigning False across the whole ticker's
        # slice wiped all 821 of its windows and left T0 unrepresented in the results
        # -- an exclusion that removes the entire ticker, which is not §M at all but
        # looked like it was working, because the self-match was gone.  Only the
        # starts the mask actually marks may be dropped.
        idx_time = np.flatnonzero(owner_arr == exclude_ticker)
        valid_arr[idx_time[time_mask]] = False
        n_excluded_time = int(time_excluded)
    finite = np.isfinite(pooled)

    # ---- Amplitude: the component the normalised distance cannot see -------- #
    # Applied to the pooled distances *before* anything reads them, and therefore
    # before the §E rank pool is built, so the percentile describes exactly the
    # population the matches were scored within.  Adding it afterwards would leave
    # the rank describing a different metric than the one it is reported against --
    # the defect the single-ticker path calls out when it insists the penalty go
    # into the *squared* distance rather than be applied as a post-hoc tie-break.
    if amplitude_weight and moves and q_move is not None and np.isfinite(q_move):
        stack = np.concatenate([np.asarray(moves[s], dtype=float) for s in dists
                                if s in moves])
        stack = stack[np.isfinite(stack)]
        if stack.size:
            scale = float(np.std(stack))
            # Zero spread means every window moved the same amount, so there is
            # no amplitude signal to add and dividing by it would manufacture one.
            if scale > 1e-12:
                ref = float(np.median(pooled[finite & valid_arr])) \
                    if (finite & valid_arr).any() else 0.0
                if ref > 1e-12:
                    pen_parts = []
                    for sym, d in dists.items():
                        mv = np.asarray(moves.get(sym, np.full(d.shape, np.nan)),
                                        dtype=float)
                        p = amplitude_weight * ref * (mv - q_move) / scale
                        pen_parts.append(np.where(np.isfinite(mv), p, 0.0))
                    pooled = np.sqrt(pooled**2 + np.concatenate(pen_parts)**2)
                    finite = np.isfinite(pooled)

    if not finite.any():
        return PanelResult([], PanelQuery(q, home_ticker or "", 0, m), n_candidates,
                           len(dists), 0, "stumpy")

    # Global rank over the whole panel, restricted to admissible windows.  PLAN.md
    # §BX: a window the matcher could never have returned must not be part of the
    # distribution the returned windows are ranked against, or the rank describes a
    # population that does not exist.
    rank_pool = np.sort(pooled[finite & valid_arr])
    n_ranked = int(rank_pool.size)

    def global_pct(value: float) -> float:
        if not np.isfinite(value):
            return float("nan")
        lo = int(np.searchsorted(rank_pool, value, side="left"))
        return float(lo / n_ranked)

    # Per-ticker and per-sector distributions, for the two local ranks.  Each is
    # restricted to that subgroup's admissible windows, so "rarest for this stock"
    # means rarest among the windows this stock could actually have contributed
    # (§BX) -- otherwise a stock that happens to trade around the clock would look
    # uniformly unusual at the open.
    per_ticker: dict[str, np.ndarray] = {}
    per_ticker_valid: dict[str, np.ndarray] = {}
    for sym, d in dists.items():
        v = _slice(valid_arr, sym, dists)
        per_ticker_valid[sym] = v
        vals = d[np.isfinite(d) & v]
        if vals.size:
            per_ticker[sym] = np.sort(vals)
    home_sector = sectors.get(home_ticker or "")
    sector_pool = np.concatenate([
        d[np.isfinite(d) & per_ticker_valid[sym]] for sym, d in dists.items()
        if home_sector and sectors.get(sym) == home_sector
    ]) if home_sector else np.array([])
    sector_pool = np.sort(sector_pool)

    def local_pct(pool: np.ndarray, value: float) -> float:
        """Rank within ``pool``, using the same convention as the global rank.

        ``side="left"`` on both sides, deliberately.  The global rank counts windows
        strictly closer than this one; if the local rank instead counted ties, the
        two percentiles would answer slightly different questions and a window that
        tied for the best score of its own ticker could come back *rarer* globally
        than it was locally -- which is arithmetically impossible and reads as a bug
        in the number a reader trusts most.
        """
        if pool.size == 0 or not np.isfinite(value):
            return float("nan")
        return float((np.searchsorted(pool, value, side="left")) / pool.size)

    # ---- Stage 2: exclusion, prefilter ------------------------------------ #
    keep = valid_arr.copy()
    # ``valid_arr`` carries *two* different suppressions by the time it gets here:
    # the §BX forward-horizon mask and, if requested, the §M timestamp self-match
    # guard.  They are counted apart because they mean different things to a reader:
    # one is a data-quality correction the app applied, the other is the app refusing
    # to let the query match itself.  Counting the §M windows as "masked" would
    # inflate the §BX figure and make the forward-return correction look bigger than
    # it is.
    n_masked = int((~valid_arr).sum()) - n_excluded_time
    n_excluded = int(n_excluded_time)
    if home_ticker and home_ticker in dists and query_span is not None:
        q0, q1 = query_span
        starts = offset_arr[owner_arr == home_ticker]
        span_mask = exclusion_mask(starts, _Span(q0, q1), sep)
        idx_home = np.flatnonzero(owner_arr == home_ticker)
        n_excluded += int(span_mask.sum())
        keep[idx_home[span_mask]] = False

    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return PanelResult([], PanelQuery(q, home_ticker or "", 0, m), n_candidates,
                           len(dists), n_excluded, "stumpy", n_masked=n_masked)

    if top_n and idx.size > top_n:
        best = idx[np.argsort(pooled[idx], kind="stable")[:top_n]]
        idx = np.sort(best)

    # ---- Stage 3: NMS within a ticker, then a per-ticker cap ------------- #
    chosen_idx: list[int] = []
    per_ticker_used: dict[str, int] = {}
    for sym in sorted(dists):
        sel = idx[owner_arr[idx] == sym]
        if sel.size == 0:
            continue
        take = non_max_suppression(offset_arr[sel], pooled[sel], sep, k)
        chosen = sel[take]
        if max_per_ticker and max_per_ticker > 0:
            chosen = chosen[:max_per_ticker]
        for c in chosen:
            per_ticker_used[sym] = per_ticker_used.get(sym, 0) + 1
        chosen_idx.extend(chosen.tolist())

    if not chosen_idx:
        return PanelResult([], PanelQuery(q, home_ticker or "", 0, m), n_candidates,
                           len(dists), n_excluded, "stumpy", n_masked=n_masked)

    chosen_idx_arr = np.asarray(chosen_idx, dtype=np.int64)
    # Best-first ordering is load-bearing: callers label these "match #1", "#2"...
    chosen_idx_arr = chosen_idx_arr[np.argsort(pooled[chosen_idx_arr], kind="stable")][:k]

    sector_n_windows = int(
        sum(per_ticker[x].size for x in dists
            if home_sector and sectors.get(x) == home_sector)
    ) if home_sector else 0

    matches: list[PanelMatch] = []
    for ci in chosen_idx_arr:
        sym = str(owner_arr[ci])
        s = int(offset_arr[ci])
        d = float(pooled[ci])
        _matrix, frame = panel[sym]
        ts = frame["timestamp"].iloc[s] if s < len(frame) else pd.NaT
        sess = frame["session"].iloc[s] if s < len(frame) and "session" in frame else pd.NaT
        matches.append(PanelMatch(
            ticker=sym, start=s, stop=s + m,
            session=str(pd.Timestamp(sess).date()) if pd.notna(sess) else "",
            timestamp=ts, distance=d,
            percentile=global_pct(d),
            percentile_same_ticker=local_pct(per_ticker.get(sym, np.array([])), d),
            percentile_same_sector=local_pct(sector_pool, d),
            n_windows_in_ticker=int(dists[sym].size),
            n_windows_in_sector=sector_n_windows,
        ))

    return PanelResult(
        matches=matches,
        query=PanelQuery(q, home_ticker or "", *(query_span or (0, m))),
        n_candidates=n_candidates,
        n_tickers=len(dists),
        n_excluded=n_excluded,
        method="stumpy",
        n_masked=n_masked,
        tickers_represented=sorted({m.ticker for m in matches}),
        sectors_represented=sorted({
            sectors.get(m.ticker, "?") for m in matches if sectors.get(m.ticker)
        }),
    )


@dataclass
class _Span:
    """Minimal stand-in for :class:`timeseries.matching.Query` for the exclusion mask.

    ``exclusion_mask`` only reads ``query.start``, ``query.stop`` and
    ``query.length``; this supplies them without re-normalising anything, because the
    exclusion zone is a question about bar positions, not about the pattern.
    """

    start: int
    stop: int

    @property
    def length(self) -> int:
        return int(self.stop - self.start)


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #
class PanelSearch:
    """Cross-sectional search over a :class:`~timeseries.store.PanelStore`.

    This is the object the UI and any script should use: it owns the feature build,
    the per-ticker matrix cache, and the forecast-with-baseline wrapper, so a caller
    cannot accidentally search the panel *without* the baseline that §D requires.
    """

    def __init__(self, store: PanelStore, *, rolling: int = 20,
                 sectors: Optional[dict[str, str]] = None,
                 tickers: Optional[Sequence[str]] = None) -> None:
        self.store = store
        self.rolling = int(rolling)
        self.sectors = dict(sectors or {})
        self._tickers = list(tickers) if tickers else None
        self._panel: Optional[dict[str, tuple[np.ndarray, pd.DataFrame]]] = None
        self._bars: Optional[dict[str, pd.DataFrame]] = None
        self._close: Optional[dict[str, np.ndarray]] = None
        #: Realised-move profiles, keyed on window length.  See :meth:`moves`.
        self._moves: dict[int, dict[str, np.ndarray]] = {}

    # -- data ------------------------------------------------------------- #
    @property
    def bars(self) -> dict[str, pd.DataFrame]:
        if self._bars is None:
            self._bars = self.store.per_ticker(tickers=self._tickers)
        return self._bars

    @property
    def panel(self) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
        if self._panel is None:
            self._panel = build_panel(self.bars, rolling=self.rolling)
        return self._panel

    def close(self, ticker: str) -> np.ndarray:
        """The ticker's raw close series, in **raw bar** index space.

        Kept for callers that want the untrimmed series (the full tape, including the
        warm-up rows that have no features).  **Do not index it with a match's
        ``start``/``stop``** -- those address :attr:`panel`, whose frames are shorter
        by the feature warm-up.  Use :meth:`close_aligned` for anything a match
        points at.
        """
        if self._close is None:
            self._close = {s: f["close"].to_numpy(dtype=float) for s, f in self.bars.items()}
        return self._close.get(ticker, np.array([]))

    def close_aligned(self, ticker: str) -> np.ndarray:
        """The ticker's close series in the **same index space as a match's span**.

        :attr:`panel` maps each ticker to ``(matrix, aligned_frame)``, and a
        :class:`PanelMatch`'s ``start``/``stop`` are row indices into *that* matrix.
        This returns ``aligned_frame``'s close column, so an index taken from a match
        addresses the bar it actually names.

        The distinction is not cosmetic.  :meth:`close` returns the raw bars, which
        are longer than the matrix by the feature warm-up -- **20 rows** at
        ``rolling=20``, but per-ticker rather than constant, because rows are dropped
        for non-finite values as well as for the warm-up.  Indexing the raw series
        with a match index therefore reads a bar roughly 20 positions earlier than
        intended: the forward return is measured from the wrong close and against the
        wrong bars, and nothing raises.  Measured on the archived panel, that is a
        0.05%-0.95% error in the anchor price alone, on every horizon of every match.

        ``self.panel`` is consulted rather than :attr:`bars` so the two index spaces
        are produced by the same object that defines the match indices, which is the
        only way they are guaranteed to correspond.
        """
        entry = self.panel.get(ticker)
        if entry is None:
            return np.array([], dtype=float)
        return entry[1]["close"].to_numpy(dtype=float)

    def ready(self) -> bool:
        """A panel needs at least two tickers and two windows' worth of candidates."""
        return len(self.panel) >= 2

    def session_of(self, ticker: str, index: int) -> str:
        frame = self.bars.get(ticker)
        if frame is None or index >= len(frame):
            return ""
        row = frame.iloc[index]
        return str(pd.Timestamp(row["session"]).date()) if pd.notna(row.get("session")) else ""

    def timestamp_of(self, ticker: str, index: int) -> pd.Timestamp:
        frame = self.bars.get(ticker)
        if frame is None or index >= len(frame):
            return pd.NaT
        return frame["timestamp"].iloc[index]

    # -- queries ----------------------------------------------------------- #
    def prepare(self, ticker: str, start: int, stop: int) -> Optional[PanelQuery]:
        """Build a query from a bar span in one ticker's feature matrix.

        The span is re-normalised per channel, matching
        :meth:`timeseries.matching.Query.from_span`, so the query carries no residual
        level from the larger series (§BC: query and candidate are normalised
        identically, and STUMPY re-normalises the query again internally).
        """
        from .matching import zscore

        entry = self.panel.get(ticker)
        if entry is None:
            return None
        matrix, _frame = entry
        if not 0 <= start < stop <= len(matrix):
            return None
        vec = matrix[start:stop]
        vec = np.stack([zscore(vec[:, c]) for c in range(vec.shape[1])], axis=-1)
        ts = self.timestamp_of(ticker, start)
        return PanelQuery(
            vector=vec, ticker=ticker, start=int(start), stop=int(stop),
            label=str(ts)[:19],
        )

    def latest_query(self, ticker: str, length: int = 60) -> Optional[PanelQuery]:
        """The most recent full window in one ticker -- the 'live' path."""
        entry = self.panel.get(ticker)
        if entry is None:
            return None
        n = len(entry[0])
        if n < length:
            return None
        return self.prepare(ticker, n - length, n)

    # -- the search + forecast --------------------------------------------- #
    def moves(self, length: int) -> dict[str, np.ndarray]:
        r"""Per-ticker realised-move profiles, keyed on window length.

        The amplitude term needs one of these per ticker, and recomputing
        :func:`~timeseries.matrix_profile.amplitude_profile` over ~500 tickers on
        every interactive search would double the cost of the one thing the Price
        tab runs on every rerun.  Cached because it is a pure function of
        ``(ticker bars, length)``, both fixed for a session.

        Two things here are load-bearing, and both were wrong in the first
        version of this method:

        * **The series is log *returns*, not close.**  ``amplitude_profile`` sums its
          input over a window, so the input has to accumulate.  Passing close prices
          would make the "realised move" the price *level* itself, so every candidate
          would be compared to the query on where it trades rather than how far it
          moved -- and, cross-sectionally, would rank a $700 name against a $4 one
          purely on nominal price.  Mirrors ``Pipeline.log_returns``.

        * **It is the feature-*aligned* frame, not the raw bars.**  ``build_panel``
          drops ~20 warm-up rows, so ``matrix`` is shorter than the raw bar frame and
          a profile built from the raw bars is longer than the distance profile it
          has to align with element-wise.  The aligned frame's rows are what the
          matrix rows correspond to, which is the same invariant
          ``_horizon_admissible`` relies on when it builds a per-ticker mask from
          ``panel[sym][1]``.  Using anything else here raised a broadcast
          ``ValueError`` the moment the warm-up was non-zero.
        """
        cached = self._moves.get(int(length))
        if cached is None:
            cached = {}
            for sym, (_matrix, frame) in self.panel.items():
                r = _log_returns(frame)
                if r is not None and r.size >= int(length):
                    cached[sym] = MP.amplitude_profile(r, int(length))
            self._moves[int(length)] = cached
        return cached

    def search(self, query: PanelQuery, *, k: int = 20, max_per_ticker: int = 3,
               nms_separation: Optional[int] = None,
               max_horizon: Optional[int] = None,
               amplitude_weight: float = DEFAULT_AMPLITUDE_WEIGHT,
               q_move: Optional[float] = None,
               exclude_times: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
               exclude_ticker: Optional[str] = None) -> PanelResult:
        """Score ``query`` against the panel, honouring the shared amplitude dial.

        ``amplitude_weight`` is threaded straight through to
        :func:`find_panel_matches` rather than being re-derived, so the number the
        sidebar's *Size of move* slider produces here and on the single-ticker tabs
        means the same thing.

        ``q_move`` is the query's own realised move, and it **cannot always be read
        off the home ticker**.  When the query was built by :meth:`prepare` its
        ``start`` indexes that ticker's feature matrix, so the move profile's entry
        is the right one.  When a caller built the vector from a *different* series --
        a live Yahoo fetch, as the Price tab does -- those indices address different
        bars, and reading them here would silently measure some unrelated ticker's
        window.

        ``exclude_times``/``exclude_ticker`` are the matching answer to the §M guard:
        see :func:`find_panel_matches`.  A caller in that position must pass both,
        because neither the index-based ``query_span`` nor the empty home ticker is
        available to it.
        """
        length = int(query.length)
        profiles = self.moves(length) if amplitude_weight else {}
        if q_move is None:
            home = profiles.get(query.ticker)
            if home is not None and 0 <= int(query.start) < home.size:
                q_move = float(home[int(query.start)])
        return find_panel_matches(
            query.vector, self.panel,
            sectors=self.sectors,
            home_ticker=query.ticker,
            query_span=(query.start, query.stop),
            k=k, max_per_ticker=max_per_ticker, nms_separation=nms_separation,
            max_horizon=max_horizon,
            amplitude_weight=amplitude_weight,
            moves=profiles or None,
            q_move=q_move,
            exclude_times=exclude_times,
            exclude_ticker=exclude_ticker,
        )

    def run(self, query: Optional[PanelQuery] = None, *, ticker: Optional[str] = None,
            k: int = 20, horizons=(5, 15, 30, 60), max_per_ticker: int = 3,
            n_baseline: int = 400, min_matches: int = MIN_MATCHES,
            block: Optional[int] = None, seed: int = 0,
            session_mask: bool = True,
            amplitude_weight: float = DEFAULT_AMPLITUDE_WEIGHT) -> dict:
        """Search the panel, then forecast *with* a baseline, across all tickers.

        The baseline is drawn from random windows in the **same** set of tickers the
        matches came from, so the comparison is like-for-like: a cross-sector
        forecast measured against a single-sector baseline would be meaningless.

        PLAN.md §BX: unlike :meth:`find_panel_matches`, the correction defaults to
        **on** here.  This method is where the horizons are known, and it is the only
        caller that produces a forecast -- a panel match whose forward return spans an
        overnight gap is exactly the defect §BX describes, so there is no defensible
        reason for the one entry point that forecasts to leave it uncorrected.
        """
        if not self.ready():
            return {"ok": False, "reason": "panel needs at least 2 tickers with enough bars"}

        q = query
        if q is None:
            q = self.latest_query(ticker or self._default_ticker())
        if q is None:
            return {"ok": False, "reason": "could not build a query; check the ticker and span"}

        res = self.search(q, k=k, max_per_ticker=max_per_ticker,
                          max_horizon=int(max(horizons)) if session_mask else None,
                          amplitude_weight=amplitude_weight)

        # Forward returns, per matched ticker.  A match is located in its own
        # ticker's price series, so its forward return must be measured there too.
        by_ticker: dict[str, list[int]] = {}
        for mat in res.matches:
            by_ticker.setdefault(mat.ticker, []).append(mat.start)
        fwd: dict[int, list[np.ndarray]] = {int(h): [] for h in horizons}
        for sym, starts in by_ticker.items():
            close = self.close_aligned(sym)
            if close.size == 0:
                continue
            out = _forward_returns(close, np.asarray(starts, dtype=np.int64), q.length, horizons)
            for h in horizons:
                fwd[int(h)].append(out[int(h)])
        forward = {int(h): (np.concatenate(fwd[int(h)]) if fwd[int(h)] else np.array([]))
                   for h in horizons}

        base = self._baseline(horizons, n=n_baseline, seed=seed, length=q.length)
        b = block if block is not None else q.length + max(horizons)
        forecasts = conditional_forecast(forward, base, block=b, seed=seed,
                                         min_matches=min_matches)
        return {
            "ok": True, "query": q, "result": res, "forecasts": forecasts,
            "forward": forward, "baseline": base, "horizons": horizons, "method": res.method,
        }

    def _default_ticker(self) -> str:
        if not self.panel:
            return ""
        # Most liquid by bar count is a defensible default and avoids a ticker with
        # a single session being the implicit subject of the query.
        return max(self.panel, key=lambda s: len(self.panel[s][0]))

    def _baseline(self, horizons, *, n: int, seed: int, length: int) -> dict:
        """Forward returns from random windows, drawn across all panel tickers.

        Parameters
        ----------
        length
            The window length to use.  This is the **query's** length, not the panel's
            shortest ticker: the control group has to be the same statistic over the
            same kind of window, or "lift" compares a 30-bar forecast to a 1,443-bar
            one.  Using ``min(bars)`` instead silently made every start invalid on
            every ticker, so the baseline came back empty and every lift was ``NaN``.

        Each ticker contributes windows in proportion to its length, so a ticker with
        2,000 bars is sampled more often than one with 200 -- matching how the matcher
        actually sees the panel.  The draw is a single seeded pass, so the baseline is
        reproducible for a given seed and a reader can re-derive it.

        PLAN.md §BX: the draw is restricted to the same admissible windows the matches
        came from, computed per ticker from that ticker's own aligned frame.  Draw the
        control from a wider pool than the treatment and any lift measured against it is
        partly an artefact of the treatment's censoring rather than of the pattern.
        """
        rng = np.random.default_rng(seed)
        syms = list(self.panel)
        hs = [int(h) for h in horizons]
        if not syms or length <= 0:
            return {h: np.array([]) for h in hs}
        lengths = np.array([len(self.panel[s][1]) for s in syms], dtype=float)
        probs = lengths / lengths.sum() if lengths.sum() > 0 else None

        picks = rng.choice(len(syms), size=n, p=probs)
        out: dict[int, list[np.ndarray]] = {h: [] for h in hs}
        maxh = max(hs)
        for i in np.unique(picks):
            sym = syms[int(i)]
            _matrix, frame = self.panel[sym]
            # ``close_aligned``, not ``close``: ``_horizon_admissible`` below is built
            # from ``frame``, so the starts it returns are row indices into that same
            # aligned frame.  Pairing them with the raw series would measure the
            # control group's forward returns ~20 bars from the treatment's, making
            # every "lift" partly an artefact of the offset.  See ``close_aligned``.
            close = self.close_aligned(sym)
            n_bars = len(close)
            # A random window is only valid if its whole forward horizon exists, so
            # the draw is bounded by the *end* of the series, not just its length.
            hi = n_bars - length - maxh - 1
            if hi <= 0:
                continue
            count = int((picks == i).sum())
            allowed = np.flatnonzero(
                _horizon_admissible(frame, length, maxh)[:hi]
            )
            if allowed.size == 0:
                continue
            starts = allowed[rng.integers(0, allowed.size, size=count)]
            res = _forward_returns(close, starts, length, hs)
            for h in hs:
                out[h].append(res[h])
        return {h: (np.concatenate(out[h]) if out[h] else np.array([])) for h in hs}


def _forward_returns(close: np.ndarray, starts: np.ndarray, m: int, horizons) -> dict:
    """Forward log-return from the last bar of each window, per §Z1.

    Delegates to the single implementation in :mod:`timeseries.matching` so the
    anchor rule (the return must start *after* the window, never inside it) is defined
    once.  Re-deriving it here would be a second place for the §Z1 bug to reappear.
    """
    from .matching import forward_returns as _fwd
    return _fwd(close, starts, m, horizons)
