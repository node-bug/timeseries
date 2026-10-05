"""End-to-end pipeline: load bars -> features -> match -> forecast.

This is the module the CLI and the Streamlit app both call.  Its job is to make the
statistical requirements of PLAN.md §D/§E unavoidable -- in particular, the baseline
comparison.  There is deliberately no code path that produces a forecast without also
producing the random-window baseline beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .features import build_features, finalize_features, quality_report
from .forecast import MIN_MATCHES, conditional_forecast, Forecast
from .store import OHLCV, session_et
from .timeframes import DEFAULT_TIMEFRAME, get_timeframe, resolve_timeframe
from . import matching as M
from . import matrix_profile as MP

__all__ = [
    "Pipeline",
    "load_bars",
    "normalize_bars",
    "session_boundary_mask",
    "horizons_for",
    "DEFAULT_HORIZONS",
    "DEFAULT_K",
    "SCORER",
]

#: Forward-return horizons, in bars, for the 1-minute timeframe.
#:
#: Kept as a module constant rather than only in :mod:`timeseries.timeframes` because
#: it is re-exported and imported directly by callers and tests that are explicitly
#: about intraday.  :func:`horizons_for` is the entry point for anything that has to
#: work at both resolutions.
DEFAULT_HORIZONS = get_timeframe("1m").horizons

# PLAN.md §BE: `k` must default to at least ``MIN_MATCHES`` (30), otherwise every
# default run trips the §E evidence gate and reports "insufficient evidence" with
# lift/p-value NaN -- a forecast that is suppressed out of the box.
DEFAULT_K = 50


def horizons_for(timeframe: object = DEFAULT_TIMEFRAME) -> tuple:
    """Forward-return horizons, in bars, for ``timeframe``.

    In bars rather than in wall-clock time, because that is what the matcher
    indexes with.  The daily horizons are therefore *longer* in calendar terms than
    the intraday ones at the same numeral: 5 daily bars is a trading week, where 5
    intraday bars is five minutes.  Both sets are "a few bars through a few months",
    which is the property the Forecast tab's horizon selector actually needs.
    """
    return resolve_timeframe(timeframe).horizons

# STUMPY is the one and only scorer (PLAN.md §C/§S), so there is nothing left to pick.
# The name is re-exported here because the pipeline is the entry point every caller
# already imports from.
SCORER = M.SCORER


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw bar frame: UTC timestamps, sorted, de-duplicated, numeric prices.

    Split out of :func:`load_bars` so that bars arriving *in memory* -- downloaded by
    :mod:`timeseries.fetch`, or read out of the :mod:`timeseries.store` archive -- go
    through byte-identical preparation to bars read from a CSV.  Duplicating these
    steps per source is how a fetched ticker and an on-disk archive end up subtly
    disagreeing about what a duplicate timestamp means.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.loc[df["timestamp"].notna()].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for c in OHLCV:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_bars(path: str) -> pd.DataFrame:
    """Load a bar CSV, parse timestamps to UTC, sort and de-duplicate."""
    return normalize_bars(pd.read_csv(path))


def session_boundary_mask(timestamps: "pd.Series | Sequence") -> np.ndarray:
    """Boolean per bar: does the bar *start a new session* relative to the previous one?

    PLAN.md §BX.  Two different, easily-conflated questions live on a session boundary,
    and only one of them corrupts a reported number:

    1. **A window that straddles a boundary** contains one gap bar, and that bar is
       *large*, not small -- measured on the live QQQ archive, its median ``|log
       return|`` is **3.6e-03** against **1.6e-04** for a typical bar, i.e. **22x**.
       That does distort the window's shape distance, but it distorts the *ranking*,
       and only **2 of 50** matched windows straddle at all (NMS's one-window
       separation rejects the rest).  Excluding them costs 31% of the candidate pool
       at L=60 and 62% at L=240, and moves the matched forward mean by less than the
       gap between two adjacent horizons.  **Not excluded.**

    2. **A window whose LAST bar is a session's final bar** has a forward return
       (§Z1 anchors on that last bar) that therefore measures the overnight move to
       the next opening print.  Measured on the same archive at L=60: mean forward
       return **+5.6e-04** against **+2.8e-05** for a window ending mid-session -- a
       **20x** inflation, and the app's own help text already warns that a straddling
       query "can manufacture a forward return out of the opening auction".  Only 20
       such windows exist, which is why it survived every earlier test.  **This is
       the one that matters.**

    So this helper marks session *starts*, and the caller uses it to exclude windows
    whose forward horizon would run across the closure.  Bar 0 is left unmarked: no
    forward return can reach before it.

    Sessions are US/Eastern days (``store.session_et``), not UTC days -- bucketing in
    UTC splits a session in half at 20:00 ET and would mark every evening bar as a
    boundary.
    """
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    keys = session_et(ts)
    out = np.zeros(len(keys), dtype=bool)
    if len(keys) > 1:
        out[1:] = keys.to_numpy()[1:] != keys.to_numpy()[:-1]
    return out


@dataclass
class Pipeline:
    """Holds the prepared series and exposes query operations over it."""

    bars: pd.DataFrame
    features: pd.DataFrame
    matrix: np.ndarray
    length: int
    ready: bool = True
    warnings: list = field(default_factory=list)
    _profile: Optional[MP.ProfileResult] = None
    timeframe: str = DEFAULT_TIMEFRAME

    @property
    def tf(self):
        """The resolved :class:`~timeseries.timeframes.Timeframe` this pipeline is on.

        Stored as a key rather than a resolved object so a ``Pipeline`` stays
        picklable and hashable-friendly for Streamlit's caches, and so a value written
        into session state by an older build still resolves.
        """
        return resolve_timeframe(self.timeframe)

    @classmethod
    def from_csv(cls, path: str, length: int = 60, rolling: Optional[int] = None,
                 timeframe: object = DEFAULT_TIMEFRAME) -> "Pipeline":
        return cls.from_frame(load_bars(path), length=length, rolling=rolling,
                              timeframe=timeframe)

    @classmethod
    def from_frame(cls, bars: pd.DataFrame, length: int = 60,
                   rolling: Optional[int] = None,
                   timeframe: object = DEFAULT_TIMEFRAME) -> "Pipeline":
        """Build the pipeline from bars already in memory.

        The in-memory twin of :meth:`from_csv`, for data that was never on disk --
        downloaded live by :func:`timeseries.fetch.fetch_ticker` or read from the
        :mod:`timeseries.store` archive.  Both constructors funnel into this one, so
        a fetched ticker is warmed up, cleaned and windowed exactly like a CSV, and
        the two cannot drift apart.

        ``rolling=None`` (the default) resolves the z-score base from ``timeframe``
        rather than from a literal, so a daily pipeline is not built with an intraday
        20-*minute* warm-up by accident.  An explicit ``rolling`` still wins, because
        the Backtest tab sweeps it deliberately.
        """
        tf = resolve_timeframe(timeframe)
        bars = normalize_bars(bars)
        if len(bars) < length + 100:
            return cls(bars=bars, features=bars, matrix=np.empty((0, 2)), length=length,
                       ready=False, timeframe=tf.key,
                       warnings=[f"only {len(bars)} bars; need >= {length + 100}"])

        feats = build_features(bars, rolling=rolling, timeframe=tf.key)
        frame, mat, _ = finalize_features(feats, how="drop")
        return cls(bars=frame, features=frame, matrix=mat, length=length, ready=True,
                   timeframe=tf.key)

    # -- introspection ---------------------------------------------------- #
    @property
    def close(self) -> np.ndarray:
        return self.bars["close"].to_numpy(dtype=float)

    @property
    def log_returns(self) -> np.ndarray:
        """Raw per-bar log returns, the series the amplitude term must be measured on.

        :func:`timeseries.matrix_profile.amplitude_profile` sums a window to recover its
        realised move, so it needs a difference series in accumulating units.  The
        ``return_z`` leg is a *rolling-z* of exactly this, and summing that instead
        divides every bar by its local volatility -- which makes a violent move in a
        quiet hour and a small move in a frantic one look alike, inverting the signal
        the penalty is supposed to measure.

        Rebased to the frame's own first bar, since bar indices here index the cleaned
        frame and the first row has no return.  A non-positive close yields ``nan``
        rather than ``inf``, so a bad print cannot poison a whole window's sum.
        """
        if "log_return" in self.bars.columns:
            r = pd.to_numeric(self.bars["log_return"], errors="coerce").to_numpy(dtype=float)
        else:
            c = self.close
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.diff(np.log(c), prepend=np.nan)
        return np.where(np.isfinite(r), r, np.nan)

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    def timestamp_at(self, index: int) -> pd.Timestamp:
        return self.bars["timestamp"].iloc[index]

    def quality(self) -> dict:
        return quality_report(self.bars, timeframe=self.timeframe)

    # -- querying --------------------------------------------------------- #
    def matrix_profile(self) -> Optional[MP.ProfileResult]:
        """The cached matrix profile for the current window length, if one exists.

        §S: the full profile is a batch/nightly artefact, not something to compute on a
        click, so this only returns a profile the caller has already built via
        :meth:`set_profile`.  The interactive path uses ``stumpy.match``-equivalent
        per-query scoring instead.
        """
        if self._profile is not None and self._profile.length == self.length:
            return self._profile
        return None

    def set_profile(self, res: MP.ProfileResult) -> None:
        """Attach a precomputed profile.  Rejected if it is for a different length.

        §T: a profile is only valid at the ``m`` it was built for, so accepting a
        mismatched one would silently answer a different question than the user asked.
        """
        if res.length != self.length:
            raise ValueError(
                f"profile length {res.length} != pipeline length {self.length}"
            )
        self._profile = res

    def build_profile(self, channel: int = 0) -> MP.ProfileResult:
        """Compute and cache the matrix profile for one feature channel.

        STUMPY's profile is per-channel.  The price leg (``return_z``, ``channel=0``)
        is the meaningful default; §U3's ``mstump`` would give a genuine cross-channel
        consensus, but it is markedly slower and is deferred by the plan.
        """
        series = np.ascontiguousarray(self.matrix[:, channel], dtype=float)
        res = MP.profile(series, self.length)
        self._profile = res
        return res

    def query_latest(self) -> M.Query:
        """Query from the most recent full window (the 'live' path)."""
        start = self.n_bars - self.length
        return M.Query.from_span(
            self.matrix, start, self.n_bars,
            label=str(self.timestamp_at(start))[:19],
        )

    def query_span(self, start: int, stop: int, label: str = "") -> M.Query:
        """Query from an explicit bar span (the manual-selection path, §L)."""
        return M.Query.from_span(
            self.matrix, start, stop,
            label=label or f"{str(self.timestamp_at(start))[:19]} .. {str(self.timestamp_at(stop - 1))[:19]}",
        )

    def match(self, query: M.Query, *, k: int = DEFAULT_K,
              amplitude_weight: float = M.DEFAULT_AMPLITUDE_WEIGHT,
              max_horizon: Optional[int] = None) -> M.MatchResult:
        """Find matches for ``query``.

        STUMPY slides over the contiguous feature matrix directly, so this skips
        materialising a window library entirely -- the O(n x m) materialisation was the
        single largest cost in the old pipeline, and dropping it removes both the memory
        spike and the interpreted scoring loop with it.

        ``amplitude_series`` is passed explicitly rather than left to ``find_matches``'
        fallback: the fallback would sum ``return_z``, which is a rolling-z of the real
        returns and therefore measures each bar relative to its *local* volatility --
        a defect that rewards wild windows.  :attr:`log_returns` is the series the
        amplitude term actually means.

        PLAN.md §BX: candidates whose forward horizon would cross a session closure are
        dropped by default.  Measured on the live QQQ archive, a window ending on a
        session's final bar reports a forward return **20x** larger than a window ending
        mid-session -- not because the market moved differently, but because the "h
        bars forward" it reports are really one overnight gap plus ``h - 1`` minutes of
        trading.  Pass ``max_horizon=None`` to switch the correction off;
        ``max_horizon`` defaults to the largest horizon the caller will forecast.
        """
        return M.find_matches(
            self.matrix, query, k=k,
            amplitude_weight=amplitude_weight,
            amplitude_series=self.log_returns,
            valid_mask=self.forward_horizon_mask(query.length, max_horizon),
        )

    def forward_horizon_mask(self, window_length: int,
                             max_horizon: Optional[int]) -> Optional[np.ndarray]:
        """PLAN.md §BX candidate mask; ``None`` when the correction is off or does
        not apply.

        Deliberately *not* a straddle filter.  A window that spans a session boundary
        contains one bar whose return is ~22x a typical bar's, so it does distort the
        ranking -- but only 2 of 50 matched windows straddle at all, and excluding the
        rest would cost 31% of the pool at L=60 and 62% at L=240 for a matched forward
        mean that moves less than the gap between two adjacent horizons.  The window
        that actually lies is the one whose *horizon* crosses the boundary, which
        inflates its forward return 20x.

        **The correction does not apply on daily, and this is the load-bearing
        reason the daily path works at all.**  §BX fixes a forward return that would
        otherwise measure an overnight closure: the horizon runs off the end of a
        session and the "next N bars" are really one overnight gap plus N-1 minutes of
        trading, inflating the return ~20x.

        On daily there is no such thing to fix.  A daily bar *is* a session, so the
        boundary mask is true for every bar, and :func:`session_boundary_mask` then
        marks every window as crossing a closure.  Measured on live QQQ daily data:
        ``6264 boundaries in 6265 bars``, and the mask admits **0 of 6206 candidates**
        at every horizon -- the search returns nothing at all.

        Silently, and in the most convincing direction available: zero matches reads
        as "this shape has no historical analogue", which is a claim about the archive
        rather than about a mask.  So the correction is skipped where its premise does
        not hold, rather than being left to produce an empty result.
        """
        if max_horizon is None or "timestamp" not in self.bars.columns:
            return None
        # No interior to cross: a daily horizon already measures whole sessions.
        if self.tf.bars_per_session == 1:
            return None
        boundary = session_boundary_mask(self.bars["timestamp"])
        return M.forward_horizon_valid_mask(boundary, window_length, max_horizon)

    def forward_returns(self, starts: Sequence[int], horizons=None,
                        window_length: Optional[int] = None) -> dict:
        """Forward returns from ``starts``, assuming a window of ``window_length``.

        ``window_length`` defaults to ``self.length`` but **must** be passed as the
        query's own length whenever the query does not have one.  The forward return is
        anchored on the last bar of the window (§Z1), so using the wrong length moves
        the anchor: for a 37-bar window evaluated as if it were 20 bars, every return is
        measured 17 bars too early and silently reports the wrong number rather than
        failing.
        """
        L = int(self.length if window_length is None else window_length)
        if horizons is None:
            horizons = self.tf.horizons
        return M.forward_returns(self.close, np.asarray(starts, dtype=np.int64),
                                 L, horizons)

    def baseline_returns(self, n: int, horizons=None, *, seed: int = 0,
                         block: int = 1, window_length: Optional[int] = None,
                         match_percentiles: Optional[np.ndarray] = None,
                         query: Optional[M.Query] = None,
                         valid_mask: Optional[np.ndarray] = None) -> dict:
        """Forward returns from RANDOM windows -- the §D control.

        Windows are drawn uniformly from the same region the matcher could see, and are
        filtered so the forward horizon exists.  This is the number a matched forecast
        has to beat to be interesting.

        ``window_length`` must match the length of the windows actually being compared,
        for the same reason as :meth:`forward_returns` -- a control group built from
        20-bar windows cannot be the control for a 37-bar matched set.  The control has
        to measure the same thing as the treatment, or the lift is not a lift.

        ``valid_mask`` applies the PLAN.md §BX admissibility filter to the control group
        as well.  It has to: a control that may include windows the treatment group
        excluded is not a control, it is a handicap, and any lift measured against it is
        partly an artefact of the treatment's own censoring.

        Parameters
        ----------
        match_percentiles, query
            When both are given, the control is drawn from a band of the **distance
            ranking** centred on each match's percentile rather than uniformly.

            PLAN.md §BW.  Forward return is correlated with distance rank -- the mean
            falls monotonically across deciles -- and the matched set is the extreme
            low-distance tail, so a uniform control compares the bottom ~1% of the
            ranking against a population spread over all of it.

        .. note::
           The measured cause of the §BU/§BV miscalibration is *not* this population
           mismatch.  It is that the matched mean is **unbiased but ~2.15x more variable**
           than an equivalent random draw (mean z = +0.08, sd z = 2.15), which no
           choice of baseline population can repair; see
           :func:`timeseries.forecast.selection_aware_test`, which is where the fix
           lives.  This banding is retained only as an optional, off-by-default
           refinement and is **not** applied by :meth:`run` -- measured, it does not
           reduce the placebo failure rate on its own.
        """
        rng = np.random.default_rng(seed)
        L = int(self.length if window_length is None else window_length)
        if horizons is None:
            horizons = self.tf.horizons
        hi = self.n_bars - L - max(horizons) - 1
        if hi <= 0:
            return {h: np.array([]) for h in horizons}

        # §BX: draw the control from the same admissible pool the treatment came from.
        vm = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        pool_starts = np.arange(hi, dtype=np.int64)
        if vm is not None:
            if vm.shape != (self.n_bars - L + 1,):
                raise ValueError(
                    f"valid_mask must have {self.n_bars - L + 1} entries, "
                    f"got {vm.shape}"
                )
            pool_starts = pool_starts[vm[:hi]]

        pct = None if match_percentiles is None else np.asarray(match_percentiles, float)
        if query is None or pct is None or pct.size == 0:
            picks = (rng.choice(pool_starts, size=n, replace=True)
                     if pool_starts.size else np.empty(0, dtype=np.int64))
        else:
            starts = np.arange(hi, dtype=np.int64)
            eligible = starts[~M.exclusion_mask(starts, query, None)]
            if vm is not None:
                eligible = eligible[vm[:hi]]
            if eligible.size == 0:
                return {h: np.array([]) for h in horizons}
            d = MP.distance_profile(
                self.matrix[query.start:query.stop], self.matrix, query_idx=query.start
            )
            finite = np.isfinite(d)
            pool = eligible[finite[eligible]]
            if pool.size == 0:
                return {h: np.array([]) for h in horizons}
            pool_pct = np.searchsorted(
                np.sort(d[finite]), d[pool]
            ) / max(1, int(finite.sum()))
            per = max(1, int(np.ceil(n / pct.size)))
            chunks = []
            for p in pct:
                near = pool[np.abs(pool_pct - p) <= 0.02]
                if near.size == 0:
                    near = pool[np.argsort(np.abs(pool_pct - p))[:per]]
                chunks.append(rng.choice(near, size=per, replace=True))
            picks = np.concatenate(chunks)

        out = M.forward_returns(self.close, picks, L, horizons)
        return out

    def run(self, query: Optional[M.Query] = None, *, k: int = DEFAULT_K,
            horizons=None, seed: int = 0,
            n_baseline: int = 400, min_matches: int = MIN_MATCHES,
            block: int = None,
            amplitude_weight: float = M.DEFAULT_AMPLITUDE_WEIGHT,
            session_mask: bool = True) -> dict:
        """Match, then forecast with baseline.  Returns a dict for the UI.

        ``horizons=None`` resolves from **this pipeline's** resolution rather than
        from the module-level intraday default.  That is load-bearing and was a real
        defect: a daily pipeline reached this method and computed horizons of
        ``(5, 15, 30, 60)`` **bars**, which on a daily frame is 5 weeks through 3
        months rather than 5 days through 2 months.

        It failed silently in the worst way available -- the horizon-60 branch produced
        *no* candidates at all (a 60-bar horizon plus the 60-bar window needs 120
        forward bars, and the filter left nothing), so the query came back with zero
        matches and a `sufficient=False` forecast.  That reads as "this shape has no
        historical analogue", which is a claim about the archive rather than about the
        horizon set.
        """
        if not self.ready:
            return {"ok": False, "reason": "; ".join(self.warnings)}

        if horizons is None:
            horizons = self.tf.horizons

        q = query or self.query_latest()

        # §BX: the query's own length is the window length for everything downstream.
        # ``self.length`` is only the *default* window -- used when no query is given.
        # `find_matches` already reads `query.length`, so without this the matched set
        # and the baseline would be measured over different window lengths and the
        # reported lift would be comparing two different quantities.
        L = int(q.length)
        max_h = int(max(horizons))

        # One mask, shared by the treatment search and the control draw.  The treatment
        # and the baseline must be drawn from the *same* admissible population, or the
        # lift they produce measures the censoring rather than the pattern.
        vmask = self.forward_horizon_mask(L, max_h) if session_mask else None

        res = M.find_matches(
            self.matrix, q, k=k,
            amplitude_weight=amplitude_weight,
            amplitude_series=self.log_returns,
            valid_mask=vmask,
        )

        starts = np.array([m.start for m in res.matches], dtype=np.int64)
        fwd = self.forward_returns(starts, horizons, window_length=L)
        base = self.baseline_returns(max(n_baseline, len(starts) * 4), horizons,
                                     seed=seed, window_length=L, valid_mask=vmask)

        b = block if block is not None else L + max(horizons)
        forecasts = conditional_forecast(fwd, base, block=b, seed=seed, min_matches=min_matches)

        return {
            "ok": True,
            "query": q,
            "result": res,
            "forecasts": forecasts,
            "forward": fwd,
            "baseline": base,
            "horizons": horizons,
            "method": res.method,
            # Recorded so the UI can state which rule produced these percentiles.  A
            # percentile is only interpretable against the metric that made it.
            "amplitude_weight": float(amplitude_weight),
        }