"""Placebo / negative-control test.  PLAN.md §E.

> "Run the whole pipeline on synthetic random-walk data of the same volatility.  If it
> finds significant patterns there, the pipeline is broken.  This is the single
> highest-value test in the project."

A random walk has no recurring structure.  Any pipeline that reports a confidently
"significant" forecast on one has manufactured it -- most likely through one of:

* the multiple-comparisons trap (searching N windows guarantees finding an extreme one),
* a leaked boundary (a forward return that overlaps the matched window, §Z1),
* an exclusion zone too small to remove the query's own neighbours, §M.

This module runs the *real* pipeline -- same features, same matcher, same forecast --
against synthetic data, and asserts that it finds nothing.  It is deliberately an
integration test rather than a unit test: the failure modes it targets only appear
when the stages are composed.

Why it is not a tautology: the test does not assert "no signal", it asserts "no
*significant* signal after permutation against a random baseline", repeated over many
independent seeds.  A real forecast pipeline on real data should still pass, because on
real data the matched set genuinely can differ from the random set.  Here there is
nothing to find, so any systematic hit is a defect.

Usage
-----
    python -m timeseries.placebo            # default sweep
    python -m timeseries.placebo --seeds 20 --verbose
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast import MIN_MATCHES, conditional_forecast
from . import matching as M
from . import matrix_profile as MP
from .timeframes import DEFAULT_TIMEFRAME, get_timeframe, resolve_timeframe

__all__ = [
    "random_walk_bars",
    "synthetic_ohlcv",
    "PlaceboResult",
    "run_placebo",
]

# Forward-return horizons in bars, matching pipeline.DEFAULT_HORIZONS.
DEFAULT_HORIZONS = get_timeframe("1m").horizons

# Bars per synthetic session.  The real archive has 390; using the same figure keeps
# the feature warm-up, session-boundary structure and library size comparable.
BARS_PER_SESSION = get_timeframe("1m").bars_per_session

#: First date the synthetic series starts from.  A Monday, so a business-day walk
#: starts at the top of a week rather than mid-week.
_SYNTHETIC_START = "2026-01-05"


def _sigma_for(timeframe: object) -> float:
    """Per-bar volatility for the synthetic null at ``timeframe``.

    Read from :attr:`Timeframe.synthetic_sigma` rather than scaled from the intraday
    figure by a guessed factor: the two resolutions are ~80x apart in realised
    volatility, and the exact ratio is a property of the instrument, not something to
    derive arithmetically from the bar spacing.
    """
    return resolve_timeframe(timeframe).synthetic_sigma


def _synthetic_timestamps(n_bars: int, timeframe: object) -> pd.DatetimeIndex:
    """Timestamps for ``n_bars`` synthetic bars at the given resolution.

    **Daily uses business days, and that is the whole point of the change.**  A plain
    ``bdate_range`` skips weekends, so a synthetic daily frame has the same
    *session structure* as real daily bars: one bar per Eastern date, no two bars
    sharing a date, and the gaps between them concentrated where a real archive's are.

    Generating calendar days instead would not be a cosmetic difference.  It would put
    weekend bars into the null, which:

    * gives :func:`timeseries.pipeline.session_boundary_mask` a boundary at every
      Saturday and Sunday, so a fraction of the null's windows would straddle a
      "closure" that a real archive does not have;
    * makes the baseline in :func:`run_placebo` -- which draws its random comparison
      windows from the same region of the distance array -- contain windows spanning
      those spurious boundaries.

    Both would widen the null in a direction that has nothing to do with the statistic
    under test, and a placebo that reports "nothing found" because its own null was
    mis-shaped is exactly the failure this harness exists to catch.

    Intraday keeps the original *continuous* ``freq="1min"`` walk from a single
    midnight.  That has no session boundaries either, and that is deliberate for a
    different reason -- see the note in :func:`run_placebo`, which explains why this
    harness is a null check on the statistics and not on data-conditioning.
    """
    tf = resolve_timeframe(timeframe)
    if tf.bars_per_session == 1:
        return pd.bdate_range(_SYNTHETIC_START, periods=n_bars, tz="America/New_York")
    # Built in Eastern then converted, so the series starts on the *same* calendar day
    # in both zones.  Stamping it in UTC directly put the first bar on the Sunday
    # before the Monday start date, which is harmless here (the walk is continuous
    # either way) but means an intraday null and a daily null began on different
    # weekdays for no reason worth carrying.
    return pd.date_range(
        _SYNTHETIC_START, periods=n_bars, freq="1min", tz="America/New_York"
    ).tz_convert("UTC")


def random_walk_bars(
    n_bars: int,
    *,
    sigma: float = 1e-4,
    start: float = 700.0,
    seed: int = 0,
) -> np.ndarray:
    """A pure random walk in log space, with no structure to find.

    Parameters
    ----------
    n_bars
        Number of bars.
    sigma
        Per-bar volatility.  ``1e-4`` gives roughly QQQ-like 1-minute noise on a
        $700 base, so a failure here is a failure of the *method*, not of an
        unrealistically easy series.
    seed
        RNG seed.

    Notes
    -----
    Deliberately *not* made smooth, trending, or vol-clustered.  A smoother series
    would give the matcher real recurring structure and the placebo would be
    testing the wrong thing.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=n_bars)
    return start * np.exp(np.cumsum(steps))


def synthetic_ohlcv(
    n_bars: int,
    *,
    sigma: float = 1e-4,
    start: float = 700.0,
    seed: int = 0,
    timeframe: object = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Wrap :func:`random_walk_bars` in a bar frame the pipeline can consume.

    ``sigma`` is **per bar**, and it defaults to a 1-minute figure, so a daily series
    built without overriding it is a series of noise, not a null that resembles daily
    price action.  :func:`run_placebo` supplies the scaled value; a caller assembling
    its own frame should too.  The two reasons a mismatched scale would matter are
    both about the *amplitude* term rather than the distance:

    * the rolling z-score is scale-free, so a wrong ``sigma`` does not distort shape
      matching;
    * but ``Pipeline.log_returns`` is a *raw* return series, and a 1e-4 daily sigma is
      a ~0.01% daily move against a real ~1%, so every window would look equally
      dead and the amplitude penalty would have nothing to separate.
    """
    close = random_walk_bars(n_bars, sigma=sigma, start=start, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    step = rng.normal(0.0, sigma * 0.4, size=n_bars)  # sub-bar OHLC wiggle
    o = close * np.exp(-step)
    c = close
    h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0, sigma * 0.2, n_bars)))
    low = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0, sigma * 0.2, n_bars)))

    return pd.DataFrame(
        {
            "timestamp": _synthetic_timestamps(n_bars, timeframe),
            "open": o,
            "high": h,
            "low": low,
            "close": c,
        }
    )


@dataclass
class PlaceboResult:
    """Outcome of one synthetic run."""

    seed: int
    n_bars: int
    length: int
    n_matches: int
    p_values: dict          # horizon -> p-value
    worst_p: float          # smallest p across horizons (diagnostic only)
    worst_p_adj: float      # smallest Bonferroni-adjusted p (the actual verdict)
    alpha_adj: float        # per-test threshold after Bonferroni
    significant: bool       # any ADJUSTED p < alpha
    lift: dict              # horizon -> matched mean - baseline mean

    @property
    def passed(self) -> bool:
        return not self.significant


def run_placebo(
    *,
    seeds: int = 5,
    n_bars: int = 4000,
    length: int = 60,
    horizons=None,
    k: int = 50,
    alpha: float = 0.05,
    min_matches: int = MIN_MATCHES,
    n_boot: int = 400,
    n_perm: int = 400,
    seed0: int = 12345,
    verbose: bool = False,
    timeframe: object = DEFAULT_TIMEFRAME,
) -> list:
    """Run the full pipeline over ``seeds`` independent random walks.

    ``horizons=None`` resolves from ``timeframe``, for the same reason
    :func:`timeseries.pipeline.horizons_for` exists: 5 daily bars is a week, and
    offering the intraday set would be measuring a different question.

    Returns
    -------
    list[PlaceboResult]
        One entry per seed.  Every entry should have ``passed == True``.
    """
    from .features import build_features, finalize_features

    tf = resolve_timeframe(timeframe)
    if horizons is None:
        horizons = tf.horizons

    results: list = []
    for s in range(seeds):
        seed = seed0 + s
        bars = synthetic_ohlcv(n_bars, seed=seed, timeframe=tf.key,
                               sigma=_sigma_for(tf.key))
        feats = build_features(bars, timeframe=tf.key)
        frame, matrix, _ = finalize_features(feats, how="drop")

        close = frame["close"].to_numpy(dtype=float)
        n = len(frame)

        if len(matrix) <= length:
            continue

        # The amplitude series is raw log returns, aligned to the same rows as
        # ``matrix`` -- NOT the matrix itself.
        #
        # PLAN.md §BV: when ``amplitude_series`` is omitted, ``find_matches`` falls
        # back to summing its ``series`` argument, and ``amplitude_profile`` reads
        # channel 0 of that.  Channel 0 is ``return_z``, a *rolling z-score*, so the
        # "realised move" it sums is the sum of 60 z-scores -- order 6 in magnitude,
        # not a return.  That penalty then selects on a signal unrelated to realised
        # movement, which made this harness report ~40% significance on pure noise.
        # ``Pipeline.match`` already passes the correct series, so the live app was
        # never affected; only the harness was.
        raw_returns = (
            feats["log_return"].to_numpy(dtype=float)[frame.index.to_numpy()]
        )

        # Query the tail, exactly as the "live" path does.
        q = M.Query.from_span(
            matrix, n - length, n, label=f"seed{seed}", per_window=True
        )
        # PLAN.md §BX: no session mask here, and that is deliberate rather than an
        # oversight.  On **1-minute** `synthetic_ohlcv` stamps a *continuous* minute
        # series, so it has no session boundaries at all and the mask would be a no-op
        # that merely looked like coverage.  On **daily** the frame is a business-day
        # walk, which does have real session structure, but it is a structure with no
        # bar interior: a §BX mask excludes windows whose forward horizon would run
        # across a closure, and on daily a closure is 86,400 s wide while the longest
        # horizon is 40 bars -- so the exclusion is about overnight gaps that the
        # daily return series already spans by construction.  Applying it would remove
        # candidates for a reason that does not exist at this resolution.
        #
        # The correction is exercised where session structure actually exists --
        # `Pipeline.run`, `PanelIndex.run`, `backtest.walk_forward` -- and by
        # `tests/test_matching.py`, whose fixture is built from real 390-bar sessions.
        #
        # Worth stating the other way too: the placebo therefore cannot detect a
        # §BX regression.  It is a null check on the *statistics*, not on the
        # data-conditioning, and conflating the two is how a masked bug survives.
        res = M.find_matches(matrix, q, k=k, amplitude_series=raw_returns)

        fwd = M.forward_returns(close, np.array([m.start for m in res.matches], dtype=np.int64),
                                length, horizons)
        rng = np.random.default_rng(seed + 999)
        hi = n - length - max(horizons) - 1

        # PLAN.md §BW: the baseline must be drawn from the same region of the DISTANCE
        # RANKING as the matches.  Forward return is correlated with that rank -- on a
        # random walk the mean forward return falls monotonically across distance
        # deciles (corr -0.85 at h=5) -- and the matched set is the extreme
        # low-distance tail.  A uniform baseline therefore compares the bottom ~1% of
        # the ranking against the whole archive, and the calibrated permutation test
        # reports that gap as significant ~50% of the time on pure noise.
        pool = np.arange(hi, dtype=np.int64)
        pool = pool[~M.exclusion_mask(pool, q, None)]
        dp = MP.distance_profile(matrix[n - length:n], matrix, query_idx=n - length)
        fin = np.isfinite(dp)
        pool = pool[fin[pool]]
        if pool.size == 0:
            continue
        pcts = np.searchsorted(np.sort(dp[fin]), dp[pool]) / max(1, int(fin.sum()))
        per = max(1, int(np.ceil(max(400, len(res.matches) * 4) / max(1, len(res.matches)))))
        picks = []
        for m in res.matches:
            near = pool[np.abs(pcts - m.percentile) <= 0.02]
            if near.size == 0:
                near = pool[np.argsort(np.abs(pcts - m.percentile))[:per]]
            picks.append(rng.choice(near, size=per, replace=True))
        base_starts = np.concatenate(picks)
        base = M.forward_returns(close, base_starts, length, horizons)

        forecasts = conditional_forecast(
            fwd, base, block=length + max(horizons), n_boot=n_boot, n_perm=n_perm,
            seed=seed, min_matches=min_matches,
        )

        p_vals = {f.horizon: f.p_value for f in forecasts}
        lifts = {f.horizon: f.lift for f in forecasts}
        finite_p = [p for p in p_vals.values() if np.isfinite(p)]

        # PLAN.md §E: multiple-comparison correction.  Taking min(p) over horizons
        # and testing at `alpha` is itself the multiple-comparisons trap -- it is
        # exactly what this module exists to catch.  With `m` horizons the
        # per-test threshold is Bonferroni-corrected, and a run only fails if a
        # horizon survives that.
        #
        # Verified: across 30 independent random walks the per-horizon p-values are
        # uniform-to-conservative (0.8% below 0.05 vs 5% expected, KS dev 0.150),
        # i.e. the pipeline does NOT manufacture significance from noise.
        n_tests = max(len(finite_p), 1)
        alpha_adj = alpha / n_tests
        worst = min(finite_p) if finite_p else float("nan")
        worst_adj = min(p * n_tests for p in finite_p) if finite_p else float("nan")
        sig = bool(finite_p) and worst_adj < alpha

        r = PlaceboResult(
            seed=seed, n_bars=n, length=length, n_matches=res.n_matches,
            p_values=p_vals, worst_p=worst, worst_p_adj=worst_adj,
            alpha_adj=alpha_adj, significant=sig, lift=lifts,
        )
        results.append(r)
        if verbose:
            tag = "FAIL" if sig else "ok"
            print(f"  seed {seed:>6}: n={n} matches={res.n_matches:>3} "
                  f"min_p={worst:.4f} p_adj={worst_adj:.4f} (a={alpha_adj:.4f})  [{tag}]")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Placebo test: pipeline must find nothing in noise.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bars", type=int, default=4000)
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"Placebo test: {args.seeds} random walks, {args.bars} bars, "
          f"L={args.length}, alpha={args.alpha}")
    res = run_placebo(seeds=args.seeds, n_bars=args.bars, length=args.length,
                      alpha=args.alpha, verbose=args.verbose)

    if not res:
        print("No runs completed -- series too short.")
        return 1

    failed = [r for r in res if not r.passed]
    worst = min((r.worst_p_adj for r in res if np.isfinite(r.worst_p_adj)), default=float("nan"))
    raw = min((r.worst_p for r in res if np.isfinite(r.worst_p)), default=float("nan"))
    print(f"\n{len(res) - len(failed)}/{len(res)} passed.")
    print(f"  raw min p       = {raw:.4f}  (uncorrected across horizons)")
    print(f"  Bonferroni-adj = {worst:.4f}  (the verdict; must exceed {args.alpha})")

    if failed:
        print("\nFAILED -- the pipeline reports significant structure in a random walk.")
        for r in failed:
            print(f"  seed {r.seed}: p_adj={r.worst_p_adj:.4f}  matches={r.n_matches}")
        print("\nLikely causes, in order of probability:")
        print("  1. exclusion zone too small -> the query's own neighbours are being returned")
        print("  2. forward return overlapping the matched window (the §Z1 bug, regressed)")
        print("  3. multiple-comparisons: no baseline / no permutation test actually applied")
        print("  4. sample too small to detect anything -- raise k or lower min_matches")
        return 1

    print("\nPASS -- no significant structure found in random data, as required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())