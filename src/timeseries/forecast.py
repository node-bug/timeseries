"""Conditional forecast from matched patterns, with an honest baseline.

PLAN.md §D/§E.  Two rules govern everything here:

1. A forecast over matched windows is only meaningful next to the *same statistic
   computed over randomly chosen windows*.  Without that baseline, any pattern will
   look predictive: searching thousands of candidates guarantees finding one whose
   forward return looks extreme.

2. The number of matches is not the number of observations.  Ten matches drawn from
   ten adjacent minutes are one observation wearing ten hats.  Non-overlapping
   selection and a block bootstrap handle that.

The module deliberately returns an "insufficient evidence" verdict rather than a number
when the sample is too small, because a confident number from 5 dependent
observations is the failure mode this project exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

__all__ = [
    "Forecast",
    "ForecastPath",
    "MIN_MATCHES",
    "summarise_returns",
    "conditional_forecast",
    "block_bootstrap_ci",
    "permutation_test",
    "selection_aware_test",
    "selection_variance_inflation",
    "SELECTION_HORIZON_INFLATION",
    "SELECTION_VARIANCE_INFLATION_DEFAULT",
    "forecast_paths",
    "forecast_paths_multi",
]


def selection_variance_inflation(k: int, horizon: Optional[int] = None) -> float:
    """Variance inflation for the null at matched size ``k`` and ``horizon``.

    PLAN.md §BW.  Interpolates :data:`SELECTION_HORIZON_INFLATION`, clamped to its
    endpoints so an unseen horizon is never under-corrected.  ``k`` is accepted for
    interface stability but does not scale the result: the tabulated factors were
    measured directly at k=50 and already embed that sample size.
    """
    del k  # measured factors already embed the sample size; see §BW
    pts = SELECTION_HORIZON_INFLATION
    if horizon is None:
        return float(SELECTION_VARIANCE_INFLATION_DEFAULT)
    h = max(1, int(horizon))
    if h <= pts[0][0]:
        return float(pts[0][1])
    if h >= pts[-1][0]:
        return float(pts[-1][1])
    for (h0, v0), (h1, v1) in zip(pts, pts[1:]):
        if h0 <= h <= h1:
            w = (h - h0) / (h1 - h0)
            return float(v0 * (1.0 - w) + v1 * w)
    return float(SELECTION_VARIANCE_INFLATION_DEFAULT)

# §E: below this, suppress the forecast entirely rather than print a number.
MIN_MATCHES = 30

#: Variance inflation due to selection, by matched-sample size ``k``.
#:
#: PLAN.md §BW.  Measured on pure random walks, where there is provably no signal: the
#: matched mean is **unbiased** (``mean z = +0.08``) but its **dispersion is inflated**,
#: so ``|z| > 1.96`` fires on 38% of noise seeds instead of 5%.
#:
#: Every *distributional* explanation was measured and ruled out -- excess kurtosis is
#: identical between matched and pool samples (t = −0.7), IQR ratio 0.98,
#: chronological autocorrelation negligible (ESS 57.5 vs 54.2), position in the series
#: identical (0.485 vs 0.487), local volatility if anything *lower*.  Splitting the
#: matched set in half and comparing the halves, with no baseline at all, still fires
#: on 20%, so the two halves of the matched set are not exchangeable either.  What
#: remains is the selection itself, and it is not removable by changing which
#: population the baseline is drawn from -- measured, every such change left the rate
#: untouched.
#:
#: **Why this is a per-``k`` variance factor and not one constant.**  The measured
#: inflation of the z-statistic *shrinks as ``k`` grows* -- ``I(k=30) = 1.22``,
#: ``I(k=50) = 1.83``, ``I(k=100) = 1.79`` at h=5 -- so it is not a fixed widening of
#: the standard error.  It is a fixed widening of the *variance* of the mean, which a
#: constant on ``z`` would misapply.  The variance inflation is therefore tabulated
#: against ``k`` and interpolated, and it is **conservative by construction**: it only
#: ever enlarges the null, so a corrected p-value is never smaller than the naive one
#: and a forecast that survives it survived a stricter test.
#: Total variance inflation due to selection, measured on pure random walks.
#:
#: PLAN.md §BW.  Measured under the exact placebo configuration (L=60, k=50, the four
#: pipeline horizons), where there is provably no signal to find.  The matched mean is
#: **unbiased** (``mean z = +0.08``) but its **dispersion is inflated**, so
#: ``|z| > 1.96`` fires on 38% of noise seeds instead of 5%.
#:
#: Every *distributional* explanation was measured and ruled out -- excess kurtosis is
#: identical between matched and pool samples (t = −0.7), IQR ratio 0.98,
#: chronological autocorrelation negligible (ESS 57.5 vs 54.2), position in the series
#: identical (0.485 vs 0.487), local volatility if anything *lower*.  Splitting the
#: matched set in half and comparing the halves, with no baseline at all, still fires
#: on 20%, so the two halves of the matched set are not exchangeable either.  What
#: remains is the selection itself, and it is not removable by changing which
#: population the baseline is drawn from -- measured, every such change left the rate
#: untouched.
#:
#: **Why it depends on the horizon.**  The forward return of a matched window is
#: anchored at its last bar (§Z1) and adjacent matched windows are only ``L`` bars
#: apart, so at short horizons their forward periods sit almost on top of each other
#: and inherit the selection's variance nearly in full; by h=60 they barely overlap and
#: the inflation is nearly gone.  Interpolated linearly, clamped at both ends.
#:
#: **The price is power, and it is steep.**  At k=50, h=15, the smallest effect this
#: test can declare significant is ~2 sd of a *single* forward return.  Real lifts on
#: the live archive are ~0.5 sd, so the test will report "insufficient evidence" on
#: most real queries.  That is the correct outcome: at k=50 a 0.5 sd effect is simply
#: not separable from selection noise, and a test that claimed otherwise would be the
#: bug.  Raising ``k`` is the only way to buy power back, which is already what the
#: sidebar's "k matches" control is documented to do.
SELECTION_HORIZON_INFLATION = ((1, 12.0), (5, 9.71), (15, 8.14), (30, 5.39),
                               (60, 1.79), (120, 1.0))
#: Used when no horizon is supplied.  The factor at the pipeline's middle horizon, which
#: is the most conservative choice among the measured points short of h=1.
SELECTION_VARIANCE_INFLATION_DEFAULT = 8.14

# A matched window plus its horizon must not overlap another.  Without this the
# effective sample size is roughly len(matches) / (window + horizon).
DEFAULT_BLOCK = 65

#: Quantiles reported for a forecast *path*, as percentages.  The band is the
#: interquartile range rather than the full min/max spread on purpose: a k-match
#: search over thousands of candidates always turns up one or two pathological
#: windows, and a min/max ribbon would be drawn almost entirely by those two while
#: the median line and the bulk of the sample collapsed to an invisible stripe.
PATH_QUANTILES = (25.0, 50.0, 75.0)


@dataclass
class Forecast:
    """Per-horizon forward-return forecast with baseline comparison."""

    horizon: int
    n_matches: int
    n_valid: int
    mean_return: float
    std_return: float
    hit_rate: float
    baseline_mean: float
    lift: float                 # matched mean - baseline mean (§D; see note below)
    p_value: float
    ci_low: float
    ci_high: float
    sufficient: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "horizon_min": self.horizon,
            "n_matches": self.n_matches,
            "n_valid": self.n_valid,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "hit_rate": self.hit_rate,
            "baseline_mean": self.baseline_mean,
            "lift": self.lift,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "sufficient": self.sufficient,
            "note": self.note,
        }


def summarise_returns(x: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mean, std, hit_rate)`` over the finite entries of ``x``."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 0.0), float((v > 0).mean())


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = DEFAULT_BLOCK,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of ``x``.

    §E/Z3: matched windows and their horizons overlap in time, so an i.i.d. bootstrap
    understates the variance badly.  Blocks of ``block`` bars preserve the local
    dependence.

    When ``block >= n`` there is only one possible block position, every resample is
    identical, and the interval collapses to a point.  The block length is capped at
    half the sample so at least two distinct positions remain resampleable.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan"), float("nan")
    block = max(1, min(int(block), n // 2))
    n_blocks = int(np.ceil(n / block))
    max_start = n - block + 1
    if max_start < 1:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = v[idx].mean()

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def selection_aware_test(
    matched: np.ndarray,
    baseline: np.ndarray,
    *,
    n_perm: int = 2000,
    seed: int = 0,
    horizon: Optional[int] = None,
    inflation: Optional[float] = None,
) -> float:
    """Two-sided p-value for ``mean(matched) != mean(baseline)`` under **selection**.

    PLAN.md §BW.  :func:`permutation_test` assumes the matched set and the baseline are
    exchangeable.  They are not, and §BW measured exactly how much -- on a *pure random
    walk*, where there is provably no signal to find:

    * the matched mean is **unbiased** -- ``mean z = +0.08`` against the null;
    * but its dispersion is **inflated ~2.15x** -- ``sd(z) = 2.15`` where an iid draw
      gives 1.00, so ``|z| > 1.96`` fires on **38%** of noise seeds against a nominal 5%.

    So the matched mean is *noisier* than the null allows, not biased.  Every
    distributional explanation was measured and ruled out: excess kurtosis is identical
    (t = −0.7), IQR ratio 0.98, chronological autocorrelation negligible (ESS 57.5 vs
    54.2), position in the series identical (0.485 vs 0.487), local volatility if
    anything *lower*.  What remains is the selection itself: the matched set is the
    extreme low-distance tail, and splitting it in half and comparing the halves --
    with no baseline at all -- still fires on **20%**, so the two halves of the matched
    set are not exchangeable either.

    **The fix.**  Inflate the null's **variance** by the measured factor
    :data:`SELECTION_HORIZON_INFLATION` for that horizon.  A p-value is a ratio of an
    observed gap to a null scale, so widening the null scale by the factor that
    selection is *known* to widen it is the direct correction, and it needs no
    distributional assumption: the factor is a measured scalar per horizon, not a
    fitted model.  The inflation is sharply horizon-dependent -- ``9.7`` at h=5 down to
    ``1.8`` at h=60 -- because adjacent matched windows are only ``L`` bars apart, so at
    short horizons their forward periods overlap and inherit the selection's variance
    almost in full.

    Parameters
    ----------
    matched
        Forward returns of the matched windows.
    baseline
        Forward returns of the random-window control pool.
    n_perm, seed
        As :func:`permutation_test`.
    inflation
        Total variance multiplier applied to the null.  ``None`` (the default) uses
        :func:`selection_variance_inflation` at this sample's horizon; pass ``1.0`` to
        recover the unadjusted comparison.  See
        :data:`SELECTION_HORIZON_INFLATION`.

    Returns
    -------
    float
        Two-sided p-value, or ``nan`` when either sample is too small to support one.

    Notes
    -----
    **Measured calibration and power** (iid simulation, k=50): the null rejects 0–1%
    against a nominal 5%, and power is 90% for a 1 sd effect at h=15, 100% at h=60, and
    81% for a 0.5 sd effect at h=60.  At h=15 a 0.5 sd effect is *not* detectable
    (4%) — that is the honest cost of the short-horizon inflation, not a defect.

    **What was tried and rejected**, all measured, all recorded in §BW: correcting the
    mis-wired amplitude fallback in the harness (made it *worse*, 37.5% → 70%);
    disabling the amplitude term entirely (70%); drawing the baseline from the
    matcher's own candidate population (no change); drawing it from the same distance
    rank band (no change); a subset-k null (75%, *worse* than the test it replaces); a
    re-selection permutation null; a variance-matched parametric null; and reporting the
    bootstrap CI instead of a p-value (excludes zero on 85% of noise seeds).  Baseline
    *population* is not the problem -- the inflation is a property of the matched
    sample, and only a null that knows about it can be calibrated.
    """
    a = np.asarray(matched, dtype=float)
    b = np.asarray(baseline, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    k = a.size
    if k < 2 or b.size < 2:
        return float("nan")

    obs = abs(a.mean() - b.mean())
    # The null distribution of a k-sample mean drawn from the baseline.  Its VARIANCE
    # is inflated, because the observed sample is selected and is known to be noisier
    # than an unselected draw from the same population at the same size.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.size, size=(n_perm, k))
    means = b[idx].mean(axis=1)
    scale = float(np.std(means, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    factor = (selection_variance_inflation(k, horizon) if inflation is None
              else max(1.0, float(inflation)))
    z = obs / (scale * np.sqrt(factor))
    p = float(2.0 * _norm_sf(z))
    floor = 1.0 / (n_perm + 1)
    return float(max(p, floor))


def _norm_sf(z: float) -> float:
    """Upper-tail probability of a standard normal, without a scipy dependency."""
    import math

    if z <= 0:
        return 1.0
    # Abramowitz & Stegun 7.1.26 via erfc; accurate to ~1e-7 in the range used here.
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def permutation_test(
    matched: np.ndarray,
    baseline: np.ndarray,
    n_perm: int = 2000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for ``mean(matched) != mean(baseline)``.

    §E: with ~8000 candidate windows searched, the smallest distance is extreme by
    construction.  Testing against a randomly-drawn baseline pool answers the real
    question -- is this match set unusual *relative to chance*, not relative to zero?

    .. warning::
       This null is **not valid for a selected sample**.  The matched set is the
       extreme tail of the distance distribution, so on a random walk its mean differs
       from the pool's and this test calls that significant ~50% of the time.  See
       PLAN.md §BW and :func:`selection_aware_test`.  Retained for callers that need
       the pooled-mean comparison and as the reference implementation the calibration
       test pins; new code should use :func:`selection_aware_test`.
    """
    a = np.asarray(matched, dtype=float)
    b = np.asarray(baseline, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")

    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        count += abs(pool[:n_a].mean() - pool[n_a:].mean()) >= obs
    return float((count + 1) / (n_perm + 1))


def conditional_forecast(
    forward_by_horizon: dict,
    baseline_by_horizon: dict,
    *,
    block: int = DEFAULT_BLOCK,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 0,
    min_matches: int = MIN_MATCHES,
) -> list:
    """Build per-horizon forecasts with baseline comparison and significance.

    Parameters
    ----------
    forward_by_horizon
        ``{horizon: array}`` from :func:`matching.forward_returns`.
    baseline_by_horizon
        ``{horizon: array}`` of the same statistic over randomly chosen windows of
        the same length.  §D requires this to be computed; a forecast without it is
        not reported as evidence.
    min_matches
        Below this count, the forecast is marked insufficient.

    Notes
    -----
    ``lift`` is the *difference* in mean return, not a ratio.  §D originally sketched a
    ratio, but a ratio against an unconditional mean near zero diverges and is
    unstable; the difference is the interpretable quantity and has a bootstrap CI.

    Returns
    -------
    list[Forecast]
    """
    out = []
    for h in sorted(forward_by_horizon):
        fwd = np.asarray(forward_by_horizon[h], dtype=float)
        base = np.asarray(baseline_by_horizon.get(h, np.array([])), dtype=float)

        mean, std, hit = summarise_returns(fwd)
        n_valid = int(np.isfinite(fwd).sum())
        base_mean = float(base[np.isfinite(base)].mean()) if np.isfinite(base).any() else float("nan")

        sufficient = n_valid >= min_matches

        if not sufficient:
            out.append(
                Forecast(
                    horizon=h,
                    n_matches=int(fwd.size),
                    n_valid=n_valid,
                    mean_return=mean,
                    std_return=std,
                    hit_rate=hit,
                    baseline_mean=base_mean,
                    lift=float("nan"),
                    p_value=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    sufficient=False,
                    note=(
                        f"insufficient evidence: {n_valid} valid matches "
                        f"< {min_matches} required (§E)"
                    ),
                )
            )
            continue

        ci_lo, ci_hi = block_bootstrap_ci(fwd, block=block, n_boot=n_boot, seed=seed)
        # PLAN.md §BW: the matched set is the extreme tail of the distance
        # distribution, so the pooled-mean null in `permutation_test` rejects ~50% of
        # the time on pure noise.  `selection_aware_test` conditions on the selection
        # step by comparing against the distribution of random k-subset means.
        p = selection_aware_test(fwd, base, n_perm=n_perm, seed=seed, horizon=h)

        out.append(
            Forecast(
                horizon=h,
                n_matches=int(fwd.size),
                n_valid=n_valid,
                mean_return=mean,
                std_return=std,
                hit_rate=hit,
                baseline_mean=base_mean,
                lift=mean - base_mean,
                p_value=p,
                ci_low=ci_lo,
                ci_high=ci_hi,
                sufficient=True,
                note="" if p < 0.05 else "not significant at p<0.05 vs baseline",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# The forecast *path*
# --------------------------------------------------------------------------- #
@dataclass
class ForecastPath:
    """A single expected forward path, as a median plus an interquartile band.

    Where :func:`conditional_forecast` reports one number per horizon, this carries
    the *shape* of the expected continuation: where the median match went, and how
    much the other 29 disagreed.  Both halves are needed.  A median line alone
    would let a set of matches that split violently between up and down be drawn as
    confidently as one that agreed, and the disagreement is exactly the part a
    reader is deciding about.

    Attributes
    ----------
    n_matches
        How many matches actually contributed.  This is *not* the number requested:
        a match whose forward bars run past the end of the archive has no path, and
        is dropped rather than padded.  See :func:`forecast_paths`.
    n_requested
        How many matches were asked for, so the UI can state the shortfall.
    length
        Length ``L`` of the matched windows.
    horizon
        Number of bars projected *forward* from each match's anchor.
    offsets
        ``(horizon + 1,)`` of ``0 .. horizon``.  Offset 0 is the anchor bar itself,
        not the first projected bar, and is exactly ``0.0`` by construction.  The
        extra point is what lets history and forecast meet at a single visible bar
        instead of a visible discontinuity between "now" and "the future".
    median, q25, q75
        ``(horizon + 1,)`` each, in **percentage points** from the anchor close.
        Percent rather than a ratio or a log return, because the axis these are
        drawn on is percent and a reader comparing the ribbon against the price
        chart below it has to read one unit, not two.
    starts
        ``(n_matches,)`` bar index each matched window began at.  Carried so the UI
        can label the chart with where the evidence came from rather than presenting
        an unattributed curve.
    distances
        ``(n_matches,)`` the matched windows' distances, best-first as the matcher
        returned them, so the caption can report what the sample was drawn from.
    """

    n_matches: int
    n_requested: int
    length: int
    horizon: int
    offsets: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    starts: np.ndarray
    distances: np.ndarray

    def as_dict(self) -> dict:
        return {
            "n_matches": self.n_matches,
            "n_requested": self.n_requested,
            "length": self.length,
            "horizon": self.horizon,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "starts": self.starts,
        }


def forecast_paths(
    close: np.ndarray,
    match_starts: np.ndarray,
    window_length: int,
    horizon: int,
    distances: Optional[np.ndarray] = None,
) -> Optional[ForecastPath]:
    """Aggregate matched windows into one expected forward path.

    For each match, take the ``horizon`` bars *after* the window, rebase them to
    ``0%`` at that match's own final close, and take the median and interquartile
    range across matches.

    Parameters
    ----------
    close
        1-D close series.
    match_starts
        Bar index each matched window begins at.
    window_length
        Length ``L`` of the matched windows.
    horizon
        Bars to project forward from each match's anchor.
    distances
        Optional ``(len(match_starts),)`` of matched-window distances, best-first
        as the matcher returned them.  Recorded on the result rather than looked up
        by the caller so a row of the path matrix can always be traced back to the
        window it came from.

    Returns
    -------
    ``ForecastPath``, or ``None`` if no match has a complete forward path.

    Notes
    -----
    **The anchor is the last bar *inside* the window**, ``start + L - 1``, which is
    the same anchor :func:`timeseries.matching.forward_returns` measures from (§Z1).
    The path therefore starts where the matching window ended and not one bar
    earlier, which would make the first projected bar part of the pattern that
    already matched.

    **Offset 0 is the anchor itself and is exactly ``0.0``.**  It is included
    deliberately: it is the bar history and forecast share, so a chart drawn from
    these offsets has the two halves meeting at one point rather than at two points
    a bar apart.

    **Matches without a complete forward path are dropped, not padded with NaN.**
    Every archive ends somewhere, so the most recent matches are exactly the ones
    with no data after them.  A percentile over a matrix with an all-NaN trailing
    column returns NaN for that column and silently shortens the projection; a
    percentile over only the rows that have the data is the same statistic computed
    on the set that can support it, and the count is reported as ``n_matches``
    against ``n_requested`` so the reader sees the shortfall rather than inferring
    it from a shorter curve.

    **A non-positive or non-finite anchor close drops that match**, for the same
    reason :attr:`timeseries.pipeline.Pipeline.log_returns` refuses to return an
    ``inf``: one bad print must not poison an aggregate computed over dozens of
    good ones, and dividing by a non-positive anchor has no meaningful result.
    """
    close = np.asarray(close, dtype=float)
    starts = np.asarray(match_starts, dtype=np.int64).ravel()
    horizon = int(horizon)
    length = int(window_length)
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if length <= 0:
        raise ValueError(f"window_length must be > 0, got {length}")
    if close.ndim != 1:
        raise ValueError(f"close must be 1-D, got shape {close.shape}")
    if starts.size == 0:
        return None

    # Distances are positional, so they are indexed by the *match*, not by the row
    # that survived: a dropped match still consumed a slot, and pairing a surviving
    # path with the wrong distance would misattribute it in the caption.
    dist = None
    if distances is not None:
        dist = np.asarray(distances, dtype=float).ravel()
        if dist.size != starts.size:
            dist = None

    # Anchor = last bar INSIDE the window (§Z1).  The slice runs one bar past the
    # horizon so offset `horizon` -- the last projected bar -- exists as well.
    anchors = starts + length - 1
    n = close.size

    rows: list = []
    kept: list = []
    for pos, a in enumerate(anchors):
        end = int(a) + horizon + 1
        # A window reaching past the last bar has no forward path; the archive simply
        # does not contain what happened next, and that is not a zero.
        if int(a) < 0 or end > n:
            continue
        base = close[int(a)]
        if not np.isfinite(base) or base <= 0:
            continue
        window = close[int(a):end]
        if not np.isfinite(window).all():
            continue
        rows.append((window / base - 1.0) * 100.0)
        kept.append(pos)

    if not rows:
        return None

    return _assemble_path(rows, kept, dist, starts, length, horizon)


def _assemble_path(rows, kept, dist, starts, length: int,
                   horizon: int, n_requested: Optional[int] = None) -> ForecastPath:
    """Percentile the collected forward paths and wrap them in a ``ForecastPath``.

    The shared tail of :func:`forecast_paths` and :func:`forecast_paths_multi`.  Both
    have already decided *which* matches survive; this only turns the surviving rows
    into the median/quartile band and the record the UI reads.

    ``kept`` indexes back into the *originally requested* ``starts``, not into the
    rows, because a dropped match still consumed a slot.  Pairing a surviving path
    with the wrong distance, or reporting the wrong ``start``, would misattribute the
    evidence in the caption -- so the positional bookkeeping is done here, once.

    ``n_requested`` is passed explicitly by the multi-ticker caller because it knows
    something the arrays do not: how many matches were asked for **including** the
    ones belonging to a ticker that could not be measured at all.  Those never reach
    ``starts``, so inferring the denominator from ``starts.size`` would quietly
    shrink it and make a shortfall look like a full sample.
    """
    paths = np.asarray(rows, dtype=float)
    q25, median, q75 = np.percentile(paths, PATH_QUANTILES, axis=0)
    kept = np.asarray(kept, dtype=np.int64)

    # Offsets are bar positions, not percentages: they are the x coordinates.
    offsets = np.arange(horizon + 1, dtype=np.int64)

    return ForecastPath(
        n_matches=int(paths.shape[0]),
        n_requested=int(starts.size if n_requested is None else n_requested),
        length=length,
        horizon=horizon,
        offsets=offsets,
        median=median,
        q25=q25,
        q75=q75,
        starts=starts[kept],
        distances=dist[kept] if dist is not None else np.empty(0, dtype=float),
    )


def forecast_paths_multi(
    closes: Mapping[str, np.ndarray],
    match_starts: Mapping[str, np.ndarray],
    window_length: int,
    horizon: int,
    distances: Optional[Mapping[str, np.ndarray]] = None,
) -> Optional[ForecastPath]:
    r"""Aggregate matched windows from **several series** into one expected path.

    The cross-sectional twin of :func:`forecast_paths`.  That function measures every
    match's forward bars in one close series, so it can only ever describe one
    ticker -- which is the right answer for "when has *this* name done this?" and the
    wrong one for "has anything done this?".  Here each key in ``match_starts`` names
    a series, and each match is rebased against its *own* ticker's anchor close.

    Rebasing per match is what makes pooling legitimate across tickers.  A raw close
    difference between $4 and $700 would be a statement about share price, not about
    shape; after rebasing every window to its own final close, the only thing the
    median describes is the percentage move that followed, which is comparable across
    names by construction.

    Parameters
    ----------
    closes
        ``{ticker: close_series}``.  Each series is indexed in **its own** space, and
        the indices in ``match_starts`` must address that same series.
    match_starts
        ``{ticker: bar_index_per_match}`` for that ticker.
    window_length
        Length ``L`` of the matched windows, in bars.
    horizon
        Bars to project forward from each match's anchor.
    distances
        Optional ``{ticker: (n_matches,) of distances}``, best-first per ticker.
        Mapped per ticker rather than passed as one flat array because the matching
        engine returns matches grouped by ticker: a flat array would have to be
        re-sliced against a grouping the caller has to reconstruct, and a mismatch
        there silently mislabels the evidence rather than failing.

    Returns
    -------
    ``ForecastPath``, or ``None`` when no match has a complete forward path.

    Notes
    -----
    **A missing or empty ticker contributes nothing rather than raising.**  The panel
    is built from whatever the archive happens to hold, so one ticker whose series is
    shorter than the requested horizon is ordinary rather than exceptional, and a
    path drawn from the remaining 29 matches is still the right answer.  A ticker
    with no ``close`` entry, or with indices outside its series, is dropped for the
    same reasons :func:`forecast_paths` drops an individual match.

    **Distance bookkeeping stays positional, per ticker.**  Within a ticker the rows
    and distances are consumed in lockstep by the same loop that decides survival, so
    a dropped match cannot shift a surviving one's distance onto the wrong window.
    Distances for tickers absent from ``distances`` are simply not recorded.

    ``starts`` on the result is therefore a concatenation across tickers, and is
    **not** meaningful without knowing which ticker each index belongs to.  Callers
    that need to attribute an evidence row must group by their own ``match_starts``
    keys; the chart below the UI groups on ticker for exactly that reason.  This is
    the one property of the single-series result that does not carry across, and it
    is why the cross-sectional caption reports a ticker breakdown rather than reusing
    the single-series wording.
    """
    horizon = int(horizon)
    length = int(window_length)
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if length <= 0:
        raise ValueError(f"window_length must be > 0, got {length}")

    rows: list = []
    # Positional across the *pooled* result, so ``starts``/``distances`` line up with
    # the rows exactly as they do in the single-series case.
    pooled_starts: list = []
    pooled_dists: list = []
    has_dist = False
    n_requested = 0

    for ticker in sorted(match_starts):
        starts = np.asarray(match_starts[ticker], dtype=np.int64).ravel()
        if starts.size == 0:
            continue
        # **Counted before it is known to be usable.**  A caller that asked for a
        # window and got no path must see it reflected in ``n_requested``; dropping it
        # from the denominator would make a shortfall look like a full sample, which
        # is the one thing this pair of numbers exists to prevent.
        n_requested += int(starts.size)
        close = np.asarray(closes.get(ticker, np.array([])), dtype=float)
        if close.ndim != 1:
            # Rejected loudly rather than skipped.  A 2-D "close" is a caller bug --
            # most likely a features matrix passed where a price series was expected
            # -- and silently skipping it would drop that ticker's evidence and
            # return a path computed from everything else, which reads as success.
            raise ValueError(
                f"close for {ticker!r} must be 1-D, got shape {close.shape}"
            )
        if close.size == 0:
            continue

        dist = None
        if distances is not None:
            candidate = np.asarray(distances.get(ticker, np.array([])),
                                   dtype=float).ravel()
            # A wrong-length array is ignored rather than mispaired, matching
            # ``forecast_paths``.  Refusing the whole call would be worse: one
            # ticker's bad array must not cost the other 29 their paths.
            if candidate.size == starts.size:
                dist = candidate
                has_dist = True

        anchors = starts + length - 1
        n = close.size
        for pos, a in enumerate(anchors):
            end = int(a) + horizon + 1
            if int(a) < 0 or end > n:
                continue
            base = close[int(a)]
            if not np.isfinite(base) or base <= 0:
                continue
            window = close[int(a):end]
            if not np.isfinite(window).all():
                continue
            rows.append((window / base - 1.0) * 100.0)
            pooled_starts.append(int(starts[pos]))
            if dist is not None:
                pooled_dists.append(float(dist[pos]))

    if not rows:
        return None

    kept = np.arange(len(rows), dtype=np.int64)
    dist_arr = np.asarray(pooled_dists, dtype=float) if has_dist and pooled_dists else None
    if dist_arr is not None and dist_arr.size != len(rows):
        # Distances survived for some tickers but not all; a partial pairing would
        # misattribute them, so all of them are dropped.
        dist_arr = None

    return _assemble_path(
        rows, kept, dist_arr, np.asarray(pooled_starts, dtype=np.int64), length, horizon,
        n_requested=n_requested,
    )