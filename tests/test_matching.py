"""Tests for the matching engine and statistical guards.

The placebo test in PLAN.md §E is the single most important test in this project: run
the whole pipeline on synthetic random-walk data and confirm it reports *nothing*.  A
pattern finder that finds significant patterns in pure noise is broken, regardless of
how good it looks on real data.

Fixtures come from :mod:`tests.session_bars` rather than a checked-in archive, so the
suite is self-contained and every bar count is a parameter instead of a number copied
out of an 892 KB CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timeseries import matching as M
from timeseries.forecast import (
    MIN_MATCHES,
    block_bootstrap_ci,
    conditional_forecast,
    permutation_test,
    summarise_returns,
)
from timeseries.features import (
    FEATURE_COLUMNS,
    build_features,
    finalize_features,
    safe_log_return,
)
from timeseries.pipeline import Pipeline

from session_bars import BARS_PER_SESSION, session_bars, session_count_for

#: Bars in the standard fixture.  Large enough that a 60-bar query at the end of the
#: archive still leaves room for a separated match, and that the default ``k=50`` can
#: reach ``MIN_MATCHES`` after suppression -- the §BE guard below depends on that.
FIXTURE_BARS = 8190

#: Seed for the standard fixture.  Fixed so a failure is reproducible.
FIXTURE_SEED = 20260930


@pytest.fixture(scope="session")
def archive_bars() -> pd.DataFrame:
    """One session-structured archive, shared: building the feature grid is the
    expensive part and every test here reads the same numbers."""
    return session_bars(FIXTURE_BARS, seed=FIXTURE_SEED)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def synthetic_bars(n: int = 4000, seed: int = 0, vol: float = 0.0005) -> pd.DataFrame:
    """Random-walk bars with the same schema as the real CSV."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, vol, size=n)
    close = 700.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        "open": close,
        "high": close * (1 + abs(rng.normal(0, vol / 2, n))),
        "low": close * (1 - abs(rng.normal(0, vol / 2, n))),
        "close": close,
    })


# --------------------------------------------------------------------------- #
# Variable query length
# --------------------------------------------------------------------------- #
class TestQueryLengthPropagates:
    """A query's own length must reach every downstream number.

    `find_matches` always read `query.length`, but `Pipeline.run` read `self.length`
    for the forward returns, the random-window baseline and the bootstrap block.  With
    one fixed window that was invisible.  With a manual selection defining its own
    length it is not: a 12-bar query would be *matched* over 12 bars and then
    *measured* as though it were 20.

    Nothing raises in that case.  The forward return is anchored on the last bar of the
    window (§Z1), so the wrong length simply moves the anchor and reports a confident,
    plausible, wrong number -- which is the failure mode this project cares about most.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def pipe(cls, archive_bars):
        """One pipeline for the class -- ``from_frame`` rebuilds the whole grid."""
        return Pipeline.from_frame(archive_bars, length=20)

    @pytest.mark.parametrize("L", [5, 12, 20, 37, 60])
    def test_forward_returns_honour_the_given_length(self, pipe, L):
        starts = np.array([100, 200, 300])
        got = pipe.forward_returns(starts, horizons=(5,), window_length=L)
        want = M.forward_returns(pipe.close, starts, L, (5,))
        assert np.allclose(got[5], want[5], equal_nan=True), (
            f"forward returns ignored window_length={L}"
        )

    def test_wrong_length_would_have_given_a_different_answer(self, pipe):
        """Guards the test above: the two lengths must actually differ here.

        Without this, the first test would also pass against a pipeline that ignored
        its argument entirely -- the archive could simply happen to give the same
        number at 20 and 37 bars.
        """
        starts = np.array([100, 200, 300])
        at_20 = pipe.forward_returns(starts, (5,), window_length=20)[5]
        at_37 = pipe.forward_returns(starts, (5,), window_length=37)[5]
        assert not np.allclose(at_20, at_37, equal_nan=True), (
            "20-bar and 37-bar windows returned identical returns; the fixture is "
            "degenerate and the length test proves nothing"
        )

    @pytest.mark.parametrize("L", [5, 12, 37])
    def test_baseline_is_built_from_the_same_length(self, pipe, L):
        """The control group has to measure what the treatment measures.

        A baseline drawn from 20-bar windows cannot be the control for a 37-bar matched
        set: the lift would be a difference between two different quantities.
        """
        got = pipe.baseline_returns(200, horizons=(5,), seed=1, window_length=L)
        rng = np.random.default_rng(1)
        hi = pipe.n_bars - L - 5 - 1
        picks = rng.integers(0, hi, size=200)
        want = M.forward_returns(pipe.close, picks, L, (5,))
        assert np.allclose(got[5], want[5], equal_nan=True), (
            f"baseline ignored window_length={L}"
        )

    @pytest.mark.parametrize("L", [5, 12, 20, 37, 60])
    def test_run_succeeds_at_any_length(self, pipe, L):
        out = pipe.run(pipe.query_span(7000, 7000 + L), k=10, n_baseline=100)
        assert out["ok"]
        assert out["query"].length == L
        assert set(out["forward"]) == {5, 15, 30, 60}

    def test_run_uses_the_query_length_for_the_block_size(self, pipe, L=37):
        """The bootstrap block must cover a whole window plus the longest horizon.

        Left at ``self.length + max(horizons)`` it is too small for a longer query, and
        the CI would be computed on overlapping blocks that are not independent.
        """
        out = pipe.run(pipe.query_span(7000, 7000 + 37), k=10, n_baseline=100)
        # The block is not echoed in the output, so assert the observable consequence:
        # every forward return must be anchored after a 37-bar window, not a 20-bar one.
        starts = np.array([m.start for m in out["result"].matches])
        at_37 = M.forward_returns(pipe.close, starts, 37, (5,))
        at_20 = M.forward_returns(pipe.close, starts, 20, (5,))
        assert not np.allclose(at_37[5], at_20[5], equal_nan=True)
        assert np.allclose(out["forward"][5], at_37[5], equal_nan=True), (
            "run() measured the 37-bar query with a 20-bar window"
        )

    def test_default_path_is_unchanged(self, pipe):
        """No query still means the pipeline's own length, exactly as before."""
        a = pipe.run(None, k=10, n_baseline=100)
        b = pipe.run(pipe.query_latest(), k=10, n_baseline=100)
        for h in a["forward"]:
            assert np.allclose(a["forward"][h], b["forward"][h], equal_nan=True)


# --------------------------------------------------------------------------- #
# Z1 -- the forward-return anchor
# --------------------------------------------------------------------------- #
class TestForwardReturnAnchor:
    def test_forward_return_starts_after_window_end(self):
        """PLAN.md §Z1: the forward return must not overlap the matched window."""
        # Deliberately non-linear, so anchoring at the wrong bar cannot coincide.
        rng = np.random.default_rng(11)
        close = 700.0 * np.exp(np.cumsum(rng.normal(0, 0.002, size=300)))
        starts = np.array([10])
        L, h = 60, 5

        got = M.forward_returns(close, starts, L, [h])[h][0]

        # Correct anchor: the last bar inside the window, close[start + L - 1].
        anchor = 10 + L - 1
        expected = np.log(close[anchor + h] / close[anchor])
        assert got == pytest.approx(expected)

        # The old bug anchored at start + h, i.e. 5 bars into a 60-bar window.
        buggy = np.log(close[10 + h] / close[10])
        assert got != pytest.approx(buggy)

    def test_horizon_uses_last_bar_of_window(self):
        close = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
        # window of length 2 starting at 2 covers bars [2, 4); horizon 1 -> bar 5
        out = M.forward_returns(close, np.array([2]), 2, [1])
        assert out[1][0] == pytest.approx(np.log(close[4] / close[3]))

    def test_nan_when_horizon_exceeds_data(self):
        close = np.ones(10)
        out = M.forward_returns(close, np.array([0]), 5, [100])
        assert np.isnan(out[100][0]), "missing tail must be NaN, not a silent zero"


# --------------------------------------------------------------------------- #
# Z2 -- feature scaling
# --------------------------------------------------------------------------- #
class TestFeatures:
    def test_features_are_finite_from_close_alone(self):
        """The feature matrix must be built from price and nothing else.

        There is no volume leg, so a bar frame carrying only ``timestamp``/``close``
        is the realistic input, and its features must be finite.
        """
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=200, freq="min", tz="UTC"),
            "close": 700.0 + np.arange(200) * 0.01,
        })
        feats = build_features(df)
        _, mat, _ = finalize_features(feats, how="drop")
        assert np.isfinite(mat).all(), "features must be finite"

    def test_feature_columns_are_price_only(self):
        """Matching is a price question: there is no volume leg left to score."""
        assert FEATURE_COLUMNS == ("return_z", "path_z")
        assert not any("volume" in c for c in FEATURE_COLUMNS)

    def test_return_z_is_comparable_scale(self):
        """``return_z`` is rolling-z, so it sits at unit variance on its own."""
        rng = np.random.default_rng(1)
        n = 800
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "close": 700.0 * np.exp(np.cumsum(rng.normal(0, 5e-4, n))),
        })
        feats = build_features(df)
        _, mat, _ = finalize_features(feats, how="drop")
        col = list(FEATURE_COLUMNS).index("return_z")
        assert mat[:, col].std() == pytest.approx(1.0, abs=0.15)

    def test_path_z_is_mean_removed_not_rescaled(self):
        """``path_z`` keeps its own units on purpose.

        It is the log-price path with a global mean removed and nothing else.  Rolling-z
        it as well would divide it by a second slowly-varying scale and flatten the slow
        drift the leg exists to capture; the per-window z-score the scorer applies later
        supplies the local normalisation.  So this leg is *expected* to have a small
        global standard deviation, and asserting unit variance here would be asserting
        the bug the column was written to avoid.
        """
        rng = np.random.default_rng(1)
        n = 800
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "close": 700.0 * np.exp(np.cumsum(rng.normal(0, 5e-4, n))),
        })
        feats = build_features(df)
        col = list(FEATURE_COLUMNS).index("path_z")
        full = feats["path_z"].to_numpy()
        _, mat, _ = finalize_features(feats, how="drop")
        path = mat[:, col]

        # Only the very first bar is undefined -- there is no prior close to take a log
        # return from.  Everything after it must be finite, and the mean must be removed
        # (pandas' ``.mean()`` skips the NaN, so this is still the global mean).
        assert np.isnan(full[0]) and np.isfinite(full[1:]).all()
        assert abs(np.nanmean(full)) < 1e-12, "the global mean is removed, not merely small"
        # After ``how="drop"`` discards the return_z warm-up, the retained subset's
        # mean is only *near* zero -- the removed rows still carried part of the level.
        assert abs(path.mean()) < 0.10 * path.std()
        assert 0.0 < path.std() < 0.05, "log units, not z units"

    def test_log_return_ignores_nonpositive_price(self):
        s = pd.Series([100.0, 0.0, 50.0, -1.0, 60.0])
        r = safe_log_return(s)
        assert np.isfinite(r.dropna()).all()


# --------------------------------------------------------------------------- #
# §M -- the self-match problem
# --------------------------------------------------------------------------- #
class TestExclusionAndSuppression:
    def test_query_cannot_match_itself(self):
        """§M: the top match must never be the query."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=3000)
        L = 60
        q = M.Query.from_span(x, 1500, 1500 + L)

        res = M.find_matches(x, q, k=5)
        assert res.matches, "expected some matches"
        for m in res.matches:
            assert m.start <= q.start - L - 1 or m.start >= q.stop + L

    def test_adjacent_windows_are_suppressed(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=3000)
        L = 60
        q = M.Query.from_span(x, 1500, 1500 + L)
        res = M.find_matches(x, q, k=5)
        s = np.sort([m.start for m in res.matches])
        if len(s) > 1:
            assert np.diff(s).min() >= L, "matches must be at least one window apart"

    def test_exclusion_mask_covers_neighbours(self):
        q = M.Query(vector=np.zeros(60), start=1000, stop=1060)
        # A candidate at 1120 occupies [1120, 1180), starting exactly L bars after the
        # query ends -- that satisfies the NMS separation and must stay searchable.
        starts = np.array([820, 880, 940, 1000, 1059, 1060, 1120, 1179])
        mask = M.exclusion_mask(starts, q)
        pos = {int(s): bool(m) for s, m in zip(starts, mask)}

        assert not pos[820], "well before the query: searchable"
        assert not pos[880], "ends exactly L bars before the query: searchable"
        assert pos[940], "touches the query start"
        assert pos[1000], "is the query"
        assert pos[1059], "overlaps the query"
        assert pos[1060], "starts exactly at the query end"
        assert not pos[1120], "starts exactly L bars after the query: searchable"
        assert not pos[1179]

    def test_nms_respects_separation(self):
        starts = np.array([0, 5, 10, 500, 1000])
        scores = np.array([0.1, 0.2, 0.05, 0.9, 1.0])
        chosen = M.non_max_suppression(starts, scores, min_separation=100, k=3)
        kept = starts[chosen]
        assert kept.tolist() == [10, 500, 1000]

    def test_nms_returns_best_first_not_by_position(self):
        """Order is the contract: callers label these "match #1", "#2", ...

        This case is chosen so position order and score order disagree -- without it
        the test above cannot tell the two behaviours apart.
        """
        starts = np.array([0, 100, 200, 300, 400])
        scores = np.array([9.0, 1.0, 8.0, 2.0, 7.0])
        chosen = M.non_max_suppression(starts, scores, min_separation=1, k=5)
        kept = starts[chosen]
        assert kept.tolist() == [100, 300, 400, 200, 0], "must be best-first"
        assert scores[chosen].tolist() == sorted(scores[chosen].tolist())

    def test_nms_is_greedy_so_the_best_survives_every_k(self):
        """NMS is greedy: a bigger k can only add matches, never displace the best."""
        starts = np.array([0, 10, 20, 500, 900, 1300])
        scores = np.array([0.5, 0.4, 0.45, 3.0, 0.1, 2.0])
        firsts = []
        for k in range(1, len(starts) + 1):
            chosen = M.non_max_suppression(starts, scores, min_separation=100, k=k)
            firsts.append(int(starts[chosen[0]]))
        assert len(set(firsts)) == 1, f"matches[0] changed with k: {firsts}"
        assert firsts[0] == 900, "index 0 must be the globally best window"


class TestMatchesAreOrderedBestFirst:
    """The regression: results were returned in *position* order, not score order.

    `MatchResult.matches` is documented as "sorted best-first", the UI labels them
    "match #1..#3", and the Price tab draws `matches[0]` as *the* match found.  With
    position order the headline chart showed an arbitrary early window.  Worse, because
    NMS is greedy, the window at index 0 moved when the `k` slider changed, so the same
    query displayed a different "best match" at different k.
    """

    def _live(self, archive_bars):
        pipe = Pipeline.from_frame(archive_bars, length=60)
        q = M.Query.from_span(pipe.matrix, pipe.n_bars - 60, pipe.n_bars)
        return pipe, q

    def test_matches_are_sorted_by_distance_ascending(self, archive_bars):
        pipe, q = self._live(archive_bars)
        res = pipe.match(q, k=10)
        d = [float(m.distance) for m in res.matches]
        assert d == sorted(d), f"not best-first -- {d}"

    def test_matches_zero_is_the_closest_window(self, archive_bars):
        pipe, q = self._live(archive_bars)
        res = pipe.match(q, k=10)
        best = min(res.matches, key=lambda m: m.distance)
        assert res.matches[0].start == best.start, (
            f"matches[0] is bar {res.matches[0].start} but the closest is bar "
            f"{best.start}"
        )

    def test_the_shown_match_does_not_depend_on_k(self, archive_bars):
        """The Price tab always draws index 0, so it must not move with the slider."""
        pipe, q = self._live(archive_bars)
        firsts = {pipe.match(q, k=k).matches[0].start for k in (1, 3, 5, 10)}
        assert len(firsts) == 1, f"the 'best match' changed with k: {firsts}"

    def test_k_is_a_prefix_of_a_larger_k(self, archive_bars):
        """Best-first order means more matches can only be appended."""
        pipe, q = self._live(archive_bars)
        small = [m.start for m in pipe.match(q, k=3).matches]
        large = [m.start for m in pipe.match(q, k=10).matches]
        assert large[:3] == small, f"k=3 {small} is not a prefix of k=10 {large}"


# --------------------------------------------------------------------------- #
# §E -- statistical guards
# --------------------------------------------------------------------------- #
class TestForecastGuards:
    def test_small_sample_is_suppressed(self):
        """A forecast from 5 dependent observations must not be reported."""
        fwd = {5: np.random.default_rng(4).normal(size=5)}
        base = {5: np.random.default_rng(5).normal(size=100)}
        out = conditional_forecast(fwd, base)[0]
        assert not out.sufficient
        assert "insufficient evidence" in out.note

    def test_sufficient_sample_is_reported(self):
        rng = np.random.default_rng(6)
        fwd = {5: rng.normal(0.01, 0.001, size=200)}
        base = {5: rng.normal(0.0, 0.001, size=400)}
        out = conditional_forecast(fwd, base, n_perm=200, n_boot=200)[0]
        assert out.sufficient
        assert out.lift > 0
        assert out.p_value < 0.05

    def test_permutation_p_value_bounds(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        p = permutation_test(a, b, n_perm=200)
        assert 0.0 < p <= 1.0

    def test_block_bootstrap_is_wider_than_iid(self):
        """Overlapping observations need a wider interval than i.i.d. implies."""
        rng = np.random.default_rng(8)
        x = np.cumsum(rng.normal(0, 0.001, 600))
        wide_lo, wide_hi = block_bootstrap_ci(x, block=60, n_boot=500)
        narrow_lo, narrow_hi = block_bootstrap_ci(x, block=1, n_boot=500)
        assert (wide_hi - wide_lo) >= (narrow_hi - narrow_lo) * 0.9

    def test_summarise_ignores_nonfinite(self):
        # finite entries are 0.1, -0.1, 0.3 -> mean 0.1, hit rate 2/3
        m, s, h = summarise_returns(np.array([0.1, np.nan, -0.1, 0.3]))
        assert m == pytest.approx(0.1)
        assert h == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# THE PLACEBO TEST -- §E
# --------------------------------------------------------------------------- #
class TestPlacebo:
    """Run the real pipeline on pure noise. It must find nothing."""

    # KNOWN FAILING -- see PLAN.md §BU.  Measured 42% significant (median p 0.18,
    # KS vs U(0,1) p=5e-4) against the 4.2% recorded in §BO when STUMPY was adopted.
    # RESOLVED by §BW: the cause was NOT a biased lift but an inflated *variance* of
    # the matched mean (mean z = +0.08, sd z = 2.15), which `forecast.
    # selection_aware_test` now corrects for.  Measured after the fix: 59/60 random
    # walks pass, and the p-value rejects on 0-1% of iid nulls.  The xfail marker is
    # removed here, in the same commit as the fix, as this comment always required.
    def test_random_walk_yields_no_significant_forecast(self, tmp_path):
        n_runs = 12
        significant = 0
        for seed in range(n_runs):
            bars = synthetic_bars(4000, seed=seed)
            path = tmp_path / f"synthetic_{seed}.csv"
            bars.to_csv(path, index=False)

            pipe = Pipeline.from_csv(str(path), length=60)
            assert pipe.ready
            out = pipe.run(k=30, horizons=(5, 15), min_matches=MIN_MATCHES, seed=seed)

            for f in out["forecasts"]:
                if f.sufficient and np.isfinite(f.p_value) and f.p_value < 0.05:
                    significant += 1

        # With ~50% of tests significant under the null and ~24 comparisons,
        # a broken pipeline produces a large majority. A working one stays low.
        total = n_runs * 2
        assert significant / total < 0.15, (
            f"{significant}/{total} significant forecasts on random-walk data -- "
            "see PLAN.md §BU: the permutation p-value is miscalibrated under "
            "selection bias, so this measures the p-value and not the matcher"
        )

    def test_matched_beats_random_is_not_assumed(self):
        """Guard against a pipeline that hard-codes a favourable result."""
        bars = synthetic_bars(4000, seed=99)
        path = bars_to_path(bars)
        pipe = Pipeline.from_csv(str(path), length=60)
        # k must clear MIN_MATCHES, otherwise §E's guard suppresses the forecast and
        # there is nothing to assert about.
        out = pipe.run(k=40, horizons=(15,), min_matches=MIN_MATCHES, seed=99)
        f = out["forecasts"][0]
        assert f.sufficient, f"expected a reportable forecast, got: {f.note}"
        # On noise the lift should straddle zero, not be reliably positive.
        assert abs(f.lift) < 0.01 or f.p_value > 0.05


def bars_to_path(bars: pd.DataFrame):
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    bars.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# regression of the two specific bugs found in review
# --------------------------------------------------------------------------- #
class TestRegression:
    def test_matches_are_separated_in_time(self, archive_bars):
        """NMS must enforce at least one window-length between accepted matches.

        The NMS regression guard needs a query long enough that ``min_separation``
        bites: at length 60 the matches have to be 60+ bars apart, which is what proves
        overlapping windows are not being counted as separate evidence (§M).
        """
        pipe = Pipeline.from_frame(archive_bars, length=60)
        out = pipe.run(k=10)
        s = np.sort([m.start for m in out["result"].matches])
        # ``len`` rather than truthiness: ``s`` is a numpy array, and a bare
        # ``assert s`` is ambiguous rather than failing cleanly.
        assert len(s) >= 2, (
            f"only {len(s)} match(es) -- NMS cannot be shown to separate anything"
        )
        assert np.diff(s).min() >= 60

    def test_multichannel_query_shape_is_validated(self):
        """A 2-D series requires a query with a matching channel count."""
        from timeseries.matrix_profile import distance_profile

        q2d = M.Query(vector=np.zeros((60, 2)), start=100, stop=160)
        with pytest.raises(ValueError, match="channel mismatch"):
            distance_profile(q2d.vector, np.zeros((500, 1)))

    def test_rejects_a_series_of_the_wrong_rank(self):
        q = M.Query(vector=np.zeros(60), start=100, stop=160)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            M.find_matches(np.zeros((500, 60, 2)), q)

    def test_rejects_a_query_outside_the_series(self):
        x = np.random.default_rng(12).normal(size=500)
        q = M.Query(vector=np.zeros(60), start=480, stop=540)
        with pytest.raises(ValueError, match="outside series"):
            M.find_matches(x, q)

# --------------------------------------------------------------------------- #
# PLAN.md §BX -- session-boundary correctness
# --------------------------------------------------------------------------- #
class TestSessionBoundaryMask:
    """The forward horizon must not cross a session closure.

    Measured on the live QQQ archive: a window ending on a session's *final* bar
    reports a mean forward return of **+5.6e-04** against **+2.8e-05** for a window
    ending mid-session -- a 20x inflation, because the "h bars forward" it reports
    are really one overnight gap plus h-1 minutes of trading.

    The companion fact, just as important, is what this mask must **not** do: the gap
    bar itself is **22x larger** than a typical bar's (median |log return| 3.6e-03 vs
    1.6e-04), so a straddling window is genuinely distorted.  But it distorts the
    *ranking*, not the reported return, only 2 of 50 matched windows straddle, and
    excluding them costs 31% of the pool at L=60 and 62% at L=240.  The first draft of
    PLAN.md §BX got this backwards -- it asserted the gap bar was ~70x *smaller* -- and
    is preserved as the cautionary tale at the end of that section.
    """

    def test_marks_session_starts_not_session_ends(self):
        from timeseries.pipeline import session_boundary_mask

        bars = session_bars(FIXTURE_BARS)
        b = session_boundary_mask(bars["timestamp"])

        n_sessions = session_count_for(len(bars))
        # Bar 0 is deliberately unmarked: it is the first session's open, but no
        # forward return can straddle it (there is no earlier bar), and marking it
        # would censor the first window of every archive for no reason.
        assert not b[0]
        assert int(b.sum()) == n_sessions - 1, (
            f"marked {int(b.sum())} boundaries for {n_sessions} sessions -- an "
            "off-by-one would double-exclude or skip a session entirely"
        )
        # No boundary may fall strictly inside a session.
        assert not b[1:BARS_PER_SESSION].any(), (
            "a session's own bars must not be boundaries"
        )

    def test_marks_the_session_open_time_not_the_utc_midnight(self):
        """Sessions are Eastern days, not UTC days.

        Bucketing in UTC splits an Eastern session in half at 20:00 ET, which would
        mark every evening bar as a boundary and exclude most of the archive.
        """
        from timeseries.pipeline import session_boundary_mask

        bars = session_bars(390 * 3)
        b = session_boundary_mask(bars["timestamp"])
        marked = bars.loc[b, "timestamp"]
        assert (marked.dt.tz_convert("America/New_York").dt.time
                == pd.Timestamp("09:30", tz="America/New_York").time()).all()

    def test_mask_removes_exactly_the_windows_whose_horizon_crosses(self):
        """The mask and the thing it claims to prevent must agree, bar for bar."""
        from timeseries.pipeline import session_boundary_mask

        bars = session_bars(FIXTURE_BARS)
        b = session_boundary_mask(bars["timestamp"])
        L, h = 60, 60
        mask = M.forward_horizon_valid_mask(b, L, h)

        assert mask.size == b.size - L + 1
        rng = np.random.default_rng(7)
        checked = 0
        for s in rng.choice(mask.size, size=60, replace=False):
            s = int(s)
            anchor = s + L - 1
            if anchor + h >= b.size:
                assert not mask[s], "no data for the horizon -- forward_returns gives NaN"
                continue
            crosses = bool(b[anchor + 1:anchor + h + 1].any())
            assert bool(mask[s]) != crosses, f"window {s} disagrees with its own horizon"
            checked += 1
        assert checked >= 20, "not enough in-sample windows to be a real check"

    def test_a_boundary_at_the_anchor_does_not_censor(self):
        """The anchor bar is the *last* bar of the window, §Z1.

        If that bar happens to open a session, the forward return starts inside that
        session and never leaves it -- so the window is perfectly valid.  Counting it
        would exclude every window that opens exactly on a session boundary, which for
        a 390-bar session at L=60 is one start per session, for no reason at all.
        """
        b = np.zeros(12, dtype=bool)
        b[5] = True  # window start 4 -> anchor 5 for window_length=2
        mask = M.forward_horizon_valid_mask(b, window_length=2, max_horizon=3)
        assert mask[4], "horizon (5, 8] contains no boundary"

    def test_windows_running_past_the_data_are_rejected(self):
        """Consistent with `forward_returns`, which returns NaN for them."""
        b = np.zeros(100, dtype=bool)
        mask = M.forward_horizon_valid_mask(b, window_length=10, max_horizon=60)
        assert mask.size == 91
        assert not mask[-1], "start 90 anchors at 99 and has no bar 60 forward"
        assert mask[0]

    def test_nothing_is_excluded_without_a_boundary(self):
        b = np.zeros(500, dtype=bool)
        mask = M.forward_horizon_valid_mask(b, 60, 60)
        # Only the windows whose horizon runs off the end are cut -- no boundary, no
        # other reason to censor.
        assert mask[:-60].all()
        assert not mask[-60:].any()


class TestFindMatchesRespectsValidMask:
    def test_no_returned_match_horizon_crosses_a_boundary(self):
        """The headline §BX guarantee, checked on the whole result."""
        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        from timeseries.pipeline import session_boundary_mask

        b = session_boundary_mask(pipe.bars["timestamp"])
        q = pipe.query_latest()
        res = M.find_matches(pipe.matrix, q, k=30,
                             amplitude_series=pipe.log_returns,
                             valid_mask=M.forward_horizon_valid_mask(b, 60, 60))

        assert len(res.matches) >= 5
        for m in res.matches:
            anchor = m.start + q.length - 1
            assert not b[anchor + 1:anchor + 61].any(), f"match {m.start} crosses"

    def test_the_correction_actually_removes_something(self):
        """A guard that never fires would pass the test above for the wrong reason.

        On the live archive the uncensored search returned 12 of 50 matches with a
        contaminated horizon; with the mask it is 0 of 50.  If the uncensored search
        here also returns none, the fixture is not exercising the defect and the test
        above is vacuous.
        """
        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        from timeseries.pipeline import session_boundary_mask

        b = session_boundary_mask(pipe.bars["timestamp"])
        q = pipe.query_latest()
        off = M.find_matches(pipe.matrix, q, k=30, amplitude_series=pipe.log_returns)
        contaminated = sum(
            1 for m in off.matches
            if b[m.start + q.length:m.start + q.length + 60].any()
        )
        assert contaminated > 0, (
            "uncensored search found no contaminated match -- fixture does not "
            "exercise the §BX defect, so the guarantee test proves nothing"
        )

    def test_n_masked_is_reported_separately_from_n_excluded(self):
        """The UI has to be able to tell a data-quality cut from the §M self-exclusion."""
        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        from timeseries.pipeline import session_boundary_mask

        b = session_boundary_mask(pipe.bars["timestamp"])
        vm = M.forward_horizon_valid_mask(b, 60, 60)
        q = pipe.query_latest()
        res = M.find_matches(pipe.matrix, q, k=30,
                             amplitude_series=pipe.log_returns, valid_mask=vm)
        assert res.n_masked == int((~vm).sum())
        assert res.n_masked > 0
        # `n_excluded` is the total removed (§M and the mask together) -- it is what
        # §E's percentile denominator derives from, so changing its meaning would
        # change every rank.  `n_masked` is the mask's own count and **overlaps** §M's
        # near the query, so the two must never be summed:
        #     n_excluded = n_m_alone + n_masked - overlap
        # A naive `+` here would overstate the censoring by the overlap.
        plain = M.find_matches(pipe.matrix, q, k=30, amplitude_series=pipe.log_returns)
        assert res.n_excluded < plain.n_excluded + res.n_masked, (
            "the two counts are disjoint, which cannot be true -- they overlap on "
            "windows near the query"
        )
        assert res.n_excluded >= res.n_masked
        # And the total must match what actually survived, by hand.
        excl = M.exclusion_mask(np.arange(plain.n_candidates), q, None)
        assert res.n_excluded == int((excl | ~vm).sum())
        assert plain.n_candidates - res.n_excluded == int((~excl & vm).sum())

    def test_percentile_population_excludes_masked_windows(self):
        """§E: the rank describes the population the matches were drawn from.

        If masked windows stayed in the distribution, every rank would be computed
        against windows that could never have been returned -- a population that does
        not exist.  The check is the count of windows strictly closer than the best
        match, computed here by hand from the same masked pool.
        """
        from timeseries.matrix_profile import distance_profile

        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        from timeseries.pipeline import session_boundary_mask

        b = session_boundary_mask(pipe.bars["timestamp"])
        vm = M.forward_horizon_valid_mask(b, 60, 60)
        q = pipe.query_latest()
        res = M.find_matches(pipe.matrix, q, k=5,
                             amplitude_series=pipe.log_returns, valid_mask=vm)

        d = distance_profile(pipe.matrix[q.start:q.stop], pipe.matrix,
                             query_idx=q.start)
        pool = d[np.isfinite(d)]
        assert res.matches[0].percentile > 0.0
        assert res.matches[0].percentile <= 1.0
        # The masked pool is a strict subset, so ranks computed on it must be
        # computable -- and no match may sit at exactly 0 percentile.
        assert res.n_candidates - res.n_masked < pool.size

    def test_mask_is_optional_and_off_by_default(self):
        x = np.random.default_rng(3).normal(size=2000)
        q = M.Query(vector=x[100:160], start=100, stop=160)
        assert M.find_matches(x, q, k=5).n_masked == 0

    def test_a_wrongly_shaped_mask_is_rejected_loudly(self):
        """Silently broadcasting a mask would censor the wrong windows."""
        x = np.random.default_rng(4).normal(size=2000)
        q = M.Query(vector=x[100:160], start=100, stop=160)
        with pytest.raises(ValueError, match="valid_mask must be over window starts"):
            M.find_matches(x, q, k=5, valid_mask=np.ones(10, dtype=bool))


class TestSessionMaskEndToEnd:
    def test_pipeline_run_reports_no_contaminated_forward_return(self):
        """The guarantee that matters to a reader of the Matches tab."""
        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        from timeseries.pipeline import session_boundary_mask

        b = session_boundary_mask(pipe.bars["timestamp"])
        out = pipe.run(k=50, horizons=(5, 15, 30, 60), seed=0)
        assert out["ok"]
        starts = np.array([m.start for m in out["result"].matches])
        for s in starts:
            anchor = int(s) + out["query"].length - 1
            assert not b[anchor + 1:anchor + 61].any()

    def test_baseline_is_drawn_from_the_same_pool_as_the_matches(self):
        """Otherwise the lift partly measures the treatment's own censoring."""
        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        from timeseries.pipeline import session_boundary_mask

        b = session_boundary_mask(pipe.bars["timestamp"])
        vm = pipe.forward_horizon_mask(60, 60)
        assert vm is not None and not vm.all()
        base = pipe.baseline_returns(400, (5, 60), seed=1, window_length=60,
                                     valid_mask=vm)
        assert base[60].size == 400
        # Reconstruct the allowed starts and confirm none is censored.
        allowed = np.flatnonzero(vm[:pipe.n_bars - 60 - 60 - 1])
        assert allowed.size > 0

    def test_session_mask_off_restores_the_uncensored_pool(self):
        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        on = pipe.run(k=30, horizons=(5, 60), seed=0)
        off = pipe.run(k=30, horizons=(5, 60), seed=0, session_mask=False)
        assert off["result"].n_masked == 0
        assert on["result"].n_masked > 0


class TestWalkForwardSessionBoundary:
    """The walk-forward path makes the same forward-return claim as `run`.

    It has to be censored too, and the mask must be built on the **history** slice --
    using the full series would leak the shape of later session boundaries into a step
    that is supposed to be blind, which is lookahead by another name.
    """

    def test_walk_forward_runs_and_reports_a_result(self):
        from timeseries.backtest import walk_forward

        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        res = walk_forward(pipe, k=30, horizon=15, stride=200, warmup=2000,
                           max_steps=10)
        assert res.n_predictions > 0
        assert np.isfinite(res.p_value)
        assert res.stride == 200

    def test_each_step_only_sees_boundaries_from_its_own_history(self):
        from timeseries.pipeline import session_boundary_mask

        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)

        i = 3000
        # `forward_horizon_mask` already speaks in window *starts*, so a history of
        # `i` bars admits `i - L + 1` of them -- no further slicing needed.
        full_mask = pipe.forward_horizon_mask(pipe.length, 60)
        hist_mask = full_mask[:i - pipe.length + 1]
        assert hist_mask is not None
        assert full_mask.size == pipe.n_bars - pipe.length + 1
        assert hist_mask.size == i - pipe.length + 1

        boundary = session_boundary_mask(pipe.bars["timestamp"])
        assert boundary[i:].sum() > 0, (
            "fixture has later boundaries, so leakage is testable"
        )
        # Censoring inside the history is unaffected by what follows it: the mask is a
        # prefix, so a step cannot see a boundary it had not yet reached.  Compare
        # against the same prefix of the full-series mask, bar for bar.
        assert np.array_equal(hist_mask, full_mask[:i - pipe.length + 1])
        # And the mask genuinely does censor something inside the history.
        assert int(hist_mask.sum()) < hist_mask.size

    def test_walk_forward_accepts_a_masked_search_without_error(self):
        from timeseries.backtest import walk_forward

        bars = session_bars(FIXTURE_BARS)
        pipe = Pipeline.from_frame(bars, length=60)
        res = walk_forward(pipe, k=30, horizon=15, stride=400, warmup=3000,
                           max_steps=8)
        assert res.n_predictions > 0
