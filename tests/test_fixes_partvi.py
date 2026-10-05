"""Regression tests for the four Part VI fixes, plus the placebo harness.

Each test names the plan section it protects.  These are the guards that stop the
defects from coming back -- particularly BC (normalisation asymmetry), which is
invisible in the output: every number stays plausible while the ranking is subtly
wrong.

    §BC  query and library must be normalised identically
    §BD  a metric that cannot be ranked must not report a percentile
    §BE  default k must be able to satisfy MIN_MATCHES
    §BF  overnight closure must not be counted as a data hole
    §E   the placebo test must find nothing in a random walk

The archive-dependent guards use :mod:`tests.session_bars` rather than a checked-in
CSV.  They previously called ``pytest.skip`` when the file was absent, which meant
§BE and §BF could stop running without anything failing -- two guards for exactly the
bug class this project exists to prevent, going quietly missing.  A seeded generator
has no absent case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timeseries import matching as M
from timeseries.features import quality_report
from timeseries.forecast import MIN_MATCHES
from timeseries.pipeline import Pipeline, DEFAULT_K
from timeseries.placebo import random_walk_bars, run_placebo

from session_bars import session_bars

#: Large enough that the default k=50 clears MIN_MATCHES after suppression (§BE) and
#: that 60-bar queries leave room for separated matches (§BF needs many sessions).
FIXTURE_BARS = 8190
FIXTURE_SEED = 20260930


@pytest.fixture(scope="module")
def archive_bars() -> pd.DataFrame:
    """Session-structured bars: 21 real trading days with 20 overnight closures."""
    return session_bars(FIXTURE_BARS, seed=FIXTURE_SEED)


# --------------------------------------------------------------------------- #
# §BC -- normalisation symmetry
# --------------------------------------------------------------------------- #
class TestPerWindowNormalisation:
    """§BC: the query was z-scored per window while the library was not.

    Consequence: a globally quiet stretch and a locally volatile one are compared on
    different scales, so the distance conflates "shape similarity" with "how volatile
    was this stretch".

    With STUMPY as the only scorer this asymmetry is now structurally impossible --
    ``normalize=True`` z-scores the query and every subsequence in the same pass.  What
    is asserted here is that the guarantee actually holds on the numbers that come out,
    which is the property the old asymmetry broke.
    """

    def test_a_query_built_at_any_scale_scores_the_same(self):
        """§BC's real test: the query must be normalised exactly once.

        ``Query.from_span`` z-scores its vector so the caller can *draw* it.  If the
        scorer trusted that vector instead of re-reading the raw span from the series,
        the query would be normalised twice and every candidate once -- leaving the
        query's scale smaller than the candidates' by exactly that factor, which is
        the asymmetry §BC was filed about.
        """
        feats = np.random.default_rng(0).normal(size=(600, 2))
        q = M.Query.from_span(feats, 300, 340, per_window=True)

        # ``amplitude_weight=0`` so the distances below are the pure shape distance.
        # At its default, ``find_matches`` folds in the §BD amplitude penalty and the
        # reported distance is intentionally not the raw MASS value.
        as_built = M.find_matches(feats, q, k=5, amplitude_weight=0.0)
        # A query object identical except for an already-normalised vector must give
        # the identical answer, because the vector is not what gets scored.
        rebuilt = M.Query(vector=M.zscore(q.vector[:, 0]), start=q.start,
                          stop=q.stop, label=q.label)
        again = M.find_matches(feats, rebuilt, k=5, amplitude_weight=0.0)
        assert [m.start for m in again.matches] == [m.start for m in as_built.matches]

        from timeseries.matrix_profile import distance_profile

        raw = distance_profile(feats[300:340], feats, query_idx=300)
        for m in as_built.matches:
            assert m.distance == pytest.approx(float(raw[m.start]), rel=1e-9)

    def test_query_built_from_the_series_sits_on_the_same_scale(self):
        """A query built from the series must sit on the scale STUMPY will score it at."""
        feats = np.random.default_rng(1).normal(size=(400, 2))
        q = M.Query.from_span(feats, 100, 130, per_window=True)
        assert q.vector.mean() == pytest.approx(0.0, abs=1e-9)
        assert q.vector.std() == pytest.approx(1.0, abs=1e-9)

    def test_constant_window_does_not_poison_the_search(self):
        """A flat window is a valid (if dull) candidate, not a divide-by-zero."""
        feats = np.zeros((100, 2))
        assert np.all(np.isfinite(M.zscore(feats[:, 0])))

    def test_zscore_without_per_window_is_the_raw_slice(self):
        feats = np.random.default_rng(2).normal(size=(100, 1))
        q = M.Query.from_span(feats, 10, 20, per_window=False)
        assert np.allclose(q.vector, feats[10:20])

    def test_scoring_is_invariant_to_the_queries_own_scale(self):
        """§BC's real test: a globally rescaled series must rank identically.

        If only one side of each comparison were normalised, scaling the series would
        change the distances and could reorder near-tied candidates.  Normalising both
        sides inside the scorer makes the ranking scale-free by construction.
        """
        feats = np.random.default_rng(5).normal(size=(800, 2))
        q = M.Query.from_span(feats, 400, 460, per_window=True)

        base = M.find_matches(feats, q, k=5)
        scaled = M.find_matches(feats * 137.5 + 42.0, q, k=5)

        assert [m.start for m in base.matches] == [m.start for m in scaled.matches]
        for a, b in zip(base.matches, scaled.matches):
            assert a.distance == pytest.approx(b.distance, rel=1e-6)


# --------------------------------------------------------------------------- #
# §BD -- a metric that cannot be ranked must not report a percentile
# --------------------------------------------------------------------------- #
class TestPercentileIsAlwaysReportable:
    """§BD: the DTW path ranked a DTW distance against a *Euclidean* distribution.

    The percentile is the module's honesty feature, so a wrong one is worse than none.
    The chosen fix was to remove the unrankable metric rather than blank the column --
    which is only sound if every remaining match carries a real rank.
    """

    def test_every_match_reports_a_finite_percentile(self):
        feats = np.random.default_rng(3).normal(size=(400, 2))
        q = M.Query.from_span(feats, 0, 30, per_window=True)
        res = M.find_matches(feats, q, k=5)
        assert res.matches
        assert all(np.isfinite(m.percentile) for m in res.matches)

    def test_percentile_is_a_fraction_of_the_searchable_population(self):
        """§E: the rank is against every window that could have matched."""
        feats = np.random.default_rng(4).normal(size=(600, 1))
        L = 20
        q = M.Query.from_span(feats, 0, L, per_window=True)
        res = M.find_matches(feats, q, k=5)

        n_searchable = res.n_candidates - res.n_excluded
        assert n_searchable > 0
        best = min(res.matches, key=lambda m: m.distance)
        assert best.percentile == pytest.approx(1.0 / n_searchable, rel=0.05)

    def test_the_closest_eligible_candidate_is_selected(self):
        """Suppressing a rank must never suppress the ranking itself."""
        from timeseries.matrix_profile import distance_profile

        feats = np.random.default_rng(5).normal(size=(300, 1))
        L = 20
        q = M.Query.from_span(feats, 0, L, per_window=True)
        # Shape distance only -- see the note in the §BC test above.
        res = M.find_matches(feats, q, k=4, amplitude_weight=0.0)
        assert len(res.matches) == 4

        starts = np.arange(len(feats) - L + 1, dtype=np.int64)
        eligible = ~M.exclusion_mask(starts, q)
        dists = distance_profile(feats[:L], feats, query_idx=0)
        best = int(np.argmin(np.where(eligible, dists, np.inf)))

        assert int(starts[best]) in {m.start for m in res.matches}
        assert min(m.distance for m in res.matches) == pytest.approx(
            dists[best], rel=1e-6
        )


# --------------------------------------------------------------------------- #
# §BE -- defaults must be able to clear the evidence gate
# --------------------------------------------------------------------------- #
class TestDefaultK:
    """§BE: k=10 against MIN_MATCHES=30 meant every default run reported
    'insufficient evidence' with lift/p-value NaN -- suppressed out of the box."""

    def test_default_k_clears_min_matches(self):
        assert DEFAULT_K >= MIN_MATCHES

    def test_default_run_can_reach_a_verdict(self, archive_bars):
        p = Pipeline.from_frame(archive_bars, length=60)
        assert p.ready, p.warnings
        out = p.run()
        assert out["ok"]
        assert out["result"].n_matches >= MIN_MATCHES


# --------------------------------------------------------------------------- #
# §BF -- session closure is not a data hole
# --------------------------------------------------------------------------- #
class TestQualityGate:
    """§BF: `gaps_over_180s` reported 20 on a clean archive -- every one was an
    overnight closure, which reads as corruption to an operator."""

    def test_clean_archive_has_no_holes_and_counts_boundaries(self, archive_bars):
        p = Pipeline.from_frame(archive_bars, length=60)
        assert p.ready, p.warnings
        q = p.quality()
        assert q["gaps_over_180s"] == 0
        assert q["session_boundaries"] == q["sessions"] - 1

    def test_real_hole_is_still_detected(self):
        """A regression guard in the other direction: tightening the filter must not
        hide genuine corruption."""
        bars = synthetic_bars()
        holed = pd.concat([bars.iloc[:900], bars.iloc[905:]], ignore_index=True)
        assert quality_report(holed)["gaps_over_180s"] == 1

    def test_sessions_are_eastern_days_not_utc_days(self):
        """UTC bucketing splits a session at 20:00 ET, misclassifying every
        overnight closure as an intra-session hole.  14:00Z/19:00Z are 09:00/14:00
        ET -- the same Eastern trading day, 5 hours apart."""
        bars = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-03-02 14:00Z", "2026-03-02 14:01Z", "2026-03-02 19:00Z"],
                    utc=True,
                ),
                "close": [700.0, 700.1, 700.2],
            }
        )
        r = quality_report(bars)
        assert r["sessions"] == 1, "all three bars are the same Eastern session"
        assert r["gaps_over_180s"] == 1, "the 4h59m jump is an intra-session hole"


def synthetic_bars(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    """Contiguous 1-minute bars in market hours (no gaps at all)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-02 14:30", periods=n, freq="1min", tz="UTC")
    close = 700 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
        }
    )


# --------------------------------------------------------------------------- #
# §E -- the placebo test
# --------------------------------------------------------------------------- #
class TestPlacebo:
    """§E: run the real pipeline on a random walk; it must report nothing.

    Note the Bonferroni correction is not optional.  The first implementation took
    min(p) across horizons and failed 2 of 8 seeds -- but across 30 walks the
    per-horizon p-values turned out uniform-to-conservative (0.8% below 0.05 vs 5%
    expected).  The test had the multiple-comparisons bug it was written to catch.
    """

    def test_random_walk_has_no_autocorrelation_to_find(self):
        r = np.diff(np.log(random_walk_bars(5000, seed=7)))
        assert abs(pd.Series(r).autocorr(1)) < 0.05

    # RESOLVED -- see PLAN.md §BW and the note in test_matching.py's copy of this
    # test.  The cause was variance inflation in the selected matched mean, corrected
    # by ``forecast.selection_aware_test``.  The ``strict=True`` xfail marker is
    # removed here, in the same commit as the fix, as this comment always required.
    def test_pipeline_finds_nothing_in_noise(self):
        res = run_placebo(seeds=3, n_bars=3000, length=40, n_boot=200, n_perm=200)
        assert res
        assert all(r.passed for r in res), [
            (r.seed, r.worst_p, r.worst_p_adj) for r in res if not r.passed
        ]

    def test_bonferroni_correction_is_applied(self):
        """A run must be judged on the adjusted p, not the raw minimum."""
        res = run_placebo(seeds=2, n_bars=3000, length=40, n_boot=200, n_perm=200)
        for r in res:
            assert r.worst_p_adj >= r.worst_p
            assert r.alpha_adj < 0.05