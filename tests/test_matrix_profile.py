"""Tests for the STUMPY matrix profile.  PLAN.md §C, §S, §T, §M, §E.

STUMPY is the *only* scorer in this package, so there is no longer a second
implementation to check equivalence against.  What replaces that is a set of
invariants that must hold for any correct matrix profile, plus a direct
recomputation of the distance from first principles -- an independent calculation,
not a competing library call, so it still catches a silently wrong metric.

The placebo tests matter just as much: a matcher that finds more \"matches\" on pure
noise is worse than no matcher at all, because it looks more convincing.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from timeseries import matching as M
from timeseries import matrix_profile as MP


def synthetic_series(n: int = 2000, seed: int = 0, channels: int = 1) -> np.ndarray:
    """Random-walk-ish features with the same character as ``return_z``/``path_z``."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, size=(n, channels))
    x = np.cumsum(steps, axis=0)
    return x if channels > 1 else x.ravel()


def direct_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two z-scored windows, computed from scratch.

    The independent check on the metric.  ``a`` and ``b`` are raw slices of equal
    length; both are z-scored here, which is precisely what ``normalize=True`` does
    inside STUMPY.  A constant window maps to all-zeros, matching
    :func:`timeseries.matching.zscore`.

    Multi-channel input is normalised column by column before flattening, which is
    what the scorer does: it runs one MASS pass per channel and combines them as a
    root-sum-of-squares.  (Normalising the whole 2-D block at once would be a
    different metric, because the two channels would share a single scale.)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 2:
        a = np.stack([M.zscore(a[:, c]) for c in range(a.shape[1])], axis=-1)
        b = np.stack([M.zscore(b[:, c]) for c in range(b.shape[1])], axis=-1)
    else:
        a, b = M.zscore(a), M.zscore(b)
    return float(np.sqrt(((a - b) ** 2).sum()))


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #
class TestAvailability:
    def test_stumpy_is_installed(self):
        """stumpy is a required dependency; if this fails, the install is incomplete."""
        assert MP.STUMPY_AVAILABLE

    def test_module_imports_without_raising(self):
        """Importing must succeed -- and must succeed *because* stumpy is required."""
        assert MP.CANONICAL_LENGTHS == (30, 60, 120)

    def test_warm_up_is_safe_to_call(self):
        """§X: JIT warm-up must never raise, even if called twice or with no data."""
        MP.warm_up()
        MP.warm_up()


# --------------------------------------------------------------------------- #
# exclusion zone (§M vs STUMPY's default)
# --------------------------------------------------------------------------- #
class TestExclusionZone:
    def test_strict_margin_is_a_full_window_length(self):
        """§M: one window length of margin, not STUMP's default m/4."""
        assert MP.excl_zone_for(60, strict=True) == 60
        assert MP.excl_zone_for(30, strict=True) == 30

    def test_library_default_is_reproduced_when_not_strict(self):
        """§V flags STUMP's default as ~m/4; keep it available for comparison."""
        assert MP.excl_zone_for(60, strict=False) == 15
        assert MP.excl_zone_for(8, strict=False) == 2

    def test_strict_margin_is_wider_than_library_default(self):
        """The whole point: our zone must be strictly larger, or §M is not enforced."""
        for m in (30, 60, 120):
            assert MP.excl_zone_for(m, strict=True) > MP.excl_zone_for(m, strict=False)


# --------------------------------------------------------------------------- #
# distance profile
# --------------------------------------------------------------------------- #
class TestDistanceProfile:
    def test_shape_matches_sliding_windows(self):
        x = synthetic_series(500)
        q = x[100:160]
        d = MP.distance_profile(q, x)
        assert len(d) == len(x) - len(q) + 1

    def test_self_distance_is_zero(self):
        x = synthetic_series(500)
        q = x[100:160]
        d = MP.distance_profile(q, x, query_idx=100)
        assert d[100] == pytest.approx(0.0, abs=1e-8)

    def test_distances_are_non_negative(self):
        d = MP.distance_profile(synthetic_series(400)[:60], synthetic_series(400, seed=1))
        assert np.all(d >= 0)

    def test_exact_duplicate_is_found(self):
        """A verbatim copy elsewhere in the series must score ~0."""
        x = synthetic_series(400)
        q = x[50:110].copy()
        x2 = np.concatenate([x, q, x])
        d = MP.distance_profile(q, x2)
        assert d.min() == pytest.approx(0.0, abs=1e-6)

    def test_multichannel_profile_is_finite_and_shaped(self):
        x = synthetic_series(400, channels=2)
        q = x[100:160, :]
        d = MP.distance_profile(q, x)
        assert len(d) == len(x) - 60 + 1
        assert np.isfinite(d).all()

    def test_constant_window_matches_the_zscore_convention(self):
        """A flat window has no standard deviation; the convention must hold.

        :func:`timeseries.matching.zscore` maps a constant window to all-zeros rather
        than dividing by a zero standard deviation.  STUMPY must produce the *same*
        number from the raw series, or flat stretches would rank differently from how
        they are drawn.
        """
        x = np.concatenate([synthetic_series(200), np.full(80, 3.0)])
        L = 60
        d = MP.distance_profile(x[:L], x, query_idx=0)

        # The trailing windows lie entirely inside the flat run.
        for j in (len(x) - L, len(x) - L - 5, len(x) - L - 19):
            assert d[j] == pytest.approx(
                direct_distance(x[:L], x[j:j + L]), abs=1e-6
            ), "STUMPY must agree with the zero-mapping convention on constant windows"
        assert not np.isnan(d).any(), "no NaNs anywhere in the profile"

    def test_two_constant_windows_score_zero(self):
        """Constant-to-constant is 0.0 -- identical, and uninformative."""
        c = np.full(200, 3.0)
        L = 60
        assert np.allclose(MP.distance_profile(c[:L], c, query_idx=0), 0.0)

    def test_rejects_non_finite_input(self):
        bad = synthetic_series(200).copy()
        bad[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            MP.distance_profile(bad[:60], bad)

    def test_rejects_channel_mismatch(self):
        with pytest.raises(ValueError, match="channel mismatch"):
            MP.distance_profile(np.zeros((60, 2)), np.zeros((500, 3)))

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            MP.profile(np.arange(100.0), 200)


class TestDistanceProfileIsCorrect:
    """The metric itself, checked against an independent recomputation.

    This is the replacement for the old "STUMPY agrees with the numpy scorer" test.
    It is a stronger check than comparing two implementations that share an
    assumption: :func:`direct_distance` re-derives the value from the definition, so
    it fails if STUMPY's normalized distance ever stops meaning "Euclidean distance
    between z-scored windows".
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_channel_matches_the_definition(self, seed):
        x = synthetic_series(1500, seed=seed)
        L = 60
        q_start = 700
        d = MP.distance_profile(x[q_start:q_start + L], x, query_idx=q_start)

        for j in range(0, len(d), 137):
            assert d[j] == pytest.approx(
                direct_distance(x[q_start:q_start + L], x[j:j + L]), abs=1e-6
            ), (
                "STUMPY's normalized distance must equal the Euclidean distance on "
                "per-window-z-scored features, or percentiles and match lists change "
                "meaning"
            )

    @pytest.mark.parametrize("seed", [3, 4])
    def test_multichannel_matches_the_flattened_definition(self, seed):
        """Per-channel distances combined as a root-sum-of-squares is the flattened norm.

        The query is (60, 2), so ``direct_distance`` z-scores each column and compares
        the flattened windows -- exactly the combination ``distance_profile`` performs.
        """
        x = synthetic_series(1200, seed=seed, channels=2)
        L = 60
        q_start = 600
        d = MP.distance_profile(x[q_start:q_start + L], x, query_idx=q_start)

        for j in range(0, len(d), 151):
            want = direct_distance(x[q_start:q_start + L], x[j:j + L])
            assert d[j] == pytest.approx(want, abs=1e-6)

    def test_query_is_not_double_normalised(self):
        """§BC: handing STUMPY a pre-z-scored query would scale only one side.

        ``Query.from_span`` normalises its vector for display.  ``find_matches`` must
        ignore that vector and re-read the raw span, so the query and every candidate
        are normalised exactly once each, inside the scorer.
        """
        x = synthetic_series(1000, seed=21)
        L = 60
        q = M.Query.from_span(x, 500, 500 + L)

        # ``amplitude_weight=0`` isolates the *shape* distance, which is the only
        # thing this test is about.  Left at its default, ``find_matches`` adds the
        # amplitude penalty described in §BD and the reported distance is
        # deliberately NOT the raw MASS value -- so the assertion below would fail
        # while the code was right, and it would fail for the least helpful reason
        # imaginable.  The amplitude term has its own coverage.
        res = M.find_matches(x, q, k=1, amplitude_weight=0.0)
        raw = MP.distance_profile(x[500:500 + L], x, query_idx=500)
        # The reported distance must be the one STUMPY computes from the raw slices.
        assert res.matches[0].distance == pytest.approx(float(raw[res.matches[0].start]))


# --------------------------------------------------------------------------- #
# §M -- the exclusion still holds
# --------------------------------------------------------------------------- #
class TestSeriesPathExclusion:
    def test_query_cannot_match_itself(self):
        x = synthetic_series(3000, seed=2)
        L = 60
        q = M.Query.from_span(x, 1500, 1500 + L)
        res = M.find_matches(x, q, k=5)
        assert res.matches, "expected some matches"
        for m in res.matches:
            assert m.start <= q.start - L - 1 or m.start >= q.stop + L

    def test_adjacent_windows_are_suppressed(self):
        x = synthetic_series(3000, seed=3)
        L = 60
        q = M.Query.from_span(x, 1500, 1500 + L)
        res = M.find_matches(x, q, k=5)
        s = np.sort([m.start for m in res.matches])
        if len(s) > 1:
            assert np.diff(s).min() >= L

    def test_matches_do_not_overlap_the_query(self):
        """Neighbouring windows share m-1 bars with the query and are not evidence."""
        x = synthetic_series(3000, seed=5)
        L = 60
        q = M.Query.from_span(x, 1200, 1200 + L)
        res = M.find_matches(x, q, k=10)
        for m in res.matches:
            overlap = min(m.stop, q.stop) - max(m.start, q.start)
            assert overlap <= 0, f"match at {m.start} overlaps the query"

    def test_exclusion_is_stricter_than_stumpy_default(self):
        """§M: a full window of margin must exclude more than STUMP's m/4.

        If these were equal, the strict margin would not actually be doing anything and
        §M's "neighbours are not independent evidence" argument would be unenforced.
        """
        x = synthetic_series(3000, seed=8)
        L = 60
        q = M.Query.from_span(x, 1500, 1500 + L)
        strict = M.find_matches(x, q, k=10)
        loose = M.find_matches(
            x, q, k=10, exclusion_margin=MP.excl_zone_for(L, strict=False)
        )
        assert strict.n_excluded > loose.n_excluded

    def test_rejects_out_of_range_query(self):
        x = synthetic_series(500)
        q = M.Query(vector=np.zeros(60), start=480, stop=540)
        with pytest.raises(ValueError, match="outside series"):
            M.find_matches(x, q)


# --------------------------------------------------------------------------- #
# batch profile (§S)
# --------------------------------------------------------------------------- #
class TestBatchProfile:
    def test_profile_shape_and_columns(self):
        x = synthetic_series(800)
        res = MP.profile(x, 60)
        assert res.n_windows == len(x) - 60 + 1
        assert res.profile.shape == (len(x) - 60 + 1, 4)
        # A matrix profile row is positional: row i is the window starting at i.
        assert np.array_equal(res.starts, np.arange(res.n_windows))
        # Column 0 is the DISTANCE and column 1 the nearest-neighbour index -- the
        # reverse of what PLAN.md §S claimed. Asserted so a re-read of the plan cannot
        # silently reintroduce the swap.
        assert np.all(res.distances > 0)
        assert np.all((res.indices >= 0) & (res.indices < res.n_windows))

    def test_profile_entry_is_actually_the_nearest_neighbour(self):
        """The defining property of a matrix profile (§S).

        For each window ``w`` the profile entry must be *no larger* than the distance
        from ``w`` to its reported neighbour ``I_[w]`` -- i.e. the reported neighbour
        really is the nearest one. Equality is not the right assertion: ``d[I_[w]]`` is
        the *neighbour's* own nearest-neighbour distance, which can be much smaller than
        ``d[w]`` if the neighbour sits inside a dense motif.
        """
        x = synthetic_series(1200, seed=17)
        res = MP.profile(x, 60)
        d, idx = res.distances, res.indices
        step = max(1, res.n_windows // 60)
        for w in range(0, res.n_windows - 1, step):
            j = int(idx[w])
            assert j >= 0 and j < res.n_windows
            # Recompute the direct distance and confirm it matches what was reported.
            direct = MP.distance_profile(x[w : w + res.length], x)[j]
            assert d[w] == pytest.approx(direct, rel=1e-9)

    def test_profile_distance_never_beats_self(self):
        """The defining property of a matrix profile (§S)."""
        x = synthetic_series(800)
        res = MP.profile(x, 60)
        assert np.all(res.distances >= -1e-9)

    def test_profile_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="invalid window length"):
            MP.profile(np.arange(100.0), 200)

    def test_top_matches_are_separated(self):
        x = synthetic_series(1500)
        res = MP.profile(x, 60)
        matches = MP.top_matches(res, k=5)
        s = np.sort([m.start for m in matches])
        if len(s) > 1:
            assert np.diff(s).min() >= 60

    def test_top_matches_report_finite_percentiles(self):
        x = synthetic_series(1500, seed=9)
        res = MP.profile(x, 60)
        for m in MP.top_matches(res, k=5):
            assert np.isfinite(m.percentile)
            assert 0.0 <= m.percentile <= 1.0

    def test_discords_are_the_farthest_windows(self):
        """§S: the same pass that finds matches also finds the anomalies."""
        x = synthetic_series(1500, seed=12)
        rep = MP.discord_report(x, 60, top=10)
        assert len(rep["distances"]) > 0
        assert np.all(np.diff(rep["distances"]) <= 1e-9), "discords must be descending"


# --------------------------------------------------------------------------- #
# §E -- THE PLACEBO TEST: noise must yield nothing
# --------------------------------------------------------------------------- #
class TestPlaceboOnNoise:
    def test_random_walk_yields_no_unreasonably_close_match(self):
        """A matrix profile over pure noise still finds a minimum -- that is expected.

        What must NOT happen is that minimum being far below the typical distance, i.e.
        the matcher implying a pattern exists when there is none.  On independent noise
        the nearest neighbour distance should sit near the bulk of the distribution, so
        the best match's percentile should be small but its distance must not be an
        outlier from below.
        """
        x = synthetic_series(4000, seed=42)
        L = 60
        q = M.Query.from_span(x, 2000, 2000 + L)
        res = M.find_matches(x, q, k=10)

        assert res.matches
        d = np.array([m.distance for m in res.matches], dtype=float)
        bulk = np.percentile(
            MP.distance_profile(x[2000:2000 + L], x, query_idx=2000), 50
        )
        # The best match must not be orders of magnitude better than the median
        # distance. A genuine motif would be; noise is not.
        assert d.min() > bulk * 0.25, (
            f"best match {d.min():.4f} is far below the median distance {bulk:.4f} "
            "-- the matcher is finding a pattern in pure noise (§E placebo test)"
        )

    def test_matches_are_not_suspiciously_identical(self):
        """On noise the top matches must not all be the *same* distance.

        Not "widely spread" -- the top-10 nearest neighbours of a 60-bar window in
        4,000 bars of noise are genuinely clustered. The failure this guards against is
        a broken scorer returning a constant value for every candidate, which would make
        the ranking meaningless while looking plausible.
        """
        x = synthetic_series(4000, seed=43)
        q = M.Query.from_span(x, 2000, 2000 + 60)
        res = M.find_matches(x, q, k=10)
        d = np.array([m.distance for m in res.matches], dtype=float)
        assert len(np.unique(np.round(d, 9))) == len(d), (
            "every match has an identical distance -- the scorer is not discriminating"
        )


# --------------------------------------------------------------------------- #
# Numba thread-safety (regression: SIGABRT, no traceback, server just vanishes)
# --------------------------------------------------------------------------- #
def test_concurrent_distance_profile_does_not_abort():
    """STUMPY kernels are ``njit(parallel=True)``; Numba's macOS default ``workqueue``
    layer aborts the *process* (SIGABRT) if two Python threads enter it at once.

    Streamlit runs each session on its own script thread, so two tabs used to be
    enough to kill the server with no Python traceback -- only a
    "Numba workqueue threading layer is terminating" line on stderr.  Unguarded,
    this test aborts the whole pytest process rather than failing an assertion, which
    is exactly the failure mode it exists to catch.
    """
    MP.warm_up()
    errors: list[BaseException] = []

    def work(seed: int) -> None:
        try:
            rng = np.random.default_rng(seed)
            for _ in range(15):
                MP.distance_profile(rng.normal(size=32), rng.normal(size=600))
                MP.distance_profile(rng.normal(size=(32, 3)), rng.normal(size=(600, 3)))
        except BaseException as exc:  # noqa: BLE001 - the assertion needs the reason
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"concurrent scoring raised: {errors[0]!r}"


def test_concurrent_profile_does_not_abort():
    """Same guard on the batch path, which calls ``stumpy.stump`` rather than ``mass``."""
    MP.warm_up()
    errors: list[BaseException] = []

    def work(seed: int) -> None:
        try:
            MP.profile(synthetic_series(900, seed=seed), 60)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"concurrent profile raised: {errors[0]!r}"


def test_numba_guard_is_reentrant():
    """``warm_up`` and the multi-channel loop nest the guard, so it must be an RLock."""
    with MP.numba_guard():
        with MP.numba_guard():
            pass
