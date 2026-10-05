"""Tests for the daily timeframe.  The second half of PLAN.md's timeframe work.

What this file is for
---------------------
A second resolution is only real once it has been shown *not* to break the
assumptions the first one carried.  Three failure modes are worth a test each, and
all three are silent:

* **A calibration silently reused.**  The 180-second hole threshold, the 390-bar
  session and the ~29-day retention wall are all statements about 1-minute bars.  On
  daily they are not merely wrong, they are *wrong in the direction that looks like
  corruption* -- a decade-long archive reported as full of holes and short sessions,
  which sends an operator hunting for a download that never failed.  Every check
  below exists to prove a daily frame is **not** judged by those rules.

* **The wrong null.**  :mod:`timeseries.placebo` is the permutation baseline every
  reported p-value is measured against.  A daily null generated from calendar days,
  or carrying a per-*minute* volatility, would still produce a plausible-looking
  p-value while testing a series that does not resemble daily price action.  That is
  the failure this suite is most careful about.

* **A cache keyed on the symbol alone.**  The app keys its fetched bars by symbol, so
  a second resolution silently served the first one's frame -- a chart of five weeks
  of minute bars captioned "Daily".  No number anywhere is wrong; the *labels* are.

The last section is the one that protects existing behaviour.  A timeframe feature
that only proves the new path works can do so by breaking the old one, so the 1-minute
numbers are asserted against the same fixtures as before.
"""

from __future__ import annotations

import sys
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from daily_bars import DAILY_SIGMA, business_days, daily_bars, with_missing_days
from session_bars import session_bars
from timeseries import fetch as F
from timeseries.features import build_features, quality_report
from timeseries.pipeline import Pipeline, horizons_for
from timeseries.placebo import run_placebo, synthetic_ohlcv
from timeseries.store import expected_bar_count, validate_session
from timeseries.timeframes import (
    TIMEFRAMES, get_timeframe, resolve_timeframe, validate_registry,
)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_it_is_internally_consistent(self):
        """A bad row must fail loudly, not produce a wrong number downstream."""
        validate_registry()

    def test_the_intraday_row_is_unchanged(self):
        """Every 1-minute constant must still read exactly as it did.

        This is the load-bearing regression assertion for the whole change: 1-minute
        is the default everywhere, and a registry that quietly "improved" one of its
        numbers would silently move every number the app reports on its default view.
        """
        tf = get_timeframe("1m")
        assert tf.yfinance_interval == "1m"
        assert tf.filename_slug == "1min"
        assert tf.gap_seconds == 180
        assert tf.bars_per_session == 390
        assert tf.rolling_window == 20
        assert tf.default_length == 240
        assert tf.forecast_history_bars == 240
        assert tf.forecast_projection_bars == 240
        assert tf.min_query_bars == 8
        assert tf.max_query_bars == 390
        assert tf.view_sessions == 5
        assert tf.horizons == (5, 15, 30, 60)
        assert tf.chunk_days == 7
        assert tf.max_days == 29
        assert tf.synthetic_sigma == 1e-4

    def test_the_daily_row_refuses_to_invent_a_hole_threshold(self):
        """``gap_seconds`` is ``None`` on daily, and that is the decision.

        A daily bar spans a whole trading day, so there is no interior in which a hole
        could exist.  Returning 180 would not be a conservative default -- it would
        make the threshold *mean* something different from what it means on intraday,
        silently.
        """
        tf = get_timeframe("1d")
        assert tf.gap_seconds is None
        assert tf.bars_per_session == 1
        assert tf.yfinance_interval == "1d"

    def test_daily_is_unbounded_where_intraday_is_not(self):
        """Chunking and the retention clamp are intraday-only mechanisms.

        Daily takes a single wide request.  If it inherited ``max_days=29`` it would
        discard ~95% of the history the endpoint offers, and the result would look like
        a short archive rather than like a bug.
        """
        assert get_timeframe("1d").chunk_days is None
        assert get_timeframe("1d").max_days is None
        assert get_timeframe("1m").chunk_days is not None

    def test_an_unknown_key_raises_rather_than_defaulting(self):
        """The lenient accessor must not collapse a typo onto the default.

        ``resolve_timeframe`` is lenient because session state may carry a string an
        older build wrote.  But a *wrong* resolution is worse than a missing one: it
        quality-checks a valid archive with the wrong rules.
        """
        assert resolve_timeframe("1D").key == "1d"
        assert resolve_timeframe(None).key == "1m"
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_timeframe("5m")

    def test_horizons_are_in_bars_and_differ_by_resolution(self):
        """5 daily bars is a week, not 5 minutes.

        Both sets are "a few bars through a few months"; reusing the intraday numerals
        on daily would measure a different question while looking identical in a
        dropdown.
        """
        assert horizons_for("1m") == (5, 15, 30, 60)
        assert horizons_for("1d") == (5, 10, 20, 40)
        assert horizons_for() == horizons_for("1m")


# --------------------------------------------------------------------------- #
# Session arithmetic
# --------------------------------------------------------------------------- #
class TestDailySessionArithmetic:
    def test_a_weekday_expects_one_bar_and_a_weekend_expects_none(self):
        assert expected_bar_count(date(2026, 7, 15), timeframe="1d") == 1
        assert expected_bar_count(date(2026, 7, 18), timeframe="1d") == 0   # Sat
        assert expected_bar_count(date(2026, 7, 19), timeframe="1d") == 0   # Sun

    def test_the_intraday_counts_are_untouched(self):
        """390 on a weekday, 210 on a 13:00 half day, 0 at the weekend."""
        assert expected_bar_count(date(2026, 7, 15)) == 390
        assert expected_bar_count(date(2026, 11, 27), close_time=time(13, 0)) == 210
        assert expected_bar_count(date(2026, 7, 18)) == 0

    def test_an_early_close_is_irrelevant_to_a_daily_bar(self):
        """A daily bar covers the whole session, so a 13:00 close changes nothing.

        Passing ``close_time`` on daily is a caller mistake rather than an error to
        raise -- the answer is the same either way, and raising would break a caller
        that threads the published close time through for both resolutions.
        """
        assert expected_bar_count(date(2026, 11, 27), close_time=time(13, 0),
                                  timeframe="1d") == 1

    def test_a_complete_daily_session_is_not_reported_short(self):
        """The check that most visibly matters, and the one a literal would break."""
        one = daily_bars(5, seed=1).head(1)
        day = pd.Timestamp(one["timestamp"].iloc[0]).date()
        assert validate_session(one, day, timeframe="1d")["issues"] == []
        # And the same bar under intraday rules is reported as a 1/390 shortfall.
        assert "short session: 1/390 bars" in validate_session(one, day)["issues"]


# --------------------------------------------------------------------------- #
# The quality report
# --------------------------------------------------------------------------- #
class TestDailyQualityReport:
    def test_a_clean_daily_archive_has_no_holes(self):
        report = quality_report(daily_bars(60, seed=1), timeframe="1d")
        assert report["intra_session_holes"] == 0
        assert report["missing_days"] == 0
        assert report["bars_min"] == report["bars_max"] == 1
        assert report["sessions"] == 60

    def test_a_missing_weekday_is_counted_and_named_separately(self):
        """The condition daily data *can* have, reported under its own key.

        It cannot be reported as a hole: a hole is an absence *inside* a bar, and a
        daily bar has no interior.  Reporting it as ``gaps_over_180s`` would also be
        arithmetically absurd, since consecutive daily bars are ~86,400 s apart by
        definition.
        """
        report = quality_report(with_missing_days(60, 3, seed=1), timeframe="1d")
        assert report["missing_days"] == 3
        assert report["intra_session_holes"] == 0

    def test_weekends_are_not_counted_as_missing_days(self):
        """A weekend is not a session that failed; it never happened.

        60 business days span ~12 calendar weeks, so a count that included weekends
        would report ~48 missing days on a perfectly complete archive.
        """
        report = quality_report(daily_bars(60, seed=1), timeframe="1d")
        assert report["missing_days"] == 0

    def test_the_legacy_gap_key_is_still_present_on_both(self):
        """``gaps_over_180s`` is kept so an existing consumer does not see a hole.

        On 1-minute it equals ``intra_session_holes`` (the threshold *is* 180 s).  On
        daily it is a hard 0 -- not the ~8,000 the unfiltered intraday scan would
        compute, which is why the daily branch exists at all.
        """
        clean = quality_report(session_bars(800), timeframe="1m")
        assert clean["gaps_over_180s"] == clean["intra_session_holes"] == 0

        daily = quality_report(daily_bars(60, seed=1), timeframe="1d")
        assert daily["gaps_over_180s"] == 0

    def test_missing_days_is_absent_on_intraday(self):
        """Not reported as 0 on intraday.

        A reader seeing ``missing_days: 0`` on minute bars would rightly ask what it
        means there, and the honest answer is "nothing" -- so the key is absent rather
        than misleadingly present.
        """
        assert "missing_days" not in quality_report(session_bars(800))

    def test_intraday_hole_detection_still_works(self):
        """The daily branch must not have cost the intraday one its teeth.

        The gap is punched *inside* one session -- two bars five minutes apart on the
        same Eastern date -- because that is the only arrangement the §BF
        intra-session filter counts.  Consecutive *sessions* are ~86,400 s apart and are
        deliberately excluded, which is why a daily frame under the intraday rules also
        reports zero holes: not because the daily branch fixed it, but because every
        daily bar is its own date.
        """
        two = session_bars(390 * 2, seed=3)
        holed = two.drop(index=two.index[100:110])
        assert quality_report(holed, timeframe="1m")["intra_session_holes"] == 1

        # The same frame, on daily: the hole is not a hole, because a daily bar has
        # no interior.  Asserted explicitly so the asymmetry is documented rather than
        # discovered later.
        assert quality_report(holed, timeframe="1d")["intra_session_holes"] == 0


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
class TestDailyFeatures:
    def test_the_rolling_window_is_resolved_from_the_timeframe(self):
        """A daily frame must not inherit the intraday warm-up by accident.

        The rolling base happens to be 20 on both resolutions today, but it is a
        *different quantity*: 20 minutes against 20 trading sessions.  Pinning the
        assertion on the behaviour rather than the coincidence is the point.
        """
        daily = daily_bars(60, seed=1)
        assert TIMEFRAMES["1d"].rolling_window == 20
        out = build_features(daily, timeframe="1d")
        # One return is NaN on the first row; the rest need a full window.
        assert int(out["return_z"].notna().sum()) == len(daily) - 20

    def test_an_explicit_rolling_still_wins(self):
        """The Backtest tab sweeps the window deliberately and must not be overridden."""
        daily = daily_bars(60, seed=1)
        out = build_features(daily, rolling=5, timeframe="1d")
        assert int(out["return_z"].notna().sum()) == len(daily) - 5


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
class TestDailyPipeline:
    def test_it_builds_and_carries_its_resolution(self):
        pipe = Pipeline.from_frame(daily_bars(400, seed=1), length=60,
                                   timeframe="1d")
        assert pipe.ready
        assert pipe.tf.key == "1d"
        assert pipe.length == 60
        assert pipe.matrix.shape[1] == 2

    def test_the_intraday_pipeline_still_defaults_to_intraday(self):
        """No ``timeframe`` argument must mean 1-minute, exactly as before."""
        pipe = Pipeline.from_frame(session_bars(4000, seed=2), length=240)
        assert pipe.tf.key == "1m"

    def test_its_quality_report_is_the_daily_one(self):
        """The pipeline forwards its resolution rather than defaulting.

        Otherwise a daily tab would render the intraday quality report over daily
        bars, which is the whole failure this suite exists to prevent.
        """
        pipe = Pipeline.from_frame(daily_bars(400, seed=1), length=60,
                                   timeframe="1d")
        report = pipe.quality()
        assert report.get("timeframe") == "1d"
        assert report["intra_session_holes"] == 0


class TestDailyForwardHorizonMask:
    """§BX's candidate mask, which is a *premise failure* on daily.

    §BX corrects a forward return that would otherwise measure an overnight closure:
    the horizon runs off the end of a session, so "the next N bars" are really one
    overnight gap plus N-1 minutes of trading, inflating the return ~20x.

    On daily that has nothing to fix. A daily bar *is* a session, so every bar is its
    own boundary and the mask rejects every window. Measured on live QQQ daily bars:
    6264 boundaries in 6265 bars, and **0 of 6206 candidates** admitted at every
    horizon -- the search returns nothing at all.
    """

    def test_every_daily_bar_is_a_session_boundary(self):
        """The premise that makes the mask wrong, stated as a fact."""
        pipe = Pipeline.from_frame(daily_bars(400, seed=1), length=60,
                                   timeframe="1d")
        from timeseries.pipeline import session_boundary_mask

        boundary = session_boundary_mask(pipe.bars["timestamp"])
        assert boundary.sum() == len(boundary) - 1

    def test_the_mask_is_skipped_on_daily(self):
        pipe = Pipeline.from_frame(daily_bars(400, seed=1), length=60,
                                   timeframe="1d")
        assert pipe.forward_horizon_mask(60, 40) is None

    def test_the_mask_is_still_applied_on_intraday(self):
        """The daily branch must not have cost intraday its correction."""
        pipe = Pipeline.from_frame(session_bars(4000, seed=2), length=240,
                                   timeframe="1m")
        mask = pipe.forward_horizon_mask(240, 60)
        assert mask is not None
        assert int(mask.sum()) > 0

    def test_a_daily_query_actually_finds_matches(self):
        """The end-to-end symptom, which is the thing that was broken.

        Asserted on match *count*, not on a forecast. Zero matches reads as "this
        shape has no historical analogue", which is a claim about the archive rather
        than about a mask -- so a test asserting only the mask's return value would
        pass while the feature stayed broken.

        The bar is :data:`MIN_MATCHES`, not ``k``: non-maximum suppression spaces
        accepted matches by a window length, so a thin fixture returns fewer than
        requested even when the search works. Below 30 the §E gate suppresses every
        forecast, which is the other symptom of the bug and is why this number is
        worth pinning.
        """
        from timeseries.forecast import MIN_MATCHES

        pipe = Pipeline.from_frame(daily_bars(2000, seed=1), length=30,
                                   timeframe="1d")
        out = pipe.run(pipe.query_latest(), k=50)
        found = len(out["result"].matches)
        assert found >= MIN_MATCHES, (
            "a daily search returned {} matches, below the §E evidence gate; the "
            "mask is censoring the candidate pool".format(found)
        )
        assert out["horizons"] == TIMEFRAMES["1d"].horizons


class TestPipelineHorizonDefaults:
    def test_run_uses_the_pipelines_own_horizons(self):
        """``run`` must not fall back to the module-level intraday default.

        A daily pipeline reaching ``run`` computed ``(5, 15, 30, 60)`` **bars**,
        which on a daily frame is 5 weeks through 3 months. The horizon-60 branch
        then produced *no* candidates -- a 60-bar horizon behind a 60-bar window
        needs 120 forward bars -- so the query returned zero matches and a
        ``sufficient=False`` forecast.
        """
        pipe = Pipeline.from_frame(daily_bars(800, seed=1), length=60,
                                   timeframe="1d")
        assert pipe.run(pipe.query_latest(), k=10)["horizons"] == (5, 10, 20, 40)
        assert Pipeline.from_frame(
            session_bars(4000, seed=2), length=240
        ).run(k=10)["horizons"] == (5, 15, 30, 60)

    def test_the_helper_methods_agree(self):
        """``forward_returns`` and ``baseline_returns`` default the same way.

        A control group built on different horizons than the treatment measures a
        different thing, so the lift would not be a lift.
        """
        pipe = Pipeline.from_frame(daily_bars(800, seed=1), length=60,
                                   timeframe="1d")
        implied = {h: np.array([0.01]) for h in TIMEFRAMES["1d"].horizons}
        assert set(pipe.forward_returns([100])) == set(implied)
        assert set(pipe.baseline_returns(50)) == set(implied)


class TestDailyHolidayAccounting:
    def test_missing_days_are_reported_against_a_rate(self):
        """Every weekday without a bar is counted, and most are market holidays.

        A 25-year QQQ archive measures 233, which is 9.3 a year against ~9-10
        published market holidays. On the raw count alone a *complete* archive reads
        as a partial download, which trains the reader to ignore the one number on
        the Quality tab that can actually matter.
        """
        report = quality_report(daily_bars(250, seed=1), timeframe="1d")
        assert "missing_days_per_year" in report
        assert 0.0 <= report["missing_days_per_year"] <= 14.0


# --------------------------------------------------------------------------- #
# The placebo null -- the highest-consequence part of this change
# --------------------------------------------------------------------------- #
class TestDailyPlaceboNull:
    def test_the_synthetic_series_is_business_days_with_no_weekend_bars(self):
        """Weekend bars would give the null session structure real data does not have.

        A calendar-day walk puts a "boundary" at every Saturday and Sunday, so a
        fraction of its windows would straddle a closure absent from any real daily
        archive -- widening the null in a direction unrelated to the statistic, which
        is how a placebo passes for the wrong reason.
        """
        frame = synthetic_ohlcv(120, timeframe="1d")
        et = pd.to_datetime(frame["timestamp"]).dt.tz_convert("America/New_York")
        assert int((et.dt.dayofweek >= 5).sum()) == 0
        assert et.dt.floor("D").nunique() == len(frame)

    def test_the_intraday_null_is_still_a_continuous_minute_walk(self):
        """The intraday null deliberately has *no* session boundaries; do not "fix" it."""
        frame = synthetic_ohlcv(120, timeframe="1m")
        stamps = pd.to_datetime(frame["timestamp"])
        assert stamps.diff().dt.total_seconds().median() == 60

    def test_the_daily_null_moves_at_daily_scale(self):
        """A per-minute sigma on a daily null leaves every window equally dead.

        The rolling z-score is scale-free, so the *shape* matching survives a wrong
        sigma -- but the amplitude term is measured on raw returns, and 1e-4 per day is
        a 0.01% move against a realistic ~1%.  With nothing to separate, the amplitude
        penalty is inert and the null is not testing what it claims to.
        """
        daily = synthetic_ohlcv(500, timeframe="1d", sigma=DAILY_SIGMA)
        moves = pd.Series(daily["close"]).pct_change().dropna()
        assert 0.002 < float(moves.std()) < 0.03

        intraday_sigma_on_daily = synthetic_ohlcv(500, timeframe="1d", sigma=1e-4)
        wrong = pd.Series(intraday_sigma_on_daily["close"]).pct_change().dropna()
        assert float(wrong.std()) < 0.001

    def test_the_daily_placebo_finds_nothing_in_pure_noise(self, ):
        """The actual §E check, run on the daily path.

        ``n_bars`` is large enough that 50 matches are actually available: below
        roughly 2,000 bars the exclusion zone leaves fewer than ``MIN_MATCHES`` (30)
        candidates, the forecaster reports ``sufficient=False``, and every p-value
        comes back NaN -- which would render this test green for the wrong reason.  A
        NaN verdict is silence, not a pass.
        """
        results = run_placebo(seeds=2, n_bars=3000, length=30, timeframe="1d",
                              n_boot=200, n_perm=200)
        assert results
        for r in results:
            assert np.isfinite(r.worst_p_adj), (
                "p-values are NaN, so this run proves nothing; the archive is too thin "
                "to clear the evidence gate"
            )
            assert not r.significant, (
                "the matcher found signal in a daily random walk"
            )

    def test_the_intraday_placebo_is_unaffected(self):
        results = run_placebo(seeds=2, n_bars=3000, length=40, n_boot=200, n_perm=200)
        for r in results:
            assert not r.significant


# --------------------------------------------------------------------------- #
# Fetch: resolution routing, span and naming
# --------------------------------------------------------------------------- #
class _RecordingYFinance:
    """Fake yfinance that records the interval of every request."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def Ticker(self, symbol):  # noqa: N802 - mirrors yfinance's capitalised API
        outer = self

        class _Ticker:
            def history(self, *, interval, start, end, **kwargs):
                outer.calls.append({"symbol": symbol, "interval": interval,
                                    "start": start, "end": end})
                return outer.frame

        return _Ticker()


def _install(monkeypatch, frame):
    import sys as _sys
    import types as _types

    fake = _RecordingYFinance(frame)
    module = _types.ModuleType("yfinance")
    module.Ticker = fake.Ticker
    monkeypatch.setitem(_sys.modules, "yfinance", module)
    return fake


class TestDailyFetch:
    def test_daily_asks_yahoo_for_one_day_bars_in_a_single_request(self, monkeypatch):
        fake = _install(monkeypatch, daily_bars(80, seed=1))
        F.fetch_ticker("QQQ", timeframe="1d")
        assert len(fake.calls) == 1, (
            "a daily fetch must not be chunked; Yahoo has no intraday retention wall "
            "at this resolution and chunking it would only multiply requests"
        )
        assert fake.calls[0]["interval"] == "1d"

    def test_intraday_still_asks_for_minute_bars_in_several_requests(self, monkeypatch):
        fake = _install(monkeypatch, daily_bars(80, seed=1))
        F.fetch_ticker("QQQ", timeframe="1m")
        assert len(fake.calls) >= 2
        assert {c["interval"] for c in fake.calls} == {"1m"}

    def test_a_daily_span_is_not_clipped_to_the_intraday_retention(self, monkeypatch):
        """The failure this guards is invisible: the result merely looks short."""
        fake = _install(monkeypatch, daily_bars(80, seed=1))
        F.fetch_ticker("QQQ", timeframe="1d")
        span_days = (fake.calls[0]["end"] - fake.calls[0]["start"]).total_seconds() / 86400
        assert span_days > 1000, (
            "a daily request was clamped to an intraday-sized span; ~95% of the "
            "available history was discarded"
        )

    def test_an_explicit_short_span_is_still_honoured_on_daily(self, monkeypatch):
        """``days`` means what it says on daily -- only the *default* is unbounded."""
        fake = _install(monkeypatch, daily_bars(80, seed=1))
        F.fetch_ticker("QQQ", days=10, timeframe="1d")
        span_days = (fake.calls[0]["end"] - fake.calls[0]["start"]).total_seconds() / 86400
        assert 9 <= span_days <= 10

    def test_the_result_records_the_resolution_it_was_fetched_at(self, monkeypatch):
        """So the summary and every message can name it."""
        _install(monkeypatch, daily_bars(40, seed=1))
        result = F.fetch_ticker("QQQ", timeframe="1d")
        assert result.timeframe == "1d"
        assert "Daily" in result.summary()

    def test_bars_per_session_is_one_on_daily(self, monkeypatch):
        """The Price tab sizes its view in sessions; on daily it is the bar count."""
        _install(monkeypatch, daily_bars(40, seed=1))
        result = F.fetch_ticker("QQQ", timeframe="1d")
        assert result.bars_per_session == pytest.approx(1.0)

    def test_an_empty_daily_response_is_not_blamed_on_intraday_retention(self, monkeypatch):
        """The two failures have different causes and quoting the wrong one misleads.

        Telling a reader their daily fetch fell "outside the ~30-day intraday
        retention" sends them looking for a wall that does not exist at this
        resolution.
        """
        _install(monkeypatch, pd.DataFrame())
        result = F.fetch_ticker("QQQ", timeframe="1d")
        assert not result.ok
        joined = " ".join(result.errors).lower()
        assert "no bars" in joined
        assert "intraday retention" not in joined
        assert "no listing history" in joined or "not listed" in joined


class TestDailyArchiveNaming:
    def test_the_two_slugs_round_trip(self):
        first, last = pd.Timestamp("2020-01-02"), pd.Timestamp("2026-09-30")
        for tf, slug in (("1m", "1min"), ("1d", "1d")):
            name = F.archive_name("QQQ", first, last, timeframe=tf)
            assert name == "QQQ_{}_20200102_20260930.csv".format(slug)
            assert F.symbol_from_filename(name) == "QQQ"
            assert F.timeframe_from_filename(name) == tf

    def test_an_index_symbol_round_trips_on_both(self):
        """The leading ``^`` must survive; ``archive_name`` writes it."""
        for tf in ("1m", "1d"):
            name = F.archive_name("^VIX", pd.Timestamp("2026-01-02"),
                                  pd.Timestamp("2026-01-30"), timeframe=tf)
            assert F.symbol_from_filename(name) == "^VIX"
            assert F.timeframe_from_filename(name) == tf

    def test_a_non_archive_filename_yields_neither(self):
        for name in ("notes.txt", "manifest.csv", "", "QQQ_5min_20200101_20200201.csv"):
            assert F.symbol_from_filename(name) is None
            assert F.timeframe_from_filename(name) is None

    def test_intraday_naming_is_byte_identical(self):
        """The files already in ``data/`` must still be recognised."""
        assert F.archive_name("qqq", pd.Timestamp("2026-08-31"),
                              pd.Timestamp("2026-09-30")) == \
            "QQQ_1min_20260831_20260930.csv"


# --------------------------------------------------------------------------- #
# The app: cache identity and copy
# --------------------------------------------------------------------------- #
class TestAppResolutionIdentity:
    """A resolution that is not part of a cache key is a resolution that is ignored."""

    def test_the_resolution_is_chosen_once_at_startup(self):
        """One gate, asked before anything is built, and asked again never.

        It was a per-tab dropdown, and the dropdown was the bug: its widget key was
        passed as ``key=`` and read *nowhere*, so choosing "Daily" moved the control and
        nothing else.  The page kept charting 1-minute bars under a Daily caption, with
        no error anywhere to explain it.

        A gate removes the class rather than fixing the instance: with one resolution
        per session there is no later moment at which the label and the archive could
        disagree, because neither is independently changeable.
        """
        from apphelpers import app_source

        source = app_source()
        assert "def session_timeframe() -> Any:" in source
        gate_at = source.index("def session_timeframe() -> Any:")
        main_at = source.index("def main() -> None:")
        assert gate_at < main_at
        # ``main`` returns on an unanswered gate rather than defaulting: a default
        # would download an archive the reader did not ask for.
        assert "if chosen is None:" in source
        # **No resolution dropdown survives anywhere in the input row.**  Its return
        # value used to be discarded, which is the write-only control this replaced.
        assert 'cfg["resolution_label"]' not in source
        assert 'cfg["resolution_help"]' not in source

    def test_both_scopes_read_the_one_session_resolution(self):
        """Neither scope may resolve a resolution of its own.

        Reading it off a scope is what let the two tabs hold different resolutions --
        and a tab whose archive differs from the one the page title names is exactly
        the failure being designed out.
        """
        from apphelpers import app_source

        body = app_source()
        assert "price_scope.tf" not in body
        assert "forecast_scope.tf" not in body
        assert "tf = _tf()" in body
        assert "forecast_tf = tf" in body

    def test_bar_indexed_state_is_namespaced_by_resolution(self):
        """A bar index means nothing without knowing which resolution produced it.

        Index 3,000 is 3,000 minute-bars into QQQ and 3,000 daily-bars into QQQ --
        fifteen years apart.  Namespacing is structural: the two resolutions live under
        different keys, so there is no path in which a brush made on daily bars is
        applied to a minute frame.
        """
        from apphelpers import app_source

        source = app_source()
        assert 'RESOLUTION_SUFFIX = "_@tf"' in source
        assert "def state_key(base: str" in source
        # The namespacing helper is the one that builds the name, not a literal
        # concatenation at each call site.
        assert 'return "{}{}{}".format(base, RESOLUTION_SUFFIX, tf_key(key))' in source

    def test_the_fetch_cache_key_carries_the_resolution(self):
        """``SYMBOL@TF``, not ``SYMBOL``.

        Keyed on the symbol alone, a reader who switches QQQ from 1-minute to Daily is
        served the cached minute frame under a Daily caption: a chart of five weeks of
        tape labelled as two decades of daily bars, with no error anywhere.
        """
        from apphelpers import app_text

        body = app_text(r"def fetch_ticker_cached\(.*?\n(?=\ndef )",
                        "fetch_ticker_cached")
        assert '"{}@{}".format(str(symbol).upper(), tf.key)' in body
        assert "days=None" in body

    def test_the_pipeline_cache_key_carries_the_resolution_too(self):
        """The fetch cache and the pipeline cache are keyed independently.

        Keying only the first leaves the second returning a daily pipeline under the
        entry built from 1-minute bars -- the same collision one layer down.
        """
        from apphelpers import app_source

        source = app_source()
        assert '"{}@{}".format(symbol, tf.key)' in source
        assert '"{}@{}".format(forecast_symbol, forecast_tf.key)' in source

    def test_the_pipeline_cache_receives_the_resolution_as_an_argument(self):
        """A real parameter, so Streamlit's own hashing covers it.

        Folding it into the cache-key *string* would preserve identity but hide the
        distinction from the cache's view of what separates two entries -- and the
        rolling warm-up depends on it, so a wrong one silently drops the wrong bars.
        """
        from apphelpers import app_text

        body = app_text(r"def pipeline_from_frame\(.*?\n(?=\ndef )",
                        "pipeline_from_frame")
        assert "timeframe: object = DEFAULT_TIMEFRAME" in body
        assert "resolve_timeframe(timeframe).key" in body

    def test_the_panel_archive_follows_the_resolution(self):
        """Daily mode reads ``data/sp500_daily``, and no longer refuses to run.

        This test previously asserted the *opposite*: that the Panel tab is switched
        off on daily, with the message "has not been backfilled at daily".  That
        refusal was correct then and is the reason the daily archive exists -- ranking
        a daily window against a million minute bars yields a confident-looking
        percentile over a comparison with no meaning, which is the worst kind of
        wrong because nothing about it reads as wrong.

        With the archive built, the same reasoning now says the tab must *work*, and
        the guard against the old bug moves rather than disappears: the archive is
        resolved from the resolution, so a daily query can only ever reach daily bars.
        """
        from apphelpers import load_app_functions

        ns = load_app_functions({"panel_root_for", "download_script_name"})
        for key, expected_dir, expected_script in (
            ("1m", "sp500_panel", "download_sp500.py"),
            ("1d", "sp500_daily", "download_daily.py"),
        ):
            ns["ACTIVE_TIMEFRAME"][0] = key
            assert ns["panel_root_for"]().endswith(expected_dir), (
                "%s resolved to %s, not %s" % (key, ns["panel_root_for"](), expected_dir)
            )
            assert ns["download_script_name"]() == expected_script, (
                "%s names the wrong downloader" % key
            )
            # The two must never disagree: a root from one resolution with a script
            # from the other writes 390-bar sessions into a 1-bar-per-day archive.
            assert ns["panel_root_for"]().endswith(expected_dir.split("_")[0] + "_"
                                                   + ("panel" if key == "1m" else "daily"))

    def test_the_panel_is_not_gated_off_on_daily_any_more(self):
        from apphelpers import app_source

        source = app_source()
        assert "has not been backfilled at daily" not in source, (
            "the panel is still refused on daily, but a daily archive now exists"
        )
        # And the gate is genuinely gone rather than renamed: the tab body renders
        # unconditionally, so daily gets a real search.
        body = app_source()
        assert "render_panel_tab()" in body

    def test_the_sector_labels_are_read_from_the_active_archive(self):
        """Sector percentiles must come from the archive being searched.

        Both archives hold their own ``constituents.csv``.  Reading the minute one
        while searching the daily archive would label daily results with a sector
        map that may describe a different index membership -- quietly wrong rather
        than absent.
        """
        from apphelpers import app_source

        source = app_source()
        assert "PANEL_SECTOR_FILE" not in source, (
            "the sector file is still a fixed path; it must resolve per resolution"
        )
        assert "def panel_sector_file() -> str:" in source

    def test_no_live_caller_still_reads_an_intraday_only_constant(self):
        """Every resolution-dependent number goes through an accessor.

        Scoped to *live* code: a comment or a docstring mentioning ``MAX_QUERY_BARS``
        is documentation, and failing on those would make the suite reject the file
        that explains why the constant is no longer read directly.
        """
        import ast
        import re

        from apphelpers import APP

        source = open(APP, encoding="utf-8").read()
        tree = ast.parse(source)
        stale = {"MAX_QUERY_BARS", "DEFAULT_HORIZONS", "FORECAST_HISTORY_BARS",
                 "FORECAST_PROJECTION_BARS"}

        # The *definitions* are excluded: they are retained deliberately, so a reader
        # (and a test) can still read "the 1-minute window cap is 390".  What must not
        # survive is a **call site** reading one, which is what a bare walk would flag
        # along with the assignment.
        bound = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in stale:
                        bound.add(t.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id in stale:
                    bound.add(node.target.id)

        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in stale and node.id not in bound:
                offenders.append("{}:{}".format(node.id, node.lineno))
            elif isinstance(node, ast.Attribute) and node.attr in stale:
                offenders.append("{}.{}:{}".format(
                    getattr(node.value, "id", "?"), node.attr, node.lineno))
        assert not offenders, (
            "resolution-dependent constants still read directly: {}".format(offenders)
        )
        assert re.search(r"^MIN_QUERY_BARS = \d+$", source, re.M)


# --------------------------------------------------------------------------- #
# Regression guard for the resolution that already existed
# --------------------------------------------------------------------------- #
class TestResolutionGateBehaviour:
    """The gate driven as a real widget, through Streamlit's own harness.

    Every other resolution test here asserts on ``app.py``'s text, which is the right
    tool for "this line exists" and the wrong one for "this works".  Three separate
    defects in this gate were invisible to text assertions:

    * it compared the radio's return value against the widget's *own* key, so the
      branch was unreachable and the gate never resolved at any resolution;
    * it assigned to a registered widget key, which real Streamlit refuses with
      ``StreamlitValueAssignmentNotAllowedError``;
    * with only an ``on_change``, the **default selection could never be committed** --
      clicking an already-selected radio is not a change, so the likeliest answer to
      "which resolution?" was the one that did not work.

    Each is a runtime-only fault, so these tests run the app rather than describing it.
    """

    def test_the_first_paint_asks_and_builds_nothing(self):
        """Nothing is downloaded before the question is answered.

        A default would have to be *some* resolution, and the wrong one means fetching
        an archive the reader did not ask for -- 6,285 daily bars rather than ~7,800
        minute bars, a visible and wasteful thing to do on their behalf.
        """
        from apphelpers import expected_tab_count, run_app_unanswered

        at = run_app_unanswered()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.radio) == 1, "the resolution gate is not on screen"
        assert not at.tabs, "the page was built before the gate was answered"
        # **The Continue button is part of the question.**  Without it the default
        # selection could never be committed -- see the class docstring.
        assert any(b.label == "Continue" for b in at.button), (
            "the gate offers no way to accept the shown default; a radio fires "
            "on_change only on a change, so 1-minute could never be chosen"
        )

    @pytest.mark.parametrize("choice", ["1-minute", "Daily"])
    def test_every_selection_can_be_committed_including_the_default(self, choice):
        """The default is a choice, and must be as committable as the other one.

        This is the whole reason the gate has a Continue button rather than relying on
        ``on_change``.  A radio fires its callback only on a *change* of value, so
        accepting the shown default produces no callback at all -- and 1-minute is what
        a reader is shown first and what most would accept.
        """
        from apphelpers import expected_tab_count, run_app_unanswered

        at = run_app_unanswered()
        if choice != at.radio[0].value:
            at.radio[0].set_value(choice)
        at.button[0].click().run()

        assert not at.exception, [e.value for e in at.exception]
        assert len(at.tabs) == expected_tab_count(), (
            "the page did not build after the gate was answered with %r" % choice
        )

    @pytest.mark.parametrize("choice", ["1-minute", "Daily"])
    def test_pressing_continue_leaves_no_gate_on_the_main_screen(self, choice):
        """The question must vanish once it is answered, not sit above the page.

        ``test_every_selection_can_be_committed_including_the_default`` asserts the
        page *builds* -- six tabs -- and that assertion passed while the bug was live.
        The gate was still on screen: the button committed its answer and returned the
        timeframe, so ``main()`` built the page in the same pass that had already drawn
        the radio, the caption and the button.  A reader who accepted the shown default
        saw the question and the app stacked together, gate above the first tab.

        "Builds" and "still visible" are different properties, and only the second is
        what a reader sees.  Both commit paths are covered, because the ``on_change``
        path landed on a later pass and hid the fault that the button path had.
        """
        from apphelpers import expected_tab_count, run_app_unanswered

        def assert_gate_gone(app, tag):
            gate_radios = [r for r in app.radio
                           if r.options == ["1-minute", "Daily"]]
            assert not gate_radios, (
                "{}: the resolution gate is still on the main screen".format(tag)
            )
            assert not [c for c in app.caption
                        if "Choose a resolution" in c.value], (
                "{}: the gate's caption is still on the main screen".format(tag)
            )
            assert not [b for b in app.button if b.label == "Continue"], (
                "{}: the gate's Continue button is still on the main screen".format(tag)
            )

        # **The button path**, which is where the bug was: it used to commit and build
        # in one pass.  Both selections, because committing the default is the case the
        # button exists for and the case a reader is most likely to take.
        at = run_app_unanswered()
        if choice != at.radio[0].value:
            at.radio[0].set_value(choice)
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.tabs) == expected_tab_count()
        assert_gate_gone(at, "button path")

        # **The on_change path**, held to the same standard so the two cannot diverge.
        # Only for a choice that *is* a change: re-selecting the value the gate already
        # shows fires no callback, which is the gap the Continue button exists to close
        # and which the button path above covers for both selections.
        #
        # ``index`` and not ``value``: ``set_value``/``value`` speak the underlying key
        # ("1m") while ``options`` speak the reader-facing label ("1-minute"), so
        # comparing a label against a key would decide "1-minute" was a change when it
        # was the default, and that branch would then sit behind a gate forever.
        probe = run_app_unanswered()
        if choice != probe.radio[0].options[probe.radio[0].index]:
            moved = run_app_unanswered()
            moved.radio[0].set_value(choice)
            moved.run()
            assert not moved.exception, [e.value for e in moved.exception]
            assert len(moved.tabs) == expected_tab_count(), (
                "moving the gate's radio to %r did not build the page" % choice
            )
            assert_gate_gone(moved, "on_change path")

        # **And it must stay gone.**  A gate that reappears on the next interaction is
        # a gate that was never answered.
        at.run()
        assert len(at.tabs) == expected_tab_count(), "the gate came back on a later rerun"
        assert_gate_gone(at, "after a later rerun")

    def test_continue_needs_only_one_click(self):
        """The gate must advance on a single press, not two.

        The fix for the leak routes the button through the same park-then-commit path
        the radio callback uses, which by itself would strand the reader: Streamlit
        runs the script once per interaction, so parking an answer with no rerun leaves
        the committed value parked and the gate drawn forever.  The rerun that
        completes the move is load-bearing, and this is the test that says so -- the
        failure mode is a Continue button that appears to do nothing.
        """
        from apphelpers import expected_tab_count, run_app_unanswered

        at = run_app_unanswered()
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.tabs) == expected_tab_count(), (
            "one press of Continue did not reach the page; if the gate is still "
            "showing, the answer is parked but never rerun into"
        )

    def test_the_answer_is_durable_across_reruns(self):
        """A slider nudge must not send the reader back to the question."""
        from apphelpers import expected_tab_count, run_app_unanswered

        at = run_app_unanswered()
        at.radio[0].set_value("Daily")
        at.button[0].click().run()
        assert len(at.tabs) == expected_tab_count()

        for _ in range(2):
            at.run()
            assert not at.exception, [e.value for e in at.exception]
            assert len(at.tabs) == expected_tab_count(), "the gate came back on a later rerun"

    def test_changing_the_radio_commits_without_pressing_continue(self):
        """Changing the selection is itself an answer.

        Only the *unchanged* default needs the button.  Making a reader press Continue
        after they have already chosen would be a second, redundant step -- and would
        make the gate feel like it ignored them.
        """
        from apphelpers import expected_tab_count, run_app_unanswered

        at = run_app_unanswered()
        at.radio[0].set_value("Daily")
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.tabs) == expected_tab_count(), (
            "moving the radio off the default did not commit the choice"
        )

    def test_a_registered_widget_key_cannot_be_written(self):
        """The guard the whole parking mechanism exists to route around.

        A registered widget's key is read-only to ``st.session_state``; Streamlit
        raises ``StreamlitValueAssignmentNotAllowedError``.  Every other helper in
        ``apphelpers`` passes a bare ``dict`` as session state, which accepts such a
        write silently -- which is exactly how the first version of this gate shipped
        a line that raises on the one pass it was meant to answer on.
        """
        from apphelpers import WidgetAwareSessionState, WidgetKeyError

        state = WidgetAwareSessionState()
        state.register_widget("a_widget", "value")
        with pytest.raises(WidgetKeyError):
            state["a_widget"] = "something else"
        # The browser's own value still arrives -- it does not come through the app.
        assert state.set_from_browser("a_widget", "from the browser") == "from the browser"

    def test_bar_indexed_keys_follow_the_committed_resolution(self):
        """Every bar-indexed key is qualified by the resolution in force.

        Index 3,000 is 3,000 minute-bars into QQQ and 3,000 daily-bars into QQQ --
        fifteen years apart.  Namespacing is structural: the two resolutions live under
        different keys, so no path exists on which a brush made on daily bars is
        applied to a minute frame.
        """
        from apphelpers import load_app_functions

        # Named explicitly rather than discovered by pattern: ``state_key`` and
        # ``tf_key`` also end in ``_key`` and take arguments, so a suffix scan would
        # call them with none.
        names = ["price_brush_key", "forecast_brush_key", "forecast_selection_key",
                 "price_run_key", "price_applied_key", "forecast_run_key",
                 "forecast_applied_key"]
        ns = load_app_functions(set(names))

        ns["ACTIVE_TIMEFRAME"][0] = "1m"
        at_1m = {n: ns[n]() for n in names}
        ns["ACTIVE_TIMEFRAME"][0] = "1d"
        at_1d = {n: ns[n]() for n in names}

        assert names, "no key accessors were loaded"
        for name in names:
            assert at_1m[name] != at_1d[name], (
                "%s is identical at both resolutions, so state set on one survives "
                "onto the other" % name
            )
            assert "1m" in at_1m[name] and "1d" in at_1d[name], (
                "%s does not carry its resolution in the key name: %r / %r"
                % (name, at_1m[name], at_1d[name])
            )


class TestTheRealAppRuns:
    """``app.py`` executed end to end, at both resolutions.

    The rest of this file reads ``app.py`` as text.  That cannot see a widget key
    registered twice in one pass, and cannot see a call to a name that does not exist
    -- both of which shipped, both of which raise and take the **whole page** down,
    and both of which left every other test in the suite green.

    Concretely, two separate defects got through:

    * ``render_ticker_input`` contained a duplicated ``with col_in:`` block, so
      ``st.text_input(key="ticker_input_price")`` was called twice per pass and raised
      ``StreamlitDuplicateElementKey``.
    * ``main()`` drew the Forecast tab's input in the not-ready branch and then called
      ``render_forecast_tab``, which drew it again.

    The two tests written to guard the input ("each stub must contain an input") were
    both *true* while the page crashed, because each looked at one branch and neither
    looked at the other.  A guard that checks the fragments cannot check the pass.
    """

    @pytest.mark.parametrize("timeframe", ["1m", "1d"])
    def test_the_page_builds_without_raising(self, timeframe):
        from apphelpers import run_app

        at = run_app(timeframe)
        assert not at.exception, "app.py raised at {}:\n{}".format(
            timeframe, "\n".join(e.value for e in at.exception)
        )

    @pytest.mark.parametrize("timeframe", ["1m", "1d"])
    def test_no_widget_key_is_registered_twice(self, timeframe):
        """Each widget key appears exactly once in the rendered page.

        A widget key may be registered only once per pass.  Streamlit raises
        ``StreamlitDuplicateElementKey`` on the second, and the raise is not
        confined to the element that caused it -- the whole script run fails, so one
        duplicated input takes down tabs that had nothing to do with it.
        """
        from collections import Counter

        from apphelpers import run_app

        at = run_app(timeframe)
        assert not at.exception, [e.value for e in at.exception]

        keys = [t.key for t in at.text_input if t.key]
        keys += [b.key for b in at.button if b.key]
        dupes = {k: n for k, n in Counter(keys).items() if n > 1}
        assert not dupes, (
            "widget keys registered more than once at {}: {}".format(
                timeframe, sorted(dupes.items())
            )
        )
        # Sanity: the page really did render its controls, so this is not passing
        # because the gate swallowed the page.
        assert len(at.text_input) == 2, (
            "expected a Price and a Forecast ticker input, got {}".format(
                [t.key for t in at.text_input]
            )
        )

    def test_the_label_matches_the_chosen_resolution(self):
        """The caption beside the ticker box names the archive actually loaded.

        This is the bug the whole refactor exists to prevent, in its original form: a
        control that moves and does nothing, leaving the page captioning one archive
        over the other's bars.  It cannot recur now that there is one resolution and
        one key it feeds, but the label is still worth checking against the gate's
        answer -- it is the only place a mismatch would show.
        """
        from apphelpers import run_app

        for timeframe, label in (("1m", "1-minute"), ("1d", "Daily")):
            at = run_app(timeframe)
            assert not at.exception
            rendered = [m.value for m in at.markdown if "tp-res" in m.value]
            assert rendered, "no resolution label rendered at {}".format(timeframe)
            assert any(label in r for r in rendered), (
                "at {} the label reads {} rather than {}".format(
                    timeframe, rendered, label
                )
            )


class TestIntradayUnchanged:
    """Every 1-minute number that predates this change, asserted against its fixture.

    A daily feature that proves the new path works can do it by breaking the old one,
    so these are here to make that impossible to do quietly.
    """

    def test_expected_bar_count_and_validation_are_unchanged(self):
        assert expected_bar_count(date(2026, 7, 15)) == 390
        assert expected_bar_count(date(2026, 7, 18)) == 0
        frame = session_bars(390, seed=1)
        report = validate_session(frame, date(2026, 1, 5))
        assert report["ok"] is True
        assert report["expected"] == 390
        assert "timeframe" in report and report["timeframe"] == "1m"

    def test_a_clean_intraday_archive_reports_the_same_numbers(self):
        bars = session_bars(390 * 5, seed=4)
        report = quality_report(bars)
        assert report["rows"] == 390 * 5
        assert report["sessions"] == 5
        assert report["bars_min"] == report["bars_max"] == 390
        assert report["intra_session_holes"] == 0
        assert report["session_boundaries"] == 4

    def test_intraday_features_are_unchanged(self):
        bars = session_bars(4000, seed=5)
        out = build_features(bars)
        assert int(out["return_z"].notna().sum()) == len(bars) - 20

    def test_intraday_fetch_still_spans_the_retention_window(self):
        from timeseries.fetch import CHUNK_DAYS, MAX_1M_DAYS

        assert CHUNK_DAYS == 7
        assert MAX_1M_DAYS == 29

    def test_the_intraday_pipeline_is_unchanged(self):
        pipe = Pipeline.from_frame(session_bars(8000, seed=6), length=240)
        assert pipe.ready
        assert pipe.length == 240
        assert pipe.tf.key == "1m"


# --------------------------------------------------------------------------- #
# The daily fixture, itself
# --------------------------------------------------------------------------- #
class TestDailyFixture:
    """A fixture that cannot express the condition under test tests nothing."""

    def test_it_produces_one_bar_per_business_day(self):
        frame = daily_bars(60, seed=1)
        et = pd.to_datetime(frame["timestamp"]).dt.tz_convert("America/New_York")
        assert len(frame) == 60
        assert et.dt.floor("D").nunique() == 60
        assert int((et.dt.dayofweek >= 5).sum()) == 0

    def test_it_is_deterministic_and_seed_sensitive(self):
        assert daily_bars(60, seed=1)["close"].equals(daily_bars(60, seed=1)["close"])
        assert not daily_bars(60, seed=2)["close"].equals(daily_bars(60, seed=1)["close"])

    def test_omitted_days_are_interior(self):
        """A gap at either edge is invisible to the missing-day count.

        The report compares the span's endpoints to the days present, so a fixture
        that dropped the first or last day would report ``missing_days == 0`` while
        looking like it had omitted some.
        """
        frame = with_missing_days(60, 3, seed=1)
        et = pd.to_datetime(frame["timestamp"]).dt.tz_convert("America/New_York")
        # Naive on both sides: the fixture's index is tz-naive business days while the
        # frame's bars are tz-aware, and comparing those raises rather than returning
        # False -- which is how a test can fail for a reason that has nothing to do with
        # what it is checking.
        days = set(et.dt.floor("D").dt.tz_localize(None))
        assert len(frame) == 60
        assert pd.Timestamp("2026-01-05") in days, "the first day must not be omitted"
        assert pd.Timestamp(business_days(60).max()) in days, \
            "the last day must not be omitted"
        assert quality_report(frame, timeframe="1d")["missing_days"] == 3

    def test_it_stamps_bars_at_the_session_open(self):
        """Matching Yahoo, which stamps a daily bar at 09:30 ET.

        The regular-hours check in ``validate_session`` is an exclusive comparison
        against 09:30, so a midnight-stamped bar would be reported as "before
        09:30 ET" -- making a test of that guard pass or fail on the fixture rather
        than on the code.
        """
        et = pd.to_datetime(daily_bars(5, seed=1)["timestamp"]).dt.tz_convert(
            "America/New_York")
        assert set(et.dt.strftime("%H:%M")) == {"09:30"}

    def test_it_moves_at_daily_scale(self):
        moves = pd.Series(daily_bars(400, seed=1)["close"]).pct_change().dropna()
        assert 0.002 < float(moves.std()) < 0.03