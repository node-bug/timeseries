"""Tests for how a bar's stamp is printed, per resolution.

The request this file exists for: *all charts in the daily frequency are showing time
in the chart x axis; date should be sufficient.*  That is one sentence with three
separate obligations, and each is a way to half-fix it:

* **The charts themselves.**  ``build_price_figure`` and
  ``build_forecast_path_figure`` label their x axes by hand, and both did it the same
  way -- ``str(ts)[:16].replace("T", " ")`` -- which puts ``09:30`` on a daily axis.
* **Everything else that shows a stamp.**  Captions, the match table, the Panel tab's
  metrics and the search spinners all had their own copy of that expression.  Fixing
  the axes alone would leave ``start (UTC)`` above a bare date and a summary line
  ending ``(UTC)`` describing an Eastern one, so the helpers are exercised at each of
  those sites rather than only where the complaint came from.
* **The zone claim.**  A daily label is an *Eastern* date (see
  :func:`timeseries.store.stamp_label`), so the ``(UTC)`` suffix is no longer true
  there.  That is the part a change like this usually leaves behind, because the
  suffix is a separate literal from the format.

Every helper is ``exec``d out of ``app.py`` by :func:`apphelpers.load_app_functions`,
so these run the app's real code.  The chart tests assert on the *built figure*
rather than on the formatting helper, because a helper that is correct while a chart
stops calling it has changed nothing a reader can see.
"""

from __future__ import annotations

from contextlib import contextmanager

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pytest

from timeseries.pipeline import Pipeline
from timeseries.store import stamp_label as store_stamp_label

from apphelpers import load_app_functions
from daily_bars import daily_bars
from session_bars import session_bars

#: A weekday well in the past.  Matches ``test_daily_archive.PAST`` deliberately: the
#: stamp is a 09:30 *Eastern* open, which is 13:30 UTC under EDT -- so a helper that
#: printed the UTC calendar date would look right for the wrong reason in summer and
#: be off by an hour, never a day, which is the easy mistake to make here.
DAY = "2026-09-03"
OPEN_ET = pd.Timestamp("2026-09-03 13:30:00", tz="UTC")   # 09:30 EDT
OPEN_EST = pd.Timestamp("2026-01-05 14:30:00", tz="UTC")  # 09:30 EST

_NS = load_app_functions({
    "stamp_label", "stamp_span", "stamp_zone", "stamp_zone_name", "stamp_column",
    "build_price_figure", "build_forecast_path_figure",
    "_closest_match_path", "_closest_distance_label",
}, namespace={"go": go})

# **One** ``load_app_functions`` call for the whole module, deliberately.  Each call
# builds its own namespace, and the resolution helpers read a module-level
# ``ACTIVE_TIMEFRAME`` *list* that ``timeframe_scope`` mutates -- so a figure loaded
# through a second namespace resolves its own copy of that list and would keep
# reporting 1-minute while the test believed it had switched to daily.  That is not a
# hypothetical: it is how this file's forecast test first reported "+0 min" on a daily
# pipeline.  One namespace means one resolution for everything in it.
stamp_label = _NS["stamp_label"]
stamp_span = _NS["stamp_span"]
stamp_zone = _NS["stamp_zone"]
stamp_zone_name = _NS["stamp_zone_name"]
stamp_column = _NS["stamp_column"]
build_price_figure = _NS["build_price_figure"]
build_forecast_path_figure = _NS["build_forecast_path_figure"]


@contextmanager
def resolution(key):
    """Run a block with ``ACTIVE_TIMEFRAME`` set to ``key``, then restore it.

    The app holds its resolution in one module-level list the gate mutates, so the only
    faithful way to render something at the other resolution is to set that list --
    exactly what :func:`app.timeframe_scope` does.  Restored on the way out including
    on failure: these tests share one list, and a leaked ``"1d"`` would silently
    re-label every later assertion in the file.
    """
    previous = _NS["ACTIVE_TIMEFRAME"][0]
    _NS["ACTIVE_TIMEFRAME"][0] = key
    try:
        yield
    finally:
        _NS["ACTIVE_TIMEFRAME"][0] = previous


def _labels(fig):
    return [str(t) for t in fig.layout.xaxis.ticktext]


# --------------------------------------------------------------------------- #
# The formatter
# --------------------------------------------------------------------------- #
class TestStampFormat:
    """A daily label is a date, and an intraday one is an instant."""

    def test_daily_prints_no_time_of_day(self):
        with resolution("1d"):
            assert stamp_label(OPEN_ET) == DAY

    def test_intraday_keeps_the_clock(self):
        """Unchanged, and asserted because a daily feature can pass by breaking this."""
        with resolution("1m"):
            assert stamp_label(OPEN_ET) == "2026-09-03 13:30"

    def test_no_daily_label_carries_a_colon(self):
        """The whole defect, as one assertion.

        Not "no label equals another": a daily chart's defect is precisely that every
        tick repeats *the same* time, so uniqueness would not catch it.  A colon cannot
        appear in an ISO date, which is the property that must hold.
        """
        with resolution("1d"):
            assert all(":" not in stamp_label(ts) for ts in (OPEN_ET, OPEN_EST))

    def test_the_date_is_the_eastern_one(self):
        """Not the UTC calendar date -- the two differ for any bar before 20:00 ET.

        This is the reason :func:`timeseries.store.stamp_label` buckets through
        ``session_et`` rather than slicing ``str(ts)[:10]``.  Yahoo's daily stamp is a
        session's open, so a midnight-UTC bar would file itself under the *previous*
        Eastern day, and the chart would then disagree with the ``session`` column the
        match table reports for the very same bar.
        """
        midnight_utc = pd.Timestamp("2026-09-03 00:00:00", tz="UTC")  # 20:00 ET, the day before
        with resolution("1d"):
            assert stamp_label(midnight_utc) == "2026-09-02"

    def test_it_survives_both_daylight_saving_offsets(self):
        """09:30 EDT is 13:30 UTC; 09:30 EST is 14:30 UTC.  Both land on their own day."""
        with resolution("1d"):
            assert stamp_label(OPEN_ET) == DAY
            assert stamp_label(OPEN_EST) == "2026-01-05"

    def test_a_missing_stamp_is_an_em_dash_not_a_crash(self):
        """The Panel tab has a live ``pd.isna`` guard on this path; keep it total."""
        for missing in (None, pd.NaT):
            with resolution("1d"):
                assert stamp_label(missing) == "—"
            with resolution("1m"):
                assert stamp_label(missing) == "—"

    def test_an_explicit_key_overrides_the_active_resolution(self):
        """For a figure built under one tab's resolution and drawn under another's.

        The Forecast tab may hold a different resolution from the Price tab, so a chart
        has to be able to label bars by the resolution of the *bars it was handed*.
        Asserted through the keyword because this is the only way the two can differ.
        """
        assert stamp_label(OPEN_ET, key="1d") == DAY
        assert stamp_label(OPEN_ET, key="1m") == "2026-09-03 13:30"


class TestZoneClaim:
    """``(UTC)`` is a claim about the stamp, so it goes when the stamp changes."""

    def test_the_suffix_is_present_intraday_and_absent_daily(self):
        with resolution("1m"):
            assert stamp_zone() == " (UTC)"
        with resolution("1d"):
            assert stamp_zone() == ""

    def test_the_zone_name_follows_the_same_rule(self):
        with resolution("1m"):
            assert stamp_zone_name() == "UTC"
        with resolution("1d"):
            assert stamp_zone_name() == "Eastern time"

    def test_a_column_header_does_not_promise_utc_over_a_bare_date(self):
        """``start (UTC)`` above ``2026-09-03`` is the §BZ failure at header size.

        Asserted as a *pair*, because either half alone is satisfiable by accident: a
        header that lost its suffix while the cells kept a UTC clock time, or the
        reverse.  The claim has to be dropped from both together.
        """
        with resolution("1m"):
            assert stamp_column("start") == "start (UTC)"
        with resolution("1d"):
            assert stamp_column("start") == "start"


class TestStampSpan:
    def test_it_is_the_arrow_pair_the_captions_open_with(self):
        with resolution("1d"):
            assert stamp_span(OPEN_ET, OPEN_EST) == "2026-09-03 → 2026-01-05"
        with resolution("1m"):
            assert stamp_span(OPEN_ET, OPEN_EST) == \
                "2026-09-03 13:30 → 2026-01-05 14:30"


class TestOneDefinition:
    """The app and the package must not be able to print the same bar two ways."""

    def test_the_app_delegates_to_the_package_helper(self):
        """A wrapper is fine; a second format is not.

        ``fetch.py``'s summary line and a chart axis both print bar stamps, and a
        reader comparing those two is the only thing that would ever notice a
        discrepancy.  So the app's helper must call the package's rather than
        re-implement it -- this asserts the delegation, since two correct copies are
        still two copies.
        """
        from apphelpers import app_text

        body = app_text(r"def stamp_label\(.*?\n(?=\ndef )", "stamp_label")
        assert "store_stamp_label(" in body


# --------------------------------------------------------------------------- #
# The charts -- the thing the reader actually complained about
# --------------------------------------------------------------------------- #
class TestPriceChartAxis:
    """``build_price_figure`` is the chart on the Price, Matches and Backtest tabs."""

    @pytest.fixture()
    def daily_fig(self):
        pipe = Pipeline.from_frame(daily_bars(400, seed=7), length=60, timeframe="1d")
        with resolution("1d"):
            return build_price_figure(pipe, pipe.n_bars - 200, pipe.n_bars)

    @pytest.fixture()
    def intraday_fig(self):
        pipe = Pipeline.from_frame(session_bars(3000, seed=7), length=240)
        with resolution("1m"):
            return build_price_figure(pipe, pipe.n_bars - 200, pipe.n_bars)

    def test_no_daily_tick_carries_a_time(self, daily_fig):
        assert _labels(daily_fig), "the axis must still be labelled"
        assert all(":" not in t for t in _labels(daily_fig)), _labels(daily_fig)

    def test_every_daily_tick_is_a_real_date_in_the_archive(self, daily_fig):
        """Not merely colon-free -- a date the archive actually contains.

        Guards the opposite failure: a formatter that fell back to an index, or to the
        frame's first date, would also pass the colon test above.
        """
        pipe = Pipeline.from_frame(daily_bars(400, seed=7), length=60, timeframe="1d")
        sessions = {str(s)[:10] for s in pipe.bars["timestamp"]}
        for text in _labels(daily_fig):
            assert text in sessions, "%r is not a date in the archive" % text

    def test_intraday_ticks_still_carry_the_clock(self, intraday_fig):
        """The daily fix must not have cost the intraday axis its times."""
        assert all(":" in t for t in _labels(intraday_fig)), _labels(intraday_fig)

    def test_the_labels_sit_on_the_bars_they_name(self, daily_fig):
        """``tickvals`` and ``ticktext`` must stay index-aligned after the change.

        The tick list is built once and the text list mapped over it, so a refactor
        that filtered one and not the other would silently slide every label sideways
        by a tick -- plausible-looking, and wrong.
        """
        pipe = Pipeline.from_frame(daily_bars(400, seed=7), length=60, timeframe="1d")
        vals = list(daily_fig.layout.xaxis.tickvals)
        texts = _labels(daily_fig)
        assert len(vals) == len(texts)
        assert len(set(vals)) == len(vals), "ticks must stay distinct"
        for val, text in zip(vals, texts):
            assert store_stamp_label(pipe.bars["timestamp"].iloc[int(val)], "1d") == text


class TestForecastPathAxis:
    """The projection chart labels its history the same way, and its future differently.

    The projected side is the reason this is not a one-line change: those ticks are
    offsets (``+4 days``), not timestamps, because sessions are compressed and a
    projection spanning a close is drawn contiguously.  Dropping the time off the
    history must not make the history and the projection agree on what they say.
    """

    def test_history_is_dates_and_projection_is_offsets(self):
        from timeseries.forecast import forecast_paths

        pipe = Pipeline.from_frame(daily_bars(400, seed=7), length=60, timeframe="1d")
        # Both matches need a complete ``horizon``-bar tail after their windows, so
        # they are placed early enough in the archive to have one.
        path = forecast_paths(pipe.close, np.array([10, 40], dtype=np.int64),
                              window_length=60, horizon=20)
        assert path is not None, "fixture produced no forecast path"
        with resolution("1d"):
            fig = build_forecast_path_figure(pipe, path)

        labels = [str(t) for t in fig.layout.xaxis.ticktext]
        history = [t for t in labels if not t.startswith("+")]
        projected = [t for t in labels if t.startswith("+")]
        assert history and projected, labels
        assert all(":" not in t for t in history), history
        assert all(":" not in t for t in projected), projected
        # The projected side is measured in days on daily, not minutes.
        assert all(t.endswith("day") or t.endswith("days") for t in projected), projected


class TestIntradayUnchanged:
    """Every 1-minute number that predates this change, asserted against its fixture."""

    def test_the_app_helper_agrees_with_the_package_helper_intraday(self):
        with resolution("1m"):
            assert stamp_label(OPEN_ET) == store_stamp_label(OPEN_ET, "1m")

    def test_the_app_helper_agrees_with_the_package_helper_daily(self):
        with resolution("1d"):
            assert stamp_label(OPEN_ET) == store_stamp_label(OPEN_ET, "1d")


class TestFetchSummary:
    """``fetch.py``'s summary line prints bar stamps too, and its own zone claim."""

    def _result(self, tf, first, last):
        from timeseries.fetch import FetchResult

        frame = pd.DataFrame({
            "timestamp": pd.to_datetime([first, last], utc=True),
            "open": [1.0, 1.1], "high": [1.2, 1.3],
            "low": [0.9, 1.0], "close": [1.1, 1.2],
        })
        return FetchResult(symbol="AAPL", frame=frame, timeframe=tf.key)

    def test_daily_prints_dates_and_no_utc_claim(self):
        from timeseries.timeframes import get_timeframe

        line = self._result(get_timeframe("1d"), OPEN_ET, OPEN_EST).summary()
        assert "2026-09-03" in line and "2026-01-05" in line, line
        assert "(UTC)" not in line, line
        # Two stamps, so the arrow between them must survive.
        assert "→" in line, line

    def test_intraday_keeps_the_clock_and_the_zone(self):
        from timeseries.timeframes import get_timeframe

        line = self._result(get_timeframe("1m"), OPEN_ET, OPEN_EST).summary()
        assert "2026-09-03 13:30" in line, line
        assert "(UTC)" in line, line