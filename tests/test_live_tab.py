"""Tests for the **Live** tab's chart: the tape rolls, the projection is frozen.

The Live tab promises three things at once, and each of them was broken in a
different way:

1. **The history's start bar stays put.**  The chart is anchored at the left
   edge, so the reader watches the tape grow past a fixed origin.
2. **The projection is frozen.**  It is the forecast made when the reader
   arrived, drawn at the same place and showing the same numbers for as long
   as the tab is open.
3. **The newest bar is on the chart.**  The tape's tail is real tape, and a
   reader watching the tape roll must be able to see it.

The bugs were not independent.  The projection was recomputed on every poll
against the archive's *most recent* bars, so the query window moved with the
tape and the "frozen" projection was a *different forecast every minute*.
The history was pinned by *bar index*, which drifts as the archive's indices
shift.  And the newest bar was drawn as a synthetic point past the archive's
last index, landing outside the axis range -- off the chart entirely.

These tests assert on **timestamps** and on the figure's actual traces, because
that is the only level at which each of these claims is true or false.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

from timeseries.forecast import forecast_paths
from timeseries.pipeline import Pipeline

from apphelpers import (
    app_block,
    app_source,
    live_tape,
    load_app_functions,
    window_over_tape,
)


class _FakeSt:
    """A ``st`` stand-in whose ``session_state`` is a plain writable ``dict``.

    The Live tab's state -- the pinned moments, the frozen path -- is its
    *input* here rather than a detail of its environment.  Testing against a
    real one is what distinguishes "the pin survived the refresh" from "the
    test restated the pin and checked itself".
    """

    cache_resource = staticmethod(st.cache_resource)
    cache_data = staticmethod(st.cache_data)

    def __init__(self):
        self.session_state: dict = {}


_ST = _FakeSt()
_ns = load_app_functions(
    {
        "build_forecast_path_figure",
        "forecast_path_for",
        "live_pinned_anchor",
        "live_state_key",
        "resolve_bar_index",
        "_live_path_is_stale",
        "_closest_match_path",
        "_closest_distance_label",
        "_freeze_closest_match",
        "_frozen_match_is_usable",
        "_frozen_or_rebuilt_match",
        "_proj_ticktext",
        "chart_config",
    },
    namespace={
        "st": _ST,
        "go": go,
        "PIPE_LENGTH_FOR_HELP": [240],
        "SYMBOL_FOR_HELP": ["QQQ"],
        "ROLLING_WINDOW": 20,
        "forecast_paths": forecast_paths,
        # Module constants, supplied because ``load_app_functions`` loads
        # functions rather than the constants a function body reads.
        "LIVE_ANCHOR_KEY": "live_anchor_ts",
        "LIVE_HIST_START_KEY": "live_hist_start_ts",
        "LIVE_PATH_KEY": "live_path",
        "LIVE_MATCH_KEY": "live_match",
    },
)

live_pinned_anchor = _ns["live_pinned_anchor"]
live_state_key = _ns["live_state_key"]
resolve_bar_index = _ns["resolve_bar_index"]
build_forecast_path_figure = _ns["build_forecast_path_figure"]
forecast_path_for = _ns["forecast_path_for"]
_live_path_is_stale = _ns["_live_path_is_stale"]
_closest_match_path = _ns["_closest_match_path"]
_freeze_closest_match = _ns["_freeze_closest_match"]
_frozen_match_is_usable = _ns["_frozen_match_is_usable"]


@pytest.fixture(autouse=True)
def _fresh_state():
    """Each test starts with an empty session state, so pins cannot leak."""
    _ST.session_state = {}
    yield
    _ST.session_state = {}


_TAPE = live_tape()


def _window(start: int, stop: int) -> Pipeline:
    """A pipeline over ``_TAPE``'s bars ``[start, stop)``."""
    return window_over_tape(_TAPE, start, stop)


class TestTheLiveHistoryStartIsPinned:
    """The left edge of the Live chart does not move as the tape grows."""

    def test_the_pin_is_a_moment_not_a_bar_index(self):
        """A stored index names a different bar once the indices shift.

        The archive's bar indices are not stable across a refresh: the cleaned
        frame drops a different number of warm-up rows, and the window rolls
        forward as old bars age out.  A pin stored as an index therefore drifts
        -- and the *frozen projection* ends up attached to the wrong moment.
        """
        body = app_source()
        assert "st.session_state.live_projection_anchor" not in body, (
            "the old index-based anchor is gone: a bar index does not identify "
            "the same bar across a refresh, so the pin must be a timestamp"
        )
        assert "st.session_state.live_hist_start" not in body, (
            "the old index-based history start is gone for the same reason"
        )

    def test_the_start_bar_does_not_move_when_the_archive_grows(self):
        """The regression: the left edge holds while the tape grows rightward."""
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        # The next poll: the archive has grown, and the tab asks for its pin
        # again.  This is the step the original bug got wrong.
        anchor_ts2, start_ts2 = live_pinned_anchor(later, "QQQ", 10)
        assert anchor_ts2 == anchor_ts, "a refetch must not move the pinned anchor"
        assert start_ts2 == start_ts, "a refetch must not move the pinned start"

        # And the pinned moments resolve to the same *bars* in both frames.
        assert (first.bars["timestamp"].iloc[resolve_bar_index(first, start_ts)]
                == later.bars["timestamp"].iloc[resolve_bar_index(later, start_ts2)])

    def test_the_pinned_moment_resolves_to_the_same_bar_when_the_archive_rolls(self):
        """The archive's front moves too, and that must not move the start bar.

        This is the case an index-based pin gets wrong: the same moment sits at
        a *different index* in the rolled frame.
        """
        first = _window(0, 1170)
        rolled = _window(200, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        idx_first = resolve_bar_index(first, start_ts)
        idx_rolled = resolve_bar_index(rolled, start_ts)
        assert idx_first is not None and idx_rolled is not None
        assert idx_first != idx_rolled, (
            "the fixture must actually shift the index, or this test proves "
            "nothing about why a timestamp is used"
        )
        assert (first.bars["timestamp"].iloc[idx_first]
                == rolled.bars["timestamp"].iloc[idx_rolled]), (
            "the pinned bar must be the same bar after the archive rolls"
        )

    def test_the_history_grows_to_the_right(self):
        """The window's *end* moves forward; only the start is pinned."""
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        start_first = resolve_bar_index(first, start_ts)
        start_later = resolve_bar_index(later, start_ts)
        assert start_first == start_later, "the start bar is pinned"
        assert later.n_bars > first.n_bars, "the archive grew"

    def test_moving_the_control_re_pins(self):
        """The reader's gesture is the one thing that re-pins the chart."""
        pipe = _window(0, 1170)

        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor_ts2, start_ts2 = live_pinned_anchor(pipe, "QQQ", 20)
        assert anchor_ts2 == anchor_ts, "the anchor is the newest bar either way"
        assert start_ts2 != start_ts, (
            "moving the control must move the pinned start bar"
        )


class TestTheLiveProjectionIsFrozen:
    """The forecast is made once and does not change under the reader."""

    def test_the_projection_is_computed_once_and_reused(self):
        """A poll that brings new bars must not produce a new forecast.

        ``forecast_path_for`` with no window matches the archive's *most recent*
        bars, so recomputing it each poll re-aims the query at a moving window
        and matches a different set of historical windows each time.  The chart
        drew the path at fixed x-positions while the path's own contents
        changed underneath it.
        """
        body = app_block(
            r"def render_live_tab\(.*?\n(?=def )",
            "render_live_tab",
        )
        # The path is stored, not recomputed unconditionally.
        assert "LIVE_PATH_KEY" in body, (
            "the Live tab must store its projection so a poll can reuse it"
        )
        assert "st.session_state[path_key] = path" in body, (
            "the computed projection must be stored for the next poll"
        )
        # And the match is against the *pinned* window, not the moving tail.
        assert "window=(hist_start, projection_anchor + 1)" in body, (
            "the projection must be matched against the pinned window, or it "
            "moves with the tape and is not frozen at all"
        )

    def test_a_stored_path_is_not_stale_for_the_same_controls(self):
        """A path built for these controls is reused, not rebuilt."""
        pipe = _window(0, 1170)
        path = forecast_path_for(
            "k", pipe, length=10, horizon=20, k=1,
            amplitude_weight=0.0, window=(pipe.n_bars - 10, pipe.n_bars),
        )
        assert path is not None
        assert not _live_path_is_stale(path, 10, 20)
        assert _live_path_is_stale(path, 20, 20), (
            "a change to *Recent bars* must invalidate the frozen path"
        )
        assert _live_path_is_stale(path, 10, 30), (
            "a change to *Projection bars* must invalidate the frozen path"
        )


class TestTheClosestMatchIsFrozenToo:
    """The orange trace is a **value**, not an index into a moving archive.

    Freezing the *path* was not enough, and the gap it left is the one this
    class is about.  The path carries the median, the quartiles and the
    matched windows' **bar indices** -- but not the individual return series
    the orange trace is drawn from.  So the trace was re-derived on every poll
    by slicing the *current* archive at ``path.starts[0]``.

    A bar index is only meaningful against the archive it was computed from.
    The Live tab's archive **rolls**: old bars age out of the front of Yahoo's
    29-day window, so the same index names a *different* bar.  The forecast was
    frozen and the line drawn from it was not, which is the whole bug: the
    projection bars changed as new data was added.
    """

    def _frozen_pair(self, pipe):
        """The pair the Live tab stores: the path and the match's own series.

        The query window is the Live tab's own -- the pinned bars, ending on
        the anchor -- so the path is the one the tab would really freeze.
        """
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)
        assert anchor is not None and start is not None

        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None, "the fixture must yield a path"
        return path, _freeze_closest_match(pipe.close, path)

    def _projection(self, pipe, path, frozen, anchor, start):
        """The figure's orange trace, as the Live tab draws it."""
        fig = build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start,
            hist_stop=pipe.n_bars, frozen_match=frozen,
        )
        trace = fig.data[-1]
        return (np.asarray(trace.x, dtype=int),
                np.asarray(trace.y, dtype=float))

    def test_the_drawn_line_does_not_change_when_the_archive_rolls(self):
        """The regression: the projection bars moved when the tape rolled.

        A *grow* poll (bars appended at the tail) leaves the index space
        alone, so the re-derived series happened to survive it.  A *roll* poll
        (bars aged out of the front) shifts every index, and that is where the
        orange line changed -- which is why the bug looked intermittent.
        """
        first = _window(0, 1170)
        rolled = _window(390, 1950)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        path, frozen = self._frozen_pair(first)

        anchor = resolve_bar_index(rolled, anchor_ts)
        start = resolve_bar_index(rolled, start_ts)
        assert anchor is not None and start is not None

        _, y_first = self._projection(
            first, path, frozen,
            resolve_bar_index(first, anchor_ts),
            resolve_bar_index(first, start_ts),
        )
        _, y_rolled = self._projection(rolled, path, frozen, anchor, start)

        np.testing.assert_array_equal(
            y_rolled, y_first,
            err_msg=(
                "the drawn projection changed when the archive rolled: the "
                "frozen forecast must be drawn from the frozen series, not "
                "re-read from a moving bar index"
            ),
        )

    def test_the_drawn_line_does_not_change_when_the_archive_grows(self):
        """The same, for the ordinary case: new bars arrive at the tail."""
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        path, frozen = self._frozen_pair(first)

        _, y_first = self._projection(
            first, path, frozen,
            resolve_bar_index(first, anchor_ts),
            resolve_bar_index(first, start_ts),
        )
        _, y_later = self._projection(
            later, path, frozen,
            resolve_bar_index(later, anchor_ts),
            resolve_bar_index(later, start_ts),
        )

        np.testing.assert_array_equal(y_later, y_first)

    def test_the_frozen_series_is_what_the_old_code_would_not_have_given(self):
        """The test above could pass vacuously if the two polls agreed anyway.

        So this asserts the *counterfactual*: re-deriving the series from the
        rolled archive -- what the app used to do -- really does produce a
        different line.  Without this, a fixture that happened not to shift
        would make the regression test meaningless.
        """
        first = _window(0, 1170)
        rolled = _window(390, 1950)

        path, _ = self._frozen_pair(first)
        rederived = _closest_match_path(rolled.close, path)
        assert rederived is not None
        frozen_series = _freeze_closest_match(first.close, path)["series"]
        assert frozen_series is not None
        assert not np.allclose(rederived, frozen_series), (
            "the fixture must actually shift the matched window, or the "
            "regression test above proves nothing"
        )

    def test_the_series_is_captured_against_the_same_archive_as_the_path(self):
        """The record names the path it belongs to.

        A path and a series from *different* archives are not the same
        forecast, so the record carries the path's own length and horizon and
        is checked against them before it is drawn.
        """
        pipe = _window(0, 1170)
        path, frozen = self._frozen_pair(pipe)
        assert frozen["length"] == path.length
        assert frozen["horizon"] == path.horizon
        assert _frozen_match_is_usable(frozen, path)

    def test_a_record_from_a_different_path_is_not_drawn(self):
        """A record that describes another forecast must not be used.

        The reader changes *Recent bars* or *Projection bars*, which
        invalidates the frozen path.  The old record is then a record of a
        *different* forecast, and drawing it next to the new path would be a
        chart of nothing in particular.
        """
        pipe = _window(0, 1170)
        path, frozen = self._frozen_pair(pipe)

        other = forecast_path_for(
            "k", pipe, length=20, horizon=10, k=1,
            amplitude_weight=0.0, window=(pipe.n_bars - 30, pipe.n_bars - 10),
        )
        assert other is not None
        assert not _frozen_match_is_usable(frozen, other), (
            "a record frozen against a different path must not be usable"
        )

    def test_a_frozen_none_stays_none(self):
        """A match that could not be reconstructed is not resurrected.

        Re-deriving it on a later archive could put the line back on a bar
        that is not the one that was matched -- the same bug wearing a
        different hat.
        """
        pipe = _window(0, 1170)
        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(pipe.n_bars - 10, pipe.n_bars),
        )
        assert path is not None
        frozen = {"length": path.length, "horizon": path.horizon, "series": None}
        assert _frozen_match_is_usable(frozen, path)
        fig = build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            frozen_match=frozen,
        )
        assert len(fig.data) == 3, (
            "a frozen match that does not exist must draw no orange trace, "
            "and must not be re-derived from the archive"
        )

    def test_the_other_charts_are_unchanged(self):
        """The Forecast tab, the panel and the brushed chart re-derive.

        They draw a *live* chart over a *live* archive and want the current
        reading, so they pass no frozen record and keep the old behaviour.
        """
        pipe = _window(0, 1170)
        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(pipe.n_bars - 10, pipe.n_bars),
        )
        assert path is not None
        fig = build_forecast_path_figure(pipe, path, history_bars=10)
        assert len(fig.data) == 5, (
            "without a frozen record the closest match is still re-derived "
            "from the archive, exactly as it always was"
        )

    def test_the_live_tab_stores_the_match_alongside_the_path(self):
        """The two are written and read together.

        A path and a series from different archives are not the same
        forecast, so the tab stores one record per session and hands the
        figure the one that belongs to the path it is drawing.
        """
        body = app_block(
            r"def render_live_tab\(.*?\n(?=def )",
            "render_live_tab",
        )
        assert "LIVE_MATCH_KEY" in body, (
            "the Live tab must store its closest match, or the orange trace "
            "is re-derived from a moving index on every poll"
        )
        assert "st.session_state[match_key] = _freeze_closest_match(" in body, (
            "the closest match must be frozen and stored next to the path"
        )
        assert "frozen_match=frozen_match" in body, (
            "the frozen match must be handed to the figure"
        )


class TestTheLiveChartShowsTheWholeTape:
    """The newest bar is on the chart, and the axis covers it."""

    def _figure(self, pipe, anchor, start):
        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None, "the fixture must yield a path"
        return build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start, hist_stop=pipe.n_bars,
        )

    def test_the_newest_bar_is_inside_the_visible_axis(self):
        """The regression: the tape's tail was clipped off the chart.

        The appended-bar mechanism drew the newest bar at the archive's *next*
        index, which was outside the axis range -- so the reader could not see
        the bar the tab existed to show them.

        **The pin is taken on the *first* window and the chart drawn on a
        later, larger one.**  That is the situation the tab is actually in: the
        reader arrives, and the archive grows underneath them.  Pinning and
        drawing on the same window would not exercise the clipping at all --
        the projection would be the longer extent, and the history's tail
        would sit comfortably inside it.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        hist_x = [int(v) for v in fig.data[0].x]
        lo, hi = fig.layout.xaxis.range

        assert hist_x[-1] == later.n_bars - 1, (
            "the history must end on the archive's newest bar"
        )
        assert hist_x[-1] <= hi, (
            "the newest bar must be inside the visible axis, not clipped off it"
        )
        assert hist_x[0] >= lo

    def test_the_history_really_does_reach_the_archive_tail(self):
        """The chart draws the tape, not a fixed-width window.

        A history of exactly the requested width would be a chart that never
        changes -- the opposite of "the tape rolls".
        """
        pipe = _window(0, 1560)
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)

        fig = self._figure(pipe, anchor, start)
        hist_x = [int(v) for v in fig.data[0].x]
        assert len(hist_x) == pipe.n_bars - start, (
            "the history must run from the pinned start to the archive's tail"
        )

    def test_the_projection_starts_on_the_anchor_bar(self):
        """The projection's offset 0 **is** the anchor bar.

        ``forecast_paths`` is explicit: "Offset 0 is the anchor itself and is
        exactly 0.0 ... it is the bar history and forecast share, so a chart
        drawn from these offsets has the two halves meeting at one point
        rather than at two points a bar apart."

        The chart used to start the projection one bar further right, which
        put the anchor's timestamp on the axis twice and made every projected
        label sit one bar behind the bar it named.
        """
        pipe = _window(0, 1170)
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)

        fig = self._figure(pipe, anchor, start)
        proj_x = [int(v) for v in fig.data[-1].x]
        assert min(proj_x) == anchor, (
            "the projection's offset 0 is the anchor bar, not the one after it"
        )


class TestTheAxisHasNoDuplicateTimestamp:
    """The anchor's timestamp appears on the axis exactly once.

    The user-visible symptom: a timestamp printed twice at the centre of the
    Live chart, once as the history's last tick and again as the projection's
    first tick.  It was one moment shown twice -- and, because the projection
    was drawn a bar too far right, the two copies sat either side of the
    boundary looking like two *different* times.
    """

    def _labels(self, fig):
        return list(fig.layout.xaxis.ticktext)

    def test_the_anchor_timestamp_is_not_printed_twice(self):
        """No label may repeat on the axis.

        The two halves of the chart share the anchor bar, so the history's
        last tick and the projection's first tick are the same moment.  Only
        one of them may carry a label.
        """
        pipe = _window(0, 1170)
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)

        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None
        fig = build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start, hist_stop=anchor + 1,
        )

        labels = self._labels(fig)
        repeated = {t for t in labels if labels.count(t) > 1}
        assert not repeated, (
            "a timestamp is printed more than once on the axis: %r" % (repeated,)
        )

    def test_the_anchor_moment_is_not_shown_under_two_different_labels(self):
        """The duplicate was two *different* strings naming one moment.

        The history labels the anchor in full ("2026-01-07 20:59") and the
        projection labelled it as a bare clock time ("20:59").  Those are the
        same instant, so a test that only looks for a repeated *string* would
        sail straight past the bug -- which is exactly what the reader saw:
        one time, printed twice, in two spellings, either side of the line.
        """
        pipe = _window(0, 1170)
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)

        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None
        fig = build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start, hist_stop=anchor + 1,
        )

        anchor_local = pd.Timestamp(pipe.bars["timestamp"].iloc[anchor])
        hhmm = anchor_local.strftime("%H:%M")
        # Every label that names the anchor's time of day, in either spelling.
        naming = [t for t in self._labels(fig) if str(t).endswith(hhmm)]
        assert len(naming) == 1, (
            "the anchor's time of day is shown %d times (%r) -- it is one moment "
            "and must be printed once" % (len(naming), naming)
        )

    def test_the_anchor_bar_is_labelled_exactly_once(self):
        """The boundary is named once -- and it *is* named.

        Dropping the duplicate must not drop the boundary with it: an
        unlabelled separator is a boundary the reader cannot place.
        """
        pipe = _window(0, 1170)
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)

        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None
        fig = build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start, hist_stop=anchor + 1,
        )

        tickvals = [int(v) for v in fig.layout.xaxis.tickvals]
        labels = self._labels(fig)
        at_anchor = [t for v, t in zip(tickvals, labels) if v == anchor]
        assert len(at_anchor) == 1, (
            "the anchor bar must be labelled exactly once, got %r" % (at_anchor,)
        )
        # And the label is the anchor's own time of day.
        anchor_ts_local = pd.Timestamp(pipe.bars["timestamp"].iloc[anchor])
        assert at_anchor[0] == anchor_ts_local.strftime("%Y-%m-%d %H:%M")

    def test_the_projection_starts_on_the_anchor_bar(self):
        """Offset 0 is the anchor, so the projection and the history share it.

        The projection used to be drawn at ``anchor + 1 + offset``, putting its
        0% point -- the anchor's own close -- on the *next* bar's x-position.
        """
        pipe = _window(0, 1170)
        anchor_ts, start_ts = live_pinned_anchor(pipe, "QQQ", 10)
        anchor = resolve_bar_index(pipe, anchor_ts)
        start = resolve_bar_index(pipe, start_ts)

        path = forecast_path_for(
            "k", pipe, length=10, horizon=10, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None
        fig = build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start, hist_stop=anchor + 1,
        )

        hist_x = [int(v) for v in fig.data[0].x]
        proj_x = [int(v) for v in fig.data[-1].x]
        assert min(proj_x) == max(hist_x), (
            "the projection's offset 0 is the anchor, which is the history's "
            "last bar -- they must share it"
        )


class TestTheLiveTabIsWired:
    """The tab must use the pin rather than re-derive an anchor."""

    def test_the_tab_resolves_its_pins_against_the_refreshed_pipeline(self):
        """The pinned moments are looked up in the *refreshed* frame.

        They are moments, so the lookup finds the same bar every time -- but
        the index it lands on is not, which is exactly why the moment is what
        was stored.
        """
        body = app_block(
            r"def render_live_tab\(.*?\n(?=def )",
            "render_live_tab",
        )
        assert "live_pinned_anchor" in body, (
            "the tab must take its anchor and start from the pin"
        )
        assert "resolve_bar_index" in body, (
            "the tab must resolve the pinned moments against the refreshed frame"
        )
        assert "st.session_state.live_projection_anchor" not in body

    def test_the_dead_append_mechanism_is_gone(self):
        """The synthetic appended bar is not drawn any more.

        It was the source of the clipping: a point drawn at the archive's next
        index rather than at a bar the archive actually has.
        """
        source = app_source()
        assert "live_append_new_bar" not in source, (
            "the appended-bar mechanism is gone: it drew a point past the "
            "archive's last index, outside the axis range"
        )
        assert "LIVE_STORE" not in source, (
            "the store that held the appended points is gone with it"
        )
        assert "live_bars" not in source, (
            "the figure no longer takes a synthetic appended bar"
        )


class TestTheSeparatorIsPinned:
    """The dotted line stays on the same bar as the tape rolls.

    The line down the Live chart is the boundary between the tape and the
    projection.  It stands on the **first bar that belongs to the projection
    alone** -- the bar just after the one the frozen projection grows out of
    -- and it has to **hold still** while the tape rolls past it, otherwise
    the reader is watching a forecast that is already behind the tape, with no
    marker saying where the prediction was actually made.

    It did not hold still.  The line was drawn at the *history's last bar*,
    and the Live tab's history runs from a pinned start all the way to the
    archive's tail -- so the line sat on the newest bar of the tape and slid
    one place to the right on every poll.

    It also pointed at the wrong bar.  Drawn on the *anchor* it stood on the
    last real bar, which reads as "the tape ends here" -- and on the Live tab
    the tape does not end there at all, it rolls on past the line to the
    right.  The first bar that exists only in the projection is the one after
    the anchor, because offset 0 of a forecast path *is* the anchor.
    """

    def _figure(self, pipe, anchor, start):
        path = forecast_path_for(
            "k", pipe, length=10, horizon=20, k=1,
            amplitude_weight=0.0, window=(start, anchor + 1),
        )
        assert path is not None, "the fixture must yield a path"
        return build_forecast_path_figure(
            pipe, path, history_bars=10, title="Live", show_median=False,
            projection_anchor=anchor, hist_start=start, hist_stop=pipe.n_bars,
        )

    def _separator_x(self, fig) -> float:
        lines = [s for s in fig.layout.shapes if s.type == "line"]
        assert lines, "the chart must draw its separator line"
        return float(lines[0].x0)

    def test_the_separator_does_not_move_when_the_archive_grows(self):
        """The regression: the line walked right with the tape.

        The pin is taken on the first window and the chart drawn on a later,
        larger one -- the situation the tab is actually in, since the reader
        arrives and the archive grows underneath them.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        assert self._separator_x(fig) == float(anchor) + 1, (
            "the separator must stay on the first projected bar, the one after "
            "the pinned anchor"
        )
        # And the bar it names is the same bar, not merely the same index.
        assert later.bars["timestamp"].iloc[int(self._separator_x(fig))] == (
            later.bars["timestamp"].iloc[anchor + 1]
        )

    def test_the_separator_is_not_the_archive_tail(self):
        """The specific failure: the line was drawn at the tape's newest bar.

        A separator drawn at the tail is stable only for as long as the
        archive is, and it moves the instant a bar arrives.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        assert self._separator_x(fig) != float(later.n_bars - 1), (
            "the separator must not be the archive's newest bar -- that is what "
            "made it slide forward with every new bar"
        )
        assert later.n_bars - 1 > anchor, "the tail has moved on; the line has not"

    def test_the_shaded_region_covers_the_projection_and_is_not_inverted(self):
        """The grey band marks the projected bars, and it is never drawn backwards.

        The band ran from the separator to the projection's last bar.  On the
        *Live* tab the tape runs *past* the separator, so the band's far end
        fell behind its near end and the rectangle was drawn inside out --
        a region on the wrong side of its own boundary.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert rects, "the chart must shade the projected region"
        x0, x1 = float(rects[0].x0), float(rects[0].x1)
        assert x0 <= x1, (
            "the shaded region is drawn backwards: it starts at %r and ends at %r"
            % (x0, x1)
        )
        assert x0 == float(anchor) + 1, (
            "the shaded region must begin at the separator -- the first "
            "projected bar"
        )

        proj_x = [int(v) for v in fig.data[-1].x]
        assert x1 >= max(proj_x) - 0.5, (
            "the shaded region must cover every projected bar"
        )

    def test_the_projection_still_starts_on_the_pinned_bar(self):
        """The line and the projection must agree on where the boundary is.

        The line stands on the first bar that belongs to the projection
        *alone*, but the projection's **trace** still starts on the anchor:
        offset 0 of a forecast path *is* the anchor bar, so the two halves
        meet at one point rather than at two points a bar apart.  A pinned
        line with a projection anchored somewhere else would be a line
        pointing at nothing.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        proj_x = [int(v) for v in fig.data[-1].x]
        assert min(proj_x) == anchor, (
            "the projection's offset 0 is the anchor bar -- the one the "
            "projection grows out of"
        )
        assert min(proj_x) == int(self._separator_x(fig)) - 1, (
            "the separator stands one bar past the anchor, on the first bar "
            "that belongs to the projection alone"
        )

    def test_the_separator_stands_on_the_first_projected_bar(self):
        """The line is on the first bar that exists *only* in the projection.

        The anchor is a real bar -- the last one at the moment the forecast
        was made -- so a line drawn on it points at the tape rather than at
        the projection.  The first projected-only bar is the one after it,
        and that is where the line belongs.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        sep = int(self._separator_x(fig))

        # The bar the line stands on is *not* a bar of the real tape's
        # projected history -- it is one bar past the anchor.
        assert sep == anchor + 1
        assert sep > anchor, (
            "the separator must not sit on the anchor: the anchor is a real "
            "bar, and the line is meant to mark where the projection begins"
        )

    def test_the_separator_names_the_same_moment_in_every_frame(self):
        """The line is pinned -- it marks the same *boundary* in every frame.

        The pin is taken on the first window and the chart drawn on a later,
        larger one -- the situation the tab is actually in, since the reader
        arrives and the archive grows underneath them.

        The boundary is "the bar right after the pinned anchor".  In the
        *first* frame the anchor is the archive's tail, so that bar does not
        exist yet -- it is a projected bar, and the line is drawn on the
        position it will occupy.  In the *later* frame the same position
        names a real bar: the one that arrived immediately after the anchor.
        The line must be in the same place in both.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor_first = resolve_bar_index(first, anchor_ts)
        anchor_later = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor_first is not None and anchor_later is not None
        assert start is not None

        fig_first = self._figure(first, anchor_first, start)
        fig_later = self._figure(later, anchor_later, start)

        sep_first = int(self._separator_x(fig_first))
        sep_later = int(self._separator_x(fig_later))

        # The same position relative to the pinned anchor, in both frames.
        assert sep_first == anchor_first + 1
        assert sep_later == anchor_later + 1

        # And the anchor the line grows from is the same moment in both --
        # that is the pin holding.
        assert first.bars["timestamp"].iloc[anchor_first] == (
            later.bars["timestamp"].iloc[anchor_later]
        )

        # In the grown archive the separator's position is the real bar that
        # came straight after the anchor -- the next bar of the tape, whatever
        # its clock time (the anchor sits at a session's close, so that is the
        # next session's open).
        after = later.bars["timestamp"].iloc[sep_later]
        assert after == later.bars["timestamp"].iloc[anchor_later + 1]
        assert after > later.bars["timestamp"].iloc[anchor_later]

    def test_the_projection_meets_the_tape_at_the_anchor(self):
        """The two halves join at the anchor bar, at the same price.

        The projection is rebased onto the *anchor's* close, so its offset 0
        sits exactly on the tape's price at the anchor.  The base used to
        be the close of whatever bar the line happened to be drawn at -- the
        tape's tail -- so the join was a visible jump rather than a
        continuation, and the "frozen" projection's first point moved with
        the tape.

        The join is at the *anchor*, not at the separator: the separator
        stands on the first projected-only bar, and the projection's trace
        still starts one bar to its left, on the real bar it grows out of.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        hist_x = [int(v) for v in fig.data[0].x]
        hist_y = [float(v) for v in fig.data[0].y]
        proj_x = [int(v) for v in fig.data[-1].x]
        proj_y = [float(v) for v in fig.data[-1].y]

        # The tape's price at the anchor...
        tape_at_anchor = hist_y[hist_x.index(anchor)]
        # ...and the projection's offset 0, which *is* that bar.
        assert proj_y[proj_x.index(anchor)] == pytest.approx(
            tape_at_anchor, abs=1e-9
        ), (
            "the projection must be rebased onto the anchor bar's own close, "
            "or the two halves do not meet"
        )
        # And the base is the *anchor's* close, not the tail's.
        assert tape_at_anchor == pytest.approx(
            float(later.close[anchor]), abs=1e-9
        )

    def test_the_separator_is_named_once_on_the_axis(self):
        """The boundary carries one label, even with the tape running past it.

        The anchor sits *inside* the Live tab's history, so it is not one of
        the history's own ticks and has to be added as an explicit one.  The
        projection's offset 0 is that same bar, so keeping it would print the
        anchor twice -- once as a full timestamp and once as a bare clock
        time.

        The **separator itself** is one bar further on, and it is labelled by
        the projection's offset-1 tick: the clock time that bar *would*
        occur at.  An unlabelled separator is a boundary the reader cannot
        place, so it must be named exactly once too.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts, start_ts = live_pinned_anchor(first, "QQQ", 10)
        assert anchor_ts is not None and start_ts is not None

        anchor = resolve_bar_index(later, anchor_ts)
        start = resolve_bar_index(later, start_ts)
        assert anchor is not None and start is not None

        fig = self._figure(later, anchor, start)
        tickvals = [int(v) for v in fig.layout.xaxis.tickvals]
        labels = [str(t) for t in fig.layout.xaxis.ticktext]

        assert len(tickvals) == len(labels), (
            "every tick must carry exactly one label"
        )
        assert len(set(tickvals)) == len(tickvals), (
            "the axis must not print two labels on one bar: %r" % (tickvals,)
        )
        at_anchor = [t for v, t in zip(tickvals, labels) if v == anchor]
        assert len(at_anchor) == 1, (
            "the anchor must be labelled exactly once, got %r" % (at_anchor,)
        )
        assert at_anchor[0] == pd.Timestamp(
            later.bars["timestamp"].iloc[anchor]
        ).strftime("%Y-%m-%d %H:%M")

        # The separator is labelled exactly once as well.
        sep = int(self._separator_x(fig))
        at_sep = [t for v, t in zip(tickvals, labels) if v == sep]
        assert len(at_sep) == 1, (
            "the separator must be labelled exactly once, got %r" % (at_sep,)
        )
        # Its label is the projected clock time of the bar it stands on.
        assert at_sep[0] == (
            pd.Timestamp(later.bars["timestamp"].iloc[anchor])
            + pd.Timedelta(minutes=1)
        ).strftime("%H:%M")
