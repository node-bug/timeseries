"""Tests for the **Projection** tab's pinned separator line.

The dotted line down the middle of that chart is the boundary between the
recent bars (left) and the projected bars (right).  It is the archive's newest
bar at the moment the reader arrived, and the whole point of the tab is that
the chart is a *fixed reference*: "what usually followed a window shaped like
the most recent one?"

**The line used to be drawn at the archive's tail**, so every new bar moved it
one place to the right and dragged the entire history along with it.  Nothing
errored and the chart still looked plausible -- it was simply describing a
different window every minute, while the caption underneath said it was
"the archive's most recent N bars".

These tests assert on the **separator's position and the history's extent**,
because that is the only level at which the claim "this line does not move" is
true or false.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

from timeseries.forecast import forecast_paths
from timeseries.pipeline import Pipeline

from apphelpers import (
    app_block,
    live_tape,
    load_app_functions,
    window_over_tape,
)


class _FakeSt:
    """A ``st`` stand-in whose ``session_state`` is a plain writable ``dict``."""

    cache_resource = staticmethod(st.cache_resource)
    cache_data = staticmethod(st.cache_data)

    def __init__(self):
        self.session_state: dict = {}


_ST = _FakeSt()
_ns = load_app_functions(
    {
        "build_forecast_path_figure",
        "forecast_path_for",
        "projection_pinned_anchor",
        "projection_anchor_key",
        "projection_figure_span",
        "resolve_bar_index",
        "_closest_match_path",
        "_closest_distance_label",
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
        "PROJECTION_ANCHOR_KEY": "projection_anchor_ts",
    },
)

projection_pinned_anchor = _ns["projection_pinned_anchor"]
projection_anchor_key = _ns["projection_anchor_key"]
projection_figure_span = _ns["projection_figure_span"]
resolve_bar_index = _ns["resolve_bar_index"]
build_forecast_path_figure = _ns["build_forecast_path_figure"]
forecast_path_for = _ns["forecast_path_for"]


@pytest.fixture(autouse=True)
def _fresh_state():
    """Each test starts with an empty session state, so pins cannot leak."""
    _ST.session_state = {}
    yield
    _ST.session_state = {}


_TAPE = live_tape()

#: The tab's own window widths.
LENGTH = 240
HORIZON = 60


def _window(start: int, stop: int) -> Pipeline:
    """A pipeline over ``_TAPE``'s bars ``[start, stop)``."""
    return window_over_tape(_TAPE, start, stop)


def _separator_x(fig) -> float:
    """The x-position of the figure's dotted separator line."""
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert lines, "the chart must draw its separator line"
    return float(lines[0].x0)


class TestTheSeparatorIsPinned:
    """The line stays on the same bar as the archive grows.

    The line stands on the **first bar that belongs to the projection
    alone** -- the bar just after the one the projection grows out of -- so
    it marks where the projected bars begin rather than pointing at the last
    real bar.
    """

    def test_the_separator_does_not_move_when_the_archive_grows(self):
        """The regression: the line was drawn at the archive's tail.

        The pin is taken on the first window and the chart drawn on later,
        larger ones -- the situation the tab is actually in, since the reader
        arrives and the archive grows underneath them.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        anchor_ts = projection_pinned_anchor(first, "QQQ", LENGTH)
        assert anchor_ts is not None

        # The span the *tab* would use on the later archive.
        span = projection_figure_span(later, "QQQ", LENGTH)
        assert span is not None
        anchor, hist_start, hist_stop = span

        path = forecast_path_for(
            "k", later, length=LENGTH, horizon=HORIZON, k=1,
            amplitude_weight=0.0,
        )
        assert path is not None

        fig = build_forecast_path_figure(
            later, path, history_bars=LENGTH, title="P",
            projection_anchor=anchor,
            hist_start=hist_start, hist_stop=hist_stop,
        )

        assert _separator_x(fig) == float(anchor) + 1, (
            "the separator must stay on the first projected bar, the one after "
            "the pinned anchor"
        )
        # And the bar it names is the same bar, not merely the same index.
        assert later.bars["timestamp"].iloc[int(_separator_x(fig))] == (
            later.bars["timestamp"].iloc[anchor + 1]
        )

    def test_the_separator_is_not_the_archive_tail(self):
        """The specific failure: the line was drawn at the archive's newest bar.

        This is the assertion that says the line is *pinned* rather than
        merely *stable*.  A separator drawn at the tail is stable only for as
        long as the archive is, and it moves the instant a bar arrives.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        projection_pinned_anchor(first, "QQQ", LENGTH)
        span = projection_figure_span(later, "QQQ", LENGTH)
        assert span is not None
        anchor, _, _ = span

        assert anchor != later.n_bars - 1, (
            "the separator must not be the archive's newest bar -- that is what "
            "made it slide forward with every new bar"
        )
        # The tail has moved on; the separator has not.
        assert later.n_bars - 1 > anchor

    def test_the_history_extent_does_not_move_either(self):
        """The line and the history are two ends of the same span.

        A pinned separator with a sliding history would be a line drawn over
        different bars on every pass -- still "not moving", but no longer the
        chart the reader is reading.
        """
        first = _window(0, 1170)
        later = _window(0, 1560)

        projection_pinned_anchor(first, "QQQ", LENGTH)
        span = projection_figure_span(later, "QQQ", LENGTH)
        assert span is not None
        anchor, hist_start, hist_stop = span

        path = forecast_path_for(
            "k", later, length=LENGTH, horizon=HORIZON, k=1,
            amplitude_weight=0.0,
        )
        fig = build_forecast_path_figure(
            later, path, history_bars=LENGTH, title="P",
            projection_anchor=anchor,
            hist_start=hist_start, hist_stop=hist_stop,
        )

        hist_x = [int(v) for v in fig.data[0].x]
        assert hist_x[0] == hist_start
        assert hist_x[-1] == anchor
        assert len(hist_x) == LENGTH, (
            "the history must be exactly the requested width, ending on the "
            "separator"
        )

    def test_the_pinned_moment_resolves_to_the_same_bar_when_the_archive_rolls(self):
        """The archive's front moves too, and that must not move the line.

        This is the case an index-based pin gets wrong: the same moment sits at
        a *different index* in the rolled frame.
        """
        first = _window(0, 1170)
        rolled = _window(200, 1560)

        anchor_ts = projection_pinned_anchor(first, "QQQ", LENGTH)
        assert anchor_ts is not None

        idx_first = resolve_bar_index(first, anchor_ts)
        idx_rolled = resolve_bar_index(rolled, anchor_ts)
        assert idx_first is not None and idx_rolled is not None
        assert idx_first != idx_rolled, (
            "the fixture must actually shift the index, or this test proves "
            "nothing about why a timestamp is used"
        )
        assert (first.bars["timestamp"].iloc[idx_first]
                == rolled.bars["timestamp"].iloc[idx_rolled]), (
            "the separator must be the same bar after the archive rolls"
        )

    def test_a_refetch_does_not_re_pin(self):
        """The second pass reads the pin; it does not take a new one."""
        first = _window(0, 1170)
        later = _window(0, 1560)

        pinned_a = projection_pinned_anchor(first, "QQQ", LENGTH)
        pinned_b = projection_pinned_anchor(later, "QQQ", LENGTH)

        assert pinned_a == pinned_b, "a refetch must not move the pinned separator"

    def test_moving_the_control_re_pins(self):
        """The reader's gesture is the one thing that re-pins the chart."""
        pipe = _window(0, 1170)

        narrow = projection_pinned_anchor(pipe, "QQQ", 60)
        assert narrow is not None
        # The separator is the archive's newest bar either way; the *window*
        # it delimits changes, which is what the width records.
        again = projection_pinned_anchor(pipe, "QQQ", 60)
        assert again == narrow

    def test_an_empty_archive_pins_nothing(self):
        """There is no bar to pin, and nothing may be stored."""
        empty = Pipeline.from_frame(
            pd.DataFrame({"timestamp": pd.to_datetime([], utc=True), "close": []}),
            length=60, timeframe="1m",
        )
        assert projection_pinned_anchor(empty, "QQQ", LENGTH) is None
        assert _ST.session_state == {}, (
            "nothing may be pinned from an empty archive"
        )

    def test_the_pin_is_per_symbol(self):
        """Two instruments cannot share a pin."""
        assert projection_anchor_key("QQQ") != projection_anchor_key("AAPL")


class TestTheProjectionTabUsesThePin:
    """The tab must *use* the pin rather than re-derive the separator."""

    def test_the_reference_chart_is_drawn_against_the_pinned_anchor(self):
        """A correct helper the tab does not call fixes nothing."""
        body = app_block(
            r"def _render_forecast_path\(.*?\n(?=def )",
            "_render_forecast_path",
        )
        assert "projection_figure_span" in body, (
            "the Projection tab must take its separator from the pin"
        )
        assert "projection_anchor=anchor" in body, (
            "the figure must be told where the separator is"
        )
        assert "hist_start=hist_start" in body and "hist_stop=hist_stop" in body, (
            "the figure must be told the history's extent, or it draws the "
            "archive's tail and drags the separator with it"
        )

    def test_the_panel_chart_shares_the_separator(self):
        """The two charts are compared bar for bar.

        They are drawn on the same axes, one above the other, and the caption
        says so.  A boundary that moved on one chart and not the other would
        make the comparison meaningless -- so the panel chart reads the same
        pin.
        """
        body = app_block(
            r"def _render_panel_forecast_path\(.*?\n(?=def )",
            "_render_panel_forecast_path",
        )
        assert "projection_figure_span" in body, (
            "the panel chart must use the same pinned separator as the chart "
            "above it"
        )
        assert "projection_anchor=anchor" in body
        assert "hist_start=hist_start" in body and "hist_stop=hist_stop" in body, (
            "the panel chart must use the same history extent as the chart above"
        )

    def test_the_forecast_tab_is_unaffected(self):
        """The *Forecast* tab's chart is about the reader's own brushed window.

        It passes an explicit ``window=`` and must keep doing so: a pin would
        be wrong there, because the window is the reader's choice rather than
        the archive's tail.
        """
        body = app_block(
            r"def _render_window_forecast\(.*?\n(?=def )",
            "_render_window_forecast",
        )
        assert "window=(start, stop)" in body, (
            "the Forecast tab must keep drawing the window the reader brushed"
        )
