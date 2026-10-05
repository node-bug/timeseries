"""Tests for brush-to-span parsing -- the Price tab's first chart is brushable.

The Price tab's top chart defines the query window, so a box dragged on it has to
become a span of real bars.  `app.selection_to_span` does that parsing, and it runs
directly on whatever Streamlit hands back from a drag.  That makes it the one piece of
selection code where a bad payload is a *page-level* failure rather than a bad query:
an unhandled exception there replaces the whole app with a traceback.

`app.py` is a Streamlit script whose import runs a whole UI, so it is not importable
under pytest.  The pure functions it depends on are therefore exec'd out of the source
by name via `apphelpers.load_app_functions`, so the bodies under test are the app's own
rather than copies of them.

The behaviours worth stating up front, all pinned below:

* A box wins over a lasso when an event carries both.  Streamlit stores them in
  separate lists, so only one can be intended; box-first is the app's established
  order and stops a stale lasso from overriding a fresh box.
* Every malformed payload resolves to ``None`` -- "no brush" -- never an exception.
  Callers fall back to the From/To pickers, which is always a usable answer.
* **A brush of N bars is a query of N bars.**  There is no fixed grid to snap onto:
  `find_matches` reads `query.length`, and `Pipeline.run` reads the same length for
  the forward returns, the baseline and the bootstrap block.  `TestVariableLength`
  pins the app-side half of that; `test_pipeline.py` pins the library half.
"""

from __future__ import annotations

import ast
import re
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from apphelpers import (
    app_block,
    app_called_names,
    app_defined_names,
    app_source,
    app_text,
    load_app_functions,
    price_tab_body,
)

# The selection helpers, exec'd out of app.py.  ``to_utc``, ``nearest_index``,
# ``session_options`` and ``is_session_complete`` were pulled in here too: the first
# two backed the From/To picker path and the last two built the session dropdown's
# labels.  All four went with the *Chart* tab, which was the only thing that drew a
# picker or rendered a session dropdown, so they are gone from app.py too.
#
# ``session_for_index`` and ``latest_session_window`` were here until the session fence
# was removed.  The first computed the ``bounds`` span ``resolve_query_window`` used to
# confine a query to one trading day; the second chose the fenced *default* query.  Both
# lost their last caller at the same time, so both are gone from app.py and from here.
_NS = load_app_functions({
    "selection_to_span",
    "snap_to_grid", "brush_bar_count", "selection_summary", "resolve_query_window",
    "session_spans",
})
selection_to_span = _NS["selection_to_span"]
snap_to_grid = _NS["snap_to_grid"]
brush_bar_count = _NS["brush_bar_count"]
selection_summary = _NS["selection_summary"]
resolve_query_window = _NS["resolve_query_window"]
session_spans = _NS["session_spans"]

#: Loaded for the assertions below, which check the *rendered* stamps rather than
#: re-deriving them -- the app formats a stamp in exactly one place, and a test that
#: hand-built the expected string would be asserting against a second opinion.
stamp_label = _NS["stamp_label"]
stamp_span = _NS["stamp_span"]
stamp_zone = _NS["stamp_zone"]
stamp_column = _NS["stamp_column"]


@contextmanager
def _resolution(key):
    """Run a block with ``ACTIVE_TIMEFRAME`` set to ``key``, then restore it.

    The app resolves a resolution through one module-level list that the gate mutates,
    so the only faithful way to exercise a helper under the other resolution is to set
    that list -- exactly what :func:`app.timeframe_scope` does.  Restored on the way
    out, including on failure, because these tests share one ``ACTIVE_TIMEFRAME``
    instance and a leaked ``"1d"`` would silently re-label every later assertion in
    the file.
    """
    previous = _NS["ACTIVE_TIMEFRAME"][0]
    _NS["ACTIVE_TIMEFRAME"][0] = key
    try:
        yield
    finally:
        _NS["ACTIVE_TIMEFRAME"][0] = previous

# Read from the app rather than restated, so changing the app's floor fails here
# instead of leaving a test that disagrees with the code.
MIN_QUERY_BARS = _NS["MIN_QUERY_BARS"]

#: The app's session boundary, read from its source rather than restated.
EASTERN = _NS["EASTERN"]

SCALE = 600
STAMPS = pd.Series(
    pd.date_range("2026-09-01 13:30", periods=SCALE, freq="1min", tz="UTC")
)


def event(selection=None, **shapes):
    """Build a selection event.  A bare positional is the whole ``selection`` value."""
    if selection is not None:
        return {"selection": selection}
    return {"selection": {k: v for k, v in shapes.items()}}


def box(*x):
    return [{"x": list(x)}]


def lasso(*x):
    return [{"x": list(x)}]


class TestBarIndexAxis:
    """Every price chart draws bars against their index, so x values are bar indices.

    Closed hours are always compressed: the app has one axis and no switch.  Plotly
    hands back the numbers off that axis, and they are already bar indices.
    """

    def test_box_gives_a_span(self):
        assert selection_to_span(event(box=box(100, 300)), SCALE) == (100, 300)

    def test_endpoints_are_ordered(self):
        """A box dragged right-to-left is the same span as one dragged the other way."""
        assert selection_to_span(event(box=box(300, 100)), SCALE) == (100, 300)

    def test_box_wins_over_lasso(self):
        """Both present: the box is the intended shape, not a leftover lasso."""
        got = selection_to_span(event(box=box(50, 80), lasso=lasso(10, 20)), SCALE)
        assert got == (50, 80)

    def test_lasso_uses_spatial_extent_not_draw_order(self):
        """A lasso's first two points are where the loop started, not its left edge."""
        got = selection_to_span(event(lasso=lasso(300, 100, 400, 200)), SCALE)
        assert got == (100, 400)

    def test_fractions_round_to_bars(self):
        assert selection_to_span(event(box=box(10.2, 20.7)), SCALE) == (10, 21)

    def test_clamped_to_the_archive(self):
        assert selection_to_span(event(box=box(-50, 100)), SCALE)[0] == 0
        assert selection_to_span(event(box=box(500, 99999)), SCALE)[1] == SCALE - 1

    def test_span_narrower_than_a_bar_is_no_brush(self):
        """Both ends rounding to one bar leaves no window to query."""
        assert selection_to_span(event(box=box(7.2, 7.4)), SCALE) is None

    @pytest.mark.parametrize("payload", ["nonsense", [1, 2], {"a": 1}, None, True])
    def test_non_numeric_endpoint_is_no_brush(self, payload):
        """The axis carries indices, so anything that is not a number is not a span.

        There is no timestamp parser left to fall back on, and there should not be:
        a string on a bar-index axis cannot be a bar, and guessing a date from one
        would invent a window the reader never drew.
        """
        assert selection_to_span(event(box=box(payload, 100)), SCALE) is None

    @pytest.mark.parametrize("lo, hi", [(1e20, 1e21), (-1e300, 1e300)])
    def test_absurd_numeric_values_are_no_brush(self, lo, hi):
        """Out-of-range magnitudes clamp rather than raise -- a drag can leave the axis."""
        got = selection_to_span(event(box=box(lo, hi)), SCALE)
        assert got is None or all(0 <= i < SCALE for i in got)


class TestDegenerateSelections:
    """Anything that is not a usable span is "no brush" -- never an exception."""

    @pytest.mark.parametrize("payload", [
        None,
        {},
        {"selection": {}},
        {"selection": {"box": [], "lasso": []}},
        {"selection": {"box": [{"y": [1, 2]}]}},        # shape without x
        {"selection": {"box": [{"x": [7]}]}},           # one endpoint
    ])
    def test_returns_none(self, payload):
        assert selection_to_span(payload, SCALE) is None

    @pytest.mark.parametrize("width", [0, 0.4])
    def test_zero_width_is_not_a_span(self, width):
        """A click or a sub-bar drag covers no window, so there is nothing to query."""
        assert selection_to_span(event(box=box(7, 7 + width)), SCALE) is None

    def test_event_that_is_not_a_dict(self):
        assert selection_to_span(["not", "a", "dict"], SCALE) is None

    @pytest.mark.parametrize("payload", [
        [{"x": [1, 2]}],            # selection is a list, not a mapping
        {"box": [[1, 2]]},          # shape is a list, not a mapping
    ])
    def test_malformed_shape_does_not_raise(self, payload):
        """These come from nowhere in normal use, but a crash here kills the page."""
        assert selection_to_span(event(payload), SCALE) is None

    def test_unparseable_endpoint(self):
        got = selection_to_span(event(box=box("nonsense", "also bad")), SCALE)
        assert got is None


class TestChartWiring:
    """The brush is only useful if it is actually armed and actually read."""

    def test_price_tab_chart_is_selectable(self):
        """The first graph takes a brush; without this the parsing above is dead code."""
        body = price_tab_body()
        assert "selectable=True" in body, (
            "the Price tab's top chart is no longer brushable"
        )
        assert 'on_select="rerun"' in body, (
            "the Price tab's top chart is brushable but nothing handles the selection"
        )

    def test_match_panel_stays_read_only(self):
        """The second graph shows what the matcher found; it must not define a query."""
        body = price_tab_body()
        compact = body[body.index("render_matches_tab("):]
        assert "selectable=True" not in compact, (
            "the match panel must stay read-only -- a brush there has no handler"
        )

    def test_brush_is_read_before_the_snap(self):
        """Otherwise the query lags the brush by one click.

        Streamlit replays the whole script on a selection, so the event is already in
        session_state before the tab body runs.  Reading it after the length is
        resolved would mean the brush only takes effect on the *following* rerun.
        """
        read = app_source().index("st.session_state.get(price_brush_key())")
        # The brush used to be resolved against a fixed-length grid; a manual
        # selection now defines its own length, so the boundary is the block that
        # works the length out.  The invariant is unchanged -- read first, resolve
        # second -- only the step's name moved.
        resolve = app_source().index("# ---------------- Use the selection's own length")
        assert read < resolve, (
            "the Price tab brush is read after the length is resolved; it will lag "
            "one rerun"
        )

    def test_read_and_write_agree_on_the_key(self):
        """Reader and renderer must not drift, or the brush is silently lost."""
        declared = re.search(r'^PRICE_BRUSH_KEY\s*=\s*"([^"]+)"',
                             app_source(), re.M)
        assert declared, "PRICE_BRUSH_KEY is not declared at module level"
        # Both ends go through the same accessor, so reader and renderer cannot
        # drift.  Asserting on the *bare* constant here would be the weaker claim:
        # the whole reason the accessor exists is that the bare constant is only
        # the base name and the live key carries a resolution suffix.
        assert "st.session_state.get(price_brush_key())" in app_source()
        assert "key = price_brush_key() if selectable" in app_source(), (
            "the chart is rendered under a different key than the one read"
        )


class TestSnapToGrid:
    """A selection must be centred inside the fixed-length query, not appended to.

    This is a correctness bug, not a presentation one.  The old snap extended only
    *forwards*::

        stop = min(start + length, n);  start = stop - length

    so a 5-bar brush at bars 100-104 became a 20-bar query at 100-119 -- 15 bars the
    user never selected, three quarters of the query, treated as if they had been
    asked about.  And because the extra bars came off the *forward* side, they are
    exactly the bars the forecast measures its returns from, so the padding
    contaminated the number the app reports.

    Centring fixes that: the padding is split either side, so the query sits on the
    gesture.  The edge cases below are where centring is impossible, and the rule is
    the same -- shift inwards rather than invent tape.
    """

    N, L = 8049, 20

    def test_exact_length_selection_is_untouched(self):
        assert snap_to_grid(100, 120, self.L, self.N) == (100, 120)
        assert snap_to_grid(4000, 4020, self.L, self.N) == (4000, 4020)

    def test_short_selection_is_centred_not_extended(self):
        """The regression: padding must appear on *both* sides.

        A 5-bar brush used to become a 20-bar query at 100-119 -- every added bar
        after the gesture.  The window now straddles it instead.
        """
        lo, hi = snap_to_grid(100, 105, self.L, self.N)
        assert (lo, hi) == (92, 112)
        assert lo < 100 and hi > 105, "the query must extend on both sides"

    def test_padding_differs_by_at_most_one_bar(self):
        """Odd padding cannot split evenly; one bar of slack is the honest outcome.

        A 5-bar selection inside a 20-bar window leaves 15 bars of padding, which
        splits 7/8.  The point is that neither side is zero -- which is exactly what
        the old forward-only snap produced.
        """
        lo, hi = snap_to_grid(100, 105, self.L, self.N)
        before, after = 100 - lo, hi - 105
        assert before >= 7 and after >= 7, (before, after)
        assert abs(before - after) <= 1, (before, after)

    def test_padding_is_split_evenly(self):
        lo, hi = snap_to_grid(100, 110, self.L, self.N)
        assert (lo, hi) == (95, 115)
        assert 100 - lo == 5 and hi - 110 == 5

    def test_selection_shorter_than_the_query_is_always_inside_it(self):
        """No matter how short, the gesture is part of the query.

        Scoped to selections *up to* the window length: a longer selection cannot be
        contained by a fixed-length query, and is trimmed instead (see the test below).
        """
        for start in range(0, self.N, 97):
            for width in (1, 2, 3, 9, self.L - 1, self.L):
                lo, hi = snap_to_grid(start, start + width, self.L, self.N)
                assert lo <= start and start + width <= hi, (start, width, lo, hi)

    def test_selection_longer_than_the_query_is_trimmed(self):
        """A 21-bar gesture in a 20-bar query cannot be contained; the window wins.

        Centring still applies -- the query sits on the middle of the gesture -- but
        the ends fall outside it, which is the only honest outcome when the selection
        is over-long.
        """
        lo, hi = snap_to_grid(100, 121, self.L, self.N)
        assert hi - lo == self.L
        assert lo <= 110 and hi >= 111, "the centre of the gesture must survive"
        lo, hi = snap_to_grid(100, 140, self.L, self.N)
        assert (lo, hi) == (110, 130)

    def test_exact_length_is_always_returned(self):
        for start in range(0, self.N, 37):
            for width in (1, 2, 19, 20, 21, 60):
                lo, hi = snap_to_grid(start, start + width, self.L, self.N)
                assert hi - lo == self.L, (start, width, lo, hi)

    def test_never_leaves_the_archive(self):
        for start in (0, 1, 5, self.N - 1, self.N - self.L, self.N // 2):
            for width in (1, self.L, 100):
                lo, hi = snap_to_grid(start, start + width, self.L, self.N)
                assert 0 <= lo and hi <= self.N, (start, width, lo, hi)

    def test_edges_shift_inwards(self):
        """No tape on one side means the window moves; it never gains phantom bars."""
        selections = [
            (0, 5), (0, 20), (0, 30),                    # archive start
            (self.N - 5, self.N), (self.N - 20, self.N),  # archive end
            (self.N - 30, self.N),
        ]
        for lo_in, hi_in in selections:
            lo, hi = snap_to_grid(lo_in, hi_in, self.L, self.N)
            assert hi - lo == self.L, (lo_in, hi_in, lo, hi)
            assert 0 <= lo and hi <= self.N, (lo_in, hi_in, lo, hi)

    def test_selection_past_the_end_is_clamped(self):
        lo, hi = snap_to_grid(8019, self.N, self.L, self.N)
        assert (lo, hi) == (8024, 8044), "an over-long tail must not shift the window"
        assert hi <= self.N

    @pytest.mark.parametrize("selection", [(50, 50), (50, 10), (-5, 5), (0, 0)])
    def test_degenerate_input_still_yields_a_valid_window(self, selection):
        """A zero-width or reversed span must not raise or return junk."""
        lo, hi = snap_to_grid(*selection, self.L, self.N)
        assert hi - lo == self.L
        assert 0 <= lo and hi <= self.N

    def test_result_always_overlaps_the_selection(self):
        """The query must contain the gesture, else the brush is simply ignored."""
        for start in range(0, self.N, 53):
            for width in (1, 4, 19, 20, 33):
                lo, hi = snap_to_grid(start, start + width, self.L, self.N)
                assert start < hi and lo < start + width, (start, width, lo, hi)

class TestBrushBarCount:
    """The readout answers "how many bars did *I* draw", not "how long is the query".

    The query is always exactly ``pipe.length`` bars by construction, so it can never
    report the selection size.  The raw brush is the only place the number the reader
    actually chose is still visible -- and it is the number that tells them whether
    the box came out where they meant it to.
    """

    def test_count_is_inclusive_of_both_ends(self):
        """The off-by-one that matters: 100-104 is five bars, not four.

        A drag from one bar to another touches every bar between them, so a
        non-inclusive count would under-report every brush by exactly one.
        """
        assert brush_bar_count(event(box=box(100, 104)), SCALE) == 5
        assert brush_bar_count(event(box=box(100, 105)), SCALE) == 6

    def test_single_bar_span(self):
        assert brush_bar_count(event(box=box(100, 101)), SCALE) == 2

    @pytest.mark.parametrize("width, want", [
        (2, 2), (3, 3), (10, 10), (19, 19), (20, 20), (21, 21), (60, 60),
    ])
    def test_arbitrary_widths(self, width, want):
        assert brush_bar_count(
            event(box=box(100, 100 + width - 1)), SCALE) == want

    def test_count_always_matches_the_span(self):
        """The readout and the span it describes must never disagree.

        Both go through ``selection_to_span``, so a brush that yields a span always
        has exactly one count -- there is no second parsing path to drift.
        """
        for start in range(0, SCALE, 97):
            for width in (2, 3, 10, 19, 20, 33):
                e = event(box=box(start, start + width - 1))
                span = selection_to_span(e, SCALE)
                count = brush_bar_count(e, SCALE)
                assert span is not None, (start, width)
                assert count == span[1] - span[0] + 1, (start, width, count, span)

    def test_lasso_is_counted_by_extent(self):
        e = event(lasso=lasso(300, 100, 400, 200))
        assert brush_bar_count(e, SCALE) == 301

    def test_clamped_brush_counts_the_clamped_span(self):
        """A box dragged past the edge reports what is actually there, not the drag."""
        assert brush_bar_count(event(box=box(-50, 100)), SCALE) == 101
        assert brush_bar_count(event(box=box(500, 99999)), SCALE) == SCALE - 500

    @pytest.mark.parametrize("payload", [
        None, {}, {"selection": {}}, {"selection": {"box": []}},
    ])
    def test_no_brush_reports_nothing(self, payload):
        """Absent brush must be ``None`` -- never a stale number from a previous drag."""
        assert brush_bar_count(payload, SCALE) is None
        assert selection_summary(payload, STAMPS, SCALE) is None

    def test_degenerate_single_bar_brush_reports_nothing(self):
        """Same reason: there is no window, so there is no count to show."""
        e = event(box=box(7, 7))
        assert brush_bar_count(e, SCALE) is None
        assert selection_summary(e, STAMPS, SCALE) is None

    def test_summary_contains_the_count(self):
        text = selection_summary(event(box=box(100, 104)), STAMPS, SCALE)
        assert text.startswith("5 bars · "), text

    def test_summary_is_singular_for_one_bar(self):
        text = selection_summary(event(box=box(100, 101)), STAMPS, SCALE)
        assert text.startswith("2 bars · "), text

    def test_summary_reports_real_timestamps(self):
        """The stamps are the brushed bars' own, at the resolution's precision.

        Asserted through ``stamp_label`` rather than by re-deriving ``str(ts)[:19]``:
        the summary formats its stamps through the same helper as the chart axes, and
        a test that hand-formatted the expected value would keep asserting *second*
        precision on a chart that deliberately prints to the minute -- and would then
        pass or fail for reasons unrelated to which bars it reported.
        """
        text = selection_summary(event(box=box(100, 104)), STAMPS, SCALE)
        assert stamp_label(STAMPS[100]) in text
        assert stamp_label(STAMPS[104]) in text
        # And the ends of the brush, not some interior pair of bars.
        assert stamp_label(STAMPS[104]) not in text[:len("5 bars · ")]

    def test_summary_names_the_zone_intraday_and_omits_it_daily(self):
        """``(UTC)`` is a claim about the stamp, so it is conditional with it.

        A daily stamp is an Eastern date rather than a UTC instant, so a summary
        reading ``5 bars · 2026-09-01 → 2026-09-08 (UTC)`` names a zone the dates are
        not in -- the §BZ failure at the width of one caption.
        """
        with _resolution("1m"):
            assert selection_summary(
                event(box=box(100, 104)), STAMPS, SCALE).endswith("(UTC)")
        with _resolution("1d"):
            assert not selection_summary(
                event(box=box(100, 104)), STAMPS, SCALE).endswith("(UTC)")


class TestVariableLength:
    """A brush of N bars must be searched as N bars -- no grid, no padding.

    This replaced a fixed 20-bar grid.  The app used to snap every selection onto that
    one length because `find_matches` was assumed to need a single window size.  It
    does not: it reads `query.length` and scores the query's own slice of the series.
    Snapping therefore discarded the only thing the reader actually chose.

    The danger in getting this half-right is silent rather than loud.  `Pipeline.run`
    used to read `pipe.length` for the forward returns, the random-window baseline and
    the bootstrap block, so a 12-bar query would have been *matched* at 12 bars and
    then *measured* as if it were 20 -- anchoring every forward return 8 bars too early
    and building the control group from 20-bar windows.  Nothing would have raised;
    the numbers would simply have been about a different window than the one drawn.
    """

    DEFAULT = 20

    def resolve(self, span, n):
        """The app's real decision -- not a re-implementation of it.

        An earlier version of this class mirrored the three lines in ``main()`` and
        then asserted against the mirror.  Mutation testing showed that proves nothing:
        reverting the app to a fixed grid left 83 of 84 tests green, because the mirror
        still passed.  The decision now lives in ``app.resolve_query_window`` and is
        called here directly, so these tests fail when the app's behaviour changes.

        ``start``/``end`` only matter on the unbrushed path, where the window is the
        default length laid over the end of the archive.  ``main()`` passes
        ``start_idx = n - length`` and ``end_idx = n`` there, so that is what is
        mirrored -- an *exclusive* end one past the last bar of the default window.
        """
        return resolve_query_window(span, n - self.DEFAULT, n, self.DEFAULT, n)

    @pytest.mark.parametrize("width", [12, 13, 20, 37, 50, 100, 240])
    def test_brush_length_is_used_verbatim(self, width):
        span = (100, 100 + width - 1)
        lo, hi, length = self.resolve(span, SCALE)
        assert length == width, (width, length)
        assert (lo, hi) == (100, 100 + width), "the window moved off the selection"

    def test_no_padding_on_either_side(self):
        """The regression that mattered: 12 bars in, 12 bars searched."""
        lo, hi, length = self.resolve((100, 111), SCALE)
        assert length == 12
        assert lo == 100 and hi == 112, "bars were added outside the selection"

    def test_short_brush_is_raised_to_the_floor(self):
        """Below the floor the brush is widened -- and only upwards."""
        lo, hi, length = self.resolve((100, 104), SCALE)
        assert length == MIN_QUERY_BARS
        assert lo <= 100 and hi >= 105, "the selection must stay inside the query"

    def test_no_brush_falls_back_to_the_default(self):
        lo, hi, length = self.resolve(None, SCALE)
        assert length == self.DEFAULT
        assert (lo, hi) == (SCALE - self.DEFAULT, SCALE), (
            "the unbrushed default is the most recent full window"
        )

    def test_unbrushed_path_is_exact_not_centred(self):
        """A full-length span must come back unchanged, not re-centred.

        The default window is already exactly the requested length, so the centring in
        ``snap_to_grid`` has nothing to pad and must be a no-op.  If it is not, the
        *Latest window* query silently slides somewhere else.
        """
        lo, hi, length = self.resolve(None, SCALE)
        assert length == self.DEFAULT == hi - lo
        assert lo == SCALE - self.DEFAULT

    def test_window_never_leaves_the_archive(self):
        for span in [(0, 3), (0, 500), (SCALE - 2, SCALE - 1), (SCALE - 1, SCALE - 1)]:
            lo, hi, length = self.resolve(span, SCALE)
            assert 0 <= lo and hi <= SCALE, (span, lo, hi)
            assert hi - lo == length

    def test_count_and_query_length_agree_above_the_floor(self):
        """The readout and the searched length are the same number."""
        for width in (12, 20, 37, 100):
            e = event(box=box(100, 100 + width - 1))
            span = selection_to_span(e, SCALE)
            count = brush_bar_count(e, SCALE)
            _, _, length = self.resolve(span, SCALE)
            assert length == count, (width, length, count)

    def test_main_uses_this_function(self):
        """``main()`` must call the function rather than re-deriving the answer.

        Without this, the function could be correct while the app kept its own inline
        copy -- and these tests would be testing dead code.
        """
        source = app_source()
        assert "resolve_query_window(" in source, (
            "main() no longer routes the window decision through resolve_query_window"
        )
        call = re.search(r"start_idx, stop_idx, length = resolve_query_window\(", source)
        assert call, "main() does not unpack start/stop/length from resolve_query_window"

    def test_pipeline_run_keys_on_the_query_length(self):
        """Each run signature must name the length searched, not the default.

        Keying on ``pipe.length`` would let two brushes over the same span with
        different lengths collide in the cache.  The leading component is the
        *symbol*: the app no longer offers an archive picker, so a fetched ticker
        is the only source of bars, and it still has to be in the key -- a bare
        length would let the previous ticker's result be served against the new
        ticker's chart.

        The second element is compared for *equality*, not containment.  A
        ``"length" in "pipe.length"`` check passes for exactly the wrong value:
        ``pipe.length`` contains the substring ``length``, so the assertion this
        replaced was satisfied by the bug it was written to catch, and reverting the
        app to the default length left the suite green.

        **Both signatures are checked.**  There are two independent searches now, so
        "the run signature" is ambiguous and this assertion would silently cover only
        whichever came first in the file.
        """
        source = app_source()
        # Anchored on ``cfg["k"]`` because the backtest tab has its own unrelated
        # ``signature = (...)`` a few thousand lines earlier.
        sigs = re.findall(r"signature = \(([^,]+), ([^,]+), cfg[^\[]*\[\"k\"\]", source)
        assert len(sigs) == 2, (
            "expected one run signature per search tab, found %d" % len(sigs)
        )
        for symbol_expr, length_expr in sigs:
            assert symbol_expr.strip() == "symbol", (
                f"run signature should lead with the symbol, got {symbol_expr!r}"
            )
            # The Matches tab has ``length`` in scope from ``resolve_query_window``;
            # the Forecast tab works from a ``(start, stop)`` window, so it spells the
            # same quantity out.  Either is correct; ``pipe.length`` never is.
            assert length_expr.strip() in ("length", "stop - start"), (
                f"run signature must key on the searched length; got {length_expr!r}"
            )


class TestBrushTriggersASearch:
    """A brush must update its own tab's search, but only when it is a *new* brush.

    Streamlit keeps a selection in session_state for the page's lifetime, so
    ``price_brush`` stays non-``None`` on every later rerun -- including ones caused by
    an unrelated slider.  Auto-running whenever a brush is merely present would
    recompute the forecast behind the user's back, which is the stale-vs-fresh
    confusion the manual *Run match* button exists to prevent.  Each tab therefore
    records the span it last acted on and only treats a *different* span as a new
    brush.

    These are structural assertions on ``app.py`` for the same reason the rest of this
    file is: the script is not importable, and the decision is three lines of control
    flow whose failure mode is a silent, expensive, wrong answer rather than a crash.

    **Both tabs are checked, and they are checked for being separate.**  Before the
    decoupling there was one query, one signature and one ``run`` in ``main()``, so a
    single set of assertions covered the app.  There are now two independent searches,
    each with its own applied-span key, and a brush on one tab must be incapable of
    re-aiming the other -- which is only guaranteed if the keys differ.
    """

    @staticmethod
    def _block(name):
        return app_text(r"def %s\(.*?\n(?=\ndef |\ndef )" % name, name)

    def test_a_brush_triggers_a_run(self):
        # The Matches tab: the gate pairs the button with its own applied-span key.
        matches = app_source()
        assert "brush_moved" in matches, (
            "nothing reacts to the brush; the Matches tab will not follow it"
        )
        trigger = re.search(r'if cfg_m\["run_clicked"\] or brush_moved:', matches)
        assert trigger, "the Matches run gate not found"
        assert "brush_moved" in trigger.group(0)

        # The Forecast tab: same shape, its own key.
        forecast = app_text(r"def _forecast_run\(.*?\n(?=\ndef )", "_forecast_run")
        trigger = re.search(r'if cfg\["run_clicked"\] or brush_moved:', forecast)
        assert trigger, "the Forecast run gate not found"
        assert "brush_moved" in trigger.group(0)

    def test_brush_is_compared_against_the_last_applied_one(self):
        """The guard is the difference between *new* and *present*.

        Keyed on the *resolved* span rather than on the raw brush, so any input that
        actually moves the window fires once -- a brush, or a session change, which is
        just as explicit a gesture.  Keying on the brush alone left the session dropdown
        silently re-labelling the chart while the Matches and Forecast tabs went on
        reporting the previous session's numbers.
        """
        source = app_source()
        for base, fn in (("PRICE_APPLIED_KEY", "price_applied_key"),
                         ("FORECAST_APPLIED_KEY", "forecast_applied_key")):
            assert "st.session_state.get({}())".format(fn) in source, (
                "%s is not compared against the last applied span, so a brush will "
                "re-fire on every rerun instead of once per gesture" % base
            )
            assert "st.session_state[{0}()] = ".format(fn) in source, (
                "%s is never recorded, so no change can ever count as new" % base
            )

    def test_the_two_searches_are_separate(self):
        """A brush on one tab must not re-aim the other tab's search.

        The decoupling's whole point.  Sharing an applied-span key or a run key would
        mean a reader brushing on Forecast silently invalidated the Matches search --
        or, worse, served the Matches result against the Forecast window.
        """
        import re as _re
        declared = dict(_re.findall(
            r'^(PRICE_RUN_KEY|PRICE_APPLIED_KEY|FORECAST_RUN_KEY|FORECAST_APPLIED_KEY)'
            r'\s*=\s*"([^"]+)"', app_source(), _re.M))
        assert len(set(declared.values())) == len(declared), (
            "the two searches share a session key, so one tab can serve the other's "
            "result: %r" % declared
        )

    def test_comparison_happens_after_the_snap(self):
        """The recorded span must be the *resolved* one, or a brush re-fires forever.

        A brush defines its own length rather than snapping to one, but it is still
        clamped (to the archive, and to ``MIN_QUERY_BARS`` at the floor).  Recording
        the raw brush instead of the resolved span would make the two disagree on
        every rerun and re-trigger the search each time.
        """
        source = app_source()
        resolve = source.index("# ---------------- Use the selection's own length")
        guard = source.index("st.session_state.get(price_applied_key())")
        assert guard > resolve, (
            "the query is compared before its length is resolved, so a clamped brush "
            "never matches the recorded span and re-fires on every rerun"
        )

    def test_absent_brush_does_not_trigger(self):
        """A rerun caused by an unrelated control must not re-run the search."""
        source = app_source()
        # Was ``price_brush is not None or session_bounds is not None``: the search
        # only auto-fired when an explicit input had been set.  The session dropdown is
        # gone, so a brush is now the only such input -- and the guard is still there,
        # which is the point: without it every unrelated rerun would recompute the
        # forecast, which is the stale-and-fresh confusion the button exists to avoid.
        assert ('brush_moved = (st.session_state.get(price_applied_key()) '
                '!= (start_idx, stop_idx)') in source, (
            "the guard does not require an explicit brush to have been set"
        )
        assert "and price_brush is not None" in source, (
            "the brush guard must also require that a brush is actually in force"
        )

    def test_no_dead_run_output_state(self):
        """``run_output`` was written and never read; it must not creep back.

        A stored copy of the result is exactly the thing that can disagree with the
        query on screen, so the app keeps only the signature and recomputes.
        """
        reads = len(re.findall(r'st\.session_state\.get\(\s*"run_output"', app_source()))
        assert reads == 0, (
            "run_output is read back; a stored result can go stale against the query"
        )


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def et_stamps(days, *, bars=390, start="2026-01-05"):
    """``days`` consecutive ET sessions as a UTC timestamp Series.

    Built in Eastern time and converted, rather than by adding 24h to a UTC start,
    because the whole point of the code under test is that a session is an *Eastern*
    day.  Constructing UTC offsets by hand would bake the answer into the fixture.

    The open is 09:30 ET, so in January (EST) the first bar is 14:30 UTC -- on the
    *previous* UTC date.  That is the trap :func:`session_spans` has to avoid, and the
    tests below assert against it directly.
    """
    open_et = pd.Timedelta(hours=9, minutes=30)   # Timedelta wants hh:mm:ss
    out = []
    day0 = pd.Timestamp(start, tz=EASTERN)
    for d in range(days):
        open_at = (day0 + pd.Timedelta(days=d)).normalize() + open_et
        out.append(pd.date_range(open_at, periods=bars, freq="1min")
                   .tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(np.concatenate(out)))


class TestSessionSpans:
    """``session_spans`` must return *positions*, grouped by Eastern day.

    Both properties are bugs when wrong, and neither raises.  Grouping by UTC date
    files a 09:30 winter open (14:30 UTC) under the previous day and splits every
    session in half; returning index *labels* instead of positions feeds
    ``bars.iloc[label]`` a silently different bar, because ``finalize_features``
    drops the feature warm-up rows without resetting the index.
    """

    def test_sessions_are_eastern_days_not_utc_days(self):
        """A 09:30 ET open in winter is 14:30 UTC -- the previous UTC date."""
        stamps = et_stamps(2)
        # The first bar is 14:30 UTC, so a UTC bucket would call it the day before.
        assert stamps.iloc[0].tz_convert(EASTERN).hour == 9
        assert stamps.iloc[0].hour == 14
        sessions = session_spans(stamps)
        assert [d for d, _, _ in sessions] == ["2026-01-05", "2026-01-06"]
        for day, _, _ in sessions:
            assert pd.Timestamp(day).day == (5 if day.endswith("05") else 6)

    def test_every_session_is_contiguous_and_non_empty(self):
        stamps = et_stamps(4)
        sessions = session_spans(stamps)
        assert len(sessions) == 4
        assert sessions[0][1] == 0, "the first session must start at bar 0"
        assert sessions[-1][2] == len(stamps), "the last must run to the final bar"
        for (_day, a, b), (_next_day, na, _nb) in zip(sessions, sessions[1:]):
            assert b == na, "sessions must tile the frame with no gap or overlap"
            assert b > a, "a session with no bars would give length=0 downstream"

    def test_bounds_are_positions_not_index_labels(self):
        """The returned numbers must work on a frame whose index is not 0-based.

        ``Pipeline.from_frame`` drops the ~20-bar feature warm-up, so ``pipe.bars``
        starts at label 20 and its later labels drift further from its positions.  A
        helper that returned labels would feed ``bars.iloc[label]`` a silently
        different bar -- ``iloc`` does not check, and the error surfaces as a chart of
        the wrong bars rather than as an exception.

        The first session is where this bites hardest: it is short by exactly the
        warm-up, and every bar in it has ``label == position + 20``.
        """
        raw = et_stamps(3)
        # Reproduce the real frame: index preserved, leading rows dropped.
        trimmed = raw.iloc[20:].copy()
        assert trimmed.index[0] == 20, "fixture must not be 0-based"
        assert list(trimmed.index[:40]) != list(range(40)), "labels must differ"

        sessions = session_spans(trimmed)
        assert sessions, "no sessions found"

        # Read every returned bound positionally and land on the right clock time.
        for _day, a, b in sessions:
            assert trimmed.iloc[a].tz_convert(EASTERN).hour == 9, (
                "a session must open at the ET open, positionally"
            )
            assert trimmed.iloc[b - 1].tz_convert(EASTERN).minute in (59, 0)
            assert b > a

        # The label reading would be wrong, and provably so: the first session is 370
        # bars (390 less the 20-bar warm-up), so reading its start as a label lands
        # 20 bars into the tape rather than at the open.
        _day, a, _b = sessions[0]
        assert trimmed.iloc[a].tz_convert(EASTERN).hour == 9
        assert trimmed.index[a] == a + 20, "the label and the position must differ"

    def test_empty_input(self):
        assert session_spans(pd.Series([], dtype="datetime64[ns, UTC]")) == []

    def test_single_session(self):
        assert len(session_spans(et_stamps(1))) == 1


class TestSessionFence:
    """A brush may span sessions: the window is exactly what the reader drew.

    ``resolve_query_window`` used to take a ``bounds`` span -- one Eastern trading day
    -- and confine the window to it, capping an over-wide brush at the day's width and
    shifting an overhanging one inward.  Both helpers that computed it were deleted too.
    PLAN.md §BX has the full reasoning; the short version is that the fence did not
    keep the gap-jump out of the picture, it just moved the window without being asked,
    while the half that corrupts a *measurement* (a candidate whose forward horizon
    crosses a close) is still censored centrally by §BX.

    So the archive is now the only bound, and these tests pin that: a brush crossing a
    close keeps the width *and* position it was drawn at.  ``N`` is five 390-bar
    sessions, so a brush wider than ``N/2`` spans more than one of them by construction.
    """

    N = 1950

    def resolve(self, span, n=None):
        return resolve_query_window(span, 100, 200, 20, n or self.N)

    def test_a_query_inside_one_session_is_unchanged(self):
        got = self.resolve((900, 999))
        assert got == (900, 1000, 100), got

    def test_a_narrow_brush_keeps_its_own_length(self):
        """A brush defines its own length; nothing snaps it to a day's width."""
        got = self.resolve((900, 939))
        assert got[2] == 40, "a 40-bar brush must stay 40 bars"
        assert got[:2] == (900, 940)

    def test_a_brush_crossing_a_session_boundary_is_not_moved(self):
        """The core inversion: spanning a close keeps both width and position.

        70 bars from bar 700 reaches 100 bars past the third session's close.  Under
        the fence this became ``(780, 941)`` -- shifted 80 bars right of where it was
        drawn.
        """
        got = self.resolve((700, 769))
        assert got == (700, 770, 70), got

    def test_a_brush_wider_than_a_session_is_not_capped(self):
        """501 bars used to become 390 -- the widest single session.

        Capping there silently replaced "the 501 bars I drew" with 390 bars of a
        different day, the one outcome the reader could neither ask for nor see.
        """
        lo, hi, length = self.resolve((800, 1300))
        assert length == 501, "the brush must not be capped at a session's width"
        assert (lo, hi) == (800, 1301)

    def test_a_brush_may_span_the_whole_archive(self):
        """Both edges are the archive's; nothing interior may narrow it."""
        lo, hi, length = self.resolve((0, self.N - 1))
        assert (lo, hi, length) == (0, self.N, self.N)

    def test_the_window_still_never_leaves_the_archive(self):
        """The archive *is* still a fence -- it is just no longer the only one."""
        for span in [(0, 5), (0, 1900), (1949, 1949), (760, 1200), (900, 900)]:
            lo, hi, _ = self.resolve(span)
            assert 0 <= lo and hi <= self.N, (span, lo, hi)

    def test_a_brush_running_past_the_end_is_shifted_in_at_full_length(self):
        """The one remaining shift, and it keeps the length rather than clipping (§Z1).

        A 40-bar brush ending on the archive's final bar would overhang by one.  It
        moves left by that one, so the forward return is still measured from the bar
        the reader pointed at.
        """
        lo, hi, length = self.resolve((self.N - 40, self.N - 1))
        assert length == 40, "the length must survive the shift"
        assert (lo, hi) == (self.N - 40, self.N)

    def test_a_brush_longer_than_the_archive_yields_the_whole_archive(self):
        lo, hi, length = self.resolve((0, self.N + 500), n=self.N)
        assert (lo, hi, length) == (0, self.N, self.N)

    def test_the_minimum_length_still_applies(self):
        """The floor is a validity bound and is unrelated to sessions."""
        assert self.resolve((100, 104))[2] == MIN_QUERY_BARS
        lo, hi, length = self.resolve((900, 902))
        assert length == MIN_QUERY_BARS
        assert lo <= 900 and hi >= 903

    def test_a_degenerate_brush_still_yields_a_positive_length(self):
        """A zero-width span must not produce a query the matcher cannot score."""
        for bad in [(100, 99), (500, 500), (1949, 1948)]:
            lo, hi, length = self.resolve(bad)
            assert length > 0 and hi <= self.N, (bad, lo, hi, length)

    def test_main_does_not_fence_the_query_to_one_session(self):
        """The query is placed where the reader drew it, across closes included.

        This test has been inverted twice before, so the reasoning is worth one
        paragraph.  A fence was first asserted (driven by a *dropdown*, since removed),
        then denied, then asserted again as *inferred* from the brushed bar.  That last
        version was right about the risk -- a gap-jump is not a trade -- and wrong about
        the remedy: confining the window did not keep the gap out, it just relocated
        every overhanging brush and capped every brush wider than a session, so the
        band stopped being what the reader drew.  The part that actually corrupts a
        *measurement* is a candidate whose **forward horizon** crosses a close (~20x
        inflation), which is handled by §BX and pinned in ``test_matching.py``.

        So this asserts the fence stays gone -- a deliberate choice, not an oversight.
        """
        source = app_source()

        # Scoped to the call, not the whole file: ``pan_bounds=`` is an unrelated
        # argument of the pannable match panel and must not trip this.
        assert not re.search(r"resolve_query_window\((?:[^()]|\([^()]*\))*bounds=",
                             source, re.S), (
            "resolve_query_window is being passed bounds again, so an over-wide or "
            "overhanging brush is silently moved or capped instead of being drawn"
        )

        # The helpers that computed the fence went with it.  Checked against the AST, not
        # the text: a comment explaining *why* the fence is gone necessarily spells
        # both names out, and a grep would fail a correct ``app.py`` -- exactly the
        # failure mode ``app_called_names`` exists to avoid.
        called = app_called_names()
        for gone, why in (
            ("session_for_index", "the query is being fenced to a session"),
            ("latest_session_window", "the default is drawn from inside one session"),
        ):
            assert gone not in called, "%s is called again; %s" % (gone, why)
            # Also not *defined* again, which the call check alone would miss for a
            # helper re-added but not yet wired up.
            assert gone not in app_defined_names(), (
                "%s is defined again; with no caller it is dead code, and with one "
                "the fence is back" % gone
            )

    def test_the_default_query_is_the_archive_s_last_bars(self):
        """Unbrushed, the app opens on the most recent ``pipe.length`` bars.

        It used to be the newest *session* wide enough, found by walking backwards
        through older days -- which existed only to keep the default inside one
        trading day.  With the fence gone there is nothing to walk back for, so the
        default is simply the newest tape, and it normally does cross the final close.
        """
        source = app_source()
        assert "start_idx, end_idx = max(0, n - pipe.length), n" in source, (
            "the default query is not the archive's last pipe.length bars"
        )
        # The session-walking default must not survive anywhere.
        assert not re.search(r"^start_idx = n - pipe\.length$", source, re.M), (
            "the default is written as a bare subtraction, which returns a *start* "
            "and would leave end_idx unset"
        )

    def test_the_days_slider_and_the_session_dropdown_are_both_gone(self):
        """Neither of the two view controls that came before this is back.

        The slider counted *calendar* days, so its maximum was a range that could not
        be displayed, and its "1 day" was ``n // days`` -- an average over those
        inflated days, not any real session.  The dropdown replaced it, and was then
        removed along with the *Chart* tab that hosted it.  What is left is the fixed
        ``DEFAULT_VIEW_DAYS`` trailing window, which needs no control at all.
        """
        source = app_source()
        assert '"Days of history to display"' not in source, "the days slider is back"
        assert 'st.slider(\n        "Days of history' not in source
        assert "days_view" not in source, (
            "days_view is read or written but the slider is gone"
        )
        assert "pending_session_key" not in source, (
            "the session dropdown is back; there is no session to select"
        )
        assert 'st.selectbox(\n        "Trading session"' not in source

    def test_the_view_is_a_fixed_number_of_real_sessions(self):
        """A fixed number of real sessions, not calendar days, and not a control.

        Counting *real* sessions is what makes the fixed span correct: ``n //
        view_sessions`` would be an average over a span that includes the closed
        market overnight, so "one day" would not be any session.  Anchoring on
        ``session_spans`` keeps the view a real, contiguous span of bars.

        **The count is now resolved per resolution** (``active_view_sessions()``:
        5 sessions of 1-minute, 250 of daily) rather than being the ``DEFAULT_VIEW_DAYS``
        literal, because five daily candles is not a view -- it is a flicker.  The
        invariant this test protects is unchanged and is the part that actually
        matters: the span is counted in *sessions*, anchored on ``session_spans``, and
        is not a reader control.
        """
        source = app_source()
        assert re.search(r"^DEFAULT_VIEW_DAYS = \d+$", source, re.M), (
            "DEFAULT_VIEW_DAYS must stay a plain integer constant"
        )
        assert "def active_view_sessions()" in source, (
            "the view size must be resolved per resolution, not read from one literal"
        )
        assert "chart_from = sessions[-active_view_sessions()][1] if sessions else 0" in source, (
            "chart_from must be derived from the last view_sessions sessions"
        )
        assert 'st.slider(\n        "Days of history' not in source

    def test_the_brush_is_the_only_window_control(self):
        """No picker may choose the window -- a gesture is the only way in.

        The *Chart* tab used to hold the session dropdown and the From/To pickers, and
        the Price tab's brush was one route among three.  With that tab gone the brush
        is the app's only window input, so a regression that drops ``selectable=True``
        would leave the reader with no way to choose a query at all.

        **The Forecast tab now carries a second brush, and the rule still holds.**
        What this test protects is that the window is chosen by *gesturing* rather than
        by operating a control -- no dropdown, no slider, no datetime picker can end up
        naming the window.  The Forecast tab's brush (``FORECAST_BRUSH_KEY``) answers a
        different question from the Price tab's (which window to forecast, versus which
        window to match for the evidence table) and selects a different chart, but it
        is the same mechanism, not a reintroduced selector.  The assertion below is
        therefore deliberately about *controls*, not about the number of charts that
        accept a brush.

        Adding a widget -- a "Forecast window" slider, a session dropdown -- is the
        thing that would break the design, and the ``datetime_input`` count catches
        the most likely form of it.
        """
        source = app_source()
        assert "selectable=True" in source and 'on_select="rerun"' in source
        assert source.count("datetime_input(") == 0, (
            "a datetime picker is back; there is no longer a place to draw one"
        )
        # Both brushes must be distinct keys.  Sharing one would mean a brush on the
        # Price tab silently re-aims the Forecast tab's selection, so a reader
        # comparing two windows could only ever have one selected at a time.
        declared = dict(re.findall(r'^(PRICE_BRUSH_KEY|FORECAST_BRUSH_KEY)\s*=\s*"([^"]+)"',
                                   source, re.M))
        assert set(declared) == {"PRICE_BRUSH_KEY", "FORECAST_BRUSH_KEY"}, (
            "both brushes must declare a key: %r" % declared
        )
        assert declared["PRICE_BRUSH_KEY"] != declared["FORECAST_BRUSH_KEY"], (
            "the two brushes share a widget key, so one silently re-aims the other"
        )

    def test_every_session_helper_is_called_with_its_real_signature(self):
        """Guard against arity drift, which a linter cannot see.

        ``session_options`` lost its ``newest_stamp`` argument when completeness moved
        to the bar count, and the stale two-argument call in ``main()`` took the whole
        page down at import -- ``TypeError`` before any tab rendered.  ``pyflakes`` does
        not check call arity across a module, so nothing caught it and the browser
        smoke test was the only thing that did.

        Every call site of every helper this class pulls in is checked against the
        signature the app actually defines, so a future change to either fails here
        rather than in the running app.  The helper set has shrunk twice: once when
        the *Chart* tab took ``session_options`` and ``is_session_complete`` with it
        (they only ever built the dropdown's labels), and again when the session fence
        took ``session_for_index`` and ``latest_session_window`` with it.
        """
        tree = ast.parse(app_source())
        defined = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                defined[node.name] = node

        # Only the helpers this class exec's.  A sweep over every function in the file
        # would flag legitimate ``f(*view)`` splats elsewhere in the app, which are
        # correct code and would bury the one real error this guard exists to catch.
        #
        # The set shrank twice.  ``session_options`` and ``is_session_complete`` went
        # with the *Chart* tab, taking their call sites with them.  ``session_for_index``
        # and ``latest_session_window`` went with the session fence -- the first
        # computed the ``bounds`` span, the second chose the fenced default query --
        # and ``resolve_query_window`` lost its ``bounds`` parameter with them.  Four
        # names remain and four calls are inspected, so the floor below still holds.
        helpers = {"session_spans", "resolve_query_window", "snap_to_grid"}
        assert helpers <= set(defined), "a session helper is missing from app.py"

        calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name not in helpers:
                continue
            sig = defined[name].args           # an ast.arguments
            # ``*args`` makes any positional count legal; ``**kwargs`` any keyword.
            if sig.vararg is not None or sig.kwarg is not None:
                continue
            pos = [a.arg for a in sig.posonlyargs + sig.args]
            n_pos = len(pos)
            n_required = n_pos - len(list(sig.defaults))
            given = len(node.args)
            assert given <= n_pos, (
                f"{name}() called with {given} positional args but takes {n_pos}"
            )
            assert given >= n_required, (
                f"{name}() called with {given} positional args but needs "
                f"{n_required}"
            )
            known = set(pos) | {a.arg for a in sig.kwonlyargs}
            for kw in node.keywords:
                assert kw.arg in known, f"{name}() got unexpected keyword {kw.arg!r}"
            calls += 1
        assert calls >= 3, (
            f"only {calls} session-helper calls inspected -- the AST walk found too "
            "little to be worth guarding. If a helper lost its last caller, drop it "
            "from the set above rather than lowering this floor: a guard that inspects "
            "nothing passes while protecting nothing."
        )
