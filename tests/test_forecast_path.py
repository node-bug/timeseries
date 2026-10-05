"""Tests for the Forecast tab's projected *path* chart.

Split into two halves, because the thing being tested is split in two.

The **aggregation** (:func:`timeseries.forecast.forecast_paths`) is where the
arithmetic lives: which bar each match is anchored to, what happens to a match whose
forward history runs off the end of the archive, and whether the median and quartiles
are computed over the right set.  These are unit tests against hand-built input,
because the properties are exact — ``offsets[0]`` is ``0.0`` and not merely near it,
a dropped match reduces ``n_matches`` and nothing else — and an approximate fixture
could not pin any of them down.

The **figure and wiring** (:func:`build_forecast_path_figure` plus the ``tab_forecast``
block in ``main()``) are about a chart the reader has to be able to trust: that it
draws the history and the projection at the sizes it claims, that the two halves meet
at zero rather than a bar apart, and that it renders *without* a search run.  The last
one is the regression this file exists for: the tab used to short-circuit on
``out is None`` and replaced itself with a notice, which left the chart permanently
blank for anyone who had not pressed *Run match*.

Fixtures come from :mod:`tests.session_bars`, so the suite stays self-contained.
"""

from __future__ import annotations

# Module-level rather than function-local: several tests here read ``app.py`` as text
# and match on it, and a local ``import re`` inside one of them leaves the next one
# with a ``NameError`` that has nothing to do with what it is testing.
import ast
import re

import numpy as np
import plotly.graph_objects as go
import pytest
import streamlit as st  # noqa: F401  - needed to exec `forecast_path_for`'s decorator

from timeseries.forecast import ForecastPath, forecast_paths, forecast_paths_multi
from timeseries.pipeline import Pipeline

from apphelpers import app_block, app_source, load_app_functions
from session_bars import session_bars

#: Window length and projection both used by the app's chart.  Read out of ``app.py``
#: rather than restated, so a change to the constants cannot leave these tests
#: asserting against a shape the chart no longer draws.
#
#: ``forecast_path_for`` is pulled in too even though the figure tests do not call it,
#: so that a signature change to the cached helper surfaces here rather than as an
#: ``AttributeError`` inside ``main()``.
_ns = load_app_functions(
    {"build_forecast_path_figure", "forecast_path_for", "fill_tokens",
     "resolve_selection_window", "session_spans",
     "resolve_query_window", "selection_to_span", "chart_config",
     # Read by ``build_forecast_path_figure`` to draw the closest-match trace.
     # ``load_app_functions`` only ``exec``s the names asked for, so a helper the
     # figure calls has to be listed here or it is a ``NameError`` at draw time.
     "_closest_match_path", "_closest_distance_label"},
    namespace={
        "go": go,
        "st": st,
        # ``fill_tokens`` reads three globals that are *mutated at runtime* rather than
        # assigned, so ``_module_constants`` cannot see them (they are lists, not
        # literals).  Supplied here for the same reason a caller passes ``MIN_QUERY_BARS``
        # -- so the values come from this file, not from a restatement that can drift.
        "PIPE_LENGTH_FOR_HELP": [240],
        "SYMBOL_FOR_HELP": ["QQQ"],
        "ROLLING_WINDOW": 20,
        # ``resolve_selection_window`` delegates to these rather than re-parsing, which
        # is the point: a brush of N bars must not be counted two different ways.
        # Supplied because ``MIN_QUERY_BARS`` is annotated rather than bare.
        "forecast_paths": forecast_paths,
    },
)
build_forecast_path_figure = _ns["build_forecast_path_figure"]
HISTORY = _ns["FORECAST_HISTORY_BARS"]
PROJECTION = _ns["FORECAST_PROJECTION_BARS"]
MATCHES = _ns["FORECAST_PATH_MATCHES"]


# --------------------------------------------------------------------------- #
# Hand-built aggregation fixtures
# --------------------------------------------------------------------------- #
def _close(n=40, base=100.0, step=1.0):
    """A simple ascending close series, so a rebased path is easy to read by eye."""
    return base + step * np.arange(n, dtype=float)


def test_offset_zero_is_exactly_zero():
    """Offset 0 is the anchor bar itself, rebased against itself.

    It is the bar history and forecast share, and it is *exactly* 0.0 rather than
    approximately: a path drawn from these offsets has to meet the history line at
    zero with no visible step, and a rounding error of 1e-14 is harmless here but a
    path that started at offset 1 instead would leave a one-bar hole between the two
    halves of the chart.
    """
    path = forecast_paths(_close(), np.array([0, 5, 9]), 4, 3)
    assert path is not None
    assert path.offsets.tolist() == [0, 1, 2, 3]
    assert path.median[0] == 0.0
    assert path.q25[0] == 0.0
    assert path.q75[0] == 0.0


def test_anchored_on_the_last_bar_inside_the_window():
    """The anchor is ``start + L - 1`` — the §Z1 anchor, not the window's first bar.

    Anchoring at ``start`` would make the first "projected" bar a bar the pattern had
    already matched, which is the tautology §Z1 exists to prevent.  Measured here by
    checking the rebased path against a hand-computed slice.
    """
    close = _close()
    length, horizon = 4, 3
    start = 5
    anchor = start + length - 1          # == 8, the last bar inside the window
    expected = (close[anchor:anchor + horizon + 1] / close[anchor] - 1.0) * 100.0

    path = forecast_paths(close, np.array([start]), length, horizon)
    assert np.allclose(path.median, expected)
    assert path.starts.tolist() == [start]


def test_median_and_quartiles_match_a_hand_built_matrix():
    """The three bands are the 25th/50th/75th percentile of the per-match paths.

    Built by hand rather than recomputed with ``np.percentile`` over the same input,
    which would only prove the function calls itself.
    """
    close = np.concatenate([
        np.full(6, 100.0),          # match A: anchor 100, then 110, 120, 130
        np.full(6, 50.0),           # match B: anchor  50, then  55,  60,  65
        np.full(6, 10.0),           # match C: anchor  10, then   9,   8,   7
    ])
    length, horizon = 2, 3

    rows = [
        (close[a:a + horizon + 1] / close[a] - 1.0) * 100.0
        for a in (1, 7, 13)
    ]
    matrix = np.vstack(rows)
    expect_25, expect_50, expect_75 = np.percentile(matrix, [25, 50, 75], axis=0)

    path = forecast_paths(close, np.array([0, 6, 12]), length, horizon)
    assert np.allclose(path.q25, expect_25)
    assert np.allclose(path.median, expect_50)
    assert np.allclose(path.q75, expect_75)
    # And the ordering the chart's ``fill="tonexty"`` polygon depends on.
    assert np.all(path.q25 <= path.median)
    assert np.all(path.median <= path.q75)


def test_each_match_rebases_to_its_own_anchor():
    """Two matches ten times apart in price level produce an identical path.

    This is the property that makes aggregating across matches meaningful at all: the
    archive holds windows from many price regimes, and they are only comparable once
    each is expressed as a percentage move from where it ended.  Asserted as equality
    between the two matches rather than against a hand-computed figure, so the test is
    about the level-invariance itself and not about my arithmetic.
    """
    low = np.array([100.0, 100.0, 110.0, 120.0, 130.0])
    high = low * 10.0
    close = np.concatenate([low, high])

    path = forecast_paths(close, np.array([0, 5]), 2, 3)
    assert path.n_matches == 2
    # Both matches are identical up to a factor of ten in price, so after rebasing
    # their paths coincide exactly.  start=0, length=2 -> anchor=1, which is the
    # 100.0 bar; the projected bars are 110, 120, 130.
    assert path.median[0] == 0.0
    assert path.median[1] == pytest.approx((110.0 / 100.0 - 1.0) * 100.0)
    assert path.median[2] == pytest.approx((120.0 / 100.0 - 1.0) * 100.0)
    assert path.median[3] == pytest.approx((130.0 / 100.0 - 1.0) * 100.0)
    # A collapsed band is the observable signature of level-invariance: with the
    # levels not divided out, the 1000-scale match would sit ten times higher.
    assert path.q25 == pytest.approx(path.median)
    assert path.q75 == pytest.approx(path.median)


def test_match_without_a_full_forward_path_is_dropped_not_padded():
    """A window reaching past the last bar is dropped, and the rest still aggregate.

    Every archive ends somewhere, so the most recent matches are exactly the ones with
    no data after them.  Padding them with NaN would make ``np.percentile`` return NaN
    for the whole trailing region and silently shorten the projection; dropping them
    computes the same statistic over the set that can support it, and the shortfall
    is reported rather than hidden.
    """
    close = _close(n=20)
    length, horizon = 4, 10
    # start=12 -> anchor=15, needs bars up to 15+10 == 25, but the archive ends at 19.
    path = forecast_paths(close, np.array([0, 12]), length, horizon)
    assert path is not None
    assert path.n_matches == 1
    assert path.n_requested == 2
    assert path.starts.tolist() == [0]
    # Every projected bar is a real number, not a NaN tail.
    assert np.isfinite(path.median).all()
    assert np.isfinite(path.q25).all()
    assert np.isfinite(path.q75).all()


def test_none_when_no_match_has_a_complete_forward_path():
    close = _close(n=12)
    # start=8 -> anchor=11, needs 11 + 4 + 1 == 16 bars; the archive has 12.
    assert forecast_paths(close, np.array([8]), 4, 4) is None


def test_none_for_no_matches():
    assert forecast_paths(_close(), np.array([], dtype=np.int64), 4, 3) is None


def test_non_positive_anchor_close_drops_only_that_match():
    """One bad print cannot poison an aggregate computed over dozens of good ones.

    Dividing by a non-positive anchor has no meaningful result, so that match is
    dropped.  Zero rather than ``inf``: an ``inf`` would propagate straight through
    ``np.percentile`` and take every other match's contribution with it.
    """
    close = _close(n=20)
    # start=6, length=4 -> anchor = 6 + 3 = 9, which is the bad bar.
    close[9] = 0.0
    path = forecast_paths(close, np.array([0, 6]), 4, 3)
    assert path is not None
    assert path.n_matches == 1
    assert path.starts.tolist() == [0]
    assert np.isfinite(path.median).all()


def test_non_finite_close_inside_the_forward_path_drops_the_match():
    close = _close(n=20)
    # start=0, length=4 -> anchor=3, slice close[3:7] holds indices 3..6.
    close[5] = np.nan
    assert forecast_paths(close, np.array([0]), 4, 3) is None


def test_distances_are_kept_positionally_with_the_surviving_rows():
    """A dropped match consumed a slot, so distances must not shift onto a survivor.

    Pairing a surviving path with the wrong distance would mislabel it in the chart's
    caption — attributing a path to a window that did not produce it.
    """
    close = _close(n=20)
    # start=14, length=4 -> anchor=17, needs bars up to 17+3+1 == 21; the archive has
    # 20, so that match is dropped.  distances are [10.0, 20.0].
    path = forecast_paths(close, np.array([0, 14]), 4, 3,
                          distances=np.array([10.0, 20.0]))
    assert path.n_matches == 1
    assert path.starts.tolist() == [0]
    assert path.distances.tolist() == [10.0]


def test_mismatched_distances_are_ignored_rather_than_mispaired():
    """A wrong-length distance array must not silently misalign the pairing."""
    close = _close(n=20)
    path = forecast_paths(close, np.array([0, 5]), 4, 3,
                          distances=np.array([10.0]))
    assert path.n_matches == 2
    assert path.distances.size == 0


@pytest.mark.parametrize("length,horizon", [(0, 3), (-1, 3), (4, -1)])
def test_invalid_shapes_are_rejected(length, horizon):
    with pytest.raises(ValueError):
        forecast_paths(_close(), np.array([0]), length, horizon)


def test_two_dimensional_close_is_rejected():
    with pytest.raises(ValueError):
        forecast_paths(_close().reshape(2, -1), np.array([0]), 2, 2)


# --------------------------------------------------------------------------- #
# Multi-ticker aggregation -- the cross-sectional path
# --------------------------------------------------------------------------- #
# ``forecast_paths`` measures every match's forward bars in ONE close series, so it
# can only describe one ticker.  Pooling matches from several tickers into a single
# median is the panel's whole premise, and it is the step where a wrong index space or
# a mis-grouped start silently produces a curve that looks exactly like a real one.
# The tests below pin the three things that can go wrong quietly: attributing a
# window to the wrong ticker, letting one ticker's index space leak into another's,
# and reporting a partial distance array as though it described every match.
def test_offset_zero_is_still_exactly_zero_across_tickers():
    """Every match rebases to *its own* anchor, so offset 0 is 0 for all of them.

    This is what makes pooling across $4 and $700 names legitimate at all: without a
    per-match rebase the median would be dominated by whichever name has the largest
    nominal price, and "the projection" would really be "the share price".
    """
    path = forecast_paths_multi(
        {"A": _close(40), "B": _close(40, base=700.0, step=7.0)},
        {"A": np.array([0, 5]), "B": np.array([3])},
        4, 3,
    )
    assert path is not None
    assert path.median[0] == 0.0
    assert path.q25[0] == 0.0
    assert path.q75[0] == 0.0


def test_matches_are_measured_in_their_own_ticker_series():
    """The core property: A's window is read from A, not from whatever came last.

    Two tickers with identical *shapes* but different prices, matched at different
    offsets.  If the aggregator mixed the series -- reading B's window out of A, or
    indexing both with one flat array -- the resulting median would not equal the
    hand-computed rebased paths below.
    """
    a = np.concatenate([np.full(6, 100.0), np.array([110.0, 120.0, 130.0, 140.0])])
    # B is padded one bar longer so its window, anchored at ``start + L - 1 = 7``,
    # still has a full ``horizon + 1``-bar forward slice.  Same shape, different
    # price, so a mix-up cannot hide behind the values coinciding.
    b = np.concatenate([np.full(7, 50.0), np.array([40.0, 45.0, 55.0, 60.0])])
    starts = {"A": np.array([0]), "B": np.array([2])}
    path = forecast_paths_multi({"A": a, "B": b}, starts, 6, 3)
    assert path is not None

    # Hand-computed: anchor = start + L - 1, then 3 bars forward, rebased to 0%.
    exp_a = (a[5:9] / a[5] - 1.0) * 100.0
    exp_b = (b[7:11] / b[7] - 1.0) * 100.0
    stacked = np.stack([exp_a, exp_b])
    q25, median, q75 = np.percentile(stacked, (25.0, 50.0, 75.0), axis=0)
    np.testing.assert_allclose(path.median, median)
    np.testing.assert_allclose(path.q25, q25)
    np.testing.assert_allclose(path.q75, q75)


def test_the_single_and_multi_forms_agree_for_one_ticker():
    """One ticker through both functions must give byte-identical bands.

    The multi form is a refactor of the single one, not a reimplementation, so any
    disagreement is a bug in whichever moved.  This is the guard that keeps the two
    paths from drifting apart as either is edited.
    """
    close = _close(60)
    starts = np.array([0, 6, 12, 20])
    single = forecast_paths(close, starts, 5, 4)
    multi = forecast_paths_multi({"A": close}, {"A": starts}, 5, 4)
    assert single is not None and multi is not None
    np.testing.assert_allclose(multi.median, single.median)
    np.testing.assert_allclose(multi.q25, single.q25)
    np.testing.assert_allclose(multi.q75, single.q75)
    assert multi.n_matches == single.n_matches
    assert multi.n_requested == single.n_requested
    np.testing.assert_array_equal(multi.starts, single.starts)


def test_a_ticker_with_no_forward_history_contributes_nothing_but_does_not_fail():
    """A short ticker is ordinary, so it is dropped -- not fatal to the whole path.

    The panel holds whatever the archive has; one thin name must not cost the other
    tickers their paths.  Asserted by checking the surviving match came from the
    ticker that could support one.
    """
    long_c = _close(60)
    path = forecast_paths_multi(
        {"LONG": long_c, "SHORT": np.array([100.0, 101.0, 102.0])},
        {"LONG": np.array([0, 10]), "SHORT": np.array([0])},
        5, 4,
    )
    assert path is not None
    assert path.n_matches == 2
    assert path.n_requested == 3, "the impossible match still counts as requested"


def test_a_ticker_missing_from_closes_is_skipped_not_indexed():
    """A start with no series to read must not index into a neighbouring ticker."""
    path = forecast_paths_multi(
        {"A": _close(60)},
        {"A": np.array([0]), "GHOST": np.array([0])},
        5, 4,
    )
    assert path is not None
    assert path.n_matches == 1


def test_an_out_of_range_start_does_not_wrap_or_raise():
    """Negative and past-the-end indices are dropped, not clamped.

    Clamping would invent a forward path from bars the window never reached, which is
    the "silently all-NaN" failure this function exists to avoid.
    """
    path = forecast_paths_multi(
        {"A": _close(30)},
        {"A": np.array([-5, 0, 999])},
        5, 4,
    )
    assert path is not None
    assert path.n_matches == 1


def test_an_empty_match_map_yields_no_path():
    assert forecast_paths_multi({"A": _close(40)}, {}, 4, 3) is None


def test_distances_stay_positional_with_the_surviving_rows():
    """A dropped match must not shift a survivor's distance onto the wrong window.

    Same rule as the single-series form, applied per ticker: distances are consumed
    by the loop that decides survival, so a window that falls off the end cannot pull
    the next window's distance forward by one slot.
    """
    close = _close(40)
    path = forecast_paths_multi(
        {"A": close},
        {"A": np.array([0, 30])},          # the second has no forward path
        4, 20,
        distances={"A": np.array([1.0, 2.0])},
    )
    assert path is not None
    assert path.n_matches == 1
    assert path.distances.size == 1
    assert path.distances[0] == pytest.approx(1.0), (
        "the surviving window kept its own distance"
    )


def test_a_mismatched_distance_array_is_dropped_rather_than_mispaired():
    """One ticker's bad distance array must not mislabel the others' evidence."""
    path = forecast_paths_multi(
        {"A": _close(40), "B": _close(40)},
        {"A": np.array([0, 5]), "B": np.array([0])},
        4, 3,
        distances={"A": np.array([1.0, 2.0]), "B": np.array([9.0, 9.0])},
    )
    assert path is not None
    assert path.n_matches == 3
    # A partial pairing would misattribute, so all distances are discarded.
    assert path.distances.size == 0


@pytest.mark.parametrize("length,horizon", [(0, 3), (-1, 3), (4, -1)])
def test_invalid_shapes_are_rejected_multi(length, horizon):
    with pytest.raises(ValueError):
        forecast_paths_multi({"A": _close()}, {"A": np.array([0])}, length, horizon)


def test_a_two_dimensional_series_is_rejected():
    """A 2-D "close" is a caller bug and must not degrade into a quiet success.

    Skipping it would drop that ticker's evidence and return a path built from the
    rest -- indistinguishable, to the caller, from a complete answer.
    """
    with pytest.raises(ValueError):
        forecast_paths_multi(
            {"A": _close(20).reshape(2, -1)}, {"A": np.array([0])}, 2, 2
        )


def test_a_missing_series_is_skipped_rather_than_raising_on_shape():
    """An *absent* ticker is ordinary; a *malformed* one is not.

    The distinction matters because the panel is built from whatever the archive
    holds.  A name with no close entry costs its own matches and nothing else, while
    a name with a wrongly-shaped entry is a defect worth failing loudly on.
    """
    path = forecast_paths_multi(
        {"A": _close(20)}, {"A": np.array([0]), "ABSENT": np.array([0])}, 4, 3
    )
    assert path is not None and path.n_matches == 1


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipe():
    """A real pipeline over multi-session bars, at the app's own history length."""
    return Pipeline.from_frame(session_bars(8190, seed=20260930), length=HISTORY)


@pytest.fixture(scope="module")
def path(pipe):
    """A genuine path: the archive's last ``HISTORY`` bars matched against itself."""
    start = pipe.n_bars - HISTORY
    query = pipe.query_span(start, pipe.n_bars, label="latest")
    result = pipe.match(query, k=MATCHES, amplitude_weight=1.0)
    assert result.matches, "fixture must yield matches"
    return forecast_paths(
        pipe.close,
        np.array([m.start for m in result.matches], dtype=np.int64),
        HISTORY, PROJECTION,
        distances=np.array([m.distance for m in result.matches], dtype=float),
    )


def test_chart_draws_the_declared_history_and_projection(pipe, path):
    fig = build_forecast_path_figure(pipe, path)
    history = fig.data[0]
    projection = fig.data[-1]
    assert len(history.x) == HISTORY
    # One point more than the projection length, because offset 0 is the shared
    # anchor bar rather than the first projected bar.
    assert len(projection.x) == PROJECTION + 1


def test_history_and_projection_are_adjacent_on_the_x_axis(pipe, path):
    """No gap, and no overlap, between the two halves.

    History ends on the archive's last bar ``n - 1``; the projection's offset 0 sits
    on ``n``.  Any other arrangement either leaves a one-bar hole at the join or draws
    the anchor twice.
    """
    fig = build_forecast_path_figure(pipe, path)
    last_history = int(max(fig.data[0].x))
    first_projection = int(min(fig.data[-1].x))
    assert first_projection == last_history + 1


def test_the_two_halves_meet_at_zero(pipe, path):
    """The join is seamless: both sides are exactly 0% at the boundary.

    The history is rebased to the archive's last close and the projection's offset 0
    is that same bar, so the two meet at zero.  This is what keeps the chart in one
    unit -- a percent projection drawn against a dollar history would jump here.
    """
    fig = build_forecast_path_figure(pipe, path)
    assert float(fig.data[0].y[-1]) == pytest.approx(0.0, abs=1e-9)
    assert float(fig.data[-1].y[0]) == pytest.approx(0.0, abs=1e-9)


def test_projected_ticks_are_offsets_and_never_timestamps(pipe, path):
    """Bars that do not exist yet are not labelled with a time.

    Sessions are compressed, so a projection spanning an overnight close is drawn
    contiguously -- which is fine for spacing and wrong for labels.  Every tick right
    of the boundary is ``+K min``, so the chart never asserts a timestamp for a bar
    it has no data for.
    """
    fig = build_forecast_path_figure(pipe, path)
    labels = list(fig.layout.xaxis.ticktext)
    projected = [t for t in labels if str(t).startswith("+")]
    assert projected, "projection must carry its own tick labels"
    assert all(str(t).endswith(" min") for t in projected)
    # No tick on the projected side may look like a date.
    assert not any(str(t).startswith("+") and "20" in str(t).split(" min")[0]
                   for t in projected)


def test_the_band_is_drawn_as_a_filled_pair(pipe, path):
    """The interquartile band is two boundary traces joined by ``fill``.

    Order matters: ``tonexty`` fills to the *previous* trace, so the upper boundary has
    to be drawn first or the band is drawn inside-out.
    """
    fig = build_forecast_path_figure(pipe, path)
    names = [tr.name for tr in fig.data]
    assert "upper quartile" in names
    lower = next(tr for tr in fig.data if tr.name == "interquartile range")
    assert lower.fill == "tonexty"
    assert names.index("upper quartile") < names.index("interquartile range")


def test_chart_covers_the_full_declared_span(pipe, path):
    fig = build_forecast_path_figure(pipe, path)
    lo, hi = fig.layout.xaxis.range
    assert hi - lo == pytest.approx(HISTORY + PROJECTION)


def test_short_archive_clamps_history_rather_than_failing(pipe, path):
    """A history longer than the archive is truncated, not an IndexError.

    Reachable whenever a symbol returns less intraday history than asked for, which is
    the normal case rather than an edge case.
    """
    fig = build_forecast_path_figure(pipe, path, history_bars=10_000)
    assert len(fig.data[0].x) == pipe.n_bars


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_the_forecast_tab_renders_the_chart_without_a_search_run():
    """``tab_forecast`` calls the renderer unconditionally.

    This is the regression the chart was added to fix.  The block used to read::

        if out is None:
            st.info("**No search has been run yet.** ...")
        else:
            render_forecast_tab(out)

    which replaced the entire tab with a notice.  That was correct while the tab held
    only the evidence table, but the path chart does not depend on the reader's query
    or on a search run, so the gate left the tab's main visual permanently blank for
    anyone who had not pressed *Run match*.
    """
    body = app_block(r"with tab_projection:(.*?)\n    with tab_", "Projection tab body")
    assert "render_forecast_tab(" in body
    assert "if out is None" not in body


def test_the_renderer_draws_the_path_before_it_consults_the_run_output():
    """The chart is drawn before the search that feeds the evidence table.

    Ordering, not just presence, is the contract: moving the chart below the gate
    would keep this passing while restoring the bug -- the tab's headline visual
    appearing only once a reader has pressed a button that has nothing to do with it.

    **The two charts now live in two different functions**, because the Forecast tab
    was split into a fixed reference (``render_forecast_tab``) and the interactive
    ``Forecast`` tab (``render_window_tab``).  So there is no longer a single
    ``_render_forecast_path`` < ``_forecast_run`` ordering to assert across -- the
    reference chart does not run a search at all, and that is exactly the point of it.
    What is asserted instead is that each renderer holds the order its own tab needs:
    the reference chart is still drawn unconditionally by its own tab, and inside
    ``render_window_tab`` the projection still precedes the run it does not depend on.
    """
    reference = app_block(r"(def render_forecast_tab\(.*?)\n(?=def |\n# =)",
                          "render_forecast_tab")
    assert "_render_forecast_path(" in reference, (
        "the fixed reference chart must be drawn by the Forecast tab"
    )
    # It takes no input and runs no search -- a reference that waited on *Run match*
    # would be blank for anyone who never pressed it.
    for absent in ("_forecast_run(", "_forecast_brush_chart(",
                   "_render_forecast_evidence("):
        assert absent not in reference, (
            "the Forecast tab is a fixed reference and must not call %s; that is the "
            "*Forecast* tab's job" % absent
        )

    body = app_block(r"(def render_window_tab\(.*?)\n(?=def |\n# =)",
                     "render_window_tab")
    chart_at = body.index("_render_window_forecast(")
    run_at = body.index("_forecast_run(")
    assert chart_at < run_at, (
        "the projection must be drawn before the run that only the evidence table needs"
    )


# --------------------------------------------------------------------------- #
# The Projection / Forecast split
# --------------------------------------------------------------------------- #
def test_the_forecast_tab_is_a_fixed_reference_and_the_window_tab_is_the_rest():
    """The split is asserted in both directions, so it cannot silently re-merge.

    Either half alone is the weaker assertion.  Asserting only that ``Forecast`` is
    clean would pass on a build where ``Forecast`` had lost half its content --
    four of the five steps migrating nowhere -- and asserting only that ``Forecast``
    is complete would pass if ``Forecast`` had been left holding a second brush and a
    second evidence table, which is the arrangement this change exists to end.
    """
    reference = app_block(r"(def render_forecast_tab\(.*?)\n(?=def |\n# =)",
                          "render_forecast_tab")
    window = app_block(r"(def render_window_tab\(.*?)\n(?=def |\n# =)",
                       "render_window_tab")

    # Forecast: the reference chart, and nothing that needs a window or a run.
    assert "_render_forecast_path(" in reference
    for absent in ("_forecast_brush_chart(", "_render_window_forecast(",
                   "_forecast_run(", "_render_forecast_evidence(",
                   "_render_search_controls("):
        assert absent not in reference, (
            "the Projection tab is a fixed reference; %s belongs to *Forecast* and "
            "must not be drawn on both" % absent
        )

    # Forecast: the brush, its projection, the settings, the run and the table.
    for expected in ("_forecast_brush_chart(", "_render_window_forecast(",
                     "_render_search_controls(", "_forecast_run(",
                     "_render_forecast_evidence("):
        assert expected in window, (
            "the *Forecast* tab lost %s; the brush, projection, settings, run and "
            "evidence table belong on one tab" % expected
        )

    # **All four move together.**  The evidence table without the brush that produces
    # its window is the exact cross-tab failure this split was arranged to avoid: a
    # gesture on one tab and a claim about it on another, with nothing saying the two
    # disagree.
    assert window.index("_forecast_brush_chart(") < window.index("_forecast_run("), (
        "the window must be resolved before the run that searches it"
    )


def test_the_forecast_ticker_input_is_drawn_on_exactly_one_tab_body():
    """``ticker_input_forecast`` may be registered once per pass, and only once.

    ``st.tabs`` renders every body on every pass, so a second draw on any other tab
    raises ``StreamlitDuplicateElementKey`` and takes down the **whole page**.  This
    shipped once already -- the not-ready stub drew the input and then called the tab
    renderer, which drew its own -- and every source-level test guarding this invariant
    stayed green through it, because each asserted that one *stub* contained an input.

    Counting drawers across bodies is the assertion that could have caught it.  It is
    a source-level guard against the *shape* of the bug; ``TestTheRealAppRuns`` in
    ``test_daily.py`` is the one that actually executes the pass and would see the
    raise.

    **Counted over the whole-file AST, not per sliced body.**  Slicing each
    ``with tab_`` block and grepping it is wrong in both directions: the Price and
    Matches bodies each *explain* this rule in a comment quoting
    ``render_scope_ticker("forecast"`` verbatim (so a grep reports three drawers),
    and the last body's slice runs past the end of ``main()`` into module level (so it
    stops parsing).  Walking ``app.py``'s own AST once answers the question directly
    -- one ``render_scope_ticker("forecast", ...)`` call site in the file, full stop --
    and comments and docstrings cannot appear in it at all.
    """
    tree = ast.parse(app_source())
    call_sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "render_scope_ticker" and node.args
                and isinstance(node.args[0], ast.Constant)):
            call_sites.append((node.lineno, node.args[0].value))

    forecast_sites = [at for at, scope in call_sites if scope == "forecast"]
    assert len(forecast_sites) == 2, (
        "expected exactly two render_scope_ticker(\"forecast\", ...) call sites in "
        "app.py -- the ready path and the not-ready stub, on mutually exclusive "
        "branches -- found %r at lines %r" % (len(forecast_sites), forecast_sites)
    )

    # ...and both must be *inside* ``main()``, since a third draw anywhere else would
    # register the key on some other pass and collide with these.
    main_node = next(n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == "main")
    for lineno in forecast_sites:
        assert any(lineno in range(stmt.lineno, stmt.end_lineno + 1)
                   for stmt in ast.walk(main_node)), (
            "the forecast input is drawn at line %d, outside main(); every draw must "
            "be inside the one pass" % lineno
        )


def test_the_projection_horizon_is_drawn_once_and_shared_by_both_tabs():
    """One slider, two projections -- read rather than redrawn on the second tab.

    Two problems, opposite in direction, and both silent:

    * **Two draws of the same key** raise ``StreamlitDuplicateElementKey`` and take
      down the page, because ``st.tabs`` renders every body on every pass.
    * **Two controls under different keys** would let the reference projection and the
      reader's own describe different numbers of bars ahead, drawn one above the other
      as though they were comparable.  They would be two different questions, not a
      comparison.

    So ``render_forecast_tab`` draws the slider and ``render_window_tab`` reads it back
    through ``_resolve_projection``.  Asserting the read rather than a second draw is
    what pins "one value", not just "one widget".
    """
    reference = app_block(r"(def render_forecast_tab\(.*?)\n(?=def |\n# =)",
                          "render_forecast_tab")
    window = app_block(r"(def render_window_tab\(.*?)\n(?=def |\n# =)",
                       "render_window_tab")

    assert "_render_projection_control(" in reference, (
        "the Forecast tab owns the projection-length control"
    )
    assert "_render_projection_control(" not in window, (
        "the *Forecast* tab must not redraw the slider; a second draw of one key "
        "raises StreamlitDuplicateElementKey and takes down the page"
    )
    assert "_resolve_projection(" in window, (
        "the *Forecast* tab must read the same horizon, or its projection and the "
        "reference projection describe different numbers of bars ahead"
    )


# --------------------------------------------------------------------------- #
# The *Recent bars* control -- how far back the fixed reference looks
# --------------------------------------------------------------------------- #
def _state_ns(state):
    """A ``st`` stand-in whose ``session_state`` is a plain writable ``dict``.

    ``_resolve_recent`` reads the stored width *out* of session state, so that dict
    is the function's input rather than a detail of its environment.  Testing against
    a real one is what distinguishes "the stale value was clamped" from "the test
    restated the clamp and checked itself".
    """
    return type("_St", (), {"session_state": dict(state)})()


class TestTheRecentBarsControl:
    """The reference chart's history is a choice, and the choice has to be *one*.

    Before this control the width was the resolution's ``forecast_history_bars``, and
    the caption hard-coded the same number back at the reader.  Making it adjustable is
    small; making it adjustable *honestly* is the whole of what is asserted here, and
    each test pins a way that a single number silently stops describing what is drawn.
    """

    def _ns(self):
        return load_app_functions(
            {"_resolve_recent", "_recent_ceiling", "active_recent_bounds",
             "_snap_recent", "forecast_recent_key"},
            namespace={"st": st},
        )

    # -- wiring ------------------------------------------------------------ #
    def test_the_control_is_drawn_on_the_forecast_tab(self):
        """The tab that draws the chart is the tab that owns the control for it.

        Drawn below the chart it governs would mean the reader has already been shown
        an answer built from a width they had not chosen -- and, for this tab
        specifically, the chart draws ``length`` real bars, so the control cannot even
        be *resolved* after it.
        """
        reference = app_block(r"(def render_forecast_tab\(.*?)\n(?=def |\n# =)",
                              "render_forecast_tab")
        assert "_render_history_control(" in reference, (
            "the Forecast tab owns the recent-bars control"
        )
        assert "_render_history_control(" not in app_block(
            r"(def render_window_tab\(.*?)\n(?=def |\n# =)", "render_window_tab"), (
            "the *Forecast* tab must not draw it; a second draw of one key raises "
            "StreamlitDuplicateElementKey and takes down the page"
        )

    def test_it_is_resolved_before_the_chart_it_governs(self):
        """Ordering, not just presence -- moving the call below the chart would pass
        the previous test while restoring the bug.

        ``_render_forecast_path`` draws ``length`` bars of history, so the length has
        to be settled before it is called rather than threaded through afterwards.
        """
        reference = app_block(r"(def render_forecast_tab\(.*?)\n(?=def |\n# =)",
                              "render_forecast_tab")
        assert reference.index("_render_history_control(") < reference.index(
            "_render_forecast_path("), (
            "the history width must be resolved before the chart that draws it"
        )

    def test_the_reference_is_drawn_against_the_chosen_width(self):
        """The figure is told ``length``; omitting it silently redraws the default.

        This is the failure the whole control would otherwise ship with.  The caption
        and the title both report ``length``, ``forecast_path_for`` searches at
        ``length``, and ``build_forecast_path_figure`` -- called **without**
        ``history_bars`` -- falls back to the resolution's own default.  Nothing
        raises and the numbers all look right; the chart simply shows 240 bars while
        every caption underneath it says 60, and the blue history is no longer the
        window that was matched.
        """
        body = app_block(r"(def _render_forecast_path\(.*?)\n(?=def |\n# =)",
                         "_render_forecast_path")
        call = app_block(r"(build_forecast_path_figure\(\s*pipe, path,.*?\n    \))",
                         "build_forecast_path_figure call", flags=re.S)
        assert "history_bars=length" in call.replace(" ", "").replace("\n", ""), (
            "the reference figure must be drawn at the chosen width; without "
            "history_bars it silently falls back to the resolution default and the "
            "chart contradicts every caption on the tab"
        )
        assert body.count("build_forecast_path_figure(") == 1, (
            "a second figure call here would need the same treatment; this assertion "
            "exists so adding one cannot skip the check above"
        )

    def test_the_caption_reports_the_chosen_width_not_the_resolution_default(self):
        """No caption may fall back to ``forecast_history_bars``.

        The caption and the title are the only things telling the reader how much
        history they are looking at, and help copy is rendered from a constant -- so
        a stale number there would go stale silently rather than raising.
        """
        body = app_block(r"(def _render_forecast_path\(.*?)\n(?=def |\n# =)",
                         "_render_forecast_path")
        assert "forecast_history_bars" not in body, (
            "_render_forecast_path must caption the width it was handed; reading the "
            "resolution default would caption the chart with a number that no longer "
            "matches it once the reader moves the slider"
        )

    # -- bounds ------------------------------------------------------------ #
    def test_the_floor_is_the_validity_bound_shared_with_a_brushed_window(self):
        """The validity floor, snapped **up** onto the step grid.

        Below roughly eight bars a shape is a handful of z-score spikes and STUMPY's
        normalised distance is decided by whichever single bar is most extreme.
        Nothing raises, which is exactly why the floor has to be stated rather than
        discovered.

        **The offered floor is the first step multiple at or above that bound**, so it
        is 30 rather than 8.  Snapping up rather than down is what preserves the rule:
        the smallest width a reader can select is strictly *safer* than the smallest
        width that would otherwise have been legal, whereas snapping down to 8 would
        offer a window the validity bound exists to keep them out of.
        """
        ns = self._ns()
        step = ns["FORECAST_RECENT_STEP"]
        floor = ns["active_min_query_bars"]()
        low, high = ns["active_recent_bounds"]()

        assert low >= floor, (
            "the offered floor %d is below the validity bound %d" % (low, floor)
        )
        assert low == -(-floor // step) * step == 30, (
            "the floor must be the first step multiple at or above %d, not the raw "
            "bound -- an off-grid floor would make the first reachable width "
            "unreachable" % floor
        )
        assert high == ns["active_max_query_bars"]() // step * step

    def test_every_offered_width_is_a_step_multiple(self):
        """No tick can be off-grid, at either end, on either resolution.

        The floor is 8 and the daily ceiling is 252 -- neither is a multiple of 30.
        Left unsnapped the slider would open with a first tick at 8 and a last at 252,
        with nothing a step away from either reachable.
        """
        ns = self._ns()
        step = ns["FORECAST_RECENT_STEP"]
        for key in ("1m", "1d"):
            ns["ACTIVE_TIMEFRAME"][0] = key
            low, high = ns["active_recent_bounds"]()
            assert low % step == 0 and high % step == 0, (
                "%s bounds (%d, %d) are not both multiples of %d"
                % (key, low, high, step)
            )
        ns["ACTIVE_TIMEFRAME"][0] = "1m"

    def test_the_default_lands_on_the_grid_in_both_resolutions(self):
        """240 and 60 are both exact multiples of 30, so neither opens between ticks.

        A step sharing no factor with the default would put the reference chart's own
        starting width off the grid, and the reader would open on a value the control
        could not name.
        """
        from timeseries.timeframes import get_timeframe

        step = self._ns()["FORECAST_RECENT_STEP"]
        for key in ("1m", "1d"):
            default = get_timeframe(key).forecast_history_bars
            assert default % step == 0, (
                "%s defaults to %d bars, which is not a multiple of the %d-bar step, "
                "so the chart would open at a width the control cannot name"
                % (key, default, step)
            )

    def test_the_ceiling_is_capped_against_the_archive_not_just_the_resolution(self):
        """A 1-minute archive that returned 600 bars cannot offer 2400.

        ``forecast_path_for`` answers an archive shorter than the window with
        ``None``, which the tab reports as *"this shape has no historical analogue"* --
        a claim about the archive rather than about a slider the reader moved.

        The archive cap is ``n - 100``, and it only binds *below* the resolution cap,
        so the case worth asserting is the one just under it: an archive of 400 bars
        on 1-minute offers 300, not 390.
        """
        ceiling = self._ns()["_recent_ceiling"]
        assert ceiling(1_000_000)[1] == 390           # long archive: resolution cap
        assert ceiling(490)[1] == 390                # 490-100 = 390, right at it
        assert ceiling(400)[1] == 300                # short archive: archive cap
        assert ceiling(400)[1] < 390

    def test_a_degenerate_archive_yields_a_drawable_single_bar_range(self):
        """``min_value == max_value`` makes ``st.slider`` **raise**.

        Verified on the pinned Streamlit build: it is not a degenerate-but-legal
        slider, it is ``StreamlitInvalidParameterTypeError`` and the page is gone.
        A too-short archive is already reported by the chart's own "no forecast path
        could be built" message, so the control has to stay drawable rather than
        becoming a second, louder way of saying it.
        """
        ceiling = self._ns()["_recent_ceiling"]
        for n in (0, 1, 50, 105, 108):
            low, high = ceiling(n)
            assert low < high, (
                "an archive of %d bars produced the degenerate range (%d, %d), which "
                "st.slider rejects" % (n, low, high)
            )

    # -- resolution -------------------------------------------------------- #
    def test_the_bounds_differ_by_resolution_and_the_key_follows(self):
        """240 minutes is not 240 days, so neither the range nor the key may carry over.

        A key that was not namespaced would let a reader move the 1-minute slider to
        390 bars and then find daily offering 390 *trading days* -- ten months of
        history, as a number that looks like the one they just chose.
        """
        from timeseries.timeframes import get_timeframe

        ns = self._ns()
        minute = get_timeframe("1m")
        daily = get_timeframe("1d")
        assert minute.max_query_bars != daily.max_query_bars
        assert minute.forecast_history_bars != daily.forecast_history_bars
        # Read the suffix off the app rather than restating it, so a rename is caught.
        suffix = re.search(r'^RESOLUTION_SUFFIX = "([^"]+)"', app_source(), re.M)
        assert suffix, "RESOLUTION_SUFFIX is not declared in app.py"
        assert ns["forecast_recent_key"]() == "forecast_recent_bars" + suffix.group(1) \
            + minute.key, (
            "the key must go through state_key, so a resolution switch starts from "
            "that resolution's own default rather than inheriting the other's"
        )

    # -- behaviour --------------------------------------------------------- #
    def test_an_explicit_length_bypasses_the_control_and_its_bounds(self):
        """A pinned ``length`` is honoured verbatim.

        ``render_forecast_tab`` takes ``length`` so the tests and any other caller can
        pin a width; clamping a caller's explicit number against a *slider* range
        would silently rewrite it, which is the wrong-number failure this repo treats
        as the worst one available.
        """
        resolve = self._ns()["_resolve_recent"]
        assert resolve(77, 1_000_000) == 77
        assert resolve(3, 1_000_000) == 3            # below the slider's own floor
        assert resolve(10_000, 1_000_000) == 10_000  # above its ceiling

    def test_the_default_is_the_width_the_reference_always_used(self):
        """Nothing about the chart changes until the reader moves the slider.

        The control is new; the chart it governs is not.  A default that quietly
        differed from ``forecast_history_bars`` would make every reader's first
        impression of the tab wrong relative to the last version they saw.
        """
        from timeseries.timeframes import get_timeframe

        assert get_timeframe("1m").forecast_history_bars == 240
        assert get_timeframe("1d").forecast_history_bars == 60
        source = app_source()
        assert "_tf().forecast_history_bars" in source, (
            "the control must default to the resolution's forecast_history_bars"
        )

    def test_a_stale_value_outside_the_range_is_clamped_rather_than_raising(self):
        """The reachable case is a ticker change to a thinner archive.

        The stored value is whatever the reader last set, and the ceiling depends on
        the archive.  ``st.slider`` raises ``StreamlitValueAboveMaxError`` on a value
        outside its bounds rather than adjusting, so without the clamp a reader who
        moved the slider on QQQ and then switched the Forecast ticker to something
        thin would lose the whole page over a control they were not touching.  The
        alternative -- widening the ceiling to fit the stale value -- would offer a
        length the archive cannot support, which reads as a claim about history.

        **Asserted against a real session state rather than by restating the clamp.**
        The stub is there because the stored value is this function's *input*;
        re-implementing ``max(low, min(high, v))`` in the test would keep passing even
        if the function stopped clamping at all.
        """
        ns = self._ns()
        resolve, ceiling = ns["_resolve_recent"], ns["_recent_ceiling"]
        key = ns["forecast_recent_key"]()
        low, high = ceiling(400)          # this archive offers at most 300

        for stale in (10_000, 390, 1, 0, 300):
            ns["st"] = _state_ns({key: stale})
            got = resolve(None, 400)
            assert low <= got <= high, (
                "a stored %r resolved to %r, outside the offered range (%d, %d)"
                % (stale, got, low, high)
            )
            # Never *above* what the archive supports: the ceiling must not be widened
            # to accommodate a stale value, which is the other way this goes wrong.
            assert got <= 300

    def test_an_off_grid_value_is_snapped_rather_than_kept(self):
        """``step`` constrains what the slider *emits*, not what it *accepts*.

        Measured on the pinned build: with ``step=30`` a stored 45 is kept and
        rendered as 45, sitting between the 30 and 60 ticks, and its bounds checks
        pass because it is inside ``min``/``max``.  Without the snap the reader would
        be looking at a chart of 45 bars on a control that cannot select 45 -- and the
        value is reachable, not hypothetical: a resolution switch, a thinner ticker,
        or a session left over from before this step existed all produce one.

        **Nearest, not rounded down.**  Snapping a stale 45 *down* to 30 would
        silently shorten the window the reader had chosen, changing the chart under a
        control they did not touch.
        """
        ns = self._ns()
        resolve, snap, step = (ns["_resolve_recent"], ns["_snap_recent"],
                               ns["FORECAST_RECENT_STEP"])
        key = ns["forecast_recent_key"]()

        # Nearest in both directions, and the halfway case resolving upwards.
        assert snap(45, 30, 390) == 60
        assert snap(31, 30, 390) == 30
        assert snap(44, 30, 390) == 30
        assert snap(46, 30, 390) == 60
        # Exhaustive: every input resolves onto the grid, or onto a bound that is.
        for raw in range(0, 400):
            got = snap(raw, 30, 390)
            assert got % step == 0 or got in (30, 390)

        # And end to end, through the resolver, for values that are stale *and* off-grid.
        for stale, expected in ((45, 60), (46, 60), (44, 30), (1, 30), (10_000, 390)):
            ns["st"] = _state_ns({key: stale})
            got = resolve(None, 1_000_000)
            assert got == expected, (
                "stored %r resolved to %r, expected %r" % (stale, got, expected)
            )

    def test_the_step_is_dropped_when_the_range_cannot_hold_it(self):
        """A ``step`` wider than the range leaves a thumb that does not travel.

        ``st.slider`` accepts it -- but on the two-value range a too-thin archive
        collapses to, the reader gets two adjacent legal values and a control that
        looks broken.  A one-bar step there is honest about there being nothing to
        choose.
        """
        ns = self._ns()
        step = ns["FORECAST_RECENT_STEP"]
        for n in (0, 1, 50, 105, 108):
            low, high = ns["_recent_ceiling"](n)
            assert high - low < step, (
                "an archive of %d bars yielded (%d, %d), a range wide enough to hold "
                "the %d-bar step but too thin for the matcher to use" % (n, low, high, step)
            )
        # And the control draws a legal step for it.
        body = app_block(r"(def _render_history_control\(.*?\n(?=\ndef |\n# =))",
                         "_render_history_control")
        assert "FORECAST_RECENT_STEP if" in body and "else 1" in body, (
            "the step must fall back to 1 when the range cannot hold it"
        )

    def test_the_horizon_control_is_untouched_by_it(self):
        """Two sliders, two keys -- moving one must not decide the other.

        One looks backwards and one forwards, and they are judged on different
        grounds: a history wants to be a recognisable shape, a projection wants to
        stop before there is no tape left to compare against.  Sharing a key would
        make the *Projection bars* slider silently resize the reference's history.
        """
        source = app_source()
        assert 'FORECAST_RECENT_KEY = "forecast_recent_bars"' in source
        assert 'FORECAST_HORIZON_KEY = "forecast_horizon_bars"' in source
        assert "state_key(FORECAST_RECENT_KEY)" in source.replace("\n", "").replace(
            "  ", " ") or "state_key(FORECAST_RECENT_KEY)" in source



def test_forecast_sits_immediately_after_projection_in_the_tab_order():
    """The tab's position is load-bearing, not a preference.

    ``FORECAST_SELECTION_KEY`` records that a plotly chart's element id hashes the
    page's **chart list**.  The brush's live selection is keyed off that id, so moving
    this tab away from immediately after *Projection* changes how many charts precede it
    and invalidates every selection a reader has already made.  Nothing raises: the
    resolved span in session state survives, so the visible symptom is a chart that
    stops accepting a second drag.

    That makes this the kind of reordering that reads as harmless and is not, so it is
    pinned rather than left to taste.
    """
    order = app_block(r"TAB_ORDER\s*=\s*\((.*?)\)", "TAB_ORDER")
    names = re.findall(r'"([^"]+)"', order)
    assert names.index("Forecast") == names.index("Projection") + 1, (
        "*Forecast* must sit immediately after *Projection* so the brush chart's "
        "element id is unchanged; got %r" % (names,)
    )
    # And the bodies are rendered in that same order, or the horizon read in
    # ``render_window_tab`` would see the *previous* pass's value.
    #
    # **Anchored to a line start, not a bare substring.**  The explanatory comment
    # above the unpack names ``with tab_forecast:`` in prose, so a plain
    # ``src.index("with tab_forecast:")`` finds the *comment* -- and reports the bodies
    # reversed.  Same hazard ``test_fetch.py`` documents for its tab-body slicing.
    src = app_source()
    bodies = [m.start() for m in
              re.finditer(r"^    with tab_(?:projection|forecast):", src, re.M)]
    assert len(bodies) == 2, (
        "expected one `with tab_projection:` and one `with tab_forecast:`, found %d"
        % len(bodies)
    )
    assert src.index("    with tab_projection:") < src.index("    with tab_forecast:"), (
        "the Projection body must render before the Forecast body; it owns the "
        "projection slider that the latter reads"
    )


def test_the_chart_render_order_is_unchanged_by_the_split():
    """The chart keys must render in the order they did before the split.

    This is the concrete version of the element-id invariant: Streamlit keys a
    plotly chart's stored value off a hash that includes the serialised figure *and*
    the page's chart list, so a chart moving earlier or later in the render order gets
    a new identity.  For the brush that means a live selection is dropped.

    The order is asserted as a whole-file sequence rather than per-function, because
    the two tab bodies are separate ``with`` blocks and the risk lives in *their*
    combination -- a chart could stay inside its own function and still move relative
    to the others.
    """
    src = app_source()
    order = [
        # The reference chart and its pooled cross-symbol twin, on *Forecast*.
        src.index('key="forecast_path"'),
        src.index("def _render_panel_forecast_path("),
        # Then the brush, its projection and the evidence bars, on *Forecast*.
        src.index("forecast_brush_key(),"),
        src.index('key="forecast_path_selected"'),
        src.index('key="forecast_bars"'),
    ]
    assert order == sorted(order), (
        "chart render order changed: a plotly chart's element id hashes the page's "
        "chart list, so reordering silently invalidates the brush's live selection"
    )


def test_the_chart_is_rendered_by_its_own_helper():
    """The path chart lives in a helper, not inline in the tab body.

    It has a different lifecycle from the evidence table -- no search run, no
    dependency on the reader's query -- and keeping them in one function is how the
    single gate came to cover both.
    """
    assert "def _render_forecast_path(" in app_source()
    body = app_block(r"(def _render_forecast_path\(.*?)\n(?=def |\n# =)",
                     "_render_forecast_path")
    assert "forecast_path_for(" in body
    assert "build_forecast_path_figure(" in body


def test_the_chart_is_independent_of_the_sidebar_k():
    """The chart's match count is its own constant, not the sidebar's ``k``.

    ``k matches`` governs the evidence table below, whose sample is judged against
    ``Min matches for evidence``.  Binding both to one slider would make a control
    labelled "how many matches" silently decide two unrelated analyses, so a reader
    lowering ``k`` to inspect one horizon would quietly thin the chart too.
    """
    body = app_block(r"(def _render_forecast_path\(.*?)\n(?=def |\n# =)",
                     "_render_forecast_path")
    assert "FORECAST_PATH_MATCHES" not in body or "cfg" not in body
    assert "FORECAST_PATH_MATCHES = 30" in app_source()


def test_the_shortfall_is_reported_rather_than_hidden():
    """A count below the requested matches is stated on screen.

    ``forecast_paths`` drops matches with no forward history, which is always the most
    recent ones, so a shortfall is normal near the end of an archive -- and it is also
    exactly the situation where the median is least trustworthy.  On the standard
    21-session equity fixture this really does bite, so the warning is load-bearing
    rather than defensive.
    """
    body = app_block(r"(def _render_forecast_path\(.*?)\n(?=def |\n# =)",
                     "_render_forecast_path")
    assert "path.n_matches < FORECAST_PATH_MATCHES" in body
    assert "st.warning(" in body


def test_forecast_paths_is_exported_from_the_library():
    """``forecast_paths`` and ``ForecastPath`` are part of the public surface.

    ``app.py`` imports ``forecast_paths`` directly, so an entry dropped from
    ``__all__`` would be an undocumented coupling rather than a clean break.
    """
    from timeseries import forecast as forecast_module

    assert "forecast_paths" in forecast_module.__all__
    assert "forecast_paths_multi" in forecast_module.__all__
    assert "ForecastPath" in forecast_module.__all__
    assert isinstance(ForecastPath, type)
    assert callable(forecast_paths)
    assert callable(forecast_paths_multi)


# --------------------------------------------------------------------------- #
# The cross-sectional chart -- matching the shape anywhere in the panel
# --------------------------------------------------------------------------- #
# The single-ticker chart answers "when has *this name* done this?".  The
# cross-sectional one answers "has *anything* done this?", which has a different
# candidate population (every window of every constituent, not one series) and can
# therefore produce a different answer.  Both are shown because quoting only the
# first would report "this has happened before" as though it were notable, when it is
# only notable *for that name*.
class TestCrossSectionalForecastChart:
    """Wiring for the pooled chart: it is drawn, it is labelled, it cannot lie."""

    def _body(self):
        return app_block(r"(def _render_panel_forecast_path\(.*?)\n(?=def |\n# =)",
                         "_render_panel_forecast_path")

    def test_the_tab_renders_the_cross_sectional_chart(self):
        """The pooled path is reached from the tab, not merely defined.

        A helper nobody calls is dead code, and it would read as a working feature
        while the reader saw only the single-ticker answer.
        """
        tab = app_block(r"(def _render_forecast_path\(.*?)\n(?=def |\n# =)",
                        "_render_forecast_path")
        assert "_render_panel_forecast_path(" in tab, (
            "the single-symbol renderer must reach the cross-sectional one"
        )

    def test_it_is_drawn_with_the_figure_builder_not_a_second_chart_type(self):
        """Same builder as the chart above, so the two are actually comparable.

        Two different figure functions would mean two different rebasings or two
        different scales, and the whole point of drawing them one above the other is
        that a bar means the same thing on both.
        """
        body = self._body()
        assert "build_forecast_path_figure(" in body

    def test_it_reports_how_many_symbols_contributed(self):
        """The evidence breakdown is on screen, not just the curve.

        ``ForecastPath.starts`` is a bare concatenation across series, so the chart
        cannot attribute its own evidence.  A median drawn from 30 windows of one
        name is a far weaker claim than 30 spread across 12 names, and the reader
        cannot see the difference without being told.
        """
        body = self._body()
        assert '"n_tickers"' in body or "n_tickers" in body
        assert "st.warning(" in body, (
            "a concentrated match set must be flagged, not silently averaged"
        )

    def test_a_missing_panel_archive_does_not_break_the_tab(self):
        """The panel chart is extra evidence, never a prerequisite.

        ``data/sp500_panel/`` is built by a separate downloader and may simply not
        exist.  The tab must fall back to the single-ticker path, which is a complete
        answer to the question it asks.
        """
        body = self._body()
        assert "if bundle is None:" in body
        assert "st.info(" in body
        assert "st.error(" not in body and "st.exception(" not in body, (
            "a missing archive is not an error; it is an absent extra chart"
        )

    def test_the_panel_search_names_no_home_ticker(self):
        """The query vector comes from the *fetched* ticker, not the panel's.

        Its bar indices therefore address a different series than the panel's, and
        naming a home ticker would hand the self-match guard a range in the wrong
        index space -- suppressing an unrelated region of that ticker, silently.
        Mirrors ``cross_sectional_match``.
        """
        body = app_block(r"(def panel_forecast_path_for\(.*?)\n(?=def |\n# =)",
                         "panel_forecast_path_for")
        assert 'ticker=""' in body.replace(" ", "").replace("\n", "")

    def test_the_panel_path_uses_the_aligned_close_series(self):
        """Matches index the aligned frame, so the aligned series must be read.

        ``PanelSearch.close`` returns the raw bars, which sit ~20 rows earlier for
        each ticker.  Reading a match's forward bars from that series computes the
        projection from the wrong close -- silently, and with a plausible-looking
        number.  See ``TestPanelIndexSpaces``.
        """
        body = app_block(r"(def panel_forecast_path_for\(.*?)\n(?=def |\n# =)",
                         "panel_forecast_path_for")
        assert "close_aligned(" in body
        assert ".close(" not in body.replace("close_aligned(", "")

    def test_it_never_raises_into_the_page(self):
        """A broken archive must not take the Forecast tab down with it.

        Every failure path returns ``None``, so the caller has one code path for
        "draw nothing" instead of several, and no partial failure can surface as an
        exception in a tab that otherwise works.
        """
        body = app_block(r"(def panel_forecast_path_for\(.*?)\n(?=def |\n# =)",
                         "panel_forecast_path_for")
        assert "return None" in body
        assert "except Exception" in body


def test_every_token_in_the_forecast_guide_is_substituted():
    """No ``[[TOKEN]]`` survives into the rendered help copy.

    ``fill_tokens`` is a literal-substitution pass with no fallthrough, so a token
    added to the copy without a matching ``.replace`` renders to the reader as the
    literal text ``[[FORECAST_HISTORY_BARS]]`` -- a broken-looking guide, on a tab
    whose whole job is to be trusted.  Nothing else would catch that: the copy is
    plain strings and no assertion reads it.
    """
    import re

    source = app_source()
    start = source.index('"Projection": (')
    end = source.index('"Quality": (', start)
    body = source[start:end]

    tokens = set(re.findall(r"\[\[([A-Z_]+)\]\]", body))
    assert tokens, "the Forecast guide should quote the live constants, not hard-code them"

    fill_tokens = _ns["fill_tokens"]
    rendered = fill_tokens(body)
    assert "[[" not in rendered
    for token in tokens:
        assert "[[%s]]" % token not in rendered


# --------------------------------------------------------------------------- #
# The brush-driven section
# --------------------------------------------------------------------------- #
def _box(a, b):
    """A Plotly box-selection event spanning bar indices ``a``..``b``."""
    return {"selection": {"box": [{"x": [a, b]}]}}


@pytest.fixture(scope="module")
def sessions(pipe):
    return _ns["session_spans"](pipe.bars["timestamp"])


@pytest.fixture(scope="module")
def resolve_selection_window():
    return _ns["resolve_selection_window"]


def test_an_in_session_brush_keeps_the_width_that_was_drawn(pipe,
                                                           resolve_selection_window):
    """A brush of N bars is a window of N bars -- nothing snapped, nothing padded.

    ``snap_to_grid``'s forward-padding was a bug once already: it put bars the
    reader never drew into the window whose forward returns were about to be
    measured.  A brush now defines its own length end to end.
    """
    for a, b in [(1000, 1119), (4000, 4120)]:
        window = resolve_selection_window(_box(a, b), pipe.n_bars)
        assert window is not None
        assert window[1] - window[0] == b - a + 1


def test_a_selection_crossing_an_overnight_close_is_not_fenced(pipe, sessions,
                                                               resolve_selection_window):
    """The window spans the close, at the width and position the reader drew.

    This is the inverse of what it used to assert.  The window *was* confined to the
    session the brush started in, because a gap-jump between two bars 17.5 hours apart
    is a move that never happened as a trade.  Confining it also meant a brush
    reaching past that session's open was relocated without being asked -- which is
    what this test now guards against.

    The fixture's sessions are 370 bars wide on this seed, so ``boundary - 20`` to
    ``boundary + 50`` straddles the boundary; the resolved window must now be exactly
    the 71 bars that were brushed, in the position they were brushed.
    """
    boundary = next(lo for _day, lo, _hi in sessions if lo > 0)
    a, b = boundary - 20, boundary + 50
    window = resolve_selection_window(_box(a, b), pipe.n_bars)
    assert window == (a, b + 1), (
        "the window was moved or shortened; a brush is placed where it is drawn, "
        "across a session close or not"
    )
    # And the span really does cross two sessions, so the assertion above is not
    # passing vacuously on a fixture that happens not to have a boundary there.
    days = {day for day, lo, hi in sessions if lo <= a and b < hi}
    assert days == set(), "this brush should straddle a boundary, so no one day owns it"


def test_a_selection_to_the_archive_end_is_kept_but_warned_about(pipe,
                                                                 resolve_selection_window):
    """A window at the very end resolves -- the renderer is what refuses to forecast it.

    ``resolve_selection_window`` reports what was selected; it is not where the
    "nothing to compare against" policy lives.  Keeping the two apart means the
    resolution helper stays a pure function of the brush, and the policy stays in one
    readable place.
    """
    window = resolve_selection_window(
        _box(pipe.n_bars - 40, pipe.n_bars - 1), pipe.n_bars
    )
    assert window == (pipe.n_bars - 40, pipe.n_bars)


@pytest.mark.parametrize("event", [None, {}, {"selection": {"box": []}},
                                  {"selection": {"box": [{}]}}])
def test_no_brush_resolves_to_nothing(resolve_selection_window, pipe, event):
    """An absent or unparseable brush is ``None``, not a fabricated default.

    The first paint of this tab has no brush, so this is a normal state rather than
    an error -- and the renderer shows a prompt instead of forecasting a window the
    reader never chose.
    """
    assert resolve_selection_window(event, pipe.n_bars) is None


def test_a_window_whose_width_disagrees_with_length_is_rejected(pipe):
    """``forecast_path_for`` refuses a window that is not exactly ``length`` wide.

    A coerced or clamped width would be *searched* at one length while the reader's
    chart showed another, and every number downstream -- the forward slice, the
    median, the band -- would then describe a question nobody asked.  Refusing
    surfaces the mistake as a missing chart rather than a wrong one.
    """
    forecast_path_for = _ns["forecast_path_for"]
    common = dict(horizon=240, k=30, amplitude_weight=1.0)

    # 240-wide window against length=240: accepted.
    assert forecast_path_for("a", pipe, length=240, window=(4000, 4240),
                             **common) is not None
    # 120-wide window against length=240: rejected.
    assert forecast_path_for("b", pipe, length=240, window=(4000, 4120),
                             **common) is None
    # The same span with the matching length: accepted, so the rejection above was
    # about the disagreement and not about the span itself.
    assert forecast_path_for("c", pipe, length=120, window=(4000, 4120),
                             **common) is not None


@pytest.mark.parametrize("window", [(4000, 99999), (5000, 4000), (-5, 235), (10, 10)])
def test_an_out_of_range_window_yields_nothing(pipe, window):
    """A window outside the archive is refused rather than clipped.

    Clipping would let a caller believe it had forecast a window that does not
    exist.  Returning ``None`` keeps the caller's "draw nothing and say why" path
    as the single answer to every impossible request.
    """
    forecast_path_for = _ns["forecast_path_for"]
    assert forecast_path_for("x", pipe, length=240, window=window,
                             horizon=240, k=30, amplitude_weight=1.0) is None


def test_omitting_the_window_keeps_the_reference_behaviour(pipe):
    """``window=None`` still means the archive's most recent bars.

    The always-live chart above the brush relies on this, and it is the reason the
    parameter is an override rather than a replacement: the two callers share one
    implementation without either having to know about the other.
    """
    forecast_path_for = _ns["forecast_path_for"]
    path = forecast_path_for("n", pipe, length=240, window=None,
                             horizon=240, k=30, amplitude_weight=1.0)
    assert path is not None
    fig = build_forecast_path_figure(pipe, path)
    assert int(max(fig.data[0].x)) == pipe.n_bars - 1


def test_the_brushed_chart_draws_the_selected_window_not_the_tail(pipe):
    """The history is the brushed window, and the projection grows from its end.

    This is the whole point of the section.  Drawing the archive's tail instead --
    which is what the figure did before ``window`` existed -- would put a caption
    describing the reader's window above a chart of different price action, and
    would trail the projection off from the end of the archive rather than from
    where the window finished.
    """
    forecast_path_for = _ns["forecast_path_for"]
    start, stop = 4000, 4120
    path = forecast_path_for("b", pipe, length=120, window=(start, stop),
                             horizon=240, k=30, amplitude_weight=1.0)
    assert path is not None

    fig = build_forecast_path_figure(pipe, path, window=(start, stop))
    hist_x = list(fig.data[0].x)
    proj_x = list(fig.data[-1].x)

    assert (min(hist_x), max(hist_x)) == (start, stop - 1)
    assert len(hist_x) == stop - start
    # The seam sits at the window's end, not at n - 0.5.
    assert max(hist_x) + 1 == min(proj_x) == stop
    assert min(proj_x) != pipe.n_bars
    # Still one unit on both sides of the join.
    assert float(fig.data[0].y[-1]) == pytest.approx(0.0, abs=1e-9)
    assert float(fig.data[-1].y[0]) == pytest.approx(0.0, abs=1e-9)


def test_an_impossible_window_on_the_figure_is_clamped_not_fatal(pipe):
    """A bad ``window`` reaching the figure cannot raise.

    The figure is a drawing function and the archive boundary is arithmetic a
    caller can get wrong; clamping to something drawable keeps a mistake in the
    renderer from taking the tab down.
    """
    forecast_path_for = _ns["forecast_path_for"]
    path = forecast_path_for("c", pipe, length=240, window=(0, 240),
                             horizon=60, k=30, amplitude_weight=1.0)
    fig = build_forecast_path_figure(pipe, path, window=(9000, 99999))
    assert fig is not None
    hist_x = list(fig.data[0].x)
    assert 0 <= min(hist_x) <= max(hist_x) < pipe.n_bars


def test_a_brushable_chart_does_not_advertise_pan():
    """``pan2d`` must be stripped from the modebar on a brushable chart.

    This is the fix for *"drag does not work when pan was selected earlier"*.

    Plotly's modebar buttons do not just act, they **rebind the chart's interaction
    mode**: pressing Pan sets ``dragmode`` to ``"pan"``, and nothing restores it.
    Measured against the plotly build this app ships: after a relayout to pan the
    mode stays ``"pan"`` until a ``react()`` re-renders the figure -- and Streamlit
    only re-sends a chart when its own state changes, so on an otherwise-unchanged
    rerun the pan simply sticks.

    The result on a brushable chart is a silent, total failure of the only input: a
    drag pans instead of selecting, no selection event is emitted, and the section
    reports "no window selected" forever while looking perfectly healthy. Removing
    the button removes the mode that has no purpose there anyway.
    """
    chart_config = _ns["chart_config"]

    brushable = chart_config(selectable=True)["modeBarButtonsToRemove"]
    assert "pan2d" in brushable, (
        "a brushable chart still offers Pan, which permanently overrides dragmode "
        "and silently breaks the brush"
    )
    # The box/lasso tools must SURVIVE: they are the brush.
    assert "select2d" not in brushable
    assert "lasso2d" not in brushable


def test_a_display_chart_keeps_pan_and_loses_the_selection_tools():
    """The converse: a non-brushable chart may pan, and must not offer a selection.

    ``dragmode`` on those charts is ``"pan"`` (see :func:`build_price_figure`), so
    panning is the *intended* gesture there and stripping it would remove a
    capability the reader actually has.  Selection tools go the other way, for the
    reason :func:`chart_config` documents: never advertise a gesture nothing reads.
    """
    chart_config = _ns["chart_config"]

    display = chart_config(selectable=False)["modeBarButtonsToRemove"]
    assert "pan2d" not in display, (
        "a read-only chart should keep panning -- it is its intended drag"
    )
    assert "select2d" in display
    assert "lasso2d" in display


def test_the_forecast_section_renders_a_brushable_chart_and_its_forecast():
    """The section draws the brush above and the forecast below it.

    Order is the contract: the forecast chart is below because the brush is the
    input, and a reader who had to scroll back up to find the control that changes
    the answer would be reading the two out of order.

    The two halves are separate functions now -- ``_forecast_brush_chart`` resolves
    the window and ``_render_window_forecast`` draws for it -- so the ordering is
    asserted where it is decided, in ``render_forecast_tab``.
    """
    brush_body = app_block(r"(def _forecast_brush_chart\(.*?)\n(?=def |\n# =)",
                           "_forecast_brush_chart")
    assert "forecast_brush_key()" in brush_body
    assert "selectable=True" in brush_body
    assert 'on_select="rerun"' in brush_body

    assert "build_price_figure(" in brush_body
    assert "resolve_selection_window(" in brush_body
    assert "build_forecast_path_figure(" not in brush_body, (
        "the brush helper resolves the window; drawing for it is the other half's job"
    )

    window_body = app_block(r"(def _render_window_forecast\(.*?)\n(?=def |\n# =)",
                            "_render_window_forecast")
    assert "build_forecast_path_figure(" in window_body

    # **The ordering assertion moved with the section.**  The brush, the projection and
    # the evidence table are the *Forecast* tab, so the order they must appear in is
    # decided there and nowhere else.
    tab_body = app_block(r"(def render_window_tab\(.*?)\n(?=def |\n# =)",
                         "render_window_tab")
    assert (tab_body.index("_forecast_brush_chart(")
            < tab_body.index("_render_window_forecast(")), (
        "the brush is the input, so it is drawn before the forecast it produces"
    )


def test_the_forecast_brush_is_read_from_the_return_value_not_session_state():
    """The brush must come from ``st.plotly_chart``'s return value.

    This is the bug that made the section report *"No window selected yet"* forever,
    while the chart sat right above it happily accepting drags.

    ``st.plotly_chart`` registers its selection widget through ``register_widget``,
    whose signature has **no ``user_key`` parameter at all** -- ``user_key`` only
    arrives via a ``WidgetMetadata``, and the plotly path never builds one.  So
    ``session_state[FORECAST_BRUSH_KEY]`` is never written and ``.get(...)`` is
    permanently ``None``.  The event is delivered *only* as the return value, which
    is why the chart's return is assigned to ``event`` at all.

    Asserting the call is not enough: the chart must be *read* through that same
    binding, or the assignment is dead code and the bug returns.
    """
    body = app_block(r"(def _forecast_brush_chart\(.*?)\n(?=def |\n# =)",
                     "_forecast_brush_chart")
    assert "event = st.plotly_chart(" in body, (
        "the chart's selection event must be bound to a name"
    )
    assert "resolve_selection_window(event, n)" in body, (
        "the brush must be read from st.plotly_chart's return value; "
        "session_state is never populated for this widget"
    )
    assert "session_state.get(forecast_brush_key())" not in body, (
        "reading session_state[forecast_brush_key()] always yields None -- the "
        "widget is registered without a user_key"
    )
    # The span has to be copied somewhere that outlives the rerun, because drawing the
    # forecast chart changes the context chart's element id.  See FORECAST_SELECTION_KEY.
    assert "st.session_state[forecast_selection_key()] = fresh" in body, (
        "the resolved span must be stored; reading only the return value makes the tab "
        "revert after a single drag"
    )


def test_the_price_brush_still_comes_from_session_state():
    """The Price tab keeps its session-state read, and the reason is ordering.

    Both routes work, but only one is available to each caller.  The Price tab's
    chart is drawn far below in the tab body while the query must be resolved above
    it, so the brush can only be read from state that outlives the call -- the
    returned event would be gone by the time the query is snapped.  The Forecast
    tab draws and reads in one place, so it uses the return value.

    Guarding both keeps a well-meant "consistency" refactor from breaking the Price
    tab's brush.
    """
    source = app_source()
    assert "selection_to_span(st.session_state.get(price_brush_key()), n)" in source


def test_a_real_plotly_state_is_parsed_by_the_brush_resolver():
    """``resolve_selection_window`` accepts the actual object Streamlit returns.

    ``st.plotly_chart`` hands back a ``PlotlyState`` -- a read-only mapping, *not*
    the plain dict the Price tab's synthetic fixtures use.  The parser's guards
    (``isinstance(event, dict)`` and the per-shape checks) therefore have to hold for
    that real type, and only a real one can show that.  A dict-shaped fixture would
    pass while the live path failed.

    Built from the installed ``streamlit`` module rather than hand-rolled, so this
    keeps testing the contract if the shape is ever serialised differently.
    """
    plotly_state = pytest.importorskip(
        "streamlit.elements.plotly_chart",
        reason="streamlit plotly selection types unavailable",
    )
    resolve_selection_window = _ns["resolve_selection_window"]
    selection_to_span = _ns["selection_to_span"]

    def state(**shapes):
        base = {"points": [], "point_indices": [], "box": [], "lasso": []}
        base.update(shapes)
        return plotly_state.PlotlyState(
            {"selection": plotly_state.PlotlySelectionState(base)}
        )

    n = 8170
    # An untouched chart yields the empty state, which must read as "nothing yet".
    assert resolve_selection_window(state(), n) is None
    # A real box selection resolves to a real window.
    assert selection_to_span(state(box=[{"x": [4000, 4120]}]), n) == (4000, 4120)
    assert resolve_selection_window(state(box=[{"x": [4000, 4120]}]), n) == (4000, 4121)
    # A lasso is read spatially, not in draw order.
    assert selection_to_span(state(lasso=[{"x": [300, 100, 250, 120]}]), n) == (100, 300)


def test_the_forecast_section_is_wired_into_the_tab():
    """``render_window_tab`` calls the selection section.

    Without this the helper could exist, be correct, and never run -- which is
    exactly the shape of a dead-code bug that no other assertion here would see.

    Repointed at ``render_window_tab`` when the Forecast tab was split: the brush
    section now lives on the *Forecast* tab, so asserting it against
    ``render_forecast_tab`` would pass for the wrong reason the moment anyone moved a
    call between them.
    """
    body = app_block(r"(def render_window_tab\(.*?)\n(?=def |\n# =)",
                     "render_window_tab")
    assert "_forecast_brush_chart(" in body
    assert "_render_window_forecast(" in body


def test_the_two_brushes_use_different_widget_keys():
    """The Forecast brush must not share state with the Price brush.

    One key would mean a brush on the Price tab silently re-aims the Forecast
    tab's selection, so a reader comparing two windows could only ever have one
    selected at a time -- and the wrong one.
    """
    import re

    source = app_source()
    keys = dict(re.findall(
        r'^(PRICE_BRUSH_KEY|FORECAST_BRUSH_KEY)\s*=\s*"([^"]+)"', source, re.M))
    assert set(keys) == {"PRICE_BRUSH_KEY", "FORECAST_BRUSH_KEY"}
    assert keys["PRICE_BRUSH_KEY"] != keys["FORECAST_BRUSH_KEY"]
