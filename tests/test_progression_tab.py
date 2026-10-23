"""Tests for the **Progression** tab.

The tab asks one question at many lookbacks: for every length ``L`` from 10 bars to
500, find the historical windows most similar to the archive's *last* ``L`` bars, and
project what happened next.  All of those projections are drawn on one chart at once.

Three things are worth pinning here, and they are pinned at three different levels:

* the **library** (:mod:`timeseries.progression`) -- the sweep produces one path per
  length, each of the requested length, with the shortfall reported rather than
  hidden;
* the **figure** -- fifty traces are drawn, in percent from the anchor, with the
  consensus on top;
* the **wiring** -- the tab exists, is rendered by name, and sits where it must sit
  so that adding it cannot invalidate the brush's stored selection.

The wiring tests are the load-bearing ones for a *new* tab.  Adding a chart changes
how many charts precede the Forecast tab's brush, and a plotly chart's element id
hashes the page's chart list -- so a tab inserted in the wrong place would silently
drop every brush selection a reader had made.  That is a failure with no exception
and no message, which is exactly why it is pinned rather than trusted.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from timeseries.pipeline import pipeline_from_frame
from timeseries.progression import (
    MAX_PROGRESSION_LENGTH,
    ProgressionSweep,
    default_lengths,
    progression_paths,
)
from timeseries.timeframes import get_timeframe

from apphelpers import app_block, app_source, load_app_functions
from session_bars import session_bars


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bars():
    """Synthetic session-structured bars: real sessions, real overnight closures.

    The session structure is what matters, not the price path: the §BX forward-
    horizon mask keys on session boundaries, so a fixture without closures cannot
    exercise the censoring this tab reports on.
    """
    return session_bars(4000, seed=7)


@pytest.fixture(scope="module")
def pipe(bars):
    """A pipeline over the synthetic archive."""
    return pipeline_from_frame(
        "TEST@1m", bars, 240, timeframe=get_timeframe("1m"),
    )


@pytest.fixture(scope="module")
def sweep(pipe):
    """A real sweep over the synthetic archive, at a short horizon for speed.

    The horizon is short here purely so the suite stays quick; the *lengths* are the
    real ones, because the length range is the thing under test.
    """
    lengths = default_lengths(pipe.n_bars, step=50)
    return progression_paths(pipe, lengths=lengths, horizon=60, k=10)


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #
class TestDefaultLengths:
    """The lookback lengths the sweep asks for."""

    def test_it_starts_at_ten_and_reaches_the_ceiling(self):
        lengths = default_lengths(8000, step=10)
        assert lengths[0] == 10
        assert lengths[-1] == MAX_PROGRESSION_LENGTH
        assert MAX_PROGRESSION_LENGTH == 500

    def test_it_covers_the_requested_range_in_ten_bar_steps(self):
        lengths = default_lengths(8000, step=10)
        assert lengths[:3] == [10, 20, 30]
        # The range is *not* truncated by the archive's length: the sweep's point is
        # to show how the answer changes with the lookback, and a range that moved
        # with the archive would answer a different question on a short archive.
        assert 500 in lengths

    def test_a_short_archive_still_gets_the_whole_range(self):
        lengths = default_lengths(600, step=10)
        assert lengths[-1] == MAX_PROGRESSION_LENGTH

    def test_the_step_is_honoured(self):
        lengths = default_lengths(8000, step=25)
        assert lengths[:4] == [10, 35, 60, 85]

    def test_a_step_that_skips_the_ceiling_still_includes_it(self):
        # 10 + 10*k never lands on 500, so the ceiling is appended explicitly: the
        # longest lookback in range is the one that most often differs from the one
        # before it.
        lengths = default_lengths(8000, step=10, max_length=500)
        assert lengths[-1] == 500
        assert lengths == sorted(set(lengths))


class TestProgressionPaths:
    """The sweep itself."""

    def test_it_returns_one_entry_per_requested_length(self, pipe, sweep):
        assert len(sweep.paths) == len(sweep.lengths)

    def test_every_path_is_the_length_that_was_asked_for(self, sweep):
        for p, length in zip(sweep.paths, sweep.lengths):
            assert p.length == length

    def test_each_path_projects_the_whole_horizon(self, sweep):
        for p in sweep.answered:
            assert p.path.median.shape == (sweep.horizon + 1,)
            assert p.path.offsets[-1] == sweep.horizon

    def test_the_median_is_finite_and_starts_at_zero(self, sweep):
        # Offset 0 is the anchor bar itself, exactly 0.0 by construction -- the bar
        # history and projection share.
        for p in sweep.answered:
            assert np.isfinite(p.path.median).all()
            assert p.path.median[0] == 0.0

    def test_a_path_carries_the_best_match_too(self, sweep):
        answered = sweep.answered
        assert answered, "the sweep answered nothing at all"
        for p in answered:
            assert p.best is not None
            assert p.best.shape == (sweep.horizon + 1,)
            assert p.best[0] == 0.0
            assert np.isfinite(p.best).all()

    def test_the_best_match_is_identified(self, sweep):
        for p in sweep.answered:
            assert np.isfinite(p.distance)
            assert 0.0 <= p.percentile <= 1.0
            assert p.n_candidates > 0

    def test_the_matrix_is_one_row_per_length(self, sweep):
        assert sweep.matrix.shape == (len(sweep.paths), sweep.horizon + 1)

    def test_the_consensus_is_finite_wherever_the_lengths_answered(self, sweep):
        consensus = sweep.consensus
        assert consensus.shape == (sweep.horizon + 1,)
        assert np.isfinite(consensus).all()

    def test_the_consensus_is_the_median_of_the_answered_lengths(self, sweep):
        mat = sweep.matrix
        finite = mat[np.isfinite(mat).all(axis=1)]
        if finite.size:
            assert np.allclose(sweep.consensus, np.median(finite, axis=0))


class TestProgressionShortfalls:
    """What the sweep says when it cannot answer.

    The rule the whole app is built on is that a number is never shown without the
    sample behind it, and that a missing number is stated rather than drawn as a
    flat line.  These tests are the library half of that.
    """

    def test_a_length_past_the_archive_is_reported_not_guessed(self, pipe):
        sweep = progression_paths(
            pipe, lengths=[10, pipe.n_bars + 100], horizon=30, k=5,
        )
        assert sweep.paths[1].ok is False
        assert sweep.paths[1].reason == "archive too short"

    def test_a_failed_length_is_still_listed(self, pipe):
        sweep = progression_paths(
            pipe, lengths=[10, pipe.n_bars + 100], horizon=30, k=5,
        )
        assert len(sweep.paths) == 2
        assert sweep.failed and sweep.failed[0].length == pipe.n_bars + 100

    def test_the_shortfall_is_reported_when_matches_are_lost(self, sweep):
        # A long lookback on 1-minute data cannot fill thirty slots: non-max
        # suppression needs one full lookback of separation between accepted
        # windows.  The count is reported rather than hidden, because a median over
        # seven windows is a thinner claim than one over thirty.
        for p in sweep.answered:
            assert p.n_matches <= p.n_requested
            assert p.n_found <= p.n_requested

    def test_the_requested_count_is_the_callers_k(self, pipe):
        sweep = progression_paths(pipe, lengths=[60], horizon=30, k=7)
        assert sweep.paths[0].n_requested == 7


class TestProgressionSweep:
    """The container's own summaries."""

    def test_answered_and_failed_partition_the_paths(self, sweep):
        assert len(sweep.answered) + len(sweep.failed) == len(sweep.paths)

    def test_an_empty_sweep_has_no_failures(self):
        empty = ProgressionSweep(paths=[], lengths=[], horizon=0, n_bars=0)
        assert empty.answered == []
        assert empty.failed == []
        assert empty.matrix.shape == (0, 1)

    def test_the_dict_view_reports_the_counts(self, sweep):
        d = sweep.as_dict()
        assert d["n_answered"] == len(sweep.answered)
        assert d["n_failed"] == len(sweep.failed)
        assert d["horizon"] == sweep.horizon


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def _figure_and_sweep(pipe, sweep):
    """Build the chart with the app's own helper, loaded out of ``app.py``."""
    ns = load_app_functions(
        {
            "build_progression_figure",
            "_proj_ticktext",
            "stamp_label",
            "chart_config",
        },
        namespace={
            "go": __import__("plotly.graph_objects", fromlist=["go"]),
            "px": __import__("plotly.express", fromlist=["px"]),
            "pd": __import__("pandas"),
        },
    )
    fig = ns["build_progression_figure"](pipe, sweep, history_bars=500)
    return fig


class TestProgressionFigure:
    """What the chart draws."""

    @pytest.fixture(scope="class")
    @classmethod
    def figure(cls, request):
        pipe = request.getfixturevalue("pipe")
        sweep = request.getfixturevalue("sweep")
        return _figure_and_sweep(pipe, sweep)

    def test_it_draws_one_trace_per_answered_length(self, figure, sweep):
        # The history, the per-length projections, and the consensus.
        assert len(figure.data) == 1 + len(sweep.answered) + 1

    def test_every_projection_trace_is_the_horizon_long(self, figure, sweep):
        for trace in figure.data[1:-1]:
            assert len(trace.x) == sweep.horizon + 1
            assert len(trace.y) == sweep.horizon + 1

    def test_the_projections_are_in_percent_from_the_anchor(self, figure, sweep):
        # Every projection starts at exactly 0% -- the anchor bar -- so the fifty
        # traces share one axis and one origin.
        for trace in figure.data[1:-1]:
            assert trace.y[0] == 0.0

    def test_the_history_is_drawn_first(self, figure):
        assert "recent tape" in figure.data[0].name

    def test_the_consensus_is_drawn_last(self, figure, sweep):
        assert "median across" in figure.data[-1].name

    def test_the_boundary_line_stands_on_the_first_projected_bar(self, figure, pipe):
        # The anchor is the archive's last bar, and offset 0 *is* the anchor, so the
        # first bar that belongs to the projection alone is the one after it.
        shapes = [s for s in figure.layout.shapes if s.type == "line"]
        assert shapes, "the chart has no boundary line"
        assert shapes[0].x0 == pipe.n_bars

    def test_the_axis_covers_the_history_and_the_projection(self, figure, pipe, sweep):
        # The history is the archive's *last* 500 bars -- not the whole archive --
        # so the axis starts just before that span's left edge, not at zero.
        lo, hi = figure.layout.xaxis.range
        assert lo <= pipe.n_bars - 500
        assert hi >= pipe.n_bars + sweep.horizon

    def test_the_history_is_the_recent_bars_not_the_whole_archive(self, figure, pipe):
        # 500 recent bars, as asked for -- not the archive, which would squash the
        # projections into a sliver at the right-hand edge.
        hist = figure.data[0]
        assert len(hist.x) == 500
        assert hist.x[0] == pipe.n_bars - 500
        assert hist.x[-1] == pipe.n_bars - 1

    def test_the_y_axis_is_named_in_percent(self, figure):
        assert "percent" in figure.layout.yaxis.title.text.lower()


# --------------------------------------------------------------------------- #
# The wiring
# --------------------------------------------------------------------------- #
class TestProgressionWiring:
    """The tab is present, rendered by name, and in the right place.

    The position matters more than it looks: a plotly chart's element id hashes the
    page's chart list, so moving a chart relative to another changes that chart's
    identity and drops any stored selection on it.  The Forecast tab's brush is a
    live selection, so inserting a chart *before* it would break the brush silently.
    """

    def test_the_tab_is_in_the_tab_order(self):
        order = app_block(r"TAB_ORDER\s*=\s*\((.*?)\)", "TAB_ORDER")
        names = re.findall(r'"([^"]+)"', order)
        assert "Progression" in names

    def test_the_tab_is_unpacked_by_name(self):
        # The unpack must be by name, not positionally: a positional unpack that was
        # not updated when a tab was added raises "too many values to unpack" and
        # takes the whole page down.
        source = app_source()
        assert 'tab_progression = tab_by_name["Progression"]' in source

    def test_the_body_is_rendered(self):
        source = app_source()
        assert re.search(r"^    with tab_progression:", source, re.M)

    def test_the_not_ready_branch_has_a_stub(self):
        source = app_source()
        assert 'with tabs[TAB_ORDER.index("Progression")]:' in source

    def test_it_sits_after_forecast(self):
        # **After**, not before.  A chart inserted before the Forecast tab's brush
        # would change how many charts precede the brush, changing the brush chart's
        # element id and silently dropping every selection a reader had made.
        order = app_block(r"TAB_ORDER\s*=\s*\((.*?)\)", "TAB_ORDER")
        names = re.findall(r'"([^"]+)"', order)
        assert names.index("Progression") > names.index("Forecast")

    def test_it_does_not_disturb_the_pinned_forecast_ordering(self):
        # The existing invariant, restated here so a future edit to TAB_ORDER that
        # breaks it fails next to the tab that caused it.
        order = app_block(r"TAB_ORDER\s*=\s*\((.*?)\)", "TAB_ORDER")
        names = re.findall(r'"([^"]+)"', order)
        assert names.index("Forecast") == names.index("Projection") + 1

    def test_the_guide_has_an_entry(self):
        # A guide key with no entry renders nothing at all, so the tab would ship
        # with no explanation of what its fifty traces mean.
        block = app_block(r'TAB_GUIDE: Dict\[str, str\] = \{(.*?)\n\}', "TAB_GUIDE")
        assert '"Progression":' in block

    def test_the_guide_copy_names_the_horizon_token(self):
        # The horizon is quoted through a token so the copy cannot go stale when the
        # constant changes -- the same rule the other guide tokens follow.
        source = app_source()
        assert "[[PROGRESSION_HORIZON]]" in source
        assert 'replace("[[PROGRESSION_HORIZON]]"' in source

    def test_the_render_is_delegated_to_a_helper(self):
        # The tab's body must not inline the chart; the renderer is a separate
        # function so it can be tested and so the tab body stays readable.
        source = app_source()
        assert "def render_progression_tab(" in source
        assert "def build_progression_figure(" in source


class TestProgressionDoesNotDuplicateControls:
    """The tab must not draw a control another tab already owns.

    ``st.tabs`` renders every body on every pass, so a widget key registered twice
    in one pass raises ``StreamlitDuplicateElementKey`` and takes down the whole
    page -- not one tab.  The Projection tab owns the *Projection bars* slider and
    this tab must read it, never draw it.
    """

    def test_the_progression_tab_draws_no_widget(self):
        block = app_block(
            r"(def render_progression_tab\(.*?)\n(?=def |\n# =)",
            "render_progression_tab",
        )
        for widget in ("st.slider(", "st.number_input(", "st.button(",
                       "st.selectbox(", "st.text_input("):
            assert widget not in block, (
                "the Progression tab must not draw %s; st.tabs renders every body on "
                "every pass, so a duplicated widget key takes down the page" % widget
            )

    def test_the_progression_tab_draws_no_ticker_input(self):
        block = app_block(
            r"(def render_progression_tab\(.*?)\n(?=def |\n# =)",
            "render_progression_tab",
        )
        assert "render_scope_ticker(" not in block
