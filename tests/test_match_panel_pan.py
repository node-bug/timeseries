"""Tests for panning the Price tab's lower (match) chart.

The request this file exists for: *be able to scroll the lower graph, but only until the
match window is about to leave the chart*.  That is one sentence containing three
separate obligations, and each is a way the feature can be half-built:

* **the drag must pan.**  ``build_price_figure`` gives every display chart
  ``dragmode="zoom"``, so without ``pannable`` the drag draws a box and the reader
  moves nowhere.
* **panning must reveal something.**  The trace is sliced to exactly the visible view, so
  a chart whose only data is its view slides the same line off-screen and ends on blank
  space.  ``pan_data`` widens the arrays while ``range`` stays put.
* **the match must stay on screen.**  ``pan_axis_bounds_for`` clamps the x range to the
  interval of left edges for which the band is still fully visible.

Each helper is `exec`d out of `app.py` by `apphelpers.load_app_functions`, so these run
the app's real code.  A mirror would be worse than useless here: the half-bar axis offset
in `pan_axis_bounds_for` is exactly the sort of detail a hand-written copy silently drops,
which is why the bound assertions below test the *built figure* rather than a formula
restated from its docstring.
"""

from __future__ import annotations

import ast
import textwrap

import numpy as np
import plotly.graph_objects as go  # noqa: F401  (namespace for the exec'd body)
import pytest

from timeseries.pipeline import Pipeline

from apphelpers import app_block, app_source, load_app_functions, price_tab_body
from session_bars import session_bars

#: A window long enough that centring is meaningful and the rolling band is warm.
LENGTH = 60
VIEW = 1950            # 5 sessions at ~390 bars -- the Price tab's widest sidebar view
MATCH_AT = 4000        # a mid-archive match: room to pan both ways

_PIPE = Pipeline.from_frame(session_bars(8190, seed=20260930), length=LENGTH)
N_BARS = _PIPE.n_bars

_ns = load_app_functions(
    {"build_price_figure", "centred_view", "pan_axis_bounds_for", "rebased_view_range",
     "chart_config"},
    namespace={"go": go},
)
build_price_figure = _ns["build_price_figure"]
centred_view = _ns["centred_view"]
pan_axis_bounds_for = _ns["pan_axis_bounds_for"]
rebased_view_range = _ns["rebased_view_range"]
chart_config = _ns["chart_config"]


def _pannable_figure(match_at=MATCH_AT, view=VIEW):
    """The Price tab's lower panel, built exactly as `_render_best_match_pair` builds it.

    ``pan_data`` is the *bounds*, not the archive: the app clips the drawn tape to the
    span the pan can legally reach so that plotly's over-drag zoom (it cannot hard-stop
    a drag -- issue #887) has nowhere unreachable to land.
    """
    m_view = centred_view(N_BARS, match_at, match_at + LENGTH, view)
    bounds = pan_axis_bounds_for(N_BARS, match_at, match_at + LENGTH, view)
    return build_price_figure(
        _PIPE, *m_view,
        query_start=match_at, query_stop=match_at + LENGTH,
        selectable=False, rebase_at=match_at,
        y_range=rebased_view_range(_PIPE, m_view, match_at, 20),
        pan_data=bounds,
        pan_bounds=bounds,
        pannable=True, height=400,
    )


def _visible_bars(axis_range):
    """How many bars the axis actually shows, undoing the half-bar label padding."""
    return int(round(axis_range[1] - axis_range[0]))


class TestPanningIsOffByDefault:
    """Every other chart in the app must be untouched by all of this."""

    def test_default_figure_has_no_bounds_and_no_pan(self):
        fig = build_price_figure(
            _PIPE, MATCH_AT - 500, MATCH_AT - 500 + VIEW,
            query_start=MATCH_AT, query_stop=MATCH_AT + LENGTH,
            selectable=False, height=400,
        )
        assert fig.layout.xaxis.minallowed is None, "a display chart must not clamp its axis"
        assert fig.layout.xaxis.maxallowed is None
        # Unbounded, but still a *drag*: with zoom off app-wide, "zoom" here would be a
        # gesture that draws a rectangle and does nothing.
        assert fig.layout.dragmode == "pan", (
            "a display chart must not claim the zoom drag when no zoom control exists"
        )

    def test_default_figure_carries_only_its_view(self):
        """The widening must be opt-in: data and view agree unless pan_data says otherwise."""
        fig = build_price_figure(
            _PIPE, MATCH_AT - 500, MATCH_AT - 500 + VIEW,
            query_start=MATCH_AT, query_stop=MATCH_AT + LENGTH,
            selectable=False, height=400,
        )
        assert len(fig.data[0].x) == VIEW, (
            "without pan_data the trace must be the view and nothing else -- a chart "
            "that silently carried the whole archive would autoscale against all of it"
        )

    def test_selectable_chart_still_brushes_rather_than_pans(self):
        """The query-defining charts must keep `select` even if asked to pan."""
        fig = build_price_figure(
            _PIPE, MATCH_AT - 500, MATCH_AT - 500 + VIEW,
            selectable=True, pannable=True, height=400,
        )
        assert fig.layout.dragmode == "select", (
            "pannable must not steal the drag from a chart the app reads a query from"
        )


class TestTheDragPans:
    def test_pannable_figure_is_in_pan_mode(self):
        assert _pannable_figure().layout.dragmode == "pan"

    def test_panning_reveals_more_than_the_view(self):
        fig = _pannable_figure()
        visible = _visible_bars(fig.layout.xaxis.range)
        # Wider than the view, so a drag has something to reveal -- but *not* the whole
        # archive: the data is clipped to the pan bounds (see the over-drag test).
        assert len(fig.data[0].x) > visible, (
            "panning a chart whose only data is the visible view reveals nothing -- "
            "it just slides the same line off-screen"
        )
        assert len(fig.data[0].x) < N_BARS, (
            "the drawn tape must be clipped to the pan bounds, not the whole archive"
        )


class TestTheMatchStaysInView:
    """The actual requirement: pan right up to the edge and no further.

    ``minallowed``/``maxallowed`` are **axis endpoints**: Plotly applies the first to
    ``range[0]`` and the second to ``range[1]``.  They are not two left-edge positions,
    and conflating the two is what hid the match off the right of the screen -- see
    ``TestTheDefaultViewIsNotClamped``.
    """

    def _left_edges(self):
        """The pan limits expressed as left edges, which is what the geometry means."""
        fig = _pannable_figure()
        lo, hi = fig.layout.xaxis.minallowed, fig.layout.xaxis.maxallowed
        width = _visible_bars(fig.layout.xaxis.range)
        return lo + 0.5, hi + 0.5 - width, width

    def _band_visible_at(self, left_edge, width, match_at=MATCH_AT):
        return left_edge <= match_at and left_edge + width >= match_at + LENGTH

    def test_band_is_visible_at_both_pan_stops(self):
        leftmost, rightmost, width = self._left_edges()
        assert self._band_visible_at(leftmost, width), "band lost at the far-left stop"
        assert self._band_visible_at(rightmost, width), "band lost at the far-right stop"

    def test_band_is_visible_everywhere_in_between(self):
        """Not just the endpoints -- the bound must hold continuously."""
        leftmost, rightmost, width = self._left_edges()
        for v0 in np.linspace(leftmost, rightmost, 25):
            assert self._band_visible_at(v0, width), (
                f"band left the chart at left edge {v0}"
            )

    def test_bounds_exactly_bracket_the_legal_range(self):
        """One bar beyond either stop drops the band -- so the bounds must be tight.

        Expressed as the two flush cases the geometry produces: panned fully left the
        band's *end* sits at the right edge, panned fully right its *start* sits at the
        left edge.
        """
        leftmost, rightmost, width = self._left_edges()
        assert leftmost + width == pytest.approx(MATCH_AT + LENGTH), (
            "the leftmost pan stop must leave the band's end flush with the right edge"
        )
        assert rightmost == pytest.approx(MATCH_AT), (
            "the rightmost pan stop must leave the band's start flush with the left edge"
        )
        # A single bar further out in either direction loses the band.
        assert not self._band_visible_at(leftmost - 1, width)
        assert not self._band_visible_at(rightmost + 1, width)

    def test_bounds_are_axis_endpoints_not_left_edges(self):
        """``maxallowed`` is the right end of the axis, so it must exceed ``range[1]``.

        Publishing the *left-edge* limit here made Plotly clamp the right edge down to
        ``band_start`` -- truncating the chart and pushing the match off screen.
        """
        fig = _pannable_figure()
        lo, hi = fig.layout.xaxis.minallowed, fig.layout.xaxis.maxallowed
        r0, r1 = fig.layout.xaxis.range
        assert hi > MATCH_AT, (
            "maxallowed is the axis's right endpoint, so it must be past the band's "
            f"start ({MATCH_AT}); got {hi}"
        )
        assert lo < MATCH_AT, (
            f"minallowed must sit left of the band; got {lo}"
        )
        assert (hi - lo) > VIEW, (
            "the bounds span the view plus its travel, not the travel alone"
        )

    def test_travel_is_width_minus_span(self):
        leftmost, rightmost, _ = self._left_edges()
        assert rightmost - leftmost == pytest.approx(VIEW - LENGTH)



    def test_travel_is_position_independent(self):
        """The same number of bars whichever match was found, except at the edges."""
        for match_at in (41, 4000, 7950):
            lo, hi = pan_axis_bounds_for(N_BARS, match_at, match_at + LENGTH, VIEW)
            travel = (hi - lo) - VIEW          # bounds span the view *plus* the travel
            if match_at == MATCH_AT:
                assert travel == pytest.approx(VIEW - LENGTH), "mid-archive is the reference"
            else:
                # Edge matches get *less* travel -- there is no tape on that side -- but
                # never more, and never a negative window.
                assert travel <= VIEW - LENGTH + 1e-9, (
                    f"match at {match_at} claims extra travel"
                )
                assert travel >= 0, f"match at {match_at} cannot pan at all"


class TestTheDefaultViewIsNotClamped:
    """The bug this file now exists partly to prevent.

    ``minallowed``/``maxallowed`` are axis endpoints, and Plotly enforces them against
    the rendered ``range`` -- clamping it when it falls outside.  Publishing the *pan
    limit* (a left-edge position) as ``maxallowed`` therefore clamped the chart's right
    edge down to ``band_start``: 1,287 intended bars became 583, and the match window
    was pushed entirely off the right of the screen.  The panel still rendered, so it
    read as "the match is missing" rather than as an error.
    """

    def test_default_view_sits_inside_the_bounds(self):
        for match_at in (41, MATCH_AT, N_BARS - LENGTH - 1):
            view = centred_view(N_BARS, match_at, match_at + LENGTH, VIEW)
            lo, hi = pan_axis_bounds_for(N_BARS, match_at, match_at + LENGTH, VIEW)
            r0, r1 = view[0] - 0.5, view[1] - 0.5
            assert r0 >= lo, f"match {match_at}: left edge below minallowed"
            assert r1 <= hi, f"match {match_at}: right edge above maxallowed"

    def test_chart_shows_its_full_width_after_clamping(self):
        """Simulate Plotly's clamp and confirm nothing is lost."""
        fig = _pannable_figure()
        xa = fig.layout.xaxis
        r0 = max(xa.range[0], xa.minallowed)
        r1 = min(xa.range[1], xa.maxallowed)
        assert int(round(r1 - r0)) == VIEW, (
            f"the chart shows {int(round(r1 - r0))} of {VIEW} bars -- the bounds are "
            "clamping the default view"
        )

    def test_band_is_inside_the_rendered_window(self):
        fig = _pannable_figure()
        xa = fig.layout.xaxis
        r0 = max(xa.range[0], xa.minallowed)
        r1 = min(xa.range[1], xa.maxallowed)
        assert r0 <= MATCH_AT and r1 >= MATCH_AT + LENGTH, (
            "the match window is not on screen once the bounds are applied -- this is "
            "the reported symptom"
        )


class TestCentring:
    def test_view_width_is_preserved(self):
        """The user's constraint: the same number of bars visible as the chart above."""
        view = centred_view(N_BARS, MATCH_AT, MATCH_AT + LENGTH, VIEW)
        assert view[1] - view[0] == VIEW

    def test_band_is_centred(self):
        view = centred_view(N_BARS, MATCH_AT, MATCH_AT + LENGTH, VIEW)
        left_room = MATCH_AT - view[0]
        right_room = view[1] - (MATCH_AT + LENGTH)
        assert abs(left_room - right_room) <= 1, f"off-centre by {left_room - right_room} bars"

    def test_centring_is_the_maximum_travel_position(self):
        """Centre = the midpoint of the legal range, so travel is symmetric."""
        lo, hi = pan_axis_bounds_for(N_BARS, MATCH_AT, MATCH_AT + LENGTH, VIEW)
        view = centred_view(N_BARS, MATCH_AT, MATCH_AT + LENGTH, VIEW)
        # `lo`/`hi` are axis endpoints; the legal left edges are `lo` and `hi - VIEW`.
        mid = (lo + (hi - VIEW)) / 2.0 + 0.5
        assert abs(view[0] - mid) <= 1.0, (
            f"centred view {view[0]} is not the midpoint {mid} -- travel is asymmetric"
        )


class TestYAxisIsPinnedToTheView:
    """Widening the data must not stretch the match into a stripe.

    ``y_range=None`` means autoscale, and autoscale is taken over the *traces* -- which
    for this panel is the whole archive.  Left alone it squashed the match's own move by
    2.4x on the fixture.
    """

    def test_pannable_panel_does_not_autoscale(self):
        assert _pannable_figure().layout.yaxis.range is not None, (
            "the pannable panel carries the whole tape, so an autoscale is taken over "
            "all of it and the matched move flattens into a stripe"
        )

    def test_pinned_range_covers_the_view(self):
        view = centred_view(N_BARS, MATCH_AT, MATCH_AT + LENGTH, VIEW)
        rng = rebased_view_range(_PIPE, view, MATCH_AT, 20)
        window = _PIPE.close[view[0]:view[1]] / _PIPE.close[MATCH_AT] - 1.0
        assert rng[0] <= window.min() * 100.0, "the view's low is clipped off the axis"
        assert rng[1] >= window.max() * 100.0, "the view's high is clipped off the axis"

    def test_range_is_much_tighter_than_whole_archive_autoscale(self):
        view = centred_view(N_BARS, MATCH_AT, MATCH_AT + LENGTH, VIEW)
        tight = rebased_view_range(_PIPE, view, MATCH_AT, 20)
        whole = rebased_view_range(_PIPE, (0, N_BARS), MATCH_AT, 20)
        assert (tight[1] - tight[0]) < (whole[1] - whole[0]), (
            "the view-pinned range is no tighter than the archive-wide one, so it is "
            "not doing its job"
        )

    def test_degenerate_input_falls_back_to_autoscale(self):
        assert rebased_view_range(_PIPE, (0, 0), MATCH_AT, 20) is None


class TestPriceTabWiring:
    """Source-level guards: the right panel, and only the right panel, is pannable."""

    def test_price_tab_asks_for_a_pannable_match_panel(self):
        """The opt-in must be an *argument*, not a mention in a comment.

        Asserted on the parsed call rather than on the text: the block's own comment
        discusses ``pannable`` in prose, so a substring search passes even after the
        argument has been deleted.
        """
        # `price_tab_body` is an indented *fragment* of `main()`, so it has to be
        # dedented before it will parse.
        fragment = textwrap.dedent(price_tab_body())
        calls = [n for n in ast.walk(ast.parse(fragment))
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "render_matches_tab"]
        assert len(calls) == 1, "expected the Price tab to render exactly one match panel"
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
        assert kwargs.get("pannable") == "True", (
            "the Price tab's match panel is not pannable, so the lower graph still "
            f"draws a zoom box when dragged; got {kwargs.get('pannable')!r}"
        )
        assert kwargs.get("compact") == "True", "the Price tab draws the compact panel"

    def test_price_tab_still_passes_the_shared_width(self):
        """Consistent bar count depends on the width surviving the change."""
        body = price_tab_body()
        assert body.count("chart_width=price_width") == 2, (
            "both renderers must still receive price_width, or the two panels show "
            "different numbers of bars"
        )

    def test_query_chart_is_not_made_pannable(self):
        """The upper chart defines the query; a pan drag there would break the brush."""
        # The lookahead used to name ``def render_chart_tab``.  That tab is gone, so
        # the slice has to stop at whatever function follows instead -- otherwise the
        # pattern matches nothing and this assertion fails for a reason that has
        # nothing to do with panning.
        #
        # Asserted on the *parsed* body, not on its text.  A plain ``"pannable" not in
        # body`` check passes right up until someone documents a decision in prose --
        # which is what happened here: the ``render_price_tab`` docstring now explains
        # that the pannable panel below drops the shared anchor, and that sentence made
        # this test fail on a word that is not a call.  Parsing looks at the code.
        fragment = textwrap.dedent(
            app_block(r"def render_price_tab\(.*?\n(?=def )", "render_price_tab body"))
        tree = ast.parse(fragment)

        # The parameter must not exist at all...
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        assert "pannable" not in args, (
            "render_price_tab grew a `pannable` parameter -- its chart is the one the "
            "app reads the query window off, so a pan drag there replaces the brush"
        )

        # ...and no call anywhere in it may set the keyword.
        kwarged = [
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and any(kw.arg == "pannable" for kw in n.keywords)
        ]
        assert not kwarged, (
            f"render_price_tab passes pannable= to {kwarged} -- a pan drag on the "
            "query chart replaces the brush and the window can never be drawn again"
        )

    def test_match_panel_passes_the_pan_arguments_through(self):
        """``pannable`` must reach the figure, not just the signature."""
        body = app_block(r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
                         "_render_best_match_pair body")
        for needed in ("pan_data=m_pan_data", "pan_bounds=m_pan_bounds",
                       "pannable=pannable"):
            assert needed in body, (
                f"{needed!r} is missing from the match panel -- pannable was threaded "
                "in but never reaches the figure"
            )
        assert "chart_config(scroll_zoom=" not in body, (
            "the match panel must not re-enable scroll-zoom; zoom is off app-wide"
        )

    def test_bounds_are_computed_not_just_forwarded(self):
        """Guards the gap a pure forwarding test leaves open.

        Deleting the ``pan_bounds_for`` call leaves every argument still *passed*
        through -- and the chart silently becomes pannable with no limits at all, so a
        drag walks the match straight off the screen.  This is the mutation that left
        the whole suite green.
        """
        body = app_block(r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
                         "_render_best_match_pair body")
        assert "pan_axis_bounds_for(pipe.n_bars, m.start, m.stop, chart_width)" in body, (
            "the match panel must derive its bounds from the match's own span -- "
            "without this call the chart is pannable but unbounded"
        )
        assert "centred_view(pipe.n_bars, m.start, m.stop, chart_width)" in body, (
            "the pannable panel must centre its own band, or the travel is asymmetric "
            "and the match sits hard against one edge on load"
        )

    def test_pan_data_widens_the_data_not_the_view(self):
        """``pan_data`` must exceed the view; the *view* stays at ``chart_width``.

        Widening both would show more bars than the chart above and break the
        equal-scale comparison the Price tab exists for.
        """
        body = app_block(r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
                         "_render_best_match_pair body")
        assert "m_view = centred_view(pipe.n_bars, m.start, m.stop, chart_width)" in body, (
            "the pannable view must stay at chart_width bars wide so both Price-tab "
            "charts show the same number of bars"
        )
        fig = _pannable_figure()
        assert len(fig.data[0].x) > _visible_bars(fig.layout.xaxis.range), (
            "pan_data must be wider than the view, or a pan reveals nothing"
        )

    def test_pan_data_is_clipped_to_the_bounds_not_the_archive(self):
        """The over-drag zoom must have nowhere unreachable to land.

        plotly.js cannot hard-stop a pan (issue #887), so a drag past a limit zooms.
        With the whole tape drawn, that zoom can walk the window off the match and
        change the visible bar count.  Clipping the data to the bounds confines it to
        the span the reader can already pan across.
        """
        body = app_block(r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
                         "_render_best_match_pair body")
        assert "(m_pan_bounds[0], m_pan_bounds[1])" in body, (
            "pan_data must be the bounds themselves -- drawing the whole archive "
            "gives an over-drag zoom somewhere unreachable to go"
        )
        fig = _pannable_figure()
        lo, hi = fig.layout.xaxis.minallowed, fig.layout.xaxis.maxallowed
        first, last = fig.data[0].x[0], fig.data[0].x[-1]
        assert first >= lo - 1, f"data starts at {first}, left of the bound {lo}"
        assert last <= hi + 1, f"data ends at {last}, right of the bound {hi}"
        # ...but it must still be *wider* than the view, or there is nothing to pan to.
        assert last - first > _visible_bars(fig.layout.xaxis.range)

    def test_no_chart_opts_into_scroll_zoom(self):
        """The wheel scrolls the page everywhere, including the pannable panel.

        Scroll-zoom used to be an opt-in only the match panel passed.  It was removed:
        a wheel that zooms breaks the equal-bar-count comparison with the query chart
        above, and the reader who wanted to move along x was already served by the drag.
        """
        # Parsed rather than grepped: `scroll_zoom` also appeared in the parameter's
        # own signature and in the docstring, and a substring search cannot tell those
        # apart from an actual argument at a call site.
        call_sites = []
        for node in ast.walk(ast.parse(app_source())):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "chart_config"):
                for kw in node.keywords:
                    if kw.arg == "scroll_zoom":
                        call_sites.append(ast.unparse(kw.value))
        assert call_sites == [], (
            f"no chart may opt into scroll-zoom; found {call_sites}"
        )

    def test_scroll_zoom_flag_is_gone_from_the_signature(self):
        """A dead parameter is worse than none: it advertises a gesture that does nothing."""
        body = app_block(r"def chart_config\(.*?\n(?=def session_spans)",
                         "chart_config body")
        header = body.split(")")[0]
        assert "scroll_zoom" not in header, (
            "chart_config still takes scroll_zoom but nothing passes it -- either "
            "restore the opt-in or drop the parameter"
        )

    def test_every_zoom_modebar_button_is_stripped(self):
        """All five routes into zoom, on every chart, including the brushable ones.

        ``resetScale2d`` is the subtle one: with a pinned ``range`` it is the only
        control that can silently undo the centring ``centred_view`` chose.
        """
        for selectable in (False, True):
            cfg = chart_config(selectable=selectable)
            removed = cfg["modeBarButtonsToRemove"]
            for button in ("zoomIn", "zoomOut", "zoom2d", "autoScale2d", "resetScale2d"):
                assert button in removed, (
                    f"{button} is still advertised on a chart (selectable={selectable})"
                )
            assert cfg["scrollZoom"] is False, (
                f"scroll-zoom is on for selectable={selectable}"
            )

    def test_pan_is_still_available_after_removing_zoom(self):
        """Removing zoom must not remove the thing the user asked for."""
        fig = _pannable_figure()
        assert fig.layout.dragmode == "pan", "the drag no longer pans"
        assert fig.layout.xaxis.minallowed is not None, "the pan lost its bounds"

    def test_selectable_charts_keep_their_brush_buttons(self):
        """Stripping zoom must not take the selection buttons with it."""
        cfg = chart_config(selectable=True)
        for button in ("select2d", "lasso2d"):
            assert button not in cfg["modeBarButtonsToRemove"], (
                f"{button} was removed from a chart that needs it"
            )
        cfg = chart_config(selectable=False)
        for button in ("select2d", "lasso2d"):
            assert button in cfg["modeBarButtonsToRemove"], (
                f"a display chart still advertises {button}, which it cannot read"
            )
