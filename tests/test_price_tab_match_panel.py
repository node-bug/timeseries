"""Tests for the Price tab's match panel -- the view that answers "when has something
like this looked like this?".

`app.py` is a Streamlit script whose import runs a whole UI, so it is not directly
importable under pytest.  The three pure helpers that decide where the query band and
the match band are drawn -- `span_context_bounds`, `shared_anchor` and `aligned_view`
-- are therefore `exec`d straight out of the source by `apphelpers.load_geometry`,
so these tests exercise the app's own code rather than a copy of it.

That is a deliberate change.  This file used to re-implement all three as local
mirrors and assert, in `TestStaysInSyncWithApp`, that a few lines inside each real
body still appeared.  Those guards could not see a *signature* change:
`span_context_bounds` grew a `Pipeline` first parameter, the mirror kept the old
`n_bars` one, and all 42 tests stayed green while testing a function the app no
longer calls that way.  Running the real code means a signature change is a test
failure, which is the whole point of the file.

What matters is that the scored band sits *centred* in the visible window.  If it does
not, the query panel and the match panel show different amounts of lead-in, the eye
compares the wrong edges, and a match can look convincing purely because its window was
cropped favourably.
"""

from __future__ import annotations

import ast
import re
import textwrap

import plotly.graph_objects as go
from plotly.subplots import make_subplots as _make_subplots

from timeseries.pipeline import Pipeline

from apphelpers import (
    app_block,
    app_source,
    app_text,
    load_geometry,
    price_tab_body,
)
from session_bars import session_bars

#: Bars in the standard fixture.  Chosen to exceed ``L + 100`` comfortably so the
#: pipeline is ready, and to be a whole number of 390-bar sessions so the archive has
#: real overnight closures -- the ``N`` the geometry assertions use is *derived* from
#: it, not hard-coded, which is what the old ``N = 8049`` (this file's bar count minus
#: the feature warm-up) silently was.
FIXTURE_BARS = 8190
FIXTURE_SEED = 20260930

#: Bar count after feature cleaning, i.e. what the geometry actually operates on.
N_BARS = len(Pipeline.from_frame(
    session_bars(FIXTURE_BARS, seed=FIXTURE_SEED), length=60).bars)

# The app's own geometry, loaded once at import.
_G = load_geometry()
shared_anchor = _G["shared_anchor"]
aligned_view = _G["aligned_view"]
_app_bounds = _G["span_context_bounds"]


class _Pipe:
    """Minimal stand-in: ``span_context_bounds`` reads only ``n_bars`` off its argument.

    The app passes a real :class:`~timeseries.pipeline.Pipeline`, and the tests still
    pass one, so a change to that parameter is a failure here rather than something a
    stale mirror absorbs.
    """

    def __init__(self, n_bars: int):
        self.n_bars = int(n_bars)


def span_context_bounds(n_bars: int, start: int, stop: int, pad: int):
    """``app.span_context_bounds`` with the pipeline argument supplied.

    A one-line delegate, not a re-implementation: every bar of arithmetic comes from
    the app.  It exists only because ~12 call sites here talk in bar counts, and
    wrapping each one in ``_Pipe(...)`` would be noise.
    """
    return _app_bounds(_Pipe(n_bars), start, stop, pad)


def centred_view(n_bars: int, start: int, stop: int, width: int):
    """Independent centring, kept only as a counter-example for the misalignment guards.

    `aligned_view` replaced this in the app: each panel used to centre its own band,
    which put the live query flush right against an interior match.  Two tests below
    assert that this *would* misalign, so that the alignment tests are known to be
    load-bearing rather than passing for an unrelated reason.
    """
    return aligned_view(n_bars, start, stop, width, (width - (stop - start)) // 2)


def padding(n_bars: int, start: int, stop: int, pad: int):
    lo, hi = span_context_bounds(n_bars, start, stop, pad)
    return start - lo, hi - stop


class TestBandIsCentred:
    N, L, PAD = 8049, 60, 20

    def test_interior_match_is_exactly_centred(self):
        """The common case: match well inside the archive, equal padding both sides."""
        for start in (500, 2000, 4000, 7000):
            left, right = padding(self.N, start, start + self.L, self.PAD)
            assert left == right == self.PAD, f"start={start}: {left}/{right}"

    def test_every_interior_start_is_centred(self):
        """Exhaustive over all starts with room on both sides."""
        bad = [
            s for s in range(self.PAD, self.N - self.L - self.PAD + 1)
            if padding(self.N, s, s + self.L, self.PAD)[0]
            != padding(self.N, s, s + self.L, self.PAD)[1]
        ]
        assert not bad, f"{len(bad)} interior windows off-centre, e.g. {bad[:5]}"

    def test_window_has_the_requested_width(self):
        """A fixed total width, so both panels are the same width and line up."""
        for start in (100, 4000, 7000):
            lo, hi = span_context_bounds(self.N, start, start + self.L, self.PAD)
            assert hi - lo == self.L + 2 * self.PAD

    def test_window_is_inside_the_archive(self):
        for start in (0, 1, 20, 4000, self.N - self.L, self.N - 1):
            lo, hi = span_context_bounds(self.N, start, start + self.L, self.PAD)
            assert 0 <= lo < hi <= self.N

    def test_band_is_always_fully_visible(self):
        """The scored span must lie inside the view -- otherwise the match is cropped."""
        for start in range(0, self.N - self.L + 1, 37):
            lo, hi = span_context_bounds(self.N, start, start + self.L, self.PAD)
            assert lo <= start and start + self.L <= hi

    def test_edge_asymmetry_is_bounded_by_two_pads(self):
        """Near an edge centring is impossible; the asymmetry is arithmetic, not a bug.

        A window cannot be centred on a band with no room on one side.  At the very
        first bar the band has *no* lead-in available, so it sits flush against the
        archive start with the entire ``2 * pad`` of context on its right instead.
        That is the worst case, and it is bounded by exactly that: the band is never
        further from flush than the pad allows.
        """
        worst, at = 0, None
        for start in range(0, self.N - self.L + 1):
            left, right = padding(self.N, start, start + self.L, self.PAD)
            if abs(left - right) > worst:
                worst, at = abs(left - right), start
        assert worst == 2 * self.PAD, (
            f"worst left/right imbalance {worst} (at start={at}); expected exactly "
            f"{2 * self.PAD} = 2*pad, the band flush against an archive edge. A "
            "different value means the centring logic changed."
        )

    def test_worst_case_is_the_very_first_bar(self):
        """No lead-in exists there, so all context falls on the right."""
        assert padding(self.N, 0, self.L, self.PAD) == (0, 2 * self.PAD)
        assert padding(self.N, self.N - self.L, self.N, self.PAD) == (2 * self.PAD, 0)

    def test_only_edge_windows_are_off_centre(self):
        """The honest scope of the limitation: exactly the first/last ``pad`` starts."""
        bad = [
            s for s in range(0, self.N - self.L + 1)
            if padding(self.N, s, s + self.L, self.PAD)[0]
            != padding(self.N, s, s + self.L, self.PAD)[1]
        ]
        expected = set(range(0, self.PAD)) | set(
            range(self.N - self.L - self.PAD + 1, self.N - self.L + 1)
        )
        assert set(bad) == expected, (
            f"{len(bad)} off-centre windows; expected only the {len(expected)} at the "
            "archive edges. A new off-centre window means the centring logic regressed."
        )


class TestDegenerateInputs:
    def test_pad_zero_leaves_no_context(self):
        lo, hi = span_context_bounds(1000, 400, 460, 0)
        assert (lo, hi) == (400, 460)
        assert padding(1000, 400, 460, 0) == (0, 0)

    def test_band_near_edge_gets_what_room_exists(self):
        """A 120-bar archive cannot supply 50 bars of lead-in for a 60-bar window.

        The requested width is 100, which does fit, so the window is built at full
        width and then clamped -- the band ends up flush against the archive start with
        less lead-in than asked for.  Showing the whole 120 bars instead would crop the
        right-hand context for no benefit.
        """
        lo, hi = span_context_bounds(120, 10, 70, 20)
        assert (lo, hi) == (0, 100), "full requested width, clamped into range"
        assert padding(120, 10, 70, 20) == (10, 30)

    def test_archive_shorter_than_the_window_shows_everything(self):
        """Nothing to pad: the window cannot exist, so show the whole archive."""
        assert span_context_bounds(80, 5, 65, 20) == (0, 80)

    def test_window_equal_to_archive_shows_all(self):
        n, L = 100, 60
        lo, hi = span_context_bounds(n, 0, L, 20)
        assert (lo, hi) == (0, n)

    def test_match_at_very_first_bar_stays_visible(self):
        lo, hi = span_context_bounds(1000, 0, 60, 20)
        assert lo == 0 and hi >= 60

    def test_match_at_very_last_bar_stays_visible(self):
        n, L = 1000, 60
        lo, hi = span_context_bounds(n, n - L, n, 20)
        assert hi == n and lo <= n - L

    def test_negative_pad_is_clamped_not_crashed(self):
        lo, hi = span_context_bounds(1000, 400, 460, -50)
        assert lo <= 400 and hi >= 460


class TestSameWidthAsQueryChart:
    """The query chart above and the match chart below must span the same bars.

    A view is an *absolute bar range*, not a width, so there are two ways to get this
    wrong and one of them looks correct: passing the query's ``(lo, hi)`` through to the
    match panel makes both panels the same width while drawing the query's bars twice
    and omitting the match entirely.  These tests pin the width *and* require each panel
    to actually contain its own band.
    """

    N, L, PAD = 8049, 60, 20

    def shared_width(self, query_start: int) -> int:
        lo, hi = span_context_bounds(self.N, query_start, query_start + self.L, self.PAD)
        return hi - lo

    def test_both_panels_span_the_same_number_of_bars(self):
        width = self.shared_width(self.N - self.L)
        qv = centred_view(self.N, self.N - self.L, self.N, width)
        # A match far away in the archive -- the realistic case.
        mv = centred_view(self.N, 138, 198, width)
        assert qv[1] - qv[0] == mv[1] - mv[0] == width

    def test_each_panel_contains_its_own_band(self):
        """The regression that matters: a shared *range* empties the match panel."""
        width = self.shared_width(5000)
        for q_start in (500, 4000, self.N - self.L):
            qv = centred_view(self.N, q_start, q_start + self.L, width)
            assert qv[0] <= q_start and q_start + self.L <= qv[1], "query band visible"

            for m_start in (0, 138, 1000, 4000, self.N - self.L):
                mv = centred_view(self.N, m_start, m_start + self.L, width)
                assert mv[0] <= m_start, f"match band {m_start} cut off at left"
                assert m_start + self.L <= mv[1], f"match band {m_start} cut off at right"

    def test_panels_are_not_the_same_range(self):
        """Distinct bands must yield distinct ranges, or one panel shows the wrong tape."""
        width = self.shared_width(4000)
        a = centred_view(self.N, 4000, 4060, width)
        b = centred_view(self.N, 138, 198, width)
        assert a != b
        assert not (a[0] <= 138 and 198 <= a[1]), "the query range must not contain the match"

    def test_width_matches_the_context_padded_width(self):
        """In the interior the shared width is exactly span + 2*pad."""
        assert self.shared_width(4000) == self.L + 2 * self.PAD

    def test_edges_clamp_without_changing_the_width(self):
        """Centring is impossible at an edge, but the width must still be honoured."""
        width = self.shared_width(4000)
        for start in (0, 5, self.N - self.L - 5, self.N - self.L):
            lo, hi = centred_view(self.N, start, start + self.L, width)
            assert hi - lo == width, f"start={start} changed the width"
            assert 0 <= lo and hi <= self.N
            assert lo <= start and start + self.L <= hi

    def test_width_is_capped_by_the_archive(self):
        """A width larger than the archive cannot be honoured; the archive is shown."""
        assert centred_view(80, 5, 65, 500) == (0, 80)

    def test_price_tab_uses_the_sidebar_width_not_the_context_width(self):
        """The Price tab must show the wide tape the sidebar asked for.

        It briefly used the narrow ``span + 2*pad`` context width (100 bars), which
        made the 60-bar match band fill 60% of the panel and flattened into a slab.
        The width comes from the sidebar's own view -- ~4 window-lengths, min 600, in
        Latest mode -- so the band reads as a shape at ~10%.
        """
        body = price_tab_body()

        assert "price_width = n - chart_from" in body, (
            "the Price tab must derive its width from the sidebar view "
            "(n - chart_from), not from CONTEXT_BARS"
        )
        assert "span_context_bounds(pipe, start_idx, stop_idx" not in body, (
            "the Price tab is back to sizing itself from CONTEXT_BARS, which shrinks "
            "the tape to ~100 bars and squashes the match band"
        )

    def test_match_band_is_a_readable_share_of_the_wide_chart(self):
        """Sanity on the proportion: the band must not dominate or vanish.

        At the 600-bar price-chart width a 60-bar band is ~10%; at the 100-bar context
        width it is 60%, which is a slab rather than a shape.
        """
        L, wide = 60, 600
        assert 0.05 <= L / wide <= 0.20, f"band is {L / wide:.0%} of the wide chart"
        assert L / (L + 2 * 20) > 0.5, "the narrow context width makes the band a slab"


class TestBandsLineUpHorizontally:
    """The regression that motivated ``shared_anchor``.

    The two panels were the same *width* but the bands sat at different x positions,
    so the eye compared different parts of two different charts.

    Geometry: drawing band ``[start, stop)`` at x-offset ``a`` inside a ``width``-bar
    window needs ``a`` bars of tape to its left and ``width - span - a`` to its right.
    So every band is drawable at one shared offset iff

        max(stop) + width - span - n  <=  min(start)

    which caps the usable width at ``n - max(stop) + min(start) + span``.
    """

    #: ``N`` is the *cleaned* bar count of :data:`FIXTURE_BARS`, so it moves with the
    #: fixture instead of being a constant that a different warm-up would invalidate.
    N, L, W = N_BARS, 60, 600

    def _offsets(self, q_start, m_start, width=None):
        width = self.W if width is None else width
        spans = [(q_start, q_start + self.L), (m_start, m_start + self.L)]
        a = shared_anchor(self.N, width, spans)
        qv = aligned_view(self.N, q_start, q_start + self.L, width, a)
        mv = aligned_view(self.N, m_start, m_start + self.L, width, a)
        return q_start - qv[0], m_start - mv[0], a

    @staticmethod
    def _cap(q_start, m_start, n, length):
        spans = [(q_start, q_start + length), (m_start, m_start + length)]
        return n - max(e for _, e in spans) + min(s for s, _ in spans) + length

    def test_live_query_and_its_top_match_line_up(self):
        """The actual view the user sees: live query against its #1 match."""
        from timeseries.matching import Query

        pipe = Pipeline.from_frame(
            session_bars(FIXTURE_BARS, seed=FIXTURE_SEED), length=self.L)
        q = Query.from_span(pipe.matrix, self.N - self.L, self.N)
        res = pipe.match(q, k=5)
        assert res.matches, "expected matches for the live query"

        for m in res.matches:
            if self._cap(q.start, m.start, self.N, self.L) < self.W:
                continue  # infeasible at 600 bars; covered separately below
            q_off, m_off, a = self._offsets(q.start, m.start)
            assert abs(q_off - m_off) == 0, (
                f"live query vs match@{m.start}: bands {abs(q_off - m_off)} bars "
                f"apart (query {q_off}, match {m_off}, anchor {a})"
            )

    def test_live_query_against_a_late_match_was_402px_apart_before(self):
        """The pre-fix offsets, on a pair where alignment is actually achievable.

        The live query is the archive's last window, so it sits flush right at offset
        ``width - span`` = 540.  A match late enough in the archive can also sit there.
        """
        q_off, m_off, _ = self._offsets(self.N - self.L, 7464)
        assert self._cap(self.N - self.L, 7464, self.N, self.L) >= self.W
        assert abs(q_off - m_off) == 0, (
            f"bands {abs(q_off - m_off)} bars apart (query {q_off}, match {m_off})"
        )

    def test_mid_archive_query_aligns(self):
        q_off, m_off, _ = self._offsets(4000, 4100)
        assert abs(q_off - m_off) == 0

    def test_independent_centring_really_does_misalign(self):
        """Guard: if this ever passes, the fix below is no longer load-bearing."""
        q_start, m_start = self.N - self.L, 7464
        qv = centred_view(self.N, q_start, q_start + self.L, self.W)
        mv = centred_view(self.N, m_start, m_start + self.L, self.W)
        assert abs((q_start - qv[0]) - (m_start - mv[0])) > 0

    def test_query_derived_anchor_really_does_misalign(self):
        """Guard: deriving the anchor from the query alone is also a bug."""
        q_start, m_start = self.N - self.L, 7464
        want = (self.W - self.L) // 2
        qv = aligned_view(self.N, q_start, q_start + self.L, self.W, want)
        mv = aligned_view(self.N, m_start, m_start + self.L, self.W, want)
        assert abs((q_start - qv[0]) - (m_start - mv[0])) > 0

    def test_shared_anchor_fixes_both_guards_above(self):
        q_off, m_off, _ = self._offsets(self.N - self.L, 7464)
        assert abs(q_off - m_off) == 0

    def test_early_match_cannot_align_at_600_bars(self):
        """A documented limit, not a bug -- the archive geometry forbids it.

        Match at bar 138 needs 540 bars of lead-in to sit beside a flush-right query,
        but only 138 exist.  Even a perfect archive cannot satisfy both, and the only
        aligned width available (198) is narrower than the 100-bar framing already
        rejected.
        """
        cap = self._cap(self.N - self.L, 138, self.N, self.L)
        assert cap == 198
        assert cap < self.W, "expected 600 bars to be infeasible here"

    def test_narrowing_to_the_cap_makes_it_align(self):
        cap = self._cap(self.N - self.L, 138, self.N, self.L)
        q_off, m_off, _ = self._offsets(self.N - self.L, 138, width=cap)
        assert abs(q_off - m_off) == 0

    def test_bands_stay_fully_visible(self):
        for q_start, m_start, w in ((self.N - self.L, 7464, self.W),
                                    (4000, 4100, self.W),
                                    (self.N - self.L, 138, self.W)):
            spans = [(q_start, q_start + self.L), (m_start, m_start + self.L)]
            a = shared_anchor(self.N, w, spans)
            for start, stop in spans:
                lo, hi = aligned_view(self.N, start, stop, w, a)
                assert lo <= start and stop <= hi, "band cropped out of view"

    def test_width_is_the_same_number_of_bars(self):
        for q_start, m_start in ((self.N - self.L, 7464), (4000, 4100)):
            spans = [(q_start, q_start + self.L), (m_start, m_start + self.L)]
            a = shared_anchor(self.N, self.W, spans)
            for start, stop in spans:
                lo, hi = aligned_view(self.N, start, stop, self.W, a)
                assert hi - lo == self.W

    def test_anchor_is_feasible_for_every_span(self):
        """Only asserted where alignment is geometrically possible."""
        for q_start, m_start in ((self.N - self.L, 7464), (4000, 4100), (4000, 41)):
            spans = [(q_start, q_start + self.L), (m_start, m_start + self.L)]
            if self._cap(q_start, m_start, self.N, self.L) < self.W:
                continue
            a = shared_anchor(self.N, self.W, spans)
            span = max(e - s for s, e in spans)
            slack = self.W - span
            assert a <= min(s for s, _ in spans), "not enough tape to the left"
            assert max(e for _, e in spans) + (slack - a) <= self.N, \
                "not enough tape to the right"

    def test_centred_offset_used_when_feasible(self):
        _, _, a = self._offsets(4000, 4100)
        assert a == (self.W - self.L) // 2, "should sit centred when it can"

    def test_degenerate_inputs_do_not_crash(self):
        assert shared_anchor(self.N, self.W, []) == 0
        assert shared_anchor(self.N, 10, [(0, 60)]) == 0
        assert 0 <= shared_anchor(100, 600, [(0, 60), (40, 100)]) < 600


class _ViewPipe:
    """Supplies ``pipe.n_bars`` for ``render_price_tab``'s degenerate-window guard."""

    n_bars = 8069


class TestPriceChartWindowIsBrushInvariant:
    """The top chart's visible window must not depend on what has been brushed.

    This is the reported bug in its own terms: *"the top chart in the Price tab
    should not scroll; currently on selecting a window of data it scrolls."*

    The window was ``aligned_view(..., anchor)`` with the anchor computed by
    ``shared_anchor`` from the query **and** the match, so the brush -- the one thing
    the reader is changing -- moved the axis out from under them.  The band they were
    looking at also jumped to a new x offset each time, so even a reader who had
    internalised "the tape is the same tape" was re-orienting on every gesture.

    These pin the invariant directly, by evaluating the real view expression from
    ``app.py`` rather than restating it: a mirror of the arithmetic would prove only
    that the mirror is self-consistent, which is precisely the failure mode this
    repository already had once (see the module docstring).
    """

    N, L = 8069, 240          # DEFAULT_LENGTH
    CHART_FROM = 6630         # the trailing five sessions' left edge

    @staticmethod
    def _view(chart_from=CHART_FROM, view_stop=N):
        """``render_price_tab``'s real view arithmetic, as a callable.

        ``app.py`` cannot be imported under pytest -- importing it renders the whole
        UI -- so the statements that size the window are lifted out of the source and
        executed with only ``pipe`` in scope.  What this evaluates is therefore *the
        app's own arithmetic*: change it and these tests fail, rather than the test
        quietly keeping a mirror that still agrees with itself.  That is the same
        contract ``load_geometry`` gives the view helpers used elsewhere in this file.
        """
        body = app_block(r"def render_price_tab\(.*?\n(?=def )", "render_price_tab")
        stmt = re.search(r"^    (lo, hi = [^\n]*)$", body, re.M)
        guard = re.search(r"^    (if hi <= lo:\n        hi = [^\n]*)$", body, re.M)
        assert stmt and guard, (
            "render_price_tab no longer sizes its view the way these tests evaluate "
            "-- update them rather than letting them pass vacuously"
        )
        ns = {"pipe": _ViewPipe}
        exec("def _v(chart_from, view_stop):\n"
             "    " + stmt.group(1) + "\n"
             "    " + guard.group(1).replace("\n", "\n    ") + "\n"
             "    return lo, hi", ns)
        return ns["_v"](chart_from, view_stop)

    def test_every_brush_yields_the_same_window(self):
        """The single assertion behind the bug report.

        Every brush across the whole archive resolves to one identical view, so the
        chart cannot scroll however the window is selected.
        """
        views = {self._view() for _ in range(0, self.N - self.L + 1, 97)}
        assert views == {(self.CHART_FROM, self.N)}, (
            f"the price chart's window moves with the brush: {views}. The reader is "
            "brushing against a view that shifts underneath them."
        )

    def test_the_signature_takes_no_anchor(self):
        """A match position must have no way to reach this view.

        The old ``anchor`` parameter *was* that way in, which is why the removal is
        enforced at the signature and not just at the call site: nothing to pass means
        nothing can be reintroduced by a later edit that reaches for it.
        """
        src = app_source()
        fn = re.search(r"def render_price_tab\((.*?)\) -> Any:", src, re.S)
        assert fn, "could not find render_price_tab's signature"
        params = fn.group(1)
        assert "anchor" not in params, (
            "render_price_tab still accepts an `anchor` -- a match position could be "
            "re-framed into this chart's window again"
        )
        assert "chart_width" in params, (
            "the width must still be threaded to both renderers so a bar is the same "
            "number of pixels on each panel"
        )

    def test_brushed_band_is_always_inside_the_window(self):
        """Stability must not come at the cost of hiding the band the reader drew.

        A brush can only be drawn inside the visible window, so for every brush a
        reader can physically make, the band lands within it.  Brushes outside it are
        unreachable by definition and are not required to be visible.
        """
        lo, hi = self._view()
        for start in range(self.CHART_FROM, self.N - self.L + 1, 37):
            assert lo <= start, f"brush@{start} starts left of the window"
            assert start + self.L <= hi, f"brush@{start} overhangs the right edge"

    def test_a_re_render_with_an_unchanged_brush_is_pixel_identical(self):
        """With the view fixed, nothing about the query can shift the panel.

        The old code produced a different view for the *same* brush depending on where
        the match had landed, so an unrelated rerun could re-frame the tape.  There is
        no remaining input for it to do that with.
        """
        first = self._view()
        for _ in range(5):
            assert self._view() == first
        assert first[1] - first[0] == self.N - self.CHART_FROM

    def test_degenerate_view_does_not_collapse(self):
        """An empty window must be widened to one bar rather than drawn empty."""
        lo, hi = self._view(chart_from=10, view_stop=10)
        assert hi > lo, "a zero-width window would draw an empty chart"
        assert hi <= self.N


class TestAppAnchorWiring:
    """Source-level guards on the *wiring*.

    The geometry tests above now run the app's own helpers, but they cannot see
    `app.py` wiring them up wrongly: a mutation that put the match view back on its own
    centred offset would leave all of them green, because `aligned_view` is fine in
    isolation.  These assert the call sites, so that has to fail too.

    The scope narrowed when the Price tab's shared anchor was removed.  The Matches
    tab still aligns its query and match panels at one offset, and
    `_render_best_match_pair` is where that happens, so those guards stand.  The Price
    tab no longer anchors anything -- its top chart's window is pinned to the trailing
    view, because deriving it from the query is what made the tape scroll under the
    reader on every brush -- so it now has guards asserting the anchor stays *absent*.
    """

    @staticmethod
    def _pair_body():
        return app_text(
            r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
            "_render_best_match_pair",
        )

    def test_both_views_use_the_same_anchor_variable(self):
        calls = re.findall(
            r"([qm])_view = aligned_view\(([^)]*)\)", self._pair_body())
        assert len(calls) == 2, "expected one aligned_view call per panel"
        q_args, m_args = calls[0][1], calls[1][1]
        # The band arguments must differ (each panel draws its own band) but the
        # trailing anchor argument must be identical.
        assert q_args != m_args, "both panels are drawing the same band"
        q_anchor = q_args.split(",")[-1].strip()
        m_anchor = m_args.split(",")[-1].strip()
        assert q_anchor == m_anchor, (
            "the two panels are drawn at different offsets -- they will not line up:\n"
            f"  query anchor {q_anchor!r} vs match anchor {m_anchor!r}"
        )
        assert m_anchor == "a", (
            f"the match view must be anchored at the shared offset `a`, got {m_anchor!r}"
        )

    def test_only_one_anchor_is_computed(self):
        body = self._pair_body()
        assert body.count("aligned_view(") == 2, (
            "expected exactly two aligned_view calls (one per panel); extra calls mean "
            "one of them is being re-anchored"
        )
        assert "anchor = q_start - q_view[0]" not in body, (
            "deriving the anchor from the query alone is the original bug -- the offset "
            "must come from shared_anchor so the match has room for it"
        )

    def test_price_tab_does_not_anchor_the_price_chart(self):
        """The top chart's window must not be derived from either band.

        This is the regression that made the Price tab's top chart scroll on every
        brush.  It computed ``band_anchor = shared_anchor(n, price_width, spans)``
        over the query *and* the match, then handed that anchor to ``render_price_tab``,
        which turned it into the chart's **entire visible range** via
        ``aligned_view``.  So brushing a new window moved the band, which moved the
        anchor, which moved the view -- four different framings of the same trailing
        five sessions, with the band hopping hundreds of bars along x.

        The anchor is gone.  Asserted on the *parsed* call rather than on the text,
        because the surrounding comment names ``shared_anchor`` and
        ``anchor=band_anchor`` while explaining why they were removed: a substring
        search would pass on the prose describing the old code and fail on a correct
        fix.  That is the same mistake ``test_price_tab_asks_for_a_pannable_match_panel``
        documents in ``test_match_panel_pan.py``.
        """
        fragment = textwrap.dedent(price_tab_body())
        calls = [n for n in ast.walk(ast.parse(fragment))
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "render_price_tab"]
        assert len(calls) == 1, "expected exactly one call to render_price_tab"
        kwargs = {kw.arg for kw in calls[0].keywords}
        assert "anchor" not in kwargs, (
            "the price chart is being anchored again -- its visible window would "
            "depend on the query and the match, and re-frame the tape on every brush"
        )

    def test_price_tab_no_longer_computes_an_anchor(self):
        """No ``shared_anchor`` call survives in the Price tab body.

        Checked on the parsed call rather than the source text: the comment explaining
        the removal quotes ``shared_anchor(n, price_width, spans)`` verbatim, so
        ``assert "shared_anchor(" in body`` passes on the prose and would have stayed
        green forever after the call itself was deleted.
        """
        fragment = textwrap.dedent(price_tab_body())
        calls = [n for n in ast.walk(ast.parse(fragment))
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "shared_anchor"]
        assert not calls, (
            "the Price tab computes a shared_anchor again -- with it, the price "
            "chart's view follows the query and the match and the tape scrolls under "
            "the reader on every brush"
        )
        # The match must still be resolved before the chart that draws it.
        i_match = fragment.index("cross_sectional_match(")
        i_render = fragment.index("render_price_tab(")
        assert i_match < i_render, (
            "the match lookup must precede render_price_tab -- the match decides "
            "whether there is a lower panel at all"
        )

    def test_price_tab_draws_the_match_from_the_matched_ticker_s_pipeline(self):
        """The match panel must not be drawn from the *query* ticker.

        The search is cross-sectional, so the match can live in a different series
        than the query.  ``build_price_figure`` reads ``pipe.bars`` for every level,
        tick label and band it draws, so handing it the query's pipeline would render
        NVDA's window at QQQ's bar indices -- a chart that looks fine and shows
        entirely the wrong tape.
        """
        body = price_tab_body()
        assert "match_pipe = pipe" in body, (
            "a fallback to the query pipeline is fine, but it has to be the default "
            "that an explicit cross-ticker build then replaces"
        )
        assert "panel_pipeline_for(auto[\"search\"], m.ticker)" in body, (
            "the matched ticker's own pipeline must be built and used; otherwise the "
            "lower chart draws the query ticker's tape at the match's coordinates"
        )
        assert "render_matches_tab(match_pipe," in body, (
            "the match panel must be rendered from the matched ticker's pipeline"
        )

    def test_price_tab_labels_the_match_with_its_ticker(self):
        """A cross-ticker chart must say which ticker it is.

        An unlabelled AAPL chart under a QQQ query is indistinguishable from a chart
        of QQQ's own history, and the reader's first assumption would be that it is
        their own tape.  The ticker, its session and its panel rank are what make the
        panel readable as a cross-sectional answer.
        """
        body = price_tab_body()
        assert "best_match.ticker" in body, (
            "the caption must name the matched ticker -- an unlabelled cross-ticker "
            "panel reads as the query ticker's own history"
        )
        assert "Closest match anywhere in the S&P 500 archive" in body, (
            "the caption must say the search was cross-sectional, or a reader will "
            "assume the match came from the ticker they were looking at"
        )

    def test_price_tab_threads_a_width_not_a_range(self):
        """The regression that produced an empty match chart.

        Passing the query's view range straight through made both panels the same width
        while drawing the same bars twice.  The plumbing must carry a *width*.

        Checked by counting keyword occurrences rather than by matching
        ``"chart_width=price_width"``: the two calls sit on separate lines with
        different continuation indents, so an exact-substring check on how they wrap
        was a reformat away from failing a test about arithmetic.
        """
        body = price_tab_body()
        assert body.count("chart_width=price_width") == 2, (
            "both charts must be drawn at the query chart's bar width; found "
            f"{body.count('chart_width=price_width')} occurrences"
        )
        assert "view=price_view" not in body, (
            "passing a view range through empties the match panel -- it is an absolute "
            "bar range, not a width"
        )
        assert "compact=True" in body, (
            "the Price tab draws the match through the compact branch, not the "
            "full Matches tab"
        )

    def test_price_tab_draws_the_match_without_redrawing_the_query(self):
        """The Price tab's first chart *is* the query, so the match block must not repeat it."""
        branch = app_block(r"if compact:\n(.*?)\n        return",
                           "compact branch")
        assert "show_query=False" in branch, (
            "the compact branch redraws the query, which the price chart above already "
            "shows -- pass show_query=False"
        )
        assert "compact=True" in price_tab_body(), (
            "Price tab no longer renders the compact block"
        )

    def test_price_tab_uses_one_height_for_both_charts(self):
        """A taller plot makes the same move look bigger; the pair must agree."""
        body = price_tab_body()

        assert body.count("height=pair_height") == 2, (
            "both charts must be drawn at the same height; found "
            f"{body.count('height=pair_height')} occurrences of height=pair_height"
        )
        assert "height=560" not in body, (
            "a literal height here will drift from the match panel's default"
        )
        assert re.search(r"pair_height = PRICE_CHART_HEIGHT", body), (
            "the shared height must come from the module constant so the two charts "
            "cannot drift apart"
        )
        assert "PRICE_CHART_HEIGHT = " in app_source(), (
            "PRICE_CHART_HEIGHT is not defined"
        )


# --------------------------------------------------------------------------- #
# build_shape_figure -- subplot geometry
# --------------------------------------------------------------------------- #
class TestShapeFigureGeometry:
    """The Matches tab's per-match overlay must build a valid figure.

    ``make_subplots`` requires ``len(row_heights) == rows`` exactly and raises
    otherwise.  This function derived its row count from ``FEATURE_COLUMNS`` but wrote
    ``heights`` as a literal sliced to fit, so the two could disagree -- and when a
    channel was removed the running server kept its wider matrix in memory
    against a shorter list, which took the whole Matches tab down with a message about
    subplot geometry that named neither the cause nor the fix.
    """

    @staticmethod
    def _figure_for(columns):
        """Call the app's own ``build_shape_figure`` against a synthetic pipe."""
        import numpy as np
        from timeseries.features import FEATURE_COLUMNS
        from timeseries import matching as M
        from apphelpers import load_app_functions

        labels = tuple(columns)
        n, L = 900, 60
        rng = np.random.default_rng(0)
        matrix = np.cumsum(rng.normal(0, 0.01, size=(n, len(labels))), axis=0)

        # A class body does not close over the enclosing function's locals, so this is a
        # namespace rather than a class.
        pipe = type("_Pipe", (), {"length": L, "matrix": matrix})()

        ns = load_app_functions(
            ["build_shape_figure"],
            namespace={
                "make_subplots": _make_subplots,
                "go": go,
                "np": np,
                "M": M,
                "CHANNEL_LABELS": labels,
            },
        )
        return ns["build_shape_figure"](pipe, 100, 400, 1, 1.234)

    def test_builds_for_the_current_feature_columns(self):
        from timeseries.features import FEATURE_COLUMNS

        fig = self._figure_for(FEATURE_COLUMNS)
        assert len(fig.data) == 2 * len(FEATURE_COLUMNS), (
            "one query trace and one match trace per channel"
        )

    def test_builds_when_a_channel_is_added_or_removed(self):
        """The guard: geometry follows the channel count, not a hard-coded literal."""
        for labels in [
            ("return_z", "path_z"),
            ("return_z", "extra_z", "path_z"),
            ("return_z", "extra_z", "other_z", "path_z"),
        ]:
            fig = self._figure_for(labels)
            assert len(fig.data) == 2 * len(labels), (
                f"{labels} produced {len(fig.data)} traces"
            )

    def test_path_z_is_drawn_first_and_tallest(self):
        from timeseries.features import FEATURE_COLUMNS

        fig = self._figure_for(FEATURE_COLUMNS)
        # ``path_z`` is reordered to the front of the rows, so it is always row 1 and
        # therefore always owns the primary ``y`` axis -- while the leg that follows it
        # lands on ``y2``.  Asserting on the axis id rather than on trace order avoids
        # depending on how many traces precede it.
        assert fig.data[0].yaxis == "y", "path_z must be the first row"
        assert fig.data[2].yaxis == "y2", "the other leg must be the second row"
        # Taller row: ``make_subplots`` normalises row_heights by their total, so
        # comparing the ratio is the check that actually means something.
        domain = fig.layout.yaxis.domain
        top_domain = fig.layout.yaxis2.domain
        assert (domain[1] - domain[0]) > (top_domain[1] - top_domain[0]), (
            "path_z must be given the most room -- it is the leg the chart above draws"
        )
