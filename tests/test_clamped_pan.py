"""The Price tab's lower chart must *stop* at its bounds, not zoom past them.

Plotly cannot do this.  ``xaxis.minallowed``/``maxallowed`` bound the range, but once a
drag runs past one, plotly.js switches to **zooming** instead of refusing -- the
behaviour plotly/plotly.js#887 asks to be fixed and is still open.  So the app renders
that one panel through a small custom component (``static/clamped_pan_chart.html``)
that re-applies the bounds on every ``plotly_relayout``.

The clamping arithmetic lives in JavaScript, so these tests do two things:

* check the Python side -- that the pannable panel really is routed to the component,
  the component really carries the bounds, and the vendored plotly.js is present; and
* extract the JS and execute it under ``node``, so the clamp itself is tested as code
  rather than by reading it.  A mirror of the logic in Python would test nothing.

The JS tests are skipped when ``node`` is absent rather than silently passing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from apphelpers import app_block, app_called_names, app_source

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
ROOT = os.path.dirname(APP)
HTML = os.path.join(ROOT, "static", "clamped_pan_chart.html")
PLOTLY_JS = os.path.join(ROOT, "static", "plotly.min.js")

NODE = shutil.which("node")

# The geometry the component is exercised against: a window 1287.5 bars wide inside
# bounds that leave room to travel, with a band somewhere in the middle.
MIN_A, MAX_A, OPEN = 2714.5, 5168.5, 1287.5
BAND = (3882, 4002)


def _js_logic(tmp_path) -> str:
    """The ``clamp``/``enforce`` functions lifted out of the component's <script>."""
    src = open(HTML, encoding="utf-8").read()
    m = re.search(r"function clamp.*?(?=window\.renderChart)", src, re.S)
    assert m, "clamp/enforce not found in clamped_pan_chart.html"
    harness = (
        "var Plotly={relayout:function(gd,u){gd.layout.xaxis.range=u['xaxis.range'];}};\n"
        "function fakeGD(a,b,r,s){return{layout:{xaxis:{minallowed:a,maxallowed:b,"
        "range:r}},_minSpan:s};}\n"
    )
    path = tmp_path / "clamp.js"
    path.write_text(harness + m.group(0) + "\nmodule.exports={enforce,fakeGD};\n")
    return str(path)


def _run_js(tmp_path, body: str) -> str:
    path = _js_logic(tmp_path)
    script = "const {enforce,fakeGD}=require(%r);\n%s\n" % (path, body)
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, "node failed:\n%s" % out.stderr
    return out.stdout



class TestTheComponentExists:
    def test_html_is_present(self):
        if not os.path.exists(HTML):
            pytest.skip("clamped_pan_chart.html not present")
        assert os.path.exists(HTML)

    def test_plotly_js_is_vendored(self):
        """The component loads plotly.js from ``static/``; a CDN would break offline."""
        assert os.path.exists(PLOTLY_JS), (
            "static/plotly.min.js is missing -- the component cannot load plotly offline"
        )
        assert os.path.getsize(PLOTLY_JS) > 1_000_000, (
            "that is not a full plotly.js bundle"
        )

    def test_component_does_not_load_from_a_cdn(self):
        src = open(HTML, encoding="utf-8").read()
        for remote in ("https://", "http://", "//cdn"):
            assert remote not in src, (
                f"the component references {remote}; it must use the vendored bundle"
            )


class TestPythonSideWiring:
    def test_pannable_panel_uses_the_component(self):
        body = app_block(r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
                         "_render_best_match_pair body")
        assert "render_clamped_pan_chart(" in body, (
            "the pannable panel is not routed to the clamped component, so a drag past "
            "the bounds will zoom again"
        )

    def test_other_panels_still_use_plotly_chart(self):
        body = app_block(r"def _render_best_match_pair\(.*?\n(?=def render_matches_tab)",
                         "_render_best_match_pair body")
        assert "st.plotly_chart(" in body, (
            "the non-pannable panels must keep using st.plotly_chart"
        )

    def test_the_component_is_delivered_as_a_document_not_a_url(self):
        """``st.iframe`` (srcdoc), *not* the URL-taking ``components.v1.iframe``.

        The two are siblings, not synonyms.  The old ``components.v1.iframe(src=...)``
        writes a **URL** into the iframe's ``src``, so passing it the component's HTML
        handed the browser a ~1 MB "URL" starting ``<!DOCTYPE html>``, which it resolved
        as a relative path against the app root.  That path does not exist, so it fell
        through to the SPA catch-all and the frame loaded **Streamlit's own HTML
        shell** -- a 200, a ``text/html``, and a document with no ``renderChart`` in it.
        The panel was an empty box of exactly the right size, with an empty log, which
        is the Price tab's second chart going missing.

        Invisible from Python by construction: the proto marshalled without complaint
        and every request succeeded.  Only the browser knows, so this is pinned here.

        Checked against the AST rather than the raw text: the comment explaining this
        pitfall necessarily spells out the forbidden call, so a substring search
        matches the explanation and fails a correct file.
        """
        if not os.path.exists(APP):
            pytest.skip("app.py not present")
        called = app_called_names()
        assert "iframe" in called, (
            "render_clamped_pan_chart must render through st.iframe, which routes a "
            "non-URL string into the iframe's srcdoc so the document is the frame's "
            "document"
        )
        assert not called & {"html"}, (
            "app.py still calls st.html somewhere the old code called components.html -- "
            "st.html sanitises by default and needs unsafe_allow_javascript for the "
            "clamp's script to run at all"
        )

    def test_falls_back_when_the_asset_is_missing(self):
        """A missing file must not take the Price tab down."""
        if not os.path.exists(APP):
            pytest.skip("app.py not present")
        body = app_block(r"def render_clamped_pan_chart\(.*?\n(?=def build_shape_figure)",
                         "render_clamped_pan_chart body")
        assert "st.plotly_chart(" in body, (
            "no fallback: a missing asset would blank the panel entirely"
        )
        assert "if not os.path.exists" in body

    def test_json_and_os_are_imported(self):
        """The renderer serialises the figure with json; both were previously missing."""
        src = app_source()
        assert re.search(r"^import json$", src, re.M), "app.py does not import json"
        assert re.search(r"^import os$", src, re.M), "app.py does not import os"


@pytest.mark.skipif(NODE is None, reason="node not available")
class TestTheClampStopsThePan:
    """The actual requirement, executed as JavaScript."""

    def test_pan_reaches_the_stop_exactly(self, tmp_path):
        out = _run_js(tmp_path, f"""
          {{
            const gd = fakeGD({MIN_A}, {MAX_A}, [{MIN_A - 0.5}, {MIN_A - 0.5 + OPEN}], {OPEN});
            enforce(gd);
            console.log(JSON.stringify(gd.layout.xaxis.range));
          }}
        """)
        got = json.loads(out.strip().splitlines()[-1])
        assert got[0] == pytest.approx(MIN_A), "the pan did not stop at the left bound"

    def test_over_drag_left_does_not_zoom(self, tmp_path):
        """Dragging 5000 bars past the edge must not shrink the window."""
        out = _run_js(tmp_path, f"""
          {{
            const gd = fakeGD({MIN_A}, {MAX_A}, [{MIN_A - 5000}, {MIN_A - 5000 + OPEN}], {OPEN});
            enforce(gd);
            console.log(JSON.stringify([gd.layout.xaxis.range,
                                        gd.layout.xaxis.range[1] - gd.layout.xaxis.range[0]]));
          }}
        """)
        rng, width = json.loads(out.strip().splitlines()[-1])
        assert width == pytest.approx(OPEN), (
            f"an over-drag zoomed the window to {width} bars instead of {OPEN}"
        )
        assert rng[0] >= MIN_A - 0.01

    def test_over_drag_right_does_not_zoom(self, tmp_path):
        """The far-right case, which is the one that escaped an earlier version."""
        left = MAX_A - OPEN + 5000
        out = _run_js(tmp_path, f"""
          {{
            const gd = fakeGD({MIN_A}, {MAX_A}, [{left}, {left + OPEN}], {OPEN});
            enforce(gd);
            const r = gd.layout.xaxis.range;
            console.log(JSON.stringify([r, r[1] - r[0]]));
          }}
        """)
        rng, width = json.loads(out.strip().splitlines()[-1])
        assert width == pytest.approx(OPEN), f"over-drag changed width to {width}"
        assert rng[1] <= MAX_A + 0.01, f"range escaped past maxAllowed: {rng}"
        assert rng[0] >= MIN_A - 0.01

    def test_range_never_leaves_the_bounds(self, tmp_path):
        """A sweep of hostile ranges, including both edges out at once."""
        cases = [
            MIN_A - 5000, MIN_A - 500, MIN_A - 0.5, MIN_A, MIN_A + 500,
            MAX_A - OPEN, MAX_A - OPEN + 500, MAX_A - OPEN + 5000, MAX_A,
        ]
        body = ["const bad=[];"]
        for left in cases:
            body.append(
                f"{{const gd=fakeGD({MIN_A},{MAX_A},[{left},{left + OPEN}],{OPEN});"
                f" enforce(gd);const r=gd.layout.xaxis.range;"
                f" if(r[0]<{MIN_A}-0.01||r[1]>{MAX_A}+0.01) bad.push([{left},r]);}}"
            )
        body.append("console.log(JSON.stringify(bad));")
        out = _run_js(tmp_path, "\n".join(body))
        bad = json.loads(out.strip().splitlines()[-1])
        assert bad == [], f"ranges escaped the bounds: {bad}"

    def test_match_stays_visible_at_every_pan_stop(self, tmp_path):
        """The user's requirement, checked through the real clamp."""
        cases = [MIN_A, MIN_A + 900, MAX_A - OPEN, MAX_A - OPEN + 5000]
        body = ["const bad=[];"]
        for left in cases:
            body.append(
                f"{{const gd=fakeGD({MIN_A},{MAX_A},[{left},{left + OPEN}],{OPEN});"
                f" enforce(gd);const r=gd.layout.xaxis.range;"
                f" if(!(r[0]<={BAND[0]} && r[1]>={BAND[1]})) bad.push([{left},r]);}}"
            )
        body.append("console.log(JSON.stringify(bad));")
        out = _run_js(tmp_path, "\n".join(body))
        bad = json.loads(out.strip().splitlines()[-1])
        assert bad == [], f"the match left the chart at: {bad}"

    def test_a_zoom_attempt_is_refused(self, tmp_path):
        """Shrinking below the opening width is the failure this exists to prevent."""
        out = _run_js(tmp_path, f"""
          {{
            const mid = ({MIN_A} + {MAX_A}) / 2;
            const gd = fakeGD({MIN_A}, {MAX_A}, [mid, mid + 400], {OPEN});
            enforce(gd);
            const r = gd.layout.xaxis.range;
            console.log(JSON.stringify(r[1] - r[0]));
          }}
        """)
        width = json.loads(out.strip().splitlines()[-1])
        assert width == pytest.approx(OPEN), (
            f"a zoom to 400 bars was allowed; width became {width}"
        )


class TestTheDocumentRendersAChart:
    """Guards the failure that made the panel invisible: a frame with no script in it.

    The component used to nest an ``<iframe src='clamped_pan_chart.html'>`` inside the
    one the outer embed already gives you.  A relative src inside an iframe resolves
    against the *app's* base URL, not ``static/``, so it asked for a path that does not
    exist; Streamlit's SPA answered with its own shell, the inner frame loaded with no
    ``renderChart`` defined, and the panel rendered as an empty box.  These assert the
    shape of the document that actually gets sent, since nothing else catches it -- the
    fallback path is silent.

    The same symptom had a second, later cause on the Python side: the document was
    delivered through the URL-taking ``components.v1.iframe`` rather than a document-
    taking form, so the frame's ``src`` attribute literally began ``<!DOCTYPE html>``
    and the browser resolved that as a relative path into the SPA shell.  Both are
    silent at the network layer and invisible from Python, which is why they are
    asserted structurally rather than caught at runtime.
    """

    def _doc(self) -> str:
        if not os.path.exists(APP):
            pytest.skip("app.py not present")
        import streamlit as st

        from timeseries.pipeline import Pipeline
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from session_bars import session_bars

        captured = []
        orig_iframe = st.iframe
        orig_plotly = st.plotly_chart
        st.iframe = lambda doc, **kw: captured.append(doc)
        st.plotly_chart = lambda *a, **k: captured.append(None)
        try:
            src = open(APP, encoding="utf-8").read().replace("\nmain()\n", "\n")
            app = {"__name__": "app_under_test", "__file__": APP}
            exec(compile(src, APP, "exec"), app)  # noqa: S102
            pipe = Pipeline.from_frame(session_bars(8190, seed=20260930), length=120)
            fig = app["build_price_figure"](
                pipe, 3000, 3000 + 1287,
                query_start=4000, query_stop=4120, selectable=False, height=420,
                pan_data=(2714.5, 5168.5), pan_bounds=(2714.5, 5168.5), pannable=True,
            )
            app["render_clamped_pan_chart"](fig, height=420, key="k")
        finally:
            st.iframe = orig_iframe
            st.plotly_chart = orig_plotly
        assert captured, "render_clamped_pan_chart rendered nothing"
        assert captured[0] is not None, (
            "the panel fell back to st.plotly_chart -- the clamp is not active"
        )
        return captured[0]

    def test_document_has_no_nested_iframe(self):
        """One frame only. A nested one silently loses its script."""
        assert "<iframe" not in self._doc()

    def test_plotly_is_loaded_from_the_served_static_path(self):
        doc = self._doc()
        # Streamlit mounts an app's `static/` under `/app/static/`, NOT `/static/`. A
        # request for the un-prefixed path is not a 404 -- it falls through to the SPA
        # catch-all, which answers 200 with an HTML shell, so the browser discards the
        # script and the panel renders as an empty box with nothing in the log. That was
        # the cause of the Price tab's second chart disappearing.
        assert '/app/static/plotly.min.js' in doc, (
            "plotly.js must be loaded from /app/static/ -- Streamlit serves an app's "
            "static/ directory under that prefix, and any other path is answered with "
            "the SPA shell rather than JavaScript"
        )
        assert '<script src="plotly.min.js"' not in doc, (
            "the relative src was not rewritten"
        )

    def test_static_serving_is_enabled_in_the_shipped_config(self):
        """The URL above only works when the server actually serves ``static/``.

        ``server.enableStaticServing`` defaults to **False**. Left off, the plotly.js
        request is swallowed by the SPA catch-all and the panel is silently blank, so
        the flag is pinned in the repo's own config rather than left to the deployment.
        """
        cfg = os.path.join(ROOT, ".streamlit", "config.toml")
        if not os.path.exists(cfg):
            pytest.skip(".streamlit/config.toml not present")
        assert "enableStaticServing = true" in open(cfg, encoding="utf-8").read(), (
            "server.enableStaticServing must be true, or the Price tab's lower chart "
            "renders as an empty box because plotly.js is never served"
        )

    def test_app_warns_when_static_serving_is_off(self):
        """A blank panel is invisible from Python, so the app must check for it.

        With the flag off the document is still built and delivered correctly -- only
        the browser knows the script never arrived -- so there is nothing in the log to
        catch. The renderer therefore tests the flag and degrades to a visible warning
        plus a working (unclamped) chart instead of an empty box.
        """
        if not os.path.exists(APP):
            pytest.skip("app.py not present")
        body = app_block(r"def render_clamped_pan_chart\(.*?\n(?=def build_shape_figure)",
                         "render_clamped_pan_chart body")
        assert "static_serving_enabled()" in body, (
            "render_clamped_pan_chart does not check whether static/ is served; a "
            "misconfigured deployment blanks the panel with no diagnostic"
        )
        assert body.count("st.plotly_chart(") >= 3, (
            "every failure path must fall back to a chart that renders, not to nothing"
        )

    def test_the_component_reports_a_missing_plotly(self):
        """Client-side backstop for the same silent failure.

        If the script fails to load, ``Plotly`` is undefined and ``renderChart`` throws
        before drawing anything. The component states the cause in words, so the panel
        is never an empty box that reads as a broken figure.
        """
        if not os.path.exists(HTML):
            pytest.skip("clamped_pan_chart.html not present")
        src = open(HTML, encoding="utf-8").read()
        assert "typeof Plotly === 'undefined'" in src, (
            "the component does not check whether plotly loaded, so a failed script "
            "leaves an unexplained empty panel"
        )

    def test_the_clamp_script_is_inlined(self):
        doc = self._doc()
        assert "function enforce(gd)" in doc, "the clamp logic was not inlined"
        assert "window.renderChart = function" in doc

    def test_render_chart_is_invoked_with_a_figure(self):
        doc = self._doc()
        assert "window.renderChart({" in doc, "the figure is never handed to the chart"

    def test_modebar_config_survives(self):
        """``to_plotly_json`` omits ``config``; without it zoom comes back."""
        doc = self._doc()
        spec = json.loads(re.search(
            r"window\.renderChart\((\{.*?\}), \d+\);</script>", doc, re.S).group(1))
        assert spec.get("config"), (
            "no config on the spec -- the panel would get plotly's default toolbar and "
            "re-advertise the zoom buttons"
        )
        assert spec["config"]["scrollZoom"] is False
        for button in ("zoomIn", "zoomOut", "zoom2d", "autoScale2d", "resetScale2d"):
            assert button in spec["config"]["modeBarButtonsToRemove"], (
                f"{button} is advertised again on the pannable panel"
            )

    def test_the_figure_carries_its_pan_bounds(self):
        doc = self._doc()
        spec = json.loads(re.search(
            r"window\.renderChart\((\{.*?\}), \d+\);</script>", doc, re.S).group(1))
        xa = spec["layout"]["xaxis"]
        assert xa.get("minallowed") is not None and xa.get("maxallowed") is not None, (
            "the component cannot clamp what has no bounds"
        )
        assert spec["layout"].get("dragmode") == "pan"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
