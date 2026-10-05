"""Shared access to ``app.py`` for the tests that need it.

``app.py`` is a Streamlit script: importing it runs the whole UI, so it cannot be
imported under pytest.  Three test files need something out of it anyway, and each
used to solve that independently -- re-reading the file, re-running the same
``ast`` extraction, and re-asserting that a helper it had *copied* still matched the
original.  This module is that logic, written once.

Two access paths, for two different jobs:

* :func:`app_source` / :func:`app_text` for assertions about the *wiring* -- that a
  call is made with the right argument, that two panels are drawn at the same
  height.  Nothing better exists short of rendering the app, and a reformat should
  not be able to fail a test.
* :func:`load_app_functions` for assertions about *behaviour*.  It ``exec``s the
  named top-level functions out of the source, so the test runs the app's own code
  rather than a copy that can drift from it.

The copies this replaced were a real source of false confidence: a test suite whose
"the app matches" guard only greps for a few lines inside a function body cannot
notice the function's *signature* changing.  ``span_context_bounds`` grew a
``Pipeline`` parameter and every mirrored test stayed green, because the mirror
reimplemented the old ``n_bars`` signature and the grep still found the same lines.
:func:`load_app_functions` has no such blind spot -- it is the real code.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import os
import re
import typing

APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)

if not os.path.exists(APP):  # pragma: no cover - app.py ships with the repo
    import pytest

    pytest.skip("app.py not present", allow_module_level=True)


def run_app_unanswered(timeout=300):
    """Run the real ``app.py`` to its **first paint** -- the unanswered gate.

    Deliberately does *not* seed the answer, so the run stops at the one question the
    app asks before it builds anything.  This is the only way to observe the gate as a
    reader sees it, and the state the seeded :func:`run_app` skips straight past.

    Cheap by comparison: no pipeline is built, so it returns in seconds.
    """
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=timeout).run()


def run_app(timeframe="1d", timeout=300):
    """Run the **real** ``app.py`` through Streamlit's own ``AppTest``.

    Everything else in this file reads ``app.py`` as text or ``exec``s individual
    functions out of it.  Both are blind to a whole class of failure -- a widget key
    registered twice in one pass, a call to an undefined name, anything that only
    happens when the module is executed end to end -- and that class is not
    hypothetical: ``render_ticker_input`` shipped a duplicated ``st.text_input``
    block, and ``main()`` shipped a duplicate ``render_scope_ticker``.  Each raised
    ``StreamlitDuplicateElementKey`` and took the whole page down.

    Every existing test was green through both.  The two tests guarding the ticker
    input each asserted that *one* stub contained an input, which was true of both
    stubs; nothing checked they were not drawn in the same pass.

    This runs the script.  ``timeframe`` seeds :data:`SESSION_TIMEFRAME_KEY` so the
    gate is already answered and the page builds in a single pass; seeding it rather
    than driving the radio keeps the run from depending on widget-event ordering.

    Slow (tens of seconds: it builds two pipelines and charts them), so it is a
    deliberate handful of tests rather than a per-fixture fixture.
    """
    from streamlit.testing.v1 import AppTest

    # Read the key off the app rather than restating it: a test that hard-codes
    # "session_timeframe" seeds a key the app may no longer use, and then quietly
    # asserts nothing -- the gate simply re-asks and the page never builds.
    import re as _re

    match = _re.search(r'^SESSION_TIMEFRAME_KEY\s*=\s*"([^"]+)"', app_source(), _re.M)
    assert match, "SESSION_TIMEFRAME_KEY is not declared in app.py"

    at = AppTest.from_file(APP, default_timeout=timeout)
    at.session_state[match.group(1)] = timeframe
    at.run()
    return at


@functools.lru_cache(maxsize=1)
def app_source() -> str:
    """The whole of ``app.py`` as text."""
    with open(APP, encoding="utf-8") as fh:
        return fh.read()


def app_text(pattern: str, what: str, flags: int = re.S) -> str:
    """The text matched by ``pattern``, asserted to exist with a useful message.

    Every source-text assertion in the suite used to open the file and re-run the
    same regex inline, then re-state a "not found" message.  Centring that here
    means a pattern that stops matching reports *what* it was looking for.
    """
    match = re.search(pattern, app_source(), flags)
    assert match, "{} not found in app.py".format(what)
    return match.group(0)


def app_block(pattern: str, what: str, flags: int = re.S) -> str:
    """Like :func:`app_text`, but yields capture group 1 when the pattern has one."""
    match = re.search(pattern, app_source(), flags)
    assert match, "{} not found in app.py".format(what)
    return match.group(1) if match.groups() else match.group(0)


def app_called_names() -> set:
    """Every ``func.attr(...)`` name ``app.py`` actually *calls*, per the AST.

    Text assertions like ``"components.html(" not in app_source()`` are misleading in
    both directions.  A comment explaining *why* a call is wrong necessarily contains
    the forbidden spelling, so a grep for it matches the explanation and fails a
    correct file -- which is how the deprecated-API guards here started asserting
    against their own docstrings.  The mirror failure is worse: renaming the alias
    (``import streamlit.components.v1 as c``) slips past a text match entirely.

    The AST has neither problem.  Comments and docstrings are not statements, so they
    cannot appear here, and an alias still resolves to the attribute name it is
    called through.  Only the final segment is kept (``st.iframe`` -> ``iframe``),
    which is what these assertions actually care about: which command is invoked.
    """
    tree = ast.parse(app_source())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def expected_tab_count() -> int:
    """How many tabs ``app.py`` builds -- read from ``TAB_ORDER`` rather than counted.

    The gate tests assert that pressing *Continue* actually reached the page, and they
    did that by asserting ``len(at.tabs) == 6``.  That number is a **copy** of
    ``TAB_ORDER``: adding a tab makes every one of those assertions fail while saying
    nothing about the thing they exist to prove, which is the worst failure mode for a
    wiring test -- it looks like the gate regressed.

    Derived here so the two cannot drift.  Reads the tuple rather than counting the
    ``with tabs[...]`` blocks, because the tuple is what ``st.tabs`` is actually given
    and a stub that never renders would not change it.
    """
    block = app_block(r"TAB_ORDER\s*=\s*\((.*?)\)", "TAB_ORDER")
    return len(re.findall(r'"([^"]+)"', block))


def app_defined_names() -> set:
    """Every top-level ``def`` in ``app.py``, per the AST.

    The complement of :func:`app_called_names`, for the same reason: asserting a
    helper is *gone* from the text cannot distinguish a deleted function from an
    explanation of its deletion, which every removal comment in this repo contains.

    Being top-level only is deliberate.  ``app.py`` nests nothing that shadows a
    module-level helper, and a recursive walk would pick up the ``def``s inside
    docstring-free nested scopes -- the local ``chart_config`` closures and the like --
    making a name-based assertion ambiguous.
    """
    return {n.name for n in ast.parse(app_source()).body
            if isinstance(n, ast.FunctionDef)}


def price_tab_body() -> str:
    """The ``with tab_price:`` block of ``main()``, up to the next tab.

    Two test files slice the Price tab this way.  The slice has to stop at the
    *next* ``with tab_`` block rather than running to the end of the file, because
    the assertions are about what the Price tab draws and not about its neighbours.

    It used to stop at ``with tab_chart:``.  That tab is gone, so a pattern naming
    it no longer matched anything and ``app_block`` raised -- taking a dozen
    unrelated Price-tab assertions down with it.  Stopping at any following
    ``with tab_`` keeps the slice correct when tabs are added or removed, which is
    exactly the failure this file records having had before.
    """
    return app_block(
        r"with tab_price:(.*?)\n    with tab_",
        "Price tab body",
    )


def load_app_functions(names, namespace=None) -> dict:
    """``exec`` the named top-level functions out of ``app.py`` into one namespace.

    ``names`` is a collection of function names.  Every one must exist at module
    level or the call fails loudly -- a helper renamed in the app has to break the
    suite rather than quietly drop out of it.

    The namespace is seeded with the imports the extracted bodies reference.  Those
    names are supplied rather than executed because importing ``app.py``'s own
    import block would drag in Streamlit and Plotly and run the UI; supplying them
    directly gets the same objects without the side effects.

    Callers pass ``namespace`` to override or add entries.  A caller that needs a
    module-level constant (``MIN_QUERY_BARS`` and friends) passes it in, so the
    value comes from the app rather than being restated in the test and drifting.
    """
    source = app_source()
    tree = ast.parse(source)

    ns: dict = {
        "np": _numpy(),
        "pd": _pandas(),
        "Any": typing.Any,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "Sequence": typing.Sequence,
        "Tuple": typing.Tuple,
        # ``timeframe_scope`` is a ``@contextmanager``, so its own decorator has to
        # resolve in this namespace.  Supplied for the same reason as ``np`` above:
        # importing app.py's import block would run the UI.
        "contextmanager": contextlib.contextmanager,
    }
    if namespace:
        ns.update(namespace)

    # Module-level constants the extracted bodies read.  Reading them out of the
    # source rather than restating them is the point: a test that hard-codes
    # ``MIN_QUERY_BARS = 8`` keeps passing after the app changes it to 10.
    ns.update(_module_constants(tree))
    # Real timeframe objects, so an extracted body that resolves a resolution gets the
    # same registry the app uses rather than a stand-in.  Supplied rather than
    # ``exec``d because ``timeframes.py`` is a *module*, not app source -- importing it
    # is cheap and has no side effects, unlike importing ``app.py``.
    from timeseries.timeframes import (  # noqa: PLC0415 - deliberately local
        DEFAULT_TIMEFRAME, TIMEFRAMES, get_timeframe, resolve_timeframe,
        timeframe_keys,
    )
    from timeseries.store import stamp_label as _store_stamp_label  # noqa: PLC0415

    # The two archive roots are ``os.path.join(...)`` expressions, so
    # ``_module_constants`` skips them -- deliberately, since evaluating an
    # arbitrary module-level expression could render the app.  They are supplied
    # here instead, derived from the same repository root the app uses, so a test
    # asserting "daily resolves to sp500_daily" is reading the real path rather
    # than a literal the test invented.
    import os as _os

    _repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    ns.update({
        "PANEL_ROOT": _os.path.join(_repo, "data", "sp500_panel"),
        "DAILY_ROOT": _os.path.join(_repo, "data", "sp500_daily"),
        "DEFAULT_TIMEFRAME": DEFAULT_TIMEFRAME,
        "TIMEFRAMES": TIMEFRAMES,
        "get_timeframe": get_timeframe,
        "resolve_timeframe": resolve_timeframe,
        "timeframe_keys": timeframe_keys,
        "ACTIVE_TIMEFRAME": [get_timeframe().key],
        "FORECAST_TIMEFRAME": [get_timeframe().key],
        # ``app.py``'s ``stamp_label`` delegates to the package's one definition of
        # what a bar looks like, so an extracted body that formats a tick needs the
        # real helper here -- or it fails with a ``NameError`` at draw time.
        "store_stamp_label": _store_stamp_label,
    })

    # The resolution accessors are loaded **together with the body that reads them**,
    # in dependency order.  They are module-level functions like any other, so this is
    # the same mechanism ``names`` already uses -- it just has to happen before the
    # bodies, because an extracted function that calls ``_tf()`` fails on a missing
    # name rather than at import.
    #
    # Ordering matters only because these four are mutually referential through the
    # module-level ``ACTIVE_TIMEFRAME`` list; ``timeframe_scope`` is the context
    # manager used to render one tab under another tab's resolution.
    # ``tf_key`` and ``state_key`` come first: they are the resolution-aware key
    # builders, and every bar-indexed key accessor (``price_brush_key`` and friends)
    # is defined in terms of ``state_key``.  Extracting one without the others fails
    # with a ``NameError`` rather than a useful message, so the chain is explicit.
    # ``panel_root_for`` joins the chain because it is the *other* resolution-aware
    # resolver in the app: ``_tf``/``tf_key`` answer "what resolution is in force"
    # and ``panel_root_for`` answers "which archive serves it", so every panel code
    # path calls it.  It needs the two ``*_ROOT`` path constants, which
    # ``_module_constants`` already supplies, and ``resolve_timeframe``, seeded above.
    dependency_order = ["tf_key", "state_key", "_tf", "active_length",
                        "active_horizons", "active_label",
                        "active_unit", "bar_unit", "active_min_query_bars",
                        "active_max_query_bars", "active_view_sessions",
                        "stamp_label", "stamp_span", "stamp_zone",
                        "stamp_zone_name", "stamp_column",
                        "panel_root_for",
                        "timeframe_scope"]
    for name in dependency_order:
        _exec_top_level(tree, ns, name, required=bool(_TF_USERS & set(names)))

    wanted = set(names)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), APP, "exec"), ns)
            found.add(node.name)

    missing = wanted - found
    assert not missing, "{} not found at module level in app.py".format(
        sorted(missing)
    )
    return ns


#: Names among ``dependency_order`` whose absence should not fail a caller.  Loading a
#: helper that no requested body uses is harmless; *failing* because the app renamed an
#: unrelated helper would be noise.
def _exec_top_level(tree: ast.Module, ns: dict, name: str, *, required: bool = False):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), APP, "exec"), ns)
            return True
    if required:
        raise AssertionError("{} not found at module level in app.py".format(name))
    return False


#: Extracted bodies that resolve a resolution, and therefore need the accessors above.
#: Computed from the AST rather than listed by hand so it cannot go stale: a new
#: helper that calls ``_tf()`` is picked up without editing this file.
def _timeframe_users(tree: ast.Module) -> set:
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = {c.func.id for c in ast.walk(node)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        if calls & {"_tf", "active_length", "active_horizons", "active_label",
                    "active_unit", "bar_unit", "active_min_query_bars",
                    "active_max_query_bars", "active_view_sessions",
                    "stamp_label", "stamp_span", "stamp_zone",
                    "stamp_zone_name", "stamp_column"}:
            names.add(node.name)
    return names


_TF_USERS = _timeframe_users(ast.parse(app_source()))


def _module_constants(tree: ast.Module) -> dict:
    """Top-level ``NAME = <literal>`` assignments, for the ones we supply.

    Only simple literals are read.  Anything computed is skipped rather than
    evaluated: a module-level expression in ``app.py`` can touch Streamlit, and a
    test helper must never render the app as a side effect of loading a constant.
    """
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = literal
    return found


def _numpy():
    import numpy

    return numpy


def _pandas():
    import pandas

    return pandas


def load_geometry():
    """The three Price-tab view helpers, straight out of ``app.py``.

    ``span_context_bounds``, ``shared_anchor`` and ``aligned_view`` decide where the
    query band and the match band are drawn, so "are they aligned?" is a question
    about these exact functions.

    They were previously re-implemented here as :mod:`test_price_tab_match_panel`
    mirrors and guarded by tests that grepped ``app.py`` for a handful of lines
    inside each body.  That arrangement gave false confidence in a specific,
    demonstrable way: ``span_context_bounds`` gained a ``Pipeline`` first parameter,
    the mirrors kept the old ``n_bars`` one, and **every mirrored test stayed
    green** -- the grep still found the same arithmetic, so it could not see the
    signature change.  Running the app's own code removes the class of bug rather
    than patching one instance of it.

    ``span_context_bounds`` takes a :class:`~timeseries.pipeline.Pipeline` in the
    app but only ever reads ``n_bars`` off it, so the tests pass the small stand-in
    below.  That keeps the real call convention -- a pipeline, not a count -- so a
    future signature change breaks these tests instead of being papered over.
    """
    ns = load_app_functions(
        {"span_context_bounds", "shared_anchor", "aligned_view"}
    )
    ns["Pipeline"] = _StubPipeline
    return ns


class _StubPipeline:
    """Carries only ``n_bars``, which is all ``span_context_bounds`` reads."""

    def __init__(self, n_bars: int):
        self.n_bars = int(n_bars)

class WidgetKeyError(RuntimeError):
    """Stand-in for Streamlit's ``StreamlitValueAssignmentNotAllowedError``.

    Same name and same meaning, raised locally so the test suite does not have to
    import Streamlit's exception class to assert on it.
    """


class WidgetAwareSessionState(dict):
    """A ``session_state`` that enforces Streamlit's one rule about widget keys.

    **A registered widget's key cannot be assigned to.**  Streamlit raises
    ``StreamlitValueAssignmentNotAllowedError``, and this exists so tests can
    reproduce that.  A bare ``dict`` -- which is what every other helper here
    passes -- silently accepts the write, which is precisely how the first version
    of the startup gate shipped a line that raises on the one pass it was meant to
    answer on.  That bug was invisible to 703 passing tests because the tests were
    checking *text*, not *behaviour*, and the behaviour only exists once something
    like this is driving the code.

    ``register_widget`` models the other half: drawing a widget claims its key, so
    any later assignment to it is an error rather than a silent no-op.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget_keys: set = set()

    def register_widget(self, key, value=None):
        """Claim ``key`` the way ``st.radio(key=...)`` does, and seed its value."""
        self.widget_keys.add(key)
        if value is not None and key not in self:
            dict.__setitem__(self, key, value)
        return self.get(key)

    def set_from_browser(self, key, value):
        """Deliver a widget value the way a real rerun does.

        Bypasses the write guard deliberately: this is the browser's event arriving,
        not application code assigning to a widget, and the distinction is the whole
        point of the guard.
        """
        dict.__setitem__(self, key, value)
        return value

    def __setitem__(self, key, value):
        if key in self.widget_keys:
            raise WidgetKeyError(
                "st.session_state.{} cannot be modified after the widget with that "
                "key is instantiated. Use a separate non-widget key to hold a "
                "value the app writes.".format(key)
            )
        dict.__setitem__(self, key, value)
