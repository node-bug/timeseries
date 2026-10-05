"""Every ``module.NAME`` reference in ``app.py`` must resolve.

``app.py`` is a Streamlit script, so it cannot be imported under pytest and no amount
of reading its source proves that an attribute it reaches for actually exists.  The
failure is also unusually well camouflaged: the Price tab wraps its match in

    try:
        auto_result = pipe.match(query, k=<HERE>, ...)
    except Exception as exc:
        st.warning("Could not compute matches for this window: %s" % exc)

so a wrong constant does not crash the page.  It renders a chart with no match panel
below it and one amber line saying so -- which is exactly what happened when the
decoupled Price tab reached for ``M.DEFAULT_K`` (``matching``) rather than
``DEFAULT_K`` (``pipeline``).  Nothing else in the suite caught it: the symbol is
only evaluated inside that ``except``, so the whole suite stayed green while a
user-visible feature was dead.

This is a static check and deliberately so -- it needs no interpreter, no fixture and
no network.  It resolves every ``M.``/``F.``/``MP.``/``SYNC.``/``ST.`` attribute
reference against the real installed module, which is the one thing source reading
cannot do.
"""

from __future__ import annotations

import ast
import importlib
import re

import pytest

from apphelpers import app_source

#: Module aliases ``app.py`` imports.  Resolved against the installed package, so the
#: check fails if an alias is renamed without its uses following.
ALIASES = {
    "F": "timeseries.fetch",
    "M": "timeseries.matching",
    "MP": "timeseries.matrix_profile",
    "SYNC": "timeseries.sync",
}

#: Names imported directly (``from timeseries.pipeline import DEFAULT_K``).  These are
#: ``Name`` nodes rather than ``Attribute``, so they need their own pass.
DIRECT = {
    "walk_forward": "timeseries.backtest",
    "forecast_paths": "timeseries.forecast",
    "FEATURE_COLUMNS": "timeseries.features",
    "PanelSearch": "timeseries.panel",
    "Pipeline": "timeseries.pipeline",
    "PanelStore": "timeseries.store",
    # The symbol registry.  ``read_sectors`` is the one place the ``unknown``
    # marker is filtered, so a rename here would not crash the page -- it would
    # silently start reporting a same-sector percentile pooled over every
    # unlabelled symbol, which is exactly the kind of wrong this check exists
    # to catch.
    "CONSTITUENTS_NAME": "timeseries.store",
    "constituents_path": "timeseries.store",
    "read_sectors": "timeseries.store",
}


def _alias_attrs():
    """Every ``ALIAS.name`` referenced anywhere in ``app.py``."""
    tree = ast.parse(app_source())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ALIASES:
                found.add((node.value.id, node.attr))
    return found


def _direct_names():
    """Every ``DIRECT`` name referenced anywhere in ``app.py``."""
    tree = ast.parse(app_source())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return {name for name in DIRECT if name in used}


@pytest.mark.parametrize("alias,name", sorted(_alias_attrs()))
def test_module_attribute_resolves(alias, name):
    """``alias.name`` exists on the module ``alias`` names.

    Includes attributes inside strings and comments only by accident -- ``ast``
    cannot see them -- which is fine: the point is to catch the ones that are *called*
    or passed, and those are statements.
    """
    module = importlib.import_module(ALIASES[alias])
    assert hasattr(module, name), (
        "app.py references {}.{}, which does not exist. If the reference is inside a "
        "try/except the page will not crash -- it will silently render without the "
        "feature that uses it.".format(alias, name)
    )


@pytest.mark.parametrize("name", sorted(_direct_names()))
def test_direct_import_resolves(name):
    """A directly-imported constant resolves -- ``DEFAULT_K`` is the regression."""
    module = importlib.import_module(DIRECT[name])
    assert hasattr(module, name), (
        "app.py imports {} from {}, which does not exist".format(
            name, DIRECT[name])
    )


def test_no_attribute_is_written_but_never_read():
    """A sanity check on the check: the pass must actually find references.

    Without this the two tests above would pass vacuously if ``app_source()`` ever
    returned something ``ast`` could not parse into attribute accesses, and a green
    suite would mean nothing.
    """
    assert len(_alias_attrs()) > 10, (
        "only {} aliased attribute references found -- the extraction is broken"
        .format(len(_alias_attrs()))
    )
    assert _direct_names(), "no directly-imported names found -- extraction is broken"


def _literal_format_calls():
    """Every ``"<literal>".format(...)`` in ``app.py``, as ``(lineno, fmt, args, kw)``."""
    tree = ast.parse(app_source())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)):
            out.append((node.lineno, node.func.value.value, node.args,
                        node.keywords))
    return out


def test_format_placeholder_count_matches_the_argument_count():
    """``str.format`` raises at *runtime* on an arity mismatch, not at import.

    A caption with nine placeholders and eight arguments parses fine, passes every
    import-time check, and takes the whole page down on first render -- and because
    the Price tab's failure mode is a chart with no match panel, it reads as "no match"
    rather than as a crash.

    Calls that use ``**kwargs`` or named fields are skipped: their arity is a
    question about *which* keywords were passed, which this check cannot answer, and
    treating them as zero-argument calls would flag correct code.
    """
    bad = []
    for lineno, fmt, args, keywords in _literal_format_calls():
        if keywords or any(isinstance(a, ast.Starred) for a in args):
            continue
        n = len(re.findall(r"\{(?!\{)[^{}]*\}", fmt))
        if n != len(args):
            bad.append("line {}: {} placeholders, {} args".format(lineno, n, len(args)))
    assert not bad, "format-string arity mismatches:\n  " + "\n  ".join(bad)


def test_a_thousands_separator_is_never_applied_to_a_string():
    """``"{:,}".format("3,632,136")`` raises ``Cannot specify ',' with 's'``.

    This is the exact bug the cross-sectional match caption shipped with: the counts
    were pre-formatted with ``"{:,}".format(n)`` and then passed into an outer format
    that applied ``{:,}`` again.  Python raises ``ValueError: Cannot specify ',' with
    's'`` because the comma grouping is an integer-only spec, so a perfectly ordinary
    caption took the entire page down on render.

    The rule it encodes: **format the number once, with the separator, in the outer
    format.**  A ``{:,}`` slot must be fed a raw ``int``.
    """
    offenders = []
    for lineno, fmt, args, keywords in _literal_format_calls():
        if keywords:
            continue
        slots = list(re.finditer(r"\{(?!\{)([^{}]*)\}(?!\})", fmt))
        for i, slot in enumerate(slots):
            if "," not in slot.group(1) or i >= len(args):
                continue
            arg = args[i]
            # An argument that is itself a "{:,}"-formatted string literal is the bug.
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "format"
                    and isinstance(arg.func.value, ast.Constant)
                    and isinstance(arg.func.value.value, str)
                    and "," in arg.func.value.value):
                offenders.append(
                    "line {}: slot {!r} is fed an already-formatted string"
                    .format(lineno, slot.group(0))
                )
    assert not offenders, (
        "'{:,}' applied to an already-formatted string:\n  " + "\n  ".join(offenders)
    )


def test_the_price_tab_caption_formats_cleanly():
    """The cross-sectional caption must render, with representative values.

    Named specifically because it is the caption whose failure removed the Price
    tab's entire match panel: the exception is raised during render, and the tab's
    documented behaviour for a missing match is to draw *nothing* below the query
    chart, so a formatting bug looks identical to "the search found no matches".
    """
    calls = [c for c in _literal_format_calls()
             if "Closest match anywhere" in c[1]]
    assert len(calls) == 1, (
        "expected exactly one cross-sectional match caption, found {}"
        .format(len(calls))
    )
    _lineno, fmt, _args, _kw = calls[0]
    rendered = fmt.format(
        "NVDA", "2026-09-30", 240, "0.00%", "0.02%", 10, 3632136, 503
    )
    assert "NVDA" in rendered and "2026-09-30" in rendered
    assert "3,632,136" in rendered, "counts must be thousands-separated"
    assert "503" in rendered


def test_price_tab_match_is_not_silenced_by_its_except_clause():
    """The Price tab's quick match must be drawn from a real constant.

    The broad ``except`` around the match search is deliberate -- one bad window, or
    one unreadable archive, must not take the chart down -- but it also means a typo
    in the call renders as a silently missing match panel rather than an error.  So
    the reference that matters is pinned explicitly: the chart below the query is the
    Price tab's reason for existing, and it must be drawn from a real constant.

    **This used to pin ``pipe.match(query, k=DEFAULT_K, ...)``**, because the Price
    tab searched the fetched ticker's own history.  It now searches the whole S&P 500
    panel, so the call is ``cross_sectional_match(...)`` and the invariant moved with
    it: the match count is still ``DEFAULT_K`` from ``pipeline`` rather than anything
    out of ``matching``, which is the mistake the test was written for (``M.DEFAULT_K``
    does not exist, and reaching for it cost the match panel once already).

    Kept as text rather than behaviour because rendering the tab under pytest is not
    possible, and ``test_module_attribute_resolves`` already covers *whether the
    attribute exists*; this covers *which one is used here*, which no import check can
    see.
    """
    body = app_source()
    assert re.search(r"cross_sectional_match\(\s*pipe,\s*symbol,\s*start_idx,\s*stop_idx,", body), (
        "the Price tab's quick match must go through cross_sectional_match, which "
        "searches the whole panel; a missing call silently drops the match panel "
        "below the query chart"
    )
    assert re.search(r"k=DEFAULT_K,\s*amplitude_weight=", body), (
        "the cross-sectional match must use pipeline.DEFAULT_K; a different constant "
        "here silently drops the match panel below the query chart"
    )
    assert "M.DEFAULT_K" not in body, (
        "M.DEFAULT_K does not exist -- matching has no such constant"
    )


def test_panel_search_is_reached_through_a_module_that_exists():
    """``PNL`` must resolve to a real module and carry the names used.

    The cross-sectional path builds a ``PanelQuery`` and reads ``PanelSearch`` inside
    ``app.py``.  Both are attribute references on a module alias, and the existing
    alias pass does not know about ``PNL`` -- so a rename would reach runtime as an
    ``AttributeError`` swallowed by the Price tab's own ``except``, which is exactly
    the silenced-failure mode above.
    """
    mod = importlib.import_module("timeseries.panel")
    assert hasattr(mod, "PanelQuery"), "PNL.PanelQuery must exist"
    assert hasattr(mod, "PanelSearch"), "PNL.PanelSearch must exist"
    assert "from timeseries import panel as PNL" in app_source(), (
        "app.py must alias timeseries.panel as PNL for the cross-sectional search"
    )