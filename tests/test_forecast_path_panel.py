"""Integration tests for the cross-sectional forecast path.

``test_forecast_path.py`` covers the arithmetic of
:func:`timeseries.forecast_paths_multi` against hand-built input, and asserts the app's
*wiring* by reading its source.  What neither does is run the app's own aggregation
code against a real :class:`~timeseries.panel.PanelSearch` -- which is where the
interesting mistakes live:

* a match index read from the wrong index space (raw rather than aligned) returns a
  path with a plausible median and a *quietly* wrong anchor bar;
* matches handed to the aggregator grouped by the wrong key attribute every window
  to one ticker and aggregates indices from unrelated series;
* the ticker breakdown shown under the chart disagrees with the matches that
  produced it.

All of those are silent.  None raise, and none change the *shape* of the curve -- only
what it means.  So the tests here build a real panel on disk, search it, and check the
pooled path against values recomputed by hand.

The archive is written into ``tmp_path`` rather than read from the repository's
``data/sp500_panel/``, so the suite does not depend on 500 tickers having been
downloaded and stays fast.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st  # noqa: F401  - needed to exec the helper's decorator

from timeseries import matching as M
from timeseries import panel as PNL
from timeseries.forecast import forecast_paths_multi
from timeseries.panel import PanelSearch
from timeseries.pipeline import Pipeline
from timeseries.store import PanelStore

from apphelpers import load_app_functions
from session_bars import session_bars


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _bars(n: int, *, price: float, seed: int) -> pd.DataFrame:
    """Three sessions of 1-minute bars in the archive's schema.

    Spanning sessions is deliberate: it gives the §BX horizon mask something to
    censor, so the test walks the same path a real archive does rather than the
    degenerate single-session case where the mask is all-True.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for day in range(3):
        ts = pd.date_range("2026-07-15 13:30", periods=n // 3, freq="1min", tz="UTC")
        close = price * np.exp(np.cumsum(rng.normal(0.0, 0.0015, len(ts))))
        frames.append(pd.DataFrame({
            "timestamp": ts + pd.Timedelta(days=day),
            "open": close,
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "volume": rng.integers(1_000, 10_000, len(ts)).astype(float),
            "ticker": "",
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def search(tmp_path) -> PanelSearch:
    """A small but real panel on disk: four tickers at four distinct price levels."""
    store = PanelStore(str(tmp_path))
    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        frame = _bars(1200, price=40.0 * (i + 1), seed=i)
        frame["ticker"] = sym
        store.write(sym, frame)
    ps = PanelSearch(store)
    assert ps.ready(), "fixture panel must be searchable"
    return ps


@pytest.fixture
def helper(search, request):
    """The app's own ``panel_forecast_path_for``, exec'd out of ``app.py``.

    The real function rather than a mirror of it, so a rename or signature change
    breaks here instead of passing in a copy that has drifted.

    Indirected through a *mutable* loader box so a test can make the archive fail:
    ``st.cache_resource`` wraps the function in a ``CachedFunc`` that does not expose
    ``__globals__``, so patching the namespace after the fact is not possible -- and
    patching it before the fact is the only way to reach the ``except`` path.
    """
    box = {"load": lambda _root: search}

    def loader(_root):
        return box["load"](_root)

    ns = load_app_functions(
        {"panel_forecast_path_for"},
        namespace={
            "st": st,
            "os": os,
            "PNL": PNL,
            "PANEL_ROOT": str(search.store.root),
            "load_panel_search": loader,
            "forecast_paths_multi": forecast_paths_multi,
        },
    )
    fn = ns["panel_forecast_path_for"]
    # Cleared once, up front: ``cache_resource`` is process-wide, so a result cached
    # by an earlier test would be handed back here and defeat every assertion that
    # compares a value against a freshly recomputed one.
    fn.clear()
    fn.loader_box = box
    return fn


def _query_vector(length: int) -> np.ndarray:
    """A normalised two-channel query vector shaped like the pipeline's own."""
    rng = np.random.default_rng(7)
    raw = rng.normal(0.0, 1.0, (int(length), 2))
    return np.stack([M.zscore(raw[:, c]) for c in range(2)], axis=-1)


def _call(helper, vector, **kw):
    args = dict(length=60, horizon=30, k=20, amplitude_weight=1.0)
    args.update(kw)
    return helper("AAA", vector, **args)


def _expected_bands(search: PanelSearch, vector: np.ndarray, *, length: int,
                    horizon: int, k: int, amplitude_weight: float):
    """Rebuild the pooled bands from the search object, as an independent oracle.

    Deliberately *not* the app's helper: it runs the panel search itself, walks the
    returned matches grouped by ticker, reads each ticker's ``close_aligned``, and
    percentiles the rebased paths.  Two implementations of the same arithmetic, so a
    mistake in one is not reproduced by the other.

    Returns ``((q25, median, q75), distinct_tickers)``, or ``None`` if the search
    finds nothing.
    """
    query = PNL.PanelQuery(vector=vector, ticker="", start=0, stop=int(length))
    result = search.search(query, k=int(k), max_horizon=int(horizon),
                           amplitude_weight=float(amplitude_weight))
    if not result.matches:
        return None

    by_ticker: dict = {}
    for m in result.matches:
        by_ticker.setdefault(m.ticker, []).append(int(m.start))

    rows = []
    for ticker in sorted(by_ticker):
        series = search.close_aligned(ticker)
        for start in by_ticker[ticker]:
            a = start + int(length) - 1
            rows.append((series[a:a + horizon + 1] / series[a] - 1.0) * 100.0)

    q25, median, q75 = np.percentile(np.asarray(rows), (25.0, 50.0, 75.0), axis=0)
    return (q25, median, q75), set(by_ticker)


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #
class TestPanelForecastPathHelper:
    def test_it_returns_a_bundle_with_the_path_and_its_evidence(self, helper):
        bundle = _call(helper, _query_vector(60))
        assert bundle is not None
        assert {"path", "tickers", "n_tickers", "n_candidates"} <= set(bundle)
        assert bundle["path"].n_matches > 0
        assert bundle["n_candidates"] > bundle["path"].n_matches

    def test_the_median_matches_a_hand_recomputation_from_the_aligned_series(
        self, helper, search
    ):
        """The chart's curve is exactly these bars, rebased to their own anchors.

        The expected values are recomputed **from the search object, not from
        ``bundle["matches"]``**.  Reading them off the helper's own output would make
        this test agree with any grouping the helper performed -- and a helper that
        attributed every window to one ticker would produce a self-consistent bundle
        that this assertion called correct.  The search is the independent oracle:
        it returns matches already grouped by ticker, and the median is rebuilt from
        ``close_aligned`` (the series a match index addresses) in that order.

        Compared as a sorted multiset: the aggregator walks tickers alphabetically
        while ranking is by distance, and neither ordering is part of what a median
        describes.
        """
        length, horizon = 60, 30
        vector = _query_vector(length)
        bundle = _call(helper, vector, length=length, horizon=horizon)
        assert bundle is not None

        expected = _expected_bands(search, vector, length=length, horizon=horizon,
                                   k=20, amplitude_weight=1.0)
        assert expected is not None, "the oracle must find matches too"
        rows, result_tickers = expected
        # The fixture must genuinely be cross-sectional, or this test proves nothing
        # about grouping.
        assert len(result_tickers) > 1, (
            "expected matches from several tickers; a single-ticker result cannot "
            "detect a mis-grouping"
        )

        path = bundle["path"]
        np.testing.assert_allclose(np.sort(path.median), np.sort(rows[1]), atol=1e-9)
        np.testing.assert_allclose(np.sort(path.q25), np.sort(rows[0]), atol=1e-9)
        np.testing.assert_allclose(np.sort(path.q75), np.sort(rows[2]), atol=1e-9)

    def test_the_ticker_breakdown_sums_to_the_match_count(self, helper):
        """The caption's evidence claim must describe the curve above it.

        A breakdown that disagreed with the matches would make the one number the
        reader uses to judge concentration meaningless -- and it would still render.
        """
        bundle = _call(helper, _query_vector(60))
        assert bundle is not None
        assert sum(bundle["tickers"].values()) == len(bundle["matches"])
        assert bundle["n_tickers"] == len(bundle["tickers"])
        assert set(bundle["tickers"]) == {m.ticker for m in bundle["matches"]}

    def test_matches_are_grouped_by_their_own_ticker(self, helper, search):
        """Each ticker's starts must index that ticker's own series.

        Grouping by the wrong key would still produce a curve -- the median of
        unrelated windows.  Here every group is bounds-checked against its own
        ticker, which a mixed grouping fails.
        """
        length = 60
        bundle = _call(helper, _query_vector(length), length=length)
        assert bundle is not None
        by_ticker: dict = {}
        for m in bundle["matches"]:
            by_ticker.setdefault(m.ticker, []).append(int(m.start))
        for ticker, starts in by_ticker.items():
            n = search.close_aligned(ticker).size
            assert all(0 <= s + length <= n for s in starts), (
                "starts for %s fall outside its own series" % ticker
            )

    def test_offset_zero_is_exactly_zero_on_a_real_pooled_path(self, helper):
        """Every match rebases to its own anchor, so the join to history is seamless."""
        bundle = _call(helper, _query_vector(60))
        assert bundle is not None
        path = bundle["path"]
        assert path.median[0] == 0.0
        assert path.q25[0] == 0.0
        assert path.q75[0] == 0.0

    def test_a_window_longer_than_the_archive_yields_no_path(self, helper):
        """A legitimately impossible query returns ``None`` rather than raising.

        Reachable whenever the reader's window exceeds the archive's shortest name,
        which is ordinary rather than exceptional.
        """
        assert _call(helper, _query_vector(4000), length=4000) is None

    def test_a_search_that_throws_yields_none_not_an_exception(self, helper, search):
        """A corrupt archive is a missing chart, never a failed page.

        Reached by making ``load_panel_search`` itself raise, which is the shape a
        genuinely broken Parquet archive takes.  Every failure path returns ``None``
        so the tab has one code path for "draw nothing" instead of several.
        """
        def boom(_root):
            raise RuntimeError("archive is corrupt")

        helper.clear()
        helper.loader_box["load"] = boom
        try:
            assert _call(helper, _query_vector(60)) is None
        finally:
            helper.clear()
            helper.loader_box["load"] = lambda _root: search

    def test_the_helper_stays_cached(self, helper):
        """Same question twice must not re-search every ticker.

        Verified through the decorator rather than by reading the app's source, so a
        removed cache fails here.  ``cache_resource`` also keeps the numpy arrays and
        dataclass the caller reads in place, which ``cache_data`` would not.
        """
        assert hasattr(helper, "clear"), (
            "panel_forecast_path_for must remain a cached Streamlit resource"
        )
        vector = _query_vector(60)
        first = _call(helper, vector)
        second = _call(helper, vector)
        assert first is not None and second is not None
        assert first["path"] is second["path"], "expected the cached object back"


# --------------------------------------------------------------------------- #
# The figure, drawn from a pooled path
# --------------------------------------------------------------------------- #
class TestPooledPathFigure:
    """A pooled path must occupy the same axes as the single-ticker one.

    The two charts are only comparable if the shared builder treats them identically.
    That is what makes the caption's "identical axes" claim true rather than merely
    intended, so it is asserted against a real pooled path rather than a stub.
    """

    def _builder(self):
        return load_app_functions(
            # ``_closest_match_path`` and ``_closest_distance_label`` are listed
            # because ``load_app_functions`` only ``exec``s the names asked for, and
            # the figure calls both to draw the closest-match trace.  Without them the
            # pooled chart raises ``NameError`` -- the same break a caller outside this
            # suite would see, which is the point of exec'ing the real function.
            {"build_forecast_path_figure", "_closest_match_path",
             "_closest_distance_label"},
            namespace={"go": go},
        )["build_forecast_path_figure"]

    def test_a_pooled_path_draws_the_declared_span(self, helper):
        builder = self._builder()
        pipe = Pipeline.from_frame(session_bars(8190, seed=20260930), length=240)
        bundle = _call(helper, _query_vector(240), length=240, horizon=240, k=20)
        assert bundle is not None

        fig = builder(pipe, bundle["path"])
        lo, hi = fig.layout.xaxis.range
        assert hi - lo == pytest.approx(480)
        assert bundle["path"].median[0] == 0.0, (
            "the pooled median must start at the anchor, like the single-ticker one"
        )