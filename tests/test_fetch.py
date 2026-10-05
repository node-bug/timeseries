"""Tests for :mod:`timeseries.fetch` -- fetching an arbitrary ticker.  PLAN.md §B/§F.

The rule this module is built around is §B's: **the archive is the product, not the
download**.  Downloading must therefore be safe to do on demand from the dashboard --
it must never touch an existing file, and it must never invent history it did not get.

Three failure modes are worth the tests here, and each is silent rather than loud:

* **Yahoo answers an over-long intraday request with an empty frame**, not an error.
  Untreated, that renders as a blank chart that looks like a broken app.  So an empty
  chunk is recorded in ``FetchResult.errors`` and reported.
* **Chunk boundaries overlap by one bar.**  Leaving that in duplicates rows, which
  inflates the bar count and pushes the de-duplication question onto every consumer.
* **A user-typed symbol becomes a filename.**  ``../../etc/passwd`` reaching a path
  join would escape the data directory, so validation is enforced at normalisation
  time -- the single choke point every symbol passes through.

Nothing here touches the network.  ``yfinance`` is replaced by a fake whose response
shape mirrors the real one (Title-case columns, a tz-aware ``Datetime`` index),
because the shape is the thing most likely to change upstream and these tests are
here to pin the *translation*, not Yahoo.
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from apphelpers import app_block, app_source, app_text
from timeseries import fetch as F
from timeseries.store import OHLCV, yahoo_symbol


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def fake_yahoo_response(n: int = 5, *, start: str = "2026-09-30 09:30",
                        freq: str = "1min") -> pd.DataFrame:
    """A response shaped like the real thing: Title-case columns, ``Datetime`` index."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York", name="Datetime")
    return pd.DataFrame(
        {
            "Open": range(100, 100 + n),
            "High": range(101, 101 + n),
            "Low": range(99, 99 + n),
            "Close": range(100, 100 + n),
            "Volume": [1_000] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=idx,
    )


def moving_tape(n: int = 400) -> pd.DataFrame:
    """A fake response with *moving* prices, which features actually require.

    A flat series is the obvious way to fake OHLCV and it silently produces nothing:
    log returns are all zero, so every rolling standard deviation is zero, so every
    ``return_z`` is NaN and the warm-up drop removes the entire frame.  The pipeline
    then reports ``ready`` with zero bars, which looks like a pass.  The drift and
    wobble below exist to keep price non-degenerate.
    """
    idx = pd.date_range("2026-09-30 09:30", periods=n, freq="1min",
                        tz="America/New_York", name="Datetime")
    steps = np.sin(np.arange(n) / 7.0) * 0.05 + 0.01
    close = 100.0 + np.cumsum(steps)
    volume = 1000.0 + (np.arange(n) % 17) * 10.0
    return pd.DataFrame(
        {"Open": close, "High": close + 0.1, "Low": close - 0.1,
         "Close": close, "Volume": volume},
        index=idx,
    )


class FakeYFinance:
    """Stand-in for the ``yfinance`` module, recording the requests it received.

    ``responses`` is a frame, a callable taking ``(start, end)``, or a list cycled
    across calls -- which is how a test serves one chunk differently from the next.
    """

    def __init__(self, responses=None):
        self._responses = responses
        self.calls: list = []

    def Ticker(self, symbol):  # noqa: N802 - mirrors yfinance's capitalised API
        fake = self

        class _Ticker:
            def history(self, *, interval, start, end, **kwargs):
                fake.calls.append({"symbol": symbol, "interval": interval,
                                   "start": start, "end": end})
                resp = fake._responses
                if callable(resp):
                    return resp(start, end)
                if isinstance(resp, list):
                    return resp[min(len(fake.calls) - 1, len(resp) - 1)]
                return resp

        return _Ticker()


class FlakyYFinance(FakeYFinance):
    """Fails the ``fail_at``-th request, to prove one bad chunk is survivable."""

    def __init__(self, responses=None, fail_at: int = 2):
        super().__init__(responses)
        self.fail_at = fail_at

    def Ticker(self, symbol):  # noqa: N802
        fake = self

        class _Ticker:
            def history(self, *, interval, start, end, **kwargs):
                fake.calls.append({"symbol": symbol, "interval": interval,
                                   "start": start, "end": end})
                if len(fake.calls) == fake.fail_at:
                    raise RuntimeError("network flake")
                return fake_yahoo_response(n=20)

        return _Ticker()


@pytest.fixture
def fake_yf(monkeypatch):
    """Install a fake ``yfinance`` module and return it for assertions."""
    def _install(responses=None, **kwargs) -> FakeYFinance:
        fake = (FlakyYFinance(responses, **kwargs) if kwargs
                else FakeYFinance(responses))
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        return fake

    return _install


# --------------------------------------------------------------------------- #
# Symbol normalisation
# --------------------------------------------------------------------------- #
class TestNormalizeSymbol:
    @pytest.mark.parametrize("raw,expected", [
        ("aapl", "AAPL"),
        ("  msft  ", "MSFT"),
        ("brk.b", "BRK-B"),
        ("bf-b", "BF-B"),
        ("spy", "SPY"),
        ("^vix", "^VIX"),      # CBOE Volatility Index -- Yahoo's leading-^ marker
        ("^gspc", "^GSPC"),    # every Yahoo index symbol resolves the same way
    ])
    def test_canonicalises(self, raw, expected):
        assert F.normalize_symbol(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_rejects_empty(self, raw):
        """An empty box is the common case, and must not reach Yahoo as ``None``."""
        with pytest.raises(ValueError, match="No ticker entered"):
            F.normalize_symbol(raw)

    @pytest.mark.parametrize("raw", [
        "../../etc/passwd",
        "AA PL",
        "AA/PL",
        "A" * 40,
        "AAPL;drop",
        "A^B",              # '^' is a prefix marker, not an interior character
        "^^VIX",            # doubled prefix
        "^",                # prefix with nothing behind it
        "^" + "A" * 40,     # the length bound still applies after the caret
        "^../../etc/passwd",  # a caret must not smuggle a path through either
    ])
    def test_rejects_path_and_shell_hazards(self, raw):
        """A symbol becomes a filename, so the charset is a security boundary."""
        with pytest.raises(ValueError):
            F.normalize_symbol(raw)

    def test_class_shares_use_yahoos_dashed_spelling(self):
        """Wikipedia spells class shares ``BRK.B``; Yahoo answers only to ``BRK-B``.
        A user pasting the Wikipedia form must still be fetched."""
        assert yahoo_symbol(F.normalize_symbol("brk.b")) == "BRK-B"
        assert yahoo_symbol(F.normalize_symbol("BRK-B")) == "BRK-B"

    def test_caret_prefix_cannot_smuggle_a_path(self):
        """Admitting ``^VIX`` must not reopen the traversal hole the charset guards.

        ``normalize_symbol`` collapses ``.`` to ``-`` *before* the pattern runs, so
        ``^../../etc/passwd`` arrives at the regex as ``^--/--/etc/passwd`` and is
        refused on the mandatory leading alphanumeric.  Asserted rather than assumed,
        because the widened charset is exactly where that defence would regress.
        """
        assert F.normalize_symbol("^AAPL") == "^AAPL"
        with pytest.raises(ValueError):
            F.normalize_symbol("^../../etc/passwd")


class TestArchiveName:
    def test_round_trips_through_the_symbol_parser(self):
        """The dashboard labels an archive by parsing its filename back out."""
        first = pd.Timestamp("2026-09-01T13:30:00Z")
        last = pd.Timestamp("2026-09-30T20:00:00Z")
        name = F.archive_name("qqq", first, last)
        assert name == "QQQ_1min_20260901_20260930.csv"
        assert F.symbol_from_filename(name) == "QQQ"

    def test_caret_symbol_round_trips_through_the_parser(self):
        """``^VIX`` must be both written and read back by the naming convention.

        This is the guard for the ``_ARCHIVE_RE`` change: with the optional ``^``
        missing, ``archive_name`` would happily emit a name its own sibling parser
        could not read, and no fetch-path test would notice, because nothing on that
        path round-trips a filename.
        """
        first = pd.Timestamp("2026-09-01T13:30:00Z")
        last = pd.Timestamp("2026-09-30T20:00:00Z")
        name = F.archive_name("^vix", first, last)
        assert name == "^VIX_1min_20260901_20260930.csv"
        assert F.symbol_from_filename(name) == "^VIX"

    @pytest.mark.parametrize("name", [
        "manifest.csv", "notes.txt", "random.csv", "1min_20260901_20260930.csv", "",
    ])
    def test_unrelated_files_have_no_symbol(self, name):
        """Anything that is not an archive gets no guessed symbol."""
        assert F.symbol_from_filename(name) is None


class TestNoArchiveLookup:
    """``find_archives`` is gone, and this is the one test that says so.

    The archive picker was removed: the single-instrument tabs download their bars
    and hold them in memory, and the multi-ticker Panel tab reads its own Parquet
    archive through :mod:`timeseries.store`.  Nothing in the app resolves a symbol to
    a CSV any more.

    Its nine tests were left behind and failed on every single run with
    ``module 'timeseries.fetch' has no attribute 'find_archives'`` -- an AttributeError
    that says the suite is broken while telling you nothing about the code.  They
    were not deleted silently: this class asserts the same fact from the direction
    that actually matters, namely that the public surface does not advertise the
    capability and the app does not depend on it.  If a CSV lookup ever comes back,
    it has to come back with its own tests on purpose, rather than by restoring a
    function whose coverage quietly rotted while it was gone.
    """

    def test_the_library_does_not_advertise_an_archive_lookup(self):
        assert not hasattr(F, "find_archives"), (
            "find_archives is back -- re-add the tests that covered it, or drop this "
            "guard"
        )
        assert "find_archives" not in F.__all__

    def test_nothing_in_the_app_reads_a_symbol_csv(self):
        src = app_source()
        for gone in ("find_archives", "archive_candidates", "_resolve_source",
                     "source_cache_key", "available_csvs"):
            assert gone not in src, (
                "%s is back in app.py -- the archive/source picker was removed on "
                "purpose, so restore its behaviour and its tests together" % gone
            )


# --------------------------------------------------------------------------- #
# Response normalisation
# --------------------------------------------------------------------------- #
class TestNormalizeChunk:
    def test_title_case_index_becomes_the_schema_the_package_uses(self):
        """yfinance returns ``Close`` under a ``Datetime`` index; the app reads
        ``close`` in a ``timestamp`` column.  This is the seam most likely to break."""
        out = F._normalize_chunk(fake_yahoo_response())
        assert list(out.columns) == ["timestamp", *OHLCV]
        assert isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype)
        assert len(out) == 5

    def test_empty_response_yields_the_schema_not_a_crash(self):
        out = F._normalize_chunk(pd.DataFrame())
        assert out.empty
        assert "timestamp" in out.columns

    def test_multiindex_columns_are_flattened(self):
        """A multi-symbol request returns ``(field, ticker)`` labels.  Only the outer
        level is dropped -- that is where the price fields live."""
        raw = fake_yahoo_response(n=3)
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
        out = F._normalize_chunk(raw)
        assert list(out.columns) == ["timestamp", *OHLCV]

    def test_missing_price_columns_raise_rather_than_silently_producing_gaps(self):
        """A response shape change must surface loudly.  Returning a frame with no
        ``close`` would fail much later, inside feature construction, where the cause
        is no longer visible."""
        raw = fake_yahoo_response().drop(columns=["Close"])
        with pytest.raises(ValueError, match="missing the column"):
            F._normalize_chunk(raw)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
class TestFetchTicker:
    def test_single_chunk_returns_clean_bars(self, fake_yf):
        fake_yf(responses=fake_yahoo_response(n=300))
        result = F.fetch_ticker("aapl", days=1)
        assert result.ok
        assert result.symbol == "AAPL"
        assert result.n_bars == 300
        assert result.errors == []
        assert result.frame["ticker"].eq("AAPL").all()

    def test_symbol_is_uppercased_and_sent_to_yahoo_in_its_own_spelling(self, fake_yf):
        """Yahoo answers to ``BRK-B``, not the ``BRK.B`` Wikipedia uses.  Sending the
        wrong one returns zero bars with no error, so the conversion is not optional.

        Verified against the live endpoint rather than assumed: both spellings were
        requested and only the dashed form returned data.
        """
        fake = fake_yf(responses=fake_yahoo_response(n=10))
        F.fetch_ticker("brk.b", days=1)
        assert fake.calls[0]["symbol"] == "BRK-B"

    def test_index_prefix_reaches_yahoo_intact(self, fake_yf):
        """``^VIX`` must be sent to Yahoo with its caret.

        Stripping it would request a different instrument and return zero bars with
        no error at all -- the same silent failure the ``BRK.B`` -> ``BRK-B`` test
        above guards against, which is why it is worth pinning separately.
        """
        fake = fake_yf(responses=fake_yahoo_response(n=10))
        F.fetch_ticker("^vix", days=1)
        assert fake.calls[0]["symbol"] == "^VIX"

    def test_interval_is_always_one_minute(self, fake_yf):
        """The whole app's gap heuristics and rolling window assume 60-second bars.
        Another interval would not be a slower version of the same thing, it would be
        a wrong one.  Pinned so the constant cannot drift."""
        fake = fake_yf(responses=fake_yahoo_response(n=10))
        F.fetch_ticker("aapl", days=1)
        assert {c["interval"] for c in fake.calls} == {"1m"}

    def test_long_span_is_chunked_to_stay_inside_yahoos_limit(self, fake_yf):
        """Yahoo serves ~8 days of 1m per request, so a 20-day ask is split."""
        fake = fake_yf(responses=fake_yahoo_response(n=50))
        F.fetch_ticker("aapl", days=20)
        assert len(fake.calls) >= 3
        for call in fake.calls:
            span = (call["end"] - call["start"]).total_seconds() / 86400
            assert span <= F.CHUNK_DAYS

    def test_span_is_clamped_to_what_yahoo_can_serve(self, fake_yf):
        """Asking for a decade of 1m bars must not become a decade of requests."""
        fake = fake_yf(responses=fake_yahoo_response(n=10))
        F.fetch_ticker("aapl", days=3650)
        total = sum((c["end"] - c["start"]).total_seconds() / 86400 for c in fake.calls)
        assert total <= F.MAX_1M_DAYS + 1

    def test_bars_are_sorted_and_unique(self, fake_yf):
        """Chunks overlap at their boundary bar.  Concatenating them raw duplicates
        rows, inflating the count and pushing de-duplication onto every consumer."""
        fake_yf(responses=fake_yahoo_response(n=50))
        result = F.fetch_ticker("aapl", days=20)
        stamps = result.frame["timestamp"]
        assert stamps.is_monotonic_increasing
        assert not stamps.duplicated().any()

    def test_price_columns_are_numeric(self, fake_yf):
        """Yahoo occasionally returns these as strings or with blanks.  Being numeric
        here fails earlier and more legibly than inside feature construction."""
        fake_yf(responses=fake_yahoo_response(n=10))
        result = F.fetch_ticker("aapl", days=1)
        for col in OHLCV:
            assert pd.api.types.is_numeric_dtype(result.frame[col]), col


class TestPartialFailure:
    def test_an_empty_chunk_is_reported_not_swallowed(self, fake_yf):
        """Yahoo's refusal is an *empty frame*, not an exception.  Untreated that is
        indistinguishable from "no data exists"."""
        fake_yf(responses=lambda s, e: (
            pd.DataFrame() if s.day % 2 else fake_yahoo_response(n=20)))

        result = F.fetch_ticker("aapl", days=14)
        assert result.chunks_ok > 0
        assert result.chunks_failed > 0
        assert result.errors, "an empty chunk must be recorded"
        assert result.ok, "partial data is still data, and is usable"

    def test_one_raising_chunk_does_not_discard_the_others(self, fake_yf):
        """Failing the whole fetch would throw away four good sessions to report one
        bad day."""
        fake_yf(fail_at=2)
        result = F.fetch_ticker("aapl", days=20)
        assert result.chunks_failed == 1
        assert result.ok
        assert "network flake" in " ".join(result.errors)

    def test_all_chunks_failing_is_not_ok(self, fake_yf):
        """``ok`` is the UI's gate.  An empty archive drawn as a chart looks like a
        broken app; this flag is what lets it say *why* instead."""
        fake_yf(responses=pd.DataFrame())
        result = F.fetch_ticker("aapl", days=7)
        assert not result.ok
        assert result.n_bars == 0
        assert result.sessions == 0
        assert "No bars returned" in result.summary()

    def test_a_bad_symbol_fails_cleanly(self, fake_yf):
        fake_yf(responses=pd.DataFrame())
        result = F.fetch_ticker("NOSUCHTICKERXYZ", days=1)
        assert not result.ok
        assert result.errors


class TestFalseDelistingNoise:
    """Yahoo logs ``possibly delisted`` for an *empty* window, not only a dead symbol.

    Because the span is chunked to 7 days, the last chunk of any window ending between
    one Friday and the next session's open spans a weekend.  Measured live on a
    Saturday for QQQ, the final chunk returns 0 rows while the overall fetch succeeds
    with 7,410 bars -- so this line is printed on essentially every successful fetch and
    asserts the opposite of what happened.  The chunk is recorded in ``errors``
    regardless, so the fix suppresses the log without losing information.
    """

    def test_the_empty_weekend_chunk_does_not_claim_delisting(self, fake_yf, caplog):
        # A fixed end so the chunk boundaries do not drift with the day the test runs.
        # 2026-10-04 is a Sunday, so the final chunk (from 09-27) covers a weekend.
        end = datetime(2026, 10, 4, tzinfo=timezone.utc)
        last_chunk_start = datetime(2026, 9, 27, tzinfo=timezone.utc)

        def responses(start, end_):
            return (pd.DataFrame() if start >= last_chunk_start
                    else fake_yahoo_response(n=20))

        fake_yf(responses=responses)
        with caplog.at_level(logging.INFO, logger="yfinance"):
            result = F.fetch_ticker("qqq", days=21, end=end)

        assert result.ok, "the fetch itself must still succeed"
        assert result.chunks_failed == 1, "only the weekend chunk is empty"
        assert result.errors, "the empty chunk is still reported to the reader"
        assert "possibly delisted" not in caplog.text

    def test_a_genuinely_empty_window_is_still_recorded_as_an_error(self, fake_yf):
        """Suppressing the log must not turn a real gap into silence."""
        fake_yf(responses=pd.DataFrame())
        result = F.fetch_ticker("qqq", days=7)
        assert not result.ok
        assert result.chunks_failed > 0
        assert any("no bars" in m.lower() for m in result.errors)

    def test_a_real_delisting_is_not_swallowed_by_the_filter(self, caplog):
        """The filter matches one message, not the logger wholesale.

        yfinance uses the same ``possibly delisted`` prefix for a symbol with no
        timezone, which *is* a real fault and must stay visible -- otherwise the fix
        would trade one misleading line for a genuinely hidden one.
        """
        logger = logging.getLogger("yfinance")
        with caplog.at_level(logging.INFO, logger="yfinance"):
            with F._quiet_false_delisting():
                logger.error("$QQQ: possibly delisted; no timezone found")
        assert "no timezone found" in caplog.text

    def test_the_rate_limit_warning_is_not_swallowed_by_the_filter(self, caplog):
        logger = logging.getLogger("yfinance")
        with caplog.at_level(logging.INFO, logger="yfinance"):
            with F._quiet_false_delisting():
                logger.error("Too Many Requests. Rate limited. Try after a while.")
        assert "Rate limited" in caplog.text

    def test_the_filter_does_not_outlive_the_request(self, fake_yf):
        """Installed per request, removed in ``finally``.

        A filter left attached would silence the message for the rest of the process --
        including for unrelated callers who would then get no signal at all.
        """
        logger = logging.getLogger("yfinance")
        before = list(logger.filters)
        fake_yf(responses=fake_yahoo_response(n=5))
        F.fetch_ticker("aapl", days=1)
        assert list(logger.filters) == before

    def test_the_filter_is_removed_even_when_the_request_raises(self, fake_yf):
        logger = logging.getLogger("yfinance")
        before = list(logger.filters)
        fake_yf(fail_at=1)
        F.fetch_ticker("aapl", days=20)
        assert list(logger.filters) == before, "an exception must not leave it attached"


class TestResult:
    def test_summary_names_the_instrument(self, fake_yf):
        fake_yf(responses=fake_yahoo_response(n=100))
        assert F.fetch_ticker("nvda", days=1).summary().startswith("NVDA")

    def test_sessions_are_counted_in_eastern_time(self):
        """US sessions straddle the UTC date boundary, so bucketing on UTC would
        report roughly twice the true session count."""
        idx = pd.to_datetime([
            "2026-09-30 13:30", "2026-09-30 14:00",   # one session, spans UTC midnight
            "2026-10-01 13:30",
        ], utc=True)
        frame = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 1.0,
                              "low": 1.0, "close": 1.0})
        assert F.FetchResult(symbol="X", frame=frame).sessions == 2

    def test_empty_result_properties_are_safe(self):
        empty = F.FetchResult(symbol="X")
        assert not empty.ok
        assert empty.first is None and empty.last is None
        assert empty.n_bars == 0 and empty.sessions == 0


# --------------------------------------------------------------------------- #
# Integration with the pipeline
# --------------------------------------------------------------------------- #
class TestFromFrame:
    def test_a_fetched_ticker_produces_a_ready_pipeline(self, fake_yf):
        """The point of the feature: the downloaded bars feed the real matcher."""
        from timeseries.pipeline import Pipeline

        fake_yf(responses=moving_tape(400))
        result = F.fetch_ticker("aapl", days=1)
        pipe = Pipeline.from_frame(result.frame, length=20)
        assert pipe.ready
        # Derived from the library, not hardcoded to the two channels the package
        # had when this was written.  What matters is that every declared feature has
        # a column, so a new one cannot be silently dropped by ``from_frame``.
        from timeseries.features import FEATURE_COLUMNS

        assert pipe.matrix.shape[1] == len(FEATURE_COLUMNS)
        assert pipe.n_bars > 100

    def test_from_frame_matches_from_csv_on_identical_bars(self, tmp_path, fake_yf):
        """A fetched ticker and a CSV must be prepared identically, or the two are not
        comparable.  Same input, two constructors, one result."""
        from timeseries.pipeline import Pipeline

        fake_yf(responses=moving_tape(400))
        result = F.fetch_ticker("aapl", days=1)

        path = tmp_path / "AAPL_1min_20260930_20260930.csv"
        result.frame.to_csv(path, index=False)

        from_frame = Pipeline.from_frame(result.frame, length=20)
        from_csv = Pipeline.from_csv(str(path), length=20)

        assert from_frame.n_bars == from_csv.n_bars
        assert from_frame.bars["timestamp"].equals(from_csv.bars["timestamp"])
        # Equal to float precision, not bit-for-bit: a CSV round-trip re-parses decimal
        # text, so the feature matrix can differ in the last bits.  What matters is that
        # no bar was dropped, added or reordered -- an earlier version compared exactly
        # and failed on a 2e-12 difference, which says nothing about correctness.
        assert np.allclose(from_frame.matrix, from_csv.matrix, rtol=1e-6, atol=1e-9)

    def test_short_fetch_is_reported_not_ready(self, fake_yf):
        """Too few bars must say so.  A pipeline reporting ``ready`` over a stub
        archive would produce percentiles against a handful of candidates."""
        from timeseries.pipeline import Pipeline

        fake_yf(responses=moving_tape(30))
        result = F.fetch_ticker("aapl", days=1)
        pipe = Pipeline.from_frame(result.frame, length=20)
        assert not pipe.ready
        assert pipe.warnings


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
class TestConstants:
    def test_chunking_stays_inside_yahoos_per_request_limit(self):
        """Yahoo's intraday ceiling is ~8 days.  Seven leaves margin, because
        exceeding it returns an empty chunk rather than an error."""
        assert F.CHUNK_DAYS <= 8

    def test_max_span_stays_inside_retention(self):
        """Yahoo retains ~30 days of intraday history; asking beyond it silently
        returns nothing."""
        assert F.MAX_1M_DAYS < 30
        assert F.MAX_1M_DAYS >= F.CHUNK_DAYS

    def test_default_span_is_everything_available(self):
        """The app has no span control, so the library default has to be the whole
        window.  Defaulting to one chunk would silently fetch a week while the UI
        implied everything was loaded.

        ``None`` is what now means "everything available", because the two
        timeframes have different ceilings: pinning the default to
        ``MAX_1M_DAYS`` and reusing it on daily would clip a 25-year history to 29
        days.  So the default is asserted to *be* ``None``, and each timeframe's own
        ceiling is asserted to be the one actually requested below."""
        import inspect

        default = inspect.signature(F.fetch_ticker).parameters["days"].default
        assert default is None

    def test_default_span_still_needs_several_requests(self, fake_yf):
        """The full window is only reachable *because* it is chunked: a single
        30-day request returns zero rows from Yahoo, while the same span assembled
        from 7-day requests comes back populated."""
        fake = fake_yf(responses=moving_tape(50))
        F.fetch_ticker("aapl")
        assert len(fake.calls) >= 4
        for call in fake.calls:
            span = (call["end"] - call["start"]).total_seconds() / 86400
            assert span <= F.CHUNK_DAYS

    def test_default_span_is_clamped_not_unbounded(self, fake_yf):
        """A caller passing an absurd span must still be capped at retention, or the
        loop would run for years."""
        fake = fake_yf(responses=moving_tape(10))
        F.fetch_ticker("aapl", days=10_000)
        total = sum((c["end"] - c["start"]).total_seconds() / 86400 for c in fake.calls)
        assert total <= F.MAX_1M_DAYS + 1


# --------------------------------------------------------------------------- #
# The dashboard wiring
# --------------------------------------------------------------------------- #
def _finds_div_with_height(source: str) -> bool:
    """True if ``source`` *calls* ``st.markdown`` with a fixed-height ``<div>``.

    Deliberately AST-based rather than a substring search.  Two reasons, both learned
    the hard way in this file:

    * The explanatory comment above the call quotes the removed markup verbatim, so a
      regex over the source matches the prose describing the bug and fails a correct
      file.  Comments and docstrings are not AST nodes, so they cannot appear here.
    * The check is about *what the code does*, and a div with an inline height is a
      layout hack by nature -- its height is one magic number tuned against one theme,
      one font size and one Streamlit version.

    Matches any height, not just the 1.7rem that caused this: the specific number is
    incidental, the nudge is the problem.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # a mid-``with`` slice is not a module; let the caller slice
        return False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "markdown"):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if re.search(r"<div[^>]*\bheight\s*:", arg.value):
                    return True
    return False


class TestAppWiring:
    """Static checks on `app.py`, which cannot be imported -- importing runs the UI.

    Reads the source and asserts the shape, via the shared `apphelpers` accessors,
    rather than refactoring a large script to make one line importable.
    """

    def test_tabs_are_unpacked_by_name_not_positionally(self):
        """Regression: ``TAB_ORDER`` had gained "Panel" while the positional unpack
        below it still listed six names, so ``main()`` raised ``ValueError: too many
        values to unpack`` on *every* run.  The app did not render at all -- this is
        pinned because the failure mode was total, and because the obvious fix
        (lengthening the tuple) breaks again the next time a tab is added."""
        src = app_source()
        assert 'dict(zip(TAB_ORDER, tabs))' in src, (
            "app.py must map tab names to containers by name; a positional unpack "
            "silently breaks whenever TAB_ORDER grows"
        )
        # The old shape must not come back.
        assert not re.search(r"^\s*tab_price, tab_chart, tab_matches.*= tabs\s*$",
                             src, re.M), "positional tab unpack reintroduced"

    def test_every_tab_in_tab_order_is_rendered(self):
        """A tab in ``TAB_ORDER`` with no ``with`` block renders as an empty panel
        that looks like "nothing found" rather than "not implemented"."""
        src = app_source()
        names = re.findall(
            r'"([^"]+)"', app_block(r"TAB_ORDER\s*=\s*\((.*?)\)", "TAB_ORDER")
        )
        for name in names:
            key = 'tab_by_name["%s"]' % name
            assert key in src, "TAB_ORDER lists %r but nothing renders it" % name

    def test_main_resolves_both_tickers_before_drawing_anything(self):
        """Both symbols come from :func:`resolve_ticker`, above the tab bar.

        The ticker inputs are drawn *above* the tabs but ``main()`` needs both symbols
        to build the two pipelines, which is before any widget exists.  So the
        resolution happens first, from ``session_state``, and only the drawing happens
        later.  A version that resolved the symbols inside the tab bodies would build
        its pipelines from the *previous* ticker's bars -- a whole render out of date,
        silently.

        Price first, because it gates the page: a Forecast that fails to resolve must
        not stop the tape from being chartable.
        """
        body = app_text(r"def main\(.*?\n(?=\ndef |\Z)", "main()")
        assert 'price_scope = resolve_ticker("price")' in body, (
            "main() must resolve the Price scope itself; reading it from the sidebar "
            "reintroduces a second source of truth"
        )
        assert 'forecast_scope = resolve_ticker("forecast")' in body
        # Each scope resolves to a *pair* now, so the symbol and the resolution both
        # come from the one call.  Asserting the pair is unpacked is what stops a
        # resolution being threaded separately and disagreeing with its symbol.
        assert "symbol = price_scope.symbol if price_scope is not None else None" in body
        # **One resolution for the session, read from the session gate** rather than
        # off either scope.  Reading it off a scope is what allowed the two tabs to
        # hold different resolutions, and a resolution that can differ from the one the
        # page title names is the bug this arrangement exists to remove.
        assert "tf = _tf()" in body
        assert "forecast_tf = tf" in body
        assert "price_scope.tf" not in body and "forecast_scope.tf" not in body
        # Both must precede the pipelines that consume them.  The cache key is
        # ``"<SYMBOL>@<TF>"`` rather than the bare symbol, so the ordering is pinned
        # against the key rather than against the symbol -- a bare symbol here would
        # let a daily pipeline be served under the entry built from 1-minute bars.
        pipe_at = body.index('"{}@{}".format(symbol, tf.key)')
        assert body.index('resolve_ticker("price")') < pipe_at, (
            "the Price ticker must be resolved before its pipeline is built"
        )
        fcst_pipe_at = body.index('"{}@{}".format(forecast_symbol, forecast_tf.key)')
        assert body.index('resolve_ticker("forecast")') < fcst_pipe_at, (
            "the Forecast ticker must be resolved before its pipeline is built"
        )
        # The resolution must reach the pipeline too, or the bars are downloaded at one
        # resolution and matched as though they were another.
        assert body.index("ACTIVE_TIMEFRAME[0] = chosen.key") < pipe_at, (
            "the active resolution must be published before any pipeline is built"
        )
        assert (body.index('resolve_ticker("price")')
                < body.index('resolve_ticker("forecast")') < pipe_at), (
            "Price is resolved first because it gates the page; a Forecast problem "
            "must not stop the tape being charted"
        )

    def test_nothing_renders_until_a_ticker_is_loaded(self):
        """With no bars there is nothing to say about a pattern.

        A bare **Price** tab is still drawn, because that is where the ticker input
        lives now -- returning early would leave the reader with an error and no
        control to fix it, which is a dead end.  The recovery page carries one tab, not
        six: with no archive, every other tab would render as "this archive has no
        such pattern", which is a claim about data that does not exist.
        """
        main_body = app_text(r"def main\(.*?\n(?=\ndef |\Z)", "main()")
        assert 'price_scope = resolve_ticker("price")' in main_body
        # Both resolves happen *above* the guard -- a scope seeded after the tab that
        # draws its box would keep the wrong label forever -- so the recovery block is
        # sliced from the guard to the title, not to the second resolve.
        assert (main_body.index('resolve_ticker("price")')
                < main_body.index('resolve_ticker("forecast")')
                < main_body.index("if not symbol:")), (
            "both scopes must be resolved before the guard draws the recovery tab; "
            "seeded afterwards, its input is permanently mislabelled"
        )
        guard_at = main_body.index("if not symbol:")
        block = main_body[guard_at:main_body.index("title_col, button_col")]
        assert re.search(r'render_scope_ticker\(\s*"price"', block), (
            "the recovery page must still draw the Price input; an error with no "
            "control to fix it is a dead end"
        )
        assert "return" in block, "main() must return before drawing the full tab bar"
        # One tab, not the full set.
        assert 'st.tabs(["Price"])' in block, (
            "the recovery page must show only the Price tab; the others would all "
            "report 'no such pattern' about an archive that does not exist"
        )
        # And the sidebar must no longer be what gates the page -- it carries the
        # scoring dial only.
        sidebar = app_text(r"def render_sidebar\(.*?\n(?=\ndef )", "render_sidebar")
        assert "resolve_ticker" not in sidebar, (
            "the sidebar must not fetch or resolve tickers; each tab draws its own "
            "input, and a second resolution path is a second source of truth"
        )
        assert "amplitude_weight" in sidebar, (
            "the sidebar should still carry the shared scoring dial"
        )

    def test_each_tab_draws_its_own_ticker_input(self):
        """Price and Forecast each carry their own input, inside themselves.

        A helper nobody renders is dead code that reads as a working feature, so both
        call sites are asserted.  And they must be in the *tab bodies*: ``st.tabs``
        renders every body on every rerun but shows one at a time, which is exactly why
        a tab-local input is the thing a reader looking at that tab's charts will find,
        and why an input parked outside the tab would govern a chart it is not above.

        The two scopes must not collide, or the second draw raises
        ``StreamlitDuplicateElementId`` -- ``st.tabs`` renders both bodies every pass.
        """
        src = app_source()

        def tab_body(name):
            """One tab's body, as text, delimited by the *next* ``with tab_`` line.

            Sliced between tab markers rather than parsed, so the boundaries have to be
            statement starts -- a naive slice between two ``with tab_`` substrings can
            begin mid-block and will not parse.  Each marker is therefore located as a
            whole line, which is where ``main()`` writes them, and the body runs to the
            next one.
            """
            lines = src.split("\n")
            starts = [i for i, ln in enumerate(lines)
                      if ln.startswith("    with tab_%s:" % name)]
            assert len(starts) == 1, (
                "expected exactly one `with tab_%s:` line, found %d -- main() renders "
                "each tab body once" % (name, len(starts))
            )
            begin = starts[0]
            later = [i for i, ln in enumerate(lines)
                     if ln.startswith("    with tab_") and i > begin]
            end = later[0] if later else len(lines)
            # Dedented: a slice taken from the middle of ``main()`` keeps ``main()``'s
            # own four-space indent, and a fragment at column 4 does not parse on its
            # own.  Common leading whitespace is stripped so the body can be handed
            # straight to ``ast.parse``.
            body = lines[begin:end]
            pad = min((len(ln) - len(ln.lstrip()) for ln in body if ln.strip()),
                      default=0)
            return "\n".join(ln[pad:] if ln.strip() else "" for ln in body)

        price_body = tab_body("price")
        # **The Forecast input lives on the *Forecast* tab**, which is the tab that
        # does the work -- it carries the brush, the search settings and the evidence
        # table.  ``tab_body`` is called with the *variable* name rather than the tab
        # label, because ``with tab_forecast:`` is what ``main()`` writes.
        forecast_body = tab_body("forecast")
        # Kept alongside so the split cannot quietly put the input back on Projection.
        reference_body = tab_body("projection")

        # Matched on the scope argument alone, not on the exact indentation of the
        # call: a whitespace-exact needle here fails on a reformat rather than on a
        # behaviour change, which is how a wiring test starts being ignored.
        def draws(body, scope):
            return re.search(r'render_scope_ticker\(\s*"%s"' % scope, body) is not None

        assert draws(price_body, "price"), (
            "the Price tab must draw its own input; without it the tape has no way to "
            "change instrument"
        )
        assert draws(forecast_body, "forecast"), (
            "the *Forecast* tab must draw the Forecast input; without it the tab "
            "can only ever follow the Price ticker"
        )
        # **The Projection tab must draw neither input.**  It is a fixed reference and
        # takes no reader input.  Asserting this is what stops the input drifting back
        # onto it -- which would register ``ticker_input_forecast`` twice per pass and
        # raise ``StreamlitDuplicateElementKey``, taking down the whole page rather
        # than one tab.  No source-level test of the stubs could have caught that one
        # (see ``TestTheRealAppRuns``); what this catches is the *drift*, early.
        assert not draws(reference_body, "forecast"), (
            "the Projection tab is a fixed reference and must draw no input; drawing it "
            "here as well would register ticker_input_forecast twice in one pass"
        )
        assert not draws(reference_body, "price"), (
            "the Price input must not live on the Projection tab either"
        )
        # And the *right* scope in the *right* body -- a Forecast input in the Price
        # tab would be the worst version of this bug: it works, and it is on the wrong
        # screen.
        assert not draws(price_body, "forecast"), (
            "the Forecast input must not live in the Price tab; st.tabs shows one "
            "body at a time, so it would be invisible exactly when it is needed"
        )
        assert not draws(forecast_body, "price"), (
            "the Price input must not live on the Forecast tab, for the same reason"
        )

        # **Exactly one tab body may draw the forecast input.**  ``st.tabs`` renders
        # every body on every pass, so two draws of the same scope in one pass is the
        # duplicate-key crash regardless of whether either one is individually right.
        drawers = [name for name, body in
                   (("price", price_body), ("projection", reference_body),
                    ("forecast", forecast_body))
                   if draws(body, "forecast")]
        assert drawers == ["forecast"], (
            "exactly one tab body may draw the forecast input, found %r" % (drawers,)
        )

        # **The Forecast input must come before the availability gate**, on the live
        # path as well as in the not-ready branch.  Everything below it is conditional
        # on ``forecast_pipe``, so an input drawn after the gate is invisible to the
        # one reader who needs it -- the one whose forecast ticker failed to load, and
        # who is being told to go and load one.  Verified by mutation: moving the call
        # inside the ``else`` leaves every other assertion in this file green.
        gate_at = forecast_body.index("if forecast_pipe is None:")
        assert forecast_body.index("render_scope_ticker(") < gate_at, (
            "the Forecast input must be drawn before the `forecast_pipe is None` gate; "
            "after it, a reader with no forecast ticker is told to load one with no "
            "box on screen to do it"
        )
        # The input must precede the charts it governs, or it is below the fold.
        #
        # **Compared by line number from the AST, not by substring.**  The comment
        # block above each call explains exactly this ordering and therefore names both
        # the input and the guide -- so `body.index('guide("Price")')` finds the prose
        # first and the assertion compares a call against its own explanation.  This is
        # the exact hazard `apphelpers.app_called_names` exists to avoid, and the reason
        # an ordering assertion on commented source has to be positional.
        #
        # Scoped to *each tab's own body*, not to the file.  ``guide`` is also called
        # from ``render_matches_tab``, which runs long before the tab bar, so a
        # whole-file "first call" comparison measures a helper against the tabs and
        # fails on correct code.
        def calls_in(body_text):
            """Every called *name* in source order, for one slice of the file.

            ``ast.walk`` is breadth-first and gives **no ordering guarantee**, so its
            output cannot answer "which came first" -- it happens to look right often
            enough that a test written on it passes until the tree shape changes.  The
            nodes are sorted by position here instead, which is the property the
            assertion is actually about.  (The slice is already delimited by the tab
            markers, so the file's own offset cancels out.)
            """
            nodes = [n for n in ast.walk(ast.parse(body_text))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            return [n.func.id for n in sorted(nodes, key=lambda n: n.lineno)]

        for body, scope in ((price_body, "Price"), (forecast_body, "Forecast")):
            names = calls_in(body)
            assert "render_scope_ticker" in names, (
                "%s tab body has no ticker input" % scope
            )
            assert names.index("render_scope_ticker") < names.index("guide"), (
                "the %s input must be drawn before its guide; the guide is the long "
                "collapsed how-to, and an input below it is below the fold" % scope
            )

    def test_the_ticker_input_is_drawn_even_when_the_tab_is_disabled(self):
        """A tab that says "not enough data" must still offer a way to try another name.

        Both messages in the not-ready branch describe the *archive*, and switching
        ticker is the remedy they should be pointing at.  With the input gated behind
        the same check it protects, the tab says "use a liquid symbol" and provides no
        way to do it -- and the reader has to know that a box they cannot see exists.

        Asserted per branch because each covers a different archive: Price's own is too
        thin, and Forecast's may be fine while Price's is not.

        **The forecast input is on *Forecast*, not on Projection**, so the two stubs
        are sliced separately and each is checked against what it *should* hold.  The
        previous version sliced ``Forecast`` → ``Backtest`` and found the input there;
        after the split that span holds the reference stub and the interactive one, so
        the same slice would have kept passing for the wrong reason -- it would find the
        input in *Forecast* while claiming Projection had it.  Slicing each stub to
        the next marker is what makes the assertion mean what it says.
        """
        src = app_source()

        # **Every marker that actually exists**, found from the source rather than
        # listed here.  A hard-coded list is wrong the moment a tab is added without a
        # not-ready stub -- ``src.index`` would raise ``ValueError`` on a missing name
        # and report it as a wiring failure, which reads as "the tab bar is broken"
        # when the tab bar is fine.  Deriving them also means a new stub is covered
        # without touching this test.
        #
        # ``re.M`` with ``^[ \t]*`` rather than ``^``: these bodies are indented inside
        # ``main()``, so a bare ``^`` would anchor to column 0 and find nothing --
        # which surfaces as an empty ``markers`` list and every stub reported missing.
        markers = sorted((m.start(), m.group(1)) for m in re.finditer(
            r'^[ \t]*with tabs\[TAB_ORDER\.index\("([^"]+)"\)\]:', src, re.M))

        def stub(name):
            """The not-ready branch's body for one tab, up to the next tab marker."""
            begin = dict((n, at) for at, n in markers).get(name)
            assert begin is not None, "no not-ready stub for %r" % name
            later = [at for at, n in markers if at > begin]
            return src[begin:later[0]] if later else src[begin:]

        assert "render_scope_ticker(" in stub("Price"), (
            "the not-ready Price tab must still draw its input; switching ticker is "
            "the remedy its own message recommends"
        )
        # The Projection stub is a fixed reference: an input here would double-register
        # the forecast key on this branch, which is the crash this test's own docstring
        # describes.
        assert "render_scope_ticker(" not in stub("Projection"), (
            "the not-ready Projection tab is a fixed reference and must draw no input; "
            "the key belongs to Forecast alone"
        )
        fcst = stub("Forecast")
        assert re.search(r'render_scope_ticker\(\s*"forecast",', fcst), (
            "the not-ready Forecast tab must still draw the Forecast input"
        )
        # ...and it must come *before* the gate, or the gate protects the very control
        # that would fix what it is complaining about.
        assert fcst.index("render_scope_ticker(") < fcst.index("if forecast_pipe is None:"), (
            "the Forecast input must be drawn before the availability gate; after it, "
            "a reader with no forecast ticker has no way to load one"
        )
        # Both stubs must be gated on the Forecast pipeline rather than the Price one.
        # This branch is entered *because* Price is too thin, which says nothing about
        # Forecast's own archive -- reading ``pipe.ready`` here would switch off a tab
        # whose bars are perfectly good.
        for name in ("Projection", "Forecast"):
            body = stub(name)
            assert "forecast_pipe is None" in body, (
                "the not-ready %s stub must gate on the Forecast pipeline" % name
            )
            assert "not pipe.ready" not in body, (
                "the not-ready %s stub must not gate on the Price pipeline" % name
            )

    def test_the_fetch_button_is_not_nudged_with_a_magic_spacer(self):
        """No hard-coded ``div`` above the button, beside a collapsed-label input.

        **This is a bug this assertion was written after hitting.**  The 1.7rem spacer
        predates ``label_visibility="collapsed"``: it existed to clear the text box's
        *visible label*, which renders above the field, so a button in the next column
        would otherwise start level with the label rather than the input.  Collapsing
        the label removed the thing being compensated for and left the nudge as a 25px
        gap between two controls that were already aligned.

        Measured in Chromium against this app at viewport 1600px, ``top`` of each box:

            variant                         input   button   delta
            spacer + collapsed label         381      406    +25px
            spacer removed                   381      378     -3px

        The residual 3px is Streamlit's own base metrics -- a button is 40px tall and a
        text input 36px -- and is deliberately not asserted on here.  Neither of the
        two obvious "fixes" for it works, both verified rather than assumed: there is no
        button-height theme option in ``streamlit.config``, and a ``<style>`` rule in
        ``HELP_CSS`` is stripped by Streamlit.  See ``render_ticker_input``.

        The failure mode this guards is the general one.  A magic-number spacer encodes
        one exact configuration and fails **silently** on every other: it keeps its
        height while whatever it was compensating for changes underneath it.  So the
        assertion is that the two cannot coexist -- a spacer here is only ever correct
        while some *other* thing about the layout happens to cancel it.
        """
        src = app_source()
        body = app_text(r"def render_ticker_input\(.*?\n(?=\ndef )",
                        "render_ticker_input")
        assert "label_visibility=\"collapsed\"" in body, (
            "the ticker input's label is collapsed, which is why no spacer is needed "
            "above the button"
        )
        # Matched on the **call**, not on the string.  This test's own docstring and the
        # comment above the call both quote the old markup verbatim to explain what
        # regressed -- so a substring search over the file matches that prose and fails
        # a correct app.  This is the hazard `apphelpers.app_called_names` exists to
        # avoid, and the reason the check goes through the AST.
        assert not _finds_div_with_height(body), (
            "the spacer div is back; it cleared a label that is no longer rendered and "
            "now pushes the Fetch button below the input"
        )
        # The spacer and the collapsed label must not both be in play: one compensates
        # for the other, so together they double-count.
        assert "label_visibility" not in body or not _finds_div_with_height(body), (
            "a collapsed label and a label-clearing spacer cannot both be in play"
        )

    def test_a_click_is_parked_and_consumed_on_the_next_pass(self):
        """The button cannot be acted on in the pass that receives the click.

        The inputs are drawn *after* the pipelines are built, so a click has to survive
        to the following run.  It cannot simply be cleared by assigning to the button's
        own key: that key is a widget, and Streamlit raises
        ``StreamlitValueAssignmentNotAllowedError`` for any write to one -- verified
        against the installed version, not assumed.

        So the click parks in a **derived non-widget** key and ``resolve_ticker`` pops
        it.  Three properties, each a distinct bug:

        * the pop is what makes the fetch happen **once** rather than on every rerun;
        * the key must be derived from the ticker key, or the two halves of one fetch
          drift apart and the fetch either never fires or fires forever;
        * the flag must be **non-widget**, or the write raises and the button does
          nothing at all -- which looks exactly like a dead button.
        """
        src = app_source()
        assert "PENDING_SUFFIX" in src, "the pending-fetch flag must be a named constant"
        decl = re.search(r'PENDING_SUFFIX = "([^"]+)"', src)
        assert decl, "PENDING_SUFFIX must be declared"
        # Derived, not hand-written per scope.
        assert (src.count("PENDING_SUFFIX") >= 3), (
            "the flag must be derived from each scope's ticker key in both the setter "
            "and the consumer; a hand-written pair drifts and then fires forever"
        )
        resolve = app_text(r"def resolve_ticker\(.*?\n(?=\ndef )", "resolve_ticker")
        assert "st.session_state.pop(pending_key, None)" in resolve, (
            "the pending flag must be POPPED, so the download happens exactly once"
        )
        button = app_text(r"def render_ticker_input\(.*?\n(?=\ndef )",
                          "render_ticker_input")
        assert "st.session_state[pending_key] = True" in button, (
            "the click must be recorded in the non-widget pending key"
        )
        # The illegal write, specifically: assigning to the button's own key.
        assert re.search(r"st\.session_state\[fetch_key\]\s*=", button) is None, (
            "the Fetch button's own key is a widget and is read-only; writing it "
            "raises StreamlitValueAssignmentNotAllowedError"
        )
        assert re.search(r"st\.session_state\[input_key\]\s*=", button) is None, (
            "the text box's own key is a widget and is read-only"
        )

    def test_the_two_scopes_use_disjoint_session_keys(self):
        """Neither scope may touch the other's state.

        ``st.tabs`` renders every tab body on **every** rerun, so two un-keyed widgets
        of the same name would raise ``StreamlitDuplicateElementId`` rather than
        merely collide -- and a shared ticker key would mean the last panel rendered
        silently overwrote the first one's instrument, which is the bug this split is
        meant to prevent.
        """
        src = app_source()
        declared = dict(re.findall(
            r'^(PRICE_TICKER_KEY|PRICE_TICKER_INPUT_KEY|PRICE_TICKER_FETCH_KEY'
            r'|FORECAST_TICKER_KEY|FORECAST_TICKER_INPUT_KEY|FORECAST_TICKER_FETCH_KEY)'
            r'\s*=\s*"([^"]+)"', src, re.M))
        assert len(declared) == 6, (
            "all six per-scope keys must be declared; found %d: %r"
            % (len(declared), declared)
        )
        assert len(set(declared.values())) == 6, (
            "the two ticker scopes share a session key, so one tab can overwrite "
            "the other's instrument: %r" % declared
        )

    def test_the_ticker_key_is_not_a_widget_key(self):
        """The symbol in force must live under a **non-widget** key.

        ``reset_query_state`` sets ``session_state["_"] = {}``, which discards *widget*
        state wholesale -- and that wipe is exactly what re-seeds the text box.  If
        the ticker lived in the widget key, the reset would erase it and the box
        would re-render with the module default while every chart showed a different
        instrument.  The non-widget key survives the wipe and is what the box reads
        back.

        So the three keys per scope are not redundant: one is durable state, two are
        widgets.
        """
        src = app_source()
        widget_keys = dict(re.findall(
            r'^(PRICE_TICKER_INPUT_KEY|PRICE_TICKER_FETCH_KEY'
            r'|FORECAST_TICKER_INPUT_KEY|FORECAST_TICKER_FETCH_KEY)'
            r'\s*=\s*"([^"]+)"', src, re.M))
        ticker_keys = dict(re.findall(
            r'^(PRICE_TICKER_KEY|FORECAST_TICKER_KEY)\s*=\s*"([^"]+)"', src, re.M))
        assert len(widget_keys) == 4 and len(ticker_keys) == 2, (
            "expected two durable ticker keys and four widget keys, got %r / %r"
            % (ticker_keys, widget_keys)
        )
        # The box and the button of one scope must not collide with each other either.
        assert len(set(widget_keys.values())) == 4, (
            "a scope reuses one key for its text box and its Fetch button: %r"
            % widget_keys
        )

    def test_forecast_seeds_on_price_only_while_it_has_none(self):
        """The seed is a one-way latch, not a running subscription.

        Two properties, and the second is the one that matters:

        * **It seeds**, so a reader who never touches the Forecast panel still gets a
          working tab rather than an empty one -- and it seeds on the Price symbol, so
          the fetch is a cache hit rather than a second download of the same bars.
        * **It does not re-seed**, because ``seed_scope`` is consulted only when the
          scope's own ticker key is *absent*.  A version that re-derived Forecast's
          ticker from Price on every render would re-couple the two tabs the moment
          Price was re-fetched, which is the coupling this change exists to remove.
        """
        src = app_source()
        body = app_block(r"def resolve_ticker\(.*?\n(?=\ndef )", "resolve_ticker")
        assert "seed_scope" in body, (
            "resolve_ticker must resolve the seed from the scope's declared seed_scope"
        )
        # The write is inside the "if this scope has no ticker" guard, so the latch is
        # real.  Asserting the guard is the part that cannot be satisfied by a
        # re-derivation elsewhere in the function.
        assert re.search(
            r"if ticker_key not in st\.session_state:.*?st\.session_state\[ticker_key\]",
            body, re.S,
        ), (
            "the seed must be applied only when this scope has no ticker of its own; "
            "an unconditional assignment re-couples Forecast to Price"
        )
        # And Price must be the one that seeds, not the other way round.
        scopes = re.search(r"FETCH_SCOPES: Dict\[str, Dict\[str, str\]\] = \{(.*?)\n\}",
                           src, re.S)
        assert scopes, "FETCH_SCOPES not found"
        body_s = scopes.group(1)
        price_block = body_s.split('"forecast":')[0]
        assert '"seed_scope": ""' in price_block, (
            "the Price scope must not seed on anything -- it is the seed source"
        )
        assert '"seed_scope": "price"' in body_s.split('"forecast":')[1], (
            "the Forecast scope must seed on the Price scope"
        )

    def test_the_seed_is_resolved_before_the_text_box_is_drawn(self):
        """The box must show the symbol the tab is actually built on.

        **This is a real bug this assertion was written after finding, not a
        hypothetical.**  A keyed widget's ``value=`` is ignored once its key has been
        registered, so seeding the ticker key *after* the ``text_input`` call locks the
        box to whatever the fallback was on the very first paint -- and it never
        converges, because every later rerun finds the key already present and skips
        the seed.  Measured with Price on AAPL and Forecast unseeded:

            paint 1 -> box='BTC-USD'  in force='AAPL'
            paint 2 -> box='BTC-USD'  in force='AAPL'   <- still wrong

        The Forecast tab would chart AAPL behind a box reading BTC-USD, indefinitely.
        The latch itself is correct; only its *position* is wrong, which is why nothing
        else in this file can see it.

        Asserted across the two functions, because the ordering that matters is now
        *between* them: the seed lives in ``resolve_ticker`` and the widget in
        ``render_ticker_input``, and ``main()`` must call them in that order.  Checking
        only within one function would pass against the broken version, where the seed
        sat in the same function but below the widget.
        """
        main_body = app_text(r"def main\(.*?\n(?=\ndef |\Z)", "main()")
        # The seed is resolved for BOTH scopes at the top of ``main()``, above the tab
        # bar.  That is the whole ordering requirement: the seed has to be written
        # before any widget reads it as ``value=``, and every widget now lives *inside*
        # a tab, which is drawn after both.
        price_at = main_body.index('resolve_ticker("price")')
        fcst_at = main_body.index('resolve_ticker("forecast")')
        tabs_at = main_body.index("st.tabs(")
        assert price_at < fcst_at < tabs_at, (
            "both scopes must be seeded before the tab bar draws any widget; seeded "
            "afterwards, the widget keeps its fallback value forever and the box "
            "disagrees with the archive behind it"
        )
        # And within the input helper, the box must read that key directly rather than
        # through a fallback, so a missing key cannot silently render the module
        # default -- which is what produced a permanent BTC-USD label over an AAPL
        # chart.
        draw = app_text(r"def render_ticker_input\(.*?\n(?=\ndef )",
                        "render_ticker_input")
        assert "value=st.session_state[ticker_key]" in draw, (
            "the box must be seeded from this scope's ticker key directly; a .get() "
            "fallback here is what let the stale BTC-USD value through"
        )

    def test_a_widget_reset_clears_nothing_here(self):
        """``session_state["_"] = {}`` is inert, and the code must not lean on it.

        The reset inherited a blanket ``st.session_state["_"] = {}`` from a version of
        this app that had From/To date pickers, where it genuinely re-seeded them.  It
        is kept for continuity but does nothing now, which was verified rather than
        assumed -- this Streamlit has no special handling for the key at all:

            slid=7 -> slid=3 -> (wipe) -> slid=3

        So the actual state reset is the per-scope key list.  A future reader who
        *believed* the wipe were doing the work would add a key to the list, see no
        behaviour change, and remove it again -- which is how state that needs clearing
        silently stops being cleared.

        Asserted on the presence of the explicit loop, since that is the part carrying
        the behaviour: nothing else in the function discards anything.
        """
        body = app_block(r"def reset_query_state\(.*?\n(?=\ndef )",
                         "reset_query_state")
        assert "for key in stale:" in body and "st.session_state.pop(state_key(key), None)" in body, (
            "the per-scope stale-key list must actually be popped; if the blanket "
            "session_state wipe is being relied on, it is relying on a no-op"
        )
        # **The pop must go through ``state_key``**, because that is where the brush
        # and applied-span are read and written.  The stale lists name the *base*
        # keys, so popping them raw evicts nothing: the brush lives at
        # ``sel_price_view_@tf1d`` and a raw ``sel_price_view`` pop finds nothing to
        # remove, so the previous ticker's brush silently survives onto the new
        # ticker's chart.  Nothing raises; the chart is just wrong.
        assert "st.session_state.pop(key, None)" not in body, (
            "the reset pops the un-namespaced key, so the brush and applied-span "
            "are never actually cleared on a ticker switch"
        )
        # The list must be selected by scope rather than hard-coded, or the two tabs
        # would clear each other's state.
        assert 'stale = PRICE_STALE_KEYS if scope == "price" else FORECAST_STALE_KEYS' \
            in body, (
            "the reset must pick its stale-key list by scope; a single hard-coded list "
            "re-couples the two tabs on every ticker switch"
        )

    def test_switching_ticker_clears_query_state(self):
        """Bar indices and cached matches describe the *previous* archive.  Carrying
        them across silently re-runs a query the user never asked for."""
        src = app_source()
        reset = re.search(r"def reset_query_state\(.*?\n(?=\ndef )", src, re.S)
        assert reset, "reset_query_state not found -- a ticker switch must clear state"
        body = reset.group(0)
        # The key list moved to module constants, so the body iterates a tuple.  Both
        # keys must still be reachable from it.
        for name in ("PRICE_STALE_KEYS", "FORECAST_STALE_KEYS"):
            assert name in body, (
                "%s must be selectable by a switch -- one scope must not clear the "
                "other's state" % name
            )
        for stale in ("run_signature", "run_output"):
            assert stale in src, (
                "%s must be cleared on a Price ticker switch; it has left the "
                "per-scope key list entirely" % stale
            )
        assert '"run_signature", "run_output"' in src, (
            "the legacy keys belong to the Price scope, which owns the shared run"
        )

    def test_a_forecast_switch_leaves_the_price_tab_alone(self):
        """The independence invariant: the two tabs must not disturb each other.

        Asserted on the per-scope key lists rather than on the reset body, because the
        lists *are* the contract -- ``PRICE_STALE_KEYS`` and ``FORECAST_STALE_KEYS`` are
        disjoint, so a Forecast switch cannot evict the Price query or run, and a Price
        switch cannot evict the Forecast result.

        A shared list would reintroduce exactly the coupling the per-tab tickers exist
        to remove: change the Forecast ticker and the Price query would vanish.
        """
        src = app_source()
        price = re.search(r"PRICE_STALE_KEYS: Tuple\[str, \.\.\.\] = \((.*?)\)",
                          src, re.S)
        forecast = re.search(r"FORECAST_STALE_KEYS: Tuple\[str, \.\.\.\] = \((.*?)\)",
                             src, re.S)
        assert price and forecast, "both per-scope stale-key lists must be declared"

        price_keys = set(re.findall(r'"([^"]+)"', price.group(1))) | {
            n for n in re.findall(r"\b([A-Z_]+_KEY)\b", price.group(1))
        }
        forecast_keys = set(re.findall(r'"([^"]+)"', forecast.group(1))) | {
            n for n in re.findall(r"\b([A-Z_]+_KEY)\b", forecast.group(1))
        }
        overlap = {k for k in price_keys & forecast_keys
                   if not k.endswith("_TICKER_KEY")}
        assert not overlap, (
            "the two scopes clear overlapping state, so switching one ticker's tab "
            "discards the other's: %r" % sorted(overlap)
        )
        # And the ticker keys specifically must never be in either list -- a reset
        # that cleared its own scope's ticker would break the latch.
        for key in ("PRICE_TICKER_KEY", "FORECAST_TICKER_KEY"):
            assert key not in src.split(key + " =")[0], (
                "%s must not appear in a stale-key list: clearing it would un-seed "
                "the scope and let the two tabs re-couple" % key
            )

    def test_only_a_price_switch_rekeys_the_tab_bar(self):
        """A Forecast ticker change must not move the reader off the Forecast tab.

        The tab bar is stateless unless ``on_change`` is passed, so it is re-keyed by
        bumping a generation -- which is what makes ``default`` apply and lands the
        reader on Price.  That is right for a Price fetch and wrong for a Forecast
        one: a reader who switched the Forecast ticker while reading that tab would be
        yanked off it to look at a tape they did not ask about.

        So the bump is scoped to Price.  Unscoped, it silently defeats the whole
        point of giving the Forecast tab its own input.
        """
        reset = re.search(r"def reset_query_state\(.*?\n(?=\ndef )", app_source(),
                          re.S)
        assert reset, "reset_query_state not found"
        body = reset.group(0)
        guard = re.search(
            r'if scope == "price":\s*\n\s*st\.session_state\[TABS_GENERATION_KEY\]',
            body)
        assert guard, (
            "the tab-bar re-key must be guarded on the Price scope; an unscoped bump "
            "yanks the reader off the Forecast tab whenever its ticker changes"
        )
        # The guard must actually wrap the bump, not merely sit near it.
        assert body.index('if scope == "price":') < body.index(
            "st.session_state[TABS_GENERATION_KEY]"), (
            "the generation bump must be inside the Price-only guard"
        )

    def test_the_forecast_path_is_told_which_ticker_it_is_charting(self):
        """Forecast charts must name the Forecast ticker, not the page's.

        ``SYMBOL_FOR_HELP`` is the **Price** instrument -- it backs the page title, the
        manual and the Backtest fee copy.  A Forecast renderer that read it would
        caption an AAPL forecast as QQQ, and would also pass the wrong
        ``forecast_path_for`` cache key, so one ticker's path could be served under
        another's chart.  Both are silent: the numbers stay plausible.
        """
        src = app_source()
        for fn in ("_render_forecast_path", "_render_panel_forecast_path",
                   "_forecast_brush_chart", "_render_window_forecast"):
            body = re.search(r"def %s\(.*?\n(?=\ndef |\n# )" % fn, src, re.S)
            assert body, "%s not found" % fn
            assert "SYMBOL_FOR_HELP" not in body.group(0).split('"""')[2], (
                "%s must take its symbol as an argument; reading the Price-side "
                "SYMBOL_FOR_HELP captions this tab's charts with the wrong ticker" % fn
            )
        # And the symbol must actually reach both cached helpers' cache keys.
        # **Split across the two Forecast tab renderers**: the reference chart belongs
        # to ``render_forecast_tab`` and the brush to ``render_window_tab``.
        tab = re.search(r"def render_forecast_tab\(.*?\n(?=\ndef |\n# )", src, re.S)
        assert tab, "render_forecast_tab not found"
        assert "_render_forecast_path(pipe, symbol," in tab.group(0), (
            "the fixed chart must be handed this tab's symbol"
        )
        window_tab = re.search(r"def render_window_tab\(.*?\n(?=\ndef |\n# )", src, re.S)
        assert window_tab, "render_window_tab not found"
        assert "_forecast_brush_chart(pipe, symbol," in window_tab.group(0), (
            "the brush chart must be handed this tab's symbol"
        )

    def test_the_forecast_tab_is_gated_on_its_own_pipeline(self):
        """A broken or thin Forecast ticker must not take the Price tabs down.

        Three distinct gates, and each is a real failure the others do not catch:

        * ``forecast_pipe is None`` -- the Forecast archive failed to load.  The tab
          says so and every other tab renders normally.
        * ``not forecast_pipe.ready`` -- too few bars to form a window.  Reported
          against **the Forecast ticker**, since a thin Forecast archive says nothing
          about the Price one.
        * the normal path -- hands over ``forecast_pipe``, never ``pipe``.

        Reading ``pipe.ready`` anywhere on this path would switch off a tab whose
        archive is perfectly good, telling a reader with a usable AAPL forecast ticker
        that forecasting is disabled.

        The second gate is the one this change had to add.  That branch is entered
        because the *Price* archive is too thin, which says nothing about Forecast's --
        so the gate has to read Forecast's own readiness or the tab is disabled for a
        reason that does not apply to it.
        """
        main_body = app_text(r"def main\(.*?\n(?=\ndef |\Z)", "main()")
        forecast_block = app_block(r"with tab_projection:(.*?)\n    with tab_",
                                   "Projection tab body")
        assert "forecast_pipe is None" in forecast_block, (
            "a failed Forecast load must degrade to a message, not a traceback"
        )
        assert "render_forecast_tab(\n                    forecast_pipe, forecast_symbol," \
            in forecast_block, (
            "the Projection tab must render from its OWN pipeline and symbol; handing "
            "it the Price pair silently charts the wrong ticker"
        )
        # ...and inside its own resolution scope, or a daily Projection tab would be
        # drawn with the Price tab's intraday horizons and window lengths.
        assert "with timeframe_scope(forecast_tf.key):" in forecast_block, (
            "the Projection tab may hold a different resolution from Price; it must be "
            "rendered inside that scope or its horizons and captions describe the "
            "wrong archive"
        )
        assert "Could not build the pipeline for {}" in main_body, (
            "a Price pipeline failure must still report against the Price symbol"
        )
        assert "Could not build the Forecast pipeline for {}" in main_body, (
            "a Forecast failure must be reported separately and must not return"
        )
        # The not-ready branch, which is reached *because* Price is too thin.  Sliced on the
        # Forecast tab's own block rather than the whole branch, and forwards from the
        # ``with`` line -- the gate is *inside* that block, so slicing backwards to the
        # branch header would stop before it and pass vacuously.
        src_all = app_source()
        block_at = src_all.index('with tabs[TAB_ORDER.index("Projection")]:')
        block_end = src_all.index('with tabs[TAB_ORDER.index("Backtest")]:', block_at)
        forecast_gate = src_all[block_at:block_end]
        assert "elif not forecast_pipe.ready:" in forecast_gate, (
            "the not-ready branch must gate the Projection tab on the Forecast "
            "pipeline's readiness; inheriting pipe.ready disables a tab whose own "
            "archive is fine"
        )

    def test_a_price_fetch_cannot_reach_into_the_forecast_tab(self):
        """The independence invariant, in the direction that is easiest to break.

        A reader sets Forecast to AAPL, brushes a window and runs a search, then goes
        back and re-fetches Price.  Every one of those Forecast artefacts must survive,
        because each of them is a claim about *AAPL's* bars and the Price fetch says
        nothing about them.

        Verified against the live app before it was written down.  Price on BTC-USD,
        Forecast switched to AAPL, a brushed window plus a completed run, then Price
        re-fetched on MSFT:

            step 1  price=BTC-USD  forecast=AAPL   applied span (1000, 1240)
            step 2  price=MSFT    forecast=AAPL   applied span (1000, 1240)

        The two things that could each break that are pinned separately: the per-scope
        key lists (``test_a_forecast_switch_leaves_the_price_tab_alone``) and the
        tab-bar guard (``test_only_a_price_switch_rekeys_the_tab_bar``).  This asserts
        the wiring that ties them together -- that a Price reset reaches only the Price
        list.
        """
        body = app_block(r"def reset_query_state\(.*?\n(?=\ndef )",
                         "reset_query_state")
        # The stale list is chosen by scope, so a Price switch can only ever pop Price
        # keys.  Asserted as an exact expression: a near-miss such as a reversed
        # ternary passes every other assertion in this file while swapping the tabs.
        assert ('ticker_key = PRICE_TICKER_KEY if scope == "price" '
                'else FORECAST_TICKER_KEY') in body
        assert ('stale = PRICE_STALE_KEYS if scope == "price" '
                'else FORECAST_STALE_KEYS') in body
        # And the ticker itself is written last, from the scope's own key -- so the
        # other scope's instrument is never touched on either path.
        assert "st.session_state[ticker_key] = symbol" in body
        assert 'st.session_state["active_ticker' not in body, (
            "the ticker must be written through the scoped key; a literal key name "
            "here would make one scope overwrite the other's instrument"
        )

    def test_a_ticker_load_lands_the_reader_on_the_price_tab(self):
        """Loading a ticker should show the tape, not whatever tab was last read.

        ``st.tabs`` is *not* a registered widget unless ``on_change`` is passed --
        Streamlit sets ``is_stateful = on_change != "ignore"`` -- so its selection
        lives in the browser and ``default`` is honoured only when the container is
        newly created.  Nothing Python does can otherwise move the reader, which is
        why the bar is re-keyed instead: a new ``key`` is a new block id, and a new
        block id is the one case where ``default`` applies.
        """
        src = app_source()
        # Count call sites, not raw text: the explanatory comment above the main
        # ``st.tabs`` also contains the literal ``key=tabs_key()``.
        sites = re.findall(r"st\.tabs\(list\(TAB_ORDER\),\s*default=DEFAULT_TAB,\s*"
                           r"key=tabs_key\(\)\)", src)
        assert len(sites) == 2, (
            "both st.tabs sites must pass key=tabs_key(); without a key the bar is "
            "stateless and a ticker switch cannot return the reader to Price "
            "(found %d of 2)" % len(sites)
        )

    def test_the_tab_key_advances_on_reset_and_is_per_session(self):
        """The generation must be bumped (not popped) and kept in session state.

        A pop would restore generation 0, which the browser may still be holding, so
        the reset would silently do nothing.  A module global would be shared across
        browser sessions, so one visitor's fetch would yank another's tab bar.
        """
        src = app_source()
        reset = re.search(r"def reset_query_state\(.*?\n(?=\ndef )", src, re.S)
        assert reset
        body = reset.group(0)
        assert "TABS_GENERATION_KEY" in body, (
            "the reset must bump the tab-bar generation"
        )
        assert "st.session_state.pop(TABS_GENERATION_KEY" not in body, (
            "popping the generation restores an earlier key the browser may still "
            "hold, so the reset would quietly do nothing"
        )
        assert "global _TABS_GENERATION" not in src, (
            "the generation must live in session state; a module global is shared by "
            "every browser session on the server"
        )
        key_fn = re.search(r"def tabs_key\(.*?\n(?=\ndef |\n# |\nclass )", src, re.S)
        assert key_fn, "tabs_key not found"
        assert "st.session_state" in key_fn.group(0)

    def test_tabs_are_not_switched_into_lazy_rendering(self):
        """``on_change='rerun'`` would fix the tab problem and cost more than it buys.

        It makes tabs render *lazily*, so every click re-runs the whole script --
        including the deliberately uncached ``pipe.run()`` baseline and bootstrap.
        """
        src = app_source()
        for m in re.finditer(r"st\.tabs\(list\(TAB_ORDER\).*?\)", src, re.S):
            assert "on_change" not in m.group(0), (
                "passing on_change to the main tab bar makes it lazy and re-runs the "
                "uncached forecast on every tab click"
            )

    def test_fetched_bars_are_never_written_to_disk(self):
        """A refresh button that silently overwrote a CSV would destroy a file the
        user may be comparing against."""
        src = app_source()
        fetch_mod = re.search(r"def fetch_ticker_cached\(.*?\n(?=\ndef )", src, re.S)
        assert fetch_mod, "fetch_ticker_cached not found"
        assert "to_csv" not in fetch_mod.group(0)
        assert "to_parquet" not in fetch_mod.group(0)

    def test_no_history_span_control_is_offered(self):
        """There is nothing to choose: 1-minute history stops at ~30 days, so asking
        for less is a choice with no upside -- it only shrinks the candidate pool the
        percentile is measured against."""
        src = app_source()
        resolve = re.search(r"def resolve_ticker\(.*?\n(?=\ndef )", src, re.S)
        assert resolve, "resolve_ticker not found"
        body = resolve.group(0)
        assert 'st.slider' not in body, (
            "the ticker input must not offer a history-span slider; it downloads "
            "everything available"
        )
        assert "How much history to download" not in src
        # The span must not be carried in session state either -- there is no choice,
        # so a second place to hold it is a second place for it to go stale.
        assert "active_ticker_days" not in src

    def test_fetch_always_requests_the_full_window(self):
        """The cached helper must pin the span to "everything available" itself
        rather than trusting a caller, since the UI no longer supplies one.

        **It pins it by passing ``days=None``, not by passing ``F.MAX_1M_DAYS``.**
        ``None`` resolves per timeframe inside :func:`fetch_ticker` -- 29 days on
        1-minute, the whole listing history on daily.  Hard-coding the intraday
        figure here would clip a daily fetch to 29 days, which renders as a short
        archive rather than as a failure."""
        src = app_source()
        fetch_mod = re.search(r"def fetch_ticker_cached\(.*?\n(?=\ndef )", src, re.S)
        assert fetch_mod
        assert "days=None" in fetch_mod.group(0)

    def test_pressing_fetch_again_re_downloads(self):
        """Today's session is still filling up, so a cached frame would freeze the
        newest bars at whatever they looked like on the first click.

        The *Get latest* button this used to name is gone -- there is one **Fetch**
        button per instrument, and ``resolve_ticker`` is where the ``refresh=True``
        lives.
        """
        src = app_source()
        resolve = re.search(r"def resolve_ticker\(.*?\n(?=\ndef )", src, re.S)
        assert resolve, "resolve_ticker not found"
        assert "refresh=True" in resolve.group(0), (
            "pressing Fetch must re-download; without refresh=True the newest "
            "session stays frozen at whatever it looked like on the first click"
        )

    def test_fetch_store_survives_a_rerun(self):
        """The symbol-keyed store must live in a ``cache_resource``, not a module dict.

        A module-level dict is the obvious implementation and it is **silently broken
        under Streamlit**: the script is re-executed in a fresh namespace on every
        interaction, so the dict is re-created empty every time and the "cache" never
        hits.  The visible consequences are not a slow page, they are a wrong one --
        every rerun re-downloads the live archive, which grows by a bar or two between
        requests, so the trailing view's right edge moves and the Price tab's top chart
        creeps even though its window is correctly pinned.

        Asserted on the *decorator*, not on the docstring: the note at the definition
        explains at length why a plain dict fails and quotes ``_FETCH_CACHE``, so a
        substring search would pass on the prose describing the old bug.
        """
        src = app_source()
        fn = re.search(
            r"((?:^@st\.cache_resource[^\n]*\n)+)^def _fetch_store\(", src, re.M)
        assert fn, (
            "_fetch_store is not wrapped in st.cache_resource -- a module-level dict is "
            "re-created empty on every rerun, so nothing is cached and the live archive "
            "grows between reruns (which reads as the top chart scrolling)"
        )
        assert "_FETCH_CACHE" not in src, (
            "the module-level dict is back; it can never hit under Streamlit"
        )
        # And it must actually be the thing the fetch helper reads.
        helper = re.search(r"def fetch_ticker_cached\(.*?\n(?=\ndef )", src, re.S)
        assert "_fetch_store()" in helper.group(0), (
            "fetch_ticker_cached must read the cached store, not a local dict"
        )

    def test_help_copy_substitutes_the_live_symbol(self):
        """Copy hardcoded to QQQ would describe a different market than the chart."""
        src = app_source()
        assert '[[SYMBOL]]' in src, "no SYMBOL placeholder in the help copy"
        filler = re.search(r"def fill_tokens\(.*?\n(?=\ndef )", src, re.S)
        assert filler, "fill_tokens not found"
        assert '[[SYMBOL]]' in filler.group(0)
        assert 'SYMBOL_FOR_HELP' in filler.group(0)

    def test_price_chart_is_titled_with_the_instrument(self):
        """The heading scrolls away; a chart that does not name its instrument can be
        misread as still showing the previous ticker.

        **And it must name the resolution too**, from the same accessor every caption
        uses.  A title that reads "QQQ · close" is ambiguous once daily bars exist,
        and one that reads "QQQ · 1-minute close" while charting daily bars is worse
        than no title.
        """
        src = app_source()
        assert '"{} · {} close".format(SYMBOL_FOR_HELP[0], active_label().lower())' in src
