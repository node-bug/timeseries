"""The symbol registry: ``constituents.csv`` as a cumulative record of what is held.

Three of these tests exist because the defect they guard was found by *running* the
merge rather than reading it, and in each case the wrong behaviour looked like the
right one -- a registry that had quietly lost a symbol, or one whose symbols had been
stripped of their sectors, still reads as a perfectly plausible CSV.

The property most worth stating up front: a symbol is in the registry because **the
archive holds its bars**, not because a downloader was asked for it.  That is why
every writer here passes the symbols a run actually fetched, and why
:class:`TestOnlyFetchedSymbolsAreRegistered` asserts the negative.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

from timeseries.store import (
    CONSTITUENTS_NAME,
    UNKNOWN_SECTOR,
    PanelStore,
    constituents_path,
    read_constituents,
    read_sectors,
    register_symbols,
    verify_manifest,
)

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import download_daily as DD  # noqa: E402
import download_sp500 as DS  # noqa: E402


def sp500_table(*rows: str) -> pd.DataFrame:
    """A constituent table shaped like :func:`timeseries.store.fetch_sp500_constituents`.

    ``rows`` are ``SYMBOL|SECTOR`` pairs.  The five columns are the real ones, because
    :func:`register_symbols` selects by name -- a narrower table would exercise a
    different path than the one production callers take.
    """
    return pd.DataFrame([
        {"symbol": sym, "name": "%s Inc." % sym, "sector": sec,
         "sub_industry": "%s Sub" % sec, "yahoo_symbol": sym}
        for sym, sec in (r.split("|") for r in rows)
    ])


def sectors_of(root: str) -> dict:
    """``{yahoo_symbol: sector}`` straight off disk, without the marker filtering."""
    df = read_constituents(constituents_path(root))
    return dict(zip(df["yahoo_symbol"], df["sector"]))


# --------------------------------------------------------------------------- #
# Reading: a missing or broken registry costs labels, never data
# --------------------------------------------------------------------------- #
class TestReadingIsTotal:
    """Neither reader may raise.  A cosmetic index cannot take down a search over
    bars that are perfectly readable, so every failure mode degrades to empty."""

    def test_a_missing_registry_reads_as_empty_not_as_an_error(self, tmp_path):
        root = str(tmp_path / "never_built")
        assert read_constituents(constituents_path(root)).empty
        assert read_sectors(constituents_path(root)) == {}

    def test_an_empty_path_is_handled(self, tmp_path):
        """``read_sectors("")`` must not raise -- the app resolves the path at read
        time, and a caller with no archive has no path to offer."""
        assert read_sectors("") == {}
        assert read_constituents("").empty

    def test_a_corrupt_registry_degrades_to_no_labels(self, tmp_path):
        """Bytes that are not a CSV at all.

        The behaviour asserted is the *shape*: the caller gets an empty map rather
        than an exception, which is what lets ``load_panel_search`` stay a one-liner.
        """
        root = str(tmp_path)
        with open(constituents_path(root), "w", encoding="utf-8") as fh:
            fh.write("\x00\x01\x02 not a csv at all")
        assert read_sectors(constituents_path(root)) == {}

    def test_a_registry_missing_a_column_is_tolerated(self, tmp_path):
        """A hand-trimmed file must not make ``read_sectors`` raise on a missing key."""
        root = str(tmp_path)
        pd.DataFrame([{"yahoo_symbol": "AAPL"}]).to_csv(
            constituents_path(root), index=False)
        assert read_sectors(constituents_path(root)) == {}


# --------------------------------------------------------------------------- #
# Writing: the merge, and the two ways it can lose information
# --------------------------------------------------------------------------- #
class TestTheMergeIsCumulative:
    def test_a_second_symbol_is_appended_not_replacing_the_first(self, tmp_path):
        root = str(tmp_path)
        register_symbols(root, ["AAPL"])
        register_symbols(root, ["NVDA"])
        assert set(sectors_of(root)) == {"AAPL", "NVDA"}

    def test_re_registering_the_same_symbol_is_byte_identical(self, tmp_path):
        """Idempotency, checked on the bytes rather than on the parsed row.

        Rows being equal does not make the file equal -- key order and float
        formatting are not visible through ``read_constituents``, and a file that
        rewrites itself differently on every run turns ``git diff`` into noise.
        """
        root = str(tmp_path)
        register_symbols(root, ["AAPL", "NVDA"], table=sp500_table(
            "AAPL|Information Technology", "NVDA|Semiconductors"))
        first = open(constituents_path(root), "rb").read()

        out = register_symbols(root, ["AAPL", "NVDA"], table=sp500_table(
            "AAPL|Information Technology", "NVDA|Semiconductors"))
        second = open(constituents_path(root), "rb").read()

        assert first == second
        assert out["added"] == [], "re-registering an existing symbol is not an addition"

    def test_the_rows_are_sorted_so_the_file_does_not_depend_on_fetch_order(self, tmp_path):
        """Two runs that fetched the same symbols in different orders must produce the
        same file -- otherwise every run looks like a change to a tracked artefact."""
        one, two = str(tmp_path / "a"), str(tmp_path / "b")
        register_symbols(one, ["NVDA", "AAPL", "MSFT"])
        register_symbols(two, ["MSFT", "NVDA", "AAPL"])
        assert (open(constituents_path(one), "rb").read()
                == open(constituents_path(two), "rb").read())

    def test_symbols_are_normalised_to_the_yahoo_spelling(self, tmp_path):
        """``brk.b`` and ``BRK-B`` are one instrument and must be one row.

        Yahoo serves ``BRK-B`` and returns "symbol may be delisted" for ``BRK.B``
        (see ``_DOT_TO_DASH``), so two rows for one stock would give it two bars
        partitions and two sector ranks.
        """
        root = str(tmp_path)
        out = register_symbols(root, ["brk.b", "BRK-B", "  aapl  "])
        assert set(sectors_of(root)) == {"BRK-B", "AAPL"}
        assert out["added"] == ["AAPL", "BRK-B"]

    def test_blank_and_punctuation_only_symbols_are_dropped(self, tmp_path):
        """Normalising them would manufacture a symbol nobody asked for.

        ``None`` is included because a symbol list assembled from a scraped table can
        carry nulls, and ``yahoo_symbol(None)`` would otherwise write the literal
        string ``"NONE"`` into the registry as if it were an instrument.
        """
        root = str(tmp_path)
        register_symbols(root, ["AAPL", "", "   ", None])
        assert set(sectors_of(root)) == {"AAPL"}


class TestASectorIsNeverLost:
    """The regression that made this module necessary.

    :func:`register_symbols` emits a placeholder row only for symbols the registry
    does *not* already hold.  When it emitted one for every requested symbol, the
    ``keep="last"`` dedup let that placeholder -- carrying
    :data:`~timeseries.store.UNKNOWN_SECTOR` -- overwrite a real GICS sector already
    on disk.  Measured on the merge this guards:

        call 1 (no table)  -> AAPL = unknown
        call 2 (table)     -> AAPL = Information Technology
        call 3 (no table)  -> AAPL = unknown      <- the bug
    """

    def test_a_bare_register_never_downgrades_a_known_sector(self, tmp_path):
        root = str(tmp_path)
        register_symbols(root, ["AAPL"], table=sp500_table("AAPL|Information Technology"))
        assert sectors_of(root)["AAPL"] == "Information Technology"

        register_symbols(root, ["AAPL"])
        assert sectors_of(root)["AAPL"] == "Information Technology", (
            "re-running the downloader stripped a known GICS sector; the registry "
            "reports no sector for a symbol it already knows"
        )

    def test_a_later_scrape_upgrades_an_unknown(self, tmp_path):
        """The one overwrite the file allows, and the reason ``--all`` is worth
        re-running: a non-constituent that later joins the index gets a real label."""
        root = str(tmp_path)
        register_symbols(root, ["XYZ"])
        assert sectors_of(root)["XYZ"] == UNKNOWN_SECTOR

        register_symbols(root, ["XYZ"], table=sp500_table("XYZ|Industrials"))
        assert sectors_of(root)["XYZ"] == "Industrials"

    def test_a_later_scrape_does_not_disturb_other_rows(self, tmp_path):
        """The upgrade is surgical.  ``BTC-USD`` was registered before ``AAPL`` joined
        the index, and must survive the scrape that labels ``AAPL``."""
        root = str(tmp_path)
        register_symbols(root, ["BTC-USD", "AAPL"])
        register_symbols(root, ["AAPL", "NVDA"], table=sp500_table(
            "AAPL|Information Technology", "NVDA|Semiconductors"))

        got = sectors_of(root)
        assert set(got) == {"AAPL", "BTC-USD", "NVDA"}
        assert got["BTC-USD"] == UNKNOWN_SECTOR
        assert got["AAPL"] == "Information Technology"

    def test_a_scrape_of_the_whole_index_leaves_non_constituants_alone(self, tmp_path):
        """The original defect, end to end.

        ``--tickers BTC-USD`` registered the symbol, and the next ``--all`` run
        overwrote the file with the scrape result -- deleting the row that admitted
        it, leaving an archive that holds the bars with an index blind to them.
        """
        root = str(tmp_path)
        register_symbols(root, ["BTC-USD"])
        register_symbols(root, ["AAPL", "NVDA"], table=sp500_table(
            "AAPL|Information Technology", "NVDA|Semiconductors"))

        got = sectors_of(root)
        assert "BTC-USD" in got, "the full-index run deleted a non-constituent"
        assert got["AAPL"] == "Information Technology"


# --------------------------------------------------------------------------- #
# The unknown marker, and the NaN it exists to avoid
# --------------------------------------------------------------------------- #
class TestTheUnknownMarker:
    def test_an_unlabelable_symbol_is_recorded_with_the_marker(self, tmp_path):
        root = str(tmp_path)
        register_symbols(root, ["BTC-USD"])
        assert sectors_of(root)["BTC-USD"] == UNKNOWN_SECTOR

    def test_the_marker_is_absent_from_what_the_panel_reads(self, tmp_path):
        """``read_sectors`` is the *only* place the marker is handled, which is what
        keeps :mod:`timeseries.panel` free of any notion of it.

        The panel builds its same-sector pool from a truthy ``sectors.get(home)`` and
        would otherwise pool every unknown-sector symbol into one fabricated
        distribution.
        """
        root = str(tmp_path)
        register_symbols(root, ["BTC-USD", "AAPL"], table=sp500_table(
            "AAPL|Information Technology"))
        assert read_sectors(constituents_path(root)) == {"AAPL": "Information Technology"}

    def test_a_blank_sector_is_dropped_rather_than_read_as_nan(self, tmp_path):
        """The trap the marker exists to avoid.

        ``float('nan')`` is **truthy**.  A registry that wrote ``""`` and read it back
        with a plain ``pd.read_csv`` would produce a *present* sector label of ``NaN``
        for every unlabelled symbol, so ``sectors.get(sym) == home_sector`` would
        compare ``NaN == NaN`` -- False, mostly, but by accident rather than by
        design -- and ``sectors_represented`` would report ``nan`` as if it were a
        GICS sector.  ``keep_default_na=False`` makes the blank a blank, and the
        ``str.len() > 0`` test then drops it.
        """
        root = str(tmp_path)
        pd.DataFrame([
            {"symbol": "AAPL", "name": "Apple", "sector": "",
             "sub_industry": "", "yahoo_symbol": "AAPL"},
            {"symbol": "MSFT", "name": "Microsoft", "sector": "Information Technology",
             "sub_industry": "Software", "yahoo_symbol": "MSFT"},
        ]).to_csv(constituents_path(root), index=False)

        assert read_sectors(constituents_path(root)) == {"MSFT": "Information Technology"}, (
            "a blank sector was read as a label"
        )

    def test_a_plain_read_csv_would_have_produced_the_nan(self, tmp_path):
        """The guard on the guard: proves the previous test is not vacuous.

        Without ``keep_default_na=False`` the same file yields a truthy ``NaN``, so
        ``read_sectors`` is doing real work rather than describing a hypothetical.
        """
        root = str(tmp_path)
        path = constituents_path(root)
        pd.DataFrame([
            {"yahoo_symbol": "AAPL", "sector": ""},
        ]).to_csv(path, index=False)

        naive = pd.read_csv(path, dtype=str)
        assert pd.isna(naive.loc[0, "sector"]), "expected the naive read to see NaN"
        assert bool(naive.loc[0, "sector"]), "NaN is truthy -- that is the whole hazard"
        assert "AAPL" not in read_sectors(path)

    def test_the_marker_is_matched_case_insensitively(self, tmp_path):
        """A hand-edited ``Unknown`` must not sneak past the filter and into the
        panel as a sector named "Unknown"."""
        root = str(tmp_path)
        pd.DataFrame([
            {"yahoo_symbol": "AAPL", "sector": "Unknown"},
            {"yahoo_symbol": "NVDA", "sector": "Semiconductors"},
        ]).to_csv(constituents_path(root), index=False)
        assert read_sectors(constituents_path(root)) == {"NVDA": "Semiconductors"}


# --------------------------------------------------------------------------- #
# The write itself
# --------------------------------------------------------------------------- #
class TestTheWriteIsAtomic:
    """``to_csv`` truncates its target before it streams, so an in-place write leaves
    a window in which a concurrent reader sees a short -- or empty -- registry.  That
    presents as "never registered" rather than as an interrupted write, which is why
    it is worth the temp file.  Same rationale as
    :meth:`PanelStore._write_manifest_atomic`."""

    def test_the_data_goes_through_a_temp_file_and_os_replace(self, tmp_path, monkeypatch):
        root = str(tmp_path)
        seen: dict = {}
        real_replace = os.replace

        def spy(src, dst):
            seen["src"], seen["dst"] = src, dst
            assert os.path.isfile(src), "replace() was called before anything was written"
            seen["tmp"] = src
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        register_symbols(root, ["AAPL"])

        path = constituents_path(root)
        assert seen["dst"] == path
        assert seen["src"] != path, "wrote straight to the live file"
        assert seen["src"].startswith(path), "the temp file must share a directory"
        assert not os.path.exists(seen["src"]), "the temp file was left behind"

    def test_no_temp_file_survives_a_failed_write(self, tmp_path, monkeypatch):
        """A crash between write and replace leaves the temp file behind.  It is never
        the real file, so removing it is always safe -- but it must not be *left*."""
        root = str(tmp_path)
        register_symbols(root, ["AAPL"])

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            register_symbols(root, ["NVDA"])

        path = constituents_path(root)
        assert os.path.isdir(root)
        leftovers = [f for f in os.listdir(root) if f != CONSTITUENTS_NAME]
        assert leftovers == [], "a failed write left %r behind" % leftovers

    def test_the_root_is_created_when_it_does_not_exist(self, tmp_path):
        """The sidecar is the first thing a run of either downloader writes, and on a
        fresh archive there is no directory to write it into."""
        root = str(tmp_path / "brand_new")
        assert not os.path.isdir(root)
        register_symbols(root, ["AAPL"])
        assert os.path.isfile(constituents_path(root))


# --------------------------------------------------------------------------- #
# The scripts: only symbols that produced bars are registered
# --------------------------------------------------------------------------- #
class TestOnlyFetchedSymbolsAreRegistered:
    """End to end through ``main``, with the network replaced.

    The registry's claim is that the archive *holds* these bars.  A symbol that
    returned nothing -- a typo, a delisted ticker, an index Yahoo does not serve --
    would make that claim false, and the archive's own ``tickers()`` list would
    contradict it.
    """

    def test_a_fetched_symbol_is_registered_and_its_bars_are_stored(self, monkeypatch, tmp_path):
        def fake_fetch(symbols, **kwargs):
            return {s: daily_frame() for s in symbols}

        monkeypatch.setattr(DD, "_fetch", fake_fetch)
        root = str(tmp_path)
        assert DD.main(["--root", root, "--tickers", "ZZZZ", *WINDOW]) == 0

        store = PanelStore(root)
        assert "ZZZZ" in store.tickers(), "the bars were not written"
        assert verify_manifest(root)["consistent"]
        assert sectors_of(root) == {"ZZZZ": UNKNOWN_SECTOR}, (
            "a symbol the archive holds has no row in the registry"
        )
        assert store.tickers() == list(read_constituents(constituents_path(root))["yahoo_symbol"])

    def test_a_symbol_that_returned_no_bars_is_not_registered(self, monkeypatch, tmp_path):
        """``ZZZZ`` is asked for and ``AAPL`` comes back; only ``AAPL`` may be
        recorded.  The alternative is a registry claiming to hold a symbol with no
        partition behind it."""
        def fake_fetch(symbols, **kwargs):
            return {s: daily_frame() for s in symbols if s == "AAPL"}

        monkeypatch.setattr(DD, "_fetch", fake_fetch)
        root = str(tmp_path)
        DD.main(["--root", root, "--tickers", "AAPL", "ZZZZ", *WINDOW])

        assert set(sectors_of(root)) == {"AAPL"}
        assert "ZZZZ" not in PanelStore(root).tickers()

    def test_the_minute_downloader_registers_too(self, monkeypatch, tmp_path):
        """The same contract on the other script, which accumulates across its
        per-window date chunks rather than fetching once."""
        def fake_fetch(symbols, start, end, **kwargs):
            return {s: minute_frame() for s in symbols}

        monkeypatch.setattr(DS, "_fetch", fake_fetch)
        root = str(tmp_path)
        assert DS.main(["--root", root, "--tickers", "QQQ",
                        "--start", SETTLED, "--end", SETTLED]) == 0

        assert sectors_of(root) == {"QQQ": UNKNOWN_SECTOR}
        assert "QQQ" in PanelStore(root).tickers()
        assert verify_manifest(root)["consistent"]

    def test_a_registry_failure_does_not_fail_the_run(self, monkeypatch, tmp_path, capsys):
        """The bars are already written by the time the registry is touched, and the
        bars are what the operator needs to hear about.  A cosmetic index failing
        must not turn a completed download into a failed one."""
        def fake_fetch(symbols, **kwargs):
            return {s: daily_frame() for s in symbols}

        def boom(*a, **k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(DD, "_fetch", fake_fetch)
        monkeypatch.setattr(DD, "register_symbols", boom)
        root = str(tmp_path)
        assert DD.main(["--root", root, "--tickers", "ZZZZ", *WINDOW]) == 0, (
            "a registry failure failed a run that had stored every bar"
        )
        assert "ZZZZ" in PanelStore(root).tickers()
        assert "WARNING" in capsys.readouterr().out

    def test_a_scrape_run_labels_the_index_without_dropping_others(
            self, monkeypatch, tmp_path):
        """The regression, through the script: ``--tickers BTC-USD`` then a full
        constituent run, and ``BTC-USD`` must still be there."""
        def fake_fetch(symbols, **kwargs):
            return {s: daily_frame() for s in symbols}

        monkeypatch.setattr(DD, "_fetch", fake_fetch)
        monkeypatch.setattr(
            DD, "fetch_sp500_constituents",
            lambda: sp500_table("AAPL|Information Technology", "NVDA|Semiconductors"))

        root = str(tmp_path)
        DD.main(["--root", root, "--tickers", "BTC-USD", *WINDOW])
        DD.main(["--root", root, *WINDOW])  # no --tickers: the scraped universe

        got = sectors_of(root)
        assert got.get("BTC-USD") == UNKNOWN_SECTOR, (
            "the full-index run deleted a non-constituent from the registry"
        )
        assert got.get("AAPL") == "Information Technology"


# --------------------------------------------------------------------------- #
# Frames the script tests above are built from
# --------------------------------------------------------------------------- #
#: A weekday well in the past, so "is this session settled?" has a real answer rather
#: than depending on today's date -- matching ``tests/test_daily_archive.py``.
SETTLED = "2026-09-03"

#: The window the script tests ask for.
#:
#: **Explicit, not ``--days``.**  ``--days`` is resolved against *today*, and
#: ``download_daily`` slices the fetched frame back to the window it was asked for
#: before writing -- so a fixture dated in the past combined with ``--days 3`` writes
#: nothing at all.  The result reads as a broken store ("no tickers, manifest
#: consistent") rather than as a fixture that fell outside the range.
WINDOW = ["--start", SETTLED, "--end", "2026-09-04"]


def daily_frame(n: int = 3, first: str = SETTLED) -> pd.DataFrame:
    """``n`` daily bars at 09:30 ET on settled weekdays.

    The stamp matters: midnight UTC would be the *previous* Eastern date and would
    land in the wrong partition.  The bars run forward from ``first`` so they cover
    the whole of :data:`WINDOW`.
    """
    ts = pd.date_range("%s 13:30:00" % first, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [1.0] * n, "high": [2.0] * n,
        "low": [0.5] * n, "close": [1.5] * n,
    })


def minute_frame(bars: int = 390, day: str = SETTLED) -> pd.DataFrame:
    """One complete 1-minute session: 13:30-19:59 UTC on a settled weekday.

    Complete on purpose -- the store's quality gate rejects a short session, and a
    rejected session is not written, so a stub returning 30 bars would produce an
    archive with no partitions and a registry with nothing to match.
    """
    idx = pd.date_range("%s 13:30" % day, periods=bars, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": idx,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
    })