"""Tests for :mod:`timeseries.sync` — the daily archive-sync path.

What is worth testing here
--------------------------
``run_sync`` shells out to the downloader and reads its result out of *text*.  That
parsing layer is the only genuinely new logic: the subprocess plumbing either works
or it raises, but the mapping from log lines to "how many bars were written" is
where a silent wrong answer could hide.  ``SyncResult.summary`` is the second
candidate — it is what the user reads, so "already up to date" must not be reported
as "0 new bars" without context.

The network is never touched.  ``run_sync`` is exercised against a fake ``python``
that prints canned downloader output, which keeps these tests fast and hermetic
while still covering the real subprocess path end to end.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from timeseries import sync  # noqa: E402
from timeseries.store import EASTERN  # noqa: E402


# --------------------------------------------------------------------------- #
# parse_progress
# --------------------------------------------------------------------------- #
class TestParseProgress:
    def test_a_batch_line_yields_done_and_total(self):
        assert sync.parse_progress("  batch 3/21: 25/25 symbols, 45,890 bars") == (
            "batch",
            3,
            21,
        )

    def test_batch_counts_are_found_even_with_extra_whitespace(self):
        # The downloader pads its labels so the numbers line up in a terminal.  A
        # parser that assumed single spaces would silently report zero progress and
        # the progress bar would sit at 0% while a real sync was running.
        assert sync.parse_progress("  batch  7 / 21 :  24 / 25  symbols") == (
            "batch",
            7,
            21,
        )

    @pytest.mark.parametrize(
        "line,name,value",
        [
            ("  bars written      920,611", "bars_written", 920611),
            ("  sessions written  2,515", "sessions_written", 2515),
            ("  sessions revised  0   (content changed; stored copy kept)", "sessions_revised", 0),
            ("  sessions rejected 3   (failed the quality gate)", "sessions_rejected", 3),
        ],
    )
    def test_summary_counters_are_parsed_with_thousands_separators(self, line, name, value):
        # The comma is load-bearing: without stripping it "920,611" would raise
        # ValueError and a successful sync would be reported as a crash.
        assert sync.parse_progress(line) == ("stat", name, value)

    def test_elapsed_is_parsed_as_a_float(self):
        assert sync.parse_progress("Done in 108.2s") == ("elapsed", 108.2)

    @pytest.mark.parametrize(
        "line",
        [
            "Fetching 2026-09-03..2026-09-10 (6 session(s))",
            "  AAPL: +5 new, 0 extended, 0 unchanged, 1,822 bars",
            "",
            "some entirely unrecognised output",
        ],
    )
    def test_unrecognised_lines_are_ignored_rather_than_raising(self, line):
        # The downloader prints far more than these four shapes -- one line per
        # ticker, quality notes, constituent counts.  A parser that raised on
        # anything else would break the sync on a harmless log line.
        assert sync.parse_progress(line) is None


# --------------------------------------------------------------------------- #
# archive_status
# --------------------------------------------------------------------------- #
def _write_panel(root, sessions, bars=390):
    """Create a minimal on-disk archive: ``sessions`` = list of YYYY-MM-DD strings."""
    import pandas as pd

    os.makedirs(root, exist_ok=True)
    rows = []
    idx = pd.date_range("2026-09-01 13:30", periods=bars, freq="1min", tz="UTC")
    for day in sessions:
        part = os.path.join(root, "ticker=AAA", "date={}".format(day))
        os.makedirs(part, exist_ok=True)
        pd.DataFrame(
            {
                "timestamp": idx,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                "session": day,
                "ticker": "AAA",
            }
        ).to_parquet(os.path.join(part, "bars.parquet"))
        rows.append(
            {
                "ticker": "AAA", "session": day, "bars": bars,
                "fingerprint": "x", "observed_fingerprint": "x",
                "first_bar": str(idx[0]), "last_bar": str(idx[-1]), "n_revisions": 0,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(root, "manifest.csv"), index=False)


class TestArchiveStatus:
    def test_a_missing_archive_reports_never_synced_rather_than_raising(self, tmp_path):
        # The sidebar renders on first load, long before any sync has run.  Raising
        # here would take down the whole page instead of showing an empty state.
        st = sync.archive_status(str(tmp_path / "nope"))
        assert st["exists"] is False
        assert st["never_synced"] is True
        assert st["stale"] is True
        assert st["bars"] == 0

    def test_an_archive_synced_today_is_not_stale(self, tmp_path):
        root = str(tmp_path / "panel")
        today = datetime.now(tz=EASTERN).date()
        _write_panel(root, [today.isoformat()])
        st = sync.archive_status(root, today=datetime.combine(today, datetime.min.time(), tzinfo=EASTERN))
        assert st["never_synced"] is False
        assert st["stale"] is False

    def test_an_archive_behind_the_staleness_threshold_is_flagged(self, tmp_path):
        root = str(tmp_path / "panel")
        now = datetime(2026, 10, 2, 12, 0, tzinfo=EASTERN)
        old = (now.date() - timedelta(days=sync.STALE_AFTER_DAYS + 1)).isoformat()
        _write_panel(root, [old])
        st = sync.archive_status(root, today=now)
        assert st["stale"] is True
        assert st["age_days"] > sync.STALE_AFTER_DAYS

    def test_age_is_measured_from_the_newest_session_not_the_oldest(self, tmp_path):
        # A month-old archive that was synced *today* is current.  Measuring from
        # the oldest session would report it as stale and train the user to ignore
        # the warning, which is worse than having no warning at all.
        root = str(tmp_path / "panel")
        now = datetime(2026, 10, 2, 12, 0, tzinfo=EASTERN)
        _write_panel(root, ["2026-09-03", now.date().isoformat()])
        st = sync.archive_status(root, today=now)
        assert st["oldest"].date() == datetime(2026, 9, 3).date()
        assert st["newest"].date() == datetime(2026, 10, 2).date()
        assert st["age_days"] == 0
        assert st["stale"] is False


# --------------------------------------------------------------------------- #
# SyncResult.summary
# --------------------------------------------------------------------------- #
class TestSummary:
    def test_a_run_that_stored_nothing_does_not_say_zero_bars(self):
        # "0 new bars" reads like a failure.  The common case -- pressing the button
        # twice in a day -- must read as confirmation, not failure.
        res = sync.SyncResult(ok=True, bars_written=0)
        assert "up to date" in res.summary()

    def test_a_successful_run_reports_bars_and_sessions(self):
        res = sync.SyncResult(ok=True, bars_written=920611, sessions_written=2515, elapsed_s=108.2)
        s = res.summary()
        assert "920,611" in s and "2,515" in s

    def test_a_failed_run_leads_with_the_error(self):
        res = sync.SyncResult(ok=False, error="timed out after 900s")
        assert res.summary() == "timed out after 900s"


# --------------------------------------------------------------------------- #
# run_sync, against a fake downloader
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_downloader(tmp_path, monkeypatch):
    """A stub ``python`` that prints canned downloader output.

    Patched in *as the downloader itself* rather than mocking ``subprocess``, so the
    real argument construction, the real pipe and the real parsing are all still
    exercised -- only the network call is replaced.
    """
    stub = tmp_path / "fake_downloader.py"
    stub.write_text(
        textwrap.dedent(
            """
            import sys
            sys.stdout.write("Incremental: 2 of 22 session(s) missing.\\n")
            sys.stdout.write("Fetching 2026-09-03..2026-09-10 (2 session(s))\\n")
            sys.stdout.write("  batch 1/2: 25/25 symbols, 46,520 bars\\n")
            sys.stdout.write("  batch 2/2: 25/25 symbols, 45,851 bars\\n")
            sys.stdout.write("=============================================\\n")
            sys.stdout.write("Done in 7.5s\\n")
            sys.stdout.write("  sessions written  503\\n")
            sys.stdout.write("  sessions revised  1\\n")
            sys.stdout.write("  sessions rejected 2\\n")
            sys.stdout.write("  bars written      92,371\\n")
            """
        )
    )
    monkeypatch.setattr(sync, "stock_script_path", lambda: str(stub))
    return str(stub)


@pytest.fixture()
def fake_daily_downloader(tmp_path, monkeypatch):
    """A stub downloader for the **daily** branch, which uses a different path.

    Separate from :func:`fake_downloader` because ``run_sync`` resolves the script
    by timeframe: patching only ``stock_script_path`` leaves the daily branch looking
    for the real ``scripts/download_daily.py``, which would then hit the network.
    """
    stub = tmp_path / "fake_daily.py"
    stub.write_text(
        textwrap.dedent(
            """
            import sys
            sys.stdout.write("incremental: 3 session(s) held\\n")
            sys.stdout.write("  batch 1/1: 50/50 symbols\\n")
            sys.stdout.write("Done in 4.0s\\n")
            sys.stdout.write("  sessions written  503\\n")
            sys.stdout.write("  bars written      1,000\\n")
            """
        )
    )
    monkeypatch.setattr(sync, "daily_script_path", lambda: str(stub))
    return str(stub)


@pytest.fixture()
def argv():
    """Record the argv the downloader is actually handed.

    A fixture rather than a hand-rolled try/finally in each test, because the
    monkeypatch has to be installed *around* the ``run_sync`` call and several tests
    here also need the downloader patched -- doing that by hand in six places is how
    one of them ends up leaking a patched ``Popen`` into another test.
    """
    seen = {}
    real = subprocess.Popen

    def spy(cmd, *a, **kw):
        seen["cmd"] = list(cmd)
        return real(cmd, *a, **kw)

    sync.subprocess.Popen = spy
    try:
        yield seen
    finally:
        sync.subprocess.Popen = real


def flag(argv_seen, name):
    """The value passed to ``name`` in the recorded argv, or ``None`` if absent."""
    if name not in argv_seen["cmd"]:
        return None
    return argv_seen["cmd"][argv_seen["cmd"].index(name) + 1]


# --------------------------------------------------------------------------- #
# The daily window: a first build must not be clipped to the lookback
# --------------------------------------------------------------------------- #
class TestDailyWindowWidths:
    """``run_sync`` decides how wide a daily fetch to ask for.

    The defect these guard was invisible in the one case that was already tested:
    on a populated archive a narrow window is *correct*, because ``--incremental``
    drops the held sessions regardless of the span requested.  On an empty archive
    that same 29-day window becomes the entire run, and the archive comes out as
    ~20 sessions per ticker -- a shape indistinguishable from a short listing, and
    one nothing on the page reported.
    """

    def test_a_first_build_asks_for_full_history(self, fake_daily_downloader, tmp_path, argv):
        res = sync.run_sync(str(tmp_path / "never_built"), python_exe=sys.executable,
                            timeframe="1d")
        assert res.ok is True
        # No --days and no --start: the downloader then applies its own 17,000-day
        # default, which is the only way to reach the full listing history.
        assert "--days" not in argv["cmd"]
        assert "--start" not in argv["cmd"]
        assert "--incremental" in argv["cmd"]

    def test_a_populated_archive_keeps_the_narrow_lookback(self, fake_daily_downloader,
                                                           tmp_path, argv):
        # The cheap path has to stay cheap: a kept-up-to-date archive should not
        # re-request full listing history on every daily press.
        root = tmp_path / "built"
        (root / "ticker=AAPL").mkdir(parents=True)
        sync.run_sync(str(root), python_exe=sys.executable, timeframe="1d")
        assert flag(argv, "--days") == str(sync.DEFAULT_LOOKBACK_DAYS)

    def test_an_empty_archive_is_told_to_build_in_full(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()  # exists, but holds no ticker= partition: never synced
        assert sync._daily_archive_is_empty(str(root)) is True

    def test_a_populated_archive_is_not_empty(self, tmp_path):
        root = tmp_path / "built"
        (root / "ticker=AAPL").mkdir(parents=True)
        assert sync._daily_archive_is_empty(str(root)) is False

    def test_a_missing_root_is_treated_as_empty_rather_than_raising(self, tmp_path):
        # The sidebar renders before the first sync, so this path runs against a root
        # that does not exist yet.  It must not raise, and "empty" is the safe answer
        # because the wide window can only over-fetch, never under-fetch.
        assert sync._daily_archive_is_empty(str(tmp_path / "absent")) is True

    def test_the_minute_branch_is_untouched_by_the_daily_rule(self, fake_downloader, argv):
        # The whole point of the branch is that the two archives get different
        # arguments; a change that leaked across would put daily bars in the minute
        # archive or the reverse, and both look fine on disk.
        sync.run_sync("unused-root", python_exe=sys.executable, timeframe="1m")
        assert flag(argv, "--start") is not None
        assert "--days" not in argv["cmd"]

    def test_the_default_resolution_still_takes_the_minute_branch(self, fake_downloader, argv):
        # Guards against the branch being inverted: nothing about the daily rule may
        # change what an unqualified call does.
        sync.run_sync("unused-root", python_exe=sys.executable)
        assert "--start" in argv["cmd"]
        assert "--days" not in argv["cmd"]


class TestRunSync:
    def test_counters_are_read_out_of_the_downloader_output(self, fake_downloader):
        res = sync.run_sync("unused-root", python_exe=sys.executable)
        assert res.ok is True
        assert res.bars_written == 92371
        assert res.sessions_written == 503
        assert res.sessions_revised == 1
        assert res.sessions_rejected == 2
        assert res.elapsed_s == pytest.approx(7.5)
        assert res.batches_done == 2 and res.batches_total == 2

    def test_incremental_is_always_passed_so_the_button_is_idempotent(self, fake_downloader, tmp_path):
        # This is what makes the button safe as an unconditional daily step: without
        # --incremental the script would re-download every session already stored.
        seen = {}

        real_popen = subprocess.Popen

        def spy(cmd, *a, **kw):
            seen["cmd"] = cmd
            return real_popen(cmd, *a, **kw)

        sync.subprocess.Popen = spy
        try:
            sync.run_sync("unused-root", python_exe=sys.executable)
        finally:
            sync.subprocess.Popen = real_popen
        assert "--incremental" in seen["cmd"]

    def test_a_lookback_window_wider_than_one_day_is_requested(self, fake_downloader):
        # One day of lookback would fetch only yesterday, so a run missed for a week
        # would silently stay a week behind forever.
        seen = {}
        real_popen = subprocess.Popen

        def spy(cmd, *a, **kw):
            seen["cmd"] = cmd
            return real_popen(cmd, *a, **kw)

        sync.subprocess.Popen = spy
        try:
            sync.run_sync("unused-root", python_exe=sys.executable)
        finally:
            sync.subprocess.Popen = real_popen
        start = seen["cmd"][seen["cmd"].index("--start") + 1]
        requested = datetime.strptime(start, "%Y-%m-%d").date()
        expected = datetime.now(tz=EASTERN).date() - timedelta(days=sync.DEFAULT_LOOKBACK_DAYS)
        assert requested == expected

    def test_a_nonzero_exit_is_reported_as_a_failure(self, tmp_path, monkeypatch):
        stub = tmp_path / "boom.py"
        stub.write_text("import sys; sys.stdout.write('boom\\n'); sys.exit(3)")
        monkeypatch.setattr(sync, "stock_script_path", lambda: str(stub))
        res = sync.run_sync("unused-root", python_exe=sys.executable)
        assert res.ok is False
        assert res.returncode == 3
        assert "3" in res.error

    def test_a_missing_downloader_is_reported_not_raised(self, tmp_path, monkeypatch):
        # The app must survive an incomplete checkout; a traceback would blank the
        # page rather than explain what is missing.
        monkeypatch.setattr(sync, "stock_script_path", lambda: str(tmp_path / "absent.py"))
        res = sync.run_sync("unused-root", python_exe=sys.executable)
        assert res.ok is False
        assert "not found" in res.error

    def test_progress_is_reported_to_the_callback_as_it_arrives(self, fake_downloader):
        seen = []
        sync.run_sync("unused-root", on_progress=lambda d, t, l: seen.append((d, t)), python_exe=sys.executable)
        assert (1, 2) in seen and (2, 2) in seen

    def test_a_failing_callback_does_not_kill_the_sync(self, fake_downloader):
        # The callback renders UI.  An exception in it must not abort a download that
        # is 90 seconds in and has already stored most of its bars.

        def bad(d, t, l):
            raise RuntimeError("boom")

        res = sync.run_sync("unused-root", on_progress=bad, python_exe=sys.executable)
        assert res.ok is True
        assert res.bars_written == 92371

    def test_output_is_truncated_to_the_tail(self, fake_downloader, tmp_path, monkeypatch):
        # A 503-ticker run logs thousands of lines; none of it belongs in a browser.
        stub = tmp_path / "chatty.py"
        stub.write_text("for i in range(5000): print('  TICKER%05d: +1 new' % i)\nprint('bars written 1,000')\n")
        monkeypatch.setattr(sync, "stock_script_path", lambda: str(stub))
        res = sync.run_sync("unused-root", python_exe=sys.executable)
        assert len(res.output.splitlines()) <= 40
        assert res.output.splitlines()[-1] == "bars written 1,000"