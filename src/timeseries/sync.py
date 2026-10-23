"""Run the archive sync from inside the app.  Implements PLAN.md §B as a daily step.

Why this module exists
----------------------
Yahoo serves 1-minute bars only for roughly the **last 30 days**.  That is a
*fetch* limit, not a *retention* limit: whatever is downloaded is written to
``data/sp500_panel`` and stays there forever.  The consequence is that the archive
is a one-way ratchet -- roughly five sessions are added per trading day, and a
session that is never downloaded while it is still within the window is gone for
good.

That makes "download every day" a real operational requirement rather than a
convenience, and an operational requirement that lives only in a README is one
people stop doing.  This module exists so the requirement can be discharged from
the same place the data is read: one button in the sidebar.

Why a subprocess
----------------
The downloader is :mod:`scripts.download_sp500`, a ``__main__`` module with its own
argument parsing and logging.  Re-implementing that logic here would let the button
and the command line drift apart, which is the one failure mode a "same thing,
two entry points" design actually produces.

So the button *shells out* to the script and streams its output.  Two properties
fall out of that choice:

* the CLI stays the single source of truth for how an archive is grown -- the
  button is a front end to it, never a second implementation of it;
* the work happens in a child process, so a 90-second sync cannot block the
  Streamlit script run, cannot corrupt the parent's import state on a partial
  failure, and cannot leave the UI thread holding a half-written archive.

The trade-off is that progress is reported by parsing the script's stdout, which
is text rather than a structured event stream.  :func:`parse_progress` therefore
takes the conservative view: it reports what it can *prove*, and never guesses.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Optional, Sequence

from .store import EASTERN, PanelStore
from .timeframes import DEFAULT_TIMEFRAME, resolve_timeframe

__all__ = [
    "SyncResult",
    "archive_integrity",
    "archive_status",
    "daily_script_path",
    "parse_progress",
    "run_sync",
    "stock_script_path",
    "DEFAULT_LOOKBACK_DAYS",
    "STALE_AFTER_DAYS",
    "STALE_AFTER_SESSIONS",
]

#: Days back the sync looks for missing sessions.
#:
#: One would be enough to fetch yesterday.  The margin is deliberate: Yahoo's window
#: is 30 days, so a run that was skipped for a few days -- or that failed on a
#: throttled batch -- has to be able to catch up in one invocation.  At ~5 sessions
#: per trading day the extra 29 days cost nothing when there is nothing to fetch,
#: because ``--incremental`` skips every session the archive already holds.
DEFAULT_LOOKBACK_DAYS = 29

#: A session older than this is considered missing rather than merely absent from
#: the manifest, and is what makes "the archive is stale" a computable statement.
#:
#: Two calendar days of slack rather than zero: today is *always* short (the market
#: is still trading, so today has ~150 bars instead of 390), and a session that was
#: downloaded before the close stays short forever unless it is re-fetched.  Waiting
#: a day before calling anything stale avoids flagging a healthy archive as overdue
#: every morning.
STALE_AFTER_DAYS = 2

#: The same threshold, counted in **trading sessions** rather than calendar days.
#:
#: The calendar version of this constant was mis-calibrated, and mis-calibrated in
#: the direction that hides the problem: a Friday-to-Monday gap is three calendar
#: days but zero trading sessions, so a perfectly current Friday archive was
#: reported as three days stale on Monday morning.  A stale archive therefore
#: looked stale for the wrong reason, and a healthy one looked stale too -- which
#: is the failure mode that teaches a user to ignore a warning.
#:
#: Two sessions of slack for the same reason as :data:`STALE_AFTER_DAYS`: today is
#: always partially downloaded, and the previous session is settled only after the
#: close.
STALE_AFTER_SESSIONS = 2

# Log line emitted by ``download_sp500.main`` for one batch of tickers.  Matched rather
#: than parsed positionally so an added or reordered field does not silently shift the
# counts.  The spacing is deliberately loose: the downloader pads its labels so the
# numbers align in a terminal, and a parser pinned to single spaces would report zero
# progress -- leaving the bar at 0% through an entire real download.
_BATCH_RE = re.compile(r"batch\s+(\d+)\s*/\s*(\d+)\s*:?\s*(\d+)\s*/\s*(\d+)\s+symbols")
_DONE_RE = re.compile(r"^Done in\s+([\d.]+)s")
_BARS_RE = re.compile(r"^\s*bars written\s+([\d,]+)")
_SESSIONS_RE = re.compile(r"^\s*sessions written\s+([\d,]+)")
_REVISED_RE = re.compile(r"^\s*sessions revised\s+([\d,]+)")
_REJECTED_RE = re.compile(r"^\s*sessions rejected\s+([\d,]+)")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _repo_root() -> str:
    """The project root: two levels up from ``src/timeseries/sync.py``."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stock_script_path() -> str:
    """Absolute path to the stock ``scripts/download_sp500.py``.

    Kept as a function rather than a module constant so a test can monkeypatch a
    single name, and so the path is resolved against *this file* rather than the
    process working directory -- Streamlit may be started from anywhere.
    """
    return os.path.join(_repo_root(), "scripts", "download_sp500.py")


def daily_script_path() -> str:
    """Absolute path to the stock ``scripts/download_daily.py``.

    A second script rather than a flag on the first, because the two downloads are
    genuinely different jobs rather than one job with a switch: the minute downloader
    chunks a bounded window because Yahoo refuses to serve more, while the daily one
    makes one unbounded request per symbol.  A shared code path would mean one of
    them carrying the other's constraints -- and the chunking would silently cap the
    daily history at a window Yahoo never imposed on it.

    Resolved against *this file* rather than the process working directory, for the
    reason given on :func:`stock_script_path`: Streamlit may be started from
    anywhere.
    """
    return os.path.join(_repo_root(), "scripts", "download_daily.py")


@dataclass
class SyncResult:
    """Outcome of one sync run."""

    ok: bool = False
    returncode: Optional[int] = None
    #: Tail of the script's stdout, for display when something needs explaining.
    output: str = ""
    bars_written: int = 0
    sessions_written: int = 0
    sessions_revised: int = 0
    sessions_rejected: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    #: Batches seen, so a stalled run can report "12 of 21" rather than nothing.
    batches_done: int = 0
    batches_total: int = 0

    def summary(self) -> str:
        """One-line human summary, used in the button's success caption."""
        if not self.ok:
            return self.error or "sync failed"
        if self.bars_written == 0:
            return "already up to date - nothing new to store"
        return "{:,} new bars across {:,} sessions in {:.0f}s".format(
            self.bars_written, self.sessions_written, self.elapsed_s
        )


def parse_progress(line: str) -> Optional[tuple]:
    """Pull progress out of one line of downloader output.

    Returns ``None`` for anything not recognised, so the caller can pass every line
    through harmlessly.  Recognised shapes:

    * ``("batch", done, total)`` from a per-batch log line
    * ``("batches", done, total)`` from the final archive line
    * ``("stat", name, value)`` for one of the summary counters
    * ``("elapsed", seconds)`` from the ``Done in`` line
    """
    m = _BATCH_RE.search(line)
    if m:
        return ("batch", int(m.group(1)), int(m.group(2)))

    m = _DONE_RE.search(line)
    if m:
        return ("elapsed", float(m.group(1)))

    for name, pattern in (
        ("bars_written", _BARS_RE),
        ("sessions_written", _SESSIONS_RE),
        ("sessions_revised", _REVISED_RE),
        ("sessions_rejected", _REJECTED_RE),
    ):
        m = pattern.match(line)
        if m:
            return ("stat", name, _int(m.group(1)))
    return None


def _is_weekday(day: str) -> bool:
    """True when ``day`` (``YYYY-MM-DD``) is a Monday-Friday.

    A weekday filter, **not** a trading calendar: market holidays count as
    trading days.  The project carries no holiday table, and a count that is
    right only when a dependency is installed is worse than a count that is
    consistently one or two off in a known direction.
    """
    try:
        return datetime.strptime(day, "%Y-%m-%d").weekday() < 5
    except (TypeError, ValueError):
        return False


def _trading_sessions_between(start: datetime, end: datetime) -> int:
    """Weekdays strictly after ``start`` up to and including ``end``.

    A **weekday count**, not an exchange calendar: market holidays are counted as
    trading days.  That is a deliberate limitation and it is stated rather than
    hidden -- the alternative needs a holiday calendar this project does not
    carry, and a wrong-but-close count is more useful than a count that is right
    only when a dependency is installed.

    Why a session count at all: the archive is measured in **trading sessions**
    (one session = one day of bars for one ticker), so "three days old" is
    ambiguous over a weekend -- a Friday-to-Monday gap is zero trading sessions
    but three calendar days.  Reporting the calendar number made a healthy Friday
    archive look stale on Monday morning, which trains the user to ignore the
    warning.
    """
    if end <= start:
        return 0
    day = start.date() + timedelta(days=1)
    last = end.date()
    count = 0
    while day <= last:
        if day.weekday() < 5:  # Mon-Fri
            count += 1
        day += timedelta(days=1)
    return count


def archive_status(root: str, *, today: Optional[datetime] = None) -> dict:
    """Summarise an archive: how current it is, and how complete it is.

    Two questions, and the old version of this function answered only the first
    while *appearing* to answer both:

    1. **How current is the archive?**  Answered from the **cohort frontier** --
       the *mode* of the per-ticker newest sessions -- not from the union maximum.
    2. **How complete is it?**  Answered from the per-ticker coverage behind that
       frontier, plus the count of partitions that cannot be read at all.

    Why the frontier is a mode and not a maximum
    --------------------------------------------
    The archive holds three instruments that trade 24/7 (``BTC-USD``, ``HO=F``,
    ``NG=F``) alongside the 503-name equity cohort.  A *union* of sessions is
    therefore always at least as new as the equity cohort, and usually newer by a
    weekend, because those three tick in on days the equities do not.  Taking the
    union's maximum made the sidebar say "Up to date" while 500 of 503 equity
    tickers were **nine trading sessions** behind.  The union is the wrong
    statistic for the question being asked, and the mode is the right one: it is
    the date that *most* tickers have reached.

    ``root`` may not exist at all -- that is reported as "never synced" rather
    than raised, because the sidebar must render before the first sync.
    """
    now = today or datetime.now(tz=EASTERN)
    out = {
        "exists": os.path.isdir(root),
        "tickers": 0,
        "sessions": [],
        "bars": 0,
        "newest": None,
        "oldest": None,
        "age_days": None,
        "stale": True,
        "never_synced": True,
        # --- cohort-aware additions ------------------------------------- #
        #: The date most tickers have reached (the mode of per-ticker newest).
        "frontier": None,
        #: Tickers whose newest session is exactly the frontier.
        "at_frontier": 0,
        #: Tickers whose newest session is *behind* the frontier.
        "behind_frontier": 0,
        #: Those tickers, by symbol.
        "lagging_tickers": [],
        #: Tickers whose newest session is *ahead* of the frontier (the 24/7
        #: instruments, and anyone else ahead of the crowd).
        "ahead_frontier": 0,
        #: Trading sessions between the frontier and now.
        "age_sessions": None,
        #: Trading sessions in which the cohort is behind the frontier.
        "missing_sessions": [],
        #: Number of partition files that exist but cannot be read.
        "unreadable": 0,
        #: Tickers with at least one unreadable partition.
        "unreadable_tickers": [],
    }
    if not out["exists"]:
        return out

    try:
        store = PanelStore(root)
        per_ticker = store.session_map()
        out["tickers"] = len(store.tickers())
    except Exception:  # noqa: BLE001 - a bad archive must still render the sidebar
        return out

    if not per_ticker:
        return out

    out["never_synced"] = False

    # --- per-ticker newest sessions ------------------------------------- #
    newest_per_ticker = {sym: days[-1] for sym, days in per_ticker.items()}
    out["sessions"] = sorted({d for days in per_ticker.values() for d in days})

    # The frontier is the mode of the newest-session distribution.  Ties are
    # broken towards the *older* date, because the older date is the one the
    # cohort as a whole is safely known to have reached.
    counts: dict[str, int] = {}
    for day in newest_per_ticker.values():
        counts[day] = counts.get(day, 0) + 1
    frontier_day = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    out["at_frontier"] = counts[frontier_day]
    out["behind_frontier"] = sum(c for d, c in counts.items() if d < frontier_day)
    out["ahead_frontier"] = sum(c for d, c in counts.items() if d > frontier_day)
    out["lagging_tickers"] = sorted(
        sym for sym, day in newest_per_ticker.items() if day < frontier_day)

    # --- newest / oldest, and the two ages -------------------------------- #
    try:
        out["newest"] = datetime.strptime(out["sessions"][-1], "%Y-%m-%d").replace(tzinfo=EASTERN)
        out["oldest"] = datetime.strptime(out["sessions"][0], "%Y-%m-%d").replace(tzinfo=EASTERN)
        frontier_dt = datetime.strptime(frontier_day, "%Y-%m-%d").replace(tzinfo=EASTERN)
    except ValueError:
        return out
    out["frontier"] = frontier_dt

    # The age that matters is the frontier's age, not the union's: the union's
    # age is measured from a weekend session that the equity cohort never
    # reached, so it reads as zero while the cohort is nine sessions behind.
    out["age_days"] = (now - frontier_dt).days
    out["age_sessions"] = _trading_sessions_between(frontier_dt, now)
    out["stale"] = out["age_sessions"] > STALE_AFTER_SESSIONS

    # --- which sessions the cohort is missing ----------------------------- #
    # The sessions the *cohort* should have reached but has not: the weekdays
    # between the frontier and the newest session any ticker holds.  Restricted to
    # weekdays because the equity cohort does not trade on a weekend, and
    # reporting "missing Saturday" for a 5-day holiday weekend is the kind of
    # false alarm that makes people stop reading the warning.
    #
    # The count of these is deliberately the same number as ``age_sessions``:
    # the cohort is behind by exactly the sessions it has not caught up on.
    if out["sessions"]:
        newest_session = out["sessions"][-1]
        if newest_session > frontier_day:
            out["missing_sessions"] = [
                d for d in out["sessions"]
                if frontier_day < d <= newest_session and _is_weekday(d)
            ]

    # --- unreadable partitions ------------------------------------------- #
    # NOT computed here.  Counting them means reading every partition's data
    # pages, which costs 2.5s on the 13,179-file panel archive -- too much to
    # pay on every Streamlit rerun, which happens on every sidebar interaction.
    # It is a separate, explicitly-called check (:func:`archive_integrity`), so
    # the cost is paid when someone asks "is this archive intact?" rather than
    # on every page load.
    out["unreadable"] = 0
    out["unreadable_tickers"] = []

    # --- bar total -------------------------------------------------------- #
    # Bar total comes from the manifest when it agrees with disk, because summing
    # 10k+ parquet files on every rerun would be far too slow for a sidebar metric.
    # A missing or drifting manifest reports 0 bars rather than a wrong number.
    try:
        import pandas as pd

        manifest = os.path.join(root, "manifest.csv")
        if os.path.isfile(manifest):
            # ``engine="pyarrow"`` reads this manifest ~5x faster than the C engine
            # (measured on the 571MB daily manifest: 1.85s -> 0.38s for the two
            # columns this needs), and the sidebar recomputes it on every rerun.
            frame = pd.read_csv(manifest, usecols=["bars"], engine="pyarrow")
            out["bars"] = int(frame["bars"].sum())
    except Exception:  # noqa: BLE001 - a cosmetic metric must never break the page
        out["bars"] = 0
    return out


def _count_unreadable_partitions(root: str) -> tuple:
    """``(n_unreadable, [tickers])`` for the partitions under ``root``.

    A partition is "unreadable" when its *footer* parses but its data pages do
    not, or when it cannot be opened at all.  The footer is the cheap check and
    it is deliberately not enough on its own: a truncated or byte-shifted file
    still has a valid footer, so counting only files that fail to open would
    report zero for exactly the archives that are in the worst shape.

    The cost is one full read per partition -- 2.5s for the 13,179-file panel
    archive -- which is why :func:`archive_integrity` is a separate call rather
    than part of :func:`archive_status`.
    """
    from .store import _iter_partition_files

    n = 0
    tickers: set[str] = set()
    for sym, _day, path in _iter_partition_files(root):
        try:
            import pyarrow.parquet as pq

            pq.ParquetFile(path).read()
        except Exception:  # noqa: BLE001 - any failure means "not readable"
            n += 1
            tickers.add(sym)
    return n, sorted(tickers)


def archive_integrity(root: str) -> dict:
    """Check that every partition under ``root`` can actually be read.

    Separate from :func:`archive_status` because it is **expensive**: it reads
    every partition's data pages (2.5s for the 13,179-file panel archive, 0.1s
    for the 504-file daily one) and the sidebar recomputes on every rerun.

    Why it exists at all
    --------------------
    Because a Parquet file whose bytes have been altered still has a **valid
    footer**.  The footer records *where* the data pages are, not *what* they
    contain, so a file that has lost a byte anywhere in its data region parses
    cleanly and only fails when the pages are actually decoded.  Every reader
    in this project caught that failure and moved on -- which is the right thing
    for a reader to do and the wrong thing for a reader to do *silently*.

    The consequence was that 442 of 504 daily archives were unreadable, and the
    archive reported itself as holding 191,181 rows instead of 4,164,637, with
    no error anywhere.  An archive that has lost 94% of its data and reports
    success is worse than one that fails loudly, because nothing prompts anyone
    to look.

    The fix is not to make the readers noisier -- a reader that raises on one
    bad partition cannot serve the other 500 -- but to make the *check* a thing
    that exists and can be run, and to surface its result.

    Returns a dict with ``n_partitions``, ``n_unreadable``, ``unreadable``
    (the offending paths) and ``unreadable_tickers``.
    """
    from .store import _iter_partition_files

    total = 0
    bad: list[str] = []
    tickers: set[str] = set()
    for sym, _day, path in _iter_partition_files(root):
        total += 1
        try:
            import pyarrow.parquet as pq

            pq.ParquetFile(path).read()
        except Exception:  # noqa: BLE001 - any failure means "not readable"
            bad.append(path)
            tickers.add(sym)
    return {
        "n_partitions": total,
        "n_unreadable": len(bad),
        "unreadable": bad,
        "unreadable_tickers": sorted(tickers),
        "ok": not bad,
    }


def _iter_lines(proc: "subprocess.Popen") -> Iterator[str]:
    """Yield stdout lines from ``proc`` without blocking the caller."""
    stream = proc.stdout
    if stream is None:
        return
    for raw in iter(stream.readline, ""):
        yield raw.rstrip("\n")


def _daily_archive_is_empty(root: str) -> bool:
    """Does ``root`` hold no daily sessions yet?

    Decides the daily window width, so it is on the hot path for every sync -- and
    it must answer cheaply.  :meth:`PanelStore.sessions` walks every partition file
    on the archive, which on a full 503-ticker build is tens of thousands of
    stat() calls, paid on each of the button's reruns.

    The cheap and sufficient test is the partition directories: an archive with no
    ``ticker=`` directory has no sessions by definition, and an archive that has
    one has held at least one symbol.  A *partially* built archive -- one ticker
    written, 502 missing -- answers "not empty", which is the correct conservative
    answer here: the per-symbol loop in ``download_daily`` fetches every symbol in
    the universe regardless of what is held, so a wide window is needed to fill in
    the rest either way.

    Never raises.  A root that is absent, unreadable, or mid-rebuild answers
    "empty", which selects the wide window -- the safe direction, because that is
    the one that can only ever fetch more.
    """
    try:
        return not PanelStore(root).tickers()
    except Exception:  # noqa: BLE001 - an unreadable root must pick the safe window
        return True


def run_sync(
    root: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    tickers: Optional[Sequence[str]] = None,
    on_progress=None,
    timeout_s: Optional[float] = None,
    python_exe: Optional[str] = None,
    timeframe: object = DEFAULT_TIMEFRAME,
    full_history: Optional[bool] = None,
) -> SyncResult:
    """Run the downloader for ``timeframe`` as a child process and collect its result.

    Parameters
    ----------
    root
        Archive root to write to; passed as ``--root``.
    lookback_days
        How far back to look for missing sessions.  ``--incremental`` skips every
        session already stored, so a wide window costs nothing when there is
        nothing to fetch.
    tickers
        Optional explicit symbol subset.  Omit for the whole index.
    on_progress
        Optional ``callback(batches_done, batches_total, line)`` invoked as output
        arrives, so a UI can render progress without polling.
    timeout_s
        Optional wall-clock cap.  Exceeding it kills the child and returns a failed
        result -- a sync left running forever would hold a Streamlit session open.
    timeframe
        Which downloader to run, and therefore which archive is written.  Resolved
        here rather than left to the caller, because **passing the right root with
        the wrong script is a silent failure**: the minute downloader would happily
        write 390-bar sessions into the daily archive, or the daily one would write
        one-bar sessions into the minute one.  Both look fine on disk, and both are
        found only later -- by a search that compares one against the other.
    full_history
        Force the daily window wide or narrow regardless of what is on disk.  See
        the note on the window itself below; ``None`` (the default) reads the
        archive and decides, which is what every caller wants.

    Notes
    -----
    ``--incremental`` is always passed.  Without it the script would re-fetch and
    re-write every session the archive holds, which for a full index means
    re-downloading megabytes to discover nothing changed.

    The manifest is rebuilt after a successful run.  That is the documented repair
    for an interrupted download, and the button is exactly the situation where an
    interruption is likely -- so the button both causes and prevents the failure it
    repairs.
    """
    tf = resolve_timeframe(timeframe)
    script = stock_script_path() if tf.bars_per_session > 1 else daily_script_path()
    result = SyncResult()
    if not os.path.isfile(script):
        result.error = "downloader not found at {}".format(script)
        result.returncode = -1
        return result

    cmd = [
        python_exe or sys.executable,
        "-u",              # unbuffered: progress must reach the UI live
        script,
        "--root", str(root),
        "--incremental",
    ]
    # The daily downloader takes a *span*, not a start date, and the two need
    # different arguments for the same reason they need different scripts: the
    # minute archive is a rolling window that is extended a few days at a time,
    # while the daily archive is the full listing history.
    #
    # **The daily span now depends on whether there is anything to extend.**
    # It used to pass ``--days`` with ``lookback_days`` (29), which
    # ``download_daily.py`` resolves as ``start = end - 29 days`` -- so *every*
    # daily sync was hard-capped to a 29-day window.  On a populated archive that
    # is invisible, because ``--incremental`` drops the held sessions anyway and
    # a recent re-fetch is all a kept-up-to-date archive needs.  On an *empty* one
    # it is the entire run: pressing the button to build the archive the rest of
    # the app reads produced ~20 sessions per ticker instead of the full listing,
    # and nothing on the page said so -- the chart simply looked like a short
    # history.  The comment immediately above this block used to say the daily
    # archive "must not be handed a 29-day lookback -- that would clip it to 29
    # days, which is precisely the truncation §CG exists to avoid", and then
    # handed it exactly that.
    #
    # So: an archive with no sessions gets **no window argument at all**, and the
    # downloader resolves its own full-history default (17,000 days, past Yahoo's
    # daily floor).  A populated one keeps the narrow window, because that is the
    # cheap path and it is already correct -- this preserves the measured
    # "first run slow, later ones not" property in the README rather than making
    # every daily sync a full re-download.
    if tf.bars_per_session > 1:
        cmd += ["--start",
                (datetime.now(tz=EASTERN).date()
                 - timedelta(days=int(lookback_days))).isoformat()]
    else:
        # A first build is asked for in full -- no ``--days`` and no ``--start``,
        # so the downloader resolves its own 17,000-day default.
        first_build = (full_history if full_history is not None
                       else _daily_archive_is_empty(root))
        if not first_build:
            cmd += ["--days", str(int(lookback_days))]
    if tickers:
        cmd += ["--tickers", *[str(t) for t in tickers]]

    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=_repo_root(),
        )
    except Exception as exc:  # noqa: BLE001
        result.error = "could not start the downloader: {}".format(exc)
        result.returncode = -1
        return result

    started = datetime.now()

    def _pump() -> None:
        for line in _iter_lines(proc):
            lines.append(line)
            ev = parse_progress(line)
            if ev and ev[0] == "batch":
                result.batches_done, result.batches_total = ev[1], ev[2]
            elif ev and ev[0] == "stat":
                setattr(result, ev[1], ev[2])
            elif ev and ev[0] == "elapsed":
                result.elapsed_s = ev[1]
            if on_progress is not None:
                try:
                    on_progress(result.batches_done, result.batches_total, line)
                except Exception:  # noqa: BLE001 - a bad callback must not kill the sync
                    pass

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
    finally:
        pump.join(timeout=10)

    result.returncode = proc.returncode
    # Keep the tail only: the full log of a 500-ticker run is thousands of lines and
    # none of it belongs in a browser.
    result.output = "\n".join(lines[-40:])
    if timed_out:
        result.error = "timed out after {:.0f}s".format(timeout_s or 0)
        result.ok = False
    elif result.returncode == 0:
        result.ok = True
        if not result.elapsed_s:
            result.elapsed_s = (datetime.now() - started).total_seconds()
    else:
        result.error = "downloader exited with code {}".format(result.returncode)

    if result.ok:
        # Repair the index in-process.  Cheap next to the download, and it means a
        # button-driven sync can never leave a drifted manifest behind.
        try:
            manifest = os.path.join(root, "manifest.csv")
            if os.path.isfile(manifest):
                store = PanelStore(root)
                store.rebuild_manifest()
        except Exception:  # noqa: BLE001 - the bars are safe even if the index is not
            pass
    return result