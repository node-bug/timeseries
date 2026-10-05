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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Empty, Queue
from typing import Iterator, Optional, Sequence

from .store import EASTERN, PanelStore
from .timeframes import DEFAULT_TIMEFRAME, resolve_timeframe

__all__ = [
    "SyncResult",
    "archive_status",
    "daily_script_path",
    "parse_progress",
    "run_sync",
    "stock_script_path",
    "DEFAULT_LOOKBACK_DAYS",
    "STALE_AFTER_DAYS",
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


def archive_status(root: str, *, today: Optional[datetime] = None) -> dict:
    """Summarise an archive: sessions held, bars held, and whether it is stale.

    "Stale" is computed from the **newest session on disk**, not from the manifest,
    so a manifest that drifted from the partitions cannot make a stale archive look
    current.  ``root`` may not exist at all -- that is reported as "never synced"
    rather than raised, because the sidebar must render before the first sync.
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
    }
    if not out["exists"]:
        return out

    try:
        store = PanelStore(root)
        sessions = sorted(store.sessions())
        out["sessions"] = sessions
        out["tickers"] = len(store.tickers())
    except Exception:  # noqa: BLE001 - a bad archive must still render the sidebar
        return out

    if not sessions:
        return out

    out["never_synced"] = False
    try:
        newest = datetime.strptime(sessions[-1], "%Y-%m-%d").replace(tzinfo=EASTERN)
        oldest = datetime.strptime(sessions[0], "%Y-%m-%d").replace(tzinfo=EASTERN)
    except ValueError:
        return out
    out["newest"] = newest
    out["oldest"] = oldest
    out["age_days"] = (now - newest).days
    out["stale"] = out["age_days"] > STALE_AFTER_DAYS

    # Bar total comes from the manifest when it agrees with disk, because summing
    # 10k+ parquet files on every rerun would be far too slow for a sidebar metric.
    # A missing or drifting manifest reports 0 bars rather than a wrong number.
    try:
        import pandas as pd

        manifest = os.path.join(root, "manifest.csv")
        if os.path.isfile(manifest):
            out["bars"] = int(pd.read_csv(manifest, usecols=["bars"])["bars"].sum())
    except Exception:  # noqa: BLE001 - a cosmetic metric must never break the page
        out["bars"] = 0
    return out


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
            import pandas as pd

            manifest = os.path.join(root, "manifest.csv")
            if os.path.isfile(manifest):
                store = PanelStore(root)
                store.rebuild_manifest()
        except Exception:  # noqa: BLE001 - the bars are safe even if the index is not
            pass
    return result