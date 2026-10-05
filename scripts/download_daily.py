#!/usr/bin/env python3
"""Populate the S&P 500 **daily** archive.  PLAN.md §CG.

Yahoo serves a full listing history for ``interval="1d"`` in a single
request, with no retention wall and no per-request day limit, so this
script is structurally simpler than ``download_sp500.py``: there is no
chunking to work around, and one request per ticker covers everything.

```bash
# every constituent, full history (one batched request per 50 symbols)
python scripts/download_daily.py

# a named subset
python scripts/download_daily.py --tickers AAPL MSFT NVDA

# a symbol outside the index: stored, and registered with an ``unknown`` sector
python scripts/download_daily.py --tickers BTC-USD

# incremental: fetch only the sessions the archive is missing
python scripts/download_daily.py --incremental

# widen the existing archive by a few days
python scripts/download_daily.py --incremental --days 5

# repair the manifest index after a manual file change
python scripts/download_daily.py --rebuild-manifest
```

The symbol registry
-------------------
Every symbol this run **actually fetched** is merged into ``constituents.csv``, the
sidecar that records which symbols the archive holds and, where known, their GICS
sector.  Two properties of that file matter and both are enforced by
:func:`timeseries.store.register_symbols`:

* **it is cumulative** -- an explicit ``--tickers BTC-USD`` run adds a row, and the
  next ``--all`` run does not delete it.  The original behaviour wrote the scrape
  result unconditionally, so every non-constituent was silently removed the moment
  anyone re-ran the full download;
* **an unlabelable symbol says so** -- ``BTC-USD`` is recorded with an explicit
  ``unknown`` sector rather than a blank cell, so the gap is visible in the file and
  :func:`timeseries.store.read_sectors` can drop it.  A blank would have been read
  back as a *truthy* ``NaN`` and pooled every unlabelled symbol into a fabricated
  "same sector" distribution.

A symbol that returned no bars at all is **not** registered: the registry's claim is
that the archive holds those bars, and a zero-bar result does not support it.

Why a separate archive, not a namespace in the minute one
-----------------------------------------------------------
The two are not two views of one dataset.  A daily partition holds
**one** bar; a 1-minute partition holds ~390.  They are fetched under
different retention rules (Yahoo keeps ~29 days of minute bars and a
century of daily ones), and they answer different questions.  Sharing a
root would mean every read had to say which it meant -- the same
"two resolutions under one name" ambiguity the per-session resolution
gate exists to remove.

The on-disk layout is nevertheless **identical** to the minute archive --
``ticker=<SYM>/date=<YYYY-MM-DD>/bars.parquet`` plus ``manifest.csv`` --
so :class:`timeseries.store.PanelStore` reads and writes both with no
format branch at all.  Only the *validation* differs, and that difference
is carried by ``timeframe="1d"`` rather than by a second store class.

Why the store had to be taught about daily
------------------------------------------
Two answers inside :class:`~timeseries.store.PanelStore` depend on the
resolution, and both were silently wrong for daily bars:

* the quality gate expects 390 bars per session, so every daily bar was
  reported as ``short session: 1/390 bars`` -- one spurious warning per
  trading day, forever;
* a daily bar is stamped 09:30 ET, which is strictly *before* a
  1-minute session's final-bar boundary, so **every daily bar read as a
  session still in progress**.  A re-fetch with different content was
  merged as "growth" instead of flagged as the retro-adjustment it is,
  which disabled the one warning §B exists to raise.

Both are fixed by threading ``timeframe`` through ``PanelStore.write``;
see its docstring.  ``--verify-revisions`` below exercises the second
one on real data rather than trusting it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional, Sequence

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from timeseries.fetch import fetch_ticker  # noqa: E402
from timeseries.store import (  # noqa: E402
    EASTERN,
    MERGE_POLICY,
    PanelStore,
    fetch_sp500_constituents,
    register_symbols,
    verify_manifest,
)
from timeseries.timeframes import get_timeframe  # noqa: E402

DEFAULT_ROOT = os.path.join(_ROOT, "data", "sp500_daily")

#: The resolution this script builds.  Named once so no call site can
#: quietly disagree -- every write, validation and quality report below
#: reads it from here.
DAILY = get_timeframe("1d")

# yfinance batches these into parallel requests; larger batches are
# throttled more aggressively and start returning empty frames.  Daily is
# one request per symbol either way, so this batches *symbols*.
BATCH_SIZE = 50

#: Pause between batches, in seconds.  Same rationale as the minute
#: downloader's: Yahoo throttles on request rate, not on volume.
PAUSE_S = 0.5

#: Attempts per batch before falling back to one symbol at a time.
RETRIES = 2

#: How far back to ask Yahoo for, in days, when no window is given.
#: ~46 years, deliberately past ``DEFAULT_DAILY_DAYS`` (25 years), which measurably
#: truncates: on AAPL it stops at 2001-10-11 where Yahoo actually holds 1980-12-12.
#: Bounded rather than infinite so a request can never be unbounded by accident.
DAILY_HISTORY_DAYS = 17_000


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def log(msg: str) -> None:
    print(msg, flush=True)


def _universe(args: argparse.Namespace) -> tuple[list[str], Optional[pd.DataFrame]]:
    """The symbols to fetch, and the constituent table when it was fetched."""
    if args.tickers:
        syms = sorted({s.strip().upper() for s in args.tickers if s.strip()})
        if not syms:
            raise SystemExit("--tickers was given but no symbols survived cleaning")
        log(f"universe: {len(syms)} explicit symbol(s)")
        return syms, None
    table = fetch_sp500_constituents()
    syms = sorted({str(s).strip().upper() for s in table["yahoo_symbol"] if str(s).strip()})
    log(f"universe: {len(syms)} S&P 500 constituent(s)")
    return syms, table


def _fetch(
    symbols: Sequence[str],
    *,
    start: date,
    end: date,
    batch_size: int = BATCH_SIZE,
    pause: float = PAUSE_S,
    on_batch: Optional[Callable[[int, int], None]] = None,
) -> dict[str, pd.DataFrame]:
    """Daily bars for ``symbols``, batched, with per-batch retry.

    Returns only the symbols that produced at least one bar.  A symbol
    with no history (delisted, or an index Yahoo does not serve daily)
    is absent from the result rather than an error: an empty archive for
    one ticker is not a failure of the run.

    The per-symbol fallback exists because a batch can fail wholesale
    while every symbol in it is fine -- throttling usually hits the batch
    as a unit.  Retrying symbol-by-symbol salvages the rest instead of
    losing a day of the archive to one bad actor.
    """
    out: dict[str, pd.DataFrame] = {}
    batches = [list(symbols[i:i + batch_size]) for i in range(0, len(symbols), batch_size)]
    for n, batch in enumerate(batches, start=1):
        got = _fetch_batch(batch, start, end)
        if got is None and len(batch) > 1:
            log(f"  batch {n}/{len(batches)}: whole batch failed, retrying singly")
            got = {}
            for sym in batch:
                one = _fetch_batch([sym], start, end)
                if one:
                    got.update(one)
        out.update(got)
        if on_batch is not None:
            on_batch(n, len(batches))
        if n < len(batches) and pause:
            time.sleep(pause)
    return out


def _fetch_batch(symbols: Sequence[str], start: date, end: date) -> Optional[dict[str, pd.DataFrame]]:
    """One batched attempt.  ``None`` means the whole attempt failed."""
    # ``fetch_ticker`` treats its span as **exclusive**: it computes
    # ``start = end - span`` and asks Yahoo for ``(start, end)``.  So the archive's
    # inclusive window is one day short at each end.
    #
    # The tempting fix -- move ``stop`` forward -- is exactly wrong, and measurably so.
    # Because the span is anchored to ``stop``, pushing ``stop`` out also pushes the
    # *start* out, and the oldest history is what gets clipped.  Measured on AAPL, with
    # a fixed span of 36,530 days:
    #
    #     stop = end+0d  -> 6285 sessions, first 2001-10-10   <- everything
    #     stop = end+1d  -> 6284 sessions, first 2001-10-11
    #     stop = end+2d  -> 6283 sessions, first 2001-10-12
    #     stop = end+30d -> 6263 sessions, first 2001-11-09   <- a month lost
    #
    # So ``stop`` stays exactly on the inclusive end, and the missing days are
    # recovered by widening the *span* instead, which is free: daily has no retention
    # wall, so a span beyond the listing is simply ignored by the endpoint.  The
    # session slice then trims back to the window the caller actually asked for.
    stop = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    # The span, not ``None``: ``None`` resolves to the registry's 25-year default,
    # which truncates.  See the note at the call site.
    span = (end - start).days + 2
    for attempt in range(RETRIES):
        last: Optional[Exception] = None
        try:
            got: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                res = fetch_ticker(sym, end=stop, days=span, timeframe=DAILY.key)
                if res.frame is None or len(res.frame) == 0:
                    continue
                got[sym] = res.frame
            if got:
                return got
            # An empty result is not an exception; Yahoo simply has nothing
            # for these symbols.  Retrying every symbol individually would
            # burn a whole batch of requests to learn that.
            return {}
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < RETRIES:
                time.sleep(1.5 * (attempt + 1))
    log(f"  batch failed after {RETRIES} attempts: {last!r}")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Populate the S&P 500 daily archive (one bar per session).",
    )
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"archive root (default: {DEFAULT_ROOT})")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="explicit symbols (default: every S&P 500 constituent)")
    ap.add_argument("--days", type=int, default=None,
                    help="how far back to fetch (default: the full listing history)")
    ap.add_argument("--start", default=None, help="explicit start, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="explicit end, YYYY-MM-DD")
    ap.add_argument("--incremental", action="store_true",
                    help="fetch only sessions the archive is missing")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"symbols per request batch (default: {BATCH_SIZE})")
    ap.add_argument("--policy", default=MERGE_POLICY,
                    help=f"flag|replace|error on a fingerprint mismatch (default: {MERGE_POLICY})")
    ap.add_argument("--rebuild-manifest", action="store_true",
                    help="regenerate manifest.csv from the partitions and exit")
    ap.add_argument("--verify-revisions", action="store_true",
                    help="after building, re-read two sessions and report whether "
                         "the revision check is live (see the module docstring)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    started = time.monotonic()

    if args.rebuild_manifest:
        store = PanelStore(args.root)
        out = store.rebuild_manifest()
        log(f"rebuilt {out['rebuilt']} manifest row(s); dropped {len(out['dropped'])}")
        rep = verify_manifest(args.root)
        log(f"consistent={rep['consistent']} partitions={rep['n_partitions']} "
            f"manifest={rep['n_manifest']}")
        return 0 if rep["consistent"] else 1

    # **The window is resolved and validated before the universe is fetched.**
    # Order matters for a reason that is not tidiness: resolving the universe
    # scrapes Wikipedia, so validating afterwards meant a bad `--start`/`--end`
    # pair cost a network round-trip before being rejected -- and, in a test,
    # meant an argument-validation test reached the internet.
    #
    # `--rebuild-manifest` still short-circuits before both: it touches no window
    # and no universe.
    end = _parse_date(args.end) if args.end else datetime.now(tz=EASTERN).date()
    if args.start:
        start = _parse_date(args.start)
    elif args.days is not None:
        start = end - timedelta(days=args.days)
    else:
        # **The full listing history, not the registry's 25-year default.**
        # ``fetch_ticker``'s ``days=None`` resolves to ``DEFAULT_DAILY_DAYS`` = 9,125
        # (25 years), and that silently truncates.  Measured on AAPL, same stop date:
        #
        #     days=None              ->  6,284 sessions, first 2001-10-11
        #     days=DEFAULT_DAILY_DAYS->  6,284 sessions, first 2001-10-11
        #     days=17,000            -> 11,544 sessions, first 1980-12-12
        #
        # So the archive default would have begun 21 years late, and nothing on the
        # page would say so -- the chart would simply look like a shorter history.
        # 17,000 days (~46 years) is comfortably past Yahoo's daily floor and still
        # bounded, so a request can never be unbounded by accident.  A wider request
        # costs nothing: daily has no retention wall, and the session slice trims back
        # to the window actually asked for.
        start = end - timedelta(days=DAILY_HISTORY_DAYS)
    if start > end:
        raise SystemExit(f"--start {start} is after --end {end}")

    symbols, table = _universe(args)

    store = PanelStore(args.root)
    # The root does not exist until the first write, and ``constituents.csv`` is
    # written before that -- so a fresh run died on the very first symbol with
    # "Cannot save file into a non-existent directory".  Created here, where the
    # intent is obvious, rather than left to whichever write happens to come first.
    os.makedirs(args.root, exist_ok=True)
    held = set(store.sessions())

    if args.dry_run:
        log(f"root        : {args.root}")
        log(f"timeframe   : {DAILY.key} ({DAILY.label})")
        log(f"symbols     : {len(symbols)}")
        log(f"window      : {start} .. {end}")
        log(f"already held: {len(held)} session(s)")
        return 0

    if held:
        rep = verify_manifest(args.root)
        if not rep["consistent"]:
            log(f"WARNING: manifest is out of step with the partitions "
                f"({len(rep['missing_from_manifest'])} on disk but unindexed, "
                f"{len(rep['missing_from_disk'])} indexed but absent). "
                f"Run --rebuild-manifest to repair.")

    if table is not None:
        # The scrape result is merged, not written -- see `register_symbols`.  The
        # actual registration of *fetched* symbols happens after the write loop,
        # below; this call is only there so a run that downloads nothing new still
        # refreshes the GICS labels, which is the one thing a bare re-run can improve.
        _register(args.root, symbols, table=table)

    if args.incremental and held and args.tickers is None:
        # Only sessions the archive has never seen.  The newest session is
        # deliberately re-fetched: it was written while still trading, so it
        # is the one partition most likely to have gained bars since.
        todo = sorted(set(_weekdays(start, end)) - held)
        # ...but a re-fetch of a held day is what finds revisions, so for a
        # full universe we still walk the newest few days.
        recent = [d for d in _weekdays(end - timedelta(days=args.days or 5), end)]
        for d in recent:
            if d not in todo:
                todo.append(d)
        todo = sorted(set(todo))
        log(f"incremental: {len(held)} session(s) held; fetching {len(todo)} missing "
            f"+ the most recent days (revisions are only visible by re-fetching)")
    else:
        todo = _weekdays(start, end)
        log(f"fetching {len(todo)} weekday(s) {start}..{end}")

    if not todo:
        log("nothing to do")
        return 0

    def progress(done: int, total: int) -> None:
        log(f"  batch {done}/{total}")

    frames = _fetch(symbols, start=start, end=end,
                    batch_size=args.batch_size, on_batch=progress)
    log(f"fetched bars for {len(frames)}/{len(symbols)} symbol(s)")

    totals = {"written": 0, "extended": 0, "unchanged": 0,
              "revised": 0, "rejected": 0, "bars": 0}
    issues: list[str] = []
    # One manifest rewrite per run, not per ticker: see `PanelStore.batched_manifest`.
    # Without it this loop rewrites a ~176MB index 503 times, which is a build that
    # does not finish rather than a slow one.
    with store.batched_manifest():
        for n_done, sym in enumerate(sorted(frames), start=1):
            frame = frames[sym]
            # Slice to the requested window: fetch_ticker is asked for
            # everything and the archive should hold only what was asked for,
            # or an incremental run would write the whole history every time.
            sub = _slice(frame, min(todo), max(todo))
            if sub is None:
                continue
            res = store.write(sym, sub, policy=args.policy, timeframe=DAILY.key)
            totals["written"] += res.sessions_written
            totals["extended"] += res.sessions_extended
            totals["unchanged"] += res.sessions_unchanged
            totals["revised"] += res.sessions_revised
            totals["rejected"] += res.sessions_rejected
            totals["bars"] += res.bars_written
            issues.extend(res.fatal)
            issues.extend(res.errors)
            # Per-ticker progress.  The write loop used to report nothing at all,
            # so a run that was slow -- or wedged -- looked identical to one that
            # was working: both printed "fetched bars for 503/503 symbol(s)" and
            # then went silent for minutes with no way to tell the two apart.
            log(f"  write {n_done}/{len(frames)} {sym}: "
                f"written={res.sessions_written} unchanged={res.sessions_unchanged} "
                f"revised={res.sessions_revised} "
                f"({time.monotonic() - started:.0f}s elapsed)")
    log(f"sessions written={totals['written']} extended={totals['extended']} "
        f"unchanged={totals['unchanged']} revised={totals['revised']} "
        f"rejected={totals['rejected']} bars={totals['bars']}")
    if issues:
        log(f"{len(issues)} issue(s); first five:")
        for line in issues[:5]:
            log(f"  {line}")

    # **Register what was actually fetched, not what was asked for.**  ``frames``
    # holds the symbols that produced at least one bar, so a typo, a delisted ticker
    # or an index Yahoo does not serve daily is never recorded as held -- the
    # registry's claim is that the archive contains these bars, and a symbol that
    # returned nothing would make that claim false.
    _register(args.root, sorted(frames))

    rep = verify_manifest(args.root)
    log(f"manifest consistent={rep['consistent']} "
        f"partitions={rep['n_partitions']} rows={rep['n_manifest']}")

    if args.verify_revisions:
        _verify_revision_check(store, frames)

    log(f"done in {time.monotonic() - started:.1f}s")
    return 0 if rep["consistent"] else 1


def _register(root: str, symbols: Sequence[str], *,
              table: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """Add ``symbols`` to the archive's symbol registry, and report what changed.

    A registry write that fails must not take down a run that has already stored
    every bar it fetched: the registry is a derived index (the partition
    directories are the truth about what is held), and the failure the operator
    actually needs to see is the bar one.  So this logs and returns ``None`` rather
    than propagating, exactly as the old bare ``to_csv`` call was guarded in
    ``download_sp500.py``.
    """
    try:
        out = register_symbols(root, symbols, table=table)
    except Exception as exc:  # noqa: BLE001 - the bars are already written
        log(f"WARNING: could not update the symbol registry ({exc!r}); "
            f"the bars themselves are unaffected.")
        return None
    if out["added"]:
        log(f"registered {len(out['added'])} new symbol(s): "
            f"{', '.join(out['added'][:10])}"
            f"{' ...' if len(out['added']) > 10 else ''}")
    else:
        log(f"registry: no new symbols ({out['total']} recorded)")
    return out


def _weekdays(start: date, end: date) -> list[date]:
    """Weekdays in ``[start, end]``.

    A weekday filter is not a trading calendar: it includes holidays,
    which come back with zero bars and are skipped by the store's quality
    gate rather than special-cased here.  No holiday table is needed or
    maintained, and a holiday therefore costs nothing to get wrong.
    """
    if end < start:
        return []
    out = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _slice(frame: pd.DataFrame, first: date, last: date) -> Optional[pd.DataFrame]:
    """Rows whose Eastern session falls in ``[first, last]``, or ``None``."""
    from timeseries.store import session_et

    if frame is None or len(frame) == 0:
        return None
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()]
    sessions = session_et(out["timestamp"]).dt.strftime("%Y-%m-%d")
    keep = (sessions >= first.isoformat()) & (sessions <= last.isoformat())
    out = out.loc[keep]
    return out if len(out) else None


def _verify_revision_check(store: PanelStore, frames: dict[str, pd.DataFrame]) -> None:
    """Prove the revision check is *live* for daily bars, not merely present.

    A cheap and dishonest-looking test is to re-fetch and check nothing
    changed -- but "nothing changed" is exactly what a *disabled* revision
    check also reports, because a disabled one silently merges instead of
    flagging.  Both outcomes look identical to the caller.

    So this perturbs one stored day, re-writes it, and asserts the store
    *flagged* it and *kept* the original value.  A store that had the
    09:30-stamp bug would report "session in progress, merged" instead, and
    the stored value would have been overwritten -- which is the real
    failure, and the one that matters.
    """
    if not frames:
        log("verify-revisions: no frames to check")
        return
    sym = sorted(frames)[0]
    day = store.sessions(sym)[0] if store.sessions(sym) else None
    if day is None:
        log("verify-revisions: nothing stored for the first symbol")
        return
    path = store._partition_path(sym, "", timeframe=DAILY.key)  # noqa: SLF001 - probe
    stored = pd.read_parquet(path)
    perturbed = stored.copy()
    last = perturbed.index[-1]
    perturbed.loc[last, "close"] = float(perturbed["close"].iloc[-1]) + 1.0
    res = store.write(sym, perturbed, timeframe=DAILY.key)
    after = pd.read_parquet(path)
    kept = abs(float(after["close"].iloc[-1]) - float(stored["close"].iloc[-1])) < 1e-12
    log(f"verify-revisions: {sym} {day} -> revised={res.sessions_revised} "
        f"extended={res.sessions_extended} stored_value_kept={kept}")
    if not (res.sessions_revised >= 1 and kept):
        log("verify-revisions: FAILED -- the daily revision check is not live. "
            "A changed settled day would be silently overwritten.")
    # Restore the partition so the archive is not left holding the probe.
    stored.to_parquet(path, index=False)
    store.rebuild_manifest()
    log("verify-revisions: partition restored")


if __name__ == "__main__":
    raise SystemExit(main())