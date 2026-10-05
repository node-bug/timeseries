#!/usr/bin/env python3
"""Populate the S&P 500 1-minute archive.  PLAN.md §B/§F.

Yahoo serves only ~8 days of 1-minute bars per request, so this script is built to be
run *repeatedly*: each run fetches what it can, appends it to the partitioned archive,
and leaves the previous sessions untouched.  The archive is the product; the download
is just how it grows (§B/§F).

```bash
# every constituent, last 8 days (one request per 50 symbols, batched)
python scripts/download_sp500.py

# a named subset, and only the sessions that are missing
python scripts/download_sp500.py --tickers AAPL MSFT NVDA --days 5

# a symbol outside the index: stored, and registered with an ``unknown`` sector
python scripts/download_sp500.py --tickers BTC-USD --days 5

# incremental: fetch only sessions the archive does not already hold
python scripts/download_sp500.py --incremental --days 5

# widen the existing archive by one more session
python scripts/download_sp500.py --incremental --start 2026-09-22 --end 2026-09-26
```

The symbol registry
-------------------
Every symbol this run **actually fetched** is merged into ``constituents.csv``, the
sidecar that records which symbols the archive holds and, where known, their GICS
sector.  It is cumulative: an explicit ``--tickers BTC-USD`` run adds a row and a
later ``--all`` run does not delete it, which the original unconditional
``to_csv`` could not promise.  A symbol outside the index is recorded with an
explicit ``unknown`` sector rather than a blank cell -- see
:func:`timeseries.store.register_symbols` for why the blank is the worse choice.

Why the constituent list is fetched rather than hard-coded
---------------------------------------------------------
A hard-coded ticker list is a snapshot that quietly rots: it goes stale on index
changes, and it silently drifts from "the S&P 500" without any signal.  Fetching the
current list means ``--all`` means *all*, today, and the GICS sector labels arrive
with it -- which is what lets a cross-sectional search report same-sector percentiles
rather than one undifferentiated number.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from timeseries.store import (  # noqa: E402
    MERGE_POLICY,
    EASTERN,
    PanelStore,
    fetch_sp500_constituents,
    register_symbols,
    verify_manifest,
)

DEFAULT_ROOT = os.path.join(_ROOT, "data", "sp500_panel")

# Yahoo rejects a 1m request spanning more than ~8 days, so the fetch window is
# capped here rather than left to fail server-side with an opaque message.
MAX_REQUEST_DAYS = 7

# yfinance batches these into parallel requests; larger batches are throttled more
# aggressively and start returning empty frames.
BATCH_SIZE = 25


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _trading_days(start: date, end: date) -> list[date]:
    """Weekdays in ``[start, end]``.

    A weekday filter is not a trading calendar: it includes holidays, which come back
    with zero bars.  Those are skipped by the store's quality gate rather than
    special-cased here, so no holiday table is needed or maintained.
    """
    if end < start:
        return []
    out, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _download_batch(symbols: Sequence[str], start: date, end: date, *, log=print) -> dict[str, pd.DataFrame]:
    """Fetch 1-minute bars for a batch of symbols, returning ``{symbol: frame}``.

    Uses one ``yf.download`` call per batch rather than one call per symbol: the
    per-symbol loop in the old ``scripts/download_data.py`` issued 500 sequential
    requests, which is both slow and a reliable way to get rate-limited.  Batch
    ``start``/``end`` are Eastern midnight converted to UTC because yfinance
    interprets naive datetimes in the exchange's local time.
    """
    import yfinance as yf

    start_utc = datetime.combine(start, datetime.min.time(), tzinfo=EASTERN).astimezone(timezone.utc)
    end_utc = datetime.combine(end + timedelta(days=1), datetime.min.time(),
                                tzinfo=EASTERN).astimezone(timezone.utc)

    raw = yf.download(
        list(symbols), start=start_utc, end=end_utc, interval="1m",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
        actions=False, timeout=60,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out

    cols = raw.columns
    # yfinance returns a (Ticker, Price) MultiIndex whenever `group_by="ticker"` is
    # set -- including for a *single* symbol, so this cannot be keyed on the symbol
    # count.  Keying on `len(symbols) > 1` made every one-symbol retry fall through
    # to the flat-column branch, where the columns are still ("AAPL","Close") pairs
    # and the symbol is silently dropped.
    multi = isinstance(cols, pd.MultiIndex)
    wanted = ("Open", "High", "Low", "Close", "Volume")
    for sym in symbols:
        try:
            if multi:
                if sym not in cols.get_level_values(0):
                    continue
                frame = raw[sym].copy()
            elif len(symbols) == 1:
                frame = raw.copy()
            else:
                log(f"  {sym}: unexpected flat columns for a {len(symbols)}-symbol "
                    f"batch; skipped")
                continue
        except KeyError:
            continue
        if frame is None or frame.empty or frame.dropna(how="all").empty:
            continue
        # `keep` is resolved against the *incoming* (capitalised) column names; the
        # rename to lowercase happens only afterwards.  Doing it the other way round
        # makes `keep` empty and silently yields zero symbols -- which is exactly how
        # a full batch appeared to download successfully while storing nothing.
        keep = [c for c in wanted if c in frame.columns]
        if "Close" not in keep:
            log(f"  {sym}: no Close column (got {list(frame.columns)}); skipped")
            continue
        frame = frame[keep].dropna(subset=["Close"])
        if frame.empty:
            continue
        frame = frame.rename(columns={c: c.lower() for c in keep})
        frame.index = pd.to_datetime(frame.index, utc=True)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame["ticker"] = sym
        out[sym] = frame.reset_index(names="timestamp")
    return out


def _fetch(symbols: Sequence[str], start: date, end: date, *,
           batch_size: int = BATCH_SIZE, retries: int = 2,
           pause: float = 0.4, log=print) -> dict[str, pd.DataFrame]:
    """Fetch every symbol in ``symbols`` over ``[start, end]``, in batches.

    A failed batch is retried symbol-by-symbol on the final attempt, so one bad
    ticker cannot discard the other 24 that came back fine.  Empty results are simply
    absent from the returned dict -- a holiday or a not-yet-listed symbol has no
    bars, and that is reported as "0 bars", not treated as an error.
    """
    frames: dict[str, pd.DataFrame] = {}
    chunks = [list(symbols[i:i + batch_size]) for i in range(0, len(symbols), batch_size)]
    for n, chunk in enumerate(chunks, 1):
        got = {}
        if not chunk:
            continue
        for attempt in range(retries + 1):
            try:
                got = _download_batch(chunk, start, end, log=log)
            except Exception as exc:  # noqa: BLE001 - a batch failure must not abort the run
                log(f"  batch {n}/{len(chunks)} attempt {attempt + 1} failed: {exc!r}")
                got = {}
            if got:
                break
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

        if not got and chunk:
            # Retry one at a time so a single bad symbol does not lose the batch.
            for sym in chunk:
                try:
                    one = _download_batch([sym], start, end, log=log)
                except Exception:  # noqa: BLE001
                    one = {}
                got.update(one)

        frames.update(got)
        log(f"  batch {n}/{len(chunks)}: {len(got)}/{len(chunk)} symbols, "
            f"{sum(len(f) for f in got.values()):,} bars")
        time.sleep(pause)
    return frames


def _chunk_days(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into windows of at most ``max_days`` calendar days."""
    if end < start:
        return []
    step = timedelta(days=max_days)
    out, cur = [], start
    while cur <= end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download S&P 500 1-minute bars into the partitioned archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="archive root directory")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="explicit symbols (default: the current S&P 500 constituents)")
    ap.add_argument("--all", action="store_true",
                    help="download every constituent (the default when --tickers is absent)")
    ap.add_argument("--days", type=int, default=5,
                    help="days back from today to start (capped by Yahoo's ~8-day limit)")
    ap.add_argument("--start", default=None, help="explicit start date, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="explicit end date, YYYY-MM-DD")
    ap.add_argument("--incremental", action="store_true",
                    help="only fetch sessions already in the archive are skipped")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--policy", default=MERGE_POLICY, choices=("flag", "replace", "error"),
                    help="what to do when a re-fetch changes a stored session")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--rebuild-manifest", action="store_true",
                    help="regenerate manifest.csv from the partitions on disk and exit. "
                         "The manifest is a derived index, so this is the repair for an "
                         "interrupted run: it never touches the bars themselves.")
    args = ap.parse_args(argv)

    def log(*a):
        print(*a, flush=True)

    # ---- resolve the universe -------------------------------------------- #
    constituents = None
    if args.tickers:
        symbols = sorted({str(s).strip().upper() for s in args.tickers})
        log(f"Using {len(symbols)} explicit symbol(s).")
    else:
        try:
            constituents = fetch_sp500_constituents()
        except Exception as exc:  # noqa: BLE001
            log(f"Could not fetch the constituent list: {exc!r}")
            log("Falling back to a 20-name smoke-test list. Pass --tickers to override.")
            symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
                       "JPM", "JNJ", "V", "PG", "UNH", "MA", "HD", "XOM", "BAC", "PFE"]
        else:
            symbols = constituents["yahoo_symbol"].tolist()
            log(f"Fetched {len(symbols)} S&P 500 constituents "
                f"({constituents['sector'].nunique()} sectors).")

    # ---- resolve the date range ------------------------------------------ #
    today = datetime.now(tz=EASTERN).date()
    end = _parse_date(args.end) if args.end else today
    start = _parse_date(args.start) if args.start else end - timedelta(days=args.days)
    if end < start:
        log(f"Empty range: start {start} is after end {end}.")
        return 2
    if (end - start).days + 1 > MAX_REQUEST_DAYS + 1:
        log(f"Splitting {start}..{end} into {MAX_REQUEST_DAYS}-day requests "
            f"(Yahoo's 1m limit).")

    store = PanelStore(args.root)
    if store.tickers():
        log(f"Archive already holds {len(store.tickers())} ticker(s) "
            f"across {len(store.sessions())} session(s).")

    if args.rebuild_manifest:
        rep = store.rebuild_manifest()
        check = verify_manifest(args.root)
        log(f"Rebuilt manifest from {rep['rebuilt']:,} partition(s).")
        if rep["dropped"]:
            log(f"  {len(rep['dropped'])} unreadable/empty partition(s) skipped: "
                f"{rep['dropped'][:5]}")
        log(f"  manifest consistent: {check['consistent']}")
        return 0

    # A manifest that has drifted is reported on every run.  It costs one directory
    # walk and is the difference between noticing an interrupted download and
    # quietly treating a stale index as truth.
    drift = verify_manifest(args.root)
    if not drift["consistent"]:
        log(f"Note: manifest is out of step with the partitions "
            f"({len(drift['missing_from_manifest'])} unindexed, "
            f"{len(drift['missing_from_disk'])} indexed but absent). "
            f"Repair with --rebuild-manifest. The bars themselves are unaffected.")

    # Persist the sector map next to the archive.  It is a *cache of* the constituent
    # list, never required to read the bars: with it, the panel can report a
    # same-sector percentile; without it, the panel still searches and simply says so.
    #
    # **Merged, not overwritten.**  Writing the scrape result unconditionally deleted
    # every symbol that is not a current constituent -- so one `--tickers BTC-USD` run
    # left an archive holding the bars while the index did not know they existed, and
    # the next `--all` run erased the row that admitted it.
    #
    # The *fetched* symbols are registered after the write loop instead, so this call
    # exists only to refresh GICS labels on a run that stored nothing new.
    _register(args.root, symbols, table=constituents, log=log)

    # ---- incremental: skip days every ticker already has ------------------ #
    wanted = _trading_days(start, end)
    if args.incremental:
        held = set(store.sessions())
        if args.tickers:
            todo_days = wanted
        else:
            todo_days = [d for d in wanted if d.isoformat() not in held]
        if not todo_days:
            log(f"Nothing to do: every session in {start}..{end} is already stored.")
            return 0
        log(f"Incremental: {len(todo_days)} of {len(wanted)} session(s) missing.")
    else:
        todo_days = wanted

    if not todo_days:
        log("No weekday sessions in the requested range.")
        return 0

    log(f"Target: {len(symbols)} ticker(s) x {len(todo_days)} session(s) "
        f"-> {args.root}")
    if args.dry_run:
        for d, e in _chunk_days(min(todo_days), max(todo_days), MAX_REQUEST_DAYS):
            log(f"  would fetch {d}..{e} in {args.batch_size}-symbol batches")
        return 0

    # ---- fetch and write -------------------------------------------------- #
    t0 = time.time()
    totals = {"written": 0, "unchanged": 0, "extended": 0, "revised": 0, "rejected": 0, "bars": 0}
    issues: list[str] = []
    # Accumulated across every date chunk below, because the fetch loop is per-chunk
    # and a symbol's bars may first appear in the second of them.
    fetched: set[str] = set()
    for chunk_start, chunk_end in _chunk_days(min(todo_days), max(todo_days), MAX_REQUEST_DAYS):
        days = [d for d in todo_days if chunk_start <= d <= chunk_end]
        if not days:
            continue
        log(f"\nFetching {days[0]}..{days[-1]} ({len(days)} session(s))")
        frames = _fetch(symbols, min(days), max(days), batch_size=args.batch_size, log=log)
        if not frames:
            log("  no bars returned for this range (weekend/holiday, or throttled)")
            continue
        # Only symbols that produced bars are candidates for registration: a symbol
        # absent from `frames` is one Yahoo had nothing for, and recording it would
        # be a claim the archive does not back up.
        fetched.update(frames)
        # One manifest rewrite per batch of sessions, not per ticker: see
        # `PanelStore.batched_manifest`.
        with store.batched_manifest():
            for sym, frame in sorted(frames.items()):
                frame = frame.copy()
                frame["ticker"] = sym
                res = store.write(sym, frame, policy=args.policy)
                totals["written"] += res.sessions_written
                totals["unchanged"] += res.sessions_unchanged
                totals["extended"] += res.sessions_extended
                totals["revised"] += res.sessions_revised
                totals["rejected"] += res.sessions_rejected
                totals["bars"] += res.bars_written
                issues.extend(f"{sym}: {m}" for m in res.issues)
            if res.fatal:
                log(f"  {sym}: REJECTED {len(res.fatal)} session(s): {res.fatal[:2]}")
            if res.sessions_written or res.sessions_unchanged or res.sessions_extended:
                log(f"  {sym}: +{res.sessions_written} new, "
                    f"{res.sessions_extended} extended, "
                    f"{res.sessions_unchanged} unchanged, {res.bars_written:,} bars")

    elapsed = time.time() - t0
    log("\n" + "=" * 62)
    log(f"Done in {elapsed:,.1f}s")
    log(f"  sessions written  {totals['written']:,}")
    log(f"  sessions extended {totals['extended']:,}   (still accumulating; bars merged)")
    log(f"  sessions unchanged{totals['unchanged']:,}")
    log(f"  sessions revised  {totals['revised']:,}   (content changed; stored copy kept)")
    log(f"  sessions rejected {totals['rejected']:,}   (failed the quality gate)")
    log(f"  bars written      {totals['bars']:,}")
    log(f"  archive now       {len(store.tickers())} ticker(s), "
        f"{len(store.sessions())} session(s)")

    # **Registered last, and from `fetched` rather than `symbols`.**  A symbol Yahoo
    # had nothing for is not a symbol the archive holds, and the registry exists to
    # say what is held.  `constituents` (when the universe was a scrape) merges in
    # again here so the GICS labels land alongside the new symbols in one write.
    _register(args.root, sorted(fetched), table=constituents, log=log)

    if issues:
        log(f"\n{len(issues)} quality note(s); first 10:")
        for m in issues[:10]:
            log(f"  - {m}")
    if totals["revised"]:
        log("\nRevised sessions kept their ORIGINAL bars. Review with --policy replace")
        log("if you intend to adopt yfinance's current adjustment history.")
    return 0


def _register(root: str, symbols: Sequence[str], *,
              table: Optional[pd.DataFrame] = None, log=print) -> Optional[dict]:
    """Add ``symbols`` to the archive's symbol registry, and report what changed.

    A registry write that fails must not take down a run that has already stored
    every bar it fetched: the registry is a derived index (the partition
    directories are the truth about what is held), and the failure the operator
    actually needs to see is the bar one.  So this logs and returns ``None``.

    ``log`` is a parameter because ``main``'s is a *closure* defined inside the
    function, so a module-level ``_register`` cannot reach it; ``download_daily``
    has a module-level ``log`` instead, which is why only this one needs the seam.
    """
    try:
        out = register_symbols(root, symbols, table=table)
    except Exception as exc:  # noqa: BLE001 - the bars are already written
        log(f"WARNING: could not update the symbol registry ({exc!r}); "
            f"the bars themselves are unaffected.")
        return None
    if out["added"]:
        log(f"registry: {len(out['added'])} new symbol(s) "
            f"({out['total']} recorded) e.g. {', '.join(out['added'][:10])}"
            f"{' ...' if len(out['added']) > 10 else ''}")
    else:
        log(f"registry: no new symbols ({out['total']} recorded)")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
