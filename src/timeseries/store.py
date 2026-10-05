"""Persistent bar archive for many tickers.  Implements PLAN.md §B.

Why this module exists
----------------------
The prototype in ``scripts/download_data.py`` wrote one CSV per ticker whose
filename embedded the request window, and it **restarted the download from scratch
on every run**.  Two consequences follow, and both are the reason §B says the database
is the product rather than the download:

* Yahoo only serves ~8 days of 1-minute bars per request (§B/§F), so an archive
  that only ever holds 8 days has ~3,000 candidate windows to rank a query against.
  §E's percentile is then computed over a population too small to mean anything.
* Re-downloading an overlapping range and blindly overwriting mixes bars fetched
  today with bars fetched last month.  yfinance retro-adjusts history, so those are
  not guaranteed to be the same numbers.  §B's response is a content fingerprint
  per session: a re-fetch that produces different content is recorded as a
  *revision*, not silently replaced.

Layout
------
``<root>/ticker=<SYMBOL>/date=<YYYY-MM-DD>/bars.parquet``

One directory per (ticker, session) makes the three operations that actually
matter cheap: re-fetching one session, checking whether a session changed, and
loading a single ticker's history.  The full table is a directory tree that DuckDB,
Polars, or pandas can read in one expression, so no catalog process is required --
which matters, because §B's DuckDB catalog was specified against a library that is
not installed and is not a dependency of this package.

The manifest
------------
``<root>/manifest.csv`` holds one row per (ticker, session): the bar count, the
session's opening/closing timestamps, the content fingerprint, and the revision
count.  It is a *cache* of what the partitions already contain, not the source of
truth: :func:`scan_partitions` reads the tree directly, and :func:`verify_manifest`
compares the two.  A manifest that has drifted from the filesystem is reported, not
trusted, because a store that silently believes a stale index is worse than no index.

The symbol registry
-------------------
``<root>/constituents.csv`` records which symbols this archive holds and, where it
is known, the GICS sector each belongs to.  It is written by :func:`register_symbols`,
which the downloaders call with the symbols a run *actually fetched* -- so a symbol
is in the registry because the archive holds its bars, not because it was asked for.

Two properties of that file are load-bearing, and both were learned the hard way:

* **It is cumulative.**  :func:`register_symbols` merges into whatever is already
  there and never truncates.  The original write was an unconditional overwrite of
  the scrape result, so the first ``--tickers BTC-USD`` run registered that symbol
  and the *next* ``--all`` run silently deleted it -- an archive holding the bars
  and an index that did not know they existed.
* **An unlabelable symbol says so.**  A symbol outside the index gets an explicit
  :data:`UNKNOWN_SECTOR` rather than a blank cell.  A blank is not equivalent:
  :func:`pandas.read_csv` parses an empty field as ``NaN``, and ``NaN`` is *truthy*,
  so a blank marker would have silently pooled every unlabelled symbol into a bogus
  "same sector" distribution.  :func:`read_sectors` drops the marker instead, at the
  one boundary where it is read.

Sessions
--------
Partitioning by *date* alone is wrong for US equities.  A trading session runs
09:30-16:00 America/New_York, which is 13:30-20:00 UTC in summer and 14:30-21:00
UTC in winter -- so a UTC date boundary cuts a session in half and produces two
partitions for one trading day.  Bucketing on the **Eastern** date is what makes
``date=`` mean "trading day".
"""

from __future__ import annotations

import hashlib
import io
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Iterable, Optional, Sequence
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .timeframes import DEFAULT_TIMEFRAME, resolve_timeframe

__all__ = [
    "CONSTITUENTS_NAME",
    "OHLCV",
    "PanelStore",
    "TickerBar",
    "SP500_URL",
    "UNKNOWN_SECTOR",
    "fetch_sp500_constituents",
    "constituents_path",
    "read_constituents",
    "read_sectors",
    "register_symbols",
    "yahoo_symbol",
    "session_bounds",
    "session_et",
    "stamp_label",
    "expected_bar_count",
    "fingerprint",
    "fingerprint_per_session",
    "validate_session",
    "scan_partitions",
    "verify_manifest",
    "MERGE_POLICY",
]

#: The bar columns this package reads and writes, in write order.  Volume is
#: deliberately absent: it is not a matching feature (:data:`timeseries.features.
#: FEATURE_COLUMNS`), and on some instruments Yahoo fills 70%+ of minute bars with a
#: zero, which made a volume check report noise rather than corruption.
OHLCV = ("open", "high", "low", "close")

#: How :meth:`PanelStore.write` treats a re-fetch whose fingerprint differs.
#:
#: ``"flag"``    keep the existing bars, count a revision, report the conflict.
#: ``"replace"`` overwrite the partition with the newly fetched content.
#: ``"error"``   refuse to write.
#:
#: The default is ``"flag"`` because §B's rule is to *flag* a revised day rather than
#: silently overwriting it.  Silent replacement is how an archive ends up mixing two
#: incompatible adjustment histories while every downstream number still looks sane.
MERGE_POLICY = "flag"

# Yahoo spells class shares with a hyphen (``BRK-B``) while the constituent list
# spells them with a dot (``BRK.B``).  Verified against the live API: ``BRK-B``
# returns bars and ``BRK.B`` returns "No data found, symbol may be delisted".
# Class-share tickers are the only symbols where the two differ.
_DOT_TO_DASH = str.maketrans({".": "-"})

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Regular trading hours.  09:30-16:00 ET, i.e. 390 one-minute bars, the count §B
# uses to decide whether a session is complete.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

EASTERN = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Symbols and sessions
# --------------------------------------------------------------------------- #
def yahoo_symbol(sym: str) -> str:
    """Normalise any spelling to the one Yahoo accepts (``BRK.B`` -> ``BRK-B``)."""
    return str(sym).strip().upper().translate(_DOT_TO_DASH)


def display_symbol(sym: str) -> str:
    """Normalise to the constituent-list spelling (``BRK-B`` -> ``BRK.B``)."""
    return str(sym).strip().upper().replace("-", ".")


def session_et(ts: pd.Series) -> pd.Series:
    """The Eastern calendar date a UTC timestamp belongs to.

    Bars are stored UTC; sessions are *named* by their Eastern date.  A 09:30 ET
    open is 13:30 or 14:30 UTC depending on daylight saving, so grouping on the UTC
    date would file a winter session's morning bars under the previous UTC day.
    """
    idx = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
    return idx.dt.tz_convert(EASTERN).dt.floor("D")


def stamp_label(ts: object, tf: object = None) -> str:
    """One bar's stamp as a reader should read it, at this bar's resolution.

    ``2026-09-03 09:30`` intraday, ``2026-09-03`` daily.  **One definition, because
    ``fetch.py``'s summary line and the app's chart axes must not be able to disagree**
    about what a bar looks like -- a reader comparing the two is exactly how a
    resolution mismatch gets noticed, and only if the two are printed the same way.

    Daily drops the time for the same reason :func:`timeseries.timeframes` drops the
    hole threshold: the value cannot vary.  Every daily bar in a real archive carries
    the same 09:30 stamp, so a clock on a daily chart is either one constant repeated
    along the axis or an implication that the bars are spread across a day they are
    not.  The date is the **Eastern** one (:func:`session_et`), which is what makes it
    agree with the ``session`` column the match table reports.

    ``tf`` accepts anything :func:`~timeseries.timeframes.resolve_timeframe` does;
    the default is the package default rather than a guess about the caller.
    """
    tf = resolve_timeframe(tf) if tf is not None else resolve_timeframe(None)
    if ts is None or ts is pd.NaT:
        return "—"
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        return "—"
    if tf.bars_per_session == 1:
        return session_et(pd.Series([stamp])).iloc[0].strftime("%Y-%m-%d")
    return stamp.strftime("%Y-%m-%d %H:%M")


def session_bounds(day: date) -> tuple[datetime, datetime]:
    """UTC ``[start, end)`` covering one Eastern trading session.

    Built by localising 09:30/16:00 *Eastern* on the given Eastern date and
    converting to UTC, so DST is handled by the timezone database rather than by a
    hand-written offset.  The end is exclusive: a bar stamped exactly 16:00 is
    outside the session, which is what makes a stray after-hours print detectable.
    """
    d = pd.Timestamp(day).date()
    start = pd.Timestamp(datetime.combine(d, MARKET_OPEN), tz=EASTERN).tz_convert("UTC")
    end = pd.Timestamp(datetime.combine(d, MARKET_CLOSE), tz=EASTERN).tz_convert("UTC")
    return start.to_pydatetime(), end.to_pydatetime()


def _session_settled_after(day: date) -> pd.Timestamp:
    """UTC timestamp of the session's final bar, past which the session is settled.

    A session's bars run 09:30 through 15:59 ET, so a complete 390-bar session's
    last timestamp is ``close - 1 minute``.  Comparing the observed last bar against
    that boundary settles both cases exactly:

    * a complete session's last bar *equals* the boundary, so it is settled;
    * an in-progress session's last bar is strictly earlier, so it is not.

    Comparing against the close itself instead classifies a complete session as "in
    progress" forever, which silently disables the whole revision check: a genuine
    retro-adjustment of settled history would be recorded as normal accumulation on
    every single sync.
    """
    _open, close = session_bounds(day)
    return pd.Timestamp(close) - pd.Timedelta(minutes=1)


def expected_bar_count(day: date, *, close_time: Optional[time] = None,
                       timeframe: object = DEFAULT_TIMEFRAME) -> int:
    """Bars a complete session should contain: 390, 210 on a 13:00 half day, or 1 daily.

    A half-day close is 3.5 hours rather than 6.5, so it holds 210 bars.  Rather than
    hard-code a holiday calendar, the default count is derived from the actual close
    time and :func:`validate_session` reports a shortfall instead of hiding it.  A
    call that knows the published early-close time passes ``close_time``; an early
    close nobody told the store about reads as a "short session" -- reported, never
    silently accepted.

    On **daily** the answer is 1 on a weekday and 0 on a weekend, and ``close_time`` is
    ignored: a daily bar covers the whole session, so there is no window for an early
    close to shorten and no hour count to derive.  The weekday rule still applies,
    because a missing weekend is a real absence rather than a tolerance.

    ``timeframe`` defaults to 1-minute, so every existing caller and test keeps its
    current answer without having to say so.
    """
    tf = resolve_timeframe(timeframe)
    if pd.Timestamp(day).dayofweek >= 5:
        return 0  # weekend: no session, so 0 expected is correct, not a hole
    if tf.bars_per_session == 1:
        # One bar *is* one trading day; there is nothing to derive from a close time.
        return 1
    close = close_time or MARKET_CLOSE
    minutes = (close.hour * 60 + close.minute) - (MARKET_OPEN.hour * 60 + MARKET_OPEN.minute)
    return max(0, int(minutes))


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #
def fingerprint(frame: pd.DataFrame) -> str:
    """Content hash of a session's bars, stable across column and row order.

    §B requires a per-session fingerprint so a re-fetch can be recognised as *the
    same data* or as *a revision*.  The hash covers every OHLCV value at full float
    precision plus the session's timestamps, and is taken over a canonically ordered,
    canonically formatted copy:

    * rows sorted by timestamp, so fetch order cannot change the hash;
    * columns fixed to :data:`OHLCV` order, so column order cannot change it;
    * values written with ``%.17g``, which round-trips IEEE-754 doubles exactly, so
      the hash compares the numbers rather than their shortest repr.

    Rounding to 8 decimals -- the obvious cheaper choice -- would merge bars that
    differ in the 9th decimal, which is inside a single tick for most S&P 500 names.
    That is precisely the kind of silent merge §B's fingerprint exists to prevent.
    """
    if frame is None or len(frame) == 0:
        return "empty"
    cols = [c for c in OHLCV if c in frame.columns]
    if "timestamp" in frame.columns:
        sort_by = ["timestamp"]
    else:
        sort_by = []
    out = frame.sort_values(sort_by) if sort_by else frame
    buf = io.StringIO()
    for c in ("timestamp", *cols):
        if c in out.columns:
            buf.write(c)
            buf.write("\x1f")
            v = out[c]
            if c == "timestamp":
                buf.write(_format_timestamps(v))
            else:
                buf.write(_format_column(v))
            buf.write("\x1e")
    return hashlib.blake2b(buf.getvalue().encode("utf-8"), digest_size=16).hexdigest()


def fingerprint_per_session(frame: pd.DataFrame, sessions) -> dict:
    """``{session: fingerprint}`` for every session, without splitting the frame.

    :func:`fingerprint` on a daily ticker is called once per *day of history*, and
    each of those calls is a one-bar frame.  AAPL's history is 11,544 sessions, so
    the archive pays 11,544 pandas ``groupby`` slices and 11,544 fixed overheads to
    hash a frame whose whole-file :func:`fingerprint` takes 0.125s.  Measured on
    that frame: 4.33s for the per-session loop against 0.125s for the whole thing.

    This is the same hash, not a cheaper one.  Every session is rendered by the same
    :func:`_format_timestamps` / :func:`_format_column` pair, so the bytes hashed
    per session are byte-identical to the scalar path's -- which is the only thing
    that makes this safe to substitute, since a fingerprint that hashed different
    bytes would turn every stored session into a "revision" on the next run.  The
    equivalence is asserted in ``tests/test_daily_archive.py``.

    Sessions are formatted as one block each rather than per row, so the per-call
    overhead is paid once per *column* instead of once per session.
    """
    if frame is None or len(frame) == 0:
        return {}
    cols = [c for c in OHLCV if c in frame.columns]
    ordered = frame.sort_values(["timestamp"]) if "timestamp" in frame.columns else frame
    if "session" not in ordered.columns:
        return {str(s): fingerprint(ordered) for s in sessions}

    keys = ordered["session"].astype(str).to_numpy()
    stamps = _format_timestamps(ordered["timestamp"]) if "timestamp" in ordered.columns else None
    # Column text is formatted once for the whole frame and split into a list ONCE.
    # Splitting it inside the per-session loop would re-scan an 11k-element string
    # 11k times -- measured at 14.9s for AAPL, four times *slower* than the scalar
    # loop it replaces, because the split is O(n) and the loop makes it O(n^2).
    stamp_list = stamps.split("\n") if stamps else None
    col_list = {c: _format_column(ordered[c]).split("\n") for c in cols}

    out: dict[str, str] = {}
    # Sorted unique sessions, so group boundaries can be found with a scan instead
    # of a ``groupby`` (which materialises a sub-frame per key).
    uniq = pd.unique(ordered["session"].astype(str))
    pos = 0
    n = len(keys)
    for day in uniq:
        end = pos
        while end < n and keys[end] == day:
            end += 1
        buf = io.StringIO()
        if stamp_list is not None:
            buf.write("timestamp"); buf.write("\x1f")
            buf.write("\n".join(stamp_list[pos:end])); buf.write("\x1e")
        for c in cols:
            buf.write(c); buf.write("\x1f")
            buf.write("\n".join(col_list[c][pos:end])); buf.write("\x1e")
        out[str(day)] = hashlib.blake2b(
            buf.getvalue().encode("utf-8"), digest_size=16).hexdigest()
        pos = end
    return out


def _format_timestamps(values) -> str:
    """UTC ISO-8601 with microseconds, matching ``.dt.strftime`` exactly.

    The obvious fast path -- assuming the column is already datetime and skipping
    the ``utc=True`` normalisation -- is wrong for exactly the data that matters
    most: daily bars are stamped in **Eastern** local time, so a naive path would
    hash 09:30 as 09:30 where the original hashes 13:30.  Every session of a
    daily ticker would then fingerprint differently from the manifest written by
    an earlier run, and the whole archive would present as revised.

    So the offset is *computed* here rather than assumed, and the cost of doing
    so honestly is paid once: :func:`fingerprint` is called per session, and on
    daily a session is a single bar, so pandas' per-call overhead dominated the
    formatting by ~10x.
    """
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    if not isinstance(ts, pd.Series):  # pragma: no cover - defensive
        ts = pd.Series(ts)
    if len(ts) == 0:
        return ""
    return "\n".join(ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").fillna("").tolist())


def _format_column(values: pd.Series) -> str:
    """One column, one ``\\x1e``-terminated block, formatted at ``%.17g``.

    Formatting is vectorised because it is the single most expensive step in a
    fingerprint, and on daily it runs once per *session* rather than once per file:
    AAPL's full history is 11,544 of these.  Measured with :mod:`cProfile`, the
    Python-level ``Series.map`` cost ~33s of a 36s single-ticker write.

    ``np.char.mod("%.17g", arr)`` was checked to be **byte-identical** to the scalar
    ``"%.17g" % x`` it replaces -- over the archive's real prices and over
    subnormals, ``-0.0``, infinities and NaN -- because a faster fingerprint that
    hashes different bytes would silently change every stored fingerprint and turn
    the whole archive into a revision.  That equivalence is what makes the
    substitution safe; it is asserted in ``tests/test_daily_archive.py``.

    NaN needs no special case here: ``errors="coerce"`` maps anything unparseable to
    NaN, and ``%.17g`` renders NaN as ``"nan"`` -- the same text the scalar path
    produced via ``.fillna("nan")``.  A value that was always NaN and a value that
    only became NaN therefore hash identically, as they must.
    """
    arr = _numeric_array(values)
    if arr.size == 0:
        return ""
    return "\n".join(np.char.mod("%.17g", arr).tolist())


def _numeric_array(values) -> np.ndarray:
    """A float64 array from a column, without paying pandas' per-call overhead.

    On daily a *session* is one bar, so ``fingerprint`` is called once per day of
    history and every fixed cost is multiplied by the length of the archive.  The
    formatting itself is ~3 microseconds; wrapping it in four ``Series`` methods
    costs ~10 each.  So the conversion goes straight to the underlying array, and
    the coercion contract -- anything unparseable becomes NaN -- is preserved
    explicitly rather than delegated to :func:`pandas.to_numeric`.
    """
    if isinstance(values, pd.Series):
        if values.dtype.kind in "fiu":
            return values.to_numpy(dtype=float, na_value=np.nan)
        values = values.to_numpy()
    else:
        values = np.asarray(values)
    if values.dtype.kind == "f":
        return values
    if values.dtype.kind in "iub":
        return values.astype(float)
    # Object or string: coerce, mapping what will not parse to NaN as before.
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
def validate_session(frame: pd.DataFrame, day, *, timeframe: object = DEFAULT_TIMEFRAME) -> dict:
    """Run §B's per-session quality gate.  Reports, never silently repairs.

    Each check is a *reported* condition rather than an exception.  A session with a
    few missing prints is normal and should be stored with a note; a session that is
    empty, unparseable, or full of non-positive prices is a fetch failure and must
    not be written at all.  :attr:`PanelStore.write` gates on :attr:`Report.fatal`
    only.

    The three checks that encode an *intraday* structure are skipped on daily, and the
    reasons are not all the same:

    * **Regular hours.**  This is the one that genuinely breaks.  A daily bar carries
      whatever timestamp Yahoo chose to stamp it with, so applying the 09:30-16:00 ET
      test flags every bar.  Measured: one daily bar at 09:30 ET still reports
      ``['1 bars before 09:30 ET']`` under the intraday rules, because Yahoo stamps
      daily bars at the session *open* and the comparison is exclusive of it.
    * **Intra-session holes.**  Skipped because the threshold is meaningless at this
      resolution, *not* because it fires.  Measured: a 5-minute hole on one day is
      reported intraday and silently dropped daily.  On real daily data the intraday
      count would in fact come out 0 anyway (every bar is its own Eastern date, so no
      delta is intra-session), but a *multi-bar-per-day* frame would not be -- and
      "one bar is a day" is an assumption the validator should not be relying on.
    * **Session arity.**  "390 bars" is an intraday statement; on daily it is 1, which
      :func:`expected_bar_count` already knows.  This is the check that most visibly
      matters: measured, the same single bar is ``short session: 1/390 bars`` intraday
      and clean daily.

    Nothing here replaces the hole check, because this function is given **one day**
    and therefore cannot see whether a *neighbouring* day is missing -- a gap in the
    daily series is only visible to something holding the whole frame.  That check
    lives in :func:`timeseries.features.quality_report`, under ``missing_days``.
    """
    tf = resolve_timeframe(timeframe)
    intraday = tf.gap_seconds is not None

    issues: list[str] = []
    fatal: list[str] = []
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    n = len(frame)
    expected = expected_bar_count(day, timeframe=tf.key)

    if n == 0:
        fatal.append("no rows")
        return {"day": str(day), "ticker": None, "rows": 0, "expected": expected,
                "issues": ["no rows"], "fatal": ["no rows"], "ok": False,
                "timeframe": tf.key}

    bad_ts = int(ts.isna().sum())
    if bad_ts:
        fatal.append(f"{bad_ts} unparseable timestamps")

    close = pd.to_numeric(frame.get("close"), errors="coerce")
    n_nonpos = int((close <= 0).sum())
    if n_nonpos:
        fatal.append(f"{n_nonpos} non-positive close")
    n_nan = int(close.isna().sum())
    if n_nan:
        fatal.append(f"{n_nan} missing close")

    dups = int(ts.duplicated().sum())
    if dups:
        issues.append(f"{dups} duplicate timestamps")

    if intraday:
        # Bars outside regular hours.  Counted against the Eastern window, not a UTC
        # one.
        et = ts.dt.tz_convert("America/New_York")
        too_early = int((et.dt.time < MARKET_OPEN).sum())
        too_late = int((et.dt.time > MARKET_CLOSE).sum())
        if too_early:
            issues.append(f"{too_early} bars before 09:30 ET")
        if too_late:
            issues.append(f"{too_late} bars after 16:00 ET")

        # Intra-session holes only.  An overnight jump is the market closing, not a
        # gap; counting both as gaps makes a clean archive look corrupt (§BF).
        if bad_ts == 0:
            ordered = ts.sort_values()
            deltas = ordered.diff().dt.total_seconds()
            holes = int((deltas > tf.gap_seconds).sum() - (deltas > 3600).sum())
            if holes:
                issues.append(f"{holes} intra-session holes > {tf.gap_seconds}s")

    if expected and n < expected:
        label = "bar" if tf.bars_per_session == 1 else "bars"
        issues.append(f"short session: {n}/{expected} {label}")

    return {
        "day": str(day),
        "ticker": None,
        "rows": int(n),
        "expected": int(expected),
        "issues": issues,
        "fatal": fatal,
        "ok": not fatal,
        "timeframe": tf.key,
    }


# --------------------------------------------------------------------------- #
# Constituents
# --------------------------------------------------------------------------- #
def fetch_sp500_constituents(*, url: str = SP500_URL) -> pd.DataFrame:
    """Fetch the current S&P 500 constituent list with its sector labels.

    The sector column is not decoration: it is what lets a cross-sectional search
    report "matches in the same GICS sector" separately from all matches, which is
    the difference between a shape coincidence and a sector-wide move.

    A Wikipedia scrape is a weaker dependency than a market-data feed, and it is a
    deliberate trade: it costs no key, no quota and no vendored copy of the index
    that silently ages.  The failure mode is a layout change, which surfaces as a
    missing column and an explicit error rather than an empty download.
    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; timeseries/0.2)"})
    html = urlopen(req, timeout=30).read()  # noqa: S310 - fixed https URL from the module
    tables = pd.read_html(io.BytesIO(html))
    if not tables:
        raise ValueError(f"no tables parsed from {url}")
    df = tables[0]
    required = {"Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"constituent table from {url} is missing column(s) {sorted(missing)}; "
            f"got {list(df.columns)} -- the page layout has probably changed"
        )
    out = df.rename(columns={
        "Symbol": "symbol",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    keep = [c for c in ("symbol", "name", "sector", "sub_industry") if c in out.columns]
    out = out[keep].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["yahoo_symbol"] = out["symbol"].map(yahoo_symbol)
    out = out.loc[out["symbol"].str.len().gt(0)].drop_duplicates("symbol").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# The symbol registry
# --------------------------------------------------------------------------- #
#: Name of the symbol-registry sidecar, beside the archive root.  Defined once
#: because three call sites spelled it out separately -- ``app.py``'s
#: ``PANEL_SECTOR_NAME`` and a bare ``os.path.join`` in each downloader -- and a
#: sidecar the reader cannot find under the name the code expects is invisible
#: rather than broken: every consumer treats a missing file as "no labels".
CONSTITUENTS_NAME = "constituents.csv"

#: Sector recorded for a symbol no constituent list can describe.
#:
#: Deliberately a *word* and not an empty cell.  ``pd.read_csv`` reads an empty
#: field as ``NaN``, and ``float('nan')`` is truthy -- so a blank sector would pass
#: every truthiness guard in :mod:`timeseries.panel` and pool every unlabelled
#: symbol into one fabricated "same sector" distribution.  This string is stored
#: verbatim so the gap is legible to anyone reading the CSV, and :func:`read_sectors`
#: drops it at the single boundary where the registry is read.
UNKNOWN_SECTOR = "unknown"

#: The registry's columns, in write order.  Fixed rather than derived from the
#: incoming table so a hand-edited or partial file still merges into a known shape.
_CONSTITUENT_COLUMNS = ("symbol", "name", "sector", "sub_industry", "yahoo_symbol")

#: Values that look like symbols but are not.  Checked case-insensitively against the
#: raw text *before* normalisation, so a null in a scraped table is dropped instead of
#: being stringified into the phantom instrument ``"NONE"``.
_NOT_A_SYMBOL = frozenset({"none", "nan", "null", "na", "n/a"})


def constituents_path(root: str) -> str:
    """Where the symbol registry for the archive at ``root`` lives."""
    return os.path.join(os.path.abspath(root), CONSTITUENTS_NAME)


def read_constituents(path: str) -> pd.DataFrame:
    """The symbol registry at ``path``, or an empty frame with its canonical columns.

    Returns an empty frame rather than ``None`` for a missing or unreadable file, so
    callers can index a column without first branching on absence.  The registry is a
    *derived* index in the same sense the manifest is -- the partition directories are
    the truth about which symbols are held -- so a missing one costs labels, not data.

    **Takes a path, while :func:`register_symbols` takes a root.**  The readers answer
    "what is in this file", and a caller that already holds a path (``app.py``'s
    ``panel_sector_file()``) should not have to unwind it back to a directory just to
    rejoin it.  Use :func:`constituents_path` to turn a root into one.

    ``keep_default_na=False`` is load-bearing rather than defensive: without it the
    literal ``"unknown"`` would survive (it does) but any *blank* cell in a
    hand-edited file would become a truthy ``NaN`` and poison a sector pool.  See
    :data:`UNKNOWN_SECTOR`.
    """
    empty = pd.DataFrame(columns=list(_CONSTITUENT_COLUMNS))
    if not path or not os.path.isfile(path):
        return empty
    try:
        out = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001 - a bad sidecar costs labels, never a crash
        return empty
    for col in _CONSTITUENT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[list(_CONSTITUENT_COLUMNS)]


def read_sectors(path: str) -> dict[str, str]:
    """``{yahoo_symbol: sector}`` from the registry at ``path``, unlabelled dropped.

    **The single boundary where :data:`UNKNOWN_SECTOR` is handled.**  Everything
    downstream -- :mod:`timeseries.panel`'s same-sector percentile above all -- may
    assume every value here is a real GICS sector, because that is what it gets.
    This is why the marker does not need to be understood anywhere else: the panel
    gates its sector pool on a truthy ``sectors.get(home_ticker)``, so a symbol
    absent from this dict reports *no* same-sector percentile, which is the honest
    answer, rather than a meaningless one computed over a pool of other unknowns.

    Never raises.  A missing, malformed or unreadable registry degrades to ``{}`` --
    the behaviour every consumer had before this existed, and the one that keeps a
    cosmetic index from taking down a search over bars that are perfectly readable.
    """
    try:
        df = read_constituents(path)
    except Exception:  # noqa: BLE001
        return {}
    if df.empty:
        return {}
    sym = df["yahoo_symbol"].astype(str).str.strip()
    sec = df["sector"].astype(str).str.strip()
    # Empty, ``NaN``-in-disguise, and the marker all mean "no label", and all three
    # are dropped here rather than in the panel.
    keep = sec.str.len().gt(0) & ~sec.str.lower().eq(UNKNOWN_SECTOR)
    return dict(zip(sym[keep], sec[keep]))


def register_symbols(root: str, symbols: Iterable[str], *,
                     table: Optional[pd.DataFrame] = None) -> dict:
    """Record ``symbols`` in the registry for ``root``, merging with what is there.

    Called by the downloaders with the symbols a run **actually fetched** -- not the
    universe it asked for.  That distinction is the whole contract: a typo, a
    delisted ticker or an index Yahoo does not serve produces zero bars, and a
    registry listing it would be a claim the archive does not back up.

    Parameters
    ----------
    symbols
        Any symbol spellings.  Normalised through :func:`yahoo_symbol`, so ``brk.b``
        and ``BRK-B`` are one row rather than two, and case is folded.
    table
        A constituent table from :func:`fetch_sp500_constituents`, when the run
        scraped the index.  Its GICS labels are merged in and win over
        :data:`UNKNOWN_SECTOR`.

    Notes
    -----
    **Merge, never overwrite.**  An unconditional write of ``table`` would delete
    every non-constituent on the next ``--all`` run -- bars on disk, index blind to
    them.  So existing rows are kept, and the dedup is ``keep="last"`` on the new
    frame, which is what makes a later scrape *upgrade* an ``unknown`` while a bare
    register (no ``table``) can never downgrade a real label.

    The write is atomic for the same reason the manifest's is: ``to_csv`` truncates
    its target before it streams, so a reader can see a short registry.
    """
    wanted: list[str] = []
    for raw in symbols:
        # ``yahoo_symbol`` stringifies whatever it is given, so a null in a scraped
        # table became the literal ``"NONE"`` -- a phantom instrument the archive does
        # not hold, registered as if it did.  Anything that is not a real ticker is
        # dropped here, at the one boundary where the registry decides what a symbol
        # is.
        text = str(raw).strip() if raw is not None else ""
        if not text or text.lower() in _NOT_A_SYMBOL:
            continue
        wanted.append(yahoo_symbol(text))
    seen: set[str] = set()
    wanted = [s for s in wanted if not (s in seen or seen.add(s))]
    if not wanted and table is None:
        return {"path": constituents_path(root), "total": 0, "added": []}

    fresh = pd.DataFrame([{"yahoo_symbol": s} for s in wanted])
    for col in _CONSTITUENT_COLUMNS:
        if col not in fresh.columns:
            fresh[col] = ""
    fresh = fresh[list(_CONSTITUENT_COLUMNS)]

    existing = read_constituents(constituents_path(root))
    before = set(existing["yahoo_symbol"]) if len(existing) else set()
    if len(fresh):
        # **Only symbols the registry does not already hold get a placeholder row.**
        # Emitting a row for *every* requested symbol would put ``unknown`` on disk
        # for anything already labelled, and the ``keep="last"`` dedup below would
        # then let that placeholder win -- so re-running the downloader would quietly
        # strip the GICS sectors off every symbol it had already learned.  Measured on
        # the merge this guards, after two calls:
        #
        #     call 1 (no table)  -> AAPL = unknown
        #     call 2 (table)     -> AAPL = Information Technology
        #     call 3 (no table)  -> AAPL = unknown        <- the bug: downgraded
        #
        # ``table`` rows are deliberately *not* subject to this filter: upgrading an
        # ``unknown`` to a real sector is the one overwrite this file allows.
        fresh = fresh.loc[~fresh["yahoo_symbol"].isin(before)]
        if len(fresh):
            # The display spelling is the Yahoo one: for class shares that is the only
            # form the archive is filed under, and the registry is read by Yahoo symbol.
            fresh["symbol"] = fresh["yahoo_symbol"]
            fresh["name"] = fresh["yahoo_symbol"]
            fresh["sector"] = UNKNOWN_SECTOR

    parts = [df for df in (existing, fresh) if len(df)]
    merged = pd.concat(parts, ignore_index=True, sort=False)
    if table is not None and len(table):
        labelled = table.copy()
        for col in _CONSTITUENT_COLUMNS:
            if col not in labelled.columns:
                labelled[col] = ""
        merged = pd.concat([merged, labelled[list(_CONSTITUENT_COLUMNS)]],
                           ignore_index=True, sort=False)

    merged["yahoo_symbol"] = merged["yahoo_symbol"].astype(str).str.strip()
    merged = merged.loc[merged["yahoo_symbol"].str.len().gt(0)]
    # Last write wins, and ``fresh`` and ``table`` both come after ``existing`` -- so
    # a genuine sector label displaces an ``unknown`` and nothing displaces a label.
    merged = merged.drop_duplicates(subset=["yahoo_symbol"], keep="last")
    merged = merged.sort_values("yahoo_symbol", kind="stable").reset_index(drop=True)

    path = constituents_path(root)
    _write_csv_atomic(merged[list(_CONSTITUENT_COLUMNS)], path)
    return {
        "path": path,
        "total": int(len(merged)),
        "added": sorted(set(wanted) - before),
    }


def _write_csv_atomic(frame: pd.DataFrame, path: str) -> None:
    """Write ``frame`` to ``path`` via a temp file and ``os.replace``.

    The same exposure :meth:`PanelStore._write_manifest_atomic` closes, for the same
    reason: ``DataFrame.to_csv`` **truncates the target before it streams**, so
    writing in place leaves a window in which the live file is absent or short.  The
    manifest's window is measured in seconds because the file is ~567MB; the
    registry's is short but its failure mode is identical and just as confusing --
    a reader sees a registry with symbols missing from it, which reads as "never
    registered" rather than "truncated mid-write".

    The temp file is created in the destination directory so ``os.replace`` stays on
    one filesystem, which is what makes it atomic rather than a copy.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        # A crash between write and replace leaves the temp file behind.  It is never
        # the real file, so removing it is always safe, and after a successful
        # replace this is a no-op because the temp name no longer exists.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:  # pragma: no cover - best effort cleanup
                pass


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def _iter_partition_files(root: str) -> Iterable[tuple[str, Optional[str], str]]:
    """Yield ``(ticker, session_or_None, path)`` for every partition in ``root``.

    ``session`` is ``None`` for a whole-ticker file (daily), where the date lives in a
    column instead of the path.

    **One walker for both readers**, because :func:`scan_partitions` and
    :meth:`PanelStore.rebuild_manifest` must agree on what a partition *is*.  When
    they walked the directory separately -- which is how it started -- the daily
    layout had to be taught twice, and teaching it once would have been enough:
    a reader that missed the flat file would report an archive as empty while the
    other reported it as full.
    """
    if not os.path.isdir(root):
        return
    for tdir in sorted(os.listdir(root)):
        if not tdir.startswith("ticker="):
            continue
        sym = tdir.split("=", 1)[1].upper()
        tpath = os.path.join(root, tdir)
        if not os.path.isdir(tpath):
            continue
        flat = os.path.join(tpath, "bars.parquet")
        if os.path.isfile(flat):
            yield sym, None, flat
            continue
        for ddir in sorted(os.listdir(tpath)):
            if not ddir.startswith("date="):
                continue
            path = os.path.join(tpath, ddir, "bars.parquet")
            if os.path.isfile(path):
                yield sym, ddir.split("=", 1)[1], path


def scan_partitions(root: str, *, tickers: Optional[Sequence[str]] = None,
                    since: Optional[str] = None, until: Optional[str] = None) -> pd.DataFrame:
    """Read the archive back as one frame, from the filesystem, ignoring the manifest.

    ``root/<ticker=SYM>/date=YYYY-MM-DD>/bars.parquet``

    The directory names are parsed, not the contents: ``ticker=`` and ``date=`` are
    the authoritative record of what a partition holds, so a row's symbol and session
    come from its path.  That keeps the returned frame self-describing even if a
    partition's columns were written by an older version.
    """
    if not os.path.isdir(root):
        return pd.DataFrame(columns=["timestamp", "ticker", "session", *OHLCV])

    want = {str(t).upper() for t in tickers} if tickers else None
    frames: list[pd.DataFrame] = []
    for sym, day, path in _iter_partition_files(root):
        if want is not None and sym not in want:
            continue
        if day is not None:
            if since and day < since:
                continue
            if until and day > until:
                continue
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - an unreadable partition is skipped, not fatal
            continue
        if df.empty:
            continue
        if day is None and (since or until):
            # A whole-ticker file cannot be skipped by directory name, so the window
            # is applied to the session column instead.  It reads more than needed
            # and costs nothing at this size.
            if "session" not in df.columns:
                continue
            sess = df["session"].astype(str)
            if since:
                df = df.loc[sess >= since]
            if until:
                df = df.loc[sess <= until]
            if len(df) == 0:
                continue
        # The directory names are authoritative for a per-session partition; for a
        # whole-ticker file the columns are.  Both are filled in so the returned frame
        # is self-describing either way.
        if "ticker" not in df.columns or df["ticker"].isna().all():
            df["ticker"] = sym
        if "session" not in df.columns and day is not None:
            df["session"] = day
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "ticker", "session", *OHLCV])

    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in OHLCV:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "ticker" not in out.columns:
        out["ticker"] = pd.NA
    if "session" not in out.columns:
        out["session"] = out["ticker"].map(lambda s: pd.NaT)
    out = out.loc[out["timestamp"].notna()].sort_values(["ticker", "timestamp"])
    return out.reset_index(drop=True)


def verify_manifest(root: str) -> dict:
    """Compare the manifest against the partitions on disk.

    The manifest is a derived index (:mod:`store` never requires it to read data), so
    a disagreement is reported rather than resolved.  A store that trusts a stale
    index will report sessions it no longer has and skip sessions it does.
    """
    mpath = os.path.join(root, "manifest.csv")
    # The same walker the readers use, so "what is on disk" means one thing here too.
    # A whole-ticker file contributes **one entry per stored session**, because that is
    # what the manifest records for daily -- counting files instead would report a
    # 503-file archive as 503 sessions against a manifest of 6,000 rows, and the
    # "inconsistency" would be an artefact of counting rather than a real one.
    actual: set[tuple[str, str]] = set()
    for sym, day, path in _iter_partition_files(root):
        if day is not None:
            actual.add((sym, day))
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        if "session" not in df.columns:
            # A whole-ticker file with no session column cannot be indexed, and
            # **silently skipping it would make a corrupt archive look intact** --
            # which is the one thing a consistency check exists to prevent.  Counted
            # under a sentinel so it shows up as "on disk but not in the manifest".
            actual.add((sym, "<no session column>"))
            continue
        for one in df["session"].astype(str).unique():
            actual.add((sym, str(one)))

    if not os.path.isfile(mpath):
        return {"n_partitions": len(actual), "n_manifest": 0,
                "missing_from_manifest": sorted(actual), "missing_from_disk": [],
                "consistent": not actual}

    man = pd.read_csv(mpath, dtype={"ticker": str})
    man["ticker"] = man["ticker"].str.upper()
    recorded = set(zip(man["ticker"], man["session"].astype(str)))
    return {
        "n_partitions": len(actual),
        "n_manifest": len(recorded),
        "missing_from_manifest": sorted(actual - recorded),
        "missing_from_disk": sorted(recorded - actual),
        "consistent": (actual - recorded) == set() and (recorded - actual) == set(),
    }


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
@dataclass
class TickerBar:
    """Result of writing one ticker's bars, including what was rejected."""

    ticker: str
    sessions_written: int = 0
    sessions_unchanged: int = 0
    sessions_extended: int = 0
    sessions_revised: int = 0
    sessions_rejected: int = 0
    bars_written: int = 0
    issues: list = field(default_factory=list)
    fatal: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal and not self.errors

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker, "sessions_written": self.sessions_written,
            "sessions_unchanged": self.sessions_unchanged,
            "sessions_extended": self.sessions_extended,
            "sessions_revised": self.sessions_revised,
            "sessions_rejected": self.sessions_rejected, "bars_written": self.bars_written,
            "n_issues": len(self.issues), "n_errors": len(self.errors),
        }


class PanelStore:
    """A partitioned Parquet archive of 1-minute bars, keyed by (ticker, session)."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.manifest_path = os.path.join(self.root, "manifest.csv")
        # Read lazily; see ``_manifest_or_none``.  Eagerly parsing a ~570MB CSV
        # on every app start was measured at ~6s of dead time.
        self._manifest: Optional[pd.DataFrame] = None
        self._manifest_read = False
        # Non-None only inside `batched_manifest`; see there for why.
        self._manifest_batch: Optional[list[dict]] = None

    # -- manifest batching ------------------------------------------------- #
    @contextmanager
    def batched_manifest(self):
        """Coalesce manifest writes across many :meth:`write` calls.

        **Why this exists.**  Every :meth:`write` rewrites the *whole* manifest,
        because the manifest is an index and an index has to be written as a unit.
        That is fine for a handful of tickers and ruinous for a full build: the daily
        manifest holds one row per session (~1.3M rows, ~176MB), and measured on this
        archive a single ``_write_manifest`` call costs **5.1s**, of which 2.4s is
        the ``to_csv`` of the full frame.  Written once per ticker, 503 tickers is
        503 full rewrites -- and it does not finish, because each rewrite grows the
        manifest for the next one.

        This is not a slow build, it is a build that never completes: the measured
        run sat at 25 minutes of CPU with **zero** new partitions on disk, which is
        what sent me looking.  Batch-flushing turns 503 rewrites into one.

        **What is deferred is only the rewrite.**  Rows are still produced per write,
        and the bars are still on disk; the manifest simply catches up at the end,
        exactly as it would have between two calls.  :meth:`flush_manifest` is called
        on the way out **including on an exception**, so an interrupted batch leaves
        the rows written so far indexed rather than lost -- which is the one thing the
        interrupted build this fixes actually needed.

        The store is left consistent either way: :meth:`write` never reads a row it
        just wrote (daily compares against the Parquet file, intraday against
        ``existing_fingerprint``), so deferring cannot change a decision.
        """
        if self._manifest_batch is not None:
            # Already batching -- inner use must not flush the outer batch's rows.
            yield self
            return
        self._manifest_batch = []
        try:
            yield self
        finally:
            # ``flush_manifest`` clears the buffer itself.  Clearing it *here* first
            # would hand ``flush_manifest`` an empty list and silently drop every
            # deferred row -- which is the one thing this is here to prevent.
            self.flush_manifest()

    def flush_manifest(self) -> int:
        """Write any rows deferred by :meth:`batched_manifest`.  Returns the count."""
        pending, self._manifest_batch = self._manifest_batch, None
        if not pending:
            return 0
        self._flush_manifest(pending)
        return len(pending)

    # -- manifest --------------------------------------------------------- #
    def _load_manifest(self) -> Optional[pd.DataFrame]:
        if not os.path.isfile(self.manifest_path):
            return None
        df = pd.read_csv(self.manifest_path, dtype={"ticker": str})
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].str.upper()
        return df

    def _manifest_or_none(self) -> Optional[pd.DataFrame]:
        """The manifest, read **on first use** rather than in the constructor.

        The manifest is a *derived* index -- the tree on disk is the truth, which
        is what :func:`scan_partitions` and :func:`verify_manifest` both say -- and
        almost nothing needs it.  :meth:`tickers`, :meth:`sessions`, :meth:`coverage`
        and :meth:`load` all read the tree directly; only writing and the two
        point lookups consult it.

        So loading it eagerly put its whole cost on every app start, and on daily
        that cost is not small.  Measured: one row per *session*, and AAPL alone has
        11,544 of them, so the CSV reaches ~570MB across 503 tickers and parsing it
        costs ~6s before the first widget renders.  Nothing on the startup path
        reads a single row of it.

        The trade is deliberate: the first :meth:`write` in a process pays the read,
        and a process that only reads never pays it at all.  Writing is a batch
        operation that was going to read those bytes regardless.
        """
        if not self._manifest_read:
            self._manifest_read = True   # set first: a missing file must not retry
            self._manifest = self._load_manifest()
        return self._manifest

    def _write_manifest(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if self._manifest_batch is not None:
            # Deferred: see `batched_manifest`.  The rows are still produced and
            # still describe the bars now on disk; only the *rewrite* is postponed.
            self._manifest_batch.extend(rows)
            return
        self._flush_manifest(rows)

    def _flush_manifest(self, rows: list[dict]) -> None:
        os.makedirs(self.root, exist_ok=True)
        new = pd.DataFrame(list(rows))
        if self._manifest_or_none() is not None and len(self._manifest):
            new = pd.concat([self._manifest_or_none(), new], ignore_index=True, sort=False)
        if not len(new):
            return
        # Last write for a (ticker, session) wins; `n_revisions` accumulates separately.
        new = new.sort_values(["ticker", "session"])
        dedup = new.drop_duplicates(subset=["ticker", "session"], keep="last")
        counts = new.groupby(["ticker", "session"])["n_revisions"].sum().reset_index()
        counts = counts.rename(columns={"n_revisions": "n_revisions_total"})
        dedup = dedup.drop(columns=["n_revisions"], errors="ignore").merge(
            counts, on=["ticker", "session"], how="left"
        )
        dedup = dedup.rename(columns={"n_revisions_total": "n_revisions"})
        self._write_manifest_atomic(dedup)
        self._manifest = dedup

    def _write_manifest_atomic(self, frame: pd.DataFrame) -> None:
        """Write the manifest via a temp file and ``os.replace``.

        ``DataFrame.to_csv`` opens the target for writing, which **truncates it
        first**, so the manifest is briefly absent-or-short while the CSV is being
        streamed out.  Observed while a build was running: a reader saw the row
        count *fall* from 1,282,071 to 1,183,772 and reported a ticker on disk that
        the manifest did not contain.

        That window is small but it is not hypothetical here, because on daily the
        manifest is ~567MB and the write takes ~1.4s — long enough to lose a race
        with anything that reads the archive concurrently, including the app.  And
        the failure is nasty in kind, not just in timing: a truncated manifest
        presents as "these sessions are on disk but unindexed", which looks exactly
        like an interrupted run.

        ``os.replace`` is atomic on POSIX, so a reader sees either the old file or
        the new one, never a partial one.  The temp file is written in the same
        directory so the replace stays on one filesystem, which is what makes it
        atomic rather than a copy.
        """
        os.makedirs(self.root, exist_ok=True)
        tmp = "%s.tmp.%d" % (self.manifest_path, os.getpid())
        try:
            # **To the temp file, then replace.**  Streaming into ``manifest_path``
            # truncates the live manifest for the whole ~2.4s write, which is the
            # very race this function exists to close -- and it never replaced
            # anything, so ``tmp`` was created, named and then removed without ever
            # being written to.  Both were silent: the docstring promised atomicity
            # and a failing test asserted it, while the reader-visible window the
            # docstring describes stayed wide open.
            frame.to_csv(tmp, index=False)
            os.replace(tmp, self.manifest_path)
        finally:
            # A crash between write and replace leaves the temp file behind; it is
            # never the manifest, so removing it is always safe.  After a successful
            # ``os.replace`` this is a no-op, because the temp name no longer exists.
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:  # pragma: no cover - best effort cleanup
                    pass

    def _partition_path(self, ticker: str, day: str, *,
                        timeframe: object = DEFAULT_TIMEFRAME) -> str:
        """Where one ``(ticker, session)`` pair is stored.

        **Daily stores one file per *ticker*, not per session**, and the reason is
        measured rather than stylistic.  A daily partition holds a single bar, so the
        per-session layout costs a directory and a Parquet file -- with its footer,
        schema and compression dictionary -- to carry four numbers:

            AAPL daily, per-session   11,544 dirs   90 MB
            AAPL daily, per-ticker        1 file   ~200 KB

        Extrapolated over the index, per-session daily storage is ~45 GB and ~5.8
        million directories.  That is not a slow archive, it is an unusable one: inode
        pressure, and minutes per ticker spent in ``mkdir`` and ``to_parquet`` rather
        than fetching.

        The session is therefore a **column** (``session``) on daily rather than a
        directory component, which is where it already lives.  Every reader below
        accepts both layouts, so an archive is portable across the change and
        ``scan_partitions`` keeps its "directory names are the authoritative record"
        property by reading the session out of the filename when there is one.
        """
        tf = resolve_timeframe(timeframe)
        if tf.bars_per_session == 1:
            return os.path.join(self.root, f"ticker={ticker}", "bars.parquet")
        return os.path.join(self.root, f"ticker={ticker}", f"date={day}", "bars.parquet")

    @staticmethod
    @staticmethod
    def _file_fingerprint(path: str) -> Optional[str]:
        """The fingerprint of a whole stored partition, or ``None`` if unreadable.

        Read from the file rather than the manifest on purpose: the manifest is a
        derived index and is allowed to lag, whereas the daily writer's whole
        question is "is what is on disk the same as what I fetched?".  Asking the
        manifest would make a stale index look like a revision -- or hide a real one.
        """
        try:
            return fingerprint(pd.read_parquet(path))
        except Exception:  # noqa: BLE001 - an unreadable file is "no fingerprint"
            return None

    def _stored_rows(self, path: str) -> int:
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:  # noqa: BLE001 - an unreadable partition is handled by its caller
            return 0

    @staticmethod
    def _session_settled(day: date, frame: pd.DataFrame, *,
                         timeframe: object = DEFAULT_TIMEFRAME) -> bool:
        """True when this session has closed, so its content is final.

        The single distinguishing question for a fingerprint mismatch is *has the
        session ended?*  Measured against real data: a **closed** session re-fetched
        later is byte-identical, while a session still in progress changes on nearly
        every fetch -- not only in the last bar, but at scattered earlier timestamps
        too, as the exchange amends prints in a running session.

        That makes "still trading" the real reason for a mismatch, and it is why the
        revision test ignores content for an open session entirely.  Comparing only a
        few trailing bars, which looks like the obvious fix, is not sufficient: the
        amendments land anywhere in the session.  Whether the session has closed is
        both simpler and exactly right, and it is a property of the calendar rather
        than of when the sync happened to run.

        **Daily bars are settled by the calendar, not by their timestamp.**  A daily
        bar carries a single stamp at 09:30 ET -- the session's *open* -- so comparing
        it against a 1-minute session's final-bar boundary makes every daily bar look
        like a session still in progress, forever.  The intraday test is literally
        ``last_bar >= close - 1 minute``, and a daily bar's 09:30 stamp fails it on
        every single day.

        That failure is silent and severe.  There are exactly two reasons a
        fingerprint may differ -- "still trading" and "the vendor retro-adjusted
        history" -- and this function is what tells them apart.  With every daily bar
        reading as in-progress, a genuine revision of a settled day was merged as
        ordinary growth: the stored bar was overwritten, ``n_revisions`` never moved,
        and the one warning this whole fingerprint mechanism exists to raise never
        fired for daily data at all.  Measured, before this branch existed:

            PanelStore.write("T", daily).issues
              -> ['2026-09-03: short session: 1/390 bars',
                  '2026-09-03: session in progress, 1 -> 1 bars; merged']

        The resolution-independent question is whether the *day* is in the past, so
        that is what is asked here:

            today          -> not settled (the session may still be running)
            an earlier day -> settled
        """
        tf = resolve_timeframe(timeframe)
        if tf.bars_per_session == 1:
            # ``day`` arrives as a ``YYYY-MM-DD`` *string* from ``groupby`` on the
            # stored session column, so it is parsed rather than compared directly:
            # ``str < date`` raises, and the caller has no reason to know that.
            return pd.Timestamp(day).date() < pd.Timestamp.now(tz=EASTERN).date()
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return False
        return bool(ts.max() >= _session_settled_after(day))

    def _merge_growth(self, path: str, incoming: pd.DataFrame) -> pd.DataFrame:
        """Union the stored partition with an incoming superset, newest value winning.

        Only called once :meth:`_is_strict_growth` has confirmed the incoming frame
        agrees with every stored bar it shares, so "newest wins" resolves a genuine
        tie only in the harmless case where both sides are identical.
        """
        try:
            stored = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return incoming
        if stored.empty:
            return incoming
        merged = pd.concat([stored, incoming], ignore_index=True, sort=False)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = merged.loc[merged["timestamp"].notna()]
        for c in OHLCV:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
        return (merged.drop_duplicates("timestamp", keep="last")
                       .sort_values("timestamp")
                       .reset_index(drop=True))

    def existing_fingerprint(self, ticker: str, day: str) -> Optional[str]:
        """The stored fingerprint for one ``(ticker, session)``, or ``None`` if absent.

        Per session, always.  The daily writer does **not** use this to compare: it
        holds one file for the whole ticker, so a per-session lookup would describe a
        granularity that no longer exists on disk.  It compares the file itself, via
        :meth:`_file_fingerprint`.  The per-session fingerprints stay here as the
        human-facing record of what each day contains.
        """
        manifest = self._manifest_or_none()
        if manifest is None:
            return None
        hit = manifest.loc[
            (manifest["ticker"] == ticker) & (manifest["session"].astype(str) == day)
        ]
        if len(hit) == 0:
            return None
        return str(hit.iloc[0].get("fingerprint", "")) or None

    def revision_count(self, ticker: str, day: str) -> int:
        manifest = self._manifest_or_none()
        if manifest is None:
            return 0
        hit = manifest.loc[
            (manifest["ticker"] == ticker) & (manifest["session"].astype(str) == day)
        ]
        if len(hit) == 0:
            return 0
        try:
            return int(hit.iloc[0].get("n_revisions", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def tickers(self) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            d.split("=", 1)[1].upper()
            for d in os.listdir(self.root)
            if d.startswith("ticker=") and os.path.isdir(os.path.join(self.root, d))
        )

    def sessions(self, ticker: Optional[str] = None) -> list[str]:
        """Every session date present, ascending.

        Reads the ``session`` **column** for a whole-ticker file, not just the
        directory names.  That is the one place the two layouts must agree on an
        answer rather than an answer *about a layout*: "which days does this archive
        hold?" is the same question for a minute archive and a daily one, and the
        incremental sync asks it to decide what to fetch.  Answering it from
        directory names alone would report a daily archive as holding no sessions at
        all, so every incremental run would re-fetch everything.
        """
        out: set[str] = set()
        want = str(ticker).upper() if ticker else None
        for sym, day, path in _iter_partition_files(self.root):
            if want is not None and sym != want:
                continue
            if day is not None:
                out.add(day)
                continue
            try:
                df = pd.read_parquet(path, columns=["session"])
            except Exception:  # noqa: BLE001
                continue
            out.update(str(v) for v in df["session"].dropna().unique())
        return sorted(out)

    def rebuild_manifest(self) -> dict:
        """Regenerate the manifest from the partitions actually on disk.

        The manifest is a derived index -- :meth:`load` never reads it, because the
        directory names already say what every partition holds.  That makes it
        safe to rebuild from scratch, which is the repair for a manifest that has
        drifted: an interrupted write, a manually deleted partition, or a copy of
        the archive made without its index.

        Fingerprints are recomputed from the stored bars rather than invented, so a
        rebuilt manifest describes what is really there.  A session whose stored bars
        are simply absent is dropped from the manifest instead of being listed as
        present, and the returned report says how many, so the caller can tell a
        clean rebuild from one that quietly forgot a ticker.
        """
        rows: list[dict] = []
        dropped: list[tuple[str, str]] = []
        # ``day is None`` means a whole-ticker file, so the manifest gets **one row
        # per stored session** rather than one per file.  The manifest therefore keeps
        # its ``(ticker, session)`` meaning across both layouts, which is what lets
        # ``verify_manifest`` and ``coverage`` mean the same thing for either
        # resolution -- and it is a CSV row, not a directory, so the per-bar cost that
        # made the on-disk layout untenable does not apply here.
        for sym, day, path in _iter_partition_files(self.root):
            try:
                frame = pd.read_parquet(path)
            except Exception:  # noqa: BLE001 - an unreadable partition is reported
                dropped.append((sym, day or "*"))
                continue
            if frame.empty:
                dropped.append((sym, day or "*"))
                continue
            if day is None:
                if "session" not in frame.columns:
                    dropped.append((sym, "*"))
                    continue
                # One vectorised pass per ticker, not one per session: this loop is
                # the same quadratic shape that made a full rebuild unfinishable
                # (measured at 21+ minutes of CPU with the manifest not yet written,
                # because it hashed 4.16M single-bar sessions one at a time).
                rows.extend(self._daily_manifest_rows(sym, frame))
                continue
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            rows.append({
                "ticker": sym, "session": day, "bars": len(frame),
                "fingerprint": fingerprint(frame),
                "observed_fingerprint": fingerprint(frame),
                "n_revisions": 0,
                "first_bar": ts.min(), "last_bar": ts.max(),
            })

        self._manifest = None  # force a clean replace rather than a merge with the stale one
        if rows:
            self._write_manifest(rows)
        elif os.path.isfile(self.manifest_path):
            os.remove(self.manifest_path)
            self._manifest = None
        return {"rebuilt": len(rows), "dropped": dropped}

    def coverage(self) -> pd.DataFrame:
        """Sessions x tickers availability grid, for the Quality tab."""
        rows = []
        for sym in self.tickers():
            days = self.sessions(sym)
            rows.append({"ticker": sym, "n_sessions": len(days),
                         "first_session": days[0] if days else None,
                         "last_session": days[-1] if days else None})
        return pd.DataFrame(rows)

    # -- write ------------------------------------------------------------ #
    def write(self, ticker: str, frame: pd.DataFrame, *, policy: str = MERGE_POLICY,
              validate: bool = True,
              timeframe: object = DEFAULT_TIMEFRAME) -> TickerBar:
        """Write one ticker's bars, partitioned by Eastern session.

        Parameters
        ----------
        frame
            Bars with at least ``timestamp`` and OHLCV.  Extra columns are kept.
        policy
            See :data:`MERGE_POLICY`.  Under the default ``"flag"`` a re-fetch whose
            content differs leaves the stored bars untouched and increments a
            revision count -- §B's requirement that a revised day be *visible* rather
            than overwritten.
        validate
            Run the §B quality gate.  A session that fails fatally is not written.
        timeframe
            The resolution ``frame`` is at.  Threaded into the quality gate and into
            the "has this session closed?" test, because **both answers depend on it**
            and both were silently wrong for daily bars when this was hard-coded to
            1-minute:

            * the gate expects 390 bars per session, so every single daily bar was
              reported as ``short session: 1/390 bars`` -- one spurious warning per
              trading day, forever;
            * a daily bar carries one timestamp at 09:30 ET, which is strictly
              earlier than a 1-minute session's final-bar boundary, so **every daily
              bar read as a session still in progress**.  A re-fetch with different
              content was therefore merged as "growth" rather than flagged as the
              retro-adjustment it is -- the one warning §B exists to raise, disabled
              for daily.

            Defaults to 1-minute, so every existing caller and test keeps its current
            answer without having to say so.
        """
        if policy not in ("flag", "replace", "error"):
            raise ValueError(f"unknown policy={policy!r}")

        ticker = str(ticker).upper()
        tf = resolve_timeframe(timeframe)
        out = TickerBar(ticker=ticker)
        if frame is None or len(frame) == 0:
            out.fatal.append("no rows")
            return out

        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[frame["timestamp"].notna()].sort_values("timestamp")
        if frame.empty:
            out.fatal.append("no rows with valid timestamps")
            return out
        frame["session"] = session_et(frame["timestamp"])
        frame["ticker"] = ticker
        # Stored as a plain ``YYYY-MM-DD`` string rather than a Timestamp so the
        # partition key, the manifest and the Parquet column all compare and filter
        # identically.  A tz-aware Timestamped column made `session >= "2026-07-16"`
        # raise rather than filter, so every date-range read silently broke.
        frame["session"] = frame["session"].dt.strftime("%Y-%m-%d")

        # **Daily is written one file per ticker**, not one per session, and it is
        # routed out of the loop below for a measured reason rather than for tidiness.
        # The loop is per ``(ticker, session)``: it computes a fingerprint per session,
        # decides per session whether the difference is growth or a revision, and
        # writes one Parquet file per session.  Applied to daily that means a
        # directory and a file carrying four numbers -- 11,544 of them for AAPL,
        # 90 MB, extrapolating to ~45 GB and ~5.8 million directories for the index.
        #
        # The daily case does not need the loop's shape.  Its unit of comparison is
        # "one bar for this date", the revision rule is the same, and the whole
        # ticker's history is one file.  So daily gets its own writer below, and the
        # per-session loop stays exactly as it was for intraday.
        if tf.bars_per_session == 1:
            return self._write_daily(ticker, frame, policy=policy)

        manifest_rows: list[dict] = []
        for day, grp in frame.groupby("session", sort=True):
            day_str = str(day)
            if validate:
                rep = validate_session(grp, day, timeframe=tf)
                if not rep["ok"]:
                    out.sessions_rejected += 1
                    out.fatal.extend(f"{day_str}: {m}" for m in rep["fatal"])
                    continue
                out.issues.extend(f"{day_str}: {m}" for m in rep["issues"])

            clean = grp.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            fp = fingerprint(clean)
            prior = self.existing_fingerprint(ticker, day_str)
            path = self._partition_path(ticker, day_str)
            exists = os.path.isfile(path)

            if exists and prior is not None and prior == fp:
                out.sessions_unchanged += 1
                continue

            grew = False
            if exists and prior != fp:
                # A fingerprint can differ for two very different reasons, and the
                # default policy must distinguish them.
                #
                #   * The session is still trading.  The archive is accumulating,
                #     which is the entire point of §B -- a session written at 10:05
                #     must be replaceable at 15:00.  This is not a revision, and
                #     treating it as one makes a routine incremental sync look like
                #     corruption on every run, which trains an operator to ignore the
                #     one warning that matters.
                #   * The session has closed and the content still differs.  That is
                #     yfinance retro-adjusting history (§B's stated risk), and §B's
                #     answer is to flag it, not overwrite it.
                if not self._session_settled(day, clean, timeframe=tf):
                    grew = True
                    out.issues.append(
                        f"{day_str}: session in progress, {self._stored_rows(path)} -> "
                        f"{len(clean)} bars; merged")
                else:
                    if policy == "error":
                        out.sessions_rejected += 1
                        out.errors.append(
                            f"{day_str}: CLOSED session content differs from stored "
                            f"fingerprint ({prior} -> {fp}); policy='error'")
                        continue
                    if policy == "flag":
                        out.sessions_revised += 1
                        out.issues.append(
                            f"{day_str}: REVISED closed session differs from stored "
                            f"fingerprint ({prior} -> {fp}); kept existing partition")
                        manifest_rows.append({
                            "ticker": ticker, "session": day_str, "bars": len(clean),
                            "fingerprint": prior, "observed_fingerprint": fp,
                            "n_revisions": 1,
                            "first_bar": clean["timestamp"].iloc[0],
                            "last_bar": clean["timestamp"].iloc[-1],
                        })
                        continue
                    out.sessions_revised += 1

            os.makedirs(os.path.dirname(path), exist_ok=True)
            before = self._stored_rows(path) if exists else 0
            # A re-fetch is a *superset* of what is stored in practice (the exchange
            # only adds bars), so the union is taken and the new value wins any tie.
            # This never truncates: a fetch that starts later in the day would
            # otherwise delete the morning.
            final = self._merge_growth(path, clean) if exists else clean
            keep = [c for c in final.columns
                    if c in (*OHLCV, "timestamp", "ticker", "session")]
            final[keep].sort_values("timestamp").to_parquet(path, index=False)
            if grew:
                out.sessions_extended += 1
                # Count only the *new* bars.  Counting the merged total would report
                # a 300-bar session as 300 bars written on every incremental run,
                # which is how a growing archive ends up looking orders of magnitude
                # larger than the data actually added.
                out.bars_written += max(0, len(final) - before)
            else:
                out.sessions_written += 1
                out.bars_written += len(final)
            manifest_rows.append({
                "ticker": ticker, "session": day_str, "bars": len(final),
                "fingerprint": fingerprint(final), "observed_fingerprint": fp,
                "n_revisions": 0,
                "first_bar": final["timestamp"].iloc[0], "last_bar": final["timestamp"].iloc[-1],
            })

        if manifest_rows:
            self._write_manifest(manifest_rows)
        return out

    def _write_daily(self, ticker: str, frame: pd.DataFrame, *,
                     policy: str = MERGE_POLICY) -> TickerBar:
        """Write one ticker's **daily** history to a single file.

        The daily counterpart of :meth:`write`, and it exists because the per-session
        loop is the wrong shape for one-bar-per-day data (measured cost above).  What
        it preserves exactly is the behaviour that matters:

        * a re-fetch whose content is unchanged writes nothing and reports
          ``sessions_unchanged``;
        * a re-fetch of a day whose content differs is **flagged** when the day has
          settled, and merged when it has not -- the §B rule, unchanged;
        * the stored file is never silently overwritten under ``"flag"``.

        What it gives up is the per-session *file*, not the per-session *decision*.
        Both live on the ``session`` column, which the reader already relies on.

        Per-session fingerprints are still computed and still go in the manifest, so
        a revised day is reported by date exactly as an intraday revision is.  What
        the store reads back for comparisons is a **whole-file** fingerprint, because
        that is the granularity at which the file can actually differ; the per-session
        fingerprints remain the human-facing record.
        """
        out = TickerBar(ticker=ticker)
        path = self._partition_path(ticker, "", timeframe="1d")
        clean = frame.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        clean = clean.sort_values("timestamp").reset_index(drop=True)
        exists = os.path.isfile(path)

        stored_fp = self._file_fingerprint(path) if exists else None
        incoming_fp = fingerprint(clean)

        if exists and stored_fp is not None and stored_fp == incoming_fp:
            out.sessions_unchanged = len(clean)
            return out

        if exists and stored_fp != incoming_fp:
            # Which days moved?  Only those can be revisions, and naming them is the
            # whole point -- "the archive changed" is not actionable, "2024-03-01 was
            # revised" is.
            try:
                stored = pd.read_parquet(path)
            except Exception:  # noqa: BLE001
                stored = pd.DataFrame()
            changed = self._daily_changed_sessions(stored, clean)
            unsettled = [d for d in changed
                         if not self._session_settled(pd.Timestamp(d).date(), clean,
                                                      timeframe="1d")]
            settled = sorted(set(changed) - set(unsettled))
            if unsettled:
                out.sessions_extended = len(unsettled)
                out.issues.append(
                    "{}: {} session(s) in progress ({}); merged".format(
                        ticker, len(unsettled), ", ".join(sorted(unsettled)[:3]))
                )
            if settled:
                if policy == "error":
                    out.sessions_rejected = len(settled)
                    out.errors.append(
                        "{}: CLOSED session(s) differ from stored content: {}; "
                        "policy='error'".format(ticker, ", ".join(settled[:3])))
                    return out
                if policy == "flag":
                    out.sessions_revised = len(settled)
                    out.issues.append(
                        "{}: REVISED closed session(s) differ from stored "
                        "content ({}); kept existing file".format(
                            ticker, ", ".join(settled[:3])))
                    # A revision is recorded per session against the *stored*
                    # fingerprint, which is what the manifest is for.
                    self._record_daily_revisions(ticker, stored, clean, settled)
                    if not unsettled:
                        return out
                    # **A settled day must not be merged just because an unsettled one
                    # was.**  Dropping the revised rows and keeping only the in-progress
                    # ones is what makes the two rules coexist: otherwise a single
                    # still-trading session would let every revised settled day
                    # through, and the §B warning would be silenced by an unrelated
                    # row.  This is the case a coarse "does anything differ?" test
                    # cannot see, because it needs *both* kinds of difference at once.
                    drop = set(settled)
                    clean = clean.loc[~clean["session"].astype(str).isin(drop)]
                    clean = clean.reset_index(drop=True)
                    if clean.empty:  # pragma: no cover - settled non-empty above
                        return out
                else:
                    out.sessions_revised = len(settled)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        before = self._stored_rows(path) if exists else 0
        final = self._merge_growth(path, clean) if exists else clean
        keep = [c for c in final.columns
                if c in (*OHLCV, "timestamp", "ticker", "session")]
        final[keep].sort_values("timestamp").to_parquet(path, index=False)
        out.bars_written = max(0, len(final) - before)
        out.sessions_written = len(final)
        self._write_manifest(self._daily_manifest_rows(ticker, final))
        return out

    @staticmethod
    def _daily_changed_sessions(stored: pd.DataFrame, incoming: pd.DataFrame) -> list[str]:
        """Sessions whose stored bar differs from the incoming one."""
        if stored is None or stored.empty:
            return sorted(set(incoming["session"].astype(str)))
        keys = ("session",)
        have = stored.set_index(list(keys))[list(OHLCV)].sort_index()
        want = incoming.drop_duplicates(list(keys)).set_index(list(keys))[list(OHLCV)].sort_index()
        both = have.index.intersection(want.index)
        if len(both) == 0:
            return sorted(set(want.index.astype(str)) - set(have.index.astype(str)))
        diff = (have.loc[both].astype(float) != want.loc[both].astype(float)).any(axis=1)
        return sorted(diff[diff].index.astype(str))

    def _daily_manifest_rows(self, ticker: str, final: pd.DataFrame) -> Iterable[dict]:
        """One manifest row per stored daily session.

        The manifest keeps its ``(ticker, session)`` shape on daily too, so
        ``verify_manifest`` and ``rebuild_manifest`` keep one meaning for
        ``n_partitions`` across both resolutions.  The cost is one row per bar, which
        is a CSV and not a directory -- the thing that was actually expensive.

        Every session's fingerprint is computed in **one** pass via
        :func:`fingerprint_per_session`, and the per-session ``first_bar``/``last_bar``
        come from a single grouped aggregation.  Iterating the sessions here and
        calling ``pd.to_datetime`` on each slice was measured at 3.0s for AAPL's
        11,544 sessions -- 11,544 ``groupby`` slices and 11,544 datetime
        normalisations, to produce two timestamps per day.
        """
        if len(final) == 0:
            return []
        fps = fingerprint_per_session(final, None)
        ts = pd.to_datetime(final["timestamp"], utc=True, errors="coerce")
        # One grouped pass for the bounds, rather than a slice and a datetime
        # conversion per session.
        bounds = (pd.DataFrame({"session": final["session"].astype(str),
                                "ts": ts})
                  .groupby("session", sort=True)["ts"]
                  .agg(["min", "max", "size"]))
        rows = []
        for day in bounds.index:
            row = bounds.loc[day]
            fp = fps[str(day)]
            rows.append({
                "ticker": ticker, "session": str(day), "bars": int(row["size"]),
                "fingerprint": fp, "observed_fingerprint": fp,
                "n_revisions": 0,
                "first_bar": row["min"], "last_bar": row["max"],
            })
        return rows

    def _record_daily_revisions(self, ticker: str, stored: pd.DataFrame,
                                incoming: pd.DataFrame, sessions: Sequence[str]) -> None:
        """Record per-session revisions against the *stored* fingerprints.

        **Vectorised, because the naive shape is quadratic.**  This used to scan the
        whole stored frame for each revised session -- ``stored.loc[stored["session"]
        .astype(str) == day]`` per day -- and hash it with two more full-column
        ``pd.to_datetime`` passes.  With Yahoo retro-adjusting a ticker's whole
        history, that is ~5,410 sessions x a 6,758-row scan each.

        Measured on the real archive, ticker ``A``: the incremental run reported
        ``revised=5410`` for its **first** ticker and took 295s to write one.  Every
        fingerprint below comes out of a single vectorised pass, so the cost is
        proportional to the frame rather than to (sessions x frame).
        """
        if not len(sessions):
            return
        wanted = {str(d) for d in sessions}
        stored = stored[stored["session"].astype(str).isin(wanted)]
        incoming = incoming[incoming["session"].astype(str).isin(wanted)]
        if not len(stored):
            return
        # One hash per session for each side, computed in a single pass each.
        stored_fp = fingerprint_per_session(stored, None)
        incoming_fp = fingerprint_per_session(incoming, None)
        ts = pd.to_datetime(stored["timestamp"], utc=True, errors="coerce")
        bounds = (pd.DataFrame({"session": stored["session"].astype(str), "ts": ts})
                  .groupby("session", sort=True)["ts"].agg(["min", "max", "size"]))
        rows = []
        for day in bounds.index:
            day = str(day)
            observed = incoming_fp.get(day, "")
            rows.append({
                "ticker": ticker, "session": day, "bars": int(bounds.loc[day, "size"]),
                "fingerprint": stored_fp[day],
                "observed_fingerprint": observed,
                "n_revisions": 1,
                "first_bar": bounds.loc[day, "min"], "last_bar": bounds.loc[day, "max"],
            })
        if rows:
            self._write_manifest(rows)

    def load(self, *, tickers: Optional[Sequence[str]] = None,
             since: Optional[str] = None, until: Optional[str] = None) -> pd.DataFrame:
        """Read the archive back as one long frame (see :func:`scan_partitions`)."""
        return scan_partitions(self.root, tickers=tickers, since=since, until=until)

    def per_ticker(self, *, tickers: Optional[Sequence[str]] = None,
                   since: Optional[str] = None, until: Optional[str] = None) -> dict[str, pd.DataFrame]:
        """Bars split per ticker, each sorted by timestamp with a clean index.

        The index is reset per ticker so a bar index is meaningful only *within* a
        ticker.  The matrix-profile engine is defined over one contiguous series, and
        a global index that ran across the ticker boundary would silently let a
        window span two companies -- which is the single easiest way to make a
        cross-sectional search meaningless.
        """
        df = self.load(tickers=tickers, since=since, until=until)
        if df.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        for sym, grp in df.groupby("ticker", sort=True):
            g = grp.sort_values("timestamp").reset_index(drop=True)
            out[str(sym)] = g
        return out
