"""Fetch bars for an arbitrary ticker, at 1-minute or daily resolution.

Implements PLAN.md §B/§F.

Why this module exists
----------------------
``scripts/download_data.py`` is a superseded prototype with a hard-coded S&P 500 list
that rewrites ``data/sp500/<TICKER>_1min_<start>_<end>.csv`` from scratch on every run.
It cannot answer *"show me MSFT"*, and §B's point is that **the archive is the
product, not the download** -- so this module adds the missing half: fetching one
symbol, on demand, without disturbing what is already on disk.

What it deliberately does *not* do
----------------------------------
It does not reimplement :mod:`timeseries.store`.  ``store`` is the partitioned
Parquet archive with per-session fingerprints and revision counting; this module is
the *acquisition* step that feeds it (and the Streamlit dashboard, which loads CSVs).
They agree on two conventions, both imported rather than reimplemented:

* :func:`timeseries.store.yahoo_symbol` -- Wikipedia's constituent list spells class
  shares ``BRK.B``, while Yahoo answers only to ``BRK-B``.  A rejected symbol is
  *silent* (an empty frame, not an error), so this conversion is applied before every
  request and a user who pastes the dotted form still gets bars.
* :func:`timeseries.store.OHLCV` -- the bar columns, in the order written to CSV.
  Volume arrives from Yahoo too and is deliberately discarded: it is not a matching
  feature, and for some instruments Yahoo fills most minute bars with a zero.

Yahoo's intraday limits, and why they shape this file
----------------------------------------------------
Yahoo serves roughly **8 days of 1-minute bars per request** and keeps no more than
~30 days of intraday history at all.  Asking for ``period="1mo"`` at ``1m`` does not
raise -- it *silently returns an empty frame*, which is the worst possible failure
mode for a UI that is about to draw an empty chart.  So:

* the requested span is **chunked** into :data:`CHUNK_DAYS`-day requests, each of
  which is inside the limit;
* every chunk is checked for emptiness and reported in
  :attr:`FetchResult.errors`, so "the server refused" and "the market was closed" are
  distinguishable instead of collapsing into one blank chart;
* :data:`MAX_1M_DAYS` caps the total span at a value Yahoo can actually serve.

Daily bars are the exception, and it is a structural one rather than a loosened limit
------------------------------------------------------------------------------------
**None of the three mechanisms above apply to ``interval="1d"``.**  Yahoo serves
decades of daily history in a single request, so there is no per-request ceiling to
chunk against and no retention wall to clamp the span to.  Applying the intraday
machinery to a daily fetch would not be conservative -- it would be actively wrong:
clamping a daily request to 29 days would silently discard the ~95% of the history
the endpoint is offering for free, and doing it in 7-day pieces would multiply one
HTTP request into thousands to work around a limit that does not exist.

So :func:`fetch_ticker` branches on the timeframe and the daily path takes a *single*
wide request.  The two paths share everything downstream of the response -- the
column normalisation, the de-duplication, the error reporting -- because none of that
depends on the interval.

Which is why the interval is not a free parameter
-------------------------------------------------
This module used to argue, at length, that it should have **no** interval selector,
on the grounds that 1-minute was baked in far deeper than one constant.  That
argument was correct, and it is why the second timeframe required re-calibrating the
gap heuristics, the session arity and the rolling window *together* rather than
merely adding a string.

The lesson is not "one interval forever" -- it is that the calibration has to be
*named*.  Those constants now live in :mod:`timeseries.timeframes`, in one table with
the reasoning for each value beside it, and this module reads its interval from
there.  A hypothetical third timeframe is a new row in that table, not a second
literal threaded through six call sites.  What is still refused is the *ad hoc*
interval: :func:`get_timeframe` raises on an unsupported key rather than passing an
unvalidated string to Yahoo, because an unknown interval does not fail loudly -- it
returns an empty frame, which is precisely the failure this module is built to
prevent.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .store import OHLCV, stamp_label, yahoo_symbol
from .timeframes import DEFAULT_TIMEFRAME, TIMEFRAMES, get_timeframe, resolve_timeframe

__all__ = [
    "FetchResult",
    "BAR_INTERVAL",
    "CHUNK_DAYS",
    "MAX_1M_DAYS",
    "archive_name",
    "fetch_ticker",
    "normalize_symbol",
    "symbol_from_filename",
    "timeframe_from_filename",
    "DEFAULT_DAILY_DAYS",
]

#: Back-compat aliases for the intraday calibration, now owned by
#: :mod:`timeseries.timeframes`.  These remain because ``app.py`` interpolates
#: ``MAX_1M_DAYS`` into its help copy and a dozen tests pin these exact numbers; a
#: name that silently started describing *whatever the default timeframe is* would be
#: worse than one that stays honestly 1-minute.
#:
#: ``CHUNK_DAYS`` is Yahoo's ~8-day per-request intraday ceiling minus a day of
#: margin, because the limit is enforced against the *requested* span and an
#: off-by-one comes back as an empty chunk rather than as an error.
#:
#: ``MAX_1M_DAYS`` is Yahoo's ~30-day intraday retention.  Verified against the live
#: endpoint: asking for 29, 30 and 31 days all return the *same* ~21 sessions, so it
#: is a real wall rather than a soft edge, and 29 is the widest span worth requesting.
#: **Neither figure has a daily analogue** -- see the module docstring.
BAR_INTERVAL = get_timeframe("1m").yfinance_interval
CHUNK_DAYS = get_timeframe("1m").chunk_days
MAX_1M_DAYS = get_timeframe("1m").max_days

#: How far back a daily fetch reaches when the caller does not say.
#:
#: Deliberately wider than anything Yahoo actually retains.  There is no retention
#: wall on daily bars, so the only thing a span limit can do is truncate a result the
#: endpoint was willing to serve in full -- and the truncation would be invisible,
#: because a short history looks exactly like a short history.  Over-asking is the
#: only safe direction, and 25 years exceeds the listing age of every instrument this
#: app can name.
DEFAULT_DAILY_DAYS = 25 * 365

#: Why Yahoo returned nothing, per timeframe, for :attr:`FetchResult.errors`.
#:
#: The two cases have genuinely different causes and quoting the wrong one is worse
#: than quoting neither: telling a reader their daily fetch fell "outside the ~30-day
#: intraday retention" when no such wall exists on daily bars sends them looking for a
#: limit that will never help them.
_WHY_EMPTY = {
    "1m": "outside the ~30-day intraday retention, or the market was closed",
    "1d": "the symbol has no daily history, or it is not listed",
}

# Yahoo answers only to the dashed class-share spelling (``BRK-B``), so the dash is
# what gets stored and displayed; :func:`timeseries.store.yahoo_symbol` normalises
# whatever the user typed.
#
# A leading ``^`` is also accepted, because Yahoo marks every index symbol with one:
# ``^VIX`` (the CBOE Volatility Index), ``^GSPC``, ``^DJI``, ``^NDX``.  It is optional
# and positional -- the bare ``^?`` in the raw string is the literal character, not an
# anchor -- so ``^^VIX`` and ``A^B`` are still refused.  Widening the set does not
# weaken the traversal defence: ``^`` is not a path separator, and ``.`` has already
# been collapsed to ``-`` one line below, before this pattern ever runs.
_SYMBOL_RE = re.compile(r"^\^?[A-Z0-9=][A-Z0-9.\-=]{0,15}$")

# ``data/`` filenames are ``<SYMBOL>_1min_<YYYYMMDD>_<YYYYMMDD>.csv`` or
# ``<SYMBOL>_1d_<YYYYMMDD>_<YYYYMMDD>.csv``.  Parsing the symbol *and the resolution*
# back out is what lets the dashboard label an archive with the name a reader expects,
# instead of printing a path.
#
# The slug alternation is built from :data:`timeseries.timeframes.TIMEFRAMES` rather
# than written as a literal, so a third timeframe does not need this pattern touched.
# That matters more here than it looks: a filename this regex cannot read is not a
# parse failure the caller can recover from, it is a file that silently does not
# appear in the dashboard's listing.
#
# The optional leading ``\^?`` mirrors :data:`_SYMBOL_RE` and is load-bearing rather
# than decorative: without it :func:`archive_name` writes a ``^VIX_1min_...csv`` that
# this parser cannot read back.  Nothing on the *fetch* path round-trips a filename,
# so a fetch smoke test would never surface that asymmetry -- only the round-trip
# test does.
_ARCHIVE_RE = re.compile(
    r"^(?P<sym>\^?[A-Za-z0-9=][A-Za-z0-9.\-=]*)_(?P<slug>"
    + "|".join(re.escape(t.filename_slug) for t in TIMEFRAMES.values())
    + r")_(?P<start>\d{8})_(?P<end>\d{8})\.csv$"
)


def normalize_symbol(raw: object) -> str:
    """Validate and canonicalise a user-typed ticker.

    Returns the **display** spelling (``BRK-B``), uppercased and dash-separated, which
    is also the filename spelling.  Either the dotted or the dashed form typed by the
    user resolves to the same stored symbol.

    A Yahoo **index** symbol keeps its leading ``^``: ``^vix`` canonicalises to
    ``^VIX``.  The caret is carried through into :func:`archive_name` and into the
    request itself, because stripping it would ask Yahoo for a different instrument
    and return zero rows with no error -- the same silent failure the ``BRK.B`` to
    ``BRK-B`` conversion exists to prevent.

    Raises
    ------
    ValueError
        If the input is not a plausible exchange symbol.  Validation happens here
        rather than being passed to Yahoo because an unvalidated string is also a
        filename: a symbol of ``../../etc/passwd`` reaching :func:`archive_name`
        would escape the data directory.  The charset is the actual defence, and this
        is the cheapest place to enforce it.
    """
    if raw is None:
        raise ValueError("No ticker entered.")
    sym = str(raw).strip().upper().replace(".", "-")
    if not sym:
        raise ValueError("No ticker entered.")
    if not _SYMBOL_RE.match(sym):
        raise ValueError(
            "{!r} is not a valid ticker symbol. Use letters, digits, '-' or '.', "
            "optionally prefixed with '^' for an index — for example AAPL, MSFT, "
            "BRK-B or ^VIX.".format(str(raw).strip())
        )
    return sym


def archive_name(symbol: str, first: pd.Timestamp, last: pd.Timestamp,
                 *, timeframe: object = DEFAULT_TIMEFRAME) -> str:
    """Filename for one ticker's archive, matching the convention already in ``data/``.

    The shape is ``<SYMBOL>_<slug>_<YYYYMMDD>_<YYYYMMDD>.csv``, so a 1-minute archive
    is still named ``QQQ_1min_20260831_20260930.csv`` and the dashboard's existing
    file listing, and :func:`symbol_from_filename`, work on it without a special case.
    Daily archives carry ``_1d_`` instead.

    The slug comes from :attr:`~timeseries.timeframes.Timeframe.filename_slug` and not
    from :attr:`~timeseries.timeframes.Timeframe.yfinance_interval`, because the two
    spellings differ for intraday: Yahoo asks for ``1m`` and the file is ``1min``.
    """
    tf = resolve_timeframe(timeframe)
    return "{}_{}_{}_{}.csv".format(
        normalize_symbol(symbol),
        tf.filename_slug,
        pd.Timestamp(first).strftime("%Y%m%d"),
        pd.Timestamp(last).strftime("%Y%m%d"),
    )


def timeframe_from_filename(name: object) -> Optional[str]:
    """Timeframe key encoded in an archive filename, or ``None`` if it is not one.

    Paired with :func:`symbol_from_filename` because a filename carries *both*, and a
    dashboard that reads the symbol off ``QQQ_1d_20200101_20260930.csv`` while
    assuming 1-minute bars would label a daily archive with the wrong resolution.

    Returns ``None`` for anything unrecognised rather than defaulting, for the same
    reason :func:`symbol_from_filename` does: a file that is not one of ours should
    be labelled as unknown, not silently described as something it is not.
    """
    m = _ARCHIVE_RE.match(os.path.basename(str(name)))
    if not m:
        return None
    for tf in TIMEFRAMES.values():
        if tf.filename_slug == m.group("slug"):
            return tf.key
    return None


def symbol_from_filename(name: object) -> Optional[str]:
    """Ticker symbol encoded in an archive filename, or ``None`` if it is not one.

    Retained for the *naming* convention itself: :func:`archive_name` writes
    ``<SYMBOL>_<slug>_<start>_<end>.csv`` and this parses the symbol back out, so a
    caller holding a filename can label it with the name a reader expects instead of
    printing a path.  Returns ``None`` rather than guessing for anything that does
    not match, so an unrelated CSV is labelled "the archive" instead of being
    mislabelled with a symbol the user never asked about.

    The resolution is read separately, by :func:`timeframe_from_filename`.  Both are
    derived from one regex, so a name cannot parse as a symbol here and as an unknown
    file there.
    """
    m = _ARCHIVE_RE.match(os.path.basename(str(name)))
    if not m:
        return None
    return m.group("sym").upper()


def _normalize_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn one raw yfinance response into the bar frame this package expects.

    yfinance hands back Title-case columns under a ``Datetime`` *index*; everything
    downstream in this repo (``features``, ``panel``, ``app``) reads lowercase
    ``timestamp`` + :data:`~timeseries.store.OHLCV` as columns.  Normalising here
    keeps that translation in exactly one place.

    A ``MultiIndex`` column set is flattened defensively: it appears whenever a
    request is made for more than one symbol, and its labels are
    ``(field, ticker)``.  Only the outer level is dropped, because that is where
    ``Close``/``Volume`` live.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=["timestamp", *OHLCV])

    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(level[0]) for level in out.columns]

    # The timestamp arrives as the index name ("Datetime" on some builds, the index
    # itself unnamed on others), so normalise to a column before renaming.
    index_name = str(out.index.name or "").strip().lower()
    if index_name in ("", "datetime", "date", "timestamp", "index"):
        out = out.reset_index()

    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "timestamp" not in out.columns:
        for candidate in ("date", "datetime", "index"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "timestamp"})
                break

    keep = [c for c in ("timestamp", *OHLCV) if c in out.columns]
    missing = [c for c in ("timestamp", *OHLCV) if c not in out.columns]
    if missing:
        raise ValueError(
            "Yahoo response is missing the column(s) {}; got {}. The upstream "
            "response shape has probably changed.".format(missing, list(out.columns))
        )
    return out[keep]


#: Only ``yfinance``'s logger is silenced, and only for the one message.  Yahoo owns
#: the wording, so the match is on the substring rather than an exact string; the
#: filter still only lets through records it recognises, so anything new stays loud.
_YF_FALSE_DELISTED = "possibly delisted; no price data found"


class _SilenceFalseDelisting:
    """Log filter hiding yfinance's ``possibly delisted`` line when it is a lie.

    Yahoo answers an empty intraday window with **the same log line it uses for a
    symbol that no longer exists**, because from its side a weekend chunk, a chunk
    before the market opened, and a genuine delisting are all just "no rows".  yfinance
    turns that into ``logger.error(err_msg)`` and returns an empty frame instead of
    raising.

    That matters here because of how the span is built.  ``fetch_ticker`` chunks a
    ~29-day window into 7-day pieces, and **the final chunk of any window that ends
    between one Friday and the next session's open covers a weekend**, so this line is
    printed on essentially every successful fetch.  Measured against the live endpoint
    on a Saturday, for ``QQQ``:

        2026-10-03 → 2026-10-04:  0 bars   <- the false alarm, on a *successful* fetch
        overall:                 7,410 bars over 19 sessions

    It is strictly worse than noise.  The chunk *is* recorded -- in
    :attr:`FetchResult.chunks_failed` and :attr:`FetchResult.errors`, with a message
    that names the real cause ("outside the ~30-day intraday retention, or the market
    was closed") -- so the line both contradicts what the app reports to the reader and
    teaches the reader to ignore errors that are real.  Suppressing it at the source
    removes the contradiction without hiding anything, because the empty chunk was
    never silent to begin with.

    A filter rather than ``logging.disable`` or a level bump, because both of those are
    global and permanent: a level bump would also swallow the rate-limit and HTTP
    warnings that are worth seeing, and neither could be undone when the fetch ends.
    The suppression is installed on the ``yfinance`` logger for the duration of one
    request and removed again in :meth:`__exit__`, so it cannot leak into an
    unrelated part of the process.

    Yahoo's phrasing is not contractual, so this degrades to a no-op filter if upstream
    changes it -- the log line returns and the chunk is still recorded as it always was.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return _YF_FALSE_DELISTED not in str(record.getMessage())
        except Exception:  # noqa: BLE001 - never let a log filter break a fetch
            return True


@contextmanager
def _quiet_false_delisting():
    """Suppress only yfinance's spurious ``possibly delisted`` line, for one call."""
    logger = logging.getLogger("yfinance")
    log_filter = _SilenceFalseDelisting()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


def _fetch_chunk(symbol: str, start: datetime, end: datetime,
                 *, interval: str = BAR_INTERVAL) -> pd.DataFrame:
    """One request to Yahoo at ``interval``.

    Chunked by the caller only where a per-request limit requires it -- see
    :func:`fetch_ticker`, which sends a daily request in a single piece precisely
    because there is no limit to respect.

    ``yfinance`` is imported here rather than at module scope: it is an optional
    ``app`` extra, and importing this module must not be what breaks a library-only
    install.  A missing dependency is reported as a fetch failure with an actionable
    message rather than an ``ImportError`` at import time.

    The quiet filter is installed around the request, not around the chunk loop: an
    empty chunk is already reported through :attr:`FetchResult.errors`, so the filter
    removes a duplicate claim rather than a unique piece of information.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - an installation failure
        raise RuntimeError(
            "Fetching a ticker needs yfinance. Install it with "
            "`pip install -e \".[app]\"`."
        ) from exc
    with _quiet_false_delisting():
        return yf.Ticker(symbol).history(interval=interval, start=start, end=end)


@dataclass
class FetchResult:
    """Outcome of one :func:`fetch_ticker` call.

    ``errors`` is a list rather than an exception because partial success is real and
    useful: four of five chunks can land while one fails, and the caller wants to load
    those 4 sessions *and* say out loud that the fifth is missing.  Raising instead
    would mean discarding good data to report a bad day.

    :attr:`timeframe` is recorded rather than inferred, because the same symbol and
    the same span fetched at two resolutions are two different archives and every
    message this object produces -- the summary, the error text, the archive filename
    -- has to describe the one that actually happened.
    """

    symbol: str
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list = field(default_factory=list)
    chunks_ok: int = 0
    chunks_failed: int = 0
    timeframe: str = DEFAULT_TIMEFRAME

    @property
    def ok(self) -> bool:
        """True when there is something to chart.

        A fetch that contacted Yahoo and came back with no bars is **not** ok, and
        this is the check the UI gates on: an empty archive would otherwise render as
        a blank chart that looks like a broken app rather than a refused request.
        """
        return len(self.frame) > 0

    @property
    def tf(self):
        """The resolved :class:`~timeseries.timeframes.Timeframe` for this fetch."""
        return resolve_timeframe(self.timeframe)

    @property
    def n_bars(self) -> int:
        return int(len(self.frame))

    @property
    def first(self) -> Optional[pd.Timestamp]:
        return self.frame["timestamp"].iloc[0] if len(self.frame) else None

    @property
    def last(self) -> Optional[pd.Timestamp]:
        return self.frame["timestamp"].iloc[-1] if len(self.frame) else None

    @property
    def sessions(self) -> int:
        """Distinct Eastern trading days covered.

        Counted in Eastern time because that is what "session" means for US equities
        (:mod:`timeseries.store`); bucketing on UTC dates would split a session that
        runs past 00:00 UTC and report roughly twice the true session count.

        On daily bars this is simply the number of bars: one bar *is* one trading
        day, so dividing by sessions is vacuous.  :attr:`bars_per_session` below is
        what the UI reads instead, and it collapses to 1 there.
        """
        if not len(self.frame):
            return 0
        idx = pd.to_datetime(self.frame["timestamp"], utc=True, errors="coerce")
        return int(idx.dt.tz_convert("America/New_York").dt.floor("D").nunique())

    @property
    def bars_per_session(self) -> float:
        """Average bars per trading session over what was fetched.

        390-ish for 1-minute, and exactly 1 for daily.  This is the honest denominator
        for the Price tab's "how much tape am I looking at" view sizing, where the
        reader thinks in sessions rather than in bars; dividing by
        :attr:`sessions` would be meaningless on a daily frame.
        """
        sessions = self.sessions
        return float(self.n_bars) / sessions if sessions else float(self.n_bars)

    def summary(self) -> str:
        """One line describing what was fetched, for the UI to confirm with."""
        tf = self.tf
        if not self.ok:
            return "No bars returned for {} ({}).".format(self.symbol, tf.label.lower())
        # The stamps come from the package's one definition of what a bar looks like,
        # so this line and the app's chart axes cannot print the same bar two ways --
        # and the "(UTC)" suffix is conditional for the same reason: a daily stamp is
        # an Eastern date, not a UTC instant.
        zone = "" if tf.bars_per_session == 1 else " (UTC)"
        span = "{} → {}".format(stamp_label(self.first, tf.key),
                                stamp_label(self.last, tf.key))
        return (
            "{} · {} · {:,} bars · {} sessions · {}{}".format(
                self.symbol, tf.label, self.n_bars, self.sessions, span, zone,
            )
        )


def fetch_ticker(
    symbol: object,
    *,
    days: Optional[int] = None,
    end: Optional[datetime] = None,
    timeframe: object = DEFAULT_TIMEFRAME,
) -> FetchResult:
    """Download bars for ``symbol``, spanning the last ``days`` days.

    Resolution is chosen by ``timeframe`` and defaults to 1-minute, where the span
    defaults to :data:`MAX_1M_DAYS` -- i.e. **everything Yahoo has**.  1-minute history
    stops at roughly 30 days, so there is no useful longer span to ask for and asking
    for less is a choice with no upside: a bigger candidate pool is what §E's
    percentile is measured against, and more sessions strictly improves it.

    On 1-minute the span is fetched in :data:`CHUNK_DAYS`-day pieces, concatenated,
    de-duplicated on timestamp and sorted.  The chunking is not an optimisation, it is
    what makes the full span reachable at all -- measured against the live endpoint, a
    *single* request spanning 30 days returns zero rows, while the same 30 days
    assembled from 7-day requests returns ~7,800 bars across ~21 sessions.

    **On daily the chunking is skipped entirely** and one request covers the whole
    :data:`DEFAULT_DAILY_DAYS` span.  That is not a performance choice: Yahoo imposes
    no per-request limit and no retention wall on ``interval="1d"``, so the mechanism
    has nothing to work around, and reusing it would discard almost all of the history
    the endpoint is offering.

    De-duplication is not cosmetic on the chunked path: adjacent boundaries overlap by
    the boundary bar itself, and leaving that in would hand ``load_bars`` a duplicate
    to resolve and inflate the bar count in the archive.  It runs on daily too, where
    it is a no-op unless the vendor repeats a day.

    Parameters
    ----------
    symbol
        Any ticker Yahoo accepts.  Normalised and validated by
        :func:`normalize_symbol`, which raises ``ValueError`` on nonsense.
    days
        Total span requested, or ``None`` for **everything this timeframe has**.
        ``None`` is the default and is resolved per timeframe rather than pinned to one
        constant: on 1-minute that is :data:`MAX_1M_DAYS`, the retention window Yahoo
        will serve; on daily it is :data:`DEFAULT_DAILY_DAYS`, wider than Yahoo keeps.
        An explicit value is *clamped* to the 1-minute retention window and honoured
        as-is on daily.  Pass a smaller value on 1-minute only to test the fetch path
        cheaply -- it yields a smaller candidate pool, not a different archive.
    end
        Exclusive end of the span.  Defaults to now, in UTC.
    timeframe
        ``"1m"`` (the default) or ``"1d"``.  Resolved through
        :func:`~timeseries.timeframes.get_timeframe`, so an unsupported key raises
        rather than being passed to Yahoo as a string that would come back empty.

    Returns
    -------
    FetchResult
        With ``errors`` populated for every chunk that came back empty or raised.
        Yahoo answers an over-long intraday request with an empty frame and a log
        line rather than an exception, so an empty chunk is recorded explicitly --
        otherwise a weekend-spanning or out-of-retention request reads as "no data
        exists" with no explanation.
    """
    sym = normalize_symbol(symbol)
    tf = resolve_timeframe(timeframe)
    result = FetchResult(symbol=sym, timeframe=tf.key)

    stop = end or datetime.now(timezone.utc)
    if stop.tzinfo is None:
        stop = stop.replace(tzinfo=timezone.utc)

    if tf.max_days is not None:
        # Intraday: Yahoo only retains a fixed window, so the span is *clamped* to it.
        # ``None`` means "everything available", which is that window.
        span = max(1, min(int(days) if days is not None else tf.max_days, tf.max_days))
    else:
        # Daily: no retention wall, so the span is honoured as asked and ``None`` means
        # "everything available" too -- a span wider than Yahoo retains.  Neither
        # branch needs to know what the other's ceiling is.
        span = max(1, int(days) if days is not None else DEFAULT_DAILY_DAYS)
    start = stop - timedelta(days=span)
    y_sym = yahoo_symbol(sym)

    frames: list[pd.DataFrame] = []

    why_empty = _WHY_EMPTY.get(tf.key, _WHY_EMPTY["1m"])

    # The chunk boundaries: one request for the whole span where there is no
    # per-request limit, and ``chunk_days``-sized pieces where there is.
    if tf.chunk_days is None:
        bounds = [(start, stop)]
    else:
        bounds = []
        cursor = start
        while cursor < stop:
            bounds.append((cursor, min(cursor + timedelta(days=tf.chunk_days), stop)))
            cursor = min(cursor + timedelta(days=tf.chunk_days), stop)

    for cursor, chunk_end in bounds:
        try:
            raw = _fetch_chunk(y_sym, cursor, chunk_end, interval=tf.yfinance_interval)
            chunk = _normalize_chunk(raw)
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not lose the rest
            result.chunks_failed += 1
            result.errors.append(
                "{} → {}: {}".format(cursor.date(), chunk_end.date(), exc)
            )
        else:
            if chunk.empty:
                result.chunks_failed += 1
                result.errors.append(
                    "{} → {}: Yahoo returned no bars ({}): {}".format(
                        cursor.date(), chunk_end.date(), tf.label.lower(), why_empty
                    )
                )
            else:
                result.chunks_ok += 1
                frames.append(chunk)

    if not frames:
        return result

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.loc[combined["timestamp"].notna()]
    for c in OHLCV:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
    combined = (combined.drop_duplicates("timestamp", keep="last")
                        .sort_values("timestamp")
                        .reset_index(drop=True))
    combined["ticker"] = sym
    result.frame = combined
    return result
