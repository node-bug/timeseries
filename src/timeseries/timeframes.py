"""The two timeframes this package supports, and every constant that depends on one.

Why this module exists
----------------------
Before this, the package was 1-minute-only and the reasoning for that lived in prose
scattered across four modules: ``fetch.py`` said there was deliberately no interval
selector, ``store.py`` hard-coded 390 bars a session, ``features.py`` calibrated a
180-second hole threshold to 60-second bars, and ``app.py`` quoted 1-minute round-trip
costs in its fee assumptions.  Each of those was individually correct and locally
well-documented, and together they formed a calibration that no single place owned.

That is the failure mode this table exists to prevent.  When a second timeframe
arrives, every one of those numbers either transfers or does not, and the answer is
never obvious at the call site: 390 does not transfer to a daily bar, and neither does
a 180-second hole threshold, but a 20-bar rolling window does.  Getting one wrong is
worse than not supporting the timeframe at all, because it fails *silently* -- a
daily series with the intraday gap rule applied reports every weekend as a hole, the
Quality tab calls a two-decade archive corrupt, and nothing says why.

So the numbers live here, in one place, **next to the reasoning for each one**, and
every other module reads them through :func:`get_timeframe`.  A call site that needs
the hole threshold asks for it; it does not carry a literal of its own.

The two supported timeframes
----------------------------
``1m`` -- the original.  Yahoo serves ~8 days per request and retains ~30 days, which
is why :mod:`timeseries.fetch` chunks a 29-day window.  All of that machinery is
intraday-specific and is bypassed entirely on ``1d``.

``1d`` -- daily bars.  Yahoo has decades of history for these in a *single* request, so
the chunking, the retention wall and ``MAX_1M_DAYS`` are all irrelevant rather than
merely loosened.  A daily bar is one trading day, so ``bars_per_session`` is 1 and the
notion of an "intra-session hole" does not exist; what a daily series can have instead
is a *missing day*, which is a different condition and is reported under a different
key.

Adding a third timeframe means adding a row here and reading nothing else -- except
where a genuine gap remains (see :meth:`Timeframe.validate`), which is the point of
:func:`validate_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Timeframe",
    "TIMEFRAMES",
    "DEFAULT_TIMEFRAME",
    "get_timeframe",
    "timeframe_keys",
    "resolve_timeframe",
    "validate_registry",
]


@dataclass(frozen=True)
class Timeframe:
    """Every constant that depends on the bar interval.

    Frozen so a caller cannot mutate a shared table entry mid-session.  The app holds
    these in Streamlit session state and reads them from several call sites per rerun;
    a mutable entry would let one tab's settings leak into another's.

    Attributes whose meaning is *only* meaningful intraday are typed ``Optional`` and
    are ``None`` on ``1d``, deliberately rather than by default.  A daily bar has no
    sub-session structure, so "the largest intra-session gap allowed" has no correct
    value -- returning 180 for a daily series would make every weekend read as a hole.
    Callers are expected to branch on ``None`` rather than to coerce it.
    """

    #: Short key used everywhere: session state, the fetch API, cache keys.
    key: str

    #: Human label for the UI and for help copy ("1-minute", "Daily").
    label: str

    #: What gets passed to ``yfinance`` as ``interval=``.
    yfinance_interval: str

    #: Filename slug, as in ``QQQ_1min_20260831_20260930.csv``.  Note this is *not*
    #: :attr:`yfinance_interval`: Yahoo spells the daily interval ``1d`` and the
    #: archive spells it ``1d`` too, but the intraday one is ``1m`` on the wire and
    #: ``1min`` on disk.  The two spellings are kept apart here rather than derived at
    #: each call site, because a file written ``QQQ_1m_...csv`` would not match the
    #: 1-minute files already in ``data/``.
    filename_slug: str

    #: Largest gap *inside one trading session* that is still continuous tape, in
    #: seconds.  ``None`` where the concept does not apply (daily: a bar is a whole
    #: day, so there is no interior to have a hole in).
    #:
    #: 180 for ``1m`` is three bars of slack over the 60-second spacing.  A 1-minute
    #: equity bar that is 181 seconds after its predecessor is a dropped print or a
    #: halt, not jitter.
    gap_seconds: Optional[int]

    #: Bars one complete regular session contains.  390 for ``1m`` (09:30-16:00 ET
    #: inclusive); 1 for ``1d``.
    bars_per_session: int

    #: Bars in the rolling z-score base used by :func:`timeseries.features.build_features`.
    #: 20 in both cases, for different reasons: 20 minutes is enough for a stable
    #: rolling standard deviation on intraday noise, and 20 *sessions* is roughly a
    #: trading month, which is the shortest horizon over which a daily rolling
    #: volatility estimate is not obviously degenerate.
    rolling_window: int

    #: Bars in the default query window, and in the forecast chart's history and
    #: projection.
    default_length: int
    forecast_history_bars: int
    forecast_projection_bars: int

    #: Bounds on a window the reader may brush.  See ``MIN_QUERY_BARS`` in ``app.py``
    #: for why the floor is a validity bound and the ceiling is a readability one.
    min_query_bars: int
    max_query_bars: int

    #: Trading sessions the Price tab shows by default.
    view_sessions: int

    #: Forward-return horizons, in bars, that the Forecast and Backtest tabs offer.
    horizons: tuple

    #: Per-bar volatility of a *synthetic* null series at this resolution, as a
    #: fractional log-return standard deviation.
    #:
    #: Not a modelling choice about real instruments -- it is what
    #: :mod:`timeseries.placebo` needs so its random walk has the same order of
    #: magnitude of move as real bars, and the property being protected is the
    #: amplitude term.  A daily null built from the 1-minute figure would be a series
    #: of ~0.01% moves where every window is equally dead, which leaves the amplitude
    #: penalty with nothing to separate and would make the null pass for the wrong
    #: reason.
    #:
    #: 1e-4 is ~QQQ-like 1-minute noise on a $700 base.  8e-3 is a ~1% daily move,
    #: which is the same instrument's realistic daily range, so the two differ by the
    #: roughly 80x that separates a minute from a day.
    synthetic_sigma: float

    #: Days requested per HTTP call.  ``None`` means "one request for the whole span".
    #: The intraday value is Yahoo's ~8-day per-request ceiling minus a day of margin;
    #: daily has no such ceiling, so chunking it would only add failure modes.
    chunk_days: Optional[int]

    #: Widest span Yahoo will serve, in days.  ``None`` means effectively unbounded --
    #: daily history goes back decades, and clamping it to 29 days would silently
    #: throw away 95% of what the endpoint will hand over for free.
    max_days: Optional[int]

    #: Longest prefix of ``key`` any other key may share, so a prefix match is never
    #: ambiguous.  Not enforced here beyond documenting the intent: keys are short and
    #: the set is closed, so the only realistic failure is two keys differing in their
    #: last character (``"1m"``/``"1M"``), which :func:`get_timeframe` would silently
    #: collapse.  That is checked in :func:`validate_registry`.
    @property
    def slug(self) -> str:
        """Filesystem- and cache-key-safe identifier: ``"1m"`` / ``"1d"``.

        Used in ``cache_resource`` keys, where a timeframe has to sit alongside a
        symbol.  ``key`` already satisfies this; the property exists so the cache code
        reads as intent rather than as string concatenation.
        """
        return self.key


#: The original calibration, verbatim.  Every default in the package still points here.
_ONE_MINUTE = Timeframe(
    key="1m",
    label="1-minute",
    yfinance_interval="1m",
    filename_slug="1min",
    gap_seconds=180,
    bars_per_session=390,
    rolling_window=20,
    default_length=240,
    forecast_history_bars=240,
    forecast_projection_bars=240,
    min_query_bars=8,
    max_query_bars=390,
    view_sessions=5,
    horizons=(5, 15, 30, 60),
    synthetic_sigma=1e-4,
    chunk_days=7,
    max_days=29,
)

#: Daily bars.
#:
#: ``gap_seconds`` is ``None`` and that is the load-bearing decision here, not an
#: oversight.  A daily bar spans a whole trading day, so there is no *interior* in
#: which a hole could exist; the only discontinuity a daily series can have is a day
#: with no bar at all, which is a missing session rather than a hole.  Applying the
#: intraday 180-second rule would mark every weekend and holiday as a hole and report
#: a two-decade archive as corrupt.
#:
#: ``default_length`` of 60 is roughly one trading quarter -- long enough to describe a
#: regime, short enough that a "shape" is still recognisable rather than a smear.  It is
#: deliberately much shorter than the 240-bar intraday default in *wall-clock* terms
#: (~3 months vs ~4 hours) because those two answers are not meant to be comparable:
#: the intraday default is "most of one session", the daily default is "one quarter".
#:
#: ``max_query_bars`` of 252 is one trading year, the widest window anyone reads as a
#: shape at all.  ``view_sessions`` of 250 shows about a year of candles, which is
#: where daily candles stop being distinguishable and become a solid band.
_DAILY = Timeframe(
    key="1d",
    label="Daily",
    yfinance_interval="1d",
    filename_slug="1d",
    gap_seconds=None,
    bars_per_session=1,
    rolling_window=20,
    default_length=60,
    forecast_history_bars=60,
    forecast_projection_bars=20,
    min_query_bars=8,
    max_query_bars=252,
    view_sessions=250,
    horizons=(5, 10, 20, 40),
    synthetic_sigma=8e-3,
    chunk_days=None,
    max_days=None,
)

#: Every supported timeframe, keyed by :attr:`Timeframe.key`.  Order is the order the
#: UI offers them in.  Adding a third means adding a :class:`Timeframe` here and
#: nothing else -- except where a genuine gap remains, which :func:`validate_registry`
#: is there to catch.
TIMEFRAMES: dict = {
    "1m": _ONE_MINUTE,
    "1d": _DAILY,
}

#: What a caller that never mentions a timeframe gets.  1-minute is the default because
#: every existing caller, test and cached value in this package was written against it,
#: and changing that default would silently re-base every number the app reports.
DEFAULT_TIMEFRAME = "1m"


def get_timeframe(key: object = DEFAULT_TIMEFRAME) -> Timeframe:
    """The :class:`Timeframe` for ``key``, or the default when ``key`` is ``None``.

    Raises
    ------
    ValueError
        If ``key`` is a non-empty value that is not a supported timeframe.  Accepting
        an unknown key silently would be worse than raising: the alternative is a daily
        series being quality-checked with 390-bars-per-session intraday rules, which
        reports a valid archive as corrupt.
    """
    if key is None:
        key = DEFAULT_TIMEFRAME
    text = str(key).strip().lower()
    if not text:
        text = DEFAULT_TIMEFRAME
    try:
        return TIMEFRAMES[text]
    except KeyError:
        raise ValueError(
            "Unknown timeframe {!r}. Supported: {}.".format(
                str(key), ", ".join(TIMEFRAMES)
            )
        ) from None


def timeframe_keys() -> list:
    """The supported timeframe keys, in UI order."""
    return list(TIMEFRAMES)


def resolve_timeframe(spec: object) -> Timeframe:
    """Resolve anything that names a timeframe to a :class:`Timeframe`.

    Accepts a key (``"1d"``), a :class:`Timeframe` (returned unchanged), or ``None``
    (the default).  This is the lenient entry point for values coming from the UI or
    from session state, where a stored string may have been written by an older
    version of the app and should not raise.
    """
    if isinstance(spec, Timeframe):
        return spec
    if spec is None:
        return get_timeframe(DEFAULT_TIMEFRAME)
    text = str(spec).strip().lower()
    if text in TIMEFRAMES:
        return TIMEFRAMES[text]
    # Tolerate the spellings a reader or a filename might plausibly carry.
    for tf in TIMEFRAMES.values():
        if text in (tf.filename_slug, tf.yfinance_interval, tf.label.lower()):
            return tf
    return get_timeframe(DEFAULT_TIMEFRAME)


def validate_registry() -> None:
    """Assert the table is internally consistent.

    Worth a call in the test suite rather than at import: an inconsistent registry is a
    *programming* error, and failing at import would turn a bad edit into an
    uninstallable package.  The checks that matter:

    * every ``max_query_bars`` leaves room for a window inside a default view;
    * a timeframe with no ``gap_seconds`` cannot also claim sub-session structure,
      which would mean the ``None`` was an oversight rather than a decision;
    * ``chunk_days`` is present exactly when ``max_days`` is, since chunking exists
      only to respect a per-request limit that a bounded span also implies;
    * no key is a prefix of another, because :func:`resolve_timeframe` does a lenient
      match and would otherwise resolve the shorter one for the longer one's input.
    """
    keys = list(TIMEFRAMES)
    for a in keys:
        for b in keys:
            if a != b and b.startswith(a):
                raise ValueError(
                    f"timeframe key {a!r} is a prefix of {b!r}; resolve_timeframe's "
                    f"lenient match would resolve {b!r} as {a!r}"
                )

    for tf in TIMEFRAMES.values():
        if tf.max_query_bars < tf.min_query_bars:
            raise ValueError(
                f"{tf.key}: max_query_bars ({tf.max_query_bars}) below "
                f"min_query_bars ({tf.min_query_bars})"
            )
        if tf.default_length < tf.min_query_bars:
            raise ValueError(
                f"{tf.key}: default_length ({tf.default_length}) below "
                f"min_query_bars ({tf.min_query_bars})"
            )
        if tf.default_length > tf.max_query_bars:
            raise ValueError(
                f"{tf.key}: default_length ({tf.default_length}) above "
                f"max_query_bars ({tf.max_query_bars})"
            )
        if tf.gap_seconds is None and tf.bars_per_session > 1:
            raise ValueError(
                f"{tf.key}: gap_seconds is None but bars_per_session is "
                f"{tf.bars_per_session}; a bar smaller than a session needs a "
                f"hole threshold"
            )
        if (tf.chunk_days is None) != (tf.max_days is None):
            raise ValueError(
                f"{tf.key}: chunk_days and max_days must both be set or both be "
                f"None; chunking exists only to respect the per-request limit that "
                f"max_days also implies"
            )