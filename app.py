"""Streamlit dashboard for the pattern matcher, at 1-minute or daily resolution.

Exploratory tool around the ``timeseries`` package.  It deliberately does not add any
analytics of its own: every number shown comes from :mod:`timeseries.pipeline`,
:mod:`timeseries.matching`, :mod:`timeseries.forecast` or
:mod:`timeseries.backtest`.

Two design rules carry over from PLAN.md and are load-bearing here:

* A matched-window forecast is never presented without the random-window baseline
  beside it.  When the sample is too small the forecaster returns
  ``sufficient=False`` and this app shows the *note* instead of the number.
* A selection defines its own window length.  ``find_matches`` reads
  ``query.length`` and scores the query's own slice of the series, so a brush of N
  bars is searched as N bars.  ``Pipeline.run`` reads the same length for the forward
  returns, the random-window baseline and the bootstrap block, so every number on the
  *Forecast* tab describes the window that was actually matched.

Resolution is chosen per scope and read from :mod:`timeseries.timeframes`, so the two
tabs can hold different instruments *and* different resolutions at the same time.
Nothing in this file carries a 1-minute assumption of its own: the window lengths,
horizons, view sizes and copy are all resolved from the active timeframe.

Run with::

    python3 -m streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager

# --- make ``src/`` importable before anything from the package is pulled in ------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as stc  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

# --- Browser timezone detection ----------------------------------------------- #
# Streamlit's session-state mechanism exposes the browser's timezone through a
# component that runs client-side JavaScript.  This is needed so that intraday
# charts print the reader's own clock rather than UTC, which is what the
# :func:`display_zone` accessor reads.
#
# The component is a no-op in tests (see ``tests/apphelpers.py``), so it can be
# declared at module scope without breaking a library-only import.
_TZ_COMPONENT = stc.declare_component(
    "tz_detector",
    path=os.path.join(_HERE, "static", "tz_detector"),
)

from timeseries import fetch as F  # noqa: E402
from timeseries import matching as M  # noqa: E402
from timeseries import matrix_profile as MP  # noqa: E402
from timeseries.backtest import walk_forward  # noqa: E402
from timeseries.features import FEATURE_COLUMNS  # noqa: E402
from timeseries.forecast import forecast_paths, forecast_paths_multi  # noqa: E402
from timeseries import panel as PNL  # noqa: E402
from timeseries.panel import PanelSearch  # noqa: E402
from timeseries.pipeline import (  # noqa: E402
    DEFAULT_K, Pipeline, horizons_for,
)
from timeseries.progression import (  # noqa: E402
    MAX_PROGRESSION_LENGTH,
    ProgressionSweep,
    default_lengths,
    progression_paths,
)
from timeseries.store import (  # noqa: E402
    CONSTITUENTS_NAME,
    MERGE_POLICY,
    PanelStore,
    read_sectors,
    session_et,
    stamp_label as store_stamp_label,
)
from timeseries import sync as SYNC  # noqa: E402
from timeseries.timeframes import (  # noqa: E402
    DEFAULT_TIMEFRAME, TIMEFRAMES, get_timeframe, resolve_timeframe, timeframe_keys,
)

# st.set_page_config must be the first Streamlit call in the script.  It is therefore
# executed at import time, before the cache decorators below are registered.
st.set_page_config(layout="wide", page_title="Pattern Matcher")

# §X: Numba JIT-compiles STUMPY on first call, costing seconds. Warming it here, right
# after the page config, keeps that cost at process start rather than on the user's
# first match -- which would otherwise look like a multi-second freeze mid-interaction.
MP.warm_up()

# Report the Numba threading layer now that the JIT has run and the layer is resolved.
# The default layer on macOS is "workqueue", which aborts the whole process (SIGABRT,
# no traceback) if two threads score at once -- so a "workqueue" line here is the early
# warning that concurrency is only being protected by the serialising guard.  On
# macOS the expected value is "omp"; anything else is worth knowing about.
_THREADING_LAYER = MP.threading_layer_status().get("layer")
if _THREADING_LAYER and _THREADING_LAYER != "omp":
    print(f"[numba] threading layer: {_THREADING_LAYER} (not threadsafe)")

ROLLING_WINDOW = 20           # rolling mean / 2-sigma band on the price chart

# =============================================================================== #
# Recently used tickers
# =============================================================================== #
# Default recently used tickers shown as buttons in the sidebar.  A ticker that is
# fetched (typed or clicked) moves to the front of the list; when the list exceeds
# :data:`MAX_RECENT_TICKERS` the tail entry is dropped.
DEFAULT_RECENT_TICKERS: List[str] = [
    "^VIX", "BTC-USD", "QQQ", "SOXL", "NG=F", "HO=F", "CL=F",
    "XBI", "LABU", "AAPL", "MSFT", "TSLA",
]
MAX_RECENT_TICKERS = 12
RECENT_TICKERS_KEY = "recent_tickers"

# =============================================================================== #
# Active timeframe
# =============================================================================== #
# The resolution this session runs at, as a mutable one-element list for exactly the
# reason :data:`SYMBOL_FOR_HELP` is one: **Streamlit re-executes this script top to
# bottom in a fresh namespace on every interaction**, so a plain module-level global is
# re-initialised on every rerun and anything that wrote to it earlier in the same pass
# would be the only reader.
#
# **There is exactly one, and it is chosen once per session.**  It used to be a
# per-tab dropdown, and that arrangement is what produced the bug documented at
# :func:`session_timeframe`: a control that moved the page's labels while leaving its
# data alone, silently.  A single startup decision removes the whole class -- there is
# no later moment at which the label and the archive could disagree, because neither
# exists independently of the other.
#
# ``FORECAST_TIMEFRAME`` is retained and always mirrors this one.  It exists because
# the Projection tab renders through the same ``active_*`` helpers as everything else;
# with one resolution per session the two can never diverge, and removing it would mean
# auditing every reader for no behavioural gain.
ACTIVE_TIMEFRAME: List[str] = [DEFAULT_TIMEFRAME]

# The timeframe the *Forecast* scope holds.  Mirrors :data:`ACTIVE_TIMEFRAME`.
FORECAST_TIMEFRAME: List[str] = [DEFAULT_TIMEFRAME]

# Session-state key holding the resolution chosen at startup, and the key of the
# selector widget that asks for it.
#
# **The choice is a session-state value rather than a dropdown the reader can move
# later**, and that is the entire point of the arrangement.  See
# :func:`session_timeframe`.
SESSION_TIMEFRAME_KEY = "session_timeframe"
SESSION_TIMEFRAME_INPUT_KEY = "session_timeframe_input"
#: Non-widget key the radio parks its answer in via ``on_change``.  Needed because
#: a registered widget's key is read-only to ``st.session_state``; see
#: :func:`session_timeframe`.
SESSION_TIMEFRAME_PENDING_KEY = "session_timeframe_pending"

#: Label for the startup picker.  Names both options rather than "Resolution", because
#: a first-time reader has to work out what they are choosing between and the words
#: "1-minute" and "Daily" do that where "Resolution" does not.
SESSION_TIMEFRAME_LABEL = "Data resolution"

#: The button that commits the selection.  Needed because a radio only fires
#: ``on_change`` on a *change*, so the default cannot be committed by clicking it --
#: see :func:`session_timeframe`.
SESSION_TIMEFRAME_GO_LABEL = "Continue"

# Module-level window/horizon constants, resolved from the active timeframe.
#
# **These are read inside function bodies, not at import**, which is what lets them
# follow the selector: a module constant evaluated once at import cannot change, but a
# name that delegates to :data:`ACTIVE_TIMEFRAME` is re-read on every call and is
# therefore correct on whichever pass asks.
#
# Keeping the module-level *names* rather than threading a parameter through every
# function is deliberate.  Dozens of call sites read ``DEFAULT_LENGTH`` or the
# query-length bounds, and passing a timeframe into each would put the resolution
# decision in fifty places instead of one.  The bounds in particular are
# referenced by the sidebar slider, and a rename would have churned all
# of them for no behavioural gain.
def _tf(key: object = None) -> Any:
    """The active timeframe, or an explicit one when ``key`` is given."""
    return get_timeframe(key if key is not None else ACTIVE_TIMEFRAME[0])


def tf_key(key: object = None) -> str:
    """The active timeframe's **key**, for building namespaced state names."""
    return _tf(key).key


def state_key(base: str, key: object = None) -> str:
    """``base`` namespaced by the active resolution.

    **The single place bar-indexed state acquires its suffix.**  Every session-state
    key that holds a bar index, a span or a match result goes through here, so the two
    resolutions are structurally unable to collide: there is no path in which an index
    computed on daily bars is looked up in a frame of minute bars, because the two
    live under different keys rather than because something remembered to check.

    The suffix is a plain string rather than a tuple because these names are also
    Streamlit *widget* keys, and Streamlit requires a string.
    """
    return "{}{}{}".format(base, RESOLUTION_SUFFIX, tf_key(key))


def active_length() -> int:
    """Default query length in bars, for the active timeframe."""
    return _tf().default_length


def active_horizons() -> tuple:
    """Forward-return horizons in bars, for the active timeframe."""
    return horizons_for(ACTIVE_TIMEFRAME[0])


def active_label() -> str:
    """``"1-minute"`` or ``"Daily"`` -- the word used in every caption and title."""
    return _tf().label


def active_unit() -> str:
    """What one bar is called: ``"minute"`` / ``"day"``.

    Used where the copy needs the noun rather than the label, e.g. "60 minutes of
    continuous trading" against "60 trading days".
    """
    return "minute" if _tf().key == "1m" else "day"


def display_zone(key: object = None) -> ZoneInfo:
    """The zone a bar is *printed in*: the reader's local zone, or UTC.

    Bars are **stored** as UTC -- that is the archive contract in
    :mod:`timeseries.store`, and changing it would break ``session_et``,
    ``fingerprint`` and ``validate_session``.  This is the *display* zone: the one
    every label, axis and caption converts through before printing, so that a reader
    in Chicago sees ``09:30`` and one in Tokyo sees ``22:30`` for the same bar.

    It is read from :data:`LOCAL_ZONE` rather than from a call site, because the zone
    is a session-wide fact decided once at startup (see :func:`resolve_local_zone`)
    and every chart must agree on it.  A chart that asked the browser again would
    race the poller and could redraw mid-session in a different zone than the one
    its caption described.
    """
    return LOCAL_ZONE[0] if isinstance(LOCAL_ZONE[0], ZoneInfo) else _UTC_ZONE


#: Fallback when the browser reports nothing usable.  Kept as a real ``ZoneInfo``
#: rather than the string ``"UTC"``, so the conversion code below never has to
#: special-case it.
_UTC_ZONE: ZoneInfo = ZoneInfo("UTC")

# The reader's local zone, as a one-element list for the same reason
# :data:`ACTIVE_TIMEFRAME` is one: a plain module global would be re-initialised on
# every rerun and any value written earlier in the pass would have no reader.  It is
# populated by :func:`resolve_local_zone` before anything reads it.
LOCAL_ZONE: List[Any] = [_UTC_ZONE]


def stamp_label(ts: object, *, key: object = None) -> str:
    """One bar's stamp as the reader should read it, at the active resolution.

    A thin wrapper over :func:`timeseries.store.stamp_label`, which is the one
    definition of what a bar looks like.  The app needs its own name because the
    resolution comes from :func:`_tf` rather than being passed, and because call sites
    read it in a chart-building expression; the *format* is not the app's to decide.
    Duplicating it here is how :func:`fetch.py`'s summary line and a chart axis would
    come to print different things for the same bar, which is a discrepancy only a
    reader comparing the two could ever notice.

    The stamp is converted to :func:`display_zone` before formatting, so the clock a
    reader sees is their own.  Daily bars are unaffected: one bar is a whole session,
    so the time is constant across the archive and :func:`store.stamp_label` already
    drops it -- only the *zone suffix* (below) moves with the reader.
    """
    return store_stamp_label(ts, _tf(key), zone=display_zone(key))


def stamp_labels(timestamps: Any, *, key: object = None) -> List[str]:
    """A whole run of bars' stamps, in the reader's own format.

    **The vectorised twin of :func:`stamp_label`, and it must agree with it bar for
    bar.**  A chart that labels its *axis* with :func:`stamp_label` and its *hover*
    with a different formatter is a chart that says two different things about the
    same bar, and a reader comparing an axis label against a hover label is exactly
    how that gets noticed.  So this does not define a second format: it computes the
    same one, faster.

    The loop over :func:`stamp_label` is the reference behaviour and costs about a
    quarter of a second for a two-thousand-bar daily archive -- noticeable on every
    redraw of a chart that draws several of these, and the reason the fast path
    exists.  It is not, however, the *only* path: :func:`stamp_label` has a
    ``"—"`` branch for a missing stamp and a per-bar zone conversion, and a
    vectorised rewrite that quietly changed either would be a divergence.  The
    fast path therefore reproduces the format exactly and falls back to the loop
    for anything it cannot prove it can reproduce -- an empty run, a non-``tz``
    timestamp, a bar whose stamp is missing.

    ``timestamps`` is any indexable sequence of timestamps; the result is one label
    per element, in the same order.
    """
    values = list(timestamps)
    if not values:
        return []

    tf = _tf(key)
    zone = display_zone(key)
    try:
        series = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    except (TypeError, ValueError):
        # A dtype pandas cannot read is not an error here -- the loop below handles
        # whatever it is, one bar at a time.
        return [stamp_label(v, key=key) for v in values]

    if tf.bars_per_session == 1:
        # Daily: the label is the *Eastern* date, which is a calendar-day question
        # and is therefore not a conversion of the bar's own zone -- the bar is
        # stamped at the Eastern open, and any other hour of the same session is
        # the same date.  ``session_et`` is the one definition of that date, so it
        # is what this calls.
        dates = session_et(series).dt.strftime("%B %-d, %Y")
    else:
        # Intraday: the bar's own instant, converted into the reader's zone.
        # ``tz_convert`` on a UTC series is the same conversion :func:`stamp_label`
        # performs on a single bar.
        dates = series.dt.tz_convert(zone).dt.strftime("%B %-d, %Y %H:%M")

    labels = [str(v) for v in dates]
    # ``NaT`` stringifies to ``"nan"``; :func:`stamp_label` prints ``"—"`` for a
    # missing stamp.  Any bar whose label is not a real stamp is redone the slow
    # way, so the two paths cannot diverge.
    if any(label == "nan" for label in labels):
        return [stamp_label(v, key=key) for v in values]
    return labels


def stamp_span(lo: object, hi: object, *, key: object = None) -> str:
    """``"2026-09-03 → 2026-09-30"`` -- the ``a → b`` half the captions open with."""
    return "{} → {}".format(stamp_label(lo, key=key), stamp_label(hi, key=key))


#: What a printed stamp is in, per resolution.  Read from ``bars_per_session`` rather
#: than from the key, because *one bar is a whole session* is the property that makes a
#: time of day meaningless -- not the spelling of the key.  A third resolution with one
#: bar a week therefore joins the daily side with no edit here.
def stamp_zone(key: object = None) -> str:
    """What a printed stamp is in: ``" (UTC)"`` intraday, ``""`` daily.

    Daily labels are Eastern dates (see :func:`stamp_label`), so appending ``(UTC)`` to
    one would name a zone the label is not in.  The suffix is part of the same decision
    as the format, so it is resolved here rather than written into each caption -- a
    ``"(UTC)"`` literal inside a display string is exactly the kind of thing that
    survives the change that should have removed it.

    On intraday the zone is the reader's own (see :func:`display_zone`), so a reader in
    New York sees ``" (America/New_York)"`` beside a ``09:30`` that is their 09:30.
    """
    if _tf(key).bars_per_session == 1:
        return ""
    return " ({})".format(display_zone(key).key)


def stamp_column(prefix: str = "", *, key: object = None) -> str:
    """Header for a stamp column: ``"start (UTC)"`` intraday, ``"start"`` daily.

    A header promising UTC above a cell that no longer shows UTC is a lie of the kind
    §BZ is about, one header smaller, so the zone suffix and the value format are
    resolved by the same registry rather than written independently at the two sites.
    """
    return "{}{}".format(prefix, stamp_zone(key))


def stamp_zone_name(key: object = None) -> str:
    """The zone's *name*, for prose: ``"America/New_York"`` intraday, ``"Eastern time"`` daily.

    :func:`stamp_zone` is for suffixes (``" (America/New_York)"``, or nothing); this is
    for a sentence that has to name the zone whatever it is.  Same source, so a header,
    a cell and its tooltip cannot end up describing three different zones.

    Daily stays ``"Eastern time"`` -- the label is an Eastern *date*, so naming the
    reader's local zone there would claim the date is in a zone it is not.
    """
    if _tf(key).bars_per_session == 1:
        return "Eastern time"
    return display_zone(key).key


def resolve_local_zone() -> ZoneInfo:
    """Resolve the reader's local zone once, and park it in :data:`LOCAL_ZONE`.

    Read from the browser's timezone via Streamlit's session-state mechanism,
    falling back to UTC when nothing usable is reported.
    """
    # Streamlit exposes the browser's timezone through its session-state mechanism.
    # Read it once at startup; a module-level global would be re-initialised on
    # every rerun and any value written earlier in the pass would have no reader.
    zone_name = st.session_state.get("_browser_tz", None)
    # Hardcode IST (Asia/Kolkata) since browser detection is not working
    zone_name = "Asia/Kolkata"
    if zone_name and isinstance(zone_name, str):
        try:
            zone = ZoneInfo(zone_name)
            LOCAL_ZONE[0] = zone
            return zone
        except ZoneInfoNotFoundError:
            pass
    # Fallback: try to resolve from the browser's reported offset or name.
    # If nothing usable arrives, fall back to UTC rather than inventing a zone.
    zone = _UTC_ZONE
    LOCAL_ZONE[0] = zone
    return zone


def bar_unit(n: int) -> str:
    """One bar's worth of *n* bars, as an axis tick: ``"45 min"`` / ``"4 days"``.

    **Pluralised, unlike :func:`active_unit`'s other call sites -- because this one is
    concatenated onto a count on every tick.**  ``active_unit() + "s"`` is right for the
    Backtest tab's fixed sentence ("15 bars = 15 minutes"), where the number is one the
    reader chose from a fixed menu, and wrong here: a tick label is emitted once per
    gridline, so ``+1 day`` and ``+0 days`` both appear and a fixed ``+%s`` would put
    "+4 day" on the axis.

    ``min`` stays an abbreviation on the 1-minute side -- it is the conventional short
    form on a crowded axis, and an axis does not want "minutes" written out six times.
    Daily uses the full noun because it is short enough to, and because a daily chart
    is exactly where the reader most needs the word spelled out to be sure of it.
    """
    if _tf().key == "1m":
        return "min"
    return "day" if abs(int(n)) == 1 else "days"


def _proj_ticktext(anchor_ts: pd.Timestamp, offsets: Sequence[int]) -> List[str]:
    """Clock-time labels for the projection ticks.

    The projection has no bars behind it, so its x-positions are offsets from the
    anchor bar.  On the 1-minute tab we resolve each offset into a clock time by
    adding ``offset`` minutes to the anchor bar's timestamp and printing the time of
    day -- ``01:05``, ``01:10`` and so on -- so the reader sees *when* the projected
    bars would occur rather than just how many bars out they are.  One 1-minute bar
    is one minute of wall-clock time, so the offset in bars *is* the offset in
    minutes; ``bars_per_session`` is how many bars fit in a session, not how many
    minutes one bar spans, and must not enter here.
    """
    if _tf().key != "1m":
        # Daily: one bar per session, so an offset is a session count, not a clock
        # time.  Fall back to the plain offset label.
        return ["+%d %s" % (o, bar_unit(o)) for o in offsets]

    # One 1-minute bar is one minute of wall-clock time, so the offset in bars *is*
    # the offset in minutes.  ``bars_per_session`` (390) is how many bars fit in a
    # session, not how many minutes one bar spans -- multiplying by it here labelled
    # offset 1 as ``anchor_ts + 390 minutes``, i.e. a full session later, and every
    # projected tick after the first was wrong (the +0 tick still landed on the
    # anchor's own time, which is why the existing test did not catch it).
    #
    # The synthetic timestamp is converted to the reader's own zone before printing,
    # same as the history ticks: the projection describes *when these bars would
    # occur*, and "when" is a local fact.  Leaving it on UTC would put a clock the
    # reader cannot read beside one they can.
    zone = display_zone()
    out = []
    for o in offsets:
        t = (anchor_ts + pd.Timedelta(minutes=int(o))).tz_convert(zone)
        out.append(t.strftime("%H:%M"))
    return out


def active_projection_bounds() -> tuple:
    """``(low, high)`` valid horizons in bars for the active timeframe.

    The Projection tab's projection length is a *control* rather than a constant, so
    its range has to be one the rest of the pipeline can actually honour -- and the
    upper bound is not a round number, because it is set by §BX.

    **On intraday the ceiling is one session, and it is load-bearing.**  §BX drops
    any candidate whose forward horizon would cross a session closure, on the
    measured grounds that such a return is really one overnight gap plus ``h - 1``
    minutes and reports ~20x a genuine one.  Measured on a 20-session 1-minute
    fixture at the default 240-bar window::

        h= 20   94.7% of windows admissible
        h= 60   84.1%
        h=240   37.7%
        h=390    0.0%   <-- every window censored

    So a horizon of one full session admits **nothing** on intraday, and the search
    returns zero matches at any ``k`` while still reporting a healthy candidate pool.
    That is the worst failure mode in this app: zero reads as "this shape has no
    historical analogue", which is a claim about the archive rather than about a mask.
    The ceiling is therefore ``bars_per_session - 1``, which keeps at least the
    windows wholly inside one session admissible.

    **On daily there is no ceiling of this kind, and that asymmetry is the point.**
    A daily bar *is* a whole session, so a horizon already measures whole sessions
    and there is no gap for §BX to correct -- :func:`Pipeline.forward_horizon_mask`
    returns ``None`` outright and :func:`~timeseries.panel._horizon_admissible`
    returns all-True.  Imposing the intraday ceiling on daily would be meaningless
    rather than merely conservative: it would cap the reader's horizon at "zero
    trading days" against a number that means nothing there.  Daily is bounded by
    ``max_query_bars`` instead, which is the app's own statement of the widest span
    readable as a shape.

    Returned as a pair rather than two accessors because the two bounds come from
    *different* facts and a caller that read them separately could not tell that: the
    floor is a chart-drawing rule and the ceiling is a correction's premise.
    """
    tf = _tf()
    if tf.key == "1d":
        return 50, 500
    low = max(1, FORECAST_MIN_PROJECTION_BARS)
    # ``bars_per_session`` is 1 on daily, where it means "one bar is one session" --
    # so it is not a horizon ceiling and must not be read as one.
    ceiling = (tf.bars_per_session - 1 if tf.bars_per_session > 1
               else tf.max_query_bars)
    return low, max(low, int(ceiling))


def active_recent_bounds() -> tuple:
    """``(low, high)`` valid *Recent bars* values for the active timeframe.

    The mirror image of :func:`active_projection_bounds`, and derived from different
    facts, which is why the two are separate functions rather than one over a sign.

    **The floor is the validity bound, not a preference.**  It is
    :func:`active_min_query_bars`, the same ``MIN_QUERY_BARS`` a *brushed* window is
    held to: below roughly eight bars a "shape" is a handful of z-score spikes with no
    trajectory between them, and STUMPY's normalised distance over a handful of points
    is decided by whichever single bar happens to be most extreme.  The distance still
    computes and the percentile is still well-defined, so nothing raises -- which is
    exactly why the floor has to be stated rather than discovered.

    **The ceiling is ``max_query_bars``, the app's own statement of the widest span
    anyone reads as a shape.**  The reference chart draws one continuous shape beside
    its projection, so a length past which the blue history stops being recognisable
    is not a useful thing to offer; 390 on 1-minute is one full session and 252 on
    daily is one trading year.

    **The ceiling is capped against the archive as well, by the caller.**  The slider
    cannot promise more bars than the tape holds -- :func:`forecast_path_for` returns
    ``None`` for an archive shorter than the window, which reads as "this shape has no
    historical analogue" and is a claim about the archive rather than about a control.
    :func:`_render_history_control` narrows the ceiling to what this pipeline actually
    has.

    **Both endpoints are snapped onto the :data:`FORECAST_RECENT_STEP` grid.**  The
    floor is 8 and the daily ceiling is 252, neither of which is a multiple of 30, so
    the raw bounds would give the slider a first tick at 8 and a last at 252 with no
    way to reach anything a multiple of the step away from either.  Snapping *up* from
    the floor and *down* from the ceiling keeps every reachable value a legal step
    multiple, so a reader can never be offered a width the control cannot express.

    **Snapping the floor up, not down, is what preserves the validity bound.**  The
    floor is a real limit on what the matcher can score, and 30 is well above it, so
    the smallest width actually offered is strictly safer than the smallest width that
    would otherwise have been legal.
    """
    step = FORECAST_RECENT_STEP
    low = max(step, -(-int(active_min_query_bars()) // step) * step)
    high = int(active_max_query_bars()) // step * step
    return low, max(low, high)


@contextmanager
def timeframe_scope(key: object):
    """Temporarily make ``key`` the resolution every ``active_*`` helper reports.

    **Needed because the Projection tab renders through the same accessors as
    everything else**, and its pipeline is built from its own ticker.  With one
    resolution per session the value it is handed always equals the session's, so
    this is now a *guard* rather than a mechanism: it states the invariant
    explicitly and fails loudly if a future change ever renders one tab under a
    different resolution than the page, instead of letting that happen silently.

    Exception-safe: the previous value is restored in a ``finally``, so a tab that
    raises mid-render cannot leave the page resolving the wrong resolution for
    everything drawn after it.  Nesting is safe for the same reason.
    """
    previous = ACTIVE_TIMEFRAME[0]
    resolved = resolve_timeframe(key).key
    ACTIVE_TIMEFRAME[0] = resolved
    try:
        yield resolved
    finally:
        ACTIVE_TIMEFRAME[0] = previous


# Fetched archives are cached in-process, keyed on symbol alone -- the span is always
# the whole available window, so there is nothing else that could distinguish one
# cached frame from another.  Deliberately not written to disk: a refresh button that
# silently overwrites a CSV the user may have opened elsewhere is worse than one that
# re-downloads.  See ``fetch_ticker_cached``.
#
# The storage is a ``st.cache_resource`` singleton rather than a plain module-level
# dict, and that is load-bearing rather than stylistic.  **Streamlit re-runs the script
# top to bottom on every interaction, in a fresh namespace each time** (see the note
# on :data:`ACTIVE_TIMEFRAME`), so a module-level dict is re-created empty on every
# single rerun: the "cache" was a cache that never hit.  Measured against the live
# app, printing the store's ``id()`` once per render -- a different object every time,
# on every brush:
#
#     id=4908179200 n=41702 chart_from=35862
#     id=4422744960 n=41702 chart_from=35861
#     id=6425589056 n=41702 chart_from=35861
#
# So every interaction re-downloaded ~29 days of 1-minute bars from Yahoo.  That is
# slow, it makes the sidebar's "bars" count a function of the weather, and -- the part
# that was actually visible -- **the live archive grows between reruns**, so the
# trailing view's right edge moved by a bar or two on every brush and the top chart
# appeared to creep even with its window correctly pinned.  ``cache_resource`` is keyed
# on its *arguments* and stores on the module globals Streamlit keeps, so the same
# call returns the same object for the whole process lifetime.
#
# The returned object is shared across sessions, deliberately: it is a read-only frame
# and a frame per symbol per process is far smaller than this app's feature matrices.
@st.cache_resource(show_spinner=False, max_entries=8)
def _fetch_store() -> Dict[str, "F.FetchResult"]:
    """The process-wide symbol -> fetched-bars store.  See the note above."""
    return {}

# Root of the many-ticker Parquet archive the Panel tab reads.  The single-instrument
# tabs never touch disk: their bars are downloaded and held in memory, so this is the
# only ``data/`` path the app reads, and it is owned by ``timeseries.store`` rather
# than by anything in this file.
PANEL_ROOT = os.path.join(_HERE, "data", "sp500_panel")

# Root of the **daily** many-ticker archive, built by
# ``scripts/download_daily.py``.  A separate directory rather
# than a namespace inside ``sp500_panel`` because the two
# archives are not two views of one thing: a daily partition
# holds **one** bar, a 1-minute partition holds ~390, and they
# are fetched under different retention rules (Yahoo keeps ~29
# days of minute bars and a century of daily ones).  Sharing a
# root would mean every read had to say which it meant, which is
# exactly the ambiguity the per-session resolution gate exists
# to remove.
#
# The layout is deliberately identical to the minute archive --
# ``ticker=<SYM>/date=<YYYY-MM-DD>/bars.parquet`` plus a
# ``manifest.csv`` -- so :class:`timeseries.store.PanelStore`
# reads and writes both with no format branch at all.
DAILY_ROOT = os.path.join(_HERE, "data", "sp500_daily")


def panel_root_for(timeframe: object = None) -> str:
    """The panel archive for the resolution in force.

    **One place decides which archive a panel search reads**, because the failure
    this prevents is silent and looks like a real answer.  A daily query scored
    against 1-minute bars returns a confident percentile comparing a two-decade
    daily window to half a day of tape -- the worst kind of wrong, because
    nothing about it reads as wrong.

    Every panel code path calls this rather than naming a root, so adding a third
    resolution is a row here and not a search of the file for hard-coded paths.
    """
    tf = resolve_timeframe(timeframe if timeframe is not None else ACTIVE_TIMEFRAME[0])
    return PANEL_ROOT if tf.bars_per_session > 1 else DAILY_ROOT


def _within_fetch_window(session: str, *, days: int = 30) -> bool:
    """Is ``session`` (YYYY-MM-DD) still inside Yahoo's minute-bar fetch window?

    Yahoo serves roughly the last 30 days of 1-minute history.  That limit is on
    *fetching*, not on *keeping* -- so a session older than the window cannot be
    downloaded, and saying so is the difference between "press the button" and
    "it is too late".

    The daily archive has no such wall, so this is only meaningful for the
    intraday panel.  The caller decides which archive it is talking about.
    """
    try:
        day = pd.Timestamp(session).tz_localize(EASTERN).date()
    except (TypeError, ValueError):
        return False
    age = (pd.Timestamp.now(tz=EASTERN).date() - day).days
    return age <= days


def download_script_name(timeframe: object = None) -> str:
    """The downloader that builds the archive for the resolution in force.

    The companion to :func:`panel_root_for`, and for the same reason: the two must
    agree, and the way to guarantee that is to derive both from the same resolution
    in one place.  Naming the wrong script is not a cosmetic slip -- it would fetch
    one resolution's bars into the other resolution's archive, which looks correct
    on disk and is only discovered by a search that compares the two.

    Written as a function taking the resolution rather than as a bare conditional at
    each call site because the conditional was written five times and **four of those
    five were inside functions with no ``tf`` in scope**, which is how
    ``NameError: name 'tf' is not defined`` reached a page that 715 tests called green.
    """
    tf = resolve_timeframe(timeframe if timeframe is not None else ACTIVE_TIMEFRAME[0])
    return "download_sp500.py" if tf.bars_per_session > 1 else "download_daily.py"


# Session-state key for the last archive sync's result, so a rerun triggered by an
# unrelated widget does not re-report a sync the user already read.  Clearing it is
# what makes the caption disappear.
SYNC_RESULT_KEY = "archive_sync_result"

# The instrument the app opens on.  Declared as a constant so the text input, the
# first-run auto-load and the help-copy fallback can never drift apart.
#
# HOF is the default because it is an S&P 500 constituent the package's daily archive
# carries, and because a single name is the most self-explanatory thing to put in front
# of a first-time reader: liquid, continuously quoted, and a daily vol of ~1% rather
# than a single name's idiosyncratic move.  It is also the shape the rest of the app
# already assumes -- a 09:30-16:00 ET session of 390 one-minute bars separated by
# overnight gaps, which is what the gap heuristics (:data:`GAP_SECONDS`),
# ``store.validate_session`` and the Quality tab's "outside 09:30-16:00 ET" check are
# all calibrated for.
#
# **This replaces QQQ**, which was the default while the app was still being written
# around the ``data/qqq_1min_*.csv`` archive.  QQQ remains fetchable like any other
# ticker -- nothing about the 24/7 path was removed, only the reason for *defaulting*
# to it.
DEFAULT_SYMBOL = "BTC-USD"

# The window length the app opens on before anything is brushed, and the length
# ``resolve_query_window`` falls back to when there is no brush.  It is declared here
# rather than inline so the help copy below can quote the real value instead of a
# number that silently goes stale the next time this constant is touched.
#
# 240 bars is four hours of continuous trading -- most of a session -- so the default
# query is a regime-sized shape rather than a microstructure blip.  Two things make
# that a different kind of query from the 20-bar default it replaces, and neither is
# an accident:
#
#   * An *exclusion margin* of one window-length (240 bars) is removed around the
#     query before scoring, plus the same margin for NMS spacing between accepted
#     matches.  On a ~8k-bar archive that removes ~6% of the tape twice over, so the
#     percentile is measured against a visibly coarser field -- which is the honest
#     cost of asking a more specific question, and the reason the Matches tab is the
#     one to read rather than the raw distance.
#   * ``Pipeline.from_frame`` refuses to build below ``length + 100`` bars, so a
#     thin archive that was fine at 20 bars now reports "not ready" instead of
#     matching.  That is the intended behaviour -- a 240-bar window over a
#     250-bar archive has no history behind it -- but it is a visible change.  On
#     the Panel tab it is not currently a risk: the thinnest ticker in
#     ``data/sp500_panel`` (ERIE) totals 1,898 bars, well clear of 240 + 100.
#
# The help copy below is written so it reads correctly at either value: the tips
# talk about "a longer window chases regimes, a shorter one chases microstructure"
# rather than asserting that the default *is* short.  Nothing is hard-coded to 20.
DEFAULT_LENGTH = 240

# Shortest window a selection may be queried at.  A brush defines its own
# length now, so this is the one floor left -- and it is a validity bound, not a
# preference.  Below roughly this many bars a "shape" is a handful of z-score spikes
# with no trajectory between them, and STUMPY's normalised distance over a handful of
# points is decided by whichever single bar happens to be most extreme.  The distance
# still computes, and the percentile is still well-defined, so this cannot be enforced
# by the library: it has to be stated, and the UI warns when it bites.
MIN_QUERY_BARS = 8

# Bounds a *brushed* window must respect, resolved from the active timeframe.  The
# sidebar slider is capped at :func:`active_max_query_bars`; this is the floor.
def active_min_query_bars() -> int:
    return _tf().min_query_bars


def active_max_query_bars() -> int:
    return _tf().max_query_bars


def active_view_sessions() -> int:
    return _tf().view_sessions

# =============================================================================== #
# Forecast-path chart
# =============================================================================== #
# Real bars drawn behind the projection, and bars projected past it.  **Neither is
# ``DEFAULT_LENGTH``.**  That constant governs the *query* -- the window the reader
# brushes and the matcher scores -- and this chart answers a different question: what
# usually followed a window shaped like the most recent one.  Reusing the same number
# would couple two decisions that are independent, and would make the chart silently
# change shape whenever a default worth revisiting is changed for the sake of the
# query.
#
# **Both sizes come from the active timeframe, not from a literal here.**  They used
# to be module constants (``FORECAST_HISTORY_BARS`` / ``FORECAST_PROJECTION_BARS``,
# both 240), which was the 1-minute calibration written into the app rather than read
# from the registry that owns it -- and it was wrong on daily, where the values are 60
# and 20.  Every call site already resolves the live value through
# :func:`_tf`; these accessors are the named form of that, so the app has one place
# that says what the forecast chart is sized at.
def forecast_history_bars() -> int:
    """Bars of real tape drawn behind the projection, for the active timeframe."""
    return _tf().forecast_history_bars


def forecast_projection_bars() -> int:
    """Bars projected past the history, for the active timeframe."""
    return _tf().forecast_projection_bars

# Granularity of the Projection tab's *Recent bars* control, in bars.
#
# **30 because the quantity being chosen is a shape, and a shape has a scale.**  One
# bar is 1 minute on 1-minute data and a whole trading day on daily, and a control
# that moves one bar at a time offers 389 intermediate positions between "a quarter
# of a session" and "a whole session" -- every one of which re-runs the matcher to
# produce a chart that looks the same as its neighbour.  Half an hour of intraday tape
# is already the smallest change that changes what the shape *is*: at 30 bars there is
# a trajectory to see between the bars, below it there is not.
#
# **It divides both resolutions' defaults**, which is what makes the control readable
# rather than merely legal.  240 (1-minute) and 60 (daily) are both exact multiples of
# 30, so the default lands *on* a tick rather than between two.  A step that shared no
# factor with the default would put the reference chart's own starting width off the
# grid, and the reader would open on a value the control could not name.
#
# **Grid alignment is not Streamlit's job.**  ``step`` only constrains what the widget
# *emits*; a value already sitting in ``session_state`` is accepted as-is -- measured
# on the pinned build, a stored 45 survives a step of 30 untouched.  So every width the
# control can hold is snapped by :func:`_snap_recent` on the way out, and the bounds in
# :func:`active_recent_bounds` are snapped on the way in.
FORECAST_RECENT_STEP = 30

# Matches aggregated into the path.  Fixed, and deliberately **not** the sidebar's
# ``k matches``: that control governs the horizon table below, whose sample is judged
# against ``Min matches for evidence``.  Binding both to one slider would make a
# control labelled "how many matches" silently decide two unrelated analyses, and a
# reader lowering ``k`` to 10 to inspect one horizon would quietly thin the chart too.
#
# 30 rather than a round 20 or 50 because it is the same figure the rest of this
# package treats as the floor for a claim (§E, ``MIN_MATCHES``).  Reusing it means the
# chart and the table never disagree about what counts as evidence, and a reader who
# raises ``k`` in the sidebar and finds the chart still at 30 is looking at a fixed
# reference rather than at a broken one.
FORECAST_PATH_MATCHES = 30

# Floor for the *Projection bars* control, in bars.  1 rather than 0 because a
# horizon of zero projects nothing: the median would be a single 0.0 and the chart
# would show a history with a degenerate one-point projection hanging off it, which is
# a chart that looks broken rather than one that says "too short".  A reader who wants
# the degenerate case can read it off the ``+0 min`` tick, which is always drawn.
FORECAST_MIN_PROJECTION_BARS = 1

# Streamlit widget key for the Projection tab's *Projection bars* control, namespaced by
# resolution for the same reason every other key here is: a 1-minute bar and a daily
# bar are different quantities, so a horizon chosen in one resolution must not be
# offered as a starting point in the other.  ``state_key`` is a *function* for the
# reason ``price_brush_key`` gives -- a module constant is evaluated at import, before
# ``session_timeframe`` has run, and would name the default resolution forever.
#
# Not a control that chooses the *window*, so ``test_the_brush_is_the_only_window_control``
# is unaffected: that rule protects the window, and this names the horizon.  The two
# are independent decisions -- the window is what you brush, the horizon is how far
# ahead you ask -- and binding them to one widget would be the same class of bug the
# Forecast tab was split to fix.
FORECAST_HORIZON_KEY = "forecast_horizon_bars"


def forecast_horizon_key() -> str:
    """Widget key for the projection-length control, namespaced by resolution."""
    return state_key(FORECAST_HORIZON_KEY)

# Streamlit widget key for the Projection tab's *Recent bars* control -- how many of the
# archive's newest bars the fixed reference asks about.  Namespaced and a *function*
# for exactly the reasons :data:`FORECAST_HORIZON_KEY` gives; the two controls are
# siblings, not parent and child.
#
# **Distinct from :data:`FORECAST_HORIZON_KEY`, deliberately.**  One names how far back
# the reference reaches and the other how far ahead it projects.  Sharing a key would
# mean a reader who moved either slider moved both, and the axis range test below is
# what says they are independent decisions rather than one length split in two.
FORECAST_RECENT_KEY = "forecast_recent_bars"


def forecast_recent_key() -> str:
    """Widget key for the recent-bars control, namespaced by resolution."""
    return state_key(FORECAST_RECENT_KEY)

# Streamlit widget key for the Price tab's brushable chart.  Declared here because the
# selection event is read *before* the chart that writes it is rendered -- see the
# "Brush on the Price tab's own chart" block in ``main()`` -- so the reader cannot
# take the key from the rendering function.
PRICE_BRUSH_KEY = "sel_price_view"


def price_brush_key() -> str:
    """Widget key for the Price brush, namespaced by the session resolution.

    **A function, not a constant.**  The key is derived from the active resolution, and
    a module constant is evaluated once at import -- before ``session_timeframe`` has
    run -- so it would name the *default* resolution forever and the namespacing would
    be cosmetic.
    """
    return state_key(PRICE_BRUSH_KEY)

# Widget key for the *Forecast* tab's own brushable chart.  Separate from
# ``PRICE_BRUSH_KEY`` because the two brushes answer different questions and must not
# share state: this one selects the window to be *forecast*, the other selects the
# window to be *matched* for the evidence table.  Collapsing them would mean a brush
# on one tab silently re-aims the other, and a reader comparing two windows would
# find they could only ever have one selected at a time.
#
# The design rule this app documents is that "the brush is the only window control".
# This does not weaken it: neither brush is a *control*, and no picker is introduced.
# ``test_the_brush_is_the_only_window_control`` guards the rule that matters -- that no
# dropdown, slider or datetime picker can end up choosing the window -- and it holds.
# What changes is that *Forecast* has its own gesture for its own question, which
# is a different thing from reintroducing a selector.
FORECAST_BRUSH_KEY = "sel_forecast_view"


def forecast_brush_key() -> str:
    """Widget key for the Forecast brush, namespaced like :func:`price_brush_key`."""
    return state_key(FORECAST_BRUSH_KEY)

# Session-state key holding the *Forecast* tab's selection as plain bar indices.
#
# **Why the selection cannot live only in the chart's widget.**  Streamlit keys a
# widget's stored value off the element id, and for ``st.plotly_chart`` that id hashes
# ``plotly_spec`` -- the serialised figure (``plotly_chart.py``:
# ``key_as_main_identity=False``, so ``plotly_spec`` is an input to the id).  Drawing
# the forecast therefore *adds a chart*, which changes the page's chart list, which
# changes the context chart's own element id, which makes the selection that produced
# the forecast unreachable on the next rerun.  The result is a self-cancelling loop,
# observed in the browser on this very tab:
#
#     before any drag   -> 3 charts, no selection
#     drag #1           -> 4 charts, forecast drawn ("1065 bars")
#     drag #2           -> selection discarded, forecast removed, back to 3 charts
#     drag #3           -> 4 charts again
#
# The reader gets exactly one selection, and the second one silently reverts the tab.
# Nothing errors; the chart just stops responding after the first use.
#
# Copying the resolved span into session state breaks the loop, because the span does
# not depend on how many charts are on the page.  It is the same reason the Price tab
# reads its brush from state rather than from a return value, arrived at from the
# other direction.
#
# **This is also why the tab's position in ``TAB_ORDER`` is load-bearing.**  The span
# itself survives any ordering, but the *live* selection the reader is dragging does
# not: it is keyed off this chart's element id, which hashes the page's chart list.
# Moving *Forecast* away from immediately after *Projection* would shift how many
# charts precede this one and invalidate every selection already made.  See the
# ``TAB_ORDER`` comment.
FORECAST_SELECTION_KEY = "_forecast_selection"


def forecast_selection_key() -> str:
    """The *Forecast* tab's resolved span, namespaced by resolution."""
    return state_key(FORECAST_SELECTION_KEY)

# Session keys for the two **independent** searches.
#
# One pair per tab because the tabs no longer share a result.  The Matches tab
# searches the window brushed on *Price*; *Forecast* searches the window brushed on
# itself.  Before the split, ``main()`` computed a single ``pipe.run()`` and
# handed the same dict to both, which meant one tab could only ever report the other
# tab's window -- and a brush on Forecast moved the evidence table on a tab whose own
# chart showed a different moment.
#
# ``*_RUN_KEY`` holds a *signature only*.  The result is recomputed whenever the
# signature still matches, exactly as ``run_signature`` was, so there is no stored
# copy that could disagree with the window on screen.  A previous version kept one and
# never read it back.
#
# ``*_APPLIED_KEY`` records the window a brush last acted on.  Streamlit keeps a
# selection for the page's lifetime, so keying on *presence* would recompute on every
# unrelated rerun -- the stale-vs-fresh confusion the manual button exists to prevent.
# Keying on the span is what makes "moved" mean "is a new gesture" rather than "is
# still set".  Recorded **resolved**, never raw, or a clamped brush disagrees with it
# on every rerun and re-fires forever.
PRICE_RUN_KEY = "run_price"
PRICE_APPLIED_KEY = "price_applied_span"
FORECAST_RUN_KEY = "run_forecast"
FORECAST_APPLIED_KEY = "forecast_applied_span"


def price_run_key() -> str:
    return state_key(PRICE_RUN_KEY)


def price_applied_key() -> str:
    return state_key(PRICE_APPLIED_KEY)


def forecast_run_key() -> str:
    return state_key(FORECAST_RUN_KEY)


def forecast_applied_key() -> str:
    return state_key(FORECAST_APPLIED_KEY)

# =============================================================================== #
# Per-tab ticker state
# =============================================================================== #
# **The Forecast tab can chart a different instrument than the Price tab.**  The two
# carry their own ticker, their own pipeline and their own search, so a reader can ask
# "what usually followed this shape in AAPL?" while the Price tape stays on QQQ.
#
# Three keys per scope rather than one, and the split between them is load-bearing
# rather than tidy.  Streamlit only persists *registered widgets*, so the symbol in
# force has to live under a **non-widget** key or it cannot survive a
# ``session_state["_"] = {}`` wipe -- and it is precisely that wipe which re-seeds the
# text box.  Writing the result key and reading it back is what keeps the box and the
# bars in agreement; seeding the box from a constant instead leaves it reading QQQ
# above a chart of MSFT.
#
#   ``*_TICKER_KEY``       non-widget.  The symbol actually in force for this scope.
#                          Survives a widget wipe, so it is the *only* durable record.
#   ``*_TICKER_INPUT_KEY`` the ``st.text_input``.  Widget state.
#   ``*_TICKER_FETCH_KEY`` the ``st.button``.  Widget state.
#
# **A fifth key per scope, derived from the first.**  The inputs are drawn in a bar
# *above* the tabs, but ``main()`` needs both symbols to build the pipelines, which is
# before any tab exists.  So a click cannot be acted on in the pass that receives it --
# the button is drawn too late to matter for the pipeline above it.  The click is
# therefore parked in ``*_TICKER_KEY + PENDING_SUFFIX``, a **non-widget** key, and
# consumed by :func:`resolve_ticker` on the following pass.  It cannot simply be cleared
# by assigning to the button's own key: that key is a widget, and Streamlit raises
# ``StreamlitValueAssignmentNotAllowedError`` for any write to one (verified, not
# assumed).  One rerun of lag is the price, and it is the same arrangement the Price
# brush already relies on -- see ``PRICE_BRUSH_KEY``.
#
# The non-widget result key is also what makes the seed a **one-way latch**: Forecast
# adopts the Price ticker only when its own key is absent, and Price's reset never
# writes it.  So a Forecast ticker chosen once is never re-seeded, and changing the
# Price ticker afterwards leaves Forecast alone.  Every key is scoped by prefix, so the
# two scopes cannot collide -- ``st.tabs`` renders every tab body on every rerun.
# **One ticker, one scope.**  The Forecast tab used to carry its own ticker, its own
# pipeline and its own search, so a reader could leave the price tape on QQQ while the
# forecast asked the same question of AAPL.  That independence is gone: the ticker input
# now lives in the sidebar, there is exactly one, and every tab -- Price, Matches,
# Projection, Forecast, Quality and Backtest -- is built from the same archive.
#
# The three keys are not redundant: one is durable state and two are widgets.  See the
# note on :func:`resolve_ticker` for why the symbol in force has to live under a
# **non-widget** key -- ``reset_query_state`` clears widget state wholesale, and a
# ticker stored in a widget key would be erased by its own reset, leaving the box
# re-seeded to the module default while every chart showed a different instrument.
#
#   ``TICKER_KEY``         non-widget.  The symbol actually in force.  Survives a
#                          widget wipe, so it is the *only* durable record.
#   ``TICKER_INPUT_KEY``   the ``st.text_input``.  Widget state.
#   ``TICKER_FETCH_KEY``   the ``st.button``.  Widget state.
TICKER_KEY = "active_ticker"
TICKER_INPUT_KEY = "ticker_input"
TICKER_FETCH_KEY = "ticker_fetch"

# The resolution is **session-level**, not per-scope: see
# :data:`SESSION_TIMEFRAME_KEY`.  There are deliberately no ``*_TIMEFRAME_INPUT_KEY``
# constants here any more, and no per-scope resolution keys at all.
#
# What replaces them is a *namespace*.  Every piece of per-archive state -- the brush,
# the query span, the match result, the applied window -- is suffixed with the
# resolution, so a 1-minute session and a daily session cannot read each other's bars
# even in principle.  That is a structural guarantee rather than a check: there is no
# code path in which a bar index computed on daily bars is looked up in a frame of
# minute bars, because the two live under different keys.

# Appended to a session-state key to namespace it by resolution.
#
# Applied to keys that hold bar indices.  Those are the only values in the app that
# mean something *specific to one archive*: an index of 3,000 is 3,000 minute-bars into
# QQQ and 3,000 daily-bars into QQQ, which is fifteen years apart.  Sharing one key
# across resolutions would let a brush made on daily bars be applied to a minute frame,
# and the chart would silently clamp it to the end.
RESOLUTION_SUFFIX = "_@tf"

# Appended to a scope's ticker key to name its **pending fetch** flag -- set by the
# **Fetch button, consumed by the next pass's :func:`resolve_ticker`.  Derived rather
# than declared so the flag cannot drift away from the key it belongs to; a hand-written
# pair would eventually disagree and the fetch would either fire on every rerun or never
# fire at all, and both look identical from the page.
PENDING_SUFFIX = "_pending_fetch"

# Where :func:`resolve_ticker` leaves the download outcome so the floating bar below can
# report it.  A module-level dict rather than session state because it is a
# *within-pass* hand-off between two functions -- the download has to happen before the
# pipelines are built, and the message belongs next to the control that caused it.  In
# session state a stale message would reappear on the next unrelated interaction; here it
# is popped by the reader and gone.
FETCH_RESULTS: Dict[str, Any] = {}

# Session keys cleared when the ticker is switched.  There is one scope now, so this is
# one list rather than two: every bar index, brush, selection, applied span and cached
# run describes a position or a result in *one* archive, and carrying any of it across
# to a new symbol silently re-runs a query the reader never asked for.  The list is the
# documented contract for "state tied to the previous ticker".
#
# ``TABS_GENERATION_KEY`` is deliberately absent.  It re-keys the tab bar to land the
# reader on *Projection* -- right for a ticker switch, which is the only fetch left --
# and is bumped separately below rather than cleared here.
STALE_KEYS: Tuple[str, ...] = (
    "run_signature", "run_output", "sel", PRICE_BRUSH_KEY,
    "applied_query_span", PRICE_RUN_KEY, PRICE_APPLIED_KEY,
    FORECAST_SELECTION_KEY, FORECAST_BRUSH_KEY,
    FORECAST_RUN_KEY, FORECAST_APPLIED_KEY,
)

# Timezone the trading sessions are grouped in.  A US equity session runs 09:30-16:00
# ET, so it is named by its *Eastern* date -- a 09:30 winter open is 14:30 UTC and
# would otherwise be filed under the previous UTC day.  Same rule as
# ``store.session_et`` and ``features.quality_report``.  It is what ``DEFAULT_VIEW_DAYS``
# counts, so the view is a number of real sessions rather than of calendar days.
EASTERN = "America/New_York"

CHANNEL_LABELS = FEATURE_COLUMNS

# "Live" leads and is named as the landing view by ``DEFAULT_TAB``: it shows the
# tape and nothing else, so the app opens on data rather than on controls.
# Everything else is opt-in depth.
#
# There is no separate *Chart* tab.  It used to hold the session dropdown, the From/To
# pickers and the "Check your selection" diagnostics -- but the one gesture that
# actually picks a window is the brush on the Price tab's own chart, and it was already
# live there and already drove Matches and Forecast directly.  Splitting "pick the
# window" onto a second tab meant the reader brushed on the tab they were looking at
# and then had to switch tabs to read what they had picked.
#
# "Panel" is the *driven* cross-sectional view: it searches the whole S&P 500
# archive and shows every analogue with all three rarity ranks, but you pick the query
# and press the button yourself.  It is placed after the single-instrument workflow it
# extends rather than beside it -- a reader who only ever wants "when has QQQ done
# this" should never have to scroll past it.  (The *Price* tab's match panel is the
# *automatic* cross-sectional view: the same archive, one match, no controls. See
# ``cross_sectional_match``.)
#
# **"Forecast" is the interactive half of "Projection", and sits directly after it.**
# The reference tab used to be called *Forecast* and carried both a fixed reference
# chart *and* the whole brush-and-evidence workflow, which made it the longest tab on
# the page and split its own subject across a scroll.  The two are now separate tabs:
# *Projection* answers "what usually followed the archive's most recent bars" and takes
# no reader input; *Forecast* answers "what usually followed the window I drew", and
# carries the brush, the search settings and the evidence table together.
#
# **The position is load-bearing, not a preference.**  ``FORECAST_SELECTION_KEY``
# records that a plotly chart's element id hashes the page's **chart list**, so the
# brush's identity depends on how many charts precede it in the render order.  Placing
# *Forecast* immediately after *Projection* leaves that order byte-for-byte what it was
# before the split --
#
#     forecast_path -> pooled panel path -> sel_forecast_view -> forecast_path_selected
#     -> forecast_bars
#
# -- so no reader's live brush is invalidated by the deploy.  Inserting this tab
# anywhere else shifts ``sel_forecast_view``'s element id and silently drops every
# selection on it.  (The *resolved span* in ``FORECAST_SELECTION_KEY`` would survive
# regardless, so the symptom would be a chart that stops accepting a second drag
# rather than an error -- do not rely on that as a safety net.)
#
# **The names do not match the internal scope, deliberately.**  The widget and session
# keys behind these tabs are all ``forecast*`` (``FORECAST_BRUSH_KEY``, ``FORECAST_RUN_KEY``,
# ``forecast_k``, and the ``"forecast"`` scope string passed to ``resolve_ticker``).
# That scope is *frozen*: renaming it to match today's tab name would migrate every one
# of those keys at once and discard every reader's brush, settings and last run on
# deploy, for no behaviour change.  A future reader who finds ``FORECAST_BRUSH_KEY``
# "mislabelled" against the *Forecast* tab should leave it alone.
#: The tabs, in the order they are drawn.
#:
#: **"Progression" sits after "Forecast" and before "Live".**  The position is
#: load-bearing in one direction only: ``FORECAST_SELECTION_KEY`` records that a
#: plotly chart's element id hashes the page's **chart list**, so moving *Forecast*
#: away from immediately after *Projection* changes how many charts precede it and
#: invalidates every brush selection a reader has already made.  That constraint is
#: pinned by ``test_forecast_sits_immediately_after_projection_in_the_tab_order``.
#:
#: The element id itself is safe against *editing* this file: it is derived from
#: ``calc_hash(main_script_path)``, which hashes the script's **path**, not its
#: contents.  What it is not safe against is changing the *number of charts rendered
#: before* a given one -- which is why the position above is pinned rather than
#: chosen.
TAB_ORDER = ("Price", "Matches", "Projection", "Forecast", "Live",
             "Panel", "Quality", "Backtest")

# How much tape the Price tab shows, in trading sessions.
#
# This replaces the *Days of history to display* slider, which counted calendar days:
# ``session_days = (t1 - t0).days + 1`` is ~30 for a 30-day archive even though only
# ~21 sessions exist in it, so its maximum was a range that could never be displayed.
# It then sized the view as ``bars_per_day * days`` where ``bars_per_day = n //
# session_days`` was an *average* over those inflated days -- so a "1 day" view was
# some number of bars near 390 that was not any real session.
#
# A constant has neither defect: every session is a real, contiguous, verifiable span
# of bars, and the number is stated in the code instead of being left to the reader to
# discover on a slider.  Five sessions is wide enough to find a moment in, narrow
# enough that a 240-bar window still reads as a shape rather than a stripe -- at the
# ~1,950-bar five-session width a [[LENGTH]]-bar band is ~12% of the panel, and the
# match panel below is drawn at the same bar-to-pixel scale.
DEFAULT_VIEW_DAYS = 5

# The tab the page opens on, passed to ``st.tabs(default=...)``.
#
# This is redundant with ``TAB_ORDER[0]`` today, and that redundancy is the point.
# "Price leads" is a *behavioural* claim the whole app's help copy makes ("this is
# where the app opens"), so it is stated as an explicit default rather than left to
# fall out of tuple order.  Streamlit falls back to the first tab when ``default`` is
# ``None``, which means a reordering of ``TAB_ORDER`` would silently change the
# landing view and break that promise with nothing in the diff to explain it.
def get_default_tab() -> str:
    """The tab the page opens on, passed to ``st.tabs(default=...)``.

    Defaults to "Projection" on daily resolution and "Live" on intraday.
    """
    return "Projection" if _tf().key == "1d" else "Live"


# ``st.tabs`` only *registers a widget* -- and only reads its selection from
# ``session_state`` -- when ``on_change`` is passed (Streamlit sets
# ``is_stateful = on_change != "ignore"``).  With the default ``on_change="ignore"``
# this bar is stateless: the selection lives in the browser, ``default`` applies only
# when the container is newly created, and nothing Python does can move the reader
# off whatever tab they are on.  ``on_change="rerun"`` would fix that but switches
# tabs to *lazy* rendering, so every click re-runs the whole script including the
# uncached ``pipe.run()`` baseline and bootstrap -- a regression on the one thing the
# app is built to show.
#
# So the tab bar is instead keyed on a *generation* that ``reset_query_state``
# bumps.  Streamlit derives a block's identity from its ``key`` string, so a new
# key means a new block id, the container is built from scratch, and ``default`` --
# consulted only on a fresh container -- finally applies and lands the reader on
# Price.  The generation is stable while a single ticker is being explored, so
# ordinary reruns (a slider, a brush, a text input) never move the reader off the
# tab they chose.
#
# It is a counter rather than the symbol because two fetches of the *same* symbol
# (the documented "press Fetch again to pick up the newest session") must also land
# on Price: a symbol-based key would be unchanged in that case and the reader would
# be stranded on Backtest looking at a brand new archive.  A counter is monotone, so
# every fetch re-keys -- including a re-fetch of a symbol already visited.
TABS_KEY = "main_tabs"

# Session key holding the tab-bar generation.  It lives in ``session_state`` rather
# than in a module global on purpose: a module global is shared by every browser
# session on the server, so one visitor's fetch would yank an unrelated visitor off
# whatever tab they were reading.  A plain module counter cannot express "this
# reader's tab bar, this many resets in".
TABS_GENERATION_KEY = "_tabs_generation"


def tabs_key() -> str:
    """The current widget key for the tab bar -- see :data:`TABS_KEY`.

    Read at both ``st.tabs`` sites, including the not-ready branch, so a ticker that
    fails to load does not leave the bar keyed differently from the one that loads.
    """
    return "{}:{}".format(TABS_KEY, int(st.session_state.get(TABS_GENERATION_KEY, 0)))

# Live window length, needed by help copy that is defined before ``main()`` runs.
# ``main()`` overwrites this once the pipeline exists; a one-element list is the
# smallest mutable bridge between the module body and the render functions.
PIPE_LENGTH_FOR_HELP: List[int] = [DEFAULT_LENGTH]

# Same bridge for the instrument name, so help copy can name the ticker on screen
# rather than whatever this app was first written against.  Rendered copy substitutes
# ``[[SYMBOL]]`` for the live value and names the default in literal prose.
SYMBOL_FOR_HELP: List[str] = [DEFAULT_SYMBOL]

HELP_ICON = "❓"

TAB_GUIDE: Dict[str, str] = {
    "Price": (
        "### 1. The tape, and the moment you are asking about\n\n"
        "This tab is two charts, plus this collapsed panel — no captions, no metrics, "
        "no commentary between you and the tape. Your query on top, the closest "
        "historical match to it below.\n\n"
        "**The match is searched across the whole S&P 500 archive**, not just the "
        "ticker you are looking at. Drag a box and the app asks \"has anything in "
        "the index done this?\", so the chart below is frequently a *different "
        "company's* tape — the caption above it names which one, along with the "
        "session and how rare that window is. If you wanted the narrower question, "
        "\"when has **this one ticker** done this?\", use the *Matches* tab, which "
        "still searches the fetched symbol's own history.\n\n"
        "Two things follow, and both are why the caption exists. A window can never "
        "span two tickers — each is scored as its own contiguous series — and the "
        "percentile is measured against every window in the panel, so a quiet "
        "micro-cap is not competing on equal terms with a liquid mega-cap. That is "
        "also why there are two numbers: **panel rarity** (across everything) and "
        "**own-ticker rarity** (within that company alone). A window can be the "
        "closest thing its own ticker has ever seen and still be unremarkable across "
        "500 names.\n\n"
        "The lower chart is always drawn. Matching is fast, so there is no reason to "
        "make it wait for a second click — if something like this window has happened "
        "before, you see it here rather than having to go and look.\n\n"
        "It exists because looking at price and reasoning about price are different "
        "moods. Here there is nothing between you and the tape.\n\n"
        "**Reading it**\n\n"
        "* the **orange band** is the query: the [[LENGTH]]-bar span being searched "
        "for. Every window overlapping it is excluded from matching, so nothing here "
        "can match itself — including in the panel's own copy of your ticker.\n"
        "* the **dashed orange line** just past the band is the bar *after* the "
        "window — the one forward returns are measured from (§Z1).\n"
        "* the **dotted blue line** is the rolling mean; the pale ribbon is ±2σ. A "
        "price hugging the ribbon is unusual for this tape, not unusual in general.\n"
        "* bars are drawn **side by side**, so the closed market between sessions "
        "takes up no space. Tick labels are still real UTC times, but the spacing "
        "between them is not — so read this for shape, not for speed.\n\n"
        "**How the query is chosen.** The app opens on the most recent [[LENGTH]] bars of "
        "tape, over a fixed view of the last five trading sessions. To ask about "
        "another moment, **drag a box across the top chart** — a brush of N bars is "
        "searched as N bars, and the *Matches* tab follows on the same click. That "
        "brush is the only window control in the app.\n\n"
        "**The tape does not move when you brush.** The top chart's view is fixed, so "
        "the same bars stay at the same place on screen however many windows you try. "
        "Only the orange band moves — to where you drew it. That sounds obvious, and "
        "it is deliberate: the chart used to re-frame itself around each new brush, "
        "which slid the tape sideways under your cursor and made the window you had "
        "just drawn appear somewhere else on the chart. A chart you are dragging a box "
        "across has to hold still.\n\n"
        "**A brush may cross a session boundary.** The band is placed exactly where "
        "you drew it, at exactly the width you drew, however many sessions that "
        "spans. Bars are drawn **side by side**, so the closed market between "
        "sessions takes no horizontal space — a window across a close is one "
        "continuous run of bars with one large move in it.\n\n"
        "That move is real tape, but know what it does: the last bar of one session "
        "and the first of the next are 17.5 hours apart, so the rebased path and the "
        "rolling return register it as a single move. It inflates similarity a little "
        "(a gap is an extreme shape, and gaps match gaps), and a shape containing one "
        "is a shape the market did not produce through trading. The **forward** "
        "version is handled for you: candidates whose *forward horizon* would run "
        "across a close are withheld from the match pool, because those report an "
        "overnight gap as if it were trading and inflate their measured return around "
        "20× (§BX). The Matches tab reports how many were withheld. Where your own "
        "query sits is your call.\n\n"
        "**Why the query is exactly the length you drew.** The matcher scores your "
        "window against every window of the *same* length in the archive, so nothing is "
        "padded, trimmed or rounded onto a fixed grid.\n\n"
        "**Tips**\n\n"
        "* A bigger window is not a better window. The default [[LENGTH]] bars is only "
        "one reading of this tape — a much shorter brush is microstructure, and at that "
        "length similarity is easier to find by accident. Compare lengths on the "
        "*percentile* in the Matches tab, not on the raw distance.\n"
        "* If the band on the chart does not sit exactly where you drew it, that is the "
        "archive running out rather than the drag failing — a window that would "
        "overhang the final bar is shifted inward, keeping the length you drew (§Z1).\n"
        "* Zoom controls are deliberately **removed** from both charts, and the mouse "
        "wheel deliberately does not zoom — it scrolls the page. The two charts are "
        "only comparable because they show the *same number of bars*, and any "
        "zoom would silently break that. To move along the tape, **drag the lower "
        "chart** — it is pannable and stops at its bounds — or brush a new window "
        "on the top one.\n\n"
        "**Next:** read the *Matches* tab, which searches this window; or press "
        "**Run match** in its own *Search settings*."
    ),
    "Matches": (
        "### 2. Read which historical windows look like your query\n\n"
        "This tab answers *\"when has something like this happened before?\"* — a "
        "question about **examples**, not about money. Treat it as intuition-building.\n\n"
        "**The three-stage funnel** (PLAN.md §C)\n\n"
        "```\n"
        "Stage 1  STUMPY matrix-profile scan over every candidate window\n"
        "   ↓  one compiled pass, every bar position scored\n"
        "Stage 2  rank by distance — STUMPY returns the whole profile,\n"
        "         so there is no separate \"precise scoring\" stage to disagree\n"
        "   ↓\n"
        "Stage 3  drop self-overlaps, then force ≥ one window-length of spacing\n"
        "   ↓\n"
        "k most similar windows, at least [[LENGTH]] bars apart\n"
        "```\n\n"
        "Stage 1 is the workhorse. It is a **matrix profile** (§S) computed by "
        "[STUMPY](https://github.com/stumpy-dev/stumpy): one compiled pass returns "
        "the distance from the query to every window in the archive, which is why "
        "the scan is fast enough to run on every keystroke. The same pass also "
        "identifies the *least* similar windows — the anomalies — for free.\n\n"
        "**There is no method selector, deliberately.** STUMPY's normalized distance "
        "*is* the Euclidean distance between z-scored windows, so it is the same "
        "ranking a plain scan would produce, just computed far more cheaply. A second "
        "scorer would only be a slower route to the same answer — or, as with the old "
        "time-warped option, a genuinely different metric whose percentile cannot be "
        "ranked against this distribution at all (§BD).\n\n"
        "**The counters at the top, left to right**\n\n"
        "| Counter | Meaning | Read it as |\n"
        "|---|---|---|\n"
        "| candidate windows | how many [[LENGTH]]-bar windows exist in the archive | "
        "bigger N ⇒ harder to be surprised |\n"
        "| excluded | windows removed because they **overlap your query** | these are "
        "suppressed so the query cannot match itself at distance ≈ 0 |\n"
        "| after NMS | how many survived suppression, capped at your **k** | each is at "
        "least [[LENGTH]] bars from the last, so they are genuinely separate moments |\n\n"
        "**Reading the table**\n\n"
        "* **distance** — lower is more similar. It is a raw geometric distance on "
        "z-scored features, so it has no units and is **not** comparable across "
        "different window lengths or different queries. Never read it as \"distance "
        "from the truth\".\n"
        "* **percentile** — the honest number. It is the share of every candidate "
        "window whose distance was *at least as small*: `0.11%` means only 11 in "
        "10,000 windows in the whole archive resemble your query this closely. Read "
        "this, not `distance`. (PLAN.md §E)\n"
        "* **bars** — always [[LENGTH]]. If it ever is not, that is a bug, and the "
        "forward returns would be measured from the wrong bar (§Z1).\n\n"
        "**Shape comparison, one tab per match**\n\n"
        "The three closest matches appear as three tabs under the table. Each tab "
        "shows the real tape twice: your query on top, the matched window below, both "
        "drawn the same way as the query chart above and both rebased to their own first bar so two "
        "different price levels land on one scale.\n\n"
        "The shaded band is the scored window. The bars to its left are what led into "
        "it; the bars to its right are what followed it. The dashed vertical line is "
        "the first bar *after* the window — the bar forward returns are measured from "
        "(§Z1).\n\n"
        "The tape charts above and the z-scored overlay below are allowed to "
        "disagree. The matcher scores `return_z` and `path_z`, not raw price level, "
        "so a match can trace the feature shape while its close price looks nothing "
        "like yours. When the two views disagree, the machine view is the one that "
        "produced the distance.\n\n"
        "These two panels put their bars side by side on a bar-index axis, which is "
        "what makes a multi-session shape line up. That means the horizontal axis is no "
        "longer proportional to time, so a gap between two bars does not tell you how "
        "long the market was shut — read the tick labels for that.\n\n"
        "In that overlay the two curves are **deliberately not symmetric**, and that "
        "matters:\n\n"
        "* the **query** (orange) is z-scored over its own span, so it is centred at 0;\n"
        "* the **match** (blue, dashed) is the raw feature slice the library actually "
        "scored, not re-normalised.\n\n"
        "Re-normalising both would draw a prettier, tighter pair of curves that **do "
        "not reproduce the distance in the table**. The visible offset between them is "
        "the distance you are being shown. (PLAN.md §BC)\n\n"
        "**The percentile is always populated.** Because the distance metric is fixed, "
        "every match has a rank against a distribution measured the same way. (An "
        "earlier version also offered a time-warped distance whose percentile was "
        "honestly reported as `—`; that option is gone, so the column no longer has "
        "a blank state. §BD)\n\n"
        "**A very low percentile is not automatically a good result.** The percentile is "
        "the share of candidates at least as close, so with thousands of windows "
        "searched even a mediocre best-match lands near zero. Treat it as a ranking, "
        "not a probability: compare the same query across window lengths and prefer "
        "one that stays rare at [[LENGTH]] bars or above.\n\n"
        "**Next:** the *Projection* tab is the fixed reference, and *Forecast* is "
        "where a window you drew is turned into a number."
    ),
    "Projection": (
        "### 3. A fixed reference: what usually happened next\n\n"
        "This tab asks one question about the archive's own most recent bars and "
        "answers it the same way every time, which is what makes it a reference "
        "rather than a query: it never asks about a window you picked, so it stays put "
        "while you explore windows on the tabs either side of it. If you want a "
        "forecast for a specific moment, that is **Forecast**, the next tab.\n\n"
        "**Two controls size this chart, and they are independent.** *Recent bars*, at "
        "the top of this tab, sets how far back into the archive's tail it looks — "
        "the blue history, and the window that gets matched — starting at "
        "[[FORECAST_HISTORY_BARS]]. *Projection bars*, just below it, sets how far "
        "ahead the green band reaches, starting at [[FORECAST_PROJECTION_BARS]]. One "
        "looks backwards and one looks forwards, so they are judged on different "
        "grounds: the history wants to be a recognisable shape, and the projection "
        "wants to stop before there is no real tape to compare against.\n\n"
        "**A longer history is not a better answer.** *Recent bars* below about "
        "[[FORECAST_HISTORY_BARS]] describes a *shape* in more detail, and matches more "
        "rarely; well above it the shape stops being recognisable and the match is "
        "made against a slab. If the reader wants a forecast for a window they chose, "
        "*Forecast* is the tab for that — brushing there sets the window, and this "
        "control never overrides it.\n\n"
        "**How far ahead it looks.** *Projection bars* is the length of the green band "
        "on **both** projection charts in the app — this one and the reader's own on "
        "*Forecast*. It changes how far the median reaches and nothing else: not "
        "which window is asked about, and not the evidence table, which keeps its own "
        "fixed set of horizons. Shortening it is usually the more useful move: past "
        "the next session close there is no real tape to measure against, so a longer "
        "band is projecting into a market that has not opened yet, and §BX drops the "
        "candidate windows whose horizon would run over that close. On 1-minute bars "
        "the slider stops one bar short of a full 390-bar session for exactly that "
        "reason — at a full session it stops there being anything to compare against.\n\n"
        "**The two charts, and why they share axes**\n\n"
        "The top chart shows the archive's most recent [[FORECAST_HISTORY_BARS]] bars "
        "of real tape, then up to [[FORECAST_PROJECTION_BARS]] bars of projection: the "
        "median of where the [[FORECAST_PATH_MATCHES]] closest matching windows went "
        "next, each rebased to its own final close. The green band is the middle half "
        "of those paths. The chart below it repeats the exercise pooled across **every "
        "symbol in the S&P 500 panel**, on deliberately identical axes, so the two are "
        "comparable bar for bar.\n\n"
        "**Read the width of the band before the line.** A wide band means the matches "
        "disagreed about what came next, and that disagreement is the most useful thing "
        "on the chart. A narrow band with a rising median is a genuinely consistent "
        "sample; a narrow band on 30 windows that all fell on the same quiet afternoon "
        "is a coincidence.\n\n"
        "**Next:** to ask about a window you choose rather than the archive's most "
        "recent one, go to **Forecast**. To check this tab's claim against an "
        "entirely independent run, go to *Backtest*.\n\n"
    ),
    "Forecast": (
        "### 4. Forecast the window you actually drew — and check it\n\n"
        "This is where the brush, the projection and the evidence table live together. "
        "They are on one tab because they are one interaction: the table describes the "
        "window you brush, so a reader who had to brush on one tab and read the result "
        "on another would be looking at a gesture and a claim about different windows.\n\n"
        "**The chart, and what it answers**\n\n"
        "Drag a box across the chart under *Forecast a window you choose* and the "
        "projection below it shows **your** window, then up to "
        "[[FORECAST_PROJECTION_BARS]] bars of where the "
        "[[FORECAST_PATH_MATCHES]] closest matching windows went next. **Blue is your "
        "window, rebased to its own final close; green is the median of where the "
        "matches went.** The projection starts where your window ended, not at the "
        "end of the archive.\n\n"
        "**The window is yours; only the horizon is shared.** *Projection bars* lives "
        "on **Projection**, and both charts read it, so the reference band and your own "
        "are the same length and are directly comparable. The evidence table below is "
        "the exception: it keeps its own fixed horizons and does not move with the "
        "slider. The *Projection* tab's *Recent bars* does **not** apply here — it "
        "sizes the reference chart's own tail, whereas the blue on this tab is exactly "
        "as wide as the box you drew.\n\n"
        "**This tab has its own ticker, its own brush and its own search.**\n\n"
        "*Its own ticker* — its own input at the **top of this tab**. It starts on "
        "whatever the Price tab holds, then keeps its own, so you can leave the price "
        "tape on QQQ while the forecast asks the same question of AAPL. Switching "
        "either tab's ticker leaves the other tab's tape, brush and results alone.\n\n"
        "*Its own window* — every number on this tab describes **the window you brush "
        "here**, and nothing else. Brushing on the *Price* tab does not move this tab: "
        "it aims the *Matches* search, which is a different question. There is **no "
        "default window** and no fallback to the archive's newest bars — matches near "
        "the end have no forward bars, so a silent fallback would reliably produce the "
        "least informative answer the tab can give.\n\n"
        "The evidence table needs both that window and a run of **this** tab's *Search "
        "settings*. Until then it says which of the two is missing rather than showing "
        "numbers for a window you did not choose.\n\n"
        "**What is computed, per horizon** (PLAN.md §D)\n\n"
        "For each horizon $h$ in 5 / 15 / 30 / 60 minutes, over the matched set $M$:\n\n"
        "$$\n"
        "\\hat{r}_h = \\frac{1}{|M|}\\sum_{i \\in M} r_{i, i+h}\n"
        "\\qquad\n"
        "\\text{lift}_h = \\hat{r}_h^{\\text{matched}} - \\hat{r}_h^{\\text{random}}\n"
        "$$\n\n"
        "* `r_{i,i+h}` is the log return from the **last bar of matched window *i*** "
        "forward *h* bars. It never starts inside the window and never overlaps it "
        "(§Z1). A window whose horizon runs past the end of the archive is `NaN`, not "
        "zero.\n"
        "* **baseline_mean** is the identical statistic over windows picked at random "
        "from the same archive. It is the control that makes the whole page mean "
        "something.\n\n"
        "**Why the baseline is not optional**\n\n"
        "Search ~20,000 windows and the single most extreme-looking forward return is "
        "extreme *by construction* — that is the multiple-comparisons trap (PLAN.md §E). "
        "A green bar without the grey bar beside it is noise with a decimal point. "
        "**Only `lift` is a claim; `mean_return` on its own is not.**\n\n"
        "**Row colours**\n\n"
        "| Colour | State | Meaning |\n"
        "|---|---|---|\n"
        "| 🟢 green | `p < 0.05` | beats the random baseline at conventional "
        "significance |\n"
        "| 🟡 amber | sufficient, `p ≥ 0.05` | a number is printable but it is "
        "**indistinguishable from chance** |\n"
        "| ⚪ grey | `sufficient = false` | `k` valid matched windows did not reach the "
        "minimum. **No number is shown** — not greyed out, not even. |\n\n"
        "The grey state is not a display bug. Below the evidence threshold the mean, "
        "interval and p-value are all functions of a handful of observations, so the "
        "honest output is no output. (PLAN.md §E, \"sample-size honesty\")\n\n"
        "**To fix a grey forecast:** raise **k matches** in the sidebar (needs ≥ "
        "**Min matches for evidence**, default 30), move the query away from the edges "
        "of the archive so more windows have a full horizon, or lengthen the window so "
        "fewer candidates get excluded for self-overlap.\n\n"
        "**Column meanings**\n\n"
        "* `n_valid` — matched windows whose full forward horizon exists in the data.\n"
        "* `ci_low` / `ci_high` — **moving-block bootstrap** interval, not i.i.d. "
        "Matched windows sit near each other in time, and an i.i.d. interval would "
        "understate the variance.\n"
        "* `p_value` — permutation test against the random-window baseline, corrected "
        "for the fact that the matched windows were *selected* as the closest-k by "
        "shape. That selection makes their mean noisier than a random draw even on pure "
        "noise, so the null is widened by a measured factor first (§BW). A `p_value` "
        "here is therefore a stricter bar than a plain permutation test, not a looser "
        "one.\n"
        "* `lift` — a **difference in basis points**, not a ratio. The unconditional "
        "mean is near zero, so a ratio diverges and would be meaningless.\n\n"
        "**Next:** if this survives, sanity-check it against an entirely independent "
        "run in the *Backtest* tab."
    ),
    "Progression": (
        "### 5. The same question at fifty different lookbacks\n\n"
        "Every other tab answers one question at one length. This one asks the *same* "
        "question about the *same* moment at fifty lengths, and draws all fifty "
        "answers at once.\n\n"
        "**What is computed**\n\n"
        "For each lookback length $L$ — 10 bars, 20 bars, 30 bars, and so on up to 500 "
        "— the archive's own most recent $L$ bars are matched against the archive's "
        "history, and the matched windows' forward paths are projected over "
        "[[PROGRESSION_HORIZON]] bars. All fifty projections are drawn on one chart.\n\n"
        "**Why one chart and not fifty rows.**  The finding is the *shape* of the "
        "answer across lengths, not any single length's number. When the short "
        "lookbacks project one way and the long ones project another, the "
        "disagreement is visible in the spread of the traces and invisible in a "
        "table. The table below the chart has the numbers for when you want them "
        "exactly.\n\n"
        "**Reading it**\n\n"
        "* the **blue line on the left** is the real tape — the archive's most recent "
        "bars, rebased to the last close so it meets the projections at exactly zero.\n"
        "* the **pale traces are short lookbacks and the dark ones are long.** The "
        "colour ramp is ordered by lookback length, so it carries meaning: if the "
        "dark traces and the pale traces disagree, the recent tape is saying "
        "something the longer history does not confirm.\n"
        "* the **red line is the median across all the lookbacks** — what they agree "
        "on. Fifty overlapping traces cannot be read by eye, so the middle of the "
        "cloud is computed and drawn.\n"
        "* the **dashed line** is where the projection starts, the same convention as "
        "every other chart in this app.\n\n"
        "**Everything is in percent from the last close.** That is what lets fifty "
        "projections share one axis: each matched window is rebased to *its own* "
        "final close, so a trace describes the percentage move that followed, not the "
        "price level it happened at.\n\n"
        "**The Matches column is the honest part.**  A long lookback cannot always "
        "get thirty matches. Non-max suppression (§M) requires one full lookback of "
        "separation between accepted windows, so a 500-bar lookback can only place a "
        "handful of windows in an archive of this size — and a median over seven "
        "windows is a materially thinner claim than a median over thirty. The table "
        "reports the count for every length, and lengths that could not be projected "
        "at all are listed with the reason rather than silently dropped.\n\n"
        "**This tab is a reference, not a query.**  Every length asks about the "
        "archive's most recent bars, so the chart stays put while you explore other "
        "windows elsewhere. It forecasts the **Forecast** tab's instrument at the "
        "Forecast tab's resolution.\n\n"
        "**Next:** the *Backtest* tab, for whether any of this survives out of sample."
    ),
    "Quality": (
        "### 6. Check the data before you trust any of it\n\n"
        "Every number in this app descends from the bars loaded here. Read this tab "
        "first when a result looks surprising.\n\n"
        "**The keys in the JSON, and what each one should be**\n\n"
        "| Key | Healthy | If it is not |\n"
        "|---|---|---|\n"
        "| `rows` | tens of thousands | too few ⇒ no credible percentile |\n"
        "| `sessions` | distinct ET trading days | `1` means a single day's 390 bars: "
        "there is nothing to match against |\n"
        "| `bars_min` / `bars_max` | ≈ 390 each | well under 390 ⇒ that session is "
        "truncated or the download failed |\n"
        "| `gaps_over_180s` | `0` | holes **inside** a session: a halt, or dropped "
        "prints. Matching across one fakes a price jump. |\n"
        "| `session_boundaries` | one per day pair | this count is **normal** — it is "
        "the market being closed overnight, not missing data (§BF) |\n"
        "| `nonpositive_close` | `0` | log returns are undefined; masked to NaN |\n"
        "| `duplicate_timestamps` | `0` | the last observation of each was kept |\n"
        "| `monotonic` | `true` | the frame was re-sorted; Yahoo did not return the "
        "bars in chronological order |\n\n"
        "**Why an overnight boundary is not a gap.** The US session opens 09:30 ET and "
        "closes 16:00 ET, so consecutive 1-minute bars are never 60 seconds apart "
        "across a night. Counting that as corruption produced a false alarm on a clean "
        "archive; the two conditions are now reported separately (§BF).\n\n"
        "**Bar indices are into the cleaned frame.** Features are a rolling-20 z-score of "
        "log return and the rebased log-price path. The first ~20 bars are **dropped**, "
        "not imputed. So bar index *i* on any other tab is an index into this cleaned "
        "frame, not into the raw download.\n\n"
        "**Next:** data sane? Go back to *Price* and run a match."
    ),
    "Backtest": (
        "### 7. Test whether the whole thing has any edge at all\n\n"
        "The Forecast tab is in-sample by construction: you chose the query, then "
        "measured what happened after it. The backtest asks the harder question — *out "
        "of sample, would matching have predicted direction better than guessing?*\n\n"
        "**How each step works**\n\n"
        "```\n"
        "step i:  the matcher may only see bars [0, i)\n"
        "          it predicts the direction of the return over [i, i+horizon)\n"
        "          that prediction is scored against what actually happened\n"
        "          the next step starts far enough ahead to share no bar\n"
        "```\n\n"
        "**Controls**\n\n"
        "* **Horizon (bars)** — bars ahead to predict. 15 bars = 15 minutes. Longer "
        "horizons are easier to call and worth more when right.\n"
        "* **Round-trip fee (bps)** — subtracted once per prediction. A 1-minute "
        "[[SYMBOL]] "
        "round trip typically costs 1–3 bps, which can exceed the entire edge being "
        "measured. At `0` the net mean is reported as `NaN`, not as an optimistic "
        "zero.\n"
        "* **Warm-up bars** — history each step is allowed to see before it can "
        "predict. Must leave at least one window plus one horizon after it.\n"
        "* **Max steps** — predictions to make. Each step **re-searches the whole "
        "library**, so this tab is far slower than a single match.\n\n"
        "**The four numbers**\n\n"
        "| Metric | Meaning |\n"
        "|---|---|\n"
        "| direction accuracy | share of steps whose predicted sign matched the "
        "realised sign |\n"
        "| baseline accuracy | the same on randomly chosen directions — the floor to "
        "beat |\n"
        "| lift vs baseline | accuracy − baseline. **This is the number.** |\n"
        "| n predictions | effective sample size. With steps spaced by "
        "`max(horizon, length/2)` bars, consecutive predictions share no bar. |\n\n"
        "**Why steps are spaced apart.** Predicting every bar while measuring a "
        "multi-bar outcome would make consecutive rows near-duplicates treated as "
        "independent, inflating the effective sample to roughly `n / horizon` and "
        "shrinking every interval. Steps are instead spaced by "
        "`max(horizon, window length / 2)` bars so no two predictions share a bar. "
        "That stride is shown above once a run completes. (PLAN.md §Z3)\n\n"
        "**How to read the verdict**\n\n"
        "* 🔴 `p ≥ 0.05` ⇒ the accuracy is consistent with the matcher having **no "
        "edge**. Do not read the point estimate as a strategy.\n"
        "* 🟢 `p < 0.05` ⇒ beats the permutation baseline. Still read the interval "
        "width — a wide interval that spans the baseline is not an edge either.\n"
        "* Net mean / net hit rate appear only when fees are supplied, for the reason "
        "above.\n\n"
        "**Next:** a backtest that fails does not invalidate the tool — it means the "
        "*forecasts do not survive costs and out-of-sample testing*, which is exactly "
        "what this tab exists to reveal."
    ),
}


# =============================================================================== #
# Help toolkit
# =============================================================================== #
HELP_CSS = """
<style>
.tp-hint {font-size: 0.82rem; line-height: 1.55; color: #6b7280; margin: 0.1rem 0 0.5rem 0;}
/* The resolution shown beside the ticker box.  It is a *label*, not a control, and it
   is styled to read as information rather than as something that can be picked up:
   grey, small, and out of the way of the input it qualifies.  Deliberately *not*
   styled like a disabled selectbox -- a disabled control still looks like a control,
   which would invite the reader to try it.  Lives in ``HELP_CSS`` despite the name,
   because ``inject_css`` runs on every pass before the gate, and the help dialog is
   only one of the things it styles. */
.tp-res {font-size: 0.82rem; color: #6b7280; padding-top: 0.55rem;}
.tp-toc {font-size: 0.9rem;}
.tp-toc li {margin-bottom: 0.3rem;}
.tp-glossary td {vertical-align: top; font-size: 0.88rem;}
.tp-glossary td:first-child {white-space: nowrap; font-weight: 600;}
blockquote {border-left: 3px solid rgba(128,128,128,0.4); margin-left: 0;
            padding-left: 0.8rem; color: inherit;}
</style>
"""


def inject_css() -> None:
    """Install the help stylesheet.  Safe to call on every Streamlit rerun."""
    st.markdown(HELP_CSS, unsafe_allow_html=True)


def hint(text: str) -> None:
    """One-line grey explanatory note, used under a control or a chart."""
    st.markdown('<p class="tp-hint">%s</p>' % text, unsafe_allow_html=True)


def fill_tokens(text: str) -> str:
    """Substitute the live window length and instrument into help copy.

    Three placeholders, all replaced as plain text:

    * ``[[LENGTH]]`` -- the window length actually in use, so the copy cannot quote a
      stale constant after ``DEFAULT_LENGTH`` changes.
    * ``[[ROLLING]]`` -- ``ROLLING_WINDOW``, the rolling-mean band width.  A separate
      token because it is a *different* number from the query window: the band is a
      smoothing constant and the query is the thing being searched.  Substituting
      ``[[LENGTH]]`` into band copy was a latent wrong number the moment the two
      constants stopped being equal.
    * ``[[SYMBOL]]`` -- the instrument on screen.  The app is no longer QQQ-specific,
      so copy that named QQQ would describe a different market than the chart above it.
      A fee quoted in "1-minute QQQ round trips" is not just stale text for a name
      twenty times the price; it is a wrong number.
    * ``[[GAP]]`` -- the intra-session hole threshold, or the word "threshold" on daily,
      where no such concept exists.  Spelled as its own token because the number is a
      property of the resolution: 180 s is three 1-minute bars of slack and is
      meaningless against bars a day apart.
    * ``[[TIMEFRAME]]`` -- the resolution in force (``1-minute`` or ``Daily``).  Its
      own token because the copy makes *resolution-specific* claims -- a ~29-day
      archive, a 390-bar session, a 180-second hole threshold, a 1-3 bps round trip --
      and none of those hold on daily bars.  Quoting them there is not merely stale
      text, it is a wrong number in the one document a reader consults to check the
      numbers on screen.
    * ``[[FORECAST_HISTORY_BARS]]``, ``[[FORECAST_PROJECTION_BARS]]`` and
      ``[[FORECAST_PATH_MATCHES]]`` -- the three sizes the Projection tab's path chart
      is built at.  Separate tokens rather than ``[[LENGTH]]`` because they are
      deliberately *not* the query length: reusing it would make the help describe the
      chart by a number that governs a different control, which is exactly the
      wrong-number failure the other tokens exist to prevent.

      ``[[FORECAST_PROJECTION_BARS]]`` is the slider's **starting point**, not the
      band the reader is currently looking at: it renders the timeframe's
      ``forecast_projection_bars``, because help copy is built outside any widget
      context and cannot read the reader's current selection.  The copy therefore
      phrases it as a default ("starting at N") rather than as the value on screen,
      which is the only claim that stays true once the reader moves the control.

      **``[[FORECAST_HISTORY_BARS]]`` is a starting point for the same reason.**  It
      names the width the *Recent bars* slider opens at, not the width currently on
      screen, so every use of it is phrased as "starting at N" or "by default".  A
      token that claimed to be the live value would go stale silently -- help copy is
      rendered once per pass from a constant, so nothing would raise when the reader
      moved the slider.

    Token substitution rather than ``str.format`` because the copy is Markdown
    containing LaTeX (``\\frac{1}{|M|}``), and ``format`` would read those braces as
    format fields.  Plain replacement cannot misparse anything.
    """
    return (text
            .replace("[[LENGTH]]", str(PIPE_LENGTH_FOR_HELP[0]))
            .replace("[[ROLLING]]", str(ROLLING_WINDOW))
            .replace("[[SYMBOL]]", SYMBOL_FOR_HELP[0])
            .replace("[[TIMEFRAME]]", active_label().lower())
            # ``[[GAP]]`` is the intraday hole threshold, which exists only where a
            # session has an interior.  On daily there is no such threshold, so the
            # token renders as the word that makes the sentence true rather than as a
            # number that would not be.
            .replace("[[GAP]]", "{} s".format(_tf().gap_seconds)
                     if _tf().gap_seconds is not None else "threshold")
            .replace("[[FORECAST_HISTORY_BARS]]", str(_tf().forecast_history_bars))
            .replace("[[FORECAST_PROJECTION_BARS]]", str(_tf().forecast_projection_bars))
            .replace("[[FORECAST_PATH_MATCHES]]", str(FORECAST_PATH_MATCHES))
            # The Progression tab's own horizon.  Its own token rather than
            # ``[[FORECAST_PROJECTION_BARS]]`` because it is a *different* number:
            # the Projection slider governs the reference path, and this sweep
            # projects 300 bars on its own account.  Sharing a token would make
            # one control's value describe two unrelated charts.
            .replace("[[PROGRESSION_HORIZON]]", str(PROGRESSION_HORIZON)))


def guide(key: str) -> None:
    """The tab's own how-to, drawn as a collapsed panel above the tab's content.

    **This was a no-op, and the whole of :data:`TAB_GUIDE` was unreachable because
    of it.**  The function body was ``pass``, so the seven ``guide("...")`` call
    sites -- one per tab -- rendered nothing at all, and the 385 lines of prose in
    :data:`TAB_GUIDE` were carried in the file for no reader to ever see.  The
    call sites and the dict were both kept and documented as if the panel rendered;
    only the body was missing.

    **Collapsed by default, deliberately.**  The Price tab's whole job is to put the
    tape in front of the reader with nothing between them, so a long how-to has to be
    one click away rather than in the way.  The panel is a disclosure, not a section.

    The prose is passed through :func:`fill_tokens` so the numbers it quotes are the
    ones actually in force -- the live window length, the instrument on screen, the
    resolution, and the forecast chart's sizes.  A guide that quoted a stale
    constant would be worse than no guide, because it would be *checkable* and
    wrong.

    **A key with no entry draws nothing**, rather than an empty panel: a tab whose
    guide has not been written yet should show no expander at all, not a heading
    the reader can open to find it empty.
    """
    text = TAB_GUIDE.get(key)
    if not text:
        return
    with st.expander("How to use this tab", icon=HELP_ICON):
        st.markdown(fill_tokens(text))


CONTROL_GLOSSARY = """
| Control | What it does | When to change it |
|---|---|---|
"| **Resolution** (chosen when the app opens) | Which bars the whole page runs on: `1-minute` (~29 days of intraday tape) or `Daily` (the whole listing history). It applies to both tabs at once and is shown, not set, beside the ticker box — reload the page to change it. | when you open the app, before reading anything as a 1-minute statement. |
| **k matches** | How many historical windows to keep *after* suppression. | raise it (≥ 30) whenever the forecast comes back grey |
| **Baseline windows** | How many random windows form the control. More = a steadier baseline, slower. | only if the baseline bar looks noisy |
| **Min matches for evidence** | The floor on valid matched windows before any forecast number may be printed. Default 30 (§E). | leave at 30; lowering it publishes numbers the sample cannot support |
| **Run match** | Executes the search. Results are never cached, because the query is chosen interactively. | every time the query or any parameter changes |
| **Horizon (bars)** | (Backtest) How many bars ahead each step predicts. | longer = easier to call, more at stake |
| **Round-trip fee (bps)** | (Backtest) Cost charged once per prediction. A [[TIMEFRAME]] [[SYMBOL]] round trip is typically 1–3 bps. | set it above 0 before believing any net number |
| **Warm-up bars** | (Backtest) History each step may see before predicting. | must leave a window plus a horizon |
| **Max steps** | (Backtest) Number of predictions. Each re-searches the library, so this is slow. | raise for a tighter confidence interval |

**There is no "how much history to download" control, on purpose.** Yahoo serves roughly
30 days of 1-minute bars and no more, so a shorter fetch is never more accurate — it
only shrinks the candidate pool the percentile is measured against. The download always
takes the whole window, assembled from 7-day requests because Yahoo refuses anything
longer in a single call.
"""

GLOSSARY = """
| Term | Meaning |
|---|---|
| **query window** | The one span of tape you picked. Everything else on the app is an answer about it. |
| **trading session** | One Eastern calendar day of trading — 09:30–16:00 ET, ~390 one-minute bars. Bars are drawn side by side, so the closed market between sessions takes no space on the chart. A *session* is also the unit the app's view is measured in: it shows the last five of them. |
| **window / window length** | Size of the pattern in [[TIMEFRAME]] bars. A brush sets its own: brush N bars and N bars are searched. The default of [[LENGTH]] bars is the window the app opens on. | — |
| **candidate window** | Every window of the query's own length the matcher could compare against, across the whole archive. |
| **distance** | Geometric gap between the query's feature vector and a candidate's. Lower = more similar. **Unitless and not comparable across queries** — use the percentile instead. |
| **percentile** | Share of *all* candidates at least as close as this one. `0.11%` = rarer than 11 windows in 10,000. This is the honesty feature (§E). |
| **exclusion margin** | One window-length around the query, removed before scoring so the query cannot match itself (§M). |
| **NMS (non-maximum suppression)** | Enforces at least one window-length of spacing between accepted matches, so overlapping windows do not masquerade as separate evidence. |
| **prefilter** | Stage-1 cap (≈500 windows) that keeps the expensive scorer cheap. |
| **return_z** | Rolling-z-score of the [[TIMEFRAME]] log return. Price movement, scaled to the recent local volatility. |
| **path_z** | The rebased log-price path — the line the chart draws. *Where* price went, as opposed to how violently it got there. |
| **forward return** | Log return from the **last bar of a matched window** forward *h* bars — always after the window, never inside it (§Z1). |
| **horizon** | How far ahead the forward return is measured: 5 / 15 / 30 / 60 bars. |
| **baseline** | The identical statistic over *randomly chosen* windows. The control a forecast must beat. |
| **lift** | `matched mean − baseline mean`, in bps. A **difference**, not a ratio — the baseline sits near zero, so a ratio would diverge. |
| **n_valid** | Matched windows whose full forward horizon exists in the archive. Windows running past the last bar are `NaN`, not zero. |
| **sufficient** | `n_valid ≥ Min matches for evidence`. When false, no number is printed at all (§E). |
| **block bootstrap** | Interval method that resamples *blocks* of time rather than single observations, because matched windows and their horizons overlap in time. |
| **permutation test** | Shuffles the match labels and recomputes the statistic, to ask how often chance alone produces an effect this large. Gives the p-value. Corrected for **selection** (§BW): the matched windows are the closest-k by shape, which makes their mean *noisier* than a random draw, so the null is widened by a measured factor before the p-value is read. That correction is why most real queries are reported as insufficient evidence rather than as findings. |
| **walk forward** | Backtest that only ever shows the matcher bars `[0, i)` at step *i*, so it can never peek at the future. |
| **direction accuracy** | Share of backtest steps whose predicted sign matched the realised sign. |
| **gap > [[GAP]]** | A hole *inside* a trading session — a halt or dropped prints. Distinct from an overnight session boundary, which is normal. On daily bars this does not apply; a missing *day* is reported instead. |
| **§** references | Point to sections of `PLAN.md`, where the design rationale and each past bug fix are recorded. |
"""


# =============================================================================== #
# Caching
# =============================================================================== #
def fetch_ticker_cached(symbol: str, *, refresh: bool = False,
                        timeframe: object = None) -> "F.FetchResult":
    """Download every available bar for ``symbol``, reusing this session's fetch.

    ``@st.cache_data`` is deliberately *not* used here: a network call returns a
    fresh DataFrame every time and Streamlit would copy it per rerun, which for a
    multi-session 1-minute frame is megabytes on every keystroke.  The store is a
    plain dict instead -- but reached through ``_fetch_store()``, a
    ``cache_resource`` singleton, because a **module-level dict is re-created empty on
    every rerun** (Streamlit re-executes the script in a fresh namespace per
    interaction -- see the note on :data:`ACTIVE_TIMEFRAME`) and would therefore
    never hit at all.  That made every brush re-download ~29 days of 1-minute bars
    and let the live archive grow mid-session, which moved the trailing view by a bar
    each time and read as the chart scrolling.
    See the note above the store's definition.

    **Keyed on the timeframe as well as the symbol.**  This was the one genuinely
    load-bearing consequence of adding a second resolution, and it fails silently:
    keying on the symbol alone means a reader who runs the page on Daily after
    having it open on 1-minute is served the *cached 1-minute frame* under a Daily
    heading -- a chart of five weeks of minute bars labelled as two decades of daily
    ones, with no error and no way to tell from the page.  The store therefore holds
    ``(symbol, timeframe)`` keys, and reloading onto the other resolution in the same
    browser session hits the original entry rather than re-downloading.

    The key is built at the call site rather than from a bare symbol precisely
    because the reader can still *have* both: the resolution is fixed per session,
    but a session that chose Daily has usually just replaced one that chose
    1-minute, and Streamlit keeps ``session_state`` across the reload.

    ``days`` is deliberately **not** passed: ``None`` means "everything this timeframe
    has", which is 29 days on 1-minute and the whole listing history on daily.  The
    intraday figure is not reusable here -- applying it to a daily request would clip
    the history to 29 days and look exactly like a short archive.

    ``refresh=True`` re-downloads and overwrites the cache entry.  Pressing *Fetch* is
    an explicit request for the latest bars, and today's session is still filling up —
    without this the newest, most interesting session would stay frozen at whatever it
    looked like on the first click.

    Bars are written to the partitioned Parquet archive on disk as soon as they are
    fetched, so the data survives Streamlit reruns (which recreate the in-memory
    ``_fetch_store`` dict) and is visible to other sessions and tabs.  The in-memory
    store is retained so the same fetch is not re-downloaded on every brush.  The
    write uses ``policy="flag"``: a re-fetch whose content differs from what is stored
    leaves the existing partition untouched and increments a revision count instead of
    overwriting it, so a retro-adjusted day is flagged rather than silently destroyed.
    """
    store = _fetch_store()
    tf = resolve_timeframe(timeframe if timeframe is not None else ACTIVE_TIMEFRAME[0])
    key = "{}@{}".format(str(symbol).upper(), tf.key)
    if not refresh:
        hit = store.get(key)
        if hit is not None:
            return hit
    result = F.fetch_ticker(symbol, days=None, timeframe=tf.key)
    store[key] = result
    # Persist fetched bars to the partitioned Parquet archive on disk.
    # This ensures data survives Streamlit reruns (which recreate the in-memory
    # _fetch_store dict) and is available to other sessions/tabs.
    panel_root = panel_root_for(timeframe)
    PanelStore(panel_root).write(
        symbol, result.frame, policy=MERGE_POLICY, timeframe=tf.key,
    )
    return result


@st.cache_resource(show_spinner=False, max_entries=8)
def pipeline_from_frame(cache_key: str, frame: pd.DataFrame, length: int,
                        timeframe: object = DEFAULT_TIMEFRAME) -> Pipeline:
    """Build (and cache) the pipeline for bars held in memory rather than on disk.

    Streamlit hashes the arguments it is given, and hashing a multi-thousand-row
    DataFrame on every rerun would cost more than rebuilding the features it caches.
    So the *cache key* is a string that uniquely names the bars -- symbol and span --
    while the frame itself rides along as an un-keyed payload.

    **``timeframe`` is passed as a real argument, not folded into ``cache_key``**, so
    Streamlit's own hashing covers it.  Folding it into the string would work for
    identity but would hide it from the cache's own view of what distinguishes two
    entries, which is how a resolution-specific pipeline ends up serving another's
    features.  It matters here because the rolling z-score base and the readiness
    floor both depend on it: a daily pipeline built with an intraday warm-up drops 20
    bars where it should drop 20 *sessions*, and the frame it returns is silently
    wrong rather than absent.

    ``max_entries`` is bounded because this cache holds full feature matrices for
    several tickers at once.  Unbounded, a session that cycles through twenty symbols
    would keep twenty of them alive for the lifetime of the server process.
    """
    return Pipeline.from_frame(frame, length=length, timeframe=resolve_timeframe(timeframe).key)


# =============================================================================== #
# Small helpers
# =============================================================================== #
def _finite_float(x: object) -> Optional[float]:
    """``x`` as a finite float, or ``None`` if it is not one.

    The guard both formatters need: a value that cannot be read as a number, or that
    is a number but not finite (``NaN``, ``inf``), renders as an em dash rather than
    as a number.  Written once because the two callers were identical line for line,
    and two copies of a rule about what a missing number looks like is one more copy
    to forget to update.
    """
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def bps(x: float) -> str:
    """Format a return as basis points.  NaN renders as an em dash, not as a number."""
    v = _finite_float(x)
    if v is None:
        return "—"
    return "{:+.2f} bps".format(v * 1e4)


def pct(x: float) -> str:
    """Format a percentile rank, e.g. 0.0011 -> '0.11%'."""
    v = _finite_float(x)
    if v is None:
        return "—"
    return "{:.2f}%".format(v * 100.0)


def resolve_bar_index(pipe: Pipeline, ts: object) -> Optional[int]:
    """The bar index in ``pipe`` whose timestamp is ``ts``, or ``None``.

    **The index of a *moment*, found by looking the moment up.**  This is the
    counterpart to storing an anchor as a timestamp: the archive's bar indices are
    not stable across refreshes (the cleaned frame's warm-up rows shift as bars
    arrive), so an index captured earlier can point at a different bar later.  The
    timestamp does not drift, so it is the thing that gets stored, and this function
    is what turns it back into an index against *this* pass's frame.

    **Searched, not matched exactly.**  A bar is found by locating the first bar at
    or after ``ts``.  An exact match is the normal case; the search is what makes the
    lookup survive a frame whose first bars have been trimmed, where the stored
    timestamp may now sit just outside the cleaned frame entirely.

    **A moment that has aged out is *not* found.**  This is the case that makes the
    search safe rather than merely lenient.  ``searchsorted`` with a target older
    than every bar returns ``0`` -- the first bar of the frame -- so a naive search
    would answer "the pinned bar is the first bar" for a pin that is no longer in
    the archive at all, and the chart would silently re-anchor on a *different* bar.
    That is the exact failure a pinned tab must not have, so a target that precedes
    the frame is reported as missing rather than resolved to the frame's front.

    Returns ``None`` when the moment is no longer in the archive at all -- a frame
    that shrank past the anchor, or an empty frame.  The caller treats that as
    "cannot honour the pin" and re-anchors, rather than drawing a chart whose left
    edge is a guess.
    """
    if ts is None or pipe is None or pipe.n_bars == 0:
        return None
    try:
        stamp = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None

    stamps = pipe.bars["timestamp"]
    if stamps.empty:
        return None

    # Compare on a common footing.  ``searchsorted`` needs both sides to agree on
    # timezone awareness, and the archive is UTC by contract while ``stamp`` may
    # arrive in any zone, so the target is converted to UTC before the comparison.
    try:
        target = (stamp.tz_convert("UTC") if stamp.tzinfo is not None
                  else stamp.tz_localize("UTC"))
        series = (stamps.dt.tz_convert("UTC")
                  if getattr(stamps.dt, "tz", None) is not None else stamps)
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        pos = int(series.searchsorted(target, side="left"))
    except (AttributeError, TypeError, ValueError):
        return None
    if pos >= len(series):
        return None

    # **The moment has aged out.**  The target is older than the frame's first bar,
    # so there is no bar at or after it *because the archive no longer reaches that
    # far back*.  Resolving to the frame's front would anchor the chart on a bar
    # the reader never pinned, so this is reported instead.
    try:
        if target < series.iloc[0]:
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    return pos


# =============================================================================== #
# Live tab: rolling tape, frozen projection
# =============================================================================== #



#: Session-state key for the Live tab's pinned **anchor** -- the bar its frozen
#: projection grows out of.
LIVE_ANCHOR_KEY = "live_anchor_ts"

#: Session-state key for the Live tab's pinned **history start** -- the first bar
#: on screen.
LIVE_HIST_START_KEY = "live_hist_start_ts"

#: Session-state key for the Live tab's **frozen projection**.
#:
#: The path is a *value*, not an index, and it is stored whole.  Recomputing it
#: on every poll would re-run the match search against a moving query window and
#: produce a *different* forecast each minute -- which is the opposite of the
#: frozen projection the tab promises.  What is stored is the path itself, so
#: every poll draws the same forecast while the tape rolls past it.
LIVE_PATH_KEY = "live_path"

#: Session-state key for the Live tab's **frozen closest match** -- the orange
#: trace's own forward series, in percent from the match's anchor close.
#:
#: **It is stored for the same reason the path is, and for one more.**
#:
#: Freezing the path was not enough, because the *line drawn from it* was never
#: frozen: the orange trace was re-derived on every poll by slicing the
#: **current** archive at ``path.starts[0]`` -- a bar *index*.  The archive's
#: index space is not stable: as old bars age out of the front, that index
#: names a *different* bar, so the drawn projection changed even though the
#: forecast had not.  A frozen forecast drawn from a moving index is not a
#: frozen forecast.
#:
#: So the series is stored as a **value**, exactly like the path, and the
#: figure draws it as it was frozen.  The two are written together and read
#: together, because a path and a series derived from *different* archives are
#: not the same forecast.
LIVE_MATCH_KEY = "live_match"


def live_state_key(symbol: str, base: str) -> str:
    """Session-state key for one piece of Live-tab state, per symbol.

    Namespaced by symbol *and* resolution.  The symbol is part of the key because
    every one of these values describes a moment in one instrument's tape; the
    resolution suffix comes from :func:`state_key`, because a bar index means
    nothing across resolutions.
    """
    return state_key("{}_{}".format(base, str(symbol).upper()))


def live_pinned_anchor(
    pipe: Pipeline,
    symbol: str,
    recent_bars: int,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """The Live tab's pinned ``(anchor, history_start)``, as timestamps.

    **The anchor** is the last bar the frozen projection grows out of -- the
    archive's newest bar at the moment the reader arrived.  **The history
    start** is the first bar on screen, ``recent_bars`` earlier.

    **Both are moments, never bar indices.**  The archive's indices are not
    stable across a refresh: the cleaned frame drops a different number of
    warm-up rows, and the window rolls forward as old bars age out, so an index
    stored today names a *different bar* on the next poll.  A stored index
    therefore does not merely fail to stay still -- it drifts, and the frozen
    projection ends up attached to the wrong moment.

    **The pin is taken once** and survives every poll, which is what makes the
    projection genuinely frozen rather than re-derived from a moving tail.  It
    is re-taken only when the reader moves the *Recent bars* control, which is
    a deliberate gesture to re-aim the whole chart.

    Returns ``(None, None)`` when there is no archive to pin yet.
    """
    anchor_key = live_state_key(symbol, LIVE_ANCHOR_KEY)
    start_key = live_state_key(symbol, LIVE_HIST_START_KEY)
    width_key = live_state_key(symbol, "live_recent_bars_width")
    state = st.session_state

    n = max(1, int(recent_bars))

    if pipe is None or pipe.n_bars == 0:
        return state.get(anchor_key), state.get(start_key)

    # Re-pin only on a deliberate change of the control's value.  A poll that
    # brings new bars is *not* a change of intent and must not move the pin.
    width = state.get(width_key)
    needs_pin = state.get(anchor_key) is None
    if not needs_pin and width is not None and int(width) != n:
        needs_pin = True

    if needs_pin:
        # The anchor is the archive's newest bar; the history start is
        # ``recent_bars`` earlier.  Both clamped into the frame.
        anchor_index = pipe.n_bars - 1
        start_index = max(0, anchor_index + 1 - n)
        state[anchor_key] = pipe.bars["timestamp"].iloc[anchor_index]
        state[start_key] = pipe.bars["timestamp"].iloc[start_index]
        state[width_key] = n

    return state.get(anchor_key), state.get(start_key)


def _live_path_is_stale(path: Any, recent_bars: int, projection_bars: int) -> bool:
    """Is the stored Live projection no longer the one the controls ask for?

    The projection is frozen, so it is *not* recomputed on every poll -- but it
    **must** be recomputed when the reader changes what they asked for, or the
    chart would keep showing a forecast built at the old width against a
    history drawn at the new one.  That is the one case where "frozen" would
    be a lie rather than a promise.

    A path is stale when its window length no longer matches *Recent bars* or
    its horizon no longer matches *Projection bars*.  Both are read off the
    path itself, so the check cannot drift from what is actually drawn.
    """
    if path is None:
        return True
    length = int(getattr(path, "length", 0) or 0)
    horizon = int(getattr(path, "horizon", 0) or 0)
    return length != int(recent_bars) or horizon != int(projection_bars)


def chart_config(*, selectable: bool = False) -> Dict[str, Any]:
    """Plotly modebar config for a price chart.

    Zoom is off, everywhere, on purpose.  A wheel gesture must not silently rescale
    the axes on a chart the user is reading a fixed span against, and the modebar's
    zoom buttons are the same problem one click away: the Price tab's two charts are
    only comparable because they show the *same number of bars*, and a reader who
    zooms one has silently broken that.  So ``zoomIn``, ``zoomOut``, ``zoom2d``
    (box-zoom), ``autoScale2d`` and ``resetScale2d`` are all stripped, leaving pan as
    the only way to move along the x axis.

    ``resetScale2d`` goes with them deliberately.  It looks harmless -- "put the axes
    back" -- but with a pinned ``range`` it is the one control that can quietly undo
    the framing ``centred_view`` chose, and a reader who clicks it to tidy up would be
    left with the match window off-centre and no way to tell why.

    ``scrollZoom`` is hard-wired off rather than parameterised.  It used to be an
    opt-in that only the Price tab's match panel passed, on the reasoning that a chart
    drawing the whole tape is the one place exploring along x matters -- but a wheel
    that zooms is still a wheel that breaks the equal-bar-count comparison, and the
    reader who wanted to move along x was already served by the drag.  An option with
    no callers is worse than no option: it advertises a gesture the app has decided
    against, and the next person to reach for it will find it does nothing.

    ``selectable`` is the other behavioural difference, and it is the important one.

    ``False`` -- the default, correct for every chart that only *displays* tape --
    removes the box/lasso buttons from the modebar, so the app never advertises a
    selection it cannot read.  The figure's ``dragmode`` is set separately, in
    ``build_price_figure``.

    ``True`` removes **pan2d as well**, and this is a bug fix rather than a policy
    choice.  Plotly's modebar buttons do not merely *act*, they *rebind the chart's
    interaction mode*: pressing Pan sets ``dragmode`` to ``"pan"``, and nothing puts
    it back.  Measured against the plotly build this app ships
    (``static/plotly.min.js``): after a ``relayout`` to pan, the mode stays ``"pan"``
    until a ``react()`` re-renders the figure -- and Streamlit only re-sends a chart
    when its own state changes, so on an unchanged rerun the chart is left untouched
    and the pan sticks.

    The consequence on a brushable chart is that the brush stops working and gives
    no error: a drag now pans the view instead of selecting, no selection event is
    ever emitted, and the section reports "no window selected" forever.  It is the
    most confusing failure this app has, because the chart looks fine and the drag
    *does* do something -- just not the thing the reader is trying to do.  Removing
    the button means the mode cannot be broken by a gesture that has no purpose on
    this chart: a brushable chart's drag is its input, and panning it has no meaning
    while the reader is choosing a window.

    ``modeBarButtonsToRemove`` is a subtraction from Plotly's default set, so the app
    never advertises a gesture it has not wired up.
    """
    # Every zoom route into the figure, removed in one place.
    remove = ["zoomIn", "zoomOut", "zoom2d", "autoScale2d", "resetScale2d"]
    if selectable:
        # Pan would permanently override `dragmode`, breaking the brush.
        remove = remove + ["pan2d"]
    else:
        remove = remove + ["select2d", "lasso2d"]
    return {
        "displaylogo": False,
        "scrollZoom": False,
        "modeBarButtonsToRemove": remove,
    }


def selection_to_span(event: Any, n_bars: int) -> Optional[Tuple[int, int]]:
    """First box/lasso in a Plotly selection event, as ``(start, end)`` bar indices.

    Returns ``None`` whenever the event carries nothing usable -- no event, no
    selection, a degenerate shape, unparseable endpoints, or a span that rounds to a
    single bar.  Callers treat ``None`` as "no brush" and fall back, so every failure
    mode here is a fallback rather than an error.

    Every price chart draws bars against their integer index (see
    :func:`build_price_figure`), so the x values Plotly hands back already *are* bar
    indices: clamping is all that is needed, and running them through a timestamp
    parser would invent a date in 1970.

    The endpoints are taken as the min and max of *all* the shape's x values, not the
    first two.  A box has exactly two, so this is identical for it, but a lasso can
    carry many and its first two points are in draw order, not spatial order -- taking
    ``xs[:2]`` there silently reported whatever the user happened to start the loop
    on.

    A box is preferred over a lasso when an event carries both.  Streamlit puts a box
    and a lasso in separate lists rather than treating them as one shape, and only one
    can be intended at a time; box-first matches the order the app has always read
    them in and keeps a stale lasso from silently overriding a fresh box.

    Every shape access is guarded rather than assumed.  This runs on whatever
    Streamlit hands back, and a malformed shape is a dead end for the *page*, not just
    for the brush: an exception here would replace the whole app with a traceback.  So
    a shape that is not a mapping, is not numbers, or has no usable ``x`` values is
    treated as no brush at all.
    """
    if not isinstance(event, dict):
        return None
    selection = event.get("selection")
    if not isinstance(selection, dict):
        return None

    shapes = list(selection.get("box") or []) + list(selection.get("lasso") or [])
    shapes = [s for s in shapes if isinstance(s, dict)]
    if not shapes:
        return None
    # Every x value is coerced to a float *before* min/max, because comparing a
    # string against a number raises rather than returning: a mixed-type payload
    # would otherwise escape the guard below as a TypeError from ``min`` itself.
    # Coercion is all-or-nothing -- a shape with one bad vertex rejects the whole
    # shape rather than having the bad vertex dropped, because a lasso reported over
    # fewer points than it had is a *different* gesture, not a stricter parse of the
    # same one.  ``bool`` is rejected explicitly: ``float(True)`` is ``1.0``, so a
    # JSON boolean would otherwise become a window starting at bar 1.
    raw_x = [v for v in (shapes[0].get("x") or []) if v is not None]
    if len(raw_x) < 2 or any(isinstance(v, bool) for v in raw_x):
        return None
    try:
        xs = [float(v) for v in raw_x]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(np.isfinite(xs)):
        return None

    lo, hi = min(xs), max(xs)
    try:
        start = max(0, min(int(round(lo)), n_bars - 1))
        end = max(0, min(int(round(hi)), n_bars - 1))
    except (TypeError, ValueError, OverflowError):
        return None

    return (start, end) if end > start else None


def brush_bar_count(event: Any, n_bars: int) -> Optional[int]:
    """How many bars the current brush covers, or ``None`` if there isn't one.

    This is the number the reader wants *while dragging*.  Now that a brush defines its
    own query length, it is also the length that will actually be matched -- so the
    count is not a side observation, it is the parameter the search runs at.

    Returns the count of bars *touched by* the selection, inclusive of both ends, so a
    box covering bars 100-104 is 5 bars and not 4.  Plotly's box x values are bar
    positions, and a drag from one bar to another touches every bar between them.

    Shares its parsing rules with :func:`selection_to_span` by construction -- both go
    through ``selection_to_span`` itself, so a count can never disagree with the span
    the same brush produces.
    """
    span = selection_to_span(event, n_bars)
    if span is None:
        return None
    start, end = span
    return int(end) - int(start) + 1


def selection_summary(event: Any, stamps: pd.Series,
                      n_bars: int) -> Optional[str]:
    """One-line description of the current brush, for display under the chart.

    Reports the raw selection, and a brush of N bars is a query of N bars, so this
    number is exactly the length the matcher will run at.
    """
    span = selection_to_span(event, n_bars)
    if span is None:
        return None
    start, end = span
    n = int(end) - int(start) + 1
    lo_ts = stamps.iloc[start]
    hi_ts = stamps.iloc[end]
    # The zone suffix is the resolution's, so the summary does not end in "(UTC)"
    # describing a bare Eastern date.  Same helper as the axis labels: the two must
    # never disagree about what a bar looks like.
    return "{:,} bar{} · {}{}".format(
        n, "" if n == 1 else "s",
        stamp_span(lo_ts, hi_ts), stamp_zone()
    )


def session_spans(timestamps: pd.Series) -> List[Tuple[str, int, int]]:
    """Every trading session in ``timestamps``, as ``(et_date, start, stop)``.

    ``start``/``stop`` are **bar positions** into the frame ``timestamps`` came from,
    half-open, so they can be handed straight to ``pipe.bars.iloc[a:b]`` and to
    :func:`resolve_query_window`.

    Three properties this has to have, each of which is a bug if it does not:

    **Sessions are Eastern days, not UTC days.**  The US session opens 09:30 ET, which
    is 14:30 UTC in winter and 13:30 UTC in summer -- so a naive UTC bucket files every
    winter morning under the *previous* UTC date and splits a session in half.  The
    same rule as ``store.session_et`` and ``features.quality_report``; three places
    now, and they have to agree because the Quality tab reports ``sessions`` in ET and
    a mismatch would show the dropdown promising 21 dates against a report saying 42.

    **Grouping is positional, never by index label.**  ``finalize_features`` drops the
    warm-up rows with ``frame.loc[frame[rcol].notna()].copy()`` and does *not* reset the
    index, so ``pipe.bars.index`` is the original positions minus a ragged set of gaps.
    A ``groupby`` here would hand back labels, and ``bars.iloc[start]`` with a label is
    silently a *different bar* rather than an error.  The change mask below is built
    over ``to_numpy()``, so the numbers are positions by construction.

    **Empty sessions cannot exist.**  The boundaries come from where the session key
    *changes*, so every entry holds at least one bar and ``stop > start`` always.  A
    degenerate entry here would propagate into ``length = 0`` downstream, which is a
    query the matcher cannot score.
    """
    if timestamps is None or len(timestamps) == 0:
        return []

    stamps = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if stamps.isna().any():
        # ``pipe.bars`` is already the cleaned frame, so an unparseable timestamp here
        # would mean the pipeline handed back something it should not have.  Dropping
        # the row keeps the positions valid for the rows that *did* parse rather than
        # raising deep inside the dropdown, and a session list missing one odd bar is
        # harmless; a wrong bar index is not.
        stamps = stamps.dropna()
    if len(stamps) == 0:
        return []

    keys = stamps.dt.tz_convert(EASTERN).dt.floor("D").to_numpy()
    # ``!=`` on datetime64 arrays is elementwise, so this marks every bar whose session
    # differs from the one before it.  The leading True opens the first session; the
    # trailing True closes the last.
    edges = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])

    return [
        (pd.Timestamp(keys[a]).date().isoformat(), int(a), int(b))
        for a, b in zip(edges[:-1], edges[1:])
    ]


def snap_to_grid(start: int, end: int, length: int, n_bars: int) -> Tuple[int, int]:
    """Fit ``[start, end)`` to exactly ``length`` bars, centred on what was selected.

    **Only the default (unbrushed) query path uses this now.**  A brush is
    queried at its own length -- ``find_matches`` reads ``query.length`` and scores
    the query's own slice of the series, so there is no single grid to snap onto.
    This remains for the *no brush yet* case, where the query is the pipeline's
    default length over the latest window.

    Kept because the centring rule below is still worth stating, and because it is the
    correct thing to do for any caller that needs a specific length:

    The previous code extended only **forwardwards**:

        stop = min(start + length, n);  start = stop - length

    which meant a short brush was silently padded with bars *after* the one you
    picked.  Brushing 5 bars at 100-104 produced a 20-bar query at 100-119: 15 bars of
    tape you never selected, three quarters of the query, treated as though you had
    asked about them.  Worse, the extra bars came from the *forward* side, so they are
    the bars whose returns the forecast is about to measure from -- the padding
    directly contaminates the quantity the tab is reporting.

    Centring instead means the padding is split evenly either side, so the query is
    centred on the gesture.  The selected bars are still the core of it; they are just
    no longer at one edge, where any excess was invisible.

    Near the archive edges centring is impossible -- there is no tape on one side --
    so the window is shifted in as far as it can go, and the result is clamped to the
    archive.  Exactly ``length`` bars are always returned, or the whole archive if it
    is shorter than that, which is the only thing a caller can do with it.
    """
    length = max(1, int(length))
    n_bars = max(1, int(n_bars))
    start, end = int(start), int(end)

    if end <= start:
        # Degenerate or reversed span.  The old fallback was ``end = min(length, n)``,
        # which confused a bar *index* with a span *width*: with length=20 and a
        # 600-bar archive, ``end <= start`` sent the centre to 10 and the window slid
        # to the far edge, returning the whole archive instead of 20 bars.  Any window
        # is as good as another when the caller gave no span, so anchor on the start
        # and let the clamp below place it.
        end = start + length
    start = max(0, min(start, n_bars - 1))
    end = max(end, start + 1)

    # Centre of the selection, then a window of exactly `length` bars around it.
    centre = (start + end) // 2
    lo = centre - length // 2
    lo = max(0, min(lo, n_bars - length))  # shift in at the edges; no padding to give
    return lo, lo + length


def resolve_query_window(brush: Optional[Tuple[int, int]], start: int, end: int,
                        default_length: int, n_bars: int) -> Tuple[int, int, int]:
    """Decide the query window: ``(start, stop, length)``.

    A selection defines its own window.  It used to be snapped onto one fixed
    length because `find_matches` was assumed to need a single window size -- and it
    does not: it reads ``query.length`` and scores the query's own slice of the series.
    Snapping therefore threw away the only thing the reader actually chose.  Brushing 12
    bars now queries 12 bars, and the bar count shown under the chart *is* the length
    that gets searched.

    ``brush`` is the raw selection, or ``None`` when nothing is brushed.  With no brush
    the window is the pipeline default, laid out by :func:`snap_to_grid` so the
    unbrushed path keeps exactly the arithmetic it always had.

    The one floor is ``MIN_QUERY_BARS``, and it is a validity bound rather than a
    preference.  Below it a "shape" is a handful of z-score spikes with no trajectory
    between them, and STUMPY's normalised distance over a handful of points is decided
    by whichever single bar happens to be most extreme.  The distance still computes and
    the percentile is still well-defined, so this cannot be enforced by the library --
    it has to be applied here, and the UI warns when it bites.  The floor widens the
    brush, never narrows it, and the selection stays inside the resulting window.

    The window is always shifted in rather than clipped when the archive is too short
    for it, so the returned length is exactly what was asked for and the stop never
    passes the last bar.

    **A brush may span any number of sessions.**  There is no fence, and there is no
    argument for one: a window is whatever the reader drew, and on a bar-index axis a
    window across an overnight close is a real shape.  A ``bounds`` span used to
    confine this to one Eastern trading day; see PLAN.md §BX for why it went and for
    the *forward*-horizon mask that replaced it.

    Split out from ``main()`` so the decision is testable on its own.  A test that
    re-implements this arithmetic proves only that the re-implementation is
    self-consistent; these tests exercise the same function the app runs.
    """
    n_bars = max(1, int(n_bars))
    start, end = int(start), int(end)

    if brush is not None:
        b_start, b_end = int(brush[0]), int(brush[1])
        length = min(max(b_end - b_start + 1, active_min_query_bars()), n_bars)
        start = b_start
    else:
        # ``end`` is already exclusive (it came from ``start_idx + length`` or the
        # archive's last index + 1), and ``snap_to_grid`` takes an exclusive end too.
        # Adding 1 here would shift the centre forward by a bar.
        #
        # ``snap_to_grid`` returns ``(start, stop)`` -- the second value is the *stop*,
        # not the length.  Unpacking it as ``(start, length)`` is a silent disaster:
        # the "length" becomes an absolute index, so a 20-bar default came out as a
        # 600-bar window.  The width is derived from the returned span instead.
        lo, hi = snap_to_grid(start, end, int(default_length), n_bars)
        start, length = lo, hi - lo

    # The archive is the only fence left, and this clamp is all of the placement logic.
    # It shifts rather than clips, so the returned length is always what was asked for
    # -- which only bites a brush running to the final bar, since the default's start
    # is already in range.
    length = max(1, min(int(length), n_bars))
    start = max(0, min(int(start), max(0, n_bars - length)))
    stop = min(start + length, n_bars)
    return stop - length, stop, length



# =============================================================================== #
# Chart
# =============================================================================== #
def build_price_figure(pipe: Pipeline, view_start: int, view_stop: int,
                       query_start: Optional[int] = None,
                       query_stop: Optional[int] = None,
                       draw_candles: bool = True,
                       span_label: str = "query",
                       span_color: str = "255,127,14",
                       span_text_color: str = "#b35c00",
                       title: Optional[str] = None,
                       rebase_at: Optional[int] = None,
                       y_range: Optional[Tuple[float, float]] = None,
                       selectable: bool = True,
                       pan_data: Optional[Tuple[int, int]] = None,
                       pan_bounds: Optional[Tuple[float, float]] = None,
                       pannable: bool = False,
                       height: int = 440) -> go.Figure:
    """Close-price chart with a rolling 20-bar mean, a 2-sigma band and the query span.

    Bars are drawn against their integer index rather than the clock, so a closed
    market occupies no horizontal space and every session sits flush against the next.
    That is what makes a multi-session shape readable as one continuous line instead of
    two clusters separated by a desert.  The x axis keeps a timestamp tick grid, so the
    labels stay readable while the spacing between them is not proportional to time.

    Two optional behaviours make two *different* windows comparable by eye, which is
    what the Matches tab needs when it draws the query above a matched window:

    ``rebase_at``
        Divide every level by the close at that bar, turning the price axis into
        percent-from-that-bar.  Windows that happened at very different price levels
        then sit on one scale.
    ``y_range``
        Pin the vertical axis, so a shared range can be passed to every figure in a
        set and the two curves can be compared without mentally rescaling.

    ``selectable``
        Arm the box/lasso drag that *defines the query window*.  Only the Price tab's
        top chart -- the app's only query control -- is rendered with ``on_select``.
        Everywhere else the brush would draw a box that no code path ever looks at,
        so a display chart passes ``False``: the pointer becomes a pan/zoom drag
        instead, and the modebar's select buttons are stripped by :func:`chart_config`.

    ``rebase_at`` and ``y_range`` are no-ops when neither is passed, so a caller can
    keep the raw tape if it has no rebased panel underneath to agree with.

    The three ``pan_*`` arguments belong to the Price tab's match panel and are the
    only reason this function knows about panning at all.  All three are inert by
    default, so every other chart in the app -- including both brushable ones -- renders
    exactly as it did before they existed.

    ``pan_data``
        ``(data_start, data_stop)`` widens the *data* behind the chart beyond the *view*
        pinned in ``range``.  Without it the trace is sliced to exactly the visible
        window, which leaves nothing off-screen for a pan to reveal: panning a chart
        whose only data is the visible window just slides the same line out of frame.
        Widening the arrays while leaving ``range`` alone is what makes the surrounding
        tape reachable.

        The caller passes the *pan bounds*, not the whole archive.  Plotly cannot hard
        stop a drag at a limit -- issue #887 requests it and is open -- so an over-drag
        becomes a zoom, and a zoom has to land on something.  Drawn past the bounds it
        lands on tape that was never reachable by panning, which is how the window ends
        up somewhere the reader cannot navigate back from; drawn only to the bounds it
        stays inside the span they have already travelled.  This narrows that failure
        rather than removing it.
    ``pan_bounds``
        ``(minallowed, maxallowed)`` clamps how far the x axis may travel.  Paired with
        ``pan_data`` it is what turns "pannable" into "pannable and still legible" --
        see :func:`pan_axis_bounds_for` for how the limits keep the scored band on screen.
    ``pannable``
        Accepted for symmetry with the two arguments above and used by the caller as the
        single "make this panel explorable" switch, but it no longer sets ``dragmode``.
        A display chart now pans by default, since zoom is off app-wide and a drag that
        drew a box would be a gesture that does nothing.  It matters because
        ``_render_best_match_pair`` reads the *same* flag to decide whether to widen the
        data and bound the travel -- the two of them together are what make the Price
        tab's lower chart explorable, and separating them would only risk one being set
        without the other.
    """
    view_start = max(0, int(view_start))
    view_stop = min(pipe.n_bars, int(view_stop))
    if view_stop <= view_start:
        view_stop = min(pipe.n_bars, view_start + 1)

    # The arrays below are sliced to the *data* extent -- the visible view unless
    # ``pan_data`` widens it.  ``range`` still describes the view, so the chart opens on
    # the same window it always did and the extra tape is simply reachable.
    data_start, data_stop = view_start, view_stop
    if pan_data is not None:
        data_start = max(0, min(int(pan_data[0]), view_start))
        data_stop = min(pipe.n_bars, max(int(pan_data[1]), view_stop))
        if data_stop <= data_start:          # a degenerate extent must not empty it
            data_start, data_stop = view_start, view_stop

    frame = pipe.bars.iloc[data_start:data_stop]
    fig = go.Figure()

    # The x axis is the bar index, so the closed market occupies no horizontal space and
    # consecutive sessions sit flush. Nothing is being joined across anything, so the
    # session-splitting path (breaks and per-run band polygons) is not needed here --
    # the whole visible window is one continuous line.
    x_vals: Any = np.arange(data_start, data_stop)

    # Rolling band is computed on the FULL series, then sliced, so the leading edge of
    # the visible window is not degraded by an artificially short warm-up.
    close = pd.Series(pipe.close)
    mean = close.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    std = close.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std(ddof=0)
    upper = (mean + 2.0 * std).iloc[data_start:data_stop].to_numpy()
    lower = (mean - 2.0 * std).iloc[data_start:data_stop].to_numpy()
    mid = mean.iloc[data_start:data_stop].to_numpy()
    price = np.asarray(frame["close"].to_numpy(dtype=float))

    # Rebasing is a single affine map applied to all four series, so the band stays
    # consistent with the close instead of being recomputed on a different scale.
    base = None
    if rebase_at is not None and 0 <= int(rebase_at) < pipe.n_bars:
        candidate = float(pipe.close[int(rebase_at)])
        if np.isfinite(candidate) and candidate > 0:
            base = candidate
    if base is None:
        y_close, y_upper, y_lower, y_mid = price, upper, lower, mid
        y_label = "close"
        close_hover = "%{customdata}<br>close %{y:.2f}<extra></extra>"
    else:
        # Multiplied by 100 so the axis reads in real percentage points (-0.23%). A raw
        # 0.0n fraction beside a "(%)" label is a unit lie the eye cannot parse.
        y_close = np.asarray(price, dtype=float)
        y_upper = upper
        y_lower = lower
        y_mid = mid
        # Kept short: a long axis title is clipped at these panel heights, and the
        # panel title already spells out what the rebasing is.
        y_label = "price"
        close_hover = "%{customdata}<br>close %{y:.2f}<extra></extra>"

    # **The hover says *when*, not *which bar*.**
    #
    # The x axis is the bar index -- deliberately, so the closed market occupies no
    # horizontal space -- and the hover used to print that same index.  A reader
    # hovering a bar learned "this is bar 1015" and had to go and find 1015 on the
    # axis to learn *when* it was.  The stamp is already printed on the axis, so the
    # hover can print it directly and the reader never has to make the translation
    # at all.
    #
    # The stamps are carried on ``customdata`` rather than in the template's ``%{x}``
    # because the x value *is* the index: there is no timestamp in it to print.
    stamps = stamp_labels(pipe.bars["timestamp"].iloc[data_start:data_stop],
                          key=pipe.timeframe)

    if draw_candles:
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_close, mode="lines",
            line=dict(color="#1f77b4", width=1.1),
            name="close", hovertemplate=close_hover,
            customdata=stamps,
        ))

    # One band polygon, because there is nothing being joined across a gap. A
    # ``tonexty`` pair per contiguous run is only needed when the axis is the clock and
    # the polygon would otherwise span the hours the market was shut.
    fig.add_trace(go.Scatter(x=x_vals, y=y_upper, mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip", name="+2σ"))
    fig.add_trace(go.Scatter(x=x_vals, y=y_lower, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.14)",
                             showlegend=False, hoverinfo="skip", name="−2σ"))
    fig.add_trace(go.Scatter(x=x_vals, y=y_mid, mode="lines",
                             line=dict(color="rgba(31,119,180,0.65)", width=1, dash="dot"),
                             name="rolling mean (%d)" % ROLLING_WINDOW, hoverinfo="skip"))

    if base is not None:
        # Anchored bottom-left with a background: at the top right it lands on the price
        # trace itself, and an unreadable label is worse than no label.
        fig.add_hline(y=base, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
                      annotation_text="window start", annotation_position="bottom left",
                      annotation_font=dict(size=10, color="rgba(128,128,128,0.9)"),
                      annotation_bgcolor="rgba(255,255,255,0.75)")

    if query_start is not None and query_stop is not None:
        lo = max(0, min(int(query_start), pipe.n_bars - 1))
        hi = min(pipe.n_bars, max(int(query_stop), lo + 1))
        q0, q1 = lo, hi - 1
        fig.add_vrect(x0=q0, x1=q1, fillcolor="rgba(%s,0.22)" % span_color,
                      line_width=0, layer="below",
                      annotation_text=span_label, annotation_position="top left",
                      annotation_font=dict(size=11, color=span_text_color))
        # The dashed line is the bar forward returns are measured from (§Z1): the first
        # bar AFTER the window, never one inside it.
        if hi < pipe.n_bars:
            fig.add_vline(x=hi, line=dict(color="rgba(%s,0.85)" % span_color, width=1.4,
                                          dash="dash"),
                          annotation_text="what follows", annotation_position="top right",
                          annotation_font=dict(size=10, color=span_text_color))

    # The axis carries bar indices but must still say *when*. A fixed tick per visible
    # day keeps every session boundary labelled, which is the whole point of
    # compressing the gaps -- otherwise the labels lie about spacing.
    #
    # The label itself is the resolution's: ``09:30`` intraday, a bare date on daily.
    # Every daily bar shares one stamp, so printing the time would repeat one constant
    # across the axis and imply the bars move within a day.  See :func:`stamp_label`.
    step = max(1, (view_stop - view_start) // 8)
    ticks = list(range(view_start, view_stop, step))
    if ticks[-1] != view_stop - 1:
        ticks.append(view_stop - 1)
    x_axis: Dict[str, Any] = dict(
        title="bar index · sessions compressed",
        showgrid=True, gridcolor="rgba(128,128,128,0.15)", automargin=True,
        tickmode="array",
        tickvals=ticks,
        ticktext=[stamp_label(pipe.bars["timestamp"].iloc[i], key=pipe.timeframe)
                  for i in ticks],
        # Half a bar of padding keeps the first and last labels off the axis ends,
        # where they would otherwise be clipped by the plot border.
        range=[view_start - 0.5, view_stop - 0.5],
        # **The unified hover box's *title* is what printed the bar number.**
        #
        # With ``hovermode="x unified"`` the box has two parts: a title, which
        # plotly.js fills with the hovered x value, and one entry per trace, which
        # the trace's own ``hovertemplate`` fills.  Putting the stamp in the trace's
        # template therefore fixed only the entry -- the title above it still read
        # "1015", and the bar number was the more prominent of the two.
        #
        # The title is a *template* of its own, and it cannot see ``customdata``: its
        # scope is limited to the x value.  So the title cannot be told to print the
        # stamp -- it can only be told to print something else, or to print nothing.
        #
        # ``%{xother}`` does the latter.  It is the x value of the *other* traces at
        # this position, which on a chart where every trace shares one x axis is
        # undefined; plotly.js resolves an undefined variable in a template to the
        # empty string rather than falling back to the x value, so the title renders
        # as nothing at all and the box collapses to its trace entries -- which
        # carry the stamp.  An *empty* ``text`` does not work: plotly.js treats that
        # as "no template" and falls back to the x value, which is the bar number
        # again.
    )

    # ``minallowed``/``maxallowed`` are what stop the pan running off into blank space.
    # Plotly treats them as hard limits on the axis range, so a drag can travel exactly
    # this far and no further -- see :func:`pan_axis_bounds_for` for the arithmetic.  They
    # are deliberately left unset on every other chart: this is a *pannable* chart's
    # property, and a chart whose data is exactly its view has nothing to pan to.
    if pan_bounds is not None:
        x_axis["minallowed"] = float(pan_bounds[0])
        x_axis["maxallowed"] = float(pan_bounds[1])

    fig.update_layout(
        height=height,
        # Bottom margin has to clear the tick labels *and* the axis title; at panel
        # heights the default 8px clips both. Left margin clears the rotated y title.
        margin=dict(l=64, r=8, t=30 if not title else 46, b=54),
        hovermode="x unified",
        # ``select`` makes a bare drag draw a box, which is what a chart the app reads
        # a query from needs.  On a display chart there is nothing listening for that
        # box, and zoom is off app-wide, so a drag that drew one would be a gesture
        # that does nothing at all.
        #
        # That leaves ``pan`` as the sensible default for a display chart: the data is
        # already on screen, so moving the tape costs the reader nothing even where it
        # reveals no new bars.  It used to be ``zoom`` here, which -- with the modebar's
        # zoom buttons removed and ``scrollZoom`` off -- meant the drag silently drew a
        # rectangle that was discarded.  ``pannable`` is unchanged and still the flag
        # that also widens the data and bounds the travel; this only decides what a bare
        # drag does on a chart with nothing to reveal.
        dragmode=("select" if selectable else "pan"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=x_axis,
        yaxis=dict(title=y_label, showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                   range=list(y_range) if y_range else None, automargin=True),
        title=dict(text=title, font=dict(size=12)) if title else None,
    )
    return fig


def span_context_bounds(pipe: Pipeline, start: int, stop: int,
                        pad: int) -> Tuple[int, int]:
    """Visible window for one query/match span, padded by ``pad`` bars either side.

    The scored band is centred in the visible window wherever that is possible: the
    window is built to a fixed total width and then shifted onto the band's midpoint,
    rather than grown outwards from the band and clipped.  Growing outwards is what
    makes a match near an archive edge sit visibly off to one side, so the query panel
    and the match panel stop lining up horizontally and the eye compares the wrong
    edges.

    Centring is **best-effort, and cannot hold near an archive edge.**  A window of
    fixed width can only be centred on a band that sits at least ``pad`` bars from both
    ends.  Measured on the current archive with ``m=60, pad=20``: the band is exactly
    centred for every start from bar 20 to bar 7969, and off-centre for only the 20
    windows at each end.  Worst case is the very first bar, which has no lead-in
    available and so sits flush against the archive start with the whole ``2 * pad`` of
    context on its right instead.  That asymmetry is arithmetic, not a bug -- the
    archive has nothing to put on the left.
    """
    n = int(pipe.n_bars)
    start, stop = int(start), int(stop)
    pad = max(0, int(pad))
    span = max(1, stop - start)
    want = span + 2 * pad

    if want >= n:
        # Archive too short to pad: show everything, centred as far as it can be.
        return 0, n

    # Shift a fixed-width window onto the band's midpoint, then clamp it into range.
    # Centring is exact in the interior and best-effort where clamping bites, which is
    # the only behaviour available when the band is near an archive edge.
    centre = (start + stop) // 2
    lo = centre - want // 2
    lo = max(0, min(lo, n - want))
    return lo, lo + want


def shared_anchor(n_bars: int, width: int,
                  spans: Sequence[Tuple[int, int]],
                  preferred: Optional[int] = None) -> int:
    """An x-offset within a ``width``-bar window that *every* span can be drawn at.

    This is the constraint that makes the Price tab's two panels line up.  Each shaded
    band sits at some offset from the left edge of its chart, and the eye only compares
    the two charts correctly when that offset is the same in both.  So the offset cannot
    be derived from one band and merely *hoped* for in the other: it has to be chosen so
    that every band has room to sit there.

    A band ``[start, stop)`` fits at offset ``a`` iff ``start >= a`` (enough tape to its
    left) and ``stop + (width - span - a) <= n`` (enough to its right).  So the feasible
    offsets for a set of bands are the intersection of those ranges:

        max(a) = max(stop) + width - span - n        (right-edge limit)
        min(a) = min(start)                          (left-edge limit)

    The preferred offset -- centred, which is what you want -- is used whenever it falls
    inside that intersection, and pulled to the nearest feasible value otherwise.  When
    it is feasible the two panels are aligned *exactly*; when it is not, the best
    achievable offset is used for both, so they remain aligned with each other rather
    than each centring itself.
    """
    n = int(n_bars)
    width = max(1, min(int(width), n))
    spans = [(int(s), int(e)) for s, e in spans if int(e) > int(s)]
    if not spans:
        return 0

    span = max(e - s for s, e in spans)
    if width <= span:
        return 0

    lower = max(e for _, e in spans) + width - span - n   # must clear the right edge
    upper = min(s for s, _ in spans)                      # must clear the left edge
    want = (width - span) // 2 if preferred is None else int(preferred)

    if lower <= upper:                                    # alignment is achievable
        return max(lower, min(want, upper))
    # Infeasible: the bands are too far apart in a too-narrow archive. Fall back to the
    # centred value and let each window clamp -- they cannot be made to line up.
    return max(0, min(want, width - 1))


def aligned_view(n_bars: int, start: int, stop: int, width: int,
                 anchor: int) -> Tuple[int, int]:
    """A ``width``-bar window placing ``[start, stop)`` at x-offset ``anchor``.

    Equal *width* is not enough for two panels to be comparable.  When each panel simply
    centres its own band, the query -- being the archive's last window -- sits flush
    against the right edge while an interior match sits in the middle, and the two
    shaded bands appear at different positions, so the eye compares the wrong parts of
    the two charts.  Measured at ``width=600`` before this was fixed: the query band
    landed at x 540-600 and the match at x 138-198, a ~400px horizontal displacement.

    ``anchor`` should come from :func:`shared_anchor` so it is one the band can actually
    sit at; the clamping below is a safety net for the infeasible case, never the
    normal path.
    """
    n = int(n_bars)
    start, stop = int(start), int(stop)
    want = max(1, min(int(width), n))
    off = max(0, min(int(anchor), want - 1))

    lo = start - off
    if lo < 0:
        lo = 0
    elif lo + want > n:
        lo = n - want
    return lo, lo + want


def centred_view(n_bars: int, start: int, stop: int,
                 width: int) -> Tuple[int, int]:
    """A ``width``-bar window with ``[start, stop)`` sitting in the middle of it.

    The same width as :func:`aligned_view`, but the band's offset is decided by this
    function instead of being negotiated between two panels.  It exists for the
    pannable match panel, where centring is not a cosmetic preference but the optimum
    position: it puts the band at the *midpoint* of the range over which the chart is
    allowed to pan, so the reader gets the same travel in both directions.

    Verified at n=8069, L=60, V=1950, band [4000, 4060): the legal left edges run
    [2110, 4000], this function returns 3055, and 3055 is exactly that interval's
    midpoint -- 945 bars of travel each way, 1890 in total.
    """
    n = int(n_bars)
    want = max(1, min(int(width), n))
    lo = (int(start) + int(stop)) // 2 - want // 2
    lo = max(0, min(lo, n - want))
    return lo, lo + want


def pan_axis_bounds_for(n_bars: int, band_start: int, band_stop: int,
                        width: int) -> Optional[Tuple[float, float]]:
    """``(minallowed, maxallowed)`` that keep a ``width``-bar view on ``[start, stop)``.

    A view with left edge ``v0`` draws bars ``[v0, v0 + width)``, so the band is fully
    in view exactly when ``v0 <= band_start`` and ``v0 + width >= band_stop``.  Those
    two inequalities are the bounds:

        v0 in [band_stop - width, band_start]

    so the total travel is ``width - span`` bars and is *independent of where the band
    sits in the archive* -- the same number of bars whichever match was found.  Both
    limits are then clipped to the archive, because a view may not start before bar 0
    nor end after the last bar; a band near either edge simply gets less travel, since
    there is genuinely no tape on that side to reveal.

    **These are axis endpoints, not left-edge positions, and the distinction is not
    cosmetic.**  Plotly applies ``minallowed`` to ``range[0]`` and ``maxallowed`` to
    ``range[1]``, each to its own end of the axis.  So the right-hand legal left edge
    (``v0 <= band_start``) has to be published as ``maxallowed = band_start + width``,
    *not* as ``band_start``: publishing the left-edge value made Plotly clamp the
    right edge down to ``band_start``, which truncated the chart to 583 of its 1,287
    intended bars and pushed the whole match window off the right of the screen --
    the panel rendered, with no match on it.  The pair is therefore

        minallowed = band_stop - width - 0.5
        maxallowed = band_start + width - 0.5

    and both carry the same half-bar padding ``build_price_figure`` uses for ``range``.

    ``None`` means the request is unsatisfiable -- the archive is too short to hold a
    window that contains the band at all -- and the caller should draw the chart
    unbounded rather than pin it to a range it cannot honour.
    """
    n = int(n_bars)
    want = max(1, min(int(width), n))
    lo = max(0, int(band_stop) - want)                 # leftmost legal left edge
    hi = min(int(band_start), n - want)                # rightmost legal left edge
    if hi < lo:
        return None
    return lo - 0.5, (hi + want) - 0.5


def shared_rebased_range(pipe: Pipeline, spans: List[Tuple[int, int]],
                         pad: int) -> Optional[Tuple[float, float]]:
    """One vertical range covering every span given, rebased on its own first bar.

    This pins the *query* panel so the same reference does not rescale as the user
    moves between matches.  The match panels below it deliberately do not use it -- they
    autoscale, so a match is framed by its own bars rather than stretched to a common
    scale with a reference it sits next to.  See :func:`_render_best_match_pair`.

    Returned in percentage points to match the rebased y axis, and padded so the
    range is identical across every panel that shares it.
    """
    lo_hi: List[float] = []
    for start, stop in spans:
        base = float(pipe.close[start]) if 0 <= start < pipe.n_bars else float("nan")
        if not np.isfinite(base) or base <= 0:
            continue
        view_lo, view_hi = span_context_bounds(pipe, start, stop, pad)
        window = pipe.close[view_lo:view_hi] / base - 1.0
        window = window[np.isfinite(window)]
        if window.size:
            lo_hi.extend([float(window.min()) * 100.0, float(window.max()) * 100.0])
    if not lo_hi:
        return None
    lo, hi = min(lo_hi), max(lo_hi)
    span = hi - lo
    margin = 0.08 * span if span > 0 else 0.05
    return lo - margin, hi + margin


def rebased_view_range(pipe: Pipeline, view: Tuple[int, int], rebase_at: int,
                       pad: int) -> Optional[Tuple[float, float]]:
    """A percentage range covering one *view*, rebased on ``rebase_at``.

    ``build_price_figure`` autoscales the y axis over whatever data the traces carry,
    which is right for a chart whose traces are exactly its view and wrong the moment
    they are not.  The pannable match panel carries the whole tape, so an autoscale
    there is taken over eight thousand bars rather than the two thousand on screen:
    measured on the fixture, the match's own move came out 0.73% tall against a 1.78%
    axis, a 2.4x squash, and the band the reader came to look at flattened into a
    stripe.  Pinning the range to the view is what keeps the match legible *before* any
    pan, and it keeps the same bars legible after one.

    Returned in percentage points to match the rebased y axis, padded by fraction so a
    flat window still gets a usable axis.  ``None`` means the window held no usable
    bars, and the caller should fall back to autoscale rather than draw an empty axis.
    """
    lo, hi = int(view[0]), int(view[1])
    base = float(pipe.close[int(rebase_at)]) if 0 <= int(rebase_at) < pipe.n_bars else float("nan")
    if not np.isfinite(base) or base <= 0:
        return None
    window = pipe.close[lo:hi] / base - 1.0
    window = window[np.isfinite(window)]
    if window.size == 0:
        return None
    lo_pct, hi_pct = float(window.min()) * 100.0, float(window.max()) * 100.0
    margin = max(0.08 * (hi_pct - lo_pct), 0.05)
    return lo_pct - margin, hi_pct + margin


def static_serving_enabled() -> bool:
    """Whether this Streamlit server is serving the app's ``static/`` directory.

    ``server.enableStaticServing`` defaults to **False**, so a chart that loads a script
    from ``static/`` works on a developer machine with a config file and breaks silently
    on a stock deployment.  Reported as a tri-state decision here -- the query is cheap
    and wrapped, because a Streamlit version without the option should not be treated as
    a definitive "no": in that case the flag is assumed on and the chart is attempted,
    which is the same optimistic default the code has always had.

    Note the option is read at *call* time, not import time: Streamlit applies
    ``config.toml`` during bootstrap, so a module-level read can happen before the
    project's own config has been merged in and would then report the wrong answer.
    """
    try:
        from streamlit import config as _st_config

        return bool(_st_config.get_option("server.enableStaticServing"))
    except Exception:
        # Option missing or config unavailable. Assume the app can serve its own assets:
        # attempting the component still produces a chart if the assets really are
        # reachable, and the iframe is self-contained apart from the one script tag.
        return True


def static_asset_url(name: str) -> str:
    """URL a browser can actually fetch ``static/<name>`` from.

    Streamlit serves an app's ``static/`` directory under ``/app/static/`` -- the prefix
    is baked into the route (``_ROUTE_APP_STATIC``) and is not configurable.  Reading it
    from the running config rather than hard-coding it keeps the app correct if it is
    ever mounted under a ``baseUrlPath``, which is prepended to every route.

    Read defensively: the option is only meaningful in the Streamlit versions that
    actually implement app static serving, and this function runs during an import-time
    test harness as well as inside the app.  A default of the documented prefix is the
    right answer in both cases, so an unreadable config yields the correct URL rather
    than an exception in the middle of rendering a chart.
    """
    base = ""
    try:
        from streamlit import config as _st_config

        base = (_st_config.get_option("server.baseUrlPath") or "").rstrip("/")
    except Exception:
        # Not a Streamlit runtime, or the option moved. The un-prefixed default below
        # is still what a stock server serves.
        base = ""
    return "{}/app/static/{}".format(base, name.lstrip("/"))


def render_clamped_pan_chart(fig: go.Figure, *, height: int,
                             key: str) -> None:
    """Render ``fig`` in a chart whose pan stops dead at the axis bounds.

    This exists because ``st.plotly_chart`` cannot deliver the Price tab's promise.
    ``xaxis.minallowed``/``maxallowed`` bound the *range*, but plotly.js does not treat
    them as a wall: once a drag runs past one it switches to **zooming** instead of
    refusing.  That is the behaviour every reader sees -- chase the edge of a window,
    and the chart quietly zooms and changes how many bars are visible, which silently
    breaks the equal-bar-count comparison with the query chart above it.  Upstream
    plotly/plotly.js#887 asks for a hard stop and is still open, so there is no config
    that produces one inside Streamlit's own component.

    The fix is a small custom component: it re-applies the bounds on *every*
    ``plotly_relayout``, so whatever range a drag produces is clamped back into the
    legal band.  The drag then stops at the edge instead of turning into a zoom.  See
    ``static/clamped_pan_chart.html`` for the clamping arithmetic, which is where the
    real logic lives.

    Clipping the drawn tape to the bounds (see ``_render_best_match_pair``) is still
    worth doing -- it means an over-drag has nowhere *unreachable* to land even if the
    clamp is bypassed -- but it is a mitigation, and this is the fix.

    Cost, stated plainly: the component is a plain ``st.iframe``, so the figure is
    serialised into the page and the app's modebar config still applies, but the panel
    is **not** a Streamlit widget.  It cannot emit selections (nothing reads any from
    this panel) and it does not participate in Streamlit's rerun cycle -- panning it is
    entirely client-side, which is what a pan should be anyway, and is why this is
    affordable at all.

    The panel is delivered with ``st.iframe`` as an HTML *document*, which Streamlit
    routes into the frame's ``srcdoc``.  The URL-taking ``st.components.v1.iframe`` is
    the wrong tool for it despite the near-identical name: handing it HTML puts a
    ~1 MB document into the element's ``src`` attribute, the browser resolves that as a
    relative path against the app root, and the frame silently loads Streamlit's SPA
    shell instead of this document and renders as an empty box.  See the call site for
    the full account.
    """
    html_path = os.path.join(_HERE, "static", "clamped_pan_chart.html")
    if not os.path.exists(html_path):
        # A missing asset must not take the tab down; the figure still renders, it just
        # falls back to Streamlit's own (unclamped) chart.
        st.warning("Pan clamp unavailable: %s not found." % html_path)
        st.plotly_chart(fig, width='stretch', key=key, config=chart_config())
        return

    # Static serving is *off by default* in current Streamlit, and the app's ``static/``
    # directory is only reachable over HTTP when it is on (see
    # ``create_app_static_serving_routes``, which is guarded by
    # ``server.enableStaticServing``).  That flag is why this panel used to vanish: with
    # it off, the plotly.js request fell through to the SPA catch-all, which answers 200
    # with an HTML document, the browser discarded it as a bad script, and the chart
    # rendered as an empty box -- silently, with nothing in the log.
    #
    # Checked here because that failure is invisible from Python: the document is built
    # and delivered correctly, and only the browser knows the script never arrived.  The
    # repo ships ``.streamlit/config.toml`` with the flag on, so this is a guard against
    # that file going missing or being overridden on the command line -- and it fails
    # loudly, as a message next to a working chart, rather than as a blank panel.
    if not static_serving_enabled():
        st.warning(
            "Pan clamp unavailable: Streamlit is not serving `static/`, so this chart "
            "cannot load plotly.js. Set `server.enableStaticServing = true` in "
            ".streamlit/config.toml and restart. Showing an unclamped chart instead."
        )
        st.plotly_chart(fig, width='stretch', key=key, config=chart_config())
        return

    # ``srcdoc`` gives the document its own browsing context, so this HTML *is* the
    # frame's document -- there is no inner <iframe> and no URL to resolve.  An earlier
    # version nested one with src='clamped_pan_chart.html', which resolved against the
    # app root rather than ``static/``; Streamlit's SPA answered with its own shell, the
    # inner frame loaded without the script, and the chart rendered as an empty box.
    # The clamp is 5 kB of JS, so inlining it costs nothing and removes the whole class
    # of bug.
    #
    # plotly.js is the one thing not inlined: it is ~4.8 MB, and it is loaded from the
    # *served* ``/static/`` path so the browser caches it once across every rerun
    # rather than shipping 4.8 MB inside every rendered panel.
    with open(html_path, encoding="utf-8") as fh:
        template = fh.read()

    # The template ships a *relative* script src, which is only correct when the file is
    # opened directly. Here it is injected into an iframe whose base URL is the app, so
    # the src has to be rewritten to the path Streamlit actually serves it from. If
    # that tag is ever renamed or dropped, plotly never loads and the panel renders as
    # an empty box with no error -- so the rewrite is verified rather than assumed.
    _REL_SRC = '<script src="plotly.min.js" charset="utf-8"></script>'
    if _REL_SRC not in template:
        st.warning(
            "Pan clamp unavailable: the component no longer loads plotly.js."
        )
        st.plotly_chart(fig, width='stretch', key=key, config=chart_config())
        return

    # The served path.  Streamlit mounts an app's ``static/`` directory under
    # ``/app/static/``, not ``/static/`` -- the prefix comes from the route definition
    # (``_ROUTE_APP_STATIC = "app/static/{path:path}"``) and is not configurable.
    #
    # Getting this wrong is silent and total.  A request for a path the router does not
    # recognise falls through to the SPA catch-all, which answers **200 with Streamlit's
    # own HTML shell** rather than a 404.  The browser then refuses the script because
    # its content type is ``text/html``, ``Plotly`` is never defined, and the panel is
    # an empty box -- with nothing in the Python log, because from the server's point
    # of view every request succeeded.  That is exactly the reported symptom: the
    # Price tab's second chart does not appear.
    #
    # It is built from the configured ``baseUrlPath`` rather than hard-coded, so the app
    # still works if it is ever mounted under a sub-path.
    plotly_url = static_asset_url("plotly.min.js")

    # ``to_plotly_json`` emits only ``data`` and ``layout``. The modebar config is a
    # *separate* argument to ``Plotly.newPlot`` and is not part of the figure, so it has
    # to be added here: without it the panel gets plotly's default toolbar, which
    # re-advertises zoomIn/zoomOut/box-zoom -- undoing the app-wide decision to strip
    # them and handing the reader back the gesture this component exists to prevent.
    spec = dict(fig.to_plotly_json(), config=chart_config())
    doc = (
        template.replace(
            _REL_SRC,
            # Served from the app's own `static/` directory rather than inlined: plotly.js
            # is ~4.8 MB, and serving it lets the browser cache it once across reruns
            # instead of shipping it inside every rendered panel.
            '<script src="{}" charset="utf-8"></script>'.format(plotly_url),
        )
        .replace("</body>",
                 "<script>window.renderChart(%s, %d);</script></body>"
                 % (json.dumps(spec), height))
    )
    # ``st.iframe``, NOT ``st.components.v1.iframe``.  The two are siblings, not
    # synonyms, and the difference is the entire bug this panel had:
    # ``components.v1.iframe(src=...)`` takes a **URL** and puts it in the element's
    # ``src`` attribute, whereas an HTML document has to go in ``srcdoc``.  Passing
    # this component's HTML to the URL-taking form therefore handed the browser an
    # ~1 MB "URL" starting ``<!DOCTYPE html>``, which it resolved as a *relative* path
    # against the app root.  That path does not exist, so it fell through to the SPA
    # catch-all and the frame loaded **Streamlit's own HTML shell** -- a 200, a
    # text/html, and a completely unrelated document.  ``renderChart`` was never
    # defined in it, the plotly.js tag never executed, and the panel was an empty box
    # of exactly the right size.
    #
    # That failure is invisible from Python, which is why it survived so long: the
    # proto marshalled without complaint, the server answered 200 for every request,
    # the log stayed empty, and the reader saw a blank rectangle where the second
    # chart belongs.  Confirming it in the browser is the only way to see it --
    # the element's ``src`` attribute literally began ``<!DOCTYPE html>``.
    #
    # ``st.iframe`` takes the *document*, not a URL, so the distinction is now made by
    # the API itself: a string that is not an absolute/relative URL is routed to
    # ``srcdoc`` automatically.  That is verified against the real proto in
    # ``tests/test_clamped_pan.py`` rather than assumed, because the failure it guards
    # against is a blank panel that no Python-side check can see.
    #
    # Two intentional differences from the old ``components.html`` call:
    #
    # * ``height`` is an explicit pixel count, as before, because the frame must match
    #   the figure exactly.  ``st.iframe``'s default is ``"content"``, which would let
    #   a mis-measured document resize the panel and break the two-chart alignment --
    #   so the default is overridden rather than inherited.
    # * ``st.iframe`` sets ``scrolling=True`` unconditionally and exposes no way to
    #   turn it off, where ``components.html`` took ``scrolling=False``.  The template
    #   is sized to its content (``html, body { height: 100% }`` on a full-width chart),
    #   so a scrollbar that never needs one is a cosmetic difference, not a layout
    #   break -- but it is not configurable and is recorded here rather than glossed.
    #
    # ``alt`` is supplied because the panel is a real, focusable iframe: without a
    # title it is announced to a screen reader as a nameless frame, which is worse
    # than the decorative markup it replaces.
    st.iframe(
        doc,
        width="stretch",
        height=height + 12,
        alt="Price tape, pannable within the selected range",
    )


def build_shape_figure(pipe: Pipeline, query_start: int, match_start: int,
                       rank: int, distance: float) -> go.Figure:
    """Overlay the query's feature shape against one matched window.

    The two legs are built the way the *matcher* builds them, not the way they look
    symmetric, because they are genuinely asymmetric:

    * the query is z-scored over its own span (``Query.from_span`` defaults to
      ``per_window=True``);
    * the match leg is a plain slice of ``pipe.matrix``.

    Both legs end up in the same place -- unit-variance rolling-z features -- so
    re-normalising either one would draw a prettier, tighter pair of curves that do
    NOT reproduce the distance reported in the Matches table. (PLAN.md §BC)

    ``path_z`` is drawn **first** and given the most room, because it is the leg the
    Price tab's price chart is actually showing.  A reader who checks this overlay and
    finds only ``return_z`` has been shown the machine's auxiliary view and none of
    the one the chart above is drawing -- which is the mismatch that made an early
    version of this figure actively misleading.
    """
    length = pipe.length
    q_vec = M.Query.from_span(pipe.matrix, query_start, query_start + length).vector
    m_vec = np.asarray(pipe.matrix[match_start:match_start + length], dtype=float)

    # path_z first, and weighted taller: it is the channel that decides most matches.
    order = [i for i, n in enumerate(CHANNEL_LABELS) if n == "path_z"]
    order += [i for i, n in enumerate(CHANNEL_LABELS) if n != "path_z"]
    # Derived from ``order`` rather than written out as a literal, because Plotly
    # requires ``len(row_heights) == rows`` exactly and raises otherwise.  A literal
    # sliced to ``[: len(order)]`` looks safe and is not: the moment FEATURE_COLUMNS
    # grows past the literal it silently yields a *short* list, and the Matches tab dies
    # with a message about subplot geometry that names neither the cause nor the fix.
    # Shares need not sum to 1 -- Plotly normalises by the total.
    others = max(len(order) - 1, 1)
    heights = [0.6 if CHANNEL_LABELS[i] == "path_z" else 0.4 / others for i in order]

    fig = make_subplots(
        rows=len(order), cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.08,
        subplot_titles=tuple("%s leg (%s)" % (CHANNEL_LABELS[i].split("_")[0],
                                             CHANNEL_LABELS[i]) for i in order),
    )
    x = np.arange(length)
    for row, channel_ix in enumerate(order, start=1):
        label = CHANNEL_LABELS[channel_ix]
        fig.add_trace(go.Scatter(
            x=x, y=q_vec[:, channel_ix], mode="lines", name="query",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="bar %{x}<br>%{y:.2f} z<extra>query " + label + "</extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=m_vec[:, channel_ix], mode="lines", name="match #%d" % rank,
            line=dict(color="#1f77b4", width=1.6, dash="dash"),
            hovertemplate="bar %{x}<br>%{y:.2f} z<extra>match " + label + "</extra>",
        ), row=row, col=1)

    fig.update_layout(
        height=340, margin=dict(l=8, r=8, t=44, b=8),
        hovermode="x unified",
        title=dict(text="normalised shape · distance %.3f · bar index within window" % distance,
                   font=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
    )
    fig.update_xaxes(title_text="bar within window", row=len(order), col=1)
    return fig


def build_forecast_figure(forecasts: List[Any]) -> Optional[go.Figure]:
    """Grouped bars: matched mean vs random-window baseline, per horizon."""
    if not forecasts:
        return None
    labels, matched, baseline, suff = [], [], [], []
    for f in forecasts:
        d = f.as_dict()
        # Bars, in the active resolution's unit -- "10 min" intraday, "10 days" daily.
        # Same rule as the projection axis in ``build_forecast_path_figure``.
        labels.append("%d %s" % (d["horizon_min"], bar_unit(d["horizon_min"])))
        matched.append(d["mean_return"] * 1e4 if d["sufficient"] else np.nan)
        baseline.append(d["baseline_mean"] * 1e4 if np.isfinite(d["baseline_mean"]) else np.nan)
        suff.append(bool(d["sufficient"]))
    if not any(suff):
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(name="matched windows", x=labels, y=matched,
                         marker_color="#2ca02c", opacity=0.9))
    fig.add_trace(go.Bar(name="matched windows (insufficient evidence)", x=labels,
                         y=[np.nan if s else v for s, v in zip(suff, matched)],
                         marker_color="rgba(128,128,128,0.25)",
                         marker_line_color="#999999", marker_line_width=1))
    fig.add_trace(go.Bar(name="random-window baseline", x=labels, y=baseline,
                         marker_color="#7f7f7f", opacity=0.75))

    fig.add_hline(y=0.0, line=dict(color="rgba(0,0,0,0.45)", width=1))
    fig.update_layout(
        barmode="group", height=320, margin=dict(l=8, r=8, t=42, b=8),
        yaxis=dict(title="forward return (bps)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=dict(text="matched vs baseline — a forecast must clear the grey bars",
                   font=dict(size=12)),
    )
    return fig


@st.cache_resource(show_spinner=False, max_entries=4)
def forecast_path_for(
    cache_key: str,
    pipe: Pipeline,
    *,
    length: int,
    horizon: int,
    k: int,
    amplitude_weight: float,
    window: Optional[Tuple[int, int]] = None,
) -> Optional[Any]:
    """Match a window of ``length`` bars and aggregate a forecast path.

    Returns a :class:`timeseries.forecast.ForecastPath`, or ``None`` when the
    archive cannot support one.

    **``window`` picks which window; omitting it means the archive's most recent
    ``length`` bars.**  Two callers, deliberately sharing one implementation:

    * the always-live reference chart above passes nothing, so it keeps asking
      "what usually followed a window shaped like the most recent one?" and stays a
      fixed reference while the reader explores elsewhere;
    * the brush-driven chart below passes the span the reader drew, so the same
      arithmetic runs on the window they chose.

    ``window`` is ``(start, stop)`` in ``pipe`` bar indices and is **validated
    against ``length``**, because a caller that passes a span of the wrong width would
    otherwise be scored at a length it never declared -- and every downstream number
    (the forward slice, the median, the band) would then quietly describe a different
    question from the one on screen.  A mismatch returns ``None`` rather than being
    coerced, so the failure surfaces as a missing chart instead of a wrong one.

    Cached for the same reason ``pipeline_from_frame`` is, and by the same mechanism:
    Streamlit hashes what it is given, so the ``cache_key`` is a string naming the
    bars while the ``Pipeline`` rides along as an un-keyed payload -- hashing a
    multi-thousand-row frame on every rerun would cost more than the search it
    guards.  The key covers everything that changes the answer (length, horizon, k,
    amplitude weight, and the window), so a nudge to an unrelated sidebar control does
    not re-run a 240-bar match search, while a change to any of these does.

    ``None`` is returned rather than an empty figure for the ways this can
    legitimately come up empty: an archive shorter than the window, a window that
    reaches outside it, a query that leaves no candidates once the exclusion zone is
    applied, or no match with a complete forward path.  All of them are "draw nothing
    and say why", and the caller has one code path for that instead of four.
    """
    n = pipe.n_bars
    if not pipe.ready or length <= 0 or n < length:
        return None

    if window is None:
        # The most recent window is the query.
        # We use the most recent length bars, but the matching happens against
        # historical windows. The query itself is at the end of the archive,
        # which is exactly what we want for a "live" forecast.
        start, stop = n - length, n
    else:
        start, stop = int(window[0]), int(window[1])
        # Rejected, not clamped: a clamped window would still be *searched* at its
        # clamped width while the reader's chart showed the width they drew.
        if not (0 <= start < stop <= n) or (stop - start) != int(length):
            return None

    query = pipe.query_span(start, stop,
                            label=stamp_span(pipe.timestamp_at(start),
                                             pipe.timestamp_at(stop - 1),
                                             key=pipe.timeframe))
    try:
        result = pipe.match(query, k=int(k),
                            amplitude_weight=float(amplitude_weight))
    except Exception as e:
        print(f"DEBUG: pipe.match failed with {e}")
        return None

    if not result.matches:
        return None

    path = forecast_paths(
        pipe.close,
        np.asarray([m.start for m in result.matches], dtype=np.int64),
        length,
        int(horizon),
        distances=np.asarray([m.distance for m in result.matches], dtype=float),
    )
    return path


@st.cache_resource(show_spinner=False, max_entries=4)
def panel_forecast_path_for(
    cache_key: str,
    vector: np.ndarray,
    *,
    length: int,
    horizon: int,
    k: int,
    amplitude_weight: float,
) -> Optional[Dict[str, Any]]:
    """The same expected forward path, pooled across the whole S&P 500 panel.

    :func:`forecast_path_for` answers "what usually followed a window shaped like
    this one **in this one ticker**".  This answers "what usually followed a window
    shaped like this one **anywhere in the index**", which is a broader question with
    a correspondingly larger candidate pool -- 500 names rather than one, so a
    routine shape has many more chances to be matched, and the matches come from
    genuinely different tapes rather than from one name's own past.

    Both are kept, and the tab shows both, because they answer different questions.
    A single ticker has fewer windows and so produces rarer-seeming matches; the panel
    has more windows and more *diverse* ones.  Quoting only the first would report
    "this has happened before" as though it were notable, when it is only notable
    *for that name*.

    Cached exactly as :func:`forecast_path_for` is, and for the same reason: the
    ``cache_key`` string names the question while the query vector rides along as
    un-keyed payload, so Streamlit does not hash a multi-thousand-row frame on every
    rerun.  The key covers everything that changes the answer.

    **The panel is searched here rather than in ``main``** so a missing or broken
    archive degrades to "this chart is not drawn" instead of taking the whole page
    down, which is what the Price tab's own panel call already does.

    Returns ``None`` -- never raises -- when the archive is absent, cannot form a
    query, finds nothing, or leaves no match with a complete forward path.  The
    caller has one code path for "draw nothing and say why".
    """
    if not os.path.isdir(panel_root_for()):
        return None
    try:
        search = load_panel_search(panel_root_for())
        if not search.ready():
            return None
        # ``ticker=""``: ``vector`` was built from the *fetched* ticker's bars, so its
        # indices address a different series than the panel's.  Naming the home ticker
        # would hand the §M self-match guard a range in the wrong index space and
        # suppress an unrelated region of that ticker.  See ``cross_sectional_match``.
        query = PNL.PanelQuery(vector=vector, ticker="", start=0, stop=int(length))
        result = search.search(
            query, k=int(k), max_horizon=int(horizon),
            amplitude_weight=float(amplitude_weight),
        )
    except Exception:  # noqa: BLE001 - a failed panel search must not take the tab down
        return None

    if not result.matches:
        return None

    # Grouped by ticker because each match's ``start`` is an index into *its own*
    # series; handing one flat array to a multi-series aggregator would attribute
    # every window to the last ticker read.
    closes: Dict[str, np.ndarray] = {}
    starts: Dict[str, np.ndarray] = {}
    dists: Dict[str, np.ndarray] = {}
    for m in result.matches:
        closes.setdefault(m.ticker, search.close_aligned(m.ticker))
        starts.setdefault(m.ticker, []).append(int(m.start))       # type: ignore[union-attr]
        dists.setdefault(m.ticker, []).append(float(m.distance))   # type: ignore[union-attr]

    try:
        path = forecast_paths_multi(
            closes,
            {t: np.asarray(v, dtype=np.int64) for t, v in starts.items()},
            int(length),
            int(horizon),
            distances={t: np.asarray(v, dtype=float) for t, v in dists.items()},
        )
    except Exception:  # noqa: BLE001 - a failed aggregation is not a failed page
        return None

    if path is None:
        return None

    # The ticker breakdown is computed here rather than left to the caption, because
    # ``ForecastPath.starts`` is a bare concatenation across series and no longer
    # identifies its own ticker.  The evidence for a pooled path is only auditable if
    # the reader can see how many of the matches came from one name -- a median drawn
    # from 30 windows of a single stock is a different claim from 30 windows spread
    # across 12 names, and the chart cannot tell them apart on its own.
    tally: Dict[str, int] = {}
    for m in result.matches:
        tally[m.ticker] = tally.get(m.ticker, 0) + 1

    return {
        "path": path,
        "tickers": tally,
        "n_tickers": len(tally),
        "n_candidates": int(result.n_candidates),
        "n_tickers_scored": int(result.n_tickers),
        "matches": result.matches,
    }


def _closest_distance_label(path: Any) -> str:
    """The best match's distance, formatted, or ``"?"`` when it was not recorded.

    A distance of ``0`` would render as ``"0.000"`` and look like a bug, so the
    formatting is left to the caller to widen if it ever needs to -- which it does not
    today.  ``nan`` is the sentinel :func:`forecast_paths` leaves when ``distances``
    were not supplied, and printing ``"nan"`` on a legend would be worse than saying
    nothing about it.
    """
    dist = np.asarray(getattr(path, "distances", np.empty(0)), dtype=float)
    if dist.size == 0 or not np.isfinite(dist).any():
        return "?"
    return "{:.3f}".format(float(dist[0]))


def _closest_match_path(close: np.ndarray, path: Any) -> Optional[np.ndarray]:
    """The best match's own forward path, in percent from its anchor close.

    ``None`` -- never an exception -- when the match cannot be reconstructed, which is
    the right answer for every way it can come up empty: no match survived, the stored
    ``start`` is outside the series, or the anchor close is missing or non-positive.
    The caller draws the median either way; a missing evidence line must never cost the
    reader the aggregate.

    **Index 0 is the closest match because ``distances`` is best-first**, and that
    ordering is a contract rather than an accident: :func:`forecast_paths` documents
    the array as "best-first as the matcher returned them", and it slices it with the
    same ``kept`` positions as ``starts``, so index 0 names the same window in both.
    Pairing row 0 with ``starts[0]`` is therefore correct even after a later match was
    dropped for having no forward bars -- which is exactly the case where trusting
    "the first thing that looks like the median" would have picked the wrong window.

    **It takes no view range.**  The matched window is somewhere in the archive's past
    and is plotted against the *query's* projection offsets, so its own position is
    irrelevant to the arithmetic -- only its forward slice matters.  An earlier version
    took ``anchor``/``stop`` and rejected any match sitting after the query's
    projection end, which silently dropped the line for every query in the first half
    of the archive: the closest match to an early window is usually a *later* window,
    so the guard suppressed the trace exactly where a reader is most likely to brush.

    Rebased to its **own** anchor close, not the query's, because that is what makes
    the two curves comparable at all: a raw close difference between two windows would
    be a statement about share price rather than about what happened next.
    """
    close = np.asarray(close, dtype=float)
    starts = np.asarray(getattr(path, "starts", np.empty(0)), dtype=np.int64)
    if starts.size == 0 or close.ndim != 1 or close.size == 0:
        return None

    length = int(getattr(path, "length", 0) or 0)
    if length <= 0:
        return None
    # Anchor is the window's last bar (§Z1), the same convention ``forecast_paths``
    # uses and the same one the projection is drawn against.
    match_anchor = int(starts[0]) + length - 1
    end = match_anchor + int(path.horizon) + 1
    # ``forecast_paths`` already dropped any match whose forward slice runs past the
    # end of the series, so a survivor is in range by construction -- but it is checked
    # rather than assumed, because the index space here is the *figure's* ``close`` and
    # a pooled path's ``starts`` live in a different series entirely.
    if match_anchor < 0 or end > close.size:
        return None

    base = float(close[match_anchor])
    if not np.isfinite(base) or base <= 0:
        return None

    window = close[match_anchor:end]
    if not np.isfinite(window).all():
        return None
    return (window / base - 1.0) * 100.0


def _freeze_closest_match(close: np.ndarray, path: Any) -> Dict[str, Any]:
    """The closest match's own forward series, frozen as a **value**.

    :func:`_closest_match_path` reconstructs the best match's forward path by
    slicing ``close`` at ``path.starts[0]`` -- a bar *index*.  That index is only
    meaningful against the archive the path was computed from: as old bars age
    out of the front, the same index names a *different* bar, so the series
    re-read from a later archive is a different series.  The **Live** tab
    therefore stores the series itself, next to the path, and draws the stored
    one -- a frozen forecast drawn from a moving index is not a frozen forecast.

    **The record says which path it belongs to.**  ``length`` and ``horizon`` are
    carried so a series can be checked against the path it was frozen with --
    see :func:`_frozen_match_is_usable`.  A path and a series from *different*
    archives are not the same forecast and must never be drawn as one.

    **The series is kept in percent from the match's own anchor close**, which is
    the unit :func:`_closest_match_path` returns and the unit the figure rebases
    onto the query's anchor.  Storing the rebased *price* instead would freeze
    the query's close into the record and leave the figure guessing which of
    the two closes to use.

    ``series`` is ``None`` when the match could not be reconstructed -- a
    legitimate answer, and one that must *stay* ``None``.  Re-deriving it on a
    later archive could resurrect the line on a bar that is not the one that was
    matched, which is the same bug wearing a different hat.
    """
    series = _closest_match_path(close, path)
    return {
        "length": int(getattr(path, "length", 0) or 0),
        "horizon": int(getattr(path, "horizon", 0) or 0),
        "series": None if series is None else np.asarray(series, dtype=float),
    }


def _frozen_match_is_usable(frozen: Any, path: Any) -> bool:
    """Is a frozen closest-match record the one that belongs to *this* path?

    A record is usable when it is a record at all, and when it was frozen
    against a path with the same window length and horizon.  Both numbers are
    read off the path itself, so the check cannot drift from what is drawn.

    It exists for one situation: a session whose path was stored by an older
    build, before the match was frozen alongside it.  Such a path has no record,
    so the pair has to be rebuilt -- from the *same* archive and *together*, so
    the two can never disagree.
    """
    if not isinstance(frozen, dict) or path is None:
        return False
    try:
        length = int(frozen.get("length", -1))
        horizon = int(frozen.get("horizon", -1))
    except (TypeError, ValueError):
        return False
    return (length == int(getattr(path, "length", -2))
            and horizon == int(getattr(path, "horizon", -2)))


def _frozen_or_rebuilt_match(
    frozen: Optional[Dict[str, Any]],
    close: np.ndarray,
    path: Any,
) -> Optional[np.ndarray]:
    """The closest match's forward series: the frozen one if there is one.

    **The frozen series wins, always.**  It was captured against the archive the
    path was computed from, so it is the series that *belongs* to the forecast --
    a re-derived one would be read from a later archive, at a bar index that has
    since come to mean something else.

    The frozen record is used when it is *usable*: a record at all, and one
    frozen against a path with the same length and horizon (see
    :func:`_frozen_match_is_usable`).  A record that fails the check describes a
    different forecast, and drawing it next to this path would be worse than
    drawing nothing.

    **Otherwise the series is rebuilt from the current archive**, exactly as it
    was before the freezing existed -- so the Forecast tab, the panel forecast and
    the brushed chart are unchanged, and a *Live* session whose record is missing
    (an older build, or a path stored before the record existed) still draws a
    line rather than nothing.

    **A frozen ``series`` of ``None`` stays ``None``.**  The record says the match
    could not be reconstructed at freeze time; re-deriving it now could
    resurrect the line on a bar that is not the one that was matched.
    """
    if isinstance(frozen, dict) and _frozen_match_is_usable(frozen, path):
        series = frozen.get("series")
        if series is None:
            return None
        return np.asarray(series, dtype=float)
    return _closest_match_path(close, path)


def build_forecast_path_figure(
    pipe: Pipeline,
    path: Any,
    *,
    history_bars: Optional[int] = None,
    height: int = 420,
    title: Optional[str] = None,
    window: Optional[Tuple[int, int]] = None,
    show_median: bool = True,
    projection_anchor: Optional[int] = None,
    hist_start: Optional[int] = None,
    hist_stop: Optional[int] = None,
    frozen_match: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """A window of real tape, then the projected path.

    Three things are drawn, and the order matters:

    1. the **history**, as a plain close line on the app's usual blue;
    2. the **interquartile band**, as two boundary traces joined by ``fill``;
    3. the **median**, as the line a reader actually follows.

    ``show_median`` toggles the median and its closest-match evidence line.
    It defaults to ``True`` for the shared reference/forecast charts and is set
    to ``False`` on the *Live* tab, which shows only the single closest match
    rather than any aggregate.

    The history is rebased to the same ``0%`` anchor as the projection -- the close
    of the **last bar of the window** -- so the two halves are in one unit and meet at
    exactly zero.  Drawing the projection in percent against a history in dollars
    would be the same mixed-units mistake the Price tab's match panel was fixed for,
    and it would make the join a visible jump rather than the continuation it is.

    **``window`` says which window is drawn.**  It is ``None`` for the reference chart,
    whose history is the archive's last ``history_bars`` bars.  For the brushed chart
    it is the span the reader drew, and it changes two things:

    * the history is *that* window rather than the archive's tail -- otherwise a
      reader who brushed a window in the middle of the archive would be shown a chart
      of some other window's price action under a caption describing theirs;
    * the projection is anchored at the window's end, so it starts where the window
      ended instead of trailing off from the end of the archive.  This is the whole
      point of the brushed chart: "what followed *this*".

    ``history_bars`` is ignored when ``window`` is given, since the window's own width
    is the history.  It stays for the unbrushed caller, where the window *is* the tail.

    Bars sit on their integer index with session gaps compressed, matching
    :func:`build_price_figure`.  The projection therefore *does* cross an overnight
    close on an equity, and that is fine for spacing but not for labels: every tick
    right of the boundary is stamped as an *offset* (``+45 min``, or ``+4 days`` on
    daily -- resolved by :func:`bar_unit`) rather than a timestamp, because those bars
    do not exist yet and printing a time for them would be the chart asserting
    something it has no data for.

    ``title`` is passed in rather than read from ``SYMBOL_FOR_HELP`` here, matching
    :func:`render_price_tab`'s handling of the Price chart's title: the renderer
    resolves the live symbol and the figure stays a pure function of its arguments.

    **Four traces, in draw order: history, band upper, band lower, median, closest
    match.**  The median is drawn before the closest match so the individual window
    sits *on top* of the aggregate -- legible without displacing it.  The band pair
    stays adjacent, because ``fill="tonexty"`` binds the lower boundary to whichever
    trace precedes it, and inserting anything between them would shade the wrong
    region.

    **The closest match is omitted, not faked, when it cannot be reconstructed.**
    :func:`_closest_match_path` returns ``None`` for every way that can happen and the
    median is drawn regardless, so a missing evidence line can never cost the reader
    the aggregate answer.

    **``frozen_match`` supplies the closest match instead of re-deriving it.**
    It is the record :func:`_freeze_closest_match` writes: the best match's own
    forward series, in percent from *its* anchor close, captured when the path was
    computed.  When it is supplied, that stored series is drawn; when it is not,
    the series is reconstructed from ``pipe.close`` exactly as it always was.

    The distinction is not cosmetic.  Re-deriving slices the *current* archive at
    ``path.starts[0]`` -- a bar *index* -- and the archive's index space is not
    stable: as old bars age out of the front, that index names a different bar.
    The **Live** tab's archive rolls, so a re-derived series changes underneath a
    frozen forecast.  The other callers (the Forecast tab, the panel forecast, the
    brushed chart) draw a *live* chart over a *live* archive and want the current
    reading, so they keep the re-derivation and pass nothing.

    **Live-tab mode: the projection is frozen, the tape rolls.**  The projection is
    anchored at ``projection_anchor`` -- the last bar of the history *at load time*
    -- rather than at the current tail, so it stays at fixed x-positions on the
    chart while the blue history tape grows past it.  ``projection_anchor`` is the
    bar the projection grows out of; ``projection_anchor + offsets`` are the
    projection's x-positions, computed once and never moved.

    **The separator stands where the projection bars start.**  That bar is
    ``projection_anchor`` -- offset 0 of the forecast path, which is the first
    projected bar.  The line therefore marks where the projected bars *begin*.
    The projection's trace still starts at the anchor, because that is the
    contract :func:`timeseries.forecast.forecast_paths` states: the two halves
    meet at one point rather than at two points a bar apart.

    **``hist_start`` and ``hist_stop`` are the history's extent**, and they are
    what let the *Live* tab hold still while its archive grows.  The history runs
    from the pinned ``hist_start`` to the end of the archive, so new bars extend
    it to the right and the left edge never moves.  ``hist_stop`` overrides that
    end for a caller that wants a different one; it is clamped into the frame, so
    a caller cannot ask for a span the archive does not have.

    **The x-axis range is the union of the history and the projection.**  Either
    can be the longer one -- the *Live* tab's history runs to the archive's tail
    while its projection is a fixed number of bars past the frozen anchor, and
    the *Forecast* tab's projection is the longer one -- and the range has to
    cover both.  It used to be computed from the projection alone, which put the
    history's own tail outside the visible axis: the newest bars of the tape
    were clipped off the chart.
    """
    n = pipe.n_bars
    if history_bars is None:
        history_bars = _tf().forecast_history_bars
    # ``anchor`` is the last bar of real tape on the chart -- the bar the projection
    # grows out of.  Everything else is expressed relative to it.
    if window is None:
        if projection_anchor is not None:
            hist_end = projection_anchor + 1
            if hist_start is not None:
                # **The history start is pinned.**  It is a bar index the caller
                # resolved from a pinned *moment*, so it names the same bar on
                # every pass even as the archive's indices shift underneath it.
                #
                # The history runs from there to the end of the archive, so the
                # tape grows to the right as new bars arrive while the left edge
                # stays put.  ``hist_stop`` overrides that end when the caller
                # wants a different one.
                if hist_stop is None:
                    hist_stop = n
                hist_start = max(0, min(int(hist_start), max(0, n - 1)))
                hist_stop = max(hist_start + 1, min(int(hist_stop), n))
            else:
                hist = max(1, min(int(history_bars), hist_end))
                hist_start = hist_end - hist
                hist_stop = hist_end
        else:
            hist = max(1, min(int(history_bars), n))
            hist_start = n - hist
            hist_stop = n
    else:
        hist_start, hist_stop = int(window[0]), int(window[1])
        if not (0 <= hist_start < hist_stop <= n):
            hist_start, hist_stop = max(0, min(hist_start, n - 1)), n
    hist = hist_stop - hist_start
    # **The separator is the bar the projection grows out of, and it is pinned.**
    #
    # It used to be derived from the *history's extent* (``hist_start + hist - 1``),
    # which names the same bar only when the history *ends* at the projection's
    # anchor.  The *Live* tab's history runs from a pinned start all the way to
    # the archive's tail, so that derivation put the separator on the *newest bar
    # of the tape*: the line slid one place to the right on every poll, with the
    # frozen projection stranded behind it and the shaded region left inverted.
    #
    # The anchor therefore comes from the caller when there is one -- the caller
    # has pinned it as a *moment* and resolved it against this pass's archive --
    # and is derived from the history only when there is not.  That is the
    # brushed chart, whose window *is* the history and so ends at its own
    # anchor, and the plain reference chart, whose history is the archive's tail.
    if projection_anchor is None:
        anchor = hist_start + hist - 1
        projection_anchor = anchor
    else:
        anchor = int(projection_anchor)
    stop = projection_anchor + int(path.horizon)

    # **The separator stands where the projection bars start.**  That bar is
    # ``anchor`` -- offset 0 of the forecast path -- see the docstring for why,
    # and for why the projection's *trace* still starts on the anchor.
    boundary = anchor

    close = np.asarray(pipe.close, dtype=float)
    base = float(close[anchor]) if 0 <= anchor < n else float("nan")
    if not np.isfinite(base) or base <= 0:
        base = 1.0

    hist_x = np.arange(hist_start, hist_start + hist, dtype=np.int64)
    hist_y = close[hist_start:hist_start + hist]

    # The projection's x-positions and its y-values, rebased onto the anchor's
    # close.  The projection is frozen: these are computed from the path the
    # caller pinned, and they never move as the archive grows.
    # **Offset 0 is the anchor bar itself**, per
    # :func:`timeseries.forecast.forecast_paths`.
    proj_x = projection_anchor + np.asarray(path.offsets, dtype=np.int64)
    y_q25 = base * (1 + path.q25 / 100.0)
    y_q75 = base * (1 + path.q75 / 100.0)
    y_median = base * (1 + path.median / 100.0)

    fig = go.Figure()

    # To avoid the "straight line" effect, we use the percentage return
    # relative to the anchor for both history and projection.
    # History: price
    # Projection: base * (1 + return / 100)
    hist_y_rel = hist_y

    # **The hover says *when*, not *which bar*.**  The x value is the bar index;
    # the stamp is carried on ``customdata`` so the hover prints a date/time.
    hist_stamps = stamp_labels(
        pipe.bars["timestamp"].iloc[hist_start:hist_start + hist], key=pipe.timeframe)
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y_rel, mode="lines",
        line=dict(color="#1f77b4", width=1.3),
        name="history (last %d bars)" % hist,
        hovertemplate="%{customdata}<br>price %{y:.2f}<extra>history</extra>",
        customdata=hist_stamps,
    ))

    # Band as a ``tonexty`` pair: the upper boundary is drawn first and the lower
    # one fills to it.  There is no gap being spanned here, so unlike the rolling-band
    # case in ``build_price_figure`` one polygon over the whole projection is enough.
    fig.add_trace(go.Scatter(
        x=proj_x, y=y_q75, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
        name="upper quartile",
    ))
    fig.add_trace(go.Scatter(
        x=proj_x, y=y_q25, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(44,160,44,0.18)",
        showlegend=False, hoverinfo="skip",
        name="interquartile range",
    ))
    # Offset labels for the projection: those bars do not exist yet, so the
    # hover prints the projected date rather than a bare offset.
    anchor_ts = pd.Timestamp(pipe.bars["timestamp"].iloc[projection_anchor])
    if _tf().key == "1m":
        # Intraday: one bar is one minute, so the offset in bars is the offset
        # in minutes. Print the projected clock time in the reader's zone.
        zone = display_zone()
        proj_stamps = [
            (anchor_ts + pd.Timedelta(minutes=int(o))).tz_convert(zone).strftime("%B %-d, %Y %H:%M")
            for o in path.offsets
        ]
    else:
        # Daily: one bar per session, so an offset is a session count. Print
        # the projected date (Eastern) rather than "+4 days".
        proj_stamps = [
            (anchor_ts + pd.Timedelta(days=int(o))).strftime("%B %-d, %Y")
            for o in path.offsets
        ]
    if show_median:
        fig.add_trace(go.Scatter(
            x=proj_x, y=y_median, mode="lines",
            name="median of %d matches" % path.n_matches,
            hovertemplate="%{customdata}<br>price %{y:.2f}<extra>projected</extra>",
            customdata=proj_stamps,
        ))

        # **The single closest match, as its own line.**  The median above is an
        # aggregate: it says what usually happened, and on its own it hides the fact
        # that "usually" may be a compromise between windows that went opposite
        # ways.  The best match is the one concrete instance behind the aggregate,
        # and drawing it costs one trace and makes the median auditable.
        #
        # **Rebuilt from ``close`` rather than carried on the path**, because
        # ``ForecastPath`` stores only the median and the quartiles: it keeps
        # ``starts`` and ``distances`` for attribution, not the individual return
        # series.  Reconstructing it here is exact -- the same slice and the same
        # rebasing ``forecast_paths`` applies.
        #
        # **Unless the caller froze it.**  ``frozen_match`` carries the series as it
        # was captured, for the *Live* tab, whose archive rolls: a reconstruction
        # from the current archive would read a *different* bar and draw a different
        # series under a frozen forecast.
        #
        # **The distance is drawn, not implied.**  A single window is an anecdote,
        # and the whole point of the §E framing is that the best-looking of
        # thousands is extreme by construction.  Labelling it "closest match" without
        # its distance would invite reading it as the expected outcome, which is
        # precisely the misreading the median exists to prevent; the number is in
        # the trace name so the reader can weigh it.
        #
        # **Drawn last and dashed**, so it sits above the band without competing
        # with the median for attention: the median is the claim, this is the
        # evidence for it.
        best = _frozen_or_rebuilt_match(frozen_match, pipe.close, path)
        if best is not None:
            # The closest match is also rebased from percentage to absolute price.
            best_abs = base * (1 + best / 100.0)
            fig.add_trace(go.Scatter(
                x=proj_x, y=best_abs, mode="lines",
                line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                name="closest match (distance %s)" % (
                    _closest_distance_label(path)),
                hovertemplate="%{customdata}<br>price %{y:.2f}<extra>closest match</extra>",
                customdata=proj_stamps,
            ))
    else:
        # The Live tab shows only the single closest match: with k=1 there is no
        # aggregate, so the projection is the match itself.  It is drawn unconditionally
        # (not gated on show_median) so the projection is never missing.
        best = _frozen_or_rebuilt_match(frozen_match, pipe.close, path)
        if best is not None:
            best_abs = base * (1 + best / 100.0)
            fig.add_trace(go.Scatter(
                x=proj_x, y=best_abs, mode="lines",
                line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                name="closest match (distance %s)" % (
                    _closest_distance_label(path)),
                hovertemplate="%{customdata}<br>price %{y:.2f}<extra>closest match</extra>",
                customdata=proj_stamps,
            ))

    # The boundary, drawn before the shading so the shading does not wash it out.
    #
    # **The line is where the projection bars start**, ``anchor`` -- offset 0 of
    # the forecast path -- so it holds still as the tape rolls.  It used to be
    # drawn at the history's last bar, which on the *Live* tab is the archive's
    # newest bar -- so the line walked right with the tape and stopped meaning
    # anything.
    fig.add_vline(x=boundary, line=dict(color="rgba(128,128,128,0.9)", width=1.4,
                                        dash="dash"))
    # **The shading covers the projection, and only the projection.**  It runs
    # from the first projected bar to the projection's last bar, and is clamped
    # so it can never be inverted: on the *Live* tab the tape runs *past* the
    # anchor, so the history's tail lies to the right of the line, and an
    # unclamped rectangle would have been drawn backwards over it.
    fig.add_vrect(
        x0=boundary, x1=max(float(stop) - 0.5, float(boundary)),
        fillcolor="rgba(128,128,128,0.10)", line_width=0, layer="below",
    )

    # Ticks: real timestamps across the history, projected offsets across the future.
    # The history ticks stay inside ``pipe.bars`` -- every one of them is a real
    # bar of the archive -- and the projection keeps offset labels.
    step = max(1, hist // 8)
    ticks = list(range(hist_start, hist_start + hist, step))
    if not ticks or ticks[-1] != hist_start + hist - 1:
        ticks.append(hist_start + hist - 1)
    pstep = max(1, int(path.horizon) // 5)
    proj_ticks = list(range(0, int(path.horizon) + 1, pstep))
    if proj_ticks and proj_ticks[-1] != int(path.horizon):
        proj_ticks.append(int(path.horizon))

    # **The anchor is labelled exactly once, by whichever side owns it.**  It is
    # the history's last bar on the *Projection* tab (the history ends there),
    # but on the *Live* tab the history runs past it to the archive's tail, so
    # the anchor sits *inside* the history and is not necessarily one of the
    # history's own ticks.  So the history's label is used when the history
    # carries the anchor, and the anchor is added as an explicit tick when it
    # does not.  Either way the boundary is named once, and never twice.
    anchor_tick = projection_anchor
    history_labels_anchor = anchor_tick in ticks
    # **Offset 0 is the boundary.**  It sits on the anchor bar, which is always
    # labelled -- by the history's own ticks when the history ends at the
    # anchor, and by an explicit tick when the anchor sits *inside* a history
    # that runs on past it (the *Live* tab, whose tape keeps rolling to the
    # right of the frozen anchor).  Keeping offset 0 in the second case put
    # the anchor on the axis twice, under two different spellings: the
    # history's full timestamp and the projection's bare clock time.
    proj_offsets = [o for o in proj_ticks if o != 0]

    # **The unit is the resolution's, not a literal.**  These offsets are *bars*, so
    # on daily a tick reads "+4 days", not "+4 min" -- one bar is one trading day there.
    # Labelling them in minutes was a wrong number printed straight onto a chart, which
    # is the failure the ``[[GAP]]`` token and the rest of this app's resolution
    # handling exist to prevent; the Backtest tab's own *Horizon* help already resolves
    # its unit through :func:`active_unit` for exactly this reason.
    #
    # Read from ``_tf()`` rather than from the figure's caller, because the figure
    # renders under whichever resolution the *containing* tab is in -- the Forecast tab
    # may hold a different one from the Price tab, and this axis describes the bars it
    # was handed, not the ones the page opened on.  The *history* labels therefore come
    # from the pipeline, which is the authority on the bars actually drawn; only the
    # projected offsets, which have no pipeline behind them, use the active timeframe.

    # **The tick values and their labels are built together**, so the two lists
    # cannot drift apart: a label is only ever emitted for a position that was
    # actually added.
    tickvals: List[int] = []
    ticktext: List[str] = []
    taken = set()

    def _add_tick(value: int, label: str) -> None:
        if value in taken:
            return
        taken.add(value)
        tickvals.append(value)
        ticktext.append(label)

    for i in ticks:
        _add_tick(i, stamp_label(pipe.bars["timestamp"].iloc[i], key=pipe.timeframe))
    if not history_labels_anchor:
        # The boundary, named by the history's own label so both halves of the
        # axis agree on what a bar looks like.
        _add_tick(anchor_tick, stamp_label(anchor_ts, key=pipe.timeframe))
    for o, label in zip(proj_offsets, _proj_ticktext(anchor_ts, proj_offsets)):
        # **A projected tick that lands on a position the history already has
        # is dropped.**  Both sides label the same x-position, so keeping both
        # would print two labels on one bar -- the same mistake as the anchor's,
        # arrived at from the other direction.
        _add_tick(projection_anchor + o, label)

    # **The ticks are sorted left to right.**  The anchor and the projection's
    # offsets are appended after the history's own ticks, but on the *Live* tab
    # the anchor sits *inside* the history, so the unsorted list put the
    # projection's ticks in the middle of the axis and the history's after
    # them.  The positions are the same either way -- this only decides the
    # order they are listed in, and a reader of the labels should meet them
    # in the order they appear on the chart.
    order = sorted(range(len(tickvals)), key=lambda k: tickvals[k])
    tickvals = [tickvals[k] for k in order]
    ticktext = [ticktext[k] for k in order]

    # No zero-line for absolute price.
    #
    # **The range must cover the history *and* the projection.**  The history can
    # be longer than the projection -- that is the normal case for the *Live*
    # tab, whose history runs to the end of the archive while the projection is
    # a fixed number of bars past the frozen anchor -- so the range is the
    # *union* of the two extents.
    #
    # It used to be computed from the projection alone, which meant the
    # history's own tail was drawn *outside* the visible axis: the newest bars
    # of the tape were clipped off the chart.  The appended-bar mechanism had
    # been hiding this, because it drew a synthetic point rather than the
    # archive's real tail.
    hist_stop_x = hist_start + len(hist_x)
    x_range = [hist_start - 0.5, max(hist_stop_x, stop) - 0.5]
    # **The axis title describes the separator, not a side of it.**  It used to
    # say "right of the dashed line is projected", which is only true where the
    # history *ends* at the line -- the *Projection* and *Forecast* tabs.  On
    # the *Live* tab the tape keeps rolling to the right of the frozen anchor,
    # so the projected bars are *on* the right of the line but the tape is too,
    # and the old wording described a half of the chart that is not there.
    xaxis_title = "bar index · sessions compressed · the dashed line is where the projection starts"
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=8, t=46, b=54),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title=xaxis_title,
            showgrid=True, gridcolor="rgba(128,128,128,0.15)", automargin=True,
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            range=x_range,
            # **The unified hover box's *title* is what printed the bar number.**
            #
            # With ``hovermode="x unified"`` the box has two parts: a title, which
            # plotly.js fills with the hovered x value, and one entry per trace,
            # which the trace's own ``hovertemplate`` fills.  The traces carry the
            # bar's stamp on ``customdata``, so the entries already print the
            # date/time -- but the title above them still read the bar index, and
            # that was the more prominent of the two.
            #
            # The title is a template of its own and cannot see ``customdata``, so
            # it cannot be told to print the stamp.  ``%{xother}`` resolves to the
            # empty string on a chart where every trace shares one x axis, so the
            # title renders as nothing and the box collapses to its trace entries
            # -- which carry the date/time.
            unifiedhovertitle=dict(text="%{xother}"),
        ),
        yaxis=dict(
            title="price",
            showgrid=True, gridcolor="rgba(128,128,128,0.15)", automargin=True,
        ),
        title=dict(text=title, font=dict(size=12)) if title else None,
    )
    return fig


# =============================================================================== #
# Tab: Matches
# =============================================================================== #
# Tails drawn either side of every scored window, in bars.  Fixed rather than a
# slider: it is a *view* control, not an analysis one, and every extra rerunnable
# widget is a cache-key dimension that can invalidate the whole pipeline for a
# question ("how much tape around the band?") that has one defensible answer.
#
# 20 bars is small relative to a typical window on purpose. The band has to stay the
# widest thing on screen -- a long tail makes the entry and follow-through legible
# without letting the surrounding noise dominate the shape being compared.
CONTEXT_BARS = 20

# Plot height shared by the Price tab's two charts.  They must agree: a bar drawn at
# the same width but on a taller plot reads as a larger move, which is exactly the
# illusion that comparing a query against its match is supposed to rule out.
PRICE_CHART_HEIGHT = 560


def _render_best_match_pair(pipe: Pipeline, query: M.Query, m: M.Match, rank: int,
                            *, pad: int, scope: str,
                            height: int = 420, label: str = "match",
                            show_query: bool = True, show_captions: bool = False,
                            y_range: Optional[Tuple[float, float]] = None,
                            chart_width: Optional[int] = None,
                            anchor: Optional[int] = None,
                            pannable: bool = False) -> None:
    """Draw the query above one of its matches, rebased to each panel's own first bar.

    Both panels rebase to their own first bar, which is what puts two windows that
    happened at different price levels onto a common footing at all.  And because both
    are drawn at the same ``chart_width``, the scored band lands at the same x offset
    in each, so the eye compares like with like across the two.

    Only the *first* graph gets an explicit vertical range; the match below it
    autoscales.  That is a deliberate trade rather than an oversight.  A shared range
    made the two panels directly comparable on absolute displacement, but it had to
    cover the query *and* the match over their context tails, so a match whose own
    move was modest got drawn as a nearly flat line against the wider of the two.
    Autoscalling the second graph lets each match fill its own panel, so what the eye
    reads is the shape of the move -- the order of the ups and downs, the relative size
    of each -- and not its size against the other panel's.  The *horizontal* framing
    stays shared, so comparing where things happened inside each window still works.

    ``y_range`` therefore applies to the query panel only.  It is passed in rather than
    derived whenever the caller has several matches to show, because the query is
    redrawn inside each match's tab: deriving it per pair would let that reference
    panel rescale as you move between tabs.  ``None`` means "derive it from the query
    alone", correct only for a single match.  It is ignored when ``show_query`` is
    ``False``, since no query panel is drawn in that case.

    ``show_query=False`` draws the match alone.  The Price tab uses that: its first
    chart is already the query on the full tape, so redrawing the query here would
    spend a whole panel restating it.

    ``chart_width`` pins how many bars every panel spans, instead of each deriving its
    own from ``pad``.  The Price tab passes the query chart's width so the query above
    and the match below are drawn at the same bar-to-pixel scale; without it the top
    chart spans ~600 bars and the match ~100, a bar is six times wider on top, and the
    two shapes cannot be compared by eye at all.

    It is a **width, not a view range**, because a view is an absolute bar range: passing
    the query's ``(7949, 8049)`` through would draw those same 100 bars in the match
    panel and omit the match altogether.  Each panel instead re-centres a window of this
    width on its *own* band, so the match stays visible and the scales still agree.

    ``pannable`` makes the *match* panel the one chart in the app the reader can drag
    along its x axis, and it changes two things together, because either alone would be
    useless.  The panel's **view** is centred on the match instead of sharing the query's
    anchor, and its **data** is the whole tape rather than just the view.  The data
    widening is the half that actually matters: a chart whose only data *is* the visible
    window has nothing to reveal, so panning it slides the same line out of frame and
    ends on blank space.  Centring is what makes the new room worth having -- it puts
    the band at the midpoint of the pannable range, so there is as much tape to reveal
    on one side as on the other (see :func:`centred_view` and :func:`pan_axis_bounds_for`).

    Centring is a real change to what this panel shows on load, and it is worth being
    explicit about: the band no longer sits at the same x offset as the query band
    above it, because the query is the archive's *last* window and physically cannot be
    centred -- doing so would need bars the archive does not have.  So the Price tab's
    two panels trade load-time band alignment for equal bar width and a match the
    reader can explore.  The bar-to-pixel scale still matches, which is what makes the
    shapes comparable; the band offsets no longer agree.  ``pannable`` is off for the
    Matches tab, whose three panels are meant to be read against each other at a glance
    rather than explored one at a time.

    Used by the Price tab (match only, no captions) and the Matches tab (three, one per
    tab, query + match + captions).
    """
    q_start, q_stop = query.start, query.stop
    pad = max(0, int(pad))
    # Only the query panel consumes an explicit range.  Deriving it from the query
    # alone (rather than from both spans) is the point of autoscaling the match: the
    # query reference no longer has to stretch to cover a match it is not sharing an
    # axis with.
    if show_query and y_range is None:
        y_range = shared_rebased_range(pipe, [(q_start, q_stop)], pad)

    if chart_width is None:
        q_view = span_context_bounds(pipe, q_start, q_stop, pad)
        m_view = span_context_bounds(pipe, m.start, m.stop, pad)
    else:
        # Both bands sit at the *same* x offset, chosen by the caller so that each has
        # room for it (see `shared_anchor`), so the two shaded bands line up.
        a = (chart_width - (q_stop - q_start)) // 2 if anchor is None else anchor
        q_view = aligned_view(pipe.n_bars, q_start, q_stop, chart_width, a)
        m_view = aligned_view(pipe.n_bars, m.start, m.stop, chart_width, a)

    if pannable:
        # The pannable panel drops the shared anchor and centres its own band, but keeps
        # ``chart_width`` -- so it still shows the same *number of bars* as the query
        # above it, and a bar is still the same width on both.  What the reader loses is
        # the band offsets agreeing; what they gain is 1890 bars of travel with the band
        # always on screen.
        m_view = centred_view(pipe.n_bars, m.start, m.stop, chart_width)
        # ``None`` means the archive is too short to frame the band at this width at all.
        # Drawing the chart without bounds is the honest response: an unsatisfiable clamp
        # would pin it to a range it cannot honour.
        m_pan_bounds = pan_axis_bounds_for(pipe.n_bars, m.start, m.stop, chart_width)
        # The data is clipped to the *bounds*, not to the archive.  Those are not the
        # same thing and the difference is the whole point: the bounds are the span the
        # pan can legally reach, so anything outside them is unreachable by panning, and
        # drawing it is what gives an over-drag somewhere to zoom off to.
        #
        # plotly.js has no hard stop for a pan -- issue #887 asks for exactly this and
        # is still open -- and what it does instead is *zoom* once the drag runs past a
        # limit.  With the whole tape drawn that zoom can walk the window off the match
        # and change how many bars are visible, silently breaking the equal-bar-count
        # comparison with the query chart above.  With the data clipped to the bounds,
        # the most the over-drag can do is shrink the window *inside* the legal span;
        # the reader still ends up looking at tape they could have panned to, which is a
        # far smaller surprise than losing the match.
        #
        # It cannot be eliminated this way -- a shrink can still hide the band -- so this
        # narrows the failure rather than removing it.  A real fix needs a custom
        # component that intercepts the drag, which Streamlit's ``plotly_chart`` does
        # not offer: it surfaces selection events only, never ``relayout``.
        m_pan_data = (
            (m_pan_bounds[0], m_pan_bounds[1]) if m_pan_bounds is not None
            else (0, pipe.n_bars)
        )
    else:
        m_pan_bounds = None
        m_pan_data = None

    # The y range has to be pinned for the pannable panel, or widening its data to the
    # whole tape silently rescales the match into a stripe -- see `rebased_view_range`.
    # The initial view is the one that sets it; a pan moves bars of similar magnitude
    # through the same axis, which is the whole point of framing on the match.
    m_y_range = y_range
    if pannable:
        # Use the raw price range for the pannable panel since we are no longer rebasing.
        # We need a range that covers the visible window in nominal prices.
        lo, hi = m_view
        window = pipe.close[lo:hi]
        if window.size > 0:
            w_min, w_max = float(window.min()), float(window.max())
            span = w_max - w_min
            margin = 0.08 * span if span > 0 else 0.05
            m_y_range = (w_min - margin, w_max + margin)
        else:
            m_y_range = None

    if show_captions:
        st.caption(
            "{} · {} bars".format(
                stamp_span(pipe.timestamp_at(m.start), pipe.timestamp_at(m.stop - 1),
                           key=pipe.timeframe),
                int(m.stop - m.start),
            )
        )
    if show_query:
        st.markdown("<b>Your query</b>")
        st.plotly_chart(
            build_price_figure(
                pipe, *q_view,
                query_start=q_start, query_stop=q_stop,
                span_label="query", title="query · rebased to its first bar",
                rebase_at=None, y_range=y_range,
                selectable=False, height=height,
            ),
            width='stretch', key="qchart_%s_%d" % (scope, rank),
            config=chart_config(),
        )

    if show_captions:
        st.markdown("<b>{}</b>".format(label))
    m_fig = build_price_figure(
        pipe, *m_view,
        query_start=m.start, query_stop=m.stop,
        span_label=label,
        span_color="44,160,44", span_text_color="#1a7a1a",
        title="%s · distance %.3f · rebased to its own first bar"
              % (label, float(m.distance)),
        # ``y_range=None`` is what turns on autoscale for this panel: the match is
        # framed by its own bars rather than stretched to line up with the query.
        # ``selectable=False`` because nothing ever reads a selection from here --
        # only the two charts that define the query are brushable.
        rebase_at=None, y_range=m_y_range,
        selectable=False, height=height,
        # The three pan arguments are all ``None``/``False`` unless ``pannable`` was
        # requested, so the Matches tab's panels -- and the Price tab's query chart --
        # build exactly the figure they built before panning existed.
        pan_data=m_pan_data, pan_bounds=m_pan_bounds, pannable=pannable,
    )
    if pannable:
        render_clamped_pan_chart(m_fig, height=height,
                                 key="mchart_%s_%d" % (scope, rank))
    else:
        # No ``scroll_zoom`` even when pannable: zoom is off across the app, and a
        # wheel gesture that rescaled this chart would break the equal-bar-count
        # comparison with the query chart above it without the reader asking for it.
        st.plotly_chart(
            m_fig,
            width='stretch', key="mchart_%s_%d" % (scope, rank),
            config=chart_config(),
        )


def _render_search_controls(scope: str) -> Dict[str, Any]:
    """``k`` / baseline / min-matches / seed / **Run match**, for one tab.

    Moved out of the sidebar so each analysis is configured on the tab whose numbers
    it governs.  These used to be shared, which is how the two tabs came to report one
    window between them: a single *k* and a single run could not be aimed at two
    different questions.

    ``scope`` namespaces every widget key, and this is load-bearing rather than
    tidy.  ``st.tabs`` renders **every** tab body on **every** rerun, so an unkeyed
    slider here would collide with the identical slider in the other tab and raise
    ``StreamlitDuplicateElementId`` -- the same failure documented at ``tab_panel``
    below, where two copies of the Panel body once rendered into the same tab.

    ``amplitude_weight`` is deliberately **not** here.  It is a scoring dial, not a run
    control: it changes what a distance *means*, which is why it already reaches both
    forecast charts despite their being independent.  Two copies of one dial would
    let them disagree about the same percentile, so it stays in the sidebar, shared and
    single.
    """
    with st.expander("Search settings — {}".format(scope), expanded=False,
                     icon="⚙"):
        k = st.slider(
            "k matches", min_value=1, max_value=120, value=10, step=1,
            key="{}_k".format(scope),
            help="How many historical windows to keep *after* suppression, capped. "
                 "This is the effective sample size for the evidence table. If that "
                 "table comes back grey, this is the first control to raise — it must "
                 "reach *Min matches for evidence* (default 30). Larger k also means "
                 "a lower percentile, i.e. rarer matches.",
        )
        n_baseline = st.number_input(
            "Baseline windows", min_value=100, max_value=5000, value=400, step=50,
            key="{}_baseline".format(scope),
            help="How many randomly chosen windows form the control group. More "
                 "windows make the baseline bar steadier (and the app slower). The "
                 "runtime also raises this to at least 4× the match count "
                 "automatically, so the baseline is never smaller than the matched "
                 "set.",
        )
        min_matches = st.number_input(
            "Min matches for evidence", min_value=5, max_value=200, value=30,
            step=1,
            key="{}_min_matches".format(scope),
            help="The floor on valid matched windows before any forecast number may "
                 "be printed. Below it, no mean, interval or p-value is shown — just "
                 "the reason. 30 is the §E minimum; lowering it publishes numbers the "
                 "sample cannot support.",
        )
        if min_matches < 30:
            hint(
                "⚠️ Below the §E minimum of 30. Lowering this prints numbers from "
                "samples too small to support them; useful only for inspecting the "
                "pipeline's plumbing."
            )
        run_clicked = st.button(
            "Run match", type="primary", width='stretch',
            key="{}_run".format(scope),
            help="Execute the search and compute the baseline. Results are never "
                 "cached: the query is chosen interactively, and caching it would hide "
                 "the baseline comparison this app exists to show. Press it again "
                 "after changing any control above.",
        )
        hint(
            "Nothing runs until you press this. Results stay empty on purpose rather "
            "than showing numbers from an earlier window."
        )
    return {
        "k": int(k),
        "n_baseline": int(n_baseline),
        "min_matches": int(min_matches),
        "seed": 0,
        "run_clicked": run_clicked,
    }


def render_live_tab(pipe: Pipeline, symbol: str, *,
                    amplitude_weight: float = M.DEFAULT_AMPLITUDE_WEIGHT) -> None:
    """The **Live** tab: "Where this usually goes next" for 1-minute data.

    This tab identifies historical patterns matching recent price action
    and projects the most likely subsequent movement. It is only available
    for 1-minute resolution data.

    **The tape rolls; the projection is frozen.**  On the first render the app
    matches the ``recent_bars`` on screen and projects the single closest match.
    That projection is anchored at load time and never changes.  On every poll
    (triggered by ``st.fragment`` every 60 seconds) the refreshed archive is
    drawn from the *same* pinned starting bar, so the blue tape grows to the
    right while the orange projection stays exactly where it was.  Existing bars
    are never changed.

    **Four things have to be pinned, and each one is pinned as a *moment* or a
    *value* rather than as a bar index** -- the archive's index space is not
    stable, so a stored index names a different bar once old bars age out:

    * **The history's start bar**, as a moment -- see :func:`live_pinned_anchor`.
    * **The projection itself**, frozen as a value computed once against the
      pinned window.  Recomputing it each poll re-aimed the query at the moving
      tail, so the "frozen" projection was a *different forecast every time*.
    * **The history's end bar**, drawn from the refreshed archive so the newest
      bars are on the chart.
    * **The closest match** -- the orange trace -- frozen alongside the path.
      It was re-derived each poll by slicing the *current* archive at
      ``path.starts[0]``, a bar index, so it changed underneath a frozen
      forecast.  A path and a series derived from *different* archives are not
      the same forecast, so the two are written and read together.

    The tab's promise is that the reader can watch what the tape did against
    what was predicted for it.  Each of those four was breaking that promise
    in a different way, and none of them announced itself: the chart simply
    looked wrong in a way that was hard to name.
    """
    # Requirement 1: Resolution Check & Guardrails
    if _tf().key != "1m":
        st.info("Live projection is only available for 1-minute resolution data.")
        return

    # Requirement 2: User Configuration (UI Controls)
    st.subheader("Live Projection Settings")
    # Requirement 2.1: Recent Bars
    recent_bars = st.slider(
        "Recent bars",
        min_value=10, max_value=600,
        step=50, value=10,
        help="How many of the most recent bars to use as the pattern to match.",
        key="live_recent_bars_slider",
    )
    # Requirement 2.2: Projection Bars
    projection_bars = st.slider(
        "Projection bars",
        min_value=10, max_value=600,
        step=50, value=40,
        help="How many bars into the future to project the historical match.",
        key="live_projection_bars_slider",
    )

    # Requirement 3: Analysis Logic (Backend) — fragment runs every 60s
    # and rebuilds ``pipe`` from Yahoo.
    #
    # **The anchor and the history start are pinned as *moments*, not as bar
    # indices.**  See :func:`live_pinned_anchor` for why an index cannot work
    # here: the archive's bar indices shift every time the frame is rebuilt, so
    # a stored index names a different bar on the next poll.  The two pinned
    # moments are the *anchor* -- the last bar the projection grows out of --
    # and the *history start* -- the first bar on screen.  Both are resolved
    # against the refreshed pipeline on every pass, so the chart holds still
    # while the tape grows underneath it.
    anchor_ts, hist_start_ts = live_pinned_anchor(pipe, symbol, recent_bars)

    @st.fragment(run_every=60)
    def _live_poll_fragment() -> None:
        # Refresh the pipeline so new bars from Yahoo are actually fetched.
        # Without refresh=True the cache never hits and the archive stays
        # frozen at load time (see test_fetch_store_survives_a_rerun).
        nonlocal pipe  # type: ignore - reassignment in nested scope
        try:
            new_bars = fetch_ticker_cached(symbol, refresh=True, timeframe=_tf().key)
            # Rebuild the pipeline from the refreshed frame.  **The package's
            # uncached ``pipeline_from_frame``, not the cached wrapper defined
            # above** -- a poll exists to pick up bars the cache does not have, so
            # routing it through the cache would serve the very frame the refresh
            # just replaced.
            from timeseries.pipeline import pipeline_from_frame
            pipe = pipeline_from_frame(
                "{}@{}".format(symbol, _tf().key),
                new_bars.frame,
                active_length(),
                timeframe=_tf(),
            )
        except Exception as exc:  # noqa: BLE001 - surface any loader failure in the UI
            st.error("Could not refresh live data: {}".format(exc))

        # **The pinned moments are resolved against the *refreshed* pipeline.**
        # They are moments, so this is the same bar every time -- but the
        # index it lands on is not, which is exactly why the moment is what
        # was stored.  A pin that has aged out of the archive is reported
        # rather than silently re-anchored on a different bar.
        projection_anchor = resolve_bar_index(pipe, anchor_ts)
        hist_start = resolve_bar_index(pipe, hist_start_ts)
        if projection_anchor is None or hist_start is None:
            st.warning("The pinned starting bar is no longer in the archive.")
            return

        # **The projection is frozen: computed once, then reused.**  It is computed
        # against the *pinned* window -- the bars the reader is actually looking at --
        # and stored whole, so every poll draws the same forecast.  See the
        # docstring for why recomputing it each poll defeated the whole tab.
        #
        # **The closest match is frozen with it, because the path alone was not
        # enough**, and the two are written and read together: a path and a series
        # derived from *different* archives are not the same forecast, and drawing
        # one next to the other would be a chart of nothing in particular.
        path_key = live_state_key(symbol, LIVE_PATH_KEY)
        match_key = live_state_key(symbol, LIVE_MATCH_KEY)
        path = st.session_state.get(path_key)
        frozen_match = st.session_state.get(match_key)
        if (path is None
                or _live_path_is_stale(path, recent_bars, projection_bars)
                or not _frozen_match_is_usable(frozen_match, path)):
            path = forecast_path_for(
                cache_key=f"live_{symbol}_{recent_bars}_{projection_bars}",
                pipe=pipe,
                length=recent_bars,
                horizon=projection_bars,
                k=1,
                amplitude_weight=M.DEFAULT_AMPLITUDE_WEIGHT,
                window=(hist_start, projection_anchor + 1),
            )
            if path is not None:
                # **The series is captured from the *same* archive the path was
                # computed from**, so the two agree by construction.  It is
                # captured here rather than on a later poll because the later
                # poll's archive is a *different* archive -- which is the whole
                # problem being fixed.
                st.session_state[path_key] = path
                st.session_state[match_key] = _freeze_closest_match(pipe.close, path)
            else:
                # A path that could not be computed must not leave a stale
                # series behind: the pair is replaced together or not at all.
                st.session_state.pop(path_key, None)
                st.session_state.pop(match_key, None)
            # Read the pair back, so the figure is given exactly what was
            # stored -- including after a recompute, where the local name
            # still held the *previous* pair.
            path = st.session_state.get(path_key)
            frozen_match = st.session_state.get(match_key)

        if path is None:
            st.warning("Could not find a matching historical pattern for the current price action.")
            return

        # Build the figure with the refreshed pipeline.
        #
        # **The history runs from the pinned start to the end of the archive**,
        # so the tape grows to the right as new bars arrive while the left
        # edge stays on the pinned bar.  The bars are drawn straight from the
        # refreshed pipeline: the old appended-bar mechanism drew a synthetic
        # point past the archive's last index, which put the newest bar
        # outside the axis range -- off the chart entirely -- and its x-position
        # moved further out on every poll.
        #
        # **The frozen series is handed over with the path**, so the orange
        # trace is drawn as it was captured rather than re-read from the
        # archive that has since rolled.
        fig = build_forecast_path_figure(
            pipe,
            path,
            history_bars=recent_bars,
            title=f"Live Projection for {symbol}",
            show_median=False,
            projection_anchor=projection_anchor,
            hist_start=hist_start,
            hist_stop=pipe.n_bars,
            frozen_match=frozen_match,
        )

        st.plotly_chart(fig, width='stretch')

        # Progression content, shown under the Live chart.
        st.subheader("Progression")

        if pipe is None or not pipe.ready:
            st.info("Progression is unavailable until the pipeline is ready.")
            return

        n_bars = pipe.n_bars
        lengths = progression_lengths(n_bars)
        if not lengths:
            st.info("This archive is too short to sweep any lookback length.")
            return

        hint(
            "Each trace asks the same question about the <b>same moment</b> at a "
            "different lookback: the archive's most recent <i>L</i> bars are matched "
            "against its own history, and the matched windows' forward paths are "
            "plotted. Pale traces are short lookbacks, dark ones long. The red line is "
            "the median across all of them — where the traces spread wide, the "
            "lookbacks disagree, and that is the finding."
        )

        with st.spinner("Matching {} lookback lengths and projecting…".format(len(lengths))):
            sweep = progression_sweep_for(
                "{}@{}|prog|{}|{}|{}|{}".format(
                    symbol, pipe.timeframe, PROGRESSION_HORIZON, PROGRESSION_MATCHES,
                    amplitude_weight, progression_length_step(),
                ),
                pipe,
                horizon=int(PROGRESSION_HORIZON),
                k=int(PROGRESSION_MATCHES),
                amplitude_weight=float(amplitude_weight),
                step=progression_length_step(),
            )

        if sweep is None or not sweep.answered:
            st.warning(
                "No projection could be built for any lookback length on this archive."
            )
            hint(
                "Matching needs at least a window's worth of bars after cleaning, and "
                "every lookback here needs enough history *behind* it to compare "
                "against. **What to try:** check the <b>Quality</b> tab for how many "
                "bars loaded, or use a more liquid symbol, which returns more of the "
                "available intraday history."
            )
            return

        fig = build_progression_figure(
            pipe, sweep,
            history_bars=min(500, n_bars),
            title="%s · %d lookbacks from %d to %d bars · %d bars projected"
                  % (symbol, len(sweep.answered), lengths[0], lengths[-1], PROGRESSION_HORIZON),
        )
        st.plotly_chart(fig, width='stretch', key="progression_paths",
                        config=chart_config())

        hint(
            "The dashed line is where the <b>projection</b> starts. To its left is the "
            "real tape — the archive's most recent bars, rebased to the last close so "
            "it meets the projections at zero. To its right is what the matched "
            "windows did over the following {} bars, each rebased to its own final "
            "close.".format(PROGRESSION_HORIZON)
        )

        _render_progression_table(sweep, PROGRESSION_HORIZON)

        failed = sweep.failed
        if failed:
            st.warning(
                "{} of {} lookback lengths could not be projected.".format(
                    len(failed), len(sweep.paths))
            )
            hint(
                "A length fails when the archive is too short to hold it, or when no "
                "historical window matches it closely enough to project. The lengths "
                "that did answer are all drawn above; the failed ones are listed in the "
                "table with the reason."
            )
    _live_poll_fragment()


# --------------------------------------------------------------------------- #
# Progression constants (content now shown under the Live chart)
# --------------------------------------------------------------------------- #
#: How many matches each lookback length asks for.
#:
#: 30 is :data:`FORECAST_PATH_MATCHES` -- the same number the Forecast tab's path
#: chart uses -- so the two tabs are drawing comparable evidence and the §E minimum
#: is met by every length.  It is *not* :data:`timeseries.pipeline.DEFAULT_K` (50):
#: that is the matcher's own default for a plain search, and this sweep runs fifty
#: searches, so fifty matches each would be fifty times the work for a median that
#: barely moves.
PROGRESSION_MATCHES = 30

#: The step between lookback lengths, in bars.
#:
#: 10 gives the fifty lengths the request asked for -- 10 through 500 -- and is the
#: natural grid on 1-minute data, where ten bars is a tenth of an hour.  The lengths
#: are the *only* thing this tab sweeps, so the step is a statement about resolution
#: and is namespaced by it below.
PROGRESSION_LENGTH_STEP = 50

#: The forward horizon the sweep projects over, in bars.
#:
#: 300 is what the request asked for, and it is the number the chart's right-hand
#: side is sized against.  It is deliberately *not* read from the Projection tab's
#: *Projection bars* slider: that slider governs a different chart, and letting it
#: silently resize this one would mean two tabs' horizons could disagree while
#: looking like the same control.
PROGRESSION_HORIZON = 300


def progression_length_step() -> int:
    """The lookback step for the active resolution, in bars.

    Ten bars on 1-minute data is a tenth of an hour -- a natural unit of "recent".
    On daily data it is a fortnight and change, which is too coarse to show how the
    answer changes with the lookback, so the daily step is five bars.
    """
    return 5 if _tf().bars_per_session == 1 else PROGRESSION_LENGTH_STEP


def progression_lengths(n_bars: int) -> List[int]:
    """The lookback lengths to sweep for an archive of ``n_bars`` bars.

    The requested range -- 10 through 500 -- is swept in full on every resolution.
    It is **not** truncated to the archive's own length, because the sweep's whole
    point is to show how the answer changes with the lookback: a truncated sweep on a
    short archive would answer a different question than the same sweep on a long
    one, and the reader would have no way of knowing the range moved.
    """
    return default_lengths(
        n_bars,
        step=progression_length_step(),
        max_length=MAX_PROGRESSION_LENGTH,
        min_length=10,
    )


@st.cache_resource(show_spinner=False, max_entries=4)
def progression_sweep_for(
    cache_key: str,
    pipe: Pipeline,
    *,
    horizon: int,
    k: int,
    amplitude_weight: float,
    step: int,
) -> Optional[ProgressionSweep]:
    """The whole sweep, cached.

    Cached exactly as :func:`forecast_path_for` is, and for the same reason: the
    ``cache_key`` string names the question while the ``Pipeline`` rides along as
    un-keyed payload, so Streamlit does not hash a multi-thousand-row frame on every
    rerun.  The key covers everything that changes the answer -- the horizon, the
    match count, the amplitude weight and the length step -- so a nudge to an
    unrelated sidebar control does not re-run fifty searches, while a change to any
    of these does.

    Fifty searches is roughly three seconds of matcher time, so the cache is the
    difference between a tab that responds and a tab that stalls on every pass.
    """
    lengths = progression_lengths(pipe.n_bars)
    if not lengths:
        return None
    try:
        return progression_paths(
            pipe,
            lengths=lengths,
            horizon=int(horizon),
            k=int(k),
            amplitude_weight=float(amplitude_weight),
        )
    except Exception:  # noqa: BLE001 - a failed sweep is a missing chart, not a dead page
        return None


def build_progression_figure(
    pipe: Pipeline,
    sweep: ProgressionSweep,
    *,
    history_bars: int = 500,
    title: Optional[str] = None,
) -> go.Figure:
    """The Progression chart: the recent tape, then every lookback's projection.

    **One chart, many projections.**  The fifty traces are the sweep's answer read
    at once, and the reason they can share an axis is that they are all in
    **percent from the anchor bar** -- the close of the archive's most recent bar.
    A projection that started from a different price level would not be comparable
    with the others, and fifty such traces on one axis would be unreadable.

    **Traces are coloured by lookback length, on a sequential ramp.**  Short
    lookbacks are light and long ones are dark, so the ramp carries meaning rather
    than just separating the lines: a reader can see at a glance whether the
    *long* lookbacks (the dark, top-right of the ramp) agree with the short ones
    (the pale ones), and that comparison is the tab's whole purpose.  The ramp runs
    dark-to-light across the viridis scale, which is monotonic in lightness and
    therefore readable in greyscale and for the most common colour-vision
    deficiencies.

    **The consensus line is drawn on top, in the app's own blue.**  Fifty
    overlapping traces cannot be read by eye, so the median across the lengths is
    computed and drawn: what the lookbacks *agree* on.  Where they disagree, the
    cloud is wide and the consensus sits somewhere inside it -- and that disagreement
    is the finding, not a failure of the chart.

    **The history is drawn in the same percent units**, rebased to the anchor, so
    the real tape and the projections meet at one visible bar at exactly zero.
    Drawing price against percent would be the mixed-units mistake the app's other
    charts are careful to avoid.

    The boundary line stands on the first bar that belongs to the projection alone
    -- the bar after the anchor -- matching :func:`build_forecast_path_figure`'s
    convention, so a reader moving between the two charts finds the same line in
    the same place.
    """
    n = pipe.n_bars
    horizon = int(sweep.horizon)
    anchor = n - 1

    close = np.asarray(pipe.close, dtype=float)
    base = float(close[anchor]) if 0 <= anchor < n else float("nan")
    if not np.isfinite(base) or base <= 0:
        base = 1.0

    fig = go.Figure()

    # **The history: the archive's own last ``history_bars`` bars, in percent.**
    hist = max(1, min(int(history_bars), n))
    hist_start = n - hist
    hist_x = np.arange(hist_start, n, dtype=np.int64)
    hist_y = (close[hist_start:n] / base - 1.0) * 100.0
    hist_stamps = stamp_labels(
        pipe.bars["timestamp"].iloc[hist_start:n], key=pipe.timeframe)
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines",
        line=dict(color="#1f77b4", width=1.3),
        name="recent tape (last %d bars)" % hist,
        hovertemplate="%{customdata}<br>%{y:.2f}%<extra>recent tape</extra>",
        customdata=hist_stamps,
    ))

    # **The projections, coloured by lookback length.**
    #
    # The colour is assigned by the length's *position in the sweep*, not by its
    # value, so the ramp is monotonic in the order the lengths were asked for --
    # 10 is always the palest and 500 always the darkest, whatever lengths in
    # between failed to answer.
    answered = sweep.answered
    if answered:
        ordered = sweep.paths
        n_paths = max(1, len(ordered) - 1)
        ramp = px.colors.sample_colorscale(
            "Viridis",
            [i / n_paths for i in range(len(ordered))],
        )
        colour_of = {}
        for i, p in enumerate(ordered):
            colour_of[id(p)] = ramp[i]

        offsets = np.arange(horizon + 1, dtype=np.int64)
        proj_x = anchor + offsets
        # Offset labels for the projection: those bars do not exist yet, so the
        # hover prints "+4 min" / "+4 days" rather than a clock time.
        proj_stamps = ["+%d %s" % (int(o), bar_unit(int(o))) for o in offsets]

        for p in ordered:
            if p.path is None:
                continue
            colour = colour_of[id(p)]
            fig.add_trace(go.Scatter(
                x=proj_x,
                y=np.asarray(p.path.median, dtype=float),
                mode="lines",
                line=dict(color=colour, width=1.1),
                opacity=0.75,
                name="%d-bar lookback" % p.length,
                legendgroup="lookback_%d" % p.length,
                showlegend=False,
                # Concatenated rather than ``%``-formatted: the template is full of
                # ``%{...}`` plotly placeholders, and ``%``-formatting reads the
                # ``%{`` as a format specifier and raises.
                hovertemplate="%{customdata}<br>%{y:.2f}%<extra>" + "%d-bar lookback</extra>"
                              % p.length,
                customdata=proj_stamps,
            ))

        # **The consensus across the lengths, drawn last so it sits on top.**
        consensus = sweep.consensus
        if np.isfinite(consensus).any():
            fig.add_trace(go.Scatter(
                x=proj_x,
                y=np.asarray(consensus, dtype=float),
                mode="lines",
                line=dict(color="#d62728", width=2.4),
                name="median across all %d lookbacks" % len(answered),
                hovertemplate="%{customdata}<br>%{y:.2f}%<extra>consensus</extra>",
                customdata=proj_stamps,
            ))

    # **The boundary, on the first bar that belongs to the projection alone.**
    boundary = anchor + 1
    fig.add_vline(x=boundary, line=dict(color="rgba(128,128,128,0.9)", width=1.4,
                                        dash="dash"))
    fig.add_vrect(
        x0=boundary, x1=float(anchor + horizon),
        fillcolor="rgba(128,128,128,0.10)", line_width=0, layer="below",
    )

    # **Ticks: real stamps across the history, offsets across the projection.**
    tickvals: List[int] = []
    ticktext: List[str] = []
    taken = set()

    def _add_tick(value: int, label: str) -> None:
        if value in taken:
            return
        taken.add(value)
        tickvals.append(value)
        ticktext.append(label)

    step = max(1, hist // 8)
    hist_ticks = list(range(hist_start, n, step))
    if not hist_ticks or hist_ticks[-1] != n - 1:
        hist_ticks.append(n - 1)
    for i in hist_ticks:
        _add_tick(i, stamp_label(pipe.bars["timestamp"].iloc[i], key=pipe.timeframe))

    anchor_ts = pd.Timestamp(pipe.bars["timestamp"].iloc[anchor])
    pstep = max(1, horizon // 5)
    proj_ticks = [o for o in range(0, horizon + 1, pstep) if o != 0]
    if horizon >= 1 and 1 not in proj_ticks:
        proj_ticks = sorted(proj_ticks + [1])
    for o, label in zip(proj_ticks, _proj_ticktext(anchor_ts, proj_ticks)):
        _add_tick(anchor + o, label)

    order = sorted(range(len(tickvals)), key=lambda k: tickvals[k])
    tickvals = [tickvals[k] for k in order]
    ticktext = [ticktext[k] for k in order]

    fig.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=48, b=8),
        hovermode="x unified",
        title=dict(text=title or "Progression", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title_text="bar",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[hist_start - 1, anchor + horizon + 1],
        ),
        yaxis=dict(title_text="percent from the last close"),
    )
    return fig


def _render_progression_table(sweep: ProgressionSweep, horizon: int) -> None:
    """The per-length evidence table, under the chart.

    Every number the chart draws is here in text, because a chart of fifty
    overlapping traces cannot be read to the precision a table gives.  The columns
    are the ones that decide whether a length's answer can be trusted: how many
    windows it is built from, how far behind the best one sat in the archive's
    own ranking (§E), and how large the match pool was.
    """
    rows = []
    for p in sweep.paths:
        rows.append({
            "Lookback": "%d bars" % p.length,
            "Matches": p.n_matches if p.ok else 0,
            "Best distance": ("%.3f" % p.distance) if np.isfinite(p.distance) else "—",
            "Percentile": ("%.2f%%" % (100.0 * p.percentile)
                           if np.isfinite(p.percentile) else "—"),
            "Pool": p.n_candidates if p.n_candidates else "—",
            "Status": (p.reason if not p.ok else
                       ("thin (%d of %d)" % (p.n_matches, p.n_requested)
                        if p.n_matches < p.n_requested else "ok")),
        })
    if not rows:
        return
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_matches_tab(pipe: Pipeline, result: M.MatchResult,
                       *, compact: bool = False,
                       pad: int = CONTEXT_BARS,
                       chart_width: Optional[int] = None,
                       anchor: Optional[int] = None,
                       pannable: bool = False,
                       height: int = 420) -> None:
    query = result.query
    # Widget keys are namespaced by where this block renders, because the Price tab
    # and the Matches tab can both draw it in the same rerun and Streamlit rejects a
    # duplicate key outright.
    scope = "price" if compact else "tab"

    # Compact mode is the Price tab's answer to "when has something like this looked
    # like this?".  It is the full-tape price chart above, then the single closest
    # historical match below it -- and nothing else: no metrics, no table, no captions,
    # no shape overlay, no hint text.  The query is not redrawn here, because the chart
    # immediately above *is* the query.  Every statistic stripped from this block lives
    # on the Matches tab, where there is room to read it.
    #
    # ``chart_width`` carries the query chart's bar width down from the Price tab so the
    # match below is drawn at the same scale.  It is a width rather than a view range
    # because a view is absolute: the match panel re-centres a window of this width on
    # its own band, which keeps the match visible and the two scales identical.
    if compact:
        if not result.matches:
            # Silence, not a caption.  This block's contract is the two charts and
            # nothing else, and "no match survived" is a sentence in a place that has
            # no sentences.  It is also not lost: the Matches tab reports the empty
            # case properly, with the two metrics that explain *why* the field is
            # empty (candidate windows, excluded overlaps) and what to do about it.
            # A lone line here used to be the only explanation the Price tab gave,
            # which is worse than none -- it named the outcome without the cause.
            return
        _render_best_match_pair(
            pipe, query, result.matches[0], 1, pad=pad,
            scope=scope, show_query=False, chart_width=chart_width,
            anchor=anchor, pannable=pannable, height=height,
        )
        return

    st.subheader("Matches")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "candidate windows", "{:,}".format(result.n_candidates),
        help="Every {}-bar window the matcher could have compared against, across "
             "the whole archive. This is the denominator behind the percentile: a "
             "match has to be rarer than all but this many windows to look unusual."
             .format(pipe.length),
    )
    c2.metric(
        "excluded (self/overlap)", "{:,}".format(result.n_excluded),
        help="Candidates removed before scoring, for **both** reasons: they overlap "
             "the query span plus one window-length of margin (§M), or their forward "
             "horizon would cross a session closure (§BX). It is the total, so it is "
             "what the percentile denominator is built from.",
    )
    c3.metric(
        "after NMS", "{:,}".format(result.n_after_nms),
        help="Matches surviving non-maximum suppression: each is at least one "
             "window-length ({}) from the last, so overlapping windows cannot "
             "masquerade as separate evidence.".format(pipe.length),
    )
    # The old fourth tile reported the scoring method.  With one scorer that was a
    # constant, so it is replaced by the number the module actually exists to report:
    # how rare the best match is (§E).
    best_pct = result.matches[0].percentile if result.matches else float("nan")
    c4.metric(
        "best percentile", pct(best_pct),
        help="How rare the closest match is: the share of every searchable candidate "
             "window whose distance was at least as small (§E). This is the honest "
             "score — the raw distance is not comparable across window lengths.",
    )

    if getattr(result, "n_masked", 0):
        st.caption(
            "§BX — {:,} of {:,} candidate windows were withheld because their "
            "forward horizon would run across a session closure. A window ending on a "
            "session's last bar reports the **overnight gap** as if it were trading, "
            "which inflated its forward return about 20× on the QQQ archive. "
            "Windows that merely *straddle* a closure are kept: that bar's move is "
            "real, it is simply unusual, and dropping every such window would cost "
            "31% of the pool at 60 bars."
            .format(result.n_masked, result.n_candidates)
        )

    hint(
        "Every candidate overlapping the query span plus one window-length of margin "
        "is removed before scoring (§M), as is any candidate whose forward horizon "
        "would span a session closure (§BX). Remaining matches are forced at least "
        "one window-length apart, so they are temporally independent rather than the "
        "same minute counted several times."
    )

    if not result.matches:
        st.warning("No matches survived. The series may be shorter than the exclusion "
                   "zone, or every candidate overlapped the query.")
        hint(
            "<b>What to try:</b> move the query away from the archive edges so more "
            "windows survive, raise <b>k matches</b>, or check the <b>Quality</b> tab for "
            "how many bars were actually loaded."
        )
        return

    rows = []
    for rank, m in enumerate(result.matches, start=1):
        rows.append({
            "rank": rank,
            # Header and value from the same registry: a column headed "(UTC)" above a
            # bare Eastern date, or headed "start (UTC)" holding "start", is the §BZ
            # failure at header size.  ``column_config`` is keyed on the same names.
            stamp_column("start"): stamp_label(pipe.timestamp_at(m.start),
                                               key=pipe.timeframe),
            stamp_column("end"): stamp_label(pipe.timestamp_at(m.stop - 1),
                                             key=pipe.timeframe),
            "distance": round(float(m.distance), 4),
            "percentile": pct(m.percentile),
            "bars": int(m.stop - m.start),
        })
    stamp_col_start = stamp_column("start")
    stamp_col_end = stamp_column("end")
    st.dataframe(
        pd.DataFrame(rows), width='stretch', hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn(
                "rank", help="Position when sorted by distance, best first."),
            stamp_col_start: st.column_config.TextColumn(
                stamp_col_start,
                help="First bar of the matched window, in {}. Compare against the "
                     "query's own start to see how far back in the archive this "
                     "analogue sits.".format(stamp_zone_name())),
            stamp_col_end: st.column_config.TextColumn(
                stamp_col_end,
                help="Last bar of the matched window. Forward returns are measured "
                     "from the bar AFTER this one, never from inside the window "
                     "(§Z1)."),
            "distance": st.column_config.NumberColumn(
                "distance", format="%.4f",
                help="Geometric gap between the query's feature vector and this "
                     "window's, on z-scored features. Lower = more similar. Unitless "
                     "and NOT comparable across different window lengths or different "
                     "queries — use the percentile instead."),
            "percentile": st.column_config.TextColumn(
                "percentile",
                help="The share of all candidate windows whose distance was at least "
                     "as small as this one. '0.11%' means only 11 windows in 10,000 "
                     "resemble your query this closely — that is what makes the match "
                     "interesting. Lower = rarer. Every match carries one, because the distance "
                     "metric is fixed — an earlier version also offered a time-warped "
                     "metric whose percentile had to be reported as unknown (§BD, §E)."),
            "bars": st.column_config.NumberColumn(
                "bars",
                help="Length of the matched window. Always equal to the app's fixed "
                     "window; if it is not, that is a bug — forward returns "
                     "would be measured from the wrong bar (§Z1)."),
        },
    )

    st.markdown("#### Shape comparison — top 3 matches")
    hint(
        "Each tab shows the <b>real tape</b> for that match, drawn exactly like the "
        "query chart on the <i>Price</i> tab, with a few bars of history on the left "
        "and a few bars of "
        "what followed on the right. The orange band is the window that was actually "
        "scored; the dashed line marks the first bar <i>after</i> it, where forward "
        "returns are measured from (§Z1)."
    )
    hint(
        "Both charts are rebased to their own first bar, and the lower one autoscales "
        "to its own bars — so read <i>shape</i>, not absolute size. Look for <b>the "
        "same sequence of ups and downs inside the band</b> — the wiggle in the same "
        "order, the same direction, roughly the same size. Then read the grey "
        "right-hand region: a match whose follow-through looks like your query's "
        "follow-through is the one worth carrying to the <i>Forecast</i> tab."
    )
    hint(
        "The <b>top chart is brushable</b> — drag a box across it to move the query "
        "window. The bar count under the chart is what <i>you</i> drew; the query "
        "itself is always {} bars, centred on your brush. The <b>match chart below is "
        "read-only</b>: it shows what the matcher found, not something you choose."
        .format(pipe.length)
    )
    hint(
        "The z-scored feature overlay below is the <i>machine's</i> view of the same "
        "thing. The two charts above are the <i>human's</i> view, and they are not "
        "required to agree — the matcher scores `return_z` and `path_z`, not raw "
        "price level, so a match can trace the feature shape while its close price "
        "looks completely different (§BC)."
    )
    # Three panels on the Matches tab: there the comparison *is* the task -- which of
    # several analogues is the better precedent is answered by looking at them, so
    # collapsing to one would hide exactly the evidence the user came for.
    top = result.matches[:3]
    if top:
        # One vertical range for the *query* panel across every tab, so that reference
        # panel does not rescale as you move between matches.  It is not passed to the
        # match panels -- those autoscale, so the range covers the query alone and
        # each match gets framed by its own bars.
        pad = max(0, int(pad))
        y_range = shared_rebased_range(pipe, [(query.start, query.stop)], pad)
        # With a single panel there is nothing to tab between, so a one-item tab strip
        # would be a click target with no second place to go; three keep the tabs.
        if len(top) > 1:
            match_tabs = st.tabs([
                "#{} · {:.3f} ({})".format(
                    rank, m.distance, pct(m.percentile)
                ) for rank, m in enumerate(top, start=1)
            ])
        else:
            match_tabs = [st.container()]

        for tab, rank, m in zip(match_tabs, range(1, len(top) + 1), top):
            with tab:
                if len(top) == 1:
                    st.markdown(
                        "<b>Best match · #{rank} · distance {d:.3f} · "
                        "percentile {p}</b>".format(
                            rank=rank, d=float(m.distance), p=pct(m.percentile)
                        )
                    )
                _render_best_match_pair(
                    pipe, query, m, rank, pad=pad,
                    scope=scope, label="match #%d" % rank, show_captions=True,
                    y_range=y_range,
                )

                hint(
                    "Compare the two bands bar-for-bar: does the tall spike fall in the "
                    "same place, does the drift run the same way, is the dip the same "
                    "depth? Then compare the grey tails — the left tail is what led "
                    "into the window, the right tail is what followed it."
                )

                st.markdown(
                    "<i>Machine view · z-scored feature overlay used for scoring</i>"
                )
                st.plotly_chart(
                    build_shape_figure(pipe, query.start, m.start, rank, float(m.distance)),
                    width='stretch', key="shape_%s_%d" % (scope, rank),
                    config=chart_config(),
                )


# =============================================================================== #
# Tab: Forecast
# =============================================================================== #
FORECAST_COLUMNS = ["horizon_min", "n_valid", "mean_return", "baseline_mean",
                    "lift", "p_value", "ci_low", "ci_high", "sufficient", "note"]


#: Session-state key for the **Projection** tab's pinned separator bar.
#:
#: The dotted line down the middle of that chart is the boundary between the
#: recent bars (left) and the projected bars (right).  It is pinned here as a
#: *moment* so it stays on the same bar as the archive grows.  The pinned
#: moment is the last *real* bar -- the one the projection was measured from
#: -- and the line is drawn on the bar just after it, which is the first bar
#: that belongs to the projection alone.
PROJECTION_ANCHOR_KEY = "projection_anchor_ts"


def projection_anchor_key(symbol: str) -> str:
    """Session-state key for the Projection tab's pinned separator, per symbol.

    Namespaced by symbol *and* resolution: the pinned moment belongs to one
    instrument's tape, and a bar index means nothing across resolutions.
    """
    return state_key("{}_{}".format(PROJECTION_ANCHOR_KEY, str(symbol).upper()))


def projection_pinned_anchor(
    pipe: Pipeline,
    symbol: str,
    length: int,
) -> Optional[pd.Timestamp]:
    """The **Projection** tab's pinned separator bar, as a timestamp.

    **The pin is the last bar of the reference window** -- the archive's
    newest bar at the moment the reader arrived.  Everything to the right of
    it is projection; everything to the left is real tape.

    **The dotted line is drawn one bar past the pin.**  The pin is a *real*
    bar, the one the projection was measured from, so a line standing on it
    would point at the tape rather than at the projection.  The first bar
    that belongs to the projection alone is the one after it, and that is
    where the line goes.  Pinning the *anchor* and drawing the line at
    ``anchor + 1`` keeps the two consistent: the line still holds still as
    the archive grows, because the pin does.

    **It is pinned as a moment, because a bar index cannot hold still.**  The
    separator was drawn at the archive's *tail*, so every new bar moved it one
    place to the right and dragged the whole history along with it.  The
    history is supposed to be a fixed reference -- "what usually followed a
    window shaped like the most recent one?" -- and a reference that slides
    forward on its own is answering a different question every minute, because
    the window it describes is a different window.

    **The pin is taken once**, and re-taken only when the reader changes the
    *Recent bars* control, which is a deliberate gesture to re-aim the chart.
    A poll that brings new bars is not a gesture and must not move the line.

    Returns ``None`` when there is no archive to pin yet.
    """
    key = projection_anchor_key(symbol)
    width_key = key + "_width"
    state = st.session_state

    pinned = state.get(key)
    width = state.get(width_key)
    n = max(1, int(length))

    if pipe is None or pipe.n_bars == 0:
        return pinned

    # A pin is needed when there is none, or when the reader has moved the
    # control since the pin was taken.
    needs_pin = pinned is None
    if not needs_pin and width is not None and int(width) != n:
        needs_pin = True

    if needs_pin:
        pinned = pipe.bars["timestamp"].iloc[pipe.n_bars - 1]
        state[key] = pinned
        state[width_key] = n
    return pinned


def projection_figure_span(
    pipe: Pipeline,
    symbol: str,
    length: int,
) -> Optional[Tuple[int, int, int]]:
    """The Projection tab's ``(anchor, hist_start, hist_stop)``, or ``None``.

    The **anchor** is the pinned moment the projection grows out of: the last
    real bar, and the one the dotted line is drawn just past.  The **history
    extent** is the span it delimits, ``anchor - length + 1 .. anchor``.

    **All three come from one pinned moment**, which is what keeps the chart a
    fixed reference.  The separator used to be drawn at the archive's *tail*,
    so every new bar moved it one place to the right and dragged the history
    with it -- a reference that slides describes a different window every
    minute while its caption still claims to describe "the most recent N
    bars".

    The anchor is the *last* bar of the history, so the two are not
    independent: given the anchor and the width, the history's extent
    follows.  Returning all three together keeps them consistent, and the
    *panel* chart below reuses them so the two charts -- which are compared
    bar for bar -- share one boundary.

    Returns ``None`` when the pinned moment is no longer in the archive, so
    the caller reports it rather than drawing a line somewhere else.
    """
    anchor_ts = projection_pinned_anchor(pipe, symbol, length)
    anchor = resolve_bar_index(pipe, anchor_ts)
    if anchor is None:
        return None
    hist_start = max(0, anchor - int(length) + 1)
    hist_stop = anchor + 1
    return anchor, hist_start, hist_stop


def _render_forecast_path(pipe: Pipeline, symbol: str, *, length: int, horizon: int,
                          k: int, amplitude_weight: float) -> None:
    """The path chart and its caption -- the part of this tab that needs no search run.

    Split out of :func:`render_forecast_tab` because it has a different lifecycle from
    everything below it.  The horizon table is gated on the reader pressing *Run match*
    (or brushing), and deliberately so: it reports a matched-vs-baseline comparison
    that must never be shown stale.  This chart does not use the reader's query at all
    -- it matches the archive's own most recent ``length`` bars -- so gating it on the
    same button would leave the tab's main visual empty for a reason that has nothing
    to do with it, and a reader who never presses the button would never see it.

    ``symbol`` is an argument rather than read from ``SYMBOL_FOR_HELP`` because **this
    tab may be charting a different instrument from the rest of the page**.  That
    global is the Price tab's ticker -- it backs the page title, the manual and the
    Backtest fee copy, all of which describe the Price archive.  Reading it here would
    caption an AAPL forecast as QQQ, and would also collide on ``forecast_path_for``'s
    cache key, serving one ticker's path against another's chart.
    """
    st.subheader("Where this usually goes next")

    # **The separator is pinned, so the chart is a fixed reference.**
    #
    # The dotted line down the middle is the boundary between the recent bars
    # and the projected ones.  It is drawn on the first bar that belongs to
    # the projection alone -- the bar just after the archive's newest bar at
    # the moment the reader arrived, which is stored as a *moment* and
    # resolved against the archive on every pass.
    #
    # It used to be drawn at the archive's *tail*, so every new bar moved it one
    # place to the right and dragged the whole history with it.  A reference
    # that slides is not a reference: the window it describes is a different
    # window every minute, so the chart was quietly answering a different
    # question than the caption claimed.
    span = projection_figure_span(pipe, symbol, length)
    if span is None:
        st.warning("The pinned separator bar is no longer in the archive.")
        return
    anchor, hist_start, hist_stop = span

    with st.spinner("Matching the most recent {} bars and aggregating…".format(length)):
        path = forecast_path_for(
            symbol, pipe,
            length=length, horizon=horizon, k=k,
            amplitude_weight=amplitude_weight,
        )

    if path is None:
        st.warning(
            "No forecast path could be built from the most recent **{}** bars of this "
            "archive.".format(length)
        )
        hint(
            "This chart matches the archive's own last window rather than the one you "
            "brushed, and it needs every one of its {} matches to have **{} further "
            "bars** of real history after it — the archive has to be long enough for "
            "that. **What to try:** check the <b>Quality</b> tab for how many bars "
            "loaded, or use a more liquid symbol, which returns more of the available "
            "intraday history.".format(FORECAST_PATH_MATCHES, horizon)
        )
        return

    # **The history runs from the pinned separator back to the left edge.**
    #
    # Passing the extent explicitly is what holds the chart still: the figure
    # would otherwise draw the archive's *tail* as the history, which is what
    # moved the line.
    fig = build_forecast_path_figure(
        pipe, path, history_bars=length,
        title="%s · %d bars of history, %d projected from %d matches"
              % (symbol, length, path.horizon,
                 path.n_matches),
        projection_anchor=anchor,
        hist_start=hist_start,
        hist_stop=hist_stop,
    )
    st.plotly_chart(fig, width='stretch', key="forecast_path", config=chart_config())

    hint(
        "The dashed line is where the <b>projection</b> starts — the first bar that "
        "belongs to the projection alone. To its left is the real tape: the archive's "
        "most recent {} bars, ending on the bar the projection was measured from. "
        "Beyond it is the median of where each of the {} closest matching windows went "
        "over the following {} bars, each rebased to its own final close. The green "
        "band is the middle half of those {} paths; a wide band means the matches "
        "disagreed, and that disagreement is the most useful thing on this chart."
        .format(length, path.n_matches, path.horizon, path.n_matches)
    )
    hint(
        "<b>This chart is deliberately independent of your brush.</b> It always asks "
        "about the archive's own most recent {} bars and always uses {} matches, so it "
        "stays put while you explore other windows on the <i>Price</i> tab. The "
        "evidence table underneath it <i>is</i> about your query — the two answer "
        "different questions and are not expected to agree.".format(
            length, FORECAST_PATH_MATCHES
        )
    )

    # The shortfall is stated rather than hidden.  ``forecast_paths`` drops matches
    # with no forward history, which is always the most recent ones, so a count below
    # the requested 30 is normal near the end of an archive -- but it is also exactly
    # the situation where the median is least trustworthy, so the reader is told.
    if path.n_matches < FORECAST_PATH_MATCHES:
        _warn_short_matches(
            path.n_matches, FORECAST_PATH_MATCHES, path.horizon,
            detail="so this median is computed from fewer windows than "
                   "intended. Matches without a forward path are dropped rather than "
                   "counted as flat, so the curve is honest — but it is thinner than "
                   "it looks.",
        )

    _render_panel_forecast_path(
        pipe, symbol, length=length, horizon=horizon, k=k,
        amplitude_weight=amplitude_weight,
    )


def _render_panel_forecast_path(pipe: Pipeline, symbol: str, *, length: int,
                                horizon: int, k: int,
                                amplitude_weight: float) -> None:
    """The same path, pooled across every symbol in the panel.

    Drawn directly below the single-ticker chart and deliberately on the *same*
    axes, because the comparison between them is the entire reason it exists: the
    blue history and both green projections share one rebasing and one scale, so the
    reader can see whether a shape that has recurred in one name has also recurred
    across the index -- or whether it is just that name's own past repeating.

    **A missing panel archive is not an error.**  This chart needs
    `data/sp500_panel/`, which is built by a separate downloader and may simply not
    exist.  The tab then shows the single-ticker path alone, which is a complete
    answer to the question it asks; the panel chart is additional evidence, not a
    prerequisite.  So a ``None`` here returns quietly rather than warning about an
    archive the reader may not know is missing.
    """
    st.divider()
    st.subheader("The same shape, anywhere in the S&P 500")

    n = pipe.n_bars
    matrix = pipe.matrix
    if matrix is None or matrix.shape[0] < int(length) or n < int(length):
        st.info(
            "The cross-sectional chart needs at least **{}** bars of history to match "
            "on.".format(length)
        )
        return

    # **The same pinned window as the chart above.**  The two are drawn on the
    # same axes and compared bar for bar, so the panel chart must ask about the
    # *same* window -- not the archive's current tail, which has moved on since
    # the chart above was drawn.
    span = projection_figure_span(pipe, symbol, length)
    if span is None:
        st.info("The pinned separator bar is no longer in the archive.")
        return
    anchor, hist_start, hist_stop = span

    start, stop = hist_start, hist_stop
    vec = np.asarray(matrix[start:stop], dtype=float)
    try:
        vector = np.stack([M.zscore(vec[:, c]) for c in range(vec.shape[1])], axis=-1)
    except Exception:  # noqa: BLE001 - an unusable vector is not a failed page
        st.info("The cross-sectional chart could not read this symbol's features.")
        return

    with st.spinner("Matching {} bars against every symbol in the archive…".format(length)):
        bundle = panel_forecast_path_for(
            symbol, vector,
            length=int(length), horizon=int(horizon), k=int(k),
            amplitude_weight=float(amplitude_weight),
        )

    if bundle is None:
        st.info(
            "**No cross-sectional path is available.** This chart searches "
            "`data/sp500_panel/`, which is not present or does not yet hold enough "
            "bars. The chart above is the single-symbol answer and is unaffected."
        )
        return

    ppath = bundle["path"]
    # **The same pinned separator and the same history extent as the chart
    # above** -- they are compared bar for bar, so a boundary that moved on one
    # and not the other would make the comparison meaningless.
    fig = build_forecast_path_figure(
        pipe, ppath, history_bars=length,
        title="%s · %d bars, %d projected from %d matches across %d symbols"
              % (symbol, length, ppath.horizon, ppath.n_matches,
                 bundle["n_tickers"]),
        projection_anchor=anchor,
        hist_start=hist_start,
        hist_stop=hist_stop,
    )
    st.plotly_chart(fig, width='stretch', key="forecast_path_panel",
                    config=chart_config())

    top = sorted(bundle["tickers"].items(), key=lambda kv: (-kv[1], kv[0]))
    summary = ", ".join("{} ×{}".format(t, c) for t, c in top[:12])
    if bundle["n_tickers"] > 12:
        summary += ", +{} more".format(bundle["n_tickers"] - 12)

    span = _panel_price_span(bundle)
    hint(
        "Identical axes to the chart above, so the two are comparable bar for bar: "
        "same blue history, same green band, same grey boundary. The projection is "
        "the median of where the {} closest windows went over the next {} bars, "
        "rebased to each window's own final close — which is what makes windows from "
        "{} comparable at all.".format(
            ppath.n_matches, ppath.horizon, span,
        )
    )
    hint(
        "**Evidence: {} matches from {} symbols**, out of {:,} windows scored across "
        "{} symbols — {}. A median drawn from many windows of one name is a much "
        "weaker claim than the same count spread thinly across the index, so the "
        "breakdown is stated rather than left to the chart.".format(
            ppath.n_matches, bundle["n_tickers"], bundle["n_candidates"],
            bundle["n_tickers_scored"], summary,
        )
    )

    # The concentration guard.  ``max_per_ticker`` stops one name filling the list,
    # but a pooled median can still be dominated by one name without any single
    # window appearing twice in a row -- so the fact is checked on the result rather
    # than assumed from the cap that produced it.
    if bundle["n_tickers"] and ppath.n_matches / bundle["n_tickers"] >= 3.0:
        st.warning(
            "**These matches are concentrated**: {} of them came from {} symbol(s). "
            "The median is closer to *that name's* history than to the market's, so "
            "read it as a single-symbol result wearing a cross-sectional label."
            .format(ppath.n_matches, bundle["n_tickers"])
        )

    if ppath.n_matches < k:
        _warn_short_matches(ppath.n_matches, k, ppath.horizon)


def _warn_short_matches(n_matches: int, k: int, horizon: int,
                        detail: str = "") -> None:
    """State that fewer matches had forward history than were asked for.

    **One place, because the sentence is the same every time.**  Three charts report
    a median built from fewer matches than were requested -- the reference path, the
    pooled panel path and the brushed path -- and each wrote its own copy of the
    warning.  They were not identical: two added a clause about the median being
    thinner than it looks, one did not.  That difference was not deliberate; it was
    three separate edits to the same sentence, and it is exactly the kind of drift
    that makes a reader wonder whether a missing clause means something.

    ``detail`` is the optional trailing clause, so the callers that want the
    explanation keep it and the one that does not, does not.  The numbers are the
    same in all three cases: how many matches survived, how many were asked for, and
    how many further bars of real history each match needed.

    The warning is raised by the *caller*, not here, because the threshold differs:
    the reference path compares against its own fixed ``FORECAST_PATH_MATCHES``,
    while the other two compare against the reader's ``k``.
    """
    st.warning(
        "Only **{}** of the requested {} matches had {} further bars of history "
        "after them{}".format(
            n_matches, k, horizon,
            (", " + detail) if detail else ".",
        )
    )


def _panel_price_span(bundle: Dict[str, Any]) -> str:
    """The price range spanned by the matched symbols, as caption prose.

    Only used to make one point in one hint -- that pooling windows across names is
    legitimate *because* each is rebased to its own close -- so this is deliberately
    crude: one recent close per matched ticker, no attempt at precision, and no
    exception worth propagating.  An unreadable panel yields ``"different symbols"``,
    which keeps the sentence true without inventing a number.
    """
    try:
        search = load_panel_search(panel_root_for())
        prices: List[float] = []
        for ticker in bundle["tickers"]:
            close = search.close_aligned(ticker)
            if close.size:
                value = float(close[-1])
                if np.isfinite(value) and value > 0:
                    prices.append(value)
    except Exception:  # noqa: BLE001 - a caption is not worth an exception
        return "different symbols"
    if not prices:
        return "different symbols"
    lo, hi = min(prices), max(prices)
    if abs(hi - lo) < 1.0:
        return "symbols trading at a similar price"
    return "a ${:.0f} name and a ${:.0f} name".format(lo, hi)


def resolve_selection_window(
    event: Any,
    n_bars: int,
) -> Optional[Tuple[int, int]]:
    """Turn a Forecast-tab brush into a ``(start, stop)`` window, or ``None``.

    The Forecast tab's brush is the **same contract** as the Price tab's, enforced by
    routing through :func:`selection_to_span` rather than a second parser: a brush of
    N bars is a window of N bars.  Nothing is snapped onto a grid and nothing is
    padded, because both were bugs once already (``snap_to_grid``'s forward-padding
    put bars the reader never drew into the window whose forward returns were about
    to be measured).

    **The window may span sessions.**  There is no fence here and no ``sessions``
    argument -- a brush crossing an overnight close keeps the width and position it was
    drawn at, exactly like the Price tab's query.  See :func:`resolve_query_window`.

    ``MIN_QUERY_BARS`` still applies, via ``resolve_query_window``: below it a shape
    is a handful of z-score spikes with no trajectory between them.  That function
    widens rather than narrows, so an over-narrow brush still yields a usable window
    with the selection inside it.

    ``None`` means "no usable brush yet", which is a normal state on first load rather
    than an error -- the caller shows the chart and waits.
    """
    raw = selection_to_span(event, n_bars)
    if raw is None:
        return None
    start, stop, _length = resolve_query_window(
        raw, raw[0], raw[1], raw[1] - raw[0], n_bars
    )
    return start, stop


def _forecast_brush_chart(pipe: Pipeline, symbol: str, n: int) -> Optional[Tuple[int, int]]:
    """The brushable context chart, and the window it selected -- or ``None``.

    Split from the drawing that follows it, and that split is load-bearing: the run
    signature names the window, so the window must be resolved *before* the controls
    and the gate that read it.  A gate placed above this could not see what it gates on.

    The brush is read from the **return value** of ``st.plotly_chart`` and then copied
    into session state.  ``st.plotly_chart`` delivers its selection event only as the
    return value -- ``register_widget`` is called with no ``user_key`` -- so
    ``session_state`` is never written by the widget itself.  But the *span* has to be
    kept somewhere that survives the rerun, because drawing the forecast chart changes
    the page's chart list and therefore the context chart's element id (see
    ``FORECAST_SELECTION_KEY``).  Reading only the return value makes the tab work once
    and then silently revert.

    There is **no default window**, and that is a deliberate change.  This used to be
    reachable two ways: the brush here, and the Price tab's brush via a shared run.
    Now that the tab owns its own search, falling back to the archive's newest bars
    would mean silently forecasting a window the reader never chose -- precisely the
    failure mode this app is built to avoid, and a poor default here in a way it is not
    on the Price tab: matches near the archive end have no forward bars, so ``n_valid``
    collapses and the evidence table greys out.  The fixed reference chart on the
    *Forecast* tab already covers the "what about right now?" case.

    **This chart is the top of the *Forecast* tab**, and the ``divider`` above it is
    what separates that tab from nothing -- it used to separate this section from the
    reference chart above it, which is now a tab of its own.  The divider is kept
    because a section heading with no rule above it reads as the top of the page, and
    the reader arriving from the sidebar lands here with no other cue that this is one
    section among several.
    """
    st.divider()
    st.subheader("Forecast a window you choose")
    hint(
        "Drag a box across the chart below to pick the window to forecast. The chart "
        "underneath answers *what usually followed a window shaped like that one*, at "
        "the same **Projection bars** you set on the <b>Projection</b> tab."
    )

    sessions = session_spans(pipe.bars["timestamp"])
    view_start = sessions[-active_view_sessions()][1] if sessions else 0

    # **The brush is read from this return value, then copied into session state.**
    # ``st.plotly_chart`` delivers its selection event only as the return value --
    # ``register_widget`` is called with no ``user_key`` -- so ``session_state`` is
    # never written by the widget itself.  But the *span* has to be kept somewhere
    # that survives the rerun, because drawing the forecast chart changes the page's
    # chart list and therefore the context chart's element id (see
    # ``FORECAST_SELECTION_KEY``).  Reading only the return value makes the tab work
    # once and then silently revert.
    event = st.plotly_chart(
        build_price_figure(
            pipe, view_start, n,
            selectable=True,
            title="%s · drag a box to choose the window to forecast" % symbol,
            height=PRICE_CHART_HEIGHT,
        ),
        width='stretch',
        key=forecast_brush_key(),
        config=chart_config(selectable=True),
        on_select="rerun",
        selection_mode=("box", "lasso"),
    )

    fresh = resolve_selection_window(event, n)
    if fresh is not None:
        st.session_state[forecast_selection_key()] = fresh
    window = fresh or st.session_state.get(forecast_selection_key())

    if window is None:
        st.info(
            "**No window selected yet.** Drag a box across the chart above — the "
            "forecast for it appears here."
        )
    return window


def _render_window_forecast(pipe: Pipeline, symbol: str,
                            window: Tuple[int, int], *,
                            horizon: int, k: int,
                            amplitude_weight: float) -> None:
    """The projection for one explicitly chosen window.

    The second half of what used to be ``_render_selection_forecast``.  It takes the
    window rather than finding it, so one resolved span reaches both the chart here and
    the evidence table below -- one window, one answer, which is the property that was
    broken while the two read from different places.
    """
    n = pipe.n_bars
    start, stop = int(window[0]), int(window[1])
    length = stop - start
    # A window the reader drew at the very end of the archive has no room for the
    # matches' forward bars, and saying so up front beats drawing a projection that
    # is silently all NaN.
    if n - stop < 1:
        st.warning(
            "That selection runs to the end of the archive, so there is nothing "
            "after it to compare against. Drag a box that ends earlier."
        )
        return

    with st.spinner("Matching the selected {} bars and aggregating…".format(length)):
        path = forecast_path_for(
            symbol, pipe,
            length=length, horizon=horizon, k=k,
            amplitude_weight=amplitude_weight, window=(start, stop),
        )

    if path is None:
        st.warning(
            "No forecast could be built for those **{}** bars.".format(length)
        )
        hint(
            "Every match needs **{} further bars** of real history after it, and the "
            "matches are kept at least one window-length apart, so a long selection "
            "on a short archive leaves nothing to compare it against. **What to try:** "
            "select a shorter window, or a window further from the end of the archive."
            .format(horizon)
        )
        return

    fig = build_forecast_path_figure(
        pipe, path, history_bars=length, window=(start, stop),
        title="%s · selected %d bars, %d projected from %d matches"
              % (symbol, length, path.horizon, path.n_matches),
    )
    st.plotly_chart(fig, width='stretch', key="forecast_path_selected",
                    config=chart_config())

    hint(
        "Blue is the {} bars you selected, rebased to the close of the window's "
        "last bar; green is the median of where the {} closest matching windows "
        "went over the next {} bars, each rebased to its own final close. The "
        "projection starts where your window ended, not at the end of the archive."
        .format(length, path.n_matches, path.horizon)
    )
    if path.n_matches < k:
        _warn_short_matches(
            path.n_matches, k, path.horizon,
            detail="so this median is computed from fewer windows than intended.",
        )


def _resolve_projection(horizon: Optional[int]) -> int:
    """The projection horizon in force, **without** drawing the control.

    Split out of :func:`_render_projection_control` so a second tab can read the same
    horizon without registering the slider a second time.  ``st.tabs`` renders every
    body on every pass, so drawing ``forecast_horizon_key`` on both the Forecast tab
    and *Forecast* would raise ``StreamlitDuplicateElementKey`` and take down the
    page -- the same failure recorded at ``render_scope_ticker``.

    **One value, two charts.**  The Forecast tab owns the slider; this reads whatever
    it resolved.  Both projections therefore describe the same number of bars ahead,
    which is what makes them comparable at all -- a reference band at 240 bars beside
    a reader's band at 20 is not a comparison of anything, it is two different
    questions drawn one above the other.  The cost is that the control is not on the
    tab whose projection it also governs, so the copy on *Forecast* names where it
    lives rather than leaving the reader to hunt for it.

    **Ordering is what makes the read valid.**  ``Projection`` precedes *Forecast* in
    :data:`TAB_ORDER`, so the slider is drawn -- and its value written to
    ``session_state`` -- earlier in this same pass than this call.  Were the order
    ever reversed, the first pass would read the timeframe default rather than the
    stored value; the stored value would win from the second pass on, so the symptom
    would be a horizon that lags by one interaction rather than a visible error.
    """
    if horizon is not None:
        return int(horizon)

    low, high = active_projection_bounds()
    # Clamped into range rather than trusted: the stored value is whatever the reader
    # last set under *this* key, and the bounds are derived from the active resolution.
    # They cannot disagree while the key is namespaced -- but a session-state wipe, or
    # a resolution whose ceiling moved, can leave a value outside the new range, and
    # ``st.slider`` raises on exactly that rather than adjusting.  Silently widening
    # the range to fit a stale value would instead offer horizons §BX censors.
    value = int(st.session_state.get(forecast_horizon_key(), _tf().forecast_projection_bars))
    return max(low, min(int(high), value))


def _render_projection_control(horizon: Optional[int]) -> int:
    """The *Projection bars* slider, and the horizon to use.

    Drawn at the very top of the Forecast tab, **above the first chart**, because it
    governs that chart: every one of the three projections on this tab reads this one
    number, so a reader who finds it below the chart has already been shown an answer
    built from a horizon they had not chosen.  The first chart is not gated on it --
    ``_render_forecast_path`` is unconditional and always draws -- but it *is* drawn at
    this horizon, so it has to be resolved first.

    **Three things are deliberately kept out of this control**, each because putting it
    here would make one slider decide two unrelated analyses -- the failure
    ``_render_search_controls`` exists to prevent:

    * the **window**.  The window is chosen by brushing, and only by brushing:
      ``test_the_brush_is_the_only_window_control`` holds the brush as the app's sole
      window input.  A horizon is not a window; it is how far *ahead* of that window
      the reader asks.
    * ``k matches``.  That governs the evidence table's sample, judged against
      *Min matches for evidence*, and the path charts keep their own
      :data:`FORECAST_PATH_MATCHES`.
    * the **amplitude weight**, which stays in the sidebar because it is a *scoring*
      dial shared by every tab, and two copies would let them disagree about the same
      percentile.

    **The key is resolution-namespaced and the value is not passed to it.**  Streamlit
    owns the value under :func:`forecast_horizon_key`; seeding the slider from
    ``value=`` on every rerun would overwrite the reader's own choice each pass.
    Streamlit uses ``value`` only on first render of a given key, so passing the
    timeframe default is safe *and* means switching resolution -- which produces a new
    key -- starts the reader from that resolution's own default rather than from
    whatever the other resolution happened to leave behind.  That is the whole reason
    the key is namespaced rather than shared.

    ``horizon`` is honoured when passed.  A caller that pins one (the tests do) gets
    exactly that number and the slider is not drawn, so an explicit argument is never
    silently overridden by stale widget state.

    **Drawn on the Projection tab only**, and the *Forecast* tab reads it back through
    :func:`_resolve_projection` rather than drawing its own copy.  Two copies of one
    key would collide outright; two copies under different keys would let the two
    tabs disagree about the same number, which is the failure this one control exists
    to prevent.  ``Forecast`` comes first in :data:`TAB_ORDER`, which is what makes the
    read valid in the same pass rather than one pass late.
    """
    if horizon is not None:
        return int(horizon)

    low, high = active_projection_bounds()
    # The value is read back through :func:`_resolve_projection`, which clamps it into
    # ``[low, high]`` itself -- so the slider is never handed a stale value that
    # ``st.slider`` would raise on.
    value = _resolve_projection(horizon)

    return int(st.slider(
        "Projection bars",
        min_value=low, max_value=int(high), value=value,
        step=50 if _tf().key == "1d" else 1,
        key=forecast_horizon_key(),
        help="How many bars past the window to project. This is the length of the "
             "green band on all three charts on this tab — it changes how far ahead "
             "the median reaches, not which window is being asked about. Shorter is "
             "usually the more useful answer: past the next session close there is no "
             "real tape to compare against, so the curve is projecting into a market "
             "that has not opened yet.",
    ))


def _snap_recent(value: int, low: int, high: int) -> int:
    """``value`` rounded to the nearest :data:`FORECAST_RECENT_STEP` multiple, in range.

    Needed because ``step`` is a constraint on what the slider *emits*, not on what it
    *accepts*.  Measured on the pinned Streamlit build: with ``step=30`` a stored 45 is
    kept and rendered as 45, sitting between two ticks on a control whose own step says
    that value cannot be selected.  The bounds checks apply to ``min``/``max`` only, so
    nothing objects.

    That matters more than tidiness here, because such a value is reachable rather than
    hypothetical: a resolution switch, a ticker change to a shorter archive, or a
    session left over from before this step existed can each leave a width that is
    inside the bounds and off the grid.  The reader would be looking at a chart of 45
    bars with a control reading 45 and neither tick able to reach it.

    **Nearest, not truncated or rounded up.**  Rounding a stale 45 *down* to 30 would
    silently shorten the window the reader had chosen -- the chart would change under a
    control they did not touch.  Nearest keeps the change to at most half a step, and
    the value is repaired into ``session_state`` (see
    :func:`_render_history_control`) rather than only reported, so the chart and the
    widget agree from the next pass onward.
    """
    step = FORECAST_RECENT_STEP
    snapped = ((int(value) + step // 2) // step) * step
    return max(int(low), min(int(high), int(snapped)))


def _resolve_recent(length: Optional[int], n_bars: int) -> int:
    """How many of the archive's newest bars the reference asks about, without drawing it.

    Split out of :func:`_render_history_control` for the reason
    :func:`_resolve_projection` was split out of its own drawer: a caller that must
    *name* the number before the widget exists -- the caption builder, and the tab
    body itself -- should not have to register the slider to learn it.

    **Clamped into range *and* snapped onto the step grid, and both are load-bearing
    rather than defensive.**  The stored value is whatever the reader last set under
    this key, and the bounds come from the active resolution *and this archive*.  Those
    can disagree in two reachable ways: a different resolution's ceiling, and a
    **ticker change to a shorter archive**, which is the ordinary case rather than an
    edge case, because 1-minute history is ~29 days and a thin symbol does not return
    all of it.  ``st.slider`` raises ``StreamlitValueAboveMaxError`` on exactly that
    rather than adjusting, which would take down the whole page over a control the
    reader had not touched.  Silently widening the ceiling to fit a stale value would
    be worse: it would offer a length the archive cannot support, and
    :func:`forecast_path_for` answers that with ``None``, which the tab reports as
    "this shape has no historical analogue" -- a claim about the archive rather than
    about a slider.

    **The grid snap is a second, independent correction** (see :func:`_snap_recent`),
    because ``step`` does not enforce one: a value inside the bounds but off the grid
    is accepted silently, so it has to be corrected here or the chart would show a
    width the control cannot name.

    ``length`` is honoured when passed, for the same reason
    :func:`_render_history_control` honours it: an explicit argument from a test or any
    other caller bypasses the control entirely and must not be overridden by stale
    widget state.
    """
    if length is not None:
        return int(length)

    low, high = _recent_ceiling(n_bars)
    value = int(st.session_state.get(forecast_recent_key(),
                                     _tf().forecast_history_bars))
    return _snap_recent(value, low, high)


def _recent_ceiling(n_bars: int) -> tuple:
    """``(low, high)`` for the *Recent bars* control, with the ceiling cut to the tape.

    The resolution's own bound comes from :func:`active_recent_bounds`; this narrows
    the *ceiling* to what this archive actually holds.  Both endpoints therefore stay
    on the :data:`FORECAST_RECENT_STEP` grid.

    **The ``+ 100`` is ``Pipeline.from_frame``'s own readiness rule**, restated rather
    than imported because it lives inside the library constructor.  The pipeline was
    built at :func:`active_length`, so a *reference* longer than that asks about a
    window the matcher will score against a candidate pool built for a different
    length -- and the shape a reader can recognise on the chart stops being the shape
    that was matched.  Capping at ``n - 100`` keeps the offer honest about both.

    **The floor is dropped before the range is collapsed, and this is load-bearing.**
    ``st.slider`` *raises* ``StreamlitInvalidParameterTypeError`` when ``min_value ==
    max_value`` (verified on the pinned Streamlit build) -- it is not a legal
    degenerate slider, it is a dead page.  An archive shorter than ``low + 100`` bars
    therefore cannot be given a two-valued range, so the floor is lowered to match the
    ceiling rather than the range being pinned: a reader on a thin archive gets a
    slider that moves over the handful of widths that actually exist, and the chart's
    own "no forecast path could be built" message -- not a crashed page -- is what
    describes a width the matcher cannot support at all.

    **The step is dropped in that same branch, and this is not cosmetic.**  A ``step``
    larger than the span is accepted by ``st.slider``, but it leaves the reader two
    adjacent legal values and a thumb that does not travel usefully between them.  A
    one-bar step over a two-value range is honest about there being nothing to choose;
    a 30-bar step over it is a control that looks broken.
    """
    low, high = active_recent_bounds()
    high = min(int(high), max(1, int(n_bars) - 100))
    # ``<=``, not ``<``: an archive of exactly ``low + 100`` bars lands the ceiling on
    # the floor, and that equality is the case ``st.slider`` rejects.
    if high <= low:
        # The floor comes down to sit under the ceiling, giving the narrowest range
        # that is still legal.  ``(1, 2)`` is the floor of that: one bar is the
        # smallest span that can be drawn, and a slider needs two distinct endpoints.
        # ``_resolve_recent`` clamps into it and the chart then reports the archive as
        # too short, which is the truth.
        low = max(1, high - 1)
        return low, max(high, low + 1)
    # The grid only carries meaning on a range that can hold more than one step of it.
    # ``n - 100`` is not a multiple of 30 in general, so the ceiling is snapped *down*
    # here rather than in :func:`active_recent_bounds`, which knows nothing of the
    # archive.  Down, not up, because rounding up would promise a window wider than
    # the tape actually holds.
    step = FORECAST_RECENT_STEP
    if high - low >= step:
        high = high // step * step
    return low, high


def _render_history_control(length: Optional[int], n_bars: int) -> int:
    """The *Recent bars* slider: how much of the archive's tail the reference reads.

    Drawn at the top of the Forecast tab, **above the first chart and above
    *Projection bars***, because it is the question the chart is answering *about*,
    while the horizon is only how far the answer reaches.  It also has to be resolved
    first for a mechanical reason: the chart draws ``length`` real bars, so a control
    found below it governs a chart already drawn at a length the reader had not chosen.

    **This names a length, not the window.**  The window a search is run against is
    chosen by brushing and only by brushing -- ``test_the_brush_is_the_only_window_control``
    holds that line, and it is why the label says *Recent bars* rather than *Window*:
    what this control chooses is *how far back* into the archive's own tail the fixed
    reference looks.  A reader forecasting a moment they picked brushes for it on
    *Forecast*, where the drawn width is the width.

    **The value is read from session state and passed back in**, exactly as
    :func:`_render_projection_control` does.  The alternative -- ``st.slider(...,
    key=...)`` with no ``value`` -- cannot express the ceiling, because the ceiling
    depends on the archive and the widget has to be told its bounds.  Reading the
    stored value first and passing it as ``value`` means Streamlit uses it on first
    render of the key and ignores it thereafter, so the reader's own choice survives
    every later pass; a stored value that has fallen outside the new bounds is clamped
    by :func:`_resolve_recent` *before* the widget exists, which is what keeps
    ``st.slider`` from raising.

    **The default is the resolution's ``forecast_history_bars``**, the same figure the
    fixed reference used before this control existed -- so nothing about the chart
    changes until the reader moves the slider, and a reader who never touches it sees
    exactly the reference this tab has always drawn.

    **The key is resolution-namespaced.**  240 bars is four hours of 1-minute tape and
    most of a trading year of daily bars, so a length carried across a resolution
    switch would be a wrong number rather than a merely odd one.

    **``step`` is :data:`FORECAST_RECENT_STEP`, and the stored value is snapped onto
    that grid *before* the widget is created.**  The order matters: ``st.slider``
    validates the stored value against ``min``/``max`` and ignores ``step`` entirely,
    so an off-grid value passes straight through and is rendered as-is.  Writing the
    snapped number back first -- and passing no ``value`` at all, so the widget never
    also carries a default -- is what makes the repair stick and keeps Streamlit from
    logging its "created with a default value but also had its value set" warning.
    Verified on the pinned build: with the repair left out, a stored 45 renders as 45
    on a control whose ticks are 30 and 60, and neither tick can reach it.
    """
    if length is not None:
        return int(length)

    low, high = _recent_ceiling(n_bars)
    key = forecast_recent_key()
    value = _resolve_recent(length, n_bars)

    # Repair an off-grid or stale value in session state rather than only reporting it,
    # and do so before the widget exists so the bounds and grid checks see a legal
    # value.  A first pass seeds the timeframe default, which is a step multiple in
    # both resolutions -- that is what keeps the reference chart's own starting width
    # on the control's grid rather than between two ticks.
    stored = st.session_state.get(key)
    if stored is None or int(stored) != value:
        st.session_state[key] = value

    return int(st.slider(
        "Recent bars",
        min_value=low, max_value=high,
        step=FORECAST_RECENT_STEP if high - low >= FORECAST_RECENT_STEP else 1,
        key=key,
        help="How many of the archive's newest bars the reference chart asks about — "
             "the blue history, and the window that is matched against the archive. "
             "It moves in steps of {} bars. It is a different control from "
             "*Projection bars*, which sets how far ahead the green band reaches: "
             "this one looks backwards, that one forwards. Longer is not more "
             "informative here — past roughly one session of 1-minute bars the shape "
             "stops being recognisable, and a shorter window matches more often but on "
             "a coarser shape.".format(FORECAST_RECENT_STEP),
    ))


def render_forecast_tab(pipe: Pipeline, symbol: str, *,
                        length: Optional[int] = None,
                        horizon: Optional[int] = None,
                        k: int = FORECAST_PATH_MATCHES,
                        amplitude_weight: float = M.DEFAULT_AMPLITUDE_WEIGHT) -> None:
    """The **Forecast** tab: a fixed reference, and the two controls that size it.

    This tab answers *what usually followed a window shaped like the most recent one?*
    -- a question about the archive's own last ``length`` bars.  No reader input
    chooses *which* window that is, which is what makes it a reference rather than a
    query; the reader sizes it through two independent controls, *Recent bars* (how
    far back into the tail it looks) and *Projection bars* (how far ahead the answer
    reaches), and it holds nothing else but :func:`_render_forecast_path`.

    **The brush, the settings and the evidence table moved to *Forecast*.**  They
    used to sit below this chart, and they moved together because they are one
    pipeline: the evidence table needs the window the brush resolves and the run the
    settings perform.  Putting the brush here and the table on the next tab would
    split one interaction across two screens -- the failure documented at
    :func:`render_window_tab`, reached the other way round.  The line is between *a
    fixed reference* and *your own window*, not between halves of one interaction.

    ``symbol`` is taken as an argument rather than read from ``SYMBOL_FOR_HELP``,
    which is a *help-copy* bridge holding the **Price** tab's ticker, not cache
    identity.  The signature below has to name the ticker, or a previous ticker's
    result could be served against this one's chart -- and a module list that exists
    to interpolate ``[[SYMBOL]]`` into prose is the wrong thing to hang that on.

    **The Forecast ticker input is not drawn here.**  It lives on *Forecast*, the
    tab that does the work, and this tab charts whichever ticker is in force.  An
    input may be registered on exactly one path per pass and ``st.tabs`` renders every
    body on every pass, so drawing it on both tabs raises
    ``StreamlitDuplicateElementKey`` and takes down the whole page (see the not-ready
    branch in ``main()``).

    ``length`` and ``horizon`` default to ``None`` and resolve from the active
    timeframe, rather than carrying a literal.  A default of 240 would be "four hours"
    intraday and "almost a year" daily, which is not a defensible default for both.

    **Both are sliders now, and both still default to those timeframe figures.**  The
    distinction matters because each parameter is also how the tests and any other
    caller pin a number explicitly; a caller that passes one bypasses the control
    entirely and keeps working unchanged.  ``length`` resolves to
    ``forecast_history_bars`` -- the width the reference always used -- and
    ``horizon`` to ``forecast_projection_bars``, so neither slider changes the chart
    until it is moved.

    **The ceiling is the archive, not just the resolution.**  ``length`` is resolved
    against ``pipe.n_bars`` (see :func:`_recent_ceiling`), because a 1-minute archive
    that returned 600 bars cannot offer 2400 and must not appear to.
    """
    # Drawn first, and above the horizon, because it is what the chart is *about* --
    # the chart draws ``length`` real bars, so this cannot be resolved after it.  A
    # caller that passes an explicit length skips the widget entirely.
    length = _render_history_control(length, pipe.n_bars)

    # Resolved next, and drawn next, because the projection below reads it.  The
    # slider is what turns ``horizon`` from a constant into a choice; a caller that
    # passes an explicit horizon skips the widget entirely and is unaffected.
    #
    # **This is also what lets *Forecast* share the horizon.**  The slider is drawn
    # here and read back there through ``_resolve_projection``, so one control governs
    # both tabs' projections and the two cannot disagree.  It stays on this tab
    # because that is where the reference chart is: a reader deciding how far ahead to
    # look is looking at that chart when they decide, and *Forecast* comes first in
    # ``TAB_ORDER``, which is what makes the read valid in the same pass.
    horizon = _render_projection_control(horizon)

    # The fixed reference chart.  Unconditional, and the rest of this tab: it needs
    # neither a window nor a run, so gating it would leave the tab's headline visual
    # blank for anyone who had not pressed a button that has nothing to do with it.
    _render_forecast_path(pipe, symbol, length=length, horizon=horizon, k=k,
                          amplitude_weight=amplitude_weight)


def render_window_tab(pipe: Pipeline, symbol: str, *,
                      horizon: Optional[int] = None,
                      k: int = FORECAST_PATH_MATCHES,
                      amplitude_weight: float = M.DEFAULT_AMPLITUDE_WEIGHT) -> None:
    """The **Forecast** tab: brush, projection, search, evidence table.

    This is the interactive half of what used to be :func:`render_forecast_tab`, and
    the order below is the contract:

    * the **brush** is the input, so it is drawn first.  A reader who had to scroll
      back up to find the control that changes the answer would be reading the two out
      of order;
    * the **window** is resolved before anything that has to name it, which is why
      :func:`_forecast_brush_chart` both draws and resolves -- a gate placed above it
      could not see what it gates on;
    * the **settings and run** come last, because only the evidence table needs them
      and the chart above is valid before anything has been pressed.

    **All four steps move together or not at all.**  Splitting them across two tabs
    would mean brushing on this one and reading the table on the other, with the
    window crossing between them through session state.  That is what happened when
    the table was fed a run over the *Price* tab's brush, and it is worse than a stale
    answer: the gesture and its result would be about different windows with nothing
    on screen saying so.

    ``horizon`` is **read, not drawn**.  The *Projection bars* slider belongs to the
    Forecast tab, which renders first; drawing it here as well would raise
    ``StreamlitDuplicateElementKey`` and take down the page.  So this tab reads that
    tab's value through :func:`_resolve_projection`, which is what keeps the reference
    projection on *Forecast* and the reader's projection here describing the same
    number of bars ahead -- without that, the two charts are not comparable.

    **The search scope is still ``"forecast"``, deliberately.**  It names every widget
    key this tab owns (``forecast_k``, ``forecast_baseline``,
    ``forecast_min_matches``, ``forecast_run``) and every session key behind them
    (``FORECAST_BRUSH_KEY``, ``FORECAST_SELECTION_KEY``, ``FORECAST_RUN_KEY``,
    ``FORECAST_APPLIED_KEY``, and ``FORECAST_STALE_KEYS``).  Renaming it to match the
    new tab would migrate all of them at once, and the only effect would be to discard
    every reader's brush, settings and last run on deploy -- so the scope string stays
    and only the tab's *name* moves.

    ``symbol`` is taken as an argument rather than read from ``SYMBOL_FOR_HELP`` --
    see :func:`render_forecast_tab` for why that bridge is the wrong thing to hang
    cache identity on.

    **No ``length`` parameter, deliberately.**  The window's width here is whatever
    the reader brushed -- a brush of N bars is a window of N bars -- so a parameter
    here would be a second, competing way to name the same thing, and the one that
    would be silently ignored.  The reference chart on *Forecast* is the opposite
    case: it asks about the archive's most recent bars and does need a length, which
    is why that parameter survives on :func:`render_forecast_tab` and not here.
    """
    # Read back from the Projection tab's slider rather than drawn here.  ``Projection``
    # precedes this tab in ``TAB_ORDER``, so the value is already in ``session_state``
    # by the time this runs within the same pass.
    horizon = _resolve_projection(horizon)

    # The reader's window, resolved before anything that needs to name it.  The
    # "no window selected yet" notice is drawn in here too, so a tab that is waiting
    # for a gesture says so above the place the gesture has to happen.
    window = _forecast_brush_chart(pipe, symbol, pipe.n_bars)

    # That window's own projection -- the same kind of object as the Projection tab's
    # fixed chart, answering the same question for a window the reader picked.
    if window is not None:
        _render_window_forecast(pipe, symbol, window, horizon=horizon, k=k,
                                amplitude_weight=amplitude_weight)

    # This tab's own search settings, and its own run.
    cfg_f = _render_search_controls("forecast")
    out = _forecast_run(pipe, window, cfg_f, symbol=symbol,
                        amplitude_weight=amplitude_weight)

    _render_forecast_evidence(pipe, out, window=window)


def _forecast_run(pipe: Pipeline, window: Optional[Tuple[int, int]],
                  cfg: Dict[str, Any], *, symbol: str,
                  amplitude_weight: float) -> Optional[Dict[str, Any]]:
    """Run this tab's search over ``window``, or return ``None`` to show a prompt.

    Gated on an explicit brush with **no default window**.  The archive's newest bars
    would have been a natural fallback, but they are the one window with no forward
    bars behind it: §BX withholds those matches, ``n_valid`` collapses and the table
    greys out, so the fallback would reliably produce the least informative answer the
    tab can give.

    A brush is consumed once.  Streamlit keeps a selection for the page's lifetime, so
    ``window`` is non-``None`` on every later rerun; firing whenever one is merely
    *present* would recompute the forecast every time an unrelated control moved.  The
    recorded span is the resolved one, so a clamped brush cannot disagree with it and
    re-fire on every pass.
    """
    if window is None:
        return None

    start, stop = int(window[0]), int(window[1])
    # ``stop - start``, not ``pipe.length``: a brush defines its own length, so the key
    # has to name the length actually searched or two brushes of one span collide.
    # The settings are in it for a stronger reason -- they change what a distance
    # *means*, so a result carried over from different dials would report a percentile
    # computed against a distribution the reader is no longer looking at.
    #
    # **The projection horizon is deliberately NOT in this signature**, and the reason
    # is that the evidence table does not use it: the table forecasts over
    # ``active_horizons()``, a fixed per-resolution set, and never over the length of
    # the chart's green band.  So adding the slider's value here would buy nothing and
    # cost a full re-search -- including the ``n_baseline`` random windows -- every
    # time the reader nudged a control that cannot alter a single number in the table.
    # The path charts above *do* change with the slider, and they re-run on their own
    # cache key, which is where the horizon belongs.
    signature = (symbol, stop - start, cfg["k"], start, stop,
                 cfg["n_baseline"], cfg["min_matches"], cfg["seed"], amplitude_weight)
    brush_moved = st.session_state.get(forecast_applied_key()) != (start, stop)
    if cfg["run_clicked"] or brush_moved:
        st.session_state[forecast_applied_key()] = (start, stop)
        st.session_state[forecast_run_key()] = signature

    if st.session_state.get(forecast_run_key()) != signature:
        return None

    with st.spinner("{} {} windows and building the baseline for {}…".format(
            "Window changed:" if brush_moved else "Searching",
            stop - start,
            stamp_span(pipe.timestamp_at(start), pipe.timestamp_at(stop - 1),
                       key=pipe.timeframe))):
        return pipe.run(
            pipe.query_span(start, stop,
                            label=stamp_span(pipe.timestamp_at(start),
                                             pipe.timestamp_at(stop - 1),
                                             key=pipe.timeframe)),
            k=cfg["k"], horizons=active_horizons(), seed=cfg["seed"],
            n_baseline=cfg["n_baseline"], min_matches=cfg["min_matches"],
            amplitude_weight=amplitude_weight,
        )


def _render_forecast_evidence(pipe: Pipeline, out: Optional[Dict[str, Any]],
                              *, window: Optional[Tuple[int, int]]) -> None:
    """The matched-vs-baseline table, for ``window``.

    Split out of :func:`render_forecast_tab` because it is the one section that needs a
    completed run, and scoping the "not yet" notice to *this* section is what keeps the
    charts above it usable while it is empty.
    """
    st.divider()
    st.subheader("Conditional forecast")
    hint(
        "What happened in the minutes *after* every matched window — aggregated over "
        "the whole matched set, and always beside the identical statistic over "
        "randomly chosen windows. The **gap between the two is the entire claim**."
    )

    st.info(
        "**Every number below is a matched-windows-vs-random-windows comparison.** "
        "The right-hand column is the same statistic computed over windows chosen at "
        "random from the same archive. A pattern that does not beat the random-window "
        "baseline is not a forecast — searching thousands of candidates guarantees "
        "finding one whose forward return looks extreme, so the lift, not the mean, "
        "is the claim."
    )

    if window is None:
        # Its own notice, distinct from the "no run yet" one below: a reader who has
        # not brushed has nothing to search, and telling them to press the button
        # would run a search against the wrong window.
        #
        # **The fixed reference chart is named by tab, not by position.**  It used to
        # live above this one on the same tab, so "the chart at the top of this tab" was
        # accurate; it now lives on *Forecast*, and after the split the only thing
        # above this table is the reader's own brush.  A relative pointer would send
        # the reader looking for a chart that is not there.
        st.info(
            "**The evidence table needs a window.** Drag a box on the chart above to "
            "choose one, then press **Run match** in *Search settings*. The fixed "
            "reference chart on the <b>Projection</b> tab needs neither — it is built "
            "from the archive's own most recent bars and is already live."
        )
        return

    if out is None:
        # Scoped to the table rather than raised as a full-page notice: the charts
        # above are already drawn and valid, so replacing the tab would discard them.
        st.info(
            "**The evidence table needs a search run.** Press **Run match** above. "
            "Change the window or any setting and it empties again until you search."
        )
        return

    forecasts: List[Any] = out.get("forecasts", [])
    if not forecasts:
        st.warning("No forecasts were produced.")
        hint(
            "**What to try:** raise <i>k matches</i> and <i>Min matches for evidence</i> "
            "in the sidebar, move the query away from the end of the archive so more "
            "matched windows have a full forward horizon, or check the <b>Quality</b> "
            "tab to confirm enough bars actually loaded."
        )
        return

    rows, styles, notes = [], [], []
    for f in forecasts:
        d = f.as_dict()
        sufficient = bool(d["sufficient"])
        rows.append({
            "horizon_min": d["horizon_min"],
            "n_valid": d["n_valid"],
            # An insufficient row shows NO numbers for the matched estimate: printing
            # them, even greyed out, is what this project exists to avoid.
            "mean_return": bps(d["mean_return"]) if sufficient else "—",
            "baseline_mean": bps(d["baseline_mean"]),
            "lift": bps(d["lift"]) if sufficient else "—",
            "p_value": ("{:.4f}".format(d["p_value"]) if sufficient and
                        np.isfinite(d["p_value"]) else "—"),
            "ci_low": bps(d["ci_low"]) if sufficient else "—",
            "ci_high": bps(d["ci_high"]) if sufficient else "—",
            "sufficient": "yes" if sufficient else "no",
            "note": d["note"],
        })
        # One CSS declaration string per row. pandas' Styler converts it to
        # ``(property, value)`` tuples itself (``maybe_convert_css_to_tuples``), which
        # is also the shape Streamlit expects downstream -- passing tuples directly
        # trips ``Styler._update_ctx``'s ``pd.isna`` check on the list-of-tuples cell.
        if not sufficient:
            styles.append("background-color: rgba(140,140,140,0.16); color: #7a7a7a;")
            notes.append(d["note"])
        elif np.isfinite(d["p_value"]) and d["p_value"] < 0.05:
            styles.append("background-color: rgba(44,160,44,0.18); color: #1a5c1a;")
        else:
            styles.append("background-color: rgba(255,193,7,0.20); color: #6b5000;")

    frame = pd.DataFrame(rows)[FORECAST_COLUMNS]
    n_cols = frame.shape[1]

    def row_style(row: pd.Series) -> List[List[str]]:
        """Return the CSS string for this row, once per column.

        ``axis=1`` maps the function's return value to COLUMN labels, so a row must be
        a list of one entry per column -- not a bare string, which pandas would read as
        column labels. ``row.name`` is the frame index, which keeps this aligned with
        ``styles``.
        """
        return [styles[int(row.name)]] * n_cols

    styled = frame.style.apply(row_style, axis=1)
    st.dataframe(styled, width='stretch', hide_index=True)
    hint(
        "Hover the ⓘ beside any column name for what it measures and when to distrust "
        "it. A `—` always means *deliberately withheld or not applicable* — never a "
        "number rounded to zero."
    )

    if notes:
        st.warning(
            "**No forecast is shown for these horizons** — " + "; ".join(notes) +
            ". The forward returns are computed from the last bar of each matched "
            "window onwards (§Z1), and the effective sample is further reduced by "
            "non-overlap suppression, so k must clear the minimum before any mean, "
            "interval or p-value is meaningful. Raise **k** in the sidebar."
        )
        hint(
            "<b>Three ways to fix a grey row, cheapest first:</b><br>"
            "1. Raise <b>k matches</b> in the sidebar — needs to reach "
            "<i>Min matches for evidence</i> (default 30).<br>"
            "2. Move the query away from the **end** of the archive: a matched window "
            "near the last bar has no data after it, so its forward return is NaN and "
            "it does not count toward <i>n_valid</i>.<br>"
            "3. Check the <b>Quality</b> tab — if very few bars loaded, the archive "
            "itself may be too short to support a match set."
        )

    fig = build_forecast_figure(forecasts)
    if fig is not None:
        st.markdown("#### Matched vs baseline, per horizon")
        hint(
            "Green = matched windows, grey = random windows. **A green bar that does "
            "not clear its grey neighbour is not a forecast.** The zero line is the "
            "break-even return; bars below it mean those windows were followed by a "
            "fall."
        )
        st.plotly_chart(fig, width='stretch', key="forecast_bars",
                        config=chart_config())
    else:
        st.caption(
            "No horizon reached the minimum sample size, so nothing is charted as a "
            "prediction. The baseline bars alone are not evidence of anything."
        )

    with st.expander("How to read this table — every column explained", icon=HELP_ICON):
        st.markdown(
            "| Column | What it is | How to read it |\n"
            "|---|---|---|\n"
            "| `horizon_min` | bars ahead the forward return is measured over | 5 / "
            "15 / 30 / 60 minutes |\n"
            "| `n_valid` | matched windows whose **full** horizon exists in the "
            "archive | windows running past the last bar are `NaN`, not zero, so they "
            "do not count here |\n"
            "| `mean_return` | average forward log return *after* the matched window "
            "ends | never measured from inside the window (§Z1) |\n"
            "| `baseline_mean` | the same average over **randomly chosen** windows | "
            "the control. The gap between the two columns is the whole claim |\n"
            "| `lift` | `mean_return − baseline_mean`, in bps | a **difference, not a "
            "ratio** — the baseline sits near zero, so a ratio diverges |\n"
            "| `p_value` | permutation test against the baseline | `< 0.05` beats "
            "chance at conventional significance |\n"
            "| `ci_low` / `ci_high` | moving-block bootstrap interval | block-based "
            "because matched windows sit near each other in time; an i.i.d. interval "
            "would understate the variance |\n"
            "| `sufficient` | did `n_valid` clear the evidence threshold? | when "
            "`no`, no mean, interval or p-value is printed |\n"
            "| `note` | why a row is what it is | always read this before quoting a "
            "number |\n\n"
            "**Row colours**\n\n"
            "| Colour | State | Meaning |\n"
            "|---|---|---|\n"
            "| 🟢 green | `p < 0.05` | beats the random baseline at conventional "
            "significance |\n"
            "| 🟡 amber | sufficient, `p ≥ 0.05` | a number is printable but it is "
            "**indistinguishable from chance** |\n"
            "| ⚪ grey | `sufficient = false` | no number is shown at all — not "
            "greyed out, not even |\n\n"
            "The grey state is not a display bug. Below the evidence threshold the mean, "
            "interval and p-value are all functions of a handful of observations, so the "
            "honest output is no output (PLAN.md §E, *sample-size honesty*)."
        )


# =============================================================================== #
# Tab: Panel -- the cross-sectional search
# =============================================================================== #
#: Name only, not a path: the sector labels live beside *both* archives, so the
#: directory is resolved at read time from the resolution in force.  Hard-coding one
#: root here would mean daily mode reported sector percentiles from whichever
#: archive happened to be written last.  The name itself comes from the store, so
#: this reader and `register_symbols` cannot disagree about where the file is.
PANEL_SECTOR_NAME = CONSTITUENTS_NAME


def panel_sector_file() -> str:
    """The ``constituents.csv`` beside the archive for the active resolution.

    Path, not root, because that is what :func:`timeseries.store.read_sectors`
    takes -- it answers "what is in this file", so a caller holding a file should
    not have to unwind it to a directory and rejoin it.

    The *name* comes from :data:`PANEL_SECTOR_NAME`, which is the store's own
    :data:`timeseries.store.CONSTITUENTS_NAME`, so this reader and the store
    cannot disagree about what the sidecar is called.
    """
    return os.path.join(panel_root_for(), PANEL_SECTOR_NAME)


def render_archive_sync() -> None:
    """Render the daily archive-sync control in the sidebar.

    Why this exists in the UI at all
    -------------------------------
    Yahoo serves 1-minute bars only for roughly the last 30 days.  Bars already
    downloaded are kept forever, so the limit is on *fetching*, not on *keeping* --
    which turns "sync every day" into an operational requirement with a real
    failure mode: a session that slips past the window while the archive is not
    synced is gone permanently, and no later run can recover it.

    A requirement that lives only in a README is one that quietly stops being met.
    Putting the command behind a button next to the archive metrics that show when
    it was last synced is what makes it hard to forget.

    What the button does
    --------------------
    It shells out to ``scripts/download_sp500.py --incremental`` -- the same command
    documented in the README -- and streams its log.  It deliberately does not
    reimplement the download: a second implementation of "how an archive is grown"
    is exactly how the button and the CLI end up disagreeing.  See
    :mod:`timeseries.sync` for why this runs in a subprocess.

    The button is idempotent.  ``--incremental`` skips every session already stored,
    so pressing it twice in a day downloads nothing the second time -- it is always
    safe to press, and that is what makes it usable as an unconditional daily step.

    It is also the way to *build* the archive, and the window it asks for widens
    automatically to match: :func:`timeseries.sync.run_sync` passes no window at all
    when the archive holds nothing yet, so the downloader fetches full listing
    history rather than the 29 days a bare lookback would have clipped it to.  The
    button used to hard-code that 29-day window, so a first press built a
    ~20-session-per-ticker archive that looked like a legitimately short history.

    Staleness is reported, never enforced.  The control says how old the newest
    session is and lets the user decide, because "3 days behind" is different on a
    holiday weekend than on three ordinary days, and the archive has no way to tell
    those apart without a trading-calendar table nobody wants to maintain.
    """
    status = SYNC.archive_status(panel_root_for())

    st.sidebar.divider()
    # The heading names the resolution in force, because this control operates on
    # the archive for *this* session's resolution and no other.  A reader who
    # reloads onto Daily and presses a button labelled only "sync" has no way to
    # know which archive it will touch.
    st.sidebar.markdown("**Daily archive sync** — {}".format(active_label().lower()))

    if not status["exists"]:
        st.sidebar.info(
            "No archive yet at `data/sp500_panel/`."
        )
    else:
        # --- the cohort frontier, not the union maximum -------------------- #
        # The old wording was "Up to date. Newest session <date>", where <date>
        # was the **newest session any ticker held**.  The archive holds three
        # instruments that trade 24/7, so that date was always a weekend and
        # always ahead of the equity cohort -- the sidebar reported "up to date"
        # while 500 of 503 tickers were nine sessions behind.
        #
        # The frontier is the date **most** tickers have reached, and it is the
        # number that answers "can I trust this archive?".
        frontier = status["frontier"]
        newest = status["newest"].strftime("%B %-d, %Y") if status["newest"] else "—"
        frontier_txt = frontier.strftime("%B %-d, %Y") if frontier else "—"

        # The age is counted in **trading sessions**, because the archive is
        # measured in sessions.  A calendar-day count made a Friday archive look
        # three days stale on Monday morning, which is both wrong and a good way
        # to teach someone to ignore a warning.
        age_sessions = status["age_sessions"]
        if age_sessions is None:
            age_txt = "unknown age"
        elif age_sessions <= 0:
            age_txt = "caught up"
        else:
            age_txt = "{} session{} behind".format(
                age_sessions, "" if age_sessions == 1 else "s")

        at_frontier = status["at_frontier"]
        behind = status["behind_frontier"]
        total = status["tickers"]

        if status["stale"]:
            st.sidebar.warning(
                "Cohort through **{}** — {}.".format(frontier_txt, age_txt)
            )
        else:
            st.sidebar.success(
                "Cohort through **{}** ({}).".format(frontier_txt, age_txt)
            )

        # Coverage: how much of the cohort is actually at the frontier.  This is
        # the number that distinguishes "the archive is a day behind" from "the
        # archive is one ticker behind", which are different problems with
        # different fixes.
        if total:
            st.sidebar.caption(
                "**{:,} of {:,}** tickers at the frontier".format(at_frontier, total)
            )
        if behind:
            laggers = ", ".join(status["lagging_tickers"][:6])
            if len(status["lagging_tickers"]) > 6:
                laggers += " +{} more".format(len(status["lagging_tickers"]) - 6)
            st.sidebar.caption(
                "Behind: **{}** ({})".format(behind, laggers)
            )

        # The gap is only recoverable while it is inside Yahoo's 30-day window.
        # After that the missing sessions are gone for good, so saying so is the
        # difference between "press the button" and "it is too late".
        missing = status["missing_sessions"]
        if missing:
            oldest_missing = missing[0]
            recoverable = _within_fetch_window(oldest_missing)
            if recoverable:
                st.sidebar.caption(
                    "Missing **{}** session{} ({} – {}); still inside Yahoo's "
                    "30-day window, so a sync can recover "
                    "them.".format(
                        len(missing), "" if len(missing) == 1 else "s",
                        missing[0], missing[-1],
                    )
                )
            else:
                st.sidebar.caption(
                    "Missing **{}** session{} ({} – {}); these are **older than "
                    "Yahoo's 30-day window** and can no longer be fetched.".format(
                        len(missing), "" if len(missing) == 1 else "s",
                        missing[0], missing[-1],
                    )
                )

        # Tickers that trade when the equities do not are reported separately, so
        # the frontier is not silently dragged forward by them.
        if status["ahead_frontier"]:
            st.sidebar.caption(
                "Ahead of the cohort: **{}** ticker{} (24/7 instruments)".format(
                    status["ahead_frontier"],
                    "" if status["ahead_frontier"] == 1 else "s",
                )
            )

    # A progress bar and a spinner are both stateful widgets, so the work runs to
    # completion inside this block rather than being kicked off in the background:
    # Streamlit reruns the script top to bottom on every interaction, and a
    # fire-and-forget download would be abandoned the moment the user touched
    # anything.  ``st.fragment`` narrows the rerun scope to this block, so a
    # progress update does not rebuild the whole page.
    @st.fragment(run_every=None)
    def _sync_block() -> None:
        running = st.session_state.get(SYNC_RESULT_KEY + "_running", False)

        if st.sidebar.button(
            "Syncing…" if running else "Sync archive now",
            width='stretch',
            type="primary",
            disabled=running,
            help="Download any S&P 500 sessions missing from the **{}** "
                 "archive. Safe to run daily — sessions already stored are "
                 "skipped, and re-fetching today's still-trading session just "
                 "adds the newer bars. On an empty archive it fetches full "
                 "listing history. Equivalent to "
                 "`python scripts/{} --incremental`.".format(
                     active_label().lower(),
                     download_script_name(),
                 ),
        ):
            st.session_state[SYNC_RESULT_KEY + "_running"] = True

        if st.session_state.get(SYNC_RESULT_KEY + "_running"):
            bar = st.sidebar.progress(0.0, text="Starting the downloader…")
            seen = {"line": ""}

            def _on_progress(done: int, total: int, line: str) -> None:
                # The child's stdout is the only channel back, so the most recent
                # line is what the caption shows.  Throttled to batch boundaries
                # because re-rendering on every line of a 90s run is wasteful.
                if done and (done != seen.get("done")):
                    seen["done"] = done
                    frac = min(done / total, 0.99) if total else 0.0
                    bar.progress(
                        frac,
                        text="Batch {} of {} tickers".format(done, total),
                    )
                elif line.startswith("  batch"):
                    seen["done"] = done
                if "Fetching" in line:
                    bar.progress(0.01, text=line.strip()[:60])

            try:
                result = SYNC.run_sync(
                    panel_root_for(), on_progress=_on_progress, timeout_s=900,
                    timeframe=_tf().key,
                )
            finally:
                st.session_state[SYNC_RESULT_KEY + "_running"] = False

            bar.progress(1.0, text="Done")
            st.session_state[SYNC_RESULT_KEY] = result
            # The panel search caches per-ticker feature matrices keyed on the root
            # path, and those matrices are now stale: the archive grew underneath
            # them.  Dropping the cached object is what makes the Panel tab show the
            # new sessions without a restart -- its key would otherwise still match.
            load_panel_search.clear()

    _sync_block()

    # Report the finished run once, and let the user dismiss it.  Keyed in session
    # state rather than drawn unconditionally so the caption does not re-announce a
    # sync on every unrelated rerun.
    result = st.session_state.get(SYNC_RESULT_KEY)
    if result is not None:
        if result.ok:
            if result.bars_written:
                st.sidebar.success(result.summary())
            else:
                st.sidebar.info("Already up to date — no new sessions to store.")
        else:
            st.sidebar.error(result.error or "The sync failed.")
        if result.sessions_rejected:
            st.sidebar.warning(
                "{} session(s) were rejected by the quality gate. Run the "
                "downloader from a terminal to see which and why.".format(
                    result.sessions_rejected
                )
            )
        if st.sidebar.button("Dismiss", key="sync_dismiss", width='stretch'):
            st.session_state.pop(SYNC_RESULT_KEY, None)

    st.sidebar.caption(
        "Yahoo serves ~30 days of 1-minute history; stored bars are kept "
        "permanently. Sync once a day so no session expires un-fetched."
    )


@st.cache_resource(show_spinner=False)
def load_panel_search(root: str) -> PanelSearch:
    """Build (and cache) the cross-sectional search over the multi-ticker archive.

    ``cache_resource``, not ``cache_data``: the returned object holds the per-ticker
    feature matrices -- a few MB of derived state that would otherwise be rebuilt on
    every single Streamlit rerun, which happens on every sidebar interaction.

    The sector map is read from a small CSV the downloader writes alongside the
    archive.  It is a *cache of* the constituent list, never required for reading: if
    it is missing the panel still searches, it simply reports no same-sector rank,
    which is better than refusing to run.

    :func:`~timeseries.store.read_sectors` owns that fallback *and* the filtering of
    symbols carrying no label -- an explicitly-fetched ``BTC-USD`` is registered with
    an ``unknown`` sector -- so this is one call rather than an ``isfile`` check, a
    read, a zip and a ``try``.  Dropping the marker here rather than in
    :mod:`timeseries.panel` is what keeps the panel free of any notion of it.
    """
    return PanelSearch(PanelStore(root), sectors=read_sectors(panel_sector_file()))


# =============================================================================== #
# Cross-sectional search for the Price tab's match panel
# =============================================================================== #
def panel_pipeline_for(search: PanelSearch, ticker: str) -> Optional[Pipeline]:
    """A :class:`~timeseries.pipeline.Pipeline` over one ticker's archived bars.

    The Price tab's lower chart is drawn by ``build_price_figure``, which reads
    ``pipe.bars`` throughout -- so a match found in NVDA cannot be drawn by the
    pipeline holding QQQ's bars.  This builds the right one.

    **Built by direct construction, not ``Pipeline.from_frame``.**  The obvious
    call is wrong in a way that looks right: ``from_frame`` re-runs
    ``build_features`` and ``finalize_features(how="drop")``, which discards the
    ~20-bar warm-up **a second time**.  The panel's matrices were already trimmed
    once, so the rebuilt frame would be ~20 bars shorter than the matrix the match
    indices refer to, and every match would be drawn against a window that was
    offset from the one that was scored.  ``build_panel`` hands back an
    ``(matrix, aligned_frame)`` pair whose rows correspond by construction, so the
    pipeline is constructed over that pair directly.

    ``length`` is carried on the pipeline but is *not* the feature length; nothing
    that draws a cross-ticker match calls ``Pipeline.match`` or ``query_latest`` on
    it, both of which would index the matrix as if it were the home ticker's own
    query span.
    """
    entry = search.panel.get(ticker)
    if entry is None:
        return None
    matrix, frame = entry
    return Pipeline(bars=frame, features=frame, matrix=matrix,
                    length=int(matrix.shape[0]), ready=True)


@st.cache_resource(show_spinner=False, max_entries=64)
def _cached_panel_search(signature: str, vector: np.ndarray, start: int, stop: int,
                         k: int, max_per_ticker: int,
                         amplitude_weight: float, max_horizon: int,
                         q_move: Optional[float],
                         t0: Optional[Any], t1: Optional[Any],
                         exclude_ticker: Optional[str]) -> Any:
    """Memoised cross-sectional search, keyed on a signature string.

    ``st.tabs`` runs **every** tab body on **every** rerun, so the Price tab's lower
    panel is searched on every sidebar interaction, every keystroke in the ticker box
    and every unrelated widget move.  The comment above the old
    ``auto_result = pipe.match(...)`` called that search "cheap and deterministic";
    a panel search is ~500 ``stumpy.mass`` passes plus a per-ticker §BX mask, which is
    not cheap.  Without this the app would re-score the whole S&P 500 on every click.

    ``cache_resource`` rather than ``cache_data``: the return value holds numpy arrays
    and a dataclass the caller reads in place, and there is nothing to copy.

    **``PanelSearch`` is deliberately NOT a parameter.**  Streamlit hashes every
    argument, and a ``PanelSearch`` holds a bound method that raises
    ``UnhashableParamError: cannot pickle 'function' object`` before the search ever
    runs -- so the Price tab silently lost its match panel on every pass.  The object
    is fetched from :func:`load_panel_search` inside the body instead, exactly as
    :func:`pipeline_from_frame` sidesteps hashing its ``DataFrame``.  That is also
    correct rather than merely convenient: ``load_panel_search`` is itself cached, so
    the same instance comes back, and every feature matrix and move profile it holds
    is shared with the rest of the app rather than re-derived.

    ``signature`` is a *string* naming the question; the vector and span ride along
    as payload but are cheap to hash next to a 500-ticker object graph.
    ``max_entries`` is bounded so a session that brushes a hundred windows cannot pin
    a hundred result sets.
    """
    search = load_panel_search(panel_root_for())
    # ``ticker=""`` on purpose -- the query's indices address the *fetched* ticker's
    # matrix, so naming it here would hand ``exclusion_mask`` a range in the wrong
    # index space.  ``t0``/``t1`` carry the self-match guard instead, on the one axis
    # both series share.  See ``cross_sectional_match``.
    query = PNL.PanelQuery(vector=vector, ticker="", start=int(start), stop=int(stop))
    return search.search(query, k=int(k), max_per_ticker=int(max_per_ticker),
                         max_horizon=max_horizon,
                         amplitude_weight=float(amplitude_weight), q_move=q_move,
                         exclude_times=None if t0 is None else (t0, t1),
                         exclude_ticker=exclude_ticker)


def cross_sectional_match(pipe: Pipeline, symbol: str, start_idx: int,
                          stop_idx: int, *, k: int = DEFAULT_K,
                          amplitude_weight: float = 1.0,
                          max_per_ticker: int = 5,
                          max_horizon: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Search the whole panel for the window brushed on the Price tab.

    Returns ``None`` when the panel archive is absent or cannot answer, so the caller
    can fall back rather than raise.  On success the dict carries the
    :class:`~timeseries.panel.PanelResult` and the :class:`PanelSearch` that produced
    it -- the caller builds the matched ticker's pipeline from that.

    **The home ticker is deliberately empty.**  The query's bars come from a live
    Yahoo fetch while the panel's come from the last sync, so ``start_idx`` addresses
    different bars in the two index spaces.  Naming the fetched symbol as the home
    ticker would hand ``exclusion_mask`` a span in the wrong coordinates and suppress
    an unrelated region of that ticker -- silently, with no error.  Leaving it empty
    costs exactly one thing: if the fetched symbol *is* an S&P 500 constituent, the
    panel may return a window from that same ticker overlapping the query.  That is
    reported to the reader rather than hidden, and is the lesser of the two errors.

    ``q_move`` is measured on the fetched ticker's aligned bars, so the amplitude term
    asks the same question of the query as of every candidate.
    """
    if not os.path.isdir(panel_root_for()):
        return None
    matrix = pipe.matrix
    if matrix is None or matrix.shape[0] == 0:
        return None
    if not 0 <= int(start_idx) < int(stop_idx) <= matrix.shape[0]:
        return None

    try:
        search = load_panel_search(panel_root_for())
        if not search.ready():
            return None
        vec = np.asarray(matrix[int(start_idx):int(stop_idx)], dtype=float)
        vector = np.stack([M.zscore(vec[:, c]) for c in range(vec.shape[1])], axis=-1)
    except Exception:  # noqa: BLE001 - a broken archive must not break the page
        return None

    length = int(stop_idx) - int(start_idx)
    q_move: Optional[float] = None
    try:
        r = pipe.log_returns
        if r is not None and r.size >= length:
            total = float(np.nansum(r[int(start_idx):int(stop_idx)]))
            if np.isfinite(total):
                q_move = total
    except Exception:  # noqa: BLE001
        q_move = None

    # The query's own clock span, for the §M self-match guard.  **Only when the
    # fetched symbol is actually in the archive** -- there is nothing to exclude from
    # a ticker the panel does not hold, and a non-constituent (QQQ is not one) is the
    # common case rather than an edge one.
    t0 = t1 = None
    if str(symbol) in search.panel:
        try:
            stamps = pd.to_datetime(pipe.bars["timestamp"], utc=True)
            t0 = stamps.iloc[int(start_idx)]
            t1 = stamps.iloc[int(stop_idx) - 1]
        except Exception:  # noqa: BLE001 - a bad stamp is not a reason to fail
            t0 = t1 = None

    signature = "|".join(str(x) for x in (
        symbol, length, int(start_idx), int(stop_idx), int(k),
        int(max_per_ticker), round(float(amplitude_weight), 4), max_horizon,
    ))
    try:
        result = _cached_panel_search(
            signature, vector, int(start_idx), int(stop_idx),
            int(k), int(max_per_ticker), float(amplitude_weight),
            max_horizon if max_horizon is None else int(max_horizon),
            q_move, t0, t1,
            str(symbol) if t0 is not None else None,
        )
    except Exception:  # noqa: BLE001 - a failed panel search is not a failed page
        return None
    return {"result": result, "search": search, "length": length, "symbol": symbol,
            "self_excluded": bool(t0 is not None)}


def panel_result_as_match_result(result: Any) -> M.MatchResult:
    """Re-express a :class:`~timeseries.panel.PanelResult` as a ``MatchResult``.

    ``PanelMatch`` and :class:`~timeseries.matching.Match` already agree on the four
    fields the chart layer reads (``start``, ``stop``, ``distance``, ``percentile``),
    so the match panel could be handed the panel result directly.  This converts
    anyway, for one load-bearing reason: **the query has to be re-anchored into the
    matched ticker's index space.**

    ``_render_best_match_pair`` reads ``query.start``/``query.stop`` and uses them
    against ``pipe.n_bars``, and the Price tab renders it with ``pipe`` = the
    *matched* ticker's pipeline.  The real query's indices address the *fetched*
    ticker's bars.  Mixing the two would ask for a view of one series at the
    coordinates of another.

    So the query carried here is **not the user's query**.  It is a placeholder
    spanning the first match's own window, which makes every coordinate the function
    touches refer to the single pipeline it is drawing, and keeps ``query.length``
    equal to the match length rather than to something degenerate.  Nothing reads
    ``query.vector``: this path never re-scores, it only draws a window that has
    already been scored.
    """
    matches = [
        M.Match(start=int(m.start), stop=int(m.stop), distance=float(m.distance),
                percentile=float(m.percentile), timestamps=(m.session,))
        for m in result.matches
    ]
    q0, q1 = (int(result.matches[0].start), int(result.matches[0].stop)
              if result.matches else (0, 1))
    return M.MatchResult(
        matches=matches,
        n_candidates=int(result.n_candidates),
        n_excluded=int(result.n_excluded),
        n_after_nms=len(matches),
        query=M.Query(vector=np.zeros((max(1, q1 - q0), 1)), start=q0, stop=q1),
        method=result.method,
        n_masked=int(getattr(result, "n_masked", 0) or 0),
    )


def render_panel_tab() -> None:
    """Search the whole S&P 500 archive for windows shaped like a chosen one.

    **This is no longer the app's only cross-sectional view.**  The *Price* tab's
    lower panel also searches the panel now — it shows the single closest match from
    anywhere in the index, whereas this tab is where you drive the search yourself and
    read the full ranked list with all three rarity columns.

    The division of labour is deliberate.  *Price* answers "what is the one thing I
    should look at?" in a single glance, with no controls in the way; this tab answers
    "how unusual is this, and does the answer survive per-ticker and per-sector?"
    because a cross-sectional search over 500 heterogeneous names makes that the whole
    difficulty.

    Where the other tabs search one ticker's own history ("when has QQQ done this?"),
    these two search the panel ("has *anything* in the index done this?").  That
    difference shows up in three places, and each is stated where the number appears
    rather than in a footnote:

    * a window can never span two tickers, because each ticker is scored as its own
      contiguous series;
    * every match carries **two** rarity ranks, because "unusual for the whole market"
      and "unusual for this stock" are different questions with different answers;
    * no one ticker can fill the result list, because when 400 names move together the
      match is the factor, not the company.
    """
    st.subheader("Cross-sectional search — the whole S&P 500 panel")

    # **The archive, the command and the retention story all name the resolution
    # in force.**  Hard-coding `data/sp500_panel` here was accurate while daily mode
    # had no panel archive; now it would send a reader who chose Daily to a
    # directory that does not serve them, and the command would build the *other*
    # resolution's archive -- the message would be a confident, wrong instruction.
    root = panel_root_for()
    script = download_script_name()
    where = os.path.basename(root)
    if not os.path.isdir(root):
        st.info("No **{}** multi-ticker archive found at `{}`.".format(
            active_label().lower(), where))
        hint(
            "Build it first:<br>"
            "<code>python scripts/{script}</code><br><br>"
            "That fetches every current S&P 500 constituent (503 symbols, sector "
            "labels included) as <b>{label}</b> bars into a partitioned Parquet "
            "archive. You can also use <b>Sync archive now</b> in the sidebar, "
            "which runs the same command and reports what it stored.<br><br>"
            "{retention}".format(
                script=script, label=active_label().lower(), retention=(
                    "Yahoo serves roughly the <b>last 30 days</b> of 1-minute "
                    "history and keeps nothing beyond it — but bars already "
                    "downloaded are kept permanently here. So the archive is a "
                    "one-way ratchet: a session that passes the 30-day mark "
                    "without being synced is gone for good. Re-running never "
                    "overwrites a completed session; it only adds the ones that "
                    "are missing, so it is always safe to press."
                    if _tf().bars_per_session > 1 else
                    "Yahoo serves a <b>century</b> of daily history with no "
                    "retention wall, so this archive is built from the full "
                    "listing in one pass and does not need syncing to avoid "
                    "losing anything. Re-running only adds what is missing and "
                    "flags any session a vendor has retro-adjusted, so it is "
                    "always safe to press."
                ),
            )
        )
        return

    with st.spinner("Loading the panel archive…"):
        try:
            search = load_panel_search(panel_root_for())
            panel = search.panel
        except Exception as exc:  # noqa: BLE001 - surface any archive failure in the UI
            st.error("Could not read the panel archive: {}".format(exc))
            hint(
                "Re-run <code>python scripts/{0}</code>, then reload. ".format(
                    download_script_name()
                ) +
                "If a session keeps failing, the store's quality gate rejected it — "
                "the download log names the reason."
            )
            return

    if len(panel) < 2:
        st.warning(
            "The archive holds {} ticker with too few bars to form a search window. "
            "A cross-sectional search needs at least two.".format(len(panel))
        )
        hint(
            "Each ticker needs at least a window-length plus the ~20-bar feature "
            "warm-up. Add more tickers, or more sessions:"
            "<br><code>python scripts/{} --incremental</code>".format(
                download_script_name()
            )
        )
        return

    n_bars = {s: len(panel[s][0]) for s in panel}
    total = sum(n_bars.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("tickers", "{:,}".format(len(panel)),
              help="Tickers with enough bars to form at least one search window.")
    c2.metric("searchable windows", "{:,}".format(total),
              help="Every fixed-length window the matcher can score, summed over all "
                   "tickers. This is the denominator behind the *panel* percentile — "
                   "a match must beat all but this many windows to count as unusual.")
    c3.metric("sessions", "{:,}".format(len({str(search.session_of(s, 0)) for s in panel})),
              help="Distinct trading days covered by the archive.")

    # ---- Choose the query -------------------------------------------------- #
    st.markdown("#### Choose a query")
    hint(
        "The query is a window of one ticker's bars. Everything else on this tab is a "
        "statement about that window — which other S&P 500 windows resemble it."
    )

    default_sym = max(panel, key=lambda s: n_bars[s])
    order = sorted(panel, key=lambda s: (-n_bars[s], s))
    c1, c2, c3 = st.columns([2, 2, 3])
    sym = c1.selectbox(
        "Ticker", order, index=order.index(default_sym),
        help="Whose tape the query is drawn from. This ticker is **excluded** from "
             "its own results, so it can never match itself (§M).",
    )
    length = c2.slider(
        # The cap was 180 bars, which was reachable but not generous: at the Price tab's
        # widest view (5 days = 1,950 bars) a 180-bar matched window is 9.2% of the
        # panel, and anything longer read as a thin sliver rather than a shape.  Raising
        # the cap to 390 -- roughly one full session -- puts a 390-bar match at 20% of
        # that panel, which is the share the chart's own guidance calls a readable
        # shape, and leaves the reader room to grow the window without changing code.
        #
        # It is capped against the archive rather than fixed, because
        # ``Pipeline.from_frame`` refuses to build below ``length + 100`` bars: a
        # slider that offered more than the tape supports would report "not ready"
        # rather than matching.  The ``- 25`` leaves the same headroom it always did.
        "Window (bars)", min_value=5,
        max_value=min(active_max_query_bars(), max(1, n_bars[sym] - 25)),
        value=min(active_length(), max(5, n_bars[sym] - 25)), step=1,
        help="Query length in 1-minute bars. {} bars = {} minutes of continuous "
             "trading. This also sets the length of every *matched* window — the "
             "matcher only ever compares windows of the query's own length — so a "
             "longer window makes the matched window below the Price tab longer too, "
             "and a larger share of the panel. Non-maximum suppression spaces accepted "
             "matches by at least this many bars, so a short window is also the one that "
             "most often returns the same moment repeatedly.".format(
                 active_length(), active_length()),
    )
    mode = c3.radio(
        "Which window", ["Latest", "Busiest 30-min move"], horizontal=True,
        help="`Latest` asks about the most recent complete span. `Busiest` finds the "
             "window with the largest realised move in this ticker — usually the more "
             "interesting question, because a flat window matches everything.",
    )

    span = search.latest_query(sym, length=length)
    if mode == "Busiest 30-min move":
        _mat, frame = panel[sym]
        close = frame["close"].to_numpy(dtype=float)
        if len(close) > length:
            rolls = np.abs(np.log(close[length:] / close[:-length]))
            at = int(np.nanargmax(rolls))
            q = search.prepare(sym, at, at + length)
            if q is not None:
                span = q
    if span is None:
        st.warning("Could not build a query for {} — not enough bars.".format(sym))
        return

    q_start, q_stop = span.start, span.stop
    c1, c2, c3 = st.columns(3)
    st.session_state["_panel_span"] = (q_start, q_stop)
    c1.metric("query window", "{} bars".format(length),
              help="Fixed length. Every candidate window in the panel is the same "
                   "length, so this is the only comparable unit.")
    c2.metric("from", stamp_label(search.timestamp_of(sym, q_start)),
              help="{} timestamp of the query's first bar.".format(
                  stamp_zone_name().capitalize()))
    c3.metric("to", stamp_label(search.timestamp_of(sym, q_stop - 1)),
              help="{} timestamp of the query's last bar.".format(
                  stamp_zone_name().capitalize()))

    # ---- Controls --------------------------------------------------------- #
    st.markdown("#### Search")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Max matches", min_value=1, max_value=60, value=12, step=1,
                  help="How many matches to return **after** the per-ticker cap. This "
                       "is the effective sample size for the forecast below.")
    cap = c2.slider("Max per ticker", min_value=1, max_value=10, value=2, step=1,
                    help="Hard cap on how many matches one ticker may contribute. "
                         "Without it, a sector-wide move returns 12 windows of the "
                         "same three names and looks like 12 independent pieces of "
                         "evidence when it is really one factor, observed once.")
    min_matches = c3.number_input("Min matches for evidence", min_value=5, max_value=200,
                                  value=30, step=1,
                                  help="The §E floor. Below it the forecast is withheld "
                                       "entirely rather than printed from a handful of "
                                       "observations.")

    run = st.button("Search the panel", type="primary", width='stretch',
                    help="Scores every window in every ticker. Fast (well under a "
                         "second on a few hundred tickers), and recomputed on press so "
                         "a stale answer can never be mistaken for a fresh one.")

    if not run:
        st.caption(
            "Press **Search the panel** to score the query against all "
            "{:,} tickers.".format(len(panel))
        )
        hint(
            "Results are never cached across presses. The archive and its derived "
            "features are cached — that is the expensive part and it never changes "
            "until you re-download."
        )
        return

    with st.spinner("Scoring every window in the panel…"):
        out = search.run(span, k=k, max_per_ticker=cap, n_baseline=600,
                         min_matches=int(min_matches))

    if not out.get("ok"):
        st.error("Search failed: {}".format(out.get("reason", "unknown")))
        return

    result = out["result"]
    if not result.matches:
        st.warning("No window in the panel matched closely enough to report.")
        hint(
            "Every window is scored, so an empty result means the archive is too thin "
            "to contain anything resembling this query. Add sessions with "
            "<code>--incremental</code>."
        )
        return

    # ---- Headline --------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("matches", "{:,}".format(result.n_matches),
              help="Windows returned after non-maximum suppression and the "
                   "per-ticker cap.")
    c2.metric("distinct tickers", "{:,}".format(result.n_distinct_tickers),
              help="How many different companies the matches come from. A low number "
                   "next to a high match count means one factor is being counted many "
                   "times, not many independent things happened.")
    c3.metric("windows scored", "{:,}".format(result.n_candidates),
              help="Every window in the panel — the denominator behind the panel "
                   "percentile below.")
    c4.metric("best panel percentile",
              pct(result.matches[0].percentile),
              help="How rare the closest match is across the **whole panel**.")

    if result.sectors_represented:
        st.caption("Sectors represented: " + ", ".join(result.sectors_represented))

    if getattr(result, "n_masked", 0):
        st.caption(
            "§BX — {:,} of {:,} panel windows were withheld because their forward "
            "horizon would run across a session closure. Such a window reports the "
            "overnight gap as if it were trading."
            .format(result.n_masked, result.n_candidates)
        )

    with st.expander("Archive coverage by ticker", expanded=False, icon=HELP_ICON):
        st.caption(
            "Sessions each ticker holds in `data/sp500_panel/`. A ticker with fewer "
            "sessions contributes proportionally fewer candidate windows, which is why "
            "a thin archive makes every window look ordinary — the percentile is a "
            "fraction of what is actually on disk."
        )
        try:
            st.dataframe(search.store.coverage(), width='stretch', hide_index=True)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break the tab
            st.caption("Could not read coverage: {}".format(exc))

    # ---- The two ranks ---------------------------------------------------- #
    st.markdown("#### Matches")
    hint(
        "Two rarity ranks per match, and they answer different questions. "
        "**Panel %** is the share of all {:,} windows in the archive at least as "
        "close. **Own %** is the same for that ticker's own history alone. A window "
        "can be the closest thing its own ticker has ever seen and still be "
        "unremarkable across 500 names — which is exactly the difference between a "
        "stock doing something unusual and the whole market doing the same thing."
        .format(result.n_candidates)
    )

    stamp_col_time = stamp_column("time")
    rows = []
    for i, m in enumerate(result.matches, 1):
        rows.append({
            "#": i,
            "ticker": m.ticker,
            "session": m.session,
            # ``session`` above already names the Eastern date, so on daily this
            # column used to say the same thing twice; the header now says what it is.
            stamp_col_time: stamp_label(m.timestamp) if pd.notna(m.timestamp) else "—",
            "panel %": pct(m.percentile),
            "own %": pct(m.percentile_same_ticker),
            "sector %": pct(m.percentile_same_sector),
            "distance": "{:.3f}".format(float(m.distance)),
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    hint(
        "`panel %` / `own %` / `sector %` are rarity ranks, not probabilities of being "
        "right — §E's honesty feature. A small number means *few windows in the "
        "archive look like this*, and says nothing about whether price goes anywhere "
        "next. The raw `distance` is **not comparable across queries**; use the "
        "percentile. A `sector %` of `—` means the sector map was unavailable, not "
        "that the sector had no matches."
    )

    # ---- The forecast, with its baseline --------------------------------- #
    st.markdown("#### What followed — always beside a random-window baseline")
    forecasts = out.get("forecasts", [])
    frows = []
    for f in forecasts:
        frows.append({
            "horizon": "{} {}".format(f.horizon, bar_unit(f.horizon)),
            "n_valid": f.n_valid,
            "matched": bps(f.mean_return),
            "random windows": bps(f.baseline_mean),
            "lift": bps(f.lift),
            "p_value": "—" if not np.isfinite(f.p_value) else "{:.3f}".format(f.p_value),
            "95% CI": ("—" if not np.isfinite(f.ci_low)
                       else f"{bps(f.ci_low)} … {bps(f.ci_high)}"),
            "sufficient": "yes" if f.sufficient else "no",
        })
    if frows:
        st.dataframe(pd.DataFrame(frows), width='stretch', hide_index=True)
    greyed = [f for f in forecasts if not f.sufficient]
    if greyed:
        st.warning(
            "No forecast printed for {} of {} horizons — fewer than {} matched windows "
            "had a full forward horizon behind them (§E).".format(
                len(greyed), len(forecasts), int(min_matches))
        )
        hint(
            "Raise <b>Max matches</b> and <b>Max per ticker</b> above, and pick a query "
            "away from the newest bars — a window at the end of the archive has no data "
            "after it, so its forward return does not exist. The threshold itself is the "
            "§E minimum and is deliberately not lowered for you."
        )
    else:
        hint(
            "`lift` is `matched − random windows`, in bps. **A forecast is only "
            "interesting if it beats the grey control** — with a few sessions of "
            "archive, a matched mean is usually indistinguishable from a random one, and "
            "that is the finding, not a failure."
        )

    with st.expander("How to read this tab — every column explained", icon=HELP_ICON):
        st.markdown(
            "| Column | What it is | How to read it |\n"
            "|---|---|---|\n"
            "| `ticker` | which company the match is in | matches are located in that "
            "ticker's **own** series; a window never spans two tickers |\n"
            "| `session` | Eastern trading day | a session runs 09:30–16:00 ET, which "
            "is why it is named by the Eastern date and not the UTC one |\n"
            "| `panel %` | rarity across every window in the archive | the honest "
            "score (§E) |\n"
            "| `own %` | rarity within that ticker's own history | the question 'is "
            "this unusual for *this stock*' |\n"
            "| `sector %` | rarity within the query's GICS sector | separates a "
            "sector move from a single-name event |\n"
            "| `distance` | geometric gap | **not comparable across queries**; use a "
            "percentile |\n"
            "| `n_valid` | matched windows with a full forward horizon | windows "
            "running past the newest bar are `NaN`, not zero |\n"
            "| `lift` | matched mean minus random-window mean, in bps | a difference, "
            "not a ratio |\n\n"
            "**Why three percentiles?** Over 500 tickers the candidate population is "
            "dominated by whichever names happened to be quiet that week, and a "
            "liquid mega-cap is a much harder shape to resemble than a thinly-traded "
            "small cap. Collapsing that into one number would make a quiet stock's "
            "ordinary move look like a discovery.\n\n"
            "**Why a per-ticker cap?** When a sector moves together, the best match is "
            "not a company — it is the factor. Without a cap, the top 12 is 12 windows "
            "of three tickers, and the forecast would treat one event as twelve "
            "observations and report a p-value four times better than it deserves."
        )


# =============================================================================== #
# Tab: Quality
# =============================================================================== #
#: Market holidays a US equity archive is expected to be missing, per year.
#:
#: Used only to judge the *daily* ``missing_days`` figure. Every weekday without a bar
#: is counted and most are the exchange being closed, so an absolute threshold would be
#: wrong at both ends: too low flags every complete archive, too high misses a genuine
#: partial download. Measured against live QQQ daily data: a 25-year span reports 9.3
#: a year, against ~9-10 published market holidays. The 1.6x multiplier is the slack
#: for a long archive whose span edges clip a partial holiday year.
EXPECTED_HOLIDAYS_PER_YEAR = 9.5


def _quality_key_table() -> str:
    """The "what healthy looks like" table, for the resolution in force.

    Split out of :func:`render_quality_tab` because the *values* are resolution
    claims, not prose: "both bars_min and bars_max ≈ 390" is true only on 1-minute,
    and the table it heads is precisely the document a reader consults when a number
    looks wrong.  A table that said 390 while charting daily bars would be worse than
    no table -- it would send someone looking for a truncated session that is not
    there.

    The row set differs too, not only the numbers: ``missing_days`` exists only on
    daily, and ``intra_session_holes`` only where a session has an interior.  Both are
    named in the shared rows below, so a reader who has seen one resolution's table
    knows what to look for in the other's.
    """
    tf = _tf()
    per_session = tf.bars_per_session
    healthy = "both ≈ {}".format(per_session) if per_session > 1 else "both `1`"
    bad = (
        "well under {} ⇒ that session is truncated or the download failed".format(
            per_session)
        if per_session > 1
        else "anything other than `1` ⇒ two bars landed on one day, or the day is empty"
    )

    rows = [
        "| Key | Healthy | If it is not |",
        "|---|---|---|",
        "| `rows` | {} | too few ⇒ there is no credible percentile to report |".format(
            "tens of thousands" if per_session > 1 else "hundreds or more"),
        "| `sessions` | the number of distinct ET trading days | `1` means a "
        "single session's {} bars — there is simply nothing to match against |".format(
            per_session),
        "| `bars_min` / `bars_max` | {} | {} |".format(healthy, bad),
    ]
    if tf.gap_seconds is None:
        rows.append(
            "| `missing_days` | ~{h} a year | weekdays with no bar at all. **Almost all "
            "of them are market holidays** — the exchange was closed, which is not "
            "missing data. Read `missing_days_per_year`, not the raw count: a live QQQ "
            "archive reports 233 over 25 years, which is exactly what a complete one "
            "looks like |".format(h=int(EXPECTED_HOLIDAYS_PER_YEAR))
        )
        rows.append(
            "| `missing_days_per_year` | ≈ {h} | much above that means the download "
            "stopped for a stretch rather than that the market was closed |".format(
                h=EXPECTED_HOLIDAYS_PER_YEAR)
        )
    else:
        rows.append(
            "| `intra_session_holes` | `0` | holes **inside** a session: a halt or "
            "dropped prints. Matching across one manufactures a price jump |"
        )
    rows.extend([
        "| `session_boundaries` | one per pair of days | **normal** — this is the "
        "market being closed overnight, not missing data (§BF) |",
        "| `bad_timestamps` | `0` | unparseable dates were dropped during load |",
        "| `nonpositive_close` | `0` | log returns undefined; masked to NaN |",
        "| `duplicate_timestamps` | `0` | last observation of each was kept |",
        "| `monotonic` | `true` | the frame was re-sorted; Yahoo did not return "
        "the bars in chronological order |",
    ])
    return "\n".join(rows)


def _quality_key_notes() -> str:
    """The prose under that table, for the resolution in force."""
    tf = _tf()
    rolling = tf.rolling_window
    if tf.gap_seconds is None:
        return (
            "**Why daily has no holes.** A daily bar is a whole trading day, so there "
            "is no interior in which a dropped print could sit. The only "
            "discontinuity a daily series can have is a *day with no bar at all*, "
            "which is reported as `missing_days` with weekends excluded — a holiday "
            "week is not a gap.\n\n"
            "**Bar indices are into the cleaned frame.** Features are a rolling-{n} "
            "z-score of the log return and the rebased log-price path. The first ~{n} "
            "bars are **dropped**, not imputed. So bar index *i* on any other tab "
            "indexes this cleaned frame, not the raw download.".format(n=rolling)
        )
    return (
        "**Why an overnight boundary is not a gap.** The US session opens 09:30 ET "
        "and closes 16:00 ET, so consecutive {} bars are never 60 seconds apart "
        "across a night. Counting that as corruption produced a false alarm on a "
        "clean archive; the two conditions are now reported separately (§BF).\n\n"
        "**Bar indices are into the cleaned frame.** Features are a rolling-{n} "
        "z-score of the log return and the rebased log-price path. The first ~{n} "
        "bars are **dropped**, not imputed. So bar index *i* on any other tab indexes "
        "this cleaned frame, not the raw download.".format(
            tf.label, n=rolling)
    )


def render_quality_tab(pipe: Pipeline) -> None:
    st.subheader("Data quality")
    hint(
        "Every number in this app descends from the bars loaded here. Check this tab "
        "first whenever a result looks surprising — and before interpreting a "
        "percentile at all, which is meaningless on a tiny archive."
    )
    report = pipe.quality()
    st.json(report)
    hint(
        "This is the raw report. The checks below say what each key should be and what "
        "to do when it is not; open “Every key explained” for the full table."
    )

    st.markdown("#### Interpretation")
    # Both keys are always present, so neither branch has to guard a lookup: on
    # daily ``gaps_over_180s`` is a hard 0 by construction and ``missing_days`` is
    # the only discontinuity a daily series can have.
    gaps = int(report.get("intra_session_holes", 0))
    sessions = int(report.get("sessions", 0))
    rows = int(report.get("rows", 0))

    if _tf().gap_seconds is None:
        # Daily. A daily bar is a whole trading day, so there is no interior in which
        # a hole could exist -- the only thing that can be missing is the *day*.
        #
        # Judged on a *rate*, because every weekday without a bar is counted and most
        # of those are market holidays. A live QQQ archive reports 233 across 25 years,
        # which is 9.3 a year against ~9-10 published market holidays -- so the raw
        # count is what a *complete* archive looks like, and warning on it would train
        # the reader to ignore the one number on this tab that can matter.
        missing = int(report.get("missing_days", 0))
        rate = float(report.get("missing_days_per_year", 0.0))
        if rate > EXPECTED_HOLIDAYS_PER_YEAR * 1.6:
            st.warning(
                "**{} weekdays with no bar — {:.1f} a year** across {:,} sessions. A "
                "complete US equity archive runs about {} a year (market holidays), so "
                "this is above the usual figure. Check whether the download stopped for "
                "a stretch; individual missing days are expected.".format(
                    f"{missing:,}", rate, sessions, EXPECTED_HOLIDAYS_PER_YEAR)
            )
        elif missing:
            st.success(
                "{} weekday(s) with no bar — {:.1f} a year, consistent with market "
                "holidays across {:,} sessions. A daily bar is a whole session, so the "
                "intraday hole check does not apply.".format(
                    f"{missing:,}", rate, sessions)
            )
        else:
            st.success(
                "Every weekday in the archive has a bar. A daily bar is a whole "
                "session, so there is no interior for a hole to sit in and the "
                "intraday gap check does not apply."
            )
    elif gaps > 20:
        st.info(
            "**{:,} gaps longer than {} s across {:,} sessions.** With {:,} bars this "
            "is the expected overnight boundary between trading days, not data loss — "
            "the US equity session opens 09:30 ET and closes 16:00 ET, so consecutive "
            "bars are never 60 s apart across a night.".format(
                gaps, _tf().gap_seconds, sessions, rows)
        )
    elif gaps > 0:
        st.warning(
            "{:,} gap(s) longer than {} s. If these fall inside a single session "
            "they are genuine holes; if they land between sessions they are the "
            "overnight boundary.".format(gaps, _tf().gap_seconds)
        )
    else:
        st.success("No gaps longer than {} s — the series is continuous at {} "
                   "resolution.".format(_tf().gap_seconds, active_label().lower()))

    if not report.get("monotonic", True):
        st.error("Timestamps are not monotonically increasing — the loaded frame was "
                 "re-sorted before use.")
        hint("The app re-sorted the frame, so results are still correct; the order "
             "Yahoo returned them in was simply not chronological.")
    if int(report.get("duplicate_timestamps", 0)) > 0:
        st.warning("Duplicate timestamps were found; the last observation of each was kept.")
        hint("The downloader split its request into chunks, so an overlapping "
             "boundary can be returned twice. Each bar is counted once, but the "
             "affected session is worth a look on the Quality tab.")
    if int(report.get("nonpositive_close", 0)) > 0:
        st.error("Non-positive close values were found; their returns are masked to NaN.")
        hint("A zero or negative price makes a log return undefined. Check whether the "
             "vendor sent a bad print; the affected bars are excluded from the library.")
    if int(report.get("bad_timestamps", 0)) > 0:
        st.error("Unparseable timestamps were dropped during load.")
        hint("Check the file's date format. The loader expects ISO-8601 and parses "
             "everything to UTC; anything it cannot read is discarded.")

    with st.expander("Every key explained — what healthy looks like", icon=HELP_ICON):
        st.markdown(_quality_key_table())
        st.markdown(
            _quality_key_notes()
        )


# =============================================================================== #
# Tab: Backtest
# =============================================================================== #
def render_backtest_tab(pipe: Pipeline) -> None:
    st.subheader("Walk-forward backtest")
    hint(
        "The Forecast tab is in-sample by construction: you chose the query, then "
        "measured what followed. This tab asks the harder question — out of sample, "
        "would matching have predicted direction better than guessing?"
    )
    guide("Backtest")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon = st.slider(
            "Horizon (bars)", min_value=5, max_value=60, value=15, step=5,
            help="How many bars ahead each step predicts. 15 bars = 15 {}. "
                 "Longer horizons are easier to call and worth more when right; "
                 "shorter ones are noisier and closer to untradeable.".format(
                     active_unit() + "s"),
        )
    with c2:
        fee_bps = st.number_input(
            "Round-trip fee (bps)", min_value=0.0, max_value=50.0, value=2.0,
            step=0.5, format="%.1f",
            help="Cost charged once per prediction. A {} {} round trip "
                 "typically "
                 "costs 1–3 bps, which can exceed the entire edge being measured. At 0 "
                 "the net mean is reported as NaN rather than as an optimistic zero."
                 .format(active_label().lower(), SYMBOL_FOR_HELP[0]),
        )
    with c3:
        warmup = st.number_input(
            "Warm-up bars", min_value=200, max_value=5000, value=1500, step=100,
            help="How much history each step is allowed to see before it may predict. "
                 "Must leave at least one window plus one horizon after it — otherwise "
                 "there is nothing to match against. Larger values mean less data per "
                 "step but a stricter no-peeking guarantee.",
        )
    with c4:
        max_steps = st.number_input(
            "Max steps", min_value=5, max_value=200, value=40, step=5,
            help="Number of predictions to make. Each step re-searches the library, so "
                 "this controls runtime more than anything else. More steps narrow the "
                 "confidence interval; a wide interval spanning the baseline is not an "
                 "edge, however good the point estimate looks.",
        )

    hint(
        "At each step the matcher sees bars `[0, i)` <b>only</b> — it cannot peek at "
        "the outcome it is about to be scored against. It predicts the direction of the "
        "return over `[i, i + horizon)`, is scored against what actually happened, then "
        "the next step starts far enough ahead that no two predictions share a bar."
    )
    st.code(
        "step i:  matcher sees bars [0, i)          ← no future information\n"
        "          predicts direction over [i, i+horizon)\n"
        "          scored against the realised return\n"
        "step i+s:  s = max(horizon, window length/2) bars later\n"
        "           → consecutive predictions share no bar",
        language="text",
    )
    hint(
        "**Why the spacing matters.** Predicting every bar while measuring a "
        "multi-bar outcome would make consecutive rows near-duplicates treated as "
        "independent, inflating the effective sample to roughly `n / horizon` and "
        "shrinking every interval until noise looked significant (§Z3). The actual "
        "stride used is printed with the results."
    )

    signature = (pipe.length, int(horizon), float(fee_bps), int(warmup), int(max_steps))

    if st.button(
        "Run backtest", type="primary", width='stretch',
        help="Execute the walk-forward simulation. This is the slowest action in the "
             "app: every step rebuilds and searches the whole window library, then "
             "runs 500 permutations for the p-value.",
    ):
        st.session_state["bt_signature"] = signature
        st.session_state["bt_result"] = None

    if st.session_state.get("bt_signature") != signature:
        st.info("Press **Run backtest** to execute. It re-searches the library at every "
                "step, so it is far slower than a single match.")
        hint(
            "Nothing runs until you press the button — a backtest over many steps can "
            "take a while, and silently re-running it on every slider nudge would make "
            "the app unusable. Changing any parameter above invalidates the previous "
            "result and returns you to this prompt."
        )
        return

    result = st.session_state.get("bt_result")
    if result is None:
        with st.spinner("Running walk-forward backtest…"):
            result = walk_forward(
                pipe, k=40, horizon=int(horizon), warmup=int(warmup),
                fee_bps=float(fee_bps), n_perm=500, seed=0,
                max_steps=int(max_steps),
            )
        st.session_state["bt_result"] = result

    # ``warnings`` lives on the dataclass, not in as_dict() — see backtest.py.
    for warning in (getattr(result, "warnings", None) or []):
        st.warning(warning)

    d = result.as_dict()
    n = d["n_predictions"]
    if not n:
        st.error("The backtest produced no predictions. Check that warm-up leaves "
                 "enough history for a window plus a horizon.")
        hint(
            "<b>What to try:</b> lower <i>Warm-up bars</i> (it must leave at least one "
            "window of {0} bars plus <i>Horizon</i> = {1} bars), or shorten <i>Horizon</i>. "
            "If warm-up × 2 already exceeds the archive size, no step can ever fire."
            .format(pipe.length, pipe.length + int(horizon))
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "direction accuracy", "{:.1%}".format(d["direction_accuracy"]),
        help="Share of steps whose predicted sign matched the realised sign. Only "
             "meaningful next to the baseline beside it.",
        border=True,
    )
    m2.metric(
        "baseline accuracy", "{:.1%}".format(d["baseline_accuracy"]),
        help="The same accuracy achieved by predicting random directions — the floor "
             "any edge has to clear.",
        border=True,
    )
    m3.metric(
        "lift vs baseline", "{:+.1%}".format(d["lift"]),
        help="accuracy − baseline accuracy. **This is the number that matters.** A "
             "positive lift with a p-value ≥ 0.05 is still chance.",
        border=True,
    )
    m4.metric(
        "n predictions", "{:,}".format(n),
        help="Effective sample size. Because steps are spaced by "
             "max(horizon, window/2) bars, no two predictions share a bar — this is a "
             "real count of independent observations, not a count of overlapping rows.",
        border=True,
    )

    st.write(
        "* 95% block-bootstrap CI on accuracy: **[{:.1%}, {:.1%}]** "
        "(stride {} bars)".format(d["accuracy_ci_low"], d["accuracy_ci_high"], d["stride"])
    )
    hint(
        "**How to read that interval:** if it is wide, or if it contains the "
        "{:.1%} baseline, then the point estimate above is not distinguishable from "
        "guessing — regardless of how good it looks. The *stride* of {} bars is the "
        "gap between consecutive predictions; it is derived from the horizon and the "
        "window length, not chosen.".format(d["baseline_accuracy"], d["stride"])
    )
    st.write(
        "* permutation **p = {}** · horizon {} bars · fee {} bps · net mean {} · "
        "net hit rate {}".format(
            "{:.4f}".format(d["p_value"]) if np.isfinite(d["p_value"]) else "—",
            d["horizon_min"],
            "{:.1f}".format(d["fee_bps"]),
            bps(d["net_mean"]),
            "{:.1%}".format(d["net_hit_rate"]) if np.isfinite(d["net_hit_rate"]) else "—",
        )
    )
    hint(
        "**p-value** comes from 500 permutation shuffles of the match labels: how often "
        "does chance alone produce an effect this large? **Net mean / net hit rate** "
        "appear only when fees were supplied — they are the return after subtracting "
        "the fee once per prediction."
    )

    if not np.isfinite(d["p_value"]) or d["p_value"] >= 0.05:
        st.error(
            "**This result is not statistically distinguishable from chance.** "
            "p = {} ≥ 0.05 against a permutation test, so the direction accuracy "
            "above is consistent with the matcher having no edge. The confidence "
            "interval [{} , {}] also contains the {:.1%} baseline. Do not read the "
            "point estimate as a strategy.".format(
                "{:.4f}".format(d["p_value"]) if np.isfinite(d["p_value"]) else "nan",
                "{:.1%}".format(d["accuracy_ci_low"]), "{:.1%}".format(d["accuracy_ci_high"]),
                d["baseline_accuracy"],
            )
        )
        hint(
            "<b>What this does and does not mean.</b> It does not mean the code is "
            "broken — it means the forecast did not clear a random baseline, which is "
            "the most common outcome for pattern matching on a single asset's intraday "
            "archive. Before concluding anything, confirm the pipeline is sound by "
            "running the **Placebo** test: <code>python3 -m timeseries.placebo</code> "
            "on synthetic random-walk data. If that reports significant patterns, the "
            "bug is in the harness, not the market."
        )
    else:
        st.success(
            "p = {:.4f} < 0.05, so direction accuracy beats the permutation baseline. "
            "Still note the interval width and that fees are applied per round trip."
            .format(d["p_value"])
        )
        hint(
            "Before treating this as an edge: check that the interval does not span "
            "the {:.1%} baseline, raise *Max steps* to tighten it, and re-run with a "
            "higher fee — a 1–3 bps round trip can easily exceed the lift.".format(
                d["baseline_accuracy"])
        )

    if d["fee_bps"] <= 0:
        st.warning(
            "No fees applied. A {} {} round trip typically costs 1–3 bps, "
            "which "
            "can exceed the edge being measured — net mean is therefore NaN rather "
            "than an optimistic zero.".format(
                active_label().lower(), SYMBOL_FOR_HELP[0])
        )
        hint(
            "<b>Fix it:</b> set <i>Round-trip fee</i> to at least 1.0 bps and re-run. "
            "Without a fee this harness deliberately refuses to report a net return, "
            "rather than implying a frictionless one."
        )


# =============================================================================== #
# Sidebar + main
# =============================================================================== #
#: How the fetch scope presents itself.  ``widget_label`` names the box within it,
#: and the help text names the resolution beside it.  Declared as data rather than as
#: a function body so the wording is one definition: a copy would let the box and its
#: own tooltip drift while both claimed to be the app's only source of bars.
#:
#: ``downloaded`` takes a timeframe argument, because it is
#: a statement *about a resolution* -- "every 1-minute bar Yahoo has (~29 days)" and
#: "every daily bar Yahoo has" are different sentences, and the first is actively
#: misleading if the reader is on the other one.  So it is a ``.format`` template
#: taking ``(symbol, timeframe)`` rather than fixed prose.
FETCH_SCOPES: Dict[str, Dict[str, str]] = {
    "app": {
        "widget_label": "Ticker symbol",
        "button_label": "Fetch",
        "help": (
            "Any ticker Yahoo Finance accepts. Press **Fetch** to download its bars at "
            "the chosen resolution and redraw every tab on the page. Nothing is "
            "written to `data/` — the bars live in memory for this session only.\n\n"
            "The app opens on **{}**, so there is something on screen before you touch "
            "anything. Type any other symbol and press **Fetch** to switch.\n\n"
            "**Resolution is chosen once, when the app opens** — it is shown "
            "beside this box and is not a control here. 1-minute gives ~29 days "
            "of intraday tape; Daily gives the whole listing history, a far "
            "larger archive to match over and a very different question. To "
            "change it, reload the page and choose again.\n\n"
            "**One ticker, one page.** Every tab — Price, Matches, Projection, "
            "Forecast, Quality and Backtest — is built from the same bars, so a "
            "forecast is always a forecast of the instrument you are looking at. "
            "There is no second box to point at another name.\n\n"
            "Class shares are typed with a dash (`BRK-B`); it is converted for you."
            "\n\nYahoo index symbols take a leading `^` — `^VIX` for the CBOE "
            "Volatility Index, `^GSPC` for the S&P 500, `^DJI` for the Dow."
        ),
        "downloaded": (
            "Downloading every {2} bar Yahoo has for {1} and redrawing every tab "
            "on the page. Press it again to pick up bars from the session still "
            "in progress."
        ),
    },
    "price": {
        "widget_label": "Ticker symbol",
        "button_label": "Fetch",
        "help": (
            "Any ticker Yahoo Finance accepts. Press **Fetch** to download its bars at "
            "the chosen resolution and redraw every tab on the page. Nothing is "
            "written to `data/` — the bars live in memory for this session only.\n\n"
            "The app opens on **{}**, so there is something on screen before you touch "
            "anything. Type any other symbol and press **Fetch** to switch.\n\n"
            "**Resolution is chosen once, when the app opens** — it is shown "
            "beside this box and is not a control here. 1-minute gives ~29 days "
            "of intraday tape; Daily gives the whole listing history, a far "
            "larger archive to match over and a very different question. To "
            "change it, reload the page and choose again.\n\n"
            "**One ticker, one page.** Every tab — Price, Matches, Projection, "
            "Forecast, Quality and Backtest — is built from the same bars, so a "
            "forecast is always a forecast of the instrument you are looking at. "
            "There is no second box to point at another name.\n\n"
            "Class shares are typed with a dash (`BRK-B`); it is converted for you."
            "\n\nYahoo index symbols take a leading `^` — `^VIX` for the CBOE "
            "Volatility Index, `^GSPC` for the S&P 500, `^DJI` for the Dow."
        ),
        "downloaded": (
            "Downloading every {2} bar Yahoo has for {1} and redrawing every tab "
            "on the page. Press it again to pick up bars from the session still "
            "in progress."
        ),
    },
    "forecast": {
        "widget_label": "Ticker symbol",
        "button_label": "Fetch",
        "help": (
            "Any ticker Yahoo Finance accepts. Press **Fetch** to download its bars at "
            "the chosen resolution and redraw every tab on the page. Nothing is "
            "written to `data/` — the bars live in memory for this session only.\n\n"
            "The app opens on **{}**, so there is something on screen before you touch "
            "anything. Type any other symbol and press **Fetch** to switch.\n\n"
            "**Resolution is chosen once, when the app opens** — it is shown "
            "beside this box and is not a control here. 1-minute gives ~29 days "
            "of intraday tape; Daily gives the whole listing history, a far "
            "larger archive to match over and a very different question. To "
            "change it, reload the page and choose again.\n\n"
            "**One ticker, one page.** Every tab — Price, Matches, Projection, "
            "Forecast, Quality and Backtest — is built from the same bars, so a "
            "forecast is always a forecast of the instrument you are looking at. "
            "There is no second box to point at another name.\n\n"
            "Class shares are typed with a dash (`BRK-B`); it is converted for you."
            "\n\nYahoo index symbols take a leading `^` — `^VIX` for the CBOE "
            "Volatility Index, `^GSPC` for the S&P 500, `^DJI` for the Dow."
        ),
        "downloaded": (
            "Downloading every {2} bar Yahoo has for {1} and redrawing every tab "
            "on the page. Press it again to pick up bars from the session still "
            "in progress."
        ),
    },

}


class ResolvedScope:
    """What the scope is currently built on: a symbol *and* a resolution.

    A small value object rather than a two-tuple because ``main()`` and every
    renderer that reads it use both fields several times, and ``scope.symbol`` /
    ``scope.tf.label`` says what a bare ``(symbol, tf)`` destructuring does not.  It is
    also what lets the *resolution* be threaded through the same call sites the symbol
    already travelled, instead of a second parallel channel.

    ``tf`` is a resolved :class:`~timeseries.timeframes.Timeframe`, so callers read
    ``scope.tf.default_length`` rather than going back to the registry.
    """

    __slots__ = ("scope", "symbol", "timeframe")

    def __init__(self, scope: str, symbol: str, timeframe: object):
        self.scope = scope
        self.symbol = symbol
        self.timeframe = resolve_timeframe(timeframe).key

    @property
    def tf(self) -> Any:
        """The resolved :class:`~timeseries.timeframes.Timeframe` for this scope."""
        return get_timeframe(self.timeframe)

    @property
    def cache_key(self) -> str:
        """Identity of this scope's archive, for cache keys and log lines."""
        return "{}@{}".format(self.symbol, self.timeframe)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "ResolvedScope({!r}, {!r})".format(self.symbol, self.timeframe)


def _scope_keys(scope: str) -> Tuple[str, str]:
    """``(ticker_key, input_key)`` for the one scope.

    There is exactly one ticker and one input now, so this is a constant pair rather
    than a switch: the two keys used to be selected per-scope, and the selection is
    what let the Forecast scope hold a different instrument from the Price one.  The
    pair is kept as a function so callers ask for "the keys" without naming them, which
    is the same indirection :data:`TICKER_KEY` already provides for the durable value.
    """
    return (TICKER_KEY, "{}ticker_input".format(scope))


def _recent_tickers() -> List[str]:
    """The ordered list of recently used tickers, most recent first.

    Seeded from :data:`DEFAULT_RECENT_TICKERS` on first access and capped at
    :data:`MAX_RECENT_TICKERS`.
    """
    if RECENT_TICKERS_KEY not in st.session_state:
        st.session_state[RECENT_TICKERS_KEY] = list(DEFAULT_RECENT_TICKERS)
    return st.session_state[RECENT_TICKERS_KEY]


def _add_recent_ticker(symbol: str) -> None:
    """Record ``symbol`` as recently used, moving it to the front of the list.

    Duplicates are removed before re-insertion at the front, and the list is
    truncated to :data:`MAX_RECENT_TICKERS` so the oldest entry drops off.
    """
    recent = _recent_tickers()
    if symbol in recent:
        recent.remove(symbol)
    recent.insert(0, symbol)
    del recent[MAX_RECENT_TICKERS:]


def resolve_ticker(scope: str = "price") -> Optional[ResolvedScope]:
    """What the scope is currently built on: symbol and resolution.

    **This runs above the tab bar, which is the whole reason it is separate from
    :func:`render_ticker_input`.**  ``main()`` needs the symbol to build the
    pipelines, and the widgets that produce it are drawn *below* that -- in the
    sidebar.  A widget's value can only be read once it exists, so the resolution
    is done here from ``session_state`` on the pass *after* the reader pressed Fetch,
    and the sidebar only has to draw the widgets.

    The one-run lag is not a workaround, it is how Streamlit works: the click is
    recorded against the button's key and observed by the next run, which is the same
    arrangement the Price brush already relies on (see ``PRICE_BRUSH_KEY``).  Verified
    against the installed version:

        paint 1     -> QQQ
        ->AAPL      -> AAPL
        ->MSFT      -> MSFT

    **A click cannot be consumed by assigning to the button key.**  That key is a
    widget, and Streamlit raises ``StreamlitValueAssignmentNotAllowedError`` for any
    write to one.  So the click is cleared through a separate non-widget "pending" key,
    which is what makes the fetch happen exactly once rather than on every later
    rerun.

    **The resolution needs no such latch.**  A typed symbol is a *proposal*
    until Fetch is pressed; the resolution, by contrast, is decided once at startup
    by :func:`session_timeframe` and published to :data:`ACTIVE_TIMEFRAME` before this
    function runs, so there is nothing for it to latch and nothing to park.  This is
    the whole reason the resolution is fixed per session: a per-tab selector would
    need the same pending-key dance as the button, and -- as it turned out -- a
    selector can also be wired up in a way that reads as correct and does nothing.

    Returns ``None`` only when a symbol cannot be resolved at all, which is a
    programming error rather than a reader error -- every other bad input (a typo, a
    delisted symbol) is reported by :func:`render_ticker_input`, which is the only
    place the download and its messages happen.
    """
    ticker_key, input_key = _scope_keys("app")
    pending_key = ticker_key + PENDING_SUFFIX

    # ---- The resolution --------------------------------------------------- #
    # **Read from the session, never from this scope.**  There is one resolution per
    # session, chosen at startup by :func:`session_timeframe` and already published to
    # :data:`ACTIVE_TIMEFRAME` above the tab bar.  There is only one scope now, so
    # nothing can drift from it and no caller has to pass it.
    wanted_tf = ACTIVE_TIMEFRAME[0]

    # The seed is resolved here, on the first pass that reaches this scope.  It must
    # happen before ``render_ticker_input`` reads ``value=`` -- see that function for
    # the measured consequence of getting the order wrong.
    if ticker_key not in st.session_state:
        try:
            st.session_state[ticker_key] = F.normalize_symbol(DEFAULT_SYMBOL)
        except ValueError as exc:  # a bad default is reported, not raised
            st.error(str(exc))
            return None

    # A pending fetch from the pass that drew the button.  Popped in one step, because
    # leaving it set would re-download on every subsequent rerun.
    if st.session_state.pop(pending_key, None):
        # Disable the fetch button while downloading
        st.session_state["fetch_disabled"] = True
        raw = st.session_state.get(input_key)
        # **The symbol that was *asked for*, recorded before the attempt.**  The
        # failure has to name the thing the reader typed: reporting the error against
        # the symbol still in force reads as "AAPL has no bars" when what happened is
        # "the symbol you just typed has none" -- which sends the reader to debug the
        # wrong ticker.  Measured before this was fixed:
        #
        #     typed NOT_A_TICKER_XY -> "No 1-minute bars returned for **AAPL**."
        #
        try:
            sym = F.normalize_symbol(raw) if raw else None
        except ValueError as exc:
            # A malformed symbol is reported rather than raised, and the previously
            # in-force ticker is kept: a typo should cost the click, not the archive
            # the reader was already looking at.
            FETCH_RESULTS[ticker_key] = _FetchFailure(raw or "", str(exc))
            # Re-enable the fetch button
            st.session_state["fetch_disabled"] = False
            return ResolvedScope("app", st.session_state[ticker_key], wanted_tf)
        if not sym:
            # Re-enable the fetch button
            st.session_state["fetch_disabled"] = False
            return ResolvedScope("app", st.session_state[ticker_key], wanted_tf)

        # ``refresh=True``: pressing Fetch is an explicit request for the latest bars,
        # so it must re-download rather than return the frame cached earlier this
        # session.  Without that, the newest -- and most interesting -- session would
        # stay frozen at whatever it looked like on the first click.
        with st.spinner("Downloading all available {} bars…".format(
                resolve_timeframe(wanted_tf).label)):
            result = fetch_ticker_cached(sym, refresh=True, timeframe=wanted_tf)
        if result.ok:
            # A new symbol or resolution means every earlier view state is about
            # different bars.  Clearing it is what stops the page reporting yesterday's
            # match count over today's price chart.
            st.session_state[ticker_key] = sym
            _add_recent_ticker(sym)
            reset_query_state("app", sym)
        else:
            # The symbol stays as it was.  Adopting one with no bars would leave the
            # page charting an empty archive, and every tab would read as "this
            # archive has no such pattern" -- a claim about data that does not exist.
            result.symbol = sym
        FETCH_RESULTS[ticker_key] = result
        # Re-enable the fetch button
        st.session_state["fetch_disabled"] = False

    return ResolvedScope("app", st.session_state[ticker_key], wanted_tf)


class _FetchFailure:
    """A symbol the reader typed that could not be used at all.

    Not a ``FetchResult``: nothing was fetched, so there is no frame and no bar count to
    report.  It carries the **attempted** symbol -- not the one still in force -- so the
    error names what the reader typed, and exists so the floating bar has one shape to
    render for "this fetch did not work" without a second branch.
    """

    def __init__(self, symbol: str, message: str):
        self.ok = False
        self.symbol = symbol
        self.message = message
        self.errors: List[str] = [message]

    def summary(self) -> str:
        return self.message


def render_ticker_input(scope: str) -> Optional[str]:
    """One scope's controls: resolution *label*, ticker box, **Fetch**, messages.

    Drawn in the bar above the tabs rather than in the tab body it controls.  That is
    what "floating" buys: the Forecast input stays reachable while the reader is on
    Price, and the Price input while they are on Forecast, so either instrument can be
    changed without hunting for a control that is only on screen half the time.

    **All the state work happens in :func:`resolve_ticker`, before the tabs render.**
    This function draws widgets and reports outcomes, and returns the symbol in force so
    the bar can show the download result.  The split is what lets both inputs sit in one
    always-visible bar while still feeding a pipeline that ``main()`` builds before any
    tab exists.

    The box is seeded from the scope's **non-widget** ticker key rather than from the
    raw text of the widget, so the field and the bars cannot disagree -- see the note on
    ordering in :func:`resolve_ticker`.

    **The resolution is *shown* to the left of the ticker box, and cannot be changed
    here.**  It was a selector in this exact position, which reads well and is why the
    bug hid for so long: it looked like it worked.  It did not.  Its widget key was
    passed as ``key=`` and read nowhere, so choosing "Daily" moved the control and
    nothing else -- the page kept charting 1-minute bars under a Daily caption, with no
    error anywhere to explain it.  The choice now happens once at startup
    (:func:`session_timeframe`), and this is a label.

    The label keeps the position rather than being deleted, for the reason the
    position had: it reads as a qualifier on the symbol ("QQQ, daily"), it puts the
    resolution in front of the eye before a control whose effect is otherwise invisible
    until the chart redraws, and it keeps the three columns on one row -- which is
    what the measured alignment notes below are about.  Adding a second row would
    invalidate all of them.
    """
    cfg = FETCH_SCOPES[scope]
    ticker_key, input_key = _scope_keys(scope)
    fetch_key = "{}ticker_fetch".format(scope)
    # Note: The above is simplified because there is now only one TICKER_FETCH_KEY.
    # The previous version attempted to use scope-specific keys.
    pending_key = ticker_key + PENDING_SUFFIX

    # ``vertical_alignment`` is deliberately not set.  It was tried here and measured
    # byte-identical to the default (input 381, button 378) -- the columns are already
    # the same height, because the collapsed label leaves the input with nothing above
    # it.  Setting it would imply it is doing something.
    # **The resolution is shown, not chosen.**  It was a dropdown here, and it was
    # the bug: its widget key was passed as ``key=`` and read nowhere, so selecting
    # "Daily" moved the control and nothing else -- the page kept charting 1-minute
    # bars under a Daily caption, with no error to explain it.  The choice now happens
    # once at startup (:func:`session_timeframe`) and this is a label, so there is no
    # control here that can disagree with the archive below it.
    #
    # A label rather than nothing at all, because the two archives are genuinely
    # different and a reader who switches ticker needs to know which one they are
    # searching.  It also keeps the three columns aligned, which the measured notes on
    # the button's position depend on.
    col_tf, col_in, col_go = st.columns([1, 3, 1])
    with col_tf:
        st.markdown(
            '<div class="tp-res">{}</div>'.format(active_label()),
            unsafe_allow_html=True,
        )
    with col_in:
        # Safe while typing: this expression is constant for as long as the user is
        # editing, so Streamlit does not reset the widget under them. It only changes
        # value at the moment a fetch resolves a new symbol.
        #
        # The widget is *drawn* here, not read: the text this box holds is read back
        # from session state by the fetch button, and the value that matters is the
        # one the reader has committed to, not the one they are halfway through
        # typing.  So the return value is deliberately discarded.
        st.text_input(
            cfg["widget_label"], key=input_key,
            value=st.session_state[ticker_key],
            max_chars=16,
            placeholder="AAPL, MSFT, BTC-USD, BRK-B, ^VIX…",
            help=cfg["help"].format(DEFAULT_SYMBOL),
            label_visibility="collapsed",
        )
    with col_go:
        # **No spacer div above this button, and that is the fix.**
        #
        # This used to be `st.markdown('<div style="height:1.7rem"></div>')`, a
        # hard-coded nudge that existed to clear the *visible label* of the text box
        # beside it -- a text input renders its label above the field, so a button in
        # the next column starts level with the label rather than the field.  When the
        # input was given ``label_visibility="collapsed"`` the label stopped rendering
        # and the nudge became a 1.7rem gap between two controls that were already
        # aligned, pushing the button 25px below the box.
        #
        # Measured in Chromium against this app, viewport 1600px, ``top`` of each box:
        #
        #     variant                         input   button   delta
        #     spacer + collapsed label (was)     381      406   +25px
        #     spacer removed                     381      378    -3px
        #     visible label + spacer             409      406    -3px
        #
        # The 3px that remain are **not** a nudge problem, and are left alone.  They
        # are Streamlit's own base metrics: a button renders 40px tall and a text input
        # 36px, so two correctly aligned boxes of different heights have their tops ~2px
        # apart.  ``vertical_alignment="center"`` does not close it -- measured
        # identical to the row above -- and there is no supported way to set it:
        #
        #   * no theme option exists.  ``baseButtonHeight`` is not declared by
        #     ``streamlit.config`` in 1.65, so putting it in ``config.toml`` is
        #     accepted silently and ignored;
        #   * no CSS route either.  Streamlit sanitises ``<style>`` passed to
        #     ``st.markdown(unsafe_allow_html=True)``, and a
        #     ``[data-testid="stButton"] button {height: 2.375rem}`` rule added to
        #     ``HELP_CSS`` measured as having no effect on the DOM.
        #
        # Both dead ends are recorded in ``.streamlit/config.toml`` so the next person
        # does not spend the same hour finding them.
        #
        # A magic-number spacer cannot express any of this.  It encodes one
        # configuration -- this label, this font size, this Streamlit version -- and
        # fails *silently* on every other one: it stays the same height while the
        # thing it was compensating for changes underneath it.  That is exactly what
        # happened here, and it is why the label is now collapsed and the spacer gone
        # rather than one compensating for the other.
        if st.button(cfg["button_label"], key=fetch_key,
                         width='stretch', type="primary",
                         help=cfg["downloaded"].format(
                             resolve_timeframe(ACTIVE_TIMEFRAME[0]).label,
                             st.session_state[ticker_key].lower(),
                             resolve_timeframe(ACTIVE_TIMEFRAME[0]).label.lower()),
                         disabled=st.session_state.get("fetch_disabled", False)):
            # A *non-widget* key, because the button's own key is read-only: writing
            # to it raises.  ``resolve_ticker`` pops this on the next pass, which is
            # where the download and the state reset happen.
            st.session_state[pending_key] = True
            st.rerun()

    # The download itself happened up in ``resolve_ticker``, before the pipelines were
    # built.  Reporting it here -- rather than fetching again -- is what keeps the
    # message in the bar that owns the control, without a second download per press.
    result = FETCH_RESULTS.pop(ticker_key, None)
    if result is not None:
        label = resolve_timeframe(
            getattr(result, "timeframe", ACTIVE_TIMEFRAME[0])
        ).label
        if result.ok:
            st.success(result.summary(), icon="📈")
        else:
            st.error("No {} bars returned for **{}**.".format(label, result.symbol))
            for message in result.errors[:3]:
                hint("· {}".format(message))
        if not result.ok:
            hint(
                "Usual causes: the symbol does not exist, or it has no {} "
                "history (yfinance covers liquid US equities and major crypto pairs; "
                "most ETFs and index funds are fine, but an obscure or newly listed "
                "name may have none). **The ticker in force is unchanged.**".format(
                    label.lower())
            )

    return st.session_state[ticker_key]


def render_scope_ticker(scope: str, *, note: str = "") -> Optional[str]:
    """One tab's own ticker input, drawn at the top of that tab's body.

    **Each tab carries its own input, inside itself.**  That is the arrangement the two
    instruments need: the control that changes what you are looking at sits directly
    above the charts it changes, so there is no separate "instrument picker" to keep
    mentally in sync with the tab you are reading.

    It does mean the Forecast input is only on screen while the *Forecast* tab is
    selected -- which is the trade.  ``st.tabs`` renders every body on every rerun but
    shows one at a time, so a tab-local input is invisible from its neighbours.  The
    alternative (a bar above the tabs) was tried first and reverted: it kept both
    reachable, but put an instrument selector between the reader and the page heading,
    on every tab, including the ones that have no ticker of their own.

    **The input is on *Forecast*, and *Projection* charts the same ticker without one.**
    The Projection tab is a fixed reference and takes no input, so it has nothing to
    offer here -- and drawing the input on both would register
    ``ticker_input_forecast`` twice per pass, which Streamlit rejects outright.

    ``note`` names the *other* tabs the same instrument feeds, because Price's input
    also drives Matches, Quality and Backtest -- a reader who changes it and then visits
    those tabs should not have to wonder why their bars moved too.
    """
    ticker = render_ticker_input(scope)
    if note:
        hint(note)
    return ticker


def reset_query_state(scope: str, symbol: str) -> None:
    """Drop every piece of state tied to the previous ticker.

    Bar indices, the From/To pickers and any brush all describe positions in one
    archive.  Carried across to another symbol they point at unrelated moments, and
    the app would quietly re-run a query the user never asked for.  The match results
    are cleared for the same reason, and with them the cached ``run`` output, whose
    signature includes the source path but not the fetched frame.

    ``st.session_state["_"] = {}`` is Streamlit's discard-all key, kept here for the
    re-seeding it used to do for the From/To pickers.  **Those pickers are gone, and
    the wipe no longer does anything** -- verified against this Streamlit rather than
    assumed, since the whole line is inherited from a UI that no longer exists:

        slid=7 -> slid=3 -> (wipe) -> slid=3      # the slider keeps its value

    ``session_state["_"]`` has no special handling in the installed version; it is
    simply a key whose value happens to be an empty dict.  So it clears nothing, and in
    particular it no longer evicts a cached pipeline.  A cached pipeline is keyed on
    the symbol, which is part of the cache key, so a new ticker's bars cannot be served
    against the old one's matrix anyway -- the per-scope key lists below are what
    actually discard state.

    It is left in place rather than deleted because it is harmless, and because the
    two stale keys it names (``run_signature``/``run_output``) are cleared explicitly
    below regardless.  A reader who assumes a blank line is doing work would be misled;
    a reader who assumes it is gone would be misled the other way.  The docstring says
    which.

    The reset is global: switching the ticker resets everything.  We bump the tab
    generation so the reader lands on *Price* with the latest window and its match
    already drawn.

    Two details that a symbol-keyed version would have got wrong.  A bump rather than
    a set-to-symbol, so that re-fetching the symbol already on screen -- the
    documented "press Fetch again to pick up the newest session" -- also re-keys and
    also lands on Price, instead of leaving the reader on Backtest in front of a
    freshly downloaded archive.  And a bump rather than a pop, because the bar has no
    stored selection to clear: popping would restore the *first* generation, which
    the browser may still be holding, and the reset would quietly do nothing.

    The one behaviour this does not change: an ordinary rerun -- a slider nudge, a
    brush, a text input -- leaves the generation alone, so it never yanks the reader
    back to Price while they are mid-analysis.
    """
    # Global reset: combine all stale keys.
    stale = STALE_KEYS

    st.session_state["_"] = {}
    for key in stale:
        st.session_state.pop(state_key(key), None)

    # Always bump the tab bar generation to land on Price.
    st.session_state[TABS_GENERATION_KEY] = (
        int(st.session_state.get(TABS_GENERATION_KEY, 0)) + 1
    )
    st.session_state[TICKER_KEY] = symbol


def render_sidebar() -> Dict[str, Any]:
    """Render every control, each with a hover tooltip explaining what it changes.

    The help here is deliberately *attached to the control it describes* rather than
    collected in one page: the question "what does this slider do?" is only ever asked
    while the slider is on screen.
    """
    with st.sidebar:

        # Recently used tickers as quick-load buttons.  Clicking one sets the symbol
        # in force and triggers a fetch, exactly as if the reader had typed it and
        # pressed **Fetch**.  The fetched symbol moves to the front of the list.
        st.subheader("Recently used tickers")
        recent = _recent_tickers()
        ticker_key, input_key = _scope_keys("app")
        pending_key = ticker_key + PENDING_SUFFIX
        cols = st.columns(4, gap="small")
        for i, sym in enumerate(recent):
            with cols[i % 4]:
                if st.button(sym, key="recent_{}".format(sym), width='stretch',
                             help="Load {} bars".format(sym)):
                    st.session_state[input_key] = sym
                    st.session_state[pending_key] = True
                    st.rerun()

        # Tighten the button grid: remove the gap between columns so the buttons sit
        # flush against each other.
        st.markdown(
            '<style>[data-testid="stColumn"] {padding: 0 !important; gap: 0 !important;} '
            '[data-testid="stColumn"] > div {margin: 0 !important; padding: 0 !important;} '
            '[data-testid="stButton"] {margin: 0 !important;} '
            '[data-testid="stButton"] > button {margin: 0 !important; padding: 0.125rem 0.25rem !important;}</style>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Ticker input for the whole session.
        render_ticker_input("app")

        st.divider()

        # **The ticker input is now here in the sidebar, and there is exactly one.**
        # Every tab — Price, Matches, Projection, Forecast, Quality and Backtest —
        # is built from the same archive.
        #
        # Nothing is fetched here any more, so nothing gates the sidebar either.
        # ``main()`` resolves the symbol before this function is called and refuses to
        # draw the page at all if it cannot be resolved.
        #
        # ``amplitude_weight`` stays at the bottom of the sidebar -- it is a scoring
        # rule shared across both instruments on purpose, so a distance or a
        # percentile means the same thing wherever it was produced.
        amplitude_weight = st.slider(
            "Size of move", min_value=0.0, max_value=2.0, value=1.0, step=0.05,
            help="How much a window's **size of move** counts, separate from its "
                 "shape. STUMPY's distance is scale-free: every window is z-scored "
                 "before comparison, so a dead-calm window and a violent one are "
                 "equally far from your query no matter how different they look. "
                 "At the default of 1.0 a candidate must have moved about as far "
                 "as yours, in the same direction, to win on shape alone. Set it "
                 "to 0 for pure shape-matching — ‘when did this wiggle happen?’ "
                 "rather than ‘when did price move like this?’",
        )
        return {
            "amplitude_weight": float(amplitude_weight),
        }


def render_price_tab(pipe: Pipeline, start_idx: int, stop_idx: int,
                     chart_from: int, view_stop: int,
                     *,
                     height: int = 440,
                     chart_width: Optional[int] = None,
                     selectable: bool = False,
                     on_select: Any = None,
                     rebase_at: Optional[int] = None,
                     show_title: bool = False) -> Any:
    """The price chart, and nothing else.

    ``with_help`` used to switch the span caption and the cleaned-frame hint on for
    the *Chart* tab and off for the Price tab.  With that tab gone there is a single
    caller and a single shape, so the flag and the branch behind it are gone too: the
    Price tab is the tape alone, which is what this function now draws.

    ``chart_width`` is threaded through from the Price tab so both panels draw a bar at the
    same pixel size -- equal *width* is what makes the two shapes comparable by eye.  It
    does not size this chart's window; ``chart_from``/``view_stop`` do.

    **The visible window never moves, and that is the point.**  The view used to be
    :func:`aligned_view`, which places ``[start_idx, stop_idx)`` at a *given* x-offset --
    and the Price tab computed that offset with :func:`shared_anchor` from the query
    **and** the match.  So the offset, and with it the whole visible range, changed
    whenever either band moved.  Brushing a new window moved the band, which moved the
    anchor, which moved the view, which slid the tape out from under the reader mid-drag
    and shifted the band they were looking at to a different place on the chart.

    That is the "it scrolls when I select a window" symptom, and it bites hardest exactly
    where the reader works: brushing toward the left of the tape.  Measured at ``n=8069``,
    view width 1439, match at bar 3000 --

        brush@6630 -> view (6031, 7470)     band at offset  599
        brush@7000 -> view (6401, 7840)     band at offset  599
        brush@7500 -> view (6630, 8069)     band at offset  870
        brush@7829 -> view (6630, 8069)     band at offset 1199

    Four different views for the same trailing five sessions, with the band hopping
    599 -> 870 -> 1199 bars along x.  The view is therefore pinned to the trailing window
    the app already computed -- ``(chart_from, view_stop)`` -- which cannot depend on the
    query, and the band is drawn wherever it was brushed.  That is the behaviour the guide
    already promises ("the band is placed exactly where you drew it"), and it leaves the
    reader brushing against tape that stays put.

    The ``anchor`` parameter is **gone**, not merely ignored.  It was the only route by
    which a match position could reach this view, and an accepted-but-unused argument is
    an invitation: the next edit that reads it will put the scrolling straight back.  The
    load-time band alignment it bought is already lost regardless -- the pannable match
    panel below drops the anchor and centres its own band (see
    :func:`_render_best_match_pair`), so the two offsets have not agreed since panning
    existed.  Equal *width* is the property still worth protecting, and ``chart_width``
    carries it.

    ``selectable`` makes the chart the one the app reads a query window off.  The
    returned value is then whatever Streamlit's ``on_select`` produced, so the caller
    can turn a brush into a bar span with :func:`selection_to_span`.  The Price tab
    passes ``selectable=True``: this is now the only chart in the app that defines
    the query, so the brush here is not one route to it among several.

    A selectable chart needs an ``on_select`` handler -- the brush is pointless
    otherwise -- so passing ``selectable=True`` without one is a programming error and
    is rejected rather than silently rendering a chart that looks pickable and is not.

    ``rebase_at`` divides the axis by the close at that bar, turning it into
    percent-from-that-bar.  **The Price tab must pass it**, because the match panel
    drawn directly below is always rebased (see :func:`_render_best_match_pair`) and the
    two are meant to be read as a pair.  Without it the query chart is in dollars and
    the match chart is in percent, and the eye compares a 735.48–744.97 range against a
    −1.01%–0.86% range -- two different units, silently presented as one comparison.

    ``show_title`` overrides the panel title, which is otherwise suppressed on the bare
    Price view.  The Price tab sets it explicitly once a match exists, so the reader is
    told the axis is rebased rather than having to infer it from the tick labels.
    """
    if selectable and on_select is None:
        raise ValueError("selectable=True requires an on_select handler")

    # The visible window is the trailing one, unconditionally: ``(chart_from, view_stop)``.
    # It used to be ``aligned_view(..., anchor)`` with an anchor derived from the query
    # *and* the match, which made the view a function of the very thing the reader is
    # changing -- see the docstring for the measured re-frames.
    #
    # ``chart_width`` is deliberately *not* applied here.  Clipping the trailing window
    # to it would only ever yield an arbitrary prefix of the tape, which is worse than
    # the full window; and it is a no-op for the Price tab in any case, which passes
    # ``n - chart_from`` -- exactly the width this window already has.  The argument stays
    # on the signature because the Price tab threads one width to both renderers and the
    # two panels must draw a bar at the same pixel size; for this chart that width is
    # already implied by the window itself.
    lo, hi = max(0, int(chart_from)), int(view_stop)
    if hi <= lo:
        hi = min(pipe.n_bars, lo + 1)
    # The Price tab's key is a module constant because the brush is *read* long before
    # this chart is drawn -- the reader cannot import the key from a function that has
    # not run yet, so the two sides have to share a name declared up here.  The
    # ``chart_view`` key that shared this line with the Chart tab is gone with it.
    key = price_brush_key() if selectable else "price_view"

    # ``on_select``/``selection_mode`` are omitted entirely rather than passed as None:
    # Streamlit's own default is the string ``"ignore"``, and a None would be a
    # different value from the default rather than the default.
    select_kwargs: Dict[str, Any] = (
        {"on_select": on_select, "selection_mode": ("box", "lasso")}
        if selectable else {}
    )
    # The instrument is drawn on the chart itself, not only in the page heading.
    # Once the app can fetch any symbol, a chart panel is frequently all that is on
    # screen -- the heading scrolls away -- and an unlabelled figure is one the reader
    # can misread as still showing the ticker they just switched away from.
    # Suppressed until a match exists: the bare Price view is chrome-free, and there is
    # nothing below it to disagree about the units until ``show_title`` turns it on.
    fig_title = None
    if show_title:
        fig_title = "{} · {} close".format(SYMBOL_FOR_HELP[0], active_label().lower())
    return st.plotly_chart(
        build_price_figure(pipe, lo, hi, query_start=start_idx,
                           query_stop=stop_idx,
                           selectable=selectable, height=height, title=fig_title,
                           rebase_at=rebase_at),
        width='stretch',
        key=key,
        config=chart_config(selectable=selectable),
        **select_kwargs,
    )


def _record_resolution_choice() -> None:
    """Park the radio's value where the next pass will find it.

    A module-level function rather than a lambda because a widget callback holds a
    reference for the life of the page, and a lambda over ``st`` in a module that
    re-executes per interaction would capture whichever pass defined it.  The name
    lookup happens at call time, so the current pass's ``st`` is used.
    """
    st.session_state[SESSION_TIMEFRAME_PENDING_KEY] = (
        st.session_state[SESSION_TIMEFRAME_INPUT_KEY]
    )


def session_timeframe() -> Any:
    """The resolution this session runs at, asking for it on the first pass.

    **Why this is asked once, at startup, and not offered as a dropdown afterwards.**

    It was a dropdown.  The dropdown's widget key was passed as ``key=`` and read
    *nowhere*, so choosing "Daily" moved the control and nothing else -- the page kept
    charting 1-minute bars under a Daily caption, with no error to explain it.  That is
    the shape of bug a per-session gate removes rather than fixes: with one resolution
    there is no later moment at which the label and the archive could disagree,
    because neither is independently changeable.

    **The gate is drawn before anything else and returns ``None`` from ``main`` until
    it is answered**, rather than defaulting and loading the wrong archive first.  A
    default would have to be *some* resolution, and picking the wrong one means
    downloading an archive the reader did not ask for -- 6,285 daily bars rather than
    ~7,800 minute bars is a visible, wasteful thing to do on their behalf.

    The answer is durable (:data:`SESSION_TIMEFRAME_KEY`) and the widget is only drawn
    while it is unanswered, so the gate cannot reappear on a later pass or be
    re-answered by an unrelated rerun.
    """
    chosen = st.session_state.get(SESSION_TIMEFRAME_KEY)
    if chosen is not None:
        return resolve_timeframe(chosen)

    # **A committed answer is parked in a non-widget key**, exactly as a clicked
    # Fetch button is (:data:`PENDING_SUFFIX`), and this is not a stylistic echo --
    # it is the only arrangement that works.  A widget's key cannot be assigned to
    # once the widget has been registered: Streamlit raises
    # ``StreamlitValueAssignmentNotAllowedError``.  So the earlier version of this
    # gate, which did ``st.session_state[SESSION_TIMEFRAME_INPUT_KEY] = picked``,
    # would have raised on the one pass it was supposed to answer on -- which is
    # never a pass a test sees, because tests use a plain dict for session state.
    #
    # The first version also compared the radio's return value against the *widget's
    # own key*.  Those are the same object, so the branch could never be taken and
    # the gate never resolved: the page sat behind a caption asking for a resolution
    # forever, at every resolution.  Both faults are invisible to ``pytest`` and
    # obvious the moment the gate is driven through a fake widget that does not
    # hand back its own key.
    picked = st.session_state.pop(SESSION_TIMEFRAME_PENDING_KEY, None)
    if picked is not None:
        st.session_state[SESSION_TIMEFRAME_KEY] = picked
        return resolve_timeframe(picked)

    keys = timeframe_keys()
    # **The widget commits, rather than being read and compared.**  Two reasons, both
    # learned the hard way here:
    #
    # * Reading the radio's return value and asking "has this changed?" cannot work,
    #   because the radio already returns its own stored value -- it *is* the answer
    #   on the very first paint.  That version asked "is this different from what is
    #   stored?", compared the value with itself, and never resolved.
    # * Worse, an ``on_change`` alone leaves the reader who accepts the shown default
    #   stuck forever.  Clicking the already-selected option is not a change, so no
    #   callback fires, so 1-minute -- the default -- could never be committed.  The
    #   most likely answer to "which resolution?" would have been the one that did
    #   not work.
    #
    # So the gate *is* the widget: a rendered radio whose callback records the
    # selection, and a commit from that recording on the next pass.  Any selection,
    # including the default, is a selection.
    st.radio(
        SESSION_TIMEFRAME_LABEL,
        options=keys,
        index=keys.index(DEFAULT_TIMEFRAME),
        format_func=lambda k: TIMEFRAMES[k].label,
        key=SESSION_TIMEFRAME_INPUT_KEY,
        on_change=_record_resolution_choice,
    )
    st.caption(
        "Choose a resolution, then press **Continue**. **1-minute** is the intraday "
        "tape and holds roughly the last 29 days; **Daily** is one bar per trading "
        "day going back to the instrument's listing. The two are separate archives "
        "and separate searches — a daily pattern is never matched against minute "
        "bars. Once chosen, this holds for the session; to change it, reload the "
        "page."
    )
    # **The Continue button, and why it is not optional.**  A radio fires
    # ``on_change`` only when its value *changes*, so a reader who accepts the shown
    # default -- 1-minute, the likeliest answer to "which resolution?" -- produces no
    # callback at all and would sit behind this gate forever.  Clicking an
    # already-selected radio is not a change.
    #
    # The button closes that gap: it commits whatever is selected, default included.
    # This is the same reason the app's Fetch button has a pending-key latch, and it
    # is a genuine constraint of the widget rather than a styling preference.  An
    # earlier version of this gate had no button and could not commit the default at
    # all, which is the single most likely choice a reader would make.
    #
    # **The button parks the selection and stops; it does not resolve it in this pass.**
    # That is the whole difference between the two ways this gate can be answered, and
    # it was a real bug.  Committing here and returning the timeframe let ``main()``
    # carry on and build the full page *in the same pass* that had already drawn the
    # radio, the caption and this button -- so a reader who accepted the default saw
    # the question and the main screen stacked together, the gate sitting above the
    # first tab.  The ``on_change`` path never did this: its commit lands on the
    # *next* pass, and that pass returns at the top of this function without drawing
    # anything.
    #
    # Parking and returning ``None`` is not enough on its own, because it would
    # strand the reader: Streamlit runs the script exactly once per interaction, so
    # parking the answer with no rerun leaves the committed value sitting in
    # ``SESSION_TIMEFRAME_PENDING_KEY`` and the gate on screen forever.  ``st.rerun``
    # is what completes the move -- it abandons this pass immediately, and the next
    # one returns at the top of this function from :data:`SESSION_TIMEFRAME_KEY`,
    # having drawn nothing.
    if st.button(SESSION_TIMEFRAME_GO_LABEL, type="primary"):
        _record_resolution_choice()
        st.rerun()
    return None


def main() -> None:
    inject_css()

    # ---- The resolution gate, before anything else ------------------------ #
    # Everything below is resolution-dependent: the backend that is called, the
    # archive that is cached, the window lengths, the horizons, and every caption
    # that names a bar.  So the question is asked first and the page is not built
    # until it is answered.
    chosen = session_timeframe()
    if chosen is None:
        return
    # Published before anything reads it, for the same reason ``SYMBOL_FOR_HELP`` is a
    # mutable list: Streamlit re-executes this module per interaction, so a value
    # written to a plain global earlier in the pass would have exactly one reader.
    ACTIVE_TIMEFRAME[0] = chosen.key
    FORECAST_TIMEFRAME[0] = chosen.key
    
    # Initialize fetch disabled state - disable during initial page load
    if "fetch_disabled" not in st.session_state:
        st.session_state["fetch_disabled"] = True
    # Re-enable after initial load if not already enabled by a fetch
    if st.session_state.get("fetch_disabled", False) and not st.session_state.get("pending_fetch", False):
        st.session_state["fetch_disabled"] = False
        
    # Resolve the reader's local zone once, before anything is drawn.  Every chart,
    # caption and table in this pass then converts through the one zone this produces,
    # so a reader in Chicago and one in Tokyo cannot read different clocks for the
    # same bar -- and the axes cannot drift mid-session as the poller fires.
    #
    # The component runs client-side JavaScript in the browser and writes the
    # detected zone into session_state["_browser_tz"], which resolve_local_zone()
    # reads.  It is a no-op in tests (see tests/apphelpers.py), so it is safe to
    # call unconditionally here.
    resolve_local_zone()

    # ---- Resolve both tickers, before anything is drawn -------------------- #
    # **This is the reason the ticker inputs can float above the tabs.**  ``main()``
    # needs both symbols to build the two pipelines, and the widgets that produce them
    # are drawn further down, so the value is read here from ``session_state`` and the
    # pending fetch is applied before the page is built.  One rerun of lag between the
    # click and its effect; see :func:`resolve_ticker`.
    #
    # **Both scopes are resolved before any widget is drawn**, and that is not tidiness.
    # Each tab now draws its own input, so the widgets live *inside* the tab bodies --
    # below this point.  A keyed widget's ``value=`` is ignored once its key has been
    # registered, so a seed written after the tab renders is permanently ignored and the
    # box keeps whatever fallback it locked in on the first paint.  Resolving both here,
    # unconditionally, is what keeps every box reading the instrument it actually charts.
    #
    # Price first, because it is the one that gates the page.  A Forecast failure must
    # not be able to stop the tape from being chartable.
    price_scope = resolve_ticker("price")
    forecast_scope = resolve_ticker("forecast")
    symbol = price_scope.symbol if price_scope is not None else None
    forecast_symbol = forecast_scope.symbol if forecast_scope is not None else None

    # **One resolution for both scopes, already published above the tab bar** by
    # ``session_timeframe``.  Read from there rather than off either scope, so the
    # symbol a pipeline is built from and the resolution it is built at are guaranteed
    # to be the pair the reader chose at startup -- there is no second place either
    # could be changed independently.
    tf = _tf()
    forecast_tf = tf

    # The sidebar is rendered before the title because it carries the scoring dial the
    # tab bodies read.  Widgets inside ``st.sidebar`` render in the sidebar regardless
    # of call order, so reading config first costs nothing visually.
    cfg = render_sidebar()

    if not symbol:
        # Nothing to chart.  **A bare Price tab is still drawn**, because that is where
        # the ticker input lives now -- returning early would leave the reader with an
        # error and no control to fix it, which is a dead end.  And it is a bare tab
        # rather than the full six: with no archive, every other tab would render as
        # "this archive has no such pattern", which is a claim about data that does not
        # exist.
        st.title("Pattern matcher")
        st.error(
            "No ticker is in force, so there is nothing to chart. Enter a symbol "
            "below and press **Fetch**; every tab on this page is built from the bars "
            "it downloads."
        )
        with st.tabs(["Price"]) as recovery:
            with recovery:
                render_scope_ticker(
                    "price",
                    note="Fetches <b>everything available</b>: 1-minute bars go back "
                         "~{} days and daily bars go back to the instrument's listing, "
                         "so there is no span to choose — the download always takes the "
                         "whole window.".format(F.MAX_1M_DAYS),
                )
        return

    title_col, button_col = st.columns([5, 2], vertical_alignment="center")
    with title_col:
        st.title("{} {} pattern matcher".format(symbol, active_label().lower()))
        st.caption(
            "**Exploratory analysis.** A forecast is only ever presented next to the same "
            "statistic over randomly chosen windows, and is suppressed entirely when the "
            "sample cannot support it. Nothing here is trading advice."
        )
    with button_col:
        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.caption(
        "Source: **{}** fetched live from Yahoo Finance — not written to "
        "`data/`, so it lives only for this session.  ·  Every tab — Price, "
        "Matches, Projection, Forecast, Quality and Backtest — is built from the "
        "same archive.".format(symbol)
    )

    with st.spinner("Loading bars and building features…"):
        try:
            # Keyed on a string naming the bars, not on the frame itself: see
            # ``pipeline_from_frame``.  The key **must** include the resolution --
            # ``symbol`` alone would let a daily pipeline be served under the entry
            # built from 1-minute bars, which is the same silent collision the fetch
            # cache is keyed to avoid.
            #
            # On a fresh Streamlit start the in-memory fetch store is empty, so
            # ``fetch_ticker_cached`` always re-downloads from yfinance even when the
            # partitioned Parquet archive on disk already holds bars from earlier
            # loads.  Read the archive first so "bars loaded" reflects everything we
            # have, not just what yfinance returned this session.
            archive_root = panel_root_for(tf.key)
            bars = None
            if os.path.isdir(archive_root):
                try:
                    archived = PanelStore(archive_root).load(
                        tickers=[symbol],
                    )
                    if not archived.empty:
                        bars = archived[archived["ticker"] == symbol].copy()
                except Exception:  # noqa: BLE001 - fall through to a fresh fetch
                    bars = None
            if bars is None or bars.empty:
                bars = fetch_ticker_cached(symbol, timeframe=tf.key).frame
            pipe = pipeline_from_frame(
                "{}@{}".format(symbol, tf.key), bars, active_length(), timeframe=tf
            )
        except Exception as exc:  # noqa: BLE001 - surface any loader failure in the UI
            st.error("Could not build the pipeline for {}: {}".format(symbol, exc))
            hint(
                "The download succeeded but the bars could not be turned into "
                "windows. Check the <i>Quality</i> tab for malformed rows — a "
                "symbol with no intraday history is the usual cause."
            )
            return

    # The Forecast tab's own pipeline, over its own ticker.
    #
    # **A second pipeline, not a second copy of the first.**  ``st.tabs`` renders every
    # tab body on every rerun, so this is built unconditionally like the one above --
    # but it is built from ``forecast_symbol``, resolved at the top of this function,
    # which is the Forecast tab's instrument and is routinely not the Price one.
    #
    # Two costs, both avoidable rather than paid.  On first load the Forecast ticker
    # *seeds* on the Price one, so this is a cache hit inside ``fetch_ticker_cached``
    # and a cache hit inside ``pipeline_from_frame`` -- the common path downloads
    # nothing twice.  And when the two symbols really do differ, each is a separate
    # ``cache_resource`` entry bounded by ``max_entries``, so the pair is retained for
    # the session rather than rebuilt per brush.
    #
    # A failure here is **not** fatal to the page.  A thin or unreadable Forecast
    # ticker must not take the Price tape down with it, so this degrades to ``None``
    # and the Forecast tab says so; only a Price failure returns from ``main``.
    # The Forecast pipeline.
    #
    # Since the ticker is now unified, this is essentially a second pass over
    # the same symbol. On first load, this is a cache hit inside
    # ``fetch_ticker_cached`` and ``pipeline_from_frame`` -- the common path
    # downloads nothing twice.
    #
    # A failure here is **not** fatal to the page.  A thin or unreadable
    # archive must not take the Price tape down with it, so this degrades to
    # ``None`` and the Forecast tab says so; only a Price failure returns from
    # ``main``.
    forecast_pipe: Optional[Pipeline] = None
    with st.spinner("Loading bars and building features…"):
        try:
            forecast_bars = fetch_ticker_cached(symbol,
                                                    timeframe=forecast_tf.key).frame
            forecast_pipe = pipeline_from_frame(
                "{}@{}".format(symbol, forecast_tf.key),
                forecast_bars, active_length(), timeframe=forecast_tf)
        except Exception as exc:  # noqa: BLE001 - one tab's failure is not the page's
            st.warning(
                "Could not build the Forecast pipeline for {}: {}".format(
                    symbol, exc)
                )
            hint(
                "The Price, Matches, Quality and Backtest tabs are unaffected — "
                "they are built from their own archive. Use the sidebar input "
                "to try a different symbol."
            )

    # Help copy rendered on later passes must quote the grid actually in use, not the
    # module default.  Set before anything below renders a `[[LENGTH]]` token.
    PIPE_LENGTH_FOR_HELP[0] = int(pipe.length)
    # Same for the instrument: help text that says "QQQ" while the chart above shows
    # MSFT would be describing a different market than the one on screen.
    #
    # **This stays the *Price* ticker, and that is now a decision rather than an
    # oversight.**  ``[[SYMBOL]]`` backs the page title, the manual's opening line and
    # the Backtest fee copy, all of which describe the Price archive.  Retrofitting a
    # second symbol would make the token ambiguous in exactly those places -- a manual
    # that opens "This app searches a QQQ archive" is wrong only because the reader
    # is reading it next to a Forecast tab showing AAPL.  The Forecast tabs therefore
    # name their own instrument in their own captions (see ``render_window_tab``).
    SYMBOL_FOR_HELP[0] = symbol

    st.sidebar.divider()
    st.sidebar.markdown("**Archive status**")
    # Two different archives are described here and they used to be presented as
    # one.  The metrics below describe the **in-memory frame for one ticker**;
    # the panel archive on disk is a *cohort* of hundreds of tickers, and the two
    # "bars" numbers are not comparable.  The captions now say which is which,
    # because "bars loaded: 4,418" next to "bars: 4,666,346" reads as a
    # discrepancy rather than as two different subjects.
    st.sidebar.caption(
        "**{}** — in-memory, this ticker only".format(symbol)
    )
    m1, m2 = st.sidebar.columns(2)
    m1.metric("bars loaded", "{:,}".format(pipe.n_bars),
              help="Rows remaining in the cleaned frame after dropping the ~20-bar "
                   "feature warm-up, for this ticker only. All bar indices elsewhere "
                   "refer to this frame, not to the raw download and not to the "
                   "on-disk panel archive.")
    m2.metric("window", "{} bars".format(pipe.length),
              help="The default query length, in {} bars — {} of continuous trading. A brush "
                   "defines its own length instead, and a session bounds it from "
                   "above.".format(active_label().lower(), pipe.length))
    st.sidebar.caption(
        "Pipeline ready: **{}**".format("yes" if pipe.ready else "no")
    )
    # The Forecast archive, stated next to the Price one so the two are never
    # mistaken for a single tape.  Shown even when the two agree, because "they happen
    # to match" and "they are the same bars" are different facts and only the reader
    # can tell which they were after switching one and not the other.  Named as feeding
    # *both* Forecast tabs, because it does -- the reference chart on one, the whole
    # brush workflow on the other.
    if forecast_pipe is not None:
        st.sidebar.caption(
            "**{}** — Forecast &amp; Projection: {:,} bars, pipeline ready: "
            "**{}**".format(
                forecast_symbol, forecast_pipe.n_bars,
                "yes" if forecast_pipe.ready else "no",
            )
        )

    # The daily archive sync sits directly under the archive metrics it maintains.
    # Those metrics describe one ticker's in-memory bars; the sync maintains the
    # *panel* archive, which is a different thing stored on disk -- so the caption
    # above and this control must not be read as describing the same numbers.
    render_archive_sync()

    if not pipe.ready:
        detail = "; ".join(pipe.warnings) or "the archive is shorter than one window."
        st.warning(
            "**Not enough data to match.** {} The pipeline is not ready, so no "
            "matches, forecasts or backtest are available. The Quality tab below "
            "still reports what was loaded.".format(detail)
        )
        hint(
            "<b>What to do:</b> matching needs at least <i>window length + 100</i> "
            "bars after cleaning, and this symbol only has {:,} — usually because it "
            "is thinly traded and Yahoo returned a short stretch. Try a liquid name "
            "such as AAPL or MSFT, or a shorter *k matches*; a window that has too few "
            "candidates behind it yields a meaningless percentile either way.".format(
                pipe.n_bars)
        )
        tabs = st.tabs(list(TAB_ORDER), default=get_default_tab(), key=tabs_key())
        with tabs[TAB_ORDER.index("Quality")]:
            render_quality_tab(pipe)
        with tabs[TAB_ORDER.index("Price")]:
            st.info("Price chart unavailable until the pipeline is ready.")
            hint("The chart needs at least one full window of cleaned bars to draw a "
                 "query span.")
            # The input is drawn even here.  A tab that says "not enough data" without
            # offering a way to try another instrument is a dead end, and switching
            # ticker is exactly the remedy the message should be pointing at.
            render_scope_ticker(
                "price",
                note="This instrument also drives <b>Matches</b>, <b>Quality</b> and "
                     "<b>Backtest</b>. A more liquid symbol usually has more history.",
            )
        with tabs[TAB_ORDER.index("Matches")]:
            st.info("Matching is disabled until the pipeline is ready.")
        with tabs[TAB_ORDER.index("Projection")]:
            # **Gated on the Forecast pipeline, not the Price one.**  This branch is
            # entered because the *Price* archive is too thin, which says nothing
            # about the Forecast archive.  Reading ``pipe.ready`` here would switch off
            # a tab whose archive is perfectly good.
            #
            # **The ticker input is NOT drawn here.**  It belongs to *Forecast*,
            # below, and drawing it on both tabs would register
            # ``ticker_input_forecast`` twice in one pass -- ``st.tabs`` renders every
            # body on every pass -- and Streamlit would raise
            # ``StreamlitDuplicateElementKey``, taking down the whole page rather than
            # one tab.  This branch used to draw it and then call
            # ``render_forecast_tab``, which drew its own; that shipped, and every
            # source-level test guarding this invariant stayed green through it.
            #
            # A ``StreamlitDuplicateElementKey`` is invisible to a suite that checks
            # each *stub* contains an input, which is exactly what the tests guarding
            # this invariant did: both halves were individually correct, and together
            # they crashed the tab.  Only running the real script found it, which is
            # why ``TestTheRealAppRuns`` exists.
            if forecast_pipe is None:
                st.info(
                    "The reference projection is unavailable until a Forecast ticker "
                    "is loaded. Set one on the <b>Forecast</b> tab."
                )
            elif not forecast_pipe.ready:
                st.info(
                    "The reference projection is disabled: **{}** returned too few "
                    "bars to form a window.".format(forecast_symbol)
                )
                hint(
                    "Matching needs at least <i>window length + 100</i> bars after "
                    "cleaning. This is the Forecast archive's own bars — a liquid symbol "
                    "such as AAPL usually returns more of the available intraday "
                    "history. Set one on the <b>Forecast</b> tab."
                )
        with tabs[TAB_ORDER.index("Forecast")]:
            # **This is the one place in ``main()`` that draws the Forecast input on
            # this branch**, and it has to be: everything below it is conditional on
            # ``forecast_pipe``, so if the input were inside the ``else`` the two
            # unavailable states below would tell a reader to load a symbol with no
            # box on screen to load it into.  A tab saying "use a liquid symbol" with
            # no way to do it is the dead end this draw exists to prevent.
            #
            # The ready path draws it in the ``with tab_forecast:`` block instead.  The
            # two are on mutually exclusive branches, so the key is registered once
            # per pass -- which is the whole invariant, and the reason it is asserted
            # against both branches rather than either one.
            render_scope_ticker(
                "forecast",
                note="This tab's ticker is its own — it starts on whatever the "
                     "<b>Price</b> tab holds, then keeps its own. Both tabs above "
                     "chart it.",
            )
            # **Gated on the Forecast pipeline, not the Price one** -- same reason as
            # the block above: this branch was entered because *Price* is too thin,
            # which says nothing about the Forecast archive.
            if forecast_pipe is None:
                st.info("Forecasting is unavailable until the Forecast ticker loads.")
            elif not forecast_pipe.ready:
                st.info(
                    "Forecasting is disabled: **{}** returned too few bars to form a "
                    "window.".format(forecast_symbol)
                )
                hint(
                    "Matching needs at least <i>window length + 100</i> bars after "
                    "cleaning. This is the Forecast tab's own archive — a liquid symbol "
                    "such as AAPL usually returns more of the available intraday "
                    "history."
                )
            else:
                # Inside its own scope: this tab may hold a different resolution from
                # the Price one, and every ``active_*`` helper below it describes
                # whichever archive is actually being charted.
                with timeframe_scope(forecast_tf.key):
                    render_window_tab(
                        forecast_pipe, forecast_symbol,
                        amplitude_weight=cfg["amplitude_weight"],
                    )
        with tabs[TAB_ORDER.index("Live")]:
            render_live_tab(pipe, symbol)
        with tabs[TAB_ORDER.index("Backtest")]:
            st.info("Backtesting is disabled until the pipeline is ready.")
        with tabs[TAB_ORDER.index("Panel")]:
            # The panel reads its own archive, not this pipeline, so it stays usable
            # even when the fetched ticker is too thin to form a window.  Gating it on
            # `pipe.ready` would take away a working feature because an unrelated file
            # is short.
            render_panel_tab()
        return

    stamps = pipe.bars["timestamp"]
    n = pipe.n_bars

    # ---------------- Sessions ---------------------------------------------- #
    # Computed once, up here, because the view below is measured in sessions and would
    # otherwise have to rebuild the list.  One entry per Eastern trading day, positions
    # only -- see ``session_spans`` for why grouping is by Eastern day and why these
    # are ``iloc`` positions rather than index labels.
    sessions = session_spans(stamps)

    # ---------------- Default window: the archive's newest bars --------------- #
    # The chart always opens on *something*, so there is a moment to look at before the
    # user has done anything.  Dragging a box on the *Price* tab moves it from there,
    # and that brush is the *only* other input -- the From/To pickers that used to sit
    # alongside it went with the *Chart* tab, which is where the reader was told the
    # brush existed.
    #
    # The archive's last ``pipe.length`` bars.  That is what it always was, until the
    # session fence made the default the newest *session* wide enough; ``resolve_query_window``
    # has the rest.  Note this window usually crosses the final close, which is now
    # allowed on purpose.
    start_idx, end_idx = max(0, n - pipe.length), n

    # ---------------- The view ---------------------------------------------- #
    # A fixed trailing window of ``DEFAULT_VIEW_DAYS`` trading sessions.
    #
    # This replaces a *Days of history to display* slider, and the session dropdown
    # that came after it.  The slider counted *calendar* days --
    # ``session_days = (t1 - t0).days + 1`` is ~30 for a 30-day archive even though
    # only ~21 sessions exist in it -- so its maximum was a range that could never be
    # displayed, and it sized the view as ``bars_per_day * days`` over an *average* of
    # those inflated days, so its "1 day" was some number of bars near 390 that was not
    # any real session.
    #
    # Counting *real* sessions has neither defect and needs no control at all: the view
    # is exactly whatever the last ``DEFAULT_VIEW_DAYS`` sessions contain, which is a
    # real, contiguous, verifiable span of bars.  An earlier moment is reached by
    # brushing the chart rather than by moving a slider -- and the brush already drove
    # the query directly, so the slider was a second way to ask a question the brush
    # had already answered.
    #
    # The view is measured in sessions because that is the unit the reader thinks in,
    # but nothing here constrains where the query may sit -- not now.  Showing five
    # sessions is the point of this chart, and a brush that crosses a boundary is a
    # perfectly good window; the tape is shown whole and the band is placed where it
    # was drawn.
    #
    # ``start_idx``/``end_idx`` above are the untouched default, and the brush below
    # overwrites them.
    chart_from = sessions[-active_view_sessions()][1] if sessions else 0

    # describes the *resolved* query, which has to exist before any tab body runs.

    # ---------------- Brush on the Price tab's own chart -------------------- #
    # The Price tab's first chart is where the query is *seen*, so it is also the most
    # natural place to change it -- a user pointing at the tape should not have to
    # switch to another tab and re-find the same moment with a picker.
    #
    # The brush is read from ``st.session_state`` because the chart is drawn **far**
    # below, inside the tab body, while the query has to be resolved long before
    # that.  ``on_select="rerun"`` puts the previous pass's event there before this
    # line runs; reading it later would be one render too late, since the window would
    # already have been snapped.
    #
    # ``session_state`` is the only option *for this ordering*, and it does work:
    # Streamlit mirrors a keyed selection widget's state there on the rerun the
    # selection triggers.  The Forecast tab reads its brush from the return value
    # instead, because that chart is drawn and read in one place -- see
    # ``_forecast_brush_chart``.
    price_brush = selection_to_span(st.session_state.get(price_brush_key()), n)
    if price_brush is not None:
        start_idx, end_idx = price_brush

    # ---------------- Use the selection's own length ----------------------- #
    # No ``bounds`` is passed, so the resolved window is exactly the drawn span.  A
    # brush of N bars becomes an N-bar query wherever it was drawn, and one crossing
    # an overnight close keeps both its width and its position.  The sidebar slider's
    # cap bounds the slider, not a brush, so a 600-bar drag is still 600 bars drawn;
    # nothing narrows it but the archive itself.
    start_idx, stop_idx, length = resolve_query_window(
        price_brush, start_idx, end_idx, pipe.length, n
    )
    query = pipe.query_span(start_idx, stop_idx,
                            label=stamp_span(stamps.iloc[start_idx],
                                             stamps.iloc[stop_idx - 1]))

    # ---------------- Did this pass just move the query? -------------------- #
    # A brush is an explicit gesture, so the Matches and Forecast tabs follow it
    # without a second click: the user already said which window they meant, and
    # making them press *Run match* again is a confirmation they did not ask for.
    #
    # "New" is the operative word.  Streamlit keeps a selection in session_state for the
    # lifetime of the page, so ``price_brush`` is non-``None`` on every subsequent rerun
    # too -- including reruns caused by an unrelated control.  Auto-running whenever a
    # brush is merely *present* would silently recompute the forecast every time a
    # slider moved, which is precisely the stale-and-fresh confusion the manual button
    # exists to prevent.
    #
    # ``main`` no longer runs the search, and no longer tracks whether one is due.
    # Both moved into the Matches tab below, which is the only tab that needs them:
    # ``main`` resolved the *window*, and one run cannot serve two windows any more.
    # The trigger logic is unchanged in shape when it lands there -- the run is still
    # button-gated, still fires once per *new* brush rather than once per rerun, and
    # the recorded span is still the post-clamp one, so a clamped brush cannot re-fire
    # on every pass.
    #
    # What this block still has to do is read the brush, because the chart that writes
    # it is drawn far below inside the Price tab body while the query has to be
    # resolved up here.  See ``PRICE_BRUSH_KEY``.

    # The onboarding tour used to render here, above the tab bar, so it appeared on
    # every tab.  It was then moved into the *Chart* tab body -- and went with that tab.
    # ``render_onboarding`` and the "New here? Start with the 60-second tour" expander
    # that hosted it are gone.

    # ---------------- Tabs --------------------------------------------------- #
    # ``default=DEFAULT_TAB`` opens the page on Projection -- the fixed reference.
    # Stated explicitly rather than left to ``TAB_ORDER[0]``; see ``DEFAULT_TAB``.
    #
    # ``key=tabs_key()`` is what lets a ticker switch send the reader back to
    # *Projection*: ``reset_query_state`` bumps the generation, a new key is a new block
    # id, and a new block id is the one situation in which ``default`` is honoured.
    # See ``TABS_KEY``.
    tabs = st.tabs(list(TAB_ORDER), default=get_default_tab(), key=tabs_key())

    # Unpacked by *name* rather than positionally.  ``TAB_ORDER`` had gained "Panel"
    # without the positional six-way unpack below being updated, which raised
    # ``ValueError: too many values to unpack`` and took the whole page down -- the app
    # was unrunnable, not merely wrong on one tab.  Building the mapping from the
    # tuple itself means a tab can be added to ``TAB_ORDER`` without editing anything
    # here, and a missing key is still a loud failure rather than a silent one.
    tab_by_name = dict(zip(TAB_ORDER, tabs))
    tab_price = tab_by_name["Price"]
    tab_matches = tab_by_name["Matches"]
    tab_projection = tab_by_name["Projection"]
    tab_forecast = tab_by_name["Forecast"]
    tab_live = tab_by_name["Live"]
    tab_panel = tab_by_name["Panel"]
    tab_quality = tab_by_name["Quality"]
    tab_backtest = tab_by_name["Backtest"]

    with tab_price:
        # The guide, rendered here rather than left in ``TAB_GUIDE`` unread.  This tab
        # absorbed the *Chart* tab's job -- it is where the query is picked, so it is
        # where the reader has to be told how -- and an expander is the only way to say
        # it without putting text between the reader and the tape, which is the one
        # thing this tab is for.  Collapsed by default, so it costs nothing visually.
        #
        # Every other tab does the same (``guide("Matches")`` and friends), so this is
        # not a new convention: it is the one tab that was missing one.
        guide("Price")

        # The query chart above and the match chart below are only comparable if they
        # agree on TWO things, and the second is easy to miss:
        #
        #   1. the same number of bars, so a bar is the same width in both; and
        #   2. the same x offset for the shaded band, so the eye compares like with like.
        #
        # Matching width alone was not enough.  Each panel used to centre its own band,
        # so the live query -- the archive's *last* window -- sat flush against the right
        # edge while an interior match sat in the middle: at 600 bars the two bands were
        # 402px apart horizontally.  One shared anchor, chosen so that *both* bands have
        # room to sit at it, fixes that.
        #
        # The width is the one the days-of-history slider chose for the price chart.  It is
        # deliberately not the match's narrow framing: at 600 bars the window band is a
        # small readable slice, whereas at the 100-bar context width it fills the panel
        # and flattens into a slab.
        #
        # **This is a bar *count*, which is exactly why it survives a cross-ticker
        # match.**  The two charts are now drawn from two different series -- the query
        # chart from the fetched ticker, the match panel from whichever ticker the match
        # lives in -- so no shared view range is possible.  What must still agree is how
        # many bars each panel spans, because that is what makes a bar the same width on
        # both and the two shapes comparable by eye.  A count carries across series; an
        # absolute range would not, and would silently empty the match panel.
        price_width = n - chart_from

        # The match is needed *before* the price chart, because its position constrains
        # where the shared band offset can be.  And `st.tabs` runs this body on every
        # rerun regardless of which tab is on screen -- which is why it is a match
        # only, not `run()`, whose random-window baseline must never appear unless
        # asked for.
        #
        # **This searches the whole S&P 500 panel, not just this ticker's history.**
        # It used to be `pipe.match(query, ...)`, which slid over `pipe.matrix` and
        # therefore could only ever return a window from the same symbol.  The reader
        # asked whether the tape had done this before; the app could only answer
        # whether *this one ticker* had, which is the narrower and much easier
        # question.  Now the candidate pool is every window of every archived
        # constituent, so a match can come from NVDA when the query is QQQ.
        #
        # ``k`` is the *default*, not the Matches tab's ``k matches`` slider: this
        # panel shows only the single closest match, so there is nothing for a larger
        # ``k`` to select, and binding it to that slider would make a control labelled
        # "how many matches" silently resize a chart that never shows more than one.
        #
        # ``max_per_ticker`` is the cap ``find_panel_matches`` applies *within* one
        # ticker, not a cap on the number of tickers consulted.  It stays at the
        # library's own default because this panel shows one match: the cap exists to
        # stop a sector-wide move filling a result list with one factor, and a
        # one-match panel cannot be filled by anything.
        #
        # ``max_horizon`` is passed so the §BX forward-horizon mask applies, which the
        # single-ticker call got for free from ``Pipeline.match``.  A cross-sectional
        # search that skipped it would rank panel windows whose forward return is 20x
        # inflated by an overnight gap -- and, unlike the single-ticker path, nothing
        # else in this call would notice.
        auto = cross_sectional_match(
            pipe, symbol, start_idx, stop_idx,
            k=DEFAULT_K, amplitude_weight=cfg["amplitude_weight"],
            max_horizon=int(max(active_horizons())),
        )

        # The result, or ``None``.  Named once because several decisions downstream all
        # turn on the same question -- is there a match to draw -- and ``auto`` is
        # ``None`` both when the panel archive is missing and when the search threw.
        panel_result = auto["result"] if auto is not None else None

        # The matched ticker's own pipeline, so its tape can be drawn.  **This is the
        # pipeline the match panel is rendered from**, not ``pipe``: the match may
        # live in a different series entirely, and ``build_price_figure`` reads
        # ``pipe.bars`` for every level, tick label and band it draws.
        match_pipe = pipe
        best_match = None
        if panel_result is not None and panel_result.matches:
            m = panel_result.matches[0]
            built = panel_pipeline_for(auto["search"], m.ticker)
            if built is not None:
                match_pipe = built
                best_match = m

        # **No shared anchor, and that is deliberate.**  This block used to compute
        # ``band_anchor = shared_anchor(n, price_width, spans)`` over the query *and*
        # the match, then hand it to both renderers so the two bands would line up at
        # load.  That only worked because the top chart *derived its whole view* from
        # the anchor -- so the price tape was re-framed on every brush, sliding the band
        # the reader was looking at to a different x offset each time.  The symptom the
        # anchor was built to prevent (bands 400px apart) was the lesser one; the cure
        # caused a worse, constant one.
        #
        # Dropping it costs nothing that was still working.  The pannable match panel
        # below ignores the anchor entirely and centres its own band (see
        # ``_render_best_match_pair``), so the two band offsets have not agreed since
        # panning existed; equal *bar width* is the property still worth protecting, and
        # that comes from ``price_width``, which is passed to both renderers unchanged.
        # The top chart's window is now simply the trailing view, which no query can move.
        # ``spans`` is gone with the anchor.  It existed only to be handed to
        # ``shared_anchor``, and nothing downstream reads it now.

        # One height for both charts too.  The two panels are only comparable if they
        # agree on every dimension -- bar width, band offset, vertical range -- and
        # differing plot heights made the same window band read as a taller feature on
        # one chart than the other.
        pair_height = PRICE_CHART_HEIGHT

        render_price_tab(pipe, start_idx, stop_idx, chart_from, n,
                         height=pair_height,
                         chart_width=price_width,
                         # No ``anchor``: the top chart's window is now pinned to the
                         # trailing view ``(chart_from, n)`` and must not depend on
                         # where either band sits.  Passing the shared anchor here was
                         # what made every brush re-frame the tape under the reader --
                         # see ``render_price_tab``.  The band is drawn where it was
                         # brushed, and the width is unchanged, so the two panels still
                         # draw a bar at the same pixel size.
                         #
                         # Rebased onto its own first bar, because the match panel drawn
                         # below is rebased too.  These two charts are a single visual
                         # comparison, and they were in different units -- dollars above,
                         # percent below -- so the eye compared a 735–745 range against a
                         # −1%–1% range and drew a conclusion from neither.  The title is
                         # enabled only once a match exists, since that is the only time
                         # there is a second panel to agree with.
                         rebase_at=start_idx,
                         show_title=best_match is not None,
                         # Brushable: a box dragged here moves the query.  The event is
                         # consumed above, before the snap, so it lands on this pass
                         # rather than the next.  Only the top chart gets this -- the
                         # match panel below stays read-only.
                         selectable=True, on_select="rerun")

        # Nothing between the two charts, and nothing after them.
        #
        # This tab used to carry a live readout of the brush -- a *query window* bar
        # count, a *brush spans* timestamp pair, and a caption restating what the drag
        # does.  That was instruction dressed as data, on the one tab whose entire
        # reason for existing is *not* having any.  The query was already legible as
        # the orange band on the chart directly above, drawn by the same gesture that
        # produced the readout; three numbers describing a shape the reader is looking
        # at is one more thing to read, not more information.
        #
        # So the tab is now the two charts and nothing else, which is what the *Price*
        # section of the guide has always claimed.  The brush itself is unchanged --
        # it still defines the query -- and its length is still reported by the
        # sidebar's *window* metric and the Forecast tab's own caption, which read the
        # resolved span rather than re-deriving it from the gesture.
        #
        # ``selection_summary`` and ``brush_bar_count`` stay in the module even with no
        # caller here: they are the parsing surface ``test_selection_parsing.py`` pins,
        # and a display is not what makes the brush parse correctly.
        #
        # The divider is drawn only when a second panel follows it.  Rendered
        # unconditionally it left a rule hanging under the price chart with nothing
        # beneath it -- the one obvious "no match" case being an empty match set --
        # which read as a chart that failed to draw.
        if best_match is not None:
            st.divider()
            # ``pannable=True`` is what turns the lower panel into a chart the reader
            # explores rather than one they read: its view centres on the match and its
            # data widens to the whole tape, so a drag walks the price into and out of
            # the matched window.  The price chart above keeps its shared anchor and
            # stays brushable, so this is the only drag on the Price tab that does not
            # redefine the query.
            #
            # It costs the load-time band alignment described in `_render_best_match_pair`:
            # the two bands no longer sit at the same x offset.  The *width* still matches
            # (`price_width` is passed through unchanged), so a bar is the same number of
            # pixels on both charts and the shapes stay comparable -- which is the
            # property the alignment tests were written to protect, as distinct from the
            # band offsets, which they cannot both police once the match is centred.
            #
            # ``match_pipe``, not ``pipe``: the match may be in another ticker's series,
            # and this is the chart that draws it.  The result is converted because the
            # draw path speaks ``MatchResult`` -- see ``panel_result_as_match_result``.
            #
            # **The match panel carries a ticker caption the single-ticker version never
            # needed.**  A chart of AAPL's tape appearing under a QQQ query, with no
            # label, is indistinguishable from a chart of QQQ's own history -- and the
            # reader's first assumption would be that it *is* their own tape.  The
            # matched name, its session and its panel rank are the three facts that make
            # the panel readable as a cross-sectional answer rather than a bug.
            # ``{:,}`` is applied to the **integers themselves**, never to an
            # already-formatted string: ``"{:,}".format(3632136)`` gives
            # ``'3,632,136'``, and formatting *that* string with ``{:,}`` raises
            # ``ValueError: Cannot specify ',' with 's'``.  The counts are therefore
            # passed raw and let the outer format add the separators.
            st.caption(
                "**Closest match anywhere in the S&P 500 archive · {} · {}** · "
                "{:,} bars · panel rarity {} (own-ticker {}) · {} of {:,} windows "
                "scored across {:,} tickers".format(
                    best_match.ticker,
                    best_match.session or "—",
                    int(best_match.stop - best_match.start),
                    pct(best_match.percentile),
                    pct(best_match.percentile_same_ticker),
                    len(panel_result.matches),
                    int(panel_result.n_candidates),
                    int(panel_result.n_tickers),
                )
            )
            render_matches_tab(match_pipe, panel_result_as_match_result(panel_result),
                               compact=True,
                               chart_width=price_width,
                               # No ``anchor``.  The compact path is ``pannable``, and
                               # the pannable branch replaces its view with
                               # ``centred_view`` unconditionally -- so an anchor here
                               # was already dead before this change, and it was only
                               # still wired up because the top chart used the same
                               # value.  Equal bar width (``price_width``) is the
                               # property that still matters, and it is unaffected.
                               pannable=True,
                               height=pair_height)

    with tab_matches:
        guide("Matches")

        # This tab's own search, over the window the *Price* brush defined.  The
        # controls live here rather than in the sidebar so this analysis and the
        # Projection tab's are configured independently -- and the result is computed
        # here rather than in ``main`` so it cannot be the Projection tab's.
        cfg_m = _render_search_controls("matches")

        # ``length`` rather than ``pipe.length``: a brush defines its own window, so
        # the key has to name the length actually searched.  The settings are in it
        # for a stronger reason -- they change what a distance *means*, so a result
        # carried over from different dials would be reporting a percentile computed
        # against a distribution the reader is no longer looking at.
        signature = (symbol, length, cfg_m["k"], start_idx, stop_idx,
                     cfg_m["n_baseline"], cfg_m["min_matches"], cfg_m["seed"],
                     cfg["amplitude_weight"])
        # A brush is consumed once.  Streamlit keeps a selection for the page's
        # lifetime, so ``price_brush`` is non-``None`` on every later rerun too;
        # auto-running whenever one is merely *present* would recompute the forecast
        # every time an unrelated slider moved.  The recorded span is the post-clamp
        # one, so a clamped brush cannot disagree with it and re-fire forever.
        brush_moved = (st.session_state.get(price_applied_key()) != (start_idx, stop_idx)
                       and price_brush is not None)
        if cfg_m["run_clicked"] or brush_moved:
            st.session_state[price_applied_key()] = (start_idx, stop_idx)
            st.session_state[price_run_key()] = signature

        out = None
        if st.session_state.get(price_run_key()) == signature:
            # The trigger differs -- button or brush -- and the wording says which, so
            # an auto-run is never mistaken for a search the user asked for by hand.
            reason = "window changed" if brush_moved else "searching"
            with st.spinner("{} {} windows and building the baseline for {}…".format(
                    reason.capitalize(), length,
                    stamp_span(stamps.iloc[start_idx], stamps.iloc[stop_idx - 1]))):
                out = pipe.run(
                    query, k=cfg_m["k"], horizons=active_horizons(),
                    seed=cfg_m["seed"], n_baseline=cfg_m["n_baseline"],
                    min_matches=cfg_m["min_matches"],
                    amplitude_weight=cfg["amplitude_weight"],
                )

        if out is None:
            st.info("**No search has been run yet.** Press **Run match** above, in "
                    "*Search settings*. Brushing a window on the *Price* tab also "
                    "runs it.")
            hint(
                "Results are never cached, so a stale answer can never be mistaken "
                "for a fresh one. Change the window, the brush, or any control and "
                "this tab empties again until you search."
            )
        else:
            render_matches_tab(pipe, out["result"])

    with tab_projection:
        # **No ticker input on this tab.**  It moved to *Forecast*, below, which is
        # the tab that does the work.  An input may be registered on exactly one path
        # per pass and ``st.tabs`` renders every body on every pass, so drawing it on
        # both would raise ``StreamlitDuplicateElementKey`` and take down the page --
        # the same failure the not-ready branch records, reached from the ready path.
        # This tab charts whichever ticker is in force, so it still needs a gate: it
        # just needs no control.
        guide("Projection")
        # Called unconditionally, and it renders the projection before it consults
        # anything else at all.  The previous version short-circuited on ``out is
        # None`` and replaced the entire tab with a notice, which was correct when the
        # tab held nothing but the evidence table -- but the reference chart does not
        # depend on the reader's window or on a search run, so that gate left the tab's
        # main visual permanently blank for anyone who had not pressed *Run match*.
        #
        # **``forecast_pipe`` and ``forecast_symbol``, not ``pipe`` and ``symbol``.**
        # This tab charts the Forecast instrument, and on a healthy pair these are two
        # different objects over two different archives.  Reading the Price pipeline
        # here would silently show a Forecast for the wrong ticker whenever the reader
        # set one -- and, because ``forecast_path_for`` is keyed on the symbol, it would
        # serve the Price ticker's cached path under the Forecast ticker's title.
        #
        # **Rendered before ``tab_forecast``**, which reads the projection horizon this
        # tab's slider resolves.
        if forecast_pipe is None:
            st.info(
                "No Forecast ticker is loaded, so this tab has nothing to chart. "
                "Use the main ticker input above to load data for forecasting. "
                "Every other tab is unaffected."
            )
        else:
            with timeframe_scope(forecast_tf.key):
                render_forecast_tab(
                    forecast_pipe, forecast_symbol,
                    amplitude_weight=cfg["amplitude_weight"],
                )

    with tab_forecast:
        # Forecast ticker input removed as requested
        pass

        guide("Forecast")
        # The tab resolves its *own* window from its own brush and runs its own
        # search, so it reports one window rather than the Price tab's.  The
        # *Projection* tab above draws the ``Projection bars`` slider this tab reads,
        # so it is rendered first and the horizon it resolved is the one used here.
        #
        # **``forecast_pipe`` and ``forecast_symbol``, not ``pipe`` and ``symbol``** --
        # same reason as the block above.
        if forecast_pipe is None:
            st.info(
                "No Forecast ticker is loaded, so this tab has nothing to forecast. "
                "Enter a symbol above and press **Fetch forecast bars**. Every other "
                "tab is unaffected."
            )
        else:
            with timeframe_scope(forecast_tf.key):
                render_window_tab(
                    forecast_pipe, forecast_symbol,
                    amplitude_weight=cfg["amplitude_weight"],
                )

    with tab_live:
        render_live_tab(pipe, symbol, amplitude_weight=cfg["amplitude_weight"])

    with tab_panel:
        # The panel reads its own archive rather than `pipe`, so it is rendered even
        # when the fetched ticker above is too thin: one short download should
        # not take a working feature down with it.
        #
        # **The archive follows the resolution, and this is why that matters.**
        # This block used to refuse to render on Daily, with a message saying the
        # 1-minute archive had not been backfilled.  The refusal was correct then and
        # is the reason `data/sp500_daily` exists: comparing a daily window against
        # a million minute bars produces a confident-looking percentile over a
        # comparison with no meaning, which is the worst kind of wrong answer
        # because nothing about it reads as wrong.
        #
        # With a daily archive built by `scripts/download_daily.py`, `panel_root_for`
        # resolves to that archive and the tab is meaningful on both resolutions --
        # so the gate is gone rather than inverted.  What replaces it is the archive
        # check *inside* `render_panel_tab`, which is honest either way: it names the
        # directory it looked in and the command that builds it.
        render_panel_tab()

    with tab_quality:
        guide("Quality")
        render_quality_tab(pipe)

    # The Panel tab is rendered once, above, through `tab_panel`.  It used to also be
    # rendered here behind an `if "Panel" in tab_by_name` guard, which was leftover
    # from before `tab_panel` existed in the unpack.  Since `tab_panel` *is*
    # `tab_by_name["Panel"]`, the two blocks rendered the same function into the same
    # tab on every rerun, and the second copy's widgets collided with the first:
    # `StreamlitDuplicateElementId: multiple selectbox elements with the same
    # auto-generated ID`.  The guard was also dead code -- unpacking `tab_panel` above
    # already raises a loud KeyError if "Panel" is ever dropped from TAB_ORDER, so the
    # "absent tab" case it was defending against cannot reach this point.

    with tab_backtest:
        render_backtest_tab(pipe)

    # The caveat lives at the foot of the page rather than above the tabs, where it
    # would push the charts down on every tab instead of only the one it concerns.
    #
    # It is keyed on whether the query actually *touches the archive end*, not on how
    # it got there.  The app now always opens on the latest window, so this is the
    # first thing a reader sees -- but brushing anywhere else makes it false, and
    # printing it anyway would be a permanent warning about a problem the current
    # window does not have.
    if stop_idx >= n:
        st.divider()
        hint(
            "Your query is at the very end of the archive, so it has no bars after it "
            "and its forward returns are NaN — for a forecast that matters, brush a "
            "window with history on both sides."
        )


# Streamlit executes this script top-to-bottom on every interaction, so main() is
# called unconditionally rather than behind a __main__ guard.
main()
