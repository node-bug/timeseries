"""Matrix-profile primitives backed by STUMPY.  Implements PLAN.md §C Stage 1, §S, §U.

Replaces the interpreted ``n x m`` scoring loop in :mod:`timeseries.matching` with
STUMPY's JIT-compiled, SIMD-accelerated primitives.  Three corrections to the plan's
prior assumptions are baked in here, because they are properties of the library rather
than of this project:

1.  **``stumpy.match`` is not the right Stage-1 primitive.**  PLAN.md §V lists it as
    query-vs-archive returning "k nearest matches".  The real signature is
    ``match(Q, T, max_distance, max_matches, ...)`` -- it returns every occurrence
    within a *distance threshold*, so how many come back depends on the data, not on a
    caller-chosen ``k``, and it cannot rank against a known candidate population either,
    which §E requires.  :func:`distance_profile` uses ``stumpy.mass`` instead, which
    returns the *full* distance profile -- giving both an exact ranking and a
    population to compute percentiles against.

2.  **The exclusion zone is ours, not STUMPY's.**  STUMP's default exclusion zone is
    ``m/4``, which suppresses only the trivial match and still returns the query's
    immediate neighbours -- windows sharing ``m-1`` bars with it, which are not
    independent evidence.  §M wants a full window length.  :func:`excl_zone_for`
    encodes that margin, and the actual filtering stays in :mod:`timeseries.matching`
    so one implementation of §M governs every method.

3.  **GPU is unavailable here.**  §W already rules out ``gpu_stump`` (needs CUDA; no
    Metal backend exists), so nothing in this module reaches for it.

STUMPY is a **required** dependency, not an optional accelerator.  The package has a
single scoring path, so there is no second implementation for this module to degrade
to; a missing stumpy is an installation failure that must surface at import rather
than become a slow, differently-ranked fallback discovered mid-query.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

__all__ = [
    "STUMPY_AVAILABLE",
    "CANONICAL_LENGTHS",
    "ProfileResult",
    "warm_up",
    "excl_zone_for",
    "distance_profile",
    "profile",
    "top_matches",
    "discord_report",
    "numba_guard",
]

try:
    import stumpy
except Exception as _exc:  # pragma: no cover - an installation failure, not a branch
    raise ImportError(
        "STUMPY is required: it is the only scoring engine in this package "
        "(PLAN.md §C/§S). Install it with `pip install stumpy` -- it needs "
        "Python 3.10+.\n"
        f"Underlying import error: {_exc!r}"
    ) from _exc

# Kept as a module attribute so callers and tests can assert the dependency is present
# without re-implementing the check.
STUMPY_AVAILABLE = True


# --------------------------------------------------------------------------- #
# Numba thread-safety
# --------------------------------------------------------------------------- #
# STUMPY's kernels are ``@numba.njit(parallel=True)``.  Numba's default threading
# layer on macOS is ``workqueue``, which is explicitly *not* threadsafe: if two Python
# threads enter a parallel region at the same time, the layer detects it, prints
# "Concurrent access has been detected", and aborts the whole process with SIGABRT.
# There is no Python traceback and no ``st.exception`` -- the server simply vanishes,
# which is why this presents as "the app crashed".
#
# Streamlit runs each session on its own script thread, so two browser tabs, or a tab
# plus a ``st.cache_resource``/background warm-up, is enough to hit it.  Setting
# ``NUMBA_NUM_THREADS=1`` does *not* help (the parallel region still launches); the
# threadsafe ``tbb`` layer is not installed in this venv.  Serialising the entry
# points is the fix that works with what is actually installed.
#
# The guard is module-level and re-entrant on purpose: :func:`warm_up` and the
# multi-channel loop in :func:`distance_profile` both call back into it, and RLock
# keeps a nested acquisition from self-deadlocking.
_NUMBA_LOCK = threading.RLock()


@contextmanager
def numba_guard() -> Iterator[None]:
    """Serialise access to STUMPY's Numba-parallel kernels across Python threads.

    Everything that reaches ``stumpy`` must go through this.  Holding the lock costs
    only the wall-clock overlap between concurrent queries -- the kernels themselves
    still use all cores -- so the single-user case is unaffected.
    """
    with _NUMBA_LOCK:
        yield


# §T option 1: a small canonical grid, so a handful of cached profiles covers the UX.
# Lengths outside this grid fall back to the per-query ``mass`` path, which is why no
# single profile is required to serve every selection.
CANONICAL_LENGTHS = (30, 60, 120)


def warm_up() -> None:
    """Force STUMPY's Numba JIT so the first real query does not stall the UI.

    §X: numba compiles on first call, costing seconds.  Calling this at startup moves
    that cost to process start instead of the user's first match.  Failures are
    swallowed -- a warm-up problem must never stop the app from starting.
    """
    try:
        rng = np.random.default_rng(0)
        with numba_guard():
            stumpy.mass(rng.normal(size=16), rng.normal(size=256), normalize=True)
    except Exception:  # pragma: no cover
        pass


def excl_zone_for(length: int, *, strict: bool = True) -> int:
    """Trivial-match exclusion radius, in bars, for a window of ``length``.

    §V flags that STUMP's default is ``m/4``.  §M wants a full window length so that
    *neighbouring* windows are suppressed, not just the query.  ``strict=True`` gives
    the plan's margin; ``strict=False`` reproduces the library default for comparison.
    """
    if length <= 0:
        return 0
    return int(length) if strict else max(1, int(np.ceil(length / 4)))


# --------------------------------------------------------------------------- #
# Stage 1 -- the online path
# --------------------------------------------------------------------------- #
def distance_profile(
    query: np.ndarray,
    series: np.ndarray,
    *,
    query_idx: Optional[int] = None,
) -> np.ndarray:
    """Distance from ``query`` to every length-``m`` window of ``series``.

    This is the Stage-1 coarse score of §C, and it is the *whole* candidate
    distribution -- not just the survivors -- so §E percentiles stay honest.

    Parameters
    ----------
    query
        ``(m,)`` or ``(m, d)``.  A 2-D query is treated as ``d`` independent per-channel
        windows and combined as ``sqrt(sum_c d_c**2)``, which is exactly the flattened
        Euclidean norm over the per-window-z-scored feature matrix.
    series
        ``(n,)`` or ``(n, d)``, matching ``query``'s dimensionality.
    query_idx
        Position of ``query`` inside ``series``, when it came from there.  Passed through
        so STUMPY sets the self-distance to exactly zero rather than recomputing it; the
        query and its neighbours are filtered by the caller regardless (§M).

    Returns
    -------
    ndarray
        ``(n - m + 1,)`` distances; index ``i`` corresponds to window start ``i``.

    Notes
    -----
    **Constant windows.**  A constant window has no standard deviation, so its
    z-normalised form is undefined.  STUMPY's normalised distance handles this by
    scoring such a window as all-zeros, and :func:`timeseries.matching.zscore` uses the
    same convention, so a flat window stays a valid, simply-uninteresting candidate on
    both sides and never leaks a NaN into the profile.

    **The distance is amplitude-blind, and that is now a documented consequence rather
    than an accident.**  STUMPY z-scores each window, so every window is forced to unit
    variance and a violent one is exactly as far from a flat one as it is from another
    window of its own size.  A shape-only question ("when did this wiggle happen?")
    wants that.  A price question ("when did price move like this?") does not, because
    a dead-calm window and a violent one are visually opposite.  :func:`amplitude_profile`
    measures that missing component; combining the two is the caller's decision, and
    :func:`timeseries.matching.find_matches` is where it is made.
    """
    q = np.asarray(query, dtype=float)
    t = np.asarray(series, dtype=float)
    if q.ndim not in (1, 2) or t.ndim not in (1, 2):
        raise ValueError(f"query/series must be 1-D or 2-D, got {q.ndim}/{t.ndim}")
    if q.shape[0] == 0:
        return np.empty(0, dtype=float)
    if not np.isfinite(q).all():
        raise ValueError("query contains non-finite values")
    if not np.isfinite(t).all():
        raise ValueError("series contains non-finite values")

    if q.ndim == 1 and t.ndim == 1:
        with numba_guard():
            d = stumpy.mass(q, t, normalize=True, query_idx=query_idx)
        return np.asarray(d, dtype=float)

    q2 = q if q.ndim == 2 else q[:, None]
    t2 = t if t.ndim == 2 else t[:, None]
    if q2.shape[1] != t2.shape[1]:
        raise ValueError(f"channel mismatch: query {q2.shape[1]} vs series {t2.shape[1]}")

    # Multi-channel: one MASS pass per channel, combined as a root-sum-of-squares.
    acc = None
    with numba_guard():
        for c in range(q2.shape[1]):
            d = np.asarray(
                stumpy.mass(
                    np.ascontiguousarray(q2[:, c]),
                    np.ascontiguousarray(t2[:, c]),
                    normalize=True,
                    query_idx=query_idx,
                ),
                dtype=float,
            )
            acc = d**2 if acc is None else acc + d**2
    return np.sqrt(acc)


# --------------------------------------------------------------------------- #
# Amplitude
# --------------------------------------------------------------------------- #
def amplitude_profile(series: np.ndarray, window_length: int) -> np.ndarray:
    """Per-window realised move, in log units, for every window of ``series``.

    The distance in :func:`distance_profile` cannot see this: STUMPY z-scores each
    window, so a window that moved 0.01% and one that moved 0.8% are both forced to unit
    variance and end up the same distance apart.  On a price chart those are visually
    opposite -- a flat line and a swing -- so a match chosen purely on the normalised
    distance can be a shape the reader would never have picked.

    This returns the signed net move over each window, which is the component the
    normalised distance discards.  It is a windowed **sum**, so the input must be a
    *difference* series whose units accumulate: pass **raw log returns**, not
    ``return_z``.  Summing a rolling-z rescales every bar by its local volatility, so a
    3-sigma move in a quiet hour and a 1-sigma move in a frantic one both sum to a
    similar total, and the resulting penalty *rewards* wild windows for having been
    wild.  The error is not small: measured on the live archive, summing ``return_z``
    rated a window that moved 3× further than the query as the better match.

    Parameters
    ----------
    series
        ``(n,)`` or ``(n, d)``.  Channel 0 is used.
    window_length
        ``m``; the window spans ``[i, i + m)``.

    Returns
    -------
    ndarray
        ``(n - m + 1,)`` of net log moves.  ``nan`` for any window that *contains* a
        non-finite bar, so callers can count valid windows instead of reading a zero as
        a real "no move" observation.
    """
    s = np.asarray(series, dtype=float)
    if s.ndim == 2:
        if s.shape[1] == 0:
            raise ValueError("series has no channels")
        s = s[:, 0]
    s = s.ravel()
    m = int(window_length)
    if m <= 0 or m > len(s):
        raise ValueError(f"invalid window length {m} for series of length {len(s)}")
    # NaNs are tolerated rather than rejected, because a leading one is unavoidable:
    # the first bar of any return series has no previous close, and dropping warm-up
    # bars upstream leaves it wherever it lands.  A NaN invalidates every window that
    # *contains* it and no others -- the running total is still exact for any window
    # starting after the last bad bar -- so one leading NaN costs at most one window
    # rather than the whole profile.  Rejecting the call instead would make the
    # amplitude term unusable on exactly the frames this package builds.
    bad = ~np.isfinite(s)
    if bad.all():
        raise ValueError("series contains no finite values")
    if bad.any():
        # A window [i, i+m) is unusable iff it contains a bad bar.  A prefix count of
        # bad bars makes this O(n) rather than O(n*m).
        prefix = np.concatenate(([0], np.cumsum(bad)))
        contains_bad = (prefix[m:] - prefix[:-m]) > 0
    else:
        contains_bad = np.zeros(len(s) - m + 1, dtype=bool)
    # Bad bars contribute 0 to the running total so the sum itself stays finite, and
    # are excluded from every window result by the mask above.
    csum = np.concatenate(([0.0], np.cumsum(np.where(bad, 0.0, s))))
    total = csum[m:] - csum[:-m]
    total[contains_bad] = np.nan
    return total



# --------------------------------------------------------------------------- #
# Batch profile (cacheable, §S)
# --------------------------------------------------------------------------- #
@dataclass
class ProfileResult:
    """A cached matrix profile plus the lookups the UI needs.

    Attributes
    ----------
    profile
        ``(n - m + 1, 4)`` exactly as :func:`stumpy.stump` returns it.  **Column 0 is
        the distance, column 1 the nearest-neighbour index**, columns 2-3 the left and
        right indices that §U1's chains consume.  PLAN.md §S states "column 1 is the
        nearest-neighbour distance"; that is wrong and was corrected against the
        installed library (see §BR).  Row ``i`` corresponds to window start ``i``, so a
        start index is simply ``arange(n_windows)``.
    length
        The ``m`` this profile was built at.  A profile is valid only for this length,
        which is the §T constraint made explicit rather than discovered at query time.
    discords
        Indices into ``profile`` ordered by *descending* distance -- the anomalies.
    """

    profile: np.ndarray
    length: int
    series_length: int
    discords: Optional[np.ndarray] = None

    @property
    def starts(self) -> np.ndarray:
        """Window start index of each profile row.

        A matrix profile row is positional: row ``i`` *is* the window starting at ``i``.
        (There is no start column to read -- see the class docstring.)
        """
        return np.arange(self.n_windows, dtype=np.int64)

    @property
    def distances(self) -> np.ndarray:
        """Nearest-neighbour distance for every window (profile column 0)."""
        return np.asarray(self.profile[:, 0], dtype=float)

    @property
    def indices(self) -> np.ndarray:
        """Nearest-neighbour index for every window (profile column 1)."""
        return np.asarray(self.profile[:, 1], dtype=np.int64)

    @property
    def n_windows(self) -> int:
        return int(len(self.profile))

    def as_dict(self) -> dict:
        return {
            "length": self.length,
            "n_windows": self.n_windows,
            "series_length": self.series_length,
        }


def profile(series: np.ndarray, length: int) -> ProfileResult:
    """Compute the full matrix profile of ``series`` at window ``length``.

    §S makes this a *batch* artefact, never an on-demand computation: at ~490k bars it
    costs minutes on a laptop, where the per-query :func:`distance_profile` costs
    milliseconds.  Caching one profile per canonical length is what makes it affordable.
    """
    series = np.asarray(series, dtype=float).ravel()
    if length <= 0 or length > len(series):
        raise ValueError(f"invalid window length {length} for series of length {len(series)}")
    if not np.isfinite(series).all():
        raise ValueError("series contains non-finite values")

    with numba_guard():
        mp = stumpy.stump(series, length)
    res = ProfileResult(
        profile=np.asarray(mp, dtype=float), length=length, series_length=len(series)
    )
    # Discords are ranked from the distance column directly.  STUMPY ships no
    # `discords()` helper in 1.14.x, and the ranking is a plain argsort -- there is no
    # reason to depend on a function that may not exist in the installed version.
    d = res.distances
    finite = np.where(np.isfinite(d))[0]
    if finite.size:
        res.discords = finite[np.argsort(-d[finite], kind="stable")].astype(np.int64)
    else:  # pragma: no cover - requires an entirely degenerate profile
        res.discords = None
    return res


def top_matches(
    res: ProfileResult,
    *,
    k: int = 10,
    nms_separation: Optional[int] = None,
) -> list:
    """Nearest-neighbour matches from a cached profile, after suppression.

    Reads column 1 rather than re-scoring, so re-ranking for the UI is cheap.  The
    percentile is taken against the full profile, keeping §E's "report a rank, not a
    bare distance" rule intact on the cached path too.  Any non-finite distance (a
    window shorter than the profile, or a degenerate input) is kept out of the
    percentile population, so one bad window cannot inflate every reported rank.
    """
    from .matching import Match, non_max_suppression

    if res.n_windows == 0:
        return []
    sep = res.length if nms_separation is None else nms_separation
    starts, dists = res.starts, res.distances
    finite = np.isfinite(dists)
    population = dists[finite]
    chosen = non_max_suppression(starts, np.where(finite, dists, np.inf), sep, k)

    out = []
    for i in chosen:
        d = float(dists[i])
        s = int(starts[i])
        pct = float((population <= d).mean()) if population.size and np.isfinite(d) else float("nan")
        out.append(Match(start=s, stop=s + res.length, distance=d, percentile=pct))
    return out


# --------------------------------------------------------------------------- #
# Discords
# --------------------------------------------------------------------------- #
def discord_report(series: np.ndarray, length: int, *, top: int = 20) -> dict:
    """Rank the most *unlike* windows in the archive.

    §S: the same pass that answers "what matched?" also answers "what did not?".  The
    largest distances are anomalous stretches, which doubles as the archive-quality
    signal §U4 asks for -- and costs nothing extra, since the profile already exists.
    """
    res = profile(series, length)
    starts, dists = res.starts, res.distances
    if res.discords is None or len(res.discords) == 0:
        return {
            "length": length,
            "starts": np.empty(0, dtype=np.int64),
            "distances": np.empty(0, dtype=float),
            "indices": np.empty(0, dtype=np.int64),
        }

    idx = res.discords[: min(top, len(res.discords))]
    idx = idx[(idx >= 0) & (idx < len(starts))]
    return {
        "length": length,
        "starts": starts[idx],
        "distances": dists[idx],
        "indices": idx,
    }
