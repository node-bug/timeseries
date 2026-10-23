"""Select a threadsafe Numba threading layer before any parallel kernel runs.

Why this module exists
----------------------
STUMPY's kernels are ``@numba.njit(parallel=True)``.  Numba's default threading layer
on macOS is ``workqueue``, which is explicitly *not* threadsafe.  When two Python
threads enter a parallel region at the same time -- or when a parallel region is
launched from inside another parallel region -- the layer detects the overlap, prints::

    Numba workqueue threading layer is terminating: Concurrent access has been detected.

and aborts the whole process with ``SIGABRT``.  There is no Python traceback and no
exception to catch: the server simply vanishes mid-request, which is why this
presents as "the app just crashed".

Numba's error message suggests TBB as the alternative.  That advice does not apply on
macOS: Intel's ``tbb`` and ``intel-openmp`` PyPI wheels publish no macOS build at all
(``manylinux``/``win_amd64`` only), so ``pip install tbb`` cannot work here and numba
ships no ``tbbpool`` extension for this platform either.  The layer that *is* both
available and genuinely threadsafe is **OpenMP** -- numba ships ``omppool`` for macOS,
it just needs the ``libomp`` runtime that the wheel does not bundle.

So this module vendors that runtime (``libomp.dylib``, taken from the conda-forge
``llvm-openmp`` package) and preloads it into the process at import time.  Preloading
is what makes it work: numba's ``omppool`` extension is linked against
``@rpath/libomp.dylib``, and dyld resolves that only if the library is already loaded
with a global symbol scope.  Once it is, numba's default layer priority
(``tbb``, ``omp``, ``workqueue``) picks ``omp`` automatically -- no environment
variable and no ``DYLD_LIBRARY_PATH`` are required.

``libomp`` is preloaded *before* the first parallel region launches.  Numba resolves
and caches its threading layer lazily, on the first ``parallel=True`` call, so the
preload has to happen before then; after that point the choice is locked in and the
only way to change it is to restart the process.

Everything here is best-effort.  On a platform with no vendored library, or on a
machine where the library will not load, this module degrades silently to whatever
Numba picks on its own -- and :func:`timeseries.matrix_profile.numba_guard` remains
as the serialisation backstop.  A missing accelerator must never be the reason the
app fails to start.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Optional

__all__ = [
    "VENDORED_LIBOMP",
    "threading_layer_status",
    "preload_threaded_runtime",
]

# The vendored OpenMP runtime.  Numba's ``omppool`` extension links against
# ``@rpath/libomp.dylib``, so this file's install name deliberately matches.
VENDORED_LIBOMP = Path(__file__).resolve().parent / "_vendor" / "libomp.dylib"

# Set once the preload has been attempted, so the (idempotent) work is done once.
_PRELOAD_ATTEMPTED = False
_VENDORED_LOADED: Optional[str] = None


def _preload_vendored_libomp() -> Optional[str]:
    """Load the vendored ``libomp.dylib`` into the process with global symbols.

    Returns the path that was loaded, or ``None`` if it could not be loaded.  The
    ``RTLD_GLOBAL`` flag is the whole point: it publishes the OpenMP symbols
    (``__kmpc_*``, ``_omp_*``) into the global namespace so that numba's
    ``omppool`` extension resolves them when it is dlopen'd later.
    """
    if platform.system() != "Darwin":
        # Only the macOS build is vendored; other platforms pick their own layer.
        return None

    if not VENDORED_LIBOMP.is_file():
        return None

    try:
        ctypes.CDLL(str(VENDORED_LIBOMP), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # A malformed or incompatible binary must not take the app down; Numba will
        # fall back to its default layer and the guard covers the gap.
        return None

    return str(VENDORED_LIBOMP)


def preload_threaded_runtime() -> Optional[str]:
    """Preload the threadsafe OpenMP runtime so Numba selects the ``omp`` layer.

    Safe to call more than once; the work only happens on the first call.  Returns the
    path of the library that was loaded, or ``None`` if no threadsafe runtime could be
    made available.
    """
    global _PRELOAD_ATTEMPTED, _VENDORED_LOADED

    if _PRELOAD_ATTEMPTED:
        return _VENDORED_LOADED

    _PRELOAD_ATTEMPTED = True
    _VENDORED_LOADED = _preload_vendored_libomp()
    return _VENDORED_LOADED


def threading_layer_status() -> dict:
    """Report the Numba threading layer in use, for diagnostics.

    Numba resolves its layer lazily and raises ``ValueError`` if asked before the
    first parallel launch, so this reports ``None`` in that case rather than failing.
    """
    status = {
        "layer": None,
        "vendored_libomp": None,
        "libomp_loadable": False,
    }

    try:
        from numba.np.ufunc import parallel as _numba_parallel
    except Exception:
        return status

    try:
        status["layer"] = _numba_parallel.threading_layer()
    except Exception:
        # "Threading layer is not initialized." -- no parallel region has run yet.
        pass

    status["vendored_libomp"] = str(VENDORED_LIBOMP)
    status["libomp_loadable"] = VENDORED_LIBOMP.is_file()
    return status


# Load the runtime at import time.  This module is imported by
# :mod:`timeseries.matrix_profile` before STUMPY is used, which is early enough to
# beat Numba's lazy, first-launch layer resolution.  Importing ``numba`` or
# ``stumpy`` on their own does not trigger a parallel launch, so the ordering
# holds even if the application imports them first.
preload_threaded_runtime()