"""Test-suite bootstrap.

The only job here is to make ``tests/session_bars.py`` importable as a plain module
(``from session_bars import ...``) rather than as part of a package.  ``pyproject.toml``
puts ``src`` on the path for the library itself but not ``tests``, and pytest's default
``prepend`` import mode only adds the *rootdir* of a test file when that directory has
no ``__init__.py`` -- which ``tests/`` does not, so the directory itself should already
be on the path.  This file makes that explicit rather than leaving it to an import-mode
detail, because a shared fixture generator that fails to import would surface as nine
mysterious collection errors instead of one obvious one.
"""

from __future__ import annotations

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
