"""The repository must not let Git rewrite its binary archives.

Why this test exists
--------------------
``.gitattributes`` used to say ``* text eol=lf`` with no rule for ``*.parquet``.
Git's *text* filter rewrites line endings by deleting the CR byte of every CRLF
pair, and it applies that rewrite on the way into the object store.

For source code that is harmless.  For a Parquet file it is destructive, and
silently so: a Parquet file is a sequence of length-prefixed compressed pages
with absolute byte offsets recorded in its footer, so deleting one byte shifts
every offset after it and the footer no longer describes the file.  Pyarrow then
reports ``Corrupt snappy compressed data`` or ``Couldn't deserialize thrift``.

The measured damage in this repository:

* **480 tracked archives** were committed with their CRLF pairs collapsed;
* **442 of 504** daily archives became unreadable;
* the daily archive reported itself as holding 191,181 rows instead of
  4,164,637 -- with no error anywhere, because every reader caught the failure
  and moved on;
* ``git checkout`` wrote the corrupted bytes back over the working tree, so
  the damage travelled with the clone.

The single-byte difference is the whole story:

    worktree   194,840 bytes, 2 CRLF pairs  -> reads fine, 4,418 rows
    committed  194,838 bytes, 0 CRLF pairs  -> "Corrupt snappy compressed data"

These tests are static -- they read ``.gitattributes`` and ask Git what it would
actually do -- because the failure mode is "Git silently did the wrong thing",
and a test that only checked the file's *contents* would not have caught it.
"""

from __future__ import annotations

import os
import subprocess

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GITATTRIBUTES = os.path.join(_ROOT, ".gitattributes")


def _attr(path: str, name: str) -> str:
    """Git's effective value of one attribute for ``path``."""
    out = subprocess.run(
        ["git", "check-attr", name, "--", path],
        cwd=_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    # "<path>: <name>: <value>"
    return out.rsplit(": ", 1)[-1]


def _has_git() -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--git-dir"],
                       cwd=_ROOT, capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


requires_git = pytest.mark.skipif(
    not _has_git(), reason="needs a git checkout to resolve attributes"
)


class TestParquetIsNeverTreatedAsText:
    """The attribute that matters is ``text``, and it must be *unset*."""

    @requires_git
    @pytest.mark.parametrize("path", [
        "data/sp500_daily/ticker=AAPL/bars.parquet",
        "data/sp500_panel/ticker=AAPL/date=2026-10-01/bars.parquet",
        "some/other/place/bars.parquet",
    ])
    def test_text_is_unset_for_a_parquet_path(self, path):
        # ``text: unset`` is what turns the line-ending filter off.  ``binary``
        # is the macro that produces it.
        assert _attr(path, "text") == "unset"

    @requires_git
    def test_binary_is_set_for_a_parquet_path(self):
        assert _attr("data/sp500_daily/ticker=AAPL/bars.parquet", "binary") == "set"

    @requires_git
    def test_a_parquet_path_is_not_marked_for_diff_or_merge(self):
        # ``binary`` expands to ``-diff -merge -text``; all three matter for a
        # file that must never be touched by a content filter.
        path = "data/sp500_daily/ticker=AAPL/bars.parquet"
        assert _attr(path, "diff") == "unset"
        assert _attr(path, "merge") == "unset"


class TestTheAttributeFileSaysSo:
    """A static reading of ``.gitattributes``, so the rule cannot be dropped.

    The attribute check above proves Git's *effective* behaviour.  This one
    proves the rule is stated in the file, where the next person editing it
    will see it.
    """

    def test_the_file_exists(self):
        assert os.path.isfile(_GITATTRIBUTES)

    def test_parquet_is_declared_binary(self):
        with open(_GITATTRIBUTES, encoding="utf-8") as fh:
            text = fh.read()
        rules = [ln.strip() for ln in text.splitlines()]
        assert "*.parquet binary" in rules

    def test_the_catch_all_text_rule_is_still_there_for_source_files(self):
        # The fix must not disable line-ending normalisation for the code.
        with open(_GITATTRIBUTES, encoding="utf-8") as fh:
            text = fh.read()
        assert "* text eol=lf" in text


class TestOtherBinaryPayloadsAreExcluded:
    @requires_git
    @pytest.mark.parametrize("path", [
        "data/sp500_daily/ticker=AAPL/bars.feather",
        "data/sp500_daily/ticker=AAPL/bars.npy",
        "data/sp500_daily/ticker=AAPL/archive.pkl",
    ])
    def test_text_is_unset_for_other_binary_payloads(self, path):
        assert _attr(path, "text") == "unset"


class TestCsvStaysText:
    """CSV is text, and must stay text.

    The archive's CSVs are read with ``engine="pyarrow"`` for speed, but they
    are still CSV: the reader parses them, and a CRLF->LF rewrite would change
    bytes without changing what any reader sees.  Declaring them binary would
    be over-correction.
    """

    @requires_git
    def test_text_is_set_for_a_csv_path(self):
        assert _attr("data/sp500_daily/manifest.csv", "text") == "set"
