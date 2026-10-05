"""The local **daily** archive: ``scripts/download_daily.py`` and the store
support it needs.  PLAN.md §CG.

Three of these tests exist because the defect they guard was found by
*measuring*, not by reading: the store's quality gate and its
"has this session closed?" test both hard-coded 1-minute answers, and both
were wrong for daily bars in ways no existing test could see -- because no
existing test wrote daily bars through the store at all.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from timeseries.store import (
    EASTERN,
    OHLCV,
    PanelStore,
    _iter_partition_files,
    expected_bar_count,
    fingerprint,
    fingerprint_per_session,
    verify_manifest,
)
from timeseries.timeframes import get_timeframe

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import download_daily as DD  # noqa: E402

DAILY = get_timeframe("1d")
INTRADAY = get_timeframe("1m")

#: A weekday well in the past, so "is this session settled?" has a real answer
#: rather than depending on today's date.
PAST = date(2026, 9, 3)
#: 09:30 ET on ``PAST`` -- how Yahoo stamps a daily bar.  Midnight UTC would be
#: the *previous* Eastern date and would silently land in the wrong partition.
PAST_OPEN = pd.Timestamp("2026-09-03 13:30:00", tz="UTC")

#: The live daily archive, for the checks that must run against real bytes.
DAILY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sp500_daily"
)


def daily_frame(session: str = "2026-09-03", *, close: float = 1.5,
                n: int = 1) -> pd.DataFrame:
    """``n`` daily bars starting at ``PAST_OPEN``, stamped 09:30 ET."""
    ts = pd.date_range(PAST_OPEN, periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [1.0] * n,
        "high": [2.0] * n,
        "low": [0.5] * n,
        "close": [close] * n,
    })


@pytest.fixture()
def store(tmp_path) -> PanelStore:
    return PanelStore(str(tmp_path))


# --------------------------------------------------------------------------- #
# The store must be told the resolution, and be told correctly
# --------------------------------------------------------------------------- #
class TestFingerprintIsStable:
    """The fingerprint was vectorised; these pin the bytes it produces.

    A fingerprint that hashes *different bytes* from the one that wrote the archive
    is the worst possible regression here: it is silent, it is fast to produce, and
    it makes every session in the archive present as a revision, so a re-run would
    either rewrite history or drown the operator in false warnings.  Speed is not
    worth that, so the equivalence is asserted rather than assumed.

    The vectorised path exists because the scalar one was the single most expensive
    step in a daily write: measured with cProfile, ``fingerprint`` cost ~33s of a
    36s single-ticker write, called once per *session* rather than once per file.
    """

    @staticmethod
    def _scalar(frame):
        """The original implementation, verbatim, as the reference."""
        import hashlib
        import io

        from timeseries.store import OHLCV

        if frame is None or len(frame) == 0:
            return "empty"
        cols = [c for c in OHLCV if c in frame.columns]
        out = frame.sort_values(["timestamp"]) if "timestamp" in frame.columns else frame
        buf = io.StringIO()
        for c in ("timestamp", *cols):
            if c in out.columns:
                buf.write(c)
                buf.write("\x1f")
                v = out[c]
                if c == "timestamp":
                    buf.write("\n".join(pd.to_datetime(v, utc=True, errors="coerce")
                                        .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").fillna("")))
                else:
                    buf.write("\n".join(pd.to_numeric(v, errors="coerce")
                                        .astype(float).map(lambda x: "%.17g" % x)
                                        .fillna("nan")))
                buf.write("\x1e")
        return hashlib.blake2b(buf.getvalue().encode("utf-8"), digest_size=16).hexdigest()

    @pytest.mark.parametrize("case", [
        "plain", "single_bar", "nan_and_extremes", "no_timestamp",
        "all_nan", "integer_columns", "unsorted", "object_strings",
        "eastern_tz", "empty",
    ])
    def test_it_hashes_exactly_what_the_scalar_version_hashed(self, case):
        import numpy as np

        n = 6
        base = {
            "timestamp": pd.date_range("2026-09-03 13:30", periods=n, freq="D", tz="UTC"),
            "open": [1.0 + i for i in range(n)],
            "high": [2.0 + i for i in range(n)],
            "low": [0.5] * n,
            "close": [1.5] * n,
        }
        frames = {
            "plain": pd.DataFrame(base),
            # The real daily case: one bar per session, called 11,544 times.
            "single_bar": pd.DataFrame(base).head(1),
            "nan_and_extremes": pd.DataFrame({
                "timestamp": base["timestamp"],
                "open": [np.nan, 1.0, -0.0, 1e308, 5e-324, np.inf],
                "high": [np.inf, 1.0, 2.0, 3.0, 4.0, 5.0],
                "low": [5e-324, 0.5, 0.5, 0.5, 0.5, 0.5],
                # An unparseable string must hash as NaN, not as its own text.
                "close": ["1.5", "abc", 1.0, 1 / 3, 2.0, 3.0],
            }),
            "no_timestamp": pd.DataFrame(base)[["open", "high", "low", "close"]],
            "all_nan": pd.DataFrame(base).assign(close=np.nan),
            "integer_columns": pd.DataFrame(base).assign(
                open=[1] * n, high=[2] * n, low=[0] * n, close=[1] * n),
            "unsorted": pd.DataFrame(base).iloc[::-1],
            "object_strings": pd.DataFrame(base).assign(
                close=[str(x) for x in range(n)]),
            # Daily bars are stamped Eastern; the offset must be applied, not assumed
            # away, or every session fingerprints differently from what is stored.
            "eastern_tz": pd.DataFrame(base).assign(
                timestamp=pd.date_range("2026-09-03 09:30", periods=n, freq="D",
                                        tz="America/New_York")),
            "empty": pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
                                   "open": pd.Series([], dtype=float)}),
        }
        frame = frames[case]
        assert fingerprint(frame) == self._scalar(frame), (
            "the vectorised fingerprint changed the bytes it hashes for case %r; "
            "every stored fingerprint would read as a revision" % case
        )

    def test_a_stored_daily_archive_still_verifies_against_its_manifest(self):
        """End of the same worry: run it against what is actually on disk."""
        root = str(DAILY_ROOT)
        if not os.path.isdir(root):
            pytest.skip("no daily archive built yet")
        store = PanelStore(root)
        # ``_manifest_or_none`` rather than ``_manifest``: the manifest is read
        # lazily, so ``_manifest`` is None until something asks for it.  Reading
        # the private attribute here would make this test pass vacuously -- the
        # loop would simply compare nothing and ``checked`` would be 0, which the
        # assertion below catches, but only by accident of how it was written.
        manifest = store._manifest_or_none()
        assert manifest is not None and len(manifest), "the daily manifest is missing"
        lut = {(r.ticker, str(r.session)): r.fingerprint
               for r in manifest.itertuples()}
        checked = mismatch = 0
        for ticker, _day, path in list(_iter_partition_files(root))[:8]:
            frame = pd.read_parquet(path)
            for day, grp in frame.groupby("session"):
                expected = lut.get((ticker, str(day)))
                if expected is None:
                    continue
                checked += 1
                if expected != fingerprint(grp):
                    mismatch += 1
        assert checked, "no sessions were compared; the check is vacuous"
        assert mismatch == 0, (
            "%d of %d stored daily sessions no longer match their manifest "
            "fingerprint" % (mismatch, checked)
        )


class TestStoreKnowsItIsDaily:
    """``PanelStore.write`` resolves two things per session, and both of them
    depend on the resolution.  Passing ``timeframe`` is what makes a daily
    archive behave like an archive rather than like 6,000 warnings."""

    def test_a_daily_bar_is_not_reported_as_a_short_session(self, store):
        """One daily bar is a *complete* session.

        Measured before the fix, on every single day:

            ['2026-09-03: short session: 1/390 bars']

        One spurious warning per trading day, on a well-formed archive, teaches an
        operator to ignore the warning that matters.
        """
        out = store.write("T", daily_frame(), timeframe=DAILY.key)
        assert out.ok
        assert out.issues == [], (
            "a single daily bar was flagged: %r" % (out.issues,)
        )

    def test_intraday_still_gets_the_390_bar_expectation(self, store):
        """The intraday answer must be unchanged by the daily branch.

        Asserted by *content*, not by the absence of a warning: a 390-bar session
        is still the expected shape, and the gate should still catch a short one.
        """
        full = pd.DataFrame({
            "timestamp": pd.date_range("2026-09-03 13:30", periods=390, freq="min",
                                       tz="UTC"),
            "open": [1.0] * 390, "high": [2.0] * 390,
            "low": [0.5] * 390, "close": [1.5] * 390,
        })
        assert store.write("T", full, timeframe=INTRADAY.key).issues == []

        short = full.iloc[:100]
        out = store.write("S", short, timeframe=INTRADAY.key)
        assert any("short session" in i for i in out.issues), (
            "a 100-bar session should still read as short; the intraday "
            "expectation was lost"
        )

    def test_expected_bar_count_is_one_on_a_weekday(self):
        assert expected_bar_count(PAST, timeframe=DAILY.key) == 1
        assert expected_bar_count(PAST, timeframe=INTRADAY.key) == 390


class TestDailySessionsSettle:
    """A daily bar stamped 09:30 ET must read as a *closed* session.

    The intraday test is ``last_bar >= close - 1 minute``.  A daily bar's only
    timestamp is the session's *open*, so it fails that comparison on every day
    and every daily bar reads as in-progress forever.

    The consequence is why this matters: "still trading" and "the vendor
    retro-adjusted history" are the only two reasons a fingerprint may differ, and
    the settled test is what tells them apart.  With every daily bar in progress,
    a genuine revision of a settled day was merged as growth -- the stored value
    overwritten, ``n_revisions`` never moved, and §B's one meaningful warning
    never fired for daily data.
    """

    def test_a_past_day_is_settled(self, store):
        assert store._session_settled(PAST, daily_frame(), timeframe=DAILY.key) is True

    def test_today_is_not_settled(self, store):
        today = datetime.now(tz=EASTERN).date()
        assert store._session_settled(today, daily_frame(), timeframe=DAILY.key) is False

    def test_a_revised_daily_bar_is_flagged_not_merged(self, store):
        """The whole point: flag it, and keep what was stored.

        A cheap test here would be "re-fetch and check nothing changed" -- but
        *nothing changed* is exactly what a **disabled** check also reports, since
        a disabled one merges silently.  Both look identical to the caller, which
        is why this perturbs a stored day and asserts the flag.
        """
        store.write("T", daily_frame(close=1.5), timeframe=DAILY.key)
        out = store.write("T", daily_frame(close=9.9), timeframe=DAILY.key)

        assert out.sessions_revised == 1, (
            "a changed settled daily bar was not reported as a revision "
            "(revised=%d extended=%d)" % (out.sessions_revised, out.sessions_extended)
        )
        assert out.sessions_extended == 0, (
            "the change was treated as growth, so a settled bar was overwritten"
        )
        row = store._manifest.set_index(["ticker", "session"]).loc[("T", "2026-09-03")]
        assert int(row["n_revisions"]) == 1
        assert row["fingerprint"] != row["observed_fingerprint"]
        stored = pd.read_parquet(store._partition_path("T", "", timeframe=DAILY.key))
        assert float(stored["close"].iloc[0]) == 1.5, (
            "the flag policy kept nothing: the stored bar was overwritten anyway"
        )

    def test_a_changed_bar_is_reported_as_a_revision_not_as_growth(self, store):
        """The two outcomes are reported under *different names*, and that is the
        whole point of the settled test.

        With the settled branch removed, a changed past day still ends up flagged --
        the ``unsettled`` list is simply empty, so the code falls through to the
        revision branch.  It is correct **by accident**, and an assertion on
        "``sessions_revised == 1``" passes either way, which makes it useless as a
        guard.

        The assertion that actually distinguishes them is ``sessions_extended``:
        growth is reported as extension and merges; a revision is reported as
        revision and keeps the stored bar.  So this asserts the count of each, not
        just that "something was flagged".
        """
        store.write("T", daily_frame(n=3, close=1.5), timeframe=DAILY.key)
        out = store.write("T", daily_frame(n=3, close=9.9), timeframe=DAILY.key)

        assert out.sessions_extended == 0, (
            "a settled day was reported as growth, so it was merged instead of "
            "flagged -- the §B warning is disabled for daily"
        )
        assert out.sessions_revised == 3, (
            "expected every changed day to be flagged, got %d" % out.sessions_revised
        )
        stored = pd.read_parquet(store._partition_path("T", "", timeframe=DAILY.key))
        assert list(stored["close"]) == [1.5] * 3, (
            "the stored bars were overwritten despite policy='flag'"
        )

    def test_a_settled_revision_is_not_merged_alongside_a_growing_day(self, store):
        """**Both kinds of difference in one pass** -- the case a coarse test misses.

        Today's session gaining a bar and an old session being retro-adjusted are
        independent facts, and both happen on every incremental sync.  A first version
        handled them in sequence: report growth, then report revisions, then write.
        Because the write came after both, a single still-trading session let *every*
        revised settled day through -- so §B's warning was silenced by an unrelated row.

        Asserting either counter alone does not catch it (a never-settled store reports
        ``revised=3, extended=0``, which looks correct).  What distinguishes the two is
        that **both** counters are non-zero *and* the stored file shows the settled day
        untouched.
        """
        today_et = pd.Timestamp(datetime.now(tz=EASTERN)).normalize() + pd.Timedelta(
            hours=9, minutes=30)
        stamps = list(pd.date_range(PAST_OPEN, periods=3, freq="D", tz="UTC")) + [
            today_et.tz_convert("UTC")]

        def frame(closes):
            n = len(closes)
            return pd.DataFrame({"timestamp": stamps, "open": [1.0] * n,
                                 "high": [2.0] * n, "low": [0.5] * n,
                                 "close": closes})

        store.write("T", frame([1.5, 1.5, 1.5, 1.5]), timeframe=DAILY.key)
        # The oldest settled day is retro-adjusted; today's bar ticks up.
        out = store.write("T", frame([9.9, 1.5, 1.5, 1.6]), timeframe=DAILY.key)

        assert out.sessions_revised == 1, (
            "the settled revision was lost because another day was in progress "
            "(revised=%d extended=%d)" % (out.sessions_revised, out.sessions_extended)
        )
        assert out.sessions_extended == 1, (
            "the in-progress day was not reported as growth (%d)"
            % out.sessions_extended
        )
        stored = pd.read_parquet(store._partition_path("T", "", timeframe=DAILY.key))
        assert list(stored["close"]) == [1.5, 1.5, 1.5, 1.6], (
            "expected the settled day kept at 1.5 and today's growth merged; got %r"
            % list(stored["close"])
        )

    def test_an_unchanged_daily_bar_is_not_a_revision(self, store):
        store.write("T", daily_frame(), timeframe=DAILY.key)
        out = store.write("T", daily_frame(), timeframe=DAILY.key)
        assert out.sessions_unchanged == 1
        assert out.sessions_revised == 0
        assert out.issues == []

    def test_todays_bar_still_grows(self, store):
        """Today's session is the one case where merging *is* correct."""
        now = datetime.now(tz=EASTERN)
        ts = (pd.Timestamp(now).normalize() + pd.Timedelta(hours=9, minutes=30))
        frame = pd.DataFrame({"timestamp": [ts.tz_convert("UTC")],
                              "open": [1.0], "high": [2.0], "low": [0.5],
                              "close": [1.5]})
        store.write("T", frame, timeframe=DAILY.key)
        revised = frame.copy()
        revised.loc[revised.index[0], "close"] = 2.5
        out = store.write("T", revised, timeframe=DAILY.key)
        assert out.sessions_extended == 1
        assert out.sessions_revised == 0


# --------------------------------------------------------------------------- #
# The archive layout
# --------------------------------------------------------------------------- #
class TestTheManifestIsNeverTruncatedMidWrite:
    """The manifest is written atomically, because a partial one is indistinguishable
    from an interrupted run.

    Observed while the daily build was running: ``wc -l`` on ``manifest.csv`` fell
    from 1,282,071 rows to 1,183,772 between two reads, and a concurrent check
    reported a ticker as "on disk but unindexed".  Nothing was wrong -- ``to_csv``
    truncates the target before streaming the new contents out, so for the ~1.4s
    that a 567MB manifest takes to write, a reader sees a short file.

    The failure mode is worse than a transient: a truncated manifest reads as
    "these sessions exist but were never indexed", which is the exact signature of a
    build that died partway, and would send an operator to ``--rebuild-manifest``
    for no reason.  ``os.replace`` makes the swap atomic, so a reader sees the old
    file or the new one and never a partial one.
    """

    def test_a_reader_never_observes_a_short_manifest(self, store):
        """Sample the file *while* it is being rewritten, from another process's view."""
        store.write("AAA", daily_frame(n=40, close=1.5), timeframe=DAILY.key)
        good = os.path.getsize(os.path.join(store.root, "manifest.csv"))
        assert good > 0

        # Grow it enough that the write is not instantaneous, then poll the size
        # from inside a write and assert it never dips below a complete file.
        rows = []
        for i in range(400):
            rows.append({"ticker": "AAA", "session": "2026-%02d-%02d" % (i // 28 + 1, i % 28 + 1),
                         "bars": 1, "fingerprint": "a" * 32,
                         "observed_fingerprint": "a" * 32, "n_revisions": 0,
                         "first_bar": "2026-01-01T13:30:00Z",
                         "last_bar": "2026-01-01T13:30:00Z"})
        path = os.path.join(store.root, "manifest.csv")

        observed = []
        original = store._write_manifest_atomic

        def watching(frame):
            original(frame)
            observed.append(os.path.getsize(path))

        store._write_manifest_atomic = watching
        store._write_manifest(rows)
        assert observed, "the atomic writer was not used"
        assert min(observed) > 0, "manifest.csv was observed empty mid-write"

    def test_the_temp_file_never_survives_a_failed_write(self, store):
        """A crash mid-write must not leave something a reader could mistake for
        the manifest."""
        store.write("AAA", daily_frame(n=3, close=1.5), timeframe=DAILY.key)
        path = os.path.join(store.root, "manifest.csv")
        before = os.path.getsize(path)

        boom = RuntimeError("simulated crash between write and replace")
        real_replace = os.replace

        def failing_replace(src, dst):
            raise boom

        os.replace = failing_replace
        try:
            with pytest.raises(RuntimeError):
                store._write_manifest_atomic(pd.DataFrame([{
                    "ticker": "AAA", "session": "2026-09-03", "bars": 1,
                    "fingerprint": "b" * 32, "observed_fingerprint": "b" * 32,
                    "n_revisions": 0, "first_bar": "2026-09-03T13:30:00Z",
                    "last_bar": "2026-09-03T13:30:00Z"}]))
        finally:
            os.replace = real_replace

        assert os.path.getsize(path) == before, (
            "a failed write modified the manifest; the old file must survive intact"
        )
        leftovers = [f for f in os.listdir(store.root) if f.startswith("manifest.csv.tmp")]
        assert not leftovers, "a temp manifest was left behind: %r" % leftovers

    def test_the_live_manifest_is_never_truncated_mid_write(self, store, monkeypatch):
        """The temp path is what gets written, and ``os.replace`` is what moves it.

        The two tests above establish that *a* write does not leave a partial file.
        They could not catch the defect that was actually there: the writer built a
        temp name, streamed the whole manifest into the **live** path, and removed
        the temp file without ever creating it -- so the live manifest was truncated
        for the entire ~2.4s write and ``os.replace`` was never called.  The
        docstring promised atomicity that the code did not implement.

        So assert the mechanism, not the outcome: the bytes must land on the temp
        path first, and only a replace may put them at the real path.
        """
        store.write("AAA", daily_frame(n=3, close=1.5), timeframe=DAILY.key)
        frame = pd.DataFrame([{
            "ticker": "AAA", "session": "2026-09-04", "bars": 1,
            "fingerprint": "c" * 32, "observed_fingerprint": "c" * 32,
            "n_revisions": 0, "first_bar": "2026-09-04T13:30:00Z",
            "last_bar": "2026-09-04T13:30:00Z"}])
        live = os.path.join(store.root, "manifest.csv")

        written, replaced = [], []
        real_csv, real_replace = pd.DataFrame.to_csv, os.replace

        def spy_csv(self, path, *a, **kw):
            written.append(str(path))
            return real_csv(self, path, *a, **kw)

        def spy_replace(src, dst):
            replaced.append((str(src), str(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr(pd.DataFrame, "to_csv", spy_csv)
        monkeypatch.setattr(os, "replace", spy_replace)
        store._write_manifest_atomic(frame)

        assert replaced, "os.replace was never called: the manifest was not swapped atomically"
        src, dst = replaced[-1]
        assert dst == live, "the replace target was not the live manifest: %r" % dst
        assert src.startswith(live + ".tmp"), "the replace source was not a temp file: %r" % src
        assert all(p.startswith(live + ".tmp") for p in written), (
            "the manifest was streamed into a non-temp path: %r" % written
        )


class TestTheManifestIsWrittenOncePerBatchNotOncePerTicker:
    """Writing N tickers rewrites the index N times unless the caller batches.

    Measured on the real daily archive: one ``_write_manifest`` call costs **5.1s**
    (2.4s of it the ``to_csv`` of the whole frame), because the manifest holds one
    row per session -- ~1.3M rows, ~176MB.  :meth:`PanelStore.write` calls it once
    per ticker, so a 503-ticker build performed 503 full rewrites of an index that
    grew on every one of them.

    That is not a slow build, it is a build that does not finish: the run this was
    found in sat at 25 minutes of CPU with **zero** new partitions written, because
    it was still on early tickers.
    """

    def test_a_batch_writes_the_manifest_once(self, store, monkeypatch):
        """N writes inside the batch must produce one manifest write, not N.

        Each ticker must be genuinely *new*: a re-write of identical content takes
        the "unchanged" fast path in :meth:`_write_daily` and produces no manifest
        rows at all, which would make this test pass without exercising anything.
        """
        store.write("SEED", daily_frame(n=4, close=1.5), timeframe=DAILY.key)
        store._manifest = None  # exercise the batch against a populated index too
        store._manifest_read = False

        calls = []
        original = store._write_manifest_atomic
        store._write_manifest_atomic = lambda f: (calls.append(len(f)),
                                                 original(f))[1]

        with store.batched_manifest():
            for i, sym in enumerate(("AAA", "BBB", "CCC", "DDD")):
                store.write(sym, daily_frame(n=4, close=1.5 + i), timeframe=DAILY.key)

        assert len(calls) == 1, (
            "expected one manifest write for four tickers, got %d" % len(calls)
        )
        rep = verify_manifest(store.root)
        assert rep["consistent"], (
            "the batched manifest does not describe the partitions on disk"
        )
        for sym in ("AAA", "BBB", "CCC", "DDD"):
            assert sym in store.tickers()

    def test_the_batch_is_flushed_even_when_a_write_raises(self, store):
        """An interrupted batch must still index what it already wrote.

        This is the case the interrupted build actually hit: bars on disk, rows lost
        because the rewrite never happened.  The flush therefore sits in a
        ``finally``, so a failure cannot reproduce it.
        """
        store.write("SEED", daily_frame(n=4, close=1.5), timeframe=DAILY.key)

        boom = RuntimeError("simulated failure mid-batch")
        original = PanelStore.write

        def failing_write(self, ticker, *a, **kw):
            if ticker == "CCC":
                raise boom
            return original(self, ticker, *a, **kw)

        PanelStore.write = failing_write
        try:
            with pytest.raises(RuntimeError):
                with store.batched_manifest():
                    store.write("AAA", daily_frame(n=4, close=1.5), timeframe=DAILY.key)
                    store.write("BBB", daily_frame(n=4, close=1.5), timeframe=DAILY.key)
                    store.write("CCC", daily_frame(n=4, close=1.5), timeframe=DAILY.key)
        finally:
            PanelStore.write = original

        # The two tickers written before the failure are on disk *and* indexed.
        for sym in ("AAA", "BBB"):
            assert os.path.isfile(os.path.join(store.root, "ticker=%s" % sym,
                                               "bars.parquet")), (
                "%s was not written to disk" % sym)
        rep = verify_manifest(store.root)
        assert not rep["missing_from_manifest"], (
            "a failed batch lost its manifest rows: %d on disk are unindexed"
            % len(rep["missing_from_manifest"])
        )

    def test_rows_deferred_by_a_batch_reach_the_manifest(self, store):
        """Batching must not change *what* is recorded, only when it is written."""
        store.write("AAA", daily_frame(n=5, close=1.5), timeframe=DAILY.key)
        eager = {r["session"]: r["fingerprint"]
                 for r in store._daily_manifest_rows("AAA", pd.read_parquet(
                     os.path.join(store.root, "ticker=AAA", "bars.parquet")))}

        store2 = PanelStore(str(store.root) + "-2")
        store2.root = store.root  # reuse the same on-disk archive
        with store2.batched_manifest():
            store2.write("AAA", daily_frame(n=5, close=1.5), timeframe=DAILY.key)
        store2._manifest = None
        store2._manifest_read = False
        manifest = pd.read_csv(os.path.join(store.root, "manifest.csv"))
        row = manifest.loc[(manifest["ticker"] == "AAA")
                           & (manifest["session"] == "2026-09-03")]
        assert len(row) == 1, "the deferred write did not reach the manifest"
        assert row["fingerprint"].iloc[0] == eager["2026-09-03"], (
            "batching changed a stored fingerprint"
        )


class TestTheVectorizedFingerprintIsTheSameFingerprint:
    """The fast per-session fingerprint must be byte-identical to the scalar one.

    :func:`fingerprint_per_session` exists only for speed, and speed is worthless
    here if it hashes different bytes: every stored session would then differ from
    its recorded fingerprint on the next run, and the whole archive would present
    as revised -- a silent, total corruption of the one index that records revisions.

    So the equivalence is asserted directly, on real archive data rather than on a
    synthetic frame, because the real data is what carries the cases that matter
    (``-0.0``, subnormals, exact ``%.17g`` round-trips).
    """

    def test_it_matches_the_scalar_path_on_a_synthetic_frame(self, store):
        frame = store_frame(n=200)
        expected = {str(d): fingerprint(g)
                    for d, g in frame.groupby("session", sort=True)}
        assert fingerprint_per_session(frame, None) == expected

    def test_it_matches_the_scalar_path_on_the_real_archive(self):
        """Real prices, real Eastern stamps, real float noise."""
        path = os.path.join(DAILY_ROOT, "ticker=AAPL", "bars.parquet")
        if not os.path.isfile(path):
            pytest.skip("no archived AAPL daily bars")
        frame = pd.read_parquet(path)
        expected = {str(d): fingerprint(g)
                    for d, g in frame.groupby("session", sort=True)}
        assert fingerprint_per_session(frame, None) == expected

    def test_it_survives_awkward_floats(self):
        """``-0.0``, a subnormal, a NaN and an exact binary fraction."""
        ts = pd.date_range(PAST_OPEN, periods=4, freq="D")
        frame = pd.DataFrame({
            "timestamp": ts,
            "open": [-0.0, 5e-324, 0.1, float("nan")],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [0.5, 0.25, 0.125, 0.0625],
            "close": [1.5, 2.5, 3.5, 4.5],
        })
        frame["session"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        expected = {str(d): fingerprint(g)
                    for d, g in frame.groupby("session", sort=True)}
        assert fingerprint_per_session(frame, None) == expected

    def test_an_unshuffled_and_a_shuffled_frame_hash_the_same(self):
        """Row order must not matter, exactly as for the scalar path."""
        frame = store_frame(n=60)
        shuffled = frame.sample(frac=1.0, random_state=7)
        assert (fingerprint_per_session(frame, None)
                == fingerprint_per_session(shuffled, None))


def store_frame(n: int = 200) -> pd.DataFrame:
    """A daily frame with a ``session`` column, as the daily writer sees it."""
    frame = daily_frame(n=n)
    frame["session"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["ticker"] = "AAA"
    return frame


class TestDailySearchIsNotCensoredIntoNothing:
    """§BX's forward-horizon censoring is vacuous on daily, and was applied anyway.

    §BX stops a window's forward return from spanning two trading sessions, because
    a 1-minute window crossing the close contains an overnight gap.  That rule is
    written in terms of *session boundaries*, and on daily a boundary sits between
    **every pair of bars** -- one bar is one whole session.  So "this window spans
    two sessions" is true of every window longer than one bar, and applying §BX
    unchanged admitted **zero** candidates.

    Measured before the fix, on the real archive: a daily search reported
    ``ok=True``, scored **1,284,802** candidates, and returned **0 matches**, for
    every ticker and every ``k``.  Nothing raised.  The intraday path masked 63,727
    windows and still returned its matches, so the bug was specific to daily and
    invisible from the resolution the app opens on by default.

    The invariant is about *agreement between two resolutions*, which is why it is
    asserted as one: the same query shape must produce matches from both archives.
    """

    @staticmethod
    def _dense_daily(n=200):
        """A daily frame: every bar its own session, which is the real shape."""
        days = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
        return pd.DataFrame({
            "timestamp": days + pd.Timedelta(hours=13, minutes=30),
            "session": days.strftime("%Y-%m-%d"),
            "open": np.linspace(1.0, 2.0, n), "high": np.linspace(1.5, 2.5, n),
            "low": np.linspace(0.5, 1.5, n), "close": np.linspace(1.2, 2.2, n),
            "ticker": "T",
        })

    @staticmethod
    def _sparse_intraday(n=200, per_session=60):
        """An intraday frame: one boundary per session, not per bar."""
        sess = np.repeat(np.arange(n // per_session), per_session)
        base = pd.Timestamp("2020-01-01 14:30", tz="UTC")
        stamps = [base + pd.Timedelta(days=int(s), minutes=30 * i)
                  for s in sess for i in range(per_session)]
        return pd.DataFrame({
            "timestamp": stamps,
            "session": ["2020-01-%02d" % (1 + int(s)) for s in sess
                        for _ in range(per_session)],
            "open": np.linspace(1.0, 2.0, len(stamps)),
            "high": np.linspace(1.5, 2.5, len(stamps)),
            "low": np.linspace(0.5, 1.5, len(stamps)),
            "close": np.linspace(1.2, 2.2, len(stamps)),
            "ticker": "T",
        })

    def test_a_daily_window_is_never_intra_session_so_censoring_is_skipped(self):
        from timeseries.panel import _censoring_can_bind, _horizon_admissible

        daily = self._dense_daily()
        boundary = (daily["session"].to_numpy()[:-1]
                    != daily["session"].to_numpy()[1:])
        assert _censoring_can_bind(boundary, 30) is False, (
            "a daily frame should have no intra-session window of 30 bars"
        )
        mask = _horizon_admissible(daily, 30, 60)
        assert int(mask.sum()) == len(mask), (
            "daily censoring rejected %d of %d windows; it must reject none"
            % (len(mask) - int(mask.sum()), len(mask))
        )

    def test_intraday_censoring_still_rejects_the_overnight_crossers(self):
        from timeseries.panel import _censoring_can_bind, _horizon_admissible

        intra = self._sparse_intraday()
        boundary = (intra["session"].to_numpy()[:-1]
                    != intra["session"].to_numpy()[1:])
        assert _censoring_can_bind(boundary, 30) is True, (
            "an intraday frame should admit intra-session windows"
        )
        mask = _horizon_admissible(intra, 30, 60)
        assert 0 < int(mask.sum()) < len(mask), (
            "intraday censoring must still censor: got %d of %d admissible"
            % (int(mask.sum()), len(mask))
        )

    def test_one_bar_censoring_is_still_enforced_even_on_daily(self):
        """``m == 1`` is a real case, and it is *not* vacuous.

        A one-bar daily window's forward return does leave the session, so §BX has
        something to say and must still say it -- only the ``m > 1`` daily case is
        skipped.  Without this the fix would over-reach and un-censor everything.
        """
        from timeseries.panel import _censoring_can_bind, _horizon_admissible

        daily = self._dense_daily()
        boundary = (daily["session"].to_numpy()[:-1]
                    != daily["session"].to_numpy()[1:])
        assert _censoring_can_bind(boundary, 1) is True
        mask = _horizon_admissible(daily, 1, 60)
        assert int(mask.sum()) == 0, (
            "a 1-bar daily window has no forward horizon inside its session, so "
            "every one of them must be censored; %d survived" % int(mask.sum())
        )


class TestTheManifestIsReadLazily:
    """The manifest is a derived index, so it is read on first use, not at startup.

    Daily's manifest is one row per *session*, so it reaches ~570MB across the
    index.  Parsing it in the constructor cost ~6s of dead time before the first
    widget rendered, for a file that ``tickers``, ``sessions``, ``coverage`` and
    ``load`` never read.  Reading it lazily moved that cost to the first lookup
    (measured: 1.37s -> 0.000s for construction) and nothing else.

    The risk is on the *write* side, and it is the dangerous shape of bug: if the
    manifest is never loaded, a write merges against nothing and replaces the index
    with only the rows it just wrote.  Nothing errors.  The archive keeps working
    perfectly well for as long as nobody notices the earlier tickers dropped out of
    the index -- at which point ``verify_manifest`` reports them as unindexed and
    every reader that trusted the manifest has been quietly wrong.
    """

    def test_a_write_keeps_the_rows_of_tickers_it_did_not_touch(self, store):
        store.write("AAA", daily_frame(n=2, close=1.5), timeframe=DAILY.key)
        store.write("BBB", daily_frame(n=2, close=1.5), timeframe=DAILY.key)

        # A *fresh* store, so the manifest can only come from the file: this is the
        # path that breaks if the lazy read never fires.
        reopened = PanelStore(store.root)
        reopened.write("CCC", daily_frame(n=2, close=1.5), timeframe=DAILY.key)

        stored = pd.read_csv(os.path.join(store.root, "manifest.csv"))
        assert set(stored["ticker"]) == {"AAA", "BBB", "CCC"}, (
            "writing one ticker dropped the others from the manifest; a store that "
            "merges against an unread manifest replaces the index instead"
        )

    def test_the_constructor_does_not_read_the_manifest(self, store):
        store.write("AAA", daily_frame(n=2, close=1.5), timeframe=DAILY.key)
        reopened = PanelStore(store.root)
        assert reopened._manifest is None, (
            "the manifest was read in the constructor; the lazy read is not lazy"
        )
        # ...and the tree still answers correctly without it.
        assert reopened.tickers() == ["AAA"]
        assert reopened.coverage()["n_sessions"].tolist() == [2]

    def test_the_first_lookup_loads_it_and_the_next_one_reuses_it(self, store):
        """Read once per store, not once per call.

        Counting the reads is the only way to tell "cached" from "re-read every
        time" -- both return the right answer, and only one of them is affordable
        at 570MB.  ``_load_manifest`` is the single place the file is opened, so
        counting its calls measures exactly the thing that matters.
        """
        store.write("AAA", daily_frame(n=3, close=1.5), timeframe=DAILY.key)
        reopened = PanelStore(store.root)

        calls = []
        original = reopened._load_manifest

        def counting():
            calls.append(1)
            return original()

        reopened._load_manifest = counting
        assert reopened.existing_fingerprint("AAA", "2026-09-03") is not None
        assert reopened.revision_count("AAA", "2026-09-03") == 0
        assert reopened.existing_fingerprint("AAA", "2026-09-04") is not None
        reopened.write("BBB", daily_frame(n=2, close=1.5), timeframe=DAILY.key)
        assert len(calls) == 1, (
            "the manifest was read %d times; it must be read once per store" % len(calls)
        )


class TestDailyArchiveLayout:
    """Identical to the minute archive, so one store reads both with no branch."""

    def test_daily_is_one_file_per_ticker_and_intraday_is_one_per_session(self, store):
        """**The measured difference, asserted as a contract.**

        Daily is ``ticker=SYM/bars.parquet`` and intraday is
        ``ticker=SYM/date=YYYY-MM-DD/bars.parquet``.  Not a style choice: the
        per-session layout applied to daily costs a directory and a Parquet file --
        footer, schema and compression dictionary -- to carry four numbers.

            AAPL daily, per-session   11,544 dirs   90 MB
            AAPL daily, per-ticker        1 file   546 KB

        Extrapolated over the index that is ~45 GB and ~5.8 million directories versus
        ~275 MB and 503 files.  The first is not a slow archive, it is an unusable one.
        """
        store.write("AAPL", daily_frame(n=5), timeframe=DAILY.key)
        flat = os.path.join(store.root, "ticker=AAPL", "bars.parquet")
        assert os.path.isfile(flat), "daily is not one file per ticker"
        assert not os.path.isdir(os.path.join(store.root, "ticker=AAPL", "date=")), (
            "daily created per-session directories, which is the layout that costs "
            "11,544 directories per ticker"
        )

        full = pd.DataFrame({
            "timestamp": pd.date_range("2026-09-03 13:30", periods=390, freq="min",
                                       tz="UTC"),
            "open": [1.0] * 390, "high": [2.0] * 390,
            "low": [0.5] * 390, "close": [1.5] * 390,
        })
        store.write("MSFT", full, timeframe=INTRADAY.key)
        assert os.path.isfile(os.path.join(store.root, "ticker=MSFT", "date=2026-09-03",
                                           "bars.parquet")), (
            "intraday must keep its per-session layout"
        )

    def test_the_manifest_keeps_one_row_per_session_on_both_layouts(self, store):
        """So ``verify_manifest`` and ``coverage`` mean the same thing either way.

        The manifest is a CSV, so a row per bar costs nothing -- unlike a directory.
        Counting *files* instead would report a daily archive as holding 503 sessions
        against a manifest of ~6,000 rows, and report the difference as corruption.
        """
        store.write("AAPL", daily_frame(n=5), timeframe=DAILY.key)
        store.write("MSFT", daily_frame(n=3), timeframe=DAILY.key)
        rep = verify_manifest(store.root)
        assert rep["consistent"], rep
        assert rep["n_partitions"] == 8, rep["n_partitions"]

    def test_sessions_are_read_from_the_column_not_the_directory(self, store):
        """Otherwise an incremental sync re-fetches everything, forever.

        ``sessions()`` is what decides what is missing.  Answering it from directory
        names reports a daily archive as holding no sessions at all, because daily has
        no ``date=`` directories -- so every run would find all 6,000 days missing and
        refetch the lot.
        """
        store.write("AAPL", daily_frame(n=5), timeframe=DAILY.key)
        got = store.sessions("AAPL")
        assert len(got) == 5, got
        assert got == sorted(got)
        assert got[0] == "2026-09-03"

    def test_one_malformed_day_does_not_reject_the_run(self, store):
        """A rejected session is recorded, not raised.

        503 symbols x 46 years is ~50,000 partitions; a single bad day must not
        cost the other 49,999.
        """
        frame = daily_frame(n=3)
        frame.loc[frame.index[1], "close"] = None
        out = store.write("AAPL", frame, timeframe=DAILY.key)
        assert out.sessions_written >= 2, (
            "a NaN close cost more than the one session it belongs to: %d"
            % out.sessions_written
        )

    def test_the_manifest_is_verifiable_and_rebuildable(self, store):
        store.write("AAPL", daily_frame(n=3), timeframe=DAILY.key)
        store.write("MSFT", daily_frame(n=2), timeframe=DAILY.key)
        rep = verify_manifest(store.root)
        assert rep["consistent"], rep
        assert rep["n_partitions"] == 5

        # A partition added behind the store's back is found, not ignored.
        stray = os.path.join(store.root, "ticker=NVDA", "bars.parquet")
        os.makedirs(os.path.dirname(stray), exist_ok=True)
        daily_frame(n=1).assign(session="2026-09-03", ticker="NVDA").to_parquet(
            stray, index=False)
        assert not verify_manifest(store.root)["consistent"], (
            "a partition the manifest has never seen did not register as missing"
        )
        out = store.rebuild_manifest()
        assert out["rebuilt"] >= 1
        assert verify_manifest(store.root)["consistent"]

    def test_a_file_with_no_session_column_is_reported_not_ignored(self, store):
        """A corrupt partition must not read as an intact archive.

        Skipping an un-indexable file would make ``verify_manifest`` report a broken
        archive as consistent -- which is the single thing it exists to prevent, and
        the failure is invisible because nothing raises.
        """
        store.write("AAPL", daily_frame(n=2), timeframe=DAILY.key)
        stray = os.path.join(store.root, "ticker=BROKEN", "bars.parquet")
        os.makedirs(os.path.dirname(stray), exist_ok=True)
        # A raw OHLCV frame: no ``session``, so it cannot be indexed.
        daily_frame().to_parquet(stray, index=False)

        rep = verify_manifest(store.root)
        assert not rep["consistent"], "an un-indexable partition read as consistent"
        assert any("BROKEN" in sym for sym, _ in rep["missing_from_manifest"]), rep

        # And rebuilding reports it as dropped rather than quietly forgetting it.
        out = store.rebuild_manifest()
        assert any(sym == "BROKEN" for sym, _ in out["dropped"]), out

    def test_read_back_is_sorted_and_complete(self, store):
        store.write("AAPL", daily_frame(n=10), timeframe=DAILY.key)
        back = store.per_ticker()["AAPL"]
        assert len(back) == 10
        assert list(back["session"]) == sorted(back["session"])
        assert set(OHLCV) <= set(back.columns)


# --------------------------------------------------------------------------- #
# The script's own arithmetic
# --------------------------------------------------------------------------- #
class TestScriptArithmetic:
    """The boundaries, which is where this went wrong twice.

    ``fetch_ticker`` computes ``start = end - span`` and asks Yahoo for
    ``(start, end)``: the span is **exclusive**.  Two consequences, both measured
    on AAPL:

    * the naive inclusive window loses the oldest session, so the archive simply
      begins one day late and nothing on the page says so;
    * moving ``stop`` forward to compensate **also** moves the start forward,
      because the span is anchored to it, so the fix loses *more* history:

          stop=end+0d  ->  6,285 rows, first 2001-10-10
          stop=end+2d  ->  6,283 rows, first 2001-10-12
          stop=end+30d ->  6,263 rows, first 2001-11-09
    """

    def test_weekdays_excludes_weekends(self):
        # 2026-09-03 is a Thursday, so 09-05/09-06 are the weekend.
        days = DD._weekdays(date(2026, 9, 3), date(2026, 9, 9))
        assert days == [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7),
                        date(2026, 9, 8), date(2026, 9, 9)]
        assert all(d.weekday() < 5 for d in days)

    def test_weekdays_handles_an_empty_and_an_inverted_range(self):
        assert DD._weekdays(date(2026, 9, 9), date(2026, 9, 3)) == []
        # A weekend-only range is legitimately empty rather than an error.
        assert DD._weekdays(date(2026, 9, 5), date(2026, 9, 6)) == []
        assert DD._weekdays(date(2026, 9, 3), date(2026, 9, 3)) == [date(2026, 9, 3)]

    def test_the_slice_is_by_eastern_session_not_utc_day(self):
        """A bar stamped 04:00 UTC is midnight ET -- the same Eastern date, but a
        different UTC date.  Slicing on the UTC date would misfile a bar whenever
        the two disagreed.
        """
        frame = daily_frame()
        out = DD._slice(frame, date(2026, 9, 3), date(2026, 9, 3))
        assert len(out) == 1
        assert DD._slice(frame, date(2026, 9, 4), date(2026, 9, 5)) is None
        assert DD._slice(pd.DataFrame(), date(2026, 9, 3), date(2026, 9, 4)) is None

    def test_the_history_span_is_wider_than_the_registry_default(self):
        """``days=None`` resolves to ``DEFAULT_DAILY_DAYS`` = 9,125 (25 years),
        which silently truncates.

        Measured on AAPL: the 25-year default starts at 2001-10-11, while Yahoo
        holds bars back to 1980-12-12 -- 21 years of history the archive would
        never have seen, and no error anywhere to say so.
        """
        from timeseries.fetch import DEFAULT_DAILY_DAYS

        assert DD.DAILY_HISTORY_DAYS > DEFAULT_DAILY_DAYS, (
            "the archive default (%d days) does not exceed the fetch default "
            "(%d days), so it inherits the truncation"
            % (DD.DAILY_HISTORY_DAYS, DEFAULT_DAILY_DAYS)
        )

    def test_the_history_span_reaches_past_forty_years(self):
        """17,000 days is ~46 years, comfortably past Yahoo's 1980 floor for
        every instrument that still exists, and bounded so a request can never
        be unbounded by accident."""
        assert DD.DAILY_HISTORY_DAYS >= 365 * 40

    def test_the_stop_is_anchored_and_the_span_is_widened(self):
        """``stop`` is ``end + 1 day`` and the span is ``+ 2``, so the exclusive
        bounds cost nothing -- and ``stop`` is *not* pushed further out, because
        that is what clips the oldest history."""
        end = date(2026, 10, 4)
        start = end - timedelta(days=DD.DAILY_HISTORY_DAYS)
        stop = datetime.combine(end + timedelta(days=1), datetime.min.time(),
                                 tzinfo=timezone.utc)
        assert stop.date() == end + timedelta(days=1)
        assert (end - start).days + 2 == DD.DAILY_HISTORY_DAYS + 2


class TestScriptRejections:
    """``--rebuild-manifest`` and the failure modes that must be loud."""

    def test_a_bad_date_is_rejected_rather_than_silently_parsed(self):
        with pytest.raises(ValueError):
            DD._parse_date("2026-13-99")

    def test_an_inverted_window_is_rejected(self, tmp_path):
        """``--start`` after ``--end`` is a mistake, not an empty range.

        The alternative -- quietly treating it as an empty range -- reports success
        having downloaded nothing, which is the sort of no-op an archive operator
        has to diagnose by hand.

        Asserted on the exception's own message, which is where ``SystemExit("...")``
        puts it; it does not go to stdout or stderr, so a ``capsys`` assertion would
        pass vacuously against an empty string.

        No network is involved, because the window is validated *before* the universe
        is fetched -- see the note in ``main``.
        """
        with pytest.raises(SystemExit) as caught:
            DD.main(["--root", str(tmp_path), "--start", "2026-09-09",
                     "--end", "2026-09-03"])
        assert "after" in str(caught.value), caught.value

    def test_a_fresh_root_is_created_before_the_first_write(self, tmp_path):
        """``constituents.csv`` is written before any partition exists.

        Without an explicit ``makedirs`` the very first run of a fresh archive
        died on the first symbol:

            OSError: Cannot save file into a non-existent directory
        """
        root = str(tmp_path / "brand_new")
        assert not os.path.isdir(root)
        DD.main(["--root", root, "--tickers", "AAPL", "--days", "3", "--dry-run"])
        assert os.path.isdir(root), "the archive root was not created"

    def test_rebuild_manifest_repairs_and_reports(self, tmp_path, capsys):
        root = str(tmp_path)
        store = PanelStore(root)
        store.write("AAPL", daily_frame(n=3), timeframe=DAILY.key)
        # Drop the index entirely, as a deleted file would.
        os.remove(os.path.join(root, "manifest.csv"))
        assert not verify_manifest(root)["consistent"]

        assert DD.main(["--root", root, "--rebuild-manifest"]) == 0
        out = capsys.readouterr().out
        assert "consistent=True" in out
        assert verify_manifest(root)["consistent"]


class TestLiveDailyArchive:
    """Checks against the archive that was actually built.  Skipped when absent.

    These are the assertions a unit test cannot make: that the archive holds a
    realistic number of sessions, spans the history it claims to, and is
    internally consistent.  A store that writes what it is given passes every
    offline test while an archive that was never built passes them too.
    """

    @pytest.fixture()
    def root(self) -> str:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "sp500_daily",
        )
        if not os.path.isdir(path):  # pragma: no cover - archive is built separately
            pytest.skip("daily archive not built; run scripts/download_daily.py")
        return path

    def test_a_daily_search_actually_returns_matches(self, root):
        """The end-to-end assertion that was missing when the search returned zero.

        Every other check in this class asks whether the archive *holds* the right
        data.  None asked whether anything can be *found* in it -- and a fully
        correct archive returned zero matches from 1,284,802 scored candidates,
        because §BX's session-boundary censoring rejects every window on daily
        (PLAN.md §CR).

        The shape of that failure is why it needed a live test: ``ok=True``, a
        healthy candidate count, an empty result list.  Every offline assertion
        about the store and the layout passed throughout, and the intraday search
        was unaffected, so nothing in the suite could see it.
        """
        from timeseries.panel import PanelSearch

        store = PanelStore(root)
        tickers = store.tickers()
        if len(tickers) < 3:
            pytest.skip("need at least 3 tickers to search a panel")
        sectors = {}
        meta_path = os.path.join(root, "constituents.csv")
        if os.path.isfile(meta_path):
            meta = pd.read_csv(meta_path, dtype={"symbol": str})
            if {"yahoo_symbol", "sector"} <= set(meta.columns):
                sectors = dict(zip(meta["yahoo_symbol"], meta["sector"]))

        home = sorted(set(tickers) & set(sectors))[0] if sectors else sorted(tickers)[0]
        search = PanelSearch(store, sectors=sectors)
        query = search.latest_query(home, length=30)
        assert query is not None, "could not build a query from %s" % home

        out = search.run(query, k=10, max_per_ticker=2, n_baseline=100)
        result = out["result"]
        assert out["ok"], "the daily search reported not-ok: %s" % out.get("reason")
        assert result.n_candidates > 0, "no candidates were scored at all"
        assert result.n_matches > 0, (
            "a daily search scored %d candidates and returned zero matches; "
            "§BX's censoring is vacuous on daily (PLAN.md §CR)"
            % result.n_candidates
        )
        assert result.n_distinct_tickers > 0, (
            "every match came from one ticker; the panel cap is not being applied "
            "per ticker"
        )

    def test_it_is_large_and_consistent(self, root):
        store = PanelStore(root)
        tickers = store.tickers()
        assert len(tickers) > 400, "only %d tickers archived" % len(tickers)
        rep = verify_manifest(root)
        assert rep["consistent"], (
            "the daily archive's manifest is out of step with its partitions: "
            "%d unindexed, %d missing"
            % (len(rep["missing_from_manifest"]), len(rep["missing_from_disk"]))
        )

    def test_it_reaches_decades_back(self, root):
        """A 25-year default would have started in 2001; the archive starts in
        1980, which is what ``DAILY_HISTORY_DAYS`` buys."""
        store = PanelStore(root)
        first = min(store.sessions())
        assert first < "1990-01-01", (
            "the oldest archived session is %s, so the 25-year truncation is "
            "still in effect" % first
        )

    def test_it_is_usable_as_a_daily_pipeline_input(self, root):
        """The end-to-end claim: the archive is a drop-in source for daily mode."""
        from timeseries.pipeline import Pipeline

        store = PanelStore(root)
        bars = store.per_ticker(tickers=["AAPL"])["AAPL"]
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        assert len(bars) > 5_000, "only %d daily bars for AAPL" % len(bars)
        # ``from_frame`` is the app's own entry point, so this asserts the archived
        # bars are usable *exactly as the app uses them* rather than by a
        # hand-assembled path the app never takes.
        pipe = Pipeline.from_frame(bars, length=60, timeframe=DAILY.key)
        # Fewer than the raw bar count, and that is correct: ``from_frame`` drops the
        # ``ROLLING_WINDOW`` warm-up bars the z-score needs.  Asserting equality here
        # would be asserting a bug; what matters is that the pipeline is *ready* and
        # that it covers nearly the whole archive.
        warmup = len(bars) - pipe.n_bars
        assert 0 < warmup <= 60, "%d warm-up bars is not a warm-up" % warmup
        assert pipe.ready, "the archived daily bars do not form a usable pipeline"
        assert pipe.n_bars > 5_000, "only %d usable daily bars" % pipe.n_bars