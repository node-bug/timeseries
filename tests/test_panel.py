"""Tests for the multi-ticker archive and the cross-sectional search.

These cover the two claims that are easy to get quietly wrong and expensive to get
wrong quietly:

* **The archive must be idempotent and must not confuse normal accumulation with a
  data revision.**  A re-fetch of today's session changes its fingerprint every time
  simply because more minutes have traded.  Treating that as a revision makes every
  routine sync look like corruption, which teaches an operator to ignore the one
  warning that matters.  Conversely, failing to flag a *closed* session whose
  content genuinely changed is how a retro-adjusted history sneaks in (§B).
* **A cross-sectional search must never let a window span two tickers**, and must
  report ranks against a population it can name.  Both are silent failures: a
  spliced window is a pattern that never happened, and a rank against a shortlist is
  a rank that always looks good.
"""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from timeseries import matrix_profile as MP
from timeseries.matching import DEFAULT_AMPLITUDE_WEIGHT
from timeseries.panel import (
    PanelSearch,
    _exclude_window_by_time,
    _horizon_admissible,
    _log_returns,
    build_panel,
    find_panel_matches,
)
from timeseries.store import (
    MERGE_POLICY,
    OHLCV,
    PanelStore,
    display_symbol,
    expected_bar_count,
    fetch_sp500_constituents,
    fingerprint,
    scan_partitions,
    session_bounds,
    session_et,
    validate_session,
    verify_manifest,
    yahoo_symbol,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_bars(start: str, n: int, *, price: float = 100.0, seed: int = 0,
              step_min: int = 1) -> pd.DataFrame:
    """A run of 1-minute bars with the archive's schema, starting at ``start`` (UTC)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=f"{step_min}min", tz="UTC")
    close = price * np.exp(np.cumsum(rng.normal(0.0, 0.0004, n)))
    return pd.DataFrame({
        "timestamp": ts,
        "open": close * (1 + rng.normal(0, 1e-4, n)),
        "high": close * (1 + abs(rng.normal(0, 3e-4, n))),
        "low": close * (1 - abs(rng.normal(0, 3e-4, n))),
        "close": close,
    })


def full_session(day: str, *, price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """A complete 390-bar regular session for one Eastern date."""
    start, end = session_bounds(date.fromisoformat(day))
    n = int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() // 60)
    return make_bars(pd.Timestamp(start).strftime("%Y-%m-%d %H:%M"), n,
                     price=price, seed=seed)


# --------------------------------------------------------------------------- #
# symbols and sessions
# --------------------------------------------------------------------------- #
class TestSymbols:
    def test_constituent_spelling_normalises_to_yahoo(self):
        # Wikipedia says BRK.B; Yahoo only answers to BRK-B.
        assert yahoo_symbol("BRK.B") == "BRK-B"
        assert yahoo_symbol("brk-b") == "BRK-B"
        assert yahoo_symbol(" aapl ") == "AAPL"

    def test_ordinary_symbols_are_unchanged(self):
        assert yahoo_symbol("AAPL") == "AAPL"
        assert yahoo_symbol("GOOGL") == "GOOGL"
        assert yahoo_symbol("BF.B") == "BF-B"

    def test_display_symbol_is_the_constituent_spelling(self):
        assert display_symbol("BRK-B") == "BRK.B"
        assert display_symbol("AAPL") == "AAPL"


class TestSessions:
    def test_session_is_named_by_eastern_date_not_utc(self):
        """A 09:30 ET open is 13:30/14:30 UTC, so UTC bucketing misfiles the morning."""
        summer = pd.Series(pd.to_datetime(["2026-07-15 13:30:00Z"]))
        winter = pd.Series(pd.to_datetime(["2026-01-15 14:30:00Z"]))
        assert str(session_et(summer).iloc[0].date()) == "2026-07-15"
        assert str(session_et(winter).iloc[0].date()) == "2026-01-15"

    def test_evening_utc_bars_stay_in_their_own_eastern_session(self):
        # 19:59 ET close is 23:59 UTC in summer -- still the same trading day.
        late = pd.Series(pd.to_datetime(["2026-07-15 23:59:00Z"]))
        assert str(session_et(late).iloc[0].date()) == "2026-07-15"

    def test_session_bounds_handle_dst(self):
        summer_open, summer_close = session_bounds(date(2026, 7, 15))
        winter_open, winter_close = session_bounds(date(2026, 1, 15))
        assert summer_open.hour == 13 and winter_open.hour == 14
        for o, c in ((summer_open, summer_close), (winter_open, winter_close)):
            assert int((c - o).total_seconds() // 60) == 390

    def test_expected_bar_count(self):
        assert expected_bar_count(date(2026, 7, 15)) == 390
        assert expected_bar_count(date(2026, 1, 15)) == 390
        assert expected_bar_count(date(2026, 7, 18)) == 0   # Saturday
        assert expected_bar_count(date(2026, 7, 19)) == 0   # Sunday
        # A 13:00 early close is 3.5 hours, not 6.5.
        assert expected_bar_count(date(2026, 11, 27), close_time=time(13, 0)) == 210


# --------------------------------------------------------------------------- #
# fingerprinting
# --------------------------------------------------------------------------- #
class TestFingerprint:
    def test_identical_content_matches(self):
        a = full_session("2026-07-15")
        assert fingerprint(a) == fingerprint(a.copy())

    def test_row_and_column_order_do_not_matter(self):
        a = full_session("2026-07-15")
        shuffled = a.sample(frac=1.0, random_state=0).reset_index(drop=True)
        reordered = shuffled[["timestamp", "close", "low", "high", "open"]]
        assert fingerprint(a) == fingerprint(reordered)

    def test_tiny_price_change_changes_the_fingerprint(self):
        """A 9th-decimal difference is a real tick, not formatting noise."""
        a = full_session("2026-07-15")
        b = a.copy()
        b.loc[5, "close"] = a.loc[5, "close"] * (1 + 1e-9)
        assert fingerprint(a) != fingerprint(b)

    def test_empty_frame(self):
        assert fingerprint(pd.DataFrame(columns=["timestamp", *OHLCV])) == "empty"


# --------------------------------------------------------------------------- #
# the quality gate
# --------------------------------------------------------------------------- #
class TestQualityGate:
    def test_clean_session_passes(self):
        rep = validate_session(full_session("2026-07-15"), date(2026, 7, 15))
        assert rep["ok"], rep
        assert rep["rows"] == 390

    def test_empty_is_fatal(self):
        rep = validate_session(pd.DataFrame(columns=["timestamp", *OHLCV]), date(2026, 7, 15))
        assert not rep["ok"]
        assert "no rows" in rep["fatal"]

    def test_non_positive_close_is_fatal(self):
        f = full_session("2026-07-15")
        f.loc[10, "close"] = 0.0
        rep = validate_session(f, date(2026, 7, 15))
        assert not rep["ok"]
        assert any("non-positive" in m for m in rep["fatal"])

    def test_short_session_is_reported_but_not_fatal(self):
        """An in-progress session is normal and must still be storable."""
        f = full_session("2026-07-15").iloc[:100]
        rep = validate_session(f, date(2026, 7, 15))
        assert rep["ok"], rep
        assert any("short session" in m for m in rep["issues"])

    def test_intra_session_hole_is_reported(self):
        f = full_session("2026-07-15")
        f = f.drop(f.index[200:210])   # a 10-minute halt
        rep = validate_session(f, date(2026, 7, 15))
        assert any("hole" in m for m in rep["issues"])

    def test_overnight_boundary_is_not_a_hole(self):
        """§BF: an overnight closure is the market being shut, not missing bars."""
        a = full_session("2026-07-15")
        b = full_session("2026-07-16")
        both = pd.concat([a, b], ignore_index=True)
        rep = validate_session(both, date(2026, 7, 15))
        assert not any("hole" in m for m in rep["issues"]), rep["issues"]

    def test_after_hours_bar_is_reported(self):
        f = full_session("2026-07-15")
        late = f.iloc[-1:].copy()
        late["timestamp"] = late["timestamp"] + pd.Timedelta(hours=3)
        rep = validate_session(pd.concat([f, late], ignore_index=True), date(2026, 7, 15))
        assert any("after 16:00" in m for m in rep["issues"])


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
class TestPanelStoreWrite:
    def test_writes_one_partition_per_session(self, tmp_path):
        st = PanelStore(str(tmp_path))
        res = st.write("AAPL", pd.concat([full_session("2026-07-15"),
                                          full_session("2026-07-16")], ignore_index=True))
        assert res.sessions_written == 2
        assert res.ok
        assert st.sessions("AAPL") == ["2026-07-15", "2026-07-16"]
        p = tmp_path / "ticker=AAPL" / "date=2026-07-15" / "bars.parquet"
        assert p.is_file()

    def test_round_trips_exactly(self, tmp_path):
        st = PanelStore(str(tmp_path))
        src = full_session("2026-07-15")
        st.write("AAPL", src)
        back = st.load()
        assert len(back) == len(src)
        a = src.set_index("timestamp")["close"].sort_index()
        b = back[back["ticker"] == "AAPL"].set_index("timestamp")["close"].sort_index()
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=0)

    def test_rewriting_identical_data_is_a_no_op(self, tmp_path):
        """The idempotency property the incremental sync depends on."""
        st = PanelStore(str(tmp_path))
        src = full_session("2026-07-15")
        st.write("AAPL", src)
        again = st.write("AAPL", src.copy())
        assert again.sessions_unchanged == 1
        assert again.sessions_written == 0
        assert again.sessions_revised == 0

    def test_rejects_unknown_policy(self, tmp_path):
        st = PanelStore(str(tmp_path))
        with pytest.raises(ValueError):
            st.write("AAPL", full_session("2026-07-15"), policy="yolo")

    def test_fatal_session_is_not_written(self, tmp_path):
        st = PanelStore(str(tmp_path))
        bad = full_session("2026-07-15")
        bad["close"] = -1.0
        res = st.write("AAPL", bad)
        assert res.sessions_rejected == 1
        assert not res.ok
        assert st.sessions() == []


class TestPanelStoreAccumulation:
    """An unfinished session must merge; a finished one must be flagged."""

    def test_open_session_grows_without_being_called_a_revision(self, tmp_path):
        day = date.today().isoformat()
        partial = full_session(day).iloc[:200]
        st = PanelStore(str(tmp_path))
        st.write("AAPL", partial)

        grown = full_session(day).iloc[:260]
        res = st.write("AAPL", grown)

        assert res.sessions_extended == 1
        assert res.sessions_revised == 0, res.issues
        assert res.bars_written == 60      # only the new bars, not the merged total
        stored = st.load()
        assert len(stored) == 260
        # A later fetch that starts mid-session must not truncate the morning.
        assert stored["timestamp"].min() == partial["timestamp"].min()

    def test_open_session_keeps_existing_partition_untouched(self, tmp_path):
        day = date.today().isoformat()
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session(day).iloc[:200])
        before = pd.read_parquet(tmp_path / "ticker=AAPL" / f"date={day}" / "bars.parquet")
        st.write("AAPL", full_session(day).iloc[:300])
        after = pd.read_parquet(tmp_path / "ticker=AAPL" / f"date={day}" / "bars.parquet")
        # Every originally stored bar survives, with its original values.
        merged = before.set_index("timestamp").join(
            after.set_index("timestamp"), lsuffix="_old", rsuffix="_new", how="left"
        )
        np.testing.assert_allclose(
            merged.loc[before["timestamp"], "close_old"].to_numpy(),
            merged.loc[before["timestamp"], "close_new"].to_numpy(), rtol=0, atol=0,
        )

    def test_closed_session_revision_is_flagged_not_overwritten(self, tmp_path):
        # A completed session: its last bar is 15:59 ET, so the store treats it as
        # settled and a content change is a real revision, not accumulation.
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        original = pd.read_parquet(
            tmp_path / "ticker=AAPL" / "date=2026-07-15" / "bars.parquet"
        )

        retro = full_session("2026-07-15")
        retro.loc[100, "close"] *= 0.98   # yfinance retro-adjusted this bar
        res = st.write("AAPL", retro, policy=MERGE_POLICY)

        assert res.sessions_revised == 1
        assert res.sessions_written == 0
        assert any("REVISED" in m for m in res.issues)
        kept = pd.read_parquet(tmp_path / "ticker=AAPL" / "date=2026-07-15" / "bars.parquet")
        np.testing.assert_allclose(
            original["close"].to_numpy(), kept["close"].to_numpy(), rtol=0, atol=0
        )
        assert st.revision_count("AAPL", "2026-07-15") == 1

    def test_policy_replace_adopts_the_new_content(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        retro = full_session("2026-07-15")
        retro.loc[100, "close"] *= 0.98
        res = st.write("AAPL", retro, policy="replace")
        assert res.sessions_revised == 1
        kept = pd.read_parquet(tmp_path / "ticker=AAPL" / "date=2026-07-15" / "bars.parquet")
        assert kept["close"].iloc[100] == pytest.approx(retro["close"].iloc[100])

    def test_policy_error_refuses_to_write(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        retro = full_session("2026-07-15")
        retro.loc[100, "close"] *= 0.98
        res = st.write("AAPL", retro, policy="error")
        assert res.sessions_rejected == 1
        assert res.errors


class TestPanelStoreRead:
    def test_scan_partitions_returns_self_describing_rows(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        st.write("MSFT", full_session("2026-07-15"))
        df = st.load()
        assert set(df["ticker"]) == {"AAPL", "MSFT"}
        assert set(df.columns) >= {"timestamp", "ticker", "session", *OHLCV}

    def test_ticker_filter(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        st.write("MSFT", full_session("2026-07-15"))
        assert set(st.load(tickers=["AAPL"])["ticker"]) == {"AAPL"}

    def test_date_filter(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        st.write("AAPL", full_session("2026-07-16"))
        got = st.load(since="2026-07-16")
        assert set(got["session"].astype(str)) == {"2026-07-16"}

    def test_per_ticker_resets_the_index(self, tmp_path):
        """A bar index is only meaningful within one ticker (no spliced windows)."""
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        st.write("MSFT", full_session("2026-07-16"))
        per = st.per_ticker()
        for sym, frame in per.items():
            assert list(frame.index) == list(range(len(frame)))
            assert frame["ticker"].eq(sym).all()
            assert frame["timestamp"].is_monotonic_increasing

    def test_missing_root_reads_as_empty(self, tmp_path):
        df = scan_partitions(str(tmp_path / "nope"))
        assert df.empty
        assert "ticker" in df.columns

    def test_manifest_tracks_the_partitions(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        st.write("MSFT", full_session("2026-07-16"))
        rep = verify_manifest(str(tmp_path))
        assert rep["consistent"], rep

    def test_manifest_drift_is_reported_not_hidden(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        import shutil
        shutil.rmtree(tmp_path / "ticker=AAPL" / "date=2026-07-15")
        rep = verify_manifest(str(tmp_path))
        assert not rep["consistent"]
        assert rep["missing_from_disk"] == [("AAPL", "2026-07-15")]

    def test_coverage_summarises_each_ticker(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("AAPL", full_session("2026-07-15"))
        st.write("AAPL", full_session("2026-07-16"))
        cov = st.coverage()
        assert set(cov["ticker"]) == {"AAPL"}
        assert int(cov.loc[cov["ticker"] == "AAPL", "n_sessions"].iloc[0]) == 2


# --------------------------------------------------------------------------- #
# the cross-sectional search
# --------------------------------------------------------------------------- #
def synthetic_panel(n_tickers: int = 4, n_bars: int = 900, seed: int = 0):
    """A panel of independent tickers, each with a distinct price level and scale."""
    bars = {}
    for i in range(n_tickers):
        bars[f"T{i}"] = make_bars("2026-07-15 13:30", n_bars,
                                  price=50.0 * (i + 1), seed=seed + i)
    return bars


def _panel_with_pattern(amplitude: float, seed: int, n_bars: int = 900):
    """One ticker walking a repeating zigzag, at a chosen size of move.

    The shape is identical whichever ``amplitude`` is passed -- the same sign
    sequence every ``step`` bars -- and only the size of each move changes.  That
    is what makes it usable for testing the amplitude term: any difference in score
    between two of these is attributable to amplitude and nothing else, because the
    normalised distance is scale-free by construction.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-07-15 13:30", periods=n_bars, freq="1min", tz="UTC")
    step = np.zeros(n_bars)
    step[::10] = amplitude
    step[5::10] = -amplitude
    close = 100.0 * np.exp(np.cumsum(step + rng.normal(0.0, amplitude * 0.02, n_bars)))
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close * 1.0001,
        "low": close * 0.9999, "close": close,
    })


class TestBuildPanel:
    def test_builds_a_matrix_aligned_to_its_frame(self):
        panel = build_panel(synthetic_panel())
        assert len(panel) == 4
        # Derived from the library rather than hardcoded: the panel's job is to lay
        # out whatever ``FEATURE_COLUMNS`` says it does, so a channel being added
        # (it grew a third, ``path_z``) must not be a test failure.  Pinning the
        # literal here only ever told us the number we had already read.
        from timeseries.features import FEATURE_COLUMNS

        for sym, (matrix, frame) in panel.items():
            assert matrix.shape[0] == len(frame)
            assert matrix.shape[1] == len(FEATURE_COLUMNS)
            assert np.isfinite(matrix).all()

    def test_drops_tickers_with_too_few_bars(self):
        bars = synthetic_panel()
        bars["TINY"] = make_bars("2026-07-15 13:30", 3)
        panel = build_panel(bars)
        assert "TINY" not in panel


class TestPanelSearchInvariants:
    def test_no_match_may_come_from_another_ticker_s_at_the_same_index(self):
        """The core cross-sectional invariant: a match is located in its own ticker."""
        bars = synthetic_panel()
        panel = build_panel(bars)
        home = "T0"
        mat, _frame = panel[home]
        query = np.ascontiguousarray(mat[100:160])
        res = find_panel_matches(query, panel, home_ticker=home, query_span=(100, 160),
                                 k=20, max_per_ticker=5)
        for m in res.matches:
            assert m.ticker in panel
            assert 0 <= m.start < len(panel[m.ticker][0])
            # The reported session must actually match the frame at that index.
            assert str(panel[m.ticker][1]["session"].iloc[m.start]) == m.session

    def test_a_window_never_spans_two_tickers(self):
        """Regression guard for the splicing failure.

        Concatenating the panel into one array and profiling it would let a window at
        a ticker boundary compare the tail of one company to the head of another -- a
        pattern that never happened in the market.  Each ticker is scored as its own
        series, so every match's span must lie inside one ticker's own bar range.
        """
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[10:70]), panel,
                                 home_ticker="T0", query_span=(10, 70), k=30,
                                 max_per_ticker=30)
        for m in res.matches:
            n_bars = len(panel[m.ticker][1])
            assert 0 <= m.start
            assert m.stop <= n_bars
            ts = panel[m.ticker][1]["timestamp"].iloc[m.start:m.stop]
            # One ticker, one session, a contiguous run of minutes.
            assert ts.is_monotonic_increasing
            assert ts.nunique() == len(ts)

    def test_excludes_the_query_itself_from_its_own_ticker(self):
        """§M: a query cannot be its own best match."""
        bars = synthetic_panel()
        panel = build_panel(bars)
        home = "T0"
        mat, _ = panel[home]
        span = (300, 360)
        res = find_panel_matches(np.ascontiguousarray(mat[span[0]:span[1]]), panel,
                                 home_ticker=home, query_span=span, k=50,
                                 max_per_ticker=50, nms_separation=60)
        assert res.n_excluded > 0
        for m in res.matches:
            if m.ticker == home:
                assert not (m.start < span[1] + 60 and m.start + 60 > span[0] - 60)

    def test_self_distance_is_zero_and_would_win_without_exclusion(self):
        """Guards that the exclusion is doing the work rather than luck."""
        bars = synthetic_panel()
        panel = build_panel(bars)
        home = "T0"
        mat, _ = panel[home]
        span = (300, 360)
        vec = np.ascontiguousarray(mat[span[0]:span[1]])
        # Unrestricted: the query's own position is the closest thing in the panel.
        d_self = MP.distance_profile(vec, np.ascontiguousarray(mat))[span[0]]
        # Tolerance is set by MASS's sliding dot product, not by floating point in
        # general.  STUMPY computes the distance at the self-position by differencing
        # two running sums that are individually O(m) larger than the difference they
        # resolve, so the rounding is O(eps * m) and lands around 1e-7 for a 60-bar
        # window.  The original 1e-8 was below the algorithm's own noise floor, so it
        # was asserting a property MASS does not have rather than one the panel needs.
        # The claim under test is "the query matches itself far better than anything
        # else", and 1e-6 is four orders below the smallest competing distance.
        assert d_self < 1e-6
        res = find_panel_matches(vec, panel, home_ticker=home, query_span=span, k=10)
        assert all(m.start != span[0] for m in res.matches if m.ticker == home)

    def test_matches_are_best_first(self):
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[50:110]), panel,
                                 home_ticker="T0", query_span=(50, 110), k=10)
        d = [m.distance for m in res.matches]
        assert d == sorted(d)

    def test_per_ticker_cap_stops_one_name_flooding_the_results(self):
        """A factor move must not return 20 windows of the same three tickers."""
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[50:110]), panel,
                                 home_ticker="T0", query_span=(50, 110),
                                 k=40, max_per_ticker=2)
        counts = {}
        for m in res.matches:
            counts[m.ticker] = counts.get(m.ticker, 0) + 1
        assert all(c <= 2 for c in counts.values()), counts

    def test_percentiles_are_monotone_in_distance(self):
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[50:110]), panel,
                                 home_ticker="T0", query_span=(50, 110), k=15)
        pct = [m.percentile for m in res.matches]
        assert pct == sorted(pct)
        assert all(0.0 <= p <= 1.0 for p in pct)

    def test_local_pool_can_only_contain_at_least_as_many_closer_windows(self):
        """The honest invariant between the two ranks, stated in counts.

        The global population is a strict superset of any ticker's own, so the number
        of windows closer than a given distance can only grow when going global.  It
        is tempting to conclude the *percentiles* must be ordered too, and they are
        not: both are fractions, so dividing the smaller count by the much larger
        denominator can land lower.  A window that is the closest in its own ticker
        has a local rank of 0 while still sitting at 0.06% globally, because two
        windows in *other* tickers happen to be closer.

        Asserting ``local >= global`` therefore fails on correct code, and asserting
        ``local <= global`` would have passed for entirely the wrong reason.  The
        count is the claim that actually holds, so that is what is asserted.
        """
        from timeseries import matrix_profile as mp_mod

        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        q = np.ascontiguousarray(mat[50:110])
        res = find_panel_matches(q, panel, home_ticker="T0", query_span=(50, 110), k=15)

        pooled = np.sort(np.concatenate([
            mp_mod.distance_profile(q, np.ascontiguousarray(panel[s][0])) for s in panel
        ]))
        for m in res.matches:
            n_closer_global = int(np.searchsorted(pooled, m.distance, side="left"))
            own = np.sort(mp_mod.distance_profile(q, np.ascontiguousarray(panel[m.ticker][0])))
            n_closer_own = int(np.searchsorted(own, m.distance, side="left"))
            assert n_closer_own <= n_closer_global
            # And each reported percentile is that count over its own pool size.
            assert m.percentile == pytest.approx(n_closer_global / pooled.size, abs=1e-12)
            assert m.percentile_same_ticker == pytest.approx(
                n_closer_own / own.size, abs=1e-12
            )

    def test_a_ticker_local_minimum_can_still_be_unremarkable_globally(self):
        """Documents *why* the two ranks are reported side by side.

        A window may be the closest thing its own ticker has ever seen and still rank
        mid-pack across the panel, because a dozen other names moved the same way.
        Collapsing this into one number would hide exactly the distinction a
        cross-sectional search exists to make, so the two ranks are reported
        separately and neither is allowed to imply the other.
        """
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[50:110]), panel,
                                 home_ticker="T0", query_span=(50, 110), k=30,
                                 max_per_ticker=30)
        for m in res.matches:
            if m.percentile_same_ticker == 0.0 and m.n_windows_in_ticker:
                # Its own record, yet not globally rare -- both true at once.
                assert m.percentile > 0.0
                return
        pytest.skip("no ticker-local minimum surfaced in this draw")

    def test_percentile_denominators_are_the_population_sizes(self):
        """Both ranks are fractions, and the denominators must be the real counts."""
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        q = np.ascontiguousarray(mat[50:110])
        res = find_panel_matches(q, panel, home_ticker="T0", query_span=(50, 110), k=15)

        # Recompute the global rank from scratch and require an exact match.
        from timeseries import matrix_profile as mp_mod
        pooled = np.concatenate([
            mp_mod.distance_profile(q, np.ascontiguousarray(panel[s][0]))
            for s in panel
        ])
        for m in res.matches:
            expected = float(np.searchsorted(np.sort(pooled), m.distance, side="left")) / pooled.size
            assert m.percentile == pytest.approx(expected, abs=1e-12)

            own = mp_mod.distance_profile(q, np.ascontiguousarray(panel[m.ticker][0]))
            expected_own = float(
                np.searchsorted(np.sort(own), m.distance, side="left")
            ) / own.size
            assert m.percentile_same_ticker == pytest.approx(expected_own, abs=1e-12)
            assert m.n_windows_in_ticker == own.size

    def test_percentiles_use_one_consistent_tie_convention(self):
        """Identical distances must rank identically globally and locally."""
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[50:110]), panel,
                                 home_ticker="T0", query_span=(50, 110), k=30,
                                 max_per_ticker=30)
        by_distance: dict[float, list] = {}
        for m in res.matches:
            by_distance.setdefault(round(m.distance, 9), []).append(m)
        for _d, group in by_distance.items():
            # Every window at the same distance shares the same local rank.
            locals_ = [g.percentile_same_ticker for g in group if g.n_windows_in_ticker]
            if len(set(round(x, 12) for x in locals_)) > 1:
                pytest.fail(f"tied distance {group[0].distance} got local ranks {locals_}")
            globals_ = [g.percentile for g in group]
            assert len(set(round(x, 12) for x in globals_)) == 1

    def test_candidate_population_covers_every_ticker(self):
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _ = panel["T0"]
        res = find_panel_matches(np.ascontiguousarray(mat[50:110]), panel,
                                 home_ticker="T0", query_span=(50, 110), k=10)
        expected = sum(len(panel[s][0]) - 60 + 1 for s in panel)
        assert res.n_candidates == expected
        assert res.n_tickers == len(panel)

    def test_channel_mismatch_is_rejected(self):
        panel = build_panel(synthetic_panel())
        bad = np.zeros((60, 5))
        with pytest.raises(ValueError):
            find_panel_matches(bad, panel, home_ticker="T0", query_span=(0, 60))


# --------------------------------------------------------------------------- #
# the amplitude term
# --------------------------------------------------------------------------- #
class TestPanelAmplitude:
    """The *size of move* term must mean the same thing here as on the single-ticker path.

    The sidebar sells one dial as "the same for every search in the app", so a
    cross-sectional search that ignored it would make one `amplitude_weight` name
    two metrics.  These pin that it is actually wired up, that it is measured on
    *moves* rather than on price *levels*, and that switching it off is exactly
    pure shape-matching.
    """

    L = 60

    def _panel_and_query(self):
        bars = synthetic_panel()
        panel = build_panel(bars)
        mat, _frame = panel["T0"]
        return panel, np.ascontiguousarray(mat[100:100 + self.L])

    def _moves(self, panel):
        """Realised-move profiles, built the way ``PanelSearch.moves`` builds them."""
        out = {}
        for sym, (_m, frame) in panel.items():
            r = _log_returns(frame)
            out[sym] = MP.amplitude_profile(r, self.L)
        return out

    def test_zero_weight_is_pure_shape_even_with_moves_supplied(self):
        """``amplitude_weight=0`` must recover shape-matching exactly.

        Also the reason every exact-value percentile test in this file still passes:
        a caller that supplies no move profiles scores shape only, so the default
        call is byte-identical to what this module always did.  The dial only bites
        when a caller both supplies the profiles and leaves the weight non-zero.
        """
        panel, q = self._panel_and_query()
        plain = find_panel_matches(q, panel, home_ticker="T0",
                                   query_span=(100, 100 + self.L), k=15)
        off = find_panel_matches(
            q, panel, home_ticker="T0", query_span=(100, 100 + self.L), k=15,
            amplitude_weight=0.0, moves=self._moves(panel), q_move=0.0,
        )
        assert plain.n_candidates == off.n_candidates
        assert [m.distance for m in plain.matches] == pytest.approx(
            [m.distance for m in off.matches], abs=1e-12
        ), "a zero weight must recover the shape-only score exactly"
        assert [m.percentile for m in plain.matches] == pytest.approx(
            [m.percentile for m in off.matches], abs=1e-12
        ), "and the rank it reports must be the shape-only rank"

    def test_no_move_profiles_means_no_amplitude_term(self):
        """The default call stays shape-only, so no existing percentile moves."""
        panel, q = self._panel_and_query()
        res = find_panel_matches(q, panel, home_ticker="T0",
                                 query_span=(100, 100 + self.L), k=15)
        # Recompute the raw pooled shape distance and require the reported rank to
        # match it, which is only true when no penalty was folded into the metric.
        pooled = np.concatenate([
            MP.distance_profile(q, np.ascontiguousarray(panel[s][0])) for s in panel
        ])
        for m in res.matches:
            expected = float(np.searchsorted(np.sort(pooled), m.distance, side="left")) / pooled.size
            assert m.percentile == pytest.approx(expected, abs=1e-12)

    def test_the_term_actually_moves_the_numbers(self):
        """A non-zero penalty must not be a no-op.

        An earlier version scaled the penalty by the pooled median distance and it
        was numerically present but functionally absent -- the failure mode the
        single-ticker path documents in detail.  This asserts the returned ranking
        genuinely differs once moves are supplied.
        """
        panel, q = self._panel_and_query()
        moves = self._moves(panel)
        q_move = float(np.median(np.concatenate(list(moves.values()))))

        shape = find_panel_matches(q, panel, home_ticker="T0",
                                   query_span=(100, 100 + self.L), k=15)
        penalised = find_panel_matches(
            q, panel, home_ticker="T0", query_span=(100, 100 + self.L), k=15,
            amplitude_weight=DEFAULT_AMPLITUDE_WEIGHT, moves=moves, q_move=q_move,
        )
        assert shape.n_candidates == penalised.n_candidates, "same pool, only the metric differs"
        changed = any(
            abs(a.distance - b.distance) > 1e-9
            for a, b in zip(shape.matches, penalised.matches)
        )
        assert changed, (
            "the amplitude term changed nothing -- it is wired in but numerically "
            "inert, which is exactly the bug it was added to fix"
        )

    def test_the_penalty_grows_with_the_gap_in_move(self):
        """A candidate that moved further than the query must be penalised.

        The distance is scale-free, so without this a violent window and a dead-calm
        one score identically.  Comparing two candidates of *equal* shape but very
        different realised move isolates the term from everything else the metric
        does.
        """
        panel = synthetic_panel(n_tickers=2, n_bars=900)
        L = self.L
        # T0 is calm, T1 is violent, and both share one shape: same return_z *and*
        # path_z pattern, different amplitude.
        calm = _panel_with_pattern(amplitude=0.0002, seed=1)
        loud = _panel_with_pattern(amplitude=0.02, seed=2)
        panel = {"CALM": calm, "LOUD": loud}
        built = build_panel(panel)
        mat, _ = built["CALM"]
        q = np.ascontiguousarray(mat[100:100 + L])

        moves = {s: MP.amplitude_profile(_log_returns(f), L)
                 for s, (_m, f) in built.items()}
        res = find_panel_matches(
            q, built, home_ticker="CALM", query_span=(100, 100 + L), k=20,
            max_per_ticker=5, amplitude_weight=DEFAULT_AMPLITUDE_WEIGHT,
            moves=moves, q_move=float(moves["CALM"][100]),
        )
        calm_d = [m.distance for m in res.matches if m.ticker == "CALM"]
        loud_d = [m.distance for m in res.matches if m.ticker == "LOUD"]
        assert calm_d and loud_d, f"expected both tickers to be represented, got {res.matches}"
        assert min(calm_d) < min(loud_d), (
            "the violent ticker was rated at least as close as the calm one -- the "
            "size-of-move term is not doing its job"
        )

    def test_a_missing_move_profile_scores_that_ticker_as_pure_shape(self):
        """A ticker with no move profile must still be scored, not dropped.

        The profiles come from the caller, so a partial dict is legitimate.  Padding
        with NaN rather than skipping is what keeps the pooled arrays aligned; the
        NaN becomes a zero penalty, i.e. that ticker contributes shape only.
        """
        panel, q = self._panel_and_query()
        moves = self._moves(panel)
        partial = {s: moves[s] for s in list(moves)[:1]}
        res = find_panel_matches(
            q, panel, home_ticker="T0", query_span=(100, 100 + self.L), k=15,
            amplitude_weight=DEFAULT_AMPLITUDE_WEIGHT, moves=partial, q_move=0.0,
        )
        assert res.n_candidates == sum(len(panel[s][0]) - self.L + 1 for s in panel), (
            "a ticker missing its move profile must still contribute its windows"
        )

    def test_empty_panel_returns_no_matches(self):
        res = find_panel_matches(np.zeros((60, 2)), {}, home_ticker="T0")
        assert res.n_matches == 0


# --------------------------------------------------------------------------- #
# timestamp-based self-exclusion
# --------------------------------------------------------------------------- #
class TestTimeBasedSelfExclusion:
    """A query built from a *different* series must still not match itself.

    ``query_span`` is an index range, so it is only meaningful against the home
    ticker's own matrix.  The Price tab builds its query from a live Yahoo fetch while
    the panel's bars come from the last archive sync, so neither the index guard nor a
    home ticker is available to it -- and leaving both off makes the panel return the
    query's own window at distance 0.000 as the top match, which is the one result
    §M exists to prevent.  Measured on the live 503-ticker archive: without the time
    guard the top match was the query's own window at d=0.000.
    """

    L = 60

    def _panel(self):
        return build_panel(synthetic_panel(n_tickers=3, n_bars=900))

    def _query_and_span(self, panel, ticker="T0", start=100):
        mat, frame = panel[ticker]
        stop = start + self.L
        vec = np.ascontiguousarray(mat[start:stop])
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        return vec, (ts.iloc[start], ts.iloc[stop - 1]), start, stop

    def test_the_query_does_not_match_itself_without_a_home_ticker(self):
        """The regression: empty home ticker, no index span, distance 0.000."""
        panel = self._panel()
        vec, _span, _s, _e = self._query_and_span(panel)
        unguarded = find_panel_matches(vec, panel, k=20, max_per_ticker=5)
        assert unguarded.matches, "expected the fixture to produce matches at all"
        assert unguarded.matches[0].distance == pytest.approx(0.0, abs=1e-6), (
            "this fixture is supposed to reproduce the self-match; if it no longer "
            "does the guard is no longer being tested"
        )

    def test_the_time_guard_removes_the_self_match(self):
        panel = self._panel()
        vec, (t0, t1), s, e = self._query_and_span(panel)
        guarded = find_panel_matches(vec, panel, k=20, max_per_ticker=5,
                                     exclude_times=(t0, t1), exclude_ticker="T0")
        assert guarded.matches, "the guard removed the pool rather than the self-match"
        self_match = [m for m in guarded.matches
                      if m.ticker == "T0" and m.distance < 1e-6]
        assert not self_match, (
            "the query's own window survived the time-based exclusion"
        )

    def test_the_guard_only_removes_the_query_neighbourhood(self):
        """It must be narrow: a distant window in the same ticker is still a match.

        An exclusion that swallowed the whole ticker would silently reduce the panel
        whenever the query's symbol happens to be an S&P 500 constituent.
        """
        panel = self._panel()
        vec, (t0, t1), s, e = self._query_and_span(panel)
        guarded = find_panel_matches(vec, panel, k=40, max_per_ticker=40,
                                     exclude_times=(t0, t1), exclude_ticker="T0")
        assert any(m.ticker == "T0" for m in guarded.matches), (
            "no window at all survived from the query's own ticker -- the exclusion "
            "is far wider than the query's neighbourhood"
        )

    def test_the_excluded_window_is_out_of_the_percentile_population(self):
        """§E: the rank must be computed over a pool the query was removed from.

        Recomputed from scratch rather than asserted loosely, because the obvious
        check is wrong in both directions.  Percentile 0 is *legitimate* after the
        removal -- nothing in the panel is then closer than the best genuine window --
        and the removal also **lowers** the percentile of a given distance, since the
        denominator shrinks along with the numerator.  Neither of those is inflation.

        What would be inflation is the query's own distance-0 window still occupying
        the head of the pool, which is exactly what this checks by rebuilding the
        pool without the excluded starts.
        """
        panel = self._panel()
        vec, (t0, t1), s, e = self._query_and_span(panel)
        guarded = find_panel_matches(vec, panel, k=20, max_per_ticker=5,
                                     exclude_times=(t0, t1), exclude_ticker="T0")

        mask, n_excluded = _exclude_window_by_time(panel["T0"][1], self.L, (t0, t1), self.L)
        assert mask is not None and n_excluded > 0

        # Rebuild the ranked pool with the excluded starts removed.
        parts = []
        for sym in sorted(panel):
            d = MP.distance_profile(vec, np.ascontiguousarray(panel[sym][0]))
            keep = np.ones(d.size, dtype=bool)
            if sym == "T0":
                keep[mask] = False
            parts.append(d[keep])
        pool = np.sort(np.concatenate(parts))

        for m in guarded.matches:
            expected = float(np.searchsorted(pool, m.distance, side="left")) / pool.size
            assert m.percentile == pytest.approx(expected, abs=1e-12), (
                f"{m.ticker}@{m.distance:.4f} ranked {m.percentile:.6f}, but a pool "
                f"with the query excluded says {expected:.6f} -- the reported rank is "
                f"not describing the population it claims to"
            )

    def test_an_unknown_ticker_excludes_nothing(self):
        """A symbol the panel does not hold (BTC-USD) must not break or narrow it."""
        panel = self._panel()
        vec, span, s, e = self._query_and_span(panel)
        guarded = find_panel_matches(vec, panel, k=20, max_per_ticker=5,
                                     exclude_times=span, exclude_ticker="NOPE")
        assert guarded.n_matches > 0
        assert guarded.n_tickers == len(panel), (
            "an unmatched exclusion ticker must not remove anything from the panel"
        )

    def test_the_helper_matches_the_index_based_guard_on_its_own_ticker(self):
        """The two mechanisms must agree, or one of them is wrong.

        Built on the same ticker with no fetch/archive skew, the timestamp guard has to
        reject exactly the starts ``exclusion_mask`` rejects.
        """
        from timeseries.matching import Query, exclusion_mask

        panel = self._panel()
        _mat, frame = panel["T0"]
        vec, (t0, t1), s, e = self._query_and_span(panel)
        n_starts = len(frame) - self.L + 1
        by_time, n_time = _exclude_window_by_time(frame, self.L, (t0, t1), self.L)
        by_index = exclusion_mask(np.arange(n_starts, dtype=np.int64),
                                  Query(vector=np.zeros((self.L, 2)), start=s, stop=e),
                                  self.L)
        assert by_time is not None
        assert n_time == int(by_index.sum()), (
            f"time-based guard rejected {n_time} windows, index-based rejected "
            f"{int(by_index.sum())} -- the two disagree on the same series"
        )
        assert np.array_equal(np.flatnonzero(by_time), np.flatnonzero(by_index)), (
            "the two guards exclude different starts on the same series, so one of "
            "them is mis-aligning the window with its timestamps"
        )

    def test_a_frame_with_no_timestamps_excludes_nothing(self):
        """No clock information means no exclusion -- never a guessed index."""
        frame = pd.DataFrame({"close": np.linspace(100.0, 110.0, 900)})
        mask, count = _exclude_window_by_time(
            frame, self.L,
            (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC")),
            self.L)
        assert mask is None and count == 0


class TestPanelSearchForecast:
    def _search(self, tmp_path, **kw):
        st = PanelStore(str(tmp_path))
        for sym, frame in synthetic_panel().items():
            st.write(sym, frame)
        sectors = {s: f"S{i}" for i, s in enumerate(sorted(synthetic_panel()))}
        return PanelSearch(st, sectors=sectors)

    def test_run_reports_a_baseline_for_every_horizon(self, tmp_path):
        """§D: a forecast is never produced without the control beside it."""
        ps = self._search(tmp_path)
        out = ps.run(ticker="T0", k=30, max_per_ticker=6, n_baseline=400,
                     min_matches=5)
        assert out["ok"]
        for h in out["horizons"]:
            assert out["baseline"][h].size > 0
            assert np.isfinite(out["baseline"][h]).any()
            f = next(x for x in out["forecasts"] if x.horizon == h)
            assert np.isfinite(f.baseline_mean)
            assert f.baseline_mean == pytest.approx(
                out["baseline"][h][np.isfinite(out["baseline"][h])].mean(), abs=1e-12
            )

    def test_baseline_uses_the_query_window_length(self, tmp_path):
        """A control group of a different window length is not a control group.

        Regression: the baseline was drawn at the panel's *shortest* ticker, which
        made every start invalid, so the baseline came back empty and lift was NaN
        while ``sufficient`` still read True.
        """
        ps = self._search(tmp_path)
        out = ps.run(ticker="T0", k=30, max_per_ticker=6, n_baseline=300,
                     min_matches=5)
        query_len = out["query"].length
        maxh = max(out["horizons"])
        for h in out["horizons"]:
            assert out["baseline"][h].size > 0
            # Every baseline draw must have had room for its full horizon.
            n_bars = min(len(f) for f in ps.bars.values())
            assert n_bars - query_len - maxh - 1 > 0

    def test_lift_is_the_difference_from_the_baseline(self, tmp_path):
        ps = self._search(tmp_path)
        out = ps.run(ticker="T0", k=30, max_per_ticker=6, n_baseline=400,
                     min_matches=5)
        for f in out["forecasts"]:
            assert f.lift == pytest.approx(f.mean_return - f.baseline_mean, abs=1e-12)

    def test_insufficient_evidence_suppresses_the_numbers(self, tmp_path):
        """§E: a small sample gets a note, not a confident number."""
        ps = self._search(tmp_path)
        out = ps.run(ticker="T0", k=5, max_per_ticker=1, n_baseline=200)
        assert out["ok"]
        for f in out["forecasts"]:
            if not f.sufficient:
                assert f.note
                assert np.isnan(f.p_value) or f.p_value != f.p_value

    def test_forward_returns_start_after_the_window(self, tmp_path):
        """§Z1: a match's forward return must not include the matched bars.

        Measured against ``close_aligned``, not ``close``.  ``m.stop`` is an index
        into the *aligned* frame, so pairing it with the raw series would read a bar
        ~20 positions early and pass a §Z1 test while violating §Z1 -- the anchor
        would sit inside the matched window it is supposed to follow.  See
        :class:`TestPanelIndexSpaces`.
        """
        ps = self._search(tmp_path)
        out = ps.run(ticker="T0", k=20, max_per_ticker=5, n_baseline=200,
                     min_matches=1)
        m = out["result"].matches[0]
        close = ps.close_aligned(m.ticker)
        h = min(out["horizons"])
        expected = np.log(close[m.stop - 1 + h] / close[m.stop - 1])
        got = out["forward"][h][0]
        assert got == pytest.approx(expected, abs=1e-12)

    def test_baseline_is_reproducible_for_a_seed(self, tmp_path):
        ps = self._search(tmp_path)
        a = ps.run(ticker="T0", k=10, max_per_ticker=5, n_baseline=200,
                   min_matches=1, seed=7)["baseline"]
        b = ps.run(ticker="T0", k=10, max_per_ticker=5, n_baseline=200,
                   min_matches=1, seed=7)["baseline"]
        for h in a:
            np.testing.assert_allclose(a[h], b[h])

    def test_readiness_requires_two_tickers(self, tmp_path):
        st = PanelStore(str(tmp_path))
        st.write("T0", make_bars("2026-07-15 13:30", 400))
        assert PanelSearch(st).ready() is False
        out = PanelSearch(st).run(ticker="T0")
        assert not out["ok"] and out["reason"]


    def test_baseline_spans_all_tickers_not_just_the_query_ticker(self, tmp_path):
        ps = self._search(tmp_path)
        base = ps._baseline([5], n=600, seed=1, length=30)
        assert base[5].size > 0
        # A panel-wide draw must be able to reach tickers other than the query's.
        closes = {s: ps.close(s) for s in ps.panel}
        assert len(closes) == 4


# --------------------------------------------------------------------------- #
# constituents
# --------------------------------------------------------------------------- #
# These call ``fetch_sp500_constituents()``, which hits Wikipedia over the network.
# They were previously unmarked, so ``pytest tests/`` failed on any machine that was
# offline, behind a proxy, or merely slow -- a failure that says nothing about the
# code under test and trains people to ignore red.  They assert on the *shape* of a
# live third-party list, so they cannot be rewritten against a fixture without
# ceasing to check what they exist to check: that the parser still reads today's
# page.  Opt in with ``-m network`` (or ``--run-network`` via conftest).
@pytest.mark.network
class TestConstituents:
    def test_shape_and_normalisation(self):
        df = fetch_sp500_constituents()
        assert len(df) > 400
        assert {"symbol", "sector", "yahoo_symbol"} <= set(df.columns)
        # The list spells class shares with a dot; Yahoo wants a hyphen.  Verified
        # against the live API: BRK-B returns bars, BRK.B returns "no data found".
        assert df["symbol"].str.contains(r"\.").any()
        assert not df["yahoo_symbol"].str.contains(r"\.").any()
        assert df["yahoo_symbol"].str.contains("-").any()   # BRK-B, BF-B
        assert df["symbol"].is_unique

    def test_yahoo_symbol_column_is_usable_by_the_downloader(self):
        df = fetch_sp500_constituents()
        brk = df.loc[df["symbol"] == "BRK.B", "yahoo_symbol"]
        assert len(brk) == 1 and brk.iloc[0] == "BRK-B"

    def test_sector_labels_are_present(self):
        df = fetch_sp500_constituents()
        assert df["sector"].notna().all()
        assert df["sector"].nunique() > 5


# --------------------------------------------------------------------------- #
# PLAN.md §BX -- session boundaries in the cross-sectional search
# --------------------------------------------------------------------------- #
def multi_session_panel(n_days: int = 6, n_sym: int = 4, seed: int = 0):
    """Several tickers over several real sessions, so closures actually exist."""
    out = {}
    days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
            "2026-01-09", "2026-01-12", "2026-01-13"]
    for i in range(n_sym):
        frames = [full_session(days[d], price=100 + 10 * i, seed=seed + 100 * i + d)
                  for d in range(n_days)]
        out[f"S{i}"] = pd.concat(frames, ignore_index=True)
    return out


class TestPanelSessionBoundary:
    """The panel's forward return has the same closure hazard as the single-ticker one."""

    def test_horizon_admissible_defaults_to_all_true_when_off(self):
        panel = build_panel(multi_session_panel())
        _mat, frame = panel["S0"]
        a = _horizon_admissible(frame, 60, None)
        assert a.size == len(frame) - 60 + 1
        assert a.all()

    def test_mask_cuts_only_the_tail_and_the_boundary_neighbourhood(self):
        """At h=60 with 390-bar sessions, a window's horizon can reach at most one
        closure, so the censored set is a narrow band before each boundary plus the
        data-end tail -- not a large fraction of the archive.  A much larger cut would
        mean the mask is over-eager."""
        panel = build_panel(multi_session_panel())
        _mat, frame = panel["S0"]
        a = _horizon_admissible(frame, 60, 60)
        assert 0 < (~a).mean() < 0.25, (
            f"masked {(~a).mean():.1%} of the pool -- expected a narrow band"
        )

    def test_no_panel_match_horizon_crosses_a_closure(self):
        bars = multi_session_panel()
        panel = build_panel(bars)
        mat, _frame = panel["S0"]
        res = find_panel_matches(np.ascontiguousarray(mat[60:120]), panel,
                                 home_ticker="S0", query_span=(60, 120),
                                 k=10, max_horizon=60)
        assert res.matches
        for m in res.matches:
            sess = panel[m.ticker][1]["session"].to_numpy()
            anchor = m.start + 60 - 1
            if anchor + 60 >= len(sess):
                continue
            assert (sess[anchor + 1:anchor + 61] == sess[anchor]).all(), (
                f"{m.ticker}@{m.start} crosses a closure"
            )

    def test_n_masked_is_reported_and_offsets_the_percentile_denominator(self):
        """§E: censored windows must leave the population the ranks are computed over."""
        bars = multi_session_panel()
        panel = build_panel(bars)
        mat, _frame = panel["S0"]
        res = find_panel_matches(np.ascontiguousarray(mat[60:120]), panel,
                                 home_ticker="S0", query_span=(60, 120),
                                 k=10, max_horizon=60)
        assert res.n_masked > 0
        plain = find_panel_matches(np.ascontiguousarray(mat[60:120]), panel,
                                   home_ticker="S0", query_span=(60, 120), k=10)
        assert plain.n_masked == 0
        assert res.n_candidates == plain.n_candidates, (
            "n_candidates is the pre-mask population; the denominator must shrink"
        )
        # Every returned rank must be computable against the *smaller* population.
        assert all(0.0 < m.percentile <= 1.0 for m in res.matches)

    def test_ranks_stay_monotone_in_distance_under_the_mask(self):
        bars = multi_session_panel()
        panel = build_panel(bars)
        mat, _frame = panel["S0"]
        res = find_panel_matches(np.ascontiguousarray(mat[60:120]), panel,
                                 home_ticker="S0", query_span=(60, 120),
                                 k=10, max_horizon=60)
        ordered = sorted(res.matches, key=lambda m: m.distance)
        pcts = [m.percentile for m in ordered]
        assert all(a <= b + 1e-12 for a, b in zip(pcts, pcts[1:]))

    def test_a_frame_without_dates_is_uncensored_not_discarded(self):
        """No date information means "cannot tell", which must not mean "drop"."""
        frame = pd.DataFrame({"close": np.linspace(100, 101, 500)})
        assert _horizon_admissible(frame, 60, 60).all()

    def test_a_frame_with_timestamps_but_no_session_column_still_works(self):
        panel = build_panel(multi_session_panel())
        _mat, frame = panel["S0"]
        stripped = frame.drop(columns=["session"])
        a = _horizon_admissible(stripped, 60, 60)
        assert a.size == len(stripped) - 60 + 1
        assert (~a).any()


class TestPanelIndexSpaces:
    """A match's ``start``/``stop`` address the **aligned** frame, not the raw bars.

    ``build_panel`` drops the feature warm-up, so every ticker's matrix and aligned
    frame are shorter than its raw bar frame by a per-ticker amount -- 20 rows at
    ``rolling=20``, but not a constant, because rows are also dropped for
    non-finite values.  :meth:`PanelSearch.close` returns the *raw* series while a
    match index points into the *aligned* one, so indexing one with the other reads a
    bar ~20 positions early.

    Nothing raises.  The forward return is simply computed from the wrong close
    against the wrong bars, which is the worst failure mode in this package: it
    returns a plausible number.  These tests pin the two series apart explicitly so
    the next reader to reach for ``close()`` with a match index is stopped by a
    failing test instead of by a colleague noticing.
    """

    def _panel(self, tmp_path):
        st = PanelStore(str(tmp_path))
        for sym, frame in synthetic_panel().items():
            st.write(sym, frame)
        return PanelSearch(st)

    def test_close_aligned_is_the_panel_frames_close_column(self, tmp_path):
        ps = self._panel(tmp_path)
        for sym, (_matrix, frame) in ps.panel.items():
            np.testing.assert_allclose(
                ps.close_aligned(sym), frame["close"].to_numpy(dtype=float)
            )

    def test_close_aligned_is_shorter_than_raw_by_the_warmup(self, tmp_path):
        """The two spaces must actually differ, or the bug below cannot be caught."""
        ps = self._panel(tmp_path)
        for sym in ps.panel:
            aligned = ps.close_aligned(sym)
            raw = ps.close(sym)
            assert aligned.size < raw.size
            # The gap is exactly the rows ``build_panel`` dropped, which is the
            # whole reason the two series cannot be used interchangeably.
            offset = raw.size - aligned.size
            assert offset >= 1, "expected a warm-up drop at %s" % sym
            assert offset != len(ps.panel[sym][0]), (
                "the offset must be the dropped warm-up, not the matrix's own rows"
            )

    def test_a_match_index_addresses_the_aligned_series(self, tmp_path):
        """The property the bug broke: one index names two different bars.

        The synthetic panel is a single session, so session *labels* cannot tell the
        two spaces apart -- they are the same date either way.  Timestamps can: the
        aligned frame's row ``i`` is the raw frame's row ``i + offset``, and asserting
        that offset is what proves an index is being resolved consistently.
        """
        ps = self._panel(tmp_path)
        for sym, (_matrix, frame) in ps.panel.items():
            offset = ps.close(sym).size - ps.close_aligned(sym).size
            raw_ts = ps.bars[sym]["timestamp"].to_numpy()
            aligned_ts = frame["timestamp"].to_numpy()
            # Every aligned row is the raw row `offset` later -- and is *not* the
            # raw row at the same index, which is the whole bug.
            assert np.array_equal(aligned_ts, raw_ts[offset:]), (
                "aligned frame for %s is not raw shifted by %d" % (sym, offset)
            )
            assert offset > 0 and not np.array_equal(aligned_ts, raw_ts[:aligned_ts.size]), (
                "the two index spaces are identical for %s; the test cannot fail" % sym
            )

    def test_run_measures_forward_returns_in_the_aligned_space(self, tmp_path):
        """The regression: a known window's forward return, recomputed by hand.

        Anchored on real matches, so the expected values are read from
        ``close_aligned`` -- the series a match index addresses -- rather than from
        ``close``.  This is the assertion that fails if the raw series is used.
        """
        ps = self._panel(tmp_path)
        out = ps.run(ticker="T0", k=10, max_per_ticker=4, n_baseline=200,
                     horizons=(30,), min_matches=1)
        assert out["ok"]
        h = 30
        assert out["forward"][h].size > 0

        res = out["result"]
        length = int(out["query"].length)
        expected = []
        for m in res.matches:
            series = ps.close_aligned(m.ticker)
            a = int(m.start) + length - 1
            expected.append(np.log(series[a + h] / series[a]))
        expected = np.asarray(expected)
        got = out["forward"][h]

        # Compared as sorted multisets, not element-wise.  ``run`` measures
        # ticker-by-ticker (``by_ticker`` groups the starts first, so each series is
        # sliced once) and concatenates in dict order, which is *not* the ranked order
        # of ``res.matches``.  Every value must appear exactly once; which position it
        # lands in is an implementation detail this test deliberately does not pin.
        assert np.isfinite(got).sum() == np.isfinite(expected).sum()
        np.testing.assert_allclose(
            np.sort(expected[np.isfinite(expected)]),
            np.sort(got[np.isfinite(got)]),
            atol=1e-12,
        )
