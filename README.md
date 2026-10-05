# Time Series Project

Pattern matching and conditional forecasting for **any** ticker, at 1-minute or daily
resolution.

On first load the app asks for a **data resolution** — `1-minute` or `Daily` — and
builds nothing until you pick. That choice holds for the session; see
[The resolution is chosen once](#the-resolution-is-chosen-once-when-the-app-opens).

Then type a ticker into the **Price** tab's box at the top of that tab and press
**Fetch**, and the tabs redraw on that instrument. The **Forecast** tab has a second
box of its own, so you can leave the price tape on QQQ while the forecast runs on
AAPL.

The output is a **conditional forecast distribution**, not a list of pretty charts:
given a query window, the tool finds historically similar windows and reports what
happened after them — always beside a random-window baseline, because a matched
forecast that cannot beat a baseline is not a forecast.

`PLAN.md` is the design document, including the reasoning behind each statistical
constraint and a record of the mistakes found and fixed during review.

## Requirements

- **Python 3.10+** — required by `stumpy`, which is a core dependency (see below)
- `uv` (recommended) or `pip`

## Installation

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Or with pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The dashboard and data download need extra packages:

```bash
pip install -e ".[dev,app]"
```

## Usage

```bash
pytest                                    # 634 tests
python -m streamlit run app.py            # dashboard
```

### Choosing a ticker and its bars

Each tab that has an instrument of its own carries the input at the **top of its own
body**, above the charts it changes:

| Where | Feeds | Notes |
|---|---|---|
| **Price** tab's box | Price, Matches, Quality, Backtest | The app title, price chart and help text follow this one |
| **Forecast** tab's box | Forecast and Projection | Starts on Price's ticker, then keeps its own; **Projection** charts the same ticker without a box of its own |

Either box takes any symbol Yahoo accepts (`AAPL`, `MSFT`, `BRK-B`, …). Switching
symbol clears the previous query for that tab only, so no stale match survives — and a
switch in one box leaves the other tab's ticker, brush and results untouched.

The point of the split is to compare instruments: leave the price tape on QQQ while
the forecast asks what usually followed a given shape **in AAPL**. *Forecast* owns
its own window brush, its own search settings and its own run; *Projection* is a fixed
reference over the same archive and takes no reader input. The Forecast box seeds
on the Price ticker the first time, so nothing is downloaded twice on a fresh load;
after that they are independent.

### The two forecast tabs

*Projection* is the reference: the archive's most recent bars and where the closest
matching windows went next, answered identically every time. It also carries the
**Projection bars** slider.

*Forecast* is the interactive half — brush a box, get the projection for it, and
read the evidence table beneath. The brush, the search settings and the table are on
one tab because they describe one window: splitting them would mean brushing on one
tab and reading the result on another. Both projections use the same *Projection bars*
value, so the reference band and your own are directly comparable.

### The resolution is chosen once, when the app opens

The app asks before it builds anything:

> **Data resolution** — `1-minute` · `Daily` — **Continue**

Nothing is drawn and nothing is downloaded until that is answered, because every
number below it depends on the answer: the Yahoo interval, the archive that is cached,
the window lengths, the forecast horizons, and every caption that names a bar. One
resolution applies to the whole session and to every tab.

This was a per-tab dropdown beside each ticker box, and it was **inert**. Its widget
key was passed as `key=` and read nowhere, so choosing *Daily* moved the control and
nothing else — the page kept charting 1-minute bars under a Daily caption, with no
error anywhere to explain it. It is now a label, and the choice happens once at
startup.

The reason that is the fix rather than a patch: with one resolution per session there
is no later moment at which the label and the archive can disagree, because neither is
independently changeable. A resolution a reader cannot change cannot be out of step
with the data.

To change resolution, reload the page and choose again. Both archives are cached
separately, so switching back costs nothing. Matching only ever compares like with
like: a daily query is scored against daily bars exclusively.

## Two resolutions, and why the constants are where they are

The package supports `1m` and `1d`. Every number that depends on the resolution — the
Yahoo interval, the filename slug, the gap threshold, bars per session, the rolling
z-score base, window bounds, horizons, the view size, and the volatility of the
synthetic null — lives in one table:

```python
from timeseries.timeframes import TIMEFRAMES, get_timeframe

tf = get_timeframe("1d")
tf.yfinance_interval      # '1d'      -- what Yahoo is asked for
tf.filename_slug          # '1d'      -- what the archive is called (1m -> '1min')
tf.gap_seconds            # None      -- a daily bar has no interior to hole
tf.bars_per_session       # 1         -- 390 on 1m
tf.default_length         # 60        -- 240 on 1m
tf.horizons               # (5,10,20,40)
```

This is a deliberate structure rather than tidiness. Each of those values transfers
between resolutions, but *not for the same reason*, and guessing wrong fails silently:
a daily archive judged by the 180-second hole rule looks corrupt, and one judged by
the 390-bar session rule looks truncated. Both readings send an operator hunting for a
download that never failed. Adding a third resolution is a new row here, not a second
literal threaded through the call sites.

`1m` remains the default everywhere, so anything that never mentions a resolution
behaves exactly as it did before.

**Fetching always downloads.** The fetched bars live in memory for the session only —
nothing is written to `data/`, and a refresh never overwrites a file you may be
comparing against. Use `timeseries.store` if you want a durable archive.

**The app never writes to `data/`; the download scripts do.** The rule above is about
the Streamlit app's **Fetch** button, and it is deliberate: a refresh that silently
overwrote an archive would destroy the file you may be comparing against, so bars
fetched in the app are held in memory and never persisted. The two downloader scripts
are the opposite case and are the supported way to grow an archive — they write
partitions, index them in `manifest.csv`, and register each fetched symbol in
`constituents.csv`. If you want a symbol kept, run the script for it:

```bash
python scripts/download_sp500.py --tickers BTC-USD --days 5   # stored + registered
```

**There is no history-span control.** A download always takes *everything* available,
and what that means depends on the resolution: roughly 29 days / ~21 sessions / ~7,800
bars on 1-minute, or the instrument's whole listing history on daily. On 1-minute a
shorter fetch is never more accurate — it only shrinks the candidate pool the
percentile is measured against — and Yahoo will not serve older bars at all.

`data/` is only ever **read** *by the app*. Downloaded bars fetched in the app are held
in memory, so nothing there can overwrite an archive you are comparing against. To
grow an archive durably, use `timeseries.store` — or the downloader scripts, which are
the supported way to do it.

```python
from timeseries import fetch

result = fetch.fetch_ticker("nvda")     # whole window; chunked into 7-day requests
print(result.summary())
# NVDA · 1-minute · 7,794 bars · 21 sessions · 2026-09-02 19:02 → 2026-10-01 19:00 (UTC)
print(result.errors)                    # chunks Yahoo refused, if any

daily = fetch.fetch_ticker("nvda", timeframe="1d")   # one request, full history
print(daily.summary())
# NVDA · Daily · 6,415 bars · 6,415 sessions · 2000-01-03 → 2026-10-01 (UTC)

shorter = fetch.fetch_ticker("nvda", days=7)   # only for cheap test runs
```

Two limits are worth knowing, because Yahoo enforces them by returning *nothing*
rather than by raising:

| Limit | Value | Consequence |
|---|---|---|
| Intraday history depth | ~30 days | older 1-minute bars cannot be fetched at all |
| Intraday history per request | ~8 days | longer spans **must** be split, or a single 30-day request silently returns zero rows |
| Daily history depth | decades | none |
| Daily history per request | unbounded | one request covers everything |

**Both limits are intraday-only, and that is why the chunking is not reused on
daily.** `fetch_ticker` branches on the resolution: a daily request goes out as a
single wide call. Reusing the intraday path would not be conservative, it would be
actively wrong — clamping to 29 days would discard ~95% of the history the endpoint
offers, and doing it in 7-day pieces would multiply one request into thousands to
work around a limit that does not exist.

An explicit `days=` is honoured as asked on daily and clamped to the retention window
on 1-minute. `days=None` (the default) means "everything this resolution has".

Because the refusal is silent, `FetchResult.ok` and `FetchResult.errors` exist so
"Yahoo returned nothing" is never mistaken for "this symbol has no data".

Matching directly:

```python
from timeseries.pipeline import Pipeline
from timeseries import matrix_profile as MP

MP.warm_up()                              # pay the Numba JIT cost at startup, not mid-click

pipe = Pipeline.from_csv("data/qqq_1min_20260831_20260930.csv", length=60)
out = pipe.run(k=50, horizons=(5, 15, 30, 60))

for f in out["forecasts"]:
    print(f.h, f.lift, f.p_value, f.sufficient)
```

From downloaded bars instead of a CSV — `from_frame` runs the identical preparation,
so a fetched ticker and a saved archive are directly comparable:

```python
from timeseries import fetch
from timeseries.pipeline import Pipeline

bars = fetch.fetch_ticker("nvda").frame   # every available bar
pipe = Pipeline.from_frame(bars, length=60)
```

## The S&P 500 panel (cross-sectional search)

Each tab searches **one ticker's own history** — "when has QQQ done this?". The
**Panel** tab searches the whole index: "has *anything* in the S&P 500 done this?".

### Fetching the panel

```bash
python scripts/download_sp500.py                      # all 503 constituents, last ~5 sessions
python scripts/download_sp500.py --tickers AAPL NVDA  # a named subset
python scripts/download_sp500.py --incremental        # only sessions not already stored
python scripts/download_sp500.py --rebuild-manifest   # repair the index after an interrupted run
```

**Yahoo only serves ~30 days of 1-minute bars per request**, so this script is built to
be run repeatedly. Each run fetches what it can and appends to the archive; the
archive is the product, the download is just how it grows. See
[Daily upkeep](#daily-upkeep-one-command-a-day) — this is not optional.

Bars are stored partitioned by ticker and Eastern trading day:

```
data/sp500_panel/
  constituents.csv                      # symbol → GICS sector, for same-sector ranks
  manifest.csv                          # one row per (ticker, session)
  ticker=AAPL/date=2026-09-30/bars.parquet
  ticker=NVDA/date=2026-09-30/bars.parquet
  ...
```

`constituents.csv` is the archive's **symbol registry**: it records which symbols the
archive holds and, where it is known, each one's GICS sector. Three properties of it
are worth knowing, because each one is a way the obvious implementation goes wrong:

* **It is cumulative — it merges, it never truncates.** Every symbol a run *actually
  fetched* is added, so `--tickers BTC-USD` puts `BTC-USD` in the archive **and** in the
  registry, and a later full-index run does not delete it. (The original behaviour
  overwrote the file with the scrape result, so the first `--tickers BTC-USD` run
  registered the symbol and the next `--all` run silently erased the row — an archive
  holding the bars with an index blind to them.)
* **A symbol the index cannot label is recorded as `unknown`,** not left blank. The
  distinction matters: `pandas.read_csv` reads a blank cell as `NaN`, and `NaN` is
  *truthy*, so a blank would have passed every sector check in the panel and pooled
  every unlabelled symbol into one fabricated "same sector" distribution. The panel
  reads the registry through `read_sectors`, which drops the marker, so an
  unlabelable symbol reports **no** same-sector percentile rather than a meaningless one.
* **A symbol that returned no bars is not registered.** The registry's claim is that
  the archive holds those bars; a typo or a delisted ticker that yielded nothing does
  not support that claim, and is left out.

Two decisions in that layout are load-bearing:

* **Sessions are named by the Eastern date, not UTC.** A trading session runs
  09:30–16:00 ET, which is 13:30–20:00 UTC in summer and 14:30–21:00 UTC in winter.
  Bucketing on the UTC date cuts a session in half and files the morning of a winter
  day under the previous day.
* **A session is fingerprinted, and a re-fetch is classified rather than
  overwritten.** yfinance retro-adjusts history, so bars fetched today need not equal
  bars fetched last month. A **closed** session whose content changed is flagged as a
  *revision* and the stored copy is kept. A session **still trading** is simply
  accumulated — re-fetching it after two more minutes of trading *must* produce a
  different fingerprint, and treating that as corruption would make every routine
  sync look like a warning, which trains you to ignore the one that matters.

### Daily upkeep: one command a day

```bash
python scripts/download_sp500.py --incremental    # 1-minute panel
python scripts/download_daily.py --incremental    # daily panel
```

**Run the 1-minute one once a trading day.** That archive grows about five sessions
per day, and the window in which those sessions can be downloaded is only ~30 days
wide. The daily archive has no such window and only needs a run after a new session
closes.

The 30-day limit is on **fetching**, not **keeping** — anything downloaded stays on
disk permanently. That asymmetry is the whole operational story:

| | Limit |
|---|---|
| Yahoo will *serve* 1m bars older than | ~30 days |
| Yahoo will *serve* daily bars older than | decades |
| The archive can *hold* bars for | forever |

So the archive is a one-way ratchet. A session that passes the 30-day mark without
being downloaded is **gone permanently** — no later run can recover it, and the gap
silently shrinks the candidate pool every match is ranked against. A week off costs
you a week of history that never comes back.

`--incremental` makes the command safe to run unconditionally: every session already
stored is skipped, so a second run in the same day downloads nothing. Run it again
after the close to fill out the still-trading session; that merges new bars rather
than replacing them.

**From the UI:** the sidebar has a **Sync archive now** button under *Archive
status*, which runs exactly this command, streams its progress, and reports what it
stored. It is the same script, not a second implementation, so the button and the
CLI cannot drift apart. On daily it runs `download_daily.py` against
`data/sp500_daily/` instead — the button follows the resolution chosen at startup.

To automate it, a launchd job or cron entry that runs the same command after the
close is sufficient:

```cron
30 17 * * 1-5 cd /path/to/timeseries && .venv/bin/python scripts/download_sp500.py --incremental >> /tmp/sp500_sync.log 2>&1
```

If a run is interrupted, `python scripts/download_sp500.py --rebuild-manifest` repairs
the index from the partitions on disk. It never touches the bars themselves.

### The daily archive

Daily mode has **its own archive**, at `data/sp500_daily/`. It is not the 1-minute
archive queried at a different interval, and it is not gated off.

```bash
python scripts/download_daily.py                    # all 503 constituents, full history
python scripts/download_daily.py --tickers AAPL NVDA
python scripts/download_daily.py --incremental      # only sessions not already stored
python scripts/download_daily.py --rebuild-manifest # repair the index after an interrupted run
```

Three things make daily different from the 1-minute archive:

* **No retention wall.** Yahoo serves daily bars going back decades, so this is a
  one-time build, not a daily chore. Run it once; `--incremental` afterwards only
  adds new sessions.
* **Full history, not a window.** The default lookback is **17,000 days (~46.5
  years)**, chosen to reach past the oldest listing in the index. AAPL resolves to
  11,544 sessions from 1980-12-12. A shorter default would not raise an error — it
  would produce a shorter archive that passes every consistency check and is simply
  missing the past, which is why the script logs the earliest date it observed.
* **One file per ticker, not one per session.** Daily is one row per session, so the
  1-minute layout would create one file *and one directory per day of history*:
  measured on AAPL alone, **90MB and 11,544 directories**. Across the panel that
  projects to ~45GB and 5.8 million directories. Storing one file per ticker instead
  is **546KB** for AAPL and ~275MB for the whole index:

  ```
  data/sp500_daily/
    constituents.csv
    manifest.csv                       # still one row per (ticker, session)
    ticker=AAPL/bars.parquet           # AAPL's entire daily history, one file
    ticker=NVDA/bars.parquet
    ...
  ```

**Why the build is fast enough to finish.** Writing one file per ticker is what makes
the layout affordable, but it means the manifest — an index written as a unit — is
rewritten once per ticker, and on daily it holds one row per session. Measured on the
real archive, a single rewrite cost **5.1s** (2.4s of it the `to_csv` of the whole
frame). Written 503 times that is 503 growing rewrites, and the build **does not
finish**: the first run of this sat at 25 minutes of CPU with zero new partitions on
disk.

Three fixes, all in `timeseries/store.py`:

| Change | Measured |
|---|---|
| `batched_manifest()` — one manifest rewrite per run, not per ticker | 5 new tickers: 20.6s → 8.3s |
| `fingerprint_per_session()` — all of a ticker's session hashes in one pass | AAPL, 11,544 sessions: 3.5s → 0.10s (**36×**), byte-identical hashes |
| `_record_daily_revisions()` — vectorised; was scanning the whole frame per revised session | ticker A, 5,410 revised sessions: ~295s → 0.28s |

Full build now: **503 tickers, 4.16M sessions, ~9 minutes.** Two properties make the
fast path safe rather than merely quick:

* **Deferred is not dropped.** `batched_manifest` flushes in a `finally`, so an
  interrupted run indexes what it already wrote — which is exactly the state the
  original interrupted build was left in, with 11,730 sessions on disk and absent
  from the manifest.
* **The hashes are the same hashes.** `fingerprint_per_session` exists only for
  speed, and speed is worthless if it hashes different bytes: every stored session
  would then differ from its recorded fingerprint and the whole archive would
  present as revised. The equivalence is asserted in `tests/test_daily_archive.py`
  against the real AAPL history and against `-0.0`, subnormals and NaN.

The manifest is also written **atomically** — to a temp file, then `os.replace` — so
a concurrent reader (including the app) sees the old index or the new one, never a
half-written 545MB CSV that would read as "these sessions exist but were never
indexed".

The `session` column is what carries the per-session decision — unchanged, revised,
in-progress — so nothing is lost by putting the sessions in one file. The manifest
still has **one row per session** on both layouts, which is what lets
`verify_manifest` mean the same thing either way.

The **Sync archive now** sidebar button and the *build this archive* link both
follow the resolution chosen at startup, so on daily they run `download_daily.py`
against `data/sp500_daily/`. There is no separate daily UI: the Panel tab reads
whichever archive the resolution selected.

The button is also a valid way to *build* the archive from empty. It widens the
requested window automatically: when the archive holds no sessions it passes no
window at all, so the downloader fetches full listing history rather than the
29-day lookback. A populated archive keeps the narrow window, which is what keeps
daily presses cheap.

### Keeping the daily archive current

```bash
python scripts/download_daily.py --incremental
```

Daily has no 30-day wall, so this is a convenience rather than an obligation —
nothing is lost by waiting. It is also cheap and safe to run as often as you like:
measured against the built archive, an unchanged re-write of AAPL's full 11,544
sessions takes **0.23s** and reports `sessions_unchanged=11544, sessions_revised=0`
without touching a byte. A re-fetch only writes when content actually differs, and
a session that is still trading is merged rather than flagged.

Two operational notes:

* **The first run is slow and later ones are not.** The initial build fetches 503
  tickers of full history. Afterwards, `--incremental` fetches only sessions the
  archive is missing.
* **`manifest.csv` is larger than the data — this is expected.** It holds one row
  per *session*, so it projects to **~4.1M rows / ~567MB** against ~250MB of
  Parquet. That is the price of keeping per-session rows on a layout that stores one
  file per ticker, and it buys the thing that matters: `verify_manifest` and a
  revised day reported *by date* mean the same thing on both layouts. It is a
  derived index — the archive works from the tree on disk — and it is read
  **lazily**, so app startup never pays for it.

### Searching it

The two archives are searched the same way — only the root and the horizon differ,
because a horizon is in **bars**, and a daily bar is a different length of time
than a minute one:

```python
import pandas as pd
from timeseries.store import PanelStore
from timeseries.panel import PanelSearch

meta = pd.read_csv("data/sp500_panel/constituents.csv", dtype={"symbol": str})
sectors = dict(zip(meta["yahoo_symbol"], meta["sector"]))

# 1-minute: the last 30 bars is half an hour of trading.
minute = PanelSearch(PanelStore("data/sp500_panel"), sectors=sectors)
out = minute.run(minute.latest_query("AAPL", length=30), k=15, max_per_ticker=2)

# Daily: the last 30 bars is a trading quarter, from the daily archive.
daily = PanelSearch(PanelStore("data/sp500_daily"), sectors=sectors)
out = daily.run(daily.latest_query("AAPL", length=30), k=15, max_per_ticker=2)

for m in out["result"].matches:
    print(m.ticker, m.session, m.percentile, m.percentile_same_ticker)
```

493 tickers / ~684,000 candidate windows score in **under a second** on the
1-minute archive.

Use `horizons_for("1m")` and `horizons_for("1d")` rather than hard-coding lengths
if you want the app's own defaults — the UI derives them from the resolution chosen
at startup.

### Why three percentiles

Over one ticker, a rank is comparable to itself. Over 500 names the candidate
population is dominated by whichever tickers happened to be quiet that week, and a
liquid mega-cap is a much harder shape to resemble than a thinly-traded small cap. So
every match reports three ranks:

| Field | Question it answers |
|---|---|
| `percentile` | how unusual across the **whole panel** |
| `percentile_same_ticker` | how unusual for **this stock** |
| `percentile_same_sector` | how unusual within its **GICS sector** |

A window can be the closest thing its own ticker has ever seen and still be
unremarkable across the index — because a dozen other names moved the same way.
Collapsing that into one number would hide exactly the distinction this search exists
to make.

Note that both are *fractions*, so they are not ordered against each other: 0 windows
closer out of 821 is `0.0%`, while 2 out of 3,284 is `0.06%`. What holds is the
*count*: a global population is a superset of any ticker's own, so the number of
windows closer to a match can only grow when you widen the pool.

### Why a per-ticker cap

When a sector moves together, the best match is not a company — it is the factor.
Without `max_per_ticker`, the top 12 is 12 windows of the same three names, and the
forecast would treat one event as twelve observations and report a p-value several
times better than it deserves.

### Why a window never spans two tickers

A matrix profile is defined by sliding a query over **one contiguous series**. Were
the panel concatenated into a single array, a window at a ticker boundary would
compare the tail of one company to the head of another — a pattern that never happened
in the market. Each ticker is therefore scored with its own `stumpy.mass` pass and the
distances are pooled, which is why a match carries both a ticker and an index within
that ticker.

## Matching: STUMPY

Stage 1 of the funnel is a **matrix profile** computed by
[`stumpy`](https://github.com/stumpy-dev/stumpy). One compiled pass scores every
candidate window in the archive, which is what makes interactive search viable.

**STUMPY is the only scorer, and there is no `method` argument.** This is not a
default that can be overridden — it is the only implementation.

Two alternatives were removed rather than kept, and both for reasons worth stating:

- **A plain numpy Euclidean scan.** It produced the *same ranking* — STUMPY's
  normalized distance *is* the Euclidean distance between z-scored windows — so it
  bought nothing but a slower route to the same answer. STUMPY computes that same
  answer in roughly 7× less time, with no memory spike for a materialised window
  library.
- **A banded-DTW option.** A genuinely different metric, but its percentile had to be
  reported as `NaN`: a time-warped distance cannot be ranked against the Euclidean
  distribution the percentile is computed from (PLAN.md §BD). Since the percentile is
  the module's headline honesty feature, an option that could only ever show a blank
  in that column was worse than no option at all.

Because `stumpy.mass` slides a query over **one contiguous series**, it cannot score a
pre-materialised library of overlapping windows — those windows are not a contiguous
series. Any "numpy fallback" would therefore have been a different metric over a
different candidate set, not a substitute for STUMPY. That path is gone, and
`find_matches(series, query)` now takes the contiguous feature matrix directly.

STUMPY is a **required** dependency. `matrix_profile.py` raises at import if it is
missing rather than silently degrading to a slower, differently-ranked scorer.

The exclusion zone (§M) is applied in this codebase, not delegated to STUMPY: its default
suppresses only the trivial match, while the plan requires a full window length so the
query's near-identical neighbours are excluded too.

## Project Structure

- `src/timeseries/` — the library
  - `features.py` — z-scored feature construction
  - `matrix_profile.py` — STUMPY primitives (profile, distance, discords)
  - `matching.py` — the funnel, exclusion and suppression over one STUMPY scorer
  - `forecast.py` — conditional forecast with baseline, permutation test, block bootstrap
  - `backtest.py` — walk-forward evaluation
  - `pipeline.py` — the single entry point; every forecast has a baseline
  - `fetch.py` — on-demand download for an arbitrary ticker, at either resolution
  - `timeframes.py` — the resolution registry: every per-timeframe constant, with the
    reasoning for each, in one table
  - `store.py` — partitioned Parquet archive with per-session fingerprints
  - `panel.py` — cross-sectional search across many tickers
  - `sync.py` — runs the daily archive sync as a child process, for the UI button
  - `placebo.py` — the guard that runs the pipeline on noise
- `tests/` — including the placebo test, the single most important test here
- `data/` — bar CSVs, and `data/sp500_panel/` / `data/sp500_daily/` for the multi-ticker archives
- `scripts/` — `download_sp500.py` builds the 1-minute panel, `download_daily.py` the
  daily one; both are what the sidebar's *Sync archive now* button shells out to
- `app.py` — Streamlit dashboard
- `PLAN.md` — design, review findings, and corrections

## Testing

`pytest` runs 346 tests. The one that matters most is `TestPlacebo`: it runs the whole
pipeline on synthetic random-walk data and asserts it reports **nothing significant**. A
pattern finder that finds patterns in pure noise is broken, however good it looks on
real data.

`tests/test_panel.py` covers the two claims that fail *quietly* if broken: the archive
must stay idempotent (a re-fetch must not be mistaken for a data revision), and a
cross-sectional window must never span two tickers.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Starting the UI

To launch the Streamlit dashboard:

1. Ensure you are in the project directory:
   ```bash
   cd /Users/thomasdsilva/Projects/timeseries
   ```

2. Activate the virtual environment (if not already activated):
   ```bash
   source .venv/bin/activate
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

   Alternatively, you can run it directly with the virtual environment's Python:
   ```bash
   .venv/bin/python -m streamlit run app.py
   ```

The UI will be available at **http://localhost:8599**.

The port is pinned in `.streamlit/config.toml`, so it applies to every launch path
above — no `--server.port` flag needed. If 8599 is already taken, Streamlit falls back
to a random port and prints the URL it actually bound to; trust the terminal over this
document in that case.