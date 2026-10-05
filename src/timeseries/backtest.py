"""Walk-forward backtest.  Fixes PLAN.md §Z3.

The original ``scripts/backtest.py`` reported a direction accuracy and a Sharpe ratio
that were not interpretable:

1. No exclusion zone, so overlapping stride-1 windows returned the same moment as
   several "distinct" matches.
2. Predictions were stepped by one bar while sharing a multi-bar forward horizon, so
   consecutive rows were near-duplicates treated as independent.  The effective
   sample size was roughly ``len(rows) / horizon``.
3. Sharpe was annualised with ``sqrt(252 * 60 * 5)``, implying ~13,600 periods per
   year.  That is not QQQ's trading calendar, and with dependent rows the numerator
   and denominator are both wrong.

This harness steps by a non-overlapping stride, reports accuracy with a block
bootstrap interval rather than a bare percentage, and reports no Sharpe at all unless
costs are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import matching as M
from .forecast import block_bootstrap_ci, permutation_test
from .pipeline import Pipeline

__all__ = ["BacktestResult", "walk_forward", "evaluate_predictions"]


@dataclass
class BacktestResult:
    n_predictions: int
    stride: int
    direction_accuracy: float
    accuracy_ci: tuple
    n_baseline: int
    baseline_accuracy: float
    lift: float
    p_value: float
    mean_actual: float
    horizon: int
    fee_bps: float = 0.0
    net_mean: float = float("nan")
    net_hit_rate: float = float("nan")
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_predictions": self.n_predictions,
            "stride": self.stride,
            "direction_accuracy": self.direction_accuracy,
            "accuracy_ci_low": self.accuracy_ci[0],
            "accuracy_ci_high": self.accuracy_ci[1],
            "baseline_accuracy": self.baseline_accuracy,
            "lift": self.lift,
            "p_value": self.p_value,
            "mean_actual": self.mean_actual,
            "horizon_min": self.horizon,
            "fee_bps": self.fee_bps,
            "net_mean": self.net_mean,
            "net_hit_rate": self.net_hit_rate,
        }


def evaluate_predictions(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    horizon: int,
    stride: int,
    fee_bps: float = 0.0,
    n_perm: int = 2000,
    seed: int = 0,
) -> BacktestResult:
    """Score a set of predictions against realised returns.

    ``predicted`` holds the expected forward return implied by each match set;
    ``actual`` holds the realised forward return.  A "hit" is
    ``sign(predicted) == sign(actual)``, which is the honest measure here -- the
    magnitude of the conditional mean is not reliably predictable even when its sign
    sometimes is.

    The baseline is the hit rate obtained by predicting with randomly chosen windows of
    the same length.  A strategy that cannot beat it is not a strategy (§D).
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(predicted) & np.isfinite(actual)
    predicted, actual = predicted[ok], actual[ok]

    if predicted.size == 0:
        return BacktestResult(0, stride, float("nan"), (float("nan"), float("nan")),
                              0, float("nan"), float("nan"), float("nan"),
                              float("nan"), horizon, fee_bps)

    hits = (np.sign(predicted) == np.sign(actual))
    acc = float(hits.mean())
    ci = block_bootstrap_ci(hits.astype(float), block=max(1, stride), n_boot=1000, seed=seed)

    # Permutation baseline (§D): the accuracy obtainable by shuffling predictions
    # against realised returns.  Because signs are roughly balanced this sits near
    # 0.5, and a strategy that cannot beat it has nothing to offer.
    p = permutation_test(predicted, actual, n_perm=n_perm, seed=seed)
    base_hits = (np.sign(np.random.default_rng(seed).permutation(predicted)) == np.sign(actual))
    base_acc = float(base_hits.mean())

    warnings = []
    net_mean = float("nan")
    net_hit = float("nan")
    if fee_bps > 0:
        cost = fee_bps / 1e4
        net = np.sign(predicted) * actual - cost
        net_mean = float(net.mean())
        net_hit = float((net > 0).mean())
    else:
        warnings.append(
            "no transaction costs applied; 1-minute QQQ round-trip costs are "
            "typically 1-3 bps, which can exceed the edge being measured"
        )

    return BacktestResult(
        n_predictions=int(predicted.size),
        stride=stride,
        direction_accuracy=acc,
        accuracy_ci=ci,
        n_baseline=int(base_hits.size),
        baseline_accuracy=base_acc,
        lift=acc - base_acc,
        p_value=p,
        mean_actual=float(actual.mean()),
        horizon=horizon,
        fee_bps=fee_bps,
        net_mean=net_mean,
        net_hit_rate=net_hit,
        warnings=warnings,
    )


def walk_forward(
    pipe: Pipeline,
    *,
    k: int = 40,
    horizon: int = 15,
    stride: int | None = None,
    warmup: int = 500,
    fee_bps: float = 0.0,
    n_perm: int = 500,
    seed: int = 0,
    max_steps: int | None = None,
) -> BacktestResult:
    """Re-run the matcher at successive points in time, using only prior data.

    At each step ``i`` the matcher sees bars ``[0, i)``, predicts the return from
    ``i`` to ``i + horizon``, and the realised return is compared.  Steps are spaced by
    ``stride >= horizon`` so that no two predictions share bars -- the fix for Z3.

    A rolling re-search over every step is expensive; ``stride`` trades resolution for
    runtime.  This is still a walk-forward test, not a single split.
    """
    if not pipe.ready:
        return BacktestResult(0, stride or horizon, float("nan"),
                              (float("nan"), float("nan")), 0, float("nan"),
                              float("nan"), float("nan"), float("nan"), horizon)

    n = pipe.n_bars
    step = stride or max(horizon, pipe.length // 2)
    preds, acts = [], []

    # Raw log returns for the amplitude term, on the same rows as ``pipe.matrix``.
    # PLAN.md §BV: omitting this falls back to summing channel 0 of the feature
    # matrix, which is ``return_z`` -- a rolling z-score, so the "realised move" is a
    # sum of z-scores rather than a return.  ``Pipeline.match`` passes this same
    # series; without it here the walk-forward search ranks windows by an artefact
    # and the p-value ``evaluate_predictions`` reports is measuring that artefact.
    amp_series = pipe.log_returns

    i = warmup
    n_steps = 0
    while i + horizon < n:
        history_matrix = pipe.matrix[:i]
        if len(history_matrix) < pipe.length + 50:
            break

        # PLAN.md §BX: same censoring as `Pipeline.run`.  The walk-forward forward
        # return is a forward return, so a match whose horizon crosses an overnight
        # closure reports the gap instead of `horizon` minutes of trading.
        #
        # The mask is built on the **history** slice, and this is the whole subtlety:
        # building it from the full series would leak the shape of later session
        # boundaries into a step that is supposed to be blind, which is lookahead by
        # another name.  `n_starts` bars of history admit `n_starts - L + 1` window
        # starts, and that is exactly the shape `find_matches` requires -- an earlier
        # draft passed the bar-length mask and tripped the validation.
        n_starts = len(history_matrix)
        vm_full = pipe.forward_horizon_mask(pipe.length, horizon)
        vm = None if vm_full is None else vm_full[:n_starts - pipe.length + 1]

        # Search history only -- no lookahead.  STUMPY slides over the contiguous
        # feature matrix, so there is no window library to rebuild per step.
        q = M.Query.from_span(history_matrix, n_starts - pipe.length, n_starts)
        res = M.find_matches(history_matrix, q, k=k,
                             amplitude_series=amp_series[:i],
                             valid_mask=vm)

        mstarts = np.array([m.start for m in res.matches], dtype=np.int64)
        if mstarts.size >= 5:
            # `pipe.close` is the *whole* series, so forward returns are anchored on
            # the match's own last bar -- which, in a walk-forward step, is always in
            # the past.  The mask above guarantees none of those anchors sits close
            # enough to a closure for the horizon to reach it.
            fwd = M.forward_returns(pipe.close, mstarts, pipe.length, [horizon])[horizon]
            fwd = fwd[np.isfinite(fwd)]
            if fwd.size >= 5:
                preds.append(float(fwd.mean()))
                acts.append(float(np.log(pipe.close[i + horizon] / pipe.close[i])))

        i += step
        n_steps += 1
        if max_steps and n_steps >= max_steps:
            break

    return evaluate_predictions(
        np.array(preds), np.array(acts),
        horizon=horizon, stride=step, fee_bps=fee_bps, n_perm=n_perm, seed=seed,
    )