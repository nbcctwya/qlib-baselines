"""Paper-convention evaluation metrics.

This module implements the metric definitions specified for the paper, which
differ from qlib's built-in `risk_analysis` (sum mode, x238). The functions here
are intentionally separate so the existing `collect_results.py` (qlib convention)
keeps working unchanged.

Conventions (documented once here):
  * All returns fed to the portfolio metrics are **daily simple returns AFTER
    cost**. From qlib's report_normal, `daily_portfolio_return_after_cost =
    report["return"] - report["cost"]` (verified: report["return"] is the
    without-fee portfolio return, report["cost"] is the fee; their net cumulative
    product equals the account growth exactly, so subtracting cost once is
    correct and there is NO double-counting).
  * We then work in **daily log return**  g_t = log(1 + daily_simple_ret_after_cost).
  * Annualisation factor is **252** (trading days), applied as sqrt(252) for
    ratios and *252 in the exponent for compounded return.
  * Risk-free rate R_f = 0.
  * `std` uses **ddof=1** (sample standard deviation) everywhere — this is the
    pandas `Series.std()` default and matches qlib's `calc_ic` ICIR. numpy's
    default is ddof=0, so we always pass ddof explicitly.
  * NaN daily IC/RankIC (from days with <2 stocks, all-constant, or missing
    values) are dropped by `.mean()`/`.std()` automatically (skipna=True default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN_FACTOR = 252  # trading days per year


# --------------------------------------------------------------------------- #
# 1. Ranking / prediction metrics (IC family)
# --------------------------------------------------------------------------- #
def compute_prediction_metrics(pred: pd.Series, label: pd.Series) -> dict:
    """Daily cross-sectional IC / RankIC and their information ratios.

    For each trading day we correlate prediction vs label across instruments
    (Pearson -> IC, Spearman -> RankIC), then average over the test period.
    Reuses qlib's `calc_ic` so the result matches `sig_analysis/ic.pkl`,
    `ric.pkl` exactly.

    IC      = mean(IC_t)
    ICIR    = mean(IC_t) / std(IC_t)
    RankIC  = mean(RankIC_t)
    RankICIR= mean(RankIC_t) / std(RankIC_t)

    ddof=1 (pandas default). Days that yield NaN correlation (constant column,
    <2 valid stocks, all-NaN) are skipped by mean/std.
    """
    from qlib.contrib.eva.alpha import calc_ic  # local import to avoid hard dep at import time

    # Accept DataFrames too: take the first (only) column. pred col is 'score'.
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    if pred is None or label is None or len(pred) == 0 or len(label) == 0:
        return {k: np.nan for k in ("IC", "ICIR", "RankIC", "RankICIR")}

    ic, ric = calc_ic(pred, label)  # daily series indexed by datetime

    def _mean_ir(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return np.nan, np.nan
        m = s.mean()
        sd = s.std(ddof=1)
        ir = m / sd if sd and sd > 0 else np.nan
        return float(m), float(ir)

    ic_m, ic_ir = _mean_ir(ic)
    ric_m, ric_ir = _mean_ir(ric)
    return {"IC": ic_m, "ICIR": ic_ir, "RankIC": ric_m, "RankICIR": ric_ir}


# --------------------------------------------------------------------------- #
# 2. Portfolio metrics (log-return convention, sqrt(252) annualisation)
# --------------------------------------------------------------------------- #
def _to_log(daily_ret_after_cost) -> pd.Series:
    """daily simple return (after cost) -> daily log return g_t = log(1+r)."""
    s = pd.Series(daily_ret_after_cost).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return np.log1p(s)


def compute_ar(daily_log_ret: pd.Series) -> float:
    """Annualised return (compounded): AR = exp(mean(g)*252) - 1."""
    if len(daily_log_ret) == 0:
        return np.nan
    return float(np.exp(daily_log_ret.mean() * ANN_FACTOR) - 1)


def compute_std(daily_log_ret: pd.Series) -> float:
    """Annualised volatility: STD = std(g, ddof=1) * sqrt(252)."""
    if len(daily_log_ret) < 2:
        return np.nan
    return float(daily_log_ret.std(ddof=1) * np.sqrt(ANN_FACTOR))


def compute_mdd(daily_log_ret: pd.Series) -> float:
    """Max drawdown on the log-nav curve. nav = exp(cumsum(g)); returns a <=0 number."""
    if len(daily_log_ret) == 0:
        return np.nan
    nav = np.exp(daily_log_ret.cumsum())
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def compute_log_return_sharpe(daily_log_ret: pd.Series) -> float:
    """Sharpe = sqrt(252) * mean(g) / std(g, ddof=1), R_f = 0."""
    if len(daily_log_ret) < 2:
        return np.nan
    sd = daily_log_ret.std(ddof=1)
    if not sd or sd == 0:
        return np.nan
    return float(np.sqrt(ANN_FACTOR) * daily_log_ret.mean() / sd)


def compute_sortino(daily_log_ret: pd.Series) -> float:
    """Sortino = sqrt(252) * mean(g) / std(negative g, ddof=1).

    Downside set = {g_t : g_t < 0} (strictly negative log returns only).
    """
    if len(daily_log_ret) == 0:
        return np.nan
    neg = daily_log_ret[daily_log_ret < 0]
    if len(neg) < 2:
        # No (or too few) down-days -> downside deviation undefined.
        return np.nan
    dd = neg.std(ddof=1)
    if not dd or dd == 0:
        return np.nan
    return float(np.sqrt(ANN_FACTOR) * daily_log_ret.mean() / dd)


def compute_calmar(ar: float, mdd: float) -> float:
    """Calmar = AR / abs(MDD). Uses abs so the sign tracks AR (MDD is <=0)."""
    if np.isnan(ar) or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(ar / abs(mdd))


def compute_portfolio_metrics(daily_ret_after_cost) -> dict:
    """All six portfolio metrics from a daily AFTER-cost simple-return series."""
    g = _to_log(daily_ret_after_cost)
    ar = compute_ar(g)
    std = compute_std(g)
    mdd = compute_mdd(g)
    return {
        "AR": ar,
        "STD": std,
        "MDD": mdd,
        "Sharpe": compute_log_return_sharpe(g),
        "Sortino": compute_sortino(g),
        "Calmar": compute_calmar(ar, mdd),
        "num_test_days": int(len(g)),
    }
