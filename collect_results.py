"""Collect metrics from every (market, model) MLflow experiment into a comparison
table + bar charts.

For each experiment it reads these recorder artifacts (paths verified in qlib
0.9.7):
  sig_analysis/ic.pkl                 daily IC series
  sig_analysis/ric.pkl                daily rank-IC series
  portfolio_analysis/port_analysis_1day.pkl   excess-return risk metrics
  portfolio_analysis/report_normal_1day.pkl   daily return/cost/turnover

All return metrics use qlib's risk_analysis (mode="sum", freq="day"), so they are
on the SAME annualisation basis: annualized = mean(daily) * 238 (arithmetic).

Column meaning
  IC, ICIR, RankIC, RankICIR   signal quality (daily mean / mean-over-std)
  AnnReturn_Gross              年化收益·不扣费·不减基准   = risk_analysis(return)
  AnnReturn_Net                年化收益·扣费  ·不减基准   = risk_analysis(return - cost)
  AnnExcess_Gross              毛超额 = 组合-基准·不扣费  = risk_analysis(return - bench)
  AnnExcess_Net                净超额 = 组合-基准·扣费    = risk_analysis(return - bench - cost)
  Sharpe                       information ratio of NET excess return
  MaxDrawdown                  max drawdown of NET excess return
  BenchReturn                  benchmark 年化 (SH000300 / ^gspc)
  Turnover                     mean daily turnover

Outputs (under reports/):
  comparison_table.csv         one row per (market, model)
  bar_{<metric>}.png

Usage:
    python collect_results.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qlib
from qlib.contrib.evaluate import risk_analysis
from qlib.workflow import R

from configs import MARKETS, MODELS, MLRUNS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

# Charts: signal quality + the two most decision-relevant returns.
METRICS_TO_PLOT = ["IC", "RankIC", "AnnExcess_Net", "AnnReturn_Net"]


def _safe_mean_std(s: pd.Series):
    """(mean, mean/std) of a daily series; ICIR = mean/std."""
    if s is None or len(s) == 0:
        return np.nan, np.nan
    m, sd = s.mean(), s.std()
    return m, (m / sd if sd != 0 else np.nan)


def _ann(r: pd.Series) -> float:
    """Annualised return (sum mode, x238) of a daily return series."""
    if r is None or len(r) == 0:
        return np.nan
    return float(risk_analysis(r, freq="day").loc["annualized_return", "risk"])


def collect_one(market_key: str, model_name: str) -> dict:
    exp_name = f"{market_key}_{model_name}"
    try:
        rec = R.get_recorder(experiment_name=exp_name)
    except Exception as e:
        print(f"  skip {exp_name}: no recorder ({e})")
        return None

    row = {"market": market_key, "model": model_name}

    # ---- signal quality (IC family) ----
    try:
        ic = rec.load_object("sig_analysis/ic.pkl")
        ric = rec.load_object("sig_analysis/ric.pkl")
        row["IC"], row["ICIR"] = _safe_mean_std(ic)
        row["RankIC"], row["RankICIR"] = _safe_mean_std(ric)
    except Exception as e:
        print(f"  {exp_name}: IC artifacts missing ({e})")

    # ---- excess-return risk metrics (precomputed by PortAnaRecord) ----
    try:
        pa = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")
        # pa index: (excess_return_without_cost|excess_return_with_cost,
        #            mean|std|annualized_return|information_ratio|max_drawdown); col "risk"
        def pick(group, sub):
            return pa.loc[(group, sub), "risk"]
        row["AnnExcess_Gross"] = pick("excess_return_without_cost", "annualized_return")  # 毛超额
        row["AnnExcess_Net"] = pick("excess_return_with_cost", "annualized_return")       # 净超额
        row["Sharpe"] = pick("excess_return_with_cost", "information_ratio")
        row["MaxDrawdown"] = pick("excess_return_with_cost", "max_drawdown")
    except Exception as e:
        print(f"  {exp_name}: port_analysis missing ({e})")

    # ---- portfolio-only returns + benchmark + turnover (from daily report) ----
    try:
        rep = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        row["AnnReturn_Gross"] = _ann(rep["return"])               # 年化收益·不扣费·不减基准
        row["AnnReturn_Net"] = _ann(rep["return"] - rep["cost"])   # 年化收益·扣费  ·不减基准
        row["BenchReturn"] = _ann(rep["bench"])                    # 基准年化
        row["Turnover"] = rep["turnover"].mean()
    except Exception as e:
        print(f"  {exp_name}: report_normal missing ({e})")

    return row


# display order
COLUMN_ORDER = [
    "IC", "ICIR", "RankIC", "RankICIR",
    "AnnReturn_Gross", "AnnReturn_Net", "AnnExcess_Gross", "AnnExcess_Net",
    "Sharpe", "MaxDrawdown", "BenchReturn", "Turnover",
]


def make_charts(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS_TO_PLOT:
        if metric not in df.columns:
            continue
        pivot = df.pivot(index="model", columns="market", values=metric)
        ax = pivot.plot.bar(figsize=(8, 5))
        ax.set_title(f"{metric} by model x market")
        ax.set_ylabel(metric)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(title="market")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"bar_{metric}.png", dpi=120)
        plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # R.get_recorder touches qlib's global config, so qlib must be initialised.
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    rows = []
    for mk in MARKETS:
        R.set_uri(f"file:{(MLRUNS_DIR / mk).resolve()}")   # required before get_recorder outside `with`
        for mdl in MODELS:
            row = collect_one(mk, mdl)
            if row is not None:
                rows.append(row)

    if not rows:
        raise SystemExit("no completed experiments found - run `python run.py` first")

    df = pd.DataFrame(rows).set_index(["market", "model"]).sort_index()
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]
    csv_path = REPORTS_DIR / "comparison_table.csv"
    df.to_csv(csv_path)
    print(f"\n=== comparison table -> {csv_path} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string())

    make_charts(df.reset_index())
    print(f"charts -> {REPORTS_DIR}/bar_{{{'|'.join(METRICS_TO_PLOT)}}}.png")


if __name__ == "__main__":
    main()
