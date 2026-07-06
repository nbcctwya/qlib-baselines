"""Collect metrics from every (market, model, seed) MLflow experiment, aggregate
across seeds, and emit comparison tables + bar charts (with seed-std error bars).

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
  comparison_table_raw.csv    every (market, model, seed) row
  comparison_table.csv        MEAN across seeds, one row per (market, model)
  comparison_table_std.csv    STD across seeds
  bar_{<metric>}.png          mean with seed-std error bars

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

from configs import MARKETS, MODELS, SEEDS, MLRUNS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

METRICS_TO_PLOT = ["IC", "RankIC", "AnnExcess_Net", "AnnReturn_Net"]
COLUMN_ORDER = [
    "IC", "ICIR", "RankIC", "RankICIR",
    "AnnReturn_Gross", "AnnReturn_Net", "AnnExcess_Gross", "AnnExcess_Net",
    "Sharpe", "MaxDrawdown", "BenchReturn", "Turnover",
]


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


def collect_one(market_key: str, model_name: str, seed: int) -> dict:
    exp_name = f"{market_key}_{model_name}_seed{seed}"
    try:
        rec = R.get_recorder(experiment_name=exp_name)
    except Exception as e:
        print(f"  skip {exp_name}: no recorder ({e})")
        return None

    row = {"market": market_key, "model": model_name, "seed": seed}

    try:
        ic = rec.load_object("sig_analysis/ic.pkl")
        ric = rec.load_object("sig_analysis/ric.pkl")
        row["IC"], row["ICIR"] = _safe_mean_std(ic)
        row["RankIC"], row["RankICIR"] = _safe_mean_std(ric)
    except Exception as e:
        print(f"  {exp_name}: IC artifacts missing ({e})")

    try:
        pa = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")
        def pick(group, sub):
            return pa.loc[(group, sub), "risk"]
        row["AnnExcess_Gross"] = pick("excess_return_without_cost", "annualized_return")
        row["AnnExcess_Net"] = pick("excess_return_with_cost", "annualized_return")
        row["Sharpe"] = pick("excess_return_with_cost", "information_ratio")
        row["MaxDrawdown"] = pick("excess_return_with_cost", "max_drawdown")
    except Exception as e:
        print(f"  {exp_name}: port_analysis missing ({e})")

    try:
        rep = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        row["AnnReturn_Gross"] = _ann(rep["return"])
        row["AnnReturn_Net"] = _ann(rep["return"] - rep["cost"])
        row["BenchReturn"] = _ann(rep["bench"])
        row["Turnover"] = rep["turnover"].mean()
    except Exception as e:
        print(f"  {exp_name}: report_normal missing ({e})")

    return row


def make_charts(mean: pd.DataFrame, std: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mean_r = mean.reset_index()
    std_r = std.reset_index()
    for metric in METRICS_TO_PLOT:
        if metric not in mean_r.columns:
            continue
        m = mean_r.pivot(index="model", columns="market", values=metric)
        s = std_r.pivot(index="model", columns="market", values=metric) if metric in std_r.columns else None
        ax = m.plot.bar(figsize=(8, 5), yerr=s, capsize=3,
                        error_kw={"elinewidth": 0.8, "alpha": 0.6})
        ax.set_title(f"{metric} (mean over seeds, error bars = std)")
        ax.set_ylabel(metric)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(title="market")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"bar_{metric}.png", dpi=120)
        plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    rows = []
    for mk in MARKETS:
        R.set_uri(f"file:{(MLRUNS_DIR / mk).resolve()}")
        for mdl in MODELS:
            for seed in SEEDS:
                row = collect_one(mk, mdl, seed)
                if row is not None:
                    rows.append(row)
    if not rows:
        raise SystemExit("no completed experiments found - run `python run.py` first")

    raw = pd.DataFrame(rows).set_index(["market", "model", "seed"]).sort_index()
    raw = raw[[c for c in COLUMN_ORDER if c in raw.columns]]
    raw.to_csv(REPORTS_DIR / "comparison_table_raw.csv")

    mean = raw.groupby(level=["market", "model"]).mean()
    std = raw.groupby(level=["market", "model"]).std()
    mean = mean[[c for c in COLUMN_ORDER if c in mean.columns]]
    std = std[[c for c in COLUMN_ORDER if c in std.columns]]
    mean.to_csv(REPORTS_DIR / "comparison_table.csv")
    std.to_csv(REPORTS_DIR / "comparison_table_std.csv")

    print(f"\n=== MEAN across {len(SEEDS)} seeds -> reports/comparison_table.csv ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(mean.to_string())

    print(f"\n=== STD across seeds -> reports/comparison_table_std.csv ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(std.to_string())

    make_charts(mean, std)
    print(f"\ncharts -> {REPORTS_DIR}/bar_{{{'|'.join(METRICS_TO_PLOT)}}}.png  (error bars = seed std)")


if __name__ == "__main__":
    main()
