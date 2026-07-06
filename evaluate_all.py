"""Unified evaluation of all trained (market, model, seed) checkpoints.

For every (market, model, seed) it reads the artifacts that `run.py` already
saved in MLflow:
  pred.pkl                                  test-segment predictions (col 'score')
  label.pkl                                 test-segment labels
  portfolio_analysis/report_normal_1day.pkl daily return/cost/bench/turnover

and computes:
  - IC / ICIR / RankIC / RankICIR  (daily cross-sectional; see eval_metrics.compute_prediction_metrics)
  - AR / STD / MDD / Sharpe / Sortino / Calmar (log-return, sqrt(252); see eval_metrics.compute_portfolio_metrics)

The TopK-DropN backtest (K=30, N=5, buy 5bps / sell 15bps) is REUSED from run.py -
this script only recomputes metrics with the paper convention, it does NOT
re-backtest. The actual backtest parameters are read from configs.py and recorded
into results/eval_run_config.json for traceability.

Outputs (under <out>/, default results/):
  eval_detail.csv    one row per (market, model, seed)
  eval_summary.csv   mean & std across seeds, one row per (market, model)
  eval_table.csv     paper-friendly "mean +/- std" strings
  eval_run_config.json   recorded backtest/eval config

Usage:
    python evaluate_all.py                                # all markets x models x seeds
    python evaluate_all.py --markets csi300 --models XGBoost --seeds 0   # single cell
    python evaluate_all.py --config configs/eval_all.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.workflow import R

from configs import MARKETS, MODELS, SEEDS, MLRUNS_DIR, STRATEGY_CONFIG, backtest_config
from eval_metrics import compute_prediction_metrics, compute_portfolio_metrics

METRIC_COLS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]

logger = logging.getLogger("evaluate_all")


# --------------------------------------------------------------------------- #
# artifact loading
# --------------------------------------------------------------------------- #
def _load_artifacts(market: str, model: str, seed: int):
    """Return (recorder, pred_df, label_df, report_df) or raise."""
    exp = f"{market}_{model}_seed{seed}"
    rec = R.get_recorder(experiment_name=exp)
    pred = rec.load_object("pred.pkl")
    label = rec.load_object("label.pkl")
    report = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
    return rec, pred, label, report


def evaluate_one(market: str, model: str, seed: int) -> dict:
    """Compute all metrics for a single (market, model, seed). Never raises - returns a row with NaNs on failure."""
    row = {"market": market, "model": model, "seed": seed}
    try:
        rec, pred, label, report = _load_artifacts(market, model, seed)
    except Exception as e:  # missing experiment / artifacts
        logger.warning("  [%s/%s/seed%s] SKIP: cannot load artifacts (%s)", market, model, seed, e)
        row.update({k: np.nan for k in METRIC_COLS})
        row.update({"num_test_days": 0, "pred_path_or_ckpt_path": ""})
        return row

    # prediction metrics (IC family) - reuses qlib calc_ic via compute_prediction_metrics
    try:
        row.update(compute_prediction_metrics(pred, label))
    except Exception as e:
        logger.warning("  [%s/%s/seed%s] prediction metrics failed: %s", market, model, seed, e)
        row.update({k: np.nan for k in ["IC", "ICIR", "RankIC", "RankICIR"]})

    # portfolio metrics (log-return paper convention). report["return"] is gross
    # (without fee), so after-cost daily return = return - cost (verified).
    try:
        daily_after_cost = report["return"] - report["cost"]
        port = compute_portfolio_metrics(daily_after_cost)
        row.update(port)
    except Exception as e:
        logger.warning("  [%s/%s/seed%s] portfolio metrics failed: %s", market, model, seed, e)
        row.update({k: np.nan for k in ["AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]})
        row["num_test_days"] = 0

    # pred artifact filesystem path (traceability)
    try:
        row["pred_path_or_ckpt_path"] = str(
            Path(MLRUNS_DIR / market) / rec.experiment_id / rec.id / "artifacts" / "pred.pkl")
    except Exception:
        row["pred_path_or_ckpt_path"] = ""

    logger.info("  [%s/%s/seed%s] IC=%.4f RankIC=%.4f AR=%.4f Sharpe=%.4f Calmar=%.4f",
                market, model, seed,
                row.get("IC", np.nan), row.get("RankIC", np.nan),
                row.get("AR", np.nan), row.get("Sharpe", np.nan), row.get("Calmar", np.nan))
    return row


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """mean / std across seeds per (market, model)."""
    g = detail.groupby(["market", "model"])[METRIC_COLS]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=1).add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    # stable column order: metric_mean, metric_std, ...
    cols = [c for m in METRIC_COLS for c in (f"{m}_mean", f"{m}_std")]
    return out[[c for c in cols if c in out.columns]]


def to_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std string per metric, one row per (market, model)."""
    rows = []
    for (market, model), r in summary.iterrows():
        cells = {"market": market, "model": model}
        for m in METRIC_COLS:
            mean = r.get(f"{m}_mean", np.nan)
            std = r.get(f"{m}_std", np.nan)
            cells[m] = "nan" if pd.isna(mean) else (
                f"{mean:.4f}" if pd.isna(std) else f"{mean:.4f} ± {std:.4f}")
        rows.append(cells)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# config / cli
# --------------------------------------------------------------------------- #
def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("pyyaml is required for --config: pip install pyyaml") from e
    with open(path) as f:
        return yaml.safe_load(f) or {}


def record_run_config(out_dir: Path, cli_args) -> dict:
    """Persist the actual backtest config (reused) + eval settings for traceability."""
    cfg = {
        "note": "Backtest is REUSED from run.py outputs; metrics recomputed with paper convention.",
        "test_period": {"train": ["2009-01-01", "2020-12-31"],
                        "valid": ["2021-01-01", "2022-12-31"],
                        "test": ["2023-01-01", "2025-12-31"]},
        "strategy": {"class": "TopkDropoutStrategy", "topk": STRATEGY_CONFIG["kwargs"]["topk"],
                     "n_drop": STRATEGY_CONFIG["kwargs"]["n_drop"], "weighting": "equal"},
        "cost_by_market": {m: backtest_config(m)["exchange_kwargs"] for m in MARKETS},
        "metric_convention": {"annualization_factor": 252, "std_ddof": 1, "rf": 0,
                              "sharpe": "sqrt(252)*mean(log(1+r_after_cost))/std",
                              "sortino": "sqrt(252)*mean(g)/std(negative g)",
                              "AR": "exp(mean(g)*252)-1", "STD": "std(g)*sqrt(252)",
                              "MDD": "min(nav/cummax-1), nav=exp(cumsum(g))",
                              "Calmar": "AR/abs(MDD)"},
        "evaluated": {"markets": cli_args.markets, "models": cli_args.models, "seeds": cli_args.seeds},
    }
    (out_dir / "eval_run_config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="optional yaml with defaults: markets/models/seeds/out")
    ap.add_argument("--markets", nargs="+", default=list(MARKETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default="results", help="output directory")
    # Informational only (backtest is reused, not re-run). If they differ from the
    # actual backtest config, we warn so the recorded metrics are not misread.
    ap.add_argument("--topk", type=int, default=STRATEGY_CONFIG["kwargs"]["topk"])
    ap.add_argument("--dropn", type=int, default=STRATEGY_CONFIG["kwargs"]["n_drop"])
    args = ap.parse_args()

    if args.config:
        cfg = _load_yaml_config(args.config)
        # config seeds defaults; explicit CLI (non-default) still wins because argparse already ran
        args.markets = cfg.get("markets", args.markets)
        args.models = cfg.get("models", args.models)
        args.seeds = cfg.get("seeds", args.seeds)
        args.out = cfg.get("out", args.out)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity: warn if requested topk/dropn differ from the actual backtest
    if args.topk != STRATEGY_CONFIG["kwargs"]["topk"] or args.dropn != STRATEGY_CONFIG["kwargs"]["n_drop"]:
        logger.warning("NOTE: --topk/--dropn (%s/%s) differ from the REUSED backtest (%s/%s); "
                       "metrics still come from the existing backtest. Re-run run.py to change them.",
                       args.topk, args.dropn, STRATEGY_CONFIG["kwargs"]["topk"], STRATEGY_CONFIG["kwargs"]["n_drop"])

    record_run_config(out_dir, args)

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")  # R.get_recorder needs qlib initialized

    rows = []
    for mk in args.markets:
        if mk not in MARKETS:
            logger.warning("unknown market %s, skipping", mk); continue
        R.set_uri(f"file:{(MLRUNS_DIR / mk).resolve()}")
        logger.info("\n=== market: %s ===", mk)
        for mdl in args.models:
            if mdl not in MODELS:
                logger.warning("unknown model %s, skipping", mdl); continue
            for seed in args.seeds:
                rows.append(evaluate_one(mk, mdl, seed))

    detail = pd.DataFrame(rows)
    detail_cols = ["market", "model", "seed"] + METRIC_COLS + ["num_test_days", "pred_path_or_ckpt_path"]
    detail = detail[[c for c in detail_cols if c in detail.columns]]
    detail.to_csv(out_dir / "eval_detail.csv", index=False)

    summary = summarize(detail)
    summary.to_csv(out_dir / "eval_summary.csv")

    table = to_paper_table(summary)
    table.to_csv(out_dir / "eval_table.csv", index=False)

    logger.info("\n=== %s/eval_detail.csv (%d rows) ===", out_dir, len(detail))
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(detail.drop(columns=["pred_path_or_ckpt_path"]).to_string(index=False))
    logger.info("\n=== %s/eval_table.csv (mean +/- std) ===", out_dir)
    print(table.to_string(index=False))
    logger.info("\noutputs: %s/{eval_detail,eval_summary,eval_table}.csv + eval_run_config.json", out_dir)


if __name__ == "__main__":
    main()
