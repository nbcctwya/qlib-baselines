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

Optional seed-ENSEMBLE evaluation (--ensemble): for each (market, model), average
the 5 seeds' pred scores (inner join by default) into one ensemble score, re-run
the SAME TopK-DropN backtest on it, and compute AR/STD/MDD/Sharpe/Sortino/Calmar.
The ensemble's IC/ICIR/RankIC/RankICIR are the MEAN of the 5 single-seed values
(same convention as the main table), NOT recomputed on the ensemble score.

Outputs (under <out>/, default results/):
  eval_detail.csv           one row per (market, model, seed)
  eval_summary.csv          mean & std across seeds, one row per (market, model)
  eval_table.csv            paper-friendly "mean +/- std" strings
  eval_ensemble_detail.csv  (--ensemble) one row per (market, model)
  eval_ensemble_table.csv   (--ensemble) paper-friendly, 4-decimal, no mean/std
  eval_run_config.json      recorded backtest/eval/ensemble config

Usage:
    python evaluate_all.py                                # all markets x models x seeds
    python evaluate_all.py --markets csi300 --models XGBoost --seeds 0   # single cell
    python evaluate_all.py --ensemble                     # + seed-ensemble eval (raw avg)
    python evaluate_all.py --ensemble --ensemble-normalize rank   # rank-percentile avg
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.workflow import R

from configs import MARKETS, MODELS, SEEDS, MLRUNS_DIR, STRATEGY_CONFIG, backtest_config
from eval_metrics import compute_prediction_metrics, compute_portfolio_metrics

METRIC_COLS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
IC_COLS = ["IC", "ICIR", "RankIC", "RankICIR"]
PORT_COLS = ["AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]

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

    try:
        row.update(compute_prediction_metrics(pred, label))
    except Exception as e:
        logger.warning("  [%s/%s/seed%s] prediction metrics failed: %s", market, model, seed, e)
        row.update({k: np.nan for k in IC_COLS})

    # report["return"] is gross (without fee); after-cost daily return = return - cost (verified).
    try:
        daily_after_cost = report["return"] - report["cost"]
        port = compute_portfolio_metrics(daily_after_cost)
        row.update(port)
    except Exception as e:
        logger.warning("  [%s/%s/seed%s] portfolio metrics failed: %s", market, model, seed, e)
        row.update({k: np.nan for k in PORT_COLS})
        row["num_test_days"] = 0

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
# seed ensemble
# --------------------------------------------------------------------------- #
def build_ensemble_score(seed_preds, normalize: str = "none", join: str = "inner"):
    """Average per-seed prediction scores into one ensemble score.

    seed_preds: list of pd.Series (or 1-col DataFrame) indexed by
                (datetime, instrument); uses iloc[:,0] for DataFrames (col may be
                'score' or anything).
    normalize:
      none   - mean of raw scores
      zscore - per-day cross-sectional z-score per seed (ddof=0), then mean.
               constant-score days yield std=0 -> those entries set to 0 (neutral).
      rank   - per-day cross-sectional rank percentile (0..1) per seed, then mean.
    join:
      inner  - keep only (datetime, instrument) present in ALL seeds (recommended;
               guarantees the ensemble score is the average of exactly 5 seeds).
      outer  - union; missing seed scores are NaN, mean skips them (so a row may
               average fewer than 5 seeds).

    Returns (ensemble_score: pd.Series named 'score', info: dict).
    """
    series = []
    for i, p in enumerate(seed_preds):
        if isinstance(p, pd.DataFrame):
            p = p.iloc[:, 0]
        series.append(p.rename(f"seed{i}").astype(float))

    df = pd.concat(series, axis=1)  # outer join on index by default
    n_per_seed = {f"seed{i}": int(s.notna().sum()) for i, s in enumerate(series)}
    if join == "inner":
        df = df.dropna()

    if df.index.names[0] != "datetime":
        # be defensive: pred index should be (datetime, instrument)
        df = df.swaplevel(0, 1) if "datetime" in df.index.names else df

    if normalize == "zscore":
        df = df.groupby(level="datetime").transform(
            lambda x: (x - x.mean()) / x.std(ddof=0))
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif normalize == "rank":
        df = df.groupby(level="datetime").transform(lambda x: x.rank(pct=True))
    # normalize == "none": leave raw

    if join == "inner":
        df = df.dropna()  # safety: normalize should not introduce NaN for inner, but guard anyway
    ensemble = df.mean(axis=1).rename("score")
    info = {"n_per_seed": n_per_seed, "n_joined": int(len(ensemble)),
            "n_nan_ensemble": int(ensemble.isna().sum())}
    return ensemble, info


def evaluate_ensemble_one(market: str, model: str, seed_preds, seed_paths,
                          ic_means: dict, normalize: str, join: str) -> dict:
    """Backtest the ensemble score and compute portfolio metrics.

    ic_means: {IC,ICIR,RankIC,RankICIR} = mean of the 5 single-seed values.
    Backtest uses the SAME TopK-DropN + exchange/cost config as single seed
    (configs.backtest_config(market)), so conventions are identical.

    Caller must have already run qlib.init(provider=<market>) + R.set_uri(<market>).
    """
    ensemble, info = build_ensemble_score(seed_preds, normalize, join)
    logger.info("  [%s/%s] ensemble samples: per-seed(non-null)=%s | joined=%d | NaN=%d",
                market, model, info["n_per_seed"], info["n_joined"], info["n_nan_ensemble"])
    if info["n_nan_ensemble"] > 0:
        logger.warning("  [%s/%s] dropping %d NaN ensemble scores", market, model, info["n_nan_ensemble"])
        ensemble = ensemble.dropna()
    if len(ensemble) == 0:
        raise ValueError("ensemble score is empty after alignment")

    bt = backtest_config(market)
    exkw = dict(bt["exchange_kwargs"])
    exkw.setdefault("freq", "day")
    strategy = TopkDropoutStrategy(
        signal=ensemble,
        topk=STRATEGY_CONFIG["kwargs"]["topk"],
        n_drop=STRATEGY_CONFIG["kwargs"]["n_drop"],
    )
    report, _ = backtest_daily(
        start_time=bt["start_time"], end_time=bt["end_time"],
        strategy=strategy, account=bt["account"],
        benchmark=bt["benchmark"], exchange_kwargs=exkw,
    )
    # report["return"] is gross; after-cost = return - cost (same as single seed, no double count)
    port = compute_portfolio_metrics(report["return"] - report["cost"])

    row = {"market": market, "model": model, "ensemble_method": f"avg_{normalize}"}
    row.update({k: ic_means[k] for k in IC_COLS})       # ranking metrics: mean of 5 seeds
    row.update({k: port[k] for k in PORT_COLS})          # backtest metrics: ensemble backtest
    row["num_test_days"] = port["num_test_days"]
    row["seeds"] = ",".join(str(s) for s in SEEDS)
    row["pred_paths"] = ";".join(seed_paths)
    return row


def run_ensemble(detail: pd.DataFrame, args, out_dir: Path):
    """Run seed-ensemble evaluation for every (market, model). Returns (detail_df, table_df)."""
    ens_rows = []
    for mk in args.markets:
        if mk not in MARKETS:
            continue
        # qlib.init per market (the backtest needs the market's data provider) AND
        # re-set the MLflow uri afterwards: qlib.init re-creates the exp_manager and
        # would otherwise leave R pointing at the wrong tracking root, breaking the
        # subsequent R.get_recorder calls in this market's model loop.
        qlib.init(provider_uri=MARKETS[mk]["provider_uri"], region=MARKETS[mk]["region"])
        R.set_uri(f"file:{(MLRUNS_DIR / mk).resolve()}")
        logger.info("\n=== ensemble market: %s (normalize=%s, join=%s) ===",
                    mk, args.ensemble_normalize, args.ensemble_join)
        ic_mean_by_model = (detail[detail["market"] == mk]
                            .groupby("model")[IC_COLS].mean())
        for mdl in args.models:
            if mdl not in MODELS:
                continue
            seed_preds, seed_paths = [], []
            for s in args.seeds:
                try:
                    rec = R.get_recorder(experiment_name=f"{mk}_{mdl}_seed{s}")
                    seed_preds.append(rec.load_object("pred.pkl"))
                    seed_paths.append(str(
                        Path(MLRUNS_DIR / mk) / rec.experiment_id / rec.id / "artifacts" / "pred.pkl"))
                except Exception as e:
                    logger.warning("  [%s/%s] missing seed%s pred: %s", mk, mdl, s, e)
            if len(seed_preds) < 2:
                logger.warning("  [%s/%s] SKIP ensemble (need >=2 seed preds, got %d)", mk, mdl, len(seed_preds))
                continue
            try:
                icm = ic_mean_by_model.loc[mdl].to_dict() if mdl in ic_mean_by_model.index else \
                    {k: np.nan for k in IC_COLS}
                row = evaluate_ensemble_one(mk, mdl, seed_preds, seed_paths, icm,
                                            args.ensemble_normalize, args.ensemble_join)
                ens_rows.append(row)
                logger.info("  [%s/%s] ensemble IC=%.4f AR=%.4f Sharpe=%.4f Calmar=%.4f",
                            mk, mdl, row["IC"], row["AR"], row["Sharpe"], row["Calmar"])
            except Exception as e:
                logger.warning("  [%s/%s] ensemble FAILED: %s", mk, mdl, e)

    ens_detail = pd.DataFrame(ens_rows)
    detail_cols = (["market", "model", "ensemble_method"] + IC_COLS + PORT_COLS
                   + ["num_test_days", "seeds", "pred_paths"])
    if len(ens_detail):
        ens_detail = ens_detail[[c for c in detail_cols if c in ens_detail.columns]]
    ens_detail.to_csv(out_dir / "eval_ensemble_detail.csv", index=False)

    table_cols = ["market", "model", "ensemble_method"] + IC_COLS + PORT_COLS
    ens_table = ens_detail[[c for c in table_cols if c in ens_detail.columns]].copy()
    for c in [c for c in table_cols if c not in ("market", "model", "ensemble_method")]:
        ens_table[c] = ens_table[c].map(lambda v: "nan" if pd.isna(v) else f"{v:.4f}")
    ens_table.to_csv(out_dir / "eval_ensemble_table.csv", index=False)
    return ens_detail, ens_table


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
        "ensemble": {
            "enabled": bool(getattr(cli_args, "ensemble", False)),
            "normalize": getattr(cli_args, "ensemble_normalize", "none"),
            "join": getattr(cli_args, "ensemble_join", "inner"),
            "ranking_metrics": "mean of the 5 single-seed values (from eval_detail.csv)",
            "backtest": "TopkDropoutStrategy on the ensemble score; same TopK/DropN/cost as single seed",
            "score_formula": "mean(score_seed0..4); optional per-day cross-sectional zscore (ddof=0) or rank-pct",
        },
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
    ap.add_argument("--topk", type=int, default=STRATEGY_CONFIG["kwargs"]["topk"])
    ap.add_argument("--dropn", type=int, default=STRATEGY_CONFIG["kwargs"]["n_drop"])
    ap.add_argument("--ensemble", action="store_true",
                    help="also run seed-ensemble evaluation -> eval_ensemble_*.csv")
    ap.add_argument("--ensemble-normalize", choices=["none", "zscore", "rank"], default="none",
                    help="per-day cross-sectional normalization before averaging seed scores")
    ap.add_argument("--ensemble-join", choices=["inner", "outer"], default="inner",
                    help="inner = keep only samples present in all seeds (recommended)")
    args = ap.parse_args()

    if args.config:
        cfg = _load_yaml_config(args.config)
        args.markets = cfg.get("markets", args.markets)
        args.models = cfg.get("models", args.models)
        args.seeds = cfg.get("seeds", args.seeds)
        args.out = cfg.get("out", args.out)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.topk != STRATEGY_CONFIG["kwargs"]["topk"] or args.dropn != STRATEGY_CONFIG["kwargs"]["n_drop"]:
        logger.warning("NOTE: --topk/--dropn (%s/%s) differ from the REUSED backtest (%s/%s); "
                       "single-seed metrics still come from the existing backtest. "
                       "Ensemble backtest always uses the configured TopK/DropN.",
                       args.topk, args.dropn, STRATEGY_CONFIG["kwargs"]["topk"], STRATEGY_CONFIG["kwargs"]["n_drop"])

    record_run_config(out_dir, args)

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")  # R.get_recorder needs qlib initialized

    # ---- single-seed pass (unchanged) ----
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

    # ---- ensemble pass (additive, separate output files) ----
    if args.ensemble:
        ens_detail, ens_table = run_ensemble(detail, args, out_dir)
        logger.info("\n=== %s/eval_ensemble_detail.csv (%d rows) ===", out_dir, len(ens_detail))
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(ens_detail.drop(columns=["pred_paths"]).to_string(index=False))
        logger.info("\n=== %s/eval_ensemble_table.csv ===", out_dir)
        print(ens_table.to_string(index=False))

    logger.info("\noutputs: %s/{eval_detail,eval_summary,eval_table}.csv%s + eval_run_config.json",
                out_dir, ", eval_ensemble_{detail,table}.csv" if args.ensemble else "")


if __name__ == "__main__":
    main()
