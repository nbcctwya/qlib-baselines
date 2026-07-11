"""Sanity-check the evaluation outputs in results/.

Two stages:
  1. Single-seed checks on metrics/seed_metrics.csv (range/sanity of every metric).
  2. Ensemble checks on metrics/ensemble_metrics.csv + tables/ensemble.csv, only
     if those files exist. Verifies row counts, ranges, table-vs-detail
     consistency. Ensemble ranking metrics are recomputed from ensemble scores
     by evaluate_all.py rather than inferred from single-seed aggregates.

Exit code 0 only if every check passes.

Usage:
    python inspect_eval_results.py
    python inspect_eval_results.py --results-dir results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from configs import MARKETS, MODELS, SEEDS
from eval_metrics import compute_portfolio_metrics

METRIC_COLS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]

# (column, predicate, description) for range checks on per-row metrics
RANGE_CHECKS = [
    ("MDD", lambda v: pd.isna(v) or v <= 1e-9, "MDD <= 0"),
    ("STD", lambda v: pd.isna(v) or v >= 0, "STD >= 0"),
    ("IC", lambda v: pd.isna(v) or abs(v) <= 1.0 + 1e-9, "|IC| <= 1"),
    ("RankIC", lambda v: pd.isna(v) or abs(v) <= 1.0 + 1e-9, "|RankIC| <= 1"),
    ("Sharpe", lambda v: pd.isna(v) or abs(v) <= 5.0 + 1e-9, "|Sharpe| <= 5"),
    ("Sortino", lambda v: pd.isna(v) or abs(v) <= 10.0 + 1e-9, "|Sortino| <= 10"),
    ("Calmar", lambda v: pd.isna(v) or abs(v) <= 10.0 + 1e-9, "|Calmar| <= 10"),
]


class Reporter:
    def __init__(self):
        self.fails = 0
        self.passes = 0
        self.checks = []

    def check(self, name: str, ok: bool, detail: str = ""):
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
        if ok:
            self.passes += 1
        else:
            self.fails += 1
        self.checks.append({"name": name, "passed": bool(ok), "detail": detail})

    def all_ok(self):
        return self.fails == 0


def _expected_combos(markets, models, seeds):
    return {(m, mdl, s) for m in markets for mdl in models for s in seeds}


def check_single_seed(r: Reporter, path: Path, markets, models, seeds):
    print("\n=== single-seed: metrics/seed_metrics.csv ===")
    if not path.exists():
        r.check("seed_metrics.csv exists", False, str(path))
        return
    df = pd.read_csv(path)
    r.check("seed_metrics.csv exists", True)

    expected = _expected_combos(markets, models, seeds)
    got = set(map(tuple, df[["market", "model", "seed"]].values.tolist()))
    r.check(f"row count == {len(expected)} (markets x models x seeds)",
            len(df) == len(expected) and got == expected,
            f"got {len(df)} rows, {len(got & expected)} of {len(expected)} expected combos")

    num = df[METRIC_COLS]
    r.check("no NaN/Inf in metrics", bool(np.isfinite(num.to_numpy()).all()),
            f"{int((~np.isfinite(num.to_numpy())).sum())} bad cells")

    for col, pred, desc in RANGE_CHECKS:
        bad = int((~num[col].map(pred)).sum())
        r.check(f"{desc} (per row)", bad == 0, f"{bad} rows violate" if bad else "")


def check_ensemble(r: Reporter, detail_path: Path, table_path: Path, markets, models):
    print("\n=== ensemble: metrics/ensemble_metrics.csv / tables/ensemble.csv ===")
    if not detail_path.exists():
        r.check("ensemble_metrics.csv exists (run `evaluate_all.py --ensemble`)", False, str(detail_path))
        return
    r.check("ensemble_metrics.csv exists", True)

    det = pd.read_csv(detail_path)
    expected_pairs = {(m, mdl) for m in markets for mdl in models}
    got_pairs = set(map(tuple, det[["market", "model"]].values.tolist()))
    r.check(f"ensemble detail row count == {len(expected_pairs)} (markets x models)",
            len(det) == len(expected_pairs) and got_pairs == expected_pairs,
            f"got {len(det)} rows")
    r.check("exactly one row per (market, model)",
            det.duplicated(["market", "model"]).sum() == 0)

    num = det[METRIC_COLS]
    r.check("no NaN/Inf in ensemble metrics", bool(np.isfinite(num.to_numpy()).all()),
            f"{int((~np.isfinite(num.to_numpy())).sum())} bad cells")
    for col, pred, desc in RANGE_CHECKS:
        bad = int((~num[col].map(pred)).sum())
        r.check(f"ensemble {desc} (per row)", bad == 0, f"{bad} rows violate" if bad else "")

    # table-vs-detail consistency
    if table_path.exists():
        tbl = pd.read_csv(table_path)
        r.check("tables/ensemble.csv exists", True)
        mismatches = 0
        for _, trow in tbl.iterrows():
            drow = det[(det["market"] == trow["market"]) & (det["model"] == trow["model"])]
            if drow.empty:
                mismatches += len(METRIC_COLS); continue
            drow = drow.iloc[0]
            for m in METRIC_COLS:
                try:
                    if abs(float(trow[m]) - float(drow[m])) > 1e-4:
                        mismatches += 1
                except (ValueError, TypeError):
                    mismatches += 1
        r.check("ensemble table == ensemble detail (within 1e-4)", mismatches == 0,
                f"{mismatches} cell mismatches")
    else:
        r.check("tables/ensemble.csv exists", False)


def check_manifest(r: Reporter, results_dir: Path):
    path = results_dir / "metadata" / "manifest.json"
    r.check("metadata/manifest.json exists", path.exists(), str(path) if not path.exists() else "")
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text())
        r.check("manifest schema_version == 1.0", manifest.get("schema_version") == "1.0")
        missing = []
        for rel in manifest.get("files", {}).values():
            matches = list(results_dir.glob(rel)) if "*" in rel else [results_dir / rel]
            if not matches or not all(p.exists() for p in matches):
                missing.append(rel)
        r.check("all manifest file references resolve", not missing,
                f"missing: {missing}" if missing else "")
    except (OSError, ValueError, TypeError) as exc:
        r.check("manifest is valid JSON", False, str(exc))


def check_aggregate(r: Reporter, results_dir: Path):
    seed_path = results_dir / "metrics" / "seed_metrics.csv"
    agg_path = results_dir / "metrics" / "aggregate_metrics.csv"
    if not seed_path.exists() or not agg_path.exists():
        r.check("aggregate inputs exist", False)
        return
    seed = pd.read_csv(seed_path)
    actual = pd.read_csv(agg_path).set_index(["market", "model"])
    grouped = seed.groupby(["market", "model"])[METRIC_COLS]
    expected = pd.concat([grouped.mean().add_suffix("_mean"),
                          grouped.std(ddof=1).add_suffix("_std")], axis=1)
    expected = expected[[c for m in METRIC_COLS for c in (f"{m}_mean", f"{m}_std")]]
    ok = actual.index.equals(expected.index) and np.allclose(
        actual[expected.columns].to_numpy(), expected.to_numpy(), equal_nan=True)
    r.check("aggregate_metrics equals seed mean/std", bool(ok))
    table_path = results_dir / "tables" / "seed_mean_std.csv"
    table_mismatches = 0
    if table_path.exists():
        table = pd.read_csv(table_path).set_index(["market", "model"])
        for key, row in table.iterrows():
            if key not in expected.index:
                table_mismatches += len(METRIC_COLS)
                continue
            for metric in METRIC_COLS:
                cell = str(row[metric]).replace("+/-", "±")
                try:
                    mean_text, std_text = (part.strip() for part in cell.split("±"))
                    if (abs(float(mean_text) - expected.loc[key, f"{metric}_mean"]) > 5e-5 or
                            abs(float(std_text) - expected.loc[key, f"{metric}_std"]) > 5e-5):
                        table_mismatches += 1
                except (ValueError, TypeError):
                    table_mismatches += 1
    else:
        table_mismatches = 1
    r.check("seed table equals aggregate metrics (4 decimals)", table_mismatches == 0,
            f"{table_mismatches} mismatch(es)" if table_mismatches else "")
    absolute = seed["pred_path_or_ckpt_path"].fillna("").map(lambda p: Path(p).is_absolute()).sum()
    r.check("seed artifact references are portable", absolute == 0,
            f"{absolute} absolute path(s)" if absolute else "")


def check_curves(r: Reporter, results_dir: Path, markets, models):
    ens_path = results_dir / "metrics" / "ensemble_metrics.csv"
    if not ens_path.exists():
        return
    ens = pd.read_csv(ens_path).set_index(["market", "model"])
    failures = []
    for market in markets:
        for model in models:
            path = results_dir / "curves" / "ensemble" / f"{market}_{model}.csv"
            if not path.exists() or (market, model) not in ens.index:
                failures.append(f"missing {market}/{model}")
                continue
            curve = pd.read_csv(path)
            dates = pd.to_datetime(curve["datetime"])
            if dates.duplicated().any() or not dates.is_monotonic_increasing:
                failures.append(f"bad dates {market}/{model}")
            if not np.allclose(curve["daily_ret_net"], curve["daily_ret_gross"] - curve["cost"]):
                failures.append(f"net relation {market}/{model}")
            nav = (1.0 + curve["daily_ret_net"]).cumprod()
            bench_nav = (1.0 + curve["bench_ret"]).cumprod()
            if not np.allclose(curve["nav"], nav) or not np.allclose(curve["bench_nav"], bench_nav):
                failures.append(f"nav {market}/{model}")
            got = compute_portfolio_metrics(curve["daily_ret_net"])
            row = ens.loc[(market, model)]
            if any(not np.isclose(got[col], row[col], rtol=1e-10, atol=1e-12)
                   for col in ["AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]):
                failures.append(f"metrics {market}/{model}")
    absolute = ens["pred_paths"].fillna("").str.split(";").explode().map(lambda p: Path(p).is_absolute()).sum()
    if absolute:
        failures.append(f"{absolute} absolute ensemble artifact path(s)")
    r.check("ensemble curves and metrics are reproducible", not failures,
            "; ".join(failures[:5]))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--markets", nargs="+", default=list(MARKETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    d = Path(args.results_dir)
    r = Reporter()
    check_manifest(r, d)
    seed_path = d / "metrics" / "seed_metrics.csv"
    ensemble_path = d / "metrics" / "ensemble_metrics.csv"
    check_single_seed(r, seed_path, args.markets, args.models, args.seeds)
    single_ok_for_ensemble = seed_path.exists()
    check_ensemble(r, ensemble_path, d / "tables" / "ensemble.csv",
                   args.markets, args.models)
    check_aggregate(r, d)
    check_curves(r, d, args.markets, args.models)

    print(f"\n=== summary: {r.passes} passed, {r.fails} failed ===")
    if r.fails == 0:
        print("All single-seed evaluation checks passed.")
        if single_ok_for_ensemble and ensemble_path.exists():
            print("All ensemble evaluation checks passed.")
    diagnostics = d / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    report = {"passed": r.all_ok(), "passes": r.passes, "failures": r.fails,
              "checks": r.checks}
    (diagnostics / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if r.all_ok() else 1)


if __name__ == "__main__":
    main()
