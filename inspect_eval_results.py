"""Sanity-check the evaluation outputs in results/.

Two stages:
  1. Single-seed checks on metrics/seed_metrics.csv (range/sanity of every metric).
  2. Ensemble checks on metrics/ensemble_metrics.csv + tables/ensemble.csv, only
     if those files exist. Verifies row counts, ranges, table-vs-detail
     consistency, and that ensemble IC/RankIC equal the mean of the 5 single-seed
     values in metrics/seed_metrics.csv.

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

    metric_cols = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    num = df[metric_cols]
    r.check("no NaN/Inf in metrics", bool(np.isfinite(num.to_numpy()).all()),
            f"{int((~np.isfinite(num.to_numpy())).sum())} bad cells")

    for col, pred, desc in RANGE_CHECKS:
        bad = int((~num[col].map(pred)).sum())
        r.check(f"{desc} (per row)", bad == 0, f"{bad} rows violate" if bad else "")


def check_ensemble(r: Reporter, detail_path: Path, table_path: Path,
                   single_path: Path, markets, models):
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

    metric_cols = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    num = det[metric_cols]
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
                mismatches += len(metric_cols); continue
            drow = drow.iloc[0]
            for m in metric_cols:
                try:
                    if abs(float(trow[m]) - float(drow[m])) > 1e-4:
                        mismatches += 1
                except (ValueError, TypeError):
                    mismatches += 1
        r.check("ensemble table == ensemble detail (within 1e-4)", mismatches == 0,
                f"{mismatches} cell mismatches")
    else:
        r.check("tables/ensemble.csv exists", False)

    # ensemble IC / RankIC == mean of 5 single-seed values
    if single_path.exists():
        sdf = pd.read_csv(single_path)
        means = sdf.groupby(["market", "model"])[["IC", "RankIC"]].mean()
        bad = 0
        for _, drow in det.iterrows():
            key = (drow["market"], drow["model"])
            if key not in means.index:
                bad += 1; continue
            for col in ["IC", "RankIC"]:
                if abs(float(drow[col]) - float(means.loc[key, col])) > 1e-6:
                    bad += 1
        r.check("ensemble IC/RankIC == mean of 5 single-seed values (1e-6)", bad == 0,
                f"{bad} mismatches")
    else:
        r.check("seed_metrics.csv available for IC-mean cross-check", False,
                "metrics/seed_metrics.csv missing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--markets", nargs="+", default=list(MARKETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    d = Path(args.results_dir)
    r = Reporter()
    seed_path = d / "metrics" / "seed_metrics.csv"
    ensemble_path = d / "metrics" / "ensemble_metrics.csv"
    check_single_seed(r, seed_path, args.markets, args.models, args.seeds)
    single_ok_for_ensemble = seed_path.exists()
    check_ensemble(r, ensemble_path, d / "tables" / "ensemble.csv",
                   seed_path, args.markets, args.models)

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
