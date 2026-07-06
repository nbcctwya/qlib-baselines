# qlib-baselines

Paper baselines for cross-sectional stock-return prediction, built on
[Microsoft Qlib](https://github.com/microsoft/qlib) (pyqlib 0.9.7).

Four models — **XGBoost, GRU, TCN, Transformer** — are trained on two stock pools
— **CSI300 (A-shares)** and **S&P 500 (US)** — with a single shared dataset, then
backtested with a **TopK-DropN (K=30, N=5)** portfolio under explicit trading
costs (buy 5bps / sell 15bps).

## Dataset split

| segment | range |
|---|---|
| train | 2009-01-01 → 2020-12-31 |
| valid | 2021-01-01 → 2022-12-31 |
| test  | 2023-01-01 → 2025-12-31 |

Processor fitting (normalisation, etc.) uses the **train segment only** — no
leakage into valid/test.

## Why a cached handler

Computing the Alpha158 feature/label matrix (~17 years × thousands of
instruments × 158 features) is the expensive step. `prepare_data.py` does it
**once per market** and dumps the processed handler
(`data_cache/{market}_handler.pkl`, `dump_all=True`). `run.py` loads that
pickle and wraps it directly in `DatasetH` (XGBoost) or `TSDatasetH`
(GRU/TCN/Transformer) — **no feature recomputation** across the four models.
Per-segment inspection pkls (`{market}_{train,valid,test}.pkl`) are also written.

## Layout

```
configs.py            single source of truth (segments, markets, HPs, backtest, SEEDS)
prepare_data.py       build + dump Alpha158 handler + 3 segment pkls per market
run.py                load handler -> 4 models x 2 markets x 5 seeds -> fit/predict/backtest
collect_results.py    qlib-convention aggregation  -> reports/*.csv + bar charts
eval_metrics.py       paper-convention metric functions (log return, sqrt(252))
evaluate_all.py       paper-convention evaluation  -> results/*.csv
data_cache/           {market}_handler.pkl (+ segment inspection pkls); gitignored
mlruns/               MLflow file store (one subdir per market); trained_model.pkl per run; gitignored
reports/              qlib convention: comparison_table*.csv + bar_*.png  (charts gitignored)
results/              paper convention: eval_detail/summary/table.csv + eval_run_config.json
```

## How to run

Environment: conda env `qlib-env` (Python 3.9, pyqlib 0.9.7, torch 2.8+cu129).
Qlib data expected at `~/.qlib/qlib_data/{cn_data,us_data}`.

```bash
# 1. Build the reusable handler (once per market)
python prepare_data.py

# 2. Smoke gate (fast end-to-end check, seed 0 only, tiny HPs)
python run.py --markets csi300 --models XGBoost --smoke

# 3. Full runs: 4 models x 2 markets x 5 seeds = 40 experiments (~50 min, warm cache)
python run.py

# 4. Aggregate metrics — two conventions, both read the SAME MLflow artifacts:
python collect_results.py    # -> reports/  (qlib risk_analysis, sum x238)
python evaluate_all.py       # -> results/  (paper log-return, sqrt(252))

# Single cell (re-evaluate one market/model/seed only):
python evaluate_all.py --markets csi300 --models XGBoost --seeds 0
```

Trained checkpoints: each (market, model, seed) run is an MLflow experiment
`{market}_{model}_seed{N}`; load a model with
`R.get_recorder(experiment_name=...).load_object("trained_model.pkl")` (see
`mlruns/<market>/<exp_id>/<run_id>/artifacts/`).

## Configuration highlights (see `configs.py`)

- **Alpha158 processors** (DNN benchmark config): `FilterCol(20 cols)` →
  `RobustZScoreNorm(clip)` → `Fillna` (infer); `DropnaLabel` → `CSRankNorm` (learn).
  `FilterCol` runs before `RobustZScoreNorm` so the `_ts` models see `d_feat=20`.
- **Label**: `Ref($close, -2) / Ref($close, -1) - 1`.
- **Backtest**: `TopkDropoutStrategy` `topk=30 n_drop=5`; `open_cost=0.0005`,
  `close_cost=0.0015`, `min_cost=0`, `deal_price=close`; `limit_threshold` per
  market (0.095 cn / None us); `trade_unit` from `qlib.init(region=...)`.
- **Model hyperparameters**: official Qlib benchmark defaults
  (`examples/benchmarks/{XGBoost,GRU,TCN,Transformer}`).

## Metric conventions (two CSV sets)

Both `collect_results.py` and `evaluate_all.py` read the **same** MLflow
artifacts; they only differ in how portfolio metrics are computed. IC/RankIC are
identical across the two (both use qlib `calc_ic`, daily cross-sectional, ddof=1).

| | `reports/` (collect_results.py) | `results/` (evaluate_all.py) |
|---|---|---|
| basis | qlib `risk_analysis`, sum mode, **×238** | **log return** g=log(1+r), **√252** |
| AR | `AnnReturn_Gross/Net` = mean(daily)×238 | `AR` = exp(mean(g)·252)−1 |
| vol | — | `STD` = std(g)·√252 |
| drawdown | `MaxDrawdown` (qlib) | `MDD` on nav=exp(cumsum(g)) |
| Sharpe | **excess-of-benchmark** information ratio (组合−基准) | **portfolio own** net-return: √252·mean(g)/std(g) |
| extra | AnnExcess_Gross/Net (vs benchmark), BenchReturn, Turnover | Sortino, Calmar=AR/abs(MDD) |
| use case | compare to qlib benchmarks / "did we beat the index?" | paste into the paper table |

Files: `reports/comparison_table{,_std,_raw}.csv` + `bar_*.png`; `results/eval_{detail,summary,table}.csv` + `eval_run_config.json` (records the exact conventions used).

## Results

Every (market, model) is run over **5 seeds (0–4)**. Test segment 2023-01-01 →
2025-12-31, TopK-DropN K=30 N=5, buy 5bps / sell 15bps. Values are **mean ± std**
over the 5 seeds. The table below is the **paper convention** (`results/eval_table.csv`).

| market | model       | IC            | RankIC        | AR           | Sharpe        | Sortino       | Calmar        |
|--------|-------------|---------------|---------------|--------------|---------------|---------------|---------------|
| csi300 | XGBoost     | 0.0136±0.0007 | 0.0263±0.0008 | 0.076±0.017  | **0.410±0.088** | **0.577±0.123** | **0.335±0.103** |
| csi300 | GRU         | 0.0152±0.0058 | 0.0358±0.0080 | 0.044±0.040  | 0.264±0.237   | 0.376±0.338   | 0.211±0.204   |
| csi300 | TCN         | 0.0026±0.0005 | 0.0010±0.0006 | 0.007±0.039  | 0.040±0.242   | 0.055±0.342   | 0.043±0.145   |
| csi300 | Transformer | 0.0142±0.0038 | 0.0307±0.0026 | 0.049±0.060  | 0.276±0.355   | 0.399±0.522   | 0.213±0.244   |
| sp500  | XGBoost     | 0.0028±0.0014 | 0.0040±0.0011 | 0.062±0.026  | 0.338±0.139   | 0.457±0.190   | 0.283±0.143   |
| sp500  | GRU         | 0.0033±0.0033 | 0.0058±0.0029 | 0.064±0.018  | 0.328±0.078   | 0.452±0.113   | 0.278±0.076   |
| sp500  | TCN         | -0.0013±0.0010| -0.0013±0.0007| 0.007±0.032  | 0.049±0.227   | 0.076±0.325   | 0.045±0.178   |
| sp500  | Transformer | 0.0031±0.0017 | 0.0063±0.0016 | 0.013±0.018  | 0.074±0.102   | 0.093±0.131   | 0.056±0.076   |

Formulas (`g = log(1 + daily_return_after_cost)`, daily_after_cost = `report["return"] − report["cost"]`,
verified to match account growth):
`AR = exp(mean(g)·252) − 1`; `STD = std(g)·√252`; `MDD = min(nav/cummax − 1)`,
`nav = exp(cumsum(g))`; `Sharpe = √252·mean(g)/std(g)`;
`Sortino = √252·mean(g)/std(negative g, ddof=1)`; `Calmar = AR/abs(MDD)`; R_f = 0.

**Interpretation.**
- **Why the seed sweep matters:** the std on the return/ratio columns is
  comparable to the means. A single seed is misleading — e.g. Transformer on
  CSI300 prints Sharpe 0.59 at seed 0 but averages **0.28 ± 0.35** over 5 seeds.
- **Signal vs return stability:** IC/RankIC are tight across seeds (std ≤ 0.008),
  so the models extract consistent cross-sectional information; the *returns* are
  noisy because a small signal is multiplied by a high-cost, high-turnover
  strategy (~0.33 daily turnover → ~6–8%/yr cost drag).
- **CSI300:** XGBoost is the most stable and strongest (Sharpe 0.41±0.09,
  Calmar 0.34±0.10); the DNN models have higher IC but much larger return
  variance (Sharpe std ±0.24–0.36).
- **SP500 vs benchmark:** the paper-convention Sharpe above is on the portfolio's
  *own* return (it looks positive because the portfolio rose with the bull
  market). To see whether the strategy *beat the index*, use the qlib-convention
  `reports/comparison_table.csv` (`AnnExcess_Net`): SP500 is uniformly negative
  there — the +19.3%/yr benchmark was too strong for Alpha158-style signals, even
  before cost.
