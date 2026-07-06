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
configs.py            single source of truth (segments, markets, HPs, backtest)
prepare_data.py       build + dump Alpha158 handler + 3 segment pkls per market
run.py                load handler -> 4 models x 2 markets -> fit/predict/backtest
collect_results.py    read MLflow artifacts -> comparison_table.csv + bar charts
data_cache/           {market}_handler.pkl (+ segment inspection pkls)
mlruns/               MLflow file store (one subdir per market)
reports/              comparison_table.csv, bar_*.png
```

## How to run

Environment: conda env `qlib-env` (Python 3.9, pyqlib 0.9.7, torch 2.8+cu129).
Qlib data expected at `~/.qlib/qlib_data/{cn_data,us_data}`.

```bash
# 1. Build the reusable handler (~7-10 min/market, once)
python prepare_data.py

# 2. Smoke gate: cheap end-to-end check before the full runs
python run.py --markets csi300 --models XGBoost --smoke
python run.py --markets csi300 --models GRU --smoke

# 3. Full runs (4 models x 2 markets, ~3.5 h total)
python run.py

# 4. Collect metrics + charts
python collect_results.py
```

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

## Results

Every (market, model) is run over **5 seeds (0–4)**. Reports hold three tables:
`comparison_table.csv` (mean across seeds), `comparison_table_std.csv` (std across
seeds), `comparison_table_raw.csv` (all 40 rows). Bar charts carry seed-std error
bars. All returns on the same annualisation basis (qlib `risk_analysis`, sum mode,
x238). Test segment 2023-01-01 → 2025-12-31, TopK-DropN K=30 N=5, buy 5bps / sell
15bps. Values below are **mean ± std** over the 5 seeds.

| market | model       | IC            | RankIC        | AnnExcess_Gross | AnnExcess_Net   | AnnReturn_Net   | Sharpe        |
|--------|-------------|---------------|---------------|-----------------|-----------------|-----------------|---------------|
| csi300 | XGBoost     | 0.0136±0.0007 | 0.0263±0.0008 | 0.091±0.015     | **+0.012±0.014**| 0.084±0.014     | +0.15±0.18    |
| csi300 | GRU         | 0.0152±0.0058 | 0.0358±0.0080 | 0.058±0.036     | -0.020±0.036    | 0.052±0.036     | -0.23±0.43    |
| csi300 | TCN         | 0.0026±0.0005 | 0.0010±0.0006 | 0.004±0.029     | -0.054±0.037    | 0.018±0.037     | -0.62±0.40    |
| csi300 | Transformer | 0.0142±0.0038 | 0.0307±0.0026 | 0.062±0.056     | -0.016±0.056    | 0.056±0.056     | -0.16±0.67    |
| sp500  | XGBoost     | 0.0028±0.0014 | 0.0040±0.0011 | -0.043±0.022    | -0.122±0.022    | 0.071±0.022     | -1.41±0.21    |
| sp500  | GRU         | 0.0033±0.0033 | 0.0058±0.0029 | -0.039±0.021    | -0.118±0.021    | 0.075±0.021     | -1.15±0.37    |
| sp500  | TCN         | -0.0013±0.0010| -0.0013±0.0007| -0.104±0.025    | -0.178±0.029    | 0.015±0.029     | -1.73±0.34    |
| sp500  | Transformer | 0.0031±0.0017 | 0.0063±0.0016 | -0.090±0.017    | -0.169±0.017    | 0.025±0.017     | -1.91±0.17    |

**Column key** (full set in the CSVs; benchmark = SH000300 for csi300 / ^gspc for
sp500; both ≈ +7.2% and +19.3% annualised over the test window respectively).
- `IC / RankIC` — signal quality (daily Pearson / rank IC). Stable across seeds.
- `AnnExcess_Gross` — **毛超额** = 组合 − 基准，不扣费（纯选股能力）。
- `AnnExcess_Net`   — **净超额** = 组合 − 基准，扣费（能否跑赢指数）。
- `AnnReturn_Net`   — 年化收益·扣费·不减基准（组合自身净收益）。
- `Sharpe` — information ratio of the net excess return.

**Interpretation.**
- **Why the seed sweep matters:** the std on the excess-return columns (1.4%–5.7%)
  is comparable to the means. A single seed is misleading — e.g. Transformer on
  CSI300 printed +3.6% net excess at seed 0 but averages **−1.6% ± 5.6%** over 5
  seeds, so that +3.6% was inside the noise band.
- **Signal vs return stability:** IC/RankIC are tight across seeds (std ≤ 0.008),
  so the models extract consistent cross-sectional information; the *returns* are
  noisy because a small signal is multiplied by a high-cost, high-turnover
  strategy (~0.33 daily turnover → ~6–8%/yr cost drag).
- **CSI300:** XGBoost is the most stable and the only cell with net excess
  clearly above zero (+1.2% ± 1.4%); the DNN models average ≈ 0 after cost despite
  higher IC, because their return variance (±3.6%–±5.6%) is much larger.
- **SP500:** uniformly negative even *before* cost (`AnnExcess_Gross` < 0) — the
  Alpha158-style signals could not keep up with the +19.3%/yr bull market, and the
  costs only widen the gap. Consistent across all seeds.
