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

`reports/comparison_table.csv` — metrics per (market, model), all returns on the
same annualisation basis (qlib `risk_analysis`, sum mode, x238). Test segment
2023-01-01 → 2025-12-31, TopK-DropN K=30 N=5, buy 5bps / sell 15bps.

| market | model       | IC     | ICIR  | RankIC | RankICIR | AnnRet_Gross | AnnRet_Net | AnnExcess_Gross | AnnExcess_Net | Sharpe | MaxDD | Bench  | Turnover |
|--------|-------------|--------|-------|--------|----------|--------------|------------|-----------------|---------------|--------|-------|--------|----------|
| csi300 | XGBoost     | 0.0138 | 0.093 | 0.0268 | 0.181    | 0.147        | 0.069      | 0.075           | -0.003        | -0.04  | -0.15 | 0.0722 | 0.33     |
| csi300 | GRU         | 0.0154 | 0.081 | 0.0380 | 0.194    | 0.129        | 0.051      | 0.057           | -0.022        | -0.26  | -0.13 | 0.0722 | 0.33     |
| csi300 | TCN         | 0.0031 | 0.030 | 0.0016 | 0.013    | 0.094        | 0.044      | 0.022           | -0.029        | -0.33  | -0.27 | 0.0722 | 0.21     |
| csi300 | Transformer | 0.0151 | 0.096 | 0.0302 | 0.190    | 0.187        | 0.108      | 0.115           | +0.036        | +0.49  | -0.13 | 0.0722 | 0.33     |
| sp500  | XGBoost     | 0.0021 | 0.023 | 0.0048 | 0.055    | 0.180        | 0.102      | -0.013          | -0.092        | -1.10  | -0.30 | 0.1933 | 0.33     |
| sp500  | GRU         | 0.0072 | 0.055 | 0.0093 | 0.068    | 0.176        | 0.096      | -0.018          | -0.097        | -0.79  | -0.36 | 0.1933 | 0.33     |
| sp500  | TCN         | -0.0020|-0.023 |-0.0018 |-0.017   | 0.055        | -0.023     | -0.138          | -0.216        | -2.17  | -0.69 | 0.1933 | 0.33     |
| sp500  | Transformer | 0.0035 | 0.030 | 0.0061 | 0.048    | 0.083        | 0.004      | -0.110          | -0.189        | -2.09  | -0.64 | 0.1933 | 0.33     |

**Column key.**
- `IC / ICIR / RankIC / RankICIR` — signal quality (daily mean; ICIR = mean/std).
- `AnnRet_Gross` — 年化收益·**不扣费**·不减基准（组合自身毛收益）。
- `AnnRet_Net`   — 年化收益·**扣费**·不减基准（组合自身净收益）。
- `AnnExcess_Gross` — **毛超额** = 组合 − 基准，不扣费。
- `AnnExcess_Net`   — **净超额** = 组合 − 基准，扣费（要跑赢指数看这个）。
- `Sharpe / MaxDD` — net-excess 的信息比与最大回撤；`Bench` — 基准年化；`Turnover` — 日均换手。

**Interpretation.** `AnnRet_Gross − AnnRet_Net` is the realised annual cost drag
(~6–8% at ~0.33 daily turnover; less for TCN which turns over ~0.21). On CSI300
the benchmark made +7.2%/yr and **Transformer beat it even after cost** (+10.8%
net return, +3.6% net excess); XGBoost is roughly breakeven; GRU/TCN slightly
negative after cost despite positive gross. SP500 was a strong bull market
(benchmark +19.3%/yr) and the Alpha158-style signals could not keep up — most
strategies are negative even *before* cost (`AnnExcess_Gross`), so cost only
widens the gap.
