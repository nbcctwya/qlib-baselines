# qlib-baselines

基于 [Microsoft Qlib](https://github.com/microsoft/qlib)（pyqlib 0.9.7）的横截面股票收益预测 **论文 baseline** 项目。

4 个模型（**XGBoost / GRU / TCN / Transformer**）× 2 个市场（**CSI300 沪深300 / SP500 标普500**）× 5 个 seed，统一用 Alpha158 特征训练，并用 **TopK-DropN（K=30, N=5）** 策略回测（买入 5bps / 卖出 15bps）。支持单 seed 评估与 **seed ensemble** 评估两套指标口径。

---

## 环境与数据

- conda 环境：`qlib-env`（Python 3.9、pyqlib 0.9.7、torch 2.8+cu129、xgboost 2.1.4）。
- Qlib 数据：`~/.qlib/qlib_data/{cn_data,us_data}`（已覆盖 2009–2025）。
- 激活环境后即可运行下述脚本：`conda activate qlib-env`。

## 数据划分

| 段 | 区间 |
|---|---|
| train | 2009-01-01 → 2020-12-31 |
| valid | 2021-01-01 → 2022-12-31 |
| test  | 2023-01-01 → 2025-12-31 |

处理器（归一化等）只在 train 段拟合，不向 valid/test 泄漏。CSI300 与 SP500 使用各自的数据路径与交易日历（test 段分别 727 / 752 天）。

---

## 项目文件说明

| 文件 | 作用 | 用法 |
|---|---|---|
| `configs.py` | **唯一配置源**：数据划分、市场（provider/region/benchmark/limit）、Alpha158 处理器、回测（TopK-DropN、费率）、各模型超参、`SEEDS=[0,1,2,3,4]`。其余脚本都从这里读配置。 | 只读，不直接运行。 |
| `prepare_data.py` | 每个市场构建 **一次** Alpha158 handler 并 dump 成 pkl（`dump_all=True`，含已拟合处理器状态），同时写出 train/valid/test 三段检查 pkl。后续所有模型复用，**不重算特征**。 | `python prepare_data.py`（约 1 分钟/市场，一次性） |
| `run.py` | 加载缓存 handler → 遍历 4 模型 × 2 市场 × 5 seed → 训练 / 预测 / 回测（走 qlib workflow，结果存 MLflow）。seed 在运行时注入到每个模型。 | `python run.py`（全量 40 个实验）；`python run.py --markets csi300 --models XGBoost --smoke`（冒烟） |
| `collect_results.py` | **qlib 口径**聚合：读 MLflow 产物，用 qlib `risk_analysis`（sum 模式 ×238）算指标，输出 `reports/*.csv` + 柱状图。 | `python collect_results.py` |
| `eval_metrics.py` | **论文口径**纯指标函数库：IC 类（复用 qlib `calc_ic`，ddof=1）+ 组合指标（log 收益、√252，ddof=1）。只提供函数，不直接运行。 | 被 `evaluate_all.py` 调用。 |
| `evaluate_all.py` | **论文口径**统一评估入口：遍历 market×model×seed，按 `metrics/tables/curves/metadata` 分层输出；`--ensemble` 额外做 seed ensemble 评估。 | `python evaluate_all.py`；`python evaluate_all.py --ensemble` |
| `inspect_eval_results.py` | 结果体检：检查单 seed 与 ensemble 输出的行数、NaN/Inf、取值范围、table==detail、ensemble IC==5-seed 均值等。 | `python inspect_eval_results.py` |

> checkpoint 位置：每个 (market,model,seed) 是一个 MLflow 实验 `{market}_{model}_seed{N}`，模型存在 `mlruns/<market>/<exp_id>/<run_id>/artifacts/trained_model.pkl`，用 `R.get_recorder(experiment_name=...).load_object("trained_model.pkl")` 加载。

---

## 输出文件说明

两套指标口径各占一个目录，**读的是同一份 MLflow 产物**，只是组合指标算法不同；IC/RankIC 两套完全一致（都来自 qlib `calc_ic`）。

### `reports/` —— qlib 口径（`collect_results.py` 产出）

用 qlib 自带 `risk_analysis`（sum 模式，年化 ×238，**非 log**）；Sharpe 是**超额收益的信息比**（组合−基准，扣费）。

| 文件 | 行数 | 内容 |
|---|---|---|
| `comparison_table_raw.csv` | 40 | 每个 (market,model,seed) 一行，含 seed 列，全部指标 |
| `comparison_table.csv` | 8 | 5 seed 的**均值**，每个 (market,model) 一行 |
| `comparison_table_std.csv` | 8 | 5 seed 的**标准差** |
| `bar_*.png` | — | IC/RankIC/年化超额/年化收益 柱状图（带 seed 标准差误差棒，已 gitignore） |

列：`IC, ICIR, RankIC, RankICIR, AnnReturn_Gross, AnnReturn_Net, AnnExcess_Gross, AnnExcess_Net, Sharpe, MaxDrawdown, BenchReturn, Turnover`。
- `AnnReturn_Gross/Net`：组合自身年化（不扣费/扣费），不减基准。
- `AnnExcess_Gross/Net`：**毛超额/净超额**（组合−基准，不扣费/扣费）——判断能否跑赢指数看这两个。
- `Sharpe`：净超额收益的信息比；`BenchReturn`：基准年化。

### `results/` —— 论文口径（`evaluate_all.py` 产出）

按论文公式（log 收益 g=log(1+r)，√252 年化）；Sharpe 是**组合自身扣费后净收益**的夏普（不减基准）。

| 文件 | 行数 | 内容 |
|---|---|---|
| `metrics/seed_metrics.csv` | 40 | 核心单 seed 数值，每行一个 (market,model,seed) |
| `metrics/aggregate_metrics.csv` | 8 | 每个 (market,model) 的 `_mean`/`_std` 数值列 |
| `metrics/ensemble_metrics.csv` | 8 | **seed ensemble** 的完整数值结果 |
| `tables/seed_mean_std.csv` | 8 | 论文展示格式，单元格 `mean ± std` |
| `tables/ensemble.csv` | 8 | ensemble 展示格式（4 位小数） |
| `curves/ensemble/*.csv` | 8 份 | ensemble 的逐日收益、成本、策略净值和基准净值 |
| `metadata/eval_config.json` | — | 回测参数、指标口径和 ensemble 配置 |
| `metadata/manifest.json` | — | schema 版本、baseline 身份、主键和文件清单；跨 baseline 读取入口 |
| `diagnostics/validation.json` | — | `inspect_eval_results.py` 写出的机器可读检查报告 |

列：`IC, ICIR, RankIC, RankICIR, AR, STD, MDD, Sharpe, Sortino, Calmar`。
- `AR=exp(mean(g)·252)−1`、`STD=std(g)·√252`、`MDD`（累计净值最大回撤）、`Sharpe=√252·mean(g)/std(g)`、`Sortino=√252·mean(g)/std(负收益)`、`Calmar=AR/abs(MDD)`，`g=log(1+日收益_after_cost)`，`日收益_after_cost = report["return"] − report["cost"]`。

---

## 快速开始

```bash
# 0. 激活环境
conda activate qlib-env

# 1. 生成可复用的缓存 handler（一次性）
python prepare_data.py

# 2. 冒烟验证（单 seed、小超参，几分钟内）
python run.py --markets csi300 --models XGBoost --smoke

# 3. 全量训练 + 回测：4 模型 × 2 市场 × 5 seed = 40 个实验
python run.py

# 4. 评估（两种口径都读同一份 MLflow 产物）
python collect_results.py        # -> reports/  (qlib 口径)
python evaluate_all.py           # -> results/  (论文口径, 单 seed)
python evaluate_all.py --ensemble              # + seed ensemble (raw 平均)
python evaluate_all.py --ensemble --ensemble-normalize rank   # rank 百分位平均

# 5. 体检
python inspect_eval_results.py   # 全部通过时输出 "All ... checks passed."
```

单 cell 评估：`python evaluate_all.py --markets csi300 --models XGBoost --seeds 0`。

### ensemble 说明

`--ensemble` 对每个 (market,model) 把 5 个 seed 的 `pred.pkl` **inner-join 对齐后求平均**成 ensemble score，再用与单 seed **完全相同**的 TopK-DropN/费率回测，算 AR/STD/MDD/Sharpe/Sortino/Calmar。
- 排序指标 IC/ICIR/RankIC/RankICIR = 5 个单 seed 值的均值（**不**在 ensemble score 上重算）。
- `--ensemble-normalize`：`none`（默认，原始 score 平均）/ `zscore`（每日横截面 z-score 后平均）/ `rank`（每日横截面 rank 百分位后平均）。
- `--ensemble-join`：`inner`（默认，只保留 5 seed 都有的样本）/ `outer`。

---

## 指标口径

两套并存的原因：`collect_results.py` 先写（qlib 标准报表口径），后来按论文公式评估时**没有覆盖旧逻辑**，而是新增 `evaluate_all.py`。选用建议：

- **对标 qlib benchmark / 看是否跑赢指数** → `reports/`（看 `AnnExcess_Net`）。
- **继续计算或跨 baseline 合并** → `results/metrics/`。
- **贴进论文表格** → `results/tables/`。
- **画 ensemble 净值图** → `results/curves/ensemble/`。
- **追溯口径与自动发现文件** → `results/metadata/manifest.json`。

> 注意：`results/` 的 Sharpe 是**组合自身净收益**口径（牛市里会偏正）；`reports/` 的 Sharpe 是**超额收益信息比**（能否跑赢指数）。两者回答的问题不同，不要混用。

---

## 配置要点（`configs.py`）

- **Alpha158 处理器**（DNN benchmark 配置）：infer = `FilterCol(20列)` → `RobustZScoreNorm(clip)` → `Fillna`；learn = `DropnaLabel` → `CSRankNorm(label)`。`FilterCol` 必须在 `RobustZScoreNorm` 之前，保证 `_ts` 模型看到 `d_feat=20`。
- **Label**：`Ref($close, -2) / Ref($close, -1) - 1`。
- **回测**：`TopkDropoutStrategy` `topk=30 n_drop=5`；`open_cost=0.0005`、`close_cost=0.0015`、`min_cost=0`、`deal_price=close`；`limit_threshold` 按市场（cn 0.095 / us None）；`trade_unit` 由 `qlib.init(region=...)` 决定。
- **模型超参**：Qlib 官方 benchmark 默认值（`examples/benchmarks/{XGBoost,GRU,TCN,Transformer}`）。

---

## 结果（测试段 2023-01-01 → 2025-12-31，5 seed）

### 单 seed（论文口径，mean ± std，`results/tables/seed_mean_std.csv`）

| market | model | IC | ICIR | RankIC | RankICIR | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| csi300 | XGBoost | 0.0136±0.0007 | 0.0930±0.0045 | 0.0263±0.0008 | 0.1797±0.0042 | 0.076±0.017 | 0.179±0.005 | -0.232±0.023 | **0.410±0.087** | **0.577±0.123** | **0.335±0.103** |
| csi300 | GRU | 0.0152±0.0058 | 0.0855±0.0334 | 0.0358±0.0080 | 0.1958±0.0412 | 0.044±0.040 | 0.162±0.009 | -0.246±0.050 | 0.264±0.237 | 0.376±0.338 | 0.211±0.204 |
| csi300 | TCN | 0.0026±0.0005 | 0.0252±0.0049 | 0.0010±0.0006 | 0.0080±0.0050 | 0.007±0.039 | 0.161±0.003 | -0.244±0.033 | 0.040±0.242 | 0.049±0.347 | 0.043±0.145 |
| csi300 | Transformer | 0.0142±0.0038 | 0.0873±0.0271 | 0.0307±0.0026 | 0.1864±0.0222 | 0.049±0.060 | 0.160±0.011 | -0.260±0.035 | 0.276±0.354 | 0.399±0.522 | 0.212±0.244 |
| sp500 | XGBoost | 0.0028±0.0014 | 0.0308±0.0150 | 0.0040±0.0011 | 0.0443±0.0121 | 0.062±0.026 | 0.177±0.002 | -0.226±0.020 | 0.338±0.139 | 0.457±0.190 | 0.283±0.143 |
| sp500 | GRU | 0.0033±0.0033 | 0.0267±0.0256 | 0.0058±0.0029 | 0.0443±0.0222 | 0.064±0.018 | 0.187±0.034 | -0.229±0.032 | 0.327±0.078 | 0.452±0.113 | 0.278±0.076 |
| sp500 | TCN | -0.0013±0.0010 | -0.0142±0.0107 | -0.0013±0.0007 | -0.0118±0.0063 | 0.007±0.032 | 0.139±0.007 | -0.181±0.014 | 0.049±0.227 | 0.076±0.325 | 0.045±0.178 |
| sp500 | Transformer | 0.0031±0.0017 | 0.0253±0.0134 | 0.0063±0.0016 | 0.0475±0.0110 | 0.013±0.018 | 0.165±0.008 | -0.223±0.016 | 0.074±0.102 | 0.093±0.131 | 0.056±0.076 |

### seed ensemble（raw 平均，`results/tables/ensemble.csv`）

| market | model | IC | ICIR | RankIC | RankICIR | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| csi300 | XGBoost | 0.0136 | 0.0930 | 0.0263 | 0.1797 | 0.092 | 0.181 | -0.210 | 0.488 | 0.686 | 0.441 |
| csi300 | GRU | 0.0152 | 0.0855 | 0.0358 | 0.1958 | 0.050 | 0.155 | -0.204 | 0.315 | 0.448 | 0.245 |
| csi300 | TCN | 0.0026 | 0.0252 | 0.0010 | 0.0080 | 0.033 | 0.159 | -0.222 | 0.205 | 0.286 | 0.149 |
| csi300 | Transformer | 0.0142 | 0.0873 | 0.0307 | 0.1864 | 0.038 | 0.155 | -0.268 | 0.242 | 0.343 | 0.143 |
| sp500 | XGBoost | 0.0028 | 0.0308 | 0.0040 | 0.0443 | 0.049 | 0.182 | -0.227 | 0.261 | 0.354 | 0.214 |
| sp500 | GRU | 0.0033 | 0.0267 | 0.0058 | 0.0443 | 0.084 | 0.196 | -0.239 | 0.412 | 0.574 | 0.353 |
| sp500 | TCN | -0.0013 | -0.0142 | -0.0013 | -0.0118 | 0.002 | 0.134 | -0.178 | 0.018 | 0.026 | 0.014 |
| sp500 | Transformer | 0.0031 | 0.0253 | 0.0063 | 0.0475 | 0.010 | 0.171 | -0.226 | 0.059 | 0.074 | 0.045 |

**解读要点**
- **多 seed 的意义**：收益/比率列的 seed 标准差与均值同量级，单 seed 不可信（如 csi300 Transformer 单 seed Sharpe 可达 0.59，5 seed 平均 0.28±0.35）。
- **信号稳、收益不稳**：IC/RankIC 的 seed 标准差 ≤ 0.008，模型确实稳定提取了横截面信息；但高换手（~0.33/天）× 高费率带来 ~6–8%/年成本拖累，放大了收益噪声。
- **ensemble 降噪**：ensemble Sharpe 普遍高于单 seed 均值（如 csi300 XGBoost 0.488 vs 单 seed 均值 0.410）。
- **SP500 是否跑赢指数**：上表 Sharpe 是组合自身收益口径（牛市偏正）。要看是否跑赢 +19.3%/yr 的标普，看 `reports/comparison_table.csv` 的 `AnnExcess_Net`——SP500 全线为负。

---

## 关于缓存 handler

计算 Alpha158 特征矩阵（~17 年 × 数千股票 × 158 特征）是最贵的一步。`prepare_data.py` 每个市场做一次并存为 `data_cache/{market}_handler.pkl`（`dump_all=True`）；`run.py` 加载后直接包成 `DatasetH`/`TSDatasetH`，**不触发 `setup_data`、不重算特征**，4 个模型 × 5 seed 全部复用同一份 handler。`data_cache/`、`mlruns/` 体积大，已 gitignore，不在仓库里（clone 后重跑 `prepare_data.py` + `run.py` 即可重建）。
