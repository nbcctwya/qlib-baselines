# Baseline Results Protocol — Reusable AI Prompt

下面代码块是一份可直接复制给其他项目 AI 的实施 Prompt。使用时只需替换最前面的项目变量；如果暂时不知道变量值，让 AI 从仓库中检查，不要猜测。

```text
你正在一个量化预测/选股 baseline 仓库中工作。请检查现有训练、预测、回测和评估代码，并将它适配到下面定义的 Baseline Results Protocol v1.0。目标是让本项目与其他 baseline（例如 PRISM-VQ、MASTER、MATCC、Qlib baselines）产生结构、字段、指标口径和校验方式一致的 results/，从而可以直接横向比较。

【项目变量】
- BASELINE_ID：<例如 prism_vq / master / matcc；若未填写，从仓库名和配置确定>
- MARKETS_OR_DATASETS：<例如 csi300, sp500；若未填写，从配置和已有实验确定>
- MODELS：<若一个项目只有一个模型，仍然保留 model 字段>
- SEEDS：<优先读取实际配置；不要默认一定是 0..4>
- TEST_PERIOD：<从数据配置确定>
- PREDICTION_ARTIFACT：<预测分数的位置或生成方法>
- LABEL_ARTIFACT：<测试标签的位置或生成方法>
- DAILY_BACKTEST_ARTIFACT：<至少能得到 gross return、cost、benchmark return>
- ENSEMBLE_SUPPORTED：<true/false；未填写时根据是否存在至少两个 seed 判断>

先只读检查仓库，再实施修改。不要重写模型、训练逻辑或改变已有回测策略；只增加或适配统一评估与结果导出层。复用项目已有预测和回测产物，除非 ensemble 必须重新回测。保留项目原生输出，避免破坏既有工作流。

一、必须生成的目录

results/
├── metrics/
│   ├── seed_metrics.csv
│   ├── aggregate_metrics.csv
│   └── ensemble_metrics.csv       # 仅当 ensemble 可用
├── tables/
│   ├── seed_mean_std.csv
│   └── ensemble.csv               # 仅当 ensemble 可用
├── curves/
│   └── ensemble/*.csv             # 仅当 ensemble 可用
├── metadata/
│   ├── eval_config.json
│   └── manifest.json
└── diagnostics/
    └── validation.json

目录语义必须固定：
- metrics/ 只放机器可计算的原始数值，不放 "mean ± std" 字符串。
- tables/ 只放论文/README 友好展示结果。
- curves/ 放逐日序列。
- metadata/ 放口径、配置和文件发现信息。
- diagnostics/ 放机器可读校验报告。

二、seed_metrics.csv

每个 (market, model, seed) 恰好一行，列顺序固定为：

market,model,seed,IC,ICIR,RankIC,RankICIR,AR,STD,MDD,Sharpe,Sortino,Calmar,num_test_days,pred_path_or_ckpt_path

要求：
- 指标列必须是数值类型，不得写百分号或格式化字符串。
- market 可以表示市场、股票池或 dataset split 的稳定 ID，但同一批 baseline 必须使用相同命名。
- model 是模型/变体 ID；单模型项目也必须填写。
- seed 使用实际随机种子。
- pred_path_or_ckpt_path 优先使用相对项目根目录的路径、run_id 或 artifact URI；禁止写只能在当前机器生效的绝对路径。
- 缺少某个实验时不要伪造数据；记录失败并让 validation 失败。不要静默跳过后仍宣称完整。

三、统一指标口径

预测指标按每个交易日做横截面计算：
- IC_t = Pearson(prediction, label)
- RankIC_t = Spearman(prediction, label)
- IC = mean(IC_t)，跳过无法计算相关系数的日期
- ICIR = mean(IC_t) / std(IC_t, ddof=1)，不乘 sqrt(252)
- RankIC = mean(RankIC_t)
- RankICIR = mean(RankIC_t) / std(RankIC_t, ddof=1)，不年化

组合指标的输入必须是扣除一次交易成本后的日简单收益：

r_net_t = daily_return_gross_t - cost_t

若项目的原始 return 已经扣费，不得再次扣费；必须检查原始框架语义，并在 eval_config.json 记录判断依据。若任一 r_net_t <= -1，明确报错，不要让 log1p 静默产生 NaN。

定义：

g_t = log(1 + r_net_t)
A = 252
MAR_daily = 0

指标公式固定为：
- AR = exp(mean(g_t) * A) - 1
- STD = std(g_t, ddof=1) * sqrt(A)
- NAV = [1.0, exp(cumsum(g_t))]
- MDD = min(NAV / cumulative_max(NAV) - 1)
- Sharpe = sqrt(A) * mean(g_t) / std(g_t, ddof=1)，无风险利率为 0
- DownsideDeviation = sqrt(mean(min(g_t - MAR_daily, 0)^2))，mean 必须覆盖全部交易日，非负收益日贡献 0
- Sortino = sqrt(A) * mean(g_t - MAR_daily) / DownsideDeviation
- Calmar = AR / abs(MDD)
- num_test_days = 有效 g_t 数量

零分母、样本不足等数学上未定义的情况写 NaN，并让校验报告明确指出；不要用 0 冒充。所有 baseline 必须使用相同的 252、log-return、ddof 和 MAR 口径。不要把相对 benchmark 的 information ratio 写成这里的 Sharpe。

四、aggregate_metrics.csv 与 seed_mean_std.csv

对 seed_metrics.csv 按 (market, model) 聚合。aggregate_metrics.csv 每组一行，列为：

market,model,IC_mean,IC_std,ICIR_mean,ICIR_std,RankIC_mean,RankIC_std,RankICIR_mean,RankICIR_std,AR_mean,AR_std,STD_mean,STD_std,MDD_mean,MDD_std,Sharpe_mean,Sharpe_std,Sortino_mean,Sortino_std,Calmar_mean,Calmar_std

- mean/std 均跨 seed 计算。
- std 使用 ddof=1。
- seed_mean_std.csv 由 aggregate_metrics.csv 派生，每个指标格式化为四位小数的 "mean ± std"。
- tables/ 中的字符串不得反过来作为任何计算的数据源。

五、ensemble 规则

若每个 (market, model) 至少有两个 seed，则实现 ensemble；否则在 manifest 和 eval_config 中明确 enabled=false，不生成空的伪 ensemble 文件。

默认 ensemble：
- 按 (datetime, instrument) 对齐不同 seed 的 prediction score。
- 默认 inner join，确保每个样本都包含全部选定 seed。
- 默认直接平均 raw score，ensemble_method = avg_none。
- 可选 avg_zscore：先对每个 seed 每日横截面 z-score（ddof=0），再平均。
- 可选 avg_rank：先对每个 seed 每日横截面 rank percentile，再平均。

必须用 ensemble score 重新执行与单 seed 完全相同的回测策略、费用、股票池和测试期。ensemble_metrics.csv 每个 (market, model, ensemble_method) 一行，列顺序：

market,model,ensemble_method,IC,ICIR,RankIC,RankICIR,AR,STD,MDD,Sharpe,Sortino,Calmar,num_test_days,seeds,pred_paths

关键要求：
- ensemble 的 IC、ICIR、RankIC、RankICIR 必须直接从 ensemble score 与对齐后的 test label 重新计算。
- 禁止用单 seed IC/ICIR 的平均值冒充 ensemble IC/ICIR。
- AR 等组合指标来自 ensemble 的重新回测。
- seeds 用逗号连接实际 seed。
- pred_paths 使用可移植的相对路径、run_id 或 artifact URI。
- tables/ensemble.csv 是上述数值文件的四位小数展示版。

每个 ensemble 曲线文件命名为 curves/ensemble/<market>_<model>.csv，列为：

datetime,daily_ret_gross,cost,daily_ret_net,bench_ret,nav,bench_nav

要求：
- daily_ret_net = daily_ret_gross - cost，除非已确认 gross 字段实际已经扣费；语义必须记录。
- nav = cumprod(1 + daily_ret_net)，代表从初始资本 1.0 经过当日收益后的净值；不要除以第一行从而丢掉第一天收益。
- bench_nav = cumprod(1 + bench_ret)。
- 日期升序、唯一，无 NaN/Inf。

六、eval_config.json

必须记录实际运行口径，而不是复制示例值。至少包含：
- baseline ID
- train/valid/test 时间段
- 市场或数据集列表
- 实际 seeds 和 models
- 回测策略及关键参数（TopK/DropN 或项目对应策略、权重方法）
- 每个市场的交易成本和 benchmark
- return 字段是否为 gross、成本如何扣除
- 完整 metric_convention（252、ddof=1、rf=0、MAR=0 和上述公式）
- ensemble 是否启用、join、normalize、score formula、ranking metrics 的来源
- 数据版本或截止日期（如果仓库能确定）
- 当前 Git commit（如果可获得）

七、manifest.json

生成以下结构，并用实际 BASELINE_ID 和实际存在的文件填写；不要登记不存在的文件：

{
  "schema_version": "1.0",
  "baseline": "<BASELINE_ID>",
  "description": "<简短描述>",
  "primary_keys": {
    "seed_metrics": ["market", "model", "seed"],
    "aggregate_metrics": ["market", "model"],
    "ensemble_metrics": ["market", "model", "ensemble_method"]
  },
  "files": {
    "seed_metrics": "metrics/seed_metrics.csv",
    "aggregate_metrics": "metrics/aggregate_metrics.csv",
    "seed_table": "tables/seed_mean_std.csv",
    "eval_config": "metadata/eval_config.json",
    "validation": "diagnostics/validation.json",
    "ensemble_metrics": "metrics/ensemble_metrics.csv",
    "ensemble_table": "tables/ensemble.csv",
    "ensemble_curves": "curves/ensemble/*.csv"
  }
}

如果 ensemble 未启用，删除 primary_keys.ensemble_metrics 和三个 ensemble 文件条目。

八、校验器与 validation.json

增加一个可重复运行的检查入口（例如 inspect_eval_results.py 或项目等价脚本），退出码在全部通过时为 0，否则非 0，并将结果写入 diagnostics/validation.json。

至少检查：
- 文件存在且 manifest 中的非 glob 路径有效。
- seed_metrics 行数和 (market, model, seed) 组合与实际配置完全一致。
- 主键无重复。
- 指标无 NaN/Inf（若数学上确实未定义，应明确列为允许的例外，而不是静默通过）。
- |IC|<=1、|RankIC|<=1、STD>=0、MDD<=0。
- aggregate_metrics 精确等于 seed_metrics 的 mean/std。
- tables 的数值在四位小数容差内等于对应 metrics。
- ensemble 开启时，每个预期 (market, model, method) 恰好一行。
- 曲线日期升序且唯一。
- daily_ret_net 与 gross/cost 关系正确。
- nav 和 bench_nav 可从日收益反算。
- AR、STD、MDD、Sharpe、Sortino、Calmar 可从曲线反算并与 ensemble_metrics 一致。
- ensemble ranking metrics 确实来自 ensemble score；尽可能在校验中从保存的 score/label 反算。若无法保存或加载，至少在生成阶段加入单元测试，不能退回 seed 均值。

validation.json 至少包含：

{
  "passed": true,
  "passes": 0,
  "failures": 0,
  "checks": [
    {"name": "...", "passed": true, "detail": "..."}
  ]
}

九、实现与兼容原则

- 先识别项目使用的是 Qlib、独立 PyTorch、MLflow、pickle、CSV 还是其他框架，再写最小适配层。
- 指标计算应集中到一个纯函数模块，避免在多个脚本复制公式。
- 输出路径从项目根目录或 --out 参数解析，不硬编码当前机器路径。
- 保留已有结果和训练入口；如必须迁移旧 results，确保内容可追溯且不误删用户产物。
- 不要为了凑齐 schema 伪造 benchmark、cost、seed 或 ensemble。
- 如果现有策略与 TopK-DropN 不同，保留该 baseline 的真实策略，并在 eval_config 中完整记录；统一的是输出和指标口径，不是强迫所有模型采用同一投资策略。
- 如果不同 baseline 要做公平比较，发现测试期、股票池、label、费用或策略不一致时必须在最终报告中醒目标注，不能仅因为 CSV 字段相同就宣称可直接比较。
- 更新项目 README，解释 results/ 各层含义、生成命令和指标定义。

十、完成标准

完成后必须：
1. 运行语法/静态检查。
2. 运行指标边界测试：首日 -10% 时 MDD 必须为 -10%；两个相同负收益日的 Sortino 分母不得因为“负收益标准差为 0”而失效；r<=-1 必须报错。
3. 用至少一组真实日收益手工/独立反算全部组合指标。
4. 运行完整 validation，必须 0 failures。
5. 展示最终目录树、各文件行数和关键改动。
6. 汇报任何无法统一的 baseline 特有差异。

请直接检查并实施，不要只给方案。如果关键信息无法从仓库发现且不同选择会实质改变结果，才向我提一个简短问题；否则使用仓库事实推进。不要训练新模型，除非我另行授权。
```

## 使用方法

在目标 baseline 仓库中开启 AI 编码会话，把上面的整个代码块发送给 AI。建议同时补充一句：

```text
BASELINE_ID=prism_vq（或 master/matcc），请从当前仓库自动发现其余项目变量，并在修改前先报告发现的已有训练、预测和回测产物。
```

如果目标项目不使用 Qlib，也不需要改 Prompt：其中统一的是结果协议和指标纯函数，AI 应为该项目现有产物编写适配层。
