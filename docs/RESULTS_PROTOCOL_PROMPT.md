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
- QLIB_PROVIDER_URI：<该市场的 Qlib 数据目录；从项目或本机配置发现>
- QLIB_REGION：<cn/us 等>
- BENCHMARK：<例如 SH000300 / ^gspc；从项目市场配置确定>

先只读检查仓库，再实施修改。不要重写模型或训练逻辑。保留项目原生回测及输出，避免破坏既有工作流；但本协议写入统一 results/ 的组合指标必须使用下述统一 Qlib 回测规则重新回测，不能直接拿各项目口径不同的原生回测指标填充。

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

三、统一 Qlib 回测规则

为了让不同 baseline 可以公平比较，写入本协议 results/ 的单 seed 和 ensemble 组合指标必须来自同一套 Qlib 回测。即使项目原本使用自定义回测器，也要把其测试集 prediction score 转换成 Qlib 可接收的、以 (datetime, instrument) 为索引的 signal，并额外执行以下标准回测；不得覆盖或删除项目原生回测结果。

固定策略：
- 使用 Qlib `qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy`。
- `topk=30`：目标持仓数量为 30。
- `n_drop=5`：每个交易日最多从当前持仓中卖出 5 只，并根据当日预测排名补入新的高分股票。
- 日频调仓，回测频率为 `day`。
- `method_sell="bottom"`：优先淘汰低分持仓。
- `method_buy="top"`：优先买入最高分候选。
- `hold_thresh=1`：至少持有一个交易 step 后才允许卖出。
- `only_tradable=False`：构造排名候选时不预先过滤不可交易股票，订单执行仍由 Qlib Exchange 检查。
- `forbid_all_trade_at_limit=True`：触及涨跌停限制时禁止该股票的全部交易。
- `risk_degree=0.95`：每次分配买入资金时使用 95% 的可用现金。
- 上述参数必须显式传入，不得依赖所安装 Qlib 版本的默认值。
- 持仓权重使用该固定参数下 Qlib TopkDropoutStrategy 的原生资金分配和整手取整逻辑，不在适配层二次改权重。

固定交易成本：
- 买入费率 `open_cost=0.0005`，即万分之五（5 bps）。
- 卖出费率 `close_cost=0.0015`，即万分之十五（15 bps）。
- `min_cost=0`，避免不同市场的最低手续费规则破坏 baseline 间一致性。
- 日净收益必须只扣费一次：`daily_ret_net = report["return"] - report["cost"]`。必须先确认 Qlib 当前版本中 `report["return"]` 是未扣费收益。

固定回测区间：
- `start_time` 和 `end_time` 必须与该项目实际 test split 的覆盖范围一致。
- 不得使用 train/valid 数据回测，也不得为了和其他项目凑日期而擅自扩展 test 区间。
- 如果 prediction 实际覆盖范围短于声明的 test split，必须报错或在 validation 中失败，不能静默缩短回测期。
- 使用对应市场的 Qlib 交易日历；`num_test_days` 记录实际回测交易日数量。

固定账户：
- 初始资金 `account=100000000`。
- 所有 market、model、seed 和 ensemble 必须使用相同初始资金。
- 不得根据模型或市场单独缩放初始资金。

必须从项目/市场配置确定并记录、不得猜测的参数：
- `provider_uri`、`region`、股票池 instruments 和 benchmark。股票池使用 Qlib instruments 提供的动态历史成分股；适配层不要自行重建、冻结或过滤成分股列表。
- `deal_price`。若项目没有另行指定，统一使用 `close`；所有参与横向比较的 baseline 必须相同。
- 涨跌停限制、停牌与不可交易处理。中国市场按实际 Qlib 市场规则配置，美国市场不得套用 A 股涨跌停参数。
- `trade_unit`、executor 类型及 Qlib 版本。`trade_unit` 由 Qlib 对应 region 的市场配置处理，适配层不覆盖。
- 是否允许卖空或杠杆；标准 TopKDropN 比较默认只做多、无杠杆。

信号与标签要求：
- signal 必须只来自 test 时点可用的信息，禁止未来数据泄漏。
- 信号到交易日采用 Qlib TopkDropoutStrategy 的默认时序：交易日 `t` 调仓时读取前一个交易 step（`t-1`）的 signal。Qlib 0.9.7 对应实现为 `get_step_time(trade_step, shift=1)`，其中正 shift 表示 earlier bars。
- signal 的 `(datetime, instrument)` 日期必须表示该信号产生/归属的交易日；适配层不得为了回测再手工整体 shift 一次，否则会造成重复滞后。
- 保持项目原本 prediction 与 label 的预测 horizon，不得为了提高结果移动预测日期。
- 在 eval_config.json 记录：`signal_date=t-1`、`trade_date=t`、Qlib 内部 `shift=1`，以及项目 label horizon。
- 若不同 baseline 的 label horizon 不一致，必须在最终报告醒目标注；不能仅因回测规则相同就宣称完全公平。

统一回测的最小配置语义如下；具体模块路径可按已安装 Qlib 版本适配，但数值规则不得改变：

strategy:
  class: TopkDropoutStrategy
  kwargs:
    signal: <test prediction score>
    topk: 30
    n_drop: 5
    method_sell: bottom
    method_buy: top
    hold_thresh: 1
    only_tradable: false
    forbid_all_trade_at_limit: true
    risk_degree: 0.95
backtest:
  start_time: <TEST_START>
  end_time: <TEST_END>
  account: 100000000
  benchmark: <MARKET_BENCHMARK>
  exchange_kwargs:
    freq: day
    open_cost: 0.0005
    close_cost: 0.0015
    min_cost: 0
    deal_price: close

四、统一指标口径

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

五、aggregate_metrics.csv 与 seed_mean_std.csv

对 seed_metrics.csv 按 (market, model) 聚合。aggregate_metrics.csv 每组一行，列为：

market,model,IC_mean,IC_std,ICIR_mean,ICIR_std,RankIC_mean,RankIC_std,RankICIR_mean,RankICIR_std,AR_mean,AR_std,STD_mean,STD_std,MDD_mean,MDD_std,Sharpe_mean,Sharpe_std,Sortino_mean,Sortino_std,Calmar_mean,Calmar_std

- mean/std 均跨 seed 计算。
- std 使用 ddof=1。
- seed_mean_std.csv 由 aggregate_metrics.csv 派生，每个指标格式化为四位小数的 "mean ± std"。
- tables/ 中的字符串不得反过来作为任何计算的数据源。

六、ensemble 规则

若每个 (market, model) 至少有两个 seed，则实现 ensemble；否则在 manifest 和 eval_config 中明确 enabled=false，不生成空的伪 ensemble 文件。

默认 ensemble：
- 按 (datetime, instrument) 对齐不同 seed 的 prediction score。
- 默认 inner join，确保每个样本都包含全部选定 seed。
- 默认直接平均 raw score，ensemble_method = avg_none。
- 可选 avg_zscore：先对每个 seed 每日横截面 z-score（ddof=0），再平均。
- 可选 avg_rank：先对每个 seed 每日横截面 rank percentile，再平均。

必须用 ensemble score 重新执行上述统一 Qlib TopKDropN 回测，策略、费用、股票池、benchmark、交易规则和测试期必须与单 seed 完全相同。禁止平均单 seed 的收益曲线或组合指标来冒充 ensemble 回测。ensemble_metrics.csv 每个 (market, model, ensemble_method) 一行，列顺序：

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

七、eval_config.json

必须记录实际运行口径，而不是复制示例值。至少包含：
- baseline ID
- train/valid/test 时间段
- 市场或数据集列表
- 实际 seeds 和 models
- 统一回测策略：Qlib TopkDropoutStrategy、topk=30、n_drop=5、method_sell=bottom、method_buy=top、hold_thresh=1、only_tradable=false、forbid_all_trade_at_limit=true、risk_degree=0.95 和日频
- 每个市场的 provider、region、股票池、benchmark、deal_price、涨跌停和不可交易规则
- 买入费率 0.0005、卖出费率 0.0015、min_cost=0
- 初始资金 account=100000000、trade_unit、executor、只做多/杠杆设置和 Qlib 版本
- signal_date=t-1、trade_date=t、Qlib shift=1 和 label horizon 的对齐关系
- return 字段是否为 gross、成本如何扣除
- 完整 metric_convention（252、ddof=1、rf=0、MAR=0 和上述公式）
- ensemble 是否启用、join、normalize、score formula、ranking metrics 的来源
- 数据版本或截止日期（如果仓库能确定）
- 当前 Git commit（如果可获得）

八、manifest.json

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

九、校验器与 validation.json

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
- eval_config 中的策略必须是 Qlib TopkDropoutStrategy，且 topk=30、n_drop=5、method_sell=bottom、method_buy=top、hold_thresh=1、only_tradable=false、forbid_all_trade_at_limit=true、risk_degree=0.95、account=100000000、open_cost=0.0005、close_cost=0.0015、min_cost=0、freq=day。
- 回测起止日期必须等于项目声明的 test split，prediction 日期覆盖不足时检查失败。

validation.json 至少包含：

{
  "passed": true,
  "passes": 0,
  "failures": 0,
  "checks": [
    {"name": "...", "passed": true, "detail": "..."}
  ]
}

十、实现与兼容原则

- 先识别项目使用的是 Qlib、独立 PyTorch、MLflow、pickle、CSV 还是其他框架，再写最小适配层。
- 指标计算应集中到一个纯函数模块，避免在多个脚本复制公式。
- 输出路径从项目根目录或 --out 参数解析，不硬编码当前机器路径。
- 保留已有结果、训练入口和原生回测；如必须迁移旧 results，确保内容可追溯且不误删用户产物。
- 不要为了凑齐 schema 伪造 benchmark、cost、seed 或 ensemble。
- 如果现有原生策略与 TopK-DropN 不同，保留原生结果，但本协议 results/ 必须额外使用统一 Qlib TopKDropN 回测；两套结果要分开命名，不能混写。
- 如果不同 baseline 的测试期、股票池、label horizon 或数据版本不一致，必须在最终报告中醒目标注，不能仅因为回测配置和 CSV 字段相同就宣称完全公平。
- 更新项目 README，解释 results/ 各层含义、生成命令和指标定义。

十一、完成标准

完成后必须：
1. 运行语法/静态检查。
2. 运行指标边界测试：首日 -10% 时 MDD 必须为 -10%；两个相同负收益日的 Sortino 分母不得因为“负收益标准差为 0”而失效；r<=-1 必须报错。
3. 用至少一组真实日收益手工/独立反算全部组合指标。
4. 断言单 seed 与 ensemble 均显式使用本协议固定的全部 Qlib TopkDropout 参数、account=100000000、日频、open_cost=0.0005、close_cost=0.0015、min_cost=0，且回测期等于 test split、交易日 t 使用 t-1 signal。
5. 运行完整 validation，必须 0 failures。
6. 展示最终目录树、各文件行数和关键改动。
7. 汇报任何无法统一的 baseline 特有差异。

请直接检查并实施，不要只给方案。如果关键信息无法从仓库发现且不同选择会实质改变结果，才向我提一个简短问题；否则使用仓库事实推进。不要训练新模型，除非我另行授权。
```

## 使用方法

在目标 baseline 仓库中开启 AI 编码会话，把上面的整个代码块发送给 AI。建议同时补充一句：

```text
BASELINE_ID=prism_vq（或 master/matcc），请从当前仓库自动发现其余项目变量，并在修改前先报告发现的已有训练、预测和回测产物。
```

如果目标项目不使用 Qlib，也不需要改 Prompt：其中统一的是结果协议和指标纯函数，AI 应为该项目现有产物编写适配层。
