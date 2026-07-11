"""Central configuration for the Qlib baseline project.

Single source of truth for: paths, dataset segments, market specs, Alpha158
processor pipeline, backtest (TopkDropoutStrategy + costs), and the 4 model
hyperparameters.  Importing this module has no side effects.

All values below were verified against the installed pyqlib 0.9.7 source and
against the official Qlib benchmark YAMLs (examples/benchmarks/{XGBoost,GRU,
TCN,Transformer}).
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"   # cached handler + segment inspection pkls
MLRUNS_DIR = PROJECT_ROOT / "mlruns"           # MLflow file store (one subdir per market)
REPORTS_DIR = PROJECT_ROOT / "reports"         # comparison_table.csv + bar charts

# --------------------------------------------------------------------------- #
# Dataset segments (locked)
# --------------------------------------------------------------------------- #
SEGMENTS = {
    "train": ("2009-01-01", "2020-12-31"),
    "valid": ("2021-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2025-12-31"),
}
TRAIN_START, TRAIN_END = SEGMENTS["train"]   # processor fit range -> no leakage into valid/test

# Handler range.  start_time gives ~245 trading days of buffer before train
# start (TSDatasetH._extend_slice auto-pads step_len=20 before each segment).
# end_time extends to the calendar max so the 2-day-forward label has no NaNs.
HANDLER_START = "2008-01-01"

# --------------------------------------------------------------------------- #
# Markets (locked).  benchmark tickers verified present under features/.
# --------------------------------------------------------------------------- #
MARKETS = {
    "csi300": dict(
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        benchmark="SH000300",
        instrument_list="csi300",
        handler_end="2026-05-25",   # cn_data calendar max (day.txt last line)
        limit_threshold=0.095,      # A-share daily price limit
    ),
    "sp500": dict(
        provider_uri="~/.qlib/qlib_data/us_data",
        region="us",
        benchmark="^gspc",
        instrument_list="sp500",
        handler_end="2026-04-29",   # us_data calendar max
        limit_threshold=None,       # no price limit in the US market
    ),
}

# --------------------------------------------------------------------------- #
# Alpha158 processor pipeline (DNN benchmark configuration)
#   infer: FilterCol(20) -> RobustZScoreNorm -> Fillna
#   learn: DropnaLabel -> CSRankNorm(label)
# FilterCol MUST precede RobustZScoreNorm so the _ts models see d_feat=20.
# XGBoost reuses the same handler -> also gets 20 feature columns.
# --------------------------------------------------------------------------- #
FILTERCOL_20 = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20",
    "CORD60", "CORD10", "CORR20", "KLOW",
]

INFER_PROCESSORS = [
    {"class": "FilterCol", "kwargs": {"fields_group": "feature", "col_list": FILTERCOL_20}},
    {"class": "RobustZScoreNorm", "kwargs": {
        "fields_group": "feature", "clip_outlier": True,
        "fit_start_time": TRAIN_START, "fit_end_time": TRAIN_END,
    }},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]
LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]
LABEL = ["Ref($close, -2) / Ref($close, -1) - 1"]

# --------------------------------------------------------------------------- #
# Backtest (locked): TopkDropoutStrategy K=30 N=5, buy 5bps / sell 15bps.
# trade_unit flows from qlib.init(region=...); limit_threshold is set per market
# because PortAnaRecord's built-in default hard-codes 0.095.
# --------------------------------------------------------------------------- #
STRATEGY_CONFIG = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "signal": "<PRED>",
        "topk": 30,
        "n_drop": 5,
        "method_sell": "bottom",
        "method_buy": "top",
        "hold_thresh": 1,
        "only_tradable": False,
        "forbid_all_trade_at_limit": True,
        "risk_degree": 0.95,
    },
}


def backtest_config(market_key: str) -> dict:
    """Per-market backtest block for PortAnaRecord(config=...)."""
    cfg = MARKETS[market_key]
    test_start, test_end = SEGMENTS["test"]
    return {
        "start_time": test_start,
        "end_time": test_end,
        "account": 100000000,
        "benchmark": cfg["benchmark"],
        "exchange_kwargs": {
            "open_cost": 0.0005,        # buy commission 5bps
            "close_cost": 0.0015,       # sell commission 15bps
            "min_cost": 0,              # user specified rates only
            "deal_price": "close",
            "limit_threshold": cfg["limit_threshold"],
        },
    }


def port_analysis_config(market_key: str) -> dict:
    """Full config dict for PortAnaRecord (executor defaults to SimulatorExecutor)."""
    return {"strategy": STRATEGY_CONFIG, "backtest": backtest_config(market_key)}


# --------------------------------------------------------------------------- #
# Models (locked, official Qlib benchmark hyperparameters)
#   ts=True  -> TSDatasetH with step_len=20 (GRU/TCN/Transformer)
#   ts=False -> DatasetH (XGBoost)
# --------------------------------------------------------------------------- #
STEP_LEN = 20
# Random seeds swept for every (market, model). `seed` is injected into every
# model's init_kwargs at runtime (XGBoost via its `seed` param, the _ts models
# set np.random.seed + torch.manual_seed in __init__), so it is intentionally
# NOT stored in the MODELS dicts below.
SEEDS = [0, 1, 2, 3, 4]

MODELS = {
    "XGBoost": dict(
        ts=False,
        module="qlib.contrib.model.xgboost",
        cls="XGBModel",
        init_kwargs=dict(
            eval_metric="rmse",
            colsample_bytree=0.8879,
            eta=0.0421,
            max_depth=8,
            n_estimators=647,
            subsample=0.8789,
            nthread=20,
        ),
        fit_kwargs=dict(num_boost_round=647, early_stopping_rounds=50, verbose_eval=False),
    ),
    "GRU": dict(
        ts=True,
        module="qlib.contrib.model.pytorch_gru_ts",
        cls="GRU",
        init_kwargs=dict(
            d_feat=20, hidden_size=64, num_layers=2, dropout=0.0,
            n_epochs=200, lr=2e-4, early_stop=10, batch_size=800,
            metric="loss", loss="mse", n_jobs=20, GPU=0,
        ),
        fit_kwargs={},
    ),
    "TCN": dict(
        ts=True,
        module="qlib.contrib.model.pytorch_tcn_ts",
        cls="TCN",
        init_kwargs=dict(
            d_feat=20, num_layers=5, n_chans=32, kernel_size=7, dropout=0.5,
            n_epochs=200, lr=1e-4, early_stop=20, batch_size=2000,
            metric="loss", loss="mse", optimizer="adam", n_jobs=20, GPU=0,
        ),
        fit_kwargs={},
    ),
    "Transformer": dict(
        ts=True,
        module="qlib.contrib.model.pytorch_transformer_ts",
        cls="TransformerModel",
        init_kwargs=dict(
            d_feat=20, d_model=64, batch_size=8192, nhead=2, num_layers=2,
            dropout=0, n_epochs=100, lr=1e-4, early_stop=5, reg=1e-3,
            n_jobs=20, GPU=0,
        ),
        fit_kwargs={},
    ),
}
