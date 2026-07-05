"""Run all model x market experiments through the qlib.workflow Python API.

For each market: load the cached Alpha158 handler ONCE, then for each model wrap
it in DatasetH (XGBoost) or TSDatasetH (GRU/TCN/Transformer), fit, predict, and
backtest via SignalRecord / SigAnaRecord / PortAnaRecord under an MLflow recorder.

Reusing the cached handler: constructing DatasetH/TSDatasetH(handler=h, segments=...)
does NOT call handler.setup_data (DatasetH.setup_data only forwards when
handler_kwargs is given), so the pickled _data/_infer/_learn are used directly -
zero feature recomputation across models.

Usage:
    python run.py                                   # all 4 models x 2 markets
    python run.py --markets csi300 --models XGBoost GRU
    python run.py --markets csi300 --models XGBoost --smoke   # tiny HPs, fast gate
"""
import argparse
import importlib
import random
import warnings

import numpy as np
import torch

import qlib
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

from configs import (
    MARKETS, MODELS, SEGMENTS, STEP_LEN, SEED,
    port_analysis_config, DATA_CACHE_DIR, MLRUNS_DIR,
)

# xgboost 2.x emits a FutureWarning about early_stopping_rounds; harmless.
warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_class(model_name: str):
    spec = MODELS[model_name]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["cls"])


def build_dataset(handler: DataHandlerLP, model_name: str):
    """Wrap the (already loaded, processed) handler in the right dataset class.

    No setup_data call: DatasetH/TSDatasetH construction reads the in-memory
    _data/_infer/_learn directly. TSDatasetH.setup_data only builds self.cal
    from handler.fetch(CS_RAW) - also in-memory, no provider reload.
    """
    spec = MODELS[model_name]
    if spec["ts"]:
        return TSDatasetH(handler=handler, segments=SEGMENTS, step_len=STEP_LEN)
    return DatasetH(handler=handler, segments=SEGMENTS)


def smoke_overrides(model_name: str, init_kwargs: dict, fit_kwargs: dict):
    """Cut hyperparameters to a fraction of their size for a fast pipeline gate."""
    if model_name == "XGBoost":
        init_kwargs = {**init_kwargs, "n_estimators": 20}
        fit_kwargs = {**fit_kwargs, "num_boost_round": 20, "early_stopping_rounds": 5}
    else:  # DNN models
        init_kwargs = {**init_kwargs, "n_epochs": 2, "early_stop": 1}
    return init_kwargs, fit_kwargs


def run_one(market_key: str, model_name: str, handler: DataHandlerLP, smoke: bool) -> None:
    set_seed(SEED)
    ds = build_dataset(handler, model_name)

    spec = MODELS[model_name]
    init_kwargs = dict(spec["init_kwargs"])
    fit_kwargs = dict(spec["fit_kwargs"])
    if smoke:
        init_kwargs, fit_kwargs = smoke_overrides(model_name, init_kwargs, fit_kwargs)

    model_cls = get_model_class(model_name)
    model = model_cls(**init_kwargs)

    uri = f"file:{(MLRUNS_DIR / market_key).resolve()}"
    (MLRUNS_DIR / market_key).mkdir(parents=True, exist_ok=True)
    exp_name = f"{market_key}_{model_name}"

    print(f"\n=== {exp_name} ==={'  [SMOKE]' if smoke else ''}")
    with R.start(experiment_name=exp_name, uri=uri):
        R.log_params(market=market_key, model=model_name, smoke=smoke, **init_kwargs)
        model.fit(ds, **fit_kwargs)
        R.save_objects(**{"trained_model.pkl": model})

        recorder = R.get_recorder()
        SignalRecord(model=model, dataset=ds, recorder=recorder).generate()
        SigAnaRecord(recorder).generate()
        PortAnaRecord(
            recorder,
            port_analysis_config(market_key),
            risk_analysis_freq=["day"],
        ).generate()
    print(f"--- {exp_name} done")


def run_market(market_key: str, model_names, smoke: bool) -> None:
    cfg = MARKETS[market_key]
    handler_path = DATA_CACHE_DIR / f"{market_key}_handler.pkl"
    if not handler_path.exists():
        raise SystemExit(
            f"{handler_path} not found - run `python prepare_data.py {market_key}` first."
        )

    qlib.init(provider_uri=cfg["provider_uri"], region=cfg["region"])
    print(f"[{market_key}] loading cached handler ...")
    handler = DataHandlerLP.load(handler_path)   # loaded once, shared across this market's models

    for model_name in model_names:
        if model_name not in MODELS:
            raise SystemExit(f"unknown model {model_name!r}; choose from {list(MODELS)}")
        try:
            run_one(market_key, model_name, handler, smoke)
        except Exception:
            # Keep going so one failure does not block the other 7 experiments.
            import traceback
            print(f"!!! {market_key}/{model_name} FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", nargs="+", default=list(MARKETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--smoke", action="store_true", help="tiny HPs for a fast end-to-end gate")
    args = ap.parse_args()

    for mk in args.markets:
        if mk not in MARKETS:
            raise SystemExit(f"unknown market {mk!r}; choose from {list(MARKETS)}")
        run_market(mk, args.models, args.smoke)
