"""Run all model x market x seed experiments through the qlib.workflow Python API.

For each market: load the cached Alpha158 handler ONCE, then for each (model,
seed) wrap it in DatasetH (XGBoost) or TSDatasetH (GRU/TCN/Transformer), fit,
predict, and backtest via SignalRecord / SigAnaRecord / PortAnaRecord under an
MLflow recorder. `seed` is injected into every model's init_kwargs (XGBoost's
`seed` param; the _ts models set np/torch RNGs in __init__).

Reusing the cached handler: constructing DatasetH/TSDatasetH(handler=h, segments=...)
does NOT call handler.setup_data (DatasetH.setup_data only forwards when
handler_kwargs is given), so the pickled _data/_infer/_learn are used directly -
zero feature recomputation across models and seeds.

Usage:
    python run.py                                   # all 4 models x 2 markets x SEEDS
    python run.py --markets csi300 --models XGBoost --seeds 0 1
    python run.py --markets csi300 --models XGBoost --smoke   # tiny HPs, seed 0 only
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
    MARKETS, MODELS, SEGMENTS, STEP_LEN, SEEDS,
    port_analysis_config, DATA_CACHE_DIR, MLRUNS_DIR,
)

# xgboost 2.x emits a FutureWarning about early_stopping_rounds; harmless.
warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int) -> None:
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
        init_kwargs.update(n_estimators=20)
        fit_kwargs.update(num_boost_round=20, early_stopping_rounds=5)
    else:  # DNN models
        init_kwargs.update(n_epochs=2, early_stop=1)
    return init_kwargs, fit_kwargs


def run_one(market_key: str, model_name: str, handler: DataHandlerLP,
            seed: int, smoke: bool) -> None:
    set_seed(seed)
    ds = build_dataset(handler, model_name)

    spec = MODELS[model_name]
    init_kwargs = dict(spec["init_kwargs"])
    init_kwargs["seed"] = seed            # injected for ALL models (XGB + DNN)
    fit_kwargs = dict(spec["fit_kwargs"])
    if smoke:
        init_kwargs, fit_kwargs = smoke_overrides(model_name, init_kwargs, fit_kwargs)

    model = get_model_class(model_name)(**init_kwargs)

    uri = f"file:{(MLRUNS_DIR / market_key).resolve()}"
    (MLRUNS_DIR / market_key).mkdir(parents=True, exist_ok=True)
    exp_name = f"{market_key}_{model_name}_seed{seed}"

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


def run_market(market_key: str, model_names, seeds, smoke: bool) -> None:
    cfg = MARKETS[market_key]
    handler_path = DATA_CACHE_DIR / f"{market_key}_handler.pkl"
    if not handler_path.exists():
        raise SystemExit(
            f"{handler_path} not found - run `python prepare_data.py {market_key}` first."
        )

    qlib.init(provider_uri=cfg["provider_uri"], region=cfg["region"])
    print(f"[{market_key}] loading cached handler ...")
    handler = DataHandlerLP.load(handler_path)   # loaded once, shared across models+seeds

    for model_name in model_names:
        if model_name not in MODELS:
            raise SystemExit(f"unknown model {model_name!r}; choose from {list(MODELS)}")
        for seed in seeds:
            try:
                run_one(market_key, model_name, handler, seed, smoke)
            except Exception:
                # Keep going so one failure does not block the other experiments.
                import traceback
                print(f"!!! {market_key}/{model_name}/seed{seed} FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", nargs="+", default=list(MARKETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="tiny HPs, seed 0 only, fast gate")
    args = ap.parse_args()

    seeds = [0] if args.smoke else args.seeds
    for mk in args.markets:
        if mk not in MARKETS:
            raise SystemExit(f"unknown market {mk!r}; choose from {list(MARKETS)}")
        run_market(mk, args.models, seeds, args.smoke)
