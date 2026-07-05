"""Build the Alpha158 handler ONCE per market and dump it to disk for reuse.

Outputs (under data_cache/):
  {market}_handler.pkl          Alpha158 with full processed state (_data/_infer/_learn
                                + fitted processors), saved with dump_all=True.  This is
                                the only artifact run.py needs; all 4 models load it.
  {market}_train.pkl / _valid.pkl / _test.pkl
                                2-D DataFrames (feature + label) per segment, for inspection.

Re-run this only if SEGMENTS, MARKETS, or the processor pipeline change.

Usage:
    python prepare_data.py                  # both markets
    python prepare_data.py csi300           # one market
"""
import sys
from pathlib import Path

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

from configs import (
    MARKETS, SEGMENTS, HANDLER_START, TRAIN_START, TRAIN_END,
    INFER_PROCESSORS, LEARN_PROCESSORS, LABEL,
    DATA_CACHE_DIR,
)


def build_and_dump(market_key: str) -> None:
    cfg = MARKETS[market_key]
    # qlib.init is global; (re)initialise for this market before building.
    qlib.init(provider_uri=cfg["provider_uri"], region=cfg["region"])

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    handler_path = DATA_CACHE_DIR / f"{market_key}_handler.pkl"

    print(f"[{market_key}] building Alpha158 handler "
          f"({HANDLER_START} -> {cfg['handler_end']}, fit {TRAIN_START}..{TRAIN_END}) ...")
    handler = Alpha158(
        instruments=cfg["instrument_list"],
        start_time=HANDLER_START,
        end_time=cfg["handler_end"],
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        infer_processors=INFER_PROCESSORS,
        learn_processors=LEARN_PROCESSORS,
        label=LABEL,
    )

    # dump_all=True keeps _data/_infer/_learn and the fitted processor instances,
    # so run.py can reuse them without recomputing any features.
    handler.to_pickle(handler_path, dump_all=True)
    size_mb = handler_path.stat().st_size / 1e6
    print(f"[{market_key}] dumped handler -> {handler_path} ({size_mb:.1f} MB)")

    # 3 inspection pkls (2-D feature+label DataFrame per segment).  Wrapping the
    # in-memory handler in DatasetH does NOT trigger any provider reload.
    ds = DatasetH(handler=handler, segments=SEGMENTS)
    for seg in ("train", "valid", "test"):
        df = ds.prepare(seg)
        seg_path = DATA_CACHE_DIR / f"{market_key}_{seg}.pkl"
        df.to_pickle(seg_path)
        print(f"[{market_key}] {seg:>5s}: {df.shape[0]:>8d} rows -> {seg_path}")


if __name__ == "__main__":
    markets = sys.argv[1:] if len(sys.argv) > 1 else list(MARKETS)
    for mk in markets:
        if mk not in MARKETS:
            raise SystemExit(f"unknown market {mk!r}; choose from {list(MARKETS)}")
        build_and_dump(mk)
