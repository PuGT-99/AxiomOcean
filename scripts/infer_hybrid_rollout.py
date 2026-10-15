#!/usr/bin/env python3
"""AxiomOcean hybrid 10-day rollout (1-day + 5-day split ONNX).

Schedule (default ``one_day_through_lead=5``), matching the manuscript operator:

- Leads 1..N1: 1-day model on consecutive states
- Leads N1+1..max: 5-day model using earlier 1-day states
  (lead L uses state indices L-5 and L-4; not chained 5-day outputs)

State index convention (same as the manuscript harness):

- states[0] = T0-1 truth, states[1] = T0 truth
- states[k+1] = forecast written for lead k
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Dict, Mapping, Optional, Set, Tuple

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

import numpy as np
from tqdm import tqdm

# Shared I/O / NetCDF helpers (same package).
import infer_1day_autoregressive as base


def _load_module(path: str, module_name: str) -> ModuleType:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Runtime module does not exist: {source}")
    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import runtime module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _empty_ocean() -> Dict[str, np.ndarray]:
    shape = (1, 2, base.N_LEVELS, base.GRID_H, base.GRID_W)
    return {name: np.empty(shape, dtype=np.float32) for name in base.OCEAN_VARS}


def _empty_surf() -> Dict[str, np.ndarray]:
    shape = (1, 2, base.GRID_H, base.GRID_W)
    return {name: np.empty(shape, dtype=np.float32) for name in base.SURFACE_VARS}


class HybridONNXRolloutTester:
    def __init__(
        self,
        *,
        runner_class,
        encoder_1d: str,
        decoder_1d: str,
        encoder_5d: str,
        decoder_5d: str,
        decoder_chunk_size: int,
        ort_gpu_id: int,
        one_day_through_lead: int,
        test_data_dir: str,
        era5_data_dir: str,
        climatology_dir: str,
        output_dir: str,
        nc_output_dir: str,
        crop_bbox: Optional[Tuple[float, float, float, float]],
        save_leadtimes: Optional[Set[int]],
        skip_existing: bool,
        progress_every: int,
        allow_missing_era5: bool,
        session_mode: str,
    ) -> None:
        self.test_data_dir = test_data_dir
        self.era5_data_dir = era5_data_dir
        self.climatology_dir = climatology_dir
        self.output_dir = output_dir
        self.nc_output_dir = nc_output_dir
        self.crop_bbox = crop_bbox
        self.save_leadtimes = save_leadtimes
        self.skip_existing = skip_existing
        self.progress_every = max(0, int(progress_every))
        self.allow_missing_era5 = allow_missing_era5
        self.one_day_through_lead = int(one_day_through_lead)
        self.session_mode = session_mode
        self.runner_class = runner_class
        self.decoder_chunk_size = decoder_chunk_size
        self.ort_gpu_id = ort_gpu_id
        self.paths = {
            "1d": (encoder_1d, decoder_1d),
            "5d": (encoder_5d, decoder_5d),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.nc_output_dir, exist_ok=True)

        # Borrow ERA5 indexing / loaders from the 1-day tester without loading ONNX.
        self._io = base.ONNXAutoregressiveTester.__new__(base.ONNXAutoregressiveTester)
        self._io.test_data_dir = test_data_dir
        self._io.era5_data_dir = era5_data_dir
        self._io.climatology_dir = climatology_dir
        self._io.nc_output_dir = nc_output_dir
        self._io.crop_bbox = crop_bbox
        self._io.allow_missing_era5 = allow_missing_era5
        self._io.era5_map = base.ONNXAutoregressiveTester._index_era5_files(self._io)

        self.runners: Dict[str, object] = {}
        if session_mode == "both":
            self.runners["1d"] = self._make_runner("1d")
            self.runners["5d"] = self._make_runner("5d")
        elif session_mode != "switch":
            raise ValueError("session_mode must be 'both' or 'switch'")

    def _make_runner(self, kind: str):
        enc, dec = self.paths[kind]
        print(f"[ONNX] Loading {kind} encoder/decoder ...")
        runner = self.runner_class(
            enc,
            dec,
            decoder_chunk_size=self.decoder_chunk_size,
            device_id=self.ort_gpu_id,
        )
        print(f"[ONNX] {kind} runner ready (sessions recreated each forward)")
        return runner

    def _get_runner(self, kind: str):
        if self.session_mode == "both":
            return self.runners[kind]
        # switch: keep only one resident
        other = "5d" if kind == "1d" else "1d"
        if other in self.runners:
            self.runners[other].close()
            del self.runners[other]
            gc.collect()
        if kind not in self.runners:
            self.runners[kind] = self._make_runner(kind)
        return self.runners[kind]

    def _pack_history(
        self,
        state_a: Mapping[str, object],
        state_b: Mapping[str, object],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        ocean = _empty_ocean()
        surf = _empty_surf()
        for name in base.OCEAN_VARS:
            ocean[name][0, 0] = state_a["ocean"][name]
            ocean[name][0, 1] = state_b["ocean"][name]
        for name in base.SURFACE_VARS:
            surf[name][0, 0] = state_a["surf"][name]
            surf[name][0, 1] = state_b["surf"][name]
        # time_days uses the second input calendar day (manuscript convention)
        time_days = np.asarray(
            [(state_b["date"] - datetime(1970, 1, 1)).total_seconds() / 86400.0],
            dtype=np.float32,
        )
        return surf, ocean, time_days

    def _process_sample(
        self,
        init_date_str: str,
        date_map: Mapping[str, str],
        max_leadtime: int,
        verbose: bool,
    ) -> Dict[str, object]:
        init_date = datetime.strptime(init_date_str, "%Y%m%d")
        previous = init_date - timedelta(days=1)
        prev_key = previous.strftime("%Y%m%d")
        if prev_key not in date_map:
            raise FileNotFoundError(f"missing previous ocean state {prev_key}")

        ocean_hist = base._allocate_ocean_history()
        grid = base._load_ocean_into_history(date_map[prev_key], ocean_hist, 0, None)
        grid = base._load_ocean_into_history(
            date_map[init_date_str], ocean_hist, 1, grid
        )

        surf0 = _empty_surf()
        self._io._load_era5_into_history(previous, grid, surf0, 0)
        self._io._load_era5_into_history(init_date, grid, surf0, 1)

        states: Dict[int, Dict[str, object]] = {
            0: {
                "date": previous,
                "ocean": {n: ocean_hist[n][0, 0].copy() for n in base.OCEAN_VARS},
                "surf": {n: surf0[n][0, 0].copy() for n in base.SURFACE_VARS},
            },
            1: {
                "date": init_date,
                "ocean": {n: ocean_hist[n][0, 1].copy() for n in base.OCEAN_VARS},
                "surf": {n: surf0[n][0, 1].copy() for n in base.SURFACE_VARS},
            },
        }

        init_out = os.path.join(self.nc_output_dir, init_date_str)
        os.makedirs(init_out, exist_ok=True)
        lead_results: Dict[int, object] = {}

        if verbose:
            print("\n" + "=" * 72)
            print(f"[Debug] Init date: {init_date:%Y-%m-%d}")
            print(
                f"[Debug] Hybrid schedule: 1d through lead "
                f"{self.one_day_through_lead}, then 5d"
            )
            print("-" * 72)

        for leadtime in range(1, max_leadtime + 1):
            if leadtime <= self.one_day_through_lead:
                kind = "1d"
                in1, in2 = leadtime - 1, leadtime
                step_days = 1
            else:
                kind = "5d"
                in1, in2 = leadtime - 5, leadtime - 4
                step_days = 5

            if in1 not in states or in2 not in states:
                raise RuntimeError(
                    f"missing states for lead {leadtime}: need {in1},{in2}"
                )

            surf, ocean, time_days = self._pack_history(states[in1], states[in2])
            runner = self._get_runner(kind)
            try:
                predicted = runner.forward(surf, ocean, time_days)
            except Exception:
                # CUDA context can be poisoned after a hard ORT failure; rebuild once.
                if kind in self.runners:
                    try:
                        self.runners[kind].close()
                    except Exception:
                        pass
                    del self.runners[kind]
                    gc.collect()
                runner = self._make_runner(kind)
                self.runners[kind] = runner
                predicted = runner.forward(surf, ocean, time_days)

            pred_date = states[in2]["date"] + timedelta(days=step_days)
            # Physical fields from decoder; keep FP32 recursive state.
            ocean_pred = {
                n: np.asarray(predicted[n][0, 0], dtype=np.float32)
                for n in base.OCEAN_VARS
            }
            del predicted

            # ERA5 at valid time for later use as a history frame
            surf_tmp = _empty_surf()
            self._io._load_era5_into_history(pred_date, grid, surf_tmp, 0)
            surf_pred = {n: surf_tmp[n][0, 0].copy() for n in base.SURFACE_VARS}

            states[leadtime + 1] = {
                "date": pred_date,
                "ocean": ocean_pred,
                "surf": surf_pred,
            }

            if self.save_leadtimes is None or leadtime in self.save_leadtimes:
                nc_path = os.path.join(
                    init_out,
                    f"fcst_{pred_date.strftime('%Y%m%d')}_leadtime{leadtime}.nc",
                )
                self._io._save_prediction_to_nc(
                    ocean_pred, grid, pred_date, nc_path
                )

            target_key = pred_date.strftime("%Y%m%d")
            if target_key in date_map:
                targets = base._load_ocean_file(date_map[target_key], grid)
                climo = self._io._load_climatology(pred_date, grid)
                metrics = self._io._calculate_metrics(
                    ocean_pred, targets, climo, grid.lat
                )
                lead_results[leadtime] = metrics
                if verbose:
                    tr = metrics.get("thetao", {})
                    print(
                        f"Lead {leadtime:02d} | {pred_date:%Y-%m-%d} | {kind} | "
                        f"thetao RMSE={tr.get('rmse', float('nan')):.4f} "
                        f"ACC={tr.get('acc', float('nan')):.4f}"
                    )

            # Drop far history to limit RAM (same idea as manuscript harness)
            drop = leadtime - 5
            if drop >= 0 and drop in states and drop not in (0, 1):
                # keep 0/1 until leads needing them finish; safe drop older
                if leadtime > self.one_day_through_lead + 1:
                    states.pop(drop, None)

            gc.collect()

        if verbose:
            print("=" * 72)
        return {
            "init": init_date_str,
            "leadtime_results": lead_results,
            "status": "success",
        }

    def run(
        self,
        *,
        max_leadtime: int,
        start_date: Optional[str],
        end_date: Optional[str],
        debug: bool,
        fail_fast: bool,
    ) -> None:
        date_map = base._build_glorys_date_map(self.test_data_dir)
        if not date_map:
            raise FileNotFoundError(f"No GLORYS files in {self.test_data_dir}")
        inits = [
            d
            for d in sorted(date_map)
            if base._date_in_range(d, start_date, end_date)
        ]
        valid = []
        for init in inits:
            prev = (
                datetime.strptime(init, "%Y%m%d") - timedelta(days=1)
            ).strftime("%Y%m%d")
            if prev in date_map:
                valid.append(init)
            else:
                print(f"[Data Warning] skip {init}: missing {prev}")

        done = failed = 0
        first = True
        for init in tqdm(valid, desc="hybrid ONNX rollout"):
            try:
                if self.skip_existing:
                    last = os.path.join(
                        self.nc_output_dir,
                        init,
                        f"fcst_"
                        f"{(datetime.strptime(init,'%Y%m%d')+timedelta(days=max_leadtime)).strftime('%Y%m%d')}"
                        f"_leadtime{max_leadtime}.nc",
                    )
                    if os.path.isfile(last):
                        continue
                self._process_sample(init, date_map, max_leadtime, debug and first)
                done += 1
                first = False
            except Exception as exc:
                failed += 1
                print(f"[Error] init={init}: {exc}", file=sys.stderr)
                if fail_fast:
                    raise
            finally:
                gc.collect()
        print(f"[Finished] done={done} failed={failed}")
        if failed:
            raise RuntimeError(f"{failed} init dates failed")


def main() -> None:
    p = argparse.ArgumentParser(description="AxiomOcean hybrid split-ONNX rollout")
    p.add_argument("--encoder_1d", required=True)
    p.add_argument("--decoder_1d", required=True)
    p.add_argument("--encoder_5d", required=True)
    p.add_argument("--decoder_5d", required=True)
    p.add_argument("--runtime_module", required=True)
    p.add_argument("--decoder_chunk_size", type=int, default=4096)
    p.add_argument("--ort_gpu_id", type=int, default=0)
    p.add_argument("--one-day-through-lead", type=int, default=5)
    p.add_argument("--session-mode", choices=("both", "switch"), default="switch")
    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--era5_data_dir", required=True)
    p.add_argument("--climatology_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--nc_output_dir", required=True)
    p.add_argument("--max-leadtime", type=int, default=10)
    p.add_argument("--start_date", default=None)
    p.add_argument("--end_date", default=None)
    p.add_argument("--save_leadtimes", default=None)
    p.add_argument("--crop_bbox", default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--allow_missing_era5", action="store_true")
    p.add_argument("--progress_every", type=int, default=25)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--fail_fast", action="store_true")
    args = p.parse_args()

    # Ensure sibling import works when launched by absolute path
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    runtime = _load_module(args.runtime_module, "axiomocean_split_runtime")
    runner_class = getattr(runtime, "SplitONNXRunner")
    save_leads = None
    if args.save_leadtimes:
        save_leads = {int(x) for x in args.save_leadtimes.split(",") if x.strip()}

    tester = HybridONNXRolloutTester(
        runner_class=runner_class,
        encoder_1d=args.encoder_1d,
        decoder_1d=args.decoder_1d,
        encoder_5d=args.encoder_5d,
        decoder_5d=args.decoder_5d,
        decoder_chunk_size=args.decoder_chunk_size,
        ort_gpu_id=args.ort_gpu_id,
        one_day_through_lead=args.one_day_through_lead,
        test_data_dir=args.test_data_dir,
        era5_data_dir=args.era5_data_dir,
        climatology_dir=args.climatology_dir,
        output_dir=args.output_dir,
        nc_output_dir=args.nc_output_dir,
        crop_bbox=base._parse_crop_bbox(args.crop_bbox),
        save_leadtimes=save_leads,
        skip_existing=args.skip_existing,
        progress_every=args.progress_every,
        allow_missing_era5=args.allow_missing_era5,
        session_mode=args.session_mode,
    )
    tester.run(
        max_leadtime=args.max_leadtime,
        start_date=args.start_date,
        end_date=args.end_date,
        debug=args.debug,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    # Make "import infer_1day_autoregressive" work when executed as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
