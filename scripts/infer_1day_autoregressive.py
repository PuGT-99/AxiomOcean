#!/usr/bin/env python3
"""Internal helpers shared by hybrid rollout (I/O, NetCDF, metrics).

Not a public entrypoint — use ``run_infer_hybrid_example.sh``.
Also contains a one-day autoregressive path used only as a library by
``infer_hybrid_rollout.py``.
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Prefer importing torch first so matching CUDA/cuDNN libs are on the path.
# Some ORT-only environments can run without torch.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


GLORYS_DEPTHS: Tuple[float, ...] = (
    0.494,
    2.645,
    5.078,
    7.929,
    11.405,
    15.810,
    21.598,
    29.444,
    40.344,
    55.764,
    77.853,
    92.326,
    109.729,
    130.666,
    155.850,
    186.125,
    222.475,
    266.040,
    318.125,
    380.208,
    453.937,
    541.088,
    643.566,
)

OCEAN_VARS: Tuple[str, ...] = ("so", "thetao", "uo", "vo")
SURFACE_VARS: Tuple[str, ...] = (
    "d2m",
    "msl",
    "mtpr",
    "ssr",
    "strd",
    "t2m",
    "u10",
    "v10",
)
GRID_H = 2040
GRID_W = 4320
N_LEVELS = len(GLORYS_DEPTHS)


@dataclass(frozen=True)
class GridMetadata:
    lat: np.ndarray
    lon: np.ndarray
    date: datetime


def _parse_ymd(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _yyyymmdd(value: datetime) -> str:
    return value.strftime("%Y%m%d")


def _date_in_range(
    date_str: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> bool:
    current = datetime.strptime(date_str, "%Y%m%d")
    if start_date and current < _parse_ymd(start_date):
        return False
    if end_date and current > _parse_ymd(end_date):
        return False
    return True


def _build_glorys_date_map(test_data_dir: str) -> Dict[str, str]:
    """Map YYYYMMDD to either legacy GLORYS or newer glo12_rg files."""
    date_map: Dict[str, str] = {}
    patterns = (
        ("mercatorglorys12v1_gl12_mean_*.nc", r"mercatorglorys12v1_gl12_mean_(\d{8})"),
        ("glo12_rg_1d-m_*.nc", r"glo12_rg_1d-m_(\d{8})"),
    )
    for pattern, regex in patterns:
        for file_path in sorted(glob.glob(os.path.join(test_data_dir, pattern))):
            match = re.search(regex, os.path.basename(file_path))
            if match:
                # The newer glo12_rg file wins if both formats contain a day.
                date_map[match.group(1)] = file_path
    return date_map


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


def _coordinate_name(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    return None


def _normalise_grid_coordinates(
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.mod(np.asarray(lon, dtype=np.float32), 360.0)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Only one-dimensional latitude/longitude coordinates are supported")

    flip_lat = bool(lat[0] < lat[-1])
    if flip_lat:
        lat = lat[::-1].copy()

    lon_order = np.argsort(lon, kind="stable")
    lon = lon[lon_order]
    return lat, lon, flip_lat, lon_order


def _extract_3d_ocean_array(
    ds: xr.Dataset,
    var_name: str,
    *,
    flip_lat: bool,
    lon_order: np.ndarray,
) -> np.ndarray:
    if var_name not in ds.variables:
        raise KeyError(f"Missing ocean variable {var_name!r} in {ds.encoding.get('source')}")
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)
    array = np.asarray(da.values, dtype=np.float32)
    array = np.squeeze(array)
    if array.shape != (N_LEVELS, GRID_H, GRID_W):
        raise ValueError(
            f"{var_name} has shape {array.shape}; expected "
            f"({N_LEVELS}, {GRID_H}, {GRID_W})"
        )
    if flip_lat:
        array = array[:, ::-1, :]
    if not np.array_equal(lon_order, np.arange(GRID_W)):
        array = array[..., lon_order]
    return np.ascontiguousarray(array, dtype=np.float32)


def _read_dataset_date(ds: xr.Dataset, file_path: str) -> datetime:
    if "time" in ds.variables and ds["time"].size:
        value = np.asarray(ds["time"].values).reshape(-1)[0]
        date_text = str(value)[:10]
        return datetime.strptime(date_text, "%Y-%m-%d")
    match = re.search(r"(\d{8})", os.path.basename(file_path))
    if not match:
        raise ValueError(f"Cannot determine date from {file_path}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def _allocate_ocean_history() -> Dict[str, np.ndarray]:
    shape = (1, 2, N_LEVELS, GRID_H, GRID_W)
    return {name: np.empty(shape, dtype=np.float32) for name in OCEAN_VARS}


def _load_ocean_into_history(
    file_path: str,
    history: Mapping[str, np.ndarray],
    history_index: int,
    expected_grid: Optional[GridMetadata] = None,
) -> GridMetadata:
    with xr.open_dataset(file_path, cache=False) as ds:
        lat_name = _coordinate_name(ds, ("latitude", "lat"))
        lon_name = _coordinate_name(ds, ("longitude", "lon"))
        if lat_name is None or lon_name is None:
            raise KeyError(f"Missing latitude/longitude coordinates in {file_path}")
        lat, lon, flip_lat, lon_order = _normalise_grid_coordinates(
            ds[lat_name].values,
            ds[lon_name].values,
        )
        if lat.shape != (GRID_H,) or lon.shape != (GRID_W,):
            raise ValueError(
                f"Unexpected grid in {file_path}: lat={lat.shape}, lon={lon.shape}; "
                f"expected ({GRID_H},), ({GRID_W},)"
            )
        date = _read_dataset_date(ds, file_path)

        if expected_grid is not None:
            if not np.allclose(lat, expected_grid.lat, atol=5e-4, rtol=0.0):
                raise ValueError(f"Latitude grid differs in {file_path}")
            if not np.allclose(lon, expected_grid.lon, atol=5e-4, rtol=0.0):
                raise ValueError(f"Longitude grid differs in {file_path}")

        for name in OCEAN_VARS:
            history[name][0, history_index] = _extract_3d_ocean_array(
                ds,
                name,
                flip_lat=flip_lat,
                lon_order=lon_order,
            )
    return GridMetadata(lat=lat, lon=lon, date=date)


def _load_ocean_file(
    file_path: str,
    expected_grid: GridMetadata,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    with xr.open_dataset(file_path, cache=False) as ds:
        lat_name = _coordinate_name(ds, ("latitude", "lat"))
        lon_name = _coordinate_name(ds, ("longitude", "lon"))
        if lat_name is None or lon_name is None:
            raise KeyError(f"Missing latitude/longitude coordinates in {file_path}")
        lat, lon, flip_lat, lon_order = _normalise_grid_coordinates(
            ds[lat_name].values,
            ds[lon_name].values,
        )
        if not np.allclose(lat, expected_grid.lat, atol=5e-4, rtol=0.0):
            raise ValueError(f"Latitude grid differs in {file_path}")
        if not np.allclose(lon, expected_grid.lon, atol=5e-4, rtol=0.0):
            raise ValueError(f"Longitude grid differs in {file_path}")
        for name in OCEAN_VARS:
            result[name] = _extract_3d_ocean_array(
                ds,
                name,
                flip_lat=flip_lat,
                lon_order=lon_order,
            )
    return result


class ONNXAutoregressiveTester:
    def __init__(
        self,
        *,
        runner_class,
        encoder_onnx: str,
        decoder_onnx: str,
        decoder_chunk_size: int,
        ort_gpu_id: int,
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

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.nc_output_dir, exist_ok=True)
        self.era5_map = self._index_era5_files()

        print("[ONNX] Preparing split ONNX runner (sessions recreated each forward) ...")
        self.runner = runner_class(
            encoder_onnx,
            decoder_onnx,
            decoder_chunk_size=decoder_chunk_size,
            device_id=ort_gpu_id,
        )
        print("[ONNX] Runner ready")

    def _index_era5_files(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        pattern = os.path.join(self.era5_data_dir, "**", "*.nc")
        for file_path in sorted(glob.glob(pattern, recursive=True)):
            match = re.search(r"(\d{8})", os.path.basename(file_path))
            if match:
                result[match.group(1)] = file_path
        print(f"[Data] Indexed {len(result)} ERA5 daily files")
        return result

    @staticmethod
    def _allocate_surface_history() -> Dict[str, np.ndarray]:
        shape = (1, 2, GRID_H, GRID_W)
        return {name: np.empty(shape, dtype=np.float32) for name in SURFACE_VARS}

    def _fill_missing_surface(
        self,
        history: Mapping[str, np.ndarray],
        history_index: int,
        message: str,
    ) -> None:
        if not self.allow_missing_era5:
            raise FileNotFoundError(message)
        print(f"[ERA5 Warning] {message}; using zeros by explicit request")
        for name in SURFACE_VARS:
            history[name][0, history_index].fill(0.0)

    def _load_era5_into_history(
        self,
        date: datetime,
        grid: GridMetadata,
        history: Mapping[str, np.ndarray],
        history_index: int,
    ) -> None:
        date_str = _yyyymmdd(date)
        file_path = self.era5_map.get(date_str)
        if not file_path or not os.path.isfile(file_path):
            self._fill_missing_surface(
                history,
                history_index,
                f"ERA5 file is missing for {date_str}",
            )
            return

        try:
            with xr.open_dataset(file_path, cache=False) as ds:
                lat_name = _coordinate_name(ds, ("latitude", "lat"))
                lon_name = _coordinate_name(ds, ("longitude", "lon"))
                flip_lat = False
                lon_order = np.arange(GRID_W)
                if lat_name is not None and lon_name is not None:
                    lat, lon, flip_lat, lon_order = _normalise_grid_coordinates(
                        ds[lat_name].values,
                        ds[lon_name].values,
                    )
                    if not np.allclose(lat, grid.lat, atol=5e-4, rtol=0.0):
                        raise ValueError("latitude grid differs from the ocean grid")
                    if not np.allclose(lon, grid.lon, atol=5e-4, rtol=0.0):
                        raise ValueError("longitude grid differs from the ocean grid")

                for name in SURFACE_VARS:
                    if name not in ds.variables:
                        raise KeyError(f"missing variable {name!r}")
                    da = ds[name]
                    if "time" in da.dims:
                        da = da.isel(time=0)
                    value = np.squeeze(np.asarray(da.values, dtype=np.float32))
                    if value.shape != (GRID_H, GRID_W):
                        raise ValueError(
                            f"{name} has shape {value.shape}; expected ({GRID_H}, {GRID_W})"
                        )
                    if flip_lat:
                        value = value[::-1, :]
                    if not np.array_equal(lon_order, np.arange(GRID_W)):
                        value = value[:, lon_order]
                    history[name][0, history_index] = value
        except Exception as exc:
            self._fill_missing_surface(
                history,
                history_index,
                f"Failed to read ERA5 {date_str} ({file_path}): {exc}",
            )

    def _load_climatology(
        self,
        target_date: datetime,
        grid: GridMetadata,
    ) -> Dict[str, np.ndarray]:
        file_path = os.path.join(
            self.climatology_dir,
            "mercatorglorys12v1_gl12_mean_1993_2016_"
            f"{target_date.strftime('%m')}.nc",
        )
        if not os.path.isfile(file_path):
            print(f"[Metric Warning] Climatology is missing: {file_path}; ACC=NaN")
            return {}
        try:
            return _load_ocean_file(file_path, grid)
        except Exception as exc:
            print(f"[Metric Warning] Failed to read climatology {file_path}: {exc}; ACC=NaN")
            return {}

    @staticmethod
    def _calculate_metrics(
        predictions: Mapping[str, np.ndarray],
        targets: Mapping[str, np.ndarray],
        climatology: Mapping[str, np.ndarray],
        lat: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute latitude-weighted metrics without global temporary arrays."""
        results: Dict[str, Dict[str, float]] = {}
        latitude_weights = np.cos(np.deg2rad(lat.astype(np.float64)))
        latitude_chunk = 128

        for name in OCEAN_VARS:
            pred = np.asarray(predictions[name], dtype=np.float32)
            target = np.asarray(targets[name], dtype=np.float32)
            if pred.shape != target.shape:
                raise ValueError(
                    f"Metric shape mismatch for {name}: {pred.shape} vs {target.shape}"
                )
            climo = climatology.get(name)
            if climo is not None:
                climo = np.asarray(climo, dtype=np.float32)

            weight_sum = 0.0
            squared_error_sum = 0.0
            absolute_error_sum = 0.0
            covariance_sum = 0.0
            pred_variance_sum = 0.0
            target_variance_sum = 0.0

            for depth_index in range(pred.shape[0]):
                for lat_start in range(0, pred.shape[1], latitude_chunk):
                    lat_end = min(lat_start + latitude_chunk, pred.shape[1])
                    pred_block = pred[depth_index, lat_start:lat_end]
                    target_block = target[depth_index, lat_start:lat_end]
                    valid = np.isfinite(pred_block) & np.isfinite(target_block)
                    climo_block = None
                    if climo is not None:
                        climo_block = climo[depth_index, lat_start:lat_end]
                        valid &= np.isfinite(climo_block)
                    if not np.any(valid):
                        continue

                    row_weights = latitude_weights[lat_start:lat_end, None]
                    weights = np.broadcast_to(row_weights, pred_block.shape)[valid]
                    pred_valid = pred_block[valid].astype(np.float64)
                    target_valid = target_block[valid].astype(np.float64)
                    error = pred_valid - target_valid

                    weight_sum += float(weights.sum(dtype=np.float64))
                    squared_error_sum += float(
                        np.sum(error * error * weights, dtype=np.float64)
                    )
                    absolute_error_sum += float(
                        np.sum(np.abs(error) * weights, dtype=np.float64)
                    )

                    if climo_block is not None:
                        climo_valid = climo_block[valid].astype(np.float64)
                        pred_anom = pred_valid - climo_valid
                        target_anom = target_valid - climo_valid
                        covariance_sum += float(
                            np.sum(pred_anom * target_anom * weights, dtype=np.float64)
                        )
                        pred_variance_sum += float(
                            np.sum(pred_anom * pred_anom * weights, dtype=np.float64)
                        )
                        target_variance_sum += float(
                            np.sum(target_anom * target_anom * weights, dtype=np.float64)
                        )

            if weight_sum <= 0.0:
                results[name] = {"rmse": np.nan, "mae": np.nan, "acc": np.nan}
                continue
            rmse = np.sqrt(squared_error_sum / weight_sum)
            mae = absolute_error_sum / weight_sum
            acc = np.nan
            if (
                climo is not None
                and pred_variance_sum > 0.0
                and target_variance_sum > 0.0
            ):
                acc = covariance_sum / np.sqrt(
                    pred_variance_sum * target_variance_sum
                )
            results[name] = {"rmse": rmse, "mae": mae, "acc": acc}
        return results

    def _crop_indices(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.crop_bbox is None:
            return np.arange(lat.size), np.arange(lon.size)
        lat_min, lat_max, lon_min, lon_max = self.crop_bbox
        lat_idx = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
        if lon_min <= lon_max:
            lon_idx = np.flatnonzero((lon >= lon_min) & (lon <= lon_max))
        else:
            lon_idx = np.flatnonzero((lon >= lon_min) | (lon <= lon_max))
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise ValueError(f"crop_bbox {self.crop_bbox} selects an empty grid")
        return lat_idx, lon_idx

    def _save_prediction_to_nc(
        self,
        prediction: Mapping[str, np.ndarray],
        grid: GridMetadata,
        predicted_date: datetime,
        save_path: str,
    ) -> None:
        lat_idx, lon_idx = self._crop_indices(grid.lat, grid.lon)
        coords = {
            "time": [np.datetime64(predicted_date)],
            "depth": np.asarray(GLORYS_DEPTHS, dtype=np.float32),
            "latitude": grid.lat[lat_idx],
            "longitude": grid.lon[lon_idx],
        }
        data_vars = {}
        for name in OCEAN_VARS:
            value = np.asarray(prediction[name], dtype=np.float32)
            if self.crop_bbox is None:
                cropped = value
            elif (
                np.all(np.diff(lat_idx) == 1)
                and np.all(np.diff(lon_idx) == 1)
            ):
                cropped = value[
                    :,
                    lat_idx[0] : lat_idx[-1] + 1,
                    lon_idx[0] : lon_idx[-1] + 1,
                ]
            else:
                # Only a longitude interval crossing 0 degrees should normally
                # reach this branch; cropped regional output is small enough to copy.
                cropped = value[:, lat_idx, :][:, :, lon_idx]
            data_vars[name] = (
                ("time", "depth", "latitude", "longitude"),
                cropped[None],
            )
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        try:
            dataset.to_netcdf(temporary)
            os.replace(str(temporary), str(output))
        finally:
            dataset.close()
            if temporary.exists():
                temporary.unlink()

    def _expected_fcst_path(self, init_date: str, leadtime: int) -> str:
        init = datetime.strptime(init_date, "%Y%m%d")
        valid = _yyyymmdd(init + timedelta(days=leadtime))
        return os.path.join(
            self.nc_output_dir,
            init_date,
            f"fcst_{valid}_leadtime{leadtime}.nc",
        )

    def _output_complete(
        self,
        init_date: str,
        max_leadtime: int,
    ) -> bool:
        leadtimes: Iterable[int]
        if self.save_leadtimes is None:
            leadtimes = range(1, max_leadtime + 1)
        else:
            leadtimes = sorted(self.save_leadtimes)
        for leadtime in leadtimes:
            path = self._expected_fcst_path(init_date, leadtime)
            if not os.path.isfile(path) or os.path.getsize(path) < 4096:
                return False
            try:
                with xr.open_dataset(path, cache=False) as ds:
                    if int(ds.sizes.get("time", 0)) != 1:
                        return False
                    if int(ds.sizes.get("depth", 0)) != N_LEVELS:
                        return False
                    if any(name not in ds.variables for name in OCEAN_VARS):
                        return False
            except Exception:
                return False
        return True

    def _write_progress(
        self,
        done: int,
        skipped: int,
        failed: int,
        total: int,
        last_init: str,
    ) -> None:
        row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "done": done,
            "skipped": skipped,
            "failed": failed,
            "total": total,
            "remaining": total - done - skipped - failed,
            "last_init": last_init,
            "strategy": "one_day_onnx_autoregressive",
        }
        log_path = os.path.join(self.nc_output_dir, "progress_summary_1day_onnx.jsonl")
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[Progress] done={done} skipped={skipped} failed={failed} "
            f"total={total} remaining={row['remaining']} last={last_init}"
        )

    @staticmethod
    def _time_days(date: datetime) -> np.ndarray:
        epoch = datetime(1970, 1, 1)
        return np.asarray([(date - epoch).total_seconds() / 86400.0], dtype=np.float32)

    def _process_sample(
        self,
        init_date_str: str,
        date_map: Mapping[str, str],
        max_leadtime: int,
        verbose: bool,
    ) -> Dict[str, object]:
        init_date = datetime.strptime(init_date_str, "%Y%m%d")
        previous_date = init_date - timedelta(days=1)
        previous_key = _yyyymmdd(previous_date)
        if previous_key not in date_map:
            raise FileNotFoundError(
                f"Initial history is incomplete: missing ocean state {previous_key}"
            )

        ocean_history = _allocate_ocean_history()
        grid_t0 = _load_ocean_into_history(
            date_map[previous_key], ocean_history, 0, expected_grid=None
        )
        grid_t1 = _load_ocean_into_history(
            date_map[init_date_str], ocean_history, 1, expected_grid=grid_t0
        )
        if grid_t0.date.date() != previous_date.date():
            raise ValueError(
                f"First input date is {grid_t0.date:%Y-%m-%d}; expected {previous_date:%Y-%m-%d}"
            )
        if grid_t1.date.date() != init_date.date():
            raise ValueError(
                f"Second input date is {grid_t1.date:%Y-%m-%d}; expected {init_date:%Y-%m-%d}"
            )

        surface_history = self._allocate_surface_history()
        self._load_era5_into_history(previous_date, grid_t1, surface_history, 0)
        self._load_era5_into_history(init_date, grid_t1, surface_history, 1)

        init_output_dir = os.path.join(self.nc_output_dir, init_date_str)
        os.makedirs(init_output_dir, exist_ok=True)
        leadtime_results: Dict[int, Dict[str, object]] = {}

        if verbose:
            print("\n" + "=" * 72)
            print(f"[Debug] Init date: {init_date:%Y-%m-%d}")
            print("[Debug] Strategy : one-day ONNX autoregression for every lead")
            print("-" * 72)

        current_date = init_date
        for leadtime in range(1, max_leadtime + 1):
            expected_date = init_date + timedelta(days=leadtime)
            predicted = self.runner.forward(
                surface_history,
                ocean_history,
                self._time_days(current_date),
            )

            # Shift the two-frame history and store the new forecast in FP32.
            # The exported graph may output FP16; keeping the recursive state in
            # FP32 avoids an additional CPU-side quantisation step.
            for name in OCEAN_VARS:
                np.copyto(ocean_history[name][0, 0], ocean_history[name][0, 1])
                np.copyto(
                    ocean_history[name][0, 1],
                    predicted[name][0, 0],
                    casting="unsafe",
                )
            del predicted
            current_date = expected_date

            prediction_view = {
                name: ocean_history[name][0, 1] for name in OCEAN_VARS
            }
            if self.save_leadtimes is None or leadtime in self.save_leadtimes:
                nc_path = self._expected_fcst_path(init_date_str, leadtime)
                self._save_prediction_to_nc(
                    prediction_view,
                    grid_t1,
                    expected_date,
                    nc_path,
                )

            target_key = _yyyymmdd(expected_date)
            target_path = date_map.get(target_key)
            if target_path is not None:
                targets = _load_ocean_file(target_path, grid_t1)
                climatology = self._load_climatology(expected_date, grid_t1)
                metrics = self._calculate_metrics(
                    prediction_view,
                    targets,
                    climatology,
                    grid_t1.lat,
                )
                leadtime_results[leadtime] = {
                    "target_file": os.path.basename(target_path),
                    "target_date": expected_date.strftime("%Y-%m-%d"),
                    "predicted_date": expected_date.strftime("%Y-%m-%d"),
                    "rmse": {name: values["rmse"] for name, values in metrics.items()},
                    "mae": {name: values["mae"] for name, values in metrics.items()},
                    "acc": {name: values["acc"] for name, values in metrics.items()},
                }
                if verbose:
                    theta_rmse = metrics.get("thetao", {}).get("rmse", np.nan)
                    theta_acc = metrics.get("thetao", {}).get("acc", np.nan)
                    print(
                        f"Lead {leadtime:02d} | {expected_date:%Y-%m-%d} | "
                        f"thetao RMSE={theta_rmse:.4f} ACC={theta_acc:.4f}"
                    )
                del targets, climatology, metrics
            else:
                print(
                    f"[Metric Warning] Truth is missing for {target_key}; "
                    f"forecast was written but metrics were skipped"
                )

            # The next step uses forcing at the newly predicted date.  No new
            # forcing is needed after the final forecast has been generated.
            if leadtime < max_leadtime:
                for name in SURFACE_VARS:
                    np.copyto(surface_history[name][0, 0], surface_history[name][0, 1])
                self._load_era5_into_history(
                    expected_date,
                    grid_t1,
                    surface_history,
                    1,
                )
            gc.collect()

        if verbose:
            print("=" * 72 + "\n")
        return {
            "start_date": init_date.strftime("%Y-%m-%d"),
            "status": "success",
            "leadtime_results": leadtime_results,
        }

    @staticmethod
    def _merge_write_csv(
        rows: List[Dict[str, object]],
        path: str,
    ) -> None:
        new_frame = pd.DataFrame(rows)
        if os.path.isfile(path):
            old_frame = pd.read_csv(path)
            frame = pd.concat((old_frame, new_frame), ignore_index=True)
        else:
            if new_frame.empty:
                print(f"[Metric Warning] No metric rows available; not creating {path}")
                return
            frame = new_frame
        if frame.empty:
            frame.to_csv(path, index=False)
            return
        key_columns = [name for name in ("start_date", "leadtime") if name in frame.columns]
        if key_columns:
            frame = frame.drop_duplicates(subset=key_columns, keep="last")
        sort_columns = [name for name in ("start_date", "leadtime") if name in frame.columns]
        if sort_columns:
            frame = frame.sort_values(sort_columns)
        temporary = path + ".tmp"
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)

    def _save_rollout_results(
        self,
        results: Sequence[Mapping[str, object]],
        max_leadtime: int,
    ) -> Tuple[str, str]:
        rmse_rows: List[Dict[str, object]] = []
        acc_rows: List[Dict[str, object]] = []
        for result in results:
            if result.get("status") != "success":
                continue
            base = {"start_date": result["start_date"]}
            lead_results = result.get("leadtime_results", {})
            for leadtime, lead_result in lead_results.items():
                rmse_row = dict(base)
                rmse_row.update(
                    {
                        "leadtime": leadtime,
                        "target_date": lead_result["target_date"],
                        "predicted_date": lead_result["predicted_date"],
                    }
                )
                for name, value in lead_result["rmse"].items():
                    rmse_row[f"{name}_rmse"] = value
                rmse_rows.append(rmse_row)

                acc_row = dict(base)
                acc_row.update(
                    {
                        "leadtime": leadtime,
                        "target_date": lead_result["target_date"],
                        "predicted_date": lead_result["predicted_date"],
                    }
                )
                for name, value in lead_result["acc"].items():
                    acc_row[f"{name}_acc"] = value
                acc_rows.append(acc_row)

        rmse_path = os.path.join(
            self.output_dir,
            f"rollout_results_rmse_lead{max_leadtime}_1day_onnx.csv",
        )
        acc_path = os.path.join(
            self.output_dir,
            f"rollout_results_acc_lead{max_leadtime}_1day_onnx.csv",
        )
        self._merge_write_csv(rmse_rows, rmse_path)
        self._merge_write_csv(acc_rows, acc_path)
        return rmse_path, acc_path

    def run(
        self,
        *,
        max_leadtime: int,
        start_date: Optional[str],
        end_date: Optional[str],
        debug: bool,
        fail_fast: bool,
    ) -> Tuple[str, str]:
        date_map = _build_glorys_date_map(self.test_data_dir)
        if not date_map:
            raise FileNotFoundError(f"No GLORYS files found in {self.test_data_dir}")
        n_legacy = sum("mercatorglorys" in path for path in date_map.values())
        n_new = sum("glo12_rg" in path for path in date_map.values())
        print(
            f"[Data] GLORYS index: {len(date_map)} days "
            f"(mercator={n_legacy}, glo12_rg={n_new})"
        )

        init_dates = [
            value
            for value in sorted(date_map)
            if _date_in_range(value, start_date, end_date)
        ]
        print(
            f"[Data] Init dates in range {start_date} to {end_date}: "
            f"{len(init_dates)}"
        )

        valid_init_dates = []
        for init_date in init_dates:
            previous = _yyyymmdd(
                datetime.strptime(init_date, "%Y%m%d") - timedelta(days=1)
            )
            if previous not in date_map:
                print(
                    f"[Data Warning] Skip init {init_date}: "
                    f"missing previous ocean state {previous}"
                )
                continue
            valid_init_dates.append(init_date)

        results: List[Mapping[str, object]] = []
        done = skipped = failed = 0
        total = len(valid_init_dates)
        progress = tqdm(valid_init_dates, desc="1-day ONNX autoregression")
        first_verbose = True

        for init_date in progress:
            if self.skip_existing and self._output_complete(init_date, max_leadtime):
                skipped += 1
            else:
                try:
                    result = self._process_sample(
                        init_date,
                        date_map,
                        max_leadtime,
                        verbose=(debug and first_verbose),
                    )
                    results.append(result)
                    done += 1
                    first_verbose = False
                except Exception as exc:
                    failed += 1
                    print(f"[Error] init={init_date}: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    if fail_fast:
                        raise
                finally:
                    gc.collect()

            processed = done + skipped + failed
            if self.progress_every and processed % self.progress_every == 0:
                self._write_progress(done, skipped, failed, total, init_date)

        self._write_progress(done, skipped, failed, total, "FINAL")
        rmse_path, acc_path = self._save_rollout_results(results, max_leadtime)
        print(
            f"[Finished] done={done}, skipped={skipped}, failed={failed}; "
            f"RMSE={rmse_path}; ACC={acc_path}"
        )
        if failed:
            raise RuntimeError(f"{failed} init dates failed; inspect the log above")
        return rmse_path, acc_path


def _parse_crop_bbox(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not value:
        return None
    parts = tuple(float(item.strip()) for item in value.split(","))
    if len(parts) != 4:
        raise ValueError("--crop_bbox requires lat_min,lat_max,lon_min,lon_max")
    lat_min, lat_max, _, _ = parts
    if lat_min > lat_max:
        raise ValueError("crop_bbox latitude minimum must not exceed maximum")
    return parts


def _parse_leadtimes(value: Optional[str], max_leadtime: int) -> Optional[Set[int]]:
    if not value:
        return None
    leadtimes = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(value for value in leadtimes if value < 1 or value > max_leadtime)
    if invalid:
        raise ValueError(
            f"save_leadtimes outside 1..{max_leadtime}: {invalid}"
        )
    return leadtimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AxiomOcean split-ONNX one-day autoregressive test"
    )
    parser.add_argument("--encoder_onnx", required=True)
    parser.add_argument("--decoder_onnx", required=True)
    parser.add_argument("--runtime_module", required=True)
    parser.add_argument("--decoder_chunk_size", type=int, default=4096)
    parser.add_argument("--ort_gpu_id", type=int, default=0)
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--era5_data_dir", required=True)
    parser.add_argument("--climatology_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nc_output_dir", required=True)
    parser.add_argument("--max-leadtime", type=int, default=10)
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--crop_bbox", default=None)
    parser.add_argument("--save_leadtimes", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--allow_missing_era5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_leadtime < 1:
        raise ValueError("--max-leadtime must be at least 1")
    if args.decoder_chunk_size < 1:
        raise ValueError("--decoder_chunk_size must be positive")
    if args.start_date:
        _parse_ymd(args.start_date)
    if args.end_date:
        _parse_ymd(args.end_date)
    if args.start_date and args.end_date:
        if _parse_ymd(args.start_date) > _parse_ymd(args.end_date):
            raise ValueError("start_date must not be after end_date")

    runtime = _load_module(args.runtime_module, "axiomocean_split_runtime")
    runner_class = getattr(runtime, "SplitONNXRunner")
    runtime_ocean_vars = tuple(getattr(runtime, "OCEAN_KEYS"))
    runtime_surface_vars = tuple(getattr(runtime, "SURF_KEYS"))
    if runtime_ocean_vars != OCEAN_VARS:
        raise RuntimeError(
            f"Runtime ocean order {runtime_ocean_vars} != expected {OCEAN_VARS}"
        )
    if runtime_surface_vars != SURFACE_VARS:
        raise RuntimeError(
            f"Runtime surface order {runtime_surface_vars} != expected {SURFACE_VARS}"
        )

    tester = ONNXAutoregressiveTester(
        runner_class=runner_class,
        encoder_onnx=args.encoder_onnx,
        decoder_onnx=args.decoder_onnx,
        decoder_chunk_size=args.decoder_chunk_size,
        ort_gpu_id=args.ort_gpu_id,
        test_data_dir=args.test_data_dir,
        era5_data_dir=args.era5_data_dir,
        climatology_dir=args.climatology_dir,
        output_dir=args.output_dir,
        nc_output_dir=args.nc_output_dir,
        crop_bbox=_parse_crop_bbox(args.crop_bbox),
        save_leadtimes=_parse_leadtimes(args.save_leadtimes, args.max_leadtime),
        skip_existing=args.skip_existing,
        progress_every=args.progress_every,
        allow_missing_era5=args.allow_missing_era5,
    )
    tester.run(
        max_leadtime=args.max_leadtime,
        start_date=args.start_date,
        end_date=args.end_date,
        debug=args.debug,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()