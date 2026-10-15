#!/usr/bin/env python3
"""One-GPU ONNX Runtime driver for the compact split AxiomOcean graphs."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort


SURF_KEYS = ("d2m", "msl", "mtpr", "ssr", "strd", "t2m", "u10", "v10")
OCEAN_KEYS = ("so", "thetao", "uo", "vo")
OCEAN_LEVELS = 23
GRID_H = 2040
GRID_W = 4320
PATCH_SIZE = 6
LATENT_LEVELS_TOTAL = 4
EMBED_DIM_DECODER = 640
PATCH_H = GRID_H // PATCH_SIZE
PATCH_W = GRID_W // PATCH_SIZE
N_PATCHES = PATCH_H * PATCH_W


class SplitONNXRunner:
    """Run the encoder once and the decoder graph over spatial chunks.

    Surface inputs must already contain the ERA5-normalised two-time-step
    histories used by the manuscript inference program.  Ocean inputs are in
    physical units.  Returned arrays are physical one-day forecasts.

    The decoder session is deliberately short-lived.  It is created only after
    the encoder has completed and is destroyed before ``forward`` returns.  In
    a multi-lead autoregressive forecast this prevents the decoder CUDA arena
    from remaining resident while the next lead runs the much larger encoder.
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        *,
        decoder_chunk_size: int = 4096,
        device_id: int = 0,
    ) -> None:
        if decoder_chunk_size <= 0:
            raise ValueError("decoder_chunk_size must be positive")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable; available providers: "
                f"{available}"
            )

        self.encoder_path = str(Path(encoder_path).resolve())
        self.decoder_path = str(Path(decoder_path).resolve())
        self.providers: List[
            Union[str, Tuple[str, Dict[str, Union[int, str]]]]
        ] = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            ),
            "CPUExecutionProvider",
        ]
        self.decoder_chunk_size = int(decoder_chunk_size)
        # Sessions are created fresh inside each forward().  Reusing a long-lived
        # encoder across decoder create/destroy cycles has been observed to trip
        # CUBLAS illegal-memory errors on some ORT+CUDA builds.
        self.encoder: Optional[ort.InferenceSession] = None
        self.decoder: Optional[ort.InferenceSession] = None
        self.encoder_types: Dict[str, str] = {}
        self.decoder_types: Dict[str, str] = {}
        self.decoder_output_type = "tensor(float16)"
        probe = self._create_encoder_session()
        self.encoder_types = {item.name: item.type for item in probe.get_inputs()}
        encoder_output = probe.get_outputs()[0]
        expected_encoder_shape = [
            1,
            LATENT_LEVELS_TOTAL * N_PATCHES,
            EMBED_DIM_DECODER,
        ]
        metadata_shape = list(encoder_output.shape)
        if len(metadata_shape) != len(expected_encoder_shape):
            raise RuntimeError(
                f"Unexpected encoder output rank: metadata={metadata_shape}; "
                f"expected={expected_encoder_shape}"
            )
        for dim_index, (actual_dim, expected_dim) in enumerate(
            zip(metadata_shape, expected_encoder_shape)
        ):
            if isinstance(actual_dim, (int, np.integer)) and actual_dim != expected_dim:
                raise RuntimeError(
                    f"Unexpected encoder output dimension {dim_index}: "
                    f"metadata={metadata_shape}; expected={expected_encoder_shape}"
                )
        print(
            "[ONNX] Encoder output metadata accepted: "
            f"{metadata_shape}; runtime expected shape={expected_encoder_shape}"
        )
        probe_dec = self._create_decoder_session()
        self.decoder_types = {
            item.name: item.type for item in probe_dec.get_inputs()
        }
        self.decoder_output_type = probe_dec.get_outputs()[0].type
        del probe, probe_dec
        gc.collect()
        print(
            "[ONNX] Session mode: recreate encoder+decoder on every forward "
            "(ORT CUDA stability)"
        )

    def _create_encoder_session(self) -> ort.InferenceSession:
        return ort.InferenceSession(self.encoder_path, providers=self.providers)

    def _create_decoder_session(self) -> ort.InferenceSession:
        return ort.InferenceSession(self.decoder_path, providers=self.providers)

    def _release_sessions(self) -> None:
        """Drop ORT sessions so the next forward starts clean."""

        self.decoder = None
        self.encoder = None
        gc.collect()

    @staticmethod
    def _numpy_dtype(ort_type: str):
        if ort_type == "tensor(float16)":
            return np.float16
        if ort_type == "tensor(float)":
            return np.float32
        raise TypeError(f"Unsupported ONNX tensor type: {ort_type}")

    @classmethod
    def _cast(cls, value: np.ndarray, ort_type: str) -> np.ndarray:
        return np.ascontiguousarray(value, dtype=cls._numpy_dtype(ort_type))

    @staticmethod
    def _validate_inputs(
        surf_vars: Mapping[str, np.ndarray],
        ocean_vars: Mapping[str, np.ndarray],
        time_days: np.ndarray,
    ) -> None:
        expected_surface = (1, 2, GRID_H, GRID_W)
        expected_ocean = (1, 2, OCEAN_LEVELS, GRID_H, GRID_W)
        for name in SURF_KEYS:
            if name not in surf_vars:
                raise KeyError(f"Missing surface variable {name!r}")
            if np.asarray(surf_vars[name]).shape != expected_surface:
                raise ValueError(
                    f"surf_vars[{name!r}] has shape "
                    f"{np.asarray(surf_vars[name]).shape}; expected {expected_surface}"
                )
        for name in OCEAN_KEYS:
            if name not in ocean_vars:
                raise KeyError(f"Missing ocean variable {name!r}")
            if np.asarray(ocean_vars[name]).shape != expected_ocean:
                raise ValueError(
                    f"ocean_vars[{name!r}] has shape "
                    f"{np.asarray(ocean_vars[name]).shape}; expected {expected_ocean}"
                )
        if np.asarray(time_days).shape != (1,):
            raise ValueError(
                f"time_days has shape {np.asarray(time_days).shape}; expected (1,)"
            )

    def _encoder_feed(
        self,
        surf_vars: Mapping[str, np.ndarray],
        ocean_vars: Mapping[str, np.ndarray],
        time_days: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        feed: Dict[str, np.ndarray] = {}
        for name in SURF_KEYS:
            input_name = f"surf_{name}"
            feed[input_name] = self._cast(
                np.asarray(surf_vars[name]), self.encoder_types[input_name]
            )
        for name in OCEAN_KEYS:
            input_name = f"ocean_{name}"
            feed[input_name] = self._cast(
                np.asarray(ocean_vars[name]), self.encoder_types[input_name]
            )
        feed["time_days"] = self._cast(
            np.asarray(time_days), self.encoder_types["time_days"]
        )
        return feed

    @staticmethod
    def _extract_ocean_latents(backbone_x: np.ndarray) -> np.ndarray:
        """Take per-patch ocean latent tokens (skip the surface latent slot)."""
        expected = (
            1,
            LATENT_LEVELS_TOTAL * N_PATCHES,
            EMBED_DIM_DECODER,
        )
        if backbone_x.shape != expected:
            raise ValueError(
                f"Unexpected encoder output shape {backbone_x.shape}; expected {expected}"
            )
        x = backbone_x.reshape(
            1,
            LATENT_LEVELS_TOTAL,
            PATCH_H,
            PATCH_W,
            EMBED_DIM_DECODER,
        )
        x = x.transpose(0, 2, 3, 1, 4).reshape(
            N_PATCHES,
            LATENT_LEVELS_TOTAL,
            EMBED_DIM_DECODER,
        )
        return np.ascontiguousarray(x[:, 1:, :])

    @staticmethod
    def _patchify_ocean_t1(
        ocean_vars: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        # (1, variables, levels, H, W) -> (1, levels, H, W, variables)
        t1 = np.stack(
            [np.asarray(ocean_vars[name])[:, -1] for name in OCEAN_KEYS],
            axis=-1,
        )
        # (1, levels, patch_h, P, patch_w, P, variables)
        t1 = t1.reshape(
            1,
            OCEAN_LEVELS,
            PATCH_H,
            PATCH_SIZE,
            PATCH_W,
            PATCH_SIZE,
            len(OCEAN_KEYS),
        )
        # (N, levels, P*P, variables) — decoder residual-add patch layout.
        t1 = t1.transpose(0, 2, 4, 1, 3, 5, 6).reshape(
            N_PATCHES,
            OCEAN_LEVELS,
            PATCH_SIZE * PATCH_SIZE,
            len(OCEAN_KEYS),
        )
        return np.ascontiguousarray(t1)

    @staticmethod
    def _patches_to_ocean_vars(fcst_patches: np.ndarray) -> Dict[str, np.ndarray]:
        expected = (
            N_PATCHES,
            OCEAN_LEVELS,
            PATCH_SIZE * PATCH_SIZE,
            len(OCEAN_KEYS),
        )
        if fcst_patches.shape != expected:
            raise ValueError(
                f"Unexpected patch output shape {fcst_patches.shape}; expected {expected}"
            )
        x = fcst_patches.reshape(
            1,
            PATCH_H,
            PATCH_W,
            OCEAN_LEVELS,
            PATCH_SIZE,
            PATCH_SIZE,
            len(OCEAN_KEYS),
        )
        x = x.transpose(0, 6, 3, 1, 4, 2, 5).reshape(
            1,
            len(OCEAN_KEYS),
            OCEAN_LEVELS,
            GRID_H,
            GRID_W,
        )
        # One-step output layout (B, T=1, Z, H, W).
        return {
            name: np.ascontiguousarray(x[:, index : index + 1])
            for index, name in enumerate(OCEAN_KEYS)
        }

    def forward(
        self,
        surf_vars: Mapping[str, np.ndarray],
        ocean_vars: Mapping[str, np.ndarray],
        time_days: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        self._validate_inputs(surf_vars, ocean_vars, time_days)
        self._release_sessions()

        encoder = self._create_encoder_session()
        self.encoder = encoder
        providers = encoder.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"Encoder graph did not select CUDAExecutionProvider: {providers}"
            )

        backbone_x = encoder.run(
            ["backbone_x"],
            self._encoder_feed(surf_vars, ocean_vars, time_days),
        )[0]
        x_ocean = self._extract_ocean_latents(backbone_x)
        del backbone_x
        # Free encoder arena before the decoder stage.
        self.encoder = None
        del encoder
        gc.collect()

        ocean_t1_patches = self._patchify_ocean_t1(ocean_vars)
        output_dtype = self._numpy_dtype(self.decoder_output_type)
        fcst_patches = np.empty(
            (
                N_PATCHES,
                OCEAN_LEVELS,
                PATCH_SIZE * PATCH_SIZE,
                len(OCEAN_KEYS),
            ),
            dtype=output_dtype,
        )

        decoder: Optional[ort.InferenceSession] = None
        try:
            decoder = self._create_decoder_session()
            self.decoder = decoder

            providers = decoder.get_providers()
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "Decoder graph did not select CUDAExecutionProvider: "
                    f"{providers}"
                )

            for start in range(0, N_PATCHES, self.decoder_chunk_size):
                end = min(start + self.decoder_chunk_size, N_PATCHES)
                feed = {
                    "x_ocean_chunk": self._cast(
                        x_ocean[start:end],
                        self.decoder_types["x_ocean_chunk"],
                    ),
                    "ocean_t1_patch_chunk": self._cast(
                        ocean_t1_patches[start:end],
                        self.decoder_types["ocean_t1_patch_chunk"],
                    ),
                }
                chunk_output = decoder.run(["fcst_patch_chunk"], feed)[0]
                fcst_patches[start:end] = chunk_output
                del chunk_output, feed

            result = self._patches_to_ocean_vars(fcst_patches)
        finally:
            self.decoder = None
            if decoder is not None:
                del decoder
            self._release_sessions()

        return result

    def close(self) -> None:
        """Release both ONNX Runtime sessions owned by this runner."""

        self._release_sessions()