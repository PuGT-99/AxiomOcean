# AxiomOcean

**AxiomOcean: Global Three-Dimensional Ocean Forecasting**

This repository provides the **FP32 ONNX model weights and inference scripts** for AxiomOcean, a global three-dimensional ocean forecasting model with 23 vertical levels.

The release supports 10-day hybrid forecasts through a combination of 1-day and 5-day forecasting models.

## 1. Overview

- **Forecast variables:** potential temperature (`thetao`), salinity (`so`), and ocean currents (`uo`, `vo`)
- **Spatial coverage:** global ocean
- **Vertical resolution:** 23 depth levels
- **Forecast horizon:** 10 days
- **Inference framework:** ONNX Runtime with CUDA Execution Provider
- **Model precision:** FP32

The hybrid forecasting strategy uses the 1-day model for forecast leads 1–5 and the 5-day model for leads 6–10, with the latter initialized from earlier 1-day forecast states.

## 2. Repository Structure

```text
AxiomOcean/
├── README.md
├── requirements.txt
├── metadata/
│   └── model_io_spec.json
├── models/
│   └── README.md
└── scripts/
    ├── run_infer_hybrid_example.sh
    ├── infer_hybrid_rollout.py
    ├── infer_1day_autoregressive.py
    └── onnx_runtime.py
```

## 3. Model Weights

The trained FP32 ONNX models are distributed separately through [GitHub Releases](https://github.com/PuGT-99/AxiomOcean/releases).

Download both archives:

- `axiomocean_1d_fp32.zip` — 1-day forecasting model
- `axiomocean_5d_fp32.zip` — 5-day forecasting model

Extract both archives into the `models/` directory.

Each model consists of an encoder/backbone graph and a chunked decoder graph, together with their associated external weight files (`.onnx.data`).

Both models are required for the complete 10-day hybrid forecast.

## 4. Environment

Inference requires:

- Linux
- NVIDIA GPU with CUDA support
- Python
- ONNX Runtime with CUDA Execution Provider

Install the required dependencies:

```bash
pip install -r requirements.txt
```

CPU execution is not supported for the released graphs.

## 5. Input Data

Users must prepare the following datasets locally:

| Environment Variable | Description |
|---|---|
| `TEST_DATA_DIR` | Ocean initial conditions in physical units |
| `ERA5_DATA_DIR` | Normalized ERA5 atmospheric forcing fields |
| `CLIMATOLOGY_DIR` | Climatological fields used for forecast evaluation |

The model input/output specifications, including variable order and grid information, are documented in `metadata/model_io_spec.json`.


## 6. Run a 10-Day Forecast

Set the paths to the input datasets:

```bash
export TEST_DATA_DIR=/path/to/ocean
export ERA5_DATA_DIR=/path/to/era5_norm
export CLIMATOLOGY_DIR=/path/to/climatology
```

Run the hybrid forecasting workflow:

```bash
START_DATE=2020-01-01 \
END_DATE=2020-01-01 \
MAX_LEADTIME=10 \
bash scripts/run_infer_hybrid_example.sh
```

Forecast outputs are written to `outputs/` by default.

The recommended entry point is `scripts/run_infer_hybrid_example.sh`. The other inference scripts are internal components of the hybrid forecasting workflow.

## 7. Model Availability and Scope

This release includes:

- FP32 ONNX computational graphs
- Trained model weights
- Hybrid inference scripts
- Model input/output specifications