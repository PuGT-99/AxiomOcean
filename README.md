# AxiomOcean（ONNX 推理包）

10 天、23 层三维海洋预报。对外只提供 **ONNX 权重 + hybrid 推理脚本**。

## 目录

```text
axiomocean_manuscript_release/
├── README.md
├── requirements.txt
├── metadata/model_io_spec.json   # 输入 / 输出说明
├── models/                       # FP32 ONNX（1d + 5d）
└── scripts/
    ├── run_infer_hybrid_example.sh   # 唯一入口
    ├── infer_hybrid_rollout.py
    ├── infer_1day_autoregressive.py  # hybrid 内部复用，勿单独当入口
    └── onnx_runtime.py
```

## 环境

- Linux + NVIDIA GPU，`onnxruntime-gpu`（CUDA EP；CPU 跑不通本图）
- `pip install -r requirements.txt`

## 数据（需自行准备）

通过环境变量指向本地数据根目录：

| 变量 | 含义 |
|------|------|
| `TEST_DATA_DIR` | 海洋初值（物理量） |
| `ERA5_DATA_DIR` | 大气强迫（已归一化） |
| `CLIMATOLOGY_DIR` | 气候态 |

变量顺序、网格见 `metadata/model_io_spec.json`。

## 跑 10 天混合预报

日程与文稿一致：Lead 1–5 用 1 天模型；Lead 6–10 用 5 天模型（输入为更早的 1 天状态）。

```bash
export TEST_DATA_DIR=/path/to/ocean
export ERA5_DATA_DIR=/path/to/era5_norm
export CLIMATOLOGY_DIR=/path/to/climatology

START_DATE=2020-01-01 END_DATE=2020-01-01 MAX_LEADTIME=10 \
  bash scripts/run_infer_hybrid_example.sh
```

## 模型文件

| 文件 | 用途 |
|------|------|
| `axiomocean_1d_encoder_backbone_fp32.onnx[.data]` | 1 天 encoder |
| `axiomocean_1d_decoder_head_chunk_fp32.onnx[.data]` | 1 天 decoder 分块 |
| `axiomocean_5d_encoder_backbone_fp32.onnx[.data]` | 5 天 encoder |
| `axiomocean_5d_decoder_head_chunk_fp32.onnx[.data]` | 5 天 decoder 分块 |
