# bitnet_nvfp4

Minimal, NVFP4‑focused training harness for the BitNet EFLA model (EGGROLL). This repo contains the fastest working NVFP4 path with sensible defaults and no extra experiments.

## Setup

```bash
git submodule update --init --recursive
./scripts/get_data.sh
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run (reproducible)

```bash
./build/train_efla_lm \
  --epochs 10 --no_val --fixed_train --data_seed 123 --seed 1
```

Defaults are tuned for throughput:
- NVFP4 block‑scaled GEMM (sm120a)
- all GPUs by default
- `gpu_workers=32` (best total throughput on the current setup)
- `pop=2048, batch=32, seq=64, hidden=512, layers=4, mlp_mult=2`
- NVFP4 defaults: split‑K decomp, splits=2, autotune off
- CUDA graph capture off, fused EFLA step off

## Notes
- NVFP4 requires SM120a+ and hidden size divisible by 128; the binary will abort if misaligned.
- `gpu_workers` scales population evaluation by running multiple model replicas per GPU (more VRAM use).
