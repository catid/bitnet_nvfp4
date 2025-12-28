# Summary

## Comparison: experimental vs simplified repo (NVFP4)

**Command (both builds):**
```
--epochs 1 --no_val --fixed_train --data_seed 123 --cuda_graph --graph_full_eval \
--noise_batch --gpu_update --gpu_noise --gemm_backend nvfp4 \
--pop 2048 --batch 32 --seq_len 64 --hidden 128 --layers 4 --gpu_workers 3 \
--data ../data/tinyshakespeare.txt
```

**Experimental repo (Release build, sm120a):**
- train_loss 5.1531, train_ppl 172.967
- tok/s_eval 575,991, tok/s_total 573,909

**Simplified repo (Release build, sm120a):**
- train_loss 5.1531, train_ppl 172.967
- tok/s_eval 576,208, tok/s_total 574,096

Loss curves match exactly for the 1‑epoch fixed‑batch run; throughput matches within ~0.04%.

## gpu_workers scaling (fixed batch, 2 GPUs)

All runs used:
```
--epochs 2 --no_val --fixed_train --data_seed 123 --seed 1 --devices 0,1
```

**Epoch 1 throughput (tok/s_eval / tok/s_total):**
- workers=8:   1.176e6 / 1.164e6
- workers=12:  1.594e6 / 1.562e6
- workers=16:  1.907e6 / 1.851e6
- workers=24:  2.255e6 / 2.147e6
- workers=32:  2.473e6 / 2.309e6  **(best total throughput)**
- workers=48:  2.480e6 / 2.248e6
- workers=64:  2.485e6 / 2.188e6
- workers=128: 2.511e6 / 1.968e6

Eval throughput plateaus beyond ~32 workers/GPU; update time grows and reduces total throughput.

## Notes
- NVFP4 requires sm120a; this repo forces `CMAKE_CUDA_ARCHITECTURES=120a` unless explicitly overridden.
- `gpu_workers=32` is the default in this repo.
