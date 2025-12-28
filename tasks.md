# Tasks / Notes

## Completed highlights (from experimental repo)
- [x] CUDA graphs + full-eval capture; NVTX ranges for profiling.
- [x] FP16 EFLA state, int8 residual + abs-mean norm; fused kernels to cut bandwidth.
- [x] GPU noise generation + GPU weight update (Z + RMS + shadow update).
- [x] Fused Q/K/V (+beta) GEMM; fused FFN GEMM+GELU+quantize.
- [x] Packed int2 GEMM path (BitNet-style) and packed updates.
- [x] NVFP4 block-scaled GEMM (CUTLASS SM120) + fused NVFP4 activation quantization.
- [x] NVFP4 fused QKV weights + split kernel; activation/GELU NVFP4 quantize.
- [x] Multi-GPU data-parallel eval/update; multi-worker per GPU for higher throughput.
- [x] Kernel tuning (EFLA tile sizes, GEMM kernel tweaks, vectorized layernorm/add_scaled).

## Future work / ideas
- [ ] Finish NVFP4 stream-K/split-K stabilization (some shapes still fail prepare).
- [ ] Investigate a persistent GPU cache of NVFP4 weights across workers to reduce VRAM duplication.
- [ ] Try larger hidden sizes (aligned to 128) to improve GPU occupancy and reduce launch overhead.
- [ ] Explore CUTLASS SM120 block-scaled auto-tuning with more tile+stage variants.
- [ ] Profile multi-worker scheduling vs. CUDA MPS or CUDA graphs per worker.
- [ ] Fuse remaining memory-bound steps (noise add, residual add, norm) into GEMM epilogues.
- [ ] Compare against Microsoft BitNet kernels for further GEMM/layout ideas.
