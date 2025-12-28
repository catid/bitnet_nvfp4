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

## NVFP4 throughput focus (in progress)
- [x] Fuse MLP residual add + abs-mean norm in NVFP4 path and reuse the normalized output for the next layer (skip per-layer absmean_norm_q_nvfp4). (~+4% tok/s in 5-epoch sanity run)
- [ ] Reduce NVFP4 quant/norm bandwidth (fuse activation quantize + norm where possible, avoid redundant nvfp4_quantize_* passes).
- [ ] Rework QKV split/format to reduce HBM traffic (keep qkv in registers/shared longer, cut extra writes).
- [ ] Investigate NVFP4 epilogue fusion in CUTLASS kernels to produce quantized activations directly.
- [ ] Track and remove avoidable cudaMemsetAsync/atomics around EFLA state paths when running NVFP4 configs.

## NVFP4 hotspot investigations (from nsys kernel summary)
- [ ] EFLA kS kernel (~18.8%): reduce atomics and memory traffic.
  - [x] Experiment: add no-atomic full-D path for hidden=512 (tile_d=hidden, grid_z=1) and compare (regressed avg tok/s_total ~863k vs ~925k baseline; reverted).
  - [x] Experiment: increase tile_d to 128 for hidden>=512 (reduced grid_z/atomics; avg tok/s_total ~935k vs ~925k baseline).
  - [x] Experiment: increase tile_d to 256 for hidden>=512 (avg tok/s_total ~922k; worse than tile_d=128; reverted).
  - [x] Experiment: increase tile_n to 256 with tile_d=128 (avg tok/s_total ~932k; slightly worse than tile_n=128; reverted).
  - [ ] Experiment: block-level reduction in shared memory and single write per output.
  - [ ] Experiment: use batched GEMV/GEMM (cuBLAS/CUTLASS) for kS to reduce kernel count.
- [ ] EFLA update kernel (~16.8%): improve arithmetic intensity and caching.
  - [x] Experiment: load k_usage/diff into shared per block; reuse for multiple S rows (regressed avg tok/s_total ~848k; reverted).
  - [ ] Experiment: half2/vectorized path for S updates (ensure alignment on hidden=512).
  - [x] Experiment: fuse diff + update for H=512 (default kernel, --efla_fuse_diff) (avg tok/s_total ~920k vs ~925k baseline; left off).
  - [x] Experiment: enable efla_mixed (FP16 k_usage/diff) (avg tok/s_total ~929k vs ~935k baseline; left off).
  - [x] Experiment: enable efla_update_wmma (tensor core update) (avg tok/s_total ~750k; reverted).
- [ ] EFLA out kernel (~14.7%): avoid atomics and improve memory coalescing.
  - [ ] Experiment: no-atomic full-D path (same as kS).
  - [ ] Experiment: shared reduction + one write per output.
  - [ ] Experiment: batched GEMV/GEMM alternative.
- [ ] EFLA prepare + diff (~8.6% total): reduce kernel count.
  - [ ] Experiment: fuse prepare with kS (compute q_norm/k_usage on the fly).
  - [ ] Experiment: fuse diff into update (H=512 tuned).
- [ ] NVFP4 absmean_norm_q_nvfp4 (~11.1%) + add_scaled_to_int8_absmean_norm_q_nvfp4 (~10.9%).
  - [ ] Experiment: ensure all residual paths use the fused add+norm kernel (eliminate standalone absmean_norm where possible).
  - [ ] Experiment: vectorized loads/stores (int4/half2) and larger tile sizes.
  - [ ] Experiment: reduce writes by keeping normalized output in-place when safe.
  - [x] Experiment: cache updated int8 residuals in shared memory inside add_scaled_to_int8_absmean_norm_q_nvfp4 (regressed avg tok/s_total ~907k vs ~925k baseline; reverted).
  - [x] Experiment: char4/float4 vectorized add_scaled_to_int8_absmean_norm_q_nvfp4 update (avg tok/s_total ~904k vs ~935k baseline; reverted).
- [ ] NVFP4 quantize kernels: gelu (~5.6%), activation (~1.2%).
  - [ ] Experiment: fuse GELU + quantize into GEMM epilogue (CUTLASS).
  - [x] Experiment: fuse activation quantize with embedding lookup (avg tok/s_total ~950k vs ~935k baseline; kept).
  - [ ] Experiment: fuse activation quantize with norm output.
  - [ ] Experiment: increase quant kernel tile sizes / persistent kernel.
  - [x] Experiment: switch nvfp4 quant mode to warp4 for default H=512 (regressed avg tok/s_total ~910k vs ~925k baseline).
- [ ] QKV split kernel (~4.4%): reduce extra writes.
  - [x] Experiment: direct consume fused QKV buffer in EFLA (skip split when possible; avg tok/s_total ~935k vs ~935k baseline; kept).
  - [x] Experiment: always use vectorized split (float4) when hidden%4==0. (small +0.2% tok/s in 5-epoch sanity)
  - [x] Experiment: enable qkv_split_bias for NVFP4 and measure impact (regressed avg tok/s_total ~913k vs ~925k baseline; reverted).
- [ ] add_scaled_to_int8 (~3.2%): eliminate residual pass.
  - [ ] Experiment: replace remaining calls with fused add+norm (NVFP4) or fuse into upstream kernels.
- [ ] cross_entropy_loss (~1.7%): reduce kernel/transfer overhead.
  - [ ] Experiment: NVFP4 head+loss fusion (epilogue or dedicated kernel).
  - [ ] Experiment: compute loss in FP16 to reduce bandwidth.
- [ ] embedding_lookup_noise (~1.0%) + add_pos_gelu_quantize (~0.9%).
  - [x] Experiment: fuse embedding lookup + quantize (NVFP4 path). (avg tok/s_total ~950k vs ~935k baseline; kept).
  - [ ] Experiment: fuse add_pos + GELU + quantize into a single kernel.
- [ ] CUDA API overhead (launch + memset + sync).
  - [ ] Experiment: reduce cudaMemsetAsync by initializing outputs in-kernel.
  - [ ] Experiment: remove cudaStreamSynchronize in NVFP4 paths (events instead).
  - [ ] Experiment: re-test CUDA graph capture for specific subsets only.
