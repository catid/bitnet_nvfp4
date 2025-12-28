#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace efla_lm_cuda {

// out[b, j] = float(emb_w[tok, j]) + anti_sign * noise_scale * float(emb_noise[tok, j])
void embedding_lookup_noise(const uint8_t* d_tokens_t,
                            int token_stride,
                            const int8_t* d_emb_w,
                            const int8_t* d_emb_noise,
                            int hidden,
                            int batch,
                            float noise_scale,
                            int anti_sign,
                            float* d_out,
                            cudaStream_t stream);

// out_q[b, j] = quantize_int8(gelu(in[b,j] + pos_t[j])).
void add_pos_gelu_quantize(const float* d_in,
                           const float* d_pos_t,
                           int hidden,
                           int batch,
                           float act_scale,
                           int8_t* d_out_q,
                           cudaStream_t stream);

// out[b, j] = gelu(in[b,j] + pos_t[j]) * act_scale.
void add_pos_gelu(const float* d_in,
                  const float* d_pos_t,
                  int hidden,
                  int batch,
                  float act_scale,
                  float* d_out,
                  cudaStream_t stream);

// out_q[i] = quantize_int8(gelu(in[i]) * act_scale).
void gelu_quantize(const float* d_in,
                   int n,
                   float act_scale,
                   int8_t* d_out_q,
                   cudaStream_t stream);

// out[i] += scale * noise[i]
void add_scaled(float* d_out,
                const float* d_noise,
                int n,
                float scale,
                cudaStream_t stream);

// out[i] += scale * float(in_q[i])
void add_scaled_i8(float* d_out,
                   const int8_t* d_in_q,
                   int n,
                   float scale,
                   cudaStream_t stream);

// out[i] = clip_int8(out[i] + quantize_int8(in[i] * scale))
void add_scaled_to_int8(int8_t* d_out,
                        const float* d_in,
                        int n,
                        float scale,
                        cudaStream_t stream);

// out_q = abs-mean norm of (out += quantize_int8(in * add_scale)).
void add_scaled_to_int8_absmean_norm_q(int8_t* d_inout,
                                       const float* d_in,
                                       int hidden,
                                       int batch,
                                       float add_scale,
                                       float norm_scale,
                                       int8_t* d_out_q,
                                       cudaStream_t stream);

// Per-vector LayerNorm over `hidden`, then quantize to int8.
void layernorm_quantize(const float* d_in,
                        int hidden,
                        int batch,
                        float scale,
                        float eps,
                        int8_t* d_out_q,
                        cudaStream_t stream);

// Abs-mean normalize int8 input, then quantize back to int8.
// y = (x * scale) / max(1, mean(|x|))
void absmean_norm_q(const int8_t* d_in,
                    int hidden,
                    int batch,
                    float scale,
                    int8_t* d_out_q,
                    cudaStream_t stream);

// Initialize shadow weights from int8 weights.
void init_shadow_from_weights(const int8_t* d_w,
                              float* d_shadow,
                              int n,
                              cudaStream_t stream);

// Fill pair seeds: seed_p = mix(base_seed, mix(epoch, p)).
void fill_pair_seeds(uint64_t base_seed,
                     uint64_t epoch,
                     int pairs,
                     uint64_t* d_out,
                     cudaStream_t stream);

// Compute Z[idx] = sum_p pair_weights[p] * noise_hash(mix(pair_seed[p], salt), idx)
// and accumulate sumsq(Z) into d_sumsq (single float).
void compute_z_sumsq(const float* d_pair_weights,
                     const uint64_t* d_pair_seeds,
                     uint64_t salt,
                     int pairs,
                     int n,
                     bool use_clt,
                     int clt_k,
                     float* d_Z,
                     float* d_sumsq,
                     cudaStream_t stream);

// Update RMS EMA and compute inv_rms (stored in d_inv_rms).
void update_inv_rms(const float* d_sumsq,
                    float* d_rms_ema,
                    bool use_adaptive,
                    float adaptive_beta,
                    float adaptive_eps,
                    int n,
                    float* d_inv_rms,
                    cudaStream_t stream);

// Apply shadow update: shadow += lr * inv_rms * Z, then ternary-clip into d_w.
void update_shadow_ternary(int8_t* d_w,
                           float* d_shadow,
                           const float* d_Z,
                           int n,
                           float inv_rms,
                           float lr,
                           float thresh,
                           cudaStream_t stream);

// Apply shadow update with inv_rms stored on device.
void update_shadow_ternary_device(int8_t* d_w,
                                  float* d_shadow,
                                  const float* d_Z,
                                  int n,
                                  const float* d_inv_rms,
                                  float lr,
                                  float thresh,
                                  cudaStream_t stream);

// Apply shadow update and repack weights into int2 (16 weights per uint32).
void update_shadow_ternary_device_packed(int8_t* d_w,
                                         uint32_t* d_w_packed,
                                         float* d_shadow,
                                         const float* d_Z,
                                         int n,
                                         const float* d_inv_rms,
                                         float lr,
                                         float thresh,
                                         cudaStream_t stream);

// BitNet-style interleaved packing variant.
void update_shadow_ternary_device_packed_bitnet(int8_t* d_w,
                                                uint32_t* d_w_packed,
                                                float* d_shadow,
                                                const float* d_Z,
                                                int n,
                                                const float* d_inv_rms,
                                                float lr,
                                                float thresh,
                                                cudaStream_t stream);

// One recurrent linear-attention step.
// Updates S in-place and outputs y[b, n] = sum_d S[b, d, n] * normalize(q)[b, d].
// method: 0 = DeltaNet, 1 = EFLA.
void efla_step(float* d_S,
               const float* d_q,
               const float* d_k,
               const float* d_v,
               const float* d_beta_raw,
               float beta_bias,
               int hidden,
               int batch,
               int method,
               float eps,
               bool use_fuse_diff,
               float* d_out,
               float* d_q_norm,
               float* d_k_usage,
               float* d_kS,
               float* d_diff,
               float* d_alpha,
               cudaStream_t stream);

void efla_step_half(__half* d_S,
                    const float* d_q,
                    const float* d_k,
                    const float* d_v,
                    const float* d_beta_raw,
                    float beta_bias,
                    int hidden,
                    int batch,
                    int method,
                    float eps,
                    bool use_wmma_update,
                    bool use_mixed_kd,
                    bool use_fuse_diff,
                    float* d_out,
                    float* d_q_norm,
                    float* d_k_usage,
                    __half* d_k_usage_h,
                    float* d_kS,
                    float* d_diff,
                    __half* d_diff_h,
                    float* d_alpha,
                    cudaStream_t stream);

// EFLA step for fused QKV layout (row-major qkv with stride fused_cols).
// Only supports the standard (non-fused-diff, non-mixed, non-wmma) path.
void efla_step_qkv_fused(float* d_S,
                         const float* d_qkv,
                         int fused_cols,
                         float beta_bias,
                         int hidden,
                         int batch,
                         int method,
                         float eps,
                         float* d_out,
                         float* d_q_norm,
                         float* d_k_usage,
                         float* d_kS,
                         float* d_diff,
                         float* d_alpha,
                         cudaStream_t stream);

void efla_step_half_qkv_fused(__half* d_S,
                              const float* d_qkv,
                              int fused_cols,
                              float beta_bias,
                              int hidden,
                              int batch,
                              int method,
                              float eps,
                              float* d_out,
                              float* d_q_norm,
                              float* d_k_usage,
                              float* d_kS,
                              float* d_diff,
                              float* d_alpha,
                              cudaStream_t stream);

// One-kernel fused EFLA step (uses shared memory, no scratch buffers).
void efla_step_fused(float* d_S,
                     const float* d_q,
                     const float* d_k,
                     const float* d_v,
                     const float* d_beta_raw,
                     float beta_bias,
                     int hidden,
                     int batch,
                     int method,
                     float eps,
                     float* d_out,
                     cudaStream_t stream);

void efla_step_fused_half(__half* d_S,
                          const float* d_q,
                          const float* d_k,
                          const float* d_v,
                          const float* d_beta_raw,
                          float beta_bias,
                          int hidden,
                          int batch,
                          int method,
                          float eps,
                          float* d_out,
                          cudaStream_t stream);


// Low-rank EFLA state step.
// K/V have shape [batch, state_dim, hidden].
void efla_step_lowrank(float* d_K,
                       float* d_V,
                       const float* d_q,
                       const float* d_k,
                       const float* d_v,
                       const float* d_gate_raw,
                       float gate_bias,
                       float gate_scale,
                       float state_decay,
                       int hidden,
                       int state_dim,
                       int batch,
                       float eps,
                       float* d_out,
                       cudaStream_t stream);

// Accumulate cross-entropy loss for a single token step over the batch.
// logits: [batch, vocab], targets_t: targets for this time step (stride = token_stride).
void cross_entropy_loss(const float* d_logits,
                        const float* d_bias,
                        const float* d_bias_noise,
                        float bias_noise_scale,
                        const uint8_t* d_targets_t,
                        int token_stride,
                        int vocab,
                        int batch,
                        float* d_loss_accum,
                        cudaStream_t stream);

// Fill int8 ternary noise with deterministic hash (anti_sign=1).
void fill_ternary_noise(int8_t* d_out,
                        size_t n,
                        uint64_t seed,
                        bool use_clt,
                        int clt_k,
                        cudaStream_t stream);

struct NoiseDesc {
    int8_t* out = nullptr;
    size_t n = 0;
    uint64_t seed = 0;
};

// Fill multiple ternary noise buffers in a single launch (2D grid over buffers x blocks).
void fill_ternary_noise_batched(NoiseDesc* d_descs,
                                int num_desc,
                                int max_blocks,
                                bool use_clt,
                                int clt_k,
                                cudaStream_t stream);

// Fill float noise vector (ternary values cast to float).
void fill_float_noise(float* d_out,
                      int n,
                      uint64_t seed,
                      bool use_clt,
                      int clt_k,
                      cudaStream_t stream);

} // namespace efla_lm_cuda
