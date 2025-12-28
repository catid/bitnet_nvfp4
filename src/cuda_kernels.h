#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace bitnet_cuda {

// Base ternary GEMM: out[b, i] = out_scale * sum_j x[b, j] * w[i, j]
// Layouts are row-major: x = [batch][cols], w = [rows][cols], out = [batch][rows]
void gemm_ternary(const int8_t* d_x, const int8_t* d_w,
                  int rows, int cols, int batch,
                  float out_scale,
                  float* d_out,
                  cudaStream_t stream);

// Warp-level fused GEMM + activation + quantize to int8.
// out_q[b, i] = clip_int8(lrint(act(out_scale * sum_j scale_x[j/256] * x[b,j] * w[i,j] + noise)))
// activation matches Activation enum: 0=None, 1=ReLU, 2=Tanh.
void gemm_ternary_act_quant(const int8_t* d_x, const int8_t* d_w,
                            const float* d_scale_x,
                            int rows, int cols, int batch,
                            float out_scale,
                            int activation,
                            int8_t* d_out_q,
                            cudaStream_t stream);

// Fused GEMM + GELU + quantize with optional noise add (full-matrix noise).
// out_q[b, i] = clip_int8(lrint(gelu(out_scale * (sum x*w + noise_scale*sum x*w_noise)) * act_scale))
// If d_w_noise is null or noise_scale == 0, noise term is skipped.
void gemm_ternary_gelu_quant(const int8_t* d_x,
                             const int8_t* d_w,
                             const int8_t* d_w_noise,
                             const float* d_scale_x,
                             int rows, int cols, int batch,
                             float out_scale,
                             float noise_scale,
                             float act_scale,
                             int8_t* d_out_q,
                             cudaStream_t stream);

void gemm_ternary_lora_act_quant(const int8_t* d_x, const int8_t* d_w,
                                 const float* d_scale_x,
                                 int rows, int cols, int batch,
                                 float out_scale,
                                 const int8_t* d_A,
                                 const int32_t* d_t,
                                 int rank,
                                 float lora_scale,
                                 int activation,
                                 int8_t* d_out_q,
                                 cudaStream_t stream);

void gemm_ternary_sparse_act_quant(const int8_t* d_x, const int8_t* d_w,
                                   const float* d_scale_x,
                                   int rows, int cols, int batch,
                                   float out_scale,
                                   const int32_t* d_row_offsets,
                                   const int32_t* d_col_idx,
                                   const int8_t* d_eps,
                                   float sparse_scale,
                                   int activation,
                                   int8_t* d_out_q,
                                   cudaStream_t stream);

// Warp-level fused GEMM to float (used for head logits).
void gemm_ternary_f(const int8_t* d_x, const int8_t* d_w,
                    const float* d_scale_x,
                    int rows, int cols, int batch,
                    float out_scale,
                    float* d_out,
                    cudaStream_t stream);

// Fused head GEMM + cross-entropy loss (vocab <= 256, dp4a path).
// Computes loss over a single token step and accumulates into d_loss_accum.
void head_gemm_cross_entropy(const int8_t* d_x,
                             const int8_t* d_w,
                             const int8_t* d_w_noise,
                             const float* d_scale_x,
                             int cols,
                             int vocab,
                             int batch,
                             float out_scale,
                             float noise_scale,
                             const float* d_bias,
                             const float* d_bias_noise,
                             float bias_noise_scale,
                             const uint8_t* d_targets_t,
                             int token_stride,
                             float* d_loss_accum,
                             cudaStream_t stream);

// GEMM to float with fused noise add (full-matrix noise).
// out[b, i] = out_scale * (sum_j x*w + noise_scale*sum_j x*w_noise)
void gemm_ternary_f_noise(const int8_t* d_x, const int8_t* d_w,
                          const int8_t* d_w_noise,
                          const float* d_scale_x,
                          int rows, int cols, int batch,
                          float out_scale,
                          float noise_scale,
                          float* d_out,
                          cudaStream_t stream);

// Tiled GEMM to float with fused noise add (optional), using shared memory.
// Intended for experimenting with cp.async-style tiling.
void gemm_ternary_f_noise_tiled(const int8_t* d_x, const int8_t* d_w,
                                const int8_t* d_w_noise,
                                const float* d_scale_x,
                                int rows, int cols, int batch,
                                float out_scale,
                                float noise_scale,
                                float* d_out,
                                cudaStream_t stream);

// Fused Q/K/V (+optional beta) GEMM to float with optional noise add.
// Computes Q/K/V (rows x cols) and optional beta (1 x cols) in one pass over X.
// If noise_scale == 0 or noise pointers are null, noise path is skipped.
void gemm_qkvb_fused(const int8_t* d_x,
                     const int8_t* d_wq, const int8_t* d_wk, const int8_t* d_wv,
                     const int8_t* d_wb,
                     const int8_t* d_wq_noise, const int8_t* d_wk_noise,
                     const int8_t* d_wv_noise, const int8_t* d_wb_noise,
                     const float* d_scale_x,
                     int rows, int cols, int batch,
                     float q_out_scale, float k_out_scale, float v_out_scale, float b_out_scale,
                     float noise_scale,
                     float* d_q, float* d_k, float* d_v, float* d_beta,
                     cudaStream_t stream);

// Pack ternary weights (-1,0,1) into int2 (16 values per uint32).
void pack_ternary_i2(const int8_t* d_w,
                     int rows, int cols,
                     uint32_t* d_w_packed,
                     cudaStream_t stream);

// Pack ternary weights into int2 with BitNet-style interleaving.
void pack_ternary_i2_bitnet(const int8_t* d_w,
                            int rows, int cols,
                            uint32_t* d_w_packed,
                            cudaStream_t stream);

// Initialize LUT for int2 decode (needed before CUDA graph capture).
void init_i2_lut();

// GEMM using int2-packed weights (W2A8-style decode + dp4a).
void gemm_ternary_f_i2(const int8_t* d_x, const uint32_t* d_w_packed,
                       const float* d_scale_x,
                       int rows, int cols, int batch,
                       float out_scale,
                       float* d_out,
                       cudaStream_t stream);

void gemm_ternary_f_i2_noise(const int8_t* d_x, const uint32_t* d_w_packed,
                             const uint32_t* d_w_packed_noise,
                             const float* d_scale_x,
                             int rows, int cols, int batch,
                             float out_scale,
                             float noise_scale,
                             float* d_out,
                             cudaStream_t stream);

void gemm_ternary_f_i2_bitnet(const int8_t* d_x, const uint32_t* d_w_packed,
                              const float* d_scale_x,
                              int rows, int cols, int batch,
                              float out_scale,
                              float* d_out,
                              cudaStream_t stream);

void gemm_ternary_f_i2_bitnet_noise(const int8_t* d_x, const uint32_t* d_w_packed,
                                    const uint32_t* d_w_packed_noise,
                                    const float* d_scale_x,
                                    int rows, int cols, int batch,
                                    float out_scale,
                                    float noise_scale,
                                    float* d_out,
                                    cudaStream_t stream);

// Tensor-core GEMM backend (unpacked).
void gemm_ternary_f_cutlass(const int8_t* d_x, const int8_t* d_w,
                            int rows, int cols, int batch,
                            float out_scale,
                            float* d_out,
                            cudaStream_t stream);

void gemm_ternary_lora_f(const int8_t* d_x, const int8_t* d_w,
                         const float* d_scale_x,
                         int rows, int cols, int batch,
                         float out_scale,
                         const int8_t* d_A,
                         const int32_t* d_t,
                         int rank,
                         float lora_scale,
                         float* d_out,
                         cudaStream_t stream);

void gemm_ternary_sparse_f(const int8_t* d_x, const int8_t* d_w,
                           const float* d_scale_x,
                           int rows, int cols, int batch,
                           float out_scale,
                           const int32_t* d_row_offsets,
                           const int32_t* d_col_idx,
                           const int8_t* d_eps,
                           float sparse_scale,
                           float* d_out,
                           cudaStream_t stream);

// LoRA helper: t[b, r] = sum_j x[b, j] * B[j, r]
// Layout: x = [batch][cols], B = [cols][rank], t = [batch][rank]
void lora_compute_t(const int8_t* d_x, const int8_t* d_B,
                    int cols, int batch, int rank,
                    int32_t* d_t,
                    cudaStream_t stream);

// LoRA add: out[b, i] += scale * sum_r t[b, r] * A[i, r]
// Layout: A = [rows][rank], out = [batch][rows]
void lora_add(const int32_t* d_t, const int8_t* d_A,
              int rows, int batch, int rank,
              float scale,
              float* d_out,
              cudaStream_t stream);

// Sparse add: out[b, row] += scale * sum_{k in row} x[b, col_k] * eps_k
// CSR-style sparse row encoding:
//   row_offsets: [rows+1], col_idx: [nnz], eps: [nnz]
void sparse_add(const int32_t* d_row_offsets,
                const int32_t* d_col_idx,
                const int8_t* d_eps,
                int rows, int cols, int batch,
                const int8_t* d_x,
                float scale,
                float* d_out,
                cudaStream_t stream);

// Activation + quantize to int8 with clamp to [-127,127].
// activation: 0=None, 1=ReLU, 2=Tanh (matches Activation enum values).
void activation_quantize(const float* d_in,
                         int8_t* d_out,
                         int n,
                         int activation,
                         cudaStream_t stream);

// Activation quantize to int8 and NVFP4 (row-major).
// A shape: [rows x cols], row-major. Uses SFA layout for block scaling.
void activation_quantize_nvfp4(const float* d_in,
                               int rows,
                               int cols_in,
                               int cols_out,
                               int activation,
                               int8_t* d_out_q,
                               void* d_out_nvfp4,
                               void* d_sfa,
                               cudaStream_t stream);

// Embedding lookup (+ optional noise) fused with int8 + NVFP4 quantize (row-major).
void embedding_lookup_noise_quantize_nvfp4(const uint8_t* d_tokens_t,
                                           int token_stride,
                                           const int8_t* d_emb_w,
                                           const int8_t* d_emb_noise,
                                           int hidden,
                                           int batch,
                                           float noise_scale,
                                           int anti_sign,
                                           int8_t* d_out_q,
                                           void* d_out_nvfp4,
                                           void* d_sfa,
                                           cudaStream_t stream);

// Add positional embeddings + GELU + quantize to int8 and NVFP4 (row-major).
void add_pos_gelu_quantize_nvfp4(const float* d_in,
                                 const float* d_pos_t,
                                 int hidden,
                                 int batch,
                                 float act_scale,
                                 int8_t* d_out_q,
                                 void* d_out_nvfp4,
                                 void* d_sfa,
                                 cudaStream_t stream);

// GELU + scale quantize to int8 and NVFP4 (row-major).
void gelu_quantize_nvfp4(const float* d_in,
                         int rows,
                         int cols_in,
                         int cols_out,
                         float act_scale,
                         int8_t* d_out_q,
                         void* d_out_nvfp4,
                         void* d_sfa,
                         cudaStream_t stream);

// Abs-mean norm on int8 input, then quantize to int8 + NVFP4 (row-major).
void absmean_norm_q_nvfp4(const int8_t* d_in,
                          int hidden,
                          int batch,
                          int cols_out,
                          float scale,
                          int8_t* d_out_q,
                          void* d_out_nvfp4,
                          void* d_sfa,
                          cudaStream_t stream);

// out_q = abs-mean norm of (inout += quantize_int8(in * add_scale)), plus NVFP4 output.
void add_scaled_to_int8_absmean_norm_q_nvfp4(int8_t* d_inout,
                                             const float* d_in,
                                             int hidden,
                                             int batch,
                                             int cols_out,
                                             float add_scale,
                                             float norm_scale,
                                             int8_t* d_out_q,
                                             void* d_out_nvfp4,
                                             void* d_sfa,
                                             cudaStream_t stream);

// Embedding/state update for one timestep.
// tokens_t points at tokens[t] in a [batch][T] row-major buffer; token_stride = T.
// state is [batch][hidden] int8.
void embed_update_plain(const uint8_t* d_tokens_t,
                        int token_stride,
                        const int8_t* d_emb_w,
                        int hidden,
                        int batch,
                        int8_t* d_state,
                        cudaStream_t stream);

void embed_update_lora(const uint8_t* d_tokens_t,
                       int token_stride,
                       const int8_t* d_emb_w,
                       const int8_t* d_A,
                       const int8_t* d_B,
                       int hidden,
                       int batch,
                       int rank,
                       float noise_scale,
                       int8_t* d_state,
                       cudaStream_t stream);

// d_row_noise is dense [vocab][hidden] int16 (pre-summed sparse eps per entry).
void embed_update_sparse(const uint8_t* d_tokens_t,
                         int token_stride,
                         const int8_t* d_emb_w,
                         const int16_t* d_row_noise,
                         int hidden,
                         int batch,
                         float noise_scale,
                         int8_t* d_state,
                         cudaStream_t stream);

// NVFP4 block-scaled GEMM helpers (Blackwell/CUTLASS).
// Sizes are in bytes for packed 4-bit storage.
inline size_t nvfp4_matrix_bytes(int rows, int cols) {
    const size_t elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    return (elems * 4 + 7) / 8;
}

enum class Nvfp4Schedule { Auto = 0, Cooperative = 1, Pingpong = 2 };
enum class Nvfp4QuantMode { Warp16 = 0, Warp4 = 1 };
enum class Nvfp4StageCount { Auto = 0, Stages2 = 2, Stages3 = 3, Stages4 = 4 };
enum class Nvfp4Decomposition { Heuristic = 0, DataParallel = 1, SplitK = 2, StreamK = 3 };

size_t nvfp4_sfa_bytes(int m, int n, int k);
size_t nvfp4_sfb_bytes(int m, int n, int k);
void nvfp4_init();
void nvfp4_set_schedule(Nvfp4Schedule schedule);
void nvfp4_set_quant_mode(Nvfp4QuantMode mode);
void nvfp4_set_stage_count(Nvfp4StageCount stages);
void nvfp4_set_decomposition(Nvfp4Decomposition mode);
void nvfp4_set_verbose(bool verbose);
void nvfp4_set_splits(int splits);
bool nvfp4_prepare_gemm(const void* d_a,
                        const void* d_b,
                        const void* d_sfa,
                        const void* d_sfb,
                        int m, int n, int k,
                        float alpha,
                        float beta,
                        const float* d_c,
                        float* d_d,
                        cudaStream_t stream);

// Quantize row-major int8 matrix into NVFP4 + block scale factors.
// A: shape [m x k] (row-major), uses SFA layout.
void nvfp4_quantize_a(const int8_t* d_in,
                      int m, int n, int k_in, int k_out,
                      void* d_out,
                      void* d_sfa,
                      cudaStream_t stream);

// B: shape [n x k] (row-major storage, treated as column-major KxN), uses SFB layout.
void nvfp4_quantize_b(const int8_t* d_in,
                      int m, int n, int k_in, int k_out,
                      void* d_out,
                      void* d_sfb,
                      cudaStream_t stream);

// Quantize fused Q/K/V (+beta) weights into a single NVFP4 matrix.
// Rows are stacked as [Wq; Wk; Wv; Wbeta], valid rows = [3*H + 1].
// rows_total can be >= valid rows for alignment/padding.
// SFB scales are multiplied by per-matrix output scales.
void nvfp4_quantize_wqkv_fused(const int8_t* d_wq,
                               const int8_t* d_wk,
                               const int8_t* d_wv,
                               const int8_t* d_wb,
                               int batch,
                               int hidden,
                               int k_out,
                               int rows_total,
                               float q_scale,
                               float k_scale,
                               float v_scale,
                               float b_scale,
                               void* d_out,
                               void* d_sfb,
                               cudaStream_t stream);

void split_qkvb_fused(const float* d_in,
                      int batch,
                      int hidden,
                      int fused_cols,
                      float* d_q,
                      float* d_k,
                      float* d_v,
                      float* d_beta,
                      float beta_bias,
                      bool use_vec4,
                      cudaStream_t stream);

// NVFP4 block-scaled GEMM: D = alpha * A * B + beta * C
void gemm_ternary_f_nvfp4(const void* d_a,
                          const void* d_b,
                          const void* d_sfa,
                          const void* d_sfb,
                          int m, int n, int k,
                          float alpha,
                          float beta,
                          const float* d_c,
                          float* d_d,
                          cudaStream_t stream);

// NVFP4 block-scaled GEMM with fused GELU+scale into int8 output:
// D = GELU(alpha * A * B + beta * C) * act_scale
void gemm_ternary_f_nvfp4_gelu_i8(const void* d_a,
                                  const void* d_b,
                                  const void* d_sfa,
                                  const void* d_sfb,
                                  int m, int n, int k,
                                  float alpha,
                                  float beta,
                                  const float* d_c,
                                  float act_scale,
                                  int8_t* d_d,
                                  cudaStream_t stream);

} // namespace bitnet_cuda
