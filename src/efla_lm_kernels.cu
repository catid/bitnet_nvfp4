#include "efla_lm_kernels.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>

namespace {

__device__ __forceinline__ int8_t clip_int8(int v) {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return static_cast<int8_t>(v);
}

__device__ __forceinline__ int8_t clip_ternary(int v) {
    if (v > 1) return 1;
    if (v < -1) return -1;
    return static_cast<int8_t>(v);
}

__device__ __forceinline__ int lrint_to_int(float x) {
    return __float2int_rn(x);
}

__device__ __forceinline__ float gelu(float x) {
    // tanh approximation (same as common PyTorch fast-gelu formulation).
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline void select_efla_tiles(int hidden, int* tile_n, int* tile_d) {
    if (hidden >= 512) {
        *tile_n = 128;
        *tile_d = 128;
    } else {
        *tile_n = 64;
        *tile_d = 32;
    }
}

__device__ __forceinline__ uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t mix_u64(uint64_t a, uint64_t b) {
    uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ __forceinline__ int8_t ternary_from_u64(uint64_t x) {
    uint8_t r = static_cast<uint8_t>(x & 0xFFu);
    if (r < 85) return -1;
    if (r < 170) return 0;
    return 1;
}

__device__ __forceinline__ int8_t noise_hash_u64(uint64_t seed, uint64_t idx, bool use_clt, int clt_k) {
    if (!use_clt) {
        const uint64_t v = mix_u64(seed, idx);
        return ternary_from_u64(v);
    }
    int k = (clt_k > 0) ? clt_k : 1;
    uint64_t st = mix_u64(seed, idx);
    int sum = 0;
    for (int i = 0; i < k; ++i) {
        uint64_t v = splitmix64_next(st);
        sum += static_cast<int>(ternary_from_u64(v));
    }
    return static_cast<int8_t>(sum);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void embedding_lookup_noise_kernel(const uint8_t* __restrict__ tokens_t,
                                              int token_stride,
                                              const int8_t* __restrict__ emb_w,
                                              const int8_t* __restrict__ emb_noise,
                                              int hidden,
                                              int batch,
                                              float noise_scale,
                                              int anti_sign,
                                              float* __restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= hidden || b >= batch) return;

    const int tok = static_cast<int>(tokens_t[b * token_stride]);
    const int idx = b * hidden + j;
    float v = static_cast<float>(emb_w[tok * hidden + j]);
    if (emb_noise && noise_scale != 0.0f && anti_sign != 0) {
        v += static_cast<float>(anti_sign) * noise_scale * static_cast<float>(emb_noise[tok * hidden + j]);
    }
    out[idx] = v;
}

__global__ void add_pos_gelu_quantize_kernel(const float* __restrict__ in,
                                            const float* __restrict__ pos_t,
                                            int hidden,
                                            int batch,
                                            float act_scale,
                                            int8_t* __restrict__ out_q) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= hidden || b >= batch) return;

    const int idx = b * hidden + j;
    float v = in[idx] + pos_t[j];
    v = gelu(v);
    v *= act_scale;
    out_q[idx] = clip_int8(lrint_to_int(v));
}

__global__ void add_pos_gelu_kernel(const float* __restrict__ in,
                                   const float* __restrict__ pos_t,
                                   int hidden,
                                   int batch,
                                   float act_scale,
                                   float* __restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= hidden || b >= batch) return;

    const int idx = b * hidden + j;
    float v = in[idx] + pos_t[j];
    v = gelu(v);
    out[idx] = v * act_scale;
}

__global__ void gelu_quantize_kernel(const float* __restrict__ in,
                                    int n,
                                    float act_scale,
                                    int8_t* __restrict__ out_q) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = gelu(in[i]) * act_scale;
    out_q[i] = clip_int8(lrint_to_int(v));
}

__global__ void add_scaled_kernel(float* __restrict__ out,
                                  const float* __restrict__ noise,
                                  int n,
                                  float scale) {
    const int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx4 * 4;
    if (i >= n) return;
    if (i + 3 < n) {
        float4 o = reinterpret_cast<const float4*>(out)[idx4];
        float4 n4 = reinterpret_cast<const float4*>(noise)[idx4];
        o.x += scale * n4.x;
        o.y += scale * n4.y;
        o.z += scale * n4.z;
        o.w += scale * n4.w;
        reinterpret_cast<float4*>(out)[idx4] = o;
    } else {
        for (int j = i; j < n && j < i + 4; ++j) {
            out[j] += scale * noise[j];
        }
    }
}

__global__ void add_scaled_i8_kernel(float* __restrict__ out,
                                     const int8_t* __restrict__ in_q,
                                     int n,
                                     float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += scale * static_cast<float>(in_q[i]);
}

__global__ void add_scaled_to_int8_kernel(int8_t* __restrict__ out,
                                          const float* __restrict__ in,
                                          int n,
                                          float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i] * scale;
    int q = lrint_to_int(v);
    int sum = static_cast<int>(out[i]) + q;
    out[i] = clip_int8(sum);
}

__global__ void add_scaled_to_int8_absmean_norm_q_kernel(int8_t* __restrict__ inout,
                                                         const float* __restrict__ in,
                                                         int hidden,
                                                         float add_scale,
                                                         float norm_scale,
                                                         int8_t* __restrict__ out_q) {
    const int b = blockIdx.x;
    const int base = b * hidden;

    float sum_abs = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        const int idx = base + j;
        int q = lrint_to_int(in[idx] * add_scale);
        int sum = static_cast<int>(inout[idx]) + q;
        int8_t updated = clip_int8(sum);
        inout[idx] = updated;
        sum_abs += fabsf(static_cast<float>(updated));
    }

    sum_abs = warp_sum(sum_abs);
    __shared__ float sh_sum[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sh_sum[warp] = sum_abs;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp == 0) {
        total_sum = (threadIdx.x < (blockDim.x >> 5)) ? sh_sum[threadIdx.x] : 0.0f;
        total_sum = warp_sum(total_sum);
    }

    __shared__ float inv_abs_mean;
    if (threadIdx.x == 0) {
        float abs_mean = total_sum / static_cast<float>(hidden);
        if (abs_mean < 1.0f) abs_mean = 1.0f;
        inv_abs_mean = norm_scale / abs_mean;
    }
    __syncthreads();

    const float inv = inv_abs_mean;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        const int idx = base + j;
        float v = static_cast<float>(inout[idx]) * inv;
        out_q[idx] = clip_int8(lrint_to_int(v));
    }
}

__global__ void cross_entropy_loss_kernel(const float* __restrict__ logits,
                                          const float* __restrict__ bias,
                                          const float* __restrict__ bias_noise,
                                          float bias_noise_scale,
                                          const uint8_t* __restrict__ targets_t,
                                          int token_stride,
                                          int vocab,
                                          float* __restrict__ loss_accum) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = b * vocab;

    extern __shared__ float shmem[];
    float* shmax = shmem;
    float* shsum = shmem + blockDim.x;

    float local_max = -INFINITY;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = logits[offset + i];
        if (bias) v += bias[i];
        if (bias_noise) v += bias_noise_scale * bias_noise[i];
        local_max = fmaxf(local_max, v);
    }
    shmax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmax[tid] = fmaxf(shmax[tid], shmax[tid + stride]);
        }
        __syncthreads();
    }

    const float maxv = shmax[0];
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = logits[offset + i];
        if (bias) v += bias[i];
        if (bias_noise) v += bias_noise_scale * bias_noise[i];
        local_sum += expf(v - maxv);
    }
    shsum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shsum[tid] += shsum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int y = static_cast<int>(targets_t[b * token_stride]);
        float vy = logits[offset + y];
        if (bias) vy += bias[y];
        if (bias_noise) vy += bias_noise_scale * bias_noise[y];
        const float logprob = vy - maxv - logf(shsum[0] + 1e-9f);
        atomicAdd(loss_accum, -logprob);
    }
}

__global__ void absmean_norm_q_kernel(const int8_t* __restrict__ in,
                                      int hidden,
                                      float scale,
                                      int8_t* __restrict__ out_q) {
    const int b = blockIdx.x;
    float sum_abs = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        sum_abs += fabsf(static_cast<float>(in[b * hidden + j]));
    }

    sum_abs = warp_sum(sum_abs);
    __shared__ float sh_sum[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sh_sum[warp] = sum_abs;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp == 0) {
        total_sum = (threadIdx.x < (blockDim.x >> 5)) ? sh_sum[threadIdx.x] : 0.0f;
        total_sum = warp_sum(total_sum);
    }

    __shared__ float inv_abs_mean;
    if (threadIdx.x == 0) {
        float abs_mean = total_sum / static_cast<float>(hidden);
        if (abs_mean < 1.0f) abs_mean = 1.0f;
        inv_abs_mean = scale / abs_mean;
    }
    __syncthreads();

    const float inv = inv_abs_mean;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = static_cast<float>(in[b * hidden + j]) * inv;
        out_q[b * hidden + j] = clip_int8(lrint_to_int(v));
    }
}

__global__ void fill_ternary_noise_kernel(int8_t* __restrict__ out,
                                          size_t n,
                                          uint64_t seed,
                                          int use_clt,
                                          int clt_k) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = noise_hash_u64(seed, static_cast<uint64_t>(idx), use_clt != 0, clt_k);
}

__global__ void fill_ternary_noise_batched_kernel(const efla_lm_cuda::NoiseDesc* __restrict__ descs,
                                                  int num_desc,
                                                  int max_blocks,
                                                  int use_clt,
                                                  int clt_k) {
    const int desc_idx = static_cast<int>(blockIdx.y);
    const int block_idx = static_cast<int>(blockIdx.x);
    if (desc_idx >= num_desc || block_idx >= max_blocks) return;
    const efla_lm_cuda::NoiseDesc desc = descs[desc_idx];
    const size_t idx = static_cast<size_t>(block_idx) * blockDim.x + threadIdx.x;
    if (idx >= desc.n) return;
    desc.out[idx] = noise_hash_u64(desc.seed, static_cast<uint64_t>(idx), use_clt != 0, clt_k);
}

__global__ void fill_float_noise_kernel(float* __restrict__ out,
                                        int n,
                                        uint64_t seed,
                                        int use_clt,
                                        int clt_k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int8_t v = noise_hash_u64(seed, static_cast<uint64_t>(idx), use_clt != 0, clt_k);
    out[idx] = static_cast<float>(v);
}

__global__ void layernorm_quantize_kernel(const float* __restrict__ in,
                                         int hidden,
                                         float scale,
                                         float eps,
                                         int8_t* __restrict__ out_q) {
    const int b = blockIdx.x;

    float sum = 0.0f;
    float sumsq = 0.0f;
    const int base = b * hidden;
    if ((hidden & 3) == 0) {
        int j = threadIdx.x * 4;
        const int stride = blockDim.x * 4;
        for (; j + 3 < hidden; j += stride) {
            const float4 v = *reinterpret_cast<const float4*>(in + base + j);
            sum += v.x + v.y + v.z + v.w;
            sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            float v = in[base + j];
            sum += v;
            sumsq += v * v;
        }
    }

    sum = warp_sum(sum);
    sumsq = warp_sum(sumsq);

    __shared__ float sh_sum[8];
    __shared__ float sh_sumsq[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sh_sum[warp] = sum;
        sh_sumsq[warp] = sumsq;
    }
    __syncthreads();

    float total_sum = 0.0f;
    float total_sumsq = 0.0f;
    if (warp == 0) {
        total_sum = (threadIdx.x < (blockDim.x >> 5)) ? sh_sum[threadIdx.x] : 0.0f;
        total_sumsq = (threadIdx.x < (blockDim.x >> 5)) ? sh_sumsq[threadIdx.x] : 0.0f;
        total_sum = warp_sum(total_sum);
        total_sumsq = warp_sum(total_sumsq);
    }

    __shared__ float sh_mean;
    __shared__ float sh_invstd;
    if (threadIdx.x == 0) {
        float mean = total_sum / static_cast<float>(hidden);
        float var = total_sumsq / static_cast<float>(hidden) - mean * mean;
        if (var < 0.0f) var = 0.0f;
        sh_mean = mean;
        sh_invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = sh_mean;
    const float invstd = sh_invstd;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = in[b * hidden + j];
        float y = (v - mean) * invstd * scale;
        out_q[b * hidden + j] = clip_int8(lrint_to_int(y));
    }
}

__global__ void efla_step_kernel(float* __restrict__ S,
                                const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                const float* __restrict__ beta_raw,
                                float beta_bias,
                                int hidden,
                                int method,
                                float eps,
                                float* __restrict__ out) {
    const int b = blockIdx.x;
    float* Sb = S + static_cast<size_t>(b) * hidden * hidden;

    extern __shared__ float sh[];
    float* q_sh = sh;
    float* k_sh = q_sh + hidden;
    float* v_sh = k_sh + hidden;
    float* k_usage = v_sh + hidden;
    float* kS = k_usage + hidden;
    float* diff = kS + hidden;

    // Load q/k/v.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] = q[b * hidden + j];
        k_sh[j] = k[b * hidden + j];
        v_sh[j] = v[b * hidden + j];
    }
    __syncthreads();

    // Normalize q.
    float q_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = q_sh[j];
        q_ss += x * x;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    q_ss = warp_sum(q_ss);
    __shared__ float sh_qss[8];
    if (lane == 0) sh_qss[warp] = q_ss;
    __syncthreads();

    float q_total = 0.0f;
    if (warp == 0) {
        q_total = (lane < (blockDim.x >> 5)) ? sh_qss[lane] : 0.0f;
        q_total = warp_sum(q_total);
    }

    __shared__ float q_inv;
    if (threadIdx.x == 0) q_inv = rsqrtf(q_total + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] *= q_inv;
    }
    __syncthreads();

    // Beta (sigmoid of linear projection + bias).
    __shared__ float beta;
    if (threadIdx.x == 0) beta = sigmoid(beta_raw[b] + beta_bias);
    __syncthreads();

    // k usage and alpha.
    float k_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = k_sh[j];
        k_ss += x * x;
    }
    k_ss = warp_sum(k_ss);
    __shared__ float sh_kss[8];
    if (lane == 0) sh_kss[warp] = k_ss;
    __syncthreads();

    float k_total = 0.0f;
    if (warp == 0) {
        k_total = (lane < (blockDim.x >> 5)) ? sh_kss[lane] : 0.0f;
        k_total = warp_sum(k_total);
    }

    __shared__ float k_norm_sum;
    if (threadIdx.x == 0) k_norm_sum = (k_total < eps) ? eps : k_total;
    __syncthreads();

    __shared__ float alpha;
    if (threadIdx.x == 0) {
        if (method == 0) { // DeltaNet
            alpha = beta;
        } else { // EFLA
            const float lambda = k_norm_sum;
            const float tmp = beta * lambda;
            alpha = -expm1f(-tmp) / (lambda + eps);
        }
    }
    __syncthreads();

    if (method == 0) { // DeltaNet uses normalized k.
        const float inv = rsqrtf(k_norm_sum + eps);
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[j] = k_sh[j] * inv;
        }
    } else {
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[j] = k_sh[j];
        }
    }
    __syncthreads();

    // kS[n] = sum_d k_usage[d] * S[d,n].
    for (int n = threadIdx.x; n < hidden; n += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            acc += k_usage[d] * Sb[d * hidden + n];
        }
        kS[n] = acc;
        diff[n] = v_sh[n] - acc;
    }
    __syncthreads();

    // S[d,n] += alpha * k_usage[d] * diff[n].
    const float a = alpha;
    for (int idx = threadIdx.x; idx < hidden * hidden; idx += blockDim.x) {
        const int d = idx / hidden;
        const int n = idx - d * hidden;
        Sb[idx] += a * k_usage[d] * diff[n];
    }
    __syncthreads();

    // out[n] = sum_d S[d,n] * q_sh[d]
    for (int n = threadIdx.x; n < hidden; n += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            acc += Sb[d * hidden + n] * q_sh[d];
        }
        out[b * hidden + n] = acc;
    }
}

__global__ void efla_step_kernel_half(__half* __restrict__ S,
                                     const float* __restrict__ q,
                                     const float* __restrict__ k,
                                     const float* __restrict__ v,
                                     const float* __restrict__ beta_raw,
                                     float beta_bias,
                                     int hidden,
                                     int method,
                                     float eps,
                                     float* __restrict__ out) {
    const int b = blockIdx.x;
    __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;

    extern __shared__ float sh[];
    float* q_sh = sh;
    float* k_sh = q_sh + hidden;
    float* v_sh = k_sh + hidden;
    float* k_usage = v_sh + hidden;
    float* kS = k_usage + hidden;
    float* diff = kS + hidden;

    // Load q/k/v.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] = q[b * hidden + j];
        k_sh[j] = k[b * hidden + j];
        v_sh[j] = v[b * hidden + j];
    }
    __syncthreads();

    // Normalize q.
    float q_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = q_sh[j];
        q_ss += x * x;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    q_ss = warp_sum(q_ss);
    __shared__ float sh_qss[8];
    if (lane == 0) sh_qss[warp] = q_ss;
    __syncthreads();

    float q_total = 0.0f;
    if (warp == 0) {
        q_total = (lane < (blockDim.x >> 5)) ? sh_qss[lane] : 0.0f;
        q_total = warp_sum(q_total);
    }

    __shared__ float q_inv;
    if (threadIdx.x == 0) q_inv = rsqrtf(q_total + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] *= q_inv;
    }
    __syncthreads();

    // Beta (sigmoid of linear projection + bias).
    __shared__ float beta;
    if (threadIdx.x == 0) beta = sigmoid(beta_raw[b] + beta_bias);
    __syncthreads();

    // k usage and alpha.
    float k_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = k_sh[j];
        k_ss += x * x;
    }
    k_ss = warp_sum(k_ss);
    __shared__ float sh_kss[8];
    if (lane == 0) sh_kss[warp] = k_ss;
    __syncthreads();

    float k_total = 0.0f;
    if (warp == 0) {
        k_total = (lane < (blockDim.x >> 5)) ? sh_kss[lane] : 0.0f;
        k_total = warp_sum(k_total);
    }

    __shared__ float k_norm_sum;
    if (threadIdx.x == 0) k_norm_sum = (k_total < eps) ? eps : k_total;
    __syncthreads();

    __shared__ float alpha;
    if (threadIdx.x == 0) {
        if (method == 0) { // DeltaNet
            alpha = beta;
        } else { // EFLA
            const float lambda = k_norm_sum;
            const float tmp = beta * lambda;
            alpha = -expm1f(-tmp) / (lambda + eps);
        }
    }
    __syncthreads();

    if (method == 0) { // DeltaNet uses normalized k.
        const float inv = rsqrtf(k_norm_sum + eps);
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[j] = k_sh[j] * inv;
        }
    } else {
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[j] = k_sh[j];
        }
    }
    __syncthreads();

    // kS[n] = sum_d k_usage[d] * S[d,n].
    for (int n = threadIdx.x; n < hidden; n += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            acc += k_usage[d] * __half2float(Sb[d * hidden + n]);
        }
        kS[n] = acc;
        diff[n] = v_sh[n] - acc;
    }
    __syncthreads();

    // S[d,n] += alpha * k_usage[d] * diff[n].
    const float a = alpha;
    for (int idx = threadIdx.x; idx < hidden * hidden; idx += blockDim.x) {
        const int d = idx / hidden;
        const int n = idx - d * hidden;
        float s = __half2float(Sb[idx]);
        s += a * k_usage[d] * diff[n];
        Sb[idx] = __float2half_rn(s);
    }
    __syncthreads();

    // out[n] = sum_d S[d,n] * q_sh[d]
    for (int n = threadIdx.x; n < hidden; n += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            acc += __half2float(Sb[d * hidden + n]) * q_sh[d];
        }
        out[b * hidden + n] = acc;
    }
}

__global__ void efla_prepare_kernel(const float* __restrict__ q,
                                    const float* __restrict__ k,
                                    const float* __restrict__ beta_raw,
                                    float beta_bias,
                                    int hidden,
                                    int method,
                                    float eps,
                                    float* __restrict__ q_norm,
                                    float* __restrict__ k_usage,
                                    float* __restrict__ alpha_out) {
    const int b = blockIdx.x;

    extern __shared__ float sh[];
    float* q_sh = sh;
    float* k_sh = q_sh + hidden;

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] = q[b * hidden + j];
        k_sh[j] = k[b * hidden + j];
    }
    __syncthreads();

    float q_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = q_sh[j];
        q_ss += x * x;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    q_ss = warp_sum(q_ss);
    __shared__ float sh_qss[8];
    if (lane == 0) sh_qss[warp] = q_ss;
    __syncthreads();

    float q_total = 0.0f;
    if (warp == 0) {
        q_total = (lane < (blockDim.x >> 5)) ? sh_qss[lane] : 0.0f;
        q_total = warp_sum(q_total);
    }

    __shared__ float q_inv;
    if (threadIdx.x == 0) q_inv = rsqrtf(q_total + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_norm[b * hidden + j] = q_sh[j] * q_inv;
    }

    float k_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = k_sh[j];
        k_ss += x * x;
    }
    k_ss = warp_sum(k_ss);
    __shared__ float sh_kss[8];
    if (lane == 0) sh_kss[warp] = k_ss;
    __syncthreads();

    float k_total = 0.0f;
    if (warp == 0) {
        k_total = (lane < (blockDim.x >> 5)) ? sh_kss[lane] : 0.0f;
        k_total = warp_sum(k_total);
    }

    __shared__ float k_norm_sum;
    if (threadIdx.x == 0) k_norm_sum = (k_total < eps) ? eps : k_total;
    __syncthreads();

    __shared__ float alpha;
    if (threadIdx.x == 0) {
        const float beta = sigmoid(beta_raw[b] + beta_bias);
        if (method == 0) { // DeltaNet
            alpha = beta;
        } else { // EFLA
            const float lambda = k_norm_sum;
            const float tmp = beta * lambda;
            alpha = -expm1f(-tmp) / (lambda + eps);
        }
        alpha_out[b] = alpha;
    }
    __syncthreads();

    if (method == 0) { // DeltaNet uses normalized k.
        const float inv = rsqrtf(k_norm_sum + eps);
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[b * hidden + j] = k_sh[j] * inv;
        }
    } else {
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[b * hidden + j] = k_sh[j];
        }
    }
}

__global__ void efla_prepare_kernel_fused(const float* __restrict__ qkv,
                                          int fused_cols,
                                          float beta_bias,
                                          int hidden,
                                          int method,
                                          float eps,
                                          float* __restrict__ q_norm,
                                          float* __restrict__ k_usage,
                                          float* __restrict__ alpha_out) {
    const int b = blockIdx.x;
    const float* qkv_row = qkv + static_cast<size_t>(b) * fused_cols;

    extern __shared__ float sh[];
    float* q_sh = sh;
    float* k_sh = q_sh + hidden;

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] = qkv_row[j];
        k_sh[j] = qkv_row[hidden + j];
    }
    __syncthreads();

    float q_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = q_sh[j];
        q_ss += x * x;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    q_ss = warp_sum(q_ss);
    __shared__ float sh_qss[8];
    if (lane == 0) sh_qss[warp] = q_ss;
    __syncthreads();

    float q_total = 0.0f;
    if (warp == 0) {
        q_total = (lane < (blockDim.x >> 5)) ? sh_qss[lane] : 0.0f;
        q_total = warp_sum(q_total);
    }

    __shared__ float q_inv;
    if (threadIdx.x == 0) q_inv = rsqrtf(q_total + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_norm[b * hidden + j] = q_sh[j] * q_inv;
    }

    float k_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = k_sh[j];
        k_ss += x * x;
    }
    k_ss = warp_sum(k_ss);
    __shared__ float sh_kss[8];
    if (lane == 0) sh_kss[warp] = k_ss;
    __syncthreads();

    float k_total = 0.0f;
    if (warp == 0) {
        k_total = (lane < (blockDim.x >> 5)) ? sh_kss[lane] : 0.0f;
        k_total = warp_sum(k_total);
    }

    __shared__ float k_norm_sum;
    if (threadIdx.x == 0) k_norm_sum = (k_total < eps) ? eps : k_total;
    __syncthreads();

    __shared__ float alpha;
    if (threadIdx.x == 0) {
        const float beta_raw = qkv_row[hidden * 3];
        const float beta = sigmoid(beta_raw + beta_bias);
        if (method == 0) { // DeltaNet
            alpha = beta;
        } else { // EFLA
            const float lambda = k_norm_sum;
            const float tmp = beta * lambda;
            alpha = -expm1f(-tmp) / (lambda + eps);
        }
        alpha_out[b] = alpha;
    }
    __syncthreads();

    if (method == 0) { // DeltaNet uses normalized k.
        const float inv = rsqrtf(k_norm_sum + eps);
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[b * hidden + j] = k_sh[j] * inv;
        }
    } else {
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[b * hidden + j] = k_sh[j];
        }
    }
}

__global__ void efla_prepare_kernel_halfk(const float* __restrict__ q,
                                          const float* __restrict__ k,
                                          const float* __restrict__ beta_raw,
                                          float beta_bias,
                                          int hidden,
                                          int method,
                                          float eps,
                                          float* __restrict__ q_norm,
                                          __half* __restrict__ k_usage,
                                          float* __restrict__ alpha_out) {
    const int b = blockIdx.x;

    extern __shared__ float sh[];
    float* q_sh = sh;
    float* k_sh = q_sh + hidden;

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] = q[b * hidden + j];
        k_sh[j] = k[b * hidden + j];
    }
    __syncthreads();

    float q_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = q_sh[j];
        q_ss += x * x;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    q_ss = warp_sum(q_ss);
    __shared__ float sh_qss[8];
    if (lane == 0) sh_qss[warp] = q_ss;
    __syncthreads();

    float q_total = 0.0f;
    if (warp == 0) {
        q_total = (lane < (blockDim.x >> 5)) ? sh_qss[lane] : 0.0f;
        q_total = warp_sum(q_total);
    }

    __shared__ float q_inv;
    if (threadIdx.x == 0) q_inv = rsqrtf(q_total + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_norm[b * hidden + j] = q_sh[j] * q_inv;
    }

    float k_ss = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float x = k_sh[j];
        k_ss += x * x;
    }
    k_ss = warp_sum(k_ss);
    __shared__ float sh_kss[8];
    if (lane == 0) sh_kss[warp] = k_ss;
    __syncthreads();

    float k_total = 0.0f;
    if (warp == 0) {
        k_total = (lane < (blockDim.x >> 5)) ? sh_kss[lane] : 0.0f;
        k_total = warp_sum(k_total);
    }

    __shared__ float k_norm_sum;
    if (threadIdx.x == 0) k_norm_sum = (k_total < eps) ? eps : k_total;
    __syncthreads();

    __shared__ float alpha;
    if (threadIdx.x == 0) {
        const float beta = sigmoid(beta_raw[b] + beta_bias);
        if (method == 0) { // DeltaNet
            alpha = beta;
        } else { // EFLA
            const float lambda = k_norm_sum;
            const float tmp = beta * lambda;
            alpha = -expm1f(-tmp) / (lambda + eps);
        }
        alpha_out[b] = alpha;
    }
    __syncthreads();

    if (method == 0) { // DeltaNet uses normalized k.
        const float inv = rsqrtf(k_norm_sum + eps);
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[b * hidden + j] = __float2half_rn(k_sh[j] * inv);
        }
    } else {
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            k_usage[b * hidden + j] = __float2half_rn(k_sh[j]);
        }
    }
}

__global__ void efla_kS_kernel(const float* __restrict__ S,
                               const float* __restrict__ k_usage,
                               int hidden,
                               int tile_d,
                               float* __restrict__ kS) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x;
    const int n = blockIdx.y * tile_n + threadIdx.x;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const float* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* kb = k_usage + b * hidden;
    float acc = 0.0f;
    for (int d = d0; d < d1; ++d) {
        acc += kb[d] * Sb[d * hidden + n];
    }
    atomicAdd(&kS[b * hidden + n], acc);
}

__global__ void efla_kS_kernel_half(const __half* __restrict__ S,
                                    const float* __restrict__ k_usage,
                                    int hidden,
                                    int tile_d,
                                    float* __restrict__ kS) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x;
    const int n = blockIdx.y * tile_n + threadIdx.x;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* kb = k_usage + b * hidden;
    float acc = 0.0f;
    for (int d = d0; d < d1; ++d) {
        acc += kb[d] * __half2float(Sb[d * hidden + n]);
    }
    atomicAdd(&kS[b * hidden + n], acc);
}

__global__ void efla_kS_kernel_half2(const __half* __restrict__ S,
                                     const float* __restrict__ k_usage,
                                     int hidden,
                                     int tile_d,
                                     float* __restrict__ kS) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* kb = k_usage + b * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = d0; d < d1; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + d * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float k = kb[d];
        acc0 += k * sf.x;
        acc1 += k * sf.y;
    }
    atomicAdd(&kS[b * hidden + n], acc0);
    if (n + 1 < hidden) {
        atomicAdd(&kS[b * hidden + n + 1], acc1);
    }
}

__global__ void efla_kS_kernel_half2_kh(const __half* __restrict__ S,
                                        const __half* __restrict__ k_usage,
                                        int hidden,
                                        int tile_d,
                                        float* __restrict__ kS) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const __half* kb = k_usage + b * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = d0; d < d1; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + d * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float k = __half2float(kb[d]);
        acc0 += k * sf.x;
        acc1 += k * sf.y;
    }
    atomicAdd(&kS[b * hidden + n], acc0);
    if (n + 1 < hidden) {
        atomicAdd(&kS[b * hidden + n + 1], acc1);
    }
}

__global__ void efla_kS_kernel_half2_ksh(const __half* __restrict__ S,
                                         const float* __restrict__ k_usage,
                                         int hidden,
                                         int tile_d,
                                         float* __restrict__ kS) {
    __shared__ float k_sh[128];
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const int count = d1 - d0;
    const float* kb = k_usage + b * hidden + d0;
    for (int d = threadIdx.x; d < count; d += blockDim.x) {
        k_sh[d] = kb[d];
    }
    __syncthreads();
    if (n >= hidden) return;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = 0; d < count; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + (d0 + d) * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float k = k_sh[d];
        acc0 += k * sf.x;
        acc1 += k * sf.y;
    }
    atomicAdd(&kS[b * hidden + n], acc0);
    if (n + 1 < hidden) {
        atomicAdd(&kS[b * hidden + n + 1], acc1);
    }
}

__global__ void efla_kS_kernel_half2_kh_ksh(const __half* __restrict__ S,
                                            const __half* __restrict__ k_usage,
                                            int hidden,
                                            int tile_d,
                                            float* __restrict__ kS) {
    __shared__ float k_sh[128];
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const int count = d1 - d0;
    const __half* kb = k_usage + b * hidden + d0;
    for (int d = threadIdx.x; d < count; d += blockDim.x) {
        k_sh[d] = __half2float(kb[d]);
    }
    __syncthreads();
    if (n >= hidden) return;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = 0; d < count; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + (d0 + d) * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float k = k_sh[d];
        acc0 += k * sf.x;
        acc1 += k * sf.y;
    }
    atomicAdd(&kS[b * hidden + n], acc0);
    if (n + 1 < hidden) {
        atomicAdd(&kS[b * hidden + n + 1], acc1);
    }
}
__global__ void efla_diff_kernel(const float* __restrict__ v,
                                 const float* __restrict__ kS,
                                 int n,
                                 float* __restrict__ diff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    diff[idx] = v[idx] - kS[idx];
}

__global__ void efla_diff_kernel_fused(const float* __restrict__ qkv,
                                       const float* __restrict__ kS,
                                       int hidden,
                                       int fused_cols,
                                       int n,
                                       float* __restrict__ diff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int b = idx / hidden;
    const int n_idx = idx - b * hidden;
    diff[idx] = qkv[static_cast<size_t>(b) * fused_cols + hidden * 2 + n_idx] - kS[idx];
}

__global__ void efla_diff_kernel_half(const float* __restrict__ v,
                                      const float* __restrict__ kS,
                                      int n,
                                      __half* __restrict__ diff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    diff[idx] = __float2half_rn(v[idx] - kS[idx]);
}

__global__ void efla_kS_kernel_half2_full(const __half* __restrict__ S,
                                          const float* __restrict__ k_usage,
                                          int hidden,
                                          float* __restrict__ kS) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    if (n >= hidden) return;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* kb = k_usage + b * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = 0; d < hidden; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + d * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float k = kb[d];
        acc0 += k * sf.x;
        acc1 += k * sf.y;
    }
    kS[b * hidden + n] = acc0;
    if (n + 1 < hidden) {
        kS[b * hidden + n + 1] = acc1;
    }
}

__global__ void efla_kS_kernel_half2_full_kh(const __half* __restrict__ S,
                                             const __half* __restrict__ k_usage,
                                             int hidden,
                                             float* __restrict__ kS) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    if (n >= hidden) return;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const __half* kb = k_usage + b * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = 0; d < hidden; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + d * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float k = __half2float(kb[d]);
        acc0 += k * sf.x;
        acc1 += k * sf.y;
    }
    kS[b * hidden + n] = acc0;
    if (n + 1 < hidden) {
        kS[b * hidden + n + 1] = acc1;
    }
}

__global__ void efla_update_kernel(float* __restrict__ S,
                                   const float* __restrict__ k_usage,
                                   const float* __restrict__ diff,
                                   const float* __restrict__ alpha,
                                   int hidden) {
    const int b = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = k_usage[b * hidden + d];
    const float df = diff[b * hidden + n];
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    S[idx] += a * ku * df;
}

__global__ void efla_update_kernel_fused_diff(float* __restrict__ S,
                                              const float* __restrict__ k_usage,
                                              const float* __restrict__ v,
                                              const float* __restrict__ kS,
                                              const float* __restrict__ alpha,
                                              int hidden) {
    const int b = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = k_usage[b * hidden + d];
    const float df = v[b * hidden + n] - kS[b * hidden + n];
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    S[idx] += a * ku * df;
}

__global__ void efla_update_kernel_half(__half* __restrict__ S,
                                        const float* __restrict__ k_usage,
                                        const float* __restrict__ diff,
                                        const float* __restrict__ alpha,
                                        int hidden) {
    const int b = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = k_usage[b * hidden + d];
    const float df = diff[b * hidden + n];
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    float s = __half2float(S[idx]);
    s += a * ku * df;
    S[idx] = __float2half_rn(s);
}

__global__ void efla_update_kernel_half_fused_diff(__half* __restrict__ S,
                                                   const float* __restrict__ k_usage,
                                                   const float* __restrict__ v,
                                                   const float* __restrict__ kS,
                                                   const float* __restrict__ alpha,
                                                   int hidden) {
    const int b = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = k_usage[b * hidden + d];
    const float df = v[b * hidden + n] - kS[b * hidden + n];
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    float s = __half2float(S[idx]);
    s += a * ku * df;
    S[idx] = __float2half_rn(s);
}

__global__ void efla_update_kernel_half2(__half* __restrict__ S,
                                         const float* __restrict__ k_usage,
                                         const float* __restrict__ diff,
                                         const float* __restrict__ alpha,
                                         int hidden) {
    const int b = blockIdx.x;
    const int n2 = blockIdx.y * blockDim.x + threadIdx.x;
    const int n = n2 * 2;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = k_usage[b * hidden + d];
    const float scale = a * ku;
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    if (n + 1 < hidden) {
        const float df0 = diff[b * hidden + n];
        const float df1 = diff[b * hidden + n + 1];
        __half2 s2 = reinterpret_cast<__half2*>(S + idx)[0];
        float2 sf = __half22float2(s2);
        sf.x += scale * df0;
        sf.y += scale * df1;
        reinterpret_cast<__half2*>(S + idx)[0] = __floats2half2_rn(sf.x, sf.y);
    } else {
        const float df0 = diff[b * hidden + n];
        float s = __half2float(S[idx]);
        s += scale * df0;
        S[idx] = __float2half_rn(s);
    }
}

__global__ void efla_update_kernel_half2_fused_diff(__half* __restrict__ S,
                                                    const float* __restrict__ k_usage,
                                                    const float* __restrict__ v,
                                                    const float* __restrict__ kS,
                                                    const float* __restrict__ alpha,
                                                    int hidden) {
    const int b = blockIdx.x;
    const int n2 = blockIdx.y * blockDim.x + threadIdx.x;
    const int n = n2 * 2;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = k_usage[b * hidden + d];
    const float scale = a * ku;
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    if (n + 1 < hidden) {
        const float df0 = v[b * hidden + n] - kS[b * hidden + n];
        const float df1 = v[b * hidden + n + 1] - kS[b * hidden + n + 1];
        __half2 s2 = reinterpret_cast<__half2*>(S + idx)[0];
        float2 sf = __half22float2(s2);
        sf.x += scale * df0;
        sf.y += scale * df1;
        reinterpret_cast<__half2*>(S + idx)[0] = __floats2half2_rn(sf.x, sf.y);
    } else {
        const float df0 = v[b * hidden + n] - kS[b * hidden + n];
        float s = __half2float(S[idx]);
        s += scale * df0;
        S[idx] = __float2half_rn(s);
    }
}

__global__ void efla_update_kernel_half2_kh(__half* __restrict__ S,
                                            const __half* __restrict__ k_usage,
                                            const __half* __restrict__ diff,
                                            const float* __restrict__ alpha,
                                            int hidden) {
    const int b = blockIdx.x;
    const int n2 = blockIdx.y * blockDim.x + threadIdx.x;
    const int n = n2 * 2;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = __half2float(k_usage[b * hidden + d]);
    const float scale = a * ku;
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    if (n + 1 < hidden) {
        const float df0 = __half2float(diff[b * hidden + n]);
        const float df1 = __half2float(diff[b * hidden + n + 1]);
        __half2 s2 = reinterpret_cast<__half2*>(S + idx)[0];
        float2 sf = __half22float2(s2);
        sf.x += scale * df0;
        sf.y += scale * df1;
        reinterpret_cast<__half2*>(S + idx)[0] = __floats2half2_rn(sf.x, sf.y);
    } else {
        const float df0 = __half2float(diff[b * hidden + n]);
        float s = __half2float(S[idx]);
        s += scale * df0;
        S[idx] = __float2half_rn(s);
    }
}

__global__ void efla_update_kernel_half2_fused_diff_kh(__half* __restrict__ S,
                                                       const __half* __restrict__ k_usage,
                                                       const float* __restrict__ v,
                                                       const float* __restrict__ kS,
                                                       const float* __restrict__ alpha,
                                                       int hidden) {
    const int b = blockIdx.x;
    const int n2 = blockIdx.y * blockDim.x + threadIdx.x;
    const int n = n2 * 2;
    const int d = blockIdx.z * blockDim.y + threadIdx.y;
    if (n >= hidden || d >= hidden) return;
    const float a = alpha[b];
    const float ku = __half2float(k_usage[b * hidden + d]);
    const float scale = a * ku;
    const size_t idx = static_cast<size_t>(b) * hidden * hidden + d * hidden + n;
    if (n + 1 < hidden) {
        const float df0 = v[b * hidden + n] - kS[b * hidden + n];
        const float df1 = v[b * hidden + n + 1] - kS[b * hidden + n + 1];
        __half2 s2 = reinterpret_cast<__half2*>(S + idx)[0];
        float2 sf = __half22float2(s2);
        sf.x += scale * df0;
        sf.y += scale * df1;
        reinterpret_cast<__half2*>(S + idx)[0] = __floats2half2_rn(sf.x, sf.y);
    } else {
        const float df0 = v[b * hidden + n] - kS[b * hidden + n];
        float s = __half2float(S[idx]);
        s += scale * df0;
        S[idx] = __float2half_rn(s);
    }
}

__global__ void efla_update_kernel_wmma(__half* __restrict__ S,
                                        const float* __restrict__ k_usage,
                                        const float* __restrict__ diff,
                                        const float* __restrict__ alpha,
                                        int hidden) {
    constexpr int kTile = 16;
    constexpr int kWarpM = 2;
    constexpr int kWarpN = 2;
    constexpr int kWarps = kWarpM * kWarpN;

    const int warp_id = threadIdx.x >> 5;
    if (warp_id >= kWarps) return;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp_id / kWarpN;
    const int warp_n = warp_id - warp_m * kWarpN;
    const int tile_m = blockIdx.y * kWarpM + warp_m;
    const int tile_n = blockIdx.x * kWarpN + warp_n;
    const int row_base = tile_m * kTile;
    const int col_base = tile_n * kTile;
    const int b = blockIdx.z;
    if (row_base >= hidden || col_base >= hidden) return;

    extern __shared__ float shmem[];
    uint8_t* shmem_u8 = reinterpret_cast<uint8_t*>(shmem);
    __half* sh_a = reinterpret_cast<__half*>(shmem_u8);
    __half* sh_b = sh_a + kWarps * kTile * kTile;
    float* sh_c = reinterpret_cast<float*>(sh_b + kWarps * kTile * kTile);
    __half* sh_a_w = sh_a + warp_id * kTile * kTile;
    __half* sh_b_w = sh_b + warp_id * kTile * kTile;
    float* sh_c_w = sh_c + warp_id * kTile * kTile;

    const float* k_row = k_usage + b * hidden + row_base;
    const float* d_row = diff + b * hidden + col_base;
    for (int idx = lane; idx < kTile * kTile; idx += 32) {
        const int r = idx / kTile;
        const int c = idx - r * kTile;
        const float ku = (row_base + r < hidden) ? k_row[r] : 0.0f;
        const float df = (col_base + c < hidden) ? d_row[c] : 0.0f;
        sh_a_w[idx] = __float2half_rn(ku);
        sh_b_w[c * kTile + r] = __float2half_rn(df);
    }
    __syncwarp();

    using namespace nvcuda::wmma;
    fragment<matrix_a, kTile, kTile, kTile, half, row_major> a_frag;
    fragment<matrix_b, kTile, kTile, kTile, half, col_major> b_frag;
    fragment<accumulator, kTile, kTile, kTile, float> c_frag;
    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, sh_a_w, kTile);
    load_matrix_sync(b_frag, sh_b_w, kTile);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(sh_c_w, c_frag, kTile, mem_row_major);
    __syncwarp();

    const float scale = alpha[b] * (1.0f / static_cast<float>(kTile));
    for (int idx = lane; idx < kTile * kTile; idx += 32) {
        const int r = idx / kTile;
        const int c = idx - r * kTile;
        const int row = row_base + r;
        const int col = col_base + c;
        if (row < hidden && col < hidden) {
            const size_t off = static_cast<size_t>(b) * hidden * hidden +
                               static_cast<size_t>(row) * hidden + col;
            float s = __half2float(S[off]);
            s += sh_c_w[idx] * scale;
            S[off] = __float2half_rn(s);
        }
    }
}


__global__ void efla_out_kernel(const float* __restrict__ S,
                                const float* __restrict__ q_norm,
                                int hidden,
                                int tile_d,
                                float* __restrict__ out) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x;
    const int n = blockIdx.y * tile_n + threadIdx.x;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const float* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* qb = q_norm + b * hidden;
    float acc = 0.0f;
    for (int d = d0; d < d1; ++d) {
        acc += Sb[d * hidden + n] * qb[d];
    }
    atomicAdd(&out[b * hidden + n], acc);
}

__global__ void efla_out_kernel_half(const __half* __restrict__ S,
                                     const float* __restrict__ q_norm,
                                     int hidden,
                                     int tile_d,
                                     float* __restrict__ out) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x;
    const int n = blockIdx.y * tile_n + threadIdx.x;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* qb = q_norm + b * hidden;
    float acc = 0.0f;
    for (int d = d0; d < d1; ++d) {
        acc += __half2float(Sb[d * hidden + n]) * qb[d];
    }
    atomicAdd(&out[b * hidden + n], acc);
}

__global__ void efla_out_kernel_half2(const __half* __restrict__ S,
                                      const float* __restrict__ q_norm,
                                      int hidden,
                                      int tile_d,
                                      float* __restrict__ out) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    if (n >= hidden) return;
    const int d0 = blockIdx.z * tile_d;
    int d1 = d0 + tile_d;
    if (d1 > hidden) d1 = hidden;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* qb = q_norm + b * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = d0; d < d1; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + d * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float q = qb[d];
        acc0 += sf.x * q;
        acc1 += sf.y * q;
    }
    atomicAdd(&out[b * hidden + n], acc0);
    if (n + 1 < hidden) {
        atomicAdd(&out[b * hidden + n + 1], acc1);
    }
}
__global__ void efla_lowrank_step_kernel(float* __restrict__ K,
                                         float* __restrict__ V,
                                         const float* __restrict__ q,
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         const float* __restrict__ gate_raw,
                                         float gate_bias,
                                         float gate_scale,
                                         float state_decay,
                                         int hidden,
                                         int state_dim,
                                         float eps,
                                         float* __restrict__ out) {
    const int b = blockIdx.x;
    float* Kb = K + static_cast<size_t>(b) * state_dim * hidden;
    float* Vb = V + static_cast<size_t>(b) * state_dim * hidden;

    extern __shared__ float sh[];
    float* q_sh = sh;
    float* k_sh = q_sh + hidden;
    float* v_sh = k_sh + hidden;
    float* k_red = v_sh + hidden;
    float* temp = k_red + state_dim;

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        q_sh[j] = q[b * hidden + j];
        k_sh[j] = k[b * hidden + j];
        v_sh[j] = v[b * hidden + j];
    }
    __syncthreads();

    __shared__ float alpha;
    __shared__ float inv_mass;
    if (threadIdx.x == 0) {
        float gate = sigmoid(gate_raw[b] + gate_bias);
        alpha = gate * gate_scale;
    }
    __syncthreads();

    const int chunk = hidden / state_dim;
    const float inv_sqrt_chunk = rsqrtf(static_cast<float>(chunk));
    for (int d = threadIdx.x; d < state_dim; d += blockDim.x) {
        float sum = 0.0f;
        const int base = d * chunk;
        for (int i = 0; i < chunk; ++i) {
            sum += k_sh[base + i];
        }
        k_red[d] = sum * inv_sqrt_chunk;

        float dot = 0.0f;
        const float* Krow = Kb + static_cast<size_t>(d) * hidden;
        for (int h = 0; h < hidden; ++h) {
            dot += q_sh[h] * Krow[h];
        }
        temp[d] = dot;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mass = 0.0f;
        for (int d = 0; d < state_dim; ++d) {
            float v = temp[d];
            mass += v * v;
        }
        inv_mass = 1.0f / (sqrtf(mass + 1e-6f) + 1.0f);
    }
    __syncthreads();

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < state_dim; ++d) {
            acc += temp[d] * Vb[static_cast<size_t>(d) * hidden + h];
        }
        out[b * hidden + h] = acc * inv_mass;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < state_dim * hidden; idx += blockDim.x) {
        const int d = idx / hidden;
        const int h = idx - d * hidden;
        const float factor = alpha * k_red[d];
        const float k_prev = Kb[idx];
        const float v_prev = Vb[idx];
        Kb[idx] = state_decay * k_prev + factor * k_sh[h];
        Vb[idx] = state_decay * v_prev + factor * v_sh[h];
    }
}

__global__ void efla_out_kernel_half2_full(const __half* __restrict__ S,
                                           const float* __restrict__ q_norm,
                                           int hidden,
                                           float* __restrict__ out) {
    const int b = blockIdx.x;
    const int tile_n = blockDim.x * 2;
    const int n = blockIdx.y * tile_n + threadIdx.x * 2;
    if (n >= hidden) return;
    const __half* Sb = S + static_cast<size_t>(b) * hidden * hidden;
    const float* qb = q_norm + b * hidden;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int d = 0; d < hidden; ++d) {
        const __half2 s2 = reinterpret_cast<const __half2*>(Sb + d * hidden + n)[0];
        const float2 sf = __half22float2(s2);
        const float q = qb[d];
        acc0 += sf.x * q;
        acc1 += sf.y * q;
    }
    out[b * hidden + n] = acc0;
    if (n + 1 < hidden) {
        out[b * hidden + n + 1] = acc1;
    }
}

__global__ void efla_zero_kS_out_kernel(float* __restrict__ kS,
                                        float* __restrict__ out,
                                        int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    kS[idx] = 0.0f;
    out[idx] = 0.0f;
}

} // namespace

namespace efla_lm_cuda {

void embedding_lookup_noise(const uint8_t* d_tokens_t,
                            int token_stride,
                            const int8_t* d_emb_w,
                            const int8_t* d_emb_noise,
                            int hidden,
                            int batch,
                            float noise_scale,
                            int anti_sign,
                            float* d_out,
                            cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden + block.x - 1) / block.x, batch, 1);
    embedding_lookup_noise_kernel<<<grid, block, 0, stream>>>(
        d_tokens_t, token_stride, d_emb_w, d_emb_noise, hidden, batch, noise_scale, anti_sign, d_out);
}

void add_pos_gelu_quantize(const float* d_in,
                           const float* d_pos_t,
                           int hidden,
                           int batch,
                           float act_scale,
                           int8_t* d_out_q,
                           cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden + block.x - 1) / block.x, batch, 1);
    add_pos_gelu_quantize_kernel<<<grid, block, 0, stream>>>(
        d_in, d_pos_t, hidden, batch, act_scale, d_out_q);
}

void add_pos_gelu(const float* d_in,
                  const float* d_pos_t,
                  int hidden,
                  int batch,
                  float act_scale,
                  float* d_out,
                  cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden + block.x - 1) / block.x, batch, 1);
    add_pos_gelu_kernel<<<grid, block, 0, stream>>>(
        d_in, d_pos_t, hidden, batch, act_scale, d_out);
}

void gelu_quantize(const float* d_in,
                   int n,
                   float act_scale,
                   int8_t* d_out_q,
                   cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    gelu_quantize_kernel<<<blocks, threads, 0, stream>>>(d_in, n, act_scale, d_out_q);
}

void add_scaled(float* d_out,
                const float* d_noise,
                int n,
                float scale,
                cudaStream_t stream) {
    if (scale == 0.0f) return;
    const int threads = 256;
    const int n4 = (n + 3) / 4;
    const int blocks = (n4 + threads - 1) / threads;
    add_scaled_kernel<<<blocks, threads, 0, stream>>>(d_out, d_noise, n, scale);
}

void add_scaled_i8(float* d_out,
                   const int8_t* d_in_q,
                   int n,
                   float scale,
                   cudaStream_t stream) {
    if (scale == 0.0f) return;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    add_scaled_i8_kernel<<<blocks, threads, 0, stream>>>(d_out, d_in_q, n, scale);
}

void add_scaled_to_int8(int8_t* d_out,
                        const float* d_in,
                        int n,
                        float scale,
                        cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    add_scaled_to_int8_kernel<<<blocks, threads, 0, stream>>>(d_out, d_in, n, scale);
}

void add_scaled_to_int8_absmean_norm_q(int8_t* d_inout,
                                       const float* d_in,
                                       int hidden,
                                       int batch,
                                       float add_scale,
                                       float norm_scale,
                                       int8_t* d_out_q,
                                       cudaStream_t stream) {
    const int threads = 256;
    add_scaled_to_int8_absmean_norm_q_kernel<<<batch, threads, 0, stream>>>(
        d_inout, d_in, hidden, add_scale, norm_scale, d_out_q);
}

void layernorm_quantize(const float* d_in,
                        int hidden,
                        int batch,
                        float scale,
                        float eps,
                        int8_t* d_out_q,
                        cudaStream_t stream) {
    const int threads = 256;
    layernorm_quantize_kernel<<<batch, threads, 0, stream>>>(d_in, hidden, scale, eps, d_out_q);
}

void absmean_norm_q(const int8_t* d_in,
                    int hidden,
                    int batch,
                    float scale,
                    int8_t* d_out_q,
                    cudaStream_t stream) {
    const int threads = 256;
    absmean_norm_q_kernel<<<batch, threads, 0, stream>>>(d_in, hidden, scale, d_out_q);
}

__global__ void init_shadow_kernel(const int8_t* __restrict__ w,
                                   float* __restrict__ shadow,
                                   int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    shadow[idx] = static_cast<float>(w[idx]);
}

__global__ void fill_pair_seeds_kernel(uint64_t base_seed,
                                       uint64_t epoch,
                                       int pairs,
                                       uint64_t* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const uint64_t mixed = mix_u64(base_seed, mix_u64(epoch, static_cast<uint64_t>(idx)));
    out[idx] = mixed;
}

__global__ void compute_z_sumsq_kernel(const float* __restrict__ pair_weights,
                                       const uint64_t* __restrict__ pair_seeds,
                                       uint64_t salt,
                                       int pairs,
                                       int n,
                                       int use_clt,
                                       int clt_k,
                                       float* __restrict__ Z,
                                       float* __restrict__ sumsq_out) {
    extern __shared__ float shsum[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float z = 0.0f;
    if (idx < n) {
        const uint64_t uidx = static_cast<uint64_t>(idx);
        const bool clt = (use_clt != 0);
        for (int p = 0; p < pairs; ++p) {
            const float w = pair_weights[p];
            if (w == 0.0f) continue;
            const uint64_t seed = mix_u64(pair_seeds[p], salt);
            const int8_t eps = noise_hash_u64(seed, uidx, clt, clt_k);
            z += w * static_cast<float>(eps);
        }
        Z[idx] = z;
    }
    const float z2 = (idx < n) ? (z * z) : 0.0f;
    shsum[threadIdx.x] = z2;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shsum[threadIdx.x] += shsum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, shsum[0]);
    }
}

__global__ void update_shadow_kernel(int8_t* __restrict__ w,
                                     float* __restrict__ shadow,
                                     const float* __restrict__ Z,
                                     int n,
                                     float inv_rms,
                                     float lr,
                                     float thresh) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = inv_rms * Z[idx];
    if (fabsf(g) < thresh) return;
    float s = shadow[idx] + lr * g;
    shadow[idx] = s;
    w[idx] = clip_ternary(__float2int_rn(s));
}

__global__ void update_shadow_kernel_device(int8_t* __restrict__ w,
                                            float* __restrict__ shadow,
                                            const float* __restrict__ Z,
                                            int n,
                                            const float* __restrict__ inv_rms,
                                            float lr,
                                            float thresh) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = inv_rms[0] * Z[idx];
    if (fabsf(g) < thresh) return;
    float s = shadow[idx] + lr * g;
    shadow[idx] = s;
    w[idx] = clip_ternary(__float2int_rn(s));
}

__global__ void update_shadow_kernel_device_pack16(int8_t* __restrict__ w,
                                                   uint32_t* __restrict__ w_packed,
                                                   float* __restrict__ shadow,
                                                   const float* __restrict__ Z,
                                                   int n,
                                                   const float* __restrict__ inv_rms,
                                                   float lr,
                                                   float thresh) {
    const int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = pack_idx * 16;
    if (base >= n) return;
    const float inv = inv_rms[0];
    uint32_t out = 0;
    int codes[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int idx = base + k;
        int8_t wv = 0;
        if (idx < n) {
            float g = inv * Z[idx];
            if (fabsf(g) >= thresh) {
                float s = shadow[idx] + lr * g;
                shadow[idx] = s;
                wv = clip_ternary(__float2int_rn(s));
                w[idx] = wv;
            } else {
                wv = w[idx];
            }
        }
        const int code = (wv > 0) ? 3 : (wv < 0) ? 1 : 2;
        codes[k] = code;
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        out |= static_cast<uint32_t>(codes[k] & 0x3) << (k * 2);
    }
    w_packed[pack_idx] = out;
}

__global__ void update_shadow_kernel_device_pack16_bitnet(int8_t* __restrict__ w,
                                                          uint32_t* __restrict__ w_packed,
                                                          float* __restrict__ shadow,
                                                          const float* __restrict__ Z,
                                                          int n,
                                                          const float* __restrict__ inv_rms,
                                                          float lr,
                                                          float thresh) {
    const int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = pack_idx * 16;
    if (base >= n) return;
    const float inv = inv_rms[0];
    uint32_t out = 0;
    int codes[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int idx = base + k;
        int8_t wv = 0;
        if (idx < n) {
            float g = inv * Z[idx];
            if (fabsf(g) >= thresh) {
                float s = shadow[idx] + lr * g;
                shadow[idx] = s;
                wv = clip_ternary(__float2int_rn(s));
                w[idx] = wv;
            } else {
                wv = w[idx];
            }
        }
        const int code = (wv > 0) ? 3 : (wv < 0) ? 1 : 2;
        codes[k] = code;
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int src = (k & 3) * 4 + (k >> 2);
        out |= static_cast<uint32_t>(codes[src] & 0x3) << (k * 2);
    }
    w_packed[pack_idx] = out;
}

__global__ void update_inv_rms_kernel(const float* __restrict__ sumsq,
                                      float* __restrict__ rms_ema,
                                      int use_adaptive,
                                      float beta,
                                      float eps,
                                      int n,
                                      float* __restrict__ inv_rms_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float inv = 1.0f;
    if (use_adaptive) {
        const float ms = sumsq[0] / static_cast<float>(n);
        float ema = rms_ema[0];
        if (ema == 0.0f) ema = ms;
        ema = beta * ema + (1.0f - beta) * ms;
        rms_ema[0] = ema;
        inv = rsqrtf(ema + eps);
    }
    inv_rms_out[0] = inv;
}

void init_shadow_from_weights(const int8_t* d_w,
                              float* d_shadow,
                              int n,
                              cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    init_shadow_kernel<<<blocks, threads, 0, stream>>>(d_w, d_shadow, n);
}

void fill_pair_seeds(uint64_t base_seed,
                     uint64_t epoch,
                     int pairs,
                     uint64_t* d_out,
                     cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (pairs + threads - 1) / threads;
    fill_pair_seeds_kernel<<<blocks, threads, 0, stream>>>(base_seed, epoch, pairs, d_out);
}

void compute_z_sumsq(const float* d_pair_weights,
                     const uint64_t* d_pair_seeds,
                     uint64_t salt,
                     int pairs,
                     int n,
                     bool use_clt,
                     int clt_k,
                     float* d_Z,
                     float* d_sumsq,
                     cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
    compute_z_sumsq_kernel<<<blocks, threads, shmem, stream>>>(d_pair_weights, d_pair_seeds,
                                                               salt, pairs, n,
                                                               use_clt ? 1 : 0, clt_k,
                                                               d_Z, d_sumsq);
}

void update_inv_rms(const float* d_sumsq,
                    float* d_rms_ema,
                    bool use_adaptive,
                    float adaptive_beta,
                    float adaptive_eps,
                    int n,
                    float* d_inv_rms,
                    cudaStream_t stream) {
    update_inv_rms_kernel<<<1, 1, 0, stream>>>(d_sumsq, d_rms_ema,
                                               use_adaptive ? 1 : 0,
                                               adaptive_beta, adaptive_eps,
                                               n, d_inv_rms);
}

void update_shadow_ternary(int8_t* d_w,
                           float* d_shadow,
                           const float* d_Z,
                           int n,
                           float inv_rms,
                           float lr,
                           float thresh,
                           cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    update_shadow_kernel<<<blocks, threads, 0, stream>>>(d_w, d_shadow, d_Z, n, inv_rms, lr, thresh);
}

void update_shadow_ternary_device(int8_t* d_w,
                                  float* d_shadow,
                                  const float* d_Z,
                                  int n,
                                  const float* d_inv_rms,
                                  float lr,
                                  float thresh,
                                  cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    update_shadow_kernel_device<<<blocks, threads, 0, stream>>>(d_w, d_shadow, d_Z, n, d_inv_rms, lr, thresh);
}

void update_shadow_ternary_device_packed(int8_t* d_w,
                                         uint32_t* d_w_packed,
                                         float* d_shadow,
                                         const float* d_Z,
                                         int n,
                                         const float* d_inv_rms,
                                         float lr,
                                         float thresh,
                                         cudaStream_t stream) {
    const int pack_count = (n + 15) >> 4;
    const int threads = 256;
    const int blocks = (pack_count + threads - 1) / threads;
    update_shadow_kernel_device_pack16<<<blocks, threads, 0, stream>>>(
        d_w, d_w_packed, d_shadow, d_Z, n, d_inv_rms, lr, thresh);
}

void update_shadow_ternary_device_packed_bitnet(int8_t* d_w,
                                                uint32_t* d_w_packed,
                                                float* d_shadow,
                                                const float* d_Z,
                                                int n,
                                                const float* d_inv_rms,
                                                float lr,
                                                float thresh,
                                                cudaStream_t stream) {
    const int pack_count = (n + 15) >> 4;
    const int threads = 256;
    const int blocks = (pack_count + threads - 1) / threads;
    update_shadow_kernel_device_pack16_bitnet<<<blocks, threads, 0, stream>>>(
        d_w, d_w_packed, d_shadow, d_Z, n, d_inv_rms, lr, thresh);
}


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
               cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes = static_cast<size_t>(hidden) * 2 * sizeof(float);
    efla_prepare_kernel<<<batch, threads, sh_bytes, stream>>>(
        d_q, d_k, d_beta_raw, beta_bias, hidden, method, eps, d_q_norm, d_k_usage, d_alpha);

    cudaMemsetAsync(d_kS, 0, static_cast<size_t>(batch) * hidden * sizeof(float), stream);
    int tile_n = 64;
    int tile_d = 32;
    select_efla_tiles(hidden, &tile_n, &tile_d);
    dim3 block_k(tile_n, 1, 1);
    dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, (hidden + tile_d - 1) / tile_d);
    efla_kS_kernel<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);

    dim3 block_u(32, 16, 1);
    dim3 grid_u(batch, (hidden + block_u.x - 1) / block_u.x, (hidden + block_u.y - 1) / block_u.y);
    if (use_fuse_diff) {
        efla_update_kernel_fused_diff<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_v, d_kS, d_alpha, hidden);
    } else {
        const int n = batch * hidden;
        const int diff_threads = 256;
        const int diff_blocks = (n + diff_threads - 1) / diff_threads;
        efla_diff_kernel<<<diff_blocks, diff_threads, 0, stream>>>(d_v, d_kS, n, d_diff);
        efla_update_kernel<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);
    }

    cudaMemsetAsync(d_out, 0, static_cast<size_t>(batch) * hidden * sizeof(float), stream);
    efla_out_kernel<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, tile_d, d_out);
}

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
                         cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes = static_cast<size_t>(hidden) * 2 * sizeof(float);
    efla_prepare_kernel_fused<<<batch, threads, sh_bytes, stream>>>(
        d_qkv, fused_cols, beta_bias, hidden, method, eps, d_q_norm, d_k_usage, d_alpha);

    cudaMemsetAsync(d_kS, 0, static_cast<size_t>(batch) * hidden * sizeof(float), stream);
    int tile_n = 64;
    int tile_d = 32;
    select_efla_tiles(hidden, &tile_n, &tile_d);
    dim3 block_k(tile_n, 1, 1);
    dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, (hidden + tile_d - 1) / tile_d);
    efla_kS_kernel<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);

    const int n = batch * hidden;
    const int diff_threads = 256;
    const int diff_blocks = (n + diff_threads - 1) / diff_threads;
    efla_diff_kernel_fused<<<diff_blocks, diff_threads, 0, stream>>>(
        d_qkv, d_kS, hidden, fused_cols, n, d_diff);

    dim3 block_u(32, 16, 1);
    dim3 grid_u(batch, (hidden + block_u.x - 1) / block_u.x, (hidden + block_u.y - 1) / block_u.y);
    efla_update_kernel<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);

    cudaMemsetAsync(d_out, 0, static_cast<size_t>(batch) * hidden * sizeof(float), stream);
    efla_out_kernel<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, tile_d, d_out);
}

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
                    cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes = static_cast<size_t>(hidden) * 2 * sizeof(float);
    const bool use_mixed = use_mixed_kd && d_k_usage_h && d_diff_h && ((hidden & 1) == 0);
    if (use_mixed) {
        efla_prepare_kernel_halfk<<<batch, threads, sh_bytes, stream>>>(
            d_q, d_k, d_beta_raw, beta_bias, hidden, method, eps, d_q_norm, d_k_usage_h, d_alpha);
    } else {
        efla_prepare_kernel<<<batch, threads, sh_bytes, stream>>>(
            d_q, d_k, d_beta_raw, beta_bias, hidden, method, eps, d_q_norm, d_k_usage, d_alpha);
    }

    int tile_n = 64;
    int tile_d = 32;
    select_efla_tiles(hidden, &tile_n, &tile_d);
    const bool use_full = ((hidden & 1) == 0) && (hidden <= 256);
    if (!use_full) {
        const int n_total = batch * hidden;
        const int zero_threads = 256;
        const int zero_blocks = (n_total + zero_threads - 1) / zero_threads;
        efla_zero_kS_out_kernel<<<zero_blocks, zero_threads, 0, stream>>>(d_kS, d_out, n_total);
    }
    if (use_full) {
        dim3 block_k(tile_n / 2, 1, 1);
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, 1);
        if (use_mixed) {
            efla_kS_kernel_half2_full_kh<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage_h, hidden, d_kS);
        } else {
            efla_kS_kernel_half2_full<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, d_kS);
        }
    } else {
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, (hidden + tile_d - 1) / tile_d);
        if ((hidden & 1) == 0) {
            dim3 block_k(tile_n / 2, 1, 1);
            if (use_mixed) {
                if (tile_d == 128) {
                    efla_kS_kernel_half2_kh_ksh<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage_h, hidden, tile_d, d_kS);
                } else {
                    efla_kS_kernel_half2_kh<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage_h, hidden, tile_d, d_kS);
                }
            } else {
                if (tile_d == 128) {
                    efla_kS_kernel_half2_ksh<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);
                } else {
                    efla_kS_kernel_half2<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);
                }
            }
        } else {
            dim3 block_k(tile_n, 1, 1);
            efla_kS_kernel_half<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);
        }
    }

    const bool use_wmma = use_wmma_update && !use_mixed && ((hidden & 15) == 0);
    const bool fuse_diff = use_fuse_diff && !use_wmma;
    if (!fuse_diff) {
        const int n = batch * hidden;
        const int diff_threads = 256;
        const int diff_blocks = (n + diff_threads - 1) / diff_threads;
        if (use_mixed) {
            efla_diff_kernel_half<<<diff_blocks, diff_threads, 0, stream>>>(d_v, d_kS, n, d_diff_h);
        } else {
            efla_diff_kernel<<<diff_blocks, diff_threads, 0, stream>>>(d_v, d_kS, n, d_diff);
        }
    }

    if (use_wmma) {
        constexpr int kTile = 16;
        constexpr int kWarpM = 2;
        constexpr int kWarpN = 2;
        constexpr int kWarps = kWarpM * kWarpN;
        const dim3 block(32 * kWarps, 1, 1);
        const dim3 grid((hidden + kTile * kWarpN - 1) / (kTile * kWarpN),
                        (hidden + kTile * kWarpM - 1) / (kTile * kWarpM),
                        batch);
        const size_t sh_bytes = static_cast<size_t>(kWarps) * kTile * kTile *
                                (sizeof(__half) * 2 + sizeof(float));
        efla_update_kernel_wmma<<<grid, block, sh_bytes, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);
    } else {
        dim3 block_u(32, 16, 1);
        if ((hidden & 1) == 0) {
            dim3 grid_u(batch,
                        (hidden + block_u.x * 2 - 1) / (block_u.x * 2),
                        (hidden + block_u.y - 1) / block_u.y);
            if (fuse_diff) {
                if (use_mixed) {
                    efla_update_kernel_half2_fused_diff_kh<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage_h, d_v, d_kS, d_alpha, hidden);
                } else {
                    efla_update_kernel_half2_fused_diff<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_v, d_kS, d_alpha, hidden);
                }
            } else {
                if (use_mixed) {
                    efla_update_kernel_half2_kh<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage_h, d_diff_h, d_alpha, hidden);
                } else {
                    efla_update_kernel_half2<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);
                }
            }
        } else {
            dim3 grid_u(batch, (hidden + block_u.x - 1) / block_u.x, (hidden + block_u.y - 1) / block_u.y);
            if (fuse_diff) {
                efla_update_kernel_half_fused_diff<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_v, d_kS, d_alpha, hidden);
            } else {
                efla_update_kernel_half<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);
            }
        }
    }

    if (use_full) {
        dim3 block_k(tile_n / 2, 1, 1);
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, 1);
        efla_out_kernel_half2_full<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, d_out);
    } else {
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, (hidden + tile_d - 1) / tile_d);
        if ((hidden & 1) == 0) {
            dim3 block_k(tile_n / 2, 1, 1);
            efla_out_kernel_half2<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, tile_d, d_out);
        } else {
            dim3 block_k(tile_n, 1, 1);
            efla_out_kernel_half<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, tile_d, d_out);
        }
    }
}

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
                              cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes = static_cast<size_t>(hidden) * 2 * sizeof(float);
    efla_prepare_kernel_fused<<<batch, threads, sh_bytes, stream>>>(
        d_qkv, fused_cols, beta_bias, hidden, method, eps, d_q_norm, d_k_usage, d_alpha);

    int tile_n = 64;
    int tile_d = 32;
    select_efla_tiles(hidden, &tile_n, &tile_d);
    const bool use_full = ((hidden & 1) == 0) && (hidden <= 256);
    if (!use_full) {
        const int n_total = batch * hidden;
        const int zero_threads = 256;
        const int zero_blocks = (n_total + zero_threads - 1) / zero_threads;
        efla_zero_kS_out_kernel<<<zero_blocks, zero_threads, 0, stream>>>(d_kS, d_out, n_total);
    }
    if (use_full) {
        dim3 block_k(tile_n / 2, 1, 1);
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, 1);
        efla_kS_kernel_half2_full<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, d_kS);
    } else {
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, (hidden + tile_d - 1) / tile_d);
        if ((hidden & 1) == 0) {
            dim3 block_k(tile_n / 2, 1, 1);
            if (tile_d == 128) {
                efla_kS_kernel_half2_ksh<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);
            } else {
                efla_kS_kernel_half2<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);
            }
        } else {
            dim3 block_k(tile_n, 1, 1);
            efla_kS_kernel_half<<<grid_k, block_k, 0, stream>>>(d_S, d_k_usage, hidden, tile_d, d_kS);
        }
    }

    const int n = batch * hidden;
    const int diff_threads = 256;
    const int diff_blocks = (n + diff_threads - 1) / diff_threads;
    efla_diff_kernel_fused<<<diff_blocks, diff_threads, 0, stream>>>(
        d_qkv, d_kS, hidden, fused_cols, n, d_diff);

    dim3 block_u(32, 16, 1);
    if ((hidden & 1) == 0) {
        dim3 grid_u(batch,
                    (hidden + block_u.x * 2 - 1) / (block_u.x * 2),
                    (hidden + block_u.y - 1) / block_u.y);
        efla_update_kernel_half2<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);
    } else {
        dim3 grid_u(batch, (hidden + block_u.x - 1) / block_u.x, (hidden + block_u.y - 1) / block_u.y);
        efla_update_kernel_half<<<grid_u, block_u, 0, stream>>>(d_S, d_k_usage, d_diff, d_alpha, hidden);
    }

    if (use_full) {
        dim3 block_k(tile_n / 2, 1, 1);
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, 1);
        efla_out_kernel_half2_full<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, d_out);
    } else {
        dim3 grid_k(batch, (hidden + tile_n - 1) / tile_n, (hidden + tile_d - 1) / tile_d);
        if ((hidden & 1) == 0) {
            dim3 block_k(tile_n / 2, 1, 1);
            efla_out_kernel_half2<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, tile_d, d_out);
        } else {
            dim3 block_k(tile_n, 1, 1);
            efla_out_kernel_half<<<grid_k, block_k, 0, stream>>>(d_S, d_q_norm, hidden, tile_d, d_out);
        }
    }
}

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
                     cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes = static_cast<size_t>(hidden) * 6 * sizeof(float);
    efla_step_kernel<<<batch, threads, sh_bytes, stream>>>(
        d_S, d_q, d_k, d_v, d_beta_raw, beta_bias, hidden, method, eps, d_out);
}

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
                          cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes = static_cast<size_t>(hidden) * 6 * sizeof(float);
    efla_step_kernel_half<<<batch, threads, sh_bytes, stream>>>(
        d_S, d_q, d_k, d_v, d_beta_raw, beta_bias, hidden, method, eps, d_out);
}

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
                       cudaStream_t stream) {
    const int threads = 256;
    const size_t sh_bytes =
        static_cast<size_t>(hidden) * 3 * sizeof(float) +
        static_cast<size_t>(state_dim) * 2 * sizeof(float);
    efla_lowrank_step_kernel<<<batch, threads, sh_bytes, stream>>>(
        d_K, d_V, d_q, d_k, d_v, d_gate_raw, gate_bias, gate_scale, state_decay,
        hidden, state_dim, eps, d_out);
}

void cross_entropy_loss(const float* d_logits,
                        const float* d_bias,
                        const float* d_bias_noise,
                        float bias_noise_scale,
                        const uint8_t* d_targets_t,
                        int token_stride,
                        int vocab,
                        int batch,
                        float* d_loss_accum,
                        cudaStream_t stream) {
    const int threads = 256;
    const size_t shmem = static_cast<size_t>(threads) * 2 * sizeof(float);
    cross_entropy_loss_kernel<<<batch, threads, shmem, stream>>>(
        d_logits, d_bias, d_bias_noise, bias_noise_scale, d_targets_t, token_stride, vocab, d_loss_accum);
}

void fill_ternary_noise(int8_t* d_out,
                        size_t n,
                        uint64_t seed,
                        bool use_clt,
                        int clt_k,
                        cudaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    fill_ternary_noise_kernel<<<blocks, threads, 0, stream>>>(
        d_out, n, seed, use_clt ? 1 : 0, clt_k);
}

void fill_ternary_noise_batched(NoiseDesc* d_descs,
                                int num_desc,
                                int max_blocks,
                                bool use_clt,
                                int clt_k,
                                cudaStream_t stream) {
    if (num_desc <= 0 || max_blocks <= 0) return;
    const int threads = 256;
    const dim3 blocks(static_cast<unsigned>(max_blocks),
                      static_cast<unsigned>(num_desc),
                      1u);
    fill_ternary_noise_batched_kernel<<<blocks, threads, 0, stream>>>(
        d_descs, num_desc, max_blocks, use_clt ? 1 : 0, clt_k);
}

void fill_float_noise(float* d_out,
                      int n,
                      uint64_t seed,
                      bool use_clt,
                      int clt_k,
                      cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    fill_float_noise_kernel<<<blocks, threads, 0, stream>>>(
        d_out, n, seed, use_clt ? 1 : 0, clt_k);
}

} // namespace efla_lm_cuda
