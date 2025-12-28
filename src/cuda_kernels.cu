#include "cuda_kernels.h"

#include <cuda_runtime.h>

#if defined(BITNET_EGGROLL_HAS_CUTLASS)
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/subbyte_reference.h>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>
#endif

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace {

constexpr int kGemmThreads = 128; // warp-level GEMM kernels
#if defined(BITNET_EGGROLL_HAS_CUTLASS)
constexpr int kNvfp4SfVec = 16;
#endif
static bitnet_cuda::Nvfp4Schedule g_nvfp4_schedule = bitnet_cuda::Nvfp4Schedule::Auto;
static bitnet_cuda::Nvfp4QuantMode g_nvfp4_quant_mode = bitnet_cuda::Nvfp4QuantMode::Warp16;
static bitnet_cuda::Nvfp4StageCount g_nvfp4_stage_count = bitnet_cuda::Nvfp4StageCount::Auto;
static bitnet_cuda::Nvfp4Decomposition g_nvfp4_decomp = bitnet_cuda::Nvfp4Decomposition::Heuristic;
static int g_nvfp4_splits = 1;
static bool g_nvfp4_verbose = true;

static inline bool nvfp4_use_streamk() {
    return g_nvfp4_splits > 1 ||
           g_nvfp4_decomp == bitnet_cuda::Nvfp4Decomposition::SplitK ||
           g_nvfp4_decomp == bitnet_cuda::Nvfp4Decomposition::StreamK;
}

__device__ __forceinline__ int8_t clip_int8(int v) {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return static_cast<int8_t>(v);
}

__device__ __forceinline__ int lrint_to_int(float x) {
    // Round to nearest even (default IEEE rounding), matching std::lrint() under default mode.
    return __float2int_rn(x);
}

__device__ __forceinline__ float gelu(float x) {
    // tanh approximation (matches efla_lm_kernels.cu)
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(a, b, c);
#else
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    int sum = c;
    sum += static_cast<int>(pa[0]) * static_cast<int>(pb[0]);
    sum += static_cast<int>(pa[1]) * static_cast<int>(pb[1]);
    sum += static_cast<int>(pa[2]) * static_cast<int>(pb[2]);
    sum += static_cast<int>(pa[3]) * static_cast<int>(pb[3]);
    return sum;
#endif
}

__device__ __forceinline__ int warp_sum_i32(int v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_sum_f(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;");
}
#endif

__device__ __constant__ int32_t kI2Lut[256];

__device__ __forceinline__ int8_t decode_i2_code(int code) {
    return (code == 1) ? static_cast<int8_t>(-1)
                       : (code == 3) ? static_cast<int8_t>(1)
                                     : static_cast<int8_t>(0);
}

__device__ __forceinline__ int8_t decode_i2_code_bitnet(uint32_t packed, int idx) {
    // BitNet interleaving is a 4x4 transpose: k = (idx % 4) * 4 + (idx / 4).
    const int k = (idx & 3) * 4 + (idx >> 2);
    const int code = static_cast<int>((packed >> (k * 2)) & 0x3u);
    return decode_i2_code(code);
}

__device__ __forceinline__ void decode_i2s_to_i8s_bitnet(uint32_t packed, int32_t* out_i8s) {
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint32_t bottom_mask = 0x03030303;
    static constexpr uint32_t magic = 0x00000000;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t v;
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(v)
                     : "r"(packed >> (2 * i)), "n"(bottom_mask), "n"(magic), "n"(immLut));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
        out_i8s[i] = __vsubss4(v, 0x02020202);
#else
        const uint32_t b0 = (v & 0xFFu) - 2u;
        const uint32_t b1 = ((v >> 8) & 0xFFu) - 2u;
        const uint32_t b2 = ((v >> 16) & 0xFFu) - 2u;
        const uint32_t b3 = ((v >> 24) & 0xFFu) - 2u;
        out_i8s[i] = static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
#endif
    }
}

void init_i2_lut_once() {
    static bool initialized = false;
    if (initialized) return;
    int32_t host_lut[256];
    for (int i = 0; i < 256; ++i) {
        const uint32_t bits = static_cast<uint32_t>(i);
        uint32_t out = 0;
        for (int k = 0; k < 4; ++k) {
            const int code = static_cast<int>((bits >> (k * 2)) & 0x3u);
            const int8_t v = (code == 1) ? static_cast<int8_t>(-1)
                                         : (code == 3) ? static_cast<int8_t>(1)
                                                       : static_cast<int8_t>(0);
            out |= static_cast<uint32_t>(static_cast<uint8_t>(v)) << (k * 8);
        }
        host_lut[i] = static_cast<int32_t>(out);
    }
    cudaMemcpyToSymbol(kI2Lut, host_lut, sizeof(host_lut));
    initialized = true;
}

__device__ __forceinline__ float gemm_row_accum(const int8_t* __restrict__ xrow,
                                                const int8_t* __restrict__ wrow,
                                                const float* __restrict__ scale_x,
                                                int cols) {
    const int lane = threadIdx.x & 31;
    float out_row = 0.0f;
    int j = 0;
    for (; j + 256 <= cols; j += 256) {
        // Each lane accumulates 8 products via 2 dp4a ops; warp-reduce once per 256 group.
        const int base = j + lane * 8;
        int32_t lane_sum = 0;
        lane_sum = dp4a(*reinterpret_cast<const int*>(xrow + base),
                        *reinterpret_cast<const int*>(wrow + base),
                        lane_sum);
        lane_sum = dp4a(*reinterpret_cast<const int*>(xrow + base + 4),
                        *reinterpret_cast<const int*>(wrow + base + 4),
                        lane_sum);
        const int32_t sum256 = warp_sum_i32(lane_sum);
        if (lane == 0) out_row += scale_x[j / 256] * static_cast<float>(sum256);
    }
    for (; j < cols; j += 32) {
        const int idx = j + lane;
        const int32_t prod = (idx < cols)
                                 ? static_cast<int32_t>(xrow[idx]) * static_cast<int32_t>(wrow[idx])
                                 : 0;
        const int32_t y = warp_sum_i32(prod);
        if (lane == 0) out_row += scale_x[j / 256] * static_cast<float>(y);
    }
    return out_row;
}

__device__ __forceinline__ float gemm_row_accum_i2(const int8_t* __restrict__ xrow,
                                                   const uint32_t* __restrict__ wrow_packed,
                                                   const float* __restrict__ scale_x,
                                                   int cols) {
    const int lane = threadIdx.x & 31;
    float out_row = 0.0f;
    const int full = (cols / 256) * 256;
    int j = 0;
    for (; j < full; j += 256) {
        const int pack_base = j >> 4; // /16
        const int pack_idx = pack_base + (lane >> 1);
        const uint32_t packed = wrow_packed[pack_idx];
        const int offset8 = (lane & 1) * 8;
        const int xbase = j + (lane >> 1) * 16 + offset8;

        const int x0 = *reinterpret_cast<const int*>(xrow + xbase);
        const int x1 = *reinterpret_cast<const int*>(xrow + xbase + 4);
        const uint8_t b0 = static_cast<uint8_t>((packed >> (offset8 * 2)) & 0xFFu);
        const uint8_t b1 = static_cast<uint8_t>((packed >> (offset8 * 2 + 8)) & 0xFFu);
        const int w0 = kI2Lut[b0];
        const int w1 = kI2Lut[b1];

        int32_t lane_sum = 0;
        lane_sum = dp4a(x0, w0, lane_sum);
        lane_sum = dp4a(x1, w1, lane_sum);
        const int32_t sum256 = warp_sum_i32(lane_sum);
        if (lane == 0) out_row += scale_x[j / 256] * static_cast<float>(sum256);
    }
    for (; j < cols; j += 32) {
        const int idx = j + lane;
        int32_t prod = 0;
        if (idx < cols) {
            const int pack_idx = idx >> 4;
            const int off = idx & 15;
            const uint32_t packed = wrow_packed[pack_idx];
            const int code = static_cast<int>((packed >> (off * 2)) & 0x3u);
            const int8_t wv = decode_i2_code(code);
            prod = static_cast<int32_t>(xrow[idx]) * static_cast<int32_t>(wv);
        }
        const int32_t sum32 = warp_sum_i32(prod);
        if (lane == 0) out_row += scale_x[j / 256] * static_cast<float>(sum32);
    }
    return out_row;
}

__device__ __forceinline__ int pack_x4(const int8_t* __restrict__ x,
                                       int i0, int i1, int i2, int i3) {
    const uint32_t b0 = static_cast<uint32_t>(static_cast<uint8_t>(x[i0]));
    const uint32_t b1 = static_cast<uint32_t>(static_cast<uint8_t>(x[i1]));
    const uint32_t b2 = static_cast<uint32_t>(static_cast<uint8_t>(x[i2]));
    const uint32_t b3 = static_cast<uint32_t>(static_cast<uint8_t>(x[i3]));
    return static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

__device__ __forceinline__ float gemm_row_accum_i2_bitnet(const int8_t* __restrict__ xrow,
                                                          const uint32_t* __restrict__ wrow_packed,
                                                          const float* __restrict__ scale_x,
                                                          int cols) {
    const int lane = threadIdx.x & 31;
    float out_row = 0.0f;
    const int full = (cols / 256) * 256;
    int j = 0;
    for (; j < full; j += 256) {
        const int pack_base = j >> 4;
        const int pack_idx = pack_base + (lane >> 1);
        const uint32_t packed = wrow_packed[pack_idx];
        int32_t w_dec[4];
        decode_i2s_to_i8s_bitnet(packed, w_dec);

        const int base = j + (lane >> 1) * 16;
        int32_t lane_sum = 0;
        if ((lane & 1) == 0) {
            const int32_t x0 = pack_x4(xrow, base + 0, base + 4, base + 8, base + 12);
            const int32_t x1 = pack_x4(xrow, base + 1, base + 5, base + 9, base + 13);
            lane_sum = dp4a(x0, w_dec[0], lane_sum);
            lane_sum = dp4a(x1, w_dec[1], lane_sum);
        } else {
            const int32_t x2 = pack_x4(xrow, base + 2, base + 6, base + 10, base + 14);
            const int32_t x3 = pack_x4(xrow, base + 3, base + 7, base + 11, base + 15);
            lane_sum = dp4a(x2, w_dec[2], lane_sum);
            lane_sum = dp4a(x3, w_dec[3], lane_sum);
        }
        const int32_t sum256 = warp_sum_i32(lane_sum);
        if (lane == 0) out_row += scale_x[j / 256] * static_cast<float>(sum256);
    }
    for (; j < cols; j += 32) {
        const int idx = j + lane;
        int32_t prod = 0;
        if (idx < cols) {
            const int pack_idx = idx >> 4;
            const int off = idx & 15;
            const uint32_t packed = wrow_packed[pack_idx];
            const int8_t wv = decode_i2_code_bitnet(packed, off);
            prod = static_cast<int32_t>(xrow[idx]) * static_cast<int32_t>(wv);
        }
        const int32_t sum32 = warp_sum_i32(prod);
        if (lane == 0) out_row += scale_x[j / 256] * static_cast<float>(sum32);
    }
    return out_row;
}

template <int TILE_ROWS, int TILE_B, int TILE_COLS>
__global__ void gemm_ternary_kernel(const int8_t* __restrict__ x,
                                   const int8_t* __restrict__ w,
                                   int rows, int cols, int batch,
                                   float out_scale,
                                   float* __restrict__ out) {
    static_assert((TILE_COLS % 4) == 0, "TILE_COLS must be divisible by 4");

    const int row = blockIdx.x * TILE_ROWS + threadIdx.x;
    const int b = blockIdx.y * TILE_B + threadIdx.y;
    const bool active = (row < rows) && (b < batch);

    const int cols4 = TILE_COLS / 4;

    extern __shared__ int shmem[];
    int* w_sh = shmem;                               // TILE_ROWS * cols4
    int* x_sh = shmem + (TILE_ROWS * cols4);         // TILE_B * cols4

    float acc = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += TILE_COLS) {
        // Cooperative load of weight tile (packed 4x int8 -> int).
        const int threads = blockDim.x * blockDim.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        const int w_elems = TILE_ROWS * cols4;
        for (int idx = tid; idx < w_elems; idx += threads) {
            const int tr = idx / cols4;
            const int pc = idx - tr * cols4;
            const int gr = blockIdx.x * TILE_ROWS + tr;
            const int gc = k0 + pc * 4;
            int v = 0;
            if (gr < rows && (gc + 3) < cols) {
                const int off = gr * cols + gc;
                v = *reinterpret_cast<const int*>(w + off);
            }
            w_sh[tr * cols4 + pc] = v;
        }

        // Cooperative load of x tile (packed).
        const int x_elems = TILE_B * cols4;
        for (int idx = tid; idx < x_elems; idx += threads) {
            const int tb = idx / cols4;
            const int pc = idx - tb * cols4;
            const int gb = blockIdx.y * TILE_B + tb;
            const int gc = k0 + pc * 4;
            int v = 0;
            if (gb < batch && (gc + 3) < cols) {
                const int off = gb * cols + gc;
                v = *reinterpret_cast<const int*>(x + off);
            }
            x_sh[tb * cols4 + pc] = v;
        }

        __syncthreads();

        // Compute dot over this tile with dp4a.
        int32_t tile_sum = 0;
        const int tr = threadIdx.x;
        const int tb = threadIdx.y;
        const int* w_row = &w_sh[tr * cols4];
        const int* x_row = &x_sh[tb * cols4];
        for (int pc = 0; pc < cols4; ++pc) {
            tile_sum = dp4a(x_row[pc], w_row[pc], tile_sum);
        }

        acc += static_cast<double>(tile_sum);

        __syncthreads();
    }

    const double y = acc * static_cast<double>(out_scale);
    if (active) out[b * rows + row] = static_cast<float>(y);
}

__global__ void lora_compute_t_kernel(const int8_t* __restrict__ x,
                                      const int8_t* __restrict__ B,
                                      int cols, int batch, int rank,
                                      int32_t* __restrict__ t_out) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rank;
    if (warp_id >= total) return;

    const int b = warp_id / rank;
    const int r = warp_id - b * rank;
    const int8_t* xb = x + b * cols;
    int32_t acc = 0;
    for (int j = lane; j < cols; j += 32) {
        acc += static_cast<int32_t>(xb[j]) * static_cast<int32_t>(B[j * rank + r]);
    }
    acc = warp_sum_i32(acc);
    if (lane == 0) t_out[b * rank + r] = acc;
}

__global__ void lora_add_kernel(const int32_t* __restrict__ t,
                                const int8_t* __restrict__ A,
                                int rows, int batch, int rank,
                                float scale,
                                float* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || b >= batch) return;

    int32_t pert = 0;
    const int32_t* tb = t + b * rank;
    const int8_t* Ar = A + row * rank;
    for (int r = 0; r < rank; ++r) {
        pert += tb[r] * static_cast<int32_t>(Ar[r]);
    }

    const int idx = b * rows + row;
    out[idx] += scale * static_cast<float>(pert);
}

__global__ void sparse_add_kernel(const int32_t* __restrict__ row_offsets,
                                  const int32_t* __restrict__ col_idx,
                                  const int8_t* __restrict__ eps,
                                  int rows, int cols, int batch,
                                  const int8_t* __restrict__ x,
                                  float scale,
                                  float* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || b >= batch) return;

    float v = out[b * rows + row];
    const int start = row_offsets[row];
    const int end = row_offsets[row + 1];
    const int8_t* xb = x + b * cols;
    for (int k = start; k < end; ++k) {
        const int j = col_idx[k];
        const int8_t e = eps[k];
        v += scale * static_cast<float>(static_cast<int32_t>(xb[j]) * static_cast<int32_t>(e));
    }
    out[b * rows + row] = v;
}

__global__ void activation_quantize_kernel(const float* __restrict__ in,
                                           int8_t* __restrict__ out,
                                           int n,
                                           int activation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = in[idx];
    if (activation == 1) { // ReLU
        if (v < 0.0f) v = 0.0f;
    } else if (activation == 2) { // Tanh
        v = tanhf(v / 127.0f) * 127.0f;
    }

    const int q = lrint_to_int(v);
    out[idx] = clip_int8(q);
}

__global__ void gemm_act_q_kernel(const int8_t* __restrict__ x,
                                  const int8_t* __restrict__ w,
                                  const float* __restrict__ scale_x,
                                  float out_scale,
                                  int cols, int rows, int batch,
                                  int activation,
                                  int8_t* __restrict__ out_q) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        float v = acc * out_scale;
        if (activation == 1) { // ReLU
            if (v < 0.0f) v = 0.0f;
        } else if (activation == 2) { // Tanh
            v = tanhf(v / 127.0f) * 127.0f;
        }
        const int q = lrint_to_int(v);
        out_q[b * rows + i] = clip_int8(q);
    }
}

__global__ void gemm_lora_act_q_kernel(const int8_t* __restrict__ x,
                                       const int8_t* __restrict__ w,
                                       const float* __restrict__ scale_x,
                                       float out_scale,
                                       int cols, int rows, int batch,
                                       const int8_t* __restrict__ A,
                                       const int32_t* __restrict__ t,
                                       int rank,
                                       float lora_scale,
                                       int activation,
                                       int8_t* __restrict__ out_q) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        float v = acc * out_scale;
        int32_t pert = 0;
        const int8_t* Arow = A + i * rank;
        const int32_t* tb = t + b * rank;
        for (int r = 0; r < rank; ++r) {
            pert += tb[r] * static_cast<int32_t>(Arow[r]);
        }
        v += lora_scale * static_cast<float>(pert);
        if (activation == 1) { // ReLU
            if (v < 0.0f) v = 0.0f;
        } else if (activation == 2) { // Tanh
            v = tanhf(v / 127.0f) * 127.0f;
        }
        const int q = lrint_to_int(v);
        out_q[b * rows + i] = clip_int8(q);
    }
}

__global__ void gemm_sparse_act_q_kernel(const int8_t* __restrict__ x,
                                         const int8_t* __restrict__ w,
                                         const float* __restrict__ scale_x,
                                         float out_scale,
                                         int cols, int rows, int batch,
                                         const int32_t* __restrict__ row_offsets,
                                         const int32_t* __restrict__ col_idx,
                                         const int8_t* __restrict__ eps,
                                         float sparse_scale,
                                         int activation,
                                         int8_t* __restrict__ out_q) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        float v = acc * out_scale;
        const int start = row_offsets[i];
        const int end = row_offsets[i + 1];
        for (int p = start; p < end; ++p) {
            const int j = col_idx[p];
            const int8_t e = eps[p];
            v += sparse_scale * static_cast<float>(static_cast<int32_t>(xrow[j]) * static_cast<int32_t>(e));
        }
        if (activation == 1) { // ReLU
            if (v < 0.0f) v = 0.0f;
        } else if (activation == 2) { // Tanh
            v = tanhf(v / 127.0f) * 127.0f;
        }
        const int q = lrint_to_int(v);
        out_q[b * rows + i] = clip_int8(q);
    }
}

__global__ void pack_ternary_i2_kernel(const int8_t* __restrict__ w,
                                       int rows, int cols,
                                       uint32_t* __restrict__ w_packed) {
    const int pack_cols = (cols + 15) >> 4;
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * pack_cols;
    if (idx >= total) return;
    const int row = idx / pack_cols;
    const int pack = idx - row * pack_cols;
    const int col0 = pack * 16;
    const int base = row * cols + col0;
    uint32_t out = 0;
    for (int k = 0; k < 16; ++k) {
        int code = 2; // 0 => code 2
        const int col = col0 + k;
        if (col < cols) {
            const int8_t v = w[base + k];
            code = (v > 0) ? 3 : (v < 0) ? 1 : 2;
        }
        out |= static_cast<uint32_t>(code & 0x3) << (k * 2);
    }
    w_packed[idx] = out;
}

__global__ void pack_ternary_i2_bitnet_kernel(const int8_t* __restrict__ w,
                                              int rows, int cols,
                                              uint32_t* __restrict__ w_packed) {
    const int pack_cols = (cols + 15) >> 4;
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * pack_cols;
    if (idx >= total) return;
    const int row = idx / pack_cols;
    const int pack = idx - row * pack_cols;
    const int col0 = pack * 16;
    const int base = row * cols + col0;
    uint32_t out = 0;
    int codes[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        int code = 2;
        const int col = col0 + k;
        if (col < cols) {
            const int8_t v = w[base + k];
            code = (v > 0) ? 3 : (v < 0) ? 1 : 2;
        }
        codes[k] = code & 0x3;
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int src = (k & 3) * 4 + (k >> 2); // BitNet interleaving (4x4 transpose)
        out |= static_cast<uint32_t>(codes[src]) << (k * 2);
    }
    w_packed[idx] = out;
}

__global__ void gemm_f_i2_kernel(const int8_t* __restrict__ x,
                                 const uint32_t* __restrict__ w_packed,
                                 const float* __restrict__ scale_x,
                                 float out_scale,
                                 int cols, int rows, int batch,
                                 float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int pack_cols = (cols + 15) >> 4;
    const uint32_t* wrow = w_packed + i * pack_cols;

    const float acc = gemm_row_accum_i2(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        out_f[b * rows + i] = acc * out_scale;
    }
}

__global__ void gemm_f_i2_noise_kernel(const int8_t* __restrict__ x,
                                       const uint32_t* __restrict__ w_packed,
                                       const uint32_t* __restrict__ w_packed_noise,
                                       const float* __restrict__ scale_x,
                                       float out_scale,
                                       float noise_scale,
                                       int cols, int rows, int batch,
                                       float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int pack_cols = (cols + 15) >> 4;
    const uint32_t* wrow = w_packed + i * pack_cols;

    const float acc = gemm_row_accum_i2(xrow, wrow, scale_x, cols);
    float acc_n = 0.0f;
    if (w_packed_noise && noise_scale != 0.0f) {
        const uint32_t* wrow_n = w_packed_noise + i * pack_cols;
        acc_n = gemm_row_accum_i2(xrow, wrow_n, scale_x, cols);
    }
    if (lane == 0) {
        out_f[b * rows + i] = (acc + noise_scale * acc_n) * out_scale;
    }
}

__global__ void gemm_f_i2_bitnet_kernel(const int8_t* __restrict__ x,
                                        const uint32_t* __restrict__ w_packed,
                                        const float* __restrict__ scale_x,
                                        float out_scale,
                                        int cols, int rows, int batch,
                                        float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int pack_cols = (cols + 15) >> 4;
    const uint32_t* wrow = w_packed + i * pack_cols;

    const float acc = gemm_row_accum_i2_bitnet(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        out_f[b * rows + i] = acc * out_scale;
    }
}

__global__ void gemm_f_i2_bitnet_noise_kernel(const int8_t* __restrict__ x,
                                              const uint32_t* __restrict__ w_packed,
                                              const uint32_t* __restrict__ w_packed_noise,
                                              const float* __restrict__ scale_x,
                                              float out_scale,
                                              float noise_scale,
                                              int cols, int rows, int batch,
                                              float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int pack_cols = (cols + 15) >> 4;
    const uint32_t* wrow = w_packed + i * pack_cols;

    const float acc = gemm_row_accum_i2_bitnet(xrow, wrow, scale_x, cols);
    float acc_n = 0.0f;
    if (w_packed_noise && noise_scale != 0.0f) {
        const uint32_t* wrow_n = w_packed_noise + i * pack_cols;
        acc_n = gemm_row_accum_i2_bitnet(xrow, wrow_n, scale_x, cols);
    }
    if (lane == 0) {
        out_f[b * rows + i] = (acc + noise_scale * acc_n) * out_scale;
    }
}

__global__ void gemm_f_kernel(const int8_t* __restrict__ x,
                              const int8_t* __restrict__ w,
                              const float* __restrict__ scale_x,
                              float out_scale,
                              int cols, int rows, int batch,
                              float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        out_f[b * rows + i] = acc * out_scale;
    }
}

__global__ void gemm_f_noise_kernel(const int8_t* __restrict__ x,
                                    const int8_t* __restrict__ w,
                                    const int8_t* __restrict__ w_noise,
                                    const float* __restrict__ scale_x,
                                    float out_scale,
                                    float noise_scale,
                                    int cols, int rows, int batch,
                                    float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    float acc_n = 0.0f;
    if (w_noise && noise_scale != 0.0f) {
        const int8_t* wrow_n = w_noise + i * cols;
        acc_n = gemm_row_accum(xrow, wrow_n, scale_x, cols);
    }

    if (lane == 0) {
        out_f[b * rows + i] = (acc + noise_scale * acc_n) * out_scale;
    }
}

__global__ void head_gemm_cross_entropy_kernel(const int8_t* __restrict__ x,
                                               const int8_t* __restrict__ w,
                                               const int8_t* __restrict__ w_noise,
                                               const float* __restrict__ scale_x,
                                               int cols,
                                               int vocab,
                                               float out_scale,
                                               float noise_scale,
                                               const float* __restrict__ bias,
                                               const float* __restrict__ bias_noise,
                                               float bias_noise_scale,
                                               const uint8_t* __restrict__ targets_t,
                                               int token_stride,
                                               float* __restrict__ loss_accum) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ unsigned char shmem_head[];
    size_t offset = 0;
    offset = (offset + 3) & ~static_cast<size_t>(3);
    int8_t* sh_x = reinterpret_cast<int8_t*>(shmem_head + offset);
    offset += static_cast<size_t>(cols) * sizeof(int8_t);
    offset = (offset + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float* shmax = reinterpret_cast<float*>(shmem_head + offset);
    float* shsum = shmax + blockDim.x;

    __shared__ float sh_vy;
    __shared__ int sh_y;
    if (tid == 0) {
        sh_y = static_cast<int>(targets_t[b * token_stride]);
    }
    __syncthreads();

    const int8_t* xrow = x + b * cols;
    for (int j = tid; j < cols; j += blockDim.x) {
        sh_x[j] = xrow[j];
    }
    __syncthreads();

    float logit = -INFINITY;
    if (tid < vocab) {
        const int8_t* wrow = w + tid * cols;
        const int8_t* wrow_n = (w_noise && noise_scale != 0.0f) ? (w_noise + tid * cols) : nullptr;
        float acc = 0.0f;
        float acc_n = 0.0f;
        for (int j0 = 0; j0 < cols; j0 += 256) {
            const int j_end = (j0 + 256 < cols) ? (j0 + 256) : cols;
            int32_t sum = 0;
            int32_t sum_n = 0;
            int j = j0;
            for (; j + 4 <= j_end; j += 4) {
                const int x4 = *reinterpret_cast<const int*>(sh_x + j);
                const int w4 = *reinterpret_cast<const int*>(wrow + j);
                sum = dp4a(x4, w4, sum);
                if (wrow_n) {
                    const int wn4 = *reinterpret_cast<const int*>(wrow_n + j);
                    sum_n = dp4a(x4, wn4, sum_n);
                }
            }
            for (; j < j_end; ++j) {
                const int32_t xv = static_cast<int32_t>(sh_x[j]);
                sum += xv * static_cast<int32_t>(wrow[j]);
                if (wrow_n) {
                    sum_n += xv * static_cast<int32_t>(wrow_n[j]);
                }
            }
            const float scale = scale_x[j0 / 256];
            acc += scale * static_cast<float>(sum);
            if (wrow_n) {
                acc_n += scale * static_cast<float>(sum_n);
            }
        }
        logit = (acc + noise_scale * acc_n) * out_scale;
        if (bias) logit += bias[tid];
        if (bias_noise) logit += bias_noise_scale * bias_noise[tid];
        if (tid == sh_y) {
            sh_vy = logit;
        }
    }

    shmax[tid] = (tid < vocab) ? logit : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmax[tid] = fmaxf(shmax[tid], shmax[tid + stride]);
        }
        __syncthreads();
    }

    const float maxv = shmax[0];
    float local_sum = (tid < vocab) ? expf(logit - maxv) : 0.0f;
    shsum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shsum[tid] += shsum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float logprob = sh_vy - maxv - logf(shsum[0] + 1e-9f);
        atomicAdd(loss_accum, -logprob);
    }
}

__global__ void gemm_gelu_q_kernel(const int8_t* __restrict__ x,
                                  const int8_t* __restrict__ w,
                                  const int8_t* __restrict__ w_noise,
                                  const float* __restrict__ scale_x,
                                  float out_scale,
                                  float noise_scale,
                                  float act_scale,
                                  int cols, int rows, int batch,
                                  int8_t* __restrict__ out_q) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    float acc_n = 0.0f;
    if (w_noise && noise_scale != 0.0f) {
        const int8_t* wrow_n = w_noise + i * cols;
        acc_n = gemm_row_accum(xrow, wrow_n, scale_x, cols);
    }

    if (lane == 0) {
        float v = (acc + noise_scale * acc_n) * out_scale;
        v = gelu(v) * act_scale;
        out_q[b * rows + i] = clip_int8(lrint_to_int(v));
    }
}

__global__ void gemm_qkvb_fused_kernel(const int8_t* __restrict__ x,
                                      const int8_t* __restrict__ wq,
                                      const int8_t* __restrict__ wk,
                                      const int8_t* __restrict__ wv,
                                      const int8_t* __restrict__ wb,
                                      const int8_t* __restrict__ wq_n,
                                      const int8_t* __restrict__ wk_n,
                                      const int8_t* __restrict__ wv_n,
                                      const int8_t* __restrict__ wb_n,
                                      const float* __restrict__ scale_x,
                                      float q_out_scale,
                                      float k_out_scale,
                                      float v_out_scale,
                                      float b_out_scale,
                                      float noise_scale,
                                      int cols, int rows, int batch,
                                      int beta_rows,
                                      float* __restrict__ out_q,
                                      float* __restrict__ out_k,
                                      float* __restrict__ out_v,
                                      float* __restrict__ out_b) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total_rows = rows + beta_rows;
    const int total = batch * total_rows;
    if (warp_id >= total) return;

    const int b = warp_id / total_rows;
    const int i = warp_id - b * total_rows;
    const int8_t* xrow = x + b * cols;

    const bool use_noise = (noise_scale != 0.0f);

    if (i < rows) {
        const int8_t* wqrow = wq + i * cols;
        const int8_t* wkrow = wk + i * cols;
        const int8_t* wvrow = wv + i * cols;
        const int8_t* wqrow_n = wq_n ? (wq_n + i * cols) : nullptr;
        const int8_t* wkrow_n = wk_n ? (wk_n + i * cols) : nullptr;
        const int8_t* wvrow_n = wv_n ? (wv_n + i * cols) : nullptr;

        float acc_q = 0.0f;
        float acc_k = 0.0f;
        float acc_v = 0.0f;
        float acc_qn = 0.0f;
        float acc_kn = 0.0f;
        float acc_vn = 0.0f;

        int j = 0;
        for (; j + 256 <= cols; j += 256) {
            const int base = j + lane * 8;
            const int x0 = *reinterpret_cast<const int*>(xrow + base);
            const int x1 = *reinterpret_cast<const int*>(xrow + base + 4);

            int32_t sum_q = 0;
            int32_t sum_k = 0;
            int32_t sum_v = 0;
            sum_q = dp4a(x0, *reinterpret_cast<const int*>(wqrow + base), sum_q);
            sum_q = dp4a(x1, *reinterpret_cast<const int*>(wqrow + base + 4), sum_q);
            sum_k = dp4a(x0, *reinterpret_cast<const int*>(wkrow + base), sum_k);
            sum_k = dp4a(x1, *reinterpret_cast<const int*>(wkrow + base + 4), sum_k);
            sum_v = dp4a(x0, *reinterpret_cast<const int*>(wvrow + base), sum_v);
            sum_v = dp4a(x1, *reinterpret_cast<const int*>(wvrow + base + 4), sum_v);

            const int32_t sum_qw = warp_sum_i32(sum_q);
            const int32_t sum_kw = warp_sum_i32(sum_k);
            const int32_t sum_vw = warp_sum_i32(sum_v);
            if (lane == 0) {
                const float scale = scale_x[j / 256];
                acc_q += scale * static_cast<float>(sum_qw);
                acc_k += scale * static_cast<float>(sum_kw);
                acc_v += scale * static_cast<float>(sum_vw);
            }

            if (use_noise && wqrow_n && wkrow_n && wvrow_n) {
                int32_t sum_qn = 0;
                int32_t sum_kn = 0;
                int32_t sum_vn = 0;
                sum_qn = dp4a(x0, *reinterpret_cast<const int*>(wqrow_n + base), sum_qn);
                sum_qn = dp4a(x1, *reinterpret_cast<const int*>(wqrow_n + base + 4), sum_qn);
                sum_kn = dp4a(x0, *reinterpret_cast<const int*>(wkrow_n + base), sum_kn);
                sum_kn = dp4a(x1, *reinterpret_cast<const int*>(wkrow_n + base + 4), sum_kn);
                sum_vn = dp4a(x0, *reinterpret_cast<const int*>(wvrow_n + base), sum_vn);
                sum_vn = dp4a(x1, *reinterpret_cast<const int*>(wvrow_n + base + 4), sum_vn);

                const int32_t sum_qnw = warp_sum_i32(sum_qn);
                const int32_t sum_knw = warp_sum_i32(sum_kn);
                const int32_t sum_vnw = warp_sum_i32(sum_vn);
                if (lane == 0) {
                    const float scale = scale_x[j / 256];
                    acc_qn += scale * static_cast<float>(sum_qnw);
                    acc_kn += scale * static_cast<float>(sum_knw);
                    acc_vn += scale * static_cast<float>(sum_vnw);
                }
            }
        }

        for (; j < cols; j += 32) {
            const int idx = j + lane;
            int32_t prod_q = 0;
            int32_t prod_k = 0;
            int32_t prod_v = 0;
            if (idx < cols) {
                const int8_t xv = xrow[idx];
                prod_q = static_cast<int32_t>(xv) * static_cast<int32_t>(wqrow[idx]);
                prod_k = static_cast<int32_t>(xv) * static_cast<int32_t>(wkrow[idx]);
                prod_v = static_cast<int32_t>(xv) * static_cast<int32_t>(wvrow[idx]);
            }
            const int32_t sum_qw = warp_sum_i32(prod_q);
            const int32_t sum_kw = warp_sum_i32(prod_k);
            const int32_t sum_vw = warp_sum_i32(prod_v);
            if (lane == 0) {
                const float scale = scale_x[j / 256];
                acc_q += scale * static_cast<float>(sum_qw);
                acc_k += scale * static_cast<float>(sum_kw);
                acc_v += scale * static_cast<float>(sum_vw);
            }

            if (use_noise && wqrow_n && wkrow_n && wvrow_n) {
                int32_t prod_qn = 0;
                int32_t prod_kn = 0;
                int32_t prod_vn = 0;
                if (idx < cols) {
                    const int8_t xv = xrow[idx];
                    prod_qn = static_cast<int32_t>(xv) * static_cast<int32_t>(wqrow_n[idx]);
                    prod_kn = static_cast<int32_t>(xv) * static_cast<int32_t>(wkrow_n[idx]);
                    prod_vn = static_cast<int32_t>(xv) * static_cast<int32_t>(wvrow_n[idx]);
                }
                const int32_t sum_qnw = warp_sum_i32(prod_qn);
                const int32_t sum_knw = warp_sum_i32(prod_kn);
                const int32_t sum_vnw = warp_sum_i32(prod_vn);
                if (lane == 0) {
                    const float scale = scale_x[j / 256];
                    acc_qn += scale * static_cast<float>(sum_qnw);
                    acc_kn += scale * static_cast<float>(sum_knw);
                    acc_vn += scale * static_cast<float>(sum_vnw);
                }
            }
        }

        if (lane == 0) {
            const float nq = acc_q + noise_scale * acc_qn;
            const float nk = acc_k + noise_scale * acc_kn;
            const float nv = acc_v + noise_scale * acc_vn;
            const int out_off = b * rows + i;
            out_q[out_off] = nq * q_out_scale;
            out_k[out_off] = nk * k_out_scale;
            out_v[out_off] = nv * v_out_scale;
        }
        return;
    }

    if (beta_rows == 0 || !out_b || !wb) return;

    // Beta row (single row).
    const int8_t* wbrow = wb;
    const int8_t* wbrow_n = wb_n ? wb_n : nullptr;
    float acc_b = 0.0f;
    float acc_bn = 0.0f;

    int j = 0;
    for (; j + 256 <= cols; j += 256) {
        const int base = j + lane * 8;
        const int x0 = *reinterpret_cast<const int*>(xrow + base);
        const int x1 = *reinterpret_cast<const int*>(xrow + base + 4);

        int32_t sum_b = 0;
        sum_b = dp4a(x0, *reinterpret_cast<const int*>(wbrow + base), sum_b);
        sum_b = dp4a(x1, *reinterpret_cast<const int*>(wbrow + base + 4), sum_b);
        const int32_t sum_bw = warp_sum_i32(sum_b);
        if (lane == 0) {
            acc_b += scale_x[j / 256] * static_cast<float>(sum_bw);
        }

        if (use_noise && wbrow_n) {
            int32_t sum_bn = 0;
            sum_bn = dp4a(x0, *reinterpret_cast<const int*>(wbrow_n + base), sum_bn);
            sum_bn = dp4a(x1, *reinterpret_cast<const int*>(wbrow_n + base + 4), sum_bn);
            const int32_t sum_bnw = warp_sum_i32(sum_bn);
            if (lane == 0) {
                acc_bn += scale_x[j / 256] * static_cast<float>(sum_bnw);
            }
        }
    }

    for (; j < cols; j += 32) {
        const int idx = j + lane;
        int32_t prod_b = 0;
        if (idx < cols) {
            prod_b = static_cast<int32_t>(xrow[idx]) * static_cast<int32_t>(wbrow[idx]);
        }
        const int32_t sum_bw = warp_sum_i32(prod_b);
        if (lane == 0) {
            acc_b += scale_x[j / 256] * static_cast<float>(sum_bw);
        }

        if (use_noise && wbrow_n) {
            int32_t prod_bn = 0;
            if (idx < cols) {
                prod_bn = static_cast<int32_t>(xrow[idx]) * static_cast<int32_t>(wbrow_n[idx]);
            }
            const int32_t sum_bnw = warp_sum_i32(prod_bn);
            if (lane == 0) {
                acc_bn += scale_x[j / 256] * static_cast<float>(sum_bnw);
            }
        }
    }

    if (lane == 0) {
        const float nb = acc_b + noise_scale * acc_bn;
        out_b[b] = nb * b_out_scale;
    }
}

template <int TILE_ROWS, int TILE_B, int TILE_COLS>
__global__ void gemm_f_noise_tiled_kernel(const int8_t* __restrict__ x,
                                          const int8_t* __restrict__ w,
                                          const int8_t* __restrict__ w_noise,
                                          const float* __restrict__ scale_x,
                                          float out_scale,
                                          float noise_scale,
                                          int rows, int cols, int batch,
                                          bool use_noise,
                                          bool use_cp_async,
                                          float* __restrict__ out_f) {
    const int row = blockIdx.x * TILE_ROWS + threadIdx.x;
    const int b = blockIdx.y * TILE_B + threadIdx.y;
    const bool active = (row < rows) && (b < batch);

    extern __shared__ int8_t shmem_i8[];
    int8_t* w_sh = shmem_i8;
    int8_t* wn_sh = w_sh + TILE_ROWS * TILE_COLS;
    int8_t* x_sh = wn_sh + TILE_ROWS * TILE_COLS;

    const int cols4 = TILE_COLS / 4;
    const int vecs_per_row = TILE_COLS / 16;
    const int w_vecs = TILE_ROWS * vecs_per_row;
    const int x_vecs = TILE_B * vecs_per_row;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    float acc = 0.0f;
    float acc_n = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += TILE_COLS) {
        for (int idx = tid; idx < w_vecs; idx += threads) {
            const int tr = idx / vecs_per_row;
            const int pc = idx - tr * vecs_per_row;
            const int gr = blockIdx.x * TILE_ROWS + tr;
            const int gc = k0 + pc * 16;
            int4 val = {0, 0, 0, 0};
            int4 valn = {0, 0, 0, 0};
            const bool in_bounds = (gr < rows) && (gc + 15 < cols);
            if (in_bounds) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                if (use_cp_async) {
                    cp_async16(reinterpret_cast<int4*>(w_sh) + idx,
                               reinterpret_cast<const int4*>(w + gr * cols + gc));
                    if (use_noise && w_noise) {
                        cp_async16(reinterpret_cast<int4*>(wn_sh) + idx,
                                   reinterpret_cast<const int4*>(w_noise + gr * cols + gc));
                    } else {
                        reinterpret_cast<int4*>(wn_sh)[idx] = valn;
                    }
                    continue;
                }
#endif
                val = *reinterpret_cast<const int4*>(w + gr * cols + gc);
                if (use_noise && w_noise) {
                    valn = *reinterpret_cast<const int4*>(w_noise + gr * cols + gc);
                }
            }
            reinterpret_cast<int4*>(w_sh)[idx] = val;
            reinterpret_cast<int4*>(wn_sh)[idx] = valn;
        }

        for (int idx = tid; idx < x_vecs; idx += threads) {
            const int tb = idx / vecs_per_row;
            const int pc = idx - tb * vecs_per_row;
            const int gb = blockIdx.y * TILE_B + tb;
            const int gc = k0 + pc * 16;
            int4 val = {0, 0, 0, 0};
            const bool in_bounds = (gb < batch) && (gc + 15 < cols);
            if (in_bounds) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                if (use_cp_async) {
                    cp_async16(reinterpret_cast<int4*>(x_sh) + idx,
                               reinterpret_cast<const int4*>(x + gb * cols + gc));
                    continue;
                }
#endif
                val = *reinterpret_cast<const int4*>(x + gb * cols + gc);
            }
            reinterpret_cast<int4*>(x_sh)[idx] = val;
        }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if (use_cp_async) {
            cp_async_commit();
            cp_async_wait();
        }
#endif
        __syncthreads();

        if (active) {
            const int tr = threadIdx.x;
            const int tb = threadIdx.y;
            const int* w_row = reinterpret_cast<const int*>(w_sh + tr * TILE_COLS);
            const int* wn_row = reinterpret_cast<const int*>(wn_sh + tr * TILE_COLS);
            const int* x_row = reinterpret_cast<const int*>(x_sh + tb * TILE_COLS);
            int32_t tile_sum = 0;
            int32_t tile_sum_n = 0;
            for (int pc = 0; pc < cols4; ++pc) {
                tile_sum = dp4a(x_row[pc], w_row[pc], tile_sum);
                if (use_noise) {
                    tile_sum_n = dp4a(x_row[pc], wn_row[pc], tile_sum_n);
                }
            }
            const float scale = scale_x[k0 / 256];
            acc += scale * static_cast<float>(tile_sum);
            if (use_noise) acc_n += scale * static_cast<float>(tile_sum_n);
        }

        __syncthreads();
    }

    if (active) {
        out_f[b * rows + row] = (acc + noise_scale * acc_n) * out_scale;
    }
}

__global__ void gemm_lora_f_kernel(const int8_t* __restrict__ x,
                                   const int8_t* __restrict__ w,
                                   const float* __restrict__ scale_x,
                                   float out_scale,
                                   int cols, int rows, int batch,
                                   const int8_t* __restrict__ A,
                                   const int32_t* __restrict__ t,
                                   int rank,
                                   float lora_scale,
                                   float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        float v = acc * out_scale;
        int32_t pert = 0;
        const int8_t* Arow = A + i * rank;
        const int32_t* tb = t + b * rank;
        for (int r = 0; r < rank; ++r) {
            pert += tb[r] * static_cast<int32_t>(Arow[r]);
        }
        v += lora_scale * static_cast<float>(pert);
        out_f[b * rows + i] = v;
    }
}

__global__ void gemm_sparse_f_kernel(const int8_t* __restrict__ x,
                                     const int8_t* __restrict__ w,
                                     const float* __restrict__ scale_x,
                                     float out_scale,
                                     int cols, int rows, int batch,
                                     const int32_t* __restrict__ row_offsets,
                                     const int32_t* __restrict__ col_idx,
                                     const int8_t* __restrict__ eps,
                                     float sparse_scale,
                                     float* __restrict__ out_f) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    const int total = batch * rows;
    if (warp_id >= total) return;

    const int b = warp_id / rows;
    const int i = warp_id - b * rows;
    const int8_t* xrow = x + b * cols;
    const int8_t* wrow = w + i * cols;

    const float acc = gemm_row_accum(xrow, wrow, scale_x, cols);
    if (lane == 0) {
        float v = acc * out_scale;
        const int start = row_offsets[i];
        const int end = row_offsets[i + 1];
        for (int p = start; p < end; ++p) {
            const int j = col_idx[p];
            const int8_t e = eps[p];
            v += sparse_scale * static_cast<float>(static_cast<int32_t>(xrow[j]) * static_cast<int32_t>(e));
        }
        out_f[b * rows + i] = v;
    }
}

__global__ void embed_update_plain_kernel(const uint8_t* __restrict__ tokens_t,
                                         int token_stride,
                                         const int8_t* __restrict__ emb_w,
                                         int hidden,
                                         int batch,
                                         int8_t* __restrict__ state) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= hidden || b >= batch) return;

    const int tok = static_cast<int>(tokens_t[b * token_stride]);
    const int8_t ev = emb_w[tok * hidden + j];
    const int idx = b * hidden + j;
    const int v = static_cast<int>(state[idx]) + static_cast<int>(ev);
    state[idx] = clip_int8(v);
}

__global__ void embed_update_lora_kernel(const uint8_t* __restrict__ tokens_t,
                                        int token_stride,
                                        const int8_t* __restrict__ emb_w,
                                        const int8_t* __restrict__ A,
                                        const int8_t* __restrict__ B,
                                        int hidden,
                                        int batch,
                                        int rank,
                                        float noise_scale,
                                        int8_t* __restrict__ state) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= hidden || b >= batch) return;

    const int tok = static_cast<int>(tokens_t[b * token_stride]);
    const int8_t* Arow = A + tok * rank;
    const int8_t* Bj = B + j * rank;
    int32_t pert = 0;
    for (int r = 0; r < rank; ++r) {
        pert += static_cast<int32_t>(Arow[r]) * static_cast<int32_t>(Bj[r]);
    }

    const float base = static_cast<float>(emb_w[tok * hidden + j]);
    const float ev = base + noise_scale * static_cast<float>(pert);
    const int add = lrint_to_int(ev);

    const int idx = b * hidden + j;
    const int v = static_cast<int>(state[idx]) + add;
    state[idx] = clip_int8(v);
}

__global__ void embed_update_sparse_kernel(const uint8_t* __restrict__ tokens_t,
                                          int token_stride,
                                          const int8_t* __restrict__ emb_w,
                                          const int16_t* __restrict__ row_noise,
                                          int hidden,
                                          int batch,
                                          float noise_scale,
                                          int8_t* __restrict__ state) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= hidden || b >= batch) return;

    const int tok = static_cast<int>(tokens_t[b * token_stride]);
    const float base = static_cast<float>(emb_w[tok * hidden + j]);
    const int16_t rn = row_noise[tok * hidden + j];
    const float ev = base + noise_scale * static_cast<float>(rn);
    const int add = lrint_to_int(ev);

    const int idx = b * hidden + j;
    const int v = static_cast<int>(state[idx]) + add;
    state[idx] = clip_int8(v);
}

#if defined(BITNET_EGGROLL_HAS_CUTLASS)
template <typename LayoutSF>
__global__ void nvfp4_quantize_rowmajor_warp4_kernel(const int8_t* __restrict__ in,
                                                     int rows,
                                                     int cols_in,
                                                     int cols_out,
                                                     cutlass::float_e2m1_t* __restrict__ out,
                                                     cutlass::float_ue4m3_t* __restrict__ sf,
                                                     LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base_in = row * cols_in + k0;
    const int base_out = row * cols_out + k0;
    constexpr unsigned kActiveMask = 0x0fu;
    float max_abs = 0.0f;
    int8_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    bool v0_ok = false, v1_ok = false, v2_ok = false, v3_ok = false;

    if (lane < 4) {
        const int k_base_load = k0 + lane * 4;
        if (k_base_load + 3 < cols_in) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(in + base_in + lane * 4);
            v0 = static_cast<int8_t>(raw & 0xffu);
            v1 = static_cast<int8_t>((raw >> 8) & 0xffu);
            v2 = static_cast<int8_t>((raw >> 16) & 0xffu);
            v3 = static_cast<int8_t>((raw >> 24) & 0xffu);
            v0_ok = v1_ok = v2_ok = v3_ok = true;
        } else {
            const int k0i = k_base_load + 0;
            const int k1i = k_base_load + 1;
            const int k2i = k_base_load + 2;
            const int k3i = k_base_load + 3;
            if (k0i < cols_in) { v0 = in[base_in + lane * 4 + 0]; v0_ok = true; }
            if (k1i < cols_in) { v1 = in[base_in + lane * 4 + 1]; v1_ok = true; }
            if (k2i < cols_in) { v2 = in[base_in + lane * 4 + 2]; v2_ok = true; }
            if (k3i < cols_in) { v3 = in[base_in + lane * 4 + 3]; v3_ok = true; }
        }
        if (v0_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v0)));
        if (v1_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v1)));
        if (v2_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v2)));
        if (v3_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v3)));
        float tmp = max_abs;
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 1));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 2));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 3));
        max_abs = __shfl_sync(kActiveMask, tmp, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        const int k_base = k0 + lane * 4;
        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (v0_ok) {
                const int idx = k_base + 0;
                const float vf = static_cast<float>(v0) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v1_ok) {
                const int idx = k_base + 1;
                const float vf = static_cast<float>(v1) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v2_ok) {
                const int idx = k_base + 2;
                const float vf = static_cast<float>(v2) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v3_ok) {
                const int idx = k_base + 3;
                const float vf = static_cast<float>(v3) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
        } else {
            uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
            const uint8_t q0_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v0) / scale_f_b).storage) & 0x0fu;
            const uint8_t q1_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v1) / scale_f_b).storage) & 0x0fu;
            const uint8_t q2_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v2) / scale_f_b).storage) & 0x0fu;
            const uint8_t q3_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v3) / scale_f_b).storage) & 0x0fu;

            const uint16_t packed = static_cast<uint16_t>(q0_bits | (q1_bits << 4) |
                                                          (q2_bits << 8) | (q3_bits << 12));
            const uint16_t p1 = __shfl_sync(kActiveMask, packed, 1);
            const uint16_t p2 = __shfl_sync(kActiveMask, packed, 2);
            const uint16_t p3 = __shfl_sync(kActiveMask, packed, 3);
            if (lane == 0) {
                const uint64_t packed64 = static_cast<uint64_t>(packed) |
                                          (static_cast<uint64_t>(p1) << 16) |
                                          (static_cast<uint64_t>(p2) << 32) |
                                          (static_cast<uint64_t>(p3) << 48);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_rowmajor_warp16_kernel(const int8_t* __restrict__ in,
                                                      int rows,
                                                      int cols_in,
                                                      int cols_out,
                                                      cutlass::float_e2m1_t* __restrict__ out,
                                                      cutlass::float_ue4m3_t* __restrict__ sf,
                                                      LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned kActiveMask = 0xffffu;
    const int base_in = row * cols_in + k0;
    const int base_out = row * cols_out + k0;

    if (lane < 16) {
        const int idx = k0 + lane;
        int8_t v = 0;
        bool v_ok = false;
        if (idx < cols_in) {
            v = in[base_in + lane];
            v_ok = true;
        }
        float max_abs = v_ok ? fabsf(static_cast<float>(v)) : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(kActiveMask, max_abs, offset));
        }
        max_abs = __shfl_sync(kActiveMask, max_abs, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane);
                out_ref = q;
            }
        } else {
            uint8_t nib = 0;
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                nib = static_cast<uint8_t>(cutlass::float_e2m1_t(vf).storage) & 0x0fu;
            }
            if (lane == 0) {
                uint64_t packed64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t n = static_cast<uint8_t>(__shfl_sync(kActiveMask, nib, i)) & 0x0fu;
                    packed64 |= (static_cast<uint64_t>(n) << (4 * i));
                }
                uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_rowmajor_activation_warp4_kernel(const float* __restrict__ in,
                                                                int rows,
                                                                int cols_in,
                                                                int cols_out,
                                                                int activation,
                                                                int8_t* __restrict__ out_q,
                                                                cutlass::float_e2m1_t* __restrict__ out,
                                                                cutlass::float_ue4m3_t* __restrict__ sf,
                                                                LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base_in = row * cols_in + k0;
    const int base_out = row * cols_out + k0;
    constexpr unsigned kActiveMask = 0x0fu;
    float max_abs = 0.0f;
    int8_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    bool v0_ok = false, v1_ok = false, v2_ok = false, v3_ok = false;

    if (lane < 4) {
        const int k_base_load = k0 + lane * 4;
        const int k0i = k_base_load + 0;
        const int k1i = k_base_load + 1;
        const int k2i = k_base_load + 2;
        const int k3i = k_base_load + 3;
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
        if (k0i < cols_in) { f0 = in[base_in + lane * 4 + 0]; v0_ok = true; }
        if (k1i < cols_in) { f1 = in[base_in + lane * 4 + 1]; v1_ok = true; }
        if (k2i < cols_in) { f2 = in[base_in + lane * 4 + 2]; v2_ok = true; }
        if (k3i < cols_in) { f3 = in[base_in + lane * 4 + 3]; v3_ok = true; }

        if (activation == 1) { // ReLU
            if (v0_ok && f0 < 0.0f) f0 = 0.0f;
            if (v1_ok && f1 < 0.0f) f1 = 0.0f;
            if (v2_ok && f2 < 0.0f) f2 = 0.0f;
            if (v3_ok && f3 < 0.0f) f3 = 0.0f;
        } else if (activation == 2) { // Tanh
            if (v0_ok) f0 = tanhf(f0 / 127.0f) * 127.0f;
            if (v1_ok) f1 = tanhf(f1 / 127.0f) * 127.0f;
            if (v2_ok) f2 = tanhf(f2 / 127.0f) * 127.0f;
            if (v3_ok) f3 = tanhf(f3 / 127.0f) * 127.0f;
        }

        if (v0_ok) v0 = clip_int8(lrint_to_int(f0));
        if (v1_ok) v1 = clip_int8(lrint_to_int(f1));
        if (v2_ok) v2 = clip_int8(lrint_to_int(f2));
        if (v3_ok) v3 = clip_int8(lrint_to_int(f3));

        if (v0_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v0)));
        if (v1_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v1)));
        if (v2_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v2)));
        if (v3_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v3)));
        float tmp = max_abs;
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 1));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 2));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 3));
        max_abs = __shfl_sync(kActiveMask, tmp, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        const int k_base = k0 + lane * 4;
        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool packed_out_ok = full_block && (cols_out == cols_in) && ((base_in & 3) == 0);
        if (packed_out_ok) {
            const uint32_t packed_q = static_cast<uint32_t>(static_cast<uint8_t>(v0)) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
            reinterpret_cast<uint32_t*>(out_q + base_in + (k_base - k0))[0] = packed_q;
        } else {
            if (v0_ok) out_q[base_in + (k_base - k0) + 0] = v0;
            if (v1_ok) out_q[base_in + (k_base - k0) + 1] = v1;
            if (v2_ok) out_q[base_in + (k_base - k0) + 2] = v2;
            if (v3_ok) out_q[base_in + (k_base - k0) + 3] = v3;
        }

        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (v0_ok) {
                const int idx = k_base + 0;
                const float vf = static_cast<float>(v0) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v1_ok) {
                const int idx = k_base + 1;
                const float vf = static_cast<float>(v1) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v2_ok) {
                const int idx = k_base + 2;
                const float vf = static_cast<float>(v2) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v3_ok) {
                const int idx = k_base + 3;
                const float vf = static_cast<float>(v3) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
        } else {
            uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
            const uint8_t q0_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v0) / scale_f_b).storage) & 0x0fu;
            const uint8_t q1_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v1) / scale_f_b).storage) & 0x0fu;
            const uint8_t q2_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v2) / scale_f_b).storage) & 0x0fu;
            const uint8_t q3_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v3) / scale_f_b).storage) & 0x0fu;

            const uint16_t packed = static_cast<uint16_t>(q0_bits | (q1_bits << 4) |
                                                          (q2_bits << 8) | (q3_bits << 12));
            const uint16_t p1 = __shfl_sync(kActiveMask, packed, 1);
            const uint16_t p2 = __shfl_sync(kActiveMask, packed, 2);
            const uint16_t p3 = __shfl_sync(kActiveMask, packed, 3);
            if (lane == 0) {
                const uint64_t packed64 = static_cast<uint64_t>(packed) |
                                          (static_cast<uint64_t>(p1) << 16) |
                                          (static_cast<uint64_t>(p2) << 32) |
                                          (static_cast<uint64_t>(p3) << 48);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_rowmajor_activation_warp16_kernel(const float* __restrict__ in,
                                                                 int rows,
                                                                 int cols_in,
                                                                 int cols_out,
                                                                 int activation,
                                                                 int8_t* __restrict__ out_q,
                                                                 cutlass::float_e2m1_t* __restrict__ out,
                                                                 cutlass::float_ue4m3_t* __restrict__ sf,
                                                                 LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned kActiveMask = 0xffffu;
    const int base_in = row * cols_in + k0;
    const int base_out = row * cols_out + k0;

    if (lane < 16) {
        const int idx = k0 + lane;
        float f = 0.0f;
        bool ok = false;
        if (idx < cols_in) {
            f = in[base_in + lane];
            ok = true;
        }
        if (ok) {
            if (activation == 1) { // ReLU
                if (f < 0.0f) f = 0.0f;
            } else if (activation == 2) { // Tanh
                f = tanhf(f / 127.0f) * 127.0f;
            }
        }
        int8_t v = 0;
        if (ok) v = clip_int8(lrint_to_int(f));
        float max_abs = ok ? fabsf(static_cast<float>(v)) : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(kActiveMask, max_abs, offset));
        }
        max_abs = __shfl_sync(kActiveMask, max_abs, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        if (idx < cols_in) {
            out_q[base_in + lane] = v;
        }

        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane);
                out_ref = q;
            }
        } else {
            uint8_t nib = 0;
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                nib = static_cast<uint8_t>(cutlass::float_e2m1_t(vf).storage) & 0x0fu;
            }
            if (lane == 0) {
                uint64_t packed64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t n = static_cast<uint8_t>(__shfl_sync(kActiveMask, nib, i)) & 0x0fu;
                    packed64 |= (static_cast<uint64_t>(n) << (4 * i));
                }
                uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void embedding_lookup_noise_quantize_nvfp4_warp4_kernel(const uint8_t* __restrict__ tokens_t,
                                                                   int token_stride,
                                                                   const int8_t* __restrict__ emb_w,
                                                                   const int8_t* __restrict__ emb_noise,
                                                                   int hidden,
                                                                   int batch,
                                                                   float noise_scale,
                                                                   int anti_sign,
                                                                   int8_t* __restrict__ out_q,
                                                                   cutlass::float_e2m1_t* __restrict__ out,
                                                                   cutlass::float_ue4m3_t* __restrict__ sf,
                                                                   LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= batch || k0 >= hidden) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base_out = row * hidden + k0;
    constexpr unsigned kActiveMask = 0x0fu;
    float max_abs = 0.0f;
    int8_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    bool v0_ok = false, v1_ok = false, v2_ok = false, v3_ok = false;

    if (lane < 4) {
        const int tok = static_cast<int>(tokens_t[row * token_stride]);
        const int k_base_load = k0 + lane * 4;
        const int k0i = k_base_load + 0;
        const int k1i = k_base_load + 1;
        const int k2i = k_base_load + 2;
        const int k3i = k_base_load + 3;
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
        if (k0i < hidden) { f0 = static_cast<float>(emb_w[tok * hidden + k0i]); v0_ok = true; }
        if (k1i < hidden) { f1 = static_cast<float>(emb_w[tok * hidden + k1i]); v1_ok = true; }
        if (k2i < hidden) { f2 = static_cast<float>(emb_w[tok * hidden + k2i]); v2_ok = true; }
        if (k3i < hidden) { f3 = static_cast<float>(emb_w[tok * hidden + k3i]); v3_ok = true; }

        if (emb_noise && noise_scale != 0.0f && anti_sign != 0) {
            const float scale = static_cast<float>(anti_sign) * noise_scale;
            if (v0_ok) f0 += scale * static_cast<float>(emb_noise[tok * hidden + k0i]);
            if (v1_ok) f1 += scale * static_cast<float>(emb_noise[tok * hidden + k1i]);
            if (v2_ok) f2 += scale * static_cast<float>(emb_noise[tok * hidden + k2i]);
            if (v3_ok) f3 += scale * static_cast<float>(emb_noise[tok * hidden + k3i]);
        }

        if (v0_ok) v0 = clip_int8(lrint_to_int(f0));
        if (v1_ok) v1 = clip_int8(lrint_to_int(f1));
        if (v2_ok) v2 = clip_int8(lrint_to_int(f2));
        if (v3_ok) v3 = clip_int8(lrint_to_int(f3));

        if (v0_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v0)));
        if (v1_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v1)));
        if (v2_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v2)));
        if (v3_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v3)));
        float tmp = max_abs;
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 1));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 2));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 3));
        max_abs = __shfl_sync(kActiveMask, tmp, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        if (v0_ok) out_q[base_out + lane * 4 + 0] = v0;
        if (v1_ok) out_q[base_out + lane * 4 + 1] = v1;
        if (v2_ok) out_q[base_out + lane * 4 + 2] = v2;
        if (v3_ok) out_q[base_out + lane * 4 + 3] = v3;

        const bool full_block = (k0 + kNvfp4SfVec) <= hidden;
        const bool fast_path = full_block && ((hidden & 1) == 0);
        if (!fast_path) {
            if (k0i < hidden) {
                const float vf = static_cast<float>(v0) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane * 4 + 0);
                out_ref = q;
            }
            if (k1i < hidden) {
                const float vf = static_cast<float>(v1) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane * 4 + 1);
                out_ref = q;
            }
            if (k2i < hidden) {
                const float vf = static_cast<float>(v2) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane * 4 + 2);
                out_ref = q;
            }
            if (k3i < hidden) {
                const float vf = static_cast<float>(v3) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane * 4 + 3);
                out_ref = q;
            }
        } else {
            uint8_t nib = 0;
            if (k0i < hidden) {
                const float vf = static_cast<float>(v0) / scale_f_b;
                nib = static_cast<uint8_t>(cutlass::float_e2m1_t(vf).storage) & 0x0fu;
            }
            if (lane == 0) {
                uint64_t packed64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t n = static_cast<uint8_t>(__shfl_sync(kActiveMask, nib, i)) & 0x0fu;
                    packed64 |= (static_cast<uint64_t>(n) << (4 * i));
                }
                uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void embedding_lookup_noise_quantize_nvfp4_warp16_kernel(const uint8_t* __restrict__ tokens_t,
                                                                    int token_stride,
                                                                    const int8_t* __restrict__ emb_w,
                                                                    const int8_t* __restrict__ emb_noise,
                                                                    int hidden,
                                                                    int batch,
                                                                    float noise_scale,
                                                                    int anti_sign,
                                                                    int8_t* __restrict__ out_q,
                                                                    cutlass::float_e2m1_t* __restrict__ out,
                                                                    cutlass::float_ue4m3_t* __restrict__ sf,
                                                                    LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= batch || k0 >= hidden) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned kActiveMask = 0xffffu;
    const int base_out = row * hidden + k0;

    if (lane < 16) {
        const int idx = k0 + lane;
        const int tok = static_cast<int>(tokens_t[row * token_stride]);
        float f = 0.0f;
        bool ok = false;
        if (idx < hidden) {
            f = static_cast<float>(emb_w[tok * hidden + idx]);
            if (emb_noise && noise_scale != 0.0f && anti_sign != 0) {
                f += static_cast<float>(anti_sign) * noise_scale *
                     static_cast<float>(emb_noise[tok * hidden + idx]);
            }
            ok = true;
        }
        int8_t v = 0;
        if (ok) v = clip_int8(lrint_to_int(f));
        float max_abs = ok ? fabsf(static_cast<float>(v)) : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(kActiveMask, max_abs, offset));
        }
        max_abs = __shfl_sync(kActiveMask, max_abs, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        if (idx < hidden) {
            out_q[base_out + lane] = v;
        }

        const bool full_block = (k0 + kNvfp4SfVec) <= hidden;
        const bool fast_path = full_block && ((hidden & 1) == 0);
        if (!fast_path) {
            if (idx < hidden) {
                const float vf = static_cast<float>(v) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane);
                out_ref = q;
            }
        } else {
            uint8_t nib = 0;
            if (idx < hidden) {
                const float vf = static_cast<float>(v) / scale_f_b;
                nib = static_cast<uint8_t>(cutlass::float_e2m1_t(vf).storage) & 0x0fu;
            }
            if (lane == 0) {
                uint64_t packed64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t n = static_cast<uint8_t>(__shfl_sync(kActiveMask, nib, i)) & 0x0fu;
                    packed64 |= (static_cast<uint64_t>(n) << (4 * i));
                }
                uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_rowmajor_gelu_warp4_kernel(const float* __restrict__ in,
                                                          int rows,
                                                          int cols_in,
                                                          int cols_out,
                                                          float act_scale,
                                                          int8_t* __restrict__ out_q,
                                                          cutlass::float_e2m1_t* __restrict__ out,
                                                          cutlass::float_ue4m3_t* __restrict__ sf,
                                                          LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base_in = row * cols_in + k0;
    const int base_out = row * cols_out + k0;
    constexpr unsigned kActiveMask = 0x0fu;
    float max_abs = 0.0f;
    int8_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    bool v0_ok = false, v1_ok = false, v2_ok = false, v3_ok = false;

    if (lane < 4) {
        const int k_base_load = k0 + lane * 4;
        const int k0i = k_base_load + 0;
        const int k1i = k_base_load + 1;
        const int k2i = k_base_load + 2;
        const int k3i = k_base_load + 3;
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
        if (k0i < cols_in) { f0 = in[base_in + lane * 4 + 0]; v0_ok = true; }
        if (k1i < cols_in) { f1 = in[base_in + lane * 4 + 1]; v1_ok = true; }
        if (k2i < cols_in) { f2 = in[base_in + lane * 4 + 2]; v2_ok = true; }
        if (k3i < cols_in) { f3 = in[base_in + lane * 4 + 3]; v3_ok = true; }

        if (v0_ok) f0 = gelu(f0) * act_scale;
        if (v1_ok) f1 = gelu(f1) * act_scale;
        if (v2_ok) f2 = gelu(f2) * act_scale;
        if (v3_ok) f3 = gelu(f3) * act_scale;

        if (v0_ok) v0 = clip_int8(lrint_to_int(f0));
        if (v1_ok) v1 = clip_int8(lrint_to_int(f1));
        if (v2_ok) v2 = clip_int8(lrint_to_int(f2));
        if (v3_ok) v3 = clip_int8(lrint_to_int(f3));

        if (v0_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v0)));
        if (v1_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v1)));
        if (v2_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v2)));
        if (v3_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v3)));
        float tmp = max_abs;
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 1));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 2));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 3));
        max_abs = __shfl_sync(kActiveMask, tmp, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        const int k_base = k0 + lane * 4;
        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool packed_out_ok = full_block && (cols_out == cols_in) && ((base_in & 3) == 0);
        if (packed_out_ok) {
            const uint32_t packed_q = static_cast<uint32_t>(static_cast<uint8_t>(v0)) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
            reinterpret_cast<uint32_t*>(out_q + base_in + (k_base - k0))[0] = packed_q;
        } else {
            if (v0_ok) out_q[base_in + (k_base - k0) + 0] = v0;
            if (v1_ok) out_q[base_in + (k_base - k0) + 1] = v1;
            if (v2_ok) out_q[base_in + (k_base - k0) + 2] = v2;
            if (v3_ok) out_q[base_in + (k_base - k0) + 3] = v3;
        }

        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (v0_ok) {
                const int idx = k_base + 0;
                const float vf = static_cast<float>(v0) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v1_ok) {
                const int idx = k_base + 1;
                const float vf = static_cast<float>(v1) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v2_ok) {
                const int idx = k_base + 2;
                const float vf = static_cast<float>(v2) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
            if (v3_ok) {
                const int idx = k_base + 3;
                const float vf = static_cast<float>(v3) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + (idx - k0));
                out_ref = q;
            }
        } else {
            uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
            const uint8_t q0_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v0) / scale_f_b).storage) & 0x0fu;
            const uint8_t q1_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v1) / scale_f_b).storage) & 0x0fu;
            const uint8_t q2_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v2) / scale_f_b).storage) & 0x0fu;
            const uint8_t q3_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v3) / scale_f_b).storage) & 0x0fu;

            const uint16_t packed = static_cast<uint16_t>(q0_bits | (q1_bits << 4) |
                                                          (q2_bits << 8) | (q3_bits << 12));
            const uint16_t p1 = __shfl_sync(kActiveMask, packed, 1);
            const uint16_t p2 = __shfl_sync(kActiveMask, packed, 2);
            const uint16_t p3 = __shfl_sync(kActiveMask, packed, 3);
            if (lane == 0) {
                const uint64_t packed64 = static_cast<uint64_t>(packed) |
                                          (static_cast<uint64_t>(p1) << 16) |
                                          (static_cast<uint64_t>(p2) << 32) |
                                          (static_cast<uint64_t>(p3) << 48);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_rowmajor_gelu_warp16_kernel(const float* __restrict__ in,
                                                           int rows,
                                                           int cols_in,
                                                           int cols_out,
                                                           float act_scale,
                                                           int8_t* __restrict__ out_q,
                                                           cutlass::float_e2m1_t* __restrict__ out,
                                                           cutlass::float_ue4m3_t* __restrict__ sf,
                                                           LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned kActiveMask = 0xffffu;
    const int base_in = row * cols_in + k0;
    const int base_out = row * cols_out + k0;

    if (lane < 16) {
        const int idx = k0 + lane;
        float f = 0.0f;
        bool ok = false;
        if (idx < cols_in) {
            f = in[base_in + lane];
            ok = true;
        }
        if (ok) f = gelu(f) * act_scale;
        int8_t v = 0;
        if (ok) v = clip_int8(lrint_to_int(f));
        float max_abs = ok ? fabsf(static_cast<float>(v)) : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(kActiveMask, max_abs, offset));
        }
        max_abs = __shfl_sync(kActiveMask, max_abs, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        if (idx < cols_in) {
            out_q[base_in + lane] = v;
        }

        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, base_out + lane);
                out_ref = q;
            }
        } else {
            uint8_t nib = 0;
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                nib = static_cast<uint8_t>(cutlass::float_e2m1_t(vf).storage) & 0x0fu;
            }
            if (lane == 0) {
                uint64_t packed64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t n = static_cast<uint8_t>(__shfl_sync(kActiveMask, nib, i)) & 0x0fu;
                    packed64 |= (static_cast<uint64_t>(n) << (4 * i));
                }
                uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
                reinterpret_cast<uint64_t*>(out_bytes + (base_out >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_wqkv_fused_warp4_kernel(const int8_t* __restrict__ wq,
                                                       const int8_t* __restrict__ wk,
                                                       const int8_t* __restrict__ wv,
                                                       const int8_t* __restrict__ wb,
                                                       int hidden,
                                                       int cols_out,
                                                       int rows_valid,
                                                       float q_scale,
                                                       float k_scale,
                                                       float v_scale,
                                                       float b_scale,
                                                       cutlass::float_e2m1_t* __restrict__ out,
                                                       cutlass::float_ue4m3_t* __restrict__ sf,
                                                       LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    const int rows = rows_valid;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned kActiveMask = 0x0fu;

    const int group = row / hidden;
    const int row_in_group = row - group * hidden;
    const int8_t* row_ptr = nullptr;
    float row_scale = 1.0f;
    if (group == 0) {
        row_ptr = wq + row_in_group * hidden;
        row_scale = q_scale;
    } else if (group == 1) {
        row_ptr = wk + row_in_group * hidden;
        row_scale = k_scale;
    } else if (group == 2) {
        row_ptr = wv + row_in_group * hidden;
        row_scale = v_scale;
    } else {
        row_ptr = wb;
        row_scale = b_scale;
    }

    float max_abs = 0.0f;
    int8_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    bool v0_ok = false, v1_ok = false, v2_ok = false, v3_ok = false;

    if (lane < 4) {
        const int k_base_load = k0 + lane * 4;
        if (k_base_load + 3 < hidden) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(row_ptr + k_base_load);
            v0 = static_cast<int8_t>(raw & 0xffu);
            v1 = static_cast<int8_t>((raw >> 8) & 0xffu);
            v2 = static_cast<int8_t>((raw >> 16) & 0xffu);
            v3 = static_cast<int8_t>((raw >> 24) & 0xffu);
            v0_ok = v1_ok = v2_ok = v3_ok = true;
        } else {
            const int k0i = k_base_load + 0;
            const int k1i = k_base_load + 1;
            const int k2i = k_base_load + 2;
            const int k3i = k_base_load + 3;
            if (k0i < hidden) { v0 = row_ptr[k0i]; v0_ok = true; }
            if (k1i < hidden) { v1 = row_ptr[k1i]; v1_ok = true; }
            if (k2i < hidden) { v2 = row_ptr[k2i]; v2_ok = true; }
            if (k3i < hidden) { v3 = row_ptr[k3i]; v3_ok = true; }
        }
        if (v0_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v0)));
        if (v1_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v1)));
        if (v2_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v2)));
        if (v3_ok) max_abs = fmaxf(max_abs, fabsf(static_cast<float>(v3)));
        float tmp = max_abs;
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 1));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 2));
        tmp = fmaxf(tmp, __shfl_sync(kActiveMask, tmp, 3));
        max_abs = __shfl_sync(kActiveMask, tmp, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            scale *= row_scale;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        const int out_base = row * cols_out + k0;
        const int k_base = k0 + lane * 4;
        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (v0_ok) {
                const int idx = k_base + 0;
                const float vf = static_cast<float>(v0) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, out_base + (idx - k0));
                out_ref = q;
            }
            if (v1_ok) {
                const int idx = k_base + 1;
                const float vf = static_cast<float>(v1) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, out_base + (idx - k0));
                out_ref = q;
            }
            if (v2_ok) {
                const int idx = k_base + 2;
                const float vf = static_cast<float>(v2) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, out_base + (idx - k0));
                out_ref = q;
            }
            if (v3_ok) {
                const int idx = k_base + 3;
                const float vf = static_cast<float>(v3) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, out_base + (idx - k0));
                out_ref = q;
            }
        } else {
            uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
            const uint8_t q0_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v0) / scale_f_b).storage) & 0x0fu;
            const uint8_t q1_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v1) / scale_f_b).storage) & 0x0fu;
            const uint8_t q2_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v2) / scale_f_b).storage) & 0x0fu;
            const uint8_t q3_bits = static_cast<uint8_t>(
                                    cutlass::float_e2m1_t(static_cast<float>(v3) / scale_f_b).storage) & 0x0fu;

            const uint16_t packed = static_cast<uint16_t>(q0_bits | (q1_bits << 4) |
                                                          (q2_bits << 8) | (q3_bits << 12));
            const uint16_t p1 = __shfl_sync(kActiveMask, packed, 1);
            const uint16_t p2 = __shfl_sync(kActiveMask, packed, 2);
            const uint16_t p3 = __shfl_sync(kActiveMask, packed, 3);
            if (lane == 0) {
                const uint64_t packed64 = static_cast<uint64_t>(packed) |
                                          (static_cast<uint64_t>(p1) << 16) |
                                          (static_cast<uint64_t>(p2) << 32) |
                                          (static_cast<uint64_t>(p3) << 48);
                reinterpret_cast<uint64_t*>(out_bytes + (out_base >> 1))[0] = packed64;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void nvfp4_quantize_wqkv_fused_warp16_kernel(const int8_t* __restrict__ wq,
                                                        const int8_t* __restrict__ wk,
                                                        const int8_t* __restrict__ wv,
                                                        const int8_t* __restrict__ wb,
                                                        int hidden,
                                                        int cols_out,
                                                        int rows_valid,
                                                        float q_scale,
                                                        float k_scale,
                                                        float v_scale,
                                                        float b_scale,
                                                        cutlass::float_e2m1_t* __restrict__ out,
                                                        cutlass::float_ue4m3_t* __restrict__ sf,
                                                        LayoutSF layout_sf) {
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int k0 = block * kNvfp4SfVec;
    const int rows = rows_valid;
    if (row >= rows || k0 >= cols_out) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr unsigned kActiveMask = 0xffffu;

    const int group = row / hidden;
    const int row_in_group = row - group * hidden;
    const int8_t* row_ptr = nullptr;
    float row_scale = 1.0f;
    if (group == 0) {
        row_ptr = wq + row_in_group * hidden;
        row_scale = q_scale;
    } else if (group == 1) {
        row_ptr = wk + row_in_group * hidden;
        row_scale = k_scale;
    } else if (group == 2) {
        row_ptr = wv + row_in_group * hidden;
        row_scale = v_scale;
    } else {
        row_ptr = wb;
        row_scale = b_scale;
    }

    if (lane < 16) {
        const int idx = k0 + lane;
        int8_t v = 0;
        bool ok = false;
        if (idx < hidden) {
            v = row_ptr[idx];
            ok = true;
        }
        float max_abs = ok ? fabsf(static_cast<float>(v)) : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(kActiveMask, max_abs, offset));
        }
        max_abs = __shfl_sync(kActiveMask, max_abs, 0);

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        if (lane == 0) {
            float scale = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
            scale *= row_scale;
            cutlass::float_ue4m3_t scale_q(scale);
            scale_f = static_cast<float>(scale_q);
            if (scale_f == 0.0f) {
                scale_q = cutlass::float_ue4m3_t(1.0f);
                scale_f = 1.0f;
            }
            scale_storage = scale_q.storage;
        }
        const float scale_f_b = __shfl_sync(kActiveMask, scale_f, 0);
        const auto scale_bits = static_cast<typename cutlass::float_ue4m3_t::Storage>(
            __shfl_sync(kActiveMask, static_cast<int>(scale_storage), 0));
        const uint8_t scale_u8 = static_cast<uint8_t>(scale_bits);
        if (lane == 0) {
            const auto sf_offset = layout_sf(cute::make_coord(row, k0, 0));
            reinterpret_cast<uint8_t*>(sf)[sf_offset] = scale_u8;
        }

        const int out_base = row * cols_out + k0;
        const bool full_block = (k0 + kNvfp4SfVec) <= cols_out;
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (!fast_path) {
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                const cutlass::float_e2m1_t q(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, out_base + lane);
                out_ref = q;
            }
        } else {
            uint8_t nib = 0;
            if (idx < cols_out) {
                const float vf = static_cast<float>(v) / scale_f_b;
                nib = static_cast<uint8_t>(cutlass::float_e2m1_t(vf).storage) & 0x0fu;
            }
            if (lane == 0) {
                uint64_t packed64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t n = static_cast<uint8_t>(__shfl_sync(kActiveMask, nib, i)) & 0x0fu;
                    packed64 |= (static_cast<uint64_t>(n) << (4 * i));
                }
                uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
                reinterpret_cast<uint64_t*>(out_bytes + (out_base >> 1))[0] = packed64;
            }
        }
    }
}

__global__ void split_qkvb_fused_kernel(const float* __restrict__ in,
                                        int batch,
                                        int hidden,
                                        int fused_cols,
                                        float* __restrict__ q,
                                        float* __restrict__ k,
                                        float* __restrict__ v,
                                        float* __restrict__ beta,
                                        float beta_bias) {
    const int b = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    if (c >= hidden) {
        if (c == 0) {
            const int stride = fused_cols;
            const int beta_offset = hidden * 3;
            const int base = b * stride;
            beta[b] = in[base + beta_offset];
        }
        return;
    }
    const int stride = fused_cols;
    const int beta_offset = hidden * 3;
    const int base = b * stride;
    const int out_base = b * hidden + c;
    q[out_base] = in[base + c];
    k[out_base] = in[base + hidden + c];
    v[out_base] = in[base + hidden * 2 + c];
    if (c == 0) {
        beta[b] = in[base + beta_offset] + beta_bias;
    }
}

__global__ void split_qkvb_fused_vec4_kernel(const float* __restrict__ in,
                                             int batch,
                                             int hidden,
                                             int fused_cols,
                                             float* __restrict__ q,
                                             float* __restrict__ k,
                                             float* __restrict__ v,
                                             float* __restrict__ beta,
                                             float beta_bias) {
    const int b = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int idx4 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int c = idx4 * 4;
    if (b >= batch || c >= hidden) return;
    const int stride = fused_cols;
    const int beta_offset = hidden * 3;
    const int base = b * stride;
    const int out_base = b * hidden + c;

    const float q0 = in[base + c + 0];
    const float q1 = in[base + c + 1];
    const float q2 = in[base + c + 2];
    const float q3 = in[base + c + 3];
    const float k0 = in[base + hidden + c + 0];
    const float k1 = in[base + hidden + c + 1];
    const float k2 = in[base + hidden + c + 2];
    const float k3 = in[base + hidden + c + 3];
    const float v0 = in[base + hidden * 2 + c + 0];
    const float v1 = in[base + hidden * 2 + c + 1];
    const float v2 = in[base + hidden * 2 + c + 2];
    const float v3 = in[base + hidden * 2 + c + 3];

    reinterpret_cast<float4*>(q + out_base)[0] = make_float4(q0, q1, q2, q3);
    reinterpret_cast<float4*>(k + out_base)[0] = make_float4(k0, k1, k2, k3);
    reinterpret_cast<float4*>(v + out_base)[0] = make_float4(v0, v1, v2, v3);
    if (threadIdx.x == 0) {
        beta[b] = in[base + beta_offset] + beta_bias;
    }
}

template <typename LayoutSF>
__global__ void absmean_norm_q_nvfp4_kernel(const int8_t* __restrict__ in,
                                            int hidden,
                                            int cols_out,
                                            float scale,
                                            int8_t* __restrict__ out_q,
                                            cutlass::float_e2m1_t* __restrict__ out,
                                            cutlass::float_ue4m3_t* __restrict__ sf,
                                            LayoutSF layout_sf) {
    const int b = static_cast<int>(blockIdx.x);
    const int base_in = b * hidden;
    const int base_out = b * cols_out;
    const int num_blocks = (cols_out + kNvfp4SfVec - 1) / kNvfp4SfVec;

    float sum_abs = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        sum_abs += fabsf(static_cast<float>(in[base_in + j]));
    }

    sum_abs = warp_sum_f(sum_abs);
    __shared__ float sh_sum[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sh_sum[warp] = sum_abs;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp == 0) {
        total_sum = (threadIdx.x < (blockDim.x >> 5)) ? sh_sum[threadIdx.x] : 0.0f;
        total_sum = warp_sum_f(total_sum);
    }

    __shared__ float inv_abs_mean;
    if (threadIdx.x == 0) {
        float abs_mean = total_sum / static_cast<float>(hidden);
        if (abs_mean < 1.0f) abs_mean = 1.0f;
        inv_abs_mean = scale / abs_mean;
    }
    __syncthreads();

    const float inv = inv_abs_mean;
    for (int blk = threadIdx.x; blk < num_blocks; blk += blockDim.x) {
        const int k0 = blk * kNvfp4SfVec;
        const int remain_in = hidden - k0;
        const int count_in = (remain_in >= kNvfp4SfVec) ? kNvfp4SfVec : remain_in;
        const int remain_out = cols_out - k0;
        const int count_out = (remain_out >= kNvfp4SfVec) ? kNvfp4SfVec : remain_out;
        if (count_out <= 0) continue;

        int8_t qvals[kNvfp4SfVec];
        float max_abs = 0.0f;
        #pragma unroll
        for (int i = 0; i < kNvfp4SfVec; ++i) {
            if (i < count_in) {
                const int idx = base_in + k0 + i;
                float v = static_cast<float>(in[idx]) * inv;
                int q = lrint_to_int(v);
                const int8_t q8 = clip_int8(q);
                qvals[i] = q8;
                out_q[idx] = q8;
                const float absq = fabsf(static_cast<float>(q8));
                if (absq > max_abs) max_abs = absq;
            } else {
                qvals[i] = 0;
            }
        }

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        float scale_raw = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
        cutlass::float_ue4m3_t scale_q(scale_raw);
        scale_f = static_cast<float>(scale_q);
        if (scale_f == 0.0f) {
            scale_q = cutlass::float_ue4m3_t(1.0f);
            scale_f = 1.0f;
        }
        scale_storage = scale_q.storage;
        const auto sf_offset = layout_sf(cute::make_coord(b, k0, 0));
        reinterpret_cast<uint8_t*>(sf)[sf_offset] = static_cast<uint8_t>(scale_storage);

        const bool full_block = (count_out == kNvfp4SfVec);
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (fast_path) {
            uint64_t packed64 = 0;
            #pragma unroll
            for (int i = 0; i < kNvfp4SfVec; ++i) {
                const uint8_t nib = static_cast<uint8_t>(
                    cutlass::float_e2m1_t(static_cast<float>(qvals[i]) / scale_f).storage) & 0x0fu;
                packed64 |= (static_cast<uint64_t>(nib) << (4 * i));
            }
            uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
            reinterpret_cast<uint64_t*>(out_bytes + ((base_out + k0) >> 1))[0] = packed64;
        } else {
            for (int i = 0; i < count_out; ++i) {
                const int idx = base_out + k0 + i;
                const float vf = static_cast<float>(qvals[i]) / scale_f;
                const cutlass::float_e2m1_t q4(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, idx);
                out_ref = q4;
            }
        }
    }
}

template <typename LayoutSF>
__global__ void add_scaled_to_int8_absmean_norm_q_nvfp4_kernel(int8_t* __restrict__ inout,
                                                               const float* __restrict__ in,
                                                               int hidden,
                                                               int cols_out,
                                                               float add_scale,
                                                               float norm_scale,
                                                               int8_t* __restrict__ out_q,
                                                               cutlass::float_e2m1_t* __restrict__ out,
                                                               cutlass::float_ue4m3_t* __restrict__ sf,
                                                               LayoutSF layout_sf) {
    const int b = static_cast<int>(blockIdx.x);
    const int base_in = b * hidden;
    const int base_out = b * cols_out;
    const int num_blocks = (cols_out + kNvfp4SfVec - 1) / kNvfp4SfVec;

    float sum_abs = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        const int idx = base_in + j;
        const int q = lrint_to_int(in[idx] * add_scale);
        const int sum = static_cast<int>(inout[idx]) + q;
        const int8_t updated = clip_int8(sum);
        inout[idx] = updated;
        sum_abs += fabsf(static_cast<float>(updated));
    }

    sum_abs = warp_sum_f(sum_abs);
    __shared__ float sh_sum[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sh_sum[warp] = sum_abs;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp == 0) {
        total_sum = (threadIdx.x < (blockDim.x >> 5)) ? sh_sum[threadIdx.x] : 0.0f;
        total_sum = warp_sum_f(total_sum);
    }

    __shared__ float inv_abs_mean;
    if (threadIdx.x == 0) {
        float abs_mean = total_sum / static_cast<float>(hidden);
        if (abs_mean < 1.0f) abs_mean = 1.0f;
        inv_abs_mean = norm_scale / abs_mean;
    }
    __syncthreads();

    const float inv = inv_abs_mean;
    for (int blk = threadIdx.x; blk < num_blocks; blk += blockDim.x) {
        const int k0 = blk * kNvfp4SfVec;
        const int remain_in = hidden - k0;
        const int count_in = (remain_in >= kNvfp4SfVec) ? kNvfp4SfVec : remain_in;
        const int remain_out = cols_out - k0;
        const int count_out = (remain_out >= kNvfp4SfVec) ? kNvfp4SfVec : remain_out;
        if (count_out <= 0) continue;

        int8_t qvals[kNvfp4SfVec];
        float max_abs = 0.0f;
        #pragma unroll
        for (int i = 0; i < kNvfp4SfVec; ++i) {
            if (i < count_in) {
                const int idx = base_in + k0 + i;
                float v = static_cast<float>(inout[idx]) * inv;
                int q = lrint_to_int(v);
                const int8_t q8 = clip_int8(q);
                qvals[i] = q8;
                out_q[idx] = q8;
                const float absq = fabsf(static_cast<float>(q8));
                if (absq > max_abs) max_abs = absq;
            } else {
                qvals[i] = 0;
            }
        }

        float scale_f = 1.0f;
        typename cutlass::float_ue4m3_t::Storage scale_storage = 0;
        float scale_raw = (max_abs > 0.0f) ? (max_abs / 6.0f) : 1.0f;
        cutlass::float_ue4m3_t scale_q(scale_raw);
        scale_f = static_cast<float>(scale_q);
        if (scale_f == 0.0f) {
            scale_q = cutlass::float_ue4m3_t(1.0f);
            scale_f = 1.0f;
        }
        scale_storage = scale_q.storage;
        const auto sf_offset = layout_sf(cute::make_coord(b, k0, 0));
        reinterpret_cast<uint8_t*>(sf)[sf_offset] = static_cast<uint8_t>(scale_storage);

        const bool full_block = (count_out == kNvfp4SfVec);
        const bool fast_path = full_block && ((cols_out & 1) == 0);
        if (fast_path) {
            uint64_t packed64 = 0;
            #pragma unroll
            for (int i = 0; i < kNvfp4SfVec; ++i) {
                const uint8_t nib = static_cast<uint8_t>(
                    cutlass::float_e2m1_t(static_cast<float>(qvals[i]) / scale_f).storage) & 0x0fu;
                packed64 |= (static_cast<uint64_t>(nib) << (4 * i));
            }
            uint8_t* out_bytes = reinterpret_cast<uint8_t*>(out);
            reinterpret_cast<uint64_t*>(out_bytes + ((base_out + k0) >> 1))[0] = packed64;
        } else {
            for (int i = 0; i < count_out; ++i) {
                const int idx = base_out + k0 + i;
                const float vf = static_cast<float>(qvals[i]) / scale_f;
                const cutlass::float_e2m1_t q4(vf);
                cutlass::SubbyteReference<cutlass::float_e2m1_t> out_ref(out, idx);
                out_ref = q4;
            }
        }
    }
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
using Nvfp4Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Nvfp4ElementA = Nvfp4Element;
using Nvfp4ElementB = Nvfp4Element;
using Nvfp4ElementC = float;
using Nvfp4ElementD = float;
using Nvfp4LayoutA = cutlass::layout::RowMajor;
using Nvfp4LayoutB = cutlass::layout::ColumnMajor;
using Nvfp4LayoutC = cutlass::layout::RowMajor;
using Nvfp4LayoutD = cutlass::layout::RowMajor;
constexpr int kNvfp4AlignmentA = 32;
constexpr int kNvfp4AlignmentB = 32;
constexpr int kNvfp4AlignmentC = 128 / cutlass::sizeof_bits<Nvfp4ElementC>::value;
constexpr int kNvfp4AlignmentD = 128 / cutlass::sizeof_bits<Nvfp4ElementD>::value;
using Nvfp4Accumulator = float;
using Nvfp4Arch = cutlass::arch::Sm120;
using Nvfp4OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using Nvfp4ThreadBlockShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using Nvfp4ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

using Nvfp4CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Nvfp4Arch, Nvfp4OpClass,
    Nvfp4ThreadBlockShape, Nvfp4ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Nvfp4Accumulator, Nvfp4Accumulator,
    Nvfp4ElementC, Nvfp4LayoutC, kNvfp4AlignmentC,
    Nvfp4ElementD, Nvfp4LayoutD, kNvfp4AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using Nvfp4StageAuto = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename Nvfp4CollectiveEpilogue::SharedStorage))>;

template <typename ScheduleTag, typename StageCountTag>
using Nvfp4CollectiveMainloopT = typename cutlass::gemm::collective::CollectiveBuilder<
    Nvfp4Arch, Nvfp4OpClass,
    Nvfp4ElementA, Nvfp4LayoutA, kNvfp4AlignmentA,
    Nvfp4ElementB, Nvfp4LayoutB, kNvfp4AlignmentB,
    Nvfp4Accumulator,
    Nvfp4ThreadBlockShape, Nvfp4ClusterShape,
    StageCountTag,
    ScheduleTag
  >::CollectiveOp;

template <typename ScheduleTag, typename StageCountTag, typename TileSchedulerTag>
using Nvfp4GemmKernelT = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    Nvfp4CollectiveMainloopT<ScheduleTag, StageCountTag>,
    Nvfp4CollectiveEpilogue,
    TileSchedulerTag>;

template <typename ScheduleTag, typename StageCountTag>
using Nvfp4GemmPersistentT = cutlass::gemm::device::GemmUniversalAdapter<
    Nvfp4GemmKernelT<ScheduleTag, StageCountTag, void>>;

template <typename ScheduleTag, typename StageCountTag>
using Nvfp4GemmStreamKT = cutlass::gemm::device::GemmUniversalAdapter<
    Nvfp4GemmKernelT<ScheduleTag, StageCountTag, cutlass::gemm::StreamKScheduler>>;

using Nvfp4GemmAuto = Nvfp4GemmPersistentT<cutlass::gemm::collective::KernelScheduleAuto, Nvfp4StageAuto>;
using Nvfp4GemmCooperative = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedCooperative, Nvfp4StageAuto>;
using Nvfp4GemmPingpong = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedPingpong, Nvfp4StageAuto>;

using Nvfp4GemmAutoS2 = Nvfp4GemmPersistentT<cutlass::gemm::collective::KernelScheduleAuto,
                                  cutlass::gemm::collective::StageCount<2>>;
using Nvfp4GemmAutoS3 = Nvfp4GemmPersistentT<cutlass::gemm::collective::KernelScheduleAuto,
                                  cutlass::gemm::collective::StageCount<3>>;
using Nvfp4GemmAutoS4 = Nvfp4GemmPersistentT<cutlass::gemm::collective::KernelScheduleAuto,
                                  cutlass::gemm::collective::StageCount<4>>;
using Nvfp4GemmCooperativeS2 = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                         cutlass::gemm::collective::StageCount<2>>;
using Nvfp4GemmCooperativeS3 = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                         cutlass::gemm::collective::StageCount<3>>;
using Nvfp4GemmCooperativeS4 = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                         cutlass::gemm::collective::StageCount<4>>;
using Nvfp4GemmPingpongS2 = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                                 cutlass::gemm::collective::StageCount<2>>;
using Nvfp4GemmPingpongS3 = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                                 cutlass::gemm::collective::StageCount<3>>;
using Nvfp4GemmPingpongS4 = Nvfp4GemmPersistentT<cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                                 cutlass::gemm::collective::StageCount<4>>;

using Nvfp4GemmAutoSK = Nvfp4GemmStreamKT<cutlass::gemm::collective::KernelScheduleAuto, Nvfp4StageAuto>;
using Nvfp4GemmCooperativeSK = Nvfp4GemmStreamKT<cutlass::gemm::KernelTmaWarpSpecializedCooperative, Nvfp4StageAuto>;

using Nvfp4GemmAutoSKS2 = Nvfp4GemmStreamKT<cutlass::gemm::collective::KernelScheduleAuto,
                                            cutlass::gemm::collective::StageCount<2>>;
using Nvfp4GemmAutoSKS3 = Nvfp4GemmStreamKT<cutlass::gemm::collective::KernelScheduleAuto,
                                            cutlass::gemm::collective::StageCount<3>>;
using Nvfp4GemmAutoSKS4 = Nvfp4GemmStreamKT<cutlass::gemm::collective::KernelScheduleAuto,
                                            cutlass::gemm::collective::StageCount<4>>;
using Nvfp4GemmCooperativeSKS2 = Nvfp4GemmStreamKT<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                                   cutlass::gemm::collective::StageCount<2>>;
using Nvfp4GemmCooperativeSKS3 = Nvfp4GemmStreamKT<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                                   cutlass::gemm::collective::StageCount<3>>;
using Nvfp4GemmCooperativeSKS4 = Nvfp4GemmStreamKT<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                                   cutlass::gemm::collective::StageCount<4>>;

template <typename Gemm>
struct Nvfp4GemmCacheEntry {
    int m = 0;
    int n = 0;
    int k = 0;
    Gemm gemm;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    bool initialized = false;
};

template <typename T, typename = void>
struct Nvfp4HasSplits : std::false_type {};

template <typename T>
struct Nvfp4HasSplits<T, std::void_t<decltype(std::declval<T&>().splits)>> : std::true_type {};

template <typename T, typename = void>
struct Nvfp4HasDecomposition : std::false_type {};

template <typename T>
struct Nvfp4HasDecomposition<T, std::void_t<decltype(std::declval<T&>().decomposition_mode)>> : std::true_type {};

template <typename Gemm>
static thread_local std::vector<Nvfp4GemmCacheEntry<Gemm>> nvfp4_gemm_cache;

template <typename Gemm>
static Nvfp4GemmCacheEntry<Gemm>* nvfp4_get_cache_entry(int m, int n, int k) {
    auto& cache = nvfp4_gemm_cache<Gemm>;
    for (auto& entry : cache) {
        if (entry.m == m && entry.n == n && entry.k == k) {
            return &entry;
        }
    }
    Nvfp4GemmCacheEntry<Gemm> entry;
    entry.m = m;
    entry.n = n;
    entry.k = k;
    cache.push_back(entry);
    return &cache.back();
}

template <typename Gemm>
static typename Gemm::Arguments nvfp4_make_args(const void* d_a,
                                                const void* d_b,
                                                const void* d_sfa,
                                                const void* d_sfb,
                                                int m, int n, int k,
                                                float alpha,
                                                float beta,
                                                const float* d_c,
                                                float* d_d) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    auto args = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {static_cast<const Nvfp4ElementA::DataType*>(d_a), stride_A,
         static_cast<const Nvfp4ElementB::DataType*>(d_b), stride_B,
         static_cast<const Nvfp4ElementA::ScaleFactorType*>(d_sfa), layout_sfa,
         static_cast<const Nvfp4ElementB::ScaleFactorType*>(d_sfb), layout_sfb},
        {{alpha, beta},
         d_c,
         stride_C,
         d_d,
         stride_D}
    };

    auto& sched = args.scheduler;
    if constexpr (Nvfp4HasSplits<decltype(sched)>::value) {
        sched.splits = g_nvfp4_splits;
    }
    if constexpr (Nvfp4HasDecomposition<decltype(sched)>::value) {
        using DecompT = decltype(sched.decomposition_mode);
        switch (g_nvfp4_decomp) {
            case bitnet_cuda::Nvfp4Decomposition::DataParallel:
                sched.decomposition_mode = DecompT::DataParallel;
                break;
            case bitnet_cuda::Nvfp4Decomposition::SplitK:
                sched.decomposition_mode = DecompT::SplitK;
                break;
            case bitnet_cuda::Nvfp4Decomposition::StreamK:
                sched.decomposition_mode = DecompT::StreamK;
                break;
            case bitnet_cuda::Nvfp4Decomposition::Heuristic:
            default:
                sched.decomposition_mode = DecompT::Heuristic;
                break;
        }
    }

    return args;
}

template <typename Gemm>
static bool nvfp4_prepare_entry(Nvfp4GemmCacheEntry<Gemm>* entry,
                                const typename Gemm::Arguments& args,
                                cudaStream_t stream) {
    if (!entry) return false;
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess) {
        if (g_nvfp4_verbose) {
            fprintf(stderr, "nvfp4_prepare_gemm: can_implement failed (m=%d n=%d k=%d)\n",
                    entry->m, entry->n, entry->k);
        }
        return false;
    }
    const size_t workspace_size = Gemm::get_workspace_size(args);
    if (workspace_size > entry->workspace_size) {
        if (entry->workspace) {
            cudaFree(entry->workspace);
            entry->workspace = nullptr;
            entry->workspace_size = 0;
        }
        if (workspace_size > 0) {
            if (cudaMalloc(&entry->workspace, workspace_size) != cudaSuccess) {
                if (g_nvfp4_verbose) {
                    fprintf(stderr, "nvfp4_prepare_gemm: cudaMalloc workspace failed (bytes=%zu)\n", workspace_size);
                }
                return false;
            }
            entry->workspace_size = workspace_size;
        }
    }
    if (entry->gemm.initialize(args, entry->workspace, stream) != cutlass::Status::kSuccess) {
        if (g_nvfp4_verbose) {
            fprintf(stderr, "nvfp4_prepare_gemm: initialize failed (m=%d n=%d k=%d)\n",
                    entry->m, entry->n, entry->k);
        }
        return false;
    }
    entry->initialized = true;
    return true;
}

template <typename Gemm>
static bool nvfp4_prepare_gemm_impl(const void* d_a,
                                    const void* d_b,
                                    const void* d_sfa,
                                    const void* d_sfb,
                                    int m, int n, int k,
                                    float alpha,
                                    float beta,
                                    const float* d_c,
                                    float* d_d,
                                    cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) return false;
    auto args = nvfp4_make_args<Gemm>(d_a, d_b, d_sfa, d_sfb, m, n, k, alpha, beta, d_c, d_d);
    auto* entry = nvfp4_get_cache_entry<Gemm>(m, n, k);
    return nvfp4_prepare_entry(entry, args, stream);
}

template <typename Gemm>
static void nvfp4_gemm_run_impl(const void* d_a,
                                const void* d_b,
                                const void* d_sfa,
                                const void* d_sfb,
                                int m, int n, int k,
                                float alpha,
                                float beta,
                                const float* d_c,
                                float* d_d,
                                cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    auto args = nvfp4_make_args<Gemm>(d_a, d_b, d_sfa, d_sfb, m, n, k, alpha, beta, d_c, d_d);
    auto* entry = nvfp4_get_cache_entry<Gemm>(m, n, k);
    if (!entry || !entry->initialized) {
        if (!nvfp4_prepare_entry(entry, args, stream)) {
            return;
        }
    }
    if (entry->gemm.update(args) != cutlass::Status::kSuccess) {
        return;
    }
    entry->gemm.run(stream);
}

#endif
#endif

} // namespace

namespace bitnet_cuda {

void gemm_ternary(const int8_t* d_x, const int8_t* d_w,
                  int rows, int cols, int batch,
                  float out_scale,
                  float* d_out,
                  cudaStream_t stream) {
    constexpr int TILE_ROWS = 32;
    constexpr int TILE_B = 8;
    constexpr int TILE_COLS = 256;

    dim3 block(TILE_ROWS, TILE_B, 1);
    dim3 grid((rows + TILE_ROWS - 1) / TILE_ROWS,
              (batch + TILE_B - 1) / TILE_B,
              1);

    const int cols4 = TILE_COLS / 4;
    const size_t sh_bytes = sizeof(int) * (TILE_ROWS * cols4 + TILE_B * cols4);

    gemm_ternary_kernel<TILE_ROWS, TILE_B, TILE_COLS>
        <<<grid, block, sh_bytes, stream>>>(d_x, d_w, rows, cols, batch, out_scale, d_out);
}

void gemm_ternary_act_quant(const int8_t* d_x, const int8_t* d_w,
                            const float* d_scale_x,
                            int rows, int cols, int batch,
                            float out_scale,
                            int activation,
                            int8_t* d_out_q,
                            cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_act_q_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_scale_x,
                                                      out_scale,
                                                      cols, rows, batch,
                                                      activation,
                                                      d_out_q);
}

void gemm_ternary_gelu_quant(const int8_t* d_x,
                             const int8_t* d_w,
                             const int8_t* d_w_noise,
                             const float* d_scale_x,
                             int rows, int cols, int batch,
                             float out_scale,
                             float noise_scale,
                             float act_scale,
                             int8_t* d_out_q,
                             cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_gelu_q_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_w_noise, d_scale_x,
                                                       out_scale, noise_scale, act_scale,
                                                       cols, rows, batch,
                                                       d_out_q);
}

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
                                 cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_lora_act_q_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_scale_x,
                                                           out_scale,
                                                           cols, rows, batch,
                                                           d_A, d_t, rank,
                                                           lora_scale,
                                                           activation,
                                                           d_out_q);
}

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
                                   cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_sparse_act_q_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_scale_x,
                                                             out_scale,
                                                             cols, rows, batch,
                                                             d_row_offsets, d_col_idx, d_eps,
                                                             sparse_scale,
                                                             activation,
                                                             d_out_q);
}

void gemm_ternary_f(const int8_t* d_x, const int8_t* d_w,
                    const float* d_scale_x,
                    int rows, int cols, int batch,
                    float out_scale,
                    float* d_out,
                    cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_f_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_scale_x,
                                                  out_scale,
                                                  cols, rows, batch,
                                                  d_out);
}

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
                             cudaStream_t stream) {
    const int threads = 256;
    const int blocks = batch;
    size_t shmem = 0;
    shmem = (shmem + 3) & ~static_cast<size_t>(3);
    shmem += static_cast<size_t>(cols) * sizeof(int8_t);
    shmem = (shmem + sizeof(float) - 1) & ~(sizeof(float) - 1);
    shmem += static_cast<size_t>(threads) * 2 * sizeof(float);
    head_gemm_cross_entropy_kernel<<<blocks, threads, shmem, stream>>>(
        d_x, d_w, d_w_noise, d_scale_x, cols, vocab, out_scale, noise_scale,
        d_bias, d_bias_noise, bias_noise_scale, d_targets_t, token_stride, d_loss_accum);
}

void gemm_ternary_f_noise(const int8_t* d_x, const int8_t* d_w,
                          const int8_t* d_w_noise,
                          const float* d_scale_x,
                          int rows, int cols, int batch,
                          float out_scale,
                          float noise_scale,
                          float* d_out,
                          cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_f_noise_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_w_noise, d_scale_x,
                                                        out_scale, noise_scale,
                                                        cols, rows, batch,
                                                        d_out);
}

#if defined(BITNET_EGGROLL_HAS_CUTLASS)
void gemm_ternary_f_cutlass(const int8_t* d_x, const int8_t* d_w,
                            int rows, int cols, int batch,
                            float out_scale,
                            float* d_out,
                            cudaStream_t stream) {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = float;
    using ElementAccumulator = int32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    // CUTLASS doesn't currently provide int8 tensor-op defaults for Sm100 in this version.
    // Use the Sm80 int8 Tensor Core configuration as a fallback on newer GPUs.
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<1, 1, 4>,
        cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2>;

    const int m = batch;
    const int n = rows;
    const int k = cols;
    const int lda = cols;
    const int ldb = cols;
    const int ldc = rows;

    typename Gemm::Arguments args({m, n, k},
                                  {d_x, lda},
                                  {d_w, ldb},
                                  {d_out, ldc},
                                  {d_out, ldc},
                                  {out_scale, 0.0f});
    Gemm gemm_op;
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
        return;
    }
    gemm_op(args, stream);
}

size_t nvfp4_sfa_bytes(int m, int n, int k) {
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    const size_t elems = static_cast<size_t>(cute::size(cute::filter_zeros(layout_sfa)));
    return (elems * cutlass::sizeof_bits<cutlass::float_ue4m3_t>::value + 7) / 8;
}

size_t nvfp4_sfb_bytes(int m, int n, int k) {
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    const size_t elems = static_cast<size_t>(cute::size(cute::filter_zeros(layout_sfb)));
    return (elems * cutlass::sizeof_bits<cutlass::float_ue4m3_t>::value + 7) / 8;
}

void nvfp4_init() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (nvfp4_use_streamk()) {
        (void)Nvfp4GemmAutoSK::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeSK::maximum_active_blocks();
        (void)Nvfp4GemmAutoSKS2::maximum_active_blocks();
        (void)Nvfp4GemmAutoSKS3::maximum_active_blocks();
        (void)Nvfp4GemmAutoSKS4::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeSKS2::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeSKS3::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeSKS4::maximum_active_blocks();
    } else {
        (void)Nvfp4GemmAuto::maximum_active_blocks();
        (void)Nvfp4GemmCooperative::maximum_active_blocks();
        (void)Nvfp4GemmPingpong::maximum_active_blocks();
        (void)Nvfp4GemmAutoS2::maximum_active_blocks();
        (void)Nvfp4GemmAutoS3::maximum_active_blocks();
        (void)Nvfp4GemmAutoS4::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeS2::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeS3::maximum_active_blocks();
        (void)Nvfp4GemmCooperativeS4::maximum_active_blocks();
        (void)Nvfp4GemmPingpongS2::maximum_active_blocks();
        (void)Nvfp4GemmPingpongS3::maximum_active_blocks();
        (void)Nvfp4GemmPingpongS4::maximum_active_blocks();
    }
#endif
}

void nvfp4_set_schedule(Nvfp4Schedule schedule) {
    g_nvfp4_schedule = schedule;
}

void nvfp4_set_quant_mode(Nvfp4QuantMode mode) {
    g_nvfp4_quant_mode = mode;
}

void nvfp4_set_stage_count(Nvfp4StageCount stages) {
    g_nvfp4_stage_count = stages;
}

void nvfp4_set_decomposition(Nvfp4Decomposition mode) {
    g_nvfp4_decomp = mode;
}

void nvfp4_set_verbose(bool verbose) {
    g_nvfp4_verbose = verbose;
}

void nvfp4_set_splits(int splits) {
    g_nvfp4_splits = (splits > 0) ? splits : 1;
}

bool nvfp4_prepare_gemm(const void* d_a,
                        const void* d_b,
                        const void* d_sfa,
                        const void* d_sfb,
                        int m, int n, int k,
                        float alpha,
                        float beta,
                        const float* d_c,
                        float* d_d,
                        cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (m <= 0 || n <= 0 || k <= 0) return false;
    const bool use_streamk = nvfp4_use_streamk();
    if (use_streamk) {
        switch (g_nvfp4_schedule) {
            case bitnet_cuda::Nvfp4Schedule::Pingpong:
            case bitnet_cuda::Nvfp4Schedule::Cooperative:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeSKS2>(d_a, d_b, d_sfa, d_sfb,
                                                                                m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeSKS3>(d_a, d_b, d_sfa, d_sfb,
                                                                                m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeSKS4>(d_a, d_b, d_sfa, d_sfb,
                                                                                m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeSK>(d_a, d_b, d_sfa, d_sfb,
                                                                               m, n, k, alpha, beta, d_c, d_d, stream);
                }
            case bitnet_cuda::Nvfp4Schedule::Auto:
            default:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoSKS2>(d_a, d_b, d_sfa, d_sfb,
                                                                         m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoSKS3>(d_a, d_b, d_sfa, d_sfb,
                                                                         m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoSKS4>(d_a, d_b, d_sfa, d_sfb,
                                                                         m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoSK>(d_a, d_b, d_sfa, d_sfb,
                                                                       m, n, k, alpha, beta, d_c, d_d, stream);
                }
        }
    } else {
        switch (g_nvfp4_schedule) {
            case bitnet_cuda::Nvfp4Schedule::Pingpong:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmPingpongS2>(d_a, d_b, d_sfa, d_sfb,
                                                                           m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmPingpongS3>(d_a, d_b, d_sfa, d_sfb,
                                                                           m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmPingpongS4>(d_a, d_b, d_sfa, d_sfb,
                                                                           m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmPingpong>(d_a, d_b, d_sfa, d_sfb,
                                                                         m, n, k, alpha, beta, d_c, d_d, stream);
                }
            case bitnet_cuda::Nvfp4Schedule::Cooperative:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeS2>(d_a, d_b, d_sfa, d_sfb,
                                                                              m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeS3>(d_a, d_b, d_sfa, d_sfb,
                                                                              m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperativeS4>(d_a, d_b, d_sfa, d_sfb,
                                                                              m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmCooperative>(d_a, d_b, d_sfa, d_sfb,
                                                                             m, n, k, alpha, beta, d_c, d_d, stream);
                }
            case bitnet_cuda::Nvfp4Schedule::Auto:
            default:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoS2>(d_a, d_b, d_sfa, d_sfb,
                                                                       m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoS3>(d_a, d_b, d_sfa, d_sfb,
                                                                       m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAutoS4>(d_a, d_b, d_sfa, d_sfb,
                                                                       m, n, k, alpha, beta, d_c, d_d, stream);
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        return nvfp4_prepare_gemm_impl<Nvfp4GemmAuto>(d_a, d_b, d_sfa, d_sfb,
                                                                     m, n, k, alpha, beta, d_c, d_d, stream);
                }
        }
    }
#else
    (void)d_a; (void)d_b; (void)d_sfa; (void)d_sfb;
    (void)m; (void)n; (void)k; (void)alpha; (void)beta;
    (void)d_c; (void)d_d; (void)stream;
    return false;
#endif
}

void nvfp4_quantize_a(const int8_t* d_in,
                      int m, int n, int k_in, int k_out,
                      void* d_out,
                      void* d_sfa,
                      cudaStream_t stream) {
    if (m <= 0 || k_out <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k_out, 1));
    dim3 grid(static_cast<unsigned>(m),
              static_cast<unsigned>((k_out + kNvfp4SfVec - 1) / kNvfp4SfVec),
              1);
    if (g_nvfp4_quant_mode == bitnet_cuda::Nvfp4QuantMode::Warp4) {
        nvfp4_quantize_rowmajor_warp4_kernel<<<grid, 32, 0, stream>>>(
            d_in, m, k_in, k_out,
            static_cast<cutlass::float_e2m1_t*>(d_out),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    } else {
        nvfp4_quantize_rowmajor_warp16_kernel<<<grid, 32, 0, stream>>>(
            d_in, m, k_in, k_out,
            static_cast<cutlass::float_e2m1_t*>(d_out),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    }
}

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
                                           cudaStream_t stream) {
    if (batch <= 0 || hidden <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(batch, 1, hidden, 1));
    dim3 grid(static_cast<unsigned>(batch),
              static_cast<unsigned>((hidden + kNvfp4SfVec - 1) / kNvfp4SfVec),
              1);
    if (g_nvfp4_quant_mode == bitnet_cuda::Nvfp4QuantMode::Warp4) {
        embedding_lookup_noise_quantize_nvfp4_warp4_kernel<<<grid, 32, 0, stream>>>(
            d_tokens_t,
            token_stride,
            d_emb_w,
            d_emb_noise,
            hidden,
            batch,
            noise_scale,
            anti_sign,
            d_out_q,
            static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    } else {
        embedding_lookup_noise_quantize_nvfp4_warp16_kernel<<<grid, 32, 0, stream>>>(
            d_tokens_t,
            token_stride,
            d_emb_w,
            d_emb_noise,
            hidden,
            batch,
            noise_scale,
            anti_sign,
            d_out_q,
            static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    }
}

void activation_quantize_nvfp4(const float* d_in,
                               int rows,
                               int cols_in,
                               int cols_out,
                               int activation,
                               int8_t* d_out_q,
                               void* d_out_nvfp4,
                               void* d_sfa,
                               cudaStream_t stream) {
    if (rows <= 0 || cols_out <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, cols_out, 1));
    dim3 grid(static_cast<unsigned>(rows),
              static_cast<unsigned>((cols_out + kNvfp4SfVec - 1) / kNvfp4SfVec),
              1);
    if (g_nvfp4_quant_mode == bitnet_cuda::Nvfp4QuantMode::Warp4) {
        nvfp4_quantize_rowmajor_activation_warp4_kernel<<<grid, 32, 0, stream>>>(
            d_in, rows, cols_in, cols_out,
            activation,
            d_out_q,
            static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    } else {
        nvfp4_quantize_rowmajor_activation_warp16_kernel<<<grid, 32, 0, stream>>>(
            d_in, rows, cols_in, cols_out,
            activation,
            d_out_q,
            static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    }
}

void gelu_quantize_nvfp4(const float* d_in,
                         int rows,
                         int cols_in,
                         int cols_out,
                         float act_scale,
                         int8_t* d_out_q,
                         void* d_out_nvfp4,
                         void* d_sfa,
                         cudaStream_t stream) {
    if (rows <= 0 || cols_out <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, cols_out, 1));
    dim3 grid(static_cast<unsigned>(rows),
              static_cast<unsigned>((cols_out + kNvfp4SfVec - 1) / kNvfp4SfVec),
              1);
    if (g_nvfp4_quant_mode == bitnet_cuda::Nvfp4QuantMode::Warp4) {
        nvfp4_quantize_rowmajor_gelu_warp4_kernel<<<grid, 32, 0, stream>>>(
            d_in, rows, cols_in, cols_out,
            act_scale,
            d_out_q,
            static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    } else {
        nvfp4_quantize_rowmajor_gelu_warp16_kernel<<<grid, 32, 0, stream>>>(
            d_in, rows, cols_in, cols_out,
            act_scale,
            d_out_q,
            static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
            static_cast<cutlass::float_ue4m3_t*>(d_sfa),
            layout_sfa);
    }
}

void absmean_norm_q_nvfp4(const int8_t* d_in,
                          int hidden,
                          int batch,
                          int cols_out,
                          float scale,
                          int8_t* d_out_q,
                          void* d_out_nvfp4,
                          void* d_sfa,
                          cudaStream_t stream) {
    if (batch <= 0 || cols_out <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(batch, 1, cols_out, 1));
    const int threads = 256;
    absmean_norm_q_nvfp4_kernel<<<batch, threads, 0, stream>>>(
        d_in, hidden, cols_out, scale,
        d_out_q,
        static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
        static_cast<cutlass::float_ue4m3_t*>(d_sfa),
        layout_sfa);
}

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
                                             cudaStream_t stream) {
    if (batch <= 0 || cols_out <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(batch, 1, cols_out, 1));
    const int threads = 256;
    add_scaled_to_int8_absmean_norm_q_nvfp4_kernel<<<batch, threads, 0, stream>>>(
        d_inout, d_in, hidden, cols_out, add_scale, norm_scale,
        d_out_q,
        static_cast<cutlass::float_e2m1_t*>(d_out_nvfp4),
        static_cast<cutlass::float_ue4m3_t*>(d_sfa),
        layout_sfa);
}

void nvfp4_quantize_b(const int8_t* d_in,
                      int m, int n, int k_in, int k_out,
                      void* d_out,
                      void* d_sfb,
                      cudaStream_t stream) {
    if (n <= 0 || k_out <= 0) return;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k_out, 1));
    dim3 grid(static_cast<unsigned>(n),
              static_cast<unsigned>((k_out + kNvfp4SfVec - 1) / kNvfp4SfVec),
              1);
    if (g_nvfp4_quant_mode == bitnet_cuda::Nvfp4QuantMode::Warp4) {
        nvfp4_quantize_rowmajor_warp4_kernel<<<grid, 32, 0, stream>>>(
            d_in, n, k_in, k_out,
            static_cast<cutlass::float_e2m1_t*>(d_out),
            static_cast<cutlass::float_ue4m3_t*>(d_sfb),
            layout_sfb);
    } else {
        nvfp4_quantize_rowmajor_warp16_kernel<<<grid, 32, 0, stream>>>(
            d_in, n, k_in, k_out,
            static_cast<cutlass::float_e2m1_t*>(d_out),
            static_cast<cutlass::float_ue4m3_t*>(d_sfb),
            layout_sfb);
    }
}

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
                               cudaStream_t stream) {
    if (hidden <= 0) return;
    const int rows_valid = hidden * 3 + 1;
    if (rows_total < rows_valid) rows_total = rows_valid;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SfVec>;
    auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(batch, rows_total, k_out, 1));
    dim3 grid(static_cast<unsigned>(rows_valid),
              static_cast<unsigned>((k_out + kNvfp4SfVec - 1) / kNvfp4SfVec),
              1);
    if (g_nvfp4_quant_mode == bitnet_cuda::Nvfp4QuantMode::Warp4) {
        nvfp4_quantize_wqkv_fused_warp4_kernel<<<grid, 32, 0, stream>>>(
            d_wq, d_wk, d_wv, d_wb,
            hidden, k_out, rows_valid,
            q_scale, k_scale, v_scale, b_scale,
            static_cast<cutlass::float_e2m1_t*>(d_out),
            static_cast<cutlass::float_ue4m3_t*>(d_sfb),
            layout_sfb);
    } else {
        nvfp4_quantize_wqkv_fused_warp16_kernel<<<grid, 32, 0, stream>>>(
            d_wq, d_wk, d_wv, d_wb,
            hidden, k_out, rows_valid,
            q_scale, k_scale, v_scale, b_scale,
            static_cast<cutlass::float_e2m1_t*>(d_out),
            static_cast<cutlass::float_ue4m3_t*>(d_sfb),
            layout_sfb);
    }
}

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
                      cudaStream_t stream) {
    if (batch <= 0 || hidden <= 0) return;
    if (use_vec4 && (hidden & 3) == 0) {
        constexpr int kTx = 32;
        constexpr int kTy = 4;
        dim3 block(kTx, kTy, 1);
        const int cols4 = hidden / 4;
        dim3 grid((cols4 + kTx - 1) / kTx,
                  (batch + kTy - 1) / kTy,
                  1);
        split_qkvb_fused_vec4_kernel<<<grid, block, 0, stream>>>(
            d_in, batch, hidden, fused_cols, d_q, d_k, d_v, d_beta, beta_bias);
    } else {
        constexpr int kTx = 16;
        constexpr int kTy = 8;
        dim3 block(kTx, kTy, 1);
        dim3 grid((hidden + kTx - 1) / kTx,
                  (batch + kTy - 1) / kTy,
                  1);
        split_qkvb_fused_kernel<<<grid, block, 0, stream>>>(
            d_in, batch, hidden, fused_cols, d_q, d_k, d_v, d_beta, beta_bias);
    }
}

void gemm_ternary_f_nvfp4(const void* d_a,
                          const void* d_b,
                          const void* d_sfa,
                          const void* d_sfb,
                          int m, int n, int k,
                          float alpha,
                          float beta,
                          const float* d_c,
                          float* d_d,
                          cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    const bool use_streamk = nvfp4_use_streamk();
    if (use_streamk) {
        switch (g_nvfp4_schedule) {
            case bitnet_cuda::Nvfp4Schedule::Pingpong:
            case bitnet_cuda::Nvfp4Schedule::Cooperative:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeSKS2>(d_a, d_b, d_sfa, d_sfb,
                                                                      m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeSKS3>(d_a, d_b, d_sfa, d_sfb,
                                                                      m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeSKS4>(d_a, d_b, d_sfa, d_sfb,
                                                                      m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeSK>(d_a, d_b, d_sfa, d_sfb,
                                                                    m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                }
                break;
            case bitnet_cuda::Nvfp4Schedule::Auto:
            default:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoSKS2>(d_a, d_b, d_sfa, d_sfb,
                                                               m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoSKS3>(d_a, d_b, d_sfa, d_sfb,
                                                               m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoSKS4>(d_a, d_b, d_sfa, d_sfb,
                                                               m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoSK>(d_a, d_b, d_sfa, d_sfb,
                                                             m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                }
                break;
        }
    } else {
        switch (g_nvfp4_schedule) {
            case bitnet_cuda::Nvfp4Schedule::Pingpong:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        nvfp4_gemm_run_impl<Nvfp4GemmPingpongS2>(d_a, d_b, d_sfa, d_sfb,
                                                                m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        nvfp4_gemm_run_impl<Nvfp4GemmPingpongS3>(d_a, d_b, d_sfa, d_sfb,
                                                                m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        nvfp4_gemm_run_impl<Nvfp4GemmPingpongS4>(d_a, d_b, d_sfa, d_sfb,
                                                                m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        nvfp4_gemm_run_impl<Nvfp4GemmPingpong>(d_a, d_b, d_sfa, d_sfb,
                                                              m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                }
                break;
            case bitnet_cuda::Nvfp4Schedule::Cooperative:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeS2>(d_a, d_b, d_sfa, d_sfb,
                                                                    m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeS3>(d_a, d_b, d_sfa, d_sfb,
                                                                    m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperativeS4>(d_a, d_b, d_sfa, d_sfb,
                                                                    m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        nvfp4_gemm_run_impl<Nvfp4GemmCooperative>(d_a, d_b, d_sfa, d_sfb,
                                                                  m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                }
                break;
            case bitnet_cuda::Nvfp4Schedule::Auto:
            default:
                switch (g_nvfp4_stage_count) {
                    case bitnet_cuda::Nvfp4StageCount::Stages2:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoS2>(d_a, d_b, d_sfa, d_sfb,
                                                             m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages3:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoS3>(d_a, d_b, d_sfa, d_sfb,
                                                             m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Stages4:
                        nvfp4_gemm_run_impl<Nvfp4GemmAutoS4>(d_a, d_b, d_sfa, d_sfb,
                                                             m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                    case bitnet_cuda::Nvfp4StageCount::Auto:
                    default:
                        nvfp4_gemm_run_impl<Nvfp4GemmAuto>(d_a, d_b, d_sfa, d_sfb,
                                                           m, n, k, alpha, beta, d_c, d_d, stream);
                        break;
                }
                break;
        }
    }
#else
    (void)d_a; (void)d_b; (void)d_sfa; (void)d_sfb;
    (void)m; (void)n; (void)k; (void)alpha; (void)beta; (void)d_c; (void)d_d; (void)stream;
#endif
}
#else
void gemm_ternary_f_cutlass(const int8_t* d_x, const int8_t* d_w,
                            int rows, int cols, int batch,
                            float out_scale,
                            float* d_out,
                            cudaStream_t stream) {
    (void)d_x; (void)d_w; (void)rows; (void)cols; (void)batch; (void)out_scale; (void)d_out; (void)stream;
}

size_t nvfp4_sfa_bytes(int m, int n, int k) {
    (void)m; (void)n; (void)k;
    return 0;
}

size_t nvfp4_sfb_bytes(int m, int n, int k) {
    (void)m; (void)n; (void)k;
    return 0;
}

void nvfp4_quantize_a(const int8_t* d_in,
                      int m, int n, int k_in, int k_out,
                      void* d_out,
                      void* d_sfa,
                      cudaStream_t stream) {
    (void)d_in; (void)m; (void)n; (void)k_in; (void)k_out; (void)d_out; (void)d_sfa; (void)stream;
}

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
                                           cudaStream_t stream) {
    (void)d_tokens_t; (void)token_stride; (void)d_emb_w; (void)d_emb_noise;
    (void)hidden; (void)batch; (void)noise_scale; (void)anti_sign;
    (void)d_out_q; (void)d_out_nvfp4; (void)d_sfa; (void)stream;
}

void activation_quantize_nvfp4(const float* d_in,
                               int rows,
                               int cols_in,
                               int cols_out,
                               int activation,
                               int8_t* d_out_q,
                               void* d_out_nvfp4,
                               void* d_sfa,
                               cudaStream_t stream) {
    (void)d_in; (void)rows; (void)cols_in; (void)cols_out; (void)activation;
    (void)d_out_q; (void)d_out_nvfp4; (void)d_sfa; (void)stream;
}

void gelu_quantize_nvfp4(const float* d_in,
                         int rows,
                         int cols_in,
                         int cols_out,
                         float act_scale,
                         int8_t* d_out_q,
                         void* d_out_nvfp4,
                         void* d_sfa,
                         cudaStream_t stream) {
    (void)d_in; (void)rows; (void)cols_in; (void)cols_out; (void)act_scale;
    (void)d_out_q; (void)d_out_nvfp4; (void)d_sfa; (void)stream;
}

void absmean_norm_q_nvfp4(const int8_t* d_in,
                          int hidden,
                          int batch,
                          int cols_out,
                          float scale,
                          int8_t* d_out_q,
                          void* d_out_nvfp4,
                          void* d_sfa,
                          cudaStream_t stream) {
    (void)d_in; (void)hidden; (void)batch; (void)cols_out; (void)scale;
    (void)d_out_q; (void)d_out_nvfp4; (void)d_sfa; (void)stream;
}

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
                                             cudaStream_t stream) {
    (void)d_inout; (void)d_in; (void)hidden; (void)batch; (void)cols_out;
    (void)add_scale; (void)norm_scale; (void)d_out_q;
    (void)d_out_nvfp4; (void)d_sfa; (void)stream;
}

void nvfp4_quantize_b(const int8_t* d_in,
                      int m, int n, int k_in, int k_out,
                      void* d_out,
                      void* d_sfb,
                      cudaStream_t stream) {
    (void)d_in; (void)m; (void)n; (void)k_in; (void)k_out; (void)d_out; (void)d_sfb; (void)stream;
}

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
                               cudaStream_t stream) {
    (void)d_wq; (void)d_wk; (void)d_wv; (void)d_wb;
    (void)batch; (void)hidden; (void)k_out; (void)rows_total; (void)q_scale; (void)k_scale;
    (void)v_scale; (void)b_scale; (void)d_out; (void)d_sfb; (void)stream;
}

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
                      cudaStream_t stream) {
    (void)d_in; (void)batch; (void)hidden; (void)fused_cols;
    (void)d_q; (void)d_k; (void)d_v; (void)d_beta;
    (void)beta_bias; (void)use_vec4; (void)stream;
}

void gemm_ternary_f_nvfp4(const void* d_a,
                          const void* d_b,
                          const void* d_sfa,
                          const void* d_sfb,
                          int m, int n, int k,
                          float alpha,
                          float beta,
                          const float* d_c,
                          float* d_d,
                          cudaStream_t stream) {
    (void)d_a; (void)d_b; (void)d_sfa; (void)d_sfb;
    (void)m; (void)n; (void)k; (void)alpha; (void)beta; (void)d_c; (void)d_d; (void)stream;
}
#endif

void gemm_ternary_f_noise_tiled(const int8_t* d_x, const int8_t* d_w,
                                const int8_t* d_w_noise,
                                const float* d_scale_x,
                                int rows, int cols, int batch,
                                float out_scale,
                                float noise_scale,
                                float* d_out,
                                cudaStream_t stream) {
    constexpr int TILE_ROWS = 16;
    constexpr int TILE_B = 8;
    constexpr int TILE_COLS = 256;
    if ((cols % 16) != 0) {
        gemm_ternary_f_noise(d_x, d_w, d_w_noise, d_scale_x,
                             rows, cols, batch,
                             out_scale, noise_scale,
                             d_out, stream);
        return;
    }

    dim3 block(TILE_ROWS, TILE_B, 1);
    dim3 grid((rows + TILE_ROWS - 1) / TILE_ROWS,
              (batch + TILE_B - 1) / TILE_B,
              1);
    const size_t sh_bytes = static_cast<size_t>(TILE_ROWS * TILE_COLS * 2 + TILE_B * TILE_COLS);
    const bool use_noise = (d_w_noise != nullptr) && (noise_scale != 0.0f);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const bool use_cp_async = true;
#else
    const bool use_cp_async = false;
#endif
    gemm_f_noise_tiled_kernel<TILE_ROWS, TILE_B, TILE_COLS>
        <<<grid, block, sh_bytes, stream>>>(d_x, d_w, d_w_noise, d_scale_x,
                                            out_scale, noise_scale,
                                            rows, cols, batch,
                                            use_noise, use_cp_async,
                                            d_out);
}

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
                     cudaStream_t stream) {
    const int beta_rows = (d_beta != nullptr && d_wb != nullptr) ? 1 : 0;
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * (rows + beta_rows);
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_qkvb_fused_kernel<<<blocks, threads, 0, stream>>>(
        d_x,
        d_wq, d_wk, d_wv, d_wb,
        d_wq_noise, d_wk_noise, d_wv_noise, d_wb_noise,
        d_scale_x,
        q_out_scale, k_out_scale, v_out_scale, b_out_scale,
        noise_scale,
        cols, rows, batch,
        beta_rows,
        d_q, d_k, d_v, d_beta);
}

void pack_ternary_i2(const int8_t* d_w,
                     int rows, int cols,
                     uint32_t* d_w_packed,
                     cudaStream_t stream) {
    const int pack_cols = (cols + 15) >> 4;
    const int total = rows * pack_cols;
    const int threads = kGemmThreads;
    const int blocks = (total + threads - 1) / threads;
    pack_ternary_i2_kernel<<<blocks, threads, 0, stream>>>(d_w, rows, cols, d_w_packed);
}

void pack_ternary_i2_bitnet(const int8_t* d_w,
                            int rows, int cols,
                            uint32_t* d_w_packed,
                            cudaStream_t stream) {
    const int pack_cols = (cols + 15) >> 4;
    const int total = rows * pack_cols;
    const int threads = kGemmThreads;
    const int blocks = (total + threads - 1) / threads;
    pack_ternary_i2_bitnet_kernel<<<blocks, threads, 0, stream>>>(d_w, rows, cols, d_w_packed);
}

void init_i2_lut() {
    init_i2_lut_once();
}

void gemm_ternary_f_i2(const int8_t* d_x, const uint32_t* d_w_packed,
                       const float* d_scale_x,
                       int rows, int cols, int batch,
                       float out_scale,
                       float* d_out,
                       cudaStream_t stream) {
    init_i2_lut_once();
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_f_i2_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w_packed, d_scale_x,
                                                     out_scale,
                                                     cols, rows, batch,
                                                     d_out);
}

void gemm_ternary_f_i2_noise(const int8_t* d_x, const uint32_t* d_w_packed,
                             const uint32_t* d_w_packed_noise,
                             const float* d_scale_x,
                             int rows, int cols, int batch,
                             float out_scale,
                             float noise_scale,
                             float* d_out,
                             cudaStream_t stream) {
    init_i2_lut_once();
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_f_i2_noise_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w_packed, d_w_packed_noise, d_scale_x,
                                                           out_scale, noise_scale,
                                                           cols, rows, batch,
                                                           d_out);
}

void gemm_ternary_f_i2_bitnet(const int8_t* d_x, const uint32_t* d_w_packed,
                              const float* d_scale_x,
                              int rows, int cols, int batch,
                              float out_scale,
                              float* d_out,
                              cudaStream_t stream) {
    init_i2_lut_once();
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_f_i2_bitnet_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w_packed, d_scale_x,
                                                            out_scale,
                                                            cols, rows, batch,
                                                            d_out);
}

void gemm_ternary_f_i2_bitnet_noise(const int8_t* d_x, const uint32_t* d_w_packed,
                                    const uint32_t* d_w_packed_noise,
                                    const float* d_scale_x,
                                    int rows, int cols, int batch,
                                    float out_scale,
                                    float noise_scale,
                                    float* d_out,
                                    cudaStream_t stream) {
    init_i2_lut_once();
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_f_i2_bitnet_noise_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w_packed, d_w_packed_noise, d_scale_x,
                                                                  out_scale, noise_scale,
                                                                  cols, rows, batch,
                                                                  d_out);
}

void gemm_ternary_lora_f(const int8_t* d_x, const int8_t* d_w,
                         const float* d_scale_x,
                         int rows, int cols, int batch,
                         float out_scale,
                         const int8_t* d_A,
                         const int32_t* d_t,
                         int rank,
                         float lora_scale,
                         float* d_out,
                         cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_lora_f_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_scale_x,
                                                       out_scale,
                                                       cols, rows, batch,
                                                       d_A, d_t, rank,
                                                       lora_scale,
                                                       d_out);
}

void gemm_ternary_sparse_f(const int8_t* d_x, const int8_t* d_w,
                           const float* d_scale_x,
                           int rows, int cols, int batch,
                           float out_scale,
                           const int32_t* d_row_offsets,
                           const int32_t* d_col_idx,
                           const int8_t* d_eps,
                           float sparse_scale,
                           float* d_out,
                           cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rows;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    gemm_sparse_f_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_scale_x,
                                                         out_scale,
                                                         cols, rows, batch,
                                                         d_row_offsets, d_col_idx, d_eps,
                                                         sparse_scale,
                                                         d_out);
}

void lora_compute_t(const int8_t* d_x, const int8_t* d_B,
                    int cols, int batch, int rank,
                    int32_t* d_t,
                    cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int warps_per_block = threads / 32;
    const int total_warps = batch * rank;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    lora_compute_t_kernel<<<blocks, threads, 0, stream>>>(d_x, d_B, cols, batch, rank, d_t);
}

void lora_add(const int32_t* d_t, const int8_t* d_A,
              int rows, int batch, int rank,
              float scale,
              float* d_out,
              cudaStream_t stream) {
    dim3 block(128, 2, 1);
    dim3 grid((rows + block.x - 1) / block.x,
              (batch + block.y - 1) / block.y,
              1);
    lora_add_kernel<<<grid, block, 0, stream>>>(d_t, d_A, rows, batch, rank, scale, d_out);
}

void sparse_add(const int32_t* d_row_offsets,
                const int32_t* d_col_idx,
                const int8_t* d_eps,
                int rows, int cols, int batch,
                const int8_t* d_x,
                float scale,
                float* d_out,
                cudaStream_t stream) {
    dim3 block(128, 2, 1);
    dim3 grid((rows + block.x - 1) / block.x,
              (batch + block.y - 1) / block.y,
              1);
    sparse_add_kernel<<<grid, block, 0, stream>>>(d_row_offsets, d_col_idx, d_eps,
                                                  rows, cols, batch, d_x, scale, d_out);
}

void activation_quantize(const float* d_in,
                         int8_t* d_out,
                         int n,
                         int activation,
                         cudaStream_t stream) {
    const int threads = kGemmThreads;
    const int blocks = (n + threads - 1) / threads;
    activation_quantize_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n, activation);
}

void embed_update_plain(const uint8_t* d_tokens_t,
                        int token_stride,
                        const int8_t* d_emb_w,
                        int hidden,
                        int batch,
                        int8_t* d_state,
                        cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden + block.x - 1) / block.x,
              batch,
              1);
    embed_update_plain_kernel<<<grid, block, 0, stream>>>(d_tokens_t, token_stride, d_emb_w,
                                                          hidden, batch, d_state);
}

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
                       cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden + block.x - 1) / block.x,
              batch,
              1);
    embed_update_lora_kernel<<<grid, block, 0, stream>>>(d_tokens_t, token_stride, d_emb_w, d_A, d_B,
                                                         hidden, batch, rank, noise_scale, d_state);
}

void embed_update_sparse(const uint8_t* d_tokens_t,
                         int token_stride,
                         const int8_t* d_emb_w,
                         const int16_t* d_row_noise,
                         int hidden,
                         int batch,
                         float noise_scale,
                         int8_t* d_state,
                         cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden + block.x - 1) / block.x,
              batch,
              1);
    embed_update_sparse_kernel<<<grid, block, 0, stream>>>(d_tokens_t, token_stride, d_emb_w, d_row_noise,
                                                           hidden, batch, noise_scale, d_state);
}

} // namespace bitnet_cuda
