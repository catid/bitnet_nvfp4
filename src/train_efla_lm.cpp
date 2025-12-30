#include "cuda_kernels.h"
#include "dataset.h"
#include "efla_lm_kernels.h"
#include "rng.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef BITNET_EGGROLL_HAS_NVTX
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#else
#error "NVTX enabled but nvToolsExt header not found"
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <numbers>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

static inline void cuda_check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::string msg = std::string("CUDA error: ") + what + ": " + cudaGetErrorString(err);
    throw std::runtime_error(msg);
}

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) {
        if (threads == 0) threads = 1;
        workers_.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    template <class F>
    std::future<void> enqueue(F&& f) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        std::future<void> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};


#ifdef BITNET_EGGROLL_HAS_NVTX
struct NvtxRange {
    explicit NvtxRange(bool enabled, const char* name) : enabled_(enabled) {
        if (enabled_) nvtxRangePushA(name);
    }
    ~NvtxRange() {
        if (enabled_) nvtxRangePop();
    }
    bool enabled_;
};
#else
struct NvtxRange {
    explicit NvtxRange(bool, const char*) {}
};
#endif

static inline int8_t clip_ternary(int v) {
    if (v > 1) return 1;
    if (v < -1) return -1;
    return static_cast<int8_t>(v);
}

static inline int8_t clip_int8(int v) {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return static_cast<int8_t>(v);
}

static inline int8_t noise_hash(uint64_t seed, uint64_t idx, int anti_sign,
                                bool use_clt, int clt_k) {
    if (!use_clt) return rng::ternary_hash(seed, idx, anti_sign);
    int k = std::max(1, clt_k);
    uint64_t st = rng::mix(seed, idx);
    int sum = 0;
    for (int i = 0; i < k; ++i) sum += rng::ternary_from_u64(rng::splitmix64(st));
    return clip_int8(sum * anti_sign);
}

static float cross_entropy_loss_token(const float* logits,
                                      const float* bias,
                                      const float* bias_noise,
                                      float bias_noise_scale,
                                      int vocab,
                                      int y) {
    auto at = [&](int i) -> float {
        float v = logits[i];
        if (bias) v += bias[i];
        if (bias_noise) v += bias_noise_scale * bias_noise[i];
        return v;
    };

    float maxv = at(0);
    for (int i = 1; i < vocab; ++i) maxv = std::max(maxv, at(i));
    float sumexp = 0.0f;
    for (int i = 0; i < vocab; ++i) sumexp += std::exp(at(i) - maxv);
    float logprob = at(y) - maxv - std::log(sumexp + 1e-9f);
    return -logprob;
}

struct TernaryMatrix {
    int rows = 0;
    int cols = 0;
    float out_scale = 1.0f;
    std::vector<int8_t> w;

    void init_rand(int rows_, int cols_, uint64_t seed) {
        rows = rows_;
        cols = cols_;
        w.resize(static_cast<size_t>(rows) * cols);
        uint64_t st = seed;
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = rng::ternary_from_u64(rng::splitmix64(st));
        }
        out_scale = 1.0f / std::sqrt(static_cast<float>(cols));
    }
};

struct TrainConfig {
    std::string data_path = "data/tinyshakespeare.txt";
    int device = 0;
    std::vector<int> devices; // optional multi-GPU device list
    int gpu_workers = 6; // workers per GPU device
    int update_streams = 2; // concurrent GPU update streams per device
    int epochs = 50;
    int population = 2048; // even
    int batch = 32;
    int seq_len = 128;
    int hidden = 512;
    int layers = 4;
    int mlp_mult = 2;
    int vocab = 256;
    int sigma_shift = 3;
    int sigma_shift_end = 4;
    int state_dim = 0; // 0 = full HxH state, >0 = low-rank D x H state

    bool use_clt_noise = true; // CLT/approx-Gaussian perturbations
    int clt_k = 4;             // number of ternary draws to sum
    bool use_gpu_noise = true;
    bool use_gpu_update = true;
    bool use_cuda_graph = false;
    bool graph_full_eval = true;
    bool use_nvtx = false;
    bool use_fused_qkv = true;
    bool use_qkv_split_kernel = true;
    int qkv_split_vec4 = -1; // -1 = auto, 0 = off, 1 = on
    bool use_qkv_split_bias = false;
    bool use_fused_ff1 = true;
    bool use_fused_head_loss = false;
    bool use_fused_noise_gemm = true;
    bool use_fused_efla = false;
    bool use_efla_mixed = false;
    bool use_efla_fuse_diff = false;
    bool use_efla_update_wmma = false;
    bool use_tiled_gemm = false;
    bool use_packed_gemm = false;
    bool use_packed_gemm_bitnet = false;
    bool use_packed_update = false;
    int omp_threads = 0; // 0 = auto (use all available cores unless OMP_NUM_THREADS is set)
    bool noise_batch = true;
    bool use_pinned_tokens = true;
    bool adaptive_pop = true; // ramp up population over epochs (12% faster, 1.6% PPL loss)

    float update_threshold = 0.008f;
    float update_threshold_end = 0.008f;

    float lr = 0.80f;     // optimizer lr / step size
    float lr_end = 0.05f; // optional schedule target

    float act_scale = 0.8f; // applied after GELU before quantize
    float mlp_act_scale = 1.0f; // applied after GELU in MLP before quantize
    float ln_scale = 1.0f;  // applied after LayerNorm before quantize
    float residual_scale = 1.0f; // L==1: scales x in LN(attn(x)+s*x); L>=2: scales attn in x += s*attn(LN(x))
    float mlp_residual_scale = 1.0f; // scales MLP in x += s*mlp(LN(x))
    float state_decay = 0.95f; // low-rank state decay
    float gate_scale = 0.1f;   // low-rank gate scale
    bool int8_residual = true; // keep residual stream in int8
    bool absmean_norm = true;  // use abs-mean norm on int8 (requires int8_residual)

    bool use_momentum = false;
    float momentum_beta = 0.9f;

    bool use_adam = false;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-3f;

    bool use_shadow = true;

    bool use_adaptive = true;
    float adaptive_beta = 0.95f;
    float adaptive_eps = 1e-3f;

    enum class FitnessMode { Sign = 0, ZScore = 1, CenteredRank = 2 };
    FitnessMode fitness_mode = FitnessMode::CenteredRank;
    float fitness_clip = 3.0f;

    enum class GemmBackend { Dp4a = 0, Cutlass = 1, CutlassNvfp4 = 2 };
    GemmBackend gemm_backend = GemmBackend::CutlassNvfp4;
    bitnet_cuda::Nvfp4Schedule nvfp4_schedule = bitnet_cuda::Nvfp4Schedule::Pingpong;
    bitnet_cuda::Nvfp4QuantMode nvfp4_quant_mode = bitnet_cuda::Nvfp4QuantMode::Warp16;
    bitnet_cuda::Nvfp4StageCount nvfp4_stage_count = bitnet_cuda::Nvfp4StageCount::Stages2;
    bitnet_cuda::Nvfp4Decomposition nvfp4_decomp = bitnet_cuda::Nvfp4Decomposition::SplitK;
    int nvfp4_splits = 4;
    bool nvfp4_autotune = false;
    bool nvfp4_tune_schedule = false;
    bool nvfp4_tune_quant = false;
    bool nvfp4_tune_stages = false;
    bool nvfp4_tune_decomp = false;
    bool nvfp4_tune_splits = false;

    enum class Schedule { Constant = 0, Linear = 1, Cosine = 2, Exp = 3 };
    Schedule lr_schedule = Schedule::Cosine;
    Schedule thresh_schedule = Schedule::Exp;
    Schedule sigma_schedule = Schedule::Exp;

    bool state_fp16 = true;

    bool train_pos = true;
    float pos_lr_mult = 1.0f;
    float pos_thresh_mult = 1.0f;


    bool do_val = true;
    uint64_t val_seed = 12345;
    int val_every = 1;

    bool fixed_train = false;
    uint64_t data_seed = 0; // 0 => inherit from --seed

    uint64_t seed = 1;
    enum class Method { DeltaNet = 0, EFLA = 1 };
    Method method = Method::EFLA;
};

static void configure_openmp(int omp_threads) {
#ifdef _OPENMP
    if (omp_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(omp_threads);
    } else {
        omp_set_dynamic(0);
        const int procs = omp_get_num_procs();
        if (procs > 0) omp_set_num_threads(procs);
    }
    const int max_threads = omp_get_max_threads();
    std::cout << "OpenMP threads " << max_threads << "\n";
#else
    (void)omp_threads;
#endif
}

static uint64_t count_parameters(const TrainConfig& cfg) {
    const uint64_t V = static_cast<uint64_t>(cfg.vocab);
    const uint64_t H = static_cast<uint64_t>(cfg.hidden);
    const uint64_t L = static_cast<uint64_t>(cfg.layers);
    const uint64_t M = static_cast<uint64_t>(std::max(1, cfg.mlp_mult));
    const uint64_t F = M * H;
    const uint64_t T = static_cast<uint64_t>(cfg.seq_len);

    uint64_t total = 0;
    total += V * H; // emb
    total += H * H; // win
    total += L * (3 * H * H + H); // wq/wk/wv + wbeta
    total += L * (2 * H * F); // FFN: w1=[F][H], w2=[H][F]
    total += V * H; // head
    total += V;     // head_bias
    total += L;     // beta_bias per layer
    if (cfg.train_pos) total += T * H; // pos embedding
    return total;
}

static TrainConfig::Method parse_method(const std::string& s) {
    if (s == "DeltaNet" || s == "deltanet" || s == "delta") return TrainConfig::Method::DeltaNet;
    if (s == "efla" || s == "EFLA") return TrainConfig::Method::EFLA;
    return TrainConfig::Method::EFLA;
}

static std::vector<int> parse_devices(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

static TrainConfig::FitnessMode parse_fitness_mode(const std::string& s) {
    if (s == "sign") return TrainConfig::FitnessMode::Sign;
    if (s == "zscore") return TrainConfig::FitnessMode::ZScore;
    if (s == "centered_rank" || s == "rank") return TrainConfig::FitnessMode::CenteredRank;
    return TrainConfig::FitnessMode::CenteredRank;
}

static TrainConfig::Schedule parse_schedule(const std::string& s) {
    if (s == "const" || s == "constant") return TrainConfig::Schedule::Constant;
    if (s == "linear") return TrainConfig::Schedule::Linear;
    if (s == "cosine") return TrainConfig::Schedule::Cosine;
    if (s == "exp") return TrainConfig::Schedule::Exp;
    return TrainConfig::Schedule::Constant;
}

static TrainConfig::GemmBackend parse_gemm_backend(const std::string& s) {
    if (s == "cutlass") return TrainConfig::GemmBackend::Cutlass;
    if (s == "nvfp4" || s == "cutlass_nvfp4") return TrainConfig::GemmBackend::CutlassNvfp4;
    return TrainConfig::GemmBackend::Dp4a;
}

static bitnet_cuda::Nvfp4Schedule parse_nvfp4_schedule(const std::string& s) {
    if (s == "auto") return bitnet_cuda::Nvfp4Schedule::Auto;
    if (s == "cooperative" || s == "coop") return bitnet_cuda::Nvfp4Schedule::Cooperative;
    if (s == "pingpong" || s == "pp") return bitnet_cuda::Nvfp4Schedule::Pingpong;
    return bitnet_cuda::Nvfp4Schedule::Auto;
}

static bitnet_cuda::Nvfp4QuantMode parse_nvfp4_quant_mode(const std::string& s) {
    if (s == "warp4") return bitnet_cuda::Nvfp4QuantMode::Warp4;
    if (s == "warp16" || s == "warp") return bitnet_cuda::Nvfp4QuantMode::Warp16;
    return bitnet_cuda::Nvfp4QuantMode::Warp16;
}

static bitnet_cuda::Nvfp4StageCount parse_nvfp4_stage_count(const std::string& s) {
    if (s == "2" || s == "s2" || s == "stages2") return bitnet_cuda::Nvfp4StageCount::Stages2;
    if (s == "3" || s == "s3" || s == "stages3") return bitnet_cuda::Nvfp4StageCount::Stages3;
    if (s == "4" || s == "s4" || s == "stages4") return bitnet_cuda::Nvfp4StageCount::Stages4;
    return bitnet_cuda::Nvfp4StageCount::Auto;
}

static bitnet_cuda::Nvfp4Decomposition parse_nvfp4_decomp(const std::string& s) {
    if (s == "data" || s == "dataparallel") return bitnet_cuda::Nvfp4Decomposition::DataParallel;
    if (s == "splitk" || s == "split") return bitnet_cuda::Nvfp4Decomposition::SplitK;
    if (s == "streamk" || s == "stream") return bitnet_cuda::Nvfp4Decomposition::StreamK;
    if (s == "auto" || s == "heuristic") return bitnet_cuda::Nvfp4Decomposition::Heuristic;
    return bitnet_cuda::Nvfp4Decomposition::Heuristic;
}

static float schedule_value(float start, float end, TrainConfig::Schedule schedule, float t01) {
    const float t = std::clamp(t01, 0.0f, 1.0f);
    const float e = (end >= 0.0f) ? end : start;
    if (schedule == TrainConfig::Schedule::Constant) return start;
    if (schedule == TrainConfig::Schedule::Linear) return start + (e - start) * t;
    if (schedule == TrainConfig::Schedule::Cosine) {
        const float ct = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * t));
        return start + (e - start) * ct;
    }
    // Exp
    if (start <= 0.0f || e <= 0.0f) return start + (e - start) * t;
    return start * std::pow(e / start, t);
}

struct EflaLmWeights {
    TernaryMatrix emb;   // [V][H]
    TernaryMatrix win;   // [H][H]
    std::vector<TernaryMatrix> wq;    // [L] each [H][H]
    std::vector<TernaryMatrix> wk;    // [L] each [H][H]
    std::vector<TernaryMatrix> wv;    // [L] each [H][H]
    std::vector<TernaryMatrix> wbeta; // [L] each [1][H]
    std::vector<TernaryMatrix> ff1;   // [L] each [F][H]
    std::vector<TernaryMatrix> ff2;   // [L] each [H][F]
    TernaryMatrix head;  // [V][H]
    std::vector<float> head_bias; // [V]
    std::vector<float> beta_bias; // [L]
};

static int qkv_fused_cols_aligned(int hidden, bool use_fused_qkv) {
    if (!use_fused_qkv) return 0;
    const int valid = hidden * 3 + 1;
    return (valid + 31) & ~31;
}

struct DeviceWeights {
    int device = 0;
    int V = 0;
    int H = 0;
    int L = 0;
    int F = 0;
    int B = 0;
    bool use_packed_gemm = false;
    bool use_nvfp4 = false;
    bool use_fused_qkv_nvfp4 = false;
    int qkv_fused_cols = 0;
    float win_out_scale = 1.0f;
    float q_out_scale = 1.0f;
    float k_out_scale = 1.0f;
    float v_out_scale = 1.0f;
    float beta_out_scale = 1.0f;
    float ff1_out_scale = 1.0f;
    float ff2_out_scale = 1.0f;
    float head_out_scale = 1.0f;

    int8_t* d_emb_w = nullptr;
    int8_t* d_win = nullptr;
    int8_t* d_wq = nullptr;
    int8_t* d_wk = nullptr;
    int8_t* d_wv = nullptr;
    int8_t* d_wbeta = nullptr;
    int8_t* d_ff1 = nullptr;
    int8_t* d_ff2 = nullptr;
    int8_t* d_head = nullptr;
    uint32_t* d_win_packed = nullptr;
    uint32_t* d_wq_packed = nullptr;
    uint32_t* d_wk_packed = nullptr;
    uint32_t* d_wv_packed = nullptr;
    uint32_t* d_wbeta_packed = nullptr;
    uint32_t* d_ff1_packed = nullptr;
    uint32_t* d_ff2_packed = nullptr;
    uint32_t* d_head_packed = nullptr;

    void* d_emb_nvfp4 = nullptr;
    void* d_win_nvfp4 = nullptr;
    void* d_head_nvfp4 = nullptr;
    void* d_emb_sfb = nullptr;
    void* d_win_sfb = nullptr;
    void* d_head_sfb = nullptr;
    std::vector<void*> d_wq_nvfp4;
    std::vector<void*> d_wk_nvfp4;
    std::vector<void*> d_wv_nvfp4;
    std::vector<void*> d_wbeta_nvfp4;
    std::vector<void*> d_ff1_nvfp4;
    std::vector<void*> d_ff2_nvfp4;
    std::vector<void*> d_wq_sfb;
    std::vector<void*> d_wk_sfb;
    std::vector<void*> d_wv_sfb;
    std::vector<void*> d_wbeta_sfb;
    std::vector<void*> d_ff1_sfb;
    std::vector<void*> d_ff2_sfb;
    std::vector<void*> d_wqkv_nvfp4;
    std::vector<void*> d_wqkv_sfb;

    DeviceWeights(int device_,
                  int vocab,
                  int hidden,
                  int layers,
                  int mlp_mult,
                  int batch,
                  bool use_packed_gemm_,
                  bool use_nvfp4_,
                  bool use_fused_qkv_nvfp4_)
        : device(device_),
          V(vocab),
          H(hidden),
          L(layers),
          F(hidden * std::max(1, mlp_mult)),
          B(batch),
          use_packed_gemm(use_packed_gemm_),
          use_nvfp4(use_nvfp4_),
          use_fused_qkv_nvfp4(use_fused_qkv_nvfp4_),
          qkv_fused_cols(qkv_fused_cols_aligned(hidden, use_fused_qkv_nvfp4_)) {
        cuda_check(cudaSetDevice(device), "cudaSetDevice DeviceWeights");

        const size_t emb_bytes = static_cast<size_t>(V) * H;
        const size_t hh_bytes = static_cast<size_t>(H) * H;
        const size_t hf_bytes = static_cast<size_t>(H) * F;
        const size_t h_bytes = static_cast<size_t>(H);
        const size_t head_bytes = static_cast<size_t>(V) * H;

        // Base weights.
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_emb_w), emb_bytes), "cudaMalloc emb");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_win), hh_bytes), "cudaMalloc win");
        const size_t hh_layers_bytes = static_cast<size_t>(L) * hh_bytes;
        const size_t h_layers_bytes = static_cast<size_t>(L) * h_bytes;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wq), hh_layers_bytes), "cudaMalloc wq");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wk), hh_layers_bytes), "cudaMalloc wk");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wv), hh_layers_bytes), "cudaMalloc wv");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wbeta), h_layers_bytes), "cudaMalloc wbeta");
        const size_t hf_layers_bytes = static_cast<size_t>(L) * hf_bytes;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff1), hf_layers_bytes), "cudaMalloc ff1");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff2), hf_layers_bytes), "cudaMalloc ff2");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_head), head_bytes), "cudaMalloc head");

        if (use_packed_gemm) {
            const int pack_cols_h = (H + 15) / 16;
            const int pack_cols_f = (F + 15) / 16;
            const size_t win_pack = static_cast<size_t>(H) * pack_cols_h;
            const size_t hh_layers_pack = static_cast<size_t>(L) * H * pack_cols_h;
            const size_t h_layers_pack = static_cast<size_t>(L) * pack_cols_h;
            const size_t ff1_layers_pack = static_cast<size_t>(L) * F * pack_cols_h;
            const size_t ff2_layers_pack = static_cast<size_t>(L) * H * pack_cols_f;
            const size_t head_pack = static_cast<size_t>(V) * pack_cols_h;

            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_win_packed), win_pack * sizeof(uint32_t)), "cudaMalloc win_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wq_packed), hh_layers_pack * sizeof(uint32_t)), "cudaMalloc wq_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wk_packed), hh_layers_pack * sizeof(uint32_t)), "cudaMalloc wk_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wv_packed), hh_layers_pack * sizeof(uint32_t)), "cudaMalloc wv_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wbeta_packed), h_layers_pack * sizeof(uint32_t)), "cudaMalloc wbeta_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff1_packed), ff1_layers_pack * sizeof(uint32_t)), "cudaMalloc ff1_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff2_packed), ff2_layers_pack * sizeof(uint32_t)), "cudaMalloc ff2_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_head_packed), head_pack * sizeof(uint32_t)), "cudaMalloc head_packed");
        }

        if (use_nvfp4) {
            auto alloc_nvfp4_b = [&](int rows, int cols, void** d_mat, void** d_sfb, const char* tag) {
                const size_t mat_bytes = bitnet_cuda::nvfp4_matrix_bytes(rows, cols);
                const size_t sfb_bytes = bitnet_cuda::nvfp4_sfb_bytes(B, rows, cols);
                cuda_check(cudaMalloc(reinterpret_cast<void**>(d_mat), mat_bytes), tag);
                cuda_check(cudaMalloc(reinterpret_cast<void**>(d_sfb), sfb_bytes), tag);
            };

            alloc_nvfp4_b(V, H, &d_emb_nvfp4, &d_emb_sfb, "cudaMalloc nvfp4 emb");
            alloc_nvfp4_b(H, H, &d_win_nvfp4, &d_win_sfb, "cudaMalloc nvfp4 win");
            alloc_nvfp4_b(V, H, &d_head_nvfp4, &d_head_sfb, "cudaMalloc nvfp4 head");

            d_wq_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_wk_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_wv_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_wbeta_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_ff1_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_ff2_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_wq_sfb.resize(static_cast<size_t>(L), nullptr);
            d_wk_sfb.resize(static_cast<size_t>(L), nullptr);
            d_wv_sfb.resize(static_cast<size_t>(L), nullptr);
            d_wbeta_sfb.resize(static_cast<size_t>(L), nullptr);
            d_ff1_sfb.resize(static_cast<size_t>(L), nullptr);
            d_ff2_sfb.resize(static_cast<size_t>(L), nullptr);
            d_wqkv_nvfp4.resize(static_cast<size_t>(L), nullptr);
            d_wqkv_sfb.resize(static_cast<size_t>(L), nullptr);

            for (int l = 0; l < L; ++l) {
                if (use_fused_qkv_nvfp4) {
                    alloc_nvfp4_b(qkv_fused_cols, H,
                                  &d_wqkv_nvfp4[static_cast<size_t>(l)],
                                  &d_wqkv_sfb[static_cast<size_t>(l)],
                                  "cudaMalloc nvfp4 wqkv");
                } else {
                    alloc_nvfp4_b(H, H, &d_wq_nvfp4[static_cast<size_t>(l)],
                                  &d_wq_sfb[static_cast<size_t>(l)], "cudaMalloc nvfp4 wq");
                    alloc_nvfp4_b(H, H, &d_wk_nvfp4[static_cast<size_t>(l)],
                                  &d_wk_sfb[static_cast<size_t>(l)], "cudaMalloc nvfp4 wk");
                    alloc_nvfp4_b(H, H, &d_wv_nvfp4[static_cast<size_t>(l)],
                                  &d_wv_sfb[static_cast<size_t>(l)], "cudaMalloc nvfp4 wv");
                    alloc_nvfp4_b(1, H, &d_wbeta_nvfp4[static_cast<size_t>(l)],
                                  &d_wbeta_sfb[static_cast<size_t>(l)], "cudaMalloc nvfp4 wbeta");
                }
                alloc_nvfp4_b(F, H, &d_ff1_nvfp4[static_cast<size_t>(l)],
                              &d_ff1_sfb[static_cast<size_t>(l)], "cudaMalloc nvfp4 ff1");
                alloc_nvfp4_b(H, F, &d_ff2_nvfp4[static_cast<size_t>(l)],
                              &d_ff2_sfb[static_cast<size_t>(l)], "cudaMalloc nvfp4 ff2");
            }
        }
    }

    ~DeviceWeights() {
        cuda_check(cudaSetDevice(device), "cudaSetDevice DeviceWeights free");
        if (d_head) cudaFree(d_head);
        if (d_ff2) cudaFree(d_ff2);
        if (d_ff1) cudaFree(d_ff1);
        if (d_wbeta) cudaFree(d_wbeta);
        if (d_wv) cudaFree(d_wv);
        if (d_wk) cudaFree(d_wk);
        if (d_wq) cudaFree(d_wq);
        if (d_win) cudaFree(d_win);
        if (d_emb_w) cudaFree(d_emb_w);
        if (d_head_packed) cudaFree(d_head_packed);
        if (d_ff2_packed) cudaFree(d_ff2_packed);
        if (d_ff1_packed) cudaFree(d_ff1_packed);
        if (d_wbeta_packed) cudaFree(d_wbeta_packed);
        if (d_wv_packed) cudaFree(d_wv_packed);
        if (d_wk_packed) cudaFree(d_wk_packed);
        if (d_wq_packed) cudaFree(d_wq_packed);
        if (d_win_packed) cudaFree(d_win_packed);

        for (void* p : d_ff2_sfb) if (p) cudaFree(p);
        for (void* p : d_ff1_sfb) if (p) cudaFree(p);
        for (void* p : d_wbeta_sfb) if (p) cudaFree(p);
        for (void* p : d_wv_sfb) if (p) cudaFree(p);
        for (void* p : d_wk_sfb) if (p) cudaFree(p);
        for (void* p : d_wq_sfb) if (p) cudaFree(p);
        for (void* p : d_wqkv_sfb) if (p) cudaFree(p);
        for (void* p : d_ff2_nvfp4) if (p) cudaFree(p);
        for (void* p : d_ff1_nvfp4) if (p) cudaFree(p);
        for (void* p : d_wbeta_nvfp4) if (p) cudaFree(p);
        for (void* p : d_wv_nvfp4) if (p) cudaFree(p);
        for (void* p : d_wk_nvfp4) if (p) cudaFree(p);
        for (void* p : d_wq_nvfp4) if (p) cudaFree(p);
        for (void* p : d_wqkv_nvfp4) if (p) cudaFree(p);
        if (d_head_sfb) cudaFree(d_head_sfb);
        if (d_win_sfb) cudaFree(d_win_sfb);
        if (d_emb_sfb) cudaFree(d_emb_sfb);
        if (d_head_nvfp4) cudaFree(d_head_nvfp4);
        if (d_win_nvfp4) cudaFree(d_win_nvfp4);
        if (d_emb_nvfp4) cudaFree(d_emb_nvfp4);
    }
};

class CudaEflaLm {
public:
    CudaEflaLm(int device, int vocab, int hidden, int layers, int mlp_mult, int batch, int seq_len,
               int state_dim, bool state_fp16, bool use_cuda_graph, bool graph_full_eval, bool use_nvtx,
               bool use_fused_qkv, bool use_qkv_split_kernel, int qkv_split_vec4,
               bool use_qkv_split_bias,
               bool use_fused_ff1, bool use_fused_head_loss, bool use_fused_noise_gemm,
               bool use_fused_efla, bool use_efla_mixed, bool use_efla_fuse_diff, bool use_efla_update_wmma,
               bool use_tiled_gemm, bool use_packed_gemm,
               bool use_packed_gemm_bitnet,
               TrainConfig::GemmBackend gemm_backend,
               bitnet_cuda::Nvfp4Schedule nvfp4_schedule,
               bitnet_cuda::Nvfp4QuantMode nvfp4_quant_mode,
               bitnet_cuda::Nvfp4StageCount nvfp4_stage_count,
               bitnet_cuda::Nvfp4Decomposition nvfp4_decomp,
               int nvfp4_splits,
               bool nvfp4_autotune,
               bool nvfp4_tune_schedule,
               bool nvfp4_tune_quant,
               bool nvfp4_tune_stages,
               bool nvfp4_tune_decomp,
               bool nvfp4_tune_splits,
               bool noise_batch,
               std::shared_ptr<DeviceWeights> weights)
        : device_(device),
          V_(vocab),
          H_(hidden),
          L_(layers),
          D_(state_dim),
          F_(hidden * std::max(1, mlp_mult)),
          B_(batch),
          T_(seq_len),
          use_fp16_state_(state_fp16),
          use_cuda_graph_(use_cuda_graph),
          graph_full_eval_(graph_full_eval),
          use_nvtx_(use_nvtx),
          use_fused_qkv_(use_fused_qkv),
          use_qkv_split_kernel_(use_qkv_split_kernel),
          use_qkv_split_vec4_(qkv_split_vec4 < 0 ? ((hidden & 3) == 0) : (qkv_split_vec4 != 0)),
          use_qkv_split_bias_(use_qkv_split_bias),
          use_fused_ff1_(use_fused_ff1),
          use_fused_head_loss_(use_fused_head_loss),
          use_fused_noise_gemm_(use_fused_noise_gemm),
          use_fused_efla_(use_fused_efla),
          use_efla_mixed_(use_efla_mixed),
          use_efla_fuse_diff_(use_efla_fuse_diff),
          use_efla_update_wmma_(use_efla_update_wmma),
          use_tiled_gemm_(use_tiled_gemm),
          use_packed_gemm_(use_packed_gemm),
          use_packed_gemm_bitnet_(use_packed_gemm_bitnet),
          gemm_backend_(gemm_backend),
          nvfp4_schedule_(nvfp4_schedule),
          nvfp4_quant_mode_(nvfp4_quant_mode),
          nvfp4_stage_count_(nvfp4_stage_count),
          nvfp4_decomp_(nvfp4_decomp),
          nvfp4_splits_(nvfp4_splits),
          nvfp4_autotune_(nvfp4_autotune),
          nvfp4_tune_schedule_(nvfp4_tune_schedule),
          nvfp4_tune_quant_(nvfp4_tune_quant),
          nvfp4_tune_stages_(nvfp4_tune_stages),
          nvfp4_tune_decomp_(nvfp4_tune_decomp),
          nvfp4_tune_splits_(nvfp4_tune_splits),
          use_noise_batch_(noise_batch),
          weights_(std::move(weights)) {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        if (gemm_backend_ == TrainConfig::GemmBackend::Cutlass) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device_);
            if (prop.major >= 10) {
                fprintf(stderr, "cutlass int8 GEMM selected for sm%d.\n", prop.major * 10 + prop.minor);
            }
        } else if (gemm_backend_ == TrainConfig::GemmBackend::CutlassNvfp4) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device_);
            if (prop.major < 12) {
                fprintf(stderr, "nvfp4 GEMM requires sm120+ (got sm%d); falling back to dp4a.\n",
                        prop.major * 10 + prop.minor);
                gemm_backend_ = TrainConfig::GemmBackend::Dp4a;
            } else {
                fprintf(stderr, "cutlass nvfp4 GEMM selected for sm%d.\n", prop.major * 10 + prop.minor);
            }
        }
        use_nvfp4_ = (gemm_backend_ == TrainConfig::GemmBackend::CutlassNvfp4);
        if (use_nvfp4_ && use_packed_gemm_) {
            fprintf(stderr, "nvfp4 GEMM does not support packed weights; disabling packed_gemm.\n");
            use_packed_gemm_ = false;
            use_packed_gemm_bitnet_ = false;
        }
        if (!weights_) {
            throw std::runtime_error("CudaEflaLm requires shared DeviceWeights");
        }
        if (weights_->device != device_ || weights_->V != V_ || weights_->H != H_ ||
            weights_->L != L_ || weights_->F != F_ || weights_->B != B_) {
            throw std::runtime_error("DeviceWeights shape mismatch");
        }
        if (weights_->use_nvfp4 != use_nvfp4_) {
            throw std::runtime_error("DeviceWeights NVFP4 mode mismatch");
        }
        if (use_packed_gemm_ && !weights_->use_packed_gemm) {
            throw std::runtime_error("DeviceWeights packed GEMM mismatch");
        }
        d_emb_w_ = weights_->d_emb_w;
        d_win_ = weights_->d_win;
        d_wq_ = weights_->d_wq;
        d_wk_ = weights_->d_wk;
        d_wv_ = weights_->d_wv;
        d_wbeta_ = weights_->d_wbeta;
        d_ff1_ = weights_->d_ff1;
        d_ff2_ = weights_->d_ff2;
        d_head_ = weights_->d_head;
        d_win_packed_ = weights_->d_win_packed;
        d_wq_packed_ = weights_->d_wq_packed;
        d_wk_packed_ = weights_->d_wk_packed;
        d_wv_packed_ = weights_->d_wv_packed;
        d_wbeta_packed_ = weights_->d_wbeta_packed;
        d_ff1_packed_ = weights_->d_ff1_packed;
        d_ff2_packed_ = weights_->d_ff2_packed;
        d_head_packed_ = weights_->d_head_packed;
        d_emb_nvfp4_ = weights_->d_emb_nvfp4;
        d_win_nvfp4_ = weights_->d_win_nvfp4;
        d_head_nvfp4_ = weights_->d_head_nvfp4;
        d_emb_sfb_ = weights_->d_emb_sfb;
        d_win_sfb_ = weights_->d_win_sfb;
        d_head_sfb_ = weights_->d_head_sfb;
        d_wq_nvfp4_ = weights_->d_wq_nvfp4;
        d_wk_nvfp4_ = weights_->d_wk_nvfp4;
        d_wv_nvfp4_ = weights_->d_wv_nvfp4;
        d_wbeta_nvfp4_ = weights_->d_wbeta_nvfp4;
        d_ff1_nvfp4_ = weights_->d_ff1_nvfp4;
        d_ff2_nvfp4_ = weights_->d_ff2_nvfp4;
        d_wq_sfb_ = weights_->d_wq_sfb;
        d_wk_sfb_ = weights_->d_wk_sfb;
        d_wv_sfb_ = weights_->d_wv_sfb;
        d_wbeta_sfb_ = weights_->d_wbeta_sfb;
        d_ff1_sfb_ = weights_->d_ff1_sfb;
        d_ff2_sfb_ = weights_->d_ff2_sfb;
        d_wqkv_nvfp4_ = weights_->d_wqkv_nvfp4;
        d_wqkv_sfb_ = weights_->d_wqkv_sfb;
        use_fused_qkv_nvfp4_ = weights_->use_fused_qkv_nvfp4;
        qkv_fused_cols_valid_ = H_ * 3 + 1;
        qkv_fused_cols_ = use_fused_qkv_nvfp4_ ? weights_->qkv_fused_cols : qkv_fused_cols_valid_;
        win_out_scale_ = weights_->win_out_scale;
        q_out_scale_ = weights_->q_out_scale;
        k_out_scale_ = weights_->k_out_scale;
        v_out_scale_ = weights_->v_out_scale;
        beta_out_scale_ = weights_->beta_out_scale;
        ff1_out_scale_ = weights_->ff1_out_scale;
        ff2_out_scale_ = weights_->ff2_out_scale;
        head_out_scale_ = weights_->head_out_scale;
        if (use_efla_update_wmma_ && !use_fp16_state_) {
            fprintf(stderr, "efla_update_wmma requires FP16 state; disabling.\n");
            use_efla_update_wmma_ = false;
        }
        if (use_efla_update_wmma_ && ((H_ & 15) != 0)) {
            fprintf(stderr, "efla_update_wmma requires hidden multiple of 16; disabling.\n");
            use_efla_update_wmma_ = false;
        }
        if (use_efla_mixed_ && !use_fp16_state_) {
            fprintf(stderr, "efla_mixed requires FP16 state; disabling.\n");
            use_efla_mixed_ = false;
        }
        if (use_efla_mixed_ && ((H_ & 1) != 0)) {
            fprintf(stderr, "efla_mixed requires even hidden; disabling.\n");
            use_efla_mixed_ = false;
        }
        if (use_efla_mixed_ && D_ > 0) {
            fprintf(stderr, "efla_mixed is not used with low-rank state; disabling.\n");
            use_efla_mixed_ = false;
        }
        if (use_nvfp4_ && use_cuda_graph_) {
            fprintf(stderr, "nvfp4 GEMM: attempting CUDA graph capture (relaxed mode); will fall back on failure.\n");
        }
        cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        cuda_check(cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking), "cudaStreamCreate upload");
        for (int i = 0; i < kTokenBuffers; ++i) {
            cuda_check(cudaEventCreateWithFlags(&upload_done_[i], cudaEventDisableTiming),
                       "cudaEventCreate upload_done");
        }
        if (use_nvfp4_) {
            const char* sched_name = "auto";
            switch (nvfp4_schedule_) {
                case bitnet_cuda::Nvfp4Schedule::Cooperative: sched_name = "cooperative"; break;
                case bitnet_cuda::Nvfp4Schedule::Pingpong: sched_name = "pingpong"; break;
                case bitnet_cuda::Nvfp4Schedule::Auto:
                default: sched_name = "auto"; break;
            }
            const char* quant_name = (nvfp4_quant_mode_ == bitnet_cuda::Nvfp4QuantMode::Warp4) ? "warp4" : "warp16";
            const char* stage_name = "auto";
            switch (nvfp4_stage_count_) {
                case bitnet_cuda::Nvfp4StageCount::Stages2: stage_name = "2"; break;
                case bitnet_cuda::Nvfp4StageCount::Stages3: stage_name = "3"; break;
                case bitnet_cuda::Nvfp4StageCount::Stages4: stage_name = "4"; break;
                case bitnet_cuda::Nvfp4StageCount::Auto:
                default: stage_name = "auto"; break;
            }
            const char* decomp_name = "auto";
            switch (nvfp4_decomp_) {
                case bitnet_cuda::Nvfp4Decomposition::DataParallel: decomp_name = "data"; break;
                case bitnet_cuda::Nvfp4Decomposition::SplitK: decomp_name = "splitk"; break;
                case bitnet_cuda::Nvfp4Decomposition::StreamK: decomp_name = "streamk"; break;
                case bitnet_cuda::Nvfp4Decomposition::Heuristic:
                default: decomp_name = "auto"; break;
            }
            fprintf(stderr, "nvfp4 schedule: %s\n", sched_name);
            fprintf(stderr, "nvfp4 quant: %s\n", quant_name);
            fprintf(stderr, "nvfp4 stages: %s\n", stage_name);
            fprintf(stderr, "nvfp4 decomp: %s (splits=%d)\n", decomp_name, nvfp4_splits_);
            bitnet_cuda::nvfp4_set_schedule(nvfp4_schedule_);
            bitnet_cuda::nvfp4_set_quant_mode(nvfp4_quant_mode_);
            bitnet_cuda::nvfp4_set_stage_count(nvfp4_stage_count_);
            bitnet_cuda::nvfp4_set_decomposition(nvfp4_decomp_);
            bitnet_cuda::nvfp4_set_splits(nvfp4_splits_);
            bitnet_cuda::nvfp4_init();
        }
        if (use_packed_gemm_) {
            bitnet_cuda::init_i2_lut();
        }

        const size_t emb_bytes = static_cast<size_t>(V_) * H_;
        const size_t hh_bytes = static_cast<size_t>(H_) * H_;
        const size_t hf_bytes = static_cast<size_t>(H_) * F_;
        const size_t h_bytes = static_cast<size_t>(H_);
        const size_t head_bytes = static_cast<size_t>(V_) * H_;

        const size_t hh_layers_bytes = static_cast<size_t>(L_) * hh_bytes;
        const size_t h_layers_bytes = static_cast<size_t>(L_) * h_bytes;
        const size_t hf_layers_bytes = static_cast<size_t>(L_) * hf_bytes;

        // Noise weights (full-matrix noise, materialized).
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_emb_noise_), emb_bytes), "cudaMalloc emb_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_win_noise_), hh_bytes), "cudaMalloc win_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wq_noise_), hh_layers_bytes), "cudaMalloc wq_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wk_noise_), hh_layers_bytes), "cudaMalloc wk_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wv_noise_), hh_layers_bytes), "cudaMalloc wv_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wbeta_noise_), h_layers_bytes), "cudaMalloc wbeta_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff1_noise_), hf_layers_bytes), "cudaMalloc ff1_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff2_noise_), hf_layers_bytes), "cudaMalloc ff2_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_head_noise_), head_bytes), "cudaMalloc head_noise");

        if (use_packed_gemm_) {
            pack_cols_h_ = (H_ + 15) / 16;
            pack_cols_f_ = (F_ + 15) / 16;
            const size_t win_pack = static_cast<size_t>(H_) * pack_cols_h_;
            const size_t hh_layers_pack = static_cast<size_t>(L_) * H_ * pack_cols_h_;
            const size_t h_layers_pack = static_cast<size_t>(L_) * pack_cols_h_;
            const size_t ff1_layers_pack = static_cast<size_t>(L_) * F_ * pack_cols_h_;
            const size_t ff2_layers_pack = static_cast<size_t>(L_) * H_ * pack_cols_f_;
            const size_t head_pack = static_cast<size_t>(V_) * pack_cols_h_;

            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_win_noise_packed_), win_pack * sizeof(uint32_t)), "cudaMalloc win_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wq_noise_packed_), hh_layers_pack * sizeof(uint32_t)), "cudaMalloc wq_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wk_noise_packed_), hh_layers_pack * sizeof(uint32_t)), "cudaMalloc wk_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wv_noise_packed_), hh_layers_pack * sizeof(uint32_t)), "cudaMalloc wv_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_wbeta_noise_packed_), h_layers_pack * sizeof(uint32_t)), "cudaMalloc wbeta_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff1_noise_packed_), ff1_layers_pack * sizeof(uint32_t)), "cudaMalloc ff1_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff2_noise_packed_), ff2_layers_pack * sizeof(uint32_t)), "cudaMalloc ff2_noise_packed");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_head_noise_packed_), head_pack * sizeof(uint32_t)), "cudaMalloc head_noise_packed");
        }

        if (use_noise_batch_) {
            const size_t emb_n = static_cast<size_t>(V_) * H_;
            const size_t hh = static_cast<size_t>(H_) * H_;
            const size_t hf = static_cast<size_t>(H_) * static_cast<size_t>(F_);
            noise_desc_host_.clear();
            noise_desc_host_.reserve(static_cast<size_t>(3 + 6 * L_));
            noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_emb_noise_, emb_n, 0});
            noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_win_noise_, hh, 0});
            for (int l = 0; l < L_; ++l) {
                const size_t off_hh = static_cast<size_t>(l) * hh;
                const size_t off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
                const size_t off_hf = static_cast<size_t>(l) * hf;
                noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_wq_noise_ + off_hh, hh, 0});
                noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_wk_noise_ + off_hh, hh, 0});
                noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_wv_noise_ + off_hh, hh, 0});
                noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_wbeta_noise_ + off_h, static_cast<size_t>(H_), 0});
                noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_ff1_noise_ + off_hf, hf, 0});
                noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_ff2_noise_ + off_hf, hf, 0});
            }
            noise_desc_host_.push_back(efla_lm_cuda::NoiseDesc{d_head_noise_, emb_n, 0});
            noise_desc_count_ = static_cast<int>(noise_desc_host_.size());
            size_t max_n = 0;
            for (const auto& desc : noise_desc_host_) {
                if (desc.n > max_n) max_n = desc.n;
            }
            constexpr int kNoiseThreads = 256;
            noise_desc_max_blocks_ = static_cast<int>((max_n + kNoiseThreads - 1) / kNoiseThreads);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_noise_desc_),
                                  static_cast<size_t>(noise_desc_count_) * sizeof(efla_lm_cuda::NoiseDesc)),
                       "cudaMalloc noise_desc");
            cuda_check(cudaMemcpyAsync(d_noise_desc_, noise_desc_host_.data(),
                                       static_cast<size_t>(noise_desc_count_) * sizeof(efla_lm_cuda::NoiseDesc),
                                       cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync noise_desc init");
        }

        if (use_nvfp4_) {
            const size_t a_h_bytes = bitnet_cuda::nvfp4_matrix_bytes(B_, H_);
            const size_t a_f_bytes = bitnet_cuda::nvfp4_matrix_bytes(B_, F_);
            const size_t sfa_h_bytes = bitnet_cuda::nvfp4_sfa_bytes(B_, 1, H_);
            const size_t sfa_f_bytes = bitnet_cuda::nvfp4_sfa_bytes(B_, 1, F_);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_a_nvfp4_h_), a_h_bytes), "cudaMalloc nvfp4 a_h");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sfa_h_), sfa_h_bytes), "cudaMalloc nvfp4 sfa_h");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_a_nvfp4_f_), a_f_bytes), "cudaMalloc nvfp4 a_f");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sfa_f_), sfa_f_bytes), "cudaMalloc nvfp4 sfa_f");

            auto alloc_nvfp4_b = [&](int rows, int cols, void** d_mat, void** d_sfb, const char* tag) {
                const size_t mat_bytes = bitnet_cuda::nvfp4_matrix_bytes(rows, cols);
                const size_t sfb_bytes = bitnet_cuda::nvfp4_sfb_bytes(B_, rows, cols);
                cuda_check(cudaMalloc(reinterpret_cast<void**>(d_mat), mat_bytes), tag);
                cuda_check(cudaMalloc(reinterpret_cast<void**>(d_sfb), sfb_bytes), tag);
            };
            alloc_nvfp4_b(V_, H_, &d_emb_noise_nvfp4_, &d_emb_noise_sfb_, "cudaMalloc nvfp4 emb_noise");
            alloc_nvfp4_b(H_, H_, &d_win_noise_nvfp4_, &d_win_noise_sfb_, "cudaMalloc nvfp4 win_noise");
            alloc_nvfp4_b(V_, H_, &d_head_noise_nvfp4_, &d_head_noise_sfb_, "cudaMalloc nvfp4 head_noise");

            d_wq_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
            d_wk_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
            d_wv_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
            d_wbeta_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
            d_ff1_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
            d_ff2_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
            d_wq_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            d_wk_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            d_wv_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            d_wbeta_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            d_ff1_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            d_ff2_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            if (use_fused_qkv_nvfp4_) {
                d_wqkv_noise_nvfp4_.resize(static_cast<size_t>(L_), nullptr);
                d_wqkv_noise_sfb_.resize(static_cast<size_t>(L_), nullptr);
            }

            for (int l = 0; l < L_; ++l) {
                alloc_nvfp4_b(H_, H_, &d_wq_noise_nvfp4_[static_cast<size_t>(l)],
                              &d_wq_noise_sfb_[static_cast<size_t>(l)], "cudaMalloc nvfp4 wq_noise");
                alloc_nvfp4_b(H_, H_, &d_wk_noise_nvfp4_[static_cast<size_t>(l)],
                              &d_wk_noise_sfb_[static_cast<size_t>(l)], "cudaMalloc nvfp4 wk_noise");
                alloc_nvfp4_b(H_, H_, &d_wv_noise_nvfp4_[static_cast<size_t>(l)],
                              &d_wv_noise_sfb_[static_cast<size_t>(l)], "cudaMalloc nvfp4 wv_noise");
                alloc_nvfp4_b(1, H_, &d_wbeta_noise_nvfp4_[static_cast<size_t>(l)],
                              &d_wbeta_noise_sfb_[static_cast<size_t>(l)], "cudaMalloc nvfp4 wbeta_noise");
                alloc_nvfp4_b(F_, H_, &d_ff1_noise_nvfp4_[static_cast<size_t>(l)],
                              &d_ff1_noise_sfb_[static_cast<size_t>(l)], "cudaMalloc nvfp4 ff1_noise");
                alloc_nvfp4_b(H_, F_, &d_ff2_noise_nvfp4_[static_cast<size_t>(l)],
                              &d_ff2_noise_sfb_[static_cast<size_t>(l)], "cudaMalloc nvfp4 ff2_noise");
                if (use_fused_qkv_nvfp4_) {
                    alloc_nvfp4_b(qkv_fused_cols_, H_,
                                  &d_wqkv_noise_nvfp4_[static_cast<size_t>(l)],
                                  &d_wqkv_noise_sfb_[static_cast<size_t>(l)],
                                  "cudaMalloc nvfp4 wqkv_noise");
                }
            }
        }

        // Constant scale_x (all ones).
        const int g = (H_ + 255) / 256;
        std::vector<float> ones(static_cast<size_t>(g), 1.0f);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scale_x_), ones.size() * sizeof(float)), "cudaMalloc scale_x");
        cuda_check(cudaMemcpyAsync(d_scale_x_, ones.data(), ones.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync scale_x");

        // Constant scale_f (all ones), for MLP hidden dim.
        const int gf = (F_ + 255) / 256;
        std::vector<float> ones_f(static_cast<size_t>(gf), 1.0f);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scale_f_), ones_f.size() * sizeof(float)), "cudaMalloc scale_f");
        cuda_check(cudaMemcpyAsync(d_scale_f_, ones_f.data(), ones_f.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync scale_f");

        // Positional embedding.
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pos_), static_cast<size_t>(T_) * H_ * sizeof(float)), "cudaMalloc pos");

        // Tokens and buffers (double-buffered).
        const size_t host_bytes = static_cast<size_t>(B_) * T_;
        for (int i = 0; i < kTokenBuffers; ++i) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tokens_[i]), host_bytes), "cudaMalloc tokens");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_targets_[i]), host_bytes), "cudaMalloc targets");
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&h_tokens_pinned_[i]), host_bytes, cudaHostAllocDefault),
                       "cudaHostAlloc h_tokens");
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&h_targets_pinned_[i]), host_bytes, cudaHostAllocDefault),
                       "cudaHostAlloc h_targets");
        }
        d_tokens_active_ = d_tokens_[0];
        d_targets_active_ = d_targets_[0];

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_emb_f_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc emb_f");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_emb_q_), static_cast<size_t>(B_) * H_), "cudaMalloc emb_q");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_inproj_f_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc inproj_f");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_res_f_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc res_f");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_x_q_), static_cast<size_t>(B_) * H_), "cudaMalloc x_q");
        if (use_fused_qkv_nvfp4_) {
            const size_t qkv_bytes = static_cast<size_t>(B_) * static_cast<size_t>(qkv_fused_cols_) * sizeof(float);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qkv_f_), qkv_bytes), "cudaMalloc qkv_fused");
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_q_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc q");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_k_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc k");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_v_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc v");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_beta_raw_), static_cast<size_t>(B_) * sizeof(float)), "cudaMalloc beta_raw");

        if (D_ > 0) {
            state_bytes_ = static_cast<size_t>(L_) * B_ * static_cast<size_t>(D_) * H_ * sizeof(float);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_K_f_), state_bytes_), "cudaMalloc K");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_V_f_), state_bytes_), "cudaMalloc V");
        } else {
            state_bytes_ = static_cast<size_t>(L_) * B_ * H_ * H_ *
                           (use_fp16_state_ ? sizeof(__half) : sizeof(float));
            if (use_fp16_state_) {
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_S_h_), state_bytes_), "cudaMalloc S half");
            } else {
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_S_f_), state_bytes_), "cudaMalloc S");
            }
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_q_norm_), static_cast<size_t>(B_) * H_ * sizeof(float)),
                       "cudaMalloc q_norm");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_k_usage_), static_cast<size_t>(B_) * H_ * sizeof(float)),
                       "cudaMalloc k_usage");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_kS_), static_cast<size_t>(B_) * H_ * sizeof(float)),
                       "cudaMalloc kS");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_diff_), static_cast<size_t>(B_) * H_ * sizeof(float)),
                       "cudaMalloc diff");
            if (use_efla_mixed_) {
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_k_usage_h_), static_cast<size_t>(B_) * H_ * sizeof(__half)),
                           "cudaMalloc k_usage_h");
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_diff_h_), static_cast<size_t>(B_) * H_ * sizeof(__half)),
                           "cudaMalloc diff_h");
            }
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_alpha_), static_cast<size_t>(B_) * sizeof(float)),
                       "cudaMalloc alpha");
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_y_), static_cast<size_t>(B_) * H_ * sizeof(float)), "cudaMalloc y");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_y_q_), static_cast<size_t>(B_) * H_), "cudaMalloc y_q");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff1_f_), static_cast<size_t>(B_) * F_ * sizeof(float)), "cudaMalloc ff1_f");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ff1_q_), static_cast<size_t>(B_) * F_), "cudaMalloc ff1_q");

        const size_t logits_floats = static_cast<size_t>(T_) * B_ * V_;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_logits_), logits_floats * sizeof(float)), "cudaMalloc logits");
        const int tmp_cols = std::max(std::max(V_, H_), F_);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tmp_noise_), static_cast<size_t>(B_) * tmp_cols * sizeof(float)),
                   "cudaMalloc tmp_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_head_bias_), static_cast<size_t>(V_) * sizeof(float)), "cudaMalloc head_bias");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_head_bias_noise_), static_cast<size_t>(V_) * sizeof(float)), "cudaMalloc head_bias_noise");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_loss_sum_), sizeof(float)), "cudaMalloc loss_sum");
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&h_loss_sum_pinned_), sizeof(float), cudaHostAllocDefault),
                   "cudaHostAlloc loss_sum");
        if (h_loss_sum_pinned_) *h_loss_sum_pinned_ = 0.0f;

        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize init");
    }

    ~CudaEflaLm() {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        for (auto& entry : graph_cache_) {
            if (entry.exec) cudaGraphExecDestroy(entry.exec);
            if (entry.graph) cudaGraphDestroy(entry.graph);
        }
        if (h_logits_) cudaFreeHost(h_logits_);
        if (h_loss_sum_pinned_) cudaFreeHost(h_loss_sum_pinned_);
        if (d_loss_sum_) cudaFree(d_loss_sum_);
        if (d_head_bias_noise_) cudaFree(d_head_bias_noise_);
        if (d_head_bias_) cudaFree(d_head_bias_);
        if (d_tmp_noise_) cudaFree(d_tmp_noise_);
        if (d_logits_) cudaFree(d_logits_);
        if (d_y_q_) cudaFree(d_y_q_);
        if (d_y_) cudaFree(d_y_);
        if (d_ff1_q_) cudaFree(d_ff1_q_);
        if (d_ff1_f_) cudaFree(d_ff1_f_);
        if (d_qkv_f_) cudaFree(d_qkv_f_);
        if (d_V_f_) cudaFree(d_V_f_);
        if (d_K_f_) cudaFree(d_K_f_);
        if (d_alpha_) cudaFree(d_alpha_);
        if (d_diff_h_) cudaFree(d_diff_h_);
        if (d_k_usage_h_) cudaFree(d_k_usage_h_);
        if (d_diff_) cudaFree(d_diff_);
        if (d_kS_) cudaFree(d_kS_);
        if (d_k_usage_) cudaFree(d_k_usage_);
        if (d_q_norm_) cudaFree(d_q_norm_);
        if (d_S_h_) cudaFree(d_S_h_);
        if (d_S_f_) cudaFree(d_S_f_);
        if (d_beta_raw_) cudaFree(d_beta_raw_);
        if (d_v_) cudaFree(d_v_);
        if (d_k_) cudaFree(d_k_);
        if (d_q_) cudaFree(d_q_);
        if (d_x_q_) cudaFree(d_x_q_);
        if (d_inproj_f_) cudaFree(d_inproj_f_);
        if (d_res_f_) cudaFree(d_res_f_);
        if (d_emb_q_) cudaFree(d_emb_q_);
        if (d_emb_f_) cudaFree(d_emb_f_);
        for (int i = 0; i < kTokenBuffers; ++i) {
            if (d_targets_[i]) cudaFree(d_targets_[i]);
            if (d_tokens_[i]) cudaFree(d_tokens_[i]);
            if (h_targets_pinned_[i]) cudaFreeHost(h_targets_pinned_[i]);
            if (h_tokens_pinned_[i]) cudaFreeHost(h_tokens_pinned_[i]);
            if (upload_done_[i]) cudaEventDestroy(upload_done_[i]);
        }
        if (upload_stream_) cudaStreamDestroy(upload_stream_);
        if (d_pos_) cudaFree(d_pos_);
        if (d_scale_x_) cudaFree(d_scale_x_);
        if (d_scale_f_) cudaFree(d_scale_f_);

        if (d_head_noise_) cudaFree(d_head_noise_);
        if (d_ff2_noise_) cudaFree(d_ff2_noise_);
        if (d_ff1_noise_) cudaFree(d_ff1_noise_);
        if (d_wbeta_noise_) cudaFree(d_wbeta_noise_);
        if (d_wv_noise_) cudaFree(d_wv_noise_);
        if (d_wk_noise_) cudaFree(d_wk_noise_);
        if (d_wq_noise_) cudaFree(d_wq_noise_);
        if (d_win_noise_) cudaFree(d_win_noise_);
        if (d_emb_noise_) cudaFree(d_emb_noise_);
        if (d_head_noise_packed_) cudaFree(d_head_noise_packed_);
        if (d_noise_desc_) cudaFree(d_noise_desc_);
        if (d_ff2_noise_packed_) cudaFree(d_ff2_noise_packed_);
        if (d_ff1_noise_packed_) cudaFree(d_ff1_noise_packed_);
        if (d_wbeta_noise_packed_) cudaFree(d_wbeta_noise_packed_);
        if (d_wv_noise_packed_) cudaFree(d_wv_noise_packed_);
        if (d_wk_noise_packed_) cudaFree(d_wk_noise_packed_);
        if (d_wq_noise_packed_) cudaFree(d_wq_noise_packed_);
        if (d_win_noise_packed_) cudaFree(d_win_noise_packed_);

        for (void* p : d_ff2_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_ff1_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_wbeta_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_wv_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_wk_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_wq_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_wqkv_noise_sfb_) if (p) cudaFree(p);
        for (void* p : d_ff2_noise_nvfp4_) if (p) cudaFree(p);
        for (void* p : d_ff1_noise_nvfp4_) if (p) cudaFree(p);
        for (void* p : d_wbeta_noise_nvfp4_) if (p) cudaFree(p);
        for (void* p : d_wv_noise_nvfp4_) if (p) cudaFree(p);
        for (void* p : d_wk_noise_nvfp4_) if (p) cudaFree(p);
        for (void* p : d_wq_noise_nvfp4_) if (p) cudaFree(p);
        for (void* p : d_wqkv_noise_nvfp4_) if (p) cudaFree(p);
        if (d_head_noise_sfb_) cudaFree(d_head_noise_sfb_);
        if (d_win_noise_sfb_) cudaFree(d_win_noise_sfb_);
        if (d_emb_noise_sfb_) cudaFree(d_emb_noise_sfb_);
        if (d_head_noise_nvfp4_) cudaFree(d_head_noise_nvfp4_);
        if (d_win_noise_nvfp4_) cudaFree(d_win_noise_nvfp4_);
        if (d_emb_noise_nvfp4_) cudaFree(d_emb_noise_nvfp4_);
        if (d_sfa_f_) cudaFree(d_sfa_f_);
        if (d_sfa_h_) cudaFree(d_sfa_h_);
        if (d_a_nvfp4_f_) cudaFree(d_a_nvfp4_f_);
        if (d_a_nvfp4_h_) cudaFree(d_a_nvfp4_h_);


        if (stream_) cudaStreamDestroy(stream_);
    }

    void set_pos_embedding(const std::vector<float>& pos) {
        if (pos.size() != static_cast<size_t>(T_) * H_) throw std::runtime_error("pos size mismatch");
        cuda_check(cudaMemcpyAsync(d_pos_, pos.data(), pos.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync pos");
    }

    void sync_base_weights(const EflaLmWeights& w) {
        if (static_cast<int>(w.wq.size()) != L_ ||
            static_cast<int>(w.wk.size()) != L_ ||
            static_cast<int>(w.wv.size()) != L_ ||
            static_cast<int>(w.wbeta.size()) != L_ ||
            static_cast<int>(w.ff1.size()) != L_ ||
            static_cast<int>(w.ff2.size()) != L_) {
            throw std::runtime_error("sync_base_weights: layer count mismatch");
        }

        cuda_check(cudaMemcpyAsync(d_emb_w_, w.emb.w.data(), w.emb.w.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync emb");
        cuda_check(cudaMemcpyAsync(d_win_, w.win.w.data(), w.win.w.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync win");

        const size_t hh = static_cast<size_t>(H_) * H_;
        h_wq_pack_.resize(static_cast<size_t>(L_) * hh);
        h_wk_pack_.resize(static_cast<size_t>(L_) * hh);
        h_wv_pack_.resize(static_cast<size_t>(L_) * hh);
        h_wbeta_pack_.resize(static_cast<size_t>(L_) * static_cast<size_t>(H_));
        const size_t hf = static_cast<size_t>(H_) * static_cast<size_t>(F_);
        h_ff1_pack_.resize(static_cast<size_t>(L_) * hf);
        h_ff2_pack_.resize(static_cast<size_t>(L_) * hf);
        for (int l = 0; l < L_; ++l) {
            std::copy(w.wq[static_cast<size_t>(l)].w.begin(),
                      w.wq[static_cast<size_t>(l)].w.end(),
                      h_wq_pack_.begin() + static_cast<size_t>(l) * hh);
            std::copy(w.wk[static_cast<size_t>(l)].w.begin(),
                      w.wk[static_cast<size_t>(l)].w.end(),
                      h_wk_pack_.begin() + static_cast<size_t>(l) * hh);
            std::copy(w.wv[static_cast<size_t>(l)].w.begin(),
                      w.wv[static_cast<size_t>(l)].w.end(),
                      h_wv_pack_.begin() + static_cast<size_t>(l) * hh);

            const auto& wb = w.wbeta[static_cast<size_t>(l)].w;
            std::copy(wb.begin(), wb.end(),
                      h_wbeta_pack_.begin() + static_cast<size_t>(l) * static_cast<size_t>(H_));

            std::copy(w.ff1[static_cast<size_t>(l)].w.begin(),
                      w.ff1[static_cast<size_t>(l)].w.end(),
                      h_ff1_pack_.begin() + static_cast<size_t>(l) * hf);
            std::copy(w.ff2[static_cast<size_t>(l)].w.begin(),
                      w.ff2[static_cast<size_t>(l)].w.end(),
                      h_ff2_pack_.begin() + static_cast<size_t>(l) * hf);
        }

        cuda_check(cudaMemcpyAsync(d_wq_, h_wq_pack_.data(), h_wq_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wq");
        cuda_check(cudaMemcpyAsync(d_wk_, h_wk_pack_.data(), h_wk_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wk");
        cuda_check(cudaMemcpyAsync(d_wv_, h_wv_pack_.data(), h_wv_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wv");
        cuda_check(cudaMemcpyAsync(d_wbeta_, h_wbeta_pack_.data(), h_wbeta_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wbeta");
        cuda_check(cudaMemcpyAsync(d_ff1_, h_ff1_pack_.data(), h_ff1_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync ff1");
        cuda_check(cudaMemcpyAsync(d_ff2_, h_ff2_pack_.data(), h_ff2_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync ff2");

        cuda_check(cudaMemcpyAsync(d_head_, w.head.w.data(), w.head.w.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync head");
        if (use_packed_gemm_) {
            const int rows_hh = L_ * H_;
            const int rows_hf1 = L_ * F_;
            const int rows_hf2 = L_ * H_;
            pack_i2_weights(d_win_, H_, H_, d_win_packed_);
            pack_i2_weights(d_wq_, rows_hh, H_, d_wq_packed_);
            pack_i2_weights(d_wk_, rows_hh, H_, d_wk_packed_);
            pack_i2_weights(d_wv_, rows_hh, H_, d_wv_packed_);
            pack_i2_weights(d_wbeta_, L_, H_, d_wbeta_packed_);
            pack_i2_weights(d_ff1_, rows_hf1, H_, d_ff1_packed_);
            pack_i2_weights(d_ff2_, rows_hf2, F_, d_ff2_packed_);
            pack_i2_weights(d_head_, V_, H_, d_head_packed_);
        }
        win_out_scale_ = w.win.out_scale;
        q_out_scale_ = w.wq[0].out_scale;
        k_out_scale_ = w.wk[0].out_scale;
        v_out_scale_ = w.wv[0].out_scale;
        beta_out_scale_ = w.wbeta[0].out_scale;
        ff1_out_scale_ = w.ff1[0].out_scale;
        ff2_out_scale_ = w.ff2[0].out_scale;
        head_out_scale_ = w.head.out_scale;
        if (weights_) {
            weights_->win_out_scale = win_out_scale_;
            weights_->q_out_scale = q_out_scale_;
            weights_->k_out_scale = k_out_scale_;
            weights_->v_out_scale = v_out_scale_;
            weights_->beta_out_scale = beta_out_scale_;
            weights_->ff1_out_scale = ff1_out_scale_;
            weights_->ff2_out_scale = ff2_out_scale_;
            weights_->head_out_scale = head_out_scale_;
        }
        if (use_nvfp4_) {
            quantize_nvfp4_base_weights();
        }
        const auto nvfp4_schedule_init = nvfp4_schedule_;
        const auto nvfp4_quant_init = nvfp4_quant_mode_;
        const auto nvfp4_stage_init = nvfp4_stage_count_;
        const auto nvfp4_decomp_init = nvfp4_decomp_;
        const int nvfp4_splits_init = nvfp4_splits_;
        if (use_nvfp4_ && nvfp4_autotune_) {
            maybe_autotune_nvfp4();
        }

        if (use_nvfp4_ && use_cuda_graph_) {
            bitnet_cuda::nvfp4_set_verbose(false);
            const bool prewarm_ok = prewarm_nvfp4_gemm_cache(false);
            bitnet_cuda::nvfp4_set_verbose(true);
            if (!prewarm_ok) {
                bool restored = false;
                if (nvfp4_autotune_ &&
                    (nvfp4_schedule_ != nvfp4_schedule_init ||
                     nvfp4_quant_mode_ != nvfp4_quant_init ||
                     nvfp4_stage_count_ != nvfp4_stage_init ||
                     nvfp4_decomp_ != nvfp4_decomp_init ||
                     nvfp4_splits_ != nvfp4_splits_init)) {
                    nvfp4_schedule_ = nvfp4_schedule_init;
                    nvfp4_quant_mode_ = nvfp4_quant_init;
                    nvfp4_stage_count_ = nvfp4_stage_init;
                    nvfp4_decomp_ = nvfp4_decomp_init;
                    nvfp4_splits_ = nvfp4_splits_init;
                    bitnet_cuda::nvfp4_set_schedule(nvfp4_schedule_);
                    bitnet_cuda::nvfp4_set_quant_mode(nvfp4_quant_mode_);
                    bitnet_cuda::nvfp4_set_stage_count(nvfp4_stage_count_);
                    bitnet_cuda::nvfp4_set_decomposition(nvfp4_decomp_);
                    bitnet_cuda::nvfp4_set_splits(nvfp4_splits_);
                    quantize_nvfp4_base_weights();
                    refresh_nvfp4_activation_buffers();
                    if (prewarm_nvfp4_gemm_cache()) {
                        static bool warned_revert = false;
                        if (!warned_revert) {
                            fprintf(stderr, "nvfp4 autotune config failed prewarm; reverting to defaults for cuda_graph.\n");
                            warned_revert = true;
                        }
                        restored = true;
                    }
                }
                if (!restored) {
                    static bool warned_graph = false;
                    if (!warned_graph) {
                        fprintf(stderr, "nvfp4 GEMM prewarm failed; disabling cuda_graph.\n");
                        warned_graph = true;
                    }
                    use_cuda_graph_ = false;
                }
            }
        }

    }

    void pack_i2_weights(const int8_t* d_src,
                         int rows, int cols,
                         uint32_t* d_dst) {
        if (use_packed_gemm_bitnet_) {
            bitnet_cuda::pack_ternary_i2_bitnet(d_src, rows, cols, d_dst, stream_);
        } else {
            bitnet_cuda::pack_ternary_i2(d_src, rows, cols, d_dst, stream_);
        }
    }

    void pack_base_weights() {
        if (!use_packed_gemm_) return;
        const int rows_hh = L_ * H_;
        const int rows_hf1 = L_ * F_;
        const int rows_hf2 = L_ * H_;
        pack_i2_weights(d_win_, H_, H_, d_win_packed_);
        pack_i2_weights(d_wq_, rows_hh, H_, d_wq_packed_);
        pack_i2_weights(d_wk_, rows_hh, H_, d_wk_packed_);
        pack_i2_weights(d_wv_, rows_hh, H_, d_wv_packed_);
        pack_i2_weights(d_wbeta_, L_, H_, d_wbeta_packed_);
        pack_i2_weights(d_ff1_, rows_hf1, H_, d_ff1_packed_);
        pack_i2_weights(d_ff2_, rows_hf2, F_, d_ff2_packed_);
        pack_i2_weights(d_head_, V_, H_, d_head_packed_);
    }

    void quantize_nvfp4_base_weights_range(int layer_begin,
                                           int layer_end,
                                           bool include_emb,
                                           bool include_win,
                                           bool include_head) {
        if (!use_nvfp4_) return;
        if (include_emb) {
            bitnet_cuda::nvfp4_quantize_b(d_emb_w_, B_, V_, H_, H_,
                                          d_emb_nvfp4_, d_emb_sfb_, stream_);
        }
        if (include_win) {
            bitnet_cuda::nvfp4_quantize_b(d_win_, B_, H_, H_, H_,
                                          d_win_nvfp4_, d_win_sfb_, stream_);
        }
        int lb = std::max(0, layer_begin);
        int le = std::min(L_, layer_end);
        for (int l = lb; l < le; ++l) {
            const size_t off_hh = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(H_);
            const size_t off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
            const size_t off_hf = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(F_);
            if (use_fused_qkv_nvfp4_) {
                bitnet_cuda::nvfp4_quantize_wqkv_fused(d_wq_ + off_hh,
                                                       d_wk_ + off_hh,
                                                       d_wv_ + off_hh,
                                                       d_wbeta_ + off_h,
                                                       B_, H_, H_, qkv_fused_cols_,
                                                       q_out_scale_,
                                                       k_out_scale_,
                                                       v_out_scale_,
                                                       beta_out_scale_,
                                                       d_wqkv_nvfp4_[static_cast<size_t>(l)],
                                                       d_wqkv_sfb_[static_cast<size_t>(l)],
                                                       stream_);
            } else {
                bitnet_cuda::nvfp4_quantize_b(d_wq_ + off_hh, B_, H_, H_, H_,
                                              d_wq_nvfp4_[static_cast<size_t>(l)],
                                              d_wq_sfb_[static_cast<size_t>(l)], stream_);
                bitnet_cuda::nvfp4_quantize_b(d_wk_ + off_hh, B_, H_, H_, H_,
                                              d_wk_nvfp4_[static_cast<size_t>(l)],
                                              d_wk_sfb_[static_cast<size_t>(l)], stream_);
                bitnet_cuda::nvfp4_quantize_b(d_wv_ + off_hh, B_, H_, H_, H_,
                                              d_wv_nvfp4_[static_cast<size_t>(l)],
                                              d_wv_sfb_[static_cast<size_t>(l)], stream_);
                bitnet_cuda::nvfp4_quantize_b(d_wbeta_ + off_h, B_, 1, H_, H_,
                                              d_wbeta_nvfp4_[static_cast<size_t>(l)],
                                              d_wbeta_sfb_[static_cast<size_t>(l)], stream_);
            }
            bitnet_cuda::nvfp4_quantize_b(d_ff1_ + off_hf, B_, F_, H_, H_,
                                          d_ff1_nvfp4_[static_cast<size_t>(l)],
                                          d_ff1_sfb_[static_cast<size_t>(l)], stream_);
            bitnet_cuda::nvfp4_quantize_b(d_ff2_ + off_hf, B_, H_, F_, F_,
                                          d_ff2_nvfp4_[static_cast<size_t>(l)],
                                          d_ff2_sfb_[static_cast<size_t>(l)], stream_);
        }
        if (include_head) {
            bitnet_cuda::nvfp4_quantize_b(d_head_, B_, V_, H_, H_,
                                          d_head_nvfp4_, d_head_sfb_, stream_);
        }
    }

    void quantize_nvfp4_base_weights() {
        quantize_nvfp4_base_weights_range(0, L_, true, true, true);
    }

    void quantize_nvfp4_noise_weights() {
        if (!use_nvfp4_) return;
        if (!d_emb_noise_nvfp4_ || !d_emb_noise_sfb_) return;
        bitnet_cuda::nvfp4_quantize_b(d_emb_noise_, B_, V_, H_, H_,
                                      d_emb_noise_nvfp4_, d_emb_noise_sfb_, stream_);
        bitnet_cuda::nvfp4_quantize_b(d_win_noise_, B_, H_, H_, H_,
                                      d_win_noise_nvfp4_, d_win_noise_sfb_, stream_);
        bitnet_cuda::nvfp4_quantize_b(d_head_noise_, B_, V_, H_, H_,
                                      d_head_noise_nvfp4_, d_head_noise_sfb_, stream_);
        const size_t hh = static_cast<size_t>(H_) * static_cast<size_t>(H_);
        const size_t hf = static_cast<size_t>(H_) * static_cast<size_t>(F_);
        const bool has_fused_noise = use_fused_qkv_nvfp4_ &&
                                     qkv_fused_cols_ > 0 &&
                                     !d_wqkv_noise_nvfp4_.empty() &&
                                     !d_wqkv_noise_sfb_.empty();
        for (int l = 0; l < L_; ++l) {
            const size_t off_hh = static_cast<size_t>(l) * hh;
            const size_t off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
            const size_t off_hf = static_cast<size_t>(l) * hf;
            const bool fused_qkv_noise = has_fused_noise &&
                                         d_wqkv_noise_nvfp4_[static_cast<size_t>(l)] &&
                                         d_wqkv_noise_sfb_[static_cast<size_t>(l)];
            if (fused_qkv_noise) {
                bitnet_cuda::nvfp4_quantize_wqkv_fused(d_wq_noise_ + off_hh,
                                                       d_wk_noise_ + off_hh,
                                                       d_wv_noise_ + off_hh,
                                                       d_wbeta_noise_ + off_h,
                                                       B_, H_, H_, qkv_fused_cols_,
                                                       q_out_scale_,
                                                       k_out_scale_,
                                                       v_out_scale_,
                                                       beta_out_scale_,
                                                       d_wqkv_noise_nvfp4_[static_cast<size_t>(l)],
                                                       d_wqkv_noise_sfb_[static_cast<size_t>(l)],
                                                       stream_);
            } else {
                bitnet_cuda::nvfp4_quantize_b(d_wq_noise_ + off_hh, B_, H_, H_, H_,
                                              d_wq_noise_nvfp4_[static_cast<size_t>(l)],
                                              d_wq_noise_sfb_[static_cast<size_t>(l)], stream_);
                bitnet_cuda::nvfp4_quantize_b(d_wk_noise_ + off_hh, B_, H_, H_, H_,
                                              d_wk_noise_nvfp4_[static_cast<size_t>(l)],
                                              d_wk_noise_sfb_[static_cast<size_t>(l)], stream_);
                bitnet_cuda::nvfp4_quantize_b(d_wv_noise_ + off_hh, B_, H_, H_, H_,
                                              d_wv_noise_nvfp4_[static_cast<size_t>(l)],
                                              d_wv_noise_sfb_[static_cast<size_t>(l)], stream_);
                bitnet_cuda::nvfp4_quantize_b(d_wbeta_noise_ + off_h, B_, 1, H_, H_,
                                              d_wbeta_noise_nvfp4_[static_cast<size_t>(l)],
                                              d_wbeta_noise_sfb_[static_cast<size_t>(l)], stream_);
            }
            bitnet_cuda::nvfp4_quantize_b(d_ff1_noise_ + off_hf, B_, F_, H_, H_,
                                          d_ff1_noise_nvfp4_[static_cast<size_t>(l)],
                                          d_ff1_noise_sfb_[static_cast<size_t>(l)], stream_);
            bitnet_cuda::nvfp4_quantize_b(d_ff2_noise_ + off_hf, B_, H_, F_, F_,
                                          d_ff2_noise_nvfp4_[static_cast<size_t>(l)],
                                          d_ff2_noise_sfb_[static_cast<size_t>(l)], stream_);
        }
    }

    struct Nvfp4TuneResult {
        bitnet_cuda::Nvfp4Schedule schedule = bitnet_cuda::Nvfp4Schedule::Auto;
        bitnet_cuda::Nvfp4QuantMode quant = bitnet_cuda::Nvfp4QuantMode::Warp16;
        bitnet_cuda::Nvfp4StageCount stages = bitnet_cuda::Nvfp4StageCount::Auto;
        bitnet_cuda::Nvfp4Decomposition decomp = bitnet_cuda::Nvfp4Decomposition::Heuristic;
        int splits = 1;
        float ms = std::numeric_limits<float>::infinity();
        bool valid = false;
    };

    struct Nvfp4TuneKey {
        int batch = 0;
        int hidden = 0;
        int ffn_dim = 0;
        int vocab = 0;
        int layers = 0;
        bool fused_qkv = false;
    };

    static bool nvfp4_tune_key_equal(const Nvfp4TuneKey& a, const Nvfp4TuneKey& b) {
        return a.batch == b.batch &&
               a.hidden == b.hidden &&
               a.ffn_dim == b.ffn_dim &&
               a.vocab == b.vocab &&
               a.layers == b.layers &&
               a.fused_qkv == b.fused_qkv;
    }

    void refresh_nvfp4_activation_buffers() {
        if (!use_nvfp4_) return;
        const size_t h_bytes = static_cast<size_t>(B_) * H_ * sizeof(float);
        cuda_check(cudaMemsetAsync(d_emb_f_, 0, h_bytes, stream_), "cudaMemsetAsync emb_f");
        bitnet_cuda::activation_quantize_nvfp4(d_emb_f_, B_, H_, H_, /*activation*/0,
                                               d_emb_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
        const size_t f_bytes = static_cast<size_t>(B_) * F_ * sizeof(float);
        cuda_check(cudaMemsetAsync(d_ff1_f_, 0, f_bytes, stream_), "cudaMemsetAsync ff1_f");
        bitnet_cuda::activation_quantize_nvfp4(d_ff1_f_, B_, F_, F_, /*activation*/0,
                                               d_ff1_q_, d_a_nvfp4_f_, d_sfa_f_, stream_);
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize nvfp4 act");
    }

    void run_nvfp4_workload() {
        if (!use_nvfp4_) return;
        const float* d_c = nullptr;
        d_c = d_inproj_f_;
        bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_win_nvfp4_,
                                          d_sfa_h_, d_win_sfb_,
                                          B_, H_, H_,
                                          1.0f, 0.0f,
                                          d_c, d_inproj_f_,
                                          stream_);
        for (int l = 0; l < L_; ++l) {
            if (use_fused_qkv_nvfp4_ && !d_wqkv_nvfp4_.empty()) {
                d_c = d_qkv_f_;
                bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_wqkv_nvfp4_[static_cast<size_t>(l)],
                                                  d_sfa_h_, d_wqkv_sfb_[static_cast<size_t>(l)],
                                                  B_, qkv_fused_cols_, H_,
                                                  1.0f, 0.0f,
                                                  d_c, d_qkv_f_,
                                                  stream_);
            } else {
                d_c = d_q_;
                bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_wq_nvfp4_[static_cast<size_t>(l)],
                                                  d_sfa_h_, d_wq_sfb_[static_cast<size_t>(l)],
                                                  B_, H_, H_,
                                                  1.0f, 0.0f,
                                                  d_c, d_q_,
                                                  stream_);
                d_c = d_k_;
                bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_wk_nvfp4_[static_cast<size_t>(l)],
                                                  d_sfa_h_, d_wk_sfb_[static_cast<size_t>(l)],
                                                  B_, H_, H_,
                                                  1.0f, 0.0f,
                                                  d_c, d_k_,
                                                  stream_);
                d_c = d_v_;
                bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_wv_nvfp4_[static_cast<size_t>(l)],
                                                  d_sfa_h_, d_wv_sfb_[static_cast<size_t>(l)],
                                                  B_, H_, H_,
                                                  1.0f, 0.0f,
                                                  d_c, d_v_,
                                                  stream_);
                d_c = d_beta_raw_;
                bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_wbeta_nvfp4_[static_cast<size_t>(l)],
                                                  d_sfa_h_, d_wbeta_sfb_[static_cast<size_t>(l)],
                                                  B_, 1, H_,
                                                  1.0f, 0.0f,
                                                  d_c, d_beta_raw_,
                                                  stream_);
            }
            d_c = d_ff1_f_;
            bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_ff1_nvfp4_[static_cast<size_t>(l)],
                                              d_sfa_h_, d_ff1_sfb_[static_cast<size_t>(l)],
                                              B_, F_, H_,
                                              1.0f, 0.0f,
                                              d_c, d_ff1_f_,
                                              stream_);
            d_c = d_y_;
            bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_f_, d_ff2_nvfp4_[static_cast<size_t>(l)],
                                              d_sfa_f_, d_ff2_sfb_[static_cast<size_t>(l)],
                                              B_, H_, F_,
                                              1.0f, 0.0f,
                                              d_c, d_y_,
                                              stream_);
        }
        d_c = d_logits_;
        bitnet_cuda::gemm_ternary_f_nvfp4(d_a_nvfp4_h_, d_head_nvfp4_,
                                          d_sfa_h_, d_head_sfb_,
                                          B_, V_, H_,
                                          1.0f, 0.0f,
                                          d_c, d_logits_,
                                          stream_);
    }

    float benchmark_nvfp4_workload(int warmup_iters, int iters) {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cuda_check(cudaEventCreate(&start), "cudaEventCreate nvfp4 start");
        cuda_check(cudaEventCreate(&stop), "cudaEventCreate nvfp4 stop");
        for (int i = 0; i < warmup_iters; ++i) {
            run_nvfp4_workload();
        }
        cuda_check(cudaEventRecord(start, stream_), "cudaEventRecord nvfp4 start");
        for (int i = 0; i < iters; ++i) {
            run_nvfp4_workload();
        }
        cuda_check(cudaEventRecord(stop, stream_), "cudaEventRecord nvfp4 stop");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize nvfp4 stop");
        float ms = std::numeric_limits<float>::infinity();
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime nvfp4");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (cudaGetLastError() != cudaSuccess) {
            return std::numeric_limits<float>::infinity();
        }
        return (iters > 0) ? (ms / static_cast<float>(iters)) : ms;
    }

    Nvfp4TuneResult autotune_nvfp4_impl() {
        Nvfp4TuneResult best;
        if (!use_nvfp4_) return best;
        cuda_check(cudaSetDevice(device_), "cudaSetDevice nvfp4 autotune");
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize nvfp4 autotune");

        std::vector<bitnet_cuda::Nvfp4Schedule> schedules;
        if (nvfp4_tune_schedule_) {
            schedules = {bitnet_cuda::Nvfp4Schedule::Auto,
                         bitnet_cuda::Nvfp4Schedule::Cooperative,
                         bitnet_cuda::Nvfp4Schedule::Pingpong};
        } else {
            schedules = {nvfp4_schedule_};
        }
        std::vector<bitnet_cuda::Nvfp4QuantMode> quants;
        if (nvfp4_tune_quant_) {
            quants = {bitnet_cuda::Nvfp4QuantMode::Warp16,
                      bitnet_cuda::Nvfp4QuantMode::Warp4};
        } else {
            quants = {nvfp4_quant_mode_};
        }
        std::vector<bitnet_cuda::Nvfp4StageCount> stages;
        if (nvfp4_tune_stages_) {
            stages = {bitnet_cuda::Nvfp4StageCount::Auto,
                      bitnet_cuda::Nvfp4StageCount::Stages2,
                      bitnet_cuda::Nvfp4StageCount::Stages3,
                      bitnet_cuda::Nvfp4StageCount::Stages4};
        } else {
            stages = {nvfp4_stage_count_};
        }
        std::vector<bitnet_cuda::Nvfp4Decomposition> decomps;
        if (nvfp4_tune_decomp_) {
            decomps = {bitnet_cuda::Nvfp4Decomposition::Heuristic,
                       bitnet_cuda::Nvfp4Decomposition::DataParallel,
                       bitnet_cuda::Nvfp4Decomposition::SplitK,
                       bitnet_cuda::Nvfp4Decomposition::StreamK};
        } else {
            decomps = {nvfp4_decomp_};
        }
        std::vector<int> split_candidates;
        if (nvfp4_tune_splits_) {
            split_candidates = {1, 2, 4};
        } else {
            split_candidates = {std::max(1, nvfp4_splits_)};
        }

        int total = 0;
        for (auto decomp : decomps) {
            const bool use_splits = (decomp == bitnet_cuda::Nvfp4Decomposition::SplitK ||
                                     decomp == bitnet_cuda::Nvfp4Decomposition::StreamK);
            const int splits_count = use_splits ? static_cast<int>(split_candidates.size()) : 1;
            total += static_cast<int>(schedules.size() * quants.size() * stages.size()) * splits_count;
        }
        fprintf(stderr, "nvfp4 autotune: testing %d configs\n", total);
        bitnet_cuda::nvfp4_set_verbose(false);

        bitnet_cuda::Nvfp4QuantMode current_quant = nvfp4_quant_mode_;
        bitnet_cuda::nvfp4_set_quant_mode(current_quant);
        refresh_nvfp4_activation_buffers();

        const int warmup_iters = 2;
        const int bench_iters = 5;

        for (auto quant : quants) {
            if (quant != current_quant) {
                current_quant = quant;
                nvfp4_quant_mode_ = quant;
                bitnet_cuda::nvfp4_set_quant_mode(quant);
                quantize_nvfp4_base_weights();
                refresh_nvfp4_activation_buffers();
            }
            for (auto schedule : schedules) {
                for (auto stage : stages) {
                    for (auto decomp : decomps) {
                        const bool use_splits = (decomp == bitnet_cuda::Nvfp4Decomposition::SplitK ||
                                                 decomp == bitnet_cuda::Nvfp4Decomposition::StreamK);
                        if (use_splits) {
                            for (int splits : split_candidates) {
                                nvfp4_schedule_ = schedule;
                                nvfp4_stage_count_ = stage;
                                nvfp4_decomp_ = decomp;
                                nvfp4_splits_ = std::max(1, splits);
                                bitnet_cuda::nvfp4_set_schedule(schedule);
                                bitnet_cuda::nvfp4_set_stage_count(stage);
                                bitnet_cuda::nvfp4_set_decomposition(decomp);
                                bitnet_cuda::nvfp4_set_splits(nvfp4_splits_);

                                if (!prewarm_nvfp4_gemm_cache(false)) {
                                    continue;
                                }
                                const float ms = benchmark_nvfp4_workload(warmup_iters, bench_iters);
                                if (ms < best.ms) {
                                    best.ms = ms;
                                    best.schedule = schedule;
                                    best.quant = quant;
                                    best.stages = stage;
                                    best.decomp = decomp;
                                    best.splits = nvfp4_splits_;
                                    best.valid = true;
                                }
                            }
                        } else {
                            nvfp4_schedule_ = schedule;
                            nvfp4_stage_count_ = stage;
                            nvfp4_decomp_ = decomp;
                            nvfp4_splits_ = 1;
                            bitnet_cuda::nvfp4_set_schedule(schedule);
                            bitnet_cuda::nvfp4_set_stage_count(stage);
                            bitnet_cuda::nvfp4_set_decomposition(decomp);
                            bitnet_cuda::nvfp4_set_splits(nvfp4_splits_);

                            if (!prewarm_nvfp4_gemm_cache(false)) {
                                continue;
                            }
                            const float ms = benchmark_nvfp4_workload(warmup_iters, bench_iters);
                            if (ms < best.ms) {
                                best.ms = ms;
                                best.schedule = schedule;
                                best.quant = quant;
                                best.stages = stage;
                                best.decomp = decomp;
                                best.splits = nvfp4_splits_;
                                best.valid = true;
                            }
                        }
                    }
                }
            }
        }

        if (!best.valid) {
            bitnet_cuda::nvfp4_set_verbose(true);
            fprintf(stderr, "nvfp4 autotune: no valid configs found; keeping defaults\n");
            return best;
        }

        if (best.quant != current_quant) {
            current_quant = best.quant;
            nvfp4_quant_mode_ = best.quant;
            bitnet_cuda::nvfp4_set_quant_mode(best.quant);
            quantize_nvfp4_base_weights();
            refresh_nvfp4_activation_buffers();
        }
        nvfp4_schedule_ = best.schedule;
        nvfp4_stage_count_ = best.stages;
        nvfp4_decomp_ = best.decomp;
        nvfp4_splits_ = std::max(1, best.splits);
        bitnet_cuda::nvfp4_set_schedule(best.schedule);
        bitnet_cuda::nvfp4_set_stage_count(best.stages);
        bitnet_cuda::nvfp4_set_decomposition(best.decomp);
        bitnet_cuda::nvfp4_set_splits(nvfp4_splits_);

        const char* sched_name = (best.schedule == bitnet_cuda::Nvfp4Schedule::Cooperative) ? "cooperative" :
                                 (best.schedule == bitnet_cuda::Nvfp4Schedule::Pingpong) ? "pingpong" : "auto";
        const char* quant_name = (best.quant == bitnet_cuda::Nvfp4QuantMode::Warp4) ? "warp4" : "warp16";
        const char* stage_name = "auto";
        switch (best.stages) {
            case bitnet_cuda::Nvfp4StageCount::Stages2: stage_name = "2"; break;
            case bitnet_cuda::Nvfp4StageCount::Stages3: stage_name = "3"; break;
            case bitnet_cuda::Nvfp4StageCount::Stages4: stage_name = "4"; break;
            case bitnet_cuda::Nvfp4StageCount::Auto: default: stage_name = "auto"; break;
        }
        const char* decomp_name = "auto";
        switch (best.decomp) {
            case bitnet_cuda::Nvfp4Decomposition::DataParallel: decomp_name = "data"; break;
            case bitnet_cuda::Nvfp4Decomposition::SplitK: decomp_name = "splitk"; break;
            case bitnet_cuda::Nvfp4Decomposition::StreamK: decomp_name = "streamk"; break;
            case bitnet_cuda::Nvfp4Decomposition::Heuristic: default: decomp_name = "auto"; break;
        }
        fprintf(stderr, "nvfp4 autotune: best schedule=%s quant=%s stages=%s decomp=%s splits=%d (%.3f ms)\n",
                sched_name, quant_name, stage_name, decomp_name, nvfp4_splits_, best.ms);
        bitnet_cuda::nvfp4_set_verbose(true);
        return best;
    }

    void maybe_autotune_nvfp4() {
        if (!use_nvfp4_ || !nvfp4_autotune_) return;
        static std::mutex tune_mutex;
        static std::vector<std::pair<Nvfp4TuneKey, Nvfp4TuneResult>> tune_cache;
        Nvfp4TuneKey key{B_, H_, F_, V_, L_, use_fused_qkv_nvfp4_};
        std::lock_guard<std::mutex> lock(tune_mutex);
        for (const auto& entry : tune_cache) {
            if (nvfp4_tune_key_equal(entry.first, key)) {
                const Nvfp4TuneResult& cached = entry.second;
                if (cached.valid) {
                    const bool quant_changed = (nvfp4_quant_mode_ != cached.quant);
                    nvfp4_schedule_ = cached.schedule;
                    nvfp4_quant_mode_ = cached.quant;
                    nvfp4_stage_count_ = cached.stages;
                    nvfp4_decomp_ = cached.decomp;
                    nvfp4_splits_ = std::max(1, cached.splits);
                    bitnet_cuda::nvfp4_set_schedule(nvfp4_schedule_);
                    bitnet_cuda::nvfp4_set_quant_mode(nvfp4_quant_mode_);
                    bitnet_cuda::nvfp4_set_stage_count(nvfp4_stage_count_);
                    bitnet_cuda::nvfp4_set_decomposition(nvfp4_decomp_);
                    bitnet_cuda::nvfp4_set_splits(nvfp4_splits_);
                    if (quant_changed) {
                        quantize_nvfp4_base_weights();
                    }
                    refresh_nvfp4_activation_buffers();
                    return;
                }
            }
        }

        Nvfp4TuneResult best = autotune_nvfp4_impl();
        tune_cache.emplace_back(key, best);
    }

    void set_noise_weights(const EflaLmWeights& n) {
        if (static_cast<int>(n.wq.size()) != L_ ||
            static_cast<int>(n.wk.size()) != L_ ||
            static_cast<int>(n.wv.size()) != L_ ||
            static_cast<int>(n.wbeta.size()) != L_ ||
            static_cast<int>(n.ff1.size()) != L_ ||
            static_cast<int>(n.ff2.size()) != L_) {
            throw std::runtime_error("set_noise_weights: layer count mismatch");
        }

        cuda_check(cudaMemcpyAsync(d_emb_noise_, n.emb.w.data(), n.emb.w.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync emb_noise");
        cuda_check(cudaMemcpyAsync(d_win_noise_, n.win.w.data(), n.win.w.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync win_noise");

        const size_t hh = static_cast<size_t>(H_) * H_;
        h_wq_pack_.resize(static_cast<size_t>(L_) * hh);
        h_wk_pack_.resize(static_cast<size_t>(L_) * hh);
        h_wv_pack_.resize(static_cast<size_t>(L_) * hh);
        h_wbeta_pack_.resize(static_cast<size_t>(L_) * static_cast<size_t>(H_));
        const size_t hf = static_cast<size_t>(H_) * static_cast<size_t>(F_);
        h_ff1_pack_.resize(static_cast<size_t>(L_) * hf);
        h_ff2_pack_.resize(static_cast<size_t>(L_) * hf);
        for (int l = 0; l < L_; ++l) {
            std::copy(n.wq[static_cast<size_t>(l)].w.begin(),
                      n.wq[static_cast<size_t>(l)].w.end(),
                      h_wq_pack_.begin() + static_cast<size_t>(l) * hh);
            std::copy(n.wk[static_cast<size_t>(l)].w.begin(),
                      n.wk[static_cast<size_t>(l)].w.end(),
                      h_wk_pack_.begin() + static_cast<size_t>(l) * hh);
            std::copy(n.wv[static_cast<size_t>(l)].w.begin(),
                      n.wv[static_cast<size_t>(l)].w.end(),
                      h_wv_pack_.begin() + static_cast<size_t>(l) * hh);
            const auto& wb = n.wbeta[static_cast<size_t>(l)].w;
            std::copy(wb.begin(), wb.end(),
                      h_wbeta_pack_.begin() + static_cast<size_t>(l) * static_cast<size_t>(H_));

            std::copy(n.ff1[static_cast<size_t>(l)].w.begin(),
                      n.ff1[static_cast<size_t>(l)].w.end(),
                      h_ff1_pack_.begin() + static_cast<size_t>(l) * hf);
            std::copy(n.ff2[static_cast<size_t>(l)].w.begin(),
                      n.ff2[static_cast<size_t>(l)].w.end(),
                      h_ff2_pack_.begin() + static_cast<size_t>(l) * hf);
        }

        cuda_check(cudaMemcpyAsync(d_wq_noise_, h_wq_pack_.data(), h_wq_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wq_noise");
        cuda_check(cudaMemcpyAsync(d_wk_noise_, h_wk_pack_.data(), h_wk_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wk_noise");
        cuda_check(cudaMemcpyAsync(d_wv_noise_, h_wv_pack_.data(), h_wv_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wv_noise");
        cuda_check(cudaMemcpyAsync(d_wbeta_noise_, h_wbeta_pack_.data(), h_wbeta_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync wbeta_noise");
        cuda_check(cudaMemcpyAsync(d_ff1_noise_, h_ff1_pack_.data(), h_ff1_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync ff1_noise");
        cuda_check(cudaMemcpyAsync(d_ff2_noise_, h_ff2_pack_.data(), h_ff2_pack_.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync ff2_noise");

        cuda_check(cudaMemcpyAsync(d_head_noise_, n.head.w.data(), n.head.w.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync head_noise");
        if (use_packed_gemm_) {
            const int rows_hh = L_ * H_;
            const int rows_hf1 = L_ * F_;
            const int rows_hf2 = L_ * H_;
            pack_i2_weights(d_win_noise_, H_, H_, d_win_noise_packed_);
            pack_i2_weights(d_wq_noise_, rows_hh, H_, d_wq_noise_packed_);
            pack_i2_weights(d_wk_noise_, rows_hh, H_, d_wk_noise_packed_);
            pack_i2_weights(d_wv_noise_, rows_hh, H_, d_wv_noise_packed_);
            pack_i2_weights(d_wbeta_noise_, L_, H_, d_wbeta_noise_packed_);
            pack_i2_weights(d_ff1_noise_, rows_hf1, H_, d_ff1_noise_packed_);
            pack_i2_weights(d_ff2_noise_, rows_hf2, F_, d_ff2_noise_packed_);
            pack_i2_weights(d_head_noise_, V_, H_, d_head_noise_packed_);
        }
        quantize_nvfp4_noise_weights();
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize set_noise_weights");
    }

    void set_noise_seeds(uint64_t seed_emb,
                         uint64_t seed_win,
                         const std::vector<uint64_t>& seed_wq,
                         const std::vector<uint64_t>& seed_wk,
                         const std::vector<uint64_t>& seed_wv,
                         const std::vector<uint64_t>& seed_wbeta,
                         const std::vector<uint64_t>& seed_ff1,
                         const std::vector<uint64_t>& seed_ff2,
                         uint64_t seed_head,
                         bool use_clt,
                         int clt_k) {
        const size_t emb_n = static_cast<size_t>(V_) * H_;
        const size_t hh = static_cast<size_t>(H_) * H_;
        const size_t hf = static_cast<size_t>(H_) * static_cast<size_t>(F_);
        if (use_noise_batch_ && d_noise_desc_ && !noise_desc_host_.empty()) {
            size_t idx = 0;
            noise_desc_host_[idx++].seed = seed_emb;
            noise_desc_host_[idx++].seed = seed_win;
            for (int l = 0; l < L_; ++l) {
                noise_desc_host_[idx++].seed = seed_wq[static_cast<size_t>(l)];
                noise_desc_host_[idx++].seed = seed_wk[static_cast<size_t>(l)];
                noise_desc_host_[idx++].seed = seed_wv[static_cast<size_t>(l)];
                noise_desc_host_[idx++].seed = seed_wbeta[static_cast<size_t>(l)];
                noise_desc_host_[idx++].seed = seed_ff1[static_cast<size_t>(l)];
                noise_desc_host_[idx++].seed = seed_ff2[static_cast<size_t>(l)];
            }
            if (idx < noise_desc_host_.size()) {
                noise_desc_host_[idx++].seed = seed_head;
            }
            cuda_check(cudaMemcpyAsync(d_noise_desc_, noise_desc_host_.data(),
                                       static_cast<size_t>(noise_desc_count_) * sizeof(efla_lm_cuda::NoiseDesc),
                                       cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync noise_desc");
            efla_lm_cuda::fill_ternary_noise_batched(d_noise_desc_, noise_desc_count_, noise_desc_max_blocks_,
                                                     use_clt, clt_k, stream_);
        } else {
            efla_lm_cuda::fill_ternary_noise(d_emb_noise_, emb_n, seed_emb, use_clt, clt_k, stream_);
            efla_lm_cuda::fill_ternary_noise(d_win_noise_, hh, seed_win, use_clt, clt_k, stream_);
            for (int l = 0; l < L_; ++l) {
                const size_t off_hh = static_cast<size_t>(l) * hh;
                const size_t off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
                const size_t off_hf = static_cast<size_t>(l) * hf;
                efla_lm_cuda::fill_ternary_noise(d_wq_noise_ + off_hh, hh, seed_wq[static_cast<size_t>(l)], use_clt, clt_k, stream_);
                efla_lm_cuda::fill_ternary_noise(d_wk_noise_ + off_hh, hh, seed_wk[static_cast<size_t>(l)], use_clt, clt_k, stream_);
                efla_lm_cuda::fill_ternary_noise(d_wv_noise_ + off_hh, hh, seed_wv[static_cast<size_t>(l)], use_clt, clt_k, stream_);
                efla_lm_cuda::fill_ternary_noise(d_wbeta_noise_ + off_h, static_cast<size_t>(H_),
                                                 seed_wbeta[static_cast<size_t>(l)], use_clt, clt_k, stream_);
                efla_lm_cuda::fill_ternary_noise(d_ff1_noise_ + off_hf, hf, seed_ff1[static_cast<size_t>(l)], use_clt, clt_k, stream_);
                efla_lm_cuda::fill_ternary_noise(d_ff2_noise_ + off_hf, hf, seed_ff2[static_cast<size_t>(l)], use_clt, clt_k, stream_);
            }
            efla_lm_cuda::fill_ternary_noise(d_head_noise_, emb_n, seed_head, use_clt, clt_k, stream_);
        }
        if (use_packed_gemm_) {
            const int rows_hh = L_ * H_;
            const int rows_hf1 = L_ * F_;
            const int rows_hf2 = L_ * H_;
            pack_i2_weights(d_win_noise_, H_, H_, d_win_noise_packed_);
            pack_i2_weights(d_wq_noise_, rows_hh, H_, d_wq_noise_packed_);
            pack_i2_weights(d_wk_noise_, rows_hh, H_, d_wk_noise_packed_);
            pack_i2_weights(d_wv_noise_, rows_hh, H_, d_wv_noise_packed_);
            pack_i2_weights(d_wbeta_noise_, L_, H_, d_wbeta_noise_packed_);
            pack_i2_weights(d_ff1_noise_, rows_hf1, H_, d_ff1_noise_packed_);
            pack_i2_weights(d_ff2_noise_, rows_hf2, F_, d_ff2_noise_packed_);
            pack_i2_weights(d_head_noise_, V_, H_, d_head_noise_packed_);
        }
        quantize_nvfp4_noise_weights();
    }

    void upload_tokens_async(const std::vector<std::vector<uint8_t>>& inputs,
                             const std::vector<std::vector<uint8_t>>& targets,
                             int buffer_idx) {
        if (buffer_idx < 0 || buffer_idx >= kTokenBuffers) {
            throw std::runtime_error("upload_tokens_async buffer index out of range");
        }
        if (static_cast<int>(inputs.size()) != B_ || static_cast<int>(targets.size()) != B_) {
            throw std::runtime_error("upload_tokens batch mismatch");
        }
        const size_t total = static_cast<size_t>(B_) * T_;
        uint8_t* h_tokens = h_tokens_pinned_[buffer_idx];
        uint8_t* h_targets = h_targets_pinned_[buffer_idx];
        for (int b = 0; b < B_; ++b) {
            if (static_cast<int>(inputs[b].size()) != T_ || static_cast<int>(targets[b].size()) != T_) {
                throw std::runtime_error("upload_tokens seq mismatch");
            }
            for (int t = 0; t < T_; ++t) {
                h_tokens[static_cast<size_t>(b) * T_ + t] = inputs[b][t];
                h_targets[static_cast<size_t>(b) * T_ + t] = targets[b][t];
            }
        }
        cuda_check(cudaMemcpyAsync(d_tokens_[buffer_idx], h_tokens, total,
                                   cudaMemcpyHostToDevice, upload_stream_),
                   "cudaMemcpyAsync tokens");
        cuda_check(cudaMemcpyAsync(d_targets_[buffer_idx], h_targets, total,
                                   cudaMemcpyHostToDevice, upload_stream_),
                   "cudaMemcpyAsync targets");
        cuda_check(cudaEventRecord(upload_done_[buffer_idx], upload_stream_), "cudaEventRecord upload_done");
    }

    void upload_tokens_async_packed(const uint8_t* inputs,
                                    const uint8_t* targets,
                                    int buffer_idx) {
        if (buffer_idx < 0 || buffer_idx >= kTokenBuffers) {
            throw std::runtime_error("upload_tokens_async_packed buffer index out of range");
        }
        if (!inputs || !targets) {
            throw std::runtime_error("upload_tokens_async_packed null inputs/targets");
        }
        const size_t total = static_cast<size_t>(B_) * T_;
        cuda_check(cudaMemcpyAsync(d_tokens_[buffer_idx], inputs, total,
                                   cudaMemcpyHostToDevice, upload_stream_),
                   "cudaMemcpyAsync tokens packed");
        cuda_check(cudaMemcpyAsync(d_targets_[buffer_idx], targets, total,
                                   cudaMemcpyHostToDevice, upload_stream_),
                   "cudaMemcpyAsync targets packed");
        cuda_check(cudaEventRecord(upload_done_[buffer_idx], upload_stream_), "cudaEventRecord upload_done packed");
    }

    void set_active_tokens(int buffer_idx) {
        if (buffer_idx < 0 || buffer_idx >= kTokenBuffers) {
            throw std::runtime_error("set_active_tokens buffer index out of range");
        }
        active_tokens_idx_ = buffer_idx;
        d_tokens_active_ = d_tokens_[buffer_idx];
        d_targets_active_ = d_targets_[buffer_idx];
        cuda_check(cudaStreamWaitEvent(stream_, upload_done_[buffer_idx], 0), "cudaStreamWaitEvent upload_done");
    }

    void upload_tokens(const std::vector<std::vector<uint8_t>>& inputs,
                       const std::vector<std::vector<uint8_t>>& targets,
                       int buffer_idx) {
        upload_tokens_async(inputs, targets, buffer_idx);
        set_active_tokens(buffer_idx);
    }

    void upload_tokens(const std::vector<std::vector<uint8_t>>& inputs,
                       const std::vector<std::vector<uint8_t>>& targets) {
        upload_tokens(inputs, targets, active_tokens_idx_);
    }

    int device() const { return device_; }
    cudaStream_t stream() const { return stream_; }
    int8_t* d_emb_w() const { return d_emb_w_; }
    int8_t* d_win_w() const { return d_win_; }
    int8_t* d_wq_w() const { return d_wq_; }
    int8_t* d_wk_w() const { return d_wk_; }
    int8_t* d_wv_w() const { return d_wv_; }
    int8_t* d_wbeta_w() const { return d_wbeta_; }
    int8_t* d_ff1_w() const { return d_ff1_; }
    int8_t* d_ff2_w() const { return d_ff2_; }
    int8_t* d_head_w() const { return d_head_; }
    uint32_t* d_win_packed() const { return d_win_packed_; }
    uint32_t* d_wq_packed() const { return d_wq_packed_; }
    uint32_t* d_wk_packed() const { return d_wk_packed_; }
    uint32_t* d_wv_packed() const { return d_wv_packed_; }
    uint32_t* d_wbeta_packed() const { return d_wbeta_packed_; }
    uint32_t* d_ff1_packed() const { return d_ff1_packed_; }
    uint32_t* d_ff2_packed() const { return d_ff2_packed_; }
    uint32_t* d_head_packed() const { return d_head_packed_; }
    bool use_packed_gemm() const { return use_packed_gemm_; }

    void set_head_bias(const float* head_bias) {
        if (!head_bias) {
            head_bias_cached_ = false;
            return;
        }
        cuda_check(cudaMemcpyAsync(d_head_bias_, head_bias, static_cast<size_t>(V_) * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync head_bias");
        head_bias_cached_ = true;
    }

    float evaluate_loss(int method,
                        int anti_sign,
                        int sigma_shift,
                        float act_scale,
                        float mlp_act_scale,
                        float ln_scale,
                        float residual_scale,
                        float mlp_residual_scale,
                        float gate_scale,
                        float state_decay,
                        bool int8_residual,
                        bool absmean_norm,
                        const float* head_bias,
                        const float* head_bias_noise,
                        const float* beta_bias,
                        const float* beta_bias_noise) {
        NvtxRange range_eval(use_nvtx_, "evaluate_loss");
        const float noise_scale =
            (anti_sign == 0) ? 0.0f : (1.0f / static_cast<float>(1 << std::max(0, sigma_shift)));
        const float bias_noise_scale = static_cast<float>(anti_sign) * noise_scale;
        const float eps = 1e-6f;

        const float* d_bias = nullptr;
        const float* d_bias_noise = nullptr;
        if (head_bias) {
            set_head_bias(head_bias);
            d_bias = d_head_bias_;
        } else if (head_bias_cached_) {
            d_bias = d_head_bias_;
        }
        if (head_bias_noise && bias_noise_scale != 0.0f) {
            cuda_check(cudaMemcpyAsync(d_head_bias_noise_, head_bias_noise, static_cast<size_t>(V_) * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync head_bias_noise");
            d_bias_noise = d_head_bias_noise_;
        }

        const bool use_transformer_residual = (L_ >= 2);
        const bool noise_active = (noise_scale != 0.0f);
        const bool capture_loss = use_cuda_graph_ && graph_full_eval_;
        const bool fused_head_loss =
            use_fused_head_loss_ &&
            !use_packed_gemm_ &&
            (gemm_backend_ == TrainConfig::GemmBackend::Dp4a) &&
            (V_ <= 256);

        auto efla_full_state = [&](float* d_S_f, __half* d_S_h, float beta_bias_eff, float* d_out,
                                   const float* d_qkv, int fused_cols, bool use_qkv_fused_layout) {
            if (use_qkv_fused_layout) {
                if (use_fp16_state_) {
                    efla_lm_cuda::efla_step_half_qkv_fused(d_S_h, d_qkv, fused_cols, beta_bias_eff,
                                                          H_, B_, method, eps,
                                                          d_out, d_q_norm_, d_k_usage_, d_kS_, d_diff_, d_alpha_,
                                                          stream_);
                } else {
                    efla_lm_cuda::efla_step_qkv_fused(d_S_f, d_qkv, fused_cols, beta_bias_eff,
                                                     H_, B_, method, eps,
                                                     d_out, d_q_norm_, d_k_usage_, d_kS_, d_diff_, d_alpha_,
                                                     stream_);
                }
                return;
            }
            if (use_fused_efla_) {
                if (use_fp16_state_) {
                    efla_lm_cuda::efla_step_fused_half(d_S_h, d_q_, d_k_, d_v_, d_beta_raw_, beta_bias_eff,
                                                       H_, B_, method, eps, d_out, stream_);
                } else {
                    efla_lm_cuda::efla_step_fused(d_S_f, d_q_, d_k_, d_v_, d_beta_raw_, beta_bias_eff,
                                                  H_, B_, method, eps, d_out, stream_);
                }
            } else {
                if (use_fp16_state_) {
                    efla_lm_cuda::efla_step_half(d_S_h, d_q_, d_k_, d_v_, d_beta_raw_, beta_bias_eff,
                                                 H_, B_, method, eps, use_efla_update_wmma_, use_efla_mixed_,
                                                 use_efla_fuse_diff_,
                                                 d_out, d_q_norm_, d_k_usage_, d_k_usage_h_,
                                                 d_kS_, d_diff_, d_diff_h_, d_alpha_, stream_);
                } else {
                    efla_lm_cuda::efla_step(d_S_f, d_q_, d_k_, d_v_, d_beta_raw_, beta_bias_eff,
                                            H_, B_, method, eps, use_efla_fuse_diff_,
                                            d_out, d_q_norm_, d_k_usage_, d_kS_, d_diff_, d_alpha_, stream_);
                }
            }
        };

        auto run_kernels = [&]() {
            cuda_check(cudaMemsetAsync(d_loss_sum_, 0, sizeof(float), stream_), "cudaMemsetAsync loss_sum");
            if (D_ > 0) {
                cuda_check(cudaMemsetAsync(d_K_f_, 0, state_bytes_, stream_),
                           "cudaMemsetAsync K");
                cuda_check(cudaMemsetAsync(d_V_f_, 0, state_bytes_, stream_),
                           "cudaMemsetAsync V");
            } else {
                void* d_S_ptr = use_fp16_state_ ? static_cast<void*>(d_S_h_) : static_cast<void*>(d_S_f_);
                cuda_check(cudaMemsetAsync(d_S_ptr, 0, state_bytes_, stream_),
                           "cudaMemsetAsync S");
            }

            const bool use_cutlass = (gemm_backend_ == TrainConfig::GemmBackend::Cutlass);
            const bool use_nvfp4 = (gemm_backend_ == TrainConfig::GemmBackend::CutlassNvfp4);
            struct Nvfp4ActCache {
                const int8_t* ptr = nullptr;
                int cols = 0;
                int version = -1;
            };
            Nvfp4ActCache cache_h;
            Nvfp4ActCache cache_f;
            int ver_emb_q = 0;
            int ver_x_q = 0;
            int ver_y_q = 0;
            int ver_ff1_q = 0;
            auto act_version = [&](const int8_t* ptr) -> int {
                if (ptr == d_emb_q_) return ver_emb_q;
                if (ptr == d_x_q_) return ver_x_q;
                if (ptr == d_y_q_) return ver_y_q;
                if (ptr == d_ff1_q_) return ver_ff1_q;
                return -1;
            };
            auto bump_version = [&](int8_t* ptr) {
                if (ptr == d_emb_q_) { ++ver_emb_q; return; }
                if (ptr == d_x_q_) { ++ver_x_q; return; }
                if (ptr == d_y_q_) { ++ver_y_q; return; }
                if (ptr == d_ff1_q_) { ++ver_ff1_q; return; }
            };
            auto get_nvfp4_act = [&](const int8_t* d_act, int act_cols) -> std::pair<const void*, const void*> {
                if (!use_nvfp4) return {nullptr, nullptr};
                void* d_a = nullptr;
                void* d_sfa = nullptr;
                Nvfp4ActCache* cache = nullptr;
                if (act_cols == H_) {
                    d_a = d_a_nvfp4_h_;
                    d_sfa = d_sfa_h_;
                    cache = &cache_h;
                } else if (act_cols == F_) {
                    d_a = d_a_nvfp4_f_;
                    d_sfa = d_sfa_f_;
                    cache = &cache_f;
                }
                if (!d_a || !d_sfa) return {nullptr, nullptr};
                const int ver = act_version(d_act);
                if (cache && ver >= 0 && cache->ptr == d_act && cache->cols == act_cols && cache->version == ver) {
                    return {d_a, d_sfa};
                }
                bitnet_cuda::nvfp4_quantize_a(d_act, B_, 1, act_cols, act_cols, d_a, d_sfa, stream_);
                if (cache && ver >= 0) {
                    cache->ptr = d_act;
                    cache->cols = act_cols;
                    cache->version = ver;
                }
                return {d_a, d_sfa};
            };
            auto set_nvfp4_act = [&](const int8_t* d_act, int act_cols) {
                if (!use_nvfp4) return;
                const int ver = act_version(d_act);
                if (ver < 0) return;
                if (act_cols == H_) {
                    cache_h.ptr = d_act;
                    cache_h.cols = act_cols;
                    cache_h.version = ver;
                } else if (act_cols == F_) {
                    cache_f.ptr = d_act;
                    cache_f.cols = act_cols;
                    cache_f.version = ver;
                }
            };

            auto gemm_noisy = [&](const int8_t* d_x,
                                  const int8_t* d_w,
                                  const int8_t* d_w_noise,
                                  const uint32_t* d_w_packed,
                                  const uint32_t* d_w_packed_noise,
                                  const float* d_scale,
                                  const void* d_w_nvfp4,
                                  const void* d_w_sfb,
                                  const void* d_w_noise_nvfp4,
                                  const void* d_w_noise_sfb,
                                  int rows,
                                  int cols,
                                  float out_scale,
                                  float* d_out) {
                if (use_packed_gemm_) {
                    const float fused_noise = noise_active ? static_cast<float>(anti_sign) * noise_scale : 0.0f;
                    if (noise_active && use_fused_noise_gemm_) {
                        if (use_packed_gemm_bitnet_) {
                            bitnet_cuda::gemm_ternary_f_i2_bitnet_noise(d_x, d_w_packed, d_w_packed_noise, d_scale,
                                                                        rows, cols, B_,
                                                                        out_scale,
                                                                        fused_noise,
                                                                        d_out,
                                                                        stream_);
                        } else {
                            bitnet_cuda::gemm_ternary_f_i2_noise(d_x, d_w_packed, d_w_packed_noise, d_scale,
                                                                 rows, cols, B_,
                                                                 out_scale,
                                                                 fused_noise,
                                                                 d_out,
                                                                 stream_);
                        }
                    } else {
                        if (use_packed_gemm_bitnet_) {
                            bitnet_cuda::gemm_ternary_f_i2_bitnet(d_x, d_w_packed, d_scale,
                                                                  rows, cols, B_,
                                                                  out_scale,
                                                                  d_out,
                                                                  stream_);
                        } else {
                            bitnet_cuda::gemm_ternary_f_i2(d_x, d_w_packed, d_scale,
                                                           rows, cols, B_,
                                                           out_scale,
                                                           d_out,
                                                           stream_);
                        }
                        if (noise_active) {
                            if (use_packed_gemm_bitnet_) {
                                bitnet_cuda::gemm_ternary_f_i2_bitnet(d_x, d_w_packed_noise, d_scale,
                                                                      rows, cols, B_,
                                                                      out_scale,
                                                                      d_tmp_noise_,
                                                                      stream_);
                            } else {
                                bitnet_cuda::gemm_ternary_f_i2(d_x, d_w_packed_noise, d_scale,
                                                               rows, cols, B_,
                                                               out_scale,
                                                               d_tmp_noise_,
                                                               stream_);
                            }
                            efla_lm_cuda::add_scaled(d_out, d_tmp_noise_, B_ * rows,
                                                     static_cast<float>(anti_sign) * noise_scale,
                                                     stream_);
                        }
                    }
                } else {
                    if (use_nvfp4 && d_w_nvfp4 && d_w_sfb) {
                        auto act = get_nvfp4_act(d_x, cols);
                        if (act.first && act.second) {
                            const float* d_c = d_out;
                            bitnet_cuda::gemm_ternary_f_nvfp4(act.first, d_w_nvfp4,
                                                              act.second, d_w_sfb,
                                                              B_, rows, cols,
                                                              out_scale, 0.0f,
                                                              d_c, d_out,
                                                              stream_);
                            if (noise_active && d_w_noise) {
                                if (d_w_noise_nvfp4 && d_w_noise_sfb) {
                                    const float alpha = out_scale * static_cast<float>(anti_sign) * noise_scale;
                                    bitnet_cuda::gemm_ternary_f_nvfp4(act.first, d_w_noise_nvfp4,
                                                                      act.second, d_w_noise_sfb,
                                                                      B_, rows, cols,
                                                                      alpha, 1.0f,
                                                                      d_out, d_out,
                                                                      stream_);
                                } else {
                                    bitnet_cuda::gemm_ternary_f(d_x, d_w_noise, d_scale,
                                                                rows, cols, B_,
                                                                out_scale,
                                                                d_tmp_noise_,
                                                                stream_);
                                    efla_lm_cuda::add_scaled(d_out, d_tmp_noise_, B_ * rows,
                                                             static_cast<float>(anti_sign) * noise_scale,
                                                             stream_);
                                }
                            }
                            return;
                        }
                    }
                    if (use_cutlass) {
                        bitnet_cuda::gemm_ternary_f_cutlass(d_x, d_w, rows, cols, B_, out_scale, d_out, stream_);
                        if (noise_active) {
                            bitnet_cuda::gemm_ternary_f_cutlass(d_x, d_w_noise, rows, cols, B_, out_scale, d_tmp_noise_, stream_);
                            efla_lm_cuda::add_scaled(d_out, d_tmp_noise_, B_ * rows,
                                                     static_cast<float>(anti_sign) * noise_scale,
                                                     stream_);
                        }
                    } else if (noise_active && use_fused_noise_gemm_) {
                        if (use_tiled_gemm_) {
                            bitnet_cuda::gemm_ternary_f_noise_tiled(d_x, d_w, d_w_noise, d_scale,
                                                                    rows, cols, B_,
                                                                    out_scale,
                                                                    static_cast<float>(anti_sign) * noise_scale,
                                                                    d_out,
                                                                    stream_);
                        } else {
                            bitnet_cuda::gemm_ternary_f_noise(d_x, d_w, d_w_noise, d_scale,
                                                              rows, cols, B_,
                                                              out_scale,
                                                              static_cast<float>(anti_sign) * noise_scale,
                                                              d_out,
                                                              stream_);
                        }
                    } else {
                        bitnet_cuda::gemm_ternary_f(d_x, d_w, d_scale,
                                                    rows, cols, B_,
                                                    out_scale,
                                                    d_out,
                                                    stream_);
                        if (noise_active) {
                            bitnet_cuda::gemm_ternary_f(d_x, d_w_noise, d_scale,
                                                        rows, cols, B_,
                                                        out_scale,
                                                        d_tmp_noise_,
                                                        stream_);
                            efla_lm_cuda::add_scaled(d_out, d_tmp_noise_, B_ * rows,
                                                     static_cast<float>(anti_sign) * noise_scale,
                                                     stream_);
                        }
                    }
                }
            };

            auto gemm_qkvb = [&](const int8_t* d_x,
                                 const int8_t* wq,
                                 const int8_t* wk,
                                 const int8_t* wv,
                                 const int8_t* wb,
                                 const int8_t* wq_n,
                                 const int8_t* wk_n,
                                 const int8_t* wv_n,
                                 const int8_t* wb_n,
                                 const uint32_t* wq_p,
                                 const uint32_t* wk_p,
                                 const uint32_t* wv_p,
                                 const uint32_t* wb_p,
                                 const uint32_t* wq_n_p,
                                 const uint32_t* wk_n_p,
                                 const uint32_t* wv_n_p,
                                 const uint32_t* wb_n_p,
                                 const void* wq_nvfp4,
                                 const void* wk_nvfp4,
                                 const void* wv_nvfp4,
                                 const void* wb_nvfp4,
                                 const void* wq_sfb,
                                 const void* wk_sfb,
                                 const void* wv_sfb,
                                 const void* wb_sfb,
                                 const void* wq_noise_nvfp4,
                                 const void* wk_noise_nvfp4,
                                 const void* wv_noise_nvfp4,
                                 const void* wb_noise_nvfp4,
                                 const void* wq_noise_sfb,
                                 const void* wk_noise_sfb,
                                 const void* wv_noise_sfb,
                                 const void* wb_noise_sfb,
                                 const void* wqkv_nvfp4,
                                 const void* wqkv_sfb,
                                 const void* wqkv_noise_nvfp4,
                                 const void* wqkv_noise_sfb,
                                 const float* d_scale,
                                 int rows,
                                 int cols,
                                 float q_scale,
                                 float k_scale,
                                 float v_scale,
                                 float b_scale,
                                 float beta_bias_eff,
                                 bool fuse_beta_bias,
                                 bool skip_split,
                                 float* d_q,
                                 float* d_k,
                                 float* d_v,
                                 float* d_beta) {
                if (use_fused_qkv_ && !use_packed_gemm_ &&
                    gemm_backend_ == TrainConfig::GemmBackend::Dp4a) {
                    const float fused_noise = noise_active ? static_cast<float>(anti_sign) * noise_scale : 0.0f;
                    bitnet_cuda::gemm_qkvb_fused(d_x,
                                                 wq, wk, wv, wb,
                                                 (noise_active ? wq_n : nullptr),
                                                 (noise_active ? wk_n : nullptr),
                                                 (noise_active ? wv_n : nullptr),
                                                 (noise_active ? wb_n : nullptr),
                                                 d_scale,
                                                 rows, cols, B_,
                                                 q_scale, k_scale, v_scale, b_scale,
                                                 fused_noise,
                                                 d_q, d_k, d_v, d_beta,
                                                 stream_);
                    return;
                }

                if (use_nvfp4 && !use_packed_gemm_) {
                    auto act = get_nvfp4_act(d_x, cols);
                    if (act.first && act.second &&
                        ((wqkv_nvfp4 && wqkv_sfb) ||
                         (wq_nvfp4 && wk_nvfp4 && wv_nvfp4 && wb_nvfp4 &&
                          wq_sfb && wk_sfb && wv_sfb && wb_sfb))) {
                        const int fused_rows_valid = rows * 3 + 1;
                        const int fused_rows = use_fused_qkv_nvfp4_ ? qkv_fused_cols_ : fused_rows_valid;
                        const bool nvfp4_beta_ok = ((1 & 31) == 0);
                        bool fused_noise = false;
                        const bool use_fused_qkv = use_fused_qkv_nvfp4_ && fused_rows > 0 &&
                                                   wqkv_nvfp4 && wqkv_sfb && d_qkv_f_;
                        if (use_fused_qkv) {
                            bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wqkv_nvfp4,
                                                              act.second, wqkv_sfb,
                                                              B_, fused_rows, cols,
                                                              1.0f, 0.0f,
                                                              d_qkv_f_, d_qkv_f_,
                                                              stream_);
                            if (noise_active && wqkv_noise_nvfp4 && wqkv_noise_sfb) {
                                const float alpha = static_cast<float>(anti_sign) * noise_scale;
                                bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wqkv_noise_nvfp4,
                                                                  act.second, wqkv_noise_sfb,
                                                                  B_, fused_rows, cols,
                                                                  alpha, 1.0f,
                                                                  d_qkv_f_, d_qkv_f_,
                                                                  stream_);
                                fused_noise = true;
                            }
                            if (!skip_split) {
                                if (use_qkv_split_kernel_) {
                                    const float bias = fuse_beta_bias ? beta_bias_eff : 0.0f;
                                    bitnet_cuda::split_qkvb_fused(d_qkv_f_, B_, rows, fused_rows, d_q, d_k, d_v, d_beta,
                                                                  bias, use_qkv_split_vec4_, stream_);
                                } else {
                                    const size_t src_pitch = static_cast<size_t>(fused_rows) * sizeof(float);
                                    const size_t dst_pitch = static_cast<size_t>(rows) * sizeof(float);
                                    cuda_check(cudaMemcpy2DAsync(d_q, dst_pitch,
                                                                 d_qkv_f_, src_pitch,
                                                                 static_cast<size_t>(rows) * sizeof(float),
                                                                 B_,
                                                                 cudaMemcpyDeviceToDevice,
                                                                 stream_), "cudaMemcpy2DAsync q");
                                    cuda_check(cudaMemcpy2DAsync(d_k, dst_pitch,
                                                                 d_qkv_f_ + rows, src_pitch,
                                                                 static_cast<size_t>(rows) * sizeof(float),
                                                                 B_,
                                                                 cudaMemcpyDeviceToDevice,
                                                                 stream_), "cudaMemcpy2DAsync k");
                                    cuda_check(cudaMemcpy2DAsync(d_v, dst_pitch,
                                                                 d_qkv_f_ + rows * 2, src_pitch,
                                                                 static_cast<size_t>(rows) * sizeof(float),
                                                                 B_,
                                                                 cudaMemcpyDeviceToDevice,
                                                                 stream_), "cudaMemcpy2DAsync v");
                                    cuda_check(cudaMemcpy2DAsync(d_beta, sizeof(float),
                                                                 d_qkv_f_ + rows * 3, src_pitch,
                                                                 sizeof(float),
                                                                 B_,
                                                                 cudaMemcpyDeviceToDevice,
                                                                 stream_), "cudaMemcpy2DAsync beta");
                                }
                            }
                        } else {
                            const float* d_c = d_q;
                            bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wq_nvfp4,
                                                              act.second, wq_sfb,
                                                              B_, rows, cols,
                                                              q_scale, 0.0f,
                                                              d_c, d_q,
                                                              stream_);
                            bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wk_nvfp4,
                                                              act.second, wk_sfb,
                                                              B_, rows, cols,
                                                              k_scale, 0.0f,
                                                              d_k, d_k,
                                                              stream_);
                            bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wv_nvfp4,
                                                              act.second, wv_sfb,
                                                              B_, rows, cols,
                                                              v_scale, 0.0f,
                                                              d_v, d_v,
                                                              stream_);
                            if (nvfp4_beta_ok) {
                                bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wb_nvfp4,
                                                                  act.second, wb_sfb,
                                                                  B_, 1, cols,
                                                                  b_scale, 0.0f,
                                                                  d_beta, d_beta,
                                                                  stream_);
                            } else {
                                bitnet_cuda::gemm_ternary_f(d_x, wb, d_scale,
                                                            1, cols, B_,
                                                            b_scale,
                                                            d_beta,
                                                            stream_);
                            }
                        }
                        if (noise_active && !fused_noise) {
                            if (wq_n) {
                                if (wq_noise_nvfp4 && wq_noise_sfb) {
                                    const float alpha = q_scale * static_cast<float>(anti_sign) * noise_scale;
                                    bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wq_noise_nvfp4,
                                                                      act.second, wq_noise_sfb,
                                                                      B_, rows, cols,
                                                                      alpha, 1.0f,
                                                                      d_q, d_q,
                                                                      stream_);
                                } else {
                                    bitnet_cuda::gemm_ternary_f(d_x, wq_n, d_scale,
                                                                rows, cols, B_,
                                                                q_scale,
                                                                d_tmp_noise_,
                                                                stream_);
                                    efla_lm_cuda::add_scaled(d_q, d_tmp_noise_, B_ * rows,
                                                             static_cast<float>(anti_sign) * noise_scale,
                                                             stream_);
                                }
                            }
                            if (wk_n) {
                                if (wk_noise_nvfp4 && wk_noise_sfb) {
                                    const float alpha = k_scale * static_cast<float>(anti_sign) * noise_scale;
                                    bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wk_noise_nvfp4,
                                                                      act.second, wk_noise_sfb,
                                                                      B_, rows, cols,
                                                                      alpha, 1.0f,
                                                                      d_k, d_k,
                                                                      stream_);
                                } else {
                                    bitnet_cuda::gemm_ternary_f(d_x, wk_n, d_scale,
                                                                rows, cols, B_,
                                                                k_scale,
                                                                d_tmp_noise_,
                                                                stream_);
                                    efla_lm_cuda::add_scaled(d_k, d_tmp_noise_, B_ * rows,
                                                             static_cast<float>(anti_sign) * noise_scale,
                                                             stream_);
                                }
                            }
                            if (wv_n) {
                                if (wv_noise_nvfp4 && wv_noise_sfb) {
                                    const float alpha = v_scale * static_cast<float>(anti_sign) * noise_scale;
                                    bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wv_noise_nvfp4,
                                                                      act.second, wv_noise_sfb,
                                                                      B_, rows, cols,
                                                                      alpha, 1.0f,
                                                                      d_v, d_v,
                                                                      stream_);
                                } else {
                                    bitnet_cuda::gemm_ternary_f(d_x, wv_n, d_scale,
                                                                rows, cols, B_,
                                                                v_scale,
                                                                d_tmp_noise_,
                                                                stream_);
                                    efla_lm_cuda::add_scaled(d_v, d_tmp_noise_, B_ * rows,
                                                             static_cast<float>(anti_sign) * noise_scale,
                                                             stream_);
                                }
                            }
                            if (wb_n) {
                                if (nvfp4_beta_ok && wb_noise_nvfp4 && wb_noise_sfb) {
                                    const float alpha = b_scale * static_cast<float>(anti_sign) * noise_scale;
                                    bitnet_cuda::gemm_ternary_f_nvfp4(act.first, wb_noise_nvfp4,
                                                                      act.second, wb_noise_sfb,
                                                                      B_, 1, cols,
                                                                      alpha, 1.0f,
                                                                      d_beta, d_beta,
                                                                      stream_);
                                } else {
                                    bitnet_cuda::gemm_ternary_f(d_x, wb_n, d_scale,
                                                                1, cols, B_,
                                                                b_scale,
                                                                d_tmp_noise_,
                                                                stream_);
                                    efla_lm_cuda::add_scaled(d_beta, d_tmp_noise_, B_,
                                                             static_cast<float>(anti_sign) * noise_scale,
                                                             stream_);
                                }
                            }
                        }
                        return;
                    }
                }

                gemm_noisy(d_x, wq, wq_n, wq_p, wq_n_p, d_scale, wq_nvfp4, wq_sfb,
                           wq_noise_nvfp4, wq_noise_sfb,
                           rows, cols, q_scale, d_q);
                gemm_noisy(d_x, wk, wk_n, wk_p, wk_n_p, d_scale, wk_nvfp4, wk_sfb,
                           wk_noise_nvfp4, wk_noise_sfb,
                           rows, cols, k_scale, d_k);
                gemm_noisy(d_x, wv, wv_n, wv_p, wv_n_p, d_scale, wv_nvfp4, wv_sfb,
                           wv_noise_nvfp4, wv_noise_sfb,
                           rows, cols, v_scale, d_v);
                gemm_noisy(d_x, wb, wb_n, wb_p, wb_n_p, d_scale, wb_nvfp4, wb_sfb,
                           wb_noise_nvfp4, wb_noise_sfb,
                           1, cols, b_scale, d_beta);
            };

            auto gemm_ff1_gelu = [&](const int8_t* d_x,
                                     const int8_t* d_w,
                                     const int8_t* d_w_noise,
                                     const uint32_t* d_w_packed,
                                     const uint32_t* d_w_packed_noise,
                                     const float* d_scale,
                                     const void* d_w_nvfp4,
                                     const void* d_w_sfb,
                                     const void* d_w_noise_nvfp4,
                                     const void* d_w_noise_sfb,
                                     int rows,
                                     int cols,
                                     float out_scale,
                                     float act_scale,
                                     float* d_out_f,
                                     int8_t* d_out_q) {
                if (use_fused_ff1_ && !use_packed_gemm_ &&
                    gemm_backend_ == TrainConfig::GemmBackend::Dp4a) {
                    const float fused_noise = noise_active ? static_cast<float>(anti_sign) * noise_scale : 0.0f;
                    bitnet_cuda::gemm_ternary_gelu_quant(d_x,
                                                         d_w,
                                                         (noise_active ? d_w_noise : nullptr),
                                                         d_scale,
                                                         rows, cols, B_,
                                                         out_scale,
                                                         fused_noise,
                                                         act_scale,
                                                         d_out_q,
                                                         stream_);
                    return;
                }
                if (use_nvfp4 && !use_packed_gemm_ && noise_active &&
                    d_w_nvfp4 && d_w_sfb &&
                    d_w_noise_nvfp4 && d_w_noise_sfb) {
                    auto act = get_nvfp4_act(d_x, cols);
                    if (act.first && act.second) {
                        const float* d_c = d_out_f;
                        bitnet_cuda::gemm_ternary_f_nvfp4(act.first, d_w_nvfp4,
                                                          act.second, d_w_sfb,
                                                          B_, rows, cols,
                                                          out_scale, 0.0f,
                                                          d_c, d_out_f,
                                                          stream_);
                        const float alpha = out_scale * static_cast<float>(anti_sign) * noise_scale;
                        bitnet_cuda::gemm_ternary_f_nvfp4_gelu_i8(act.first, d_w_noise_nvfp4,
                                                                  act.second, d_w_noise_sfb,
                                                                  B_, rows, cols,
                                                                  alpha, 1.0f,
                                                                  d_out_f,
                                                                  act_scale,
                                                                  d_out_q,
                                                                  stream_);
                        bitnet_cuda::nvfp4_quantize_a(d_out_q, B_, 1, rows, rows,
                                                      d_a_nvfp4_f_, d_sfa_f_, stream_);
                        return;
                    }
                }
                gemm_noisy(d_x, d_w, d_w_noise, d_w_packed, d_w_packed_noise, d_scale,
                           d_w_nvfp4, d_w_sfb,
                           d_w_noise_nvfp4, d_w_noise_sfb,
                           rows, cols, out_scale, d_out_f);
                if (use_nvfp4 && !use_packed_gemm_) {
                    bitnet_cuda::gelu_quantize_nvfp4(d_out_f, B_, rows, rows, act_scale,
                                                     d_out_q, d_a_nvfp4_f_, d_sfa_f_, stream_);
                    return;
                }
                efla_lm_cuda::gelu_quantize(d_out_f, B_ * rows, act_scale, d_out_q, stream_);
            };

            for (int t = 0; t < T_; ++t) {
                NvtxRange range_token(use_nvtx_ && !use_cuda_graph_, "token");
                const uint8_t* d_tokens_t = d_tokens_active_ + t;

                if (use_nvfp4) {
                    bitnet_cuda::embedding_lookup_noise_quantize_nvfp4(d_tokens_t, T_,
                                                                       d_emb_w_, (noise_active ? d_emb_noise_ : nullptr),
                                                                       H_, B_,
                                                                       noise_scale, anti_sign,
                                                                       d_emb_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                    bump_version(d_emb_q_);
                    set_nvfp4_act(d_emb_q_, H_);
                } else {
                    // emb_f: float, including optional full noise.
                    efla_lm_cuda::embedding_lookup_noise(d_tokens_t, T_,
                                                         d_emb_w_, (noise_active ? d_emb_noise_ : nullptr),
                                                         H_, B_,
                                                         noise_scale, anti_sign,
                                                         d_emb_f_, stream_);

                    // emb_q: quantized int8 embedding.
                    bitnet_cuda::activation_quantize(d_emb_f_, d_emb_q_, B_ * H_, /*activation*/0, stream_);
                    bump_version(d_emb_q_);
                }

                // input projection (float).
                gemm_noisy(d_emb_q_, d_win_, d_win_noise_,
                           use_packed_gemm_ ? d_win_packed_ : nullptr,
                           use_packed_gemm_ ? d_win_noise_packed_ : nullptr,
                           d_scale_x_,
                           d_win_nvfp4_, d_win_sfb_,
                           d_win_noise_nvfp4_, d_win_noise_sfb_,
                           H_, H_,
                           win_out_scale_,
                           d_inproj_f_);

                const float* d_pos_t = d_pos_ + static_cast<size_t>(t) * H_;
                if (use_transformer_residual) {
                    if (int8_residual) {
                        if (use_nvfp4) {
                            bitnet_cuda::add_pos_gelu_quantize_nvfp4(d_inproj_f_, d_pos_t, H_, B_, act_scale,
                                                                     d_x_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                        } else {
                            efla_lm_cuda::add_pos_gelu_quantize(d_inproj_f_, d_pos_t, H_, B_, act_scale, d_x_q_, stream_);
                        }
                        bump_version(d_x_q_);
                        if (use_nvfp4) {
                            set_nvfp4_act(d_x_q_, H_);
                        }
                    } else {
                        efla_lm_cuda::add_pos_gelu(d_inproj_f_, d_pos_t, H_, B_, act_scale, d_res_f_, stream_);
                    }
                } else {
                    if (use_nvfp4) {
                        bitnet_cuda::add_pos_gelu_quantize_nvfp4(d_inproj_f_, d_pos_t, H_, B_, act_scale,
                                                                 d_x_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                    } else {
                        efla_lm_cuda::add_pos_gelu_quantize(d_inproj_f_, d_pos_t, H_, B_, act_scale, d_x_q_, stream_);
                    }
                    bump_version(d_x_q_);
                    if (use_nvfp4) {
                        set_nvfp4_act(d_x_q_, H_);
                    }
                }

                const size_t hh = static_cast<size_t>(H_) * H_;
                const size_t hf = static_cast<size_t>(H_) * static_cast<size_t>(F_);
                const size_t dh = (D_ > 0) ? (static_cast<size_t>(D_) * static_cast<size_t>(H_)) : 0u;
                const int8_t* x_final = nullptr;
                if (use_transformer_residual) {
                    if (int8_residual) {
                        NvtxRange range_attn(use_nvtx_ && !use_cuda_graph_, "attn");
                        bool have_normed_x = false;
                        for (int l = 0; l < L_; ++l) {
                            if (use_nvfp4) {
                                if (!have_normed_x) {
                                    bitnet_cuda::absmean_norm_q_nvfp4(d_x_q_, H_, B_, H_, ln_scale,
                                                                      d_y_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                                    bump_version(d_y_q_);
                                    set_nvfp4_act(d_y_q_, H_);
                                }
                            } else {
                                efla_lm_cuda::absmean_norm_q(d_x_q_, H_, B_, ln_scale, d_y_q_, stream_);
                                bump_version(d_y_q_);
                            }

                            const size_t l_off_hh = static_cast<size_t>(l) * hh;
                            const size_t l_off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
                            const size_t l_off_hf = static_cast<size_t>(l) * hf;
                            const size_t l_off_hh_p = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(pack_cols_h_);
                            const size_t l_off_h_p = static_cast<size_t>(l) * static_cast<size_t>(pack_cols_h_);
                            const size_t l_off_hf1_p = static_cast<size_t>(l) * static_cast<size_t>(F_) * static_cast<size_t>(pack_cols_h_);
                            const size_t l_off_hf2_p = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(pack_cols_f_);

                            const int8_t* wq = d_wq_ + l_off_hh;
                            const int8_t* wk = d_wk_ + l_off_hh;
                            const int8_t* wv = d_wv_ + l_off_hh;
                            const int8_t* wb = d_wbeta_ + l_off_h;
                            const int8_t* wf1 = d_ff1_ + l_off_hf;
                            const int8_t* wf2 = d_ff2_ + l_off_hf;
                            const void* wq_nvfp4 = (use_nvfp4_ ? d_wq_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_nvfp4 = (use_nvfp4_ ? d_wk_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_nvfp4 = (use_nvfp4_ ? d_wv_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_nvfp4 = (use_nvfp4_ ? d_wbeta_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_nvfp4 = (use_nvfp4_ ? d_ff1_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_nvfp4 = (use_nvfp4_ ? d_ff2_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wq_noise_nvfp4 = (use_nvfp4_ ? d_wq_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_noise_nvfp4 = (use_nvfp4_ ? d_wk_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_noise_nvfp4 = (use_nvfp4_ ? d_wv_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_noise_nvfp4 = (use_nvfp4_ ? d_wbeta_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_noise_nvfp4 = (use_nvfp4_ ? d_ff1_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_noise_nvfp4 = (use_nvfp4_ ? d_ff2_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wq_sfb = (use_nvfp4_ ? d_wq_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_sfb = (use_nvfp4_ ? d_wk_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_sfb = (use_nvfp4_ ? d_wv_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_sfb = (use_nvfp4_ ? d_wbeta_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_sfb = (use_nvfp4_ ? d_ff1_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_sfb = (use_nvfp4_ ? d_ff2_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wq_noise_sfb = (use_nvfp4_ ? d_wq_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_noise_sfb = (use_nvfp4_ ? d_wk_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_noise_sfb = (use_nvfp4_ ? d_wv_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_noise_sfb = (use_nvfp4_ ? d_wbeta_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_noise_sfb = (use_nvfp4_ ? d_ff1_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_noise_sfb = (use_nvfp4_ ? d_ff2_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_nvfp4 = (use_fused_qkv_nvfp4_ ? d_wqkv_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_sfb = (use_fused_qkv_nvfp4_ ? d_wqkv_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_noise_nvfp4 =
                                (use_fused_qkv_nvfp4_ ? d_wqkv_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_noise_sfb =
                                (use_fused_qkv_nvfp4_ ? d_wqkv_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const uint32_t* wq_p = use_packed_gemm_ ? (d_wq_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wk_p = use_packed_gemm_ ? (d_wk_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wv_p = use_packed_gemm_ ? (d_wv_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wb_p = use_packed_gemm_ ? (d_wbeta_packed_ + l_off_h_p) : nullptr;
                            const uint32_t* wf1_p = use_packed_gemm_ ? (d_ff1_packed_ + l_off_hf1_p) : nullptr;
                            const uint32_t* wf2_p = use_packed_gemm_ ? (d_ff2_packed_ + l_off_hf2_p) : nullptr;

                            const int8_t* wq_n = d_wq_noise_ + l_off_hh;
                            const int8_t* wk_n = d_wk_noise_ + l_off_hh;
                            const int8_t* wv_n = d_wv_noise_ + l_off_hh;
                            const int8_t* wb_n = d_wbeta_noise_ + l_off_h;
                            const int8_t* wf1_n = d_ff1_noise_ + l_off_hf;
                            const int8_t* wf2_n = d_ff2_noise_ + l_off_hf;
                            const uint32_t* wq_n_p = use_packed_gemm_ ? (d_wq_noise_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wk_n_p = use_packed_gemm_ ? (d_wk_noise_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wv_n_p = use_packed_gemm_ ? (d_wv_noise_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wb_n_p = use_packed_gemm_ ? (d_wbeta_noise_packed_ + l_off_h_p) : nullptr;
                            const uint32_t* wf1_n_p = use_packed_gemm_ ? (d_ff1_noise_packed_ + l_off_hf1_p) : nullptr;
                            const uint32_t* wf2_n_p = use_packed_gemm_ ? (d_ff2_noise_packed_ + l_off_hf2_p) : nullptr;

                            const float bb_base = beta_bias ? beta_bias[l] : 0.0f;
                            const float bb_noise = (beta_bias_noise && noise_active)
                                                        ? beta_bias_noise[l]
                                                        : 0.0f;
                            const float beta_bias_eff =
                                bb_base + static_cast<float>(anti_sign) * noise_scale * bb_noise;
                            const bool use_qkv_fused_layout =
                                use_nvfp4 && use_fused_qkv_nvfp4_ && wqkv_nvfp4 && wqkv_sfb && d_qkv_f_ &&
                                (D_ == 0) && !use_fused_efla_ && !use_efla_mixed_ &&
                                !use_efla_fuse_diff_ && !use_efla_update_wmma_;
                            const bool fuse_beta_bias =
                                use_qkv_split_bias_ && use_nvfp4 && use_fused_qkv_nvfp4_ && use_qkv_split_kernel_ &&
                                wqkv_nvfp4 && wqkv_sfb && d_qkv_f_ && !use_qkv_fused_layout;
                            const float beta_bias_for_step = fuse_beta_bias ? 0.0f : beta_bias_eff;

                            gemm_qkvb(d_y_q_,
                                      wq, wk, wv, wb,
                                      wq_n, wk_n, wv_n, wb_n,
                                      wq_p, wk_p, wv_p, wb_p,
                                      wq_n_p, wk_n_p, wv_n_p, wb_n_p,
                                      wq_nvfp4, wk_nvfp4, wv_nvfp4, wb_nvfp4,
                                      wq_sfb, wk_sfb, wv_sfb, wb_sfb,
                                      wq_noise_nvfp4, wk_noise_nvfp4, wv_noise_nvfp4, wb_noise_nvfp4,
                                      wq_noise_sfb, wk_noise_sfb, wv_noise_sfb, wb_noise_sfb,
                                      wqkv_nvfp4, wqkv_sfb,
                                      wqkv_noise_nvfp4, wqkv_noise_sfb,
                                      d_scale_x_, H_, H_,
                                      q_out_scale_, k_out_scale_, v_out_scale_, beta_out_scale_,
                                      beta_bias_eff, fuse_beta_bias, use_qkv_fused_layout,
                                      d_q_, d_k_, d_v_, d_beta_raw_);
                            if (D_ > 0) {
                                float* d_K_l = d_K_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * dh;
                                float* d_V_l = d_V_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * dh;
                                efla_lm_cuda::efla_step_lowrank(d_K_l, d_V_l, d_q_, d_k_, d_v_, d_beta_raw_,
                                                                beta_bias_for_step, gate_scale, state_decay,
                                                                H_, D_, B_, eps,
                                                                d_y_, stream_);
                            } else {
                                float* d_S_f = use_fp16_state_
                                                   ? nullptr
                                                   : (d_S_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * hh);
                                __half* d_S_h = use_fp16_state_
                                                    ? (d_S_h_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * hh)
                                                    : nullptr;
                                efla_full_state(d_S_f, d_S_h, beta_bias_for_step, d_y_,
                                                d_qkv_f_, qkv_fused_cols_, use_qkv_fused_layout);
                            }
                            NvtxRange range_mlp(use_nvtx_ && !use_cuda_graph_, "mlp");
                            if (use_nvfp4) {
                                bitnet_cuda::add_scaled_to_int8_absmean_norm_q_nvfp4(
                                    d_x_q_, d_y_, H_, B_, H_, residual_scale, ln_scale,
                                    d_y_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                                bump_version(d_x_q_);
                                bump_version(d_y_q_);
                                set_nvfp4_act(d_y_q_, H_);
                            } else {
                                efla_lm_cuda::add_scaled_to_int8_absmean_norm_q(
                                    d_x_q_, d_y_, H_, B_, residual_scale, ln_scale, d_y_q_, stream_);
                                bump_version(d_x_q_);
                                bump_version(d_y_q_);
                            }
                            gemm_ff1_gelu(d_y_q_, wf1, wf1_n, wf1_p, wf1_n_p, d_scale_x_,
                                          wf1_nvfp4, wf1_sfb,
                                          wf1_noise_nvfp4, wf1_noise_sfb,
                                          F_, H_, ff1_out_scale_, mlp_act_scale,
                                          d_ff1_f_, d_ff1_q_);
                            bump_version(d_ff1_q_);
                            if (use_nvfp4) {
                                set_nvfp4_act(d_ff1_q_, F_);
                            }
                            gemm_noisy(d_ff1_q_, wf2, wf2_n, wf2_p, wf2_n_p, d_scale_f_,
                                       wf2_nvfp4, wf2_sfb,
                                       wf2_noise_nvfp4, wf2_noise_sfb,
                                       H_, F_, ff2_out_scale_, d_y_);
                            if (use_nvfp4) {
                                bitnet_cuda::add_scaled_to_int8_absmean_norm_q_nvfp4(
                                    d_x_q_, d_y_, H_, B_, H_, mlp_residual_scale, ln_scale,
                                    d_y_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                                bump_version(d_x_q_);
                                bump_version(d_y_q_);
                                set_nvfp4_act(d_y_q_, H_);
                                have_normed_x = true;
                            } else {
                                efla_lm_cuda::add_scaled_to_int8(d_x_q_, d_y_, B_ * H_, mlp_residual_scale, stream_);
                                bump_version(d_x_q_);
                            }
                        }

                        if (use_nvfp4) {
                            if (!have_normed_x) {
                                bitnet_cuda::absmean_norm_q_nvfp4(d_x_q_, H_, B_, H_, ln_scale,
                                                                  d_y_q_, d_a_nvfp4_h_, d_sfa_h_, stream_);
                                bump_version(d_y_q_);
                                set_nvfp4_act(d_y_q_, H_);
                            }
                        } else {
                            efla_lm_cuda::absmean_norm_q(d_x_q_, H_, B_, ln_scale, d_y_q_, stream_);
                            bump_version(d_y_q_);
                        }
                        x_final = d_y_q_;
                    } else {
                        NvtxRange range_attn(use_nvtx_ && !use_cuda_graph_, "attn");
                        for (int l = 0; l < L_; ++l) {
                            efla_lm_cuda::layernorm_quantize(d_res_f_, H_, B_, ln_scale, eps, d_x_q_, stream_);
                            bump_version(d_x_q_);

                            const size_t l_off_hh = static_cast<size_t>(l) * hh;
                            const size_t l_off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
                            const size_t l_off_hf = static_cast<size_t>(l) * hf;
                            const size_t l_off_hh_p = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(pack_cols_h_);
                            const size_t l_off_h_p = static_cast<size_t>(l) * static_cast<size_t>(pack_cols_h_);
                            const size_t l_off_hf1_p = static_cast<size_t>(l) * static_cast<size_t>(F_) * static_cast<size_t>(pack_cols_h_);
                            const size_t l_off_hf2_p = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(pack_cols_f_);

                            const int8_t* wq = d_wq_ + l_off_hh;
                            const int8_t* wk = d_wk_ + l_off_hh;
                            const int8_t* wv = d_wv_ + l_off_hh;
                            const int8_t* wb = d_wbeta_ + l_off_h;
                            const int8_t* wf1 = d_ff1_ + l_off_hf;
                            const int8_t* wf2 = d_ff2_ + l_off_hf;
                            const void* wq_nvfp4 = (use_nvfp4_ ? d_wq_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_nvfp4 = (use_nvfp4_ ? d_wk_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_nvfp4 = (use_nvfp4_ ? d_wv_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_nvfp4 = (use_nvfp4_ ? d_wbeta_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_nvfp4 = (use_nvfp4_ ? d_ff1_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_nvfp4 = (use_nvfp4_ ? d_ff2_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wq_noise_nvfp4 = (use_nvfp4_ ? d_wq_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_noise_nvfp4 = (use_nvfp4_ ? d_wk_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_noise_nvfp4 = (use_nvfp4_ ? d_wv_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_noise_nvfp4 = (use_nvfp4_ ? d_wbeta_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_noise_nvfp4 = (use_nvfp4_ ? d_ff1_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_noise_nvfp4 = (use_nvfp4_ ? d_ff2_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wq_sfb = (use_nvfp4_ ? d_wq_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_sfb = (use_nvfp4_ ? d_wk_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_sfb = (use_nvfp4_ ? d_wv_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_sfb = (use_nvfp4_ ? d_wbeta_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_sfb = (use_nvfp4_ ? d_ff1_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_sfb = (use_nvfp4_ ? d_ff2_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wq_noise_sfb = (use_nvfp4_ ? d_wq_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wk_noise_sfb = (use_nvfp4_ ? d_wk_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wv_noise_sfb = (use_nvfp4_ ? d_wv_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wb_noise_sfb = (use_nvfp4_ ? d_wbeta_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf1_noise_sfb = (use_nvfp4_ ? d_ff1_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wf2_noise_sfb = (use_nvfp4_ ? d_ff2_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_nvfp4 = (use_fused_qkv_nvfp4_ ? d_wqkv_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_sfb = (use_fused_qkv_nvfp4_ ? d_wqkv_sfb_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_noise_nvfp4 =
                                (use_fused_qkv_nvfp4_ ? d_wqkv_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                            const void* wqkv_noise_sfb =
                                (use_fused_qkv_nvfp4_ ? d_wqkv_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                            const uint32_t* wq_p = use_packed_gemm_ ? (d_wq_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wk_p = use_packed_gemm_ ? (d_wk_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wv_p = use_packed_gemm_ ? (d_wv_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wb_p = use_packed_gemm_ ? (d_wbeta_packed_ + l_off_h_p) : nullptr;
                            const uint32_t* wf1_p = use_packed_gemm_ ? (d_ff1_packed_ + l_off_hf1_p) : nullptr;
                            const uint32_t* wf2_p = use_packed_gemm_ ? (d_ff2_packed_ + l_off_hf2_p) : nullptr;

                            const int8_t* wq_n = d_wq_noise_ + l_off_hh;
                            const int8_t* wk_n = d_wk_noise_ + l_off_hh;
                            const int8_t* wv_n = d_wv_noise_ + l_off_hh;
                            const int8_t* wb_n = d_wbeta_noise_ + l_off_h;
                            const int8_t* wf1_n = d_ff1_noise_ + l_off_hf;
                            const int8_t* wf2_n = d_ff2_noise_ + l_off_hf;
                            const uint32_t* wq_n_p = use_packed_gemm_ ? (d_wq_noise_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wk_n_p = use_packed_gemm_ ? (d_wk_noise_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wv_n_p = use_packed_gemm_ ? (d_wv_noise_packed_ + l_off_hh_p) : nullptr;
                            const uint32_t* wb_n_p = use_packed_gemm_ ? (d_wbeta_noise_packed_ + l_off_h_p) : nullptr;
                            const uint32_t* wf1_n_p = use_packed_gemm_ ? (d_ff1_noise_packed_ + l_off_hf1_p) : nullptr;
                            const uint32_t* wf2_n_p = use_packed_gemm_ ? (d_ff2_noise_packed_ + l_off_hf2_p) : nullptr;

                            const float bb_base = beta_bias ? beta_bias[l] : 0.0f;
                            const float bb_noise = (beta_bias_noise && noise_active)
                                                        ? beta_bias_noise[l]
                                                        : 0.0f;
                            const float beta_bias_eff =
                                bb_base + static_cast<float>(anti_sign) * noise_scale * bb_noise;
                            const bool use_qkv_fused_layout =
                                use_nvfp4 && use_fused_qkv_nvfp4_ && wqkv_nvfp4 && wqkv_sfb && d_qkv_f_ &&
                                (D_ == 0) && !use_fused_efla_ && !use_efla_mixed_ &&
                                !use_efla_fuse_diff_ && !use_efla_update_wmma_;
                            const bool fuse_beta_bias =
                                use_qkv_split_bias_ && use_nvfp4 && use_fused_qkv_nvfp4_ && use_qkv_split_kernel_ &&
                                wqkv_nvfp4 && wqkv_sfb && d_qkv_f_ && !use_qkv_fused_layout;
                            const float beta_bias_for_step = fuse_beta_bias ? 0.0f : beta_bias_eff;

                            gemm_qkvb(d_x_q_,
                                      wq, wk, wv, wb,
                                      wq_n, wk_n, wv_n, wb_n,
                                      wq_p, wk_p, wv_p, wb_p,
                                      wq_n_p, wk_n_p, wv_n_p, wb_n_p,
                                      wq_nvfp4, wk_nvfp4, wv_nvfp4, wb_nvfp4,
                                      wq_sfb, wk_sfb, wv_sfb, wb_sfb,
                                      wq_noise_nvfp4, wk_noise_nvfp4, wv_noise_nvfp4, wb_noise_nvfp4,
                                      wq_noise_sfb, wk_noise_sfb, wv_noise_sfb, wb_noise_sfb,
                                      wqkv_nvfp4, wqkv_sfb,
                                      wqkv_noise_nvfp4, wqkv_noise_sfb,
                                      d_scale_x_, H_, H_,
                                      q_out_scale_, k_out_scale_, v_out_scale_, beta_out_scale_,
                                      beta_bias_eff, fuse_beta_bias, use_qkv_fused_layout,
                                      d_q_, d_k_, d_v_, d_beta_raw_);
                            if (D_ > 0) {
                                float* d_K_l = d_K_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * dh;
                                float* d_V_l = d_V_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * dh;
                                efla_lm_cuda::efla_step_lowrank(d_K_l, d_V_l, d_q_, d_k_, d_v_, d_beta_raw_,
                                                                beta_bias_for_step, gate_scale, state_decay,
                                                                H_, D_, B_, eps,
                                                                d_y_, stream_);
                            } else {
                                float* d_S_f = use_fp16_state_
                                                   ? nullptr
                                                   : (d_S_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * hh);
                                __half* d_S_h = use_fp16_state_
                                                    ? (d_S_h_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * hh)
                                                    : nullptr;
                                efla_full_state(d_S_f, d_S_h, beta_bias_for_step, d_y_,
                                                d_qkv_f_, qkv_fused_cols_, use_qkv_fused_layout);
                            }
                            efla_lm_cuda::add_scaled(d_res_f_, d_y_, B_ * H_, residual_scale, stream_);

                            NvtxRange range_mlp(use_nvtx_ && !use_cuda_graph_, "mlp");
                            efla_lm_cuda::layernorm_quantize(d_res_f_, H_, B_, ln_scale, eps, d_x_q_, stream_);
                            bump_version(d_x_q_);
                            gemm_ff1_gelu(d_x_q_, wf1, wf1_n, wf1_p, wf1_n_p, d_scale_x_,
                                          wf1_nvfp4, wf1_sfb,
                                          wf1_noise_nvfp4, wf1_noise_sfb,
                                          F_, H_, ff1_out_scale_, mlp_act_scale,
                                          d_ff1_f_, d_ff1_q_);
                            bump_version(d_ff1_q_);
                            if (use_nvfp4) {
                                set_nvfp4_act(d_ff1_q_, F_);
                            }
                            gemm_noisy(d_ff1_q_, wf2, wf2_n, wf2_p, wf2_n_p, d_scale_f_,
                                       wf2_nvfp4, wf2_sfb,
                                       wf2_noise_nvfp4, wf2_noise_sfb,
                                       H_, F_, ff2_out_scale_, d_y_);
                            efla_lm_cuda::add_scaled(d_res_f_, d_y_, B_ * H_, mlp_residual_scale, stream_);
                        }

                        efla_lm_cuda::layernorm_quantize(d_res_f_, H_, B_, ln_scale, eps, d_x_q_, stream_);
                        bump_version(d_x_q_);
                        x_final = d_x_q_;
                    }
                } else {
                    // Original post-norm style:
                    //   x_attn = LN(attn(x) + residual_scale*x)
                    //   x = LN(mlp(x_attn) + x_attn)
                    int8_t* x_in = d_x_q_;
                    int8_t* x_attn = d_y_q_;
                    for (int l = 0; l < L_; ++l) {
                        const size_t l_off_hh = static_cast<size_t>(l) * hh;
                        const size_t l_off_h = static_cast<size_t>(l) * static_cast<size_t>(H_);
                        const size_t l_off_hf = static_cast<size_t>(l) * hf;
                        const size_t l_off_hh_p = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(pack_cols_h_);
                        const size_t l_off_h_p = static_cast<size_t>(l) * static_cast<size_t>(pack_cols_h_);
                        const size_t l_off_hf1_p = static_cast<size_t>(l) * static_cast<size_t>(F_) * static_cast<size_t>(pack_cols_h_);
                        const size_t l_off_hf2_p = static_cast<size_t>(l) * static_cast<size_t>(H_) * static_cast<size_t>(pack_cols_f_);

                        const int8_t* wq = d_wq_ + l_off_hh;
                        const int8_t* wk = d_wk_ + l_off_hh;
                        const int8_t* wv = d_wv_ + l_off_hh;
                        const int8_t* wb = d_wbeta_ + l_off_h;
                        const int8_t* wf1 = d_ff1_ + l_off_hf;
                        const int8_t* wf2 = d_ff2_ + l_off_hf;
                        const void* wq_nvfp4 = (use_nvfp4_ ? d_wq_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wk_nvfp4 = (use_nvfp4_ ? d_wk_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wv_nvfp4 = (use_nvfp4_ ? d_wv_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wb_nvfp4 = (use_nvfp4_ ? d_wbeta_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wf1_nvfp4 = (use_nvfp4_ ? d_ff1_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wf2_nvfp4 = (use_nvfp4_ ? d_ff2_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wq_noise_nvfp4 = (use_nvfp4_ ? d_wq_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wk_noise_nvfp4 = (use_nvfp4_ ? d_wk_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wv_noise_nvfp4 = (use_nvfp4_ ? d_wv_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wb_noise_nvfp4 = (use_nvfp4_ ? d_wbeta_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wf1_noise_nvfp4 = (use_nvfp4_ ? d_ff1_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wf2_noise_nvfp4 = (use_nvfp4_ ? d_ff2_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wq_sfb = (use_nvfp4_ ? d_wq_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wk_sfb = (use_nvfp4_ ? d_wk_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wv_sfb = (use_nvfp4_ ? d_wv_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wb_sfb = (use_nvfp4_ ? d_wbeta_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wf1_sfb = (use_nvfp4_ ? d_ff1_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wf2_sfb = (use_nvfp4_ ? d_ff2_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wq_noise_sfb = (use_nvfp4_ ? d_wq_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wk_noise_sfb = (use_nvfp4_ ? d_wk_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wv_noise_sfb = (use_nvfp4_ ? d_wv_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wb_noise_sfb = (use_nvfp4_ ? d_wbeta_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wf1_noise_sfb = (use_nvfp4_ ? d_ff1_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wf2_noise_sfb = (use_nvfp4_ ? d_ff2_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wqkv_nvfp4 = (use_fused_qkv_nvfp4_ ? d_wqkv_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wqkv_sfb = (use_fused_qkv_nvfp4_ ? d_wqkv_sfb_[static_cast<size_t>(l)] : nullptr);
                        const void* wqkv_noise_nvfp4 =
                            (use_fused_qkv_nvfp4_ ? d_wqkv_noise_nvfp4_[static_cast<size_t>(l)] : nullptr);
                        const void* wqkv_noise_sfb =
                            (use_fused_qkv_nvfp4_ ? d_wqkv_noise_sfb_[static_cast<size_t>(l)] : nullptr);
                        const uint32_t* wq_p = use_packed_gemm_ ? (d_wq_packed_ + l_off_hh_p) : nullptr;
                        const uint32_t* wk_p = use_packed_gemm_ ? (d_wk_packed_ + l_off_hh_p) : nullptr;
                        const uint32_t* wv_p = use_packed_gemm_ ? (d_wv_packed_ + l_off_hh_p) : nullptr;
                        const uint32_t* wb_p = use_packed_gemm_ ? (d_wbeta_packed_ + l_off_h_p) : nullptr;
                        const uint32_t* wf1_p = use_packed_gemm_ ? (d_ff1_packed_ + l_off_hf1_p) : nullptr;
                        const uint32_t* wf2_p = use_packed_gemm_ ? (d_ff2_packed_ + l_off_hf2_p) : nullptr;

                        const int8_t* wq_n = d_wq_noise_ + l_off_hh;
                        const int8_t* wk_n = d_wk_noise_ + l_off_hh;
                        const int8_t* wv_n = d_wv_noise_ + l_off_hh;
                        const int8_t* wb_n = d_wbeta_noise_ + l_off_h;
                        const int8_t* wf1_n = d_ff1_noise_ + l_off_hf;
                        const int8_t* wf2_n = d_ff2_noise_ + l_off_hf;
                        const uint32_t* wq_n_p = use_packed_gemm_ ? (d_wq_noise_packed_ + l_off_hh_p) : nullptr;
                        const uint32_t* wk_n_p = use_packed_gemm_ ? (d_wk_noise_packed_ + l_off_hh_p) : nullptr;
                        const uint32_t* wv_n_p = use_packed_gemm_ ? (d_wv_noise_packed_ + l_off_hh_p) : nullptr;
                        const uint32_t* wb_n_p = use_packed_gemm_ ? (d_wbeta_noise_packed_ + l_off_h_p) : nullptr;
                        const uint32_t* wf1_n_p = use_packed_gemm_ ? (d_ff1_noise_packed_ + l_off_hf1_p) : nullptr;
                        const uint32_t* wf2_n_p = use_packed_gemm_ ? (d_ff2_noise_packed_ + l_off_hf2_p) : nullptr;

                        const float bb_base = beta_bias ? beta_bias[l] : 0.0f;
                        const float bb_noise = (beta_bias_noise && noise_active)
                                                    ? beta_bias_noise[l]
                                                    : 0.0f;
                        const float beta_bias_eff =
                            bb_base + static_cast<float>(anti_sign) * noise_scale * bb_noise;
                        const bool use_qkv_fused_layout =
                            use_nvfp4 && use_fused_qkv_nvfp4_ && wqkv_nvfp4 && wqkv_sfb && d_qkv_f_ &&
                            (D_ == 0) && !use_fused_efla_ && !use_efla_mixed_ &&
                            !use_efla_fuse_diff_ && !use_efla_update_wmma_;
                        const bool fuse_beta_bias =
                            use_qkv_split_bias_ && use_nvfp4 && use_fused_qkv_nvfp4_ && use_qkv_split_kernel_ &&
                            wqkv_nvfp4 && wqkv_sfb && d_qkv_f_ && !use_qkv_fused_layout;
                        const float beta_bias_for_step = fuse_beta_bias ? 0.0f : beta_bias_eff;

                        gemm_qkvb(x_in,
                                  wq, wk, wv, wb,
                                  wq_n, wk_n, wv_n, wb_n,
                                  wq_p, wk_p, wv_p, wb_p,
                                  wq_n_p, wk_n_p, wv_n_p, wb_n_p,
                                  wq_nvfp4, wk_nvfp4, wv_nvfp4, wb_nvfp4,
                                  wq_sfb, wk_sfb, wv_sfb, wb_sfb,
                                  wq_noise_nvfp4, wk_noise_nvfp4, wv_noise_nvfp4, wb_noise_nvfp4,
                                  wq_noise_sfb, wk_noise_sfb, wv_noise_sfb, wb_noise_sfb,
                                  wqkv_nvfp4, wqkv_sfb,
                                  wqkv_noise_nvfp4, wqkv_noise_sfb,
                                  d_scale_x_, H_, H_,
                                  q_out_scale_, k_out_scale_, v_out_scale_, beta_out_scale_,
                                  beta_bias_eff, fuse_beta_bias, use_qkv_fused_layout,
                                  d_q_, d_k_, d_v_, d_beta_raw_);
                        if (D_ > 0) {
                            float* d_K_l = d_K_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * dh;
                            float* d_V_l = d_V_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * dh;
                            efla_lm_cuda::efla_step_lowrank(d_K_l, d_V_l, d_q_, d_k_, d_v_, d_beta_raw_,
                                                            beta_bias_for_step, gate_scale, state_decay,
                                                            H_, D_, B_, eps,
                                                            d_y_, stream_);
                        } else {
                            float* d_S_f = use_fp16_state_
                                               ? nullptr
                                               : (d_S_f_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * hh);
                            __half* d_S_h = use_fp16_state_
                                                ? (d_S_h_ + static_cast<size_t>(l) * static_cast<size_t>(B_) * hh)
                                                : nullptr;
                            efla_full_state(d_S_f, d_S_h, beta_bias_for_step, d_y_,
                                            d_qkv_f_, qkv_fused_cols_, use_qkv_fused_layout);
                        }
                        if (int8_residual) {
                            if (use_nvfp4) {
                                bitnet_cuda::add_scaled_to_int8_absmean_norm_q_nvfp4(
                                    x_in, d_y_, H_, B_, H_, residual_scale, ln_scale,
                                    x_attn, d_a_nvfp4_h_, d_sfa_h_, stream_);
                                bump_version(x_in);
                                bump_version(x_attn);
                                set_nvfp4_act(x_attn, H_);
                            } else {
                                efla_lm_cuda::add_scaled_to_int8_absmean_norm_q(
                                    x_in, d_y_, H_, B_, residual_scale, ln_scale, x_attn, stream_);
                                bump_version(x_in);
                                bump_version(x_attn);
                            }
                        } else {
                            efla_lm_cuda::add_scaled_i8(d_y_, x_in, B_ * H_, residual_scale, stream_);
                            efla_lm_cuda::layernorm_quantize(d_y_, H_, B_, ln_scale, eps, x_attn, stream_);
                            bump_version(x_attn);
                        }

                        gemm_ff1_gelu(x_attn, wf1, wf1_n, wf1_p, wf1_n_p, d_scale_x_,
                                      wf1_nvfp4, wf1_sfb,
                                      wf1_noise_nvfp4, wf1_noise_sfb,
                                      F_, H_, ff1_out_scale_, mlp_act_scale,
                                      d_ff1_f_, d_ff1_q_);
                        bump_version(d_ff1_q_);
                        if (use_nvfp4) {
                            set_nvfp4_act(d_ff1_q_, F_);
                        }
                        gemm_noisy(d_ff1_q_, wf2, wf2_n, wf2_p, wf2_n_p, d_scale_f_,
                                   wf2_nvfp4, wf2_sfb,
                                   wf2_noise_nvfp4, wf2_noise_sfb,
                                   H_, F_, ff2_out_scale_, d_y_);
                        if (int8_residual) {
                            if (use_nvfp4) {
                                bitnet_cuda::add_scaled_to_int8_absmean_norm_q_nvfp4(
                                    x_attn, d_y_, H_, B_, H_, mlp_residual_scale, ln_scale,
                                    x_in, d_a_nvfp4_h_, d_sfa_h_, stream_);
                                bump_version(x_attn);
                                bump_version(x_in);
                                set_nvfp4_act(x_in, H_);
                            } else {
                                efla_lm_cuda::add_scaled_to_int8_absmean_norm_q(
                                    x_attn, d_y_, H_, B_, mlp_residual_scale, ln_scale, x_in, stream_);
                                bump_version(x_attn);
                                bump_version(x_in);
                            }
                        } else {
                            if (mlp_residual_scale != 1.0f) {
                                efla_lm_cuda::add_scaled(d_y_, d_y_, B_ * H_, mlp_residual_scale - 1.0f, stream_);
                            }
                            efla_lm_cuda::add_scaled_i8(d_y_, x_attn, B_ * H_, 1.0f, stream_);
                            efla_lm_cuda::layernorm_quantize(d_y_, H_, B_, ln_scale, eps, x_in, stream_);
                            bump_version(x_in);
                        }
                    }
                    x_final = x_in;
                }

                NvtxRange range_head(use_nvtx_ && !use_cuda_graph_, "head");
                if (fused_head_loss) {
                    const float fused_noise = noise_active ? static_cast<float>(anti_sign) * noise_scale : 0.0f;
                    const int8_t* head_noise = noise_active ? d_head_noise_ : nullptr;
                    bitnet_cuda::head_gemm_cross_entropy(x_final,
                                                         d_head_,
                                                         head_noise,
                                                         d_scale_x_,
                                                         H_,
                                                         V_,
                                                         B_,
                                                         head_out_scale_,
                                                         fused_noise,
                                                         d_bias,
                                                         d_bias_noise,
                                                         bias_noise_scale,
                                                         d_targets_active_ + t,
                                                         T_,
                                                         d_loss_sum_,
                                                         stream_);
                } else {
                    float* d_logits_t = d_logits_ + static_cast<size_t>(t) * B_ * V_;
                    gemm_noisy(x_final, d_head_, d_head_noise_,
                               use_packed_gemm_ ? d_head_packed_ : nullptr,
                               use_packed_gemm_ ? d_head_noise_packed_ : nullptr,
                               d_scale_x_,
                               d_head_nvfp4_, d_head_sfb_,
                               d_head_noise_nvfp4_, d_head_noise_sfb_,
                               V_, H_,
                               head_out_scale_,
                               d_logits_t);
                    efla_lm_cuda::cross_entropy_loss(d_logits_t,
                                                     d_bias,
                                                     d_bias_noise,
                                                     bias_noise_scale,
                                                     d_targets_active_ + t,
                                                     T_,
                                                     V_,
                                                     B_,
                                                     d_loss_sum_,
                                                     stream_);
                }
            }
            if (capture_loss) {
                cuda_check(cudaMemcpyAsync(h_loss_sum_pinned_, d_loss_sum_, sizeof(float),
                                           cudaMemcpyDeviceToHost, stream_),
                           "cudaMemcpyAsync loss_sum");
            }
        };

        cudaGraphExec_t exec = nullptr;
        if (use_cuda_graph_) {
            GraphKey key{};
            key.anti_sign = anti_sign;
            key.sigma_shift = sigma_shift;
            key.method = method;
            key.hidden = H_;
            key.layers = L_;
            key.batch = B_;
            key.seq = T_;
            key.state_dim = D_;
            key.tokens_idx = active_tokens_idx_;
            key.graph_full_eval = graph_full_eval_;
            key.noise_active = noise_active;
            key.int8_residual = int8_residual;
            key.absmean_norm = absmean_norm;
            key.fp16_state = use_fp16_state_;
            key.efla_mixed = use_efla_mixed_;
            key.efla_fuse_diff = use_efla_fuse_diff_;
            key.efla_update_wmma = use_efla_update_wmma_;
            key.fused_head_loss = fused_head_loss;
            key.act_scale = act_scale;
            key.mlp_act_scale = mlp_act_scale;
            key.ln_scale = ln_scale;
            key.residual_scale = residual_scale;
            key.mlp_residual_scale = mlp_residual_scale;
            key.gate_scale = gate_scale;
            key.state_decay = state_decay;

            NvtxRange range_capture(use_nvtx_, "graph_capture");
            exec = get_or_create_graph(key, run_kernels);

            NvtxRange range_launch(use_nvtx_, "graph_launch");
            if (exec) {
                cuda_check(cudaGraphLaunch(exec, stream_), "cudaGraphLaunch");
            } else {
                use_cuda_graph_ = false;
                run_kernels();
            }
        } else {
            run_kernels();
        }

        if (!capture_loss) {
            cuda_check(cudaMemcpyAsync(h_loss_sum_pinned_, d_loss_sum_, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync loss_sum");
        }
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize loss_sum");
        return h_loss_sum_pinned_ ? (*h_loss_sum_pinned_) / static_cast<float>(T_ * B_) : 0.0f;
    }

private:
    int device_ = 0;
    int V_ = 0;
    int H_ = 0;
    int L_ = 0;
    int D_ = 0;
    int F_ = 0;
    int B_ = 0;
    int T_ = 0;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<DeviceWeights> weights_;

    int8_t* d_emb_w_ = nullptr;
    int8_t* d_win_ = nullptr;
    int8_t* d_wq_ = nullptr;
    int8_t* d_wk_ = nullptr;
    int8_t* d_wv_ = nullptr;
    int8_t* d_wbeta_ = nullptr;
    int8_t* d_ff1_ = nullptr;
    int8_t* d_ff2_ = nullptr;
    int8_t* d_head_ = nullptr;
    uint32_t* d_win_packed_ = nullptr;
    uint32_t* d_wq_packed_ = nullptr;
    uint32_t* d_wk_packed_ = nullptr;
    uint32_t* d_wv_packed_ = nullptr;
    uint32_t* d_wbeta_packed_ = nullptr;
    uint32_t* d_ff1_packed_ = nullptr;
    uint32_t* d_ff2_packed_ = nullptr;
    uint32_t* d_head_packed_ = nullptr;
    int pack_cols_h_ = 0;
    int pack_cols_f_ = 0;

    void* d_a_nvfp4_h_ = nullptr;
    void* d_a_nvfp4_f_ = nullptr;
    void* d_sfa_h_ = nullptr;
    void* d_sfa_f_ = nullptr;

    void* d_emb_nvfp4_ = nullptr;
    void* d_win_nvfp4_ = nullptr;
    void* d_head_nvfp4_ = nullptr;
    void* d_emb_sfb_ = nullptr;
    void* d_win_sfb_ = nullptr;
    void* d_head_sfb_ = nullptr;
    void* d_emb_noise_nvfp4_ = nullptr;
    void* d_win_noise_nvfp4_ = nullptr;
    void* d_head_noise_nvfp4_ = nullptr;
    void* d_emb_noise_sfb_ = nullptr;
    void* d_win_noise_sfb_ = nullptr;
    void* d_head_noise_sfb_ = nullptr;
    std::vector<void*> d_wq_nvfp4_;
    std::vector<void*> d_wk_nvfp4_;
    std::vector<void*> d_wv_nvfp4_;
    std::vector<void*> d_wbeta_nvfp4_;
    std::vector<void*> d_ff1_nvfp4_;
    std::vector<void*> d_ff2_nvfp4_;
    std::vector<void*> d_wq_sfb_;
    std::vector<void*> d_wk_sfb_;
    std::vector<void*> d_wv_sfb_;
    std::vector<void*> d_wbeta_sfb_;
    std::vector<void*> d_ff1_sfb_;
    std::vector<void*> d_ff2_sfb_;
    std::vector<void*> d_wqkv_nvfp4_;
    std::vector<void*> d_wqkv_sfb_;
    std::vector<void*> d_wqkv_noise_nvfp4_;
    std::vector<void*> d_wqkv_noise_sfb_;
    std::vector<void*> d_wq_noise_nvfp4_;
    std::vector<void*> d_wk_noise_nvfp4_;
    std::vector<void*> d_wv_noise_nvfp4_;
    std::vector<void*> d_wbeta_noise_nvfp4_;
    std::vector<void*> d_ff1_noise_nvfp4_;
    std::vector<void*> d_ff2_noise_nvfp4_;
    std::vector<void*> d_wq_noise_sfb_;
    std::vector<void*> d_wk_noise_sfb_;
    std::vector<void*> d_wv_noise_sfb_;
    std::vector<void*> d_wbeta_noise_sfb_;
    std::vector<void*> d_ff1_noise_sfb_;
    std::vector<void*> d_ff2_noise_sfb_;

    int8_t* d_emb_noise_ = nullptr;
    int8_t* d_win_noise_ = nullptr;
    int8_t* d_wq_noise_ = nullptr;
    int8_t* d_wk_noise_ = nullptr;
    int8_t* d_wv_noise_ = nullptr;
    int8_t* d_wbeta_noise_ = nullptr;
    int8_t* d_ff1_noise_ = nullptr;
    int8_t* d_ff2_noise_ = nullptr;
    int8_t* d_head_noise_ = nullptr;
    uint32_t* d_win_noise_packed_ = nullptr;
    uint32_t* d_wq_noise_packed_ = nullptr;
    uint32_t* d_wk_noise_packed_ = nullptr;
    uint32_t* d_wv_noise_packed_ = nullptr;
    uint32_t* d_wbeta_noise_packed_ = nullptr;
    uint32_t* d_ff1_noise_packed_ = nullptr;
    uint32_t* d_ff2_noise_packed_ = nullptr;
    uint32_t* d_head_noise_packed_ = nullptr;
    efla_lm_cuda::NoiseDesc* d_noise_desc_ = nullptr;
    std::vector<efla_lm_cuda::NoiseDesc> noise_desc_host_;
    int noise_desc_count_ = 0;
    int noise_desc_max_blocks_ = 0;

    float* d_scale_x_ = nullptr;
    float* d_scale_f_ = nullptr;
    float* d_pos_ = nullptr;
    static constexpr int kTokenBuffers = 2;
    uint8_t* d_tokens_[kTokenBuffers] = {nullptr, nullptr};
    uint8_t* d_targets_[kTokenBuffers] = {nullptr, nullptr};
    uint8_t* d_tokens_active_ = nullptr;
    uint8_t* d_targets_active_ = nullptr;
    uint8_t* h_tokens_pinned_[kTokenBuffers] = {nullptr, nullptr};
    uint8_t* h_targets_pinned_[kTokenBuffers] = {nullptr, nullptr};
    cudaStream_t upload_stream_ = nullptr;
    cudaEvent_t upload_done_[kTokenBuffers] = {nullptr, nullptr};
    int active_tokens_idx_ = 0;

    float* d_emb_f_ = nullptr;
    int8_t* d_emb_q_ = nullptr;
    float* d_inproj_f_ = nullptr;
    float* d_res_f_ = nullptr;
    int8_t* d_x_q_ = nullptr;

    float* d_q_ = nullptr;
    float* d_k_ = nullptr;
    float* d_v_ = nullptr;
    float* d_beta_raw_ = nullptr;
    float* d_qkv_f_ = nullptr;
    int qkv_fused_cols_ = 0;
    int qkv_fused_cols_valid_ = 0;
    float* d_q_norm_ = nullptr;
    float* d_k_usage_ = nullptr;
    __half* d_k_usage_h_ = nullptr;
    float* d_kS_ = nullptr;
    float* d_diff_ = nullptr;
    __half* d_diff_h_ = nullptr;
    float* d_alpha_ = nullptr;

    float* d_S_f_ = nullptr;
    __half* d_S_h_ = nullptr;
    float* d_K_f_ = nullptr;
    float* d_V_f_ = nullptr;
    float* d_y_ = nullptr;
    int8_t* d_y_q_ = nullptr;
    float* d_ff1_f_ = nullptr;
    int8_t* d_ff1_q_ = nullptr;

    float* d_logits_ = nullptr;
    float* d_tmp_noise_ = nullptr;
    float* h_logits_ = nullptr;
    float* d_head_bias_ = nullptr;
    float* d_head_bias_noise_ = nullptr;
    float* d_loss_sum_ = nullptr;
    float* h_loss_sum_pinned_ = nullptr;
    bool head_bias_cached_ = false;

    size_t state_bytes_ = 0;
    bool use_fp16_state_ = false;

    float win_out_scale_ = 1.0f;
    float q_out_scale_ = 1.0f;
    float k_out_scale_ = 1.0f;
    float v_out_scale_ = 1.0f;
    float beta_out_scale_ = 1.0f;
    float ff1_out_scale_ = 1.0f;
    float ff2_out_scale_ = 1.0f;
    float head_out_scale_ = 1.0f;

    // Host staging for packed per-layer weights (reused to avoid reallocs in hot loops).
    std::vector<int8_t> h_wq_pack_;
    std::vector<int8_t> h_wk_pack_;
    std::vector<int8_t> h_wv_pack_;
    std::vector<int8_t> h_wbeta_pack_;
    std::vector<int8_t> h_ff1_pack_;
    std::vector<int8_t> h_ff2_pack_;

    struct GraphKey {
        int anti_sign = 0;
        int sigma_shift = 0;
        int method = 0;
        int hidden = 0;
        int layers = 0;
        int batch = 0;
        int seq = 0;
        int state_dim = 0;
        int tokens_idx = 0;
        bool graph_full_eval = false;
        bool noise_active = false;
        bool int8_residual = false;
        bool absmean_norm = false;
        bool fp16_state = false;
        bool efla_mixed = false;
        bool efla_fuse_diff = false;
        bool efla_update_wmma = false;
        bool fused_head_loss = false;
        float act_scale = 0.0f;
        float mlp_act_scale = 0.0f;
        float ln_scale = 0.0f;
        float residual_scale = 0.0f;
        float mlp_residual_scale = 0.0f;
        float gate_scale = 0.0f;
        float state_decay = 0.0f;
    };

    struct GraphEntry {
        GraphKey key;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
    };

    std::vector<GraphEntry> graph_cache_;
    bool use_cuda_graph_ = false;
    bool graph_full_eval_ = false;
    bool use_nvtx_ = false;
    bool use_fused_qkv_ = true;
    bool use_qkv_split_kernel_ = true;
    bool use_qkv_split_vec4_ = true;
    bool use_qkv_split_bias_ = false;
    bool use_fused_ff1_ = true;
    bool use_fused_head_loss_ = false;
    bool use_fused_noise_gemm_ = true;
    bool use_fused_efla_ = false;
    bool use_efla_mixed_ = false;
    bool use_efla_fuse_diff_ = false;
    bool use_efla_update_wmma_ = false;
    bool use_tiled_gemm_ = false;
    bool use_packed_gemm_ = false;
    bool use_packed_gemm_bitnet_ = false;
    bool use_nvfp4_ = false;
    bool use_fused_qkv_nvfp4_ = false;
    TrainConfig::GemmBackend gemm_backend_ = TrainConfig::GemmBackend::CutlassNvfp4;
    bitnet_cuda::Nvfp4Schedule nvfp4_schedule_ = bitnet_cuda::Nvfp4Schedule::Auto;
    bitnet_cuda::Nvfp4QuantMode nvfp4_quant_mode_ = bitnet_cuda::Nvfp4QuantMode::Warp16;
    bitnet_cuda::Nvfp4StageCount nvfp4_stage_count_ = bitnet_cuda::Nvfp4StageCount::Auto;
    bitnet_cuda::Nvfp4Decomposition nvfp4_decomp_ = bitnet_cuda::Nvfp4Decomposition::Heuristic;
    int nvfp4_splits_ = 1;
    bool nvfp4_autotune_ = false;
    bool nvfp4_tune_schedule_ = false;
    bool nvfp4_tune_quant_ = false;
    bool nvfp4_tune_stages_ = false;
    bool nvfp4_tune_decomp_ = false;
    bool nvfp4_tune_splits_ = false;
    bool use_noise_batch_ = true;

    bool prewarm_nvfp4_gemm_cache(bool verbose = true) {
        if (!use_nvfp4_) return false;
        if (!d_a_nvfp4_h_ || !d_sfa_h_) return false;
        auto prep_h = [&](int rows, const void* d_b, const void* d_sfb, const float* d_c, float* d_d, const char* tag) -> bool {
            if (!d_b || !d_sfb) {
                if (verbose) {
                    fprintf(stderr, "nvfp4 prewarm skip %s (missing weights)\n", tag);
                }
                return true;
            }
            if (!bitnet_cuda::nvfp4_prepare_gemm(d_a_nvfp4_h_, d_b, d_sfa_h_, d_sfb,
                                                 B_, rows, H_,
                                                 1.0f, 0.0f,
                                                 d_c, d_d,
                                                 stream_)) {
                if (verbose) {
                    fprintf(stderr, "nvfp4 prewarm failed for %s (m=%d n=%d k=%d)\n", tag, B_, rows, H_);
                }
                return false;
            }
            return true;
        };
        auto prep_f = [&](int rows, const void* d_b, const void* d_sfb, const float* d_c, float* d_d, const char* tag) -> bool {
            if (!d_a_nvfp4_f_ || !d_sfa_f_) {
                if (verbose) {
                    fprintf(stderr, "nvfp4 prewarm skip %s (missing activation buffers)\n", tag);
                }
                return true;
            }
            if (!d_b || !d_sfb) {
                if (verbose) {
                    fprintf(stderr, "nvfp4 prewarm skip %s (missing weights)\n", tag);
                }
                return true;
            }
            if (!bitnet_cuda::nvfp4_prepare_gemm(d_a_nvfp4_f_, d_b, d_sfa_f_, d_sfb,
                                                 B_, rows, F_,
                                                 1.0f, 0.0f,
                                                 d_c, d_d,
                                                 stream_)) {
                if (verbose) {
                    fprintf(stderr, "nvfp4 prewarm failed for %s (m=%d n=%d k=%d)\n", tag, B_, rows, F_);
                }
                return false;
            }
            return true;
        };

        bool ok = true;
        ok &= prep_h(H_, d_win_nvfp4_, d_win_sfb_, d_inproj_f_, d_inproj_f_, "win");
        const int fused_rows = qkv_fused_cols_;
        if (use_fused_qkv_nvfp4_ && fused_rows > 0) {
            if (!d_qkv_f_) return false;
            ok &= prep_h(fused_rows, d_wqkv_nvfp4_.empty() ? nullptr : d_wqkv_nvfp4_[0],
                         d_wqkv_sfb_.empty() ? nullptr : d_wqkv_sfb_[0],
                         d_qkv_f_, d_qkv_f_, "qkv_fused");
        } else {
            ok &= prep_h(H_, d_wq_nvfp4_.empty() ? nullptr : d_wq_nvfp4_[0],
                         d_wq_sfb_.empty() ? nullptr : d_wq_sfb_[0],
                         d_q_, d_q_, "wq");
            ok &= prep_h(H_, d_wk_nvfp4_.empty() ? nullptr : d_wk_nvfp4_[0],
                         d_wk_sfb_.empty() ? nullptr : d_wk_sfb_[0],
                         d_k_, d_k_, "wk");
            ok &= prep_h(H_, d_wv_nvfp4_.empty() ? nullptr : d_wv_nvfp4_[0],
                         d_wv_sfb_.empty() ? nullptr : d_wv_sfb_[0],
                         d_v_, d_v_, "wv");
        }
        ok &= prep_h(F_, d_ff1_nvfp4_.empty() ? nullptr : d_ff1_nvfp4_[0],
                     d_ff1_sfb_.empty() ? nullptr : d_ff1_sfb_[0],
                     d_ff1_f_, d_ff1_f_, "ff1");
        ok &= prep_f(H_, d_ff2_nvfp4_.empty() ? nullptr : d_ff2_nvfp4_[0],
                     d_ff2_sfb_.empty() ? nullptr : d_ff2_sfb_[0],
                     d_y_, d_y_, "ff2");
        ok &= prep_h(V_, d_head_nvfp4_, d_head_sfb_, d_logits_, d_logits_, "head");
        return ok;
    }

    static bool graph_key_equal(const GraphKey& a, const GraphKey& b) {
        return a.anti_sign == b.anti_sign &&
               a.sigma_shift == b.sigma_shift &&
               a.method == b.method &&
               a.hidden == b.hidden &&
               a.layers == b.layers &&
               a.batch == b.batch &&
               a.seq == b.seq &&
               a.state_dim == b.state_dim &&
               a.tokens_idx == b.tokens_idx &&
               a.graph_full_eval == b.graph_full_eval &&
               a.noise_active == b.noise_active &&
               a.int8_residual == b.int8_residual &&
               a.absmean_norm == b.absmean_norm &&
               a.fp16_state == b.fp16_state &&
               a.efla_mixed == b.efla_mixed &&
               a.efla_fuse_diff == b.efla_fuse_diff &&
               a.efla_update_wmma == b.efla_update_wmma &&
               a.fused_head_loss == b.fused_head_loss &&
               a.act_scale == b.act_scale &&
               a.mlp_act_scale == b.mlp_act_scale &&
               a.ln_scale == b.ln_scale &&
               a.residual_scale == b.residual_scale &&
               a.mlp_residual_scale == b.mlp_residual_scale &&
               a.gate_scale == b.gate_scale &&
               a.state_decay == b.state_decay;
    }

    template <typename F>
    cudaGraphExec_t get_or_create_graph(const GraphKey& key, F&& capture) {
        for (auto& entry : graph_cache_) {
            if (graph_key_equal(entry.key, key)) {
                return entry.exec;
            }
        }
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        const cudaStreamCaptureMode mode = use_nvfp4_ ? cudaStreamCaptureModeRelaxed
                                                      : cudaStreamCaptureModeThreadLocal;
        cudaError_t err = cudaStreamBeginCapture(stream_, mode);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(err));
            cudaGetLastError();
            return nullptr;
        }
        capture();
        err = cudaStreamEndCapture(stream_, &graph);
        if (err != cudaSuccess || graph == nullptr) {
            fprintf(stderr, "cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
            const cudaError_t last = cudaGetLastError();
            if (last != cudaSuccess) {
                fprintf(stderr, "cudaGetLastError after capture: %s\n", cudaGetErrorString(last));
            }
            return nullptr;
        }
        err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
            cudaGraphDestroy(graph);
            cudaGetLastError();
            return nullptr;
        }
        graph_cache_.push_back(GraphEntry{key, graph, exec});
        fprintf(stderr,
                "cuda graph captured (hidden=%d layers=%d batch=%d seq=%d state_dim=%d tokens_buf=%d full_eval=%d)\n",
                key.hidden, key.layers, key.batch, key.seq, key.state_dim, key.tokens_idx,
                key.graph_full_eval ? 1 : 0);
        return exec;
    }
};

constexpr uint64_t kSaltEmb = 0x111111ULL;
constexpr uint64_t kSaltWin = 0x222222ULL;
constexpr uint64_t kSaltWq = 0x333333ULL;
constexpr uint64_t kSaltWk = 0x444444ULL;
constexpr uint64_t kSaltWv = 0x555555ULL;
constexpr uint64_t kSaltWbeta = 0x666666ULL;
constexpr uint64_t kSaltFf1 = 0x999999ULL;
constexpr uint64_t kSaltFf2 = 0xAAAAAAULL;
constexpr uint64_t kSaltHead = 0x777777ULL;
constexpr uint64_t kSaltHeadBias = 0x888888ULL;
constexpr uint64_t kSaltBetaBias = 0x9999999ULL;
constexpr uint64_t kSaltPos = 0xBEEFF00DULL;

constexpr size_t kNoiseParallelThreshold = 1u << 18;

static void fill_noise_matrix(TernaryMatrix& dst,
                              const TernaryMatrix& shape,
                              uint64_t matrix_seed,
                              bool use_clt,
                              int clt_k) {
    dst.rows = shape.rows;
    dst.cols = shape.cols;
    dst.out_scale = shape.out_scale;
    if (dst.w.size() != shape.w.size()) {
        dst.w.resize(shape.w.size());
    }
    const size_t n = dst.w.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= kNoiseParallelThreshold)
    for (long long idx = 0; idx < static_cast<long long>(n); ++idx) {
        dst.w[static_cast<size_t>(idx)] =
            noise_hash(matrix_seed, static_cast<uint64_t>(idx), /*anti*/1, use_clt, clt_k);
    }
#else
    for (size_t idx = 0; idx < n; ++idx) {
        dst.w[idx] = noise_hash(matrix_seed, static_cast<uint64_t>(idx), /*anti*/1, use_clt, clt_k);
    }
#endif
}

static void fill_noise_vector(std::vector<float>& dst,
                              int n,
                              uint64_t seed,
                              bool use_clt,
                              int clt_k) {
    const size_t nn = static_cast<size_t>(n);
    if (dst.size() != nn) {
        dst.resize(nn);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (nn >= kNoiseParallelThreshold)
    for (int i = 0; i < n; ++i) {
        dst[static_cast<size_t>(i)] =
            static_cast<float>(noise_hash(seed, static_cast<uint64_t>(i), /*anti*/1, use_clt, clt_k));
    }
#else
    for (int i = 0; i < n; ++i) {
        dst[static_cast<size_t>(i)] =
            static_cast<float>(noise_hash(seed, static_cast<uint64_t>(i), /*anti*/1, use_clt, clt_k));
    }
#endif
}

struct MatrixOptState {
    std::vector<float> acc;
    std::vector<float> var;
    std::vector<float> shadow;
    float rms_ema = 0.0f;
};

struct VectorOptState {
    std::vector<float> acc;
    std::vector<float> var;
    float rms_ema = 0.0f;
};

static void shape_pair_weights(std::vector<float>& w, const TrainConfig& cfg) {
    const int n = static_cast<int>(w.size());
    if (n <= 0) return;

    if (cfg.fitness_mode == TrainConfig::FitnessMode::CenteredRank) {
        static thread_local std::vector<int> order;
        static thread_local std::vector<float> ranked;
        order.resize(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] < w[b]; });
        ranked.resize(n);
        if (n == 1) {
            ranked[0] = 0.0f;
        } else {
            for (int r = 0; r < n; ++r) {
                const float cr = static_cast<float>(r) / static_cast<float>(n - 1) - 0.5f;
                ranked[order[r]] = cr;
            }
        }
        w.swap(ranked);

        float mean = 0.0f;
        for (float d : w) mean += d;
        mean /= static_cast<float>(n);
        float var = 0.0f;
        for (float d : w) var += (d - mean) * (d - mean);
        var /= static_cast<float>(n);
        const float stdv = std::sqrt(var) + 1e-6f;
        for (float& d : w) {
            d = (d - mean) / stdv;
            if (d > cfg.fitness_clip) d = cfg.fitness_clip;
            if (d < -cfg.fitness_clip) d = -cfg.fitness_clip;
        }
        return;
    }

    if (cfg.fitness_mode == TrainConfig::FitnessMode::ZScore) {
        float mean = 0.0f;
        for (float d : w) mean += d;
        mean /= static_cast<float>(n);
        float var = 0.0f;
        for (float d : w) var += (d - mean) * (d - mean);
        var /= static_cast<float>(n);
        const float stdv = std::sqrt(var) + 1e-6f;
        for (float& d : w) {
            d = (d - mean) / stdv;
            if (d > cfg.fitness_clip) d = cfg.fitness_clip;
            if (d < -cfg.fitness_clip) d = -cfg.fitness_clip;
        }
        return;
    }

    // Sign.
    for (float& d : w) d = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
}

static void update_matrix(TernaryMatrix& m,
                          MatrixOptState& st,
                          const std::vector<float>& pair_weights,
                          const TrainConfig& cfg,
                          int epoch,
                          uint64_t salt,
                          float lr_t,
                          float thresh_t) {
    const int pairs = static_cast<int>(pair_weights.size());
    const float thresh = thresh_t * std::sqrt(static_cast<float>(pairs));

    std::vector<uint64_t> matrix_seeds(static_cast<size_t>(pairs));
    for (int p = 0; p < pairs; ++p) {
        const uint64_t pair_seed =
            rng::mix(cfg.seed, rng::mix(static_cast<uint64_t>(epoch), static_cast<uint64_t>(p)));
        matrix_seeds[static_cast<size_t>(p)] = rng::mix(pair_seed, salt);
    }

    std::vector<float> Z(m.w.size(), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m.w.size() >= kNoiseParallelThreshold)
#endif
    for (long long idx = 0; idx < static_cast<long long>(m.w.size()); ++idx) {
        float z = 0.0f;
        for (int p = 0; p < pairs; ++p) {
            const float wpair = pair_weights[p];
            if (wpair == 0.0f) continue;
            const uint64_t matrix_seed = matrix_seeds[static_cast<size_t>(p)];
            const int8_t eps = noise_hash(matrix_seed, static_cast<uint64_t>(idx), /*anti*/1,
                                          cfg.use_clt_noise, cfg.clt_k);
            if (eps == 0) continue;
            z += wpair * static_cast<float>(eps);
        }
        Z[static_cast<size_t>(idx)] = z;
    }

    float inv_rms = 1.0f;
    if (cfg.use_adaptive) {
        double ms = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : ms) schedule(static) if (Z.size() >= kNoiseParallelThreshold)
#endif
        for (long long i = 0; i < static_cast<long long>(Z.size()); ++i) {
            const double z = static_cast<double>(Z[static_cast<size_t>(i)]);
            ms += z * z;
        }
        ms /= static_cast<double>(Z.size());
        if (st.rms_ema == 0.0f) st.rms_ema = static_cast<float>(ms);
        st.rms_ema = cfg.adaptive_beta * st.rms_ema +
                     (1.0f - cfg.adaptive_beta) * static_cast<float>(ms);
        inv_rms = 1.0f / (std::sqrt(st.rms_ema) + cfg.adaptive_eps);
    }

    if (cfg.use_shadow) {
        if (st.shadow.size() != m.w.size()) {
            st.shadow.resize(m.w.size());
            for (size_t i = 0; i < m.w.size(); ++i) st.shadow[i] = static_cast<float>(m.w[i]);
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m.w.size() >= kNoiseParallelThreshold)
#endif
        for (long long idx = 0; idx < static_cast<long long>(m.w.size()); ++idx) {
            const float g = inv_rms * Z[idx];
            if (std::abs(g) < thresh) continue;
            const size_t u = static_cast<size_t>(idx);
            st.shadow[u] += lr_t * g;
            m.w[u] = clip_ternary(static_cast<int>(std::lrint(st.shadow[u])));
        }
        return;
    }

    if (cfg.use_adam) {
        if (st.acc.size() != m.w.size()) st.acc.assign(m.w.size(), 0.0f);
        if (st.var.size() != m.w.size()) st.var.assign(m.w.size(), 0.0f);
        const float b1 = cfg.momentum_beta;
        const float b2 = cfg.adam_beta2;
        const float lr = lr_t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m.w.size() >= kNoiseParallelThreshold)
#endif
        for (long long idx = 0; idx < static_cast<long long>(m.w.size()); ++idx) {
            const size_t u = static_cast<size_t>(idx);
            const float g = inv_rms * Z[u];
            float& m1 = st.acc[u];
            float& m2 = st.var[u];
            m1 = b1 * m1 + (1.0f - b1) * g;
            m2 = b2 * m2 + (1.0f - b2) * g * g;
            const float denom = std::sqrt(m2) + cfg.adam_eps;
            const float step_val = lr * m1 / denom;
            if (std::abs(step_val) < thresh) continue;
            const int step = (step_val > 0.0f) ? 1 : -1;
            m.w[u] = clip_ternary(static_cast<int>(m.w[u]) + step);
            const float lr_safe = (lr > 1e-12f) ? lr : 1e-12f;
            m1 -= static_cast<float>(step) * thresh * denom / lr_safe; // keep residual
        }
        return;
    }

    if (cfg.use_momentum) {
        if (st.acc.size() != m.w.size()) st.acc.assign(m.w.size(), 0.0f);
        const float beta = cfg.momentum_beta;
        const float scale = lr_t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m.w.size() >= kNoiseParallelThreshold)
#endif
        for (long long idx = 0; idx < static_cast<long long>(m.w.size()); ++idx) {
            const size_t u = static_cast<size_t>(idx);
            const float g = inv_rms * Z[u];
            float a = beta * st.acc[u] + (1.0f - beta) * scale * g;
            st.acc[u] = a;
            if (std::abs(a) < thresh) continue;
            const int step = (a > 0.0f) ? 1 : -1;
            m.w[u] = clip_ternary(static_cast<int>(m.w[u]) + step);
            st.acc[u] = a - static_cast<float>(step) * thresh;
        }
        return;
    }

    // Plain sign update (no optimizer).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (m.w.size() >= kNoiseParallelThreshold)
#endif
    for (long long idx = 0; idx < static_cast<long long>(m.w.size()); ++idx) {
        const size_t u = static_cast<size_t>(idx);
        const float g = inv_rms * Z[u];
        if (std::abs(g) < thresh) continue;
        const int step = (g > 0.0f) ? 1 : -1;
        m.w[u] = clip_ternary(static_cast<int>(m.w[u]) + step);
    }
}

static void update_vector(std::vector<float>& v,
                          VectorOptState& st,
                          const std::vector<float>& pair_weights,
                          const TrainConfig& cfg,
                          int epoch,
                          uint64_t salt,
                          float lr_t,
                          float thresh_t) {
    const int pairs = static_cast<int>(pair_weights.size());
    if (pairs <= 0 || v.empty()) return;
    const float thresh = thresh_t * std::sqrt(static_cast<float>(pairs));

    std::vector<uint64_t> vec_seeds(static_cast<size_t>(pairs));
    for (int p = 0; p < pairs; ++p) {
        const uint64_t pair_seed =
            rng::mix(cfg.seed, rng::mix(static_cast<uint64_t>(epoch), static_cast<uint64_t>(p)));
        vec_seeds[static_cast<size_t>(p)] = rng::mix(pair_seed, salt);
    }

    std::vector<float> Z(v.size(), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (v.size() >= kNoiseParallelThreshold)
#endif
    for (long long idx = 0; idx < static_cast<long long>(v.size()); ++idx) {
        float z = 0.0f;
        for (int p = 0; p < pairs; ++p) {
            const float wpair = pair_weights[p];
            if (wpair == 0.0f) continue;
            const uint64_t vec_seed = vec_seeds[static_cast<size_t>(p)];
            const int8_t eps = noise_hash(vec_seed, static_cast<uint64_t>(idx), /*anti*/1,
                                          cfg.use_clt_noise, cfg.clt_k);
            if (eps == 0) continue;
            z += wpair * static_cast<float>(eps);
        }
        Z[static_cast<size_t>(idx)] = z;
    }

    float inv_rms = 1.0f;
    if (cfg.use_adaptive) {
        double ms = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : ms) schedule(static) if (Z.size() >= kNoiseParallelThreshold)
#endif
        for (long long i = 0; i < static_cast<long long>(Z.size()); ++i) {
            const double z = static_cast<double>(Z[static_cast<size_t>(i)]);
            ms += z * z;
        }
        ms /= static_cast<double>(Z.size());
        if (st.rms_ema == 0.0f) st.rms_ema = static_cast<float>(ms);
        st.rms_ema = cfg.adaptive_beta * st.rms_ema +
                     (1.0f - cfg.adaptive_beta) * static_cast<float>(ms);
        inv_rms = 1.0f / (std::sqrt(st.rms_ema) + cfg.adaptive_eps);
    }

    if (cfg.use_adam) {
        if (st.acc.size() != v.size()) st.acc.assign(v.size(), 0.0f);
        if (st.var.size() != v.size()) st.var.assign(v.size(), 0.0f);
        const float b1 = cfg.momentum_beta;
        const float b2 = cfg.adam_beta2;
        const float lr = lr_t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (v.size() >= kNoiseParallelThreshold)
#endif
        for (long long idx = 0; idx < static_cast<long long>(v.size()); ++idx) {
            const size_t u = static_cast<size_t>(idx);
            const float g = inv_rms * Z[u];
            float& m1 = st.acc[u];
            float& m2 = st.var[u];
            m1 = b1 * m1 + (1.0f - b1) * g;
            m2 = b2 * m2 + (1.0f - b2) * g * g;
            const float uval = lr * m1 / (std::sqrt(m2) + cfg.adam_eps);
            if (std::abs(uval) < thresh) continue;
            v[u] += uval;
        }
        return;
    }

    if (cfg.use_momentum) {
        if (st.acc.size() != v.size()) st.acc.assign(v.size(), 0.0f);
        const float beta = cfg.momentum_beta;
        const float scale = lr_t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (v.size() >= kNoiseParallelThreshold)
#endif
        for (long long idx = 0; idx < static_cast<long long>(v.size()); ++idx) {
            const size_t u = static_cast<size_t>(idx);
            const float g = inv_rms * Z[u];
            float a = beta * st.acc[u] + (1.0f - beta) * scale * g;
            st.acc[u] = a;
            if (std::abs(a) < thresh) continue;
            v[u] += a;
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (v.size() >= kNoiseParallelThreshold)
#endif
    for (long long idx = 0; idx < static_cast<long long>(v.size()); ++idx) {
        const size_t u = static_cast<size_t>(idx);
        const float g = inv_rms * Z[u];
        if (std::abs(g) < thresh) continue;
        v[u] += lr_t * g;
    }
}

struct GpuUpdateBuffers {
    float* d_Z = nullptr;
    float* d_sumsq = nullptr;
    float* d_inv_rms = nullptr;
    float* d_pair_weights = nullptr;
    uint64_t* d_pair_seeds = nullptr;
    size_t z_capacity = 0;
    int pairs_capacity = 0;
};

struct GpuMatrixState {
    float* d_shadow = nullptr;
    float* d_rms_ema = nullptr;
    bool initialized = false;
    size_t size = 0;
};

struct GpuUpdateDevice {
    std::vector<GpuUpdateBuffers> bufs;
    std::vector<cudaEvent_t> stream_events;
    int stream_count = 1;
    GpuMatrixState emb;
    GpuMatrixState win;
    GpuMatrixState head;
    std::vector<GpuMatrixState> wq;
    std::vector<GpuMatrixState> wk;
    std::vector<GpuMatrixState> wv;
    std::vector<GpuMatrixState> wbeta;
    std::vector<GpuMatrixState> ff1;
    std::vector<GpuMatrixState> ff2;
};

static void gpu_update_free_buffers(int device, GpuUpdateBuffers& buf) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice gpu_update_free");
    if (buf.d_Z) cudaFree(buf.d_Z);
    if (buf.d_sumsq) cudaFree(buf.d_sumsq);
    if (buf.d_inv_rms) cudaFree(buf.d_inv_rms);
    if (buf.d_pair_weights) cudaFree(buf.d_pair_weights);
    if (buf.d_pair_seeds) cudaFree(buf.d_pair_seeds);
    buf = {};
}

static void gpu_update_free_matrix(int device, GpuMatrixState& st) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice gpu_update_free_matrix");
    if (st.d_shadow) cudaFree(st.d_shadow);
    if (st.d_rms_ema) cudaFree(st.d_rms_ema);
    st.d_shadow = nullptr;
    st.d_rms_ema = nullptr;
    st.initialized = false;
    st.size = 0;
}

static void gpu_update_ensure_buffers(CudaEflaLm& model,
                                      GpuUpdateBuffers& buf,
                                      size_t z_elems,
                                      int pairs) {
    cuda_check(cudaSetDevice(model.device()), "cudaSetDevice gpu_update_ensure");
    if (buf.z_capacity < z_elems) {
        if (buf.d_Z) cudaFree(buf.d_Z);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&buf.d_Z), z_elems * sizeof(float)), "cudaMalloc update Z");
        buf.z_capacity = z_elems;
    }
    if (!buf.d_sumsq) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&buf.d_sumsq), sizeof(float)), "cudaMalloc update sumsq");
    }
    if (!buf.d_inv_rms) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&buf.d_inv_rms), sizeof(float)), "cudaMalloc update inv_rms");
    }
    if (buf.pairs_capacity < pairs) {
        if (buf.d_pair_weights) cudaFree(buf.d_pair_weights);
        if (buf.d_pair_seeds) cudaFree(buf.d_pair_seeds);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&buf.d_pair_weights), pairs * sizeof(float)),
                   "cudaMalloc update pair_weights");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&buf.d_pair_seeds), pairs * sizeof(uint64_t)),
                   "cudaMalloc update pair_seeds");
        buf.pairs_capacity = pairs;
    }
}

static void gpu_update_alloc_shadow(CudaEflaLm& model, GpuMatrixState& st, size_t n) {
    if (n == 0) return;
    if (st.d_shadow && st.size == n) return;
    gpu_update_free_matrix(model.device(), st);
    cuda_check(cudaSetDevice(model.device()), "cudaSetDevice gpu_update_alloc_shadow");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&st.d_shadow), n * sizeof(float)), "cudaMalloc update shadow");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&st.d_rms_ema), sizeof(float)), "cudaMalloc update rms_ema");
    cuda_check(cudaMemsetAsync(st.d_rms_ema, 0, sizeof(float), model.stream()), "cudaMemsetAsync update rms_ema");
    st.size = n;
    st.initialized = false;
}

static void gpu_update_upload_pair_weights(CudaEflaLm& model,
                                           GpuUpdateBuffers& buf,
                                           const float* pair_weights,
                                           int pair_count) {
    if (!pair_weights || pair_count <= 0) return;
    gpu_update_ensure_buffers(model, buf, buf.z_capacity, pair_count);
    cuda_check(cudaMemcpyAsync(buf.d_pair_weights, pair_weights,
                               static_cast<size_t>(pair_count) * sizeof(float),
                               cudaMemcpyHostToDevice, model.stream()),
               "cudaMemcpyAsync pair_weights");
}

static void gpu_update_fill_pair_seeds(CudaEflaLm& model,
                                       GpuUpdateBuffers& buf,
                                       uint64_t base_seed,
                                       uint64_t epoch,
                                       int pairs) {
    if (pairs <= 0) return;
    gpu_update_ensure_buffers(model, buf, buf.z_capacity, pairs);
    efla_lm_cuda::fill_pair_seeds(base_seed, epoch, pairs, buf.d_pair_seeds, model.stream());
}

static void gpu_update_matrix(CudaEflaLm& model,
                              GpuUpdateBuffers& buf,
                              GpuMatrixState& st_gpu,
                              const TrainConfig& cfg,
                              uint64_t salt,
                              float lr_t,
                              float thresh_t,
                              int8_t* d_w,
                              size_t n,
                              int pairs,
                              uint32_t* d_w_packed) {
    if (n == 0 || pairs == 0) return;
    gpu_update_ensure_buffers(model, buf, n, pairs);
    gpu_update_alloc_shadow(model, st_gpu, n);

    if (!st_gpu.initialized) {
        efla_lm_cuda::init_shadow_from_weights(d_w, st_gpu.d_shadow, static_cast<int>(n), model.stream());
        st_gpu.initialized = true;
    }

    cuda_check(cudaMemsetAsync(buf.d_sumsq, 0, sizeof(float), model.stream()),
               "cudaMemsetAsync update sumsq");
    efla_lm_cuda::compute_z_sumsq(buf.d_pair_weights, buf.d_pair_seeds,
                                 salt, pairs, static_cast<int>(n),
                                 cfg.use_clt_noise, cfg.clt_k,
                                 buf.d_Z, buf.d_sumsq, model.stream());
    efla_lm_cuda::update_inv_rms(buf.d_sumsq, st_gpu.d_rms_ema,
                                 cfg.use_adaptive,
                                 cfg.adaptive_beta,
                                 cfg.adaptive_eps,
                                 static_cast<int>(n),
                                 buf.d_inv_rms, model.stream());

    const float thresh = thresh_t * std::sqrt(static_cast<float>(pairs));
    if (cfg.use_packed_update && d_w_packed) {
        if (cfg.use_packed_gemm_bitnet) {
            efla_lm_cuda::update_shadow_ternary_device_packed_bitnet(
                d_w, d_w_packed, st_gpu.d_shadow, buf.d_Z,
                static_cast<int>(n),
                buf.d_inv_rms,
                lr_t, thresh, model.stream());
        } else {
            efla_lm_cuda::update_shadow_ternary_device_packed(
                d_w, d_w_packed, st_gpu.d_shadow, buf.d_Z,
                static_cast<int>(n),
                buf.d_inv_rms,
                lr_t, thresh, model.stream());
        }
    } else {
        efla_lm_cuda::update_shadow_ternary_device(d_w, st_gpu.d_shadow, buf.d_Z,
                                                   static_cast<int>(n),
                                                   buf.d_inv_rms,
                                                   lr_t, thresh, model.stream());
    }
}

} // namespace

int main(int argc, char** argv) {
    TrainConfig cfg;
    bool device_specified = false;
    bool devices_specified = false;
    bool model_8m_requested = false;
    bool lr_specified = false;
    bool lr_end_specified = false;
    bool thresh_specified = false;
    bool sigma_specified = false;
    bool act_scale_specified = false;
    bool mlp_act_scale_specified = false;
    bool nvfp4_schedule_specified = (cfg.nvfp4_schedule != bitnet_cuda::Nvfp4Schedule::Auto);
    bool nvfp4_quant_specified = false;
    bool nvfp4_stage_specified = (cfg.nvfp4_stage_count != bitnet_cuda::Nvfp4StageCount::Auto);
    bool nvfp4_decomp_specified = false;
    bool nvfp4_splits_specified = false;
    bool nvfp4_autotune_specified = false;
    bool cuda_graph_specified = false;
    bool efla_fused_specified = false;
    bool lr_schedule_specified = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int& i) -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--data") cfg.data_path = next(i);
        else if (a == "--device") {
            cfg.device = std::stoi(next(i));
            device_specified = true;
        } else if (a == "--devices") {
            cfg.devices = parse_devices(next(i));
            devices_specified = true;
        } else if (a == "--gpu_workers") {
            cfg.gpu_workers = std::max(1, std::stoi(next(i)));
        } else if (a == "--update_streams") {
            cfg.update_streams = std::max(1, std::stoi(next(i)));
        }
        else if (a == "--epochs") cfg.epochs = std::stoi(next(i));
        else if (a == "--pop") cfg.population = std::stoi(next(i));
        else if (a == "--adaptive_pop") cfg.adaptive_pop = true;
        else if (a == "--no_adaptive_pop") cfg.adaptive_pop = false;
        else if (a == "--batch") cfg.batch = std::stoi(next(i));
        else if (a == "--seq") cfg.seq_len = std::stoi(next(i));
        else if (a == "--hidden") cfg.hidden = std::stoi(next(i));
        else if (a == "--layers") cfg.layers = std::stoi(next(i));
        else if (a == "--mlp_mult") cfg.mlp_mult = std::stoi(next(i));
        else if (a == "--model_8m") {
            cfg.hidden = 416;
            cfg.layers = 4;
            cfg.mlp_mult = 4;
            model_8m_requested = true;
        }
        else if (a == "--sigma_shift") {
            cfg.sigma_shift = std::stoi(next(i));
            sigma_specified = true;
        }
        else if (a == "--sigma_shift_end") cfg.sigma_shift_end = std::stoi(next(i));
        else if (a == "--sigma_schedule") cfg.sigma_schedule = parse_schedule(next(i));
        else if (a == "--state_dim") cfg.state_dim = std::stoi(next(i));
        else if (a == "--state_fp16") cfg.state_fp16 = true;
        else if (a == "--no_state_fp16") cfg.state_fp16 = false;
        else if (a == "--train_pos") cfg.train_pos = true;
        else if (a == "--no_train_pos") cfg.train_pos = false;
        else if (a == "--pos_lr_mult") cfg.pos_lr_mult = std::stof(next(i));
        else if (a == "--pos_thresh_mult") cfg.pos_thresh_mult = std::stof(next(i));
        else if (a == "--gpu_noise") cfg.use_gpu_noise = true;
        else if (a == "--no_gpu_noise") cfg.use_gpu_noise = false;
        else if (a == "--gpu_update") cfg.use_gpu_update = true;
        else if (a == "--no_gpu_update") cfg.use_gpu_update = false;
        else if (a == "--cuda_graph") { cfg.use_cuda_graph = true; cuda_graph_specified = true; }
        else if (a == "--no_cuda_graph") { cfg.use_cuda_graph = false; cuda_graph_specified = true; }
        else if (a == "--graph_full_eval") cfg.graph_full_eval = true;
        else if (a == "--no_graph_full_eval") cfg.graph_full_eval = false;
        else if (a == "--nvtx") cfg.use_nvtx = true;
        else if (a == "--no_nvtx") cfg.use_nvtx = false;
        else if (a == "--fused_qkv") cfg.use_fused_qkv = true;
        else if (a == "--no_fused_qkv") cfg.use_fused_qkv = false;
        else if (a == "--qkv_split_kernel") cfg.use_qkv_split_kernel = true;
        else if (a == "--no_qkv_split_kernel") cfg.use_qkv_split_kernel = false;
        else if (a == "--qkv_split_vec4") cfg.qkv_split_vec4 = 1;
        else if (a == "--no_qkv_split_vec4") cfg.qkv_split_vec4 = 0;
        else if (a == "--qkv_split_vec4_auto") cfg.qkv_split_vec4 = -1;
        else if (a == "--qkv_split_bias") { cfg.use_qkv_split_bias = true; }
        else if (a == "--no_qkv_split_bias") { cfg.use_qkv_split_bias = false; }
        else if (a == "--fused_ff1") cfg.use_fused_ff1 = true;
        else if (a == "--no_fused_ff1") cfg.use_fused_ff1 = false;
        else if (a == "--fused_head_loss") cfg.use_fused_head_loss = true;
        else if (a == "--no_fused_head_loss") cfg.use_fused_head_loss = false;
        else if (a == "--fused_noise_gemm") cfg.use_fused_noise_gemm = true;
        else if (a == "--no_fused_noise_gemm") cfg.use_fused_noise_gemm = false;
        else if (a == "--noise_batch") cfg.noise_batch = true;
        else if (a == "--no_noise_batch") cfg.noise_batch = false;
        else if (a == "--pinned_tokens") cfg.use_pinned_tokens = true;
        else if (a == "--no_pinned_tokens") cfg.use_pinned_tokens = false;
        else if (a == "--gemm_backend") cfg.gemm_backend = parse_gemm_backend(next(i));
        else if (a == "--cutlass_gemm") cfg.gemm_backend = TrainConfig::GemmBackend::Cutlass;
        else if (a == "--cutlass_nvfp4") cfg.gemm_backend = TrainConfig::GemmBackend::CutlassNvfp4;
        else if (a == "--nvfp4_schedule") { cfg.nvfp4_schedule = parse_nvfp4_schedule(next(i)); nvfp4_schedule_specified = true; }
        else if (a == "--nvfp4_quant") { cfg.nvfp4_quant_mode = parse_nvfp4_quant_mode(next(i)); nvfp4_quant_specified = true; }
        else if (a == "--nvfp4_stages") { cfg.nvfp4_stage_count = parse_nvfp4_stage_count(next(i)); nvfp4_stage_specified = true; }
        else if (a == "--nvfp4_decomp") { cfg.nvfp4_decomp = parse_nvfp4_decomp(next(i)); nvfp4_decomp_specified = true; }
        else if (a == "--nvfp4_splits") { cfg.nvfp4_splits = std::max(1, std::stoi(next(i))); nvfp4_splits_specified = true; }
        else if (a == "--nvfp4_autotune") { cfg.nvfp4_autotune = true; nvfp4_autotune_specified = true; }
        else if (a == "--no_nvfp4_autotune") { cfg.nvfp4_autotune = false; nvfp4_autotune_specified = true; }
        else if (a == "--efla_fused") { cfg.use_fused_efla = true; efla_fused_specified = true; }
        else if (a == "--no_efla_fused") { cfg.use_fused_efla = false; efla_fused_specified = true; }
        else if (a == "--efla_mixed") cfg.use_efla_mixed = true;
        else if (a == "--no_efla_mixed") cfg.use_efla_mixed = false;
        else if (a == "--efla_fuse_diff") cfg.use_efla_fuse_diff = true;
        else if (a == "--no_efla_fuse_diff") cfg.use_efla_fuse_diff = false;
        else if (a == "--efla_update_wmma") cfg.use_efla_update_wmma = true;
        else if (a == "--no_efla_update_wmma") cfg.use_efla_update_wmma = false;
        else if (a == "--tiled_gemm") cfg.use_tiled_gemm = true;
        else if (a == "--no_tiled_gemm") cfg.use_tiled_gemm = false;
        else if (a == "--packed_gemm") cfg.use_packed_gemm = true;
        else if (a == "--no_packed_gemm") cfg.use_packed_gemm = false;
        else if (a == "--packed_gemm_bitnet") { cfg.use_packed_gemm = true; cfg.use_packed_gemm_bitnet = true; }
        else if (a == "--no_packed_gemm_bitnet") cfg.use_packed_gemm_bitnet = false;
        else if (a == "--packed_update") cfg.use_packed_update = true;
        else if (a == "--no_packed_update") cfg.use_packed_update = false;
        else if (a == "--omp_threads") cfg.omp_threads = std::stoi(next(i));
        else if (a == "--clt") cfg.use_clt_noise = true;
        else if (a == "--no_clt") cfg.use_clt_noise = false;
        else if (a == "--clt_k") cfg.clt_k = std::stoi(next(i));
        else if (a == "--thresh") {
            cfg.update_threshold = std::stof(next(i));
            thresh_specified = true;
        }
        else if (a == "--thresh_end") cfg.update_threshold_end = std::stof(next(i));
        else if (a == "--thresh_schedule") cfg.thresh_schedule = parse_schedule(next(i));
        else if (a == "--lr") {
            cfg.lr = std::stof(next(i));
            lr_specified = true;
        }
        else if (a == "--lr_end") {
            cfg.lr_end = std::stof(next(i));
            lr_end_specified = true;
        }
        else if (a == "--lr_schedule") {
            cfg.lr_schedule = parse_schedule(next(i));
            lr_schedule_specified = true;
        }
        else if (a == "--act_scale") {
            cfg.act_scale = std::stof(next(i));
            act_scale_specified = true;
        }
        else if (a == "--mlp_act_scale") {
            cfg.mlp_act_scale = std::stof(next(i));
            mlp_act_scale_specified = true;
        }
        else if (a == "--ln_scale") cfg.ln_scale = std::stof(next(i));
        else if (a == "--residual_scale") cfg.residual_scale = std::stof(next(i));
        else if (a == "--mlp_residual_scale") cfg.mlp_residual_scale = std::stof(next(i));
        else if (a == "--state_decay") cfg.state_decay = std::stof(next(i));
        else if (a == "--gate_scale") cfg.gate_scale = std::stof(next(i));
        else if (a == "--int8_residual") cfg.int8_residual = true;
        else if (a == "--no_int8_residual") cfg.int8_residual = false;
        else if (a == "--absmean_norm") cfg.absmean_norm = true;
        else if (a == "--no_absmean_norm") cfg.absmean_norm = false;
        else if (a == "--fitness") cfg.fitness_mode = parse_fitness_mode(next(i));
        else if (a == "--fitness_clip") cfg.fitness_clip = std::stof(next(i));
        else if (a == "--momentum") cfg.use_momentum = true;
        else if (a == "--no_momentum") cfg.use_momentum = false;
        else if (a == "--momentum_beta") cfg.momentum_beta = std::stof(next(i));
        else if (a == "--adam") cfg.use_adam = true;
        else if (a == "--no_adam") cfg.use_adam = false;
        else if (a == "--adam_beta2") cfg.adam_beta2 = std::stof(next(i));
        else if (a == "--adam_eps") cfg.adam_eps = std::stof(next(i));
        else if (a == "--shadow") cfg.use_shadow = true;
        else if (a == "--no_shadow") cfg.use_shadow = false;
        else if (a == "--adaptive") cfg.use_adaptive = true;
        else if (a == "--no_adaptive") cfg.use_adaptive = false;
        else if (a == "--adaptive_beta") cfg.adaptive_beta = std::stof(next(i));
        else if (a == "--adaptive_eps") cfg.adaptive_eps = std::stof(next(i));
        else if (a == "--no_val") cfg.do_val = false;
        else if (a == "--val_seed") cfg.val_seed = std::stoull(next(i));
        else if (a == "--val_every") cfg.val_every = std::stoi(next(i));
        else if (a == "--fixed_train") cfg.fixed_train = true;
        else if (a == "--data_seed") cfg.data_seed = std::stoull(next(i));
        else if (a == "--seed") cfg.seed = std::stoull(next(i));
        else if (a == "--method") cfg.method = parse_method(next(i));
        else if (a == "--help") {
            std::cout <<
                "train_efla_lm options:\n"
                "  --data PATH        byte-level training file (default data/tinyshakespeare.txt)\n"
                "  --device N         single CUDA device (overrides default all GPUs)\n"
                "  --devices LIST     comma-separated CUDA devices for data-parallel eval (default: all GPUs)\n"
                "  --gpu_workers N    workers per GPU device (default 6)\n"
                "  --update_streams N concurrent GPU update streams per device (default 2)\n"
                "  --method efla|DeltaNet\n"
                "  --epochs N         epochs (default 50)\n"
                "  --pop N            ES population (even, default 2048)\n"
                "  --adaptive_pop     ramp up population: 50% -> 75% -> 100% (default, 12% faster)\n"
                "  --no_adaptive_pop  use fixed population throughout training\n"
                "  --batch N          batch size (default 32)\n"
                "  --seq N            sequence length (default 128)\n"
                "  --hidden H         hidden dim (default 512)\n"
                "  --layers L         number of EFLA blocks (default 4)\n"
                "  --mlp_mult M       MLP expansion ratio (default 2)\n"
                "  --model_8m         preset ~8M params (hidden=416, layers=4, mlp_mult=4)\n"
                "                    + tuned defaults if not overridden: lr=0.80, thresh=0.008,\n"
                "                      thresh_end=0.008, sigma_shift=3->4, lr_schedule=cosine,\n"
                "                      act_scale=0.8, mlp_act_scale=1.0, lr_end=0.05\n"
                "  --sigma_shift S    noise scale 1/2^S (default 3)\n"
                "  --sigma_shift_end S end sigma_shift (default 4)\n"
                "  --sigma_schedule {constant|linear|cosine|exp} (default exp)\n"
                "  --state_dim D      low-rank state dim (0 = full HxH state)\n"
                "  --state_fp16       store EFLA state in FP16 (default on)\n"
                "  --no_state_fp16    store EFLA state in FP32\n"
                "  --train_pos        train positional embeddings (default on)\n"
                "  --no_train_pos     keep positional embeddings fixed\n"
                "  --pos_lr_mult X    scale lr for pos embeddings (default 1.0)\n"
                "  --pos_thresh_mult X scale thresh for pos embeddings (default 1.0)\n"
                "  --gpu_noise        generate noise weights on GPU (default on)\n"
                "  --no_gpu_noise     generate noise weights on CPU\n"
                "  --gpu_update       update ternary weights on GPU (default on)\n"
                "  --no_gpu_update    update ternary weights on CPU\n"
                "  --cuda_graph       capture forward pass in a CUDA graph (default off)\n"
                "  --no_cuda_graph    disable CUDA graph capture\n"
                "  --graph_full_eval  include loss copy in CUDA graph (default on)\n"
                "  --no_graph_full_eval keep loss copy outside CUDA graph\n"
                "  --nvtx             enable NVTX profiling ranges (default off)\n"
                "  --no_nvtx          disable NVTX ranges\n"
                "  --fused_qkv        fuse Q/K/V(+beta) GEMMs when possible (default on)\n"
                "  --no_fused_qkv     disable fused Q/K/V path\n"
                "  --qkv_split_kernel use fused QKV split kernel when available (default on)\n"
                "  --no_qkv_split_kernel use memcpy splits for fused QKV path\n"
                "  --qkv_split_vec4  use float4 stores in QKV split kernel\n"
                "  --no_qkv_split_vec4 disable float4 stores in QKV split kernel\n"
                "  --qkv_split_vec4_auto auto-select vec4 vs scalar split (default)\n"
                "  --qkv_split_bias fuse beta bias into split kernel (default off)\n"
                "  --no_qkv_split_bias do not fuse beta bias into split kernel\n"
                "  --fused_ff1        fuse FFN GEMM + GELU + quantize when possible (default on)\n"
                "  --no_fused_ff1     disable fused FFN GEMM path\n"
                "  --fused_head_loss  fuse head GEMM + loss (dp4a, vocab<=256, default off)\n"
                "  --no_fused_head_loss disable fused head GEMM + loss\n"
                "  --fused_noise_gemm fuse GEMM + noise add for float outputs (default on)\n"
                "  --no_fused_noise_gemm disable fused noise GEMM path\n"
                "  --noise_batch     batch GPU noise fills in one launch (default on)\n"
                "  --no_noise_batch  disable batched noise fill\n"
                "  --pinned_tokens   stage token batches in pinned host memory (default on)\n"
                "  --no_pinned_tokens disable pinned token staging\n"
                "  --gemm_backend {dp4a|cutlass|nvfp4} (default nvfp4)\n"
                "  --cutlass_gemm   use CUTLASS GEMM (unpacked only)\n"
                "  --cutlass_nvfp4  use CUTLASS NVFP4 block-scaled GEMM (sm120+, hidden%128==0)\n"
                "  --nvfp4_schedule {auto|cooperative|pingpong} (default pingpong)\n"
                "  --nvfp4_quant {warp16|warp4} (default warp16)\n"
                "  --nvfp4_stages {auto|2|3|4} (default 2)\n"
                "  --nvfp4_decomp {auto|data|splitk|streamk} (default splitk)\n"
                "  --nvfp4_splits N  split-K factor for nvfp4 (default 4)\n"
                "  --nvfp4_autotune enable nvfp4 tuning for unset options (default off)\n"
                "  --no_nvfp4_autotune disable nvfp4 tuning\n"
                "  --efla_fused      use single-kernel EFLA step (default off)\n"
                "  --no_efla_fused   disable fused EFLA step\n"
                "  --efla_mixed      store k_usage/diff in FP16 during EFLA step (default off)\n"
                "  --no_efla_mixed   disable mixed-precision EFLA step\n"
                "  --efla_fuse_diff  fuse diff into EFLA update (skip diff kernel, default off)\n"
                "  --no_efla_fuse_diff disable fused diff update\n"
                "  --efla_update_wmma use tensor-core outer-product update for EFLA (FP16 state)\n"
                "  --no_efla_update_wmma disable tensor-core EFLA update (default)\n"
                "  --tiled_gemm       use tiled GEMM kernels (experimental, default off)\n"
                "  --no_tiled_gemm    disable tiled GEMM kernels\n"
                "  --packed_gemm      use int2-packed GEMM (default off)\n"
                "  --no_packed_gemm   disable packed GEMM\n"
                "  --packed_gemm_bitnet use BitNet-style interleaved int2 pack/decode (default off)\n"
                "  --no_packed_gemm_bitnet disable BitNet-style packed GEMM\n"
                "  --packed_update   update packed weights in-place during GPU update (default off)\n"
                "  --no_packed_update disable in-place packed update\n"
                "  --clt/--no_clt     use CLT/approx-Gaussian noise (default on)\n"
                "  --clt_k K          CLT sum terms (default 4)\n"
                "  --omp_threads N    set OpenMP thread count (default 0 = auto/all cores)\n"
                "  --thresh T         update threshold (default 0.008)\n"
                "  --thresh_end T     end threshold (default 0.008)\n"
                "  --thresh_schedule {constant|linear|cosine|exp} (default exp)\n"
                "  --lr LR            optimizer lr / step size (default 0.80)\n"
                "  --lr_end LR        end lr (default 0.05)\n"
                "  --lr_schedule {constant|linear|cosine|exp} (default cosine)\n"
                "  --fitness {sign|zscore|centered_rank}\n"
                "  --fitness_clip C   clip for zscore/rank (default 3)\n"
                "  --shadow/--no_shadow    float shadow weights (default on)\n"
                "  --momentum/--no_momentum (shadow must be off)\n"
                "  --momentum_beta B  momentum beta\n"
                "  --adam/--no_adam    Adam-style (shadow must be off)\n"
                "  --adam_beta2 B2     Adam beta2\n"
                "  --adam_eps E        Adam eps\n"
                "  --adaptive/--no_adaptive adaptive RMS scaling (default on)\n"
                "  --adaptive_beta B   adaptive ema beta\n"
                "  --adaptive_eps E    adaptive eps\n"
                "  --act_scale S       scale after GELU (default 0.8)\n"
                "  --mlp_act_scale S   scale after GELU in MLP (default 1)\n"
                "  --ln_scale S        scale after LayerNorm (default 1)\n"
                "  --residual_scale S  residual scale (L==1: scales x; L>=2: scales attn) (default 1)\n"
                "  --mlp_residual_scale S residual scale for MLP (default 1)\n"
                "  --state_decay D    low-rank state decay (default 0.95)\n"
                "  --gate_scale G     low-rank gate scale (default 0.1)\n"
                "  --int8_residual    keep residual stream in int8 (default on)\n"
                "  --no_int8_residual keep residual stream in float\n"
                "  --absmean_norm     use abs-mean norm on int8 (requires int8_residual) (default on)\n"
                "  --no_absmean_norm  use LayerNorm\n"
                "  --no_val            disable val eval\n"
                "  --val_seed S        fixed val seed\n"
                "  --val_every N       eval val every N epochs\n"
                "  --fixed_train       use a fixed training batch (for overfitting/debug)\n"
                "  --data_seed S       seed for data sampling (defaults to --seed)\n"
                "  --seed S           RNG seed\n";
            return 0;
        }
    }

    if (cfg.population <= 0 || (cfg.population % 2) != 0) {
        std::cerr << "Error: --pop must be positive and even\n";
        return 1;
    }
    if (cfg.use_packed_gemm_bitnet) {
        cfg.use_packed_gemm = true;
    }
    if (cfg.use_packed_update && !cfg.use_packed_gemm) {
        std::cerr << "Note: --packed_update requires --packed_gemm; disabling packed_update\n";
        cfg.use_packed_update = false;
    }
    if (cfg.use_packed_update && !cfg.use_gpu_update) {
        std::cerr << "Note: --packed_update requires --gpu_update; disabling packed_update\n";
        cfg.use_packed_update = false;
    }
#if !defined(BITNET_EGGROLL_HAS_CUTLASS)
    if (cfg.gemm_backend == TrainConfig::GemmBackend::Cutlass ||
        cfg.gemm_backend == TrainConfig::GemmBackend::CutlassNvfp4) {
        std::cerr << "Note: CUTLASS not available; falling back to dp4a GEMM\n";
        cfg.gemm_backend = TrainConfig::GemmBackend::Dp4a;
    }
#endif
    if (cfg.vocab != 256) {
        std::cerr << "Error: vocab must be 256 for byte-level LM\n";
        return 1;
    }
    if (cfg.hidden <= 0 || cfg.seq_len <= 0 || cfg.batch <= 0) {
        std::cerr << "Error: invalid hidden/seq/batch\n";
        return 1;
    }
    if (cfg.gemm_backend == TrainConfig::GemmBackend::CutlassNvfp4) {
        if ((cfg.hidden & 127) != 0) {
            const int aligned = (cfg.hidden + 127) & ~127;
            std::cerr << "Error: nvfp4 GEMM requires --hidden multiple of 128 (got "
                      << cfg.hidden << "). Use --hidden " << aligned
                      << " or switch --gemm_backend.\n";
            return 1;
        }
    }
    if (cfg.state_dim < 0) {
        std::cerr << "Error: --state_dim must be >= 0\n";
        return 1;
    }
    if (cfg.state_dim > 0) {
        if (cfg.hidden % cfg.state_dim != 0) {
            std::cerr << "Error: --state_dim must divide --hidden for low-rank state\n";
            return 1;
        }
        if (cfg.state_fp16) {
            std::cerr << "Note: --state_fp16 not yet supported with --state_dim; using FP32 state\n";
            cfg.state_fp16 = false;
        }
    }
    if (cfg.mlp_mult <= 0) {
        std::cerr << "Error: --mlp_mult must be positive\n";
        return 1;
    }
    if (cfg.layers <= 0) {
        std::cerr << "Error: --layers must be positive\n";
        return 1;
    }
    if (model_8m_requested) {
        if (!lr_specified) cfg.lr = 0.80f;
        if (!lr_end_specified) cfg.lr_end = 0.05f;
        if (!lr_schedule_specified) cfg.lr_schedule = TrainConfig::Schedule::Cosine;
        if (!thresh_specified) cfg.update_threshold = 0.008f;
        if (!sigma_specified) cfg.sigma_shift = 3;
        if (!act_scale_specified) cfg.act_scale = 0.8f;
        if (!mlp_act_scale_specified) cfg.mlp_act_scale = 1.0f;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        std::cerr << "Error: no CUDA devices found\n";
        return 1;
    }
    if (!devices_specified && !device_specified) {
        cfg.devices.clear();
        cfg.devices.reserve(device_count);
        for (int dev = 0; dev < device_count; ++dev) {
            cfg.devices.push_back(dev);
        }
    }

    std::vector<int> devices = cfg.devices.empty() ? std::vector<int>{cfg.device} : cfg.devices;
    if (devices.empty()) {
        std::cerr << "Error: no devices specified\n";
        return 1;
    }
    std::vector<int> devices_sorted = devices;
    std::sort(devices_sorted.begin(), devices_sorted.end());
    devices_sorted.erase(std::unique(devices_sorted.begin(), devices_sorted.end()), devices_sorted.end());
    if (devices_sorted.size() != devices.size()) {
        std::cerr << "Error: --devices contains duplicates\n";
        return 1;
    }
    for (int dev : devices) {
        if (dev < 0 || dev >= device_count) {
            std::cerr << "Error: invalid device index " << dev << " (device_count=" << device_count << ")\n";
            return 1;
        }
    }
    if (cfg.use_shadow && (cfg.use_momentum || cfg.use_adam)) {
        std::cerr << "Note: --shadow overrides --momentum/--adam in this prototype\n";
    }
    if (cfg.use_adam && cfg.lr <= 0.0f) {
        std::cerr << "Warning: --lr <= 0; Adam updates will be no-ops\n";
    }
    if (cfg.absmean_norm && !cfg.int8_residual) {
        std::cerr << "Error: --absmean_norm requires --int8_residual\n";
        return 1;
    }
    if (cfg.use_gpu_update && !cfg.use_gpu_noise) {
        std::cerr << "Error: --gpu_update requires --gpu_noise (CPU noise not supported)\n";
        return 1;
    }
    if (cfg.use_gpu_update && !cfg.use_shadow) {
        std::cerr << "Error: --gpu_update currently requires --shadow\n";
        return 1;
    }
    if (cfg.data_seed == 0) cfg.data_seed = cfg.seed;
    if (cfg.gemm_backend != TrainConfig::GemmBackend::CutlassNvfp4) {
        cfg.nvfp4_autotune = false;
    }
    const bool nvfp4_backend = (cfg.gemm_backend == TrainConfig::GemmBackend::CutlassNvfp4);
    if (nvfp4_backend && cfg.hidden >= 512) {
        bool set_decomp = false;
        bool set_splits = false;
        bool set_autotune = false;
        bool set_graph = false;
        if (!nvfp4_decomp_specified &&
            cfg.nvfp4_decomp != bitnet_cuda::Nvfp4Decomposition::SplitK) {
            cfg.nvfp4_decomp = bitnet_cuda::Nvfp4Decomposition::SplitK;
            set_decomp = true;
        }
        if (!nvfp4_splits_specified && cfg.nvfp4_splits != 4) {
            cfg.nvfp4_splits = 4;
            set_splits = true;
        }
        if (!nvfp4_autotune_specified && cfg.nvfp4_autotune) {
            cfg.nvfp4_autotune = false;
            set_autotune = true;
        }
        if (!cuda_graph_specified && cfg.use_cuda_graph) {
            cfg.use_cuda_graph = false;
            set_graph = true;
        }
        if (set_decomp || set_splits || set_autotune || set_graph) {
            std::cerr << "Note: nvfp4 defaults for hidden>=512:";
            bool first = true;
            if (set_decomp) {
                std::cerr << (first ? " " : ", ") << "decomp=splitk";
                first = false;
            }
            if (set_splits) {
                std::cerr << (first ? " " : ", ") << "splits=4";
                first = false;
            }
            if (set_graph) {
                std::cerr << (first ? " " : ", ") << "cuda_graph=off";
                first = false;
            }
            if (set_autotune) {
                std::cerr << (first ? " " : ", ") << "autotune=off";
                first = false;
            }
            std::cerr << ". Override with --nvfp4_*";
            if (set_graph) {
                std::cerr << " or --cuda_graph";
            }
            std::cerr << ".\n";
        }
    }
    if (cfg.method == TrainConfig::Method::EFLA && cfg.state_dim == 0 && cfg.hidden >= 512) {
        if (!efla_fused_specified && cfg.use_fused_efla) {
            cfg.use_fused_efla = false;
            std::cerr << "Note: disabling fused EFLA step for hidden>=512; override with --efla_fused.\n";
        }
    }
    if (cfg.nvfp4_autotune) {
        cfg.nvfp4_tune_schedule = !nvfp4_schedule_specified;
        cfg.nvfp4_tune_quant = !nvfp4_quant_specified;
        cfg.nvfp4_tune_stages = !nvfp4_stage_specified;
        cfg.nvfp4_tune_decomp = !nvfp4_decomp_specified;
        cfg.nvfp4_tune_splits = !nvfp4_splits_specified;
    } else {
        cfg.nvfp4_tune_schedule = false;
        cfg.nvfp4_tune_quant = false;
        cfg.nvfp4_tune_stages = false;
        cfg.nvfp4_tune_decomp = false;
        cfg.nvfp4_tune_splits = false;
    }

    configure_openmp(cfg.omp_threads);

    {
        const uint64_t params = count_parameters(cfg);
        const double params_m = static_cast<double>(params) / 1e6;
        const uint64_t h = static_cast<uint64_t>(cfg.hidden);
        const uint64_t d = static_cast<uint64_t>(std::max(0, cfg.state_dim));
        const uint64_t elem_bytes = cfg.state_fp16 ? 2ULL : 4ULL;
        uint64_t state_bytes = 0;
        const char* label = "state_S";
        if (cfg.state_dim > 0) {
            state_bytes = 2ULL * static_cast<uint64_t>(cfg.layers) *
                          static_cast<uint64_t>(cfg.batch) * h * d * elem_bytes;
            label = "state_KV";
        } else {
            state_bytes = static_cast<uint64_t>(cfg.layers) *
                          static_cast<uint64_t>(cfg.batch) * h * h * elem_bytes;
        }
        const double state_mb = static_cast<double>(state_bytes) / (1024.0 * 1024.0);
        std::cout << "model params " << params << " (~" << params_m << "M)"
                  << "  " << label << " " << state_mb << " MiB\n";
    }

    ByteDataset data;
    if (!data.load_file(cfg.data_path)) return 1;

    EflaLmWeights w;
    w.emb.init_rand(cfg.vocab, cfg.hidden, 0x1000ULL);
    w.win.init_rand(cfg.hidden, cfg.hidden, 0x2000ULL);
    w.wq.resize(static_cast<size_t>(cfg.layers));
    w.wk.resize(static_cast<size_t>(cfg.layers));
    w.wv.resize(static_cast<size_t>(cfg.layers));
    w.wbeta.resize(static_cast<size_t>(cfg.layers));
    w.ff1.resize(static_cast<size_t>(cfg.layers));
    w.ff2.resize(static_cast<size_t>(cfg.layers));
    const int ffn_dim = cfg.hidden * std::max(1, cfg.mlp_mult);
    for (int l = 0; l < cfg.layers; ++l) {
        uint64_t base = 0x3000ULL + static_cast<uint64_t>(l) * 8;
        w.wq[static_cast<size_t>(l)].init_rand(cfg.hidden, cfg.hidden, base + 0);
        w.wk[static_cast<size_t>(l)].init_rand(cfg.hidden, cfg.hidden, base + 1);
        w.wv[static_cast<size_t>(l)].init_rand(cfg.hidden, cfg.hidden, base + 2);
        w.wbeta[static_cast<size_t>(l)].init_rand(1, cfg.hidden, base + 3);
        w.ff1[static_cast<size_t>(l)].init_rand(ffn_dim, cfg.hidden, base + 4);
        w.ff2[static_cast<size_t>(l)].init_rand(cfg.hidden, ffn_dim, base + 5);
    }
    w.head.init_rand(cfg.vocab, cfg.hidden, 0x7000ULL);
    w.head_bias.assign(static_cast<size_t>(cfg.vocab), 0.0f);
    w.beta_bias.assign(static_cast<size_t>(cfg.layers), -2.0f);

    // Fixed positional embedding.
    std::vector<float> pos(static_cast<size_t>(cfg.seq_len) * cfg.hidden, 0.0f);
    {
        uint64_t st = 0xABCDEFULL;
        for (size_t i = 0; i < pos.size(); ++i) {
            // Uniform-ish in [-0.02, 0.02].
            const float u = static_cast<float>(rng::splitmix64(st) & 0xFFFF) / 65535.0f;
            pos[i] = (u - 0.5f) * 0.04f;
        }
    }

    try {
        if (devices.size() > 1) {
            std::cout << "Using devices:";
            for (int dev : devices) std::cout << " " << dev;
            std::cout << "\n";
        }
        if (cfg.gpu_workers > 1) {
            std::cout << "Workers per device: " << cfg.gpu_workers << "\n";
        }
        if (cfg.update_streams > 1) {
            std::cout << "Update streams per device: " << cfg.update_streams << "\n";
        }
        std::vector<std::shared_ptr<DeviceWeights>> device_weights;
        device_weights.reserve(devices.size());
        std::vector<std::vector<size_t>> device_model_indices(devices.size());
        std::vector<cudaEvent_t> update_done_events(devices.size(), nullptr);
        for (size_t di = 0; di < devices.size(); ++di) {
            const int dev = devices[di];
            cuda_check(cudaSetDevice(dev), "cudaSetDevice init weights");
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, dev);
            const bool device_nvfp4 = (cfg.gemm_backend == TrainConfig::GemmBackend::CutlassNvfp4) && (prop.major >= 12);
            bool device_packed = cfg.use_packed_gemm;
            if (device_nvfp4 && device_packed) {
                device_packed = false;
            }
            const bool device_fused_qkv_nvfp4 = device_nvfp4 && cfg.use_fused_qkv;
            auto wdev = std::make_shared<DeviceWeights>(dev, cfg.vocab, cfg.hidden, cfg.layers, cfg.mlp_mult,
                                                        cfg.batch, device_packed, device_nvfp4, device_fused_qkv_nvfp4);
            wdev->win_out_scale = w.win.out_scale;
            wdev->q_out_scale = w.wq[0].out_scale;
            wdev->k_out_scale = w.wk[0].out_scale;
            wdev->v_out_scale = w.wv[0].out_scale;
            wdev->beta_out_scale = w.wbeta[0].out_scale;
            wdev->ff1_out_scale = w.ff1[0].out_scale;
            wdev->ff2_out_scale = w.ff2[0].out_scale;
            wdev->head_out_scale = w.head.out_scale;
            device_weights.push_back(std::move(wdev));
            cuda_check(cudaEventCreateWithFlags(&update_done_events[di], cudaEventDisableTiming),
                       "cudaEventCreate update_done");
        }

        std::vector<std::unique_ptr<CudaEflaLm>> models;
        const int workers_per_device = std::max(1, cfg.gpu_workers);
        models.reserve(devices.size() * static_cast<size_t>(workers_per_device));
        for (size_t di = 0; di < devices.size(); ++di) {
            const int dev = devices[di];
            for (int wi = 0; wi < workers_per_device; ++wi) {
                models.emplace_back(std::make_unique<CudaEflaLm>(dev, cfg.vocab, cfg.hidden, cfg.layers, cfg.mlp_mult,
                                                                 cfg.batch, cfg.seq_len, cfg.state_dim, cfg.state_fp16,
                                                                 cfg.use_cuda_graph, cfg.graph_full_eval, cfg.use_nvtx,
                                                                 cfg.use_fused_qkv, cfg.use_qkv_split_kernel,
                                                                 cfg.qkv_split_vec4, cfg.use_qkv_split_bias,
                                                                 cfg.use_fused_ff1,
                                                                 cfg.use_fused_head_loss,
                                                                 cfg.use_fused_noise_gemm,
                                                                 cfg.use_fused_efla,
                                                                 cfg.use_efla_mixed,
                                                                 cfg.use_efla_fuse_diff,
                                                                 cfg.use_efla_update_wmma,
                                                                 cfg.use_tiled_gemm,
                                                                 cfg.use_packed_gemm,
                                                                 cfg.use_packed_gemm_bitnet,
                                                                 cfg.gemm_backend,
                                                                 cfg.nvfp4_schedule,
                                                                 cfg.nvfp4_quant_mode,
                                                                 cfg.nvfp4_stage_count,
                                                                 cfg.nvfp4_decomp,
                                                                 cfg.nvfp4_splits,
                                                                 cfg.nvfp4_autotune,
                                                                 cfg.nvfp4_tune_schedule,
                                                                 cfg.nvfp4_tune_quant,
                                                                 cfg.nvfp4_tune_stages,
                                                                 cfg.nvfp4_tune_decomp,
                                                                 cfg.nvfp4_tune_splits,
                                                                 cfg.noise_batch,
                                                                 device_weights[di]));
                device_model_indices[di].push_back(models.size() - 1);
                cuda_check(cudaSetDevice(dev), "cudaSetDevice init");
                models.back()->set_pos_embedding(pos);
                if (device_model_indices[di].size() == 1) {
                    models.back()->sync_base_weights(w);
                }
            }
        }
        for (size_t di = 0; di < device_model_indices.size(); ++di) {
            if (device_model_indices[di].empty()) continue;
            const size_t leader_idx = device_model_indices[di][0];
            cuda_check(cudaSetDevice(models[leader_idx]->device()), "cudaSetDevice init_wait");
            cuda_check(cudaEventRecord(update_done_events[di], models[leader_idx]->stream()),
                       "cudaEventRecord init_weights");
            for (size_t mi : device_model_indices[di]) {
                cuda_check(cudaStreamWaitEvent(models[mi]->stream(), update_done_events[di], 0),
                           "cudaStreamWaitEvent init_weights");
            }
        }
        CudaEflaLm& model = *models.front();
        const size_t pool_threads = std::max<size_t>(1, std::max(models.size(), devices.size()) + 1);
        ThreadPool pool(pool_threads);

        const int max_pairs = cfg.population / 2;  // max for allocations
        const int method_id = static_cast<int>(cfg.method);

        using Clock = std::chrono::steady_clock;
        auto seconds_since = [](Clock::time_point start) -> double {
            return std::chrono::duration<double>(Clock::now() - start).count();
        };

        // Helper to compute adaptive population pairs based on epoch progress
        // Conservative schedule: 50% -> 75% -> 100% to preserve model quality
        auto compute_adaptive_pairs = [&](int epoch) -> int {
            if (!cfg.adaptive_pop) return max_pairs;
            const float progress = (cfg.epochs <= 1) ? 1.0f
                : (static_cast<float>(epoch) / static_cast<float>(cfg.epochs - 1));
            int p;
            if (progress < 0.15f) {
                p = max_pairs / 2;  // 50% population for first 15% of epochs
            } else if (progress < 0.35f) {
                p = (max_pairs * 3) / 4;  // 75% population until 35% of epochs
            } else {
                p = max_pairs;      // full population thereafter
            }
            // Ensure even and at least 2
            p = std::max(2, (p / 2) * 2);
            return p;
        };

        struct EvalScratch {
            std::vector<uint64_t> seed_wq;
            std::vector<uint64_t> seed_wk;
            std::vector<uint64_t> seed_wv;
            std::vector<uint64_t> seed_wbeta;
            std::vector<uint64_t> seed_ff1;
            std::vector<uint64_t> seed_ff2;
            std::vector<float> beta_bias_noise;
            std::vector<float> pos_noise;
            std::vector<float> pos_noisy;
            std::vector<float> head_bias_noise;
            EflaLmWeights noise;
        };

        std::vector<EvalScratch> eval_scratch(models.size());
        std::vector<std::pair<int, int>> eval_ranges(models.size(), {0, 0});
        // Helper to recompute eval_ranges based on current pairs count
        auto recompute_eval_ranges = [&](int pairs) {
            if (pairs > 0 && !models.empty()) {
                const int num_workers = static_cast<int>(models.size());
                const int pairs_per_worker = (pairs + num_workers - 1) / num_workers;
                for (int wi = 0; wi < num_workers; ++wi) {
                    const int start = wi * pairs_per_worker;
                    const int end = std::min(pairs, start + pairs_per_worker);
                    eval_ranges[static_cast<size_t>(wi)] = {start, end};
                }
            }
        };
        recompute_eval_ranges(max_pairs);  // initial setup for allocations
        for (auto& scratch : eval_scratch) {
            scratch.seed_wq.resize(static_cast<size_t>(cfg.layers));
            scratch.seed_wk.resize(static_cast<size_t>(cfg.layers));
            scratch.seed_wv.resize(static_cast<size_t>(cfg.layers));
            scratch.seed_wbeta.resize(static_cast<size_t>(cfg.layers));
            scratch.seed_ff1.resize(static_cast<size_t>(cfg.layers));
            scratch.seed_ff2.resize(static_cast<size_t>(cfg.layers));
            scratch.beta_bias_noise.resize(static_cast<size_t>(cfg.layers));
            scratch.head_bias_noise.resize(static_cast<size_t>(cfg.vocab));
            if (cfg.train_pos) {
                scratch.pos_noise.resize(pos.size());
                scratch.pos_noisy.resize(pos.size());
            }
            if (!cfg.use_gpu_noise) {
                scratch.noise.wq.resize(static_cast<size_t>(cfg.layers));
                scratch.noise.wk.resize(static_cast<size_t>(cfg.layers));
                scratch.noise.wv.resize(static_cast<size_t>(cfg.layers));
                scratch.noise.wbeta.resize(static_cast<size_t>(cfg.layers));
                scratch.noise.ff1.resize(static_cast<size_t>(cfg.layers));
                scratch.noise.ff2.resize(static_cast<size_t>(cfg.layers));
            }
        }

        std::vector<uint64_t> layer_salt_wq(static_cast<size_t>(cfg.layers));
        std::vector<uint64_t> layer_salt_wk(static_cast<size_t>(cfg.layers));
        std::vector<uint64_t> layer_salt_wv(static_cast<size_t>(cfg.layers));
        std::vector<uint64_t> layer_salt_wbeta(static_cast<size_t>(cfg.layers));
        std::vector<uint64_t> layer_salt_ff1(static_cast<size_t>(cfg.layers));
        std::vector<uint64_t> layer_salt_ff2(static_cast<size_t>(cfg.layers));
        for (int l = 0; l < cfg.layers; ++l) {
            const uint64_t lu = static_cast<uint64_t>(l);
            layer_salt_wq[static_cast<size_t>(l)] = rng::mix(lu, kSaltWq);
            layer_salt_wk[static_cast<size_t>(l)] = rng::mix(lu, kSaltWk);
            layer_salt_wv[static_cast<size_t>(l)] = rng::mix(lu, kSaltWv);
            layer_salt_wbeta[static_cast<size_t>(l)] = rng::mix(lu, kSaltWbeta);
            layer_salt_ff1[static_cast<size_t>(l)] = rng::mix(lu, kSaltFf1);
            layer_salt_ff2[static_cast<size_t>(l)] = rng::mix(lu, kSaltFf2);
        }

        std::vector<float> weights(static_cast<size_t>(max_pairs), 0.0f);

        float* h_pair_weights = nullptr;
        if (cfg.use_gpu_update && max_pairs > 0) {
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&h_pair_weights),
                                     static_cast<size_t>(max_pairs) * sizeof(float),
                                     cudaHostAllocDefault),
                       "cudaHostAlloc pair_weights");
        }

        MatrixOptState st_emb, st_win, st_head;
        std::vector<MatrixOptState> st_wq(static_cast<size_t>(cfg.layers));
        std::vector<MatrixOptState> st_wk(static_cast<size_t>(cfg.layers));
        std::vector<MatrixOptState> st_wv(static_cast<size_t>(cfg.layers));
        std::vector<MatrixOptState> st_wbeta(static_cast<size_t>(cfg.layers));
        std::vector<MatrixOptState> st_ff1(static_cast<size_t>(cfg.layers));
        std::vector<MatrixOptState> st_ff2(static_cast<size_t>(cfg.layers));
        VectorOptState st_head_bias;
        VectorOptState st_beta_bias;
        VectorOptState st_pos;

        std::vector<GpuUpdateDevice> gpu_updates;

        const size_t emb_size = static_cast<size_t>(cfg.vocab) * cfg.hidden;
        const size_t hh_size = static_cast<size_t>(cfg.hidden) * cfg.hidden;
        const size_t hf_size = static_cast<size_t>(cfg.hidden) * static_cast<size_t>(cfg.hidden * std::max(1, cfg.mlp_mult));
        const size_t h_size = static_cast<size_t>(cfg.hidden);
        const size_t head_size = emb_size;
        const int pack_cols_h = (cfg.hidden + 15) >> 4;
        const int pack_cols_f = ((cfg.hidden * std::max(1, cfg.mlp_mult)) + 15) >> 4;
        const size_t hh_pack = static_cast<size_t>(cfg.hidden) * static_cast<size_t>(pack_cols_h);
        const size_t hf1_pack = static_cast<size_t>(cfg.hidden * std::max(1, cfg.mlp_mult)) * static_cast<size_t>(pack_cols_h);
        const size_t hf2_pack = static_cast<size_t>(cfg.hidden) * static_cast<size_t>(pack_cols_f);
        const size_t h_pack = static_cast<size_t>(pack_cols_h);
        const size_t head_pack = static_cast<size_t>(cfg.vocab) * static_cast<size_t>(pack_cols_h);

        if (cfg.use_gpu_update) {
            size_t max_n = emb_size;
            max_n = std::max(max_n, hh_size);
            max_n = std::max(max_n, hf_size);
            max_n = std::max(max_n, h_size);
            max_n = std::max(max_n, head_size);
            gpu_updates.resize(devices.size());
            for (size_t di = 0; di < devices.size(); ++di) {
                if (device_model_indices[di].empty()) continue;
                const size_t leader_idx = device_model_indices[di][0];
                GpuUpdateDevice& gu = gpu_updates[di];
                gu.wq.resize(static_cast<size_t>(cfg.layers));
                gu.wk.resize(static_cast<size_t>(cfg.layers));
                gu.wv.resize(static_cast<size_t>(cfg.layers));
                gu.wbeta.resize(static_cast<size_t>(cfg.layers));
                gu.ff1.resize(static_cast<size_t>(cfg.layers));
                gu.ff2.resize(static_cast<size_t>(cfg.layers));

                CudaEflaLm& leader = *models[leader_idx];
                const int available_streams = static_cast<int>(device_model_indices[di].size());
                gu.stream_count = std::max(1, std::min(cfg.update_streams, available_streams));
                gu.bufs.resize(static_cast<size_t>(gu.stream_count));
                gu.stream_events.resize(static_cast<size_t>(gu.stream_count), nullptr);
                for (int si = 0; si < gu.stream_count; ++si) {
                    gpu_update_ensure_buffers(leader, gu.bufs[static_cast<size_t>(si)], max_n, max_pairs);
                    cuda_check(cudaSetDevice(leader.device()), "cudaSetDevice update_stream_event");
                    cuda_check(cudaEventCreateWithFlags(&gu.stream_events[static_cast<size_t>(si)],
                                                        cudaEventDisableTiming),
                               "cudaEventCreate update_stream_event");
                }
                gpu_update_alloc_shadow(leader, gu.emb, emb_size);
                gpu_update_alloc_shadow(leader, gu.win, hh_size);
                gpu_update_alloc_shadow(leader, gu.head, head_size);
                for (int l = 0; l < cfg.layers; ++l) {
                    gpu_update_alloc_shadow(leader, gu.wq[static_cast<size_t>(l)], hh_size);
                    gpu_update_alloc_shadow(leader, gu.wk[static_cast<size_t>(l)], hh_size);
                    gpu_update_alloc_shadow(leader, gu.wv[static_cast<size_t>(l)], hh_size);
                    gpu_update_alloc_shadow(leader, gu.wbeta[static_cast<size_t>(l)], h_size);
                    gpu_update_alloc_shadow(leader, gu.ff1[static_cast<size_t>(l)], hf_size);
                    gpu_update_alloc_shadow(leader, gu.ff2[static_cast<size_t>(l)], hf_size);
                }
            }
        }

        std::vector<std::vector<uint8_t>> inputs_cur, targets_cur;
        std::vector<std::vector<uint8_t>> inputs_next, targets_next;
        constexpr int kTokenBuffers = 2;
        uint8_t* h_inputs_pinned[kTokenBuffers] = {nullptr, nullptr};
        uint8_t* h_targets_pinned[kTokenBuffers] = {nullptr, nullptr};
        bool use_pinned_tokens = cfg.use_pinned_tokens;
        if (use_pinned_tokens) {
            cuda_check(cudaSetDevice(models[0]->device()), "cudaSetDevice pinned_tokens_alloc");
            const size_t host_bytes = static_cast<size_t>(cfg.batch) * cfg.seq_len;
            for (int i = 0; i < kTokenBuffers; ++i) {
                cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&h_inputs_pinned[i]),
                                                host_bytes, cudaHostAllocPortable);
                if (err != cudaSuccess) {
                    use_pinned_tokens = false;
                    break;
                }
                err = cudaHostAlloc(reinterpret_cast<void**>(&h_targets_pinned[i]),
                                    host_bytes, cudaHostAllocPortable);
                if (err != cudaSuccess) {
                    use_pinned_tokens = false;
                    break;
                }
            }
            if (!use_pinned_tokens) {
                for (int i = 0; i < kTokenBuffers; ++i) {
                    if (h_inputs_pinned[i]) {
                        cudaFreeHost(h_inputs_pinned[i]);
                        h_inputs_pinned[i] = nullptr;
                    }
                    if (h_targets_pinned[i]) {
                        cudaFreeHost(h_targets_pinned[i]);
                        h_targets_pinned[i] = nullptr;
                    }
                }
                std::cerr << "Pinned token staging disabled (cudaHostAlloc failed)\n";
            }
        }
        double cur_sample_s = 0.0;
        int buf_cur = 0;
        int buf_next = 1;
        bool cur_uploaded = false;
        bool next_uploaded = false;
        {
            NvtxRange range_sample(cfg.use_nvtx, "sample_batch");
            const auto t0 = Clock::now();
            const uint64_t seed0 = cfg.fixed_train
                                       ? cfg.data_seed
                                       : rng::mix(cfg.data_seed, static_cast<uint64_t>(0));
            if (use_pinned_tokens) {
                data.sample_batch_packed(cfg.batch, cfg.seq_len, seed0,
                                         h_inputs_pinned[buf_cur], h_targets_pinned[buf_cur]);
            } else {
                data.sample_batch(cfg.batch, cfg.seq_len, seed0, inputs_cur, targets_cur);
            }
            cur_sample_s = seconds_since(t0);
        }

        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            const auto epoch_start = Clock::now();
            double sample_s = (cfg.fixed_train && epoch > 0) ? 0.0 : cur_sample_s;
            double upload_s = 0.0;
            double eval_s = 0.0;
            double update_s = 0.0;
            double train_s = 0.0;
            double val_s = 0.0;
            const float t01 = (cfg.epochs <= 1) ? 1.0f : (static_cast<float>(epoch) / static_cast<float>(cfg.epochs - 1));
            const float lr_t = schedule_value(cfg.lr, cfg.lr_end, cfg.lr_schedule, t01);
            const float thresh_t = schedule_value(cfg.update_threshold, cfg.update_threshold_end, cfg.thresh_schedule, t01);
            int sigma_t = static_cast<int>(std::lrint(schedule_value(
                static_cast<float>(cfg.sigma_shift),
                static_cast<float>(cfg.sigma_shift_end),
                cfg.sigma_schedule,
                t01)));
            if (sigma_t < 0) sigma_t = 0;
            const float noise_scale = 1.0f / static_cast<float>(1 << sigma_t);

            // Compute current pairs for adaptive population
            const int current_pairs = compute_adaptive_pairs(epoch);
            recompute_eval_ranges(current_pairs);

            double next_sample_s = 0.0;
            std::future<void> prefetch_future;
            bool prefetch_active = false;
            next_uploaded = false;
            if (!cfg.fixed_train && (epoch + 1) < cfg.epochs) {
                const uint64_t next_seed = rng::mix(cfg.data_seed, static_cast<uint64_t>(epoch + 1));
                const int buf_upload = buf_next;
                prefetch_active = true;
                prefetch_future = pool.enqueue([&, buf_upload, next_seed]() {
                    const auto t0 = Clock::now();
                    if (use_pinned_tokens) {
                        data.sample_batch_packed(cfg.batch, cfg.seq_len, next_seed,
                                                 h_inputs_pinned[buf_upload], h_targets_pinned[buf_upload]);
                    } else {
                        data.sample_batch(cfg.batch, cfg.seq_len, next_seed, inputs_next, targets_next);
                    }
                    next_sample_s = seconds_since(t0);
                    for (size_t dev_idx = 0; dev_idx < device_model_indices.size(); ++dev_idx) {
                        if (device_model_indices[dev_idx].empty()) continue;
                        const size_t leader_idx = device_model_indices[dev_idx][0];
                        cuda_check(cudaSetDevice(models[leader_idx]->device()), "cudaSetDevice prefetch_upload");
                        for (size_t mi : device_model_indices[dev_idx]) {
                            if (use_pinned_tokens) {
                                models[mi]->upload_tokens_async_packed(h_inputs_pinned[buf_upload],
                                                                       h_targets_pinned[buf_upload],
                                                                       buf_upload);
                            } else {
                                models[mi]->upload_tokens_async(inputs_next, targets_next, buf_upload);
                            }
                        }
                    }
                    next_uploaded = true;
                });
            }

            {
                NvtxRange range_upload(cfg.use_nvtx, "upload_tokens");
                const auto t0 = Clock::now();
                if (!cur_uploaded) {
                    for (size_t dev_idx = 0; dev_idx < device_model_indices.size(); ++dev_idx) {
                        if (device_model_indices[dev_idx].empty()) continue;
                        const size_t leader_idx = device_model_indices[dev_idx][0];
                        cuda_check(cudaSetDevice(models[leader_idx]->device()), "cudaSetDevice upload_tokens");
                        for (size_t mi : device_model_indices[dev_idx]) {
                            if (use_pinned_tokens) {
                                models[mi]->upload_tokens_async_packed(h_inputs_pinned[buf_cur],
                                                                       h_targets_pinned[buf_cur],
                                                                       buf_cur);
                            } else {
                                models[mi]->upload_tokens_async(inputs_cur, targets_cur, buf_cur);
                            }
                        }
                    }
                }
                for (size_t dev_idx = 0; dev_idx < device_model_indices.size(); ++dev_idx) {
                    if (device_model_indices[dev_idx].empty()) continue;
                    const size_t leader_idx = device_model_indices[dev_idx][0];
                    cuda_check(cudaSetDevice(models[leader_idx]->device()), "cudaSetDevice set_active_tokens");
                    for (size_t mi : device_model_indices[dev_idx]) {
                        models[mi]->set_active_tokens(buf_cur);
                    }
                }
                upload_s = seconds_since(t0);
                cur_uploaded = true;
            }

            auto eval_worker = [&](size_t di) {
                const auto range = eval_ranges[di];
                const int start = range.first;
                const int end = range.second;
                if (start >= end) return;
                const int dev = models[di]->device();
                cuda_check(cudaSetDevice(dev), "cudaSetDevice eval_worker");
                CudaEflaLm& worker = *models[di];

                EvalScratch& scratch = eval_scratch[di];
                std::vector<uint64_t>& seed_wq = scratch.seed_wq;
                std::vector<uint64_t>& seed_wk = scratch.seed_wk;
                std::vector<uint64_t>& seed_wv = scratch.seed_wv;
                std::vector<uint64_t>& seed_wbeta = scratch.seed_wbeta;
                std::vector<uint64_t>& seed_ff1 = scratch.seed_ff1;
                std::vector<uint64_t>& seed_ff2 = scratch.seed_ff2;
                std::vector<float>& beta_bias_noise = scratch.beta_bias_noise;
                std::vector<float>& pos_noise = scratch.pos_noise;
                std::vector<float>& pos_noisy = scratch.pos_noisy;
                std::vector<float>& head_bias_noise = scratch.head_bias_noise;
                EflaLmWeights& noise = scratch.noise;

                worker.set_head_bias(w.head_bias.data());

                for (int p = start; p < end; ++p) {
                    const uint64_t pair_seed =
                        rng::mix(cfg.seed, rng::mix(static_cast<uint64_t>(epoch), static_cast<uint64_t>(p)));

                    const uint64_t beta_seed = rng::mix(pair_seed, kSaltBetaBias);
                    for (int l = 0; l < cfg.layers; ++l) {
                        beta_bias_noise[static_cast<size_t>(l)] =
                            static_cast<float>(noise_hash(beta_seed, static_cast<uint64_t>(l),
                                                          /*anti*/1, cfg.use_clt_noise, cfg.clt_k));
                    }

                    if (cfg.train_pos) {
                        fill_noise_vector(pos_noise, static_cast<int>(pos.size()),
                                          rng::mix(pair_seed, kSaltPos),
                                          cfg.use_clt_noise, cfg.clt_k);
                        if (pos_noisy.size() != pos.size()) pos_noisy.resize(pos.size());
                        for (size_t i = 0; i < pos.size(); ++i) {
                            pos_noisy[i] = pos[i] + noise_scale * pos_noise[i];
                        }
                        worker.set_pos_embedding(pos_noisy);
                    }

                    const uint64_t seed_emb = rng::mix(pair_seed, kSaltEmb);
                    const uint64_t seed_win = rng::mix(pair_seed, kSaltWin);
                    const uint64_t seed_head = rng::mix(pair_seed, kSaltHead);
                    for (int l = 0; l < cfg.layers; ++l) {
                        seed_wq[static_cast<size_t>(l)] =
                            rng::mix(pair_seed, layer_salt_wq[static_cast<size_t>(l)]);
                        seed_wk[static_cast<size_t>(l)] =
                            rng::mix(pair_seed, layer_salt_wk[static_cast<size_t>(l)]);
                        seed_wv[static_cast<size_t>(l)] =
                            rng::mix(pair_seed, layer_salt_wv[static_cast<size_t>(l)]);
                        seed_wbeta[static_cast<size_t>(l)] =
                            rng::mix(pair_seed, layer_salt_wbeta[static_cast<size_t>(l)]);
                        seed_ff1[static_cast<size_t>(l)] =
                            rng::mix(pair_seed, layer_salt_ff1[static_cast<size_t>(l)]);
                        seed_ff2[static_cast<size_t>(l)] =
                            rng::mix(pair_seed, layer_salt_ff2[static_cast<size_t>(l)]);
                    }

                    if (cfg.use_gpu_noise) {
                        worker.set_noise_seeds(seed_emb,
                                               seed_win,
                                               seed_wq,
                                               seed_wk,
                                               seed_wv,
                                               seed_wbeta,
                                               seed_ff1,
                                               seed_ff2,
                                               seed_head,
                                               cfg.use_clt_noise,
                                               cfg.clt_k);
                    } else {
                        fill_noise_matrix(noise.emb, w.emb, seed_emb, cfg.use_clt_noise, cfg.clt_k);
                        fill_noise_matrix(noise.win, w.win, seed_win, cfg.use_clt_noise, cfg.clt_k);
                        for (int l = 0; l < cfg.layers; ++l) {
                            fill_noise_matrix(noise.wq[static_cast<size_t>(l)], w.wq[static_cast<size_t>(l)],
                                              seed_wq[static_cast<size_t>(l)], cfg.use_clt_noise, cfg.clt_k);
                            fill_noise_matrix(noise.wk[static_cast<size_t>(l)], w.wk[static_cast<size_t>(l)],
                                              seed_wk[static_cast<size_t>(l)], cfg.use_clt_noise, cfg.clt_k);
                            fill_noise_matrix(noise.wv[static_cast<size_t>(l)], w.wv[static_cast<size_t>(l)],
                                              seed_wv[static_cast<size_t>(l)], cfg.use_clt_noise, cfg.clt_k);
                            fill_noise_matrix(noise.wbeta[static_cast<size_t>(l)], w.wbeta[static_cast<size_t>(l)],
                                              seed_wbeta[static_cast<size_t>(l)], cfg.use_clt_noise, cfg.clt_k);
                            fill_noise_matrix(noise.ff1[static_cast<size_t>(l)], w.ff1[static_cast<size_t>(l)],
                                              seed_ff1[static_cast<size_t>(l)], cfg.use_clt_noise, cfg.clt_k);
                            fill_noise_matrix(noise.ff2[static_cast<size_t>(l)], w.ff2[static_cast<size_t>(l)],
                                              seed_ff2[static_cast<size_t>(l)], cfg.use_clt_noise, cfg.clt_k);
                        }
                        fill_noise_matrix(noise.head, w.head, seed_head, cfg.use_clt_noise, cfg.clt_k);
                        worker.set_noise_weights(noise);
                    }

                    fill_noise_vector(head_bias_noise, cfg.vocab, rng::mix(pair_seed, kSaltHeadBias),
                                      cfg.use_clt_noise, cfg.clt_k);

                    const float loss_p = worker.evaluate_loss(method_id, /*anti*/+1, sigma_t,
                                                              cfg.act_scale, cfg.mlp_act_scale,
                                                              cfg.ln_scale,
                                                              cfg.residual_scale, cfg.mlp_residual_scale,
                                                              cfg.gate_scale, cfg.state_decay,
                                                              cfg.int8_residual, cfg.absmean_norm,
                                                              /*head_bias*/nullptr, head_bias_noise.data(),
                                                              w.beta_bias.data(), beta_bias_noise.data());
                    if (cfg.train_pos) {
                        for (size_t i = 0; i < pos.size(); ++i) {
                            pos_noisy[i] = pos[i] - noise_scale * pos_noise[i];
                        }
                        worker.set_pos_embedding(pos_noisy);
                    }
                    const float loss_m = worker.evaluate_loss(method_id, /*anti*/-1, sigma_t,
                                                              cfg.act_scale, cfg.mlp_act_scale,
                                                              cfg.ln_scale,
                                                              cfg.residual_scale, cfg.mlp_residual_scale,
                                                              cfg.gate_scale, cfg.state_decay,
                                                              cfg.int8_residual, cfg.absmean_norm,
                                                              /*head_bias*/nullptr, head_bias_noise.data(),
                                                              w.beta_bias.data(), beta_bias_noise.data());
                    weights[static_cast<size_t>(p)] = -(loss_p - loss_m);
                }
            };

            {
                NvtxRange range_eval(cfg.use_nvtx, "eval_pairs");
                const auto t0 = Clock::now();
                if (models.size() == 1) {
                    eval_worker(0);
                } else {
                    std::vector<std::future<void>> futures;
                    futures.reserve(models.size());
                    for (size_t di = 0; di < models.size(); ++di) {
                        futures.push_back(pool.enqueue([&, di]() { eval_worker(di); }));
                    }
                    for (auto& f : futures) f.get();
                }
                eval_s = seconds_since(t0);
            }

            shape_pair_weights(weights, cfg);
            if (h_pair_weights) {
                std::copy(weights.begin(), weights.end(), h_pair_weights);
            }

            const auto update_t0 = Clock::now();
            if (cfg.use_gpu_update) {
                NvtxRange range_update(cfg.use_nvtx, "update_weights");
                auto update_worker = [&](size_t dev_idx) {
                    if (device_model_indices[dev_idx].empty()) return;
                    const size_t leader_idx = device_model_indices[dev_idx][0];
                    const int dev = models[leader_idx]->device();
                    cuda_check(cudaSetDevice(dev), "cudaSetDevice update_worker");
                    CudaEflaLm& worker = *models[leader_idx];
                    GpuUpdateDevice& gu = gpu_updates[dev_idx];
                    const bool use_packed = worker.use_packed_gemm();

                    const int stream_count = std::max(1, gu.stream_count);
                    std::vector<CudaEflaLm*> update_models;
                    update_models.reserve(static_cast<size_t>(stream_count));
                    for (int si = 0; si < stream_count; ++si) {
                        const size_t mi = device_model_indices[dev_idx][static_cast<size_t>(si)];
                        update_models.push_back(models[mi].get());
                    }

                    for (int si = 0; si < stream_count; ++si) {
                        CudaEflaLm& upd = *update_models[static_cast<size_t>(si)];
                        gpu_update_upload_pair_weights(upd,
                                                       gu.bufs[static_cast<size_t>(si)],
                                                       h_pair_weights ? h_pair_weights : weights.data(),
                                                       current_pairs);
                        gpu_update_fill_pair_seeds(upd,
                                                   gu.bufs[static_cast<size_t>(si)],
                                                   cfg.seed,
                                                   static_cast<uint64_t>(epoch),
                                                   current_pairs);
                    }

                    int stream_idx = 0;
                    auto next_stream = [&]() -> int {
                        const int idx = stream_idx;
                        stream_idx = (stream_idx + 1) % stream_count;
                        return idx;
                    };
                    auto run_update = [&](GpuMatrixState& st,
                                          uint64_t salt,
                                          int8_t* d_w,
                                          size_t n,
                                          uint32_t* d_w_packed) {
                        const int si = next_stream();
                        gpu_update_matrix(*update_models[static_cast<size_t>(si)],
                                          gu.bufs[static_cast<size_t>(si)],
                                          st,
                                          cfg,
                                          salt,
                                          lr_t,
                                          thresh_t,
                                          d_w,
                                          n,
                                          current_pairs,
                                          d_w_packed);
                    };

                    run_update(gu.emb, kSaltEmb, worker.d_emb_w(), emb_size, nullptr);
                    run_update(gu.win, kSaltWin, worker.d_win_w(), hh_size,
                               use_packed ? worker.d_win_packed() : nullptr);
                    for (int l = 0; l < cfg.layers; ++l) {
                        const uint64_t lq = layer_salt_wq[static_cast<size_t>(l)];
                        const uint64_t lk = layer_salt_wk[static_cast<size_t>(l)];
                        const uint64_t lv = layer_salt_wv[static_cast<size_t>(l)];
                        const uint64_t lb = layer_salt_wbeta[static_cast<size_t>(l)];
                        const uint64_t lff1 = layer_salt_ff1[static_cast<size_t>(l)];
                        const uint64_t lff2 = layer_salt_ff2[static_cast<size_t>(l)];
                        const size_t off_hh = static_cast<size_t>(l) * hh_size;
                        const size_t off_h = static_cast<size_t>(l) * h_size;
                        const size_t off_hf = static_cast<size_t>(l) * hf_size;
                        const size_t off_hh_p = static_cast<size_t>(l) * hh_pack;
                        const size_t off_h_p = static_cast<size_t>(l) * h_pack;
                        const size_t off_hf1_p = static_cast<size_t>(l) * hf1_pack;
                        const size_t off_hf2_p = static_cast<size_t>(l) * hf2_pack;
                        run_update(gu.wq[static_cast<size_t>(l)],
                                   lq,
                                   worker.d_wq_w() + off_hh,
                                   hh_size,
                                   use_packed ? (worker.d_wq_packed() + off_hh_p) : nullptr);
                        run_update(gu.wk[static_cast<size_t>(l)],
                                   lk,
                                   worker.d_wk_w() + off_hh,
                                   hh_size,
                                   use_packed ? (worker.d_wk_packed() + off_hh_p) : nullptr);
                        run_update(gu.wv[static_cast<size_t>(l)],
                                   lv,
                                   worker.d_wv_w() + off_hh,
                                   hh_size,
                                   use_packed ? (worker.d_wv_packed() + off_hh_p) : nullptr);
                        run_update(gu.wbeta[static_cast<size_t>(l)],
                                   lb,
                                   worker.d_wbeta_w() + off_h,
                                   h_size,
                                   use_packed ? (worker.d_wbeta_packed() + off_h_p) : nullptr);
                        run_update(gu.ff1[static_cast<size_t>(l)],
                                   lff1,
                                   worker.d_ff1_w() + off_hf,
                                   hf_size,
                                   use_packed ? (worker.d_ff1_packed() + off_hf1_p) : nullptr);
                        run_update(gu.ff2[static_cast<size_t>(l)],
                                   lff2,
                                   worker.d_ff2_w() + off_hf,
                                   hf_size,
                                   use_packed ? (worker.d_ff2_packed() + off_hf2_p) : nullptr);
                    }
                    run_update(gu.head, kSaltHead, worker.d_head_w(), head_size,
                               use_packed ? worker.d_head_packed() : nullptr);

                    if (stream_count > 1) {
                        const int layers = cfg.layers;
                        const int per_stream = (layers + stream_count - 1) / stream_count;
                        for (int si = 0; si < stream_count; ++si) {
                            const int l0 = si * per_stream;
                            const int l1 = std::min(layers, l0 + per_stream);
                            const bool include_emb = (si == 0);
                            const bool include_win = (si == 0);
                            const bool include_head = (si == 0);
                            update_models[static_cast<size_t>(si)]->quantize_nvfp4_base_weights_range(
                                l0, l1, include_emb, include_win, include_head);
                        }
                    }

                    for (int si = 0; si < stream_count; ++si) {
                        cuda_check(cudaEventRecord(gu.stream_events[static_cast<size_t>(si)],
                                                   update_models[static_cast<size_t>(si)]->stream()),
                                   "cudaEventRecord update_stream_done");
                        cuda_check(cudaStreamWaitEvent(worker.stream(),
                                                       gu.stream_events[static_cast<size_t>(si)],
                                                       0),
                                   "cudaStreamWaitEvent update_stream_done");
                    }
                    if (use_packed && !cfg.use_packed_update) {
                        worker.pack_base_weights();
                    }
                    if (stream_count == 1) {
                        worker.quantize_nvfp4_base_weights();
                    }
                    cuda_check(cudaEventRecord(update_done_events[dev_idx], worker.stream()),
                               "cudaEventRecord update_done");
                    for (size_t mi : device_model_indices[dev_idx]) {
                        cuda_check(cudaStreamWaitEvent(models[mi]->stream(), update_done_events[dev_idx], 0),
                                   "cudaStreamWaitEvent update_done");
                    }
                };

                if (devices.size() == 1) {
                    update_worker(0);
                } else {
                    std::vector<std::future<void>> futures;
                    futures.reserve(devices.size());
                    for (size_t di = 0; di < devices.size(); ++di) {
                        futures.push_back(pool.enqueue([&, di]() { update_worker(di); }));
                    }
                    for (auto& f : futures) f.get();
                }

                update_vector(w.head_bias, st_head_bias, weights, cfg, epoch, kSaltHeadBias, lr_t, thresh_t);
                update_vector(w.beta_bias, st_beta_bias, weights, cfg, epoch, kSaltBetaBias, lr_t, thresh_t);
                if (cfg.train_pos) {
                    update_vector(pos, st_pos, weights, cfg, epoch, kSaltPos,
                                  lr_t * cfg.pos_lr_mult, thresh_t * cfg.pos_thresh_mult);
                    for (size_t di = 0; di < models.size(); ++di) {
                        cuda_check(cudaSetDevice(models[di]->device()), "cudaSetDevice set_pos_embedding");
                        models[di]->set_pos_embedding(pos);
                    }
                }
            } else {
                // Update all ternary weight matrices on CPU.
                {
                    NvtxRange range_update(cfg.use_nvtx, "update_weights");
                    update_matrix(w.emb, st_emb, weights, cfg, epoch, kSaltEmb, lr_t, thresh_t);
                    update_matrix(w.win, st_win, weights, cfg, epoch, kSaltWin, lr_t, thresh_t);
                    for (int l = 0; l < cfg.layers; ++l) {
                        const uint64_t lq = rng::mix(static_cast<uint64_t>(l), kSaltWq);
                        const uint64_t lk = rng::mix(static_cast<uint64_t>(l), kSaltWk);
                        const uint64_t lv = rng::mix(static_cast<uint64_t>(l), kSaltWv);
                        const uint64_t lb = rng::mix(static_cast<uint64_t>(l), kSaltWbeta);
                        const uint64_t lff1 = rng::mix(static_cast<uint64_t>(l), kSaltFf1);
                        const uint64_t lff2 = rng::mix(static_cast<uint64_t>(l), kSaltFf2);
                        update_matrix(w.wq[static_cast<size_t>(l)], st_wq[static_cast<size_t>(l)],
                                      weights, cfg, epoch, lq, lr_t, thresh_t);
                        update_matrix(w.wk[static_cast<size_t>(l)], st_wk[static_cast<size_t>(l)],
                                      weights, cfg, epoch, lk, lr_t, thresh_t);
                        update_matrix(w.wv[static_cast<size_t>(l)], st_wv[static_cast<size_t>(l)],
                                      weights, cfg, epoch, lv, lr_t, thresh_t);
                        update_matrix(w.wbeta[static_cast<size_t>(l)], st_wbeta[static_cast<size_t>(l)],
                                      weights, cfg, epoch, lb, lr_t, thresh_t);
                        update_matrix(w.ff1[static_cast<size_t>(l)], st_ff1[static_cast<size_t>(l)],
                                      weights, cfg, epoch, lff1, lr_t, thresh_t);
                        update_matrix(w.ff2[static_cast<size_t>(l)], st_ff2[static_cast<size_t>(l)],
                                      weights, cfg, epoch, lff2, lr_t, thresh_t);
                    }
                    update_matrix(w.head, st_head, weights, cfg, epoch, kSaltHead, lr_t, thresh_t);
                    update_vector(w.head_bias, st_head_bias, weights, cfg, epoch, kSaltHeadBias, lr_t, thresh_t);
                    update_vector(w.beta_bias, st_beta_bias, weights, cfg, epoch, kSaltBetaBias, lr_t, thresh_t);
                    if (cfg.train_pos) {
                        update_vector(pos, st_pos, weights, cfg, epoch, kSaltPos,
                                      lr_t * cfg.pos_lr_mult, thresh_t * cfg.pos_thresh_mult);
                        for (size_t di = 0; di < models.size(); ++di) {
                            cuda_check(cudaSetDevice(models[di]->device()), "cudaSetDevice set_pos_embedding");
                            models[di]->set_pos_embedding(pos);
                        }
                    }
                }

                {
                    NvtxRange range_sync(cfg.use_nvtx, "sync_weights");
                    for (size_t di = 0; di < device_model_indices.size(); ++di) {
                        if (device_model_indices[di].empty()) continue;
                        const size_t leader_idx = device_model_indices[di][0];
                        cuda_check(cudaSetDevice(models[leader_idx]->device()), "cudaSetDevice sync_base_weights");
                        models[leader_idx]->sync_base_weights(w);
                        cuda_check(cudaEventRecord(update_done_events[di], models[leader_idx]->stream()),
                                   "cudaEventRecord sync_weights");
                        for (size_t mi : device_model_indices[di]) {
                            cuda_check(cudaStreamWaitEvent(models[mi]->stream(), update_done_events[di], 0),
                                       "cudaStreamWaitEvent sync_weights");
                        }
                    }
                }
            }
            update_s = seconds_since(update_t0);

            const bool do_val_epoch = cfg.do_val && cfg.val_every > 0 && (epoch % cfg.val_every) == 0;

            {
                float train_loss = 0.0f;
                {
                    NvtxRange range_train(cfg.use_nvtx, "train_loss_eval");
                    const auto t0 = Clock::now();
                    cuda_check(cudaSetDevice(models[0]->device()), "cudaSetDevice train_loss");
                    model.set_head_bias(w.head_bias.data());
                    train_loss = model.evaluate_loss(method_id, /*anti*/0, sigma_t,
                                                     cfg.act_scale, cfg.mlp_act_scale,
                                                     cfg.ln_scale,
                                                     cfg.residual_scale, cfg.mlp_residual_scale,
                                                     cfg.gate_scale, cfg.state_decay,
                                                     cfg.int8_residual, cfg.absmean_norm,
                                                     /*head_bias*/nullptr, /*head_bias_noise*/nullptr,
                                                     w.beta_bias.data(), /*beta_bias_noise*/nullptr);
                    train_s = seconds_since(t0);
                }
                const float train_ppl = std::exp(train_loss);
                std::cout << "epoch " << epoch
                          << " lr " << lr_t
                          << " thresh " << thresh_t
                          << " sigma " << sigma_t
                          << " train_loss " << train_loss
                          << " train_ppl " << train_ppl;

                if (do_val_epoch) {
                    float val_loss = 0.0f;
                    {
                        NvtxRange range_val(cfg.use_nvtx, "val_eval");
                        const auto t0 = Clock::now();
                        std::vector<std::vector<uint8_t>> vin, vtgt;
                        data.sample_batch(cfg.batch, cfg.seq_len, cfg.val_seed, vin, vtgt);
                        const int val_buf = cfg.fixed_train ? buf_next : buf_cur;
                        cuda_check(cudaSetDevice(models[0]->device()), "cudaSetDevice val_upload_tokens");
                        model.set_head_bias(w.head_bias.data());
                        model.upload_tokens(vin, vtgt, val_buf);
                        val_loss = model.evaluate_loss(method_id, /*anti*/0, sigma_t,
                                                       cfg.act_scale, cfg.mlp_act_scale,
                                                       cfg.ln_scale,
                                                       cfg.residual_scale, cfg.mlp_residual_scale,
                                                       cfg.gate_scale, cfg.state_decay,
                                                       cfg.int8_residual, cfg.absmean_norm,
                                                       /*head_bias*/nullptr, /*head_bias_noise*/nullptr,
                                                       w.beta_bias.data(), /*beta_bias_noise*/nullptr);
                        if (cfg.fixed_train && val_buf != buf_cur) {
                            model.set_active_tokens(buf_cur);
                        }
                        val_s = seconds_since(t0);
                    }
                    const float val_ppl = std::exp(val_loss);
                    std::cout << " val_loss " << val_loss
                              << " val_ppl " << val_ppl;
                }

                if (prefetch_active) {
                    prefetch_future.get();
                    if (!use_pinned_tokens) {
                        inputs_cur.swap(inputs_next);
                        targets_cur.swap(targets_next);
                    }
                    cur_sample_s = next_sample_s;
                    buf_cur = buf_next;
                    buf_next = 1 - buf_cur;
                    cur_uploaded = next_uploaded;
                }

            }

            const double wall_s = seconds_since(epoch_start);
            const double tokens_eval = static_cast<double>(current_pairs) * 2.0 *
                                       static_cast<double>(cfg.batch) * static_cast<double>(cfg.seq_len);
            const double tokens_train = static_cast<double>(cfg.batch) * static_cast<double>(cfg.seq_len);
            const double tokens_val = do_val_epoch ? tokens_train : 0.0;
            const double tokens_total = tokens_eval + tokens_train + tokens_val;
            const double tok_s_eval = (eval_s > 0.0) ? (tokens_eval / eval_s) : 0.0;
            const double tok_s_total = (wall_s > 0.0) ? (tokens_total / wall_s) : 0.0;

            std::cout << " wall_s " << wall_s
                      << " sample_s " << sample_s
                      << " upload_s " << upload_s
                      << " eval_s " << eval_s
                      << " update_s " << update_s
                      << " train_s " << train_s;
            if (do_val_epoch) {
                std::cout << " val_s " << val_s;
            }
            std::cout << " tok/s_eval " << tok_s_eval
                      << " tok/s_total " << tok_s_total;

            std::cout << "\n";
        }

        if (cfg.use_gpu_update) {
            for (size_t di = 0; di < gpu_updates.size(); ++di) {
                if (device_model_indices[di].empty()) continue;
                const size_t leader_idx = device_model_indices[di][0];
                const int dev = models[leader_idx]->device();
                GpuUpdateDevice& gu = gpu_updates[di];
                for (auto& buf : gu.bufs) {
                    gpu_update_free_buffers(dev, buf);
                }
                for (cudaEvent_t ev : gu.stream_events) {
                    if (!ev) continue;
                    cuda_check(cudaSetDevice(dev), "cudaSetDevice destroy update_stream_event");
                    cudaEventDestroy(ev);
                }
                gpu_update_free_matrix(dev, gu.emb);
                gpu_update_free_matrix(dev, gu.win);
                gpu_update_free_matrix(dev, gu.head);
                for (size_t l = 0; l < gu.wq.size(); ++l) {
                    gpu_update_free_matrix(dev, gu.wq[l]);
                    gpu_update_free_matrix(dev, gu.wk[l]);
                    gpu_update_free_matrix(dev, gu.wv[l]);
                    gpu_update_free_matrix(dev, gu.wbeta[l]);
                    gpu_update_free_matrix(dev, gu.ff1[l]);
                    gpu_update_free_matrix(dev, gu.ff2[l]);
                }
            }
        }
        for (int i = 0; i < kTokenBuffers; ++i) {
            if (h_inputs_pinned[i]) cudaFreeHost(h_inputs_pinned[i]);
            if (h_targets_pinned[i]) cudaFreeHost(h_targets_pinned[i]);
        }
        if (h_pair_weights) {
            cudaFreeHost(h_pair_weights);
        }
        for (size_t di = 0; di < update_done_events.size(); ++di) {
            if (!update_done_events[di]) continue;
            cuda_check(cudaSetDevice(devices[di]), "cudaSetDevice destroy update_done");
            cudaEventDestroy(update_done_events[di]);
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
