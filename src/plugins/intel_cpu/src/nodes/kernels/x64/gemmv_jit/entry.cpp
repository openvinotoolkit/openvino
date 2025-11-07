// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "entry.hpp"

#include <memory>
#include <cstdlib>
#include <string>
#include <thread>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include "jit_minigemm_avx512_fp32.hpp"
#include "xbyak/xbyak_util.h"
#include "jit_gemmv_avx512_vnni_s32.hpp"
#include "vnni_gemmv_intrin.hpp"
#include "amx_gemmv_intrin.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

// Forward decl: vectorized per-tensor u8 quantization helper (defined later in this file)
static inline void quantize_u8_symmetric(const float* x, int K, uint8_t* xq, float* s_x_out, int32_t* zp_out);

void run_gemmv_i8_fp32(const float* x, int K,
                       const uint8_t* wq_packed, int M, int ld_w_bytes,
                       const float* scales, const int32_t* zps,
                       float* y, const float* bias,
                       quant_granularity_t gran, bool accumulate,
                       bool is_u8) {
    gemmv_ukr_params_t p{};
    p.x = x;
    p.K = K;
    p.wq = wq_packed;
    p.ld_w_bytes = ld_w_bytes;
    p.scales = scales;
    p.zps = zps;
    p.gran = gran;
    p.y = y;
    p.bias = const_cast<float*>(bias);
    p.M = M;
    p.accumulate = accumulate;
    p.a_type = a_dtype_t::fp32;
    p.w_type = is_u8 ? w_dtype_t::u8 : w_dtype_t::i8;
    p.fuse_gate = false; p.gate_scale = 1.f; p.act_kind = 0;

    std::unique_ptr<GemmvKernel> kernel(create_gemmv_kernel(p));
    (*kernel)(&p);
}

// Forward to reference implementation declared in gemmv_ukernel.hpp
void run_minigemm_ref_q_fp32(const float* x, int K, int N,
                             const uint8_t* wq_packed, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran,
                             w_dtype_t wtype,
                             bool accumulate,
                             int group_size);

void run_gemmv_i8_fp32_ex(const float* x, int K,
                          const uint8_t* wq_packed, int M, int ld_w_bytes,
                          const float* scales, const int32_t* zps,
                          float* y, const float* bias,
                          quant_granularity_t gran, bool accumulate,
                          bool is_u8, const char** kernel_name) {
    gemmv_ukr_params_t p{};
    p.x = x;
    p.K = K;
    p.wq = wq_packed;
    p.ld_w_bytes = ld_w_bytes;
    p.scales = scales;
    p.zps = zps;
    p.gran = gran;
    p.y = y;
    p.bias = const_cast<float*>(bias);
    p.M = M;
    p.accumulate = accumulate;
    p.a_type = a_dtype_t::fp32;
    p.w_type = is_u8 ? w_dtype_t::u8 : w_dtype_t::i8;
    p.fuse_gate = false; p.gate_scale = 1.f; p.act_kind = 0;

    std::unique_ptr<GemmvKernel> kernel(create_gemmv_kernel(p));
    if (kernel_name) *kernel_name = kernel->name();
    (*kernel)(&p);
}

void run_gemmv_q_fp32_ex(const float* x, int K,
                         const uint8_t* wq_packed, int M, int ld_w_bytes,
                         const float* scales, const int32_t* zps,
                         float* y, const float* bias,
                         quant_granularity_t gran, int group_size, bool accumulate,
                         w_dtype_t wtype, const char** kernel_name) {
    // Repack cache: avoids repeated interleave->k64/k4 repacks across calls for same weights
    struct FreeDeleter { void operator()(uint8_t* p) const { if (p) free(p); } };
    struct PackedEntry {
        std::unique_ptr<uint8_t, FreeDeleter> buf{};
        std::vector<int32_t> sumW;
        size_t bytes = 0;
        int ld_bytes = 0;
        int M_pad = 0;
    };
    struct RepackKey { uintptr_t base; int M; int K; int ld; int layout; };
    struct RepackKeyHash { size_t operator()(const RepackKey& k) const noexcept {
        size_t h = std::hash<uintptr_t>{}(k.base);
        h ^= std::hash<uint64_t>{}(((uint64_t)k.M)<<32 ^ (uint32_t)k.K) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        h ^= std::hash<uint64_t>{}(((uint64_t)k.ld)<<32 ^ (uint32_t)k.layout) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        return h; } };
    struct RepackKeyEq { bool operator()(const RepackKey&a, const RepackKey&b) const noexcept { return a.base==b.base && a.M==b.M && a.K==b.K && a.ld==b.ld && a.layout==b.layout; } };
    static std::unordered_map<RepackKey, PackedEntry, RepackKeyHash, RepackKeyEq> g_repack_cache;
    static std::mutex g_repack_mtx;
    auto get_or_make_k64 = [&](const uint8_t* base, int Mv, int Kv, int ld)->PackedEntry&{
        std::lock_guard<std::mutex> lk(g_repack_mtx);
        RepackKey key{(uintptr_t)base, Mv, Kv, ld, 0};
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) return it->second;
        const int M_blk = 16, K_blk = 64; const int M_pad = ((Mv + M_blk - 1)/M_blk)*M_blk; const int K_grp = (Kv + K_blk - 1)/K_blk;
        const size_t bytes = (size_t)M_pad * (size_t)K_grp * (size_t)K_blk;
        void* mem=nullptr; size_t size_al = ((bytes + 63)/64)*64; if (posix_memalign(&mem, 64, size_al)!=0 || !mem) mem = malloc(size_al);
        PackedEntry pe; pe.buf = std::unique_ptr<uint8_t, FreeDeleter>((uint8_t*)mem); pe.bytes = bytes; pe.ld_bytes = M_blk * K_blk; pe.M_pad = M_pad; pe.sumW.assign(M_pad, 0);
        repack_interleave_m16_to_k64_m16(pe.buf.get(), base, Mv, Kv, ld, M_blk, K_blk, pe.sumW.data());
        auto ins = g_repack_cache.emplace(key, std::move(pe));
        return ins.first->second;
    };
    auto get_or_make_k4 = [&](const uint8_t* base, int Mv, int Kv, int ld)->PackedEntry&{
        std::lock_guard<std::mutex> lk(g_repack_mtx);
        RepackKey key{(uintptr_t)base, Mv, Kv, ld, 1};
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) return it->second;
        const int M_blk = 16; const int K_grp = (Kv + 3)/4; const int M_pad = ((Mv + M_blk - 1)/M_blk)*M_blk;
        const size_t bytes = (size_t)M_pad * (size_t)K_grp * 4;
        void* mem=nullptr; size_t size_al = ((bytes + 63)/64)*64; if (posix_memalign(&mem, 64, size_al)!=0 || !mem) mem = malloc(size_al);
        PackedEntry pe; pe.buf = std::unique_ptr<uint8_t, FreeDeleter>((uint8_t*)mem); pe.bytes = bytes; pe.ld_bytes = K_grp * 64; pe.M_pad = M_pad; pe.sumW.assign(M_pad, 0);
        repack_interleave_m16_to_k4_m16(pe.buf.get(), base, Mv, Kv, ld, M_blk, pe.sumW.data());
        auto ins = g_repack_cache.emplace(key, std::move(pe));
        return ins.first->second;
    };
    auto get_or_make_k64_tile = [&](const uint8_t* base, int Mv, int Kv, int ld)->PackedEntry&{
        std::lock_guard<std::mutex> lk(g_repack_mtx);
        RepackKey key{(uintptr_t)base, Mv, Kv, ld, 2};
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) return it->second;
        const int M_blk = 16, K_blk = 64; const int M_pad = ((Mv + M_blk - 1)/M_blk)*M_blk; const int K_grp = (Kv + K_blk - 1)/K_blk;
        const size_t bytes = (size_t)M_pad * (size_t)K_grp * (size_t)K_blk * 16;
        void* mem=nullptr; size_t size_al = ((bytes + 63)/64)*64; if (posix_memalign(&mem, 64, size_al)!=0 || !mem) mem = malloc(size_al);
        PackedEntry pe; pe.buf = std::unique_ptr<uint8_t, FreeDeleter>((uint8_t*)mem); pe.bytes = bytes; pe.ld_bytes = K_grp * (K_blk * 16); pe.M_pad = M_pad; pe.sumW.assign(M_pad, 0);
        repack_interleave_m16_to_k64_m16_tile(pe.buf.get(), base, Mv, Kv, ld, M_blk, K_blk, pe.sumW.data());
        auto ins = g_repack_cache.emplace(key, std::move(pe));
        return ins.first->second;
    };
    // Fast path 0: AMX INT8 (tile GEMV) with on-the-fly repack to K64 per M-block
    if (!accumulate && wtype == w_dtype_t::i8) {
        // Try AMX automatically when supported (no env flags required)
        bool diag = false; if (const char* d = std::getenv("GEMMV_AMX_DIAG")) diag = (std::string(d) == "1");
        {
            if (diag) std::fprintf(stderr, "[GEMMV-AMX-DIAG] AMX route requested\n");
            // Repack via cache + try AMX intrinsics
            const int M_blk = 16; const int K_blk = 64; const int K_grp = (K + K_blk - 1)/K_blk;
            auto &wk64_amx = get_or_make_k64_tile(wq_packed, M, K, ld_w_bytes);
            bool ok = run_gemmv_amx_i8u8_fp32(x, K, wk64_amx.buf.get(), M, /*ld_w_kbytes*/ K_grp * (K_blk * 16),
                                              scales, zps, y, bias, gran, group_size,
                                              wk64_amx.sumW.data());
            if (ok) { if (kernel_name) *kernel_name = "amx_int8"; return; }
            if (diag) {
                std::fprintf(stderr, "[GEMMV-AMX-DIAG] AMX route unavailable, fallback to VNNI/JIT\n");
                std::fprintf(stderr, "[GEMMV-AMX-DIAG] hint: set GEMMV_AMX_DIAG=1 for reasons; quick check: GEMMV_DISABLE_VNNI=1 GEMMV_AMX_DIAG=1 OMP_NUM_THREADS=1 <gemmv_bench> 512 4096\n");
            }
        }
    }

    // Fast path 1: AVX-512 VNNI and i8 weights (enabled by default; opt-out via GEMMV_DISABLE_VNNI=1)
    if (wtype == w_dtype_t::i8) {
        Xbyak::util::Cpu cpu;
        if (cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) && cpu.has(Xbyak::util::Cpu::tAVX512BW)) {
            // Quantize X to u8 per-tensor, vectorized helper
            std::vector<uint8_t> xq(K);
            float s_x = 1.f; int32_t zp_x = 128; int32_t sum_x_q = 0;
            quantize_u8_symmetric(x, K, xq.data(), &s_x, &zp_x);
            const int M_blk = 16;
            // Deterministic route:
            // - k4 for small M (<=256)
            // - k64 for larger M (>=512)
            // - for 256<M<512 choose k64 only when K is 64-aligned
            bool prefer_k64 = false;
            if (M >= 512) prefer_k64 = true;
            else if (M <= 256) prefer_k64 = false;
            else prefer_k64 = ((K & 63) == 0);
            if (prefer_k64) {
                const int K_blk = 64; const int K_grp = (K + K_blk - 1)/K_blk;
                auto &wk64 = get_or_make_k64(wq_packed, M, K, ld_w_bytes);
                run_gemmv_vnni_intrin_i8u8_fp32_k64(xq.data(), K,
                                                    wk64.buf.get(), M, /*ld_w_kbytes*/ K_grp * (M_blk*K_blk),
                                                    scales ? scales[0] : 1.f, /*zp_w*/ 0,
                                                    s_x, zp_x, y, bias ? bias[0] : 0.f,
                                                    wk64.sumW.data());
                if (kernel_name) *kernel_name = "vnni_k64_repack_intrin";
                return;
            }
            // k4 path: per-tensor
            if (gran == quant_granularity_t::per_tensor) {
                const int K_grp = (K + 3) / 4;
                auto &wk4 = get_or_make_k4(wq_packed, M, K, ld_w_bytes);
                run_gemmv_vnni_intrin_i8u8_fp32(xq.data(), K,
                                                wk4.buf.get(), M, /*ld_w_gbytes*/ K_grp * 64,
                                                scales ? scales[0] : 1.f, /*zp_w*/ 0,
                                                s_x, zp_x, y, bias ? bias[0] : 0.f,
                                                wk4.sumW.data());
                if (kernel_name) *kernel_name = "vnni_k4_repack_intrin";
                return;
            }
            // Fallback: no-repack intrinsics for other granularities
            {
                const int M_full = M / M_blk; const int M_tail = M % M_blk;
                auto meta_ptrs = [&](int bi, const float*& sc, const int32_t*& zp, const float*& bs){
                    if (gran == quant_granularity_t::per_tensor) { sc = scales; zp = zps; bs = bias; return; }
                    if (gran == quant_granularity_t::per_channel) { sc = scales + bi * M_blk; zp = zps ? (zps + bi * M_blk) : nullptr; bs = bias ? (bias + bi * M_blk) : nullptr; return; }
                    const int base_m = bi * M_blk; const int gs = group_size > 0 ? group_size : M_blk;
                    static float sc_buf[16]; static int32_t zp_buf[16]; static float b_buf[16];
                    for (int m = 0; m < 16; ++m) { int g = (base_m + m) / gs; sc_buf[m] = scales[g]; if (zps) zp_buf[m] = zps[g]; if (bias) b_buf[m] = bias[g]; }
                    sc = sc_buf; zp = zps ? zp_buf : nullptr; bs = bias ? b_buf : nullptr;
                };
                for (int bi = 0; bi < M_full; ++bi) {
                    const float* sc=nullptr; const int32_t* zp=nullptr; const float* bs=nullptr; meta_ptrs(bi, sc, zp, bs);
                    run_gemmv_vnni_intrin_i8u8_fp32_norepack(xq.data(), K, wq_packed + (size_t)bi * (size_t)ld_w_bytes,
                                                             M_blk, ld_w_bytes, sc, zp, bs, s_x, zp_x, sum_x_q,
                                                             y + bi * M_blk, 0, (int)gran, group_size);
                }
                if (M_tail) {
                    const int bi = M_full; const float* sc=nullptr; const int32_t* zp=nullptr; const float* bs=nullptr; meta_ptrs(bi, sc, zp, bs);
                    run_gemmv_vnni_intrin_i8u8_fp32_norepack(xq.data(), K, wq_packed + (size_t)bi * (size_t)ld_w_bytes,
                                                             M_tail, ld_w_bytes, sc, zp, bs, s_x, zp_x, sum_x_q,
                                                             y + bi * M_blk, M_tail, (int)gran, group_size);
                }
                if (kernel_name) *kernel_name = "vnni_norepack_intrin";
                return;
            }
        }
    }

    // Fallback to JIT/ref kernel
    gemmv_ukr_params_t p{};
    p.x = x; p.K = K; p.wq = wq_packed; p.ld_w_bytes = ld_w_bytes;
    p.scales = scales; p.zps = zps; p.gran = gran; p.group_size = group_size;
    p.y = y; p.bias = const_cast<float*>(bias); p.M = M; p.accumulate = accumulate;
    p.a_type = a_dtype_t::fp32; p.w_type = wtype; p.fuse_gate = false; p.gate_scale = 1.f; p.act_kind = 0;
    std::unique_ptr<GemmvKernel> kernel(create_gemmv_kernel(p)); if (kernel_name) *kernel_name = kernel->name(); (*kernel)(&p);
}

// Simple multi-threaded wrapper across M-blocks using std::thread.
// Partitions M across T threads (env GEMMV_THREADS or hw concurrency).
void run_gemmv_q_fp32_mt(const float* x, int K,
                         const uint8_t* wq_packed, int M, int ld_w_bytes,
                         const float* scales, const int32_t* zps,
                         float* y, const float* bias,
                         quant_granularity_t gran, int group_size, bool accumulate,
                         w_dtype_t wtype, int threads) {
    if (threads <= 1 || M <= 16) {
        run_gemmv_q_fp32_ex(x, K, wq_packed, M, ld_w_bytes,
                            scales, zps, y, bias, gran, group_size, accumulate, wtype, nullptr);
        return;
    }
    const int M_blk = 16;
    const int blocks = (M + M_blk - 1) / M_blk;
    const int T = std::max(1, std::min(threads, blocks));
    std::unique_ptr<GemmvKernel> kernel;
    {
        gemmv_ukr_params_t p{};
        p.x = x; p.K = K; p.wq = wq_packed; p.ld_w_bytes = ld_w_bytes; p.scales = scales; p.zps = zps;
        p.gran = gran; p.group_size = group_size; p.y = y; p.bias = const_cast<float*>(bias);
        p.M = std::min(M, M_blk); p.accumulate = accumulate; p.a_type = a_dtype_t::fp32; p.w_type = wtype;
        kernel.reset(create_gemmv_kernel(p));
    }
    auto worker = [&](int tid){
        int blocks_per_thr = blocks / T;
        int rem = blocks % T;
        int start_b = tid * blocks_per_thr + std::min(tid, rem);
        int my_count = blocks_per_thr + (tid < rem ? 1 : 0);
        if (my_count <= 0) return;
        int start_m = start_b * M_blk;
        int m_len = std::min(M - start_m, my_count * M_blk);
        gemmv_ukr_params_t p{};
        p.x = x; p.K = K; p.wq = wq_packed + (size_t)start_b * (size_t)ld_w_bytes;
        p.ld_w_bytes = ld_w_bytes; p.scales = scales; p.zps = zps; p.gran = gran; p.group_size = group_size;
        p.y = y + start_m; p.bias = const_cast<float*>(bias); p.M = m_len; p.accumulate = accumulate;
        p.a_type = a_dtype_t::fp32; p.w_type = wtype; p.m_base = start_m;
        (*kernel)(&p);
    };
    std::vector<std::thread> pool;
    pool.reserve(T);
    for (int t = 0; t < T; ++t) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();
}

void run_minigemm_q_fp32_ex(const float* x, int K, int N,
                            const uint8_t* wq_packed, int M, int ld_w_bytes,
                            const float* scales, const int32_t* zps,
                            float* y, const float* bias,
                            quant_granularity_t gran, int group_size,
                            w_dtype_t wtype, const char** kernel_name) {
    // Simple heuristic: prefer JIT mini-GEMM starting from small N when K is large
    // to amortize W decode across columns; otherwise fall back to reference.
    // Tunables (can be refined by calibration logs):
    auto compute_n_thr = [&](int Mv, int Kv) {
        // Simple table-inspired heuristic: shrink threshold when K grows,
        // raise when M very small. Tuned conservatively.
        if (Mv < 64) {
            if (Kv >= 4096) return 6;  // very large K still benefits earlier
            return 12;
        }
        if (Kv >= 8192) return 4;
        if (Kv >= 4096) return 6;
        if (Kv >= 2048) return 8;
        return 12;
    };
    int n_thr = compute_n_thr(M, K);

    bool used_jit = false;
    const bool enable_jit_minigemm = (N >= n_thr);
    if (enable_jit_minigemm) {
        used_jit = run_minigemm_jit_q_fp32(x, K, N, wq_packed, M, ld_w_bytes,
                                           scales, zps, y, bias, gran, wtype, group_size);
    }
    if (used_jit) {
        if (kernel_name) *kernel_name = "jit_avx512_minigemm";
        return;
    }
    run_minigemm_ref_q_fp32(x, K, N, wq_packed, M, ld_w_bytes,
                            scales, zps, y, bias, gran, wtype, /*acc=*/false, group_size);
    if (kernel_name) *kernel_name = "ref_minigemm";
}

static inline void quantize_u8_symmetric(const float* x, int K, uint8_t* xq, float* s_x_out, int32_t* zp_out) {
    float amax = 0.f; for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[k]));
    float s = (amax > 0.f) ? (amax / 127.f) : 1.f; // symmetric around 0 -> use zp=128 for u8
    int32_t zp = 128;
#if defined(__AVX512F__)
    const __m512 inv_s = _mm512_set1_ps(1.0f / s);
    const __m512i vzp = _mm512_set1_epi32(zp);
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        __m512 xf = _mm512_loadu_ps(x + k);
        xf = _mm512_mul_ps(xf, inv_s);
        __m512i xi = _mm512_cvtps_epi32(xf);
        xi = _mm512_add_epi32(xi, vzp);
        // clamp to [0,255]
        xi = _mm512_max_epi32(xi, _mm512_set1_epi32(0));
        xi = _mm512_min_epi32(xi, _mm512_set1_epi32(255));
        // pack 16 x int32 to 16 x u8
        __m256i xi16a = _mm512_cvtepi32_epi16(xi); // 16-bit lanes
        __m128i xi8 = _mm_packus_epi16(_mm256_castsi256_si128(xi16a), _mm_setzero_si128());
        _mm_storeu_si128((__m128i*)(xq + k), xi8);
    }
    for (; k < K; ++k) {
        int v = (int)std::lrintf(x[k] / s) + zp;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        xq[k] = (uint8_t)v;
    }
#else
    for (int k = 0; k < K; ++k) {
        int v = (int)std::lrintf(x[k] / s) + zp;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        xq[k] = (uint8_t)v;
    }
#endif
    *s_x_out = s; *zp_out = zp;
}

bool run_gemmv_vnni_q8s8_ex(const float* x_fp32, int K,
                            const uint8_t* wq_k4, int M, int ld_w_gbytes,
                            const float* scales, const int32_t* zps,
                            float* y, const float* bias,
                            quant_granularity_t gran,
                            int dbg_block, int32_t* dbg_acc, int32_t* dbg_sumw,
                            const int32_t* sumW_precomp) {
    // Only per-tensor supported in first cut
    if (gran != quant_granularity_t::per_tensor) return false;
    Xbyak::util::Cpu cpu; if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) || !cpu.has(Xbyak::util::Cpu::tAVX512BW)) return false;
    float s_w = scales ? scales[0] : 1.f;
    // For signed int8 weights use symmetric zp_w=0
    int32_t zp_w = 0;
    float bias0 = bias ? bias[0] : 0.f;
    // Quantize X per-tensor symmetric to u8
    std::vector<uint8_t> xq(K);
    float s_x; int32_t zp_x;
    quantize_u8_symmetric(x_fp32, K, xq.data(), &s_x, &zp_x);
    const int dump_only = (dbg_acc && dbg_sumw && dbg_block >= 0) ? 1 : 0;
    // Optional: pure C++ stub for dump-only debug (env GEMMV_VNNI_STUB=1)
    if (dump_only) {
        if (const char* st = std::getenv("GEMMV_VNNI_STUB"); st && std::string(st) == "1") {
            // Compute for block 0, group 0 only
            const int M_blk = 16; const int K_grp = (K + 3)/4;
            auto get_w4 = [&](int bi,int g,int row){ return wq_k4 + (size_t)bi*ld_w_gbytes + (size_t)g*64 + (size_t)row*4; };
            // Quantize X to u8 (we already have xq)
            std::vector<uint8_t> xq(K);
            float s_x; int32_t zp_x; quantize_u8_symmetric(x_fp32, K, xq.data(), &s_x, &zp_x);
            for (int m=0;m<M_blk;++m) {
                const uint8_t* w4 = get_w4(0, 0, m);
                int8_t w0=(int8_t)w4[0], w1=(int8_t)w4[1], w2=(int8_t)w4[2], w3=(int8_t)w4[3];
                uint8_t xb0=xq[0], xb1=xq[1], xb2=xq[2], xb3=xq[3];
                int32_t acc = (int32_t)xb0*w0 + (int32_t)xb1*w1 + (int32_t)xb2*w2 + (int32_t)xb3*w3;
                int32_t sw = (int32_t)w0 + w1 + w2 + w3;
                dbg_acc[m] = acc; dbg_sumw[m] = sw;
            }
            return true;
        }
    }
    // Try JIT first when not stub; otherwise fallback to intrinsic
    bool used = false;
    if (!std::getenv("GEMMV_VNNI_STUB")) {
        used = run_gemmv_vnni_i8u8_fp32(xq.data(), K, wq_k4, M, ld_w_gbytes,
                                        s_w, zp_w, s_x, zp_x, y, bias0,
                                        dbg_block, dbg_acc, dbg_sumw, dump_only);
    }
    if (!used && !dump_only) {
        // Heuristic: choose KU/PFD based on K and M (small-M specialization) if not set by env
        if (!std::getenv("GEMMV_VNNI_KU")) {
            int ku = (M <= 256) ? 8 : ((K >= 4096) ? 8 : (K >= 2048 ? 4 : 2));
            std::string s = std::to_string(ku);
            setenv("GEMMV_VNNI_KU", s.c_str(), 0);
        }
        if (!std::getenv("GEMMV_VNNI_PFD")) {
            int pfd = (M <= 256) ? 4 : ((K >= 4096) ? 4 : (K >= 2048 ? 3 : 2));
            std::string s = std::to_string(pfd);
            setenv("GEMMV_VNNI_PFD", s.c_str(), 0);
        }
        used = run_gemmv_vnni_intrin_i8u8_fp32(xq.data(), K, wq_k4, M, ld_w_gbytes,
                                               s_w, zp_w, s_x, zp_x, y, bias0, sumW_precomp);
    }
    return used;
}

} // namespace ov::intel_cpu::x64::gemmv_jit
