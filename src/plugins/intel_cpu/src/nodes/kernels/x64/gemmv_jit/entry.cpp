// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "entry.hpp"

#include <memory>
#include <cstdlib>
#include <string>
#include <thread>
#include <cmath>
#include <algorithm>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <limits>
#include <cstring>
#include <chrono>
#include <atomic>
#include <immintrin.h>
#include "jit_minigemm_avx512_fp32.hpp"
#include "xbyak/xbyak_util.h"
#include "jit_gemmv_avx512_vnni_s32.hpp"
#include "amx_gemmv_intrin.hpp"
#include "gemmv_force_isa.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

// Forward decl: vectorized per-tensor u8 quantization helper (defined later in this file)
static inline void quantize_u8_symmetric(const float* x, int K, uint8_t* xq,
                                         float* s_x_out, int32_t* zp_out, int32_t* sum_x_out = nullptr);

namespace {

bool read_profile_env() {
    if (const char* env = std::getenv("GEMMV_PROFILE")) {
        return env[0] != '\0' && env[0] != '0';
    }
    return false;
}

std::atomic<bool> g_profile_override{false};
thread_local gemmv_profile_snapshot g_last_profile{};

inline bool gemmv_profile_enabled() {
    static const bool env_enabled = read_profile_env();
    if (g_profile_override.load(std::memory_order_relaxed)) {
        return true;
    }
    return env_enabled;
}

} // namespace

gemmv_profile_snapshot get_last_gemmv_profile() {
    return g_last_profile;
}

void set_gemmv_profile_override(bool enable) {
    g_profile_override.store(enable, std::memory_order_relaxed);
}

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
    const bool profile_enabled = gemmv_profile_enabled();
    if (profile_enabled) {
        g_last_profile = {};
    }
    const gemmv_force_isa_t force_mode = get_gemmv_force_isa();
    const bool force_amx_int8 = (force_mode == gemmv_force_isa_t::amx_int8);
    const bool force_amx_bf16 = (force_mode == gemmv_force_isa_t::amx_bf16);
    const bool force_vnni = (force_mode == gemmv_force_isa_t::vnni);
    const bool force_avx512 = (force_mode == gemmv_force_isa_t::avx512_fp32);
    const bool force_avx2 = (force_mode == gemmv_force_isa_t::avx2_fp32);
    const bool force_ref = (force_mode == gemmv_force_isa_t::ref_fp32);
    if (const char* trace = std::getenv("GEMMV_FORCE_ISA_TRACE")) {
        if (trace[0] != '\0' && force_mode != gemmv_force_isa_t::auto_mode) {
            std::fprintf(stderr, "[GEMMV][FORCE_ISA] mode=%s\n", gemmv_force_isa_to_cstr(force_mode));
        }
    }
    const bool disable_amx = [](){
        const char* env = std::getenv("GEMMV_DISABLE_AMX");
        return env && env[0] != '\0';
    }();
    // Repack cache: avoids repeated interleave->k64/k4 repacks across calls for same weights
    struct FreeDeleter { void operator()(uint8_t* p) const { if (p) free(p); } };
    struct PackedEntry {
        std::unique_ptr<uint8_t, FreeDeleter> buf{};
        std::vector<int32_t> sumW;
        std::vector<float> lane_scales;
        std::vector<int32_t> lane_zps;
        std::vector<float> lane_bias;
        uintptr_t scales_id = 0;
        uintptr_t zps_id = 0;
        uintptr_t bias_id = 0;
        quant_granularity_t lane_gran = quant_granularity_t::per_tensor;
        int lane_group = 0;
        size_t bytes = 0;
        size_t capacity = 0;
        size_t src_bytes = 0;
        uint64_t signature = 0;
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
    auto refresh_lane_meta = [](PackedEntry& pe,
                                int M_actual,
                                const float* scales,
                                const int32_t* zps,
                                const float* bias,
                                quant_granularity_t gran,
                                int group_size) {
        const uintptr_t sc_id = reinterpret_cast<uintptr_t>(scales);
        const uintptr_t zp_id = reinterpret_cast<uintptr_t>(zps);
        const uintptr_t bias_id = reinterpret_cast<uintptr_t>(bias);
        const bool need_zps = (zps != nullptr);
        const bool need_bias = (bias != nullptr);
        const bool need_update =
                pe.scales_id != sc_id ||
                pe.zps_id != zp_id ||
                pe.bias_id != bias_id ||
                pe.lane_gran != gran ||
                pe.lane_group != group_size ||
                pe.lane_scales.size() != static_cast<size_t>(pe.M_pad) ||
                (need_zps && pe.lane_zps.size() != static_cast<size_t>(pe.M_pad)) ||
                (!need_zps && !pe.lane_zps.empty()) ||
                (need_bias && pe.lane_bias.size() != static_cast<size_t>(pe.M_pad)) ||
                (!need_bias && !pe.lane_bias.empty());
        if (!need_update) {
            return;
        }
        pe.lane_scales.assign(pe.M_pad, 1.f);
        if (need_zps) pe.lane_zps.assign(pe.M_pad, 0);
        else pe.lane_zps.clear();
        if (need_bias) pe.lane_bias.assign(pe.M_pad, 0.f);
        else pe.lane_bias.clear();
        const int last_valid = (M_actual > 0) ? (M_actual - 1) : 0;
        const int eff_group = group_size > 0 ? group_size : 16;
        auto fetch_scale = [&](int ref) -> float {
            if (!scales) return 1.f;
            switch (gran) {
                case quant_granularity_t::per_tensor: return scales[0];
                case quant_granularity_t::per_channel: return scales[ref];
                case quant_granularity_t::per_group: return scales[ref / eff_group];
            }
            return 1.f;
        };
        auto fetch_zp = [&](int ref) -> int32_t {
            if (!zps) return 0;
            switch (gran) {
                case quant_granularity_t::per_tensor: return zps[0];
                case quant_granularity_t::per_channel: return zps[ref];
                case quant_granularity_t::per_group: return zps[ref / eff_group];
            }
            return 0;
        };
        auto fetch_bias = [&](int ref) -> float {
            if (!bias) return 0.f;
            switch (gran) {
                case quant_granularity_t::per_tensor: return bias[0];
                case quant_granularity_t::per_channel: return bias[ref];
                case quant_granularity_t::per_group: return bias[ref / eff_group];
            }
            return 0.f;
        };
        for (int m = 0; m < pe.M_pad; ++m) {
            const int ref = std::min(m, last_valid);
            pe.lane_scales[m] = fetch_scale(ref);
            if (need_zps) pe.lane_zps[m] = fetch_zp(ref);
            if (need_bias) pe.lane_bias[m] = fetch_bias(ref);
        }
        pe.scales_id = sc_id;
        pe.zps_id = zp_id;
        pe.bias_id = bias_id;
        pe.lane_gran = gran;
        pe.lane_group = group_size;
    };
    auto fingerprint = [](const uint8_t* data, size_t bytes) -> uint64_t {
        if (!data || bytes == 0) return 0;
        const size_t samples = std::min<size_t>(bytes, static_cast<size_t>(64));
        const size_t stride = std::max<size_t>(1, bytes / samples);
        uint64_t sig = bytes * 1469598103934665603ULL;
        for (size_t i = 0; i < samples; ++i) {
            const size_t idx = std::min(bytes - 1, i * stride);
            sig ^= static_cast<uint64_t>(data[idx]);
            sig *= 1099511628211ULL;
        }
        sig ^= static_cast<uint64_t>(data[bytes - 1]) << 32;
        return sig;
    };
    auto repack_k4_entry = [&](PackedEntry& pe, const uint8_t* base, int Mv, int Kv, int ld){
        const int M_blk = 16;
        const int K_grp = (Kv + 3)/4;
        const int M_pad = ((Mv + M_blk - 1)/M_blk)*M_blk;
        const size_t bytes = (size_t)M_pad * (size_t)K_grp * 4;
        const size_t blocks = (size_t)M_pad / M_blk;
        const size_t src_bytes = (size_t)ld * blocks;
        if (pe.capacity < bytes) {
            void* mem = nullptr;
            size_t size_al = ((bytes + 63)/64)*64;
            if (posix_memalign(&mem, 64, size_al)!=0 || !mem) mem = malloc(size_al);
            pe.buf.reset((uint8_t*)mem);
            pe.capacity = bytes;
        }
        pe.bytes = bytes;
        pe.ld_bytes = K_grp * 64;
        pe.M_pad = M_pad;
        pe.sumW.assign(M_pad, 0);
        repack_interleave_m16_to_k4_m16(pe.buf.get(), base, Mv, Kv, ld, M_blk, pe.sumW.data());
        pe.src_bytes = src_bytes;
        pe.signature = fingerprint(base, src_bytes);
    };
    auto repack_k64_entry = [&](PackedEntry& pe, const uint8_t* base, int Mv, int Kv, int ld){
        const int M_blk = 16;
        const int K_blk = 64;
        const int M_pad = ((Mv + M_blk - 1)/M_blk)*M_blk;
        const int K_grp = (Kv + K_blk - 1)/K_blk;
        const size_t bytes = (size_t)M_pad * (size_t)K_grp * (size_t)K_blk * 16;
        const size_t blocks = (size_t)M_pad / M_blk;
        const size_t src_bytes = (size_t)ld * blocks;
        if (pe.capacity < bytes) {
            void* mem=nullptr;
            size_t size_al = ((bytes + 63)/64)*64;
            if (posix_memalign(&mem, 64, size_al)!=0 || !mem) mem = malloc(size_al);
            pe.buf.reset((uint8_t*)mem);
            pe.capacity = bytes;
        }
        pe.bytes = bytes;
        pe.ld_bytes = K_grp * (K_blk * 16);
        pe.M_pad = M_pad;
        pe.sumW.assign(M_pad, 0);
        repack_interleave_m16_to_k64_m16_tile(pe.buf.get(), base, Mv, Kv, ld, M_blk, K_blk, pe.sumW.data());
        pe.src_bytes = src_bytes;
        pe.signature = fingerprint(base, src_bytes);
    };
    auto get_or_make_k4 = [&](const uint8_t* base, int Mv, int Kv, int ld)->PackedEntry&{
        std::lock_guard<std::mutex> lk(g_repack_mtx);
        RepackKey key{(uintptr_t)base, Mv, Kv, ld, 1};
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) {
            PackedEntry& pe = it->second;
            const size_t blocks = (size_t)pe.M_pad / 16;
            const size_t src_bytes = (size_t)ld * blocks;
            const uint64_t sig = fingerprint(base, src_bytes);
            if (pe.signature != sig) {
                repack_k4_entry(pe, base, Mv, Kv, ld);
            }
            return pe;
        }
        PackedEntry pe;
        repack_k4_entry(pe, base, Mv, Kv, ld);
        auto ins = g_repack_cache.emplace(key, std::move(pe));
        return ins.first->second;
    };
    auto get_or_make_k64_tile = [&](const uint8_t* base, int Mv, int Kv, int ld)->PackedEntry&{
        std::lock_guard<std::mutex> lk(g_repack_mtx);
        RepackKey key{(uintptr_t)base, Mv, Kv, ld, 2};
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) {
            PackedEntry& pe = it->second;
            const size_t blocks = (size_t)pe.M_pad / 16;
            const size_t src_bytes = (size_t)ld * blocks;
            const uint64_t sig = fingerprint(base, src_bytes);
            if (pe.signature != sig) {
                repack_k64_entry(pe, base, Mv, Kv, ld);
            }
            return pe;
        }
        PackedEntry pe;
        repack_k64_entry(pe, base, Mv, Kv, ld);
        auto ins = g_repack_cache.emplace(key, std::move(pe));
        return ins.first->second;
    };
    // Fast path 0: AMX INT8 (tile GEMV) with on-the-fly repack to K4 per M-block (per-tensor only)
    if (!disable_amx &&
        (force_mode == gemmv_force_isa_t::auto_mode || force_amx_int8) &&
        !accumulate && wtype == w_dtype_t::i8 && gran == quant_granularity_t::per_tensor && M >= 16 && K >= 64) {
        // Allow measurement scripts to gate AMX via env; default is auto-enable when supported
        const int M_blk = 16; const int K_blk = 64;
        auto &wk4_amx = get_or_make_k4(wq_packed, M, K, ld_w_bytes);
        refresh_lane_meta(wk4_amx, M, scales, zps, bias, gran, group_size);
        amx_lane_meta_t lane_meta{};
        lane_meta.scales = wk4_amx.lane_scales.empty() ? nullptr : wk4_amx.lane_scales.data();
        lane_meta.zps = wk4_amx.lane_zps.empty() ? nullptr : wk4_amx.lane_zps.data();
        lane_meta.bias = wk4_amx.lane_bias.empty() ? nullptr : wk4_amx.lane_bias.data();
        const amx_lane_meta_t* lane_meta_ptr = (lane_meta.scales || lane_meta.zps || lane_meta.bias) ? &lane_meta : nullptr;
        bool ok = run_gemmv_amx_i8u8_fp32(x, K, wk4_amx.buf.get(), M, wk4_amx.ld_bytes,
                                          scales, zps, y, bias, gran, group_size,
                                          wk4_amx.sumW.data(),
                                          lane_meta_ptr);
        if (ok) { if (kernel_name) *kernel_name = "amx_int8"; return; }
        // Try AMX BF16 path (Pack int8->bf16 per-tensor and compute via tdpbf16ps)
        if (!disable_amx && (force_mode == gemmv_force_isa_t::auto_mode || force_amx_bf16)) {
            Xbyak::util::Cpu cpu;
            if (cpu.has(Xbyak::util::Cpu::tAMX_BF16)) {
                const float s_w = scales ? scales[0] : 1.f;
                const int32_t zp_w = zps ? zps[0] : 0;
                const int M_pad = ((M + M_blk - 1)/M_blk)*M_blk;
                const int K_grp_b = (K + K_blk - 1)/K_blk;
                const size_t bytes_bf16 = (size_t)M_pad * (size_t)K_grp_b * (size_t)K_blk * (size_t)M_blk * sizeof(uint16_t);
                std::vector<uint16_t> Wbf(bytes_bf16/2);
                repack_interleave_m16_to_k64_m16_bf16(Wbf.data(), wq_packed, M, K, ld_w_bytes, s_w, zp_w, M_blk, K_blk);
                const int ld_bf16 = K_grp_b * (K_blk * 16 * 2);
                if (run_gemmv_amx_bf16_fp32(x, K, Wbf.data(), M, ld_bf16, y, bias)) {
                    if (kernel_name) {
                        *kernel_name = "amx_bf16";
                    }
                    return;
                }
            }
        }
    }

    // Forced AMX_INT8 even when weights are only interleave-packed: repack to K64 tiles on the fly.
    if (!disable_amx &&
        force_amx_int8 &&
        wtype == w_dtype_t::i8 &&
        !accumulate && M >= 16 && K >= 64) {
        const int M_blk = 16; const int K_blk = 64;
        auto &wk4_amx = get_or_make_k4(wq_packed, M, K, ld_w_bytes);
        refresh_lane_meta(wk4_amx, M, scales, zps, bias, gran, group_size);
        amx_lane_meta_t lane_meta{};
        lane_meta.scales = wk4_amx.lane_scales.empty() ? nullptr : wk4_amx.lane_scales.data();
        lane_meta.zps = wk4_amx.lane_zps.empty() ? nullptr : wk4_amx.lane_zps.data();
        lane_meta.bias = wk4_amx.lane_bias.empty() ? nullptr : wk4_amx.lane_bias.data();
        const amx_lane_meta_t* lane_meta_ptr = (lane_meta.scales || lane_meta.zps || lane_meta.bias) ? &lane_meta : nullptr;
        bool ok = run_gemmv_amx_i8u8_fp32(x, K, wk4_amx.buf.get(), M, wk4_amx.ld_bytes,
                                          scales, zps, y, bias, gran, group_size,
                                          wk4_amx.sumW.data(),
                                          lane_meta_ptr);
        if (ok) { if (kernel_name) *kernel_name = "amx_int8"; return; }
    }

    // Fast path 1: AVX-512 VNNI and i8 weights (automatically selected; Xbyak JIT only)
    if (wtype == w_dtype_t::i8 &&
        (force_mode == gemmv_force_isa_t::auto_mode || force_vnni) &&
        !force_avx512 && !force_avx2 && !force_ref && !force_amx_int8) {
        Xbyak::util::Cpu cpu;
        if (cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) && cpu.has(Xbyak::util::Cpu::tAVX512BW)) {
            struct XCache {
                std::vector<uint8_t> xq;
                float s_x = 1.f;
                int32_t zp_x = 128;
                int32_t sum_x_q = 0;
            };
            static thread_local XCache xc;
            if (xc.xq.size() != static_cast<size_t>(K)) {
                xc.xq.assign((size_t)K, 0);
            }
            std::chrono::steady_clock::time_point t_quant_begin;
            if (profile_enabled) {
                t_quant_begin = std::chrono::steady_clock::now();
            }
            quantize_u8_symmetric(x, K, xc.xq.data(), &xc.s_x, &xc.zp_x, &xc.sum_x_q);
            if (profile_enabled) {
                const auto t_quant_end = std::chrono::steady_clock::now();
                g_last_profile.quant_ns =
                    std::chrono::duration<double, std::nano>(t_quant_end - t_quant_begin).count();
            }

        auto& wk4 = get_or_make_k4(wq_packed, M, K, ld_w_bytes);
        refresh_lane_meta(wk4, M, scales, zps, bias, gran, group_size);
        gemmv_ukr_params_t params{};
        params.x = x;
        params.K = K;
            params.wq = wk4.buf.get();
            params.ld_w_bytes = wk4.ld_bytes;
            params.sumW_precomp = wk4.sumW.empty() ? nullptr : wk4.sumW.data();
            params.scales = scales;
            params.zps = zps;
            params.gran = gran;
            params.group_size = group_size;
            params.lane_scales = wk4.lane_scales.empty() ? nullptr : wk4.lane_scales.data();
            params.lane_zps = wk4.lane_zps.empty() ? nullptr : wk4.lane_zps.data();
            params.lane_bias = wk4.lane_bias.empty() ? nullptr : wk4.lane_bias.data();
            params.y = y;
            params.bias = const_cast<float*>(bias);
            params.M = M;
            params.M_tail = M % 16;
            params.accumulate = accumulate;
            params.m_base = 0;
            params.a_type = a_dtype_t::fp32;
            params.w_type = w_dtype_t::i8;
            params.w_layout = w_layout_t::k4_m16;
            params.x_q8 = xc.xq.data();
            params.x_scale = xc.s_x;
            params.x_zp = xc.zp_x;
            params.sum_x_q = xc.sum_x_q;
            params.N_expert = 1;
            params.fuse_gate = false;
            params.gate_scale = 1.f;
            params.act_kind = 0;

            const float x_scale_cached = params.x_scale;
            const int32_t x_zp_cached = params.x_zp;
            const int32_t sum_x_cached = params.sum_x_q;
            std::unique_ptr<GemmvKernel> kernel(create_gemmv_kernel(params));
            if (kernel_name) *kernel_name = kernel->name();
            std::chrono::steady_clock::time_point t_kernel_begin;
            if (profile_enabled) {
                t_kernel_begin = std::chrono::steady_clock::now();
            }
            (*kernel)(&params);
            if (profile_enabled) {
                const auto t_kernel_end = std::chrono::steady_clock::now();
                g_last_profile.kernel_ns =
                    std::chrono::duration<double, std::nano>(t_kernel_end - t_kernel_begin).count();
                g_last_profile.total_ns = g_last_profile.quant_ns + g_last_profile.kernel_ns;
            }

            if (const char* verify = std::getenv("GEMMV_VNNI_VERIFY"); verify && verify[0] != '\0') {
                const int M_blk = 16;
                const int K_grp = (K + 3) / 4;
                const int M_pad = wk4.M_pad;
                const int eff_group = group_size > 0 ? group_size : 16;
                auto fetch_scale_src = [&](int ref) -> float {
                    if (!scales) return 1.f;
                    switch (gran) {
                        case quant_granularity_t::per_tensor: return scales[0];
                        case quant_granularity_t::per_channel: return scales[ref];
                        case quant_granularity_t::per_group: return scales[ref / eff_group];
                    }
                    return 1.f;
                };
                auto fetch_zp_src = [&](int ref) -> int32_t {
                    if (!zps) return 0;
                    switch (gran) {
                        case quant_granularity_t::per_tensor: return zps[0];
                        case quant_granularity_t::per_channel: return zps[ref];
                        case quant_granularity_t::per_group: return zps[ref / eff_group];
                    }
                    return 0;
                };
                auto fetch_bias_src = [&](int ref) -> float {
                    if (!bias) return 0.f;
                    switch (gran) {
                        case quant_granularity_t::per_tensor: return bias[0];
                        case quant_granularity_t::per_channel: return bias[ref];
                        case quant_granularity_t::per_group: return bias[ref / eff_group];
                    }
                    return 0.f;
                };
                auto lane_fetch = [&](const std::vector<float>& buf, int idx, float fallback) -> float {
                    if (!buf.empty()) return buf[idx];
                    return fallback;
                };
                auto lane_fetch_zp = [&](const std::vector<int32_t>& buf, int idx, int32_t fallback) -> int32_t {
                    if (!buf.empty()) return buf[idx];
                    return fallback;
                };
                const uint8_t* k4_base = wk4.buf.get();
                const int32_t* sumW_ptr = wk4.sumW.empty() ? nullptr : wk4.sumW.data();
                auto sum_row_interleave = [&](int m_idx) -> int32_t {
                    if (m_idx < 0 || m_idx >= M) return 0;
                    const int block = m_idx / M_blk;
                    const int lane = m_idx % M_blk;
                    const uint8_t* blk_base = wq_packed + (size_t)block * (size_t)ld_w_bytes;
                    int32_t s = 0;
                    for (int k_idx = 0; k_idx < K; ++k_idx) {
                        const uint8_t* col = blk_base + (size_t)k_idx * M_blk;
                        s += static_cast<int32_t>(static_cast<int8_t>(col[lane]));
                    }
                    return s;
                };
                auto sum_from_k4 = [&](int m_idx) -> int32_t {
                    if (sumW_ptr && m_idx < M_pad) return sumW_ptr[m_idx];
                    return sum_row_interleave(m_idx);
                };
                auto dot_from_k4 = [&](int m_idx) -> int64_t {
                    if (m_idx < 0 || m_idx >= M) return 0;
                    const int block = m_idx / M_blk;
                    const int lane = m_idx % M_blk;
                    const uint8_t* block_base = k4_base + (size_t)block * (size_t)wk4.ld_bytes;
                    int64_t acc = 0;
                    for (int g = 0; g < K_grp; ++g) {
                        const uint8_t* row_bytes = block_base + (size_t)g * 64 + lane * 4;
                        const int k0 = g * 4;
                        for (int t = 0; t < 4; ++t) {
                            const int k_idx = k0 + t;
                            if (k_idx >= K) break;
                            const int32_t x_val = static_cast<int32_t>(xc.xq[k_idx]);
                            const int32_t w_val = static_cast<int8_t>(row_bytes[t]);
                            acc += static_cast<int64_t>(x_val) * static_cast<int64_t>(w_val);
                        }
                    }
                    return acc;
                };
                std::vector<float> y_ref(M, 0.f);
                for (int mi = 0; mi < M; ++mi) {
                    const int block = mi / M_blk;
                    const int lane = mi % M_blk;
                    const int lane_idx = block * M_blk + lane;
                    const float scale_lane = lane_fetch(wk4.lane_scales, lane_idx,
                                                        fetch_scale_src(std::min(mi, M - 1)));
                    const int32_t zp_lane = lane_fetch_zp(wk4.lane_zps, lane_idx,
                                                          fetch_zp_src(std::min(mi, M - 1)));
                    const float bias_lane = lane_fetch(wk4.lane_bias, lane_idx,
                                                       fetch_bias_src(std::min(mi, M - 1)));
                    const uint8_t* block_base = k4_base + (size_t)block * (size_t)wk4.ld_bytes;
                    int64_t acc = 0;
                    for (int g = 0; g < K_grp; ++g) {
                        const uint8_t* row_bytes = block_base + (size_t)g * 64 + lane * 4;
                        const int k0 = g * 4;
                        for (int t = 0; t < 4; ++t) {
                            const int k_idx = k0 + t;
                            if (k_idx >= K) break;
                            const int32_t x_val = static_cast<int32_t>(xc.xq[k_idx]);
                            const int32_t w_val = static_cast<int8_t>(row_bytes[t]);
                            acc += static_cast<int64_t>(x_val) * static_cast<int64_t>(w_val);
                        }
                    }
                    const int32_t sumW_lane = sum_from_k4(lane_idx);
                    const int32_t comp_int = (-xc.zp_x) * sumW_lane +
                                             zp_lane * (K * xc.zp_x - xc.sum_x_q);
                    const float scale = scale_lane * xc.s_x;
                    y_ref[mi] = static_cast<float>(acc) * scale +
                                static_cast<float>(comp_int) * scale +
                                bias_lane;
                }
                float max_abs = 0.f;
                float max_rel = 0.f;
                int worst_idx = -1;
                for (int mi = 0; mi < M; ++mi) {
                    const float ref = y_ref[mi];
                    const float got = y[mi];
                    const float diff = std::abs(ref - got);
                    const float rel = ref == 0.f ? diff : diff / std::abs(ref);
                    if (diff > max_abs) {
                        max_abs = diff;
                        max_rel = rel;
                        worst_idx = mi;
                    }
                }
                const int safe_idx = worst_idx >= 0 ? worst_idx : 0;
                const int64_t dot_layout = dot_from_k4(safe_idx);
                std::fprintf(stderr, "[GEMMV][VNNI_VERIFY] M=%d K=%d max_abs=%g max_rel=%g idx=%d ref=%g got=%g (x_scale=%g zp_x=%d sum_x=%d dot_jit_layout=%lld)\n",
                             M, K, max_abs, max_rel, worst_idx,
                             (worst_idx >= 0 ? y_ref[worst_idx] : 0.f),
                             (worst_idx >= 0 ? y[worst_idx] : 0.f),
                             x_scale_cached,
                             x_zp_cached,
                             sum_x_cached,
                             static_cast<long long>(dot_layout));
                const int32_t row0_sum = sum_from_k4(0);
                const int32_t row0_ref = sum_row_interleave(0);
                const int32_t worst_sum = sum_from_k4(safe_idx);
                const int32_t worst_ref = sum_row_interleave(safe_idx);
                std::fprintf(stderr, "[GEMMV][VNNI_VERIFY] sumW_row0=%d (ref=%d) sumW_idx=%d=%d (ref=%d)\n",
                             row0_sum, row0_ref, worst_idx,
                             worst_sum, worst_ref);
                #if defined(__AVX512F__) && defined(__AVX512VNNI__)
                if (const char* sim_env = std::getenv("GEMMV_VNNI_SIM")) {
                    const int M_blk = 16;
                    int sim_block = std::max(0, std::atoi(sim_env));
                    const int total_blocks = wk4.M_pad / M_blk;
                    if (sim_block < total_blocks) {
                        const int K_grp = (K + 3) / 4;
                        const uint8_t* block_base = wk4.buf.get() + static_cast<size_t>(sim_block) * static_cast<size_t>(wk4.ld_bytes);
                        __m512i vacc = _mm512_setzero_si512();
                        for (int g = 0; g < K_grp; ++g) {
                            const uint8_t* row_bytes = block_base + static_cast<size_t>(g) * 64;
                            __m512i vw = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_bytes));
                            uint32_t xb = 0;
                            std::memcpy(&xb, xc.xq.data() + g * 4, sizeof(uint32_t));
                            __m512i vx = _mm512_set1_epi32(static_cast<int32_t>(xb));
                            vacc = _mm512_dpbusd_epi32(vacc, vx, vw);
                        }
                        alignas(64) int32_t sim_acc[16];
                        _mm512_store_si512(reinterpret_cast<__m512i*>(sim_acc), vacc);
                        std::fprintf(stderr, "[GEMMV][VNNI_SIM] block=%d acc:", sim_block);
                        for (int i = 0; i < M_blk; ++i) std::fprintf(stderr, " %d", sim_acc[i]);
                        std::fprintf(stderr, "\n");
                    }
                }
#else
                if (std::getenv("GEMMV_VNNI_SIM")) {
                    std::fprintf(stderr, "[GEMMV][VNNI_SIM] host AVX-512 VNNI not available in this build\n");
                }
#endif
                if (const char* dump_block_env = std::getenv("GEMMV_VNNI_DUMP_BLOCK")) {
                    const int M_blk = 16;
                    int dbg_block = dump_block_env[0] ? std::max(0, std::atoi(dump_block_env)) : -1;
                    if (dbg_block < 0 && worst_idx >= 0) dbg_block = worst_idx / M_blk;
                    if (dbg_block >= 0) {
                        alignas(64) int32_t dbg_acc[16]{};
                        alignas(64) int32_t dbg_sumw[16]{};
                        std::vector<float> y_dbg(static_cast<size_t>(wk4.M_pad), 0.f);
                        const float s_w_scalar = params.scales ? params.scales[0] : 1.f;
                        const int32_t zp_w_scalar = params.zps ? params.zps[0] : 0;
                        const float bias_scalar = bias ? bias[0] : 0.f;
                        const int32_t* sumW_ptr = wk4.sumW.empty() ? nullptr : wk4.sumW.data();
                        const bool ok_dump = run_gemmv_vnni_i8u8_fp32(xc.xq.data(), K,
                                                                      wk4.buf.get(), M, wk4.ld_bytes,
                                                                      s_w_scalar, zp_w_scalar,
                                                                      xc.s_x, xc.zp_x,
                                                                      y_dbg.data(), bias_scalar,
                                                                      sumW_ptr,
                                                                      dbg_block, dbg_acc, dbg_sumw,
                                                                      /*dbg_dump_only=*/0);
                        std::fprintf(stderr, "[GEMMV][VNNI_DUMP] ok=%d block=%d\n", ok_dump ? 1 : 0, dbg_block);
                        if (ok_dump) {
                            std::fprintf(stderr, "  acc:");
                            for (int i = 0; i < M_blk; ++i) std::fprintf(stderr, " %d", dbg_acc[i]);
                            std::fprintf(stderr, "\n  acc_ref:");
                            for (int i = 0; i < M_blk; ++i) {
                                const int lane = dbg_block * M_blk + i;
                                std::fprintf(stderr, " %lld", static_cast<long long>(dot_from_k4(lane)));
                            }
                            std::fprintf(stderr, "\n  sumW:");
                            for (int i = 0; i < M_blk; ++i) std::fprintf(stderr, " %d", dbg_sumw[i]);
                            std::fprintf(stderr, "\n");
                        }
                    }
                }
            }
            return;
        }
    }

    // Fallback to JIT/ref kernel
    gemmv_ukr_params_t p{};
    p.x = x; p.K = K; p.wq = wq_packed; p.ld_w_bytes = ld_w_bytes;
    p.scales = scales; p.zps = zps; p.gran = gran; p.group_size = group_size;
    p.y = y; p.bias = const_cast<float*>(bias); p.M = M; p.accumulate = accumulate;
    p.a_type = a_dtype_t::fp32; p.w_type = wtype; p.fuse_gate = false; p.gate_scale = 1.f; p.act_kind = 0;

    // Optional pre-quantized X for int8 per-tensor path (used by AMX/VNNI)
    std::vector<uint8_t> xq_tmp;
    if (wtype == w_dtype_t::i8 && gran == quant_granularity_t::per_tensor) {
        xq_tmp.resize(static_cast<size_t>(K));
        float amax = 0.f;
        for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[k]));
        float s_x = (amax > 0.f) ? (amax / 127.f) : 1.f;
        int32_t zp_x = 128;
        int32_t sum_x_q = 0;
        for (int k = 0; k < K; ++k) {
            int v = static_cast<int>(std::lrintf(x[k] / s_x)) + zp_x;
            v = std::min(255, std::max(0, v));
            xq_tmp[static_cast<size_t>(k)] = static_cast<uint8_t>(v);
            sum_x_q += v;
        }
        p.x_q8 = xq_tmp.data();
        p.x_scale = s_x;
        p.x_zp = zp_x;
        p.sum_x_q = sum_x_q;
    }

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

static inline void quantize_u8_symmetric(const float* x, int K, uint8_t* xq,
                                         float* s_x_out, int32_t* zp_out, int32_t* sum_x_out) {
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
    *s_x_out = s;
    *zp_out = zp;
    if (sum_x_out) {
        int64_t acc = 0;
        for (int i = 0; i < K; ++i) acc += (int32_t)xq[i];
        *sum_x_out = static_cast<int32_t>(acc);
    }
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
    quantize_u8_symmetric(x_fp32, K, xq.data(), &s_x, &zp_x, nullptr);
    bool dump_only_flag = false;
    if (const char* dump_env = std::getenv("GEMMV_VNNI_DUMP_ONLY")) {
        dump_only_flag = dump_env[0] != '\0' && dump_env[0] != '0';
    }
    const int dump_only = dump_only_flag ? 1 : 0;
    // Optional: pure C++ stub for dump-only debug (env GEMMV_VNNI_STUB=1)
    if (dump_only) {
        if (const char* st = std::getenv("GEMMV_VNNI_STUB"); st && std::string(st) == "1") {
            // Compute for block 0, group 0 only
            const int M_blk = 16;
            auto get_w4 = [&](int bi,int g,int row){ return wq_k4 + (size_t)bi*ld_w_gbytes + (size_t)g*64 + (size_t)row*4; };
            // Quantize X to u8 (we already have xq)
            std::vector<uint8_t> xq(K);
            float s_x; int32_t zp_x; quantize_u8_symmetric(x_fp32, K, xq.data(), &s_x, &zp_x, nullptr);
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
    return run_gemmv_vnni_i8u8_fp32(xq.data(), K, wq_k4, M, ld_w_gbytes,
                                    s_w, zp_w, s_x, zp_x, y, bias0, sumW_precomp,
                                    dbg_block, dbg_acc, dbg_sumw, dump_only);
}

} // namespace ov::intel_cpu::x64::gemmv_jit
