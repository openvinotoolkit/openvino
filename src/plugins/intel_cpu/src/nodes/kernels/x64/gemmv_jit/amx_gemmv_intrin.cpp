// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// AMX INT8 GEMV (u8 X, s8 W) -> fp32 Y microkernel (N=1)
#include "amx_gemmv_intrin.hpp"
#include "xbyak/xbyak_util.h"
#include <immintrin.h>
#include <cstdio>
#include <string>
#include <errno.h>
#if defined(__x86_64__)
#include <sys/syscall.h>
#include <asm/prctl.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>

namespace ov::intel_cpu::x64::gemmv_jit {

// Helper: per-lane metadata getters
static inline float get_scale_lane(const float* s, int idx, quant_granularity_t gran, int group_size) {
    if (!s) return 1.f;
    if (gran == quant_granularity_t::per_tensor) return s[0];
    if (gran == quant_granularity_t::per_channel) return s[idx];
    const int gs = group_size > 0 ? group_size : 16;
    return s[idx / gs];
}
static inline int32_t get_zp_lane(const int32_t* z, int idx, quant_granularity_t gran, int group_size) {
    if (!z) return 0;
    if (gran == quant_granularity_t::per_tensor) return z[0];
    if (gran == quant_granularity_t::per_channel) return z[idx];
    const int gs = group_size > 0 ? group_size : 16;
    return z[idx / gs];
}
static inline float get_bias_lane(const float* b, int idx, quant_granularity_t gran, int group_size) {
    if (!b) return 0.f;
    if (gran == quant_granularity_t::per_tensor) return b[0];
    if (gran == quant_granularity_t::per_channel) return b[idx];
    const int gs = group_size > 0 ? group_size : 16;
    return b[idx / gs];
}

// Forward decl for smoke test
static bool amx_smoke_test();

static bool enable_amx_for_thread() {
#if defined(__x86_64__)
    static thread_local bool init = false;
    static thread_local bool ok = false;
    static thread_local long last_r1 = 0;
    static thread_local long last_r2 = 0;
    static thread_local int last_errno1 = 0;
    static thread_local int last_errno2 = 0;
    if (init) return ok;
    init = true;
    // Request permission for XTILECFG (17) and XTILEDATA (18)
    errno = 0;
    long r1 = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 17);
    last_errno1 = errno; last_r1 = r1;
    if (r1 != 0) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] arch_prctl(ARCH_REQ_XCOMP_PERM,17) failed: ret=%ld errno=%d (%s)\n",
                r1, last_errno1, std::strerror(last_errno1));
        }
        return ok = false;
    }
    errno = 0;
    long r2 = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18);
    last_errno2 = errno; last_r2 = r2;
    if (r2 != 0) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] arch_prctl(ARCH_REQ_XCOMP_PERM,18) failed: ret=%ld errno=%d (%s)\n",
                r2, last_errno2, std::strerror(last_errno2));
        }
        return ok = false;
    }
    if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
        if (std::string(d) == "1") {
            unsigned long long xcr0 = _xgetbv(0);
            int b17 = (xcr0 & (1ull<<17)) ? 1 : 0;
            int b18 = (xcr0 & (1ull<<18)) ? 1 : 0;
            Xbyak::util::Cpu cpu;
            std::fprintf(stderr,
                         "[GEMMV-AMX-DIAG] arch_prctl ok; XCR0.TILECFG(bit17)=%d XTILEDATA(bit18)=%d; ISA: AMX_TILE=%d AMX_INT8=%d\n",
                         b17, b18, cpu.has(Xbyak::util::Cpu::tAMX_TILE)?1:0, cpu.has(Xbyak::util::Cpu::tAMX_INT8)?1:0);
        }
    }
    return ok = true;
#else
    return false;
#endif
}

#if defined(__GNUC__)
__attribute__((target("amx-int8,amx-tile,avx512bw,avx512vnni,avx512f")))
#endif
static inline void amx_epilogue_store_fp32(int valid, int base_idx,
                                           float* yb,
                                           const int32_t* Cbuf_row0,
                                           const int32_t* Sbuf_row0,
                                           const int32_t* sumW_precomp,
                                           int K, int32_t zp_x, int32_t sum_x_q,
                                           const float* scales, const int32_t* zps, const float* bias,
                                           quant_granularity_t gran, int group_size,
                                           float s_x) {
    __mmask16 km = valid >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << valid) - 1u);
    __m512i acc_v = _mm512_maskz_loadu_epi32(km, Cbuf_row0);
    __m512i sumw_v;
    if (sumW_precomp) {
        sumw_v = _mm512_maskz_loadu_epi32(km, sumW_precomp + base_idx);
    } else {
        sumw_v = _mm512_maskz_loadu_epi32(km, Sbuf_row0);
    }
    __m512i zpw_v = _mm512_setzero_si512();
    if (zps) {
        alignas(64) int32_t zpw_buf[16];
        if (gran == quant_granularity_t::per_tensor) {
            for (int m = 0; m < 16; ++m) zpw_buf[m] = zps[0];
        } else if (gran == quant_granularity_t::per_channel) {
            for (int m = 0; m < 16; ++m) zpw_buf[m] = zps[base_idx + m];
        } else {
            const int gs = group_size > 0 ? group_size : 16;
            for (int m = 0; m < 16; ++m) { int g = (base_idx + m) / gs; zpw_buf[m] = zps[g]; }
        }
        zpw_v = _mm512_maskz_loadu_epi32(km, zpw_buf);
    }
    __m512i neg_zpx_v = _mm512_set1_epi32(-zp_x);
    __m512i comp_v = _mm512_mullo_epi32(neg_zpx_v, sumw_v);
    __m512i c_v = _mm512_set1_epi32(K * zp_x - sum_x_q);
    __m512i zpw_term = _mm512_mullo_epi32(zpw_v, c_v);
    comp_v = _mm512_add_epi32(comp_v, zpw_term);
    __m512 yf = _mm512_cvtepi32_ps(_mm512_add_epi32(acc_v, comp_v));
    __m512 s_v;
    if (!scales) {
        s_v = _mm512_set1_ps(1.f * s_x);
    } else if (gran == quant_granularity_t::per_tensor) {
        s_v = _mm512_set1_ps(scales[0] * s_x);
    } else {
        alignas(64) float sc_buf[16];
        if (gran == quant_granularity_t::per_channel) {
            for (int m = 0; m < 16; ++m) sc_buf[m] = scales[base_idx + m] * s_x;
        } else {
            const int gs = group_size > 0 ? group_size : 16;
            for (int m = 0; m < 16; ++m) { int g = (base_idx + m) / gs; sc_buf[m] = scales[g] * s_x; }
        }
        s_v = _mm512_maskz_loadu_ps(km, sc_buf);
    }
    yf = _mm512_mul_ps(yf, s_v);
    if (bias) {
        __m512 b_v;
        if (gran == quant_granularity_t::per_tensor) {
            b_v = _mm512_set1_ps(bias[0]);
        } else {
            alignas(64) float b_buf[16];
            if (gran == quant_granularity_t::per_channel) {
                for (int m = 0; m < 16; ++m) b_buf[m] = bias[base_idx + m];
            } else {
                const int gs = group_size > 0 ? group_size : 16;
                for (int m = 0; m < 16; ++m) { int g = (base_idx + m) / gs; b_buf[m] = bias[g]; }
            }
            b_v = _mm512_maskz_loadu_ps(km, b_buf);
        }
        yf = _mm512_add_ps(yf, b_v);
    }
    bool do_nt = true;
    if (valid >= 16 && do_nt && (((uintptr_t)yb) & 63u) == 0u) _mm512_stream_ps(yb, yf);
    else _mm512_mask_storeu_ps(yb, km, yf);
}

#if defined(__GNUC__)
__attribute__((target("amx-int8,amx-tile,avx512bw,avx512vnni,avx512f")))
#endif
static bool amx_kernel_u8s8_fp32_impl_xq(const uint8_t* xq, int K, int32_t sum_x_q,
                                         float s_x, int32_t zp_x,
                                         const uint8_t* wq_k64, int M, int ld_w_kbytes,
                                         const float* scales, const int32_t* zps,
                                         float* y, const float* bias,
                                         quant_granularity_t gran, int group_size,
                                         const int32_t* sumW_precomp) {

    const int M_blk = 16;
    const int K_blk = 64;
    const int B_row_bytes = 16;                 // 16 lanes per k-row
    const int B_group_bytes = K_blk * B_row_bytes; // bytes per K64 group per M-block
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;

    // Tile config
    struct alignas(64) tilecfg_t {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    // tmm0: C (16x16, s32) -> colsb = 16*4 = 64
    cfg.colsb[0] = 64; cfg.rows[0] = 16;
    // tmm1: A (16x64, u8) -> colsb = 64
    cfg.colsb[1] = 64; cfg.rows[1] = 16;
    // tmm2: B (64x16, s8) -> colsb = 16
    cfg.colsb[2] = 16; cfg.rows[2] = 64;
    // tmm3: B1 (64x16, s8) when unroll >= 2
    cfg.colsb[3] = 16; cfg.rows[3] = 64;
    // tmm4: B2 (64x16, s8) when unroll == 4
    cfg.colsb[4] = 16; cfg.rows[4] = 64;
    // tmm5: B3 (64x16, s8) when unroll == 4
    cfg.colsb[5] = 16; cfg.rows[5] = 64;
    // tmm6: SumW accumulator (like C)
    cfg.colsb[6] = 64; cfg.rows[6] = 16;
    // tmm7: A_ones (16x64, u8)
    cfg.colsb[7] = 64; cfg.rows[7] = 16;
    _tile_loadconfig(&cfg);

    // Prepare A tile inputs as single 64B rows and broadcast into tiles via stride=0 loads
    alignas(64) uint8_t A_row[64];
    alignas(64) uint8_t A_ones_row[64];
    std::memset(A_ones_row, 0x01, sizeof(A_ones_row));

    bool dbg = false;
    // Deterministic small-M friendly defaults
    int KU = (K >= 1024) ? 4 : (K >= 512) ? 2 : 1;
    int PFD = (M <= 512) ? 4 : 2;
    const bool do_db = true;
    auto run_block = [&](int bi, int valid){
        const uint8_t* wblk = wq_k64 + (size_t)bi * (size_t)ld_w_kbytes;
        const uint8_t* wend = wblk + (size_t)ld_w_kbytes;
        // zero accumulators
        _tile_zero(0); // C
        const bool need_sumW = (sumW_precomp == nullptr);
        if (need_sumW) {
            _tile_zero(6); // SumW
            // Load a tile of ones by broadcasting a single 64B row (stride=0)
            _tile_loadd(7, A_ones_row, 0);
        }
        // Iterate K-groups with optional 2/4-stage pipeline
        const int K_grp = (K + K_blk - 1) / K_blk;
        const bool dblA = !need_sumW; // reuse tmm6 as A when sumW is precomputed
        bool pipe4 = (M <= 512);
        bool pipe2 = (!pipe4 && (M <= 1024));
        // Prefetch locality hint (0..3); default: high for small/medium M, low for large M
        int PFL = (M <= 512 ? 3 : 1);
        auto pf_lvl = [](const void* p, int lvl){
            switch (lvl) {
                default: __builtin_prefetch(p, 0, 0); break;
                case 1:  __builtin_prefetch(p, 0, 1); break;
                case 2:  __builtin_prefetch(p, 0, 2); break;
                case 3:  __builtin_prefetch(p, 0, 3); break;
            }
        };
        if (pipe4) {
            for (int g = 0; g < K_grp; g += 4) {
                for (int t = 0; t < 4 && (g + t) < K_grp; ++t) {
                    const int gi = g + t;
                    const int kx = gi * K_blk; const int kkx = std::min(K_blk, K - kx);
                    if (PFD > 0 && gi + PFD < K_grp) {
                        const uint8_t* Bpf = wblk + (size_t)(gi + PFD) * (size_t)B_group_bytes;
                        pf_lvl((const void*)Bpf, PFL);
                        const uint8_t* Xpf = xq + (size_t)(gi + PFD) * (size_t)K_blk;
                        pf_lvl((const void*)Xpf, (PFL>=1?1:0));
                    }
                    // Prepare a single 64B row and broadcast into A tile via stride=0
                    std::memset(A_row, (uint8_t)zp_x, sizeof(A_row));
                    if (kkx > 0) std::memcpy(A_row, xq + kx, (size_t)kkx);
                    bool useA6 = (dblA && ((gi & 1) != 0));
                    if (useA6) _tile_loadd(6, A_row, 0); else _tile_loadd(1, A_row, 0);
                    const uint8_t* Bi = wblk + (size_t)gi * (size_t)B_group_bytes;
                    if (dbg && (Bi < wblk || Bi + (size_t)(K_blk* M_blk) > wend)) return;
                    switch (t) {
                        case 0:
                            _tile_loadd(2, Bi, 16);
                            if (useA6) _tile_dpbusd(0, 6, 2); else _tile_dpbusd(0, 1, 2);
                            if (need_sumW) _tile_dpbusd(6, 7, 2);
                            break;
                        case 1:
                            _tile_loadd(3, Bi, 16);
                            if (useA6) _tile_dpbusd(0, 6, 3); else _tile_dpbusd(0, 1, 3);
                            if (need_sumW) _tile_dpbusd(6, 7, 3);
                            break;
                        case 2:
                            _tile_loadd(4, Bi, 16);
                            if (useA6) _tile_dpbusd(0, 6, 4); else _tile_dpbusd(0, 1, 4);
                            if (need_sumW) _tile_dpbusd(6, 7, 4);
                            break;
                        default:
                            _tile_loadd(5, Bi, 16);
                            if (useA6) _tile_dpbusd(0, 6, 5); else _tile_dpbusd(0, 1, 5);
                            if (need_sumW) _tile_dpbusd(6, 7, 5);
                            break;
                    }
                }
            }
        } else if (pipe2) {
            int g = 0;
            for (; g + 1 < K_grp; g += 2) {
                // Group g
                int k0 = g * K_blk; int kk0 = std::min(K_blk, K - k0);
                if (PFD > 0 && g + PFD < K_grp) {
                    const uint8_t* Bpf = wblk + (size_t)(g + PFD) * (size_t)B_group_bytes;
                    pf_lvl((const void*)Bpf, PFL);
                    const uint8_t* Xpf = xq + (size_t)(g + PFD) * (size_t)K_blk;
                    pf_lvl((const void*)Xpf, (PFL>=1?1:0));
                }
                std::memset(A_row, (uint8_t)zp_x, sizeof(A_row));
                if (kk0 > 0) std::memcpy(A_row, xq + k0, (size_t)kk0);
                bool useA6_0 = (dblA && (g & 1));
                if (useA6_0) _tile_loadd(6, A_row, 0); else _tile_loadd(1, A_row, 0);
                const uint8_t* B0 = wblk + (size_t)g * (size_t)B_group_bytes;
                if (dbg && (B0 < wblk || B0 + (size_t)(K_blk* M_blk) > wend)) return;
                _tile_loadd(2, B0, 16);
                if (useA6_0) _tile_dpbusd(0, 6, 2); else _tile_dpbusd(0, 1, 2);
                if (need_sumW) _tile_dpbusd(6, 7, 2);
                // Group g+1
                int k1 = (g + 1) * K_blk; int kk1 = std::min(K_blk, K - k1);
                if (PFD > 0 && g + 1 + PFD < K_grp) {
                    const uint8_t* Bpf2 = wblk + (size_t)(g + 1 + PFD) * (size_t)B_group_bytes;
                    pf_lvl((const void*)Bpf2, PFL);
                    const uint8_t* Xpf2 = xq + (size_t)(g + 1 + PFD) * (size_t)K_blk;
                    pf_lvl((const void*)Xpf2, (PFL>=1?1:0));
                }
                std::memset(A_row, (uint8_t)zp_x, sizeof(A_row));
                if (kk1 > 0) std::memcpy(A_row, xq + k1, (size_t)kk1);
                bool useA6_1 = (dblA && ((g + 1) & 1));
                if (useA6_1) _tile_loadd(6, A_row, 0); else _tile_loadd(1, A_row, 0);
                const uint8_t* B1 = wblk + (size_t)(g + 1) * (size_t)B_group_bytes;
                if (dbg && (B1 < wblk || B1 + (size_t)(K_blk* M_blk) > wend)) return;
                _tile_loadd(3, B1, 16);
                if (useA6_1) _tile_dpbusd(0, 6, 3); else _tile_dpbusd(0, 1, 3);
                if (need_sumW) _tile_dpbusd(6, 7, 3);
            }
            if (g < K_grp) {
                // Tail one group
                int k0 = g * K_blk; int kk0 = std::min(K_blk, K - k0);
                std::memset(A_row, (uint8_t)zp_x, sizeof(A_row));
                if (kk0 > 0) std::memcpy(A_row, xq + k0, (size_t)kk0);
                bool useA6_0 = (dblA && (g & 1));
                if (useA6_0) _tile_loadd(6, A_row, 0); else _tile_loadd(1, A_row, 0);
                const uint8_t* B0 = wblk + (size_t)g * (size_t)B_group_bytes;
                if (dbg && (B0 < wblk || B0 + (size_t)(K_blk* M_blk) > wend)) return;
                _tile_loadd(2, B0, 16);
                if (useA6_0) _tile_dpbusd(0, 6, 2); else _tile_dpbusd(0, 1, 2);
                if (need_sumW) _tile_dpbusd(6, 7, 2);
            }
        } else {
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * K_blk;
            const int kk0 = std::min(K_blk, K - k0);
            // Prefetch future tiles
            if (PFD > 0) {
                if (g + PFD < K_grp) {
                    const uint8_t* Bpf = wblk + (size_t)(g + PFD) * (size_t)B_group_bytes;
                    pf_lvl((const void*)Bpf, PFL);
                    const uint8_t* Xpf = xq + (size_t)(g + PFD) * (size_t)K_blk;
                    pf_lvl((const void*)Xpf, (PFL>=1?1:0));
                }
                if (do_db && g + PFD + 1 < K_grp) {
                    const uint8_t* Bpf2 = wblk + (size_t)(g + PFD + 1) * (size_t)B_group_bytes;
                    pf_lvl((const void*)Bpf2, (PFL>=1?1:0));
                    const uint8_t* Xpf2 = xq + (size_t)(g + PFD + 1) * (size_t)K_blk;
                    pf_lvl((const void*)Xpf2, (PFL>=1?1:0));
                }
            }
            // Build and load A tile (alternate 1 and 6 if available). Broadcast single 64B row via stride=0.
            std::memset(A_row, (uint8_t)zp_x, sizeof(A_row));
            if (kk0 > 0) std::memcpy(A_row, xq + k0, (size_t)kk0);
            const bool useA6 = (dblA && (g & 1));
            if (useA6) _tile_loadd(6, A_row, 0); else _tile_loadd(1, A_row, 0);
            // Load B and compute
            const uint8_t* Bptr = wblk + (size_t)g * (size_t)B_group_bytes;
            if (dbg && (Bptr < wblk || Bptr + (size_t)(K_blk* M_blk) > wend)) return;
            int bsel = g % KU;
            switch (bsel) {
                case 0:
                    _tile_loadd(2, Bptr, 16);
                    if (useA6) _tile_dpbusd(0, 6, 2); else _tile_dpbusd(0, 1, 2);
                    if (need_sumW) _tile_dpbusd(6, 7, 2);
                    break;
                case 1:
                    _tile_loadd(3, Bptr, 16);
                    if (useA6) _tile_dpbusd(0, 6, 3); else _tile_dpbusd(0, 1, 3);
                    if (need_sumW) _tile_dpbusd(6, 7, 3);
                    break;
                case 2:
                    _tile_loadd(4, Bptr, 16);
                    if (useA6) _tile_dpbusd(0, 6, 4); else _tile_dpbusd(0, 1, 4);
                    if (need_sumW) _tile_dpbusd(6, 7, 4);
                    break;
                default:
                    _tile_loadd(5, Bptr, 16);
                    if (useA6) _tile_dpbusd(0, 6, 5); else _tile_dpbusd(0, 1, 5);
                    if (need_sumW) _tile_dpbusd(6, 7, 5);
                    break;
            }
        }
        }
        // store C and SumW tiles
        alignas(64) int32_t Cbuf[16 * 16];
        alignas(64) int32_t Sbuf[16 * 16];
        _tile_stored(0, Cbuf, 64);
        if (!sumW_precomp) _tile_stored(6, Sbuf, 64);
        // Vectorized epilogue using row 0 across 16 columns
        float* yb = y + bi * M_blk;
        const int base_idx = bi * M_blk;
        amx_epilogue_store_fp32(valid, base_idx,
                                yb,
                                Cbuf, Sbuf,
                                sumW_precomp,
                                K, zp_x, sum_x_q,
                                scales, zps, bias,
                                gran, group_size,
                                s_x);
    };

    for (int bi = 0; bi < M_full; ++bi) run_block(bi, 16);
    if (M_tail) run_block(M_full, M_tail);
    _tile_release();
    return true;
}

static bool amx_kernel_u8s8_fp32_impl(const float* x_fp32, int K,
                                      const uint8_t* wq_k64, int M, int ld_w_kbytes,
                                      const float* scales, const int32_t* zps,
                                      float* y, const float* bias,
                                      quant_granularity_t gran, int group_size,
                                      const int32_t* sumW_precomp) {
    // Quantize X per-tensor u8 and compute sum_x_q
    std::vector<uint8_t> xq(K);
    float s_x = 1.f; int32_t zp_x = 128; int32_t sum_x_q = 0;
    {
        float amax = 0.f; for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x_fp32[k]));
        s_x = (amax > 0.f) ? (amax / 127.f) : 1.f; zp_x = 128;
        for (int k = 0; k < K; ++k) { int v = (int)std::lrintf(x_fp32[k] / s_x) + zp_x; v = std::min(255, std::max(0, v)); xq[k] = (uint8_t)v; sum_x_q += v; }
    }
    return amx_kernel_u8s8_fp32_impl_xq(xq.data(), K, sum_x_q, s_x, zp_x,
                                        wq_k64, M, ld_w_kbytes,
                                        scales, zps, y, bias, gran, group_size,
                                        sumW_precomp);
}

bool run_gemmv_amx_i8u8_fp32(const float* x_fp32, int K,
                             const uint8_t* wq_k64, int M, int ld_w_kbytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran, int group_size,
                             const int32_t* sumW_precomp) {
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAMX_TILE) || !cpu.has(Xbyak::util::Cpu::tAMX_INT8)) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] CPU lacks AMX support: AMX_TILE=%d AMX_INT8=%d\n",
                cpu.has(Xbyak::util::Cpu::tAMX_TILE)?1:0, cpu.has(Xbyak::util::Cpu::tAMX_INT8)?1:0);
        }
        return false;
    }
    // Request AMX permission first, then verify XCR0 state
    if (!enable_amx_for_thread()) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] enable_amx_for_thread() failed; see previous messages.\n");
        }
        return false;
    }
    unsigned long long xcr0 = _xgetbv(0);
    if (((xcr0 & (1ull<<17)) == 0ull) || ((xcr0 & (1ull<<18)) == 0ull)) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] XCR0 missing bits: TILECFG=%d XTILEDATA=%d (xcr0=0x%llx)\n",
                (int)((xcr0>>17)&1ull), (int)((xcr0>>18)&1ull), xcr0);
        }
        return false;
    }
    return amx_kernel_u8s8_fp32_impl(x_fp32, K, wq_k64, M, ld_w_kbytes,
                                     scales, zps, y, bias, gran, group_size,
                                     sumW_precomp);
}

#if defined(__GNUC__)
__attribute__((target("amx-int8,amx-tile,avx512bw,avx512vnni,avx512f")))
#endif
static bool amx_smoke_test() {
    // Configure minimal tiles and execute a single dpbusd over zeroed buffers.
    struct alignas(64) tilecfg_t {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } cfg{};
    cfg.palette_id = 1;
    cfg.colsb[0] = 64; cfg.rows[0] = 16; // C 16x16 s32
    cfg.colsb[1] = 64; cfg.rows[1] = 16; // A 16x64 u8
    cfg.colsb[2] = 16; cfg.rows[2] = 64; // B 64x16 s8
    _tile_loadconfig(&cfg);
    alignas(64) uint8_t A[16*64] = {0};
    alignas(64) uint8_t B[64*16] = {0};
    alignas(64) int32_t C[16*16] = {0};
    _tile_zero(0);
    _tile_loadd(1, A, 64);
    _tile_loadd(2, B, 16);
    _tile_dpbusd(0, 1, 2);
    _tile_stored(0, C, 64);
    _tile_release();
    return true;
}

} // namespace ov::intel_cpu::x64::gemmv_jit

namespace ov::intel_cpu::x64::gemmv_jit {

bool run_gemmv_amx_i8u8_fp32_xq(const uint8_t* xq, int K, int32_t sum_x_q,
                                const uint8_t* wq_k64, int M, int ld_w_kbytes,
                                const float* scales, const int32_t* zps,
                                float* y, const float* bias,
                                quant_granularity_t gran, int group_size,
                                const int32_t* sumW_precomp) {
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAMX_TILE) || !cpu.has(Xbyak::util::Cpu::tAMX_INT8)) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] CPU lacks AMX support: AMX_TILE=%d AMX_INT8=%d\n",
                cpu.has(Xbyak::util::Cpu::tAMX_TILE)?1:0, cpu.has(Xbyak::util::Cpu::tAMX_INT8)?1:0);
        }
        return false;
    }
    if (!enable_amx_for_thread()) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] enable_amx_for_thread() failed; see previous messages.\n");
        }
        return false;
    }
    unsigned long long xcr0 = _xgetbv(0);
    if (((xcr0 & (1ull<<17)) == 0ull) || ((xcr0 & (1ull<<18)) == 0ull)) {
        if (const char* d = std::getenv("GEMMV_AMX_DIAG")) {
            if (std::string(d) == "1") std::fprintf(stderr,
                "[GEMMV-AMX-DIAG] XCR0 missing bits: TILECFG=%d XTILEDATA=%d (xcr0=0x%llx)\n",
                (int)((xcr0>>17)&1ull), (int)((xcr0>>18)&1ull), xcr0);
        }
        return false;
    }
    // We need s_x and zp_x to scale and compensate; use per-tensor path with zp_x=128 and infer s_x from max(xq)
    // For reuse path, expect caller provides effective s_x via scales[0] (w scale)×s_x in epilogue; we pass s_x=1 and zp_x=128.
    float s_x = 1.f; int32_t zp_x = 128;
    return amx_kernel_u8s8_fp32_impl_xq(xq, K, sum_x_q, s_x, zp_x,
                                        wq_k64, M, ld_w_kbytes,
                                        scales, zps, y, bias, gran, group_size,
                                        sumW_precomp);
}

} // namespace ov::intel_cpu::x64::gemmv_jit
