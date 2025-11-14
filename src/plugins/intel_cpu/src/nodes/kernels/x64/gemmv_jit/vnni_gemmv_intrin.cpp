// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vnni_gemmv_intrin.hpp"
#include "gemmv_ukernel.hpp"
#include "xbyak/xbyak_util.h"
#include <immintrin.h>
#include <cstdlib>
#include <chrono>
#include <cstring>
// no runtime caches or env tuning needed here

namespace ov::intel_cpu::x64::gemmv_jit {

// Fixed defaults (ISA/shape-based); no env or runtime autotune
static constexpr bool g_k4_doublebuf = true;

#if defined(__GNUC__)
__attribute__((target("avx512vnni,avx512bw,avx512f")))
#endif
static void kern_block_u8s8(const uint8_t* xq, int K_actual, int K_groups,
                            const uint8_t* wq, int /*ld_w_gbytes_unused*/,
                            float s_ws_x, float bias,
                            int32_t zp_w, int32_t zp_x, int32_t sum_x_q,
                            float* y, int M_tail, const int32_t* sumW_block,
                            int M_total) {
    const int M_blk = 16;
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i sumw0 = _mm512_setzero_si512();
    __m512i sumw1 = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi8(1);
    // Deeper unroll for small-M to improve ILP
    int U = (M_total <= 128) ? 12 : (M_total <= 256) ? 8 : (K_actual >= 4096 ? 8 : (K_actual >= 2048 ? 4 : 2));
    const bool w_aligned = (((uintptr_t)wq & 63u) == 0u);
    // Slightly deeper prefetch on small-M to hide latency, match tail-heavy forms
    // Tuned ladder: 12/10/6/4 for M<=128/<=256/<=512/>512
    int P = (M_total <= 128) ? 12 : (M_total <= 256) ? 12 : (M_total <= 512) ? 8 : 6;
    // PIPE4 only for genuinely small M to avoid frontend pressure for larger M
    const bool pipe4 = (M_total <= 256);
    for (int g = 0; g < K_groups; g += U) {
        if (pipe4) {
            #pragma unroll(2)
            for (int u = 0; u < U; u += 4) {
                // Stage 1: prefetch W and prepare X broadcasts for 4-pack
                __m512i xbv[4]; int gg_idx[4]; int valid = 0;
                for (int t = 0; t < 4; ++t) {
                    int ui = u + t; if (ui >= U) break; int gg = g + ui; if (gg >= K_groups) break; gg_idx[valid] = gg;
                    const uint8_t* wptr_pf = wq + gg * 64;
                    if (gg + P < K_groups) _mm_prefetch((const char*)(wptr_pf + P * 64), _MM_HINT_T0);
                    if (g_k4_doublebuf && gg + P + 1 < K_groups) _mm_prefetch((const char*)(wptr_pf + (P + 1) * 64), _MM_HINT_T0);
                    int kbase = gg * 4;
                    if (gg + P < K_groups) { int kb_pf = (gg + P) * 4; if (kb_pf + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf), 0, 1); }
                    if (g_k4_doublebuf && gg + P + 1 < K_groups) { int kb_pf2 = (gg + P + 1) * 4; if (kb_pf2 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf2), 0, 1); }
                    uint32_t x4v;
                    if (kbase + 4 <= K_actual) x4v = *(const uint32_t*)(xq + kbase);
                    else { uint8_t b0 = (kbase + 0 < K_actual) ? xq[kbase + 0] : 0; uint8_t b1 = (kbase + 1 < K_actual) ? xq[kbase + 1] : 0; uint8_t b2 = (kbase + 2 < K_actual) ? xq[kbase + 2] : 0; uint8_t b3 = (kbase + 3 < K_actual) ? xq[kbase + 3] : 0; x4v = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24); }
                    const __m128i x128 = _mm_cvtsi32_si128((int)x4v);
                    xbv[valid++] = _mm512_broadcastd_epi32(x128);
                }
                // Stage 2: issue loads of W and dpbusd using pre-broadcast X
                for (int t = 0; t < valid; ++t) {
                    int gg = gg_idx[t]; const uint8_t* wptr = wq + gg * 64;
                    const __m512i wv = w_aligned ? _mm512_load_si512((const void*)(wptr)) : _mm512_loadu_si512((const void*)(wptr));
                    if (((u + t) & 1) == 0) { acc0 = _mm512_dpbusd_epi32(acc0, xbv[t], wv); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, wv); }
                    else                    { acc1 = _mm512_dpbusd_epi32(acc1, xbv[t], wv); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, wv); }
                }
            }
            continue;
        }
        #pragma unroll(4)
        for (int u = 0; u < U; u += 2) {
            int gg0 = g + u; if (gg0 >= K_groups) break;
            const uint8_t* wptr0 = wq + gg0 * 64;
            if (gg0 + P < K_groups) _mm_prefetch((const char*)(wptr0 + P * 64), _MM_HINT_T0);
            if (g_k4_doublebuf && gg0 + P + 1 < K_groups) _mm_prefetch((const char*)(wptr0 + (P + 1) * 64), _MM_HINT_T0);
            if (g_k4_doublebuf && gg0 + P + 1 < K_groups) _mm_prefetch((const char*)(wptr0 + (P + 1) * 64), _MM_HINT_T0);
            const __m512i w0 = w_aligned ? _mm512_load_si512((const void*)(wptr0))
                                         : _mm512_loadu_si512((const void*)(wptr0));
            int kbase0 = gg0 * 4;
            if (gg0 + P < K_groups) {
                int kb_pf = (gg0 + P) * 4;
                if (kb_pf + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf), 0, 1);
            }
            if (g_k4_doublebuf && gg0 + P + 1 < K_groups) {
                int kb_pf2 = (gg0 + P + 1) * 4;
                if (kb_pf2 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf2), 0, 1);
            }
            uint32_t x40;
            if (kbase0 + 4 <= K_actual) x40 = *(const uint32_t*)(xq + kbase0);
            else {
                uint8_t b0 = (kbase0 + 0 < K_actual) ? xq[kbase0 + 0] : 0;
                uint8_t b1 = (kbase0 + 1 < K_actual) ? xq[kbase0 + 1] : 0;
                uint8_t b2 = (kbase0 + 2 < K_actual) ? xq[kbase0 + 2] : 0;
                uint8_t b3 = (kbase0 + 3 < K_actual) ? xq[kbase0 + 3] : 0;
                x40 = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
            }
            const __m128i x128_0 = _mm_cvtsi32_si128((int)x40);
            const __m512i xb0 = _mm512_broadcastd_epi32(x128_0);
            if ((u & 1) == 0) { acc0 = _mm512_dpbusd_epi32(acc0, xb0, w0); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w0); }
            else               { acc1 = _mm512_dpbusd_epi32(acc1, xb0, w0); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w0); }

            int u1 = u + 1;
            if (u1 < U) {
                int gg1 = g + u1; if (gg1 < K_groups) {
                    const uint8_t* wptr1 = wq + gg1 * 64;
                    if (gg1 + P < K_groups) _mm_prefetch((const char*)(wptr1 + P * 64), _MM_HINT_T0);
                    if (g_k4_doublebuf && gg1 + P + 1 < K_groups) _mm_prefetch((const char*)(wptr1 + (P + 1) * 64), _MM_HINT_T0);
                    const __m512i w1 = w_aligned ? _mm512_load_si512((const void*)(wptr1))
                                                 : _mm512_loadu_si512((const void*)(wptr1));
                    int kbase1 = gg1 * 4;
                    if (gg1 + P < K_groups) {
                        int kb_pf1 = (gg1 + P) * 4;
                        if (kb_pf1 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf1), 0, 1);
                    }
                    if (g_k4_doublebuf && gg1 + P + 1 < K_groups) {
                        int kb_pf12 = (gg1 + P + 1) * 4;
                        if (kb_pf12 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf12), 0, 1);
                    }
                    uint32_t x41;
                    if (kbase1 + 4 <= K_actual) x41 = *(const uint32_t*)(xq + kbase1);
                    else {
                        uint8_t c0 = (kbase1 + 0 < K_actual) ? xq[kbase1 + 0] : 0;
                        uint8_t c1 = (kbase1 + 1 < K_actual) ? xq[kbase1 + 1] : 0;
                        uint8_t c2 = (kbase1 + 2 < K_actual) ? xq[kbase1 + 2] : 0;
                        uint8_t c3 = (kbase1 + 3 < K_actual) ? xq[kbase1 + 3] : 0;
                        x41 = (uint32_t)c0 | ((uint32_t)c1 << 8) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 24);
                    }
                    const __m128i x128_1 = _mm_cvtsi32_si128((int)x41);
                    const __m512i xb1 = _mm512_broadcastd_epi32(x128_1);
                    if ((u1 & 1) == 0) { acc0 = _mm512_dpbusd_epi32(acc0, xb1, w1); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w1); }
                    else                { acc1 = _mm512_dpbusd_epi32(acc1, xb1, w1); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w1); }
                }
            }
        }
    }
    if (sumW_block) {
        const bool s_aligned = (((uintptr_t)sumW_block & 63u) == 0u);
        const __m512i sblk = s_aligned ? _mm512_load_si512((const void*)sumW_block)
                                       : _mm512_loadu_si512((const void*)sumW_block);
        sumw0 = sblk; sumw1 = _mm512_setzero_si512();
    }
    // sum accumulators
    __m512i acc = _mm512_add_epi32(acc0, acc1);
    __m512i sumw = _mm512_add_epi32(sumw0, sumw1);
    // comp = (-x_zp)*sumW + (-zp_w)*sum_x_q + (K*zp_w*zp_x)
    const int32_t t_scalar = (-zp_w) * sum_x_q + K_actual * zp_w * zp_x;
    __m512i comp = _mm512_set1_epi32(t_scalar);
    const __m512i neg_xzp = _mm512_set1_epi32(-zp_x);
    comp = _mm512_add_epi32(comp, _mm512_mullo_epi32(neg_xzp, sumw));
    acc = _mm512_add_epi32(acc, comp);
    // to fp32
    __m512 yf = _mm512_cvtepi32_ps(acc);
    const __m512 scale = _mm512_set1_ps(s_ws_x);
    yf = _mm512_fmadd_ps(yf, scale, _mm512_set1_ps(bias));
    // store with tail if needed
    const bool do_nt = ((((uintptr_t)y) & 63u) == 0u);
    if (M_tail > 0) {
        __mmask16 k = (M_tail >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << M_tail) - 1);
        _mm512_mask_storeu_ps(y, k, yf);
    } else {
        if (do_nt && (((uintptr_t)y) & 63u) == 0u) _mm512_stream_ps(y, yf);
        else _mm512_storeu_ps(y, yf);
    }
}

#if defined(__GNUC__)
__attribute__((target("avx512vnni,avx512bw,avx512f")))
#endif
static void kern_block_u8s8_smallM(const uint8_t* xq, int K_actual, int K_groups,
                                   const uint8_t* wq, int /*ld_w_gbytes_unused*/,
                                   float s_ws_x, float bias,
                                   int32_t zp_w, int32_t zp_x, int32_t sum_x_q,
                                   float* y, int M_tail, const int32_t* sumW_block,
                                   int M_total) {
    // Aggressive unroll and prefetch for small-M cases; fixed U=8
    const int M_blk = 16;
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    __m512i acc4 = _mm512_setzero_si512();
    __m512i acc5 = _mm512_setzero_si512();
    __m512i acc6 = _mm512_setzero_si512();
    __m512i acc7 = _mm512_setzero_si512();
    __m512i sumw0 = _mm512_setzero_si512();
    __m512i sumw1 = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi8(1);
    int U = 12; // deeper unroll on tiny M to maximize ILP
    const bool w_aligned = (((uintptr_t)wq & 63u) == 0u);
    // Prefetch ladder aligned with small-M focus: 12/10/6/4
    int P = (M_total <= 128) ? 12 : (M_total <= 256) ? 12 : (M_total <= 512) ? 8 : 6;
    const bool pipe4 = (M_total <= 256);
    for (int g = 0; g < K_groups; g += U) {
        // Optionally process 4-pack per step for better ILP
        if (pipe4) {
            #pragma unroll(2)
            for (int u = 0; u < U; u += 4) {
                __m512i xbv[4]; int gg_idx[4]; int valid = 0;
                for (int t = 0; t < 4; ++t) {
                    int ui = u + t; if (ui >= U) break; int gg = g + ui; if (gg >= K_groups) break; gg_idx[valid] = gg;
                    const uint8_t* wptr_pf = wq + gg * 64;
                    if (gg + P < K_groups) _mm_prefetch((const char*)(wptr_pf + P * 64), _MM_HINT_T0);
                    if (g_k4_doublebuf && gg + P + 1 < K_groups) _mm_prefetch((const char*)(wptr_pf + (P + 1) * 64), _MM_HINT_T0);
                    int kbase = gg * 4; uint32_t x4v;
                    if (gg + P < K_groups) { int kb_pf = (gg + P) * 4; if (kb_pf + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf), 0, 1); }
                    if (g_k4_doublebuf && gg + P + 1 < K_groups) { int kb_pf2 = (gg + P + 1) * 4; if (kb_pf2 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf2), 0, 1); }
                    if (kbase + 4 <= K_actual) x4v = *(const uint32_t*)(xq + kbase);
                    else { uint8_t b0=(kbase+0<K_actual)?xq[kbase+0]:0, b1=(kbase+1<K_actual)?xq[kbase+1]:0, b2=(kbase+2<K_actual)?xq[kbase+2]:0, b3=(kbase+3<K_actual)?xq[kbase+3]:0; x4v=(uint32_t)b0|((uint32_t)b1<<8)|((uint32_t)b2<<16)|((uint32_t)b3<<24); }
                    const __m128i x128 = _mm_cvtsi32_si128((int)x4v);
                    xbv[valid++] = _mm512_broadcastd_epi32(x128);
                }
                for (int t = 0; t < valid; ++t) {
                    int ui = u + t; int gg = gg_idx[t]; const uint8_t* wptr = wq + gg * 64;
                    const __m512i wv = w_aligned ? _mm512_load_si512((const void*)(wptr)) : _mm512_loadu_si512((const void*)(wptr));
                    switch (ui & 7) {
                        case 0: acc0 = _mm512_dpbusd_epi32(acc0, xbv[t], wv); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, wv); break;
                        case 1: acc1 = _mm512_dpbusd_epi32(acc1, xbv[t], wv); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, wv); break;
                        case 2: acc2 = _mm512_dpbusd_epi32(acc2, xbv[t], wv); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, wv); break;
                        case 3: acc3 = _mm512_dpbusd_epi32(acc3, xbv[t], wv); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, wv); break;
                        case 4: acc4 = _mm512_dpbusd_epi32(acc4, xbv[t], wv); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, wv); break;
                        case 5: acc5 = _mm512_dpbusd_epi32(acc5, xbv[t], wv); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, wv); break;
                        case 6: acc6 = _mm512_dpbusd_epi32(acc6, xbv[t], wv); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, wv); break;
                        default: acc7 = _mm512_dpbusd_epi32(acc7, xbv[t], wv); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, wv); break;
                    }
                }
            }
            continue;
        }
        // Default: process pairs (u,u+1)
        #pragma unroll(4)
        for (int u = 0; u < U; u += 2) {
            // First in pair
            int gg0 = g + u; if (gg0 >= K_groups) break;
            const uint8_t* wptr0 = wq + gg0 * 64;
            if (gg0 + P < K_groups) _mm_prefetch((const char*)(wptr0 + P * 64), _MM_HINT_T0);
            const __m512i w0 = w_aligned ? _mm512_load_si512((const void*)(wptr0))
                                         : _mm512_loadu_si512((const void*)(wptr0));
            int kbase0 = gg0 * 4;
            if (gg0 + P < K_groups) {
                int kb_pf = (gg0 + P) * 4;
                if (kb_pf + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf), 0, 1);
            }
            if (g_k4_doublebuf && gg0 + P + 1 < K_groups) {
                int kb_pf2 = (gg0 + P + 1) * 4;
                if (kb_pf2 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf2), 0, 1);
            }
            uint32_t x40;
            if (kbase0 + 4 <= K_actual) x40 = *(const uint32_t*)(xq + kbase0);
            else {
                uint8_t b0 = (kbase0 + 0 < K_actual) ? xq[kbase0 + 0] : 0;
                uint8_t b1 = (kbase0 + 1 < K_actual) ? xq[kbase0 + 1] : 0;
                uint8_t b2 = (kbase0 + 2 < K_actual) ? xq[kbase0 + 2] : 0;
                uint8_t b3 = (kbase0 + 3 < K_actual) ? xq[kbase0 + 3] : 0;
                x40 = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
            }
            const __m128i x128_0 = _mm_cvtsi32_si128((int)x40);
            const __m512i xb0 = _mm512_broadcastd_epi32(x128_0);
            switch (u & 7) {
                case 0: acc0 = _mm512_dpbusd_epi32(acc0, xb0, w0); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w0); break;
                case 1: acc1 = _mm512_dpbusd_epi32(acc1, xb0, w0); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w0); break;
                case 2: acc2 = _mm512_dpbusd_epi32(acc2, xb0, w0); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w0); break;
                case 3: acc3 = _mm512_dpbusd_epi32(acc3, xb0, w0); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w0); break;
                case 4: acc4 = _mm512_dpbusd_epi32(acc4, xb0, w0); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w0); break;
                case 5: acc5 = _mm512_dpbusd_epi32(acc5, xb0, w0); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w0); break;
                case 6: acc6 = _mm512_dpbusd_epi32(acc6, xb0, w0); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w0); break;
                default: acc7 = _mm512_dpbusd_epi32(acc7, xb0, w0); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w0); break;
            }
            // Second in pair (u+1)
            int u1 = u + 1;
            if (u1 < U) {
                int gg1 = g + u1; if (gg1 < K_groups) {
                    const uint8_t* wptr1 = wq + gg1 * 64;
                    if (gg1 + P < K_groups) _mm_prefetch((const char*)(wptr1 + P * 64), _MM_HINT_T0);
                    if (g_k4_doublebuf && gg1 + P + 1 < K_groups) _mm_prefetch((const char*)(wptr1 + (P + 1) * 64), _MM_HINT_T0);
                    const __m512i w1 = w_aligned ? _mm512_load_si512((const void*)(wptr1))
                                                 : _mm512_loadu_si512((const void*)(wptr1));
                    int kbase1 = gg1 * 4;
                    if (gg1 + P < K_groups) {
                        int kb_pf1 = (gg1 + P) * 4;
                        if (kb_pf1 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf1), 0, 1);
                    }
                    if (g_k4_doublebuf && gg1 + P + 1 < K_groups) {
                        int kb_pf12 = (gg1 + P + 1) * 4;
                        if (kb_pf12 + 3 < K_actual) __builtin_prefetch((const void*)(xq + kb_pf12), 0, 1);
                    }
                    uint32_t x41;
                    if (kbase1 + 4 <= K_actual) x41 = *(const uint32_t*)(xq + kbase1);
                    else {
                        uint8_t c0 = (kbase1 + 0 < K_actual) ? xq[kbase1 + 0] : 0;
                        uint8_t c1 = (kbase1 + 1 < K_actual) ? xq[kbase1 + 1] : 0;
                        uint8_t c2 = (kbase1 + 2 < K_actual) ? xq[kbase1 + 2] : 0;
                        uint8_t c3 = (kbase1 + 3 < K_actual) ? xq[kbase1 + 3] : 0;
                        x41 = (uint32_t)c0 | ((uint32_t)c1 << 8) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 24);
                    }
                    const __m128i x128_1 = _mm_cvtsi32_si128((int)x41);
                    const __m512i xb1 = _mm512_broadcastd_epi32(x128_1);
                    switch (u1 & 7) {
                        case 0: acc0 = _mm512_dpbusd_epi32(acc0, xb1, w1); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w1); break;
                        case 1: acc1 = _mm512_dpbusd_epi32(acc1, xb1, w1); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w1); break;
                        case 2: acc2 = _mm512_dpbusd_epi32(acc2, xb1, w1); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w1); break;
                        case 3: acc3 = _mm512_dpbusd_epi32(acc3, xb1, w1); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w1); break;
                        case 4: acc4 = _mm512_dpbusd_epi32(acc4, xb1, w1); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w1); break;
                        case 5: acc5 = _mm512_dpbusd_epi32(acc5, xb1, w1); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w1); break;
                        case 6: acc6 = _mm512_dpbusd_epi32(acc6, xb1, w1); if (!sumW_block) sumw0 = _mm512_dpbusd_epi32(sumw0, ones, w1); break;
                        default: acc7 = _mm512_dpbusd_epi32(acc7, xb1, w1); if (!sumW_block) sumw1 = _mm512_dpbusd_epi32(sumw1, ones, w1); break;
                    }
                }
            }
        }
    }
    if (sumW_block) {
        const bool s_aligned = (((uintptr_t)sumW_block & 63u) == 0u);
        const __m512i sblk = s_aligned ? _mm512_load_si512((const void*)sumW_block)
                                       : _mm512_loadu_si512((const void*)sumW_block);
        sumw0 = sblk; sumw1 = _mm512_setzero_si512();
    }
    __m512i acc_lo = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    __m512i acc_hi = _mm512_add_epi32(_mm512_add_epi32(acc4, acc5), _mm512_add_epi32(acc6, acc7));
    __m512i acc = _mm512_add_epi32(acc_lo, acc_hi);
    __m512i sumw = _mm512_add_epi32(sumw0, sumw1);
    const int32_t t_scalar = (-zp_w) * sum_x_q + K_actual * zp_w * zp_x;
    __m512i comp = _mm512_set1_epi32(t_scalar);
    const __m512i neg_xzp = _mm512_set1_epi32(-zp_x);
    comp = _mm512_add_epi32(comp, _mm512_mullo_epi32(neg_xzp, sumw));
    acc = _mm512_add_epi32(acc, comp);
    __m512 yf = _mm512_cvtepi32_ps(acc);
    const __m512 scale = _mm512_set1_ps(s_ws_x);
    yf = _mm512_fmadd_ps(yf, scale, _mm512_set1_ps(bias));
    const bool do_nt = true;
    if (M_tail > 0) {
        __mmask16 k = (M_tail >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << M_tail) - 1);
        _mm512_mask_storeu_ps(y, k, yf);
    } else {
        if (do_nt && (((uintptr_t)y) & 63u) == 0u) _mm512_stream_ps(y, yf);
        else _mm512_storeu_ps(y, yf);
    }
}

bool run_gemmv_vnni_intrin_i8u8_fp32(const uint8_t* xq, int K,
                                     const uint8_t* wq_k4, int M, int ld_w_gbytes,
                                     float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                                     float* y, float bias, const int32_t* sumW_precomp) {
#if defined(__GNUC__)
    // quick compile-time check: function above is available
#else
    return false;
#endif
    if (!xq || !wq_k4 || !y) return false;
    const int M_blk = 16;
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;
    // Pad X to K64 for k4 path to eliminate K tails in inner loop (matches oneDNN copy_b behavior)
    bool pad_x = true;
    const uint8_t* xptr = xq;
    std::vector<uint8_t> xpad;
    int K_use = K;
    if (pad_x) {
        int K_pad = ((K + 63) / 64) * 64;
        xpad.resize(K_pad, (uint8_t)zp_x);
        std::memcpy(xpad.data(), xq, (size_t)K);
        xptr = xpad.data();
        K_use = K_pad;
    }
    const int K_groups = (K_use + 3) / 4;
    // sum_x_q via SIMD (only when zp_w != 0)
    int32_t sum_x_q = 0;
    if (zp_w != 0) {
        int i = 0;
#if defined(__AVX512F__)
        __m512i acc64 = _mm512_setzero_si512();
        for (; i + 64 <= K_use; i += 64) {
            __m512i vx = _mm512_loadu_si512((const void*)(xptr + i));
            __m512i sad = _mm512_sad_epu8(vx, _mm512_setzero_si512());
            acc64 = _mm512_add_epi64(acc64, sad);
        }
        alignas(64) uint64_t tmp64[8];
        _mm512_store_si512((__m512i*)tmp64, acc64);
        uint64_t s64 = 0; for (int j = 0; j < 8; ++j) s64 += tmp64[j];
        sum_x_q = (int32_t)s64;
#endif
        for (; i < K_use; ++i) sum_x_q += (int)xptr[i];
    }
    const float s_ws_x = s_w * s_x;
    const bool smallm = (M <= 256);
    for (int bi = 0; bi < M_full; ++bi) {
        const uint8_t* wblk = wq_k4 + (size_t)bi * (size_t)ld_w_gbytes;
        const int32_t* sumW_block = sumW_precomp ? (sumW_precomp + bi * M_blk) : nullptr;
        if (smallm) {
            kern_block_u8s8_smallM(xptr, K_use, K_groups, wblk, ld_w_gbytes, s_ws_x, bias, zp_w, zp_x, sum_x_q, y + bi*M_blk, 0, sumW_block, M);
        } else {
            kern_block_u8s8(xptr, K_use, K_groups, wblk, ld_w_gbytes, s_ws_x, bias, zp_w, zp_x, sum_x_q, y + bi*M_blk, 0, sumW_block, M);
        }
    }
    if (M_tail) {
        const int bi = M_full;
        const uint8_t* wblk = wq_k4 + (size_t)bi * (size_t)ld_w_gbytes;
        const int32_t* sumW_block = sumW_precomp ? (sumW_precomp + bi * M_blk) : nullptr;
        if (smallm) {
            kern_block_u8s8_smallM(xptr, K_use, K_groups, wblk, ld_w_gbytes, s_ws_x, bias, zp_w, zp_x, sum_x_q, y + bi*M_blk, M_tail, sumW_block, M);
        } else {
            kern_block_u8s8(xptr, K_use, K_groups, wblk, ld_w_gbytes, s_ws_x, bias, zp_w, zp_x, sum_x_q, y + bi*M_blk, M_tail, sumW_block, M);
        }
    }
    return true;
}

// No-repack variant using vpdpbusd with byte-position masking.
#if defined(__GNUC__)
__attribute__((target("avx512vnni,avx512bw,avx512f")))
#endif
bool run_gemmv_vnni_intrin_i8u8_fp32_norepack(const uint8_t* xq, int K,
                                              const uint8_t* wq_m16k, int M, int ld_w_bytes,
                                              const float* scales, const int32_t* zps, const float* bias,
                                              float s_x, int32_t zp_x, int32_t sum_x_q,
                                              float* y, int M_tail,
                                              int gran, int group_size) {
#if !defined(__GNUC__)
    return false;
#else
    Xbyak::util::Cpu cpu; if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) || !cpu.has(Xbyak::util::Cpu::tAVX512BW)) return false;
    const int M_blk = 16;
    const int M_full = M / M_blk;
    const int M_tail_local = M % M_blk;
    for (int bi = 0; bi < M_full; ++bi) {
        const int valid = 16;
        __m512i acc = _mm512_setzero_si512();
        __m512i sumw = _mm512_setzero_si512();
        const uint8_t* wblk = wq_m16k + (size_t)bi * (size_t)ld_w_bytes;
        for (int k = 0; k < K; k += 4) {
            // Load 4 consecutive k-slices of 16 bytes (rows)
            __m128i w0 = _mm_setzero_si128(), w1 = _mm_setzero_si128(), w2 = _mm_setzero_si128(), w3 = _mm_setzero_si128();
            if (k + 0 < K) w0 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 0) * 16));
            if (k + 1 < K) w1 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 1) * 16));
            if (k + 2 < K) w2 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 2) * 16));
            if (k + 3 < K) w3 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 3) * 16));
            // Interleave to [w0,w1,w2,w3] per row into 64B zmm (B-group)
            __m128i lo01 = _mm_unpacklo_epi8(w0, w1); // w0_0,w1_0,w0_1,w1_1,...
            __m128i hi01 = _mm_unpackhi_epi8(w0, w1);
            __m128i lo23 = _mm_unpacklo_epi8(w2, w3);
            __m128i hi23 = _mm_unpackhi_epi8(w2, w3);
            __m128i lo0123_lo = _mm_unpacklo_epi16(lo01, lo23); // w0_0,w1_0,w2_0,w3_0, w0_1,w1_1,w2_1,w3_1, ...
            __m128i lo0123_hi = _mm_unpackhi_epi16(lo01, lo23);
            __m128i hi0123_lo = _mm_unpacklo_epi16(hi01, hi23);
            __m128i hi0123_hi = _mm_unpackhi_epi16(hi01, hi23);
            // Build zmm Bgrp = [lo0123_lo, lo0123_hi, hi0123_lo, hi0123_hi]
            __m512i Bgrp = _mm512_castsi128_si512(lo0123_lo);
            Bgrp = _mm512_inserti32x4(Bgrp, lo0123_hi, 1);
            Bgrp = _mm512_inserti32x4(Bgrp, hi0123_lo, 2);
            Bgrp = _mm512_inserti32x4(Bgrp, hi0123_hi, 3);
            // sumW += Bgrp (sign-extend bytes to dwords via dpbusd with ones)
            const __m512i ones = _mm512_set1_epi32(0x01010101u);
            sumw = _mm512_dpbusd_epi32(sumw, ones, Bgrp);
            // Build A-broadcast of xq[k..k+3]
            uint32_t xb = 0;
            uint8_t x0 = (k + 0 < K) ? xq[k + 0] : 0;
            uint8_t x1 = (k + 1 < K) ? xq[k + 1] : 0;
            uint8_t x2 = (k + 2 < K) ? xq[k + 2] : 0;
            uint8_t x3 = (k + 3 < K) ? xq[k + 3] : 0;
            xb = (uint32_t)x0 | ((uint32_t)x1 << 8) | ((uint32_t)x2 << 16) | ((uint32_t)x3 << 24);
            const __m128i x128b = _mm_cvtsi32_si128((int)xb);
            __m512i Ab = _mm512_broadcastd_epi32(x128b);
            // acc += dpbusd(Ab, Bgrp)
            acc = _mm512_dpbusd_epi32(acc, Ab, Bgrp);
        }
        if (zp_x != 0) { __m512i negx = _mm512_set1_epi32(-zp_x); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(negx, sumw)); }
        if (zps) {
            const int c = K * zp_x - sum_x_q; __m512i cv = _mm512_set1_epi32(c);
            if (gran == (int)quant_granularity_t::per_tensor) { __m512i zw = _mm512_set1_epi32(zps[0]); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(zw, cv)); }
            else { __m512i zw = _mm512_loadu_si512((const void*)zps); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(zw, cv)); }
        }
        __m512 yf = _mm512_cvtepi32_ps(acc);
        __m512 sw = (gran == (int)quant_granularity_t::per_tensor) ? _mm512_set1_ps(scales[0]) : _mm512_loadu_ps(scales);
        sw = _mm512_mul_ps(sw, _mm512_set1_ps(s_x)); yf = _mm512_mul_ps(yf, sw);
        if (bias) { __m512 b = (gran == (int)quant_granularity_t::per_tensor) ? _mm512_set1_ps(bias[0]) : _mm512_loadu_ps(bias); yf = _mm512_add_ps(yf, b); }
        _mm512_storeu_ps(y + bi*16, yf);
    }
    if (M_tail_local) {
        const int bi = M_full; const int valid = M_tail_local;
        __m512i acc = _mm512_setzero_si512();
        __m512i sumw = _mm512_setzero_si512();
        const uint8_t* wblk = wq_m16k + (size_t)bi * (size_t)ld_w_bytes;
        for (int k = 0; k < K; k += 4) {
            __m128i w0 = _mm_setzero_si128(), w1 = _mm_setzero_si128(), w2 = _mm_setzero_si128(), w3 = _mm_setzero_si128();
            if (k + 0 < K) w0 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 0) * 16));
            if (k + 1 < K) w1 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 1) * 16));
            if (k + 2 < K) w2 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 2) * 16));
            if (k + 3 < K) w3 = _mm_loadu_si128((const __m128i*)(wblk + (size_t)(k + 3) * 16));
            __m128i lo01 = _mm_unpacklo_epi8(w0, w1);
            __m128i hi01 = _mm_unpackhi_epi8(w0, w1);
            __m128i lo23 = _mm_unpacklo_epi8(w2, w3);
            __m128i hi23 = _mm_unpackhi_epi8(w2, w3);
            __m128i lo0123_lo = _mm_unpacklo_epi16(lo01, lo23);
            __m128i lo0123_hi = _mm_unpackhi_epi16(lo01, lo23);
            __m128i hi0123_lo = _mm_unpacklo_epi16(hi01, hi23);
            __m128i hi0123_hi = _mm_unpackhi_epi16(hi01, hi23);
            __m512i Bgrp = _mm512_castsi128_si512(lo0123_lo);
            Bgrp = _mm512_inserti32x4(Bgrp, lo0123_hi, 1);
            Bgrp = _mm512_inserti32x4(Bgrp, hi0123_lo, 2);
            Bgrp = _mm512_inserti32x4(Bgrp, hi0123_hi, 3);
            const __m512i ones = _mm512_set1_epi32(0x01010101u);
            sumw = _mm512_dpbusd_epi32(sumw, ones, Bgrp);
            uint32_t xb = 0;
            uint8_t x0 = (k + 0 < K) ? xq[k + 0] : 0;
            uint8_t x1 = (k + 1 < K) ? xq[k + 1] : 0;
            uint8_t x2 = (k + 2 < K) ? xq[k + 2] : 0;
            uint8_t x3 = (k + 3 < K) ? xq[k + 3] : 0;
            xb = (uint32_t)x0 | ((uint32_t)x1 << 8) | ((uint32_t)x2 << 16) | ((uint32_t)x3 << 24);
            const __m128i x128b2 = _mm_cvtsi32_si128((int)xb);
            __m512i Ab = _mm512_broadcastd_epi32(x128b2);
            acc = _mm512_dpbusd_epi32(acc, Ab, Bgrp);
        }
        if (zp_x != 0) { __m512i negx = _mm512_set1_epi32(-zp_x); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(negx, sumw)); }
        if (zps) { const int c = K * zp_x - sum_x_q; __m512i cv = _mm512_set1_epi32(c); if (gran == (int)quant_granularity_t::per_tensor) { __m512i zw = _mm512_set1_epi32(zps[0]); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(zw, cv)); } else { __mmask16 k1 = (__mmask16)((1u << valid) - 1u); __m512i zw = _mm512_maskz_loadu_epi32(k1, zps); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(zw, cv)); } }
        __m512 yf = _mm512_cvtepi32_ps(acc);
        __m512 sw; if (gran == (int)quant_granularity_t::per_tensor) sw = _mm512_set1_ps(scales[0]); else { __mmask16 k1 = (__mmask16)((1u << valid) - 1u); sw = _mm512_maskz_loadu_ps(k1, scales); }
        sw = _mm512_mul_ps(sw, _mm512_set1_ps(s_x)); yf = _mm512_mul_ps(yf, sw);
        if (bias) { __m512 b; if (gran == (int)quant_granularity_t::per_tensor) b = _mm512_set1_ps(bias[0]); else { __mmask16 k1 = (__mmask16)((1u << valid) - 1u); b = _mm512_maskz_loadu_ps(k1, bias);} yf = _mm512_add_ps(yf, b); }
        __mmask16 k1 = (__mmask16)((1u << valid) - 1u); _mm512_mask_storeu_ps(y + bi*16, k1, yf);
    }
    return true;
#endif
}

#if defined(__GNUC__)
__attribute__((target("avx512vnni,avx512bw,avx512f")))
#endif
bool run_gemmv_vnni_intrin_i8u8_fp32_k64(const uint8_t* xq, int K,
                                         const uint8_t* wq_k64, int M, int ld_w_kbytes,
                                         float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                                         float* y, float bias, const int32_t* sumW_precomp) {
#if !defined(__GNUC__)
    return false;
#else
    Xbyak::util::Cpu cpu; if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) || !cpu.has(Xbyak::util::Cpu::tAVX512BW)) return false;
    if (!xq || !wq_k64 || !y) return false;
    const int M_blk = 16;
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;
    const int K_blk = 64;
    bool pad_x = true;
    const uint8_t* xptr = xq; std::vector<uint8_t> xbuf; int K_use = K;
    if (pad_x) { int K_pad = ((K + K_blk - 1) / K_blk) * K_blk; xbuf.resize(K_pad, (uint8_t)zp_x); std::memcpy(xbuf.data(), xq, (size_t)K); xptr = xbuf.data(); K_use = K_pad; }
    const int K_grp = (K_use + K_blk - 1) / K_blk;
    // sum_x_q via SIMD (only when zp_w != 0)
    int32_t sum_x_q = 0;
    if (zp_w != 0) {
        int i = 0;
#if defined(__AVX512F__)
        __m512i acc64 = _mm512_setzero_si512();
        for (; i + 64 <= K_use; i += 64) {
            __m512i vx = _mm512_loadu_si512((const void*)(xptr + i));
            __m512i sad = _mm512_sad_epu8(vx, _mm512_setzero_si512());
            acc64 = _mm512_add_epi64(acc64, sad);
        }
        alignas(64) uint64_t tmp64[8]; _mm512_store_si512((__m512i*)tmp64, acc64);
        uint64_t s64 = 0; for (int j = 0; j < 8; ++j) s64 += tmp64[j];
        sum_x_q = (int32_t)s64;
#endif
        for (; i < K_use; ++i) sum_x_q += (int)xptr[i];
    }
    const float s_ws_x = s_w * s_x;
    const bool smallM_k64 = (M <= 512);
    const bool do_nt = true;
    int P =  smallM_k64 ? 4 : 2;
    const bool k64_pipe = smallM_k64;

    for (int bi = 0; bi < M_full; ++bi) {
        const uint8_t* wblk = wq_k64 + (size_t)bi * (size_t)ld_w_kbytes;
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        __m512i sumw = _mm512_setzero_si512();
        for (int g = 0; g < K_grp; ++g) {
            const uint8_t* gbase = wblk + (size_t)g * (size_t)(K_blk * M_blk);
            const int kk_max = std::min(K_blk, K_use - g * K_blk);
            const int kg_max = (kk_max + 3) / 4;
            int kg = 0;
            // ILP-4 main: handle 4 kg groups when available
            if (k64_pipe) for (; kg + 3 < kg_max; kg += 4) {
                const uint8_t* bptr0 = gbase + (size_t)kg * 64;
                const uint8_t* bptr1 = gbase + (size_t)(kg + 1) * 64;
                const uint8_t* bptr2 = gbase + (size_t)(kg + 2) * 64;
                const uint8_t* bptr3 = gbase + (size_t)(kg + 3) * 64;
                const bool al0 = (((uintptr_t)bptr0 & 63u) == 0u);
                const bool al1 = (((uintptr_t)bptr1 & 63u) == 0u);
                const bool al2 = (((uintptr_t)bptr2 & 63u) == 0u);
                const bool al3 = (((uintptr_t)bptr3 & 63u) == 0u);
                __m512i B0 = al0 ? _mm512_load_si512((const void*)bptr0) : _mm512_loadu_si512((const void*)bptr0);
                __m512i B1 = al1 ? _mm512_load_si512((const void*)bptr1) : _mm512_loadu_si512((const void*)bptr1);
                __m512i B2 = al2 ? _mm512_load_si512((const void*)bptr2) : _mm512_loadu_si512((const void*)bptr2);
                __m512i B3 = al3 ? _mm512_load_si512((const void*)bptr3) : _mm512_loadu_si512((const void*)bptr3);
                const __m512i ones = _mm512_set1_epi32(0x01010101u);
                if (!sumW_precomp) {
                    sumw = _mm512_dpbusd_epi32(sumw, ones, B0);
                    sumw = _mm512_dpbusd_epi32(sumw, ones, B1);
                    sumw = _mm512_dpbusd_epi32(sumw, ones, B2);
                    sumw = _mm512_dpbusd_epi32(sumw, ones, B3);
                }
                const int kbase0 = g*K_blk + (kg + 0)*4;
                const int kbase1 = g*K_blk + (kg + 1)*4;
                const int kbase2 = g*K_blk + (kg + 2)*4;
                const int kbase3 = g*K_blk + (kg + 3)*4;
                uint32_t xb0 = (kbase0 + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase0)
                    : ((uint32_t)((kbase0 + 0 < K_use) ? xptr[kbase0 + 0] : 0) | ((uint32_t)((kbase0 + 1 < K_use) ? xptr[kbase0 + 1] : 0) << 8)
                     | ((uint32_t)((kbase0 + 2 < K_use) ? xptr[kbase0 + 2] : 0) << 16) | ((uint32_t)((kbase0 + 3 < K_use) ? xptr[kbase0 + 3] : 0) << 24));
                uint32_t xb1 = (kbase1 + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase1)
                    : ((uint32_t)((kbase1 + 0 < K_use) ? xptr[kbase1 + 0] : 0) | ((uint32_t)((kbase1 + 1 < K_use) ? xptr[kbase1 + 1] : 0) << 8)
                     | ((uint32_t)((kbase1 + 2 < K_use) ? xptr[kbase1 + 2] : 0) << 16) | ((uint32_t)((kbase1 + 3 < K_use) ? xptr[kbase1 + 3] : 0) << 24));
                uint32_t xb2 = (kbase2 + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase2)
                    : ((uint32_t)((kbase2 + 0 < K_use) ? xptr[kbase2 + 0] : 0) | ((uint32_t)((kbase2 + 1 < K_use) ? xptr[kbase2 + 1] : 0) << 8)
                     | ((uint32_t)((kbase2 + 2 < K_use) ? xptr[kbase2 + 2] : 0) << 16) | ((uint32_t)((kbase2 + 3 < K_use) ? xptr[kbase2 + 3] : 0) << 24));
                uint32_t xb3 = (kbase3 + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase3)
                    : ((uint32_t)((kbase3 + 0 < K_use) ? xptr[kbase3 + 0] : 0) | ((uint32_t)((kbase3 + 1 < K_use) ? xptr[kbase3 + 1] : 0) << 8)
                     | ((uint32_t)((kbase3 + 2 < K_use) ? xptr[kbase3 + 2] : 0) << 16) | ((uint32_t)((kbase3 + 3 < K_use) ? xptr[kbase3 + 3] : 0) << 24));
                const __m128i x128_0 = _mm_cvtsi32_si128((int)xb0);
                const __m128i x128_1 = _mm_cvtsi32_si128((int)xb1);
                const __m128i x128_2 = _mm_cvtsi32_si128((int)xb2);
                const __m128i x128_3 = _mm_cvtsi32_si128((int)xb3);
                __m512i A0 = _mm512_broadcastd_epi32(x128_0);
                __m512i A1 = _mm512_broadcastd_epi32(x128_1);
                __m512i A2 = _mm512_broadcastd_epi32(x128_2);
                __m512i A3 = _mm512_broadcastd_epi32(x128_3);
                acc0 = _mm512_dpbusd_epi32(acc0, A0, B0);
                acc1 = _mm512_dpbusd_epi32(acc1, A1, B1);
                acc2 = _mm512_dpbusd_epi32(acc2, A2, B2);
                acc3 = _mm512_dpbusd_epi32(acc3, A3, B3);
                if (kg + P < kg_max) {
                    const void* pf = (const void*)(gbase + (size_t)(kg + P) * 64);
                    if (smallM_k64) __builtin_prefetch(pf, 0, 3); else __builtin_prefetch(pf, 0, 1);
                    int kpf0 = g*K_blk + (kg + P)*4;
                    int kpf1 = g*K_blk + (kg + 1 + P)*4;
                    int kpf2 = g*K_blk + (kg + 2 + P)*4;
                    int kpf3 = g*K_blk + (kg + 3 + P)*4;
                    if (kpf0 + 3 < K_use) __builtin_prefetch((const void*)(xptr + kpf0), 0, 1);
                    if (kpf1 + 3 < K_use) __builtin_prefetch((const void*)(xptr + kpf1), 0, 1);
                    if (kpf2 + 3 < K_use) __builtin_prefetch((const void*)(xptr + kpf2), 0, 1);
                    if (kpf3 + 3 < K_use) __builtin_prefetch((const void*)(xptr + kpf3), 0, 1);
                }
            }
            // ILP-2 remainder
            for (; kg + 1 < kg_max; kg += 2) {
                const uint8_t* bptr0 = gbase + (size_t)kg * 64;
                const uint8_t* bptr1 = gbase + (size_t)(kg + 1) * 64;
                const bool al0 = (((uintptr_t)bptr0 & 63u) == 0u);
                const bool al1 = (((uintptr_t)bptr1 & 63u) == 0u);
                __m512i B0 = al0 ? _mm512_load_si512((const void*)bptr0)
                                 : _mm512_loadu_si512((const void*)bptr0);
                __m512i B1 = al1 ? _mm512_load_si512((const void*)bptr1)
                                 : _mm512_loadu_si512((const void*)bptr1);
                const __m512i ones = _mm512_set1_epi32(0x01010101u);
                if (!sumW_precomp) { sumw = _mm512_dpbusd_epi32(sumw, ones, B0); sumw = _mm512_dpbusd_epi32(sumw, ones, B1); }
                const int kbase0 = g*K_blk + kg*4;
                const int kbase1 = kbase0 + 4;
                uint32_t xb0 = (kbase0 + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase0)
                    : ((uint32_t)((kbase0 + 0 < K_use) ? xptr[kbase0 + 0] : 0) | ((uint32_t)((kbase0 + 1 < K_use) ? xptr[kbase0 + 1] : 0) << 8)
                     | ((uint32_t)((kbase0 + 2 < K_use) ? xptr[kbase0 + 2] : 0) << 16) | ((uint32_t)((kbase0 + 3 < K_use) ? xptr[kbase0 + 3] : 0) << 24));
                uint32_t xb1 = (kbase1 + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase1)
                    : ((uint32_t)((kbase1 + 0 < K_use) ? xptr[kbase1 + 0] : 0) | ((uint32_t)((kbase1 + 1 < K_use) ? xptr[kbase1 + 1] : 0) << 8)
                     | ((uint32_t)((kbase1 + 2 < K_use) ? xptr[kbase1 + 2] : 0) << 16) | ((uint32_t)((kbase1 + 3 < K_use) ? xptr[kbase1 + 3] : 0) << 24));
                const __m128i x128_0 = _mm_cvtsi32_si128((int)xb0);
                const __m128i x128_1 = _mm_cvtsi32_si128((int)xb1);
                __m512i A0 = _mm512_broadcastd_epi32(x128_0);
                __m512i A1 = _mm512_broadcastd_epi32(x128_1);
                acc0 = _mm512_dpbusd_epi32(acc0, A0, B0);
                acc1 = _mm512_dpbusd_epi32(acc1, A1, B1);
                if (kg + P < kg_max) {
                    const void* pf = (const void*)(gbase + (size_t)(kg + P) * 64);
                    if (smallM_k64) __builtin_prefetch(pf, 0, 3); else __builtin_prefetch(pf, 0, 1);
                    int kpf0 = g*K_blk + (kg + P)*4;
                    int kpf1 = g*K_blk + (kg + 1 + P)*4;
                    if (kpf0 + 3 < K) __builtin_prefetch((const void*)(xq + kpf0), 0, 1);
                    if (kpf1 + 3 < K) __builtin_prefetch((const void*)(xq + kpf1), 0, 1);
                }
            }
            if (kg < kg_max) {
                const uint8_t* bptr = gbase + (size_t)kg * 64;
                const bool al = (((uintptr_t)bptr & 63u) == 0u);
                __m512i B = al ? _mm512_load_si512((const void*)bptr)
                               : _mm512_loadu_si512((const void*)bptr);
                const __m512i ones = _mm512_set1_epi32(0x01010101u);
                if (!sumW_precomp) sumw = _mm512_dpbusd_epi32(sumw, ones, B);
                const int kbase = g*K_blk + kg*4;
                uint32_t xb4 = (kbase + 4 <= K_use) ? *(const uint32_t*)(xptr + kbase)
                    : ((uint32_t)((kbase + 0 < K_use) ? xptr[kbase + 0] : 0) | ((uint32_t)((kbase + 1 < K_use) ? xptr[kbase + 1] : 0) << 8)
                     | ((uint32_t)((kbase + 2 < K_use) ? xptr[kbase + 2] : 0) << 16) | ((uint32_t)((kbase + 3 < K_use) ? xptr[kbase + 3] : 0) << 24));
                const __m128i x128 = _mm_cvtsi32_si128((int)xb4);
                __m512i A = _mm512_broadcastd_epi32(x128);
                acc0 = _mm512_dpbusd_epi32(acc0, A, B);
            }
            if (g + P < K_grp) {
                const void* pf = (const void*)(wblk + (size_t)(g + P) * (size_t)(K_blk * M_blk));
                if (smallM_k64) __builtin_prefetch(pf, 0, 3); else __builtin_prefetch(pf, 0, 1);
            }
        }
        if (sumW_precomp) {
            const int32_t* sblk = sumW_precomp + bi * M_blk;
            sumw = _mm512_loadu_si512((const void*)sblk);
        }
        __m512i acc01 = _mm512_add_epi32(acc0, acc1);
        __m512i acc23 = _mm512_add_epi32(acc2, acc3);
        __m512i acc = _mm512_add_epi32(acc01, acc23);
        if (zp_x != 0) { __m512i negx = _mm512_set1_epi32(-zp_x); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(negx, sumw)); }
        if (zp_w != 0) { const int c = K * zp_x - sum_x_q; __m512i cv = _mm512_set1_epi32(c); __m512i zw = _mm512_set1_epi32(zp_w); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(zw, cv)); }
        __m512 yf = _mm512_cvtepi32_ps(acc);
        yf = _mm512_fmadd_ps(yf, _mm512_set1_ps(s_ws_x), _mm512_set1_ps(bias));
        if (do_nt && (((uintptr_t)(y + bi*16)) & 63u) == 0u) {
            _mm512_stream_ps(y + bi*16, yf);
        } else {
            _mm512_storeu_ps(y + bi*16, yf);
        }
    }
    if (M_tail) {
        const int bi = M_full; const int valid = M_tail;
        const uint8_t* wblk = wq_k64 + (size_t)bi * (size_t)ld_w_kbytes;
        __m512i acc = _mm512_setzero_si512();
        __m512i sumw = _mm512_setzero_si512();
        for (int g = 0; g < K_grp; ++g) {
            const uint8_t* gbase = wblk + (size_t)g * (size_t)(K_blk * M_blk);
            const int kk_max = std::min(K_blk, K - g * K_blk);
            const int kg_max = (kk_max + 3) / 4;
            for (int kg = 0; kg < kg_max; ++kg) {
                const uint8_t* bptr = gbase + (size_t)kg * 64;
                const bool al = (((uintptr_t)bptr & 63u) == 0u);
                __m512i Bgrp = al ? _mm512_load_si512((const void*)bptr)
                                  : _mm512_loadu_si512((const void*)bptr);
                const __m512i ones = _mm512_set1_epi32(0x01010101u);
                if (!sumW_precomp) sumw = _mm512_dpbusd_epi32(sumw, ones, Bgrp);
                const int kbase = g*K_blk + kg*4;
                uint32_t xb4;
                if (kbase + 4 <= K_use) xb4 = *(const uint32_t*)(xptr + kbase);
                else {
                    uint8_t x0 = (kbase + 0 < K_use) ? xptr[kbase + 0] : 0;
                    uint8_t x1 = (kbase + 1 < K_use) ? xptr[kbase + 1] : 0;
                    uint8_t x2 = (kbase + 2 < K_use) ? xptr[kbase + 2] : 0;
                    uint8_t x3 = (kbase + 3 < K_use) ? xptr[kbase + 3] : 0;
                    xb4 = (uint32_t)x0 | ((uint32_t)x1 << 8) | ((uint32_t)x2 << 16) | ((uint32_t)x3 << 24);
                }
                __m512i Ab = _mm512_set1_epi32((int)xb4);
                acc = _mm512_dpbusd_epi32(acc, Ab, Bgrp);
                if (kg + P < kg_max) {
                    const void* pf = (const void*)(gbase + (size_t)(kg + P) * 64);
                    if (smallM_k64) __builtin_prefetch(pf, 0, 3); else __builtin_prefetch(pf, 0, 1);
                    int kpf = g*K_blk + (kg + P)*4;
                    if (kpf + 3 < K) __builtin_prefetch((const void*)(xq + kpf), 0, 1);
                }
            }
            if (g + P < K_grp) {
                const void* pf = (const void*)(wblk + (size_t)(g + P) * (size_t)(K_blk * M_blk));
                if (smallM_k64) __builtin_prefetch(pf, 0, 3); else __builtin_prefetch(pf, 0, 1);
            }
        }
        if (sumW_precomp) {
            alignas(64) int32_t tmp[16]; const int32_t* sblk = sumW_precomp + bi * M_blk;
            for (int m = 0; m < valid; ++m) tmp[m] = sblk[m];
            for (int m = valid; m < 16; ++m) tmp[m] = 0;
            sumw = _mm512_loadu_si512((const void*)tmp);
        }
        if (zp_x != 0) { __m512i negx = _mm512_set1_epi32(-zp_x); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(negx, sumw)); }
        if (zp_w != 0) { const int c = K * zp_x - sum_x_q; __m512i cv = _mm512_set1_epi32(c); __m512i zw = _mm512_set1_epi32(zp_w); acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(zw, cv)); }
        __m512 yf = _mm512_cvtepi32_ps(acc);
        yf = _mm512_fmadd_ps(yf, _mm512_set1_ps(s_ws_x), _mm512_set1_ps(bias));
        __mmask16 mk = (__mmask16)((1u << valid) - 1);
        _mm512_mask_storeu_ps(y + bi*16, mk, yf);
    }
    return true;
#endif
}

} // namespace ov::intel_cpu::x64::gemmv_jit
