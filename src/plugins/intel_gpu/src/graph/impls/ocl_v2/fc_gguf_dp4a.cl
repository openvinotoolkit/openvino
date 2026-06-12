// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q5_K / Q6_K dp4a GEMV decode kernel (int8-activation path).
//
// Drop-in alternative to the float decode in fc_gguf_opt.cl for Q5_K / Q6_K weights at decode
// (small M). The activation has already been pre-quantized to signed int8 (Aq) plus a per-32 f32
// scale (Asc) by fc_gguf_prequant.cl, so both the unpacked weight code and the activation enter the
// hardware integer dot product (cl_khr_integer_dot_product) as packed 8-bit lanes. The decoder is
// selected at compile time by GGUF_IS_Q5_K / GGUF_IS_Q6_K.
//
// The unpack is done with SWAR: four nibbles are masked/shifted per 32-bit word and the qh high bit
// is OR-ed in while staying packed, then fed straight to dot_acc_sat_4x8packed_us_int (weights are
// 0..31 unsigned, activations signed). This keeps the per-element unpack ALU ~2x lower than the
// scalar float decode, which was the measured bottleneck (the float decoder topped out near the
// decode-only ALU ceiling); the packed-int form breaks past it. The activation absmax term Sqa
// (folded with the Q5_K per-sub-block min) is accumulated with a dot against an all-ones weight,
// reusing the already-loaded activation lanes (no extra global traffic).
//
// One subgroup (SG_SIZE lanes) cooperatively owns one (n, bm) output, the row's blocks striped
// across lanes and collapsed with a single sub_group_reduce_add — identical layout to fc_gguf_opt.

#include "include/batch_headers/common.cl"

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

inline half FUNC(dp_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

#if defined(GGUF_IS_Q5_K)
// 6-bit packed sub-block scale/min extraction (ggml get_scale_min_k4), same as fc_gguf_opt.cl.
inline void FUNC(dp_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Streaming dp4a dot of one Q5_K block (256 elements) against the int8 activation slice `aq` with its
// eight per-32 scales `asc`. Returns the block's contribution to sum_k a[k]*dequant(w[k]).
inline float FUNC(dp_block_dot_q5k)(const __global uchar* blk,
                                    const __global char*  aq,
                                    const __global float* asc) {
    const float d    = (float)FUNC_CALL(dp_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(dp_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qh     = blk + 16;   // 32 bytes (high bit-plane)
    const __global uchar* ql     = blk + 48;   // 128 bytes (low 4 bits)

    float acc = 0.0f;
    int is = 0, bit = 0;
    for (int iter = 0; iter < 4; ++iter) {
        uchar sc, m;
        FUNC_CALL(dp_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(dp_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;

        const __global char* aq_lo = aq + (2 * iter) * 32;
        const __global char* aq_hi = aq + (2 * iter + 1) * 32;
        int Sqq1 = 0, Sqa1 = 0, Sqq2 = 0, Sqa2 = 0;
        for (int l = 0; l < 32; l += 4) {
            const uint qlu = as_uint(vload4(0, ql + l));
            const uint qhu = as_uint(vload4(0, qh + l));
            const uint lo4 = qlu & 0x0F0F0F0Fu;
            const uint hi4 = (qlu >> 4) & 0x0F0F0F0Fu;
            const uint wlo = lo4 | (((qhu >> bit)       & 0x01010101u) << 4);
            const uint whi = hi4 | (((qhu >> (bit + 1)) & 0x01010101u) << 4);
            const uint au_lo = as_uint(vload4(0, aq_lo + l));
            const uint au_hi = as_uint(vload4(0, aq_hi + l));
            Sqq1 = dot_acc_sat_4x8packed_us_int(wlo, au_lo, Sqq1);
            Sqq2 = dot_acc_sat_4x8packed_us_int(whi, au_hi, Sqq2);
            Sqa1 = dot_acc_sat_4x8packed_us_int(0x01010101u, au_lo, Sqa1);
            Sqa2 = dot_acc_sat_4x8packed_us_int(0x01010101u, au_hi, Sqa2);
        }
        acc += asc[2 * iter]     * (d1 * Sqq1 - m1 * Sqa1);
        acc += asc[2 * iter + 1] * (d2 * Sqq2 - m2 * Sqa2);
        ql += 32; is += 2; bit += 2;
    }
    return acc;
}
#endif  // GGUF_IS_Q5_K

#if defined(GGUF_IS_Q6_K)
// One SWAR-unpack + dp4a step for a single packed uint (4 weights) of a low-nibble stream (A) and
// its paired high-nibble stream (C). qlc/qhc are the matching ql/qh uint lanes; the Q6_K 2 high
// bits come from qh shifted by sh_lo/sh_hi. Sqq accumulates w*a, Sqa accumulates a (to fold the -32
// dequant offset). Mirrors the proven standalone q6k_dp4a_swar.cl decoder.
#define DP_Q6K_STEP(qlc, qhc, aAc, aCc, Sa, Aa, Sc, Ac, sh_lo, sh_hi) do {                          \
    const uint _wA = ((qlc) & 0x0F0F0F0Fu)        | ((((qhc) >> (sh_lo)) & 0x03030303u) << 4);      \
    const uint _wC = (((qlc) >> 4) & 0x0F0F0F0Fu) | ((((qhc) >> (sh_hi)) & 0x03030303u) << 4);      \
    (Sa) = dot_acc_sat_4x8packed_us_int(_wA, (aAc), (Sa));                                          \
    (Aa) = dot_acc_sat_4x8packed_us_int(0x01010101u, (aAc), (Aa));                                  \
    (Sc) = dot_acc_sat_4x8packed_us_int(_wC, (aCc), (Sc));                                          \
    (Ac) = dot_acc_sat_4x8packed_us_int(0x01010101u, (aCc), (Ac));                                  \
} while (0)

// Streaming dual-dp4a dot of one Q6_K block (256 elements / 210 useful bytes) against the int8
// activation slice `aq` with its eight per-32 scales `asc`. Returns the block contribution to
// sum_k a[k]*dequant(w[k]). Wide vload16 loads cut load-instruction count; the per-16 signed Q6_K
// scales force an lo16/hi16 split -> 8 live int accumulators. blk is the block base; with an
// aligned storage stride (GGUF_BLOCK_BYTES) the 210 useful bytes are contiguous from blk.
inline float FUNC(dp_block_dot_q6k)(const __global uchar* blk,
                                    const __global char*  aq,
                                    const __global float* asc) {
    const __global uchar* ql = blk;
    const __global uchar* qh = blk + 128;
    const __global char*  sc = (const __global char*)(blk + 192);
    const float d = (float)FUNC_CALL(dp_load_f16)(blk + 208);
    float acc = 0.0f;

    for (int hf = 0; hf < 2; ++hf) {
        const int o  = hf * 128;     // weight/activation base offset within block
        const int g  = hf * 8;       // weight-scale group base (sc index)
        const int ag = hf * 4;       // activation-scale group base (asc index)

        unroll_for (int pr = 0; pr < 2; ++pr) {
            const int ql_off = pr * 32;            // pr=0 -> ql[l] (streams 0,2); pr=1 -> ql[l+32] (streams 1,3)
            const int sh_lo  = pr * 2;             // qh shift for the low-nibble stream  (0 or 2)
            const int sh_hi  = pr * 2 + 4;         // qh shift for the high-nibble stream (4 or 6)
            const __global char* act_lo = aq + o + pr * 32;        // low-nibble stream  (stream pr)
            const __global char* act_hi = aq + o + pr * 32 + 64;   // high-nibble stream (stream pr+2)

            int SloA = 0, AloA = 0, ShiA = 0, AhiA = 0;   // low-nibble stream: lo16 + hi16 sc groups
            int SloC = 0, AloC = 0, ShiC = 0, AhiC = 0;   // high-nibble stream
            unroll_for (int l = 0; l < 32; l += 16) {
                const uint4 QL = as_uint4(vload16(0, ql + ql_off + l));
                const uint4 QH = as_uint4(vload16(0, qh + l));
                const uint4 AA = as_uint4(vload16(0, (const __global uchar*)act_lo + l));
                const uint4 AC = as_uint4(vload16(0, (const __global uchar*)act_hi + l));
                if (l < 16) {
                    DP_Q6K_STEP(QL.x, QH.x, AA.x, AC.x, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                    DP_Q6K_STEP(QL.y, QH.y, AA.y, AC.y, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                    DP_Q6K_STEP(QL.z, QH.z, AA.z, AC.z, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                    DP_Q6K_STEP(QL.w, QH.w, AA.w, AC.w, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                } else {
                    DP_Q6K_STEP(QL.x, QH.x, AA.x, AC.x, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                    DP_Q6K_STEP(QL.y, QH.y, AA.y, AC.y, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                    DP_Q6K_STEP(QL.z, QH.z, AA.z, AC.z, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                    DP_Q6K_STEP(QL.w, QH.w, AA.w, AC.w, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                }
            }
            // low-nibble stream = stream pr; high-nibble stream = stream pr+2.
            acc += d * asc[ag + pr]     * ((float)sc[g + 2 * pr]     * (float)(SloA - 32 * AloA)
                                         + (float)sc[g + 2 * pr + 1] * (float)(ShiA - 32 * AhiA));
            acc += d * asc[ag + pr + 2] * ((float)sc[g + 2 * pr + 4] * (float)(SloC - 32 * AloC)
                                         + (float)sc[g + 2 * pr + 5] * (float)(ShiC - 32 * AhiC));
        }

        ql += 64; qh += 32;
    }
    return acc;
}
#endif  // GGUF_IS_Q6_K

// NROW = number of output rows one subgroup owns (register/multi-row blocking). Default 1 keeps the
// original single-output Q5_K behaviour; the Q6_K path sets NROW > 1 to read each activation slice
// once and L1-reuse it across rows, amortising activation traffic on DRAM-bound shapes.
#ifndef NROW
#define NROW 1
#endif

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_dp4a)(
    const __global char*        Aq,   // int8 activation       [BM, K]
    const __global float*       Asc,  // per-32 activation scale [BM, K/32]
    const __global uchar*       W,    // GGUF weights (aligned block stride GGUF_BLOCK_BYTES) [N, K]
          __global OUTPUT_TYPE* C)    // output                [BM, N]
{
    const int sg   = (int)(get_global_id(0) / SG_SIZE);
    const int lane = (int)get_sub_group_local_id();
    const int bm   = (int)get_global_id(1);
    const int n0   = sg * NROW;                 // first output row this subgroup owns

    if (n0 >= N_SIZE)
        return;

    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const __global char*  aq_row  = Aq  + (uint)bm * (uint)K_SIZE;
    const __global float* asc_row = Asc + (uint)bm * (uint)(K_SIZE / 32);

    float partial[NROW];
    unroll_for (int r = 0; r < NROW; ++r)
        partial[r] = 0.0f;

    // Stripe K-blocks across the subgroup lanes; for each block decode all NROW weight rows against
    // the SAME activation slice (loaded once, then L1-resident across the row loop). The (n0+r) row
    // guard keeps the unconditional unrolled row reads in-bounds when N_SIZE is not a NROW multiple
    // (it is loop-invariant per r, so the compiler hoists it out of the kb loop).
    for (int kb = lane; kb < blocks_per_row; kb += SG_SIZE) {
        const __global char*  aq_blk  = aq_row  + (uint)kb * GGUF_BLOCK_ELEM;
        const __global float* asc_blk = asc_row + (uint)kb * (GGUF_BLOCK_ELEM / 32);
        unroll_for (int r = 0; r < NROW; ++r) {
            if ((n0 + r) >= N_SIZE)
                continue;
            const __global uchar* w_blk = W + ((uint)(n0 + r) * (uint)blocks_per_row + (uint)kb) * GGUF_BLOCK_BYTES;
#if defined(GGUF_IS_Q6_K)
            partial[r] += FUNC_CALL(dp_block_dot_q6k)(w_blk, aq_blk, asc_blk);
#else
            partial[r] += FUNC_CALL(dp_block_dot_q5k)(w_blk, aq_blk, asc_blk);
#endif
        }
    }

    unroll_for (int r = 0; r < NROW; ++r) {
        const float total = sub_group_reduce_add(partial[r]);
        if (lane == 0 && (n0 + r) < N_SIZE)
            C[(uint)bm * (uint)N_SIZE + (n0 + r)] = TO_OUTPUT_TYPE(total);
    }
}

