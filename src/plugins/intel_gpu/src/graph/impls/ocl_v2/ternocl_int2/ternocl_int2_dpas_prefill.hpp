// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Prefill (large M) path for the ternary int2 FullyConnected: activations are quantized once per
// (row, 128-group) to int8 in the DPAS A layout, then an int8 x int8 DPAS GEMM reads the weights from
// a row-major copy ([N][K/16] words, built once per FC from the [K/16][N] packed weights) with scales
// [N][K/128]. The packed layout walks K at an N-word stride, which costs ~1.5x in this GEMM.
//
// Weight word w of row n holds K values 16w..16w+15 at bits [2j, 2j+1]. The packed encoding is
// {0, +1, -1} = {0, 1, 3}; the relayout turns every field into field + 1 mod 4 = the code {1, 2, 0}
// (c = (~p & lo) | (((p ^ (p >> 1)) & lo) << 1)), so the GEMM only shifts and masks, and the zero
// point 1 is subtracted through the per-group activation sum. Bit identities, operand order and
// addressing are covered by the unit tests (u2 and u1, epilogues, odd N, padded M).
// One DPAS step covers 32 K values = 2 words; operand j (0..7) is word j % 2, field j / 2, so byte t
// of operand j holds value 16 * (j % 2) + 4t + j / 2 of the step. The A layout matches that order.
//   y[m][n] = sum_g s[n][g] * xs[m][g] * (sum_{k in g} q[m][k] * c[n][k] - xsum[m][g])

namespace cldnn {
namespace ocl {

static const char kTernoclDpasPrefillSource[] = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#define SIMD 16
#define G 128
#define NG (K / G)
#define KW (K / 16)
#define KSTEPS (K / 32)
// Tile: RB 8-row blocks per subgroup, SGR subgroups stacked along rows (SGC = 8 / SGR side by side).
// ROWFAST walks the row tiles first (wide FCs: the work-groups in flight share a weight slice);
// otherwise the column tiles first (long-K FCs: they share the activation rows).
#ifndef RB
#define RB 4
#endif
#ifndef SGR
#define SGR 1
#endif
#ifndef ROWFAST
#define ROWFAST 1
#endif
#define SGC (8 / SGR)
#define WG_ROWS (8 * RB * SGR)
#define WG_COLS (16 * SGC)
#ifndef POSTOP
#define POSTOP 0
#endif
#ifdef OUT_F32
typedef float ct;
#else
typedef half ct;
#endif
#define UNROLL __attribute__((opencl_unroll_hint))
// U1: binary weights in OpenVINO's u1 layout [N][K/8] bytes (value 8b + k = bit 7 - k of byte b), zero point 0.5.
#ifndef U1
#define U1 0
#endif
#if U1
#define ZPV 0.5f
#define KWU (K / 32)
#else
#define ZPV 1.0f
#endif

inline float sigmoid_x(float x) { return native_recip(native_exp(-x) + 1.0f) * convert_float(x > -10.0f); }
inline float epi(float v, float e) {
#if POSTOP == 1
    return v * sigmoid_x(v) * e;
#elif POSTOP == 2 || POSTOP == 3
    return v + e;
#elif POSTOP == 4
    return sigmoid_x(v);
#else
    return v;
#endif
}

// One-time relayouts: packed [KW][N] words -> code rows [N][KW]; scales [NG][N] -> [N][NG].
__kernel void weights_to_rows(const __global uint* P, __global uint* R) {
    const size_t n = get_global_id(0), w = get_global_id(1);
    const uint p = P[w * N + n], lo = 0x55555555u;
    R[n * KW + w] = (~p & lo) | (((p ^ (p >> 1)) & lo) << 1);
}
__kernel void scales_to_rows(const __global half* S, __global half* R) {
    const size_t n = get_global_id(0), g = get_global_id(1);
    R[n * NG + g] = S[g * N + n];
}

// One subgroup per (row < Mp, group). Lane l holds values 8l..8l+7 of the group: step l / 4, word (l % 4) / 2,
// bytes t = 2 (l % 2) + {0, 1}; field f goes to short lane 4f + 2 word + l % 2. Rows >= M are written as zeros.
// XS / XZ are [NG][Mp]: the activation scale and scale * sum (the zero point 1 term).
__attribute__((intel_reqd_sub_group_size(SIMD)))
__kernel void quant_a(const __global half* X, __global float* XS, __global float* XZ, __global ushort* XD, int M, int Mp) {
    const uint sgid = get_global_id(0) / SIMD, lane = get_sub_group_local_id();
    const uint row = sgid / NG, g = sgid % NG;
    float8 v = (float8)(0.f);
    if ((int)row < M) v = convert_float8(vload8(0, X + (size_t)row * K + g * G + lane * 8));
    float mx = fmax(fmax(fmax(fabs(v.s0), fabs(v.s1)), fmax(fabs(v.s2), fabs(v.s3))), fmax(fmax(fabs(v.s4), fabs(v.s5)), fmax(fabs(v.s6), fabs(v.s7))));
    mx = sub_group_reduce_max(mx);
    const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
    const int8 q = convert_int8_rte(v * inv);
    const int qs = sub_group_reduce_add(q.s0 + q.s1 + q.s2 + q.s3 + q.s4 + q.s5 + q.s6 + q.s7);
    if (lane == 0) {
        XS[(size_t)g * Mp + row] = sc;
        XZ[(size_t)g * Mp + row] = ZPV * (float)qs * sc;
    }
#if U1
    // u1 operand order: step c, operand j = bit 2c + j/4 of word j%4, byte t = value 32 (j%4) + 8t + bit.
    // Lane l's value 8l + e goes to step e/2, A lane 8 (e%2) + 2 (l/4) + (l%4)/2, byte l%2.
    __global uchar* xb = (__global uchar*)XD;
    const size_t blk = ((size_t)(row / 8) * KSTEPS + g * 4) * 8 + row % 8;
    const uint lbase = 2 * (lane / 4) + (lane % 4) / 2, h = lane % 2;
    const int qa[8] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7};
    UNROLL for (int e = 0; e < 8; ++e)
        xb[((blk + (size_t)(e / 2) * 8) * 16 + 8 * (e % 2) + lbase) * 2 + h] = (uchar)(qa[e] & 0xFF);
#else
    const uint s = lane / 4, i = (lane % 4) / 2;
    __global ushort* dst = XD + ((((size_t)(row / 8) * KSTEPS) + g * 4 + s) * 8 + row % 8) * 16 + 2 * i + lane % 2;
    dst[0] = (ushort)((q.s0 & 0xFF) | ((q.s4 & 0xFF) << 8));
    dst[4] = (ushort)((q.s1 & 0xFF) | ((q.s5 & 0xFF) << 8));
    dst[8] = (ushort)((q.s2 & 0xFF) | ((q.s6 & 0xFF) << 8));
    dst[12] = (ushort)((q.s3 & 0xFF) | ((q.s7 & 0xFF) << 8));
#endif
}

)CLC"
                                                R"CLC(
// Work-group: 8 subgroups, SGR along rows and SGC = 8 / SGR along columns (tile WG_ROWS x WG_COLS).
__attribute__((intel_reqd_sub_group_size(SIMD))) __attribute__((reqd_work_group_size(8 * SIMD, 1, 1)))
__kernel void gemm(const __global ushort* XD, const __global float* XS, const __global float* XZ, const __global uint* W,
                   const __global half* S, __global ct* C, const __global half* Other, const __global ct* Bias, int M, int Mp) {
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
#if ROWFAST
    const uint mtiles = (uint)Mp / WG_ROWS;
    const uint wg_m = get_group_id(0) % mtiles, wg_n = get_group_id(0) / mtiles;
#else
    const uint ntiles = (N + WG_COLS - 1) / WG_COLS;
    const uint wg_n = get_group_id(0) % ntiles, wg_m = get_group_id(0) / ntiles;
#endif
    const uint sr = sg / SGC, sc = sg % SGC;
    const uint mb0 = (wg_m * SGR + sr) * RB;
    // a column tail (N not a multiple of the tile width, e.g. a vocab-sized LM head) reads the last column and
    // does not store
    const uint n_raw = (wg_n * SGC + sc) * 16 + lane;
    const uint n = min(n_raw, (uint)N - 1);
#if U1
    const __global uint* wp = W + (size_t)n * KWU;
#else
    const __global uint* wp = W + (size_t)n * KW;
#endif
    float acc[RB][8];
    UNROLL for (int a = 0; a < RB; ++a) UNROLL for (int r = 0; r < 8; ++r) acc[a][r] = 0.f;
    for (uint g = 0; g < NG; ++g) {
        int8 ai[RB];
        UNROLL for (int a = 0; a < RB; ++a) ai[a] = (int8)(0);
#if U1
        const uint4 wg = vload4(0, wp + g * 4);
#else
        const uint8 wg = vload8(0, wp + g * 8);
#endif
        UNROLL for (int s = 0; s < 4; ++s) {
            short8 A[RB];
            UNROLL for (int a = 0; a < RB; ++a)
                A[a] = as_short8(intel_sub_group_block_read_us8(XD + (((size_t)(mb0 + a) * KSTEPS) + g * 4 + s) * 128));
#if U1
            const uint k0 = 7 - 2 * s, k1 = 6 - 2 * s;  // shifts for bits 2s and 2s + 1 (MSB first)
            const int8 b = (int8)((wg.s0 >> k0) & 0x01010101u, (wg.s1 >> k0) & 0x01010101u, (wg.s2 >> k0) & 0x01010101u, (wg.s3 >> k0) & 0x01010101u,
                                  (wg.s0 >> k1) & 0x01010101u, (wg.s1 >> k1) & 0x01010101u, (wg.s2 >> k1) & 0x01010101u, (wg.s3 >> k1) & 0x01010101u);
#else
            const uint w0 = s == 0 ? wg.s0 : s == 1 ? wg.s2 : s == 2 ? wg.s4 : wg.s6;
            const uint w1 = s == 0 ? wg.s1 : s == 1 ? wg.s3 : s == 2 ? wg.s5 : wg.s7;
#define F(x) ((x) & 0x03030303u)
            const int8 b = (int8)(F(w0), F(w1), F(w0 >> 2), F(w1 >> 2), F(w0 >> 4), F(w1 >> 4), F(w0 >> 6), F(w1 >> 6));
#undef F
#endif
            UNROLL for (int a = 0; a < RB; ++a) ai[a] = intel_sub_group_i8_i8_matrix_mad_k32(A[a], b, ai[a]);
        }
        const float ws = (float)S[(size_t)n * NG + g];
        UNROLL for (int a = 0; a < RB; ++a) {
            // uniform-address loads stay scalar: no per-row broadcast
            const float8 xs8 = vload8(0, XS + (size_t)g * Mp + (mb0 + a) * 8);
            const float8 xz8 = vload8(0, XZ + (size_t)g * Mp + (mb0 + a) * 8);
            acc[a][0] = fma(fma((float)ai[a].s0, xs8.s0, -xz8.s0), ws, acc[a][0]);
            acc[a][1] = fma(fma((float)ai[a].s1, xs8.s1, -xz8.s1), ws, acc[a][1]);
            acc[a][2] = fma(fma((float)ai[a].s2, xs8.s2, -xz8.s2), ws, acc[a][2]);
            acc[a][3] = fma(fma((float)ai[a].s3, xs8.s3, -xz8.s3), ws, acc[a][3]);
            acc[a][4] = fma(fma((float)ai[a].s4, xs8.s4, -xz8.s4), ws, acc[a][4]);
            acc[a][5] = fma(fma((float)ai[a].s5, xs8.s5, -xz8.s5), ws, acc[a][5]);
            acc[a][6] = fma(fma((float)ai[a].s6, xs8.s6, -xz8.s6), ws, acc[a][6]);
            acc[a][7] = fma(fma((float)ai[a].s7, xs8.s7, -xz8.s7), ws, acc[a][7]);
        }
    }
    UNROLL for (int a = 0; a < RB; ++a) UNROLL for (int r = 0; r < 8; ++r) {
        const int row = (int)((mb0 + a) * 8 + r);
        if (row < M && n_raw < N) {
            const size_t i = (size_t)row * N + n;
#if POSTOP == 1 || POSTOP == 2
            const float e = (float)Other[i];
#elif POSTOP == 3
            const float e = (float)Bias[n];
#else
            const float e = 0.f;
#endif
            C[i] = (ct)epi(acc[a][r], e);
        }
    }
}
)CLC"
                                                R"CLC(
// u1 decode GEMV (M = MR <= 2): activations quantized once per call (quant_gemv_u1) into XQ [MR][NG][128 B]
// in dp4a order (word k * 4 + i, byte t = value 32 i + 8 t + k; lane l of the quantizer stores value 8 l + e at
// byte 16 e + l). No SLM, no barriers: each
// subgroup owns GROWS output columns, its 16 lanes split the column's 128-groups, then reduce.
#ifdef MR
#ifndef GROWS
#define GROWS 8
#endif
__attribute__((intel_reqd_sub_group_size(SIMD)))
__kernel void quant_gemv_u1(const __global half* X, __global uchar* XQ, __global float* XSG, __global float* XZG) {
    const uint sgid = get_global_id(0) / SIMD, lane = get_sub_group_local_id();
    const uint m = sgid / NG, g = sgid % NG;
    const float8 v = convert_float8(vload8(0, X + (size_t)m * K + g * G + lane * 8));
    float mx = fmax(fmax(fmax(fabs(v.s0), fabs(v.s1)), fmax(fabs(v.s2), fabs(v.s3))), fmax(fmax(fabs(v.s4), fabs(v.s5)), fmax(fabs(v.s6), fabs(v.s7))));
    mx = sub_group_reduce_max(mx);
    const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
    const int8 q = convert_int8_rte(v * inv);
    const int qs = sub_group_reduce_add(q.s0 + q.s1 + q.s2 + q.s3 + q.s4 + q.s5 + q.s6 + q.s7);
    if (lane == 0) { XSG[(size_t)m * NG + g] = sc; XZG[(size_t)m * NG + g] = ZPV * (float)qs * sc; }
    __global uchar* u = XQ + ((size_t)m * NG + g) * 128 + lane;
    u[0] = (uchar)(q.s0 & 0xFF); u[16] = (uchar)(q.s1 & 0xFF); u[32] = (uchar)(q.s2 & 0xFF); u[48] = (uchar)(q.s3 & 0xFF);
    u[64] = (uchar)(q.s4 & 0xFF); u[80] = (uchar)(q.s5 & 0xFF); u[96] = (uchar)(q.s6 & 0xFF); u[112] = (uchar)(q.s7 & 0xFF);
}

__attribute__((intel_reqd_sub_group_size(SIMD))) __attribute__((reqd_work_group_size(8 * SIMD, 1, 1)))
__kernel void gemv_u1(const __global uint* XQ, const __global float* XSG, const __global float* XZG, const __global uchar* W,
                       const __global half* S, __global ct* Y, const __global half* Other, const __global ct* Bias) {
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const uint n0 = (get_group_id(0) * 8 + sg) * GROWS;
    float acc[GROWS][MR];
    UNROLL for (int r = 0; r < GROWS; ++r) UNROLL for (int m = 0; m < MR; ++m) acc[r][m] = 0.f;
    for (uint p = lane; p < NG; p += SIMD) {
        uint8 xa[MR][4];
        float xs[MR], xz[MR];
        UNROLL for (int m = 0; m < MR; ++m) {
            const __global uint* xq = XQ + ((size_t)m * NG + p) * 32;
            UNROLL for (int c = 0; c < 4; ++c) xa[m][c] = vload8(c, xq);
            xs[m] = XSG[(size_t)m * NG + p];
            xz[m] = XZG[(size_t)m * NG + p];
        }
        UNROLL for (int r = 0; r < GROWS; ++r) {
            const uint n = min(n0 + r, (uint)N - 1);
            const uint4 wv = vload4(0, (const __global uint*)(W + (size_t)n * (K / 8)) + p * 4);
            const uint wa[4] = {wv.s0, wv.s1, wv.s2, wv.s3};
            const float ws = (float)S[(size_t)p * N + n];
            UNROLL for (int m = 0; m < MR; ++m) {
                const uint xw[32] = {xa[m][0].s0, xa[m][0].s1, xa[m][0].s2, xa[m][0].s3, xa[m][0].s4, xa[m][0].s5, xa[m][0].s6, xa[m][0].s7,
                                     xa[m][1].s0, xa[m][1].s1, xa[m][1].s2, xa[m][1].s3, xa[m][1].s4, xa[m][1].s5, xa[m][1].s6, xa[m][1].s7,
                                     xa[m][2].s0, xa[m][2].s1, xa[m][2].s2, xa[m][2].s3, xa[m][2].s4, xa[m][2].s5, xa[m][2].s6, xa[m][2].s7,
                                     xa[m][3].s0, xa[m][3].s1, xa[m][3].s2, xa[m][3].s3, xa[m][3].s4, xa[m][3].s5, xa[m][3].s6, xa[m][3].s7};
                int d = 0;
                UNROLL for (uint k = 0; k < 8; ++k)
                    UNROLL for (uint i = 0; i < 4; ++i) d = dot_acc_sat_4x8packed_us_int((wa[i] >> (7 - k)) & 0x01010101u, xw[k * 4 + i], d);
                acc[r][m] = fma(fma((float)d, xs[m], -xz[m]), ws, acc[r][m]);
            }
        }
    }
    UNROLL for (int r = 0; r < GROWS; ++r) {
        const uint n = n0 + r;
        UNROLL for (int m = 0; m < MR; ++m) {
            const float v = sub_group_reduce_add(acc[r][m]);
            if (lane == 0 && n < N) {
                const size_t i = (size_t)m * N + n;
#if POSTOP == 1 || POSTOP == 2
                const float e = (float)Other[i];
#elif POSTOP == 3
                const float e = (float)Bias[n];
#else
                const float e = 0.f;
#endif
                Y[i] = (ct)epi(v, e);
            }
        }
    }
}
#endif
)CLC";

}  // namespace ocl
}  // namespace cldnn
