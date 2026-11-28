// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// GatherMatmul with base-3 ternary ("t1") expert weights, one subgroup per (output column, token, expert slot).
//
// Weights: [E, N, K / 64 * 13] bytes. Each 64 values of a row are 13 bytes; byte j holds values 5j .. 5j + 4 (the
// last byte holds 4 and a padding code) as q = ceil(v * 256 / 243), v = 81 t0 + 27 t1 + 9 t2 + 3 t3 + t4, codes
// t in {0, 1, 2}, zero point 1. The codes come out most significant first: t = 3 r, code = t >> 8, r = t & 0xFF,
// starting from r = q (the same digit code as the CPU t1 path).
// Scales: [E, K / WEI_GROUP_SIZE, N] f16 (the byfx order the plugin gives GatherMatmul scales).
// A: [n_act, n_tokens, K] (n_act == 1 broadcasts one row to all slots); out: [TOP_K, n_tokens, N].

#define SIMD 16
#define KB (K_SIZE / 64)
#if WEIGHTS_U1
// Binary weights: 64 values in 8 bytes, most significant bit first, value = (bit - U1_ZP) * scale.
#    define BLOCK_BYTES 8
#else
#    define BLOCK_BYTES 13
#endif
#define ROW_BYTES (KB * BLOCK_BYTES)

#if SCALES_FP8
// FP8 E4M3 scale byte c: ((c & 0x80) << 8) | ((c & 0x7F) << 7) is exactly value / 256 as a half for every non-NaN
// code; the 256 is applied once per block.
#    define SCALE_T uchar
#    define SCALE_VAL(c) ((float)as_half((ushort)((((uint)(c) & 0x80u) << 8) | (((uint)(c) & 0x7Fu) << 7))) * 256.0f)
#else
#    define SCALE_T half
#    define SCALE_VAL(h) ((float)(h))
#endif

#if !T1_DP4A
// Reference variant: per-value decode and fp32 FMA. One subgroup per (output column, token, expert slot).
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(gathermatmul_t1)(OPTIONAL_SHAPE_INFO_ARG
                        const __global half* input_ptr,
                        const __global uchar* weights,
                        __global half* out_ptr,
                        const __global int* indices,
                        const __global SCALE_T* scales,
                        int n_tokens,
                        int n_act,
                        int top_k) {
    const uint lane = get_sub_group_local_id();
    const uint n = get_group_id(1);
    const uint flat = get_group_id(2);
    const uint token = flat / top_k;
    const uint slot = flat % top_k;
    const int expert = indices[token * top_k + slot];
    const uint a_slot = min((int)slot, n_act - 1);

    const __global half* a = input_ptr + ((size_t)a_slot * n_tokens + token) * K_SIZE;
    const __global uchar* w = weights + ((size_t)expert * N_SIZE + n) * ROW_BYTES;
    const __global SCALE_T* s = scales + (size_t)expert * (K_SIZE / WEI_GROUP_SIZE) * N_SIZE + n;

    float acc = 0.0f;
    for (uint b = lane; b < KB; b += SIMD) {
        const __global uchar* p = w + b * BLOCK_BYTES;
        const __global half* x = a + b * 64;
        float sum = 0.0f;
#if WEIGHTS_U1
        float xsum = 0.0f;
        unroll_for(uint j = 0; j < 8; ++j) {
            const uint q = p[j];
            unroll_for(uint k = 0; k < 8; ++k) {
                const float xv = (float)x[8 * j + k];
                xsum += xv;
                if ((q >> (7 - k)) & 1u) {
                    sum += xv;
                }
            }
        }
        sum = fma(-U1_ZP, xsum, sum);
#else
        unroll_for(uint j = 0; j < 13; ++j) {
            uint r = p[j];
            unroll_for(uint k = 0; k < 5; ++k) {
                if (5 * j + k < 64) {
                    const uint t = r * 3u;
                    r = t & 0xFFu;
                    sum = fma((float)(t >> 8) - 1.0f, (float)x[5 * j + k], sum);
                }
            }
        }
#endif
        acc = fma(sum, SCALE_VAL(s[(size_t)((b * 64) / WEI_GROUP_SIZE) * N_SIZE]), acc);
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) {
        out_ptr[((size_t)slot * n_tokens + token) * N_SIZE + n] = (half)acc;
    }
}
#else  // T1_DP4A
// dp4a variant for decode. A workgroup of WG_SG subgroups handles one (token, expert slot) and WG_SG * ROWS_PER_SG *
// ROW_ITERS output columns. It first quantizes the activation row once into SLM: per weight group (WEI_GROUP_SIZE
// values) a symmetric int8 scale (max|x| / 127), the group sum of the int8 values, and the int8 bytes permuted to the
// weight byte order, so that 4 consecutive weight bytes form one dp4a operand with their matching activations.
// The weights keep their stored bytes:
//  t1: a unit is 4 blocks = 52 bytes = 13 uints. Each uint is split into two 16-bit-lane halves (bytes 0,2 and 1,3);
//      multiplying a half by 3 extracts the next base-3 digit of both bytes at once. Permuted activations:
//      xq[unit][k][52 bytes], byte b <- x[unit * 256 + 64 * (b / 13) + 5 * (b % 13) + k], 0 for the padding digit.
//  u1: a unit is one group (WEI_GROUP_SIZE / 8 bytes). Bit 7 - k of 4 consecutive bytes form one operand.
//      Permuted activations: xq[unit][k][bytes], byte b <- x[unit * G + 64 * (b / 8) + 8 * (b % 8) + k].
// A t1 weight group can end inside a uint (a group is 13 * G / 64 bytes); the bytes of such a uint are masked into
// the accumulators of the two groups. Result per group: (dot - zp * sum(x_int8)) * x_scale * w_scale.
#define G WEI_GROUP_SIZE
#define NG (K_SIZE / G)
#if WEIGHTS_U1
#    define UNIT_VALUES G
#    define UNIT_UINTS (G / 32)
#    define DIGITS 8
#    define ZP_F U1_ZP
#else
#    define UNIT_VALUES 256
#    define UNIT_UINTS 13
#    define DIGITS 5
#    define ZP_F 1.0f
#    define GROUP_BYTES (13 * G / 64)
#    define GROUPS_PER_UNIT (256 / G)
#endif
#define UNITS (K_SIZE / UNIT_VALUES)
#define XQ_UNIT (DIGITS * UNIT_UINTS)
#define LANES_PER_ROW (UNITS < SIMD ? UNITS : SIMD)
#define ROWS_PER_SG (SIMD / LANES_PER_ROW)
#define WG_SIZE (WG_SG * SIMD)

#if defined(__opencl_c_integer_dot_product_input_4x8bit_packed)
#    define DP4A_US(a, b, acc) dot_acc_sat_4x8packed_us_int((a), (b), (acc))
#else
#    define DP4A_US(a, b, acc) imad_us_packed((a), (b), (acc))
inline int imad_us_packed(uint a, uint b, int acc) {
    const uchar4 ua = as_uchar4(a);
    const char4 sb = as_char4(b);
    return acc + ua.s0 * sb.s0 + ua.s1 * sb.s1 + ua.s2 * sb.s2 + ua.s3 * sb.s3;
}
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
KERNEL(gathermatmul_t1)(OPTIONAL_SHAPE_INFO_ARG
                        const __global half* input_ptr,
                        const __global uchar* weights,
                        __global half* out_ptr,
                        const __global int* indices,
                        const __global SCALE_T* scales,
#if SHARED_XQ || T1_QUANT_ONLY
                        __global uint* xqg,
                        __global float* xsg,
                        __global int* xsumg,
#endif
                        int n_tokens,
                        int n_act,
                        int top_k) {
    __local uint xq[UNITS * XQ_UNIT];
    __local float xs[NG];
    __local float xinv[NG];
    __local int xsum[NG];
    const uint lid = get_local_id(0);
    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    const uint flat = get_group_id(2);
#if T1_QUANT_ONLY
    // one work-group per activation row (token, a_slot): quantize it once for every column work-group of the gemv
    const uint token = flat / n_act;
    const uint a_slot = flat % n_act;
    const uint slot = 0;    // the column loop below is compiled but not reached
    const int expert = 0;
#else
    const uint token = flat / top_k;
    const uint slot = flat % top_k;
    const int expert = indices[token * top_k + slot];
    const uint a_slot = min((int)slot, n_act - 1);
#endif
#if SHARED_XQ || T1_QUANT_ONLY
    const size_t xrow = (size_t)a_slot * n_tokens + token;
#endif
    const __global half* a = input_ptr + ((size_t)a_slot * n_tokens + token) * K_SIZE;

#if SHARED_XQ
    // activations quantized once per row by the T1_QUANT_ONLY stage
    for (uint u = lid; u < UNITS * XQ_UNIT; u += WG_SIZE) xq[u] = xqg[xrow * (UNITS * XQ_UNIT) + u];
    for (uint g = lid; g < NG; g += WG_SIZE) {
        xs[g] = xsg[xrow * NG + g];
        xsum[g] = xsumg[xrow * NG + g];
    }
#else
    // 1. per-group int8 scale and sum (one subgroup per group)
    for (uint g = sg; g < NG; g += WG_SG) {
        float m = 0.0f;
        for (uint i = lane; i < G; i += SIMD) m = fmax(m, fabs((float)a[g * G + i]));
        m = sub_group_reduce_max(m);
        const float sc = m > 0.0f ? m / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        int q = 0;
        for (uint i = lane; i < G; i += SIMD) q += convert_int_rte((float)a[g * G + i] * inv);
        q = sub_group_reduce_add(q);
        if (lane == 0) {
            xs[g] = sc;
            xinv[g] = inv;
            xsum[g] = q;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // 2. permuted int8 activations, one uint per work-item step
    for (uint u = lid; u < UNITS * XQ_UNIT; u += WG_SIZE) {
        const uint unit = u / XQ_UNIT;
        const uint r = u % XQ_UNIT;
        const uint k = r / UNIT_UINTS;
        uint packed = 0;
        unroll_for(uint t = 0; t < 4; ++t) {
            const uint b = (r % UNIT_UINTS) * 4 + t;
#if WEIGHTS_U1
            // u1: bit 7 - k of byte b, most significant first.
            // Either way that bit is value 8 b + k.
            const uint vi = unit * UNIT_VALUES + 64 * (b / 8) + 8 * (b % 8) + k;
#else
            const uint pos = 5 * (b % 13) + k;
            const uint vi = unit * UNIT_VALUES + 64 * (b / 13) + pos;
            if (pos < 64)
#endif
                packed |= ((uint)convert_int_rte((float)a[vi] * xinv[vi / G]) & 0xFFu) << (8 * t);
        }
        xq[u] = packed;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#if T1_QUANT_ONLY
    for (uint u = lid; u < UNITS * XQ_UNIT; u += WG_SIZE) xqg[xrow * (UNITS * XQ_UNIT) + u] = xq[u];
    for (uint g = lid; g < NG; g += WG_SIZE) {
        xsg[xrow * NG + g] = xs[g];
        xsumg[xrow * NG + g] = xsum[g];
    }
    return;
#endif

    // 3. output columns
    const uint row_in_sg = lane / LANES_PER_ROW;
    const uint ul = lane % LANES_PER_ROW;
    for (uint it = 0; it < ROW_ITERS; ++it) {
        const uint n_raw = ((get_group_id(1) * ROW_ITERS + it) * WG_SG + sg) * ROWS_PER_SG + row_in_sg;
        const uint n = min(n_raw, (uint)N_SIZE - 1);
        const __global uint* w = (const __global uint*)(weights + ((size_t)expert * N_SIZE + n) * ROW_BYTES);
        const __global SCALE_T* s = scales + (size_t)expert * NG * N_SIZE + n;
        float acc = 0.0f;
        for (uint p = ul; p < UNITS; p += LANES_PER_ROW) {
            const __local uint* xp = xq + p * XQ_UNIT;
#if WEIGHTS_U1
            uint wv[UNIT_UINTS];
            unroll_for(uint i = 0; i < UNIT_UINTS; ++i) wv[i] = w[p * UNIT_UINTS + i];
            int d = 0;
            unroll_for(uint k = 0; k < 8; ++k) {
                unroll_for(uint i = 0; i < UNIT_UINTS; ++i) d = DP4A_US((wv[i] >> (7 - k)) & 0x01010101u, xp[k * UNIT_UINTS + i], d);
            }
            acc = fma(((float)d - ZP_F * (float)xsum[p]) * xs[p], SCALE_VAL(s[(size_t)p * N_SIZE]), acc);
#else
            uint lo[13], hi[13];
            unroll_for(uint i = 0; i < 13; ++i) {
                const uint v = w[p * 13 + i];
                lo[i] = v & 0x00FF00FFu;
                hi[i] = (v >> 8) & 0x00FF00FFu;
            }
            int d[GROUPS_PER_UNIT];
            unroll_for(uint gi = 0; gi < GROUPS_PER_UNIT; ++gi) d[gi] = 0;
            unroll_for(uint k = 0; k < 5; ++k) {
                unroll_for(uint i = 0; i < 13; ++i) {
                    const uint tl = lo[i] * 3u;
                    const uint th = hi[i] * 3u;
                    lo[i] = tl & 0x00FF00FFu;
                    hi[i] = th & 0x00FF00FFu;
                    const uint dg = ((tl >> 8) & 0x00030003u) | (((th >> 8) & 0x00030003u) << 8);
                    const uint xv = xp[k * 13 + i];
                    // bytes 4i .. 4i+3 of the unit; a group boundary inside the uint splits it
                    const uint g0 = (4 * i) / GROUP_BYTES;
                    const uint g1 = (4 * i + 3) / GROUP_BYTES;
                    if (g0 == g1) {
                        d[g0] = DP4A_US(dg, xv, d[g0]);
                    } else {
                        const uint split = GROUP_BYTES * g1 - 4 * i;  // leading bytes of the uint that belong to g0
                        const uint mask = (1u << (8 * split)) - 1u;
                        d[g0] = DP4A_US(dg & mask, xv, d[g0]);
                        d[g1] = DP4A_US(dg & ~mask, xv, d[g1]);
                    }
                }
            }
            unroll_for(uint gi = 0; gi < GROUPS_PER_UNIT; ++gi) {
                const uint g = p * GROUPS_PER_UNIT + gi;
                acc = fma(((float)d[gi] - ZP_F * (float)xsum[g]) * xs[g], SCALE_VAL(s[(size_t)g * N_SIZE]), acc);
            }
#endif
        }
        for (uint off = LANES_PER_ROW / 2; off > 0; off /= 2) acc += intel_sub_group_shuffle_xor(acc, off);
        if (ul == 0 && n_raw < N_SIZE) {
            out_ptr[((size_t)slot * n_tokens + token) * N_SIZE + n_raw] = (half)acc;
        }
    }
}
#endif  // T1_DP4A
