// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q5_K dp4a GEMV decode kernel (int8-activation path).
//
// Drop-in alternative to the float decode in fc_gguf_opt.cl for Q5_K weights at decode (small M).
// The activation has already been pre-quantized to signed int8 (Aq) plus a per-32 f32 scale (Asc)
// by fc_gguf_prequant.cl, so both the unpacked 5-bit weight code and the activation enter the
// hardware integer dot product (cl_khr_integer_dot_product) as packed 8-bit lanes.
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

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_dp4a)(
    const __global char*        Aq,   // int8 activation [BM, K]
    const __global float*       Asc,  // per-32 scale     [BM, K/32]
    const __global uchar*       W,    // GGUF Q5_K weights [N, K]
          __global OUTPUT_TYPE* C)    // output           [BM, N]
{
    const int n    = (int)(get_global_id(0) / SG_SIZE);
    const int lane = (int)get_sub_group_local_id();
    const int bm   = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const __global uchar* w_row   = W   + (uint)n  * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    const __global char*  aq_row  = Aq  + (uint)bm * (uint)K_SIZE;
    const __global float* asc_row = Asc + (uint)bm * (uint)(K_SIZE / 32);

    float partial = 0.0f;
    for (int kb = lane; kb < blocks_per_row; kb += SG_SIZE) {
        partial += FUNC_CALL(dp_block_dot_q5k)(w_row   + (uint)kb * GGUF_BLOCK_BYTES,
                                               aq_row  + (uint)kb * GGUF_BLOCK_ELEM,
                                               asc_row + (uint)kb * (GGUF_BLOCK_ELEM / 32));
    }

    const float total = sub_group_reduce_add(partial);
    if (lane == 0)
        C[(uint)bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(total);
}
