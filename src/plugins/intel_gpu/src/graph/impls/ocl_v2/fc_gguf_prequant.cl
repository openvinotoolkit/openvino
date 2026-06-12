// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Activation int8 pre-quantization for the Q5_K dp4a GEMV decode path.
//
// Quantizes the activation A[BM, K] (f16/f32) into a signed int8 matrix Aq[BM, K] plus a parallel
// per-32-element f32 scale Asc[BM, K/32], using symmetric absmax/127 scaling per 32-element group
// (the group granularity matches the Q5_K sub-block so the dp4a GEMV can fold the per-sub-block
// scale directly into its accumulation). This is a single one-shot pass over the (tiny, decode-time)
// activation, so its cost is negligible against the weight-bound GEMV that follows — keeping the
// dp4a path's measured roofline intact while letting both operands enter the hardware dot product as
// packed 8-bit integers.
//
// One work-item owns one (bm, group) pair: global = [K_SIZE / 32, BM, 1].

#include "include/batch_headers/common.cl"

KERNEL(fc_gguf_prequant)(
    const __global INPUT0_TYPE* A,    // activation [BM, K]
          __global char*        Aq,   // int8 activation [BM, K]
          __global float*       Asc)  // per-32 scale [BM, K/32]
{
    const int g  = (int)get_global_id(0);   // 32-element group index in [0, K/32)
    const int bm = (int)get_global_id(1);    // activation row
    const uint base = (uint)bm * (uint)K_SIZE + (uint)g * 32u;

    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = fmax(amax, fabs((float)A[base + i]));
    }
    const float scale = amax * (1.0f / 127.0f);
    const float inv   = amax > 0.0f ? (127.0f / amax) : 0.0f;

    for (int i = 0; i < 32; ++i) {
        int q = convert_int_rte((float)A[base + i] * inv);
        Aq[base + i] = (char)clamp(q, -127, 127);
    }
    Asc[(uint)bm * (uint)(K_SIZE / 32) + (uint)g] = scale;
}
