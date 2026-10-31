// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Dequantizes one compressed GatedResidual weight matrix [N, K] (u4 group 32 / u8 per-channel, scale / zp [G, N]) into a
// row-major f16 scratch matrix with the arithmetic of the decompression graph, see gr_wq.cl. The f16 GEMM kernels then run on
// the scratch (experiment: compared against dequantizing inside the GEMM kernels, see OV_GPU_GR_PD_MIN_ROWS).
//
// One subgroup owns 16 rows x GR_DQ_KCH columns; lane l owns row 16 * tile + l.
// JIT: GR_W_TYPE, GR_DQ_N, GR_DQ_K, GR_DQ_KCH, GR_DQ_NRAW (dwords per raw load), GR_DQ_RB (row bytes).
// Dispatch: gws = [N / 16 * 16, K / GR_DQ_KCH, 1], lws = [16, 1, 1].

#include "gr_common.cl"
#include "gr_wq.cl"

#if GR_W_TYPE == GR_WT_F16
    #error "gr_dequant_weights.cl: compressed weights only"
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_dequant_weights)(const __global uchar* w, const __global ushort* scale, const __global uchar* zp, __global ushort* out) {
    const uint lane = get_sub_group_local_id();
    const uint n0 = get_group_id(0) * 16;
    const uint k = get_group_id(1) * GR_DQ_KCH;
    const __global uchar* brow = w + (size_t)(n0 + lane) * GR_DQ_RB;

    uint rw[GR_DQ_NRAW];
    GR_WQ_LOAD_RAW(rw, GR_DQ_NRAW, brow, k);
#if GR_W_TYPE == GR_WT_U8
    const half2 scv = GR_WQ_SCALE2(scale, 0, GR_DQ_N, n0);
    const half2 zpv = GR_WQ_ZP2(zp, 0, GR_DQ_N, n0, lane);
#endif
    unroll_for (uint s_ = 0; s_ < GR_DQ_KCH / 32; s_++) {
#if GR_W_TYPE == GR_WT_U4
        const half2 scv = GR_WQ_SCALE2(scale, (k >> 5) + s_, GR_DQ_N, n0);
        const half2 zpv = GR_WQ_ZP2(zp, (k >> 5) + s_, GR_DQ_N, n0, lane);
#endif
        const uint16 b = GR_WQ_DEQ32(rw, s_ * GR_WQ_RAW_PER_K32, zpv, scv);
        // the B operand dwords are (k, k + 1) pairs in k order, i.e. already the row-major f16 layout
        vstore16(b, 0, (__global uint*)(out + (size_t)(n0 + lane) * GR_DQ_K + k + 32 * s_));
    }
}
