// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatedResidual WRITE / COMBINE:  R'[row, c, d] = R[row, c, d] + S[row, c] * Y[row, d]
//
// One subgroup per (d-tile, row); the branch loop sits inside the kernel so Y and the gate vector are read once.
// JIT: GR_BRANCHES, GR_HIDDEN, GR_WRITE_TILE (halves per subgroup, divides GR_HIDDEN), GR_WRITE_NV (= GR_WRITE_TILE / 16).
// Dispatch: gws = [ceil(D / TILE) * 16, rows, 1], lws = [16, 1, 1].

#include "gr_common.cl"

#if GR_BRANCHES != 4
    #error "gr_write.cl: only 4 branches are supported"
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_write)(const __global ushort* residual, const __global ushort* sublayer_y, const __global ushort* write_gate, __global ushort* residual_out, uint num_rows) {
    const uint d0 = get_group_id(0) * GR_WRITE_TILE;
    const uint row = get_group_id(1);
    if (d0 >= GR_HIDDEN || row >= num_rows) {
        return;
    }

    const float4 gate = convert_float4(as_half4(vload4(0, write_gate + row * GR_BRANCHES)));
    const GR_FT(GR_WRITE_NV) y = GR_LDF(GR_WRITE_NV, sublayer_y + row * GR_HIDDEN + d0);

    // All branch loads are issued before the first store: residual_out may alias residual (in-place), which would otherwise
    // keep the compiler from moving the next load above the previous store.
    const uint base = row * GR_BRANCHES * GR_HIDDEN + d0;
    GR_UT(GR_WRITE_NV) r[GR_BRANCHES];
    __attribute__((opencl_unroll_hint)) for (uint c = 0; c < GR_BRANCHES; c++) {
        r[c] = GR_BRU(GR_WRITE_NV, residual + base + c * GR_HIDDEN);
    }
    __attribute__((opencl_unroll_hint)) for (uint c = 0; c < GR_BRANCHES; c++) {
        const float g = c == 0 ? gate.s0 : c == 1 ? gate.s1 : c == 2 ? gate.s2 : gate.s3;
        GR_STF(GR_WRITE_NV, residual_out + base + c * GR_HIDDEN, GR_U2F(GR_WRITE_NV, r[c]) + g * y);
    }
}
