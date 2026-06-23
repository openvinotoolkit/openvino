// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// Optimized 4D Transpose that swaps the last two dimensions (Y<->X), i.e.
//   out[b, f, y, x] = in[b, f, x, y]
//
// Each workgroup processes a TILE_SIZE x TILE_SIZE block of the input and
// writes the transposed block to the output. The workgroup has fixed size
// WG_DIM x WG_DIM, so when TILE_SIZE > WG_DIM each work-item handles
// ELEMS_PER_DIM = TILE_SIZE / WG_DIM elements along each axis (a
// ELEMS_PER_DIM x ELEMS_PER_DIM sub-tile per WI).
//
// Loads and stores are coalesced along the innermost dimension of each side
// (input X on read, output X = input Y on write). SLM is used to absorb the
// transposition; +1 inner padding eliminates 32-bank conflicts on the
// transposed column read.
//
// JIT defines:
//   TILE_SIZE      - tile edge length in elements
//   WG_DIM         - WG size along each of X/Y (kept at 16 for SIMD8 occupancy)
//   ELEMS_PER_DIM  - TILE_SIZE / WG_DIM
//
// Required WG size: (WG_DIM, WG_DIM, 1).
// GWS: (INPUT0_SIZE_X / ELEMS_PER_DIM, INPUT0_SIZE_Y / ELEMS_PER_DIM, B*F).
// Input X and Y must both be multiples of TILE_SIZE (enforced by Validate).

__attribute__((reqd_work_group_size(WG_DIM, WG_DIM, 1)))
KERNEL(permute_xy_swap)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint lx = (uint)get_local_id(0);    // 0 .. WG_DIM-1
    const uint ly = (uint)get_local_id(1);    // 0 .. WG_DIM-1
    const uint tx = (uint)get_group_id(0);    // tile index along input X
    const uint ty = (uint)get_group_id(1);    // tile index along input Y
    const uint bf = (uint)get_global_id(2);
    const uint f  = bf % INPUT0_FEATURE_NUM;
    const uint b  = bf / INPUT0_FEATURE_NUM;

    // +1 inner padding avoids SLM bank conflicts on the transposed read.
    __local INPUT0_TYPE tile[TILE_SIZE][TILE_SIZE + 1];

    // ---- Phase 1: ELEMS_PER_DIM x ELEMS_PER_DIM coalesced reads per WI. ----
    // Each (dy, dx) sub-step issues a 16-wide coalesced load along input X.
    __attribute__((opencl_unroll_hint))
    for (uint dy = 0; dy < ELEMS_PER_DIM; ++dy) {
        const uint sub_y = ly + dy * WG_DIM;       // 0 .. TILE_SIZE-1
        const uint in_y  = ty * TILE_SIZE + sub_y;
        __attribute__((opencl_unroll_hint))
        for (uint dx = 0; dx < ELEMS_PER_DIM; ++dx) {
            const uint sub_x = lx + dx * WG_DIM;   // 0 .. TILE_SIZE-1
            const uint in_x  = tx * TILE_SIZE + sub_x;
            tile[sub_y][sub_x] = input[INPUT0_GET_INDEX(b, f, in_y, in_x)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Phase 2: ELEMS_PER_DIM x ELEMS_PER_DIM coalesced writes per WI. ----
    // Output tile origin is (out_y_base=tx*TILE_SIZE, out_x_base=ty*TILE_SIZE).
    // out[..., out_y, out_x] = in[..., out_x, out_y] = tile[out_x_local][out_y_local].
    __attribute__((opencl_unroll_hint))
    for (uint dy = 0; dy < ELEMS_PER_DIM; ++dy) {
        const uint sub_y = ly + dy * WG_DIM;       // local row in output tile
        const uint out_y = tx * TILE_SIZE + sub_y;
        __attribute__((opencl_unroll_hint))
        for (uint dx = 0; dx < ELEMS_PER_DIM; ++dx) {
            const uint sub_x = lx + dx * WG_DIM;   // local col in output tile
            const uint out_x = ty * TILE_SIZE + sub_x;
            const INPUT0_TYPE val = tile[sub_x][sub_y];  // transposed SLM read

#if HAS_FUSED_OPS
            FUSED_OPS;
            output[OUTPUT_GET_INDEX(b, f, out_y, out_x)] = FUSED_OPS_RESULT;
#else
            output[OUTPUT_GET_INDEX(b, f, out_y, out_x)] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
        }
    }
}
