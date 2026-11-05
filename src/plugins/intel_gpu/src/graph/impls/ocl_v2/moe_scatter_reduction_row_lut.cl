// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#if VEC_BLK_SIZE == 1
#define VLOAD(offset, ptr)         ((ptr)[offset])
#define VSTORE(val, offset, ptr)   ((ptr)[offset] = (val))
#else
#define VLOAD  CAT(vload,  VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)
#endif
#define INPUT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_BLK_SIZE)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_BLK_SIZE)
// The last work-items of a row can be past HIDDEN_SIZE when the vector blocks do not split evenly between them
#ifndef HIDDEN_SIZE_TAIL
#define HIDDEN_SIZE_TAIL 0
#endif
#if HIDDEN_SIZE_TAIL
#define IN_HIDDEN(h) ((h) < HIDDEN_SIZE)
#else
#define IN_HIDDEN(h) 1
#endif

// Weighted reduction of the expert-major GEMM output back into token order.
//
// Input rows are grouped by expert, so the row holding the (token, k)-th expert output cannot be
// derived from the token id alone. `row_lut` is the host-built inverse of the gather order:
// row_lut[token * ACTIVE_EXPERTS + k] is that row, or negative when the pair is unused. It
// replaces the per-token linear search over the expert token lists.
KERNEL(moe_scatter_reduction_row_lut)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* expert_weights,
    const __global INPUT2_TYPE* row_lut,
    __global OUTPUT_TYPE* output
)
{
    const uint token = (uint)get_group_id(0);
    const uint lane_base = (uint)get_local_id(0) * VEC_BLK_SIZE * BATCHES_PER_THREAD;

    OUTPUT_VEC_TYPE acc[BATCHES_PER_THREAD];
    for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
        acc[j] = TO_OUTPUT_TYPE(0);
    }

    for (uint k = 0; k < ACTIVE_EXPERTS; k++) {
#ifdef DATA_ROWS
        // Gather-8 indices: negative ones count from the end, out-of-range ones read zeros
        const long rows = DATA_ROWS;
        long idx = row_lut[token * ACTIVE_EXPERTS + k];
        if (idx < 0)
            idx += rows;
        if (idx < 0 || idx >= rows)
            continue;
        const uint row = (uint)idx;
#else
        const int row = row_lut[token * ACTIVE_EXPERTS + k];
        if (row < 0)
            continue;
#endif

        const INPUT1_TYPE expert_weight = expert_weights[token * ACTIVE_EXPERTS + k];
        const uint in_base = (uint)row * HIDDEN_SIZE + lane_base;

        for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
            if (!IN_HIDDEN(lane_base + j * VEC_BLK_SIZE))
                break;
            INPUT_VEC_TYPE input_data = VLOAD(0, &input[in_base + j * VEC_BLK_SIZE]);
            input_data *= expert_weight;
            acc[j] += input_data;
        }
    }

    const uint out_base = token * HIDDEN_SIZE + lane_base;
    for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
        if (!IN_HIDDEN(lane_base + j * VEC_BLK_SIZE))
            break;
        VSTORE(acc[j], 0, &output[out_base + j * VEC_BLK_SIZE]);
    }
}
