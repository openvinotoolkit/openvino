// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/unit_type.cl"

__attribute__((reqd_work_group_size(1, LWS1, 1)))
KERNEL (reorder_fs_b_yx_fsv32_to_bfyx)(
        const __global INPUT_REORDER_TYPE* input,
        __global OUTPUT_REORDER_TYPE* output
)
{
    const uint b = get_global_id(0);
    const uint f_blocked = get_group_id(1) * LWS1;
    const uint sglid = get_local_id(1);
    const uint f = f_blocked + sglid;
    const uint y = ((uint)get_global_id(2) / X_BLOCKED_SIZE);
    const uint x = ((uint)get_global_id(2) % X_BLOCKED_SIZE) * X_BLOCK_SIZE;
    __local CALC_TYPE in_data[X_BLOCK_SIZE * LWS1];

    const uint in_idx = INPUT0_GET_INDEX(b, f_blocked, y, x);

    __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
    for (int i = 0; i < X_BLOCK_SIZE; i++) {
        in_data[sglid * X_BLOCK_SIZE + i] = input[in_idx + (i * FSV) + sglid];
    }

    __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
    for (int i = 0; i < X_BLOCK_SIZE; i++) {
        const uint f_idx = f_blocked + i * (LWS1 / X_BLOCK_SIZE) + (sglid / X_BLOCK_SIZE);
        const uint x_idx = x + (sglid % X_BLOCK_SIZE);
        const uint out_idx = OUTPUT_GET_INDEX(b, f_idx, y, x_idx);
        const uint data_idx = sglid % X_BLOCK_SIZE + (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE + i * (LWS1 / X_BLOCK_SIZE) * X_BLOCK_SIZE;
#if defined(LEFTOVERS_OC) || defined(LEFTOVERS_OX)
#if defined(LEFTOVERS_OC) && defined(LEFTOVERS_OX)
        const bool skip = f_idx >= OUTPUT_FEATURE_NUM || x_idx >= OUTPUT_SIZE_X;
#elif defined(LEFTOVERS_OC)
        const bool skip = f_idx >= OUTPUT_FEATURE_NUM;
#else
        const bool skip = x_idx >= OUTPUT_SIZE_X;
#endif
        if (!skip) {
            output[out_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, in_data[data_idx], ACTIVATION_PARAMS_TYPED);
        }
#else
        output[out_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, in_data[data_idx], ACTIVATION_PARAMS_TYPED);
#endif
    }
}
