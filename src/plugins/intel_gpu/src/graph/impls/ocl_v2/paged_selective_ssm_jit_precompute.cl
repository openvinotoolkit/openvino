// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "selective_ssm_type_utils.cl"

KERNEL(paged_selective_ssm_jit_precompute)(OPTIONAL_SHAPE_INFO_ARG
                                           const __global INPUT0_TYPE* A,
                                           const __global INPUT1_TYPE* dt,
                                           __global float* precomputed_dA) {
    const size_t index = get_global_id(0);
#if IS_DYNAMIC
    const size_t token_count = INPUT1_BATCH_NUM;
#else
    const size_t token_count = SSM_TOKEN_COUNT;
#endif
    if (index >= token_count * SSM_NUM_HEADS)
        return;

    const size_t token = index / SSM_NUM_HEADS;
    const size_t h = index % SSM_NUM_HEADS;
#if IS_DYNAMIC
    const size_t A_index = GET_DATA_INDEX(INPUT0, h, 0, 0, 0);
    const size_t dt_index = GET_DATA_INDEX(INPUT1, token, h, 0, 0);
#else
    const size_t A_index = h;
    const size_t dt_index = index;
#endif
    precomputed_dA[index] = exp(ssm_to_float(A[A_index]) * ssm_to_float(dt[dt_index]));
}
