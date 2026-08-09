// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"

KERNEL(selective_ssm_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,
    const __global INPUT1_TYPE* dt,
    const __global INPUT2_TYPE* B,
    const __global INPUT3_TYPE* x,
    const __global INPUT4_TYPE* C,
    const __global INPUT5_TYPE* initial_state,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_state,
    __local float* work) {
    const size_t lane = get_local_id(0);
    const size_t lws = get_local_size(0);
    const size_t p = get_group_id(0);
    const size_t h = get_global_id(1);
    const size_t b = get_global_id(2);

    const size_t batch = INPUT3_BATCH_NUM;
    const size_t seq_len = INPUT3_FEATURE_NUM;
    const size_t num_heads = INPUT3_SIZE_Y;
    const size_t head_dim = INPUT3_SIZE_X;
    const size_t num_groups = INPUT2_SIZE_Y;
    const size_t state_size = INPUT2_SIZE_X;

    if (INPUT0_BATCH_NUM != num_heads ||
        INPUT1_BATCH_NUM != batch || INPUT1_FEATURE_NUM != seq_len || INPUT1_SIZE_Y != num_heads ||
        INPUT2_BATCH_NUM != batch || INPUT2_FEATURE_NUM != seq_len ||
        INPUT4_BATCH_NUM != batch || INPUT4_FEATURE_NUM != seq_len ||
        INPUT4_SIZE_Y != num_groups || INPUT4_SIZE_X != state_size ||
        INPUT5_BATCH_NUM != batch || INPUT5_FEATURE_NUM != num_heads ||
        INPUT5_SIZE_Y != head_dim || INPUT5_SIZE_X != state_size)
        return;

    if (b >= batch || h >= num_heads || p >= head_dim || num_groups == 0 || num_heads % num_groups != 0)
        return;

    const size_t heads_per_group = num_heads / num_groups;
    const size_t g = h / heads_per_group;
    __local float* local_state = work;
    __local float* reduction = work + state_size;

    for (size_t n = lane; n < state_size; n += lws) {
        const size_t state_idx = INPUT5_GET_INDEX(b, h, p, n);
        local_state[n] = SSM_TO_FLOAT(initial_state[state_idx]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t dt_idx = INPUT1_GET_INDEX(b, t, h, 0);
        const size_t x_idx = INPUT3_GET_INDEX(b, t, h, p);
        const float dt_value = SSM_TO_FLOAT(dt[dt_idx]);
        const float dA = exp(SSM_TO_FLOAT(A[INPUT0_GET_INDEX(h, 0, 0, 0)]) * dt_value);
        const float x_value = SSM_TO_FLOAT(x[x_idx]);
        float partial = 0.0f;

        for (size_t n = lane; n < state_size; n += lws) {
            const size_t b_idx = INPUT2_GET_INDEX(b, t, g, n);
            const size_t c_idx = INPUT4_GET_INDEX(b, t, g, n);
            const float new_state = fma(local_state[n], dA,
                                        x_value * dt_value * SSM_TO_FLOAT(B[b_idx]));
            const float stored_state = SSM_ROUND_STATE(new_state);
            local_state[n] = stored_state;
            partial = fma(stored_state, SSM_TO_FLOAT(C[c_idx]), partial);
        }

        reduction[lane] = partial;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t offset = lws / 2; offset > 0; offset /= 2) {
            if (lane < offset)
                reduction[lane] += reduction[lane + offset];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0)
            output[OUTPUT_GET_INDEX(b, t, h, p)] = TO_OUTPUT_TYPE(reduction[0]);
    }

    for (size_t n = lane; n < state_size; n += lws) {
        output_state[OUTPUT1_GET_INDEX(b, h, p, n)] = TO_OUTPUT1_TYPE(local_state[n]);
    }
}
