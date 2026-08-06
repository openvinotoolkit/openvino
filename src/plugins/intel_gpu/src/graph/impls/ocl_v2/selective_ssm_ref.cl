// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"

#if INPUT0_IS_FP
#define SSM_TO_FLOAT(value) convert_float(value)
#else
#define SSM_TO_FLOAT(value) _convert_as_bfloat16_float(value)
#endif

KERNEL(selective_ssm_ref)
(OPTIONAL_SHAPE_INFO_ARG
 __global INPUT0_TYPE* A,
 __global INPUT1_TYPE* dt,
 __global INPUT2_TYPE* B,
 __global INPUT3_TYPE* x,
 __global INPUT4_TYPE* C,
 __global INPUT5_TYPE* initial_state,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* output_state) {
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int p = get_global_id(2);
    const int batch = INPUT3_BATCH_NUM;
    const int seq_len = INPUT3_FEATURE_NUM;
    const int num_heads = INPUT3_Y_NUM;
    const int head_dim = INPUT3_X_NUM;
    const int num_groups = INPUT2_Y_NUM;
    const int state_size = INPUT2_X_NUM;
    if (b >= batch || h >= num_heads || p >= head_dim || num_groups <= 0)
        return;

    const int heads_per_group = num_heads / num_groups;
    if (heads_per_group <= 0)
        return;
    const int g = h / heads_per_group;

    for (int n = 0; n < state_size; n++) {
        const uint state_idx = INPUT5_GET_INDEX(b, h, p, n);
        output_state[OUTPUT1_GET_INDEX(b, h, p, n)] = initial_state[state_idx];
    }

    for (int t = 0; t < seq_len; t++) {
        const uint dt_idx = INPUT1_GET_INDEX(b, t, h, 0);
        const uint x_idx = INPUT3_GET_INDEX(b, t, h, p);
        const float dt_value = SSM_TO_FLOAT(dt[dt_idx]);
        const float dA = exp(SSM_TO_FLOAT(A[INPUT0_GET_INDEX(h, 0, 0, 0)]) * dt_value);
        const float x_value = SSM_TO_FLOAT(x[x_idx]);

        float acc = 0.0f;
        for (int n = 0; n < state_size; n++) {
            const uint state_idx = OUTPUT1_GET_INDEX(b, h, p, n);
            const uint bc_idx = INPUT2_GET_INDEX(b, t, g, n);
            const float prev_state = SSM_TO_FLOAT(output_state[state_idx]);
            const float new_state = prev_state * dA + x_value * dt_value * SSM_TO_FLOAT(B[bc_idx]);
            output_state[state_idx] = TO_OUTPUT1_TYPE(new_state);
            acc = fma(SSM_TO_FLOAT(output_state[state_idx]), SSM_TO_FLOAT(C[INPUT4_GET_INDEX(b, t, g, n)]), acc);
        }
        output[OUTPUT_GET_INDEX(b, t, h, p)] = TO_OUTPUT_TYPE(acc);
    }
}
