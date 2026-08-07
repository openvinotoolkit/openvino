// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(selective_ssm_ref)
(__global INPUT0_TYPE* A,
 __global INPUT1_TYPE* dt,
 __global INPUT2_TYPE* B,
 __global INPUT3_TYPE* x,
 __global INPUT4_TYPE* C,
 __global INPUT5_TYPE* initial_state,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* output_state,
 int seq_len,
 int num_heads,
 int num_groups,
 int head_dim,
 int state_size) {
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int p = get_global_id(2);
    const int heads_per_group = num_heads / num_groups;
    const int g = h / heads_per_group;

    const int state_base = ((b * num_heads + h) * head_dim + p) * state_size;
    for (int n = 0; n < state_size; n++) {
        output_state[state_base + n] = initial_state[state_base + n];
    }

    for (int t = 0; t < seq_len; t++) {
        const int dt_offset = (b * seq_len + t) * num_heads + h;
        const int bc_base = (b * seq_len + t) * num_groups * state_size + g * state_size;
        const int x_offset = ((b * seq_len + t) * num_heads + h) * head_dim + p;
        const float dt_value = convert_float(dt[dt_offset]);
        const float dA = exp(convert_float(A[h]) * dt_value);
        const float x_value = convert_float(x[x_offset]);

        float acc = 0.0f;
        for (int n = 0; n < state_size; n++) {
            const float prev_state = convert_float(output_state[state_base + n]);
            const float new_state = prev_state * dA + x_value * dt_value * convert_float(B[bc_base + n]);
            output_state[state_base + n] = TO_OUTPUT1_TYPE(new_state);
            acc = fma(new_state, convert_float(C[bc_base + n]), acc);
        }
        output[x_offset] = TO_OUTPUT_TYPE(acc);
    }
}
