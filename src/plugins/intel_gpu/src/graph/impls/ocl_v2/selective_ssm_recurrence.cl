// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The including kernel provides the layout-specific SSM_*_INDEX macros,
// SSM_TOKEN_INDEX, and the state accessors. This file is included directly
// inside the token loop to preserve the compiler-visible recurrence body.
const size_t dt_idx = SSM_DT_INDEX(SSM_TOKEN_INDEX);
const float dt_value = ssm_to_float(dt[dt_idx]);
const float dA = exp(A_value * dt_value);
float input_scales[SSM_MAX_HEAD_DIM_BLOCK];
float partial[SSM_MAX_HEAD_DIM_BLOCK];

for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
    const size_t p = p_base + (size_t)p_offset;
    input_scales[p_offset] = ssm_to_float(x[SSM_X_INDEX(SSM_TOKEN_INDEX, p)]) * dt_value;
    partial[p_offset] = 0.0f;
}

for (SSM_STATE_ITERATION_TYPE step = 0; step < state_iterations; ++step) {
    const SSM_STATE_ITERATION_TYPE state_element = step * lws + lane;
    if (state_element >= state_size)
        break;
    const size_t b_idx = SSM_B_INDEX(SSM_TOKEN_INDEX, state_element);
    const size_t c_idx = SSM_C_INDEX(SSM_TOKEN_INDEX, state_element);
    const float b_value = ssm_to_float(B[b_idx]);
    const float c_value = ssm_to_float(C[c_idx]);
    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
        const size_t recurrence_state_idx = SSM_STATE_INDEX(p_offset, state_element);
        const float new_state = fma(SSM_STATE_AT(recurrence_state_idx), dA, input_scales[p_offset] * b_value);
        SSM_STATE_AT(recurrence_state_idx) = new_state;
        partial[p_offset] = fma(new_state, c_value, partial[p_offset]);
    }
}

if (use_subgroup_reduction) {
    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
        const float subgroup_sum = sub_group_reduce_add(partial[p_offset]);
        if (subgroup_lane == 0)
            reduction[(size_t)p_offset * lws + subgroup_id] = subgroup_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (subgroup_id == 0) {
        for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
            float total = subgroup_lane < subgroup_count
                              ? reduction[(size_t)p_offset * lws + subgroup_lane]
                              : 0.0f;
            total = sub_group_reduce_add(total);
            const size_t p = p_base + (size_t)p_offset;
            if (subgroup_lane == 0)
                output[SSM_OUTPUT_INDEX(SSM_TOKEN_INDEX, p)] = TO_OUTPUT_TYPE(total);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
} else {
    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset)
        reduction[(size_t)p_offset * lws + lane] = partial[p_offset];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = lws / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
                const size_t reduction_idx = (size_t)p_offset * lws + lane;
                reduction[reduction_idx] += reduction[reduction_idx + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
            const size_t p = p_base + (size_t)p_offset;
            output[SSM_OUTPUT_INDEX(SSM_TOKEN_INDEX, p)] = TO_OUTPUT_TYPE(reduction[(size_t)p_offset * lws]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
