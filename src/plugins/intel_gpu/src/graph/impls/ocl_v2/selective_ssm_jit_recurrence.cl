// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The including JIT kernel provides compile-time dimensions, layout-specific
// SSM_*_INDEX macros, SSM_TOKEN_INDEX, and recurrence-state storage.
const float dt_lane = lane == 0 ? SSM_TO_FLOAT(dt[SSM_DT_INDEX(SSM_TOKEN_INDEX)]) : 0.0f;
const float dt_value = sub_group_broadcast(dt_lane, 0);
const float dA_lane = lane == 0 ? exp(A_value * dt_value) : 0.0f;
const float dA = sub_group_broadcast(dA_lane, 0);
float input_scales[SSM_HEAD_DIM_BLOCK];
float partial[SSM_HEAD_DIM_BLOCK];

#pragma unroll
for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
    const uint p = p_base + p_offset;
    const float x_lane = lane == 0 && p < SSM_HEAD_DIM
                             ? SSM_TO_FLOAT(x[SSM_X_INDEX(SSM_TOKEN_INDEX, p)])
                             : 0.0f;
    input_scales[p_offset] = sub_group_broadcast(x_lane, 0) * dt_value;
    partial[p_offset] = 0.0f;
}

#pragma unroll
for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
    const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
    if (state_element < SSM_STATE_SIZE) {
        const float b_value = SSM_TO_FLOAT(B[SSM_B_INDEX(SSM_TOKEN_INDEX, state_element)]);
        const float c_value = SSM_TO_FLOAT(C[SSM_C_INDEX(SSM_TOKEN_INDEX, state_element)]);
#pragma unroll
        for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
            if (p_base + p_offset < SSM_HEAD_DIM) {
                const float new_state =
                    fma(SSM_STATE_AT(p_offset, step, state_element), dA, input_scales[p_offset] * b_value);
                SSM_STATE_AT(p_offset, step, state_element) = new_state;
                partial[p_offset] = fma(new_state, c_value, partial[p_offset]);
            }
        }
    }
}

#pragma unroll
for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
    const uint p = p_base + p_offset;
    const float total = sub_group_reduce_add(partial[p_offset]);
    if (lane == 0 && p < SSM_HEAD_DIM)
        output[SSM_OUTPUT_INDEX(SSM_TOKEN_INDEX, p)] = TO_OUTPUT_TYPE(total);
}
