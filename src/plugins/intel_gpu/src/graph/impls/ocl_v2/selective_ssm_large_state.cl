// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"

#define SSM_MAX_HEAD_DIM_BLOCK 4
#if INPUT0_IS_FP
#    define SSM_TO_FLOAT(v) convert_float(v)
#else
#    define SSM_TO_FLOAT(v) _convert_as_bfloat16_float(v)
#endif

KERNEL(selective_ssm_large_state)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,
    const __global INPUT1_TYPE* dt,
    const __global INPUT2_TYPE* B,
    const __global INPUT3_TYPE* x,
    const __global INPUT4_TYPE* C,
    const __global INPUT5_TYPE* initial_state,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_state,
    __global float* state_scratch,
    int head_dim_block,
    uint use_subgroup_reduction,
    __local float* reduction) {
    const uint lane = get_local_id(0);
    const uint lws = get_local_size(0);
    const uint subgroup_lane = get_sub_group_local_id();
    const uint subgroup_id = get_sub_group_id();
    const uint subgroup_count = get_num_sub_groups();
    const size_t h = get_global_id(1);
    const size_t b = get_global_id(2);

    const size_t batch = INPUT3_BATCH_NUM;
    const size_t seq_len = INPUT3_FEATURE_NUM;
    const size_t num_heads = INPUT3_SIZE_Y;
    const size_t head_dim = INPUT3_SIZE_X;
    const size_t num_groups = INPUT2_SIZE_Y;
    const size_t state_size = INPUT2_SIZE_X;
    const int block = head_dim_block;
    const size_t state_iterations = state_size / lws + (size_t)(state_size % lws != 0);
    const size_t p_base = get_group_id(0) * (size_t)block;

    if (INPUT0_BATCH_NUM != num_heads ||
        INPUT1_BATCH_NUM != batch || INPUT1_FEATURE_NUM != seq_len || INPUT1_SIZE_Y != num_heads ||
        INPUT2_BATCH_NUM != batch || INPUT2_FEATURE_NUM != seq_len ||
        INPUT4_BATCH_NUM != batch || INPUT4_FEATURE_NUM != seq_len ||
        INPUT4_SIZE_Y != num_groups || INPUT4_SIZE_X != state_size ||
        INPUT5_BATCH_NUM != batch || INPUT5_FEATURE_NUM != num_heads ||
        INPUT5_SIZE_Y != head_dim || INPUT5_SIZE_X != state_size)
        return;

    if (b >= batch || h >= num_heads || p_base >= head_dim || num_groups == 0 ||
        num_heads % num_groups != 0 || block <= 0 || block > SSM_MAX_HEAD_DIM_BLOCK)
        return;

    const size_t heads_per_group = num_heads / num_groups;
    const size_t g = h / heads_per_group;
    const float A_value = SSM_TO_FLOAT(A[GET_DATA_INDEX(INPUT0, h, 0, 0, 0)]);
    const int valid_head_dim_block = p_base + (size_t)block <= head_dim
                                         ? block
                                         : (int)(head_dim - p_base);

    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
        const size_t p = p_base + (size_t)p_offset;
        const size_t scratch_base = ((b * num_heads + h) * head_dim + p) * state_size;
        for (size_t step = 0; step < state_iterations; ++step) {
            const size_t state_element = step * lws + lane;
            if (state_element >= state_size)
                break;
            state_scratch[scratch_base + state_element] =
                SSM_TO_FLOAT(initial_state[GET_DATA_INDEX(INPUT5, b, h, p, state_element)]);
        }
    }

    for (size_t t = 0; t < seq_len; ++t) {
        const float dt_value = SSM_TO_FLOAT(dt[GET_DATA_INDEX(INPUT1, b, t, h, 0)]);
        const float dA = exp(A_value * dt_value);
        float input_scales[SSM_MAX_HEAD_DIM_BLOCK];
        float partial[SSM_MAX_HEAD_DIM_BLOCK];

        for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
            const size_t p = p_base + (size_t)p_offset;
            input_scales[p_offset] = SSM_TO_FLOAT(x[GET_DATA_INDEX(INPUT3, b, t, h, p)]) * dt_value;
            partial[p_offset] = 0.0f;
        }

        for (size_t step = 0; step < state_iterations; ++step) {
            const size_t state_element = step * lws + lane;
            if (state_element >= state_size)
                break;
            const float b_value = SSM_TO_FLOAT(B[GET_DATA_INDEX(INPUT2, b, t, g, state_element)]);
            const float c_value = SSM_TO_FLOAT(C[GET_DATA_INDEX(INPUT4, b, t, g, state_element)]);
            for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
                const size_t p = p_base + (size_t)p_offset;
                const size_t scratch_idx = ((b * num_heads + h) * head_dim + p) * state_size + state_element;
                const float new_state = fma(state_scratch[scratch_idx], dA, input_scales[p_offset] * b_value);
                state_scratch[scratch_idx] = new_state;
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
                        output[GET_DATA_INDEX(OUTPUT, b, t, h, p)] = TO_OUTPUT_TYPE(total);
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
                    output[GET_DATA_INDEX(OUTPUT, b, t, h, p)] = TO_OUTPUT_TYPE(reduction[(size_t)p_offset * lws]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
        const size_t p = p_base + (size_t)p_offset;
        const size_t scratch_base = ((b * num_heads + h) * head_dim + p) * state_size;
        for (size_t step = 0; step < state_iterations; ++step) {
            const size_t state_element = step * lws + lane;
            if (state_element >= state_size)
                break;
            output_state[GET_DATA_INDEX(OUTPUT1, b, h, p, state_element)] =
                TO_OUTPUT1_TYPE(state_scratch[scratch_base + state_element]);
        }
    }
}

#undef SSM_MAX_HEAD_DIM_BLOCK
#undef SSM_TO_FLOAT
