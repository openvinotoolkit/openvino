// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"

#if INPUT0_IS_FP
#    define SSM_TO_FLOAT(v) convert_float(v)
#else
#    define SSM_TO_FLOAT(v) _convert_as_bfloat16_float(v)
#endif

#define SSM_DT_INDEX(token) (((token) * SSM_NUM_HEADS) + h)
#define SSM_B_INDEX(token, state_element) \
    ((((token) * SSM_NUM_GROUPS + g) * SSM_STATE_SIZE) + (state_element))
#define SSM_C_INDEX(token, state_element) SSM_B_INDEX(token, state_element)
#define SSM_X_INDEX(token, p) (((((token) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM) + (p)))
#define SSM_OUTPUT_INDEX(token, p) SSM_X_INDEX(token, p)
#define SSM_STATE_INDEX(block, p, state_element) \
    ((((((block) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM + (p)) * SSM_STATE_SIZE) + (state_element)))
#if SSM_JIT_USE_SLM
#    define SSM_STATE_AT(p_offset, step, state_element) slm_state[(p_offset) * SSM_STATE_SIZE + (state_element)]
#else
#    define SSM_STATE_AT(p_offset, step, state_element) private_state[p_offset][step]
#endif

REQD_SUB_GROUP_SIZE(SSM_SUBGROUP_SIZE)
KERNEL(SSM_JIT_KERNEL)(const __global INPUT0_TYPE* A,
                       const __global INPUT1_TYPE* dt,
                       const __global INPUT2_TYPE* B,
                       const __global INPUT3_TYPE* x,
                       const __global INPUT4_TYPE* C,
                       __global INPUT5_TYPE* recurrent_state_table,
                       const __global INPUT6_TYPE* subsequence_begins,
                       const __global INPUT7_TYPE* block_indices,
                       const __global INPUT8_TYPE* block_indices_begins,
                       const __global INPUT9_TYPE* num_processed_tokens,
                       const __global INPUT10_TYPE* cache_interval,
                       __global OUTPUT_TYPE* output
#if SSM_JIT_USE_SLM
                       , __local float* slm_state
#endif
                       ) {
    const uint lane = get_sub_group_local_id();
    const uint h = get_global_id(1);
    const uint seq = get_global_id(2);
    const uint p_base = get_group_id(0) * SSM_HEAD_DIM_BLOCK;
    const long token_begin = (long)subsequence_begins[seq];
    const long token_end = (long)subsequence_begins[seq + 1];
    const long block_begin = (long)block_indices_begins[seq];
    const long block_end = (long)block_indices_begins[seq + 1];

    if (token_begin < 0 || token_end < token_begin || (ulong)token_end > (ulong)SSM_TOKEN_COUNT)
        return;
    if (token_begin == token_end)
        return;

    if (block_begin < 0 || block_end <= block_begin || (ulong)block_end > (ulong)INPUT7_BATCH_NUM) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token) {
#pragma unroll
                for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
                    const uint p = p_base + p_offset;
                    if (p < SSM_HEAD_DIM)
                        output[SSM_OUTPUT_INDEX((uint)token, p)] = TO_OUTPUT_TYPE(0.0f);
                }
            }
        }
        return;
    }

    const long first_block = (long)block_indices[block_begin];
    if (first_block < 0 || (ulong)first_block >= (ulong)INPUT5_BATCH_NUM) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token) {
#pragma unroll
                for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
                    const uint p = p_base + p_offset;
                    if (p < SSM_HEAD_DIM)
                        output[SSM_OUTPUT_INDEX((uint)token, p)] = TO_OUTPUT_TYPE(0.0f);
                }
            }
        }
        return;
    }

    const long processed_raw = (long)num_processed_tokens[seq];
    const long interval = (long)cache_interval[seq];
    const long processed = max(processed_raw, (long)0);
    const bool cache_enabled = interval > 0;
    const ulong positive_interval = cache_enabled ? (ulong)interval : 1;
    const ulong previous_in_interval = cache_enabled ? (ulong)processed % positive_interval : 0;
    ulong tokens_until_boundary = cache_enabled ? positive_interval - previous_in_interval : 0;
    ulong write_slot = 1;
    const uint g = h / (SSM_NUM_HEADS / SSM_NUM_GROUPS);
    const float A_lane = lane == 0 ? SSM_TO_FLOAT(A[h]) : 0.0f;
    const float A_value = sub_group_broadcast(A_lane, 0);
#if !SSM_JIT_USE_SLM
    float private_state[SSM_HEAD_DIM_BLOCK][SSM_STATE_ITERATIONS];
#endif

#pragma unroll
    for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
        const uint p = p_base + p_offset;
#pragma unroll
        for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
            const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
            if (p < SSM_HEAD_DIM && state_element < SSM_STATE_SIZE) {
                SSM_STATE_AT(p_offset, step, state_element) =
                    SSM_TO_FLOAT(recurrent_state_table[SSM_STATE_INDEX((uint)first_block, p, state_element)]);
            }
        }
    }

    for (long token = token_begin; token < token_end; ++token) {
        const uint token_idx = (uint)token;
#define SSM_TOKEN_INDEX token_idx
#include "selective_ssm_jit_recurrence.cl"
#undef SSM_TOKEN_INDEX

        if (cache_enabled) {
            const bool at_boundary = --tokens_until_boundary == 0;
            const bool at_sequence_end = token + 1 == token_end;
            if (at_boundary || at_sequence_end) {
                const ulong block_position = (ulong)block_begin + write_slot++;
                if (block_position < (ulong)block_end && block_position < (ulong)INPUT7_BATCH_NUM) {
                    const long block_id = (long)block_indices[block_position];
                    if (block_id >= 0 && (ulong)block_id < (ulong)INPUT5_BATCH_NUM) {
#pragma unroll
                        for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
                            const uint p = p_base + p_offset;
#pragma unroll
                            for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
                                const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
                                if (p < SSM_HEAD_DIM && state_element < SSM_STATE_SIZE) {
                                    recurrent_state_table[SSM_STATE_INDEX((uint)block_id, p, state_element)] =
                                        TO_INPUT5_TYPE(SSM_STATE_AT(p_offset, step, state_element));
                                }
                            }
                        }
                    }
                }
                if (at_boundary)
                    tokens_until_boundary = positive_interval;
            }
        }
    }
}

#undef SSM_TO_FLOAT
#undef SSM_DT_INDEX
#undef SSM_B_INDEX
#undef SSM_C_INDEX
#undef SSM_X_INDEX
#undef SSM_OUTPUT_INDEX
#undef SSM_STATE_INDEX
#undef SSM_STATE_AT
