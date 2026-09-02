// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "selective_ssm_type_utils.cl"

#if SSM_JIT_PRECOMPUTE_DA && !SSM_PAGED
#    error "Precomputed dA is supported only by the paged SelectiveSSM JIT kernel"
#endif

// Specialize tensor indexing for paged and dense layouts.
#if SSM_PAGED
#    if IS_DYNAMIC
#        define SSM_RUNTIME_INDEX(value) ((size_t)(value))
#        define SSM_TOKEN_TYPE long
#        define SSM_COUNTDOWN_TYPE ulong
#        define SSM_A_INDEX GET_DATA_INDEX(INPUT0, h, 0, 0, 0)
#        define SSM_DT_INDEX(token) GET_DATA_INDEX(INPUT1, token, h, 0, 0)
#        define SSM_B_INDEX(token, state_element) GET_DATA_INDEX(INPUT2, token, g, state_element, 0)
#        define SSM_C_INDEX(token, state_element) GET_DATA_INDEX(INPUT4, token, g, state_element, 0)
#        define SSM_X_INDEX(token, p) GET_DATA_INDEX(INPUT3, token, h, p, 0)
#        define SSM_OUTPUT_INDEX(token, p) GET_DATA_INDEX(OUTPUT, token, h, p, 0)
#        define SSM_STATE_INDEX(block, p, state_element) GET_DATA_INDEX(INPUT5, block, h, p, state_element)
#        define SSM_SUBSEQUENCE_INDEX(index) GET_DATA_INDEX(INPUT6, index, 0, 0, 0)
#        define SSM_BLOCK_INDEX(index) GET_DATA_INDEX(INPUT7, index, 0, 0, 0)
#        define SSM_BLOCK_BEGIN_INDEX(index) GET_DATA_INDEX(INPUT8, index, 0, 0, 0)
#        define SSM_PROCESSED_INDEX(index) GET_DATA_INDEX(INPUT9, index, 0, 0, 0)
#        define SSM_INTERVAL_INDEX(index) GET_DATA_INDEX(INPUT10, index, 0, 0, 0)
#    else
#        define SSM_RUNTIME_INDEX(value) ((uint)(value))
#        define SSM_TOKEN_TYPE uint
#        define SSM_COUNTDOWN_TYPE uint
#        define SSM_A_INDEX h
#        define SSM_DT_INDEX(token) (((token) * SSM_NUM_HEADS) + h)
#        define SSM_B_INDEX(token, state_element) \
            ((((token) * SSM_NUM_GROUPS + g) * SSM_STATE_SIZE) + (state_element))
#        define SSM_C_INDEX(token, state_element) SSM_B_INDEX(token, state_element)
#        define SSM_X_INDEX(token, p) (((((token) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM) + (p)))
#        define SSM_OUTPUT_INDEX(token, p) SSM_X_INDEX(token, p)
#        define SSM_STATE_INDEX(block, p, state_element) \
            ((((((block) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM + (p)) * SSM_STATE_SIZE) + (state_element)))
#        define SSM_SUBSEQUENCE_INDEX(index) (index)
#        define SSM_BLOCK_INDEX(index) (index)
#        define SSM_BLOCK_BEGIN_INDEX(index) (index)
#        define SSM_PROCESSED_INDEX(index) (index)
#        define SSM_INTERVAL_INDEX(index) (index)
#    endif
#else
#    define SSM_RUNTIME_INDEX(value) ((uint)(value))
#    define SSM_TOKEN_TYPE uint
#    define SSM_A_INDEX h
#    define SSM_DT_INDEX(token) (((b * SSM_SEQUENCE_SIZE + (token)) * SSM_NUM_HEADS) + h)
#    define SSM_B_INDEX(token, state_element) \
        ((((b * SSM_SEQUENCE_SIZE + (token)) * SSM_NUM_GROUPS + g) * SSM_STATE_SIZE) + (state_element))
#    define SSM_C_INDEX(token, state_element) SSM_B_INDEX(token, state_element)
#    define SSM_X_INDEX(token, p) \
        (((((b * SSM_SEQUENCE_SIZE + (token)) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM) + (p)))
#    define SSM_OUTPUT_INDEX(token, p) SSM_X_INDEX(token, p)
#    define SSM_STATE_INDEX(batch, p, state_element) \
        ((((((batch) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM + (p)) * SSM_STATE_SIZE) + (state_element)))
#endif

#if SSM_JIT_PRECOMPUTE_DA
#    define SSM_PRECOMPUTED_DA_INDEX(token) (((token) * SSM_NUM_HEADS) + h)
#endif

// Select recurrence-state storage at compile time for each device specialization.
#if SSM_JIT_USE_SLM
#    define SSM_STATE_AT(p_offset, step, state_element) slm_state[(p_offset) * SSM_STATE_SIZE + (state_element)]
#else
#    define SSM_STATE_AT(p_offset, step, state_element) private_state[p_offset][step]
#endif

REQD_SUB_GROUP_SIZE(SSM_SUBGROUP_SIZE)
KERNEL(selective_ssm_jit)(OPTIONAL_SHAPE_INFO_ARG
                       const __global INPUT0_TYPE* A,
                       const __global INPUT1_TYPE* dt,
                       const __global INPUT2_TYPE* B,
                       const __global INPUT3_TYPE* x,
                       const __global INPUT4_TYPE* C,
#if SSM_PAGED
                       __global INPUT5_TYPE* state,
                       const __global INPUT6_TYPE* subsequence_begins,
                       const __global INPUT7_TYPE* block_indices,
                       const __global INPUT8_TYPE* block_indices_begins,
                       const __global INPUT9_TYPE* num_processed_tokens,
                       const __global INPUT10_TYPE* cache_interval,
                       __global OUTPUT_TYPE* output
#else
                       const __global INPUT5_TYPE* state,
                       __global OUTPUT_TYPE* output,
                       __global OUTPUT1_TYPE* output_state,
                       uint sequence_size
#endif
#if SSM_JIT_PRECOMPUTE_DA
                       , const __global float* precomputed_dA
#endif
#if SSM_JIT_USE_SLM
                       , __local float* slm_state
#endif
                       ) {
    const uint lane = get_sub_group_local_id();

// Resolve paged recurrence and cache metadata before entering the common kernel body.
#if SSM_PAGED
#    if IS_DYNAMIC
    const uint h = (uint)get_global_id(1);
    const size_t seq = get_global_id(2);
    const uint p_base = (uint)get_group_id(0) * SSM_HEAD_DIM_BLOCK;
    const size_t tokens = INPUT3_BATCH_NUM;
    const size_t sequences = INPUT6_BATCH_NUM > 0 ? INPUT6_BATCH_NUM - 1 : 0;

    if (INPUT0_BATCH_NUM != SSM_NUM_HEADS ||
        INPUT1_BATCH_NUM != tokens || INPUT1_FEATURE_NUM != SSM_NUM_HEADS ||
        INPUT2_BATCH_NUM != tokens || INPUT2_FEATURE_NUM != SSM_NUM_GROUPS || INPUT2_SIZE_Y != SSM_STATE_SIZE ||
        INPUT3_FEATURE_NUM != SSM_NUM_HEADS || INPUT3_SIZE_Y != SSM_HEAD_DIM ||
        INPUT4_BATCH_NUM != tokens || INPUT4_FEATURE_NUM != SSM_NUM_GROUPS || INPUT4_SIZE_Y != SSM_STATE_SIZE ||
        INPUT5_FEATURE_NUM != SSM_NUM_HEADS || INPUT5_SIZE_Y != SSM_HEAD_DIM || INPUT5_SIZE_X != SSM_STATE_SIZE ||
        OUTPUT_BATCH_NUM != tokens || OUTPUT_FEATURE_NUM != SSM_NUM_HEADS || OUTPUT_SIZE_Y != SSM_HEAD_DIM ||
        INPUT8_BATCH_NUM < sequences + 1 || INPUT9_BATCH_NUM < sequences || INPUT10_BATCH_NUM < sequences)
        return;

    if (seq >= sequences || h >= SSM_NUM_HEADS || p_base >= SSM_HEAD_DIM)
        return;

    const long token_begin = (long)subsequence_begins[SSM_SUBSEQUENCE_INDEX(seq)];
    const long token_end = (long)subsequence_begins[SSM_SUBSEQUENCE_INDEX(seq + 1)];
    const long block_begin = (long)block_indices_begins[SSM_BLOCK_BEGIN_INDEX(seq)];
    const long block_end = (long)block_indices_begins[SSM_BLOCK_BEGIN_INDEX(seq + 1)];
#    else
    const uint h = get_global_id(1);
    const uint seq = get_global_id(2);
    const uint p_base = get_group_id(0) * SSM_HEAD_DIM_BLOCK;
    const long token_begin = (long)subsequence_begins[seq];
    const long token_end = (long)subsequence_begins[seq + 1];
    const long block_begin = (long)block_indices_begins[seq];
    const long block_end = (long)block_indices_begins[seq + 1];
#    endif

#    if IS_DYNAMIC
    if (token_begin < 0 || token_end < token_begin || (ulong)token_end > (ulong)tokens)
#    else
    if (token_begin < 0 || token_end < token_begin || (ulong)token_end > (ulong)SSM_TOKEN_COUNT)
#    endif
        return;
    if (token_begin == token_end)
        return;

    const SSM_TOKEN_TYPE recurrence_begin = (SSM_TOKEN_TYPE)token_begin;
    const SSM_TOKEN_TYPE recurrence_end = (SSM_TOKEN_TYPE)token_end;

    if (block_begin < 0 || block_end <= block_begin || (ulong)block_end > (ulong)INPUT7_BATCH_NUM) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token) {
#pragma unroll
                for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
                    const uint p = p_base + p_offset;
                    if (p < SSM_HEAD_DIM)
                        output[SSM_OUTPUT_INDEX(SSM_RUNTIME_INDEX(token), p)] = TO_OUTPUT_TYPE(0.0f);
                }
            }
        }
        return;
    }

    const long first_block = (long)block_indices[SSM_BLOCK_INDEX((size_t)block_begin)];
    if (first_block < 0 || (ulong)first_block >= (ulong)INPUT5_BATCH_NUM) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token) {
#pragma unroll
                for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
                    const uint p = p_base + p_offset;
                    if (p < SSM_HEAD_DIM)
                        output[SSM_OUTPUT_INDEX(SSM_RUNTIME_INDEX(token), p)] = TO_OUTPUT_TYPE(0.0f);
                }
            }
        }
        return;
    }

    const uint initial_state_block = (uint)first_block;
    const long processed_raw = (long)num_processed_tokens[SSM_PROCESSED_INDEX(seq)];
    const long interval = (long)cache_interval[SSM_INTERVAL_INDEX(seq)];
    const long processed = max(processed_raw, (long)0);
    const bool cache_enabled = interval > 0;
    const ulong positive_interval = cache_enabled ? (ulong)interval : 1;
    const ulong previous_in_interval = cache_enabled ? (ulong)processed % positive_interval : 0;
#    if IS_DYNAMIC
    SSM_COUNTDOWN_TYPE tokens_until_boundary = cache_enabled ? positive_interval - previous_in_interval : 0;
#    else
    SSM_COUNTDOWN_TYPE tokens_until_boundary =
        cache_enabled ? (uint)min(positive_interval - previous_in_interval, (ulong)0xffffffffu) : 0;
#    endif
    ulong write_slot = 1;
#else
    const uint h = get_global_id(1);
    const uint b = get_global_id(2);
    const uint p_base = get_group_id(0) * SSM_HEAD_DIM_BLOCK;
    const SSM_TOKEN_TYPE recurrence_begin = 0;
    const SSM_TOKEN_TYPE recurrence_end = sequence_size;
    const uint initial_state_block = b;
#endif

    // Load the initial recurrence state into the selected private or SLM storage.
    const uint g = h / (SSM_NUM_HEADS / SSM_NUM_GROUPS);
    const float A_lane = lane == 0 ? ssm_to_float(A[SSM_A_INDEX]) : 0.0f;
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
                    ssm_to_float(state[SSM_STATE_INDEX(initial_state_block, p, state_element)]);
            }
        }
    }

    // Apply the common SelectiveSSM recurrence for the selected token range.
    for (SSM_TOKEN_TYPE token = recurrence_begin; token < recurrence_end; ++token) {
#if SSM_PAGED && IS_DYNAMIC
        const size_t token_idx = (size_t)token;
#else
        const uint token_idx = (uint)token;
#endif
        const float dt_lane = lane == 0 ? ssm_to_float(dt[SSM_DT_INDEX(token_idx)]) : 0.0f;
        const float dt_value = sub_group_broadcast(dt_lane, 0);
#if SSM_JIT_PRECOMPUTE_DA
        const float dA_lane = lane == 0 ? precomputed_dA[SSM_PRECOMPUTED_DA_INDEX(token_idx)] : 0.0f;
#else
        const float dA_lane = lane == 0 ? exp(A_value * dt_value) : 0.0f;
#endif
        const float dA = sub_group_broadcast(dA_lane, 0);
        float input_scales[SSM_HEAD_DIM_BLOCK];
        float partial[SSM_HEAD_DIM_BLOCK];

#pragma unroll
        for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
            const uint p = p_base + p_offset;
            const float x_lane = lane == 0 && p < SSM_HEAD_DIM
                                     ? ssm_to_float(x[SSM_X_INDEX(token_idx, p)])
                                     : 0.0f;
            input_scales[p_offset] = sub_group_broadcast(x_lane, 0) * dt_value;
            partial[p_offset] = 0.0f;
        }

#pragma unroll
        for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
            const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
            if (state_element < SSM_STATE_SIZE) {
                const float b_value = ssm_to_float(B[SSM_B_INDEX(token_idx, state_element)]);
                const float c_value = ssm_to_float(C[SSM_C_INDEX(token_idx, state_element)]);
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
                output[SSM_OUTPUT_INDEX(token_idx, p)] = TO_OUTPUT_TYPE(total);
        }

        // Persist paged state snapshots at cache boundaries and at sequence completion.
#if SSM_PAGED
        const bool at_boundary = cache_enabled && --tokens_until_boundary == 0;
        const bool at_sequence_end = token + 1 == recurrence_end;
        if (at_boundary || at_sequence_end) {
            const ulong block_position = (ulong)block_begin + write_slot++;
            if (block_position < (ulong)block_end && block_position < (ulong)INPUT7_BATCH_NUM) {
                const long block_id = (long)block_indices[SSM_BLOCK_INDEX((size_t)block_position)];
                if (block_id >= 0 && (ulong)block_id < (ulong)INPUT5_BATCH_NUM) {
#pragma unroll
                    for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
                        const uint p = p_base + p_offset;
#pragma unroll
                        for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
                            const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
                            if (p < SSM_HEAD_DIM && state_element < SSM_STATE_SIZE) {
                                state[SSM_STATE_INDEX((uint)block_id, p, state_element)] =
                                    TO_INPUT5_TYPE(SSM_STATE_AT(p_offset, step, state_element));
                            }
                        }
                    }
                }
            }
            if (at_boundary) {
#    if IS_DYNAMIC
                tokens_until_boundary = positive_interval;
#    else
                tokens_until_boundary = (uint)min(positive_interval, (ulong)0xffffffffu);
#    endif
            }
        }
#endif
    }

    // Dense SelectiveSSM returns the final recurrence state through its second output.
#if !SSM_PAGED
#pragma unroll
    for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
        const uint p = p_base + p_offset;
#pragma unroll
        for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
            const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
            if (p < SSM_HEAD_DIM && state_element < SSM_STATE_SIZE) {
                output_state[SSM_STATE_INDEX(b, p, state_element)] =
                    TO_OUTPUT1_TYPE(SSM_STATE_AT(p_offset, step, state_element));
            }
        }
    }
#endif
}

#undef SSM_RUNTIME_INDEX
#undef SSM_TOKEN_TYPE
#undef SSM_COUNTDOWN_TYPE
#undef SSM_A_INDEX
#undef SSM_DT_INDEX
#undef SSM_B_INDEX
#undef SSM_C_INDEX
#undef SSM_X_INDEX
#undef SSM_OUTPUT_INDEX
#undef SSM_STATE_INDEX
#undef SSM_SUBSEQUENCE_INDEX
#undef SSM_BLOCK_INDEX
#undef SSM_BLOCK_BEGIN_INDEX
#undef SSM_PROCESSED_INDEX
#undef SSM_INTERVAL_INDEX
#undef SSM_PRECOMPUTED_DA_INDEX
#undef SSM_STATE_AT
