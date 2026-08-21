// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"
#include "selective_ssm_jit_storage.cl"

#define SSM_DT_INDEX(token) (((b * SSM_SEQUENCE_SIZE + (token)) * SSM_NUM_HEADS) + h)
#define SSM_B_INDEX(token, state_element) \
    ((((b * SSM_SEQUENCE_SIZE + (token)) * SSM_NUM_GROUPS + g) * SSM_STATE_SIZE) + (state_element))
#define SSM_C_INDEX(token, state_element) SSM_B_INDEX(token, state_element)
#define SSM_X_INDEX(token, p) \
    (((((b * SSM_SEQUENCE_SIZE + (token)) * SSM_NUM_HEADS + h) * SSM_HEAD_DIM) + (p)))
#define SSM_OUTPUT_INDEX(token, p) SSM_X_INDEX(token, p)
#define SSM_STATE_INDEX(p, state_element) \
    (((((b * SSM_NUM_HEADS + h) * SSM_HEAD_DIM + (p)) * SSM_STATE_SIZE) + (state_element)))

REQD_SUB_GROUP_SIZE(SSM_SUBGROUP_SIZE)
KERNEL(SSM_JIT_KERNEL)(const __global INPUT0_TYPE* A,
                       const __global INPUT1_TYPE* dt,
                       const __global INPUT2_TYPE* B,
                       const __global INPUT3_TYPE* x,
                       const __global INPUT4_TYPE* C,
                       const __global INPUT5_TYPE* initial_state,
                       __global OUTPUT_TYPE* output,
                       __global OUTPUT1_TYPE* output_state,
                       uint sequence_size
#if SSM_JIT_USE_SLM
                       , __local float* slm_state
#endif
                       ) {
    const uint lane = get_sub_group_local_id();
    const uint h = get_global_id(1);
    const uint b = get_global_id(2);
    const uint p_base = get_group_id(0) * SSM_HEAD_DIM_BLOCK;
    const uint g = h / (SSM_NUM_HEADS / SSM_NUM_GROUPS);
    const float A_lane = lane == 0 ? SSM_TO_FLOAT(A[h]) : 0.0f;
    const float A_value = sub_group_broadcast(A_lane, 0);
#if !SSM_JIT_USE_SLM
    float private_state[SSM_HEAD_DIM_BLOCK][SSM_STATE_ITERATIONS];
#endif

#define SSM_JIT_LOAD_STATE(p, state_element) \
    SSM_TO_FLOAT(initial_state[SSM_STATE_INDEX(p, state_element)])
#include "selective_ssm_jit_load_state.cl"
#undef SSM_JIT_LOAD_STATE

    for (uint token = 0; token < sequence_size; ++token) {
#define SSM_TOKEN_INDEX token
#include "selective_ssm_jit_recurrence.cl"
#undef SSM_TOKEN_INDEX
    }

#define SSM_JIT_STORE_STATE(p, state_element, value) \
    output_state[SSM_STATE_INDEX(p, state_element)] = TO_OUTPUT1_TYPE(value)
#include "selective_ssm_jit_store_state.cl"
#undef SSM_JIT_STORE_STATE
}

#undef SSM_TO_FLOAT
#undef SSM_DT_INDEX
#undef SSM_B_INDEX
#undef SSM_C_INDEX
#undef SSM_X_INDEX
#undef SSM_OUTPUT_INDEX
#undef SSM_STATE_INDEX
#undef SSM_STATE_AT
