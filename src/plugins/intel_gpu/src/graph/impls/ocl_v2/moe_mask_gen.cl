// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(moe_mask_gen)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* topk_idx,
    __global OUTPUT_TYPE* tokens_per_expert,
    __global OUTPUT1_TYPE* experts_info_start_idx,
    __global OUTPUT2_TYPE* experts_id,
    __global OUTPUT3_TYPE* tokens_lens_per_expert,
    __global OUTPUT4_TYPE* num_actual_used_experts
#if SET_TOKEN_LEN
    , const int token_len
#endif
)
{
    const size_t expert_id = get_local_id(0);

#if SET_TOKEN_LEN
    int num_tokens = token_len;
#else
    int num_tokens = INPUT0_BATCH_NUM;
#endif

    int num_tokens_per_curr_expert = 0;
    for (int i = 0; i < num_tokens * NUM_EXPERTS_PER_TOKEN; ++i) {
        if (topk_idx[i] == expert_id) {
            num_tokens_per_curr_expert += 1;
        }
    }
    int is_used = (num_tokens_per_curr_expert > 0) ? 1 : 0;

    int tokens_per_expert_iter = work_group_scan_exclusive_add(num_tokens_per_curr_expert);
    int experts_id_iter = work_group_scan_exclusive_add(is_used);

    if ((expert_id + 1) == get_local_size(0)) {
        num_actual_used_experts[0] = experts_id_iter + is_used;
    }

    if (num_tokens_per_curr_expert == 0) {
        return;
    }

    experts_info_start_idx[experts_id_iter] = tokens_per_expert_iter;
    experts_id[experts_id_iter] = expert_id;
    tokens_lens_per_expert[experts_id_iter] = num_tokens_per_curr_expert;

    int token_idx = 0;
    for (int t = 0; t < num_tokens; ++t) {
        for (int e = 0; e < NUM_EXPERTS_PER_TOKEN; ++e) {
            if (topk_idx[token_idx] == expert_id) {
                tokens_per_expert[tokens_per_expert_iter] = t;
                tokens_per_expert_iter += 1;
            }
            token_idx += 1;
        }
    }
}
