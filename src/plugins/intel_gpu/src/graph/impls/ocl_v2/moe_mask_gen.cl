// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#if EXPERT_SORT_HIST || EXPERT_SORT_SCATTER
// Multi work-group expert sort for many (token, top-k slot) rows. Each work-group owns BLOCK_SIZE consecutive rows.
// hist: per-block expert histogram. scatter: block base per expert + stable rank inside the block.

#if EXPERT_SORT_HIST
KERNEL(moe_mask_gen)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* topk_idx,
    __global int* block_counts)
{
    const int rows = INPUT0_BATCH_NUM * NUM_EXPERTS_PER_TOKEN;
    const int lid = get_local_id(0);
    const int block = get_group_id(0);
    const int s = block * BLOCK_SIZE + lid;

    __local int hist[NUM_EXPERTS];
    for (int e = lid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        hist[e] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (s < rows) {
        atomic_inc(&hist[(int)topk_idx[s]]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = lid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        block_counts[block * NUM_EXPERTS + e] = hist[e];
    }
}
#else
KERNEL(moe_mask_gen)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* topk_idx,
    __global OUTPUT_TYPE* token_index,
    __global OUTPUT1_TYPE* expert_end_offsets,
    __global int* row_lut,
    const __global int* block_counts)
{
    const int rows = INPUT0_BATCH_NUM * NUM_EXPERTS_PER_TOKEN;
    const int lid = get_local_id(0);
    const int block = get_group_id(0);
    const int num_blocks = get_num_groups(0);
    const int s = block * BLOCK_SIZE + lid;

    __local int expert_base[NUM_EXPERTS];
    __local int expert_total[NUM_EXPERTS];
    __local int ids[BLOCK_SIZE];

    for (int e = lid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        int before = 0;
        int total = 0;
        for (int b = 0; b < num_blocks; ++b) {
            const int c = block_counts[b * NUM_EXPERTS + e];
            before += (b < block) ? c : 0;
            total += c;
        }
        expert_base[e] = before;
        expert_total[e] = total;
    }
    ids[lid] = (s < rows) ? (int)topk_idx[s] : -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int start = 0;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            expert_base[e] += start;
            start += expert_total[e];
            if (block == 0) {
                expert_end_offsets[e] = start;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (s < rows) {
        const int e = ids[lid];
        int rank = 0;
        for (int j = 0; j < lid; ++j) {
            rank += (ids[j] == e) ? 1 : 0;
        }
        const int row = expert_base[e] + rank;
        token_index[row] = s / NUM_EXPERTS_PER_TOKEN;
        row_lut[s] = row;
    }
}
#endif

#else

KERNEL(moe_mask_gen)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* topk_idx,
    __global OUTPUT_TYPE* tokens_per_expert,
    __global OUTPUT1_TYPE* experts_info_start_idx
#if !EXPERT_SORT
    , __global OUTPUT2_TYPE* experts_id,
    __global OUTPUT3_TYPE* tokens_lens_per_expert,
    __global OUTPUT4_TYPE* num_actual_used_experts
#endif
#if ROW_LUT_ENABLE
    , __global int* row_lut
#endif
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
#if !EXPERT_SORT
    int experts_id_iter = work_group_scan_exclusive_add(is_used);

    if ((expert_id + 1) == get_local_size(0)) {
        num_actual_used_experts[0] = experts_id_iter + is_used;
    }
#endif

    #if ONEDNN_GROUPED_GEMM_USED
        experts_info_start_idx[expert_id] = tokens_per_expert_iter + num_tokens_per_curr_expert;
    #endif

    if (num_tokens_per_curr_expert == 0) {
        return;
    }

#if !EXPERT_SORT
    #if !ONEDNN_GROUPED_GEMM_USED
        experts_info_start_idx[experts_id_iter] = tokens_per_expert_iter;
    #endif
    experts_id[experts_id_iter] = expert_id;
    tokens_lens_per_expert[experts_id_iter] = num_tokens_per_curr_expert;
#endif

    int token_idx = 0;
    for (int t = 0; t < num_tokens; ++t) {
        for (int e = 0; e < NUM_EXPERTS_PER_TOKEN; ++e) {
            if (topk_idx[token_idx] == expert_id) {
#if ROW_LUT_ENABLE
                // inverse mapping: (token, topk) pair -> gathered row
                row_lut[token_idx] = tokens_per_expert_iter;
#endif
                tokens_per_expert[tokens_per_expert_iter] = t;
                tokens_per_expert_iter += 1;
            }
            token_idx += 1;
        }
    }
}
#endif
