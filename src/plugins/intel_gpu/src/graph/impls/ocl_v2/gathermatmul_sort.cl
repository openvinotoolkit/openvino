/*******************************************************************************
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Counting sort: groups tokens by expert within each slot. Single-WG, sequential.

KERNEL(bgm_sort)(
    OPTIONAL_SHAPE_INFO_ARG
    const global INPUT0_TYPE* indices,
    global int* group_expert_ids,
    global int* group_slot_ids,
    global int* group_offsets,
    global int* group_sizes,
    global int* token_map,
    global int* num_groups
) {
    int n_tokens = N_TOKENS;
    int top_k = TOP_K;
    int n_all_experts = N_ALL_EXPERTS;

    int max_groups = n_all_experts * top_k;

    // Phase 1 (group_sizes scratch): histogram + compact pass — superseded below; kept
    // for behavioural parity. TODO: drop and run only the clean Pass 1-3.
    for (int i = 0; i < max_groups; i++) {
        group_sizes[i] = 0;
    }
    for (int t = 0; t < n_tokens; t++) {
        for (int s = 0; s < top_k; s++) {
            int expert_id = (int)indices[t * top_k + s];
            int bin = s * n_all_experts + expert_id;
            group_sizes[bin]++;
        }
    }
    int g = 0;
    int offset = 0;
    for (int s = 0; s < top_k; s++) {
        for (int e = 0; e < n_all_experts; e++) {
            int bin = s * n_all_experts + e;
            int count = group_sizes[bin];
            if (count > 0) {
                group_expert_ids[g] = e;
                group_slot_ids[g] = s;
                group_offsets[g] = offset;
                group_sizes[bin] = g;
                offset += count;
                g++;
            }
        }
    }
    int total_groups = g;
    num_groups[0] = total_groups;
    for (int i = 0; i < total_groups; i++) {
        group_sizes[i] = 0;
    }

    // Pass 1: histogram into token_map (overwritten in Pass 3 scatter).
    for (int i = 0; i < max_groups; i++) {
        token_map[i] = 0;
    }
    for (int t = 0; t < n_tokens; t++) {
        for (int s = 0; s < top_k; s++) {
            int expert_id = (int)indices[t * top_k + s];
            int bin = s * n_all_experts + expert_id;
            token_map[bin]++;
        }
    }

    // Pass 2: compact non-empty bins; group_sizes[bin]=group_idx as temp scratch.
    g = 0;
    offset = 0;
    for (int i = 0; i < max_groups; i++) {
        group_sizes[i] = -1;
    }
    for (int s = 0; s < top_k; s++) {
        for (int e = 0; e < n_all_experts; e++) {
            int bin = s * n_all_experts + e;
            int count = token_map[bin];
            if (count > 0) {
                group_expert_ids[g] = e;
                group_slot_ids[g] = s;
                group_offsets[g] = offset;
                group_sizes[bin] = g;
                offset += count;
                g++;
            }
        }
    }
    total_groups = g;
    num_groups[0] = total_groups;

    // Pass 3: scatter — per-group scan over tokens. O(total_groups * n_tokens), tiny in practice.
    for (int gi = 0; gi < total_groups; gi++) {
        int target_slot = group_slot_ids[gi];
        int target_expert = group_expert_ids[gi];
        int write_pos = group_offsets[gi];
        for (int t = 0; t < n_tokens; t++) {
            int expert_id = (int)indices[t * top_k + target_slot];
            if (expert_id == target_expert) {
                token_map[write_pos] = t;
                write_pos++;
            }
        }
        group_sizes[gi] = write_pos - group_offsets[gi];
    }
}
