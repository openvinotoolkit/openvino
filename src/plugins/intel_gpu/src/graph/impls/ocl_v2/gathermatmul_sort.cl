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
//
// JIT flags (default 0 for legacy OCL batched-prefill path):
//   EXPERT_MAJOR_ORDER  — pack each expert's rows contiguously; required by dnnl grouped src/dst.
//   EMIT_EXPERT_OFFSETS — append expert_offsets[N_ALL_EXPERTS]: cumulative exclusive end per expert.

KERNEL(bgm_sort)(
    OPTIONAL_SHAPE_INFO_ARG
    const global INPUT0_TYPE* indices,
    global int* group_expert_ids,
    global int* group_slot_ids,
    global int* group_offsets,
    global int* group_sizes,
    global int* token_map,
    global int* num_groups
#if EMIT_EXPERT_OFFSETS
    , global int* expert_offsets
#endif
) {
#if SORT_LOCAL_GROUPS
    // Single work-group: local histogram, serial compaction in local memory, parallel scatter.
    // Rows of one group may land in any order; every consumer addresses them through token_map.
    __local int hist[SORT_LOCAL_GROUPS];
    __local int group_of[SORT_LOCAL_GROUPS];
    __local int cursor[SORT_LOCAL_GROUPS];
    const int lid = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);
    const int n_tokens = N_TOKENS;
    const int top_k = TOP_K;
    const int n_all_experts = N_ALL_EXPERTS;
    const int rows = n_tokens * top_k;
    for (int i = lid; i < SORT_LOCAL_GROUPS; i += lsize)
        hist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = lid; r < rows; r += lsize)
        atomic_inc(&hist[(r % top_k) * n_all_experts + (int)indices[r]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int g = 0;
        int offset = 0;
  #if EXPERT_MAJOR_ORDER
        for (int e = 0; e < n_all_experts; e++) {
            for (int s = 0; s < top_k; s++) {
  #else
        for (int s = 0; s < top_k; s++) {
            for (int e = 0; e < n_all_experts; e++) {
  #endif
                const int bin = s * n_all_experts + e;
                const int count = hist[bin];
                if (count > 0) {
                    group_expert_ids[g] = e;
                    group_slot_ids[g] = s;
                    group_offsets[g] = offset;
                    group_sizes[g] = count;
                    group_of[bin] = g;
                    cursor[g] = offset;
                    offset += count;
                    g++;
                }
            }
  #if EMIT_EXPERT_OFFSETS && EXPERT_MAJOR_ORDER
            expert_offsets[e] = offset;
  #endif
        }
        num_groups[0] = g;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = lid; r < rows; r += lsize) {
        const int bin = (r % top_k) * n_all_experts + (int)indices[r];
        token_map[atomic_inc(&cursor[group_of[bin]])] = r / top_k;
    }
#else
    if (get_local_id(0) != 0)
        return;
    int n_tokens = N_TOKENS;
    int top_k = TOP_K;
    int n_all_experts = N_ALL_EXPERTS;

    int max_groups = n_all_experts * top_k;

    int g = 0;
    int offset = 0;
    int total_groups = 0;

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
    for (int i = 0; i < max_groups; i++) {
        group_sizes[i] = -1;
    }
#if EXPERT_MAJOR_ORDER
    for (int e = 0; e < n_all_experts; e++) {
        for (int s = 0; s < top_k; s++) {
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
  #if EMIT_EXPERT_OFFSETS
        expert_offsets[e] = offset;
  #endif
    }
#else
    // Slot-major (legacy OCL batched_gemm dispatch).
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
#endif
    total_groups = g;
    num_groups[0] = total_groups;

    // Pass 3: scatter in token order, using group_offsets as per-group write cursors.
    for (int t = 0; t < n_tokens; t++) {
        for (int s = 0; s < top_k; s++) {
            int bin = s * n_all_experts + (int)indices[t * top_k + s];
            token_map[group_offsets[group_sizes[bin]]++] = t;
        }
    }
    // Cursors now hold each group's end; restore starts and sizes.
    for (int gi = total_groups - 1; gi >= 0; gi--) {
        int start = gi > 0 ? group_offsets[gi - 1] : 0;
        group_sizes[gi] = group_offsets[gi] - start;
        group_offsets[gi] = start;
    }
#endif
}
