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

// Copies activation rows into a contiguous per-group buffer (driven by sort output).
// Dispatch: z=max_groups, y=token-within-group, x=K tile.

#define COPY_BLOCK 8

KERNEL(bgm_gather)(
    OPTIONAL_SHAPE_INFO_ARG
    const global half* input_ptr,
    global half* gathered_ptr,
    const global int* token_map,
    const global int* group_slot_ids,
    const global int* group_offsets,
    const global int* group_sizes,
    const global int* num_groups,
    int k_val
) {
    int group_id = get_global_id(2);
    if (group_id >= num_groups[0])
        return;

    int n_tokens_in_group = group_sizes[group_id];
    int token_in_group = get_global_id(1);
    if (token_in_group >= n_tokens_in_group)
        return;

    int offset = group_offsets[group_id];
    int slot = group_slot_ids[group_id];
    int n_act = N_ACTIVATED_EXPERTS;
    int a_slot = min(slot, n_act - 1);  // clamp for broadcast case
    int n_tokens = N_TOKENS;

    int orig_token = token_map[offset + token_in_group];

    // Source: A[a_slot, orig_token, :]
    const global half* src = input_ptr + (a_slot * n_tokens + orig_token) * k_val;
    // Dest: gathered_A[offset + token_in_group, :]
    global half* dst = gathered_ptr + (offset + token_in_group) * k_val;

    // Copy K elements using vectorized loads
    int k_offset = get_global_id(0) * COPY_BLOCK;
    if (k_offset + COPY_BLOCK <= k_val) {
        half8 val = vload8(0, src + k_offset);
        vstore8(val, 0, dst + k_offset);
    } else {
        // Tail handling
        for (int i = k_offset; i < k_val && i < k_offset + COPY_BLOCK; i++) {
            dst[i] = src[i];
        }
    }
}
