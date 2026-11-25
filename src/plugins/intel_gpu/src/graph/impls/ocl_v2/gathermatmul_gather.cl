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
// Dispatch: y=gathered row (n_tokens * top_k), x=K tile.

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
    const int row = get_global_id(1);
    const int k_offset = get_global_id(0) * COPY_BLOCK;
    if (k_offset >= k_val)
        return;

    int lo = 0;
    int hi = num_groups[0] - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (group_offsets[mid] <= row)
            lo = mid;
        else
            hi = mid - 1;
    }
    if (row >= group_offsets[lo] + group_sizes[lo])
        return;

    const int a_slot = min(group_slot_ids[lo], N_ACTIVATED_EXPERTS - 1);
    const int orig_token = token_map[row];
    const global half* src = input_ptr + (a_slot * N_TOKENS + orig_token) * k_val;
    global half* dst = gathered_ptr + row * k_val;

    if (k_offset + COPY_BLOCK <= k_val) {
        vstore8(vload8(0, src + k_offset), 0, dst + k_offset);
    } else {
        for (int i = k_offset; i < k_val; i++)
            dst[i] = src[i];
    }
}
