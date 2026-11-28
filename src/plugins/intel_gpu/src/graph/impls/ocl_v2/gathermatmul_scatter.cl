/*******************************************************************************
 * Copyright 2026 Intel Corporation
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
 ******************************************************************************/

// Inverse of bgm_gather: out[slot, orig_token, :] = packed[group_offsets[g] + tok, :].
// Dispatch: y=packed row (n_tokens * top_k), x=N tile.
// No reduction here — Multiply-by-router-weights + ReduceSum stay as separate downstream ops.

#define COPY_BLOCK 8

KERNEL(bgm_scatter)(
    OPTIONAL_SHAPE_INFO_ARG
    const global half* packed_ptr,
    global half* output_ptr,
    const global int* token_map,
    const global int* group_slot_ids,
    const global int* group_offsets,
    const global int* group_sizes,
    const global int* num_groups,
    int n_val
) {
    const int row = get_global_id(1);
    const int n_offset = get_global_id(0) * COPY_BLOCK;
    if (n_offset >= n_val)
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

    const int slot = group_slot_ids[lo];
    const int orig_token = token_map[row];
    const int n_tokens = N_TOKENS;

    const global half* src = packed_ptr + row * n_val;
    // Output leading dim is top_k (always per-slot), unlike bgm_gather's INPUT clamping.
    global half* dst = output_ptr + (slot * n_tokens + orig_token) * n_val;

    if (n_offset + COPY_BLOCK <= n_val) {
        half8 val = vload8(0, src + n_offset);
        vstore8(val, 0, dst + n_offset);
    } else {
        // Tail
        for (int i = n_offset; i < n_val && i < n_offset + COPY_BLOCK; i++) {
            dst[i] = src[i];
        }
    }
}
