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

// GatherMatmul Sort Kernel — groups tokens by expert within each slot via counting sort.
//
// Inputs:
//   indices[n_tokens, top_k] — per-token expert IDs (INPUT0)
//
// Outputs (internal buffers):
//   group_expert_ids[max_groups]  — expert_id per group
//   group_slot_ids[max_groups]    — slot per group
//   group_offsets[max_groups]     — start offset in gathered buffer
//   group_sizes[max_groups]       — token count per group
//   token_map[n_tokens * top_k]   — original token index per sorted position
//   num_groups[1]                 — actual count of active groups
//
// Dispatch: single workgroup (1,1,1) global, (1,1,1) local
// All work done sequentially — n_tokens * top_k is small enough.

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

    // Phase 1: Count tokens per (slot, expert) pair
    // Use group_sizes as histogram scratch space.
    // Max groups = n_all_experts * top_k
    int max_groups = n_all_experts * top_k;

    // Zero out histogram
    for (int i = 0; i < max_groups; i++) {
        group_sizes[i] = 0;
    }

    // Count: for each (token, slot), increment histogram[slot * n_all_experts + expert_id]
    for (int t = 0; t < n_tokens; t++) {
        for (int s = 0; s < top_k; s++) {
            int expert_id = (int)indices[t * top_k + s];
            int bin = s * n_all_experts + expert_id;
            group_sizes[bin]++;
        }
    }

    // Phase 2: Compact non-empty groups and compute offsets via prefix sum
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
                // Temporarily store bin->group mapping in group_sizes[bin]
                // We'll restore count after scatter
                group_sizes[bin] = g;
                // Store count temporarily — we need a separate array for counts
                // Reuse: store count in token_map region beyond n_tokens*top_k? No, it's exactly sized.
                // Instead, store count directly: we'll recompute from offsets or store separately.
                // Actually, let's just track counts in a second pass.
                offset += count;
                g++;
            }
        }
    }
    int total_groups = g;
    num_groups[0] = total_groups;

    // Phase 2b: Recompute group sizes from the compact representation
    // We lost the counts when we overwrote group_sizes[bin] with group index.
    // Recount from indices.
    // First, zero out the actual group_sizes for compact groups
    for (int i = 0; i < total_groups; i++) {
        group_sizes[i] = 0;
    }

    // Phase 3: Scatter token indices into sorted order
    // We need to map bin -> compact group index. We stored that in group_sizes[bin]
    // but just overwrote it. Let's redo this more carefully.

    // Restart: use a cleaner two-pass approach.
    // Reset everything and redo properly.

    // --- Clean implementation ---
    // Pass 1: histogram into a local array concept — but we're single WG.
    // Use token_map as scratch for histogram (it will be overwritten in scatter pass).

    // Re-zero token_map as scratch histogram
    for (int i = 0; i < max_groups; i++) {
        token_map[i] = 0;  // histogram[slot * n_all_experts + expert_id] = count
    }
    for (int t = 0; t < n_tokens; t++) {
        for (int s = 0; s < top_k; s++) {
            int expert_id = (int)indices[t * top_k + s];
            int bin = s * n_all_experts + expert_id;
            token_map[bin]++;
        }
    }

    // Pass 2: compact non-empty bins, prefix-sum for offsets
    g = 0;
    offset = 0;
    // We need bin->group_index mapping for scatter. Store in group_sizes temporarily (as scratch).
    // group_sizes has max_groups elements — enough.
    for (int i = 0; i < max_groups; i++) {
        group_sizes[i] = -1;  // -1 = no group for this bin
    }

    for (int s = 0; s < top_k; s++) {
        for (int e = 0; e < n_all_experts; e++) {
            int bin = s * n_all_experts + e;
            int count = token_map[bin];
            if (count > 0) {
                group_expert_ids[g] = e;
                group_slot_ids[g] = s;
                group_offsets[g] = offset;
                group_sizes[bin] = g;  // bin -> group index (temporary)
                offset += count;
                g++;
            }
        }
    }
    total_groups = g;
    num_groups[0] = total_groups;

    // Pass 3: Scatter token indices. Need per-group write counters.
    // Reuse token_map[0..max_groups-1] as write cursors (will be overwritten by actual scatter).
    // But token_map IS the output... so we need the cursors separate.
    // Use the first total_groups entries of token_map as cursors, then do the scatter
    // into the rest. But that overlaps.
    //
    // Solution: use group_offsets as base, track write position with a simple counter array.
    // We have at most max_groups groups. Store cursors in the first max_groups entries of
    // a scratch region. group_expert_ids has max_groups entries and we only use total_groups.
    // But we can't reuse those safely.
    //
    // Simplest: two-pass scatter. First, zero counters (reuse some buffer region).
    // Actually, just use the portion of token_map beyond n_tokens*top_k... but it's exactly sized.
    //
    // The cleanest approach: iterate tokens in order, for each (token, slot), find its group
    // via group_sizes[bin] (which currently holds group index), then write to
    // token_map[group_offsets[group] + cursor[group]], incrementing cursor.
    // For cursors, temporarily reuse the first total_groups slots of token_map itself,
    // since we write token_map left-to-right and cursors are read-then-incremented.
    // This is safe if group_offsets[0] >= total_groups. If not, we have a conflict.
    //
    // Safest: just allocate cursor on the stack. total_groups <= n_all_experts * top_k.
    // For typical MoE: 8 experts * 2 top_k = 16 groups max. Stack is fine.

    // We'll use group_expert_ids[total_groups .. max_groups-1] as cursor scratch.
    // This works because we only read group_expert_ids[0..total_groups-1] later.
    // Actually no — the host reads all of group_expert_ids. Let's not corrupt it.

    // Just iterate and count manually per group. O(n_tokens * top_k * total_groups) worst case
    // but total_groups is tiny. Or: simpler, reconstruct offsets.

    // Final clean approach: build token_map by iterating groups in order.
    // For each group g, iterate all tokens and collect those matching (slot, expert).
    // O(total_groups * n_tokens) — fine for small n.

    // Even simpler: since we have offsets, just do a second counting pass.
    // Zero out cursor array stored at the end of group_expert_ids buffer (which has max_groups entries).
    // We use entries [total_groups ... max_groups-1] as scratch for cursors of groups [0..total_groups-1].
    // But total_groups could equal max_groups. Not safe.

    // Ok, let's just do it the straightforward way with the mapping we have.
    // group_sizes[bin] currently = group_index. We need write cursors per group.
    // Let's repurpose group_offsets temporarily: save a copy of offsets, use group_offsets as cursors.
    // But we need offsets in the output!
    // Solution: write cursors = copy of group_offsets, then restore.

    // Copy offsets to cursor positions. Write cursors into num_groups buffer?
    // num_groups has only 1 element.

    // Pragmatic solution: just iterate groups, scan tokens, write matches.
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
        // Restore actual group size from write position
        group_sizes[gi] = write_pos - group_offsets[gi];
    }

    // Clean up: group_sizes[bin] entries beyond total_groups are garbage from the
    // bin->group mapping. That's fine — only [0, total_groups) are read by later stages.
}
