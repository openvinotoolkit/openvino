// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Reference kernel for KV-cache partial reorder.
//
// It copies a single token from src_id to dst_id for both KEY_CACHE and VALUE_CACHE.
//
// IMPORTANT:
// - src_id / dst_id in block_update_indices are treated as per-sequence *token ids* in the global (batched)
//   token space, where subsequence_begins[seq] is the base offset for that sequence.
// - The corresponding (block_in_seq, slot_in_block) are derived as:
//     local = id - subsequence_begins[seq]
//     block_in_seq = local / PAGED_ATTENTION_BLOCK_SIZE
//     slot        = local % PAGED_ATTENTION_BLOCK_SIZE
// - block_in_seq is mapped to physical block id via:
//     phys_block = block_indices[block_indices_begins[seq] + block_in_seq]

// block_update_indices is interpreted as a flat array of int32 pairs:
//   { src_id_0, dst_id_0, src_id_1, dst_id_1, ... }
//
// block_update_indices_begins is a prefix-sum array (length: sequences_num + 1) in units of *reorder ops*:
//   { 0, seq0_ops, seq0_ops + seq1_ops, ... }
//
// NOTE: This is a simple reference implementation for uncompressed KV cache layout:
// - key_cache:   [num_blocks, KV_HEADS_NUM, ADJUSTED_K_HEAD_SIZE, ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE]
// - value_cache: [num_blocks, KV_HEADS_NUM, ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE, ADJUSTED_V_HEAD_SIZE]
//
// The dispatch is expected to be:
//   gws = { sequences_num, KV_HEADS_NUM, SUBGROUP_SIZE }
//   lws = { 1, 1, SUBGROUP_SIZE }

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUBGROUP_SIZE)))
KERNEL(pa_kv_cache_reorder)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* block_indices,
    __global const INPUT1_TYPE* block_indices_begins,
    __global const INPUT2_TYPE* block_update_indices,
    __global const INPUT3_TYPE* block_update_indices_begins,
    __global const INPUT4_TYPE* subsequence_begins,
    __global OUTPUT_TYPE* key_cache,
    __global OUTPUT1_TYPE* value_cache
) {
    const uint seq_idx = (uint)get_global_id(0);
    const uint head_idx = (uint)get_global_id(1);
    const uint sglid = (uint)get_local_id(2);

    const uint block_indices_base = (uint)block_indices_begins[seq_idx];
    const uint subseq_begin = (uint)subsequence_begins[seq_idx];
    const uint op_begin = (uint)block_update_indices_begins[seq_idx];
    const uint op_end = (uint)block_update_indices_begins[seq_idx + 1];
    const uint blocks_in_seq = (uint)block_indices_begins[seq_idx + 1] - block_indices_base;

    // Sequentially apply all reorder ops for this sequence to avoid data races
    // for overlapping src/dst slots.
    for (uint op = op_begin; op < op_end; op++) {
        const uint pair_base = op * 2;
        const int src_i = (int)block_update_indices[pair_base + 0];
        const int dst_i = (int)block_update_indices[pair_base + 1];

        // Defensive checks: negative or out-of-sequence token ids would lead to OOB.
        if (src_i < 0 || dst_i < 0)
            continue;

        const uint src_id = (uint)src_i;
        const uint dst_id = (uint)dst_i;

        if (src_id < subseq_begin || dst_id < subseq_begin)
            continue;

        uint src_block = 0;
        uint src_slot = 0;
        uint dst_block = 0;
        uint dst_slot = 0;
        const uint local_src = src_id - subseq_begin;
        const uint src_block_in_seq = local_src / PAGED_ATTENTION_BLOCK_SIZE;
        if (src_block_in_seq >= blocks_in_seq)
            continue;
        src_slot = local_src - src_block_in_seq * PAGED_ATTENTION_BLOCK_SIZE;
        src_block = (uint)block_indices[block_indices_base + src_block_in_seq];

        const uint local_dst = dst_id - subseq_begin;
        const uint dst_block_in_seq = local_dst / PAGED_ATTENTION_BLOCK_SIZE;
        if (dst_block_in_seq >= blocks_in_seq)
            continue;
        dst_slot = local_dst - dst_block_in_seq * PAGED_ATTENTION_BLOCK_SIZE;
        dst_block = (uint)block_indices[block_indices_base + dst_block_in_seq];
        // KEY_CACHE: [block, head, k_hidden, token]
        #ifdef IS_KEY_BY_CHANNEL
        const uint key_src_base = OUTPUT_OFFSET + src_block * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                                  head_idx * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_dst_base = OUTPUT_OFFSET + dst_block * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                                  head_idx * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        #else
        const uint key_src_base = OUTPUT_OFFSET + src_block * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                  head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_dst_base = OUTPUT_OFFSET + dst_block * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                  head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        #endif
        // VALUE_CACHE: [block, head, token, v_hidden]
        const uint val_src_base = OUTPUT1_OFFSET + src_block * KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE +
                                  head_idx * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE;
        const uint val_dst_base = OUTPUT1_OFFSET + dst_block * KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE +
                                  head_idx * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE;
        #if !IS_KV_COMPRESSED
            for (uint k = sglid; k < (uint)K_HEAD_SIZE; k += (uint)SUBGROUP_SIZE) {
                const uint src_off = key_src_base + k * PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                const uint dst_off = key_dst_base + k * PAGED_ATTENTION_BLOCK_SIZE + dst_slot;
                key_cache[dst_off] = key_cache[src_off];
            }

            for (uint v = sglid; v < (uint)V_HEAD_SIZE; v += (uint)SUBGROUP_SIZE) {
                const uint src_off = val_src_base + src_slot * V_HEAD_SIZE + v;
                const uint dst_off = val_dst_base + dst_slot * V_HEAD_SIZE + v;
                value_cache[dst_off] = value_cache[src_off];
            }
        #else // IS_KV_COMPRESSED
            #ifdef IS_KEY_BY_CHANNEL
                // to be implemented, key by channel: need re-quantize using dst block scale/zp?
            #else
                // per-token quantization: copy quantized values and comp data for token
                for (uint k = sglid; k < (uint)K_HEAD_SIZE; k += (uint)SUBGROUP_SIZE) { // only copy to K_HEAD_SIZE
                    const uint src_off = key_src_base + k * PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                    const uint dst_off = key_dst_base + k * PAGED_ATTENTION_BLOCK_SIZE + dst_slot;
                    key_cache[dst_off] = key_cache[src_off];
                }

                if (sglid == 0) {
                    const uint comp_src_base = key_src_base + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
                    const uint comp_dst_base = key_dst_base + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
                    UNCOMPRESSED_TYPE* src_comp_ptr = key_cache + comp_src_base;
                    UNCOMPRESSED_TYPE* dst_comp_ptr = key_cache + comp_dst_base;
                    dst_comp_ptr[dst_slot] = src_comp_ptr[src_slot];
                    dst_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + dst_slot] = src_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + src_slot];
                }
            #endif

            // value cache: per-token quantization
            for (uint v = sglid; v < (uint)V_HEAD_SIZE; v += (uint)SUBGROUP_SIZE) {
                const uint src_off = val_src_base + src_slot * V_HEAD_SIZE + v;
                const uint dst_off = val_dst_base + dst_slot * V_HEAD_SIZE + v;
                value_cache[dst_off] = value_cache[src_off];
            }

            if (sglid == 0) {
                const uint comp_src_base = val_src_base + V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
                const uint comp_dst_base = val_dst_base + V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
                UNCOMPRESSED_TYPE* src_comp_ptr = value_cache + comp_src_base;
                UNCOMPRESSED_TYPE* dst_comp_ptr = value_cache + comp_dst_base;
                dst_comp_ptr[dst_slot] = src_comp_ptr[src_slot];
                dst_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + dst_slot] = src_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + src_slot];
            }
        #endif
    }
}
