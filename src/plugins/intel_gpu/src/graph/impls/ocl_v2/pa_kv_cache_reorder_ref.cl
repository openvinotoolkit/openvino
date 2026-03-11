// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Reference kernel for KV-cache partial reorder.

// - src_id / dst_id in block_update_indices are treated as per-sequence global token positions
// - The corresponding (block_in_seq, slot_in_block) are derived as:
//     block_in_seq = id / PAGED_ATTENTION_BLOCK_SIZE
//     slot        = id % PAGED_ATTENTION_BLOCK_SIZE
// - block_in_seq is mapped to physical block id as:
//     phys_block = block_indices[block_indices_begins[seq] + block_in_seq]

// block_update_indices as int32 pairs:
//   { src_id_0, dst_id_0, src_id_1, dst_id_1, ... }
//
// block_update_indices_begins:
//   { 0, seq0_slots, seq0_slots + seq1_slots, ... }
//

#define VEC_BLK_SIZE 8
#define VLOAD CAT(vload, VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)

#if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
inline void FUNC(requantize_and_store_by_channel_block)(__global OUTPUT_TYPE* key_cache,
                                                        const uint dst_base,
                                                        UNCOMPRESSED_TYPE* dequant_vals) {
    UNCOMPRESSED_TYPE min_value = UNCOMPRESSED_VAL_MAX;
    UNCOMPRESSED_TYPE max_value = UNCOMPRESSED_VAL_MIN;

    for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
        UNCOMPRESSED_TYPE val = dequant_vals[token];
        min_value = fmin(min_value, val);
        max_value = fmax(max_value, val);
    }

    // Re-quantize and store
    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE range = max_value - min_value;
    const ACCUMULATOR_TYPE min_range = fabs(max_value * 0.1f);
    if (range <= min_range) {
        // When the range is very small, expand the range to avoid zp overflow
        range += fmax(1.0f, min_range);
    }
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / range);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
    UNCOMPRESSED_TYPE scale = (UNCOMPRESSED_TYPE)(scale_tmp);
    UNCOMPRESSED_TYPE zp = (UNCOMPRESSED_TYPE)(zp_tmp);
    #undef ACCUMULATOR_TYPE

    for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
        OUTPUT_TYPE quantized = convert_char_rte(dequant_vals[token] * scale + zp);
        key_cache[dst_base + token] = quantized;
    }

    UNCOMPRESSED_TYPE* comp_ptr = key_cache + (dst_base + PAGED_ATTENTION_BLOCK_SIZE);
    comp_ptr[0] = 1.0f / scale;
    comp_ptr[1] = zp;
}
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(pa_kv_cache_reorder)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* block_indices,
    __global const INPUT1_TYPE* block_indices_begins,
    __global const INPUT2_TYPE* block_update_indices,
    __global const INPUT3_TYPE* block_update_indices_begins,
    __global OUTPUT_TYPE* key_cache,
    __global OUTPUT1_TYPE* value_cache
) {
    const uint seq_idx = (uint)get_global_id(0);
    const uint head_idx = (uint)get_global_id(1);
    const uint sglid = (uint)get_local_id(2);

    const uint block_indices_base = (uint)block_indices_begins[seq_idx];
    const uint pos_begin = (uint)block_update_indices_begins[seq_idx];
    const uint pos_end = (uint)block_update_indices_begins[seq_idx + 1];
    const uint blocks_in_seq = (uint)block_indices_begins[seq_idx + 1] - block_indices_base;

    // Sequentially apply all reorder slots for this sequence to avoid data races
    // for overlapping src/dst slots.
    for (uint pos = pos_begin; pos < pos_end; pos++) {
        const uint pair_base = pos * 2;
        const int src_i = (int)block_update_indices[pair_base + 0];
        const int dst_i = (int)block_update_indices[pair_base + 1];

        if (src_i < 0 || dst_i < 0)
            continue;

        const uint local_src = (uint)src_i;
        const uint local_dst = (uint)dst_i;

        uint src_block = 0;
        uint src_slot = 0;
        uint dst_block = 0;
        uint dst_slot = 0;
        const uint src_block_in_seq = local_src / PAGED_ATTENTION_BLOCK_SIZE;
        if (src_block_in_seq >= blocks_in_seq)
            continue;
        src_slot = local_src - src_block_in_seq * PAGED_ATTENTION_BLOCK_SIZE;
        src_block = (uint)block_indices[block_indices_base + src_block_in_seq];

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
                // Key by channel compressed mode
                // 1. load the original key value
                // 2. copy the src slot to dst slot
                // 3. if dst is in a different block, re-quantize dst block with updated scale/zp
                if (src_block == dst_block) {
                    // src and dst slot are in the same block
                    for (uint k = sglid; k < (uint)K_HEAD_SIZE; k += (uint)SUBGROUP_SIZE) {
                        const uint src_off = key_src_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                        const uint dst_base = key_dst_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
                        const uint dst_off = dst_base + dst_slot;

                        const uint dst_comp_off = key_dst_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PAGED_ATTENTION_BLOCK_SIZE;
                        UNCOMPRESSED_TYPE* dst_comp_ptr = key_cache + dst_comp_off;

                        const UNCOMPRESSED_TYPE dst_scale_inv = dst_comp_ptr[0];
                        const UNCOMPRESSED_TYPE dst_zp = dst_comp_ptr[1];

                        key_cache[dst_off] = key_cache[src_off];

                        UNCOMPRESSED_TYPE dequant_vals[PAGED_ATTENTION_BLOCK_SIZE];

                        for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
                            UNCOMPRESSED_TYPE val =
                                ((UNCOMPRESSED_TYPE)(key_cache[dst_base + token]) - dst_zp) * dst_scale_inv;
                            dequant_vals[token] = val;
                        }
                        FUNC_CALL(requantize_and_store_by_channel_block)(key_cache, dst_base, dequant_vals);
                    }
                } else {
                    // in this case, the dst_block is a full PAGED_ATTENTION_BLOCK_SIZE block
                    for (uint k = sglid; k < (uint)K_HEAD_SIZE; k += (uint)SUBGROUP_SIZE) {
                        const uint src_off = key_src_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                        const uint dst_base = key_dst_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;

                        const uint src_comp_off = key_src_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PAGED_ATTENTION_BLOCK_SIZE;
                        const uint dst_comp_off = key_dst_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PAGED_ATTENTION_BLOCK_SIZE;
                        UNCOMPRESSED_TYPE* src_comp_ptr = key_cache + src_comp_off;
                        UNCOMPRESSED_TYPE* dst_comp_ptr = key_cache + dst_comp_off;

                        const UNCOMPRESSED_TYPE src_scale_inv = src_comp_ptr[0];
                        const UNCOMPRESSED_TYPE src_zp = src_comp_ptr[1];
                        const UNCOMPRESSED_TYPE dst_scale_inv = dst_comp_ptr[0];
                        const UNCOMPRESSED_TYPE dst_zp = dst_comp_ptr[1];

                        const UNCOMPRESSED_TYPE src_val =
                            ((UNCOMPRESSED_TYPE)(key_cache[src_off]) - src_zp) * src_scale_inv;

                        UNCOMPRESSED_TYPE dequant_vals[PAGED_ATTENTION_BLOCK_SIZE];

                        for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
                            UNCOMPRESSED_TYPE val =
                                ((UNCOMPRESSED_TYPE)(key_cache[dst_base + token]) - dst_zp) * dst_scale_inv;
                            if (token == dst_slot) {
                                val = src_val;
                            }
                            dequant_vals[token] = val;
                        }
                        FUNC_CALL(requantize_and_store_by_channel_block)(key_cache, dst_base, dequant_vals);
                    }
                }
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
            #define VEC_SIZE 8
            uint v = sglid * VEC_SIZE;
            for (; v + VEC_SIZE - 1 < (uint)V_HEAD_SIZE; v += SUBGROUP_SIZE * VEC_SIZE) {
                const uint src_off = val_src_base + src_slot * V_HEAD_SIZE + v;
                const uint dst_off = val_dst_base + dst_slot * V_HEAD_SIZE + v;
                VSTORE(VLOAD(0, value_cache + src_off), 0, value_cache + dst_off);
            }

            // Handle leftovers
            for (; v < (uint)V_HEAD_SIZE; v += SUBGROUP_SIZE) {
                uint src_off = val_src_base + src_slot * V_HEAD_SIZE + v;
                uint dst_off = val_dst_base + dst_slot * V_HEAD_SIZE + v;
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