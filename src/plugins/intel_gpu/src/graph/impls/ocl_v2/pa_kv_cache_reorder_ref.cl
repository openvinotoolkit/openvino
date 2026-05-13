// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/int4_utils.cl"

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
#define UINT4_RANGE 15

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

    UNCOMPRESSED_TYPE* comp_ptr = (UNCOMPRESSED_TYPE*)(key_cache + (dst_base + PAGED_ATTENTION_BLOCK_SIZE));
    comp_ptr[0] = 1.0f / scale;
    comp_ptr[1] = zp;
}
#endif

#if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL) && IS_INT4_COMPRESSED
inline void FUNC(requantize_and_store_by_channel_block_int4)(__global OUTPUT_TYPE* key_cache,
                                                             const uint dst_base,
                                                             UNCOMPRESSED_TYPE* dequant_vals0,
                                                             UNCOMPRESSED_TYPE* dequant_vals1) {
    UNCOMPRESSED_TYPE min_value0 = UNCOMPRESSED_VAL_MAX;
    UNCOMPRESSED_TYPE max_value0 = UNCOMPRESSED_VAL_MIN;
    UNCOMPRESSED_TYPE min_value1 = UNCOMPRESSED_VAL_MAX;
    UNCOMPRESSED_TYPE max_value1 = UNCOMPRESSED_VAL_MIN;

    for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
        UNCOMPRESSED_TYPE val0 = dequant_vals0[token];
        UNCOMPRESSED_TYPE val1 = dequant_vals1[token];
        min_value0 = fmin(min_value0, val0);
        max_value0 = fmax(max_value0, val0);
        min_value1 = fmin(min_value1, val1);
        max_value1 = fmax(max_value1, val1);
    }

    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE range0 = max_value0 - min_value0;
    ACCUMULATOR_TYPE range1 = max_value1 - min_value1;
    const ACCUMULATOR_TYPE min_range0 = fabs(max_value0 * 0.1f);
    const ACCUMULATOR_TYPE min_range1 = fabs(max_value1 * 0.1f);
    if (range0 <= min_range0) {
        range0 += fmax(1.0f, min_range0);
    }
    if (range1 <= min_range1) {
        range1 += fmax(1.0f, min_range1);
    }
    ACCUMULATOR_TYPE scale_tmp0 = (ACCUMULATOR_TYPE)((UINT4_RANGE) / range0);
    ACCUMULATOR_TYPE zp_tmp0 = (ACCUMULATOR_TYPE)(-min_value0 * scale_tmp0);
    ACCUMULATOR_TYPE scale_tmp1 = (ACCUMULATOR_TYPE)((UINT4_RANGE) / range1);
    ACCUMULATOR_TYPE zp_tmp1 = (ACCUMULATOR_TYPE)(-min_value1 * scale_tmp1);
    UNCOMPRESSED_TYPE scale0 = (UNCOMPRESSED_TYPE)(scale_tmp0);
    UNCOMPRESSED_TYPE zp0 = (UNCOMPRESSED_TYPE)(zp_tmp0);
    UNCOMPRESSED_TYPE scale1 = (UNCOMPRESSED_TYPE)(scale_tmp1);
    UNCOMPRESSED_TYPE zp1 = (UNCOMPRESSED_TYPE)(zp_tmp1);
    #undef ACCUMULATOR_TYPE

    for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
        char2 quantized = 0;
        quantized.s0 = convert_char_rte(dequant_vals0[token] * scale0 + zp0);
        quantized.s1 = convert_char_rte(dequant_vals1[token] * scale1 + zp1);
        key_cache[dst_base + token] = cvt_int8x2_to_uint4x2(quantized);
    }

    UNCOMPRESSED_TYPE* comp_ptr = (UNCOMPRESSED_TYPE*)(key_cache + (dst_base + PAGED_ATTENTION_BLOCK_SIZE));
    comp_ptr[0] = 1.0f / scale0;
    comp_ptr[1] = zp0;
    comp_ptr[2] = 1.0f / scale1;
    comp_ptr[3] = zp1;
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
        #if IS_KV_COMPRESSED && IS_INT4_COMPRESSED
        const uint phys_adjusted_k_head_size = PACKED_ADJUSTED_K_HEAD_SIZE;
        const uint phys_adjusted_v_head_size = PACKED_ADJUSTED_V_HEAD_SIZE;
        #else
        const uint phys_adjusted_k_head_size = ADJUSTED_K_HEAD_SIZE;
        const uint phys_adjusted_v_head_size = ADJUSTED_V_HEAD_SIZE;
        #endif

        #ifdef IS_KEY_BY_CHANNEL
        const uint key_src_base = OUTPUT_OFFSET + src_block * KV_HEADS_NUM * phys_adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                  head_idx * phys_adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_dst_base = OUTPUT_OFFSET + dst_block * KV_HEADS_NUM * phys_adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                  head_idx * phys_adjusted_k_head_size * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        #else
        const uint key_src_base = OUTPUT_OFFSET + src_block * KV_HEADS_NUM * phys_adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE +
                      head_idx * phys_adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_dst_base = OUTPUT_OFFSET + dst_block * KV_HEADS_NUM * phys_adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE +
                      head_idx * phys_adjusted_k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
        #endif
        // VALUE_CACHE: [block, head, token, v_hidden]
        const uint val_src_base = OUTPUT1_OFFSET + src_block * KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * phys_adjusted_v_head_size +
                      head_idx * PAGED_ATTENTION_BLOCK_SIZE * phys_adjusted_v_head_size;
        const uint val_dst_base = OUTPUT1_OFFSET + dst_block * KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * phys_adjusted_v_head_size +
                      head_idx * PAGED_ATTENTION_BLOCK_SIZE * phys_adjusted_v_head_size;
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
            #if IS_INT4_COMPRESSED
            const uint phys_k_head_size = PACKED_K_HEAD_SIZE;
            const uint phys_v_head_size = PACKED_V_HEAD_SIZE;
            #else
            const uint phys_k_head_size = K_HEAD_SIZE;
            const uint phys_v_head_size = V_HEAD_SIZE;
            #endif

            #ifdef IS_KEY_BY_CHANNEL
                // Key by channel compressed mode
                // 1. load the original key value
                // 2. copy the src slot to dst slot
                // 3. if dst is in a different block, re-quantize dst block with updated scale/zp
                #if IS_INT4_COMPRESSED
                    for (uint k = sglid; k < phys_k_head_size; k += (uint)SUBGROUP_SIZE) {
                        const uint src_off = key_src_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                        const uint dst_base = key_dst_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;

                        const uint src_comp_off = key_src_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PAGED_ATTENTION_BLOCK_SIZE;
                        const uint dst_comp_off = key_dst_base + k * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PAGED_ATTENTION_BLOCK_SIZE;
                        UNCOMPRESSED_TYPE* src_comp_ptr = key_cache + src_comp_off;
                        UNCOMPRESSED_TYPE* dst_comp_ptr = key_cache + dst_comp_off;

                        const UNCOMPRESSED_TYPE src_scale_inv0 = src_comp_ptr[0];
                        const UNCOMPRESSED_TYPE src_zp0 = src_comp_ptr[1];
                        const UNCOMPRESSED_TYPE src_scale_inv1 = src_comp_ptr[2];
                        const UNCOMPRESSED_TYPE src_zp1 = src_comp_ptr[3];
                        const UNCOMPRESSED_TYPE dst_scale_inv0 = dst_comp_ptr[0];
                        const UNCOMPRESSED_TYPE dst_zp0 = dst_comp_ptr[1];
                        const UNCOMPRESSED_TYPE dst_scale_inv1 = dst_comp_ptr[2];
                        const UNCOMPRESSED_TYPE dst_zp1 = dst_comp_ptr[3];

                        OUTPUT_TYPE src_packed = key_cache[src_off];
                        char2 src_unpacked = unpack_to_char(*(uint4x2_t *)&src_packed);

                        const UNCOMPRESSED_TYPE src_val0 = ((UNCOMPRESSED_TYPE)(src_unpacked.s0) - src_zp0) * src_scale_inv0;
                        const UNCOMPRESSED_TYPE src_val1 = ((UNCOMPRESSED_TYPE)(src_unpacked.s1) - src_zp1) * src_scale_inv1;

                        UNCOMPRESSED_TYPE dequant_vals0[PAGED_ATTENTION_BLOCK_SIZE];
                        UNCOMPRESSED_TYPE dequant_vals1[PAGED_ATTENTION_BLOCK_SIZE];

                        for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
                            OUTPUT_TYPE dst_packed = key_cache[dst_base + token];
                            char2 dst_unpacked = unpack_to_char(*(uint4x2_t *)&dst_packed);

                            UNCOMPRESSED_TYPE val0 = ((UNCOMPRESSED_TYPE)(dst_unpacked.s0) - dst_zp0) * dst_scale_inv0;
                            UNCOMPRESSED_TYPE val1 = ((UNCOMPRESSED_TYPE)(dst_unpacked.s1) - dst_zp1) * dst_scale_inv1;
                            if (token == dst_slot) {
                                val0 = src_val0;
                                val1 = src_val1;
                            }
                            dequant_vals0[token] = val0;
                            dequant_vals1[token] = val1;
                        }

                        FUNC_CALL(requantize_and_store_by_channel_block_int4)(key_cache, dst_base, dequant_vals0, dequant_vals1);
                    }
                #else
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
                #endif
            #else
                // per-token quantization: copy quantized values and comp data for token
                for (uint k = sglid; k < phys_k_head_size; k += (uint)SUBGROUP_SIZE) {
                    const uint src_off = key_src_base + k * PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                    const uint dst_off = key_dst_base + k * PAGED_ATTENTION_BLOCK_SIZE + dst_slot;
                    key_cache[dst_off] = key_cache[src_off];
                }

                if (sglid == 0) {
                    const uint comp_src_base = key_src_base + phys_k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
                    const uint comp_dst_base = key_dst_base + phys_k_head_size * PAGED_ATTENTION_BLOCK_SIZE;
                    UNCOMPRESSED_TYPE* src_comp_ptr = key_cache + comp_src_base;
                    UNCOMPRESSED_TYPE* dst_comp_ptr = key_cache + comp_dst_base;
                    dst_comp_ptr[dst_slot] = src_comp_ptr[src_slot];
                    dst_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + dst_slot] = src_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + src_slot];
                }
            #endif

            // value cache: per-token quantization
            #if IS_INT4_COMPRESSED
            for (uint v = sglid; v < phys_v_head_size; v += (uint)SUBGROUP_SIZE) {
                const uint src_off = val_src_base + src_slot * phys_v_head_size + v;
                const uint dst_off = val_dst_base + dst_slot * phys_v_head_size + v;
                value_cache[dst_off] = value_cache[src_off];
            }
            #else
            #define VEC_SIZE 8
            uint v = sglid * VEC_SIZE;
            for (; v + VEC_SIZE - 1 < (uint)V_HEAD_SIZE; v += SUBGROUP_SIZE * VEC_SIZE) {
                const uint src_off = val_src_base + src_slot * V_HEAD_SIZE + v;
                const uint dst_off = val_dst_base + dst_slot * V_HEAD_SIZE + v;
                VSTORE(VLOAD(0, value_cache + src_off), 0, value_cache + dst_off);
            }

            // Handle leftovers from the first non-vectorized index.
            const uint vec_tail_begin = ((uint)V_HEAD_SIZE / VEC_SIZE) * VEC_SIZE;
            for (v = vec_tail_begin + sglid; v < (uint)V_HEAD_SIZE; v += SUBGROUP_SIZE) {
                uint src_off = val_src_base + src_slot * V_HEAD_SIZE + v;
                uint dst_off = val_dst_base + dst_slot * V_HEAD_SIZE + v;
                value_cache[dst_off] = value_cache[src_off];
            }
            #endif

            if (sglid == 0) {
                const uint comp_src_base = val_src_base + phys_v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
                const uint comp_dst_base = val_dst_base + phys_v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
                UNCOMPRESSED_TYPE* src_comp_ptr = value_cache + comp_src_base;
                UNCOMPRESSED_TYPE* dst_comp_ptr = value_cache + comp_dst_base;
                dst_comp_ptr[dst_slot] = src_comp_ptr[src_slot];
                dst_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + dst_slot] = src_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + src_slot];
            }
        #endif
    }
}