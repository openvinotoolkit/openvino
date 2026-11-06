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

// The host compiles one variant per K page layout and picks it from the cache shape at dispatch (pa_kv_reorder.cpp:
// the kv-update model of EAGLE3 has no PagedAttention op to learn it from). IS_KEY_TOKEN_MAJOR covers every
// token-major page; split by the quant mode, because the BY_CHANNEL page also moves its comp out of the columns and,
// for INT4, packs two CHANNELS per byte (and relays the INT4 V page: trailing per-token comp, row pitch
// PACKED_V_HEAD_SIZE). Layouts: pa_kv_cache_update_ref.cl.
#ifdef IS_KEY_BY_CHANNEL
    #define KEY_BY_TOKEN_TM   0
    #define KEY_BY_CHANNEL_TM IS_KEY_TOKEN_MAJOR
#else
    #define KEY_BY_TOKEN_TM   IS_KEY_TOKEN_MAJOR
    #define KEY_BY_CHANNEL_TM 0
#endif

#if KEY_BY_CHANNEL_TM
// The token-major BY_CHANNEL counterparts of the two helpers below: the same arithmetic, expression for expression, so
// a reorder gives the same values on either page; only the addressing differs (a channel's tokens K_HEAD_SIZE bytes
// apart, its (inv_scale, zp) pair in the per-channel array after the data).
inline void FUNC(requantize_and_store_by_channel_tm)(__global OUTPUT_TYPE* key_cache,
                                                     const uint dst_data,
                                                     UNCOMPRESSED_TYPE* comp_ptr,
                                                     UNCOMPRESSED_TYPE* dequant_vals) {
    UNCOMPRESSED_TYPE min_value = UNCOMPRESSED_VAL_MAX;
    UNCOMPRESSED_TYPE max_value = UNCOMPRESSED_VAL_MIN;

    for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
        UNCOMPRESSED_TYPE val = dequant_vals[token];
        min_value = fmin(min_value, val);
        max_value = fmax(max_value, val);
    }

    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE range = max_value - min_value;
    const ACCUMULATOR_TYPE min_range = fabs(max_value * 0.1f);
    if (range <= min_range) {
        range += fmax(1.0f, min_range);
    }
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / range);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
    UNCOMPRESSED_TYPE scale = (UNCOMPRESSED_TYPE)(scale_tmp);
    UNCOMPRESSED_TYPE zp = (UNCOMPRESSED_TYPE)(zp_tmp);
    #undef ACCUMULATOR_TYPE

    for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
        key_cache[dst_data + token * K_HEAD_SIZE] = convert_char_rte(dequant_vals[token] * scale + zp);
    }

    comp_ptr[0] = 1.0f / scale;
    comp_ptr[1] = zp;
}

#if IS_INT4_COMPRESSED
// Quantizes one channel's PAGED_ATTENTION_BLOCK_SIZE values to nibbles (q) and its (inv_scale, zp) pair (comp); the
// caller packs two channels per byte.
inline void FUNC(requantize_by_channel_int4_tm)(UNCOMPRESSED_TYPE* dequant_vals, char* q, UNCOMPRESSED_TYPE* comp) {
    UNCOMPRESSED_TYPE min_value = UNCOMPRESSED_VAL_MAX;
    UNCOMPRESSED_TYPE max_value = UNCOMPRESSED_VAL_MIN;
    for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
        min_value = fmin(min_value, dequant_vals[t]);
        max_value = fmax(max_value, dequant_vals[t]);
    }

    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE range = (max_value == min_value) ? 0.001f : (ACCUMULATOR_TYPE)(max_value - min_value);
    const ACCUMULATOR_TYPE min_range = fabs(max_value * 0.1f);
    if (range <= min_range) {
        range += fmax(1.0f, min_range);
    }
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((UINT4_RANGE) / range);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp);
    #undef ACCUMULATOR_TYPE

    for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
        q[t] = (char)clamp(convert_int_rte((float)(dequant_vals[t] * scale_tmp + zp_tmp)), 0, UINT4_RANGE);
    }
    comp[0] = (UNCOMPRESSED_TYPE)(1.0f / scale_tmp);
    comp[1] = (UNCOMPRESSED_TYPE)zp_tmp;
}
#endif
#endif  // KEY_BY_CHANNEL_TM

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
                                                             UNCOMPRESSED_TYPE* dequant_vals) {
    UNCOMPRESSED_TYPE min_value = UNCOMPRESSED_VAL_MAX;
    UNCOMPRESSED_TYPE max_value = UNCOMPRESSED_VAL_MIN;
    for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
        min_value = fmin(min_value, dequant_vals[t]);
        max_value = fmax(max_value, dequant_vals[t]);
    }

    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE range = (max_value == min_value) ? 0.001f : (ACCUMULATOR_TYPE)(max_value - min_value);
    const ACCUMULATOR_TYPE min_range = fabs(max_value * 0.1f);
    if (range <= min_range) {
        range += fmax(1.0f, min_range);
    }
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((UINT4_RANGE) / range);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp);
    #undef ACCUMULATOR_TYPE

    for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; t += U4_ELEMS_PER_BYTE) {
        char q0 = (char)clamp(convert_int_rte((float)(dequant_vals[t]     * scale_tmp + zp_tmp)), 0, UINT4_RANGE);
        char q1 = (char)clamp(convert_int_rte((float)(dequant_vals[t + 1] * scale_tmp + zp_tmp)), 0, UINT4_RANGE);
        char2 res = {q0, q1};
        key_cache[dst_base + t / U4_ELEMS_PER_BYTE] = cvt_int8x2_to_uint4x2(res);
    }

    UNCOMPRESSED_TYPE* comp_ptr = (UNCOMPRESSED_TYPE*)(key_cache + dst_base + PACKED_PAGED_ATTENTION_BLOCK_SIZE);
    comp_ptr[0] = (UNCOMPRESSED_TYPE)(1.0f / scale_tmp);
    comp_ptr[1] = (UNCOMPRESSED_TYPE)zp_tmp;
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
            #if IS_KV_COMPRESSED && IS_INT4_COMPRESSED
            const uint phys_adjusted_block_size = PACKED_ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
            #else
            const uint phys_adjusted_block_size = ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
            #endif
        const uint key_src_base = OUTPUT_OFFSET + src_block * KV_HEADS_NUM * phys_adjusted_k_head_size * phys_adjusted_block_size +
                  head_idx * phys_adjusted_k_head_size * phys_adjusted_block_size;
        const uint key_dst_base = OUTPUT_OFFSET + dst_block * KV_HEADS_NUM * phys_adjusted_k_head_size * phys_adjusted_block_size +
                  head_idx * phys_adjusted_k_head_size * phys_adjusted_block_size;
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
            // Token-major K makes head dims contiguous within a page (like the value copy below);
            // d-major makes tokens contiguous. Page bases above are products and so unchanged.
            for (uint k = sglid; k < (uint)K_HEAD_SIZE; k += (uint)SUBGROUP_SIZE) {
            #if KEY_BY_TOKEN_TM
                const uint src_off = key_src_base + src_slot * K_HEAD_SIZE + k;
                const uint dst_off = key_dst_base + dst_slot * K_HEAD_SIZE + k;
            #else
                const uint src_off = key_src_base + k * PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                const uint dst_off = key_dst_base + k * PAGED_ATTENTION_BLOCK_SIZE + dst_slot;
            #endif
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
                #if KEY_BY_CHANNEL_TM && IS_INT4_COMPRESSED
                    // Token-major INT4 page: byte b of token row t (pitch K_HEAD_SIZE / 2) holds channels 2b (low nibble)
                    // and 2b + 1 (high), and one (inv_scale, zp) pair per channel follows the data. A lane owns a
                    // byte column -- both of its channels -- so no byte is written by two lanes.
                    const uint row_bytes = K_HEAD_SIZE / U4_ELEMS_PER_BYTE;
                    for (uint b = sglid; b < row_bytes; b += (uint)SUBGROUP_SIZE) {
                        if (src_block == dst_block) {
                            // Same page, same pairs: move both channels' nibbles of the token as one byte, no
                            // requantization -- the d-major fast path, two channels at a time.
                            key_cache[key_dst_base + dst_slot * row_bytes + b] = key_cache[key_src_base + src_slot * row_bytes + b];
                        } else {
                            char q[U4_ELEMS_PER_BYTE][PAGED_ATTENTION_BLOCK_SIZE];
                            unroll_for (uint n = 0; n < U4_ELEMS_PER_BYTE; ++n) {
                                const uint c = b * U4_ELEMS_PER_BYTE + n;
                                UNCOMPRESSED_TYPE* src_comp_ptr =
                                    (UNCOMPRESSED_TYPE*)(key_cache + key_src_base + row_bytes * PAGED_ATTENTION_BLOCK_SIZE) + 2 * c;
                                UNCOMPRESSED_TYPE* dst_comp_ptr =
                                    (UNCOMPRESSED_TYPE*)(key_cache + key_dst_base + row_bytes * PAGED_ATTENTION_BLOCK_SIZE) + 2 * c;
                                const UNCOMPRESSED_TYPE src_scale_inv = src_comp_ptr[0];
                                const UNCOMPRESSED_TYPE src_zp        = src_comp_ptr[1];
                                const UNCOMPRESSED_TYPE dst_scale_inv = dst_comp_ptr[0];
                                const UNCOMPRESSED_TYPE dst_zp        = dst_comp_ptr[1];

                                OUTPUT_TYPE src_byte = key_cache[key_src_base + src_slot * row_bytes + b];
                                char2 src_pair = unpack_to_char(*(uint4x2_t*)&src_byte);
                                const char src_nibble = (n == 0) ? src_pair.s0 : src_pair.s1;
                                const UNCOMPRESSED_TYPE src_val = ((UNCOMPRESSED_TYPE)src_nibble - src_zp) * src_scale_inv;

                                UNCOMPRESSED_TYPE dequant_vals[PAGED_ATTENTION_BLOCK_SIZE];
                                for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
                                    OUTPUT_TYPE dst_byte = key_cache[key_dst_base + t * row_bytes + b];
                                    char2 dst_pair = unpack_to_char(*(uint4x2_t*)&dst_byte);
                                    const char dst_nibble = (n == 0) ? dst_pair.s0 : dst_pair.s1;
                                    UNCOMPRESSED_TYPE v = ((UNCOMPRESSED_TYPE)dst_nibble - dst_zp) * dst_scale_inv;
                                    if (t == dst_slot) v = src_val;
                                    dequant_vals[t] = v;
                                }
                                // The pair is written here, the bytes after both channels have read theirs.
                                FUNC_CALL(requantize_by_channel_int4_tm)(dequant_vals, q[n], dst_comp_ptr);
                            }
                            for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
                                char2 res = {q[0][t], q[1][t]};
                                key_cache[key_dst_base + t * row_bytes + b] = cvt_int8x2_to_uint4x2(res);
                            }
                        }
                    }
                #elif KEY_BY_CHANNEL_TM
                    // Token-major i8 page: token t of channel k at t * K_HEAD_SIZE + k, one (inv_scale, zp) pair per
                    // channel after the data. On the same page the source value dequantizes with the destination's
                    // pair, which is what the d-major path's raw copy followed by a dequant gives.
                    for (uint k = sglid; k < (uint)K_HEAD_SIZE; k += (uint)SUBGROUP_SIZE) {
                        UNCOMPRESSED_TYPE* src_comp_ptr =
                            (UNCOMPRESSED_TYPE*)(key_cache + key_src_base + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE) + 2 * k;
                        UNCOMPRESSED_TYPE* dst_comp_ptr =
                            (UNCOMPRESSED_TYPE*)(key_cache + key_dst_base + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE) + 2 * k;
                        const UNCOMPRESSED_TYPE src_scale_inv = src_comp_ptr[0];
                        const UNCOMPRESSED_TYPE src_zp = src_comp_ptr[1];
                        const UNCOMPRESSED_TYPE dst_scale_inv = dst_comp_ptr[0];
                        const UNCOMPRESSED_TYPE dst_zp = dst_comp_ptr[1];

                        const UNCOMPRESSED_TYPE src_val =
                            ((UNCOMPRESSED_TYPE)(key_cache[key_src_base + src_slot * K_HEAD_SIZE + k]) - src_zp) * src_scale_inv;

                        UNCOMPRESSED_TYPE dequant_vals[PAGED_ATTENTION_BLOCK_SIZE];
                        for (uint token = 0; token < PAGED_ATTENTION_BLOCK_SIZE; ++token) {
                            UNCOMPRESSED_TYPE val =
                                ((UNCOMPRESSED_TYPE)(key_cache[key_dst_base + token * K_HEAD_SIZE + k]) - dst_zp) * dst_scale_inv;
                            if (token == dst_slot) {
                                val = src_val;
                            }
                            dequant_vals[token] = val;
                        }
                        FUNC_CALL(requantize_and_store_by_channel_tm)(key_cache, key_dst_base + k, dst_comp_ptr, dequant_vals);
                    }
                #elif IS_INT4_COMPRESSED
                    if (src_block == dst_block) {
                        // Fast path: nibble shuffle within the same column. scale/zp unchanged,
                        // no dequant/requant — avoids accumulating rounding error across reorders.
                        for (uint k = sglid; k < phys_k_head_size; k += (uint)SUBGROUP_SIZE) {
                            const uint col_base = key_dst_base + k * PACKED_ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;

                            OUTPUT_TYPE src_byte = key_cache[col_base + src_slot / U4_ELEMS_PER_BYTE];
                            char2 src_pair = unpack_to_char(*(uint4x2_t*)&src_byte);
                            const char src_nibble = (src_slot % U4_ELEMS_PER_BYTE == 0) ? src_pair.s0 : src_pair.s1;

                            const uint dst_byte_off = col_base + dst_slot / U4_ELEMS_PER_BYTE;
                            OUTPUT_TYPE dst_byte = key_cache[dst_byte_off];
                            char2 dst_pair = unpack_to_char(*(uint4x2_t*)&dst_byte);
                            if (dst_slot % U4_ELEMS_PER_BYTE == 0) {
                                dst_pair.s0 = src_nibble;
                            } else {
                                dst_pair.s1 = src_nibble;
                            }
                            key_cache[dst_byte_off] = cvt_int8x2_to_uint4x2(dst_pair);
                        }
                    } else {
                        for (uint k = sglid; k < phys_k_head_size; k += (uint)SUBGROUP_SIZE) {
                            const uint src_col_base = key_src_base + k * PACKED_ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
                            const uint dst_col_base = key_dst_base + k * PACKED_ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;

                            UNCOMPRESSED_TYPE* src_comp_ptr = (UNCOMPRESSED_TYPE*)(key_cache + src_col_base + PACKED_PAGED_ATTENTION_BLOCK_SIZE);
                            UNCOMPRESSED_TYPE* dst_comp_ptr = (UNCOMPRESSED_TYPE*)(key_cache + dst_col_base + PACKED_PAGED_ATTENTION_BLOCK_SIZE);
                            const UNCOMPRESSED_TYPE src_scale_inv = src_comp_ptr[0];
                            const UNCOMPRESSED_TYPE src_zp        = src_comp_ptr[1];
                            const UNCOMPRESSED_TYPE dst_scale_inv = dst_comp_ptr[0];
                            const UNCOMPRESSED_TYPE dst_zp        = dst_comp_ptr[1];

                            OUTPUT_TYPE src_byte = key_cache[src_col_base + src_slot / U4_ELEMS_PER_BYTE];
                            char2 src_pair = unpack_to_char(*(uint4x2_t*)&src_byte);
                            const char src_nibble = (src_slot % U4_ELEMS_PER_BYTE == 0) ? src_pair.s0 : src_pair.s1;
                            const UNCOMPRESSED_TYPE src_val = ((UNCOMPRESSED_TYPE)src_nibble - src_zp) * src_scale_inv;

                            UNCOMPRESSED_TYPE dequant_vals[PAGED_ATTENTION_BLOCK_SIZE];
                            for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
                                OUTPUT_TYPE dst_byte = key_cache[dst_col_base + t / U4_ELEMS_PER_BYTE];
                                char2 dst_pair = unpack_to_char(*(uint4x2_t*)&dst_byte);
                                const char dst_nibble = (t % U4_ELEMS_PER_BYTE == 0) ? dst_pair.s0 : dst_pair.s1;
                                UNCOMPRESSED_TYPE v = ((UNCOMPRESSED_TYPE)dst_nibble - dst_zp) * dst_scale_inv;
                                if (t == dst_slot) v = src_val;
                                dequant_vals[t] = v;
                            }

                            FUNC_CALL(requantize_and_store_by_channel_block_int4)(key_cache, dst_col_base, dequant_vals);
                        }
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
                // per-token quantization: copy quantized values and comp data for token.
                // Token-major makes a token's head dims contiguous (like the value copy above);
                // d-major makes the tokens of one head dim contiguous. The trailing scale/zp arrays
                // are layout-independent, so only this data loop changes.
                for (uint k = sglid; k < phys_k_head_size; k += (uint)SUBGROUP_SIZE) {
                #if KEY_BY_TOKEN_TM
                    const uint src_off = key_src_base + src_slot * K_HEAD_SIZE + k;
                    const uint dst_off = key_dst_base + dst_slot * K_HEAD_SIZE + k;
                #else
                    const uint src_off = key_src_base + k * PAGED_ATTENTION_BLOCK_SIZE + src_slot;
                    const uint dst_off = key_dst_base + k * PAGED_ATTENTION_BLOCK_SIZE + dst_slot;
                #endif
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
            #if IS_INT4_COMPRESSED && KEY_BY_CHANNEL_TM
            // The token-major INT4 K page relays V too: rows of PACKED_V_HEAD_SIZE bytes, then per-token scale and zp
            // arrays as for INT8 (copied below). The split nibble order inside a row does not matter to a row copy.
            for (uint v = sglid; v < phys_v_head_size; v += (uint)SUBGROUP_SIZE) {
                const uint src_off = val_src_base + src_slot * phys_v_head_size + v;
                const uint dst_off = val_dst_base + dst_slot * phys_v_head_size + v;
                value_cache[dst_off] = value_cache[src_off];
            }
            #elif IS_INT4_COMPRESSED
            // u4 per-token layout: each token row = [packed_v_head_size data bytes][scale][zp]
            // Copy the whole row in one pass, scale/zp included.
            for (uint v = sglid; v < phys_adjusted_v_head_size; v += (uint)SUBGROUP_SIZE) {
                const uint src_off = val_src_base + src_slot * phys_adjusted_v_head_size + v;
                const uint dst_off = val_dst_base + dst_slot * phys_adjusted_v_head_size + v;
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

            #if !IS_INT4_COMPRESSED || KEY_BY_CHANNEL_TM
            // INT8 per-token (and the relaid-out INT4 V page): scale/zp arrays live at end of block, indexed by token slot.
            // The upstream INT4 page already copied scale/zp inline with the token row.
            if (sglid == 0) {
                const uint comp_src_base = val_src_base + phys_v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
                const uint comp_dst_base = val_dst_base + phys_v_head_size * PAGED_ATTENTION_BLOCK_SIZE;
                UNCOMPRESSED_TYPE* src_comp_ptr = value_cache + comp_src_base;
                UNCOMPRESSED_TYPE* dst_comp_ptr = value_cache + comp_dst_base;
                dst_comp_ptr[dst_slot] = src_comp_ptr[src_slot];
                dst_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + dst_slot] = src_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + src_slot];
            }
            #endif
        #endif
    }
}