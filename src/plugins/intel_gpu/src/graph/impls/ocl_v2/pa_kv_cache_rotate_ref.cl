// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/int4_utils.cl"

#if IS_KV_COMPRESSED
#define SUBGROUPS_PER_WG 1
#else
#define SUBGROUPS_PER_WG KV_HEADS_NUM
#endif
#define ACCUMULATOR_TYPE float
#define UINT4_RANGE 15

// In-page K addressing: lane (sglid) walks tokens, the loop index walks head dims. The legacy
// d-major page ([.., HEAD_SIZE, block_size]) makes tokens contiguous; token-major
// ([.., block_size, HEAD_SIZE]) makes head dims contiguous. Only these two strides differ, so both
// layouts share one code path. Compressed BY_CHANNEL uses the separate staging switch below because
// its scale/zp placement and, for u4, its packing axis also change with the layout.
#if IS_KEY_TOKEN_MAJOR
    #define KEY_TOKEN_STRIDE  HEAD_SIZE
    #define KEY_HIDDEN_STRIDE 1
#else
    #define KEY_TOKEN_STRIDE  1
    #define KEY_HIDDEN_STRIDE PAGED_ATTENTION_BLOCK_SIZE
#endif

#if IS_KV_COMPRESSED && IS_KEY_BY_CHANNEL
    #if IS_INT4_COMPRESSED
        #if IS_KEY_BY_CHANNEL_TOKEN_MAJOR
            // One byte holds two adjacent channels of one token; comp is a trailing per-channel array.
            #define BC_DATA_OFFSET(page, token, channel) \
                ((page) + (token) * (HEAD_SIZE / U4_ELEMS_PER_BYTE) + (channel) / U4_ELEMS_PER_BYTE)
            #define BC_NIBBLE_INDEX(token, channel) ((channel) % U4_ELEMS_PER_BYTE)
            #define BC_COMP_OFFSET(page, channel) \
                ((page) + PAGED_ATTENTION_BLOCK_SIZE * (HEAD_SIZE / U4_ELEMS_PER_BYTE) + \
                 2 * (channel) * (uint)sizeof(UNCOMPRESSED_TYPE))
        #else
            // One byte holds two adjacent tokens of one channel; comp remains inline in the column.
            #define BC_DATA_OFFSET(page, token, channel) \
                ((page) + (channel) * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + (token) / U4_ELEMS_PER_BYTE)
            #define BC_NIBBLE_INDEX(token, channel) ((token) % U4_ELEMS_PER_BYTE)
            #define BC_COMP_OFFSET(page, channel) \
                ((page) + (channel) * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PACKED_K_BLOCK_SIZE)
        #endif
    #else
        #if IS_KEY_BY_CHANNEL_TOKEN_MAJOR
            #define BC_DATA_OFFSET(page, token, channel) ((page) + (token) * HEAD_SIZE + (channel))
            #define BC_COMP_OFFSET(page, channel) \
                ((page) + PAGED_ATTENTION_BLOCK_SIZE * HEAD_SIZE + \
                 2 * (channel) * (uint)sizeof(UNCOMPRESSED_TYPE))
        #else
            #define BC_DATA_OFFSET(page, token, channel) \
                ((page) + (channel) * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + (token))
            #define BC_COMP_OFFSET(page, channel) \
                ((page) + (channel) * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + PAGED_ATTENTION_BLOCK_SIZE)
        #endif
    #endif
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, SUBGROUPS_PER_WG, 1)))
KERNEL(pa_kv_cache_rotate)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* rotated_block_indices,
    __global const INPUT1_TYPE* rotation_deltas,
    __global const INPUT2_TYPE* rotation_trig_lut,
    __global OUTPUT_TYPE* key_cache
) {
    // Input shapes:
    // rotated_block_indices: [num_blocks_to_rotate]
    // rotation_deltas: [num_blocks_to_rotate, PAGED_ATTENTION_BLOCK_SIZE] || [num_blocks_to_rotate, 1]
    // rotation_trig_lut: [max_num_batched_tokens / PAGED_ATTENTION_BLOCK_SIZE, HEAD_SIZE] || [max_num_batched_tokens, HEAD_SIZE]
    // key_cache: [num_blocks, HEADS_NUM, HEAD_SIZE, PAGED_ATTENTION_BLOCK_SIZE]

    // Output shapes:
    // key_cache (updated): [num_blocks, HEADS_NUM, HEAD_SIZE, PAGED_ATTENTION_BLOCK_SIZE]

    const uint head_idx = get_global_id(1);
    const uint block_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();

    __local INPUT2_TYPE rotation_coefficients[HEAD_SIZE][PAGED_ATTENTION_BLOCK_SIZE];

    const bool per_token_rotation = INPUT1_FEATURE_NUM == PAGED_ATTENTION_BLOCK_SIZE;

    if (per_token_rotation) {
        // Need to load HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE coefficients in total, each subgroup loads SUBGROUP_SIZE values
        for (uint i = sgid; i < HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE / SUBGROUP_SIZE; i += SUBGROUPS_PER_WG) {
            const uint token_idx = (i / (HEAD_SIZE / SUBGROUP_SIZE));
            const uint rotation_trig_lut_start_offset = rotation_deltas[block_idx * INPUT1_FEATURE_NUM + token_idx] * HEAD_SIZE;
            const uint inner_offset = (i % (HEAD_SIZE / SUBGROUP_SIZE)) * SUBGROUP_SIZE;
            const uint rotation_trig_lut_offset = rotation_trig_lut_start_offset + inner_offset;

            INPUT2_TYPE coefficient = rotation_trig_lut[rotation_trig_lut_offset + sglid];

            rotation_coefficients[inner_offset + sglid][token_idx] = coefficient;
        }
    } else {
        // Need to load HEAD_SIZE coefficients in total, each subgroup loads SUBGROUP_SIZE values
        for (uint i = sgid; i < HEAD_SIZE / SUBGROUP_SIZE; i += SUBGROUPS_PER_WG) {
            const uint token_idx = 0;
            const uint rotation_trig_lut_start_offset = rotation_deltas[block_idx * INPUT1_FEATURE_NUM + token_idx] * HEAD_SIZE;
            const uint inner_offset = i * SUBGROUP_SIZE;
            const uint rotation_trig_lut_offset = rotation_trig_lut_start_offset + inner_offset;

            INPUT2_TYPE coefficient = rotation_trig_lut[rotation_trig_lut_offset + sglid];

            rotation_coefficients[inner_offset + sglid][token_idx] = coefficient;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint token_coefficient_idx = per_token_rotation ? sglid : 0;
    const uint block_base_offset = rotated_block_indices[block_idx] * KV_HEADS_NUM * ADJUSTED_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                                   head_idx * ADJUSTED_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
    const uint token_offset = block_base_offset + sglid * KEY_TOKEN_STRIDE;

#if IS_KV_COMPRESSED
    #if IS_KEY_BY_CHANNEL
        UNCOMPRESSED_TYPE min_value[HEAD_SIZE];
        UNCOMPRESSED_TYPE max_value[HEAD_SIZE];
        for (uint i = 0; i < HEAD_SIZE; i++) {
            min_value[i] = UNCOMPRESSED_VAL_MAX;
            max_value[i] = UNCOMPRESSED_VAL_MIN;
        }
    #else
        UNCOMPRESSED_TYPE min_value = UNCOMPRESSED_VAL_MAX;
        UNCOMPRESSED_TYPE max_value = UNCOMPRESSED_VAL_MIN;
        const uint comp_offset = block_base_offset + HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        UNCOMPRESSED_TYPE* comp_ptr = key_cache + comp_offset;
        UNCOMPRESSED_TYPE comp_scale = comp_ptr[0 + sglid];
        UNCOMPRESSED_TYPE comp_zp = comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid];
    #endif

    // Reuse SLM to store dequantized rotated values
    __local UNCOMPRESSED_TYPE* rotated_data = (__local UNCOMPRESSED_TYPE*)(&rotation_coefficients[0][0]);
#endif

// Apply cache rotation
for (uint i = 0; i < HEAD_SIZE / 2; i++) {
    #if IS_KV_COMPRESSED && IS_KEY_BY_CHANNEL
        const uint hidden_first = i;
        const uint hidden_second = i + HEAD_SIZE / 2;
        const uint cache_offset_first = BC_DATA_OFFSET(block_base_offset, sglid, hidden_first);
        const uint cache_offset_second = BC_DATA_OFFSET(block_base_offset, sglid, hidden_second);

        __global UNCOMPRESSED_TYPE* comp_ptr_first =
            (__global UNCOMPRESSED_TYPE*)(key_cache + BC_COMP_OFFSET(block_base_offset, hidden_first));
        UNCOMPRESSED_TYPE comp_scale_first = comp_ptr_first[0];
        UNCOMPRESSED_TYPE comp_zp_first = comp_ptr_first[1];

        __global UNCOMPRESSED_TYPE* comp_ptr_second =
            (__global UNCOMPRESSED_TYPE*)(key_cache + BC_COMP_OFFSET(block_base_offset, hidden_second));
        UNCOMPRESSED_TYPE comp_scale_second = comp_ptr_second[0];
        UNCOMPRESSED_TYPE comp_zp_second = comp_ptr_second[1];

        #if IS_INT4_COMPRESSED
            const uchar packed_first = (uchar)key_cache[cache_offset_first];
            const uchar packed_second = (uchar)key_cache[cache_offset_second];
            const uint nibble_first = BC_NIBBLE_INDEX(sglid, hidden_first);
            const uint nibble_second = BC_NIBBLE_INDEX(sglid, hidden_second);
            const uchar quantized_first = (packed_first >> (4 * nibble_first)) & 0x0F;
            const uchar quantized_second = (packed_second >> (4 * nibble_second)) & 0x0F;
            UNCOMPRESSED_TYPE cache_value_first =
                (TO_UNCOMPRESSED_TYPE(quantized_first) - comp_zp_first) * comp_scale_first;
            UNCOMPRESSED_TYPE cache_value_second =
                (TO_UNCOMPRESSED_TYPE(quantized_second) - comp_zp_second) * comp_scale_second;
        #else
            UNCOMPRESSED_TYPE cache_value_first =
                (TO_UNCOMPRESSED_TYPE(key_cache[cache_offset_first]) - comp_zp_first) * comp_scale_first;
            UNCOMPRESSED_TYPE cache_value_second =
                (TO_UNCOMPRESSED_TYPE(key_cache[cache_offset_second]) - comp_zp_second) * comp_scale_second;
        #endif
    #elif IS_KV_COMPRESSED
        const uint cache_offset = token_offset + i * KEY_HIDDEN_STRIDE;
        UNCOMPRESSED_TYPE cache_value_first =
            (TO_UNCOMPRESSED_TYPE(key_cache[cache_offset]) - comp_zp) * comp_scale;
        UNCOMPRESSED_TYPE cache_value_second =
            (TO_UNCOMPRESSED_TYPE(key_cache[cache_offset + (HEAD_SIZE / 2) * KEY_HIDDEN_STRIDE]) - comp_zp) * comp_scale;
    #else
        const uint cache_offset = token_offset + i * KEY_HIDDEN_STRIDE;
        UNCOMPRESSED_TYPE cache_value_first = key_cache[cache_offset];
        UNCOMPRESSED_TYPE cache_value_second = key_cache[cache_offset + (HEAD_SIZE / 2) * KEY_HIDDEN_STRIDE];
    #endif

    INPUT2_TYPE rotation_value_cos = rotation_coefficients[i][token_coefficient_idx];
    INPUT2_TYPE rotation_value_sin = rotation_coefficients[i + (HEAD_SIZE / 2)][token_coefficient_idx];

    UNCOMPRESSED_TYPE new_cache_value_first = cache_value_first * rotation_value_cos - cache_value_second * rotation_value_sin;
    UNCOMPRESSED_TYPE new_cache_value_second = cache_value_first * rotation_value_sin + cache_value_second * rotation_value_cos;
    
    #if IS_KV_COMPRESSED
        #if !IS_KEY_BY_CHANNEL
            max_value = fmax(fmax(max_value, new_cache_value_first), new_cache_value_second);
            min_value = fmin(fmin(min_value, new_cache_value_first), new_cache_value_second);
        #endif

        rotated_data[(i + 0) * PAGED_ATTENTION_BLOCK_SIZE + sglid] = new_cache_value_first;
        rotated_data[(i + (HEAD_SIZE / 2)) * PAGED_ATTENTION_BLOCK_SIZE + sglid] = new_cache_value_second;
    #else
        key_cache[cache_offset] = new_cache_value_first;
        key_cache[cache_offset + (HEAD_SIZE / 2) * KEY_HIDDEN_STRIDE] = new_cache_value_second;
    #endif
}

#if IS_KV_COMPRESSED
    UNCOMPRESSED_TYPE scale;
    UNCOMPRESSED_TYPE zp;
    ACCUMULATOR_TYPE diff_value;
    ACCUMULATOR_TYPE scale_tmp;
    ACCUMULATOR_TYPE zp_tmp;
    // Note: absence of this explicit unrolling directive leads to automatic
    // unrolling and causes registers spill. Set unrolling to a reasonable value manually
    __attribute__((opencl_unroll_hint(8)))
    for (uint i = 0; i < HEAD_SIZE; i++) {
        // Re-quantize cache data
        ACCUMULATOR_TYPE grp_max = 0.001;
        UNCOMPRESSED_TYPE rotated_res = rotated_data[i * PAGED_ATTENTION_BLOCK_SIZE + sglid];
        #if IS_KEY_BY_CHANNEL
            max_value[i] = fmax(max_value[i], rotated_res);
            min_value[i] = fmin(min_value[i], rotated_res);

            min_value[i] = sub_group_reduce_min(min_value[i]);
            max_value[i] = sub_group_reduce_max(max_value[i]);

            #if IS_INT4_COMPRESSED
                diff_value = max_value[i] == min_value[i] ? 0.004f : (max_value[i] - min_value[i]);
                const ACCUMULATOR_TYPE min_range = fabs((ACCUMULATOR_TYPE)max_value[i] * 0.1f);
                if (diff_value <= min_range) {
                    diff_value += fmax(1.0f, min_range);
                }
                scale_tmp = (ACCUMULATOR_TYPE)(UINT4_RANGE / diff_value);
                zp_tmp = (ACCUMULATOR_TYPE)(-min_value[i] * scale_tmp);
            #else
                diff_value = max_value[i] == min_value[i] ? (grp_max) : (max_value[i] - min_value[i]);
                scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
                zp_tmp = (ACCUMULATOR_TYPE)(-min_value[i] * scale_tmp) + CHAR_MIN;
            #endif
        #else
            diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
            scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
            zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
        #endif
        scale = (UNCOMPRESSED_TYPE)(scale_tmp);
        zp = (UNCOMPRESSED_TYPE)(zp_tmp);

        #if IS_INT4_COMPRESSED && IS_KEY_BY_CHANNEL
            const char quantized_res =
                (char)clamp(convert_int_rte((float)(rotated_res * scale_tmp + zp_tmp)), 0, UINT4_RANGE);
            const uint cache_offset = BC_DATA_OFFSET(block_base_offset, sglid, i);
            #if IS_KEY_BY_CHANNEL_TOKEN_MAJOR
                const uchar packed = (uchar)key_cache[cache_offset];
                char2 pair = {(char)(packed & 0x0F), (char)(packed >> 4)};
                if ((i % U4_ELEMS_PER_BYTE) == 0) {
                    pair.s0 = quantized_res;
                } else {
                    pair.s1 = quantized_res;
                }
                key_cache[cache_offset] = cvt_int8x2_to_uint4x2(pair);
            #else
                const char paired_res = intel_sub_group_shuffle(quantized_res, sglid ^ 1);
                if ((sglid % U4_ELEMS_PER_BYTE) == 0) {
                    char2 pair = {quantized_res, paired_res};
                    key_cache[cache_offset] = cvt_int8x2_to_uint4x2(pair);
                }
            #endif

            if (sglid == 0) {
                __global UNCOMPRESSED_TYPE* comp_ptr =
                    (__global UNCOMPRESSED_TYPE*)(key_cache + BC_COMP_OFFSET(block_base_offset, i));
                comp_ptr[0] = (UNCOMPRESSED_TYPE)(1.0f / scale_tmp);
                comp_ptr[1] = (UNCOMPRESSED_TYPE)zp_tmp;
            }
        #else
            OUTPUT_TYPE quantized_res = convert_char_rte(rotated_res * scale + zp);
            #if IS_KEY_BY_CHANNEL
                const uint cache_offset = BC_DATA_OFFSET(block_base_offset, sglid, i);
                if (sglid == 0) {
                    __global UNCOMPRESSED_TYPE* comp_ptr =
                        (__global UNCOMPRESSED_TYPE*)(key_cache + BC_COMP_OFFSET(block_base_offset, i));
                    comp_ptr[0] = 1.0 / scale;
                    comp_ptr[1] = zp;
                }
            #else
                const uint cache_offset = token_offset + i * KEY_HIDDEN_STRIDE;
            #endif
            key_cache[cache_offset] = quantized_res;
        #endif
    }
    #if !IS_KEY_BY_CHANNEL
        comp_ptr[0 + sglid] = 1.0 / scale;
        comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid] = zp;
    #endif
#endif
}

#if IS_KV_COMPRESSED && IS_KEY_BY_CHANNEL
#undef BC_DATA_OFFSET
#undef BC_COMP_OFFSET
    #if IS_INT4_COMPRESSED
#undef BC_NIBBLE_INDEX
    #endif
#endif
#undef UINT4_RANGE
#undef ACCUMULATOR_TYPE
#undef SUBGROUPS_PER_WG
