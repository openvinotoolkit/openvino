// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#if IS_KV_COMPRESSED
#define SUBGROUPS_PER_WG 1
#else
#define SUBGROUPS_PER_WG KV_HEADS_NUM
#endif
#define ACCUMULATOR_TYPE float

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
    const uint block_base_offset = rotated_block_indices[block_idx] * KV_HEADS_NUM * ADJUSTED_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                   head_idx * ADJUSTED_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
    const uint token_offset = block_base_offset + sglid;

#if IS_KV_COMPRESSED
    const uint comp_offset = block_base_offset + HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
    UNCOMPRESSED_TYPE* comp_ptr = key_cache + comp_offset;
    UNCOMPRESSED_TYPE comp_scale = comp_ptr[0 + sglid];
    UNCOMPRESSED_TYPE comp_zp = comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid];

    UNCOMPRESSED_TYPE max_value = UNCOMPRESSED_VAL_MIN;
    UNCOMPRESSED_TYPE min_value = UNCOMPRESSED_VAL_MAX;

    // Reuse SLM to store dequantized rotated values
    __local UNCOMPRESSED_TYPE* rotated_data = (__local UNCOMPRESSED_TYPE*)(&rotation_coefficients[0][0]);
#endif

    // Apply cache rotation
    for (uint i = 0; i < HEAD_SIZE / 2; i++) {
        const uint cache_offset = token_offset + i * PAGED_ATTENTION_BLOCK_SIZE;

#if IS_KV_COMPRESSED
        UNCOMPRESSED_TYPE cache_value_first = TO_UNCOMPRESSED_TYPE(key_cache[cache_offset] - comp_zp) * comp_scale;
        UNCOMPRESSED_TYPE cache_value_second = TO_UNCOMPRESSED_TYPE(key_cache[cache_offset + (HEAD_SIZE / 2) * PAGED_ATTENTION_BLOCK_SIZE] - comp_zp) * comp_scale;
#else
        UNCOMPRESSED_TYPE cache_value_first = key_cache[cache_offset];
        UNCOMPRESSED_TYPE cache_value_second = key_cache[cache_offset + (HEAD_SIZE / 2) * PAGED_ATTENTION_BLOCK_SIZE];
#endif

        INPUT2_TYPE rotation_value_cos = rotation_coefficients[i][token_coefficient_idx];
        INPUT2_TYPE rotation_value_sin = rotation_coefficients[i + (HEAD_SIZE / 2)][token_coefficient_idx];

        UNCOMPRESSED_TYPE new_cache_value_first = cache_value_first * rotation_value_cos - cache_value_second * rotation_value_sin;
        UNCOMPRESSED_TYPE new_cache_value_second = cache_value_first * rotation_value_sin + cache_value_second * rotation_value_cos;

#if IS_KV_COMPRESSED
        max_value = fmax(fmax(max_value, new_cache_value_first), new_cache_value_second);
        min_value = fmin(fmin(min_value, new_cache_value_first), new_cache_value_second);

        rotated_data[(i + 0) * PAGED_ATTENTION_BLOCK_SIZE + sglid] = new_cache_value_first;
        rotated_data[(i + (HEAD_SIZE / 2)) * PAGED_ATTENTION_BLOCK_SIZE + sglid] = new_cache_value_second;
#else
        key_cache[cache_offset] = new_cache_value_first;
        key_cache[cache_offset + (HEAD_SIZE / 2) * PAGED_ATTENTION_BLOCK_SIZE] = new_cache_value_second;
#endif
    }

#if IS_KV_COMPRESSED
    // Re-quantize cache data
    ACCUMULATOR_TYPE grp_max = 0.001;
    ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
    UNCOMPRESSED_TYPE scale = (UNCOMPRESSED_TYPE)(scale_tmp);
    UNCOMPRESSED_TYPE zp = (UNCOMPRESSED_TYPE)(zp_tmp);

    // Note: absence of this explicit unrolling directive leads to automatic
    // unrolling and causes registers spill. Set unrolling to a reasonable value manually
    __attribute__((opencl_unroll_hint(8)))
    for (uint i = 0; i < HEAD_SIZE; i++) {
        OUTPUT_TYPE quantized_res = convert_char_rte(rotated_data[i * PAGED_ATTENTION_BLOCK_SIZE + sglid] * scale + zp);

        const uint cache_offset = token_offset + i * PAGED_ATTENTION_BLOCK_SIZE;
        key_cache[cache_offset] = quantized_res;
    }

    comp_ptr[0 + sglid] = 1.0 / scale;
    comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid] = zp;
#endif
}

#undef ACCUMULATOR_TYPE
#undef SUBGROUPS_PER_WG
