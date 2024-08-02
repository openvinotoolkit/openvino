// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#ifdef BATCH_AXIS
    #define VALUES_NUM INPUT0_BATCH_NUM
    #define AXIS 0
#endif
#ifdef FEATURE_AXIS
    #define VALUES_NUM INPUT0_FEATURE_NUM
    #define AXIS 1
#endif
#ifdef Z_AXIS
    #define VALUES_NUM INPUT0_SIZE_Z
    #define AXIS 2
#endif
#ifdef Y_AXIS
    #define VALUES_NUM INPUT0_SIZE_Y
    #define AXIS 3
#endif
#ifdef X_AXIS
    #define VALUES_NUM INPUT0_SIZE_X
    #define AXIS 4
#endif

#ifdef MAX_OUT
    #define COMPARE_SIGN <
    #define COMPARE_PARTIAL_SIGN >=
    #define COMPARE_MERGE_SIGN >
    #define COMPARE_PARALLEL_SIGN_1 <=
    #define COMPARE_PARALLEL_SIGN_2 <
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define COMPARE_PARTIAL_SIGN <=
    #define COMPARE_MERGE_SIGN <
    #define COMPARE_PARALLEL_SIGN_1 >=
    #define COMPARE_PARALLEL_SIGN_2 >
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

#define MINIMUM_NUMBER_FOR_PARTIAL_SORTING 100

inline void FUNC(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_ARG
                                        const uint output_idx,
                                        uint* indices)
{
#ifdef BATCH_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_SIZE_X * INPUT0_FEATURE_NUM); // Y
    const uint out_second_dim = output_idx / INPUT0_FEATURE_NUM % INPUT0_SIZE_X; // X
    const uint out_fourth_dim = output_idx % INPUT0_FEATURE_NUM; // F
    indices[1] = out_fourth_dim; indices[3] = out_first_dim; indices[4] = out_second_dim; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X); // F
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z; // Z
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y; // Y
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X; // X
    indices[1] = out_first_dim; indices[2] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
    #endif
#endif
#ifdef FEATURE_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_SIZE_X * INPUT0_BATCH_NUM); // Y
    const uint out_second_dim = output_idx / INPUT0_BATCH_NUM % INPUT0_SIZE_X; // X
    const uint out_fourth_dim = output_idx % INPUT0_BATCH_NUM; // B
    indices[0] = out_fourth_dim; indices[3] = out_first_dim; indices[4] = out_second_dim; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X); // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z; // Z
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;  // Y
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;  // X
    indices[0] = out_first_dim; indices[2] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
    #endif
#endif
#ifdef Z_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X);  // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM; // F
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y; // Y
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X; // X
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef Y_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM); // X
    const uint out_second_dim = output_idx / INPUT0_BATCH_NUM % INPUT0_FEATURE_NUM; // F
    const uint out_fourth_dim = output_idx % INPUT0_BATCH_NUM; // B
    indices[0] = out_fourth_dim; indices[1] = out_second_dim; indices[4] = out_first_dim; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_X); // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM; // F
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Z; // Z
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X; // X
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[2] = out_third_dim; indices[4] = out_fourth_dim;
    #endif
#endif
#ifdef X_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM); // Y
    const uint out_second_dim = output_idx / INPUT0_BATCH_NUM % INPUT0_FEATURE_NUM; // F
    const uint out_fourth_dim = output_idx % INPUT0_BATCH_NUM; // B
    indices[0] = out_fourth_dim; indices[1] = out_second_dim; indices[3] = out_first_dim; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_Y); // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y) % INPUT0_FEATURE_NUM; // F
    const uint out_third_dim = output_idx / INPUT0_SIZE_Y % INPUT0_SIZE_Z; // Z
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_Y; // Y
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[2] = out_third_dim; indices[3] = out_fourth_dim;
    #endif
#endif
}

KERNEL(arg_max_min_modified)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input
    ,__global OUTPUT_TYPE* output
#ifdef OUTPUT1_TYPE
    ,__global OUTPUT1_TYPE* second_output
#endif
#ifdef IS_DYNAMIC
    ,__global INPUT0_TYPE* tmp_buffer0
    ,__global INPUT0_TYPE* tmp_buffer1
    ,__global INPUT0_TYPE* tmp_buffer2
#endif
)
{
#include "include/arg_max_min_common.cl"
    const uint output_idx = (uint)get_global_id(0);
#if SORT_BY_VALUE
    const uint sort_idx = (uint)get_global_id(1);
#elif TOP_K == 1
    iav_type result[TOP_K];
#else
#ifdef IS_DYNAMIC
    const uint iav_type_size = INPUT0_TYPE_SIZE + 4;
    const uint buffer_size = iav_type_size * VALUES_NUM;
    const uint buffer_offset = buffer_size * OPERATION_NUM;
    __global iav_type *result = OFFSET_GLOBAL_PTR(iav_type, tmp_buffer0, output_idx * buffer_size);
    __global iav_type *temp_buf = OFFSET_GLOBAL_PTR(iav_type, tmp_buffer0, buffer_offset + output_idx * buffer_size);
#else
    iav_type result[VALUES_NUM], temp_buf[VALUES_NUM];
#endif
    const uint group_size = TOP_K >= 8 ? TOP_K : 8;
    const uint group_num = ((VALUES_NUM - 1) / group_size) + 1;
    const uint last_group_size = (VALUES_NUM % group_size > 0) ? (VALUES_NUM % group_size) : group_size;
    const uint last_group_offset = (group_num - 1) * group_size;
#endif // SORT_BY_VALUE
    uint indices[] = { 0, 0, 0, 0, 0 };

    if (OPERATION_NUM > 1) {
        if (output_idx >= OPERATION_NUM)
            return;
        FUNC_CALL(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_TENSOR output_idx, indices);
    }

// Using parallel sorting for sorting by values
#if SORT_BY_VALUE
    uint sort_position = 0;
    indices[AXIS] = sort_idx;

    iav_type result;
    result.value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
    result.index = sort_idx;

    for (uint i = 0; i < sort_idx / 8; i++) {
        uint index_offset = i * 8;
        indices[AXIS] = index_offset;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 1;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 2;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 3;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 4;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 5;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 6;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        indices[AXIS] = index_offset + 7;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
        if (sort_position >= TOP_K)
            return;
    }

    for (uint i = (sort_idx / 8) * 8; i < sort_idx; i++) {
        indices[AXIS] = i;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_1 test_value)
            sort_position++;
    }

    if (sort_position >= TOP_K)
        return;

    for (uint i = sort_idx + 1; i < VALUES_NUM; i++) {
        indices[AXIS] = i;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (result.value COMPARE_PARALLEL_SIGN_2 test_value)
            sort_position++;
        if (sort_position >= TOP_K)
            return;
    }

// Using simple sorting for sorting by indices and when TOP_K == 1
#elif TOP_K == 1
    INPUT0_TYPE val = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
    result[0].index = 0;
    result[0].value = val;
    bool already_exist = false;
    for (uint top_k = 0; top_k < TOP_K; ++top_k) {
        for (uint i = 0; i < VALUES_NUM; ++i) {
            for (uint j = 0; j < top_k; ++j) {
                if (result[j].index == i) {
                    already_exist = true;
                    break;
                }
            }

            if (already_exist) {
                already_exist = false;
                continue;
            }

            indices[AXIS] = i;
            INPUT0_TYPE in_data = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            if (val COMPARE_SIGN in_data) {
                result[top_k].index = i;
                result[top_k].value = in_data;
                val = in_data;
            }
        }
        val = INPUT0_FILL_VAL;
    }
#else // SORT_BY_VALUE
    // Using merge sorting for sorting by indices and when (TOP_K >= (VALUES_NUM / 2)) or (VALUES_NUM < MINIMUM_NUMBER_FOR_PARTIAL_SORTING)
    bool use_merge_sorting = (TOP_K >= (VALUES_NUM / 2)) || (VALUES_NUM < MINIMUM_NUMBER_FOR_PARTIAL_SORTING);
    if (use_merge_sorting) {
        for (uint i = 0; i < VALUES_NUM / 8; i++) {
            uint index_offset = i * 8;
            indices[AXIS] = result[index_offset].index = index_offset;
            result[index_offset].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 1].index = index_offset + 1;
            result[index_offset + 1].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 2].index = index_offset + 2;
            result[index_offset + 2].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 3].index = index_offset + 3;
            result[index_offset + 3].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 4].index = index_offset  + 4;
            result[index_offset + 4].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 5].index = index_offset + 5;
            result[index_offset + 5].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 6].index = index_offset + 6;
            result[index_offset + 6].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = result[index_offset + 7].index = index_offset + 7;
            result[index_offset + 7].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        }

        for (uint i = (VALUES_NUM / 8) * 8; i < VALUES_NUM; i++) {
            indices[AXIS] = result[i].index = i;
            result[i].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        }

        for (uint k = 1; k < VALUES_NUM; k *= 2) {
            for (uint left = 0; left + k < VALUES_NUM; left += k * 2) {
                uint i, j, m;
                uint right = left + k;
                uint right_end = right + k;
                if (right_end > VALUES_NUM) right_end = VALUES_NUM;
                m = i = left; j = right;
                while ((i < right) && (j < right_end)) {
                    if (result[i].value COMPARE_PARTIAL_SIGN result[j].value) {
                        temp_buf[m++] = result[i++];
                    } else {
                        temp_buf[m++] = result[j++];
                    }
                }
                while (i < right)
                    temp_buf[m++] = result[i++];
                while (j < right_end)
                    temp_buf[m++] = result[j++];
                for (m = left; m < right_end; m++)
                    result[m] = temp_buf[m];
            }
        }
    } else {
        // In other cases for sorting by indices using mixed partial/merge sorting
        for (uint i = 0; i < VALUES_NUM / 8; i++) {
            uint index_offset = i * 8;
            indices[AXIS] = temp_buf[index_offset].index = index_offset;
            temp_buf[index_offset].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 1].index = index_offset + 1;
            temp_buf[index_offset + 1].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 2].index = index_offset + 2;
            temp_buf[index_offset + 2].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 3].index = index_offset + 3;
            temp_buf[index_offset + 3].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 4].index = index_offset  + 4;
            temp_buf[index_offset + 4].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 5].index = index_offset + 5;
            temp_buf[index_offset + 5].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 6].index = index_offset + 6;
            temp_buf[index_offset + 6].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
            indices[AXIS] = temp_buf[index_offset + 7].index = index_offset + 7;
            temp_buf[index_offset + 7].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        }

        for (uint i = (VALUES_NUM / 8) * 8; i < VALUES_NUM; i++) {
            indices[AXIS] = temp_buf[i].index = i;
            temp_buf[i].value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        }

        for (uint group = 0; group < group_num - 1; group++) {
            uint group_offset = group * group_size;
            for (uint k = 1; k < group_size; k *= 2) {
                for (uint left = 0; left + k < group_size; left += k * 2) {
                    uint i, j, m;
                    uint right = left + k;
                    uint right_end = right + k;
                    if (right_end > group_size) right_end = group_size;
                    m = i = left; j = right;
                    while ((i < right) && (j < right_end)) {
                        if (temp_buf[group_offset + i].value COMPARE_PARTIAL_SIGN temp_buf[group_offset + j].value) {
                            result[group_offset + (m++)] = temp_buf[group_offset + (i++)];
                        } else {
                            result[group_offset + (m++)] = temp_buf[group_offset + (j++)];
                        }
                    }
                    while (i < right)
                        result[group_offset + (m++)] = temp_buf[group_offset + (i++)];
                    while (j < right_end)
                        result[group_offset + (m++)] = temp_buf[group_offset + (j++)];
                    for (m = left; m < right_end; m++)
                        temp_buf[group_offset + m] = result[group_offset + m];
                }
            }
        }

        for (uint k = 1; k < last_group_size; k *= 2) {
            for (uint left = 0; left + k < last_group_size; left += k * 2) {
                uint i, j, m;
                uint right = left + k;
                uint right_end = right + k;
                if (right_end > last_group_size) right_end = last_group_size;
                m = i = left; j = right;
                while ((i < right) && (j < right_end)) {
                    if (temp_buf[last_group_offset + i].value COMPARE_PARTIAL_SIGN temp_buf[last_group_offset + j].value) {
                        result[last_group_offset + (m++)] = temp_buf[last_group_offset + (i++)];
                    } else {
                        result[last_group_offset + (m++)] = temp_buf[last_group_offset + (j++)];
                    }
                }
                while (i < right)
                    result[last_group_offset + (m++)] = temp_buf[last_group_offset + (i++)];
                while (j < right_end)
                    result[last_group_offset + (m++)] = temp_buf[last_group_offset + (j++)];
                for (m = left; m < right_end; m++)
                    temp_buf[last_group_offset + m] = result[last_group_offset + m];
            }
        }

    #ifdef IS_DYNAMIC
        const uint counter_size = group_num * 4;
        const uint counter_offset = counter_size * OPERATION_NUM;
        __global uint* merge_counter = OFFSET_GLOBAL_PTR(uint, tmp_buffer1, output_idx * counter_size);
        __global uint* max_merge_counter = OFFSET_GLOBAL_PTR(uint, tmp_buffer1, counter_offset + output_idx * counter_size);
        __global bool* subgroup_done = OFFSET_GLOBAL_PTR(bool, tmp_buffer2, output_idx);
    #else
        uint merge_counter[group_num];
        uint max_merge_counter[group_num];
        bool subgroup_done[group_num];
    #endif
        iav_type merge_buf;

        unroll_for (uint i = 0; i < group_num - 1; i++) {
            merge_counter[i] = 0;
            max_merge_counter[i] = group_size;
            subgroup_done[i] = false;
        }

        merge_counter[group_num - 1] = 0;
        max_merge_counter[group_num - 1] = last_group_size;
        subgroup_done[group_num - 1] = false;

        for (uint i = 0; i < TOP_K; i++) {
            bool merge_buf_done = false;
            uint merge_buf_index = 0;
            for (uint j = 0; j < group_num; j++) {
                if (subgroup_done[j])
                    continue;

                uint test_index = j * group_size + merge_counter[j];

                if (!merge_buf_done) {
                    merge_buf = temp_buf[test_index];
                    merge_buf_done = true;
                    merge_buf_index = j;
                    continue;
                }

                if (temp_buf[test_index].value COMPARE_MERGE_SIGN merge_buf.value) {
                    merge_buf = temp_buf[test_index];
                    merge_buf_index = j;
                }
            }

            merge_counter[merge_buf_index]++;
            if (merge_counter[merge_buf_index] == max_merge_counter[merge_buf_index])
                subgroup_done[merge_buf_index] = true;

            result[i] = merge_buf;
        }
    }
#endif // SORT_BY_VALUE

#if SORT_BY_VALUE
    indices[AXIS] = sort_position;
#ifdef TOP_K_ORDER
    output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(result.value);
#else
    output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(result.index);
#endif
#ifdef OUTPUT1_TYPE
    #ifdef TOP_K_ORDER
    second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT1_TYPE(result.index);
    #else
    second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT1_TYPE(result.value);
    #endif
#endif

#else // SORT_BY_VALUE
    for (uint top_k = 0; top_k < TOP_K; ++top_k) {
        uint out_position = 0;

        for (uint i = 0; i < TOP_K; ++i) {
            if (i == top_k)
                continue;
            if (result[i].index < result[top_k].index)
                out_position++;
        }

        indices[AXIS] = out_position;
#ifdef TOP_K_ORDER
        output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(result[top_k].value);
#else
        output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(result[top_k].index);
#endif
#ifdef OUTPUT1_TYPE
    #ifdef TOP_K_ORDER
        second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT1_TYPE(result[top_k].index);
    #else
        second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT1_TYPE(result[top_k].value);
    #endif
#endif
    }
#endif
}

#undef COMPARE_SIGN
#undef COMPARE_PARTIAL_SIGN
#undef COMPARE_MERGE_SIGN
#undef COMPARE_PARALLEL_SIGN_1
#undef COMPARE_PARALLEL_SIGN_2
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef MINIMUM_NUMBER_FOR_PARTIAL_SORTING
