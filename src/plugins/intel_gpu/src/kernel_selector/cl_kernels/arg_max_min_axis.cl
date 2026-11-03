// Copyright (C) 2018-2026 Intel Corporation
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

// Value order shared with arg_max_min_topk_radix: NaN ranks above +inf as in PyTorch, and -0.0
// equals +0.0. Plain float comparisons are false for NaN, so with them a NaN ranks first and races
// with the true maximum for the same output position.
#if INPUT0_IS_FP
    #define ORDER_LT(a, b) (isnan(b) ? !isnan(a) : (a) < (b))
    #define ORDER_LE(a, b) (isnan(b) || (a) <= (b))
#else
    #define ORDER_LT(a, b) ((a) < (b))
    #define ORDER_LE(a, b) ((a) <= (b))
#endif
#define ORDER_GT(a, b) ORDER_LT(b, a)
#define ORDER_GE(a, b) ORDER_LE(b, a)

// The parallel sort ranks the work-item's own value a, which is not NaN there, by counting the
// values b before it that rank at or above a (SIGN_1) and the later ones that rank above it
// (SIGN_2). Ordered comparisons are false for a NaN b, which ranks above every number, so MIN
// counts those values directly, while MAX counts the complement and PARALLEL_RANK subtracts it
// from the number of compared values. A NaN check per comparison would be slower.
#ifdef MAX_OUT
    #define COMPARE_SIGN ORDER_LT
    #define COMPARE_PARTIAL_SIGN ORDER_GE
    #define COMPARE_MERGE_SIGN ORDER_GT
    #define COMPARE_PARALLEL_SIGN_1(a, b) ((b) < (a))
    #define COMPARE_PARALLEL_SIGN_2(a, b) ((b) <= (a))
    #define PARALLEL_RANK(counted, compared) ((compared) - (counted))
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN ORDER_GT
    #define COMPARE_PARTIAL_SIGN ORDER_LE
    #define COMPARE_MERGE_SIGN ORDER_LT
    #define COMPARE_PARALLEL_SIGN_1(a, b) ((a) >= (b))
    #define COMPARE_PARALLEL_SIGN_2(a, b) ((a) > (b))
    #define PARALLEL_RANK(counted, compared) (counted)
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
#ifdef USE_INTERNAL_BUFFERS
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
#ifdef USE_INTERNAL_BUFFERS
    const uint iav_type_size = sizeof(iav_type);
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

    // Values before sort_idx rank ahead when they rank at or above this one, later values when
    // they rank above it.
    uint head_end = sort_idx;
    uint tail_begin = sort_idx + 1;
    uint counted = 0;
#if INPUT0_IS_FP
    // The comparisons below need a number. A NaN ranks above every number and after the NaNs
    // before it, so its position is counted separately.
    if (isnan(result.value)) {
#ifdef MAX_OUT
        for (uint i = 0; i < sort_idx && sort_position < TOP_K; i++) {
            indices[AXIS] = i;
            sort_position += isnan(input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])]) ? 1 : 0;
        }
#else
        sort_position = sort_idx;
        for (uint i = sort_idx + 1; i < VALUES_NUM && sort_position < TOP_K; i++) {
            indices[AXIS] = i;
            sort_position += isnan(input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])]) ? 0 : 1;
        }
#endif
        head_end = 0;
        tail_begin = VALUES_NUM;
    }
#endif

    for (uint i = 0; i < head_end / 8; i++) {
        uint index_offset = i * 8;
        indices[AXIS] = index_offset;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 1;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 2;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 3;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 4;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 5;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 6;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 7;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
        if (sort_position + PARALLEL_RANK(counted, index_offset + 8) >= TOP_K)
            return;
    }

    for (uint i = (head_end / 8) * 8; i < head_end; i++) {
        indices[AXIS] = i;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_1(result.value, test_value))
            counted++;
    }

    sort_position += PARALLEL_RANK(counted, head_end);
    if (sort_position >= TOP_K)
        return;

    // As in the head loop, the unrolled blocks start at multiples of 8: f16 blocks at other offsets
    // load much slower.
    counted = 0;
    const uint tail_aligned = min((tail_begin + 7) / 8 * 8, (uint)VALUES_NUM);
    for (uint i = tail_begin; i < tail_aligned; i++) {
        indices[AXIS] = i;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
    }
    for (uint i = tail_aligned / 8; i < VALUES_NUM / 8; i++) {
        uint index_offset = i * 8;
        indices[AXIS] = index_offset;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 1;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 2;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 3;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 4;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 5;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 6;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        indices[AXIS] = index_offset + 7;
        test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
        if (sort_position + PARALLEL_RANK(counted, index_offset + 8 - tail_begin) >= TOP_K)
            return;
    }

    for (uint i = max(tail_aligned, (uint)(VALUES_NUM / 8) * 8); i < VALUES_NUM; i++) {
        indices[AXIS] = i;
        INPUT0_TYPE test_value = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        if (COMPARE_PARALLEL_SIGN_2(result.value, test_value))
            counted++;
    }

    sort_position += PARALLEL_RANK(counted, VALUES_NUM - tail_begin);
    if (sort_position >= TOP_K)
        return;

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
            if (COMPARE_SIGN(val, in_data)) {
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
                    if (COMPARE_PARTIAL_SIGN(result[i].value, result[j].value)) {
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
                        if (COMPARE_PARTIAL_SIGN(temp_buf[group_offset + i].value, temp_buf[group_offset + j].value)) {
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
                    if (COMPARE_PARTIAL_SIGN(temp_buf[last_group_offset + i].value, temp_buf[last_group_offset + j].value)) {
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

    #ifdef USE_INTERNAL_BUFFERS
        const uint counter_size = group_num * 4;
        const uint counter_offset = counter_size * OPERATION_NUM;
        __global uint* merge_counter = OFFSET_GLOBAL_PTR(uint, tmp_buffer1, output_idx * counter_size);
        __global uint* max_merge_counter = OFFSET_GLOBAL_PTR(uint, tmp_buffer1, counter_offset + output_idx * counter_size);
        __global bool* subgroup_done = OFFSET_GLOBAL_PTR(bool, tmp_buffer2, output_idx * group_num);
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

                if (COMPARE_MERGE_SIGN(temp_buf[test_index].value, merge_buf.value)) {
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
#undef PARALLEL_RANK
#undef ORDER_LT
#undef ORDER_LE
#undef ORDER_GT
#undef ORDER_GE
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef MINIMUM_NUMBER_FOR_PARTIAL_SORTING
