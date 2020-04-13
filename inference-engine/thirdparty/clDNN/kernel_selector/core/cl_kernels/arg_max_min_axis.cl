// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/include_all.cl"

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
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define COMPARE_PARTIAL_SIGN <=
    #define COMPARE_MERGE_SIGN <
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

#define MINIMUM_NUMBER_FOR_PARTIAL_SORTING 100

#define unroll_for __attribute__((opencl_unroll_hint)) for

///////////////////////// Input offset /////////////////////////
inline uint FUNC(get_input_offset)(uint b, uint f, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#else
#error arg_max_min_axis.cl: input format - not supported
#endif
}

///////////////////////// Output offset ////////////////////////
inline uint FUNC(get_output_offset)(uint b, uint f, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#else
#error arg_max_min_axis.cl: output format - not supported
#endif
}

KERNEL(arg_max_min_modified)(const __global INPUT0_TYPE* input
                                  ,__global OUTPUT_TYPE* output
#ifdef SECOND_OUTPUT_EXIST
                                  ,__global INPUT1_TYPE* second_output
#endif
                            )
{
#include "include/arg_max_min_common.cl"
#if (TOP_K == 1)
    iav_type result[TOP_K];
#else
    iav_type result[VALUES_NUM], temp_buf[VALUES_NUM];
    const uint group_size = TOP_K >= 8 ? TOP_K : 8;
    const uint group_num = ((VALUES_NUM - 1) / group_size) + 1;
    const uint last_group_size = (VALUES_NUM % group_size > 0) ? (VALUES_NUM % group_size) : group_size;
    const uint last_group_offset = (group_num - 1) * group_size;
#endif // (TOP_K == 1)

    uint output_idx = (uint)get_global_id(0);

    if (output_idx >= OPERATION_NUM)
        return;

#ifdef BATCH_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_SIZE_X * INPUT0_FEATURE_NUM); // Y
    const uint out_second_dim = output_idx / INPUT0_FEATURE_NUM % INPUT0_SIZE_X; // X
    const uint out_fourth_dim = output_idx % INPUT0_FEATURE_NUM; // F
    uint indices[] = {0, out_fourth_dim, 0, out_first_dim, out_second_dim}; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X); // F
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z; // Z
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y; // Y
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X; // X
    uint indices[] = {0, out_first_dim, out_second_dim, out_third_dim, out_fourth_dim};
    #endif
#endif
#ifdef FEATURE_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_SIZE_X * INPUT0_BATCH_NUM); // Y
    const uint out_second_dim = output_idx / INPUT0_BATCH_NUM % INPUT0_SIZE_X; // X
    const uint out_fourth_dim = output_idx % INPUT0_BATCH_NUM; // B
    uint indices[] = {out_fourth_dim, 0, 0, out_first_dim, out_second_dim}; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X); // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z; // Z
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;  // Y
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;  // X
    uint indices[] = {out_first_dim, 0, out_second_dim, out_third_dim, out_fourth_dim};
    #endif
#endif
#ifdef Z_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X);  // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM; // F
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y; // Y
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X; // X
    uint indices[] = {out_first_dim, out_second_dim, 0, out_third_dim, out_fourth_dim};
#endif
#ifdef Y_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM); // X
    const uint out_second_dim = output_idx / INPUT0_BATCH_NUM % INPUT0_FEATURE_NUM; // F
    const uint out_fourth_dim = output_idx % INPUT0_BATCH_NUM; // B
    uint indices[] = {out_fourth_dim, out_second_dim, 0, 0, out_first_dim}; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_X); // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM; // F
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Z; // Z
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X; // X
    uint indices[] = {out_first_dim, out_second_dim, out_third_dim, 0, out_fourth_dim};
    #endif
#endif
#ifdef X_AXIS
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM); // Y
    const uint out_second_dim = output_idx / INPUT0_BATCH_NUM % INPUT0_FEATURE_NUM; // F
    const uint out_fourth_dim = output_idx % INPUT0_BATCH_NUM; // B
    uint indices[] = {out_fourth_dim, out_second_dim, 0, out_first_dim, 0}; // BFZYX
    #else
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_Y); // B
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y) % INPUT0_FEATURE_NUM; // F
    const uint out_third_dim = output_idx / INPUT0_SIZE_Y % INPUT0_SIZE_Z; // Z
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_Y; // Y
    uint indices[] = {out_first_dim, out_second_dim, out_third_dim, out_fourth_dim, 0};
    #endif
#endif

// Using simple sorting for (TOP_K == 1)
#if (TOP_K == 1)

    INPUT0_TYPE val = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
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
            INPUT0_TYPE in_data = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
            if (val COMPARE_SIGN in_data) {
                result[top_k].index = i;
                result[top_k].value = in_data;
                val = in_data;
            }
        }
        val = INPUT0_FILL_VAL;
    }

// Using merge sorting when (TOP_K >= (VALUES_NUM / 2)) or (VALUES_NUM < MINIMUM_NUMBER_FOR_PARTIAL_SORTING)
#elif ((TOP_K >= (VALUES_NUM / 2)) || (VALUES_NUM < MINIMUM_NUMBER_FOR_PARTIAL_SORTING))

    for (uint i = 0; i < VALUES_NUM / 8; i++) {
        uint index_offset = i * 8;
        indices[AXIS] = result[index_offset].index = index_offset;
        result[index_offset].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 1].index = index_offset + 1;
        result[index_offset + 1].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 2].index = index_offset + 2;
        result[index_offset + 2].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 3].index = index_offset + 3;
        result[index_offset + 3].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 4].index = index_offset  + 4;
        result[index_offset + 4].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 5].index = index_offset + 5;
        result[index_offset + 5].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 6].index = index_offset + 6;
        result[index_offset + 6].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = result[index_offset + 7].index = index_offset + 7;
        result[index_offset + 7].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
    }

    for (uint i = (VALUES_NUM / 8) * 8; i < VALUES_NUM; i++) {
        indices[AXIS] = result[i].index = i;
        result[i].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
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

// In other cases using mixed partial/merge sorting
#else // (TOP_K == 1)

    for (uint i = 0; i < VALUES_NUM / 8; i++) {
        uint index_offset = i * 8;
        indices[AXIS] = temp_buf[index_offset].index = index_offset;
        temp_buf[index_offset].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 1].index = index_offset + 1;
        temp_buf[index_offset + 1].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 2].index = index_offset + 2;
        temp_buf[index_offset + 2].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 3].index = index_offset + 3;
        temp_buf[index_offset + 3].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 4].index = index_offset  + 4;
        temp_buf[index_offset + 4].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 5].index = index_offset + 5;
        temp_buf[index_offset + 5].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 6].index = index_offset + 6;
        temp_buf[index_offset + 6].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
        indices[AXIS] = temp_buf[index_offset + 7].index = index_offset + 7;
        temp_buf[index_offset + 7].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
    }

    for (uint i = (VALUES_NUM / 8) * 8; i < VALUES_NUM; i++) {
        indices[AXIS] = temp_buf[i].index = i;
        temp_buf[i].value = input[FUNC_CALL(get_input_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])];
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

    uint merge_counter[group_num];
    uint max_merge_counter[group_num];
    iav_type merge_buf;
    bool subgroup_done[group_num];

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

#endif // (TOP_K == 1)

    for (uint top_k = 0; top_k < TOP_K; ++top_k) {
#ifdef SORT_BY_VALUE
        indices[AXIS] = top_k;
#endif
#ifdef SORT_BY_INDEX
        uint out_position = 0;
        for (uint i = 0; i < TOP_K; ++i) {
            if (i == top_k)
                continue;
            if (result[i].index < result[top_k].index)
                out_position++;
        }
        indices[AXIS] = out_position;
#endif
#ifdef TOP_K_ORDER
    output[FUNC_CALL(get_output_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(result[top_k].value);
#else
    output[FUNC_CALL(get_output_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(result[top_k].index);
#endif
#ifdef SECOND_OUTPUT_EXIST
#ifdef TOP_K_ORDER
    second_output[FUNC_CALL(get_output_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])] = TO_INPUT1_TYPE(result[top_k].index);
#else
    second_output[FUNC_CALL(get_output_offset)(indices[0], indices[1], indices[2], indices[3], indices[4])] = TO_INPUT1_TYPE(result[top_k].value);
#endif
#endif
    }
}

#undef COMPARE_SIGN
#undef COMPARE_PARTIAL_SIGN
#undef COMPARE_MERGE_SIGN
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef MINIMUM_NUMBER_FOR_PARTIAL_SORTING
#undef unroll_for
