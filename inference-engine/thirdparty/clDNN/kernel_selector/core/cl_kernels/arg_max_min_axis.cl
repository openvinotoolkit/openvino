// Copyright (c) 2018 Intel Corporation
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

#include "include/fetch.cl"
#include "include/data_types.cl"

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
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

KERNEL(arg_max_min_modified)(const __global INPUT0_TYPE* input
                                  ,__global OUTPUT_TYPE* output
#ifdef SECOND_OUTPUT_EXIST
                                  ,__global OUTPUT_TYPE* second_output
#endif
                            )
{
#include "include/arg_max_min_common.cl"
    iav_type result[TOP_K];
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

    INPUT0_TYPE val = input[GET_DATA_INDEX_5D(INPUT0, indices[0], indices[1], indices[2], indices[3], indices[4])];
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
            INPUT0_TYPE in_data = input[GET_DATA_INDEX_5D(INPUT0, indices[0], indices[1], indices[2], indices[3], indices[4])];
            if (val COMPARE_SIGN in_data) {
                result[top_k].index = i;
                result[top_k].value = in_data;
                val = in_data;
            }
        }
        val = INPUT0_FILL_VAL;
    }

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
    output[GET_DATA_INDEX_5D(OUTPUT, indices[0], indices[1], indices[2], indices[3], indices[4])] = result[top_k].value;
#else
    output[GET_DATA_INDEX_5D(OUTPUT, indices[0], indices[1], indices[2], indices[3], indices[4])] = result[top_k].index;
#endif
#ifdef SECOND_OUTPUT_EXIST
#ifdef TOP_K_ORDER
    second_output[GET_DATA_INDEX_5D(OUTPUT, indices[0], indices[1], indices[2], indices[3], indices[4])] = result[top_k].index;
#else
    second_output[GET_DATA_INDEX_5D(OUTPUT, indices[0], indices[1], indices[2], indices[3], indices[4])] = result[top_k].value;
#endif
#endif
    }
}

#undef COMPARE_SIGN
#undef INPUT0_FILL_VAL
#undef VALUES_NUM
