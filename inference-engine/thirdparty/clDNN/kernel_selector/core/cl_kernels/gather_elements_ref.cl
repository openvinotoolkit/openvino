// Copyright (c) 2021 Intel Corporation
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

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)

#define ORDER b,f,y,x
#define IN_ORDER in_b,in_f,in_y,in_x

#if INPUT1_DIMS == 4
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif INPUT1_DIMS == 5
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#else
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif

#define OUT_ORDER out_b,out_f,out_y,out_x
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)

#define INDICES_MAX_DIM 6

KERNEL(gather_nd_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    // Calculate indice index
    const uint F_NUM = INPUT1_FEATURE_NUM;
    const uint idx_f = dim2 % F_NUM;
    const uint idx_b = dim2 / F_NUM;

#if INPUT1_DIMS == 4
    // const uint idx_x = dim0; // y
    // const uint idx_y = dim1; // x
    // const uint idx_z = 0;
    // const uint idx_w = 0;
    const uint idx_x = dim0;
    const uint idx_y = dim1;
#elif INPUT1_DIMS == 5
        // const uint idx_x = dim0 / INPUT1_SIZE_Y; // z
        // const uint idx_y = dim0 % INPUT1_SIZE_Y; // y
        // const uint idx_z = dim1; // x
        // const uint idx_w = 0;
    const uint idx_x = dim0 % OUTPUT_SIZE_X;
    const uint idx_y = dim0 / OUTPUT_SIZE_X;
    const uint idx_z = dim1;

#else
    // INPUT1_DIMS == 6
    const uint idx_x = dim0 % OUTPUT_SIZE_X; // x
    const uint idx_y = dim0 / OUTPUT_SIZE_X; // y
    const uint idx_z = dim1 % OUTPUT_SIZE_Z; // z
    const uint idx_w = dim1 / OUTPUT_SIZE_Z; // w
#endif

    const int out_idx = GET_UPDATES_INDEX(INPUT1, IDX_ORDER);
    // printf("%d\n", out_idx);
    int axis = AXIS;
    size_t rank = INPUT0_DIMS; // indices_shape.size(), data_shape.size()
//     printf("rank and axis: %d %d\n", rank, axis);

    size_t data_shape[10] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_X, INPUT0_SIZE_Y, INPUT0_SIZE_Z, INPUT0_SIZE_W};
    size_t indices_shape[10] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_X, INPUT1_SIZE_Y, INPUT1_SIZE_Z, INPUT1_SIZE_W};

    size_t max_inner_sum = 1, max_outer_sum = 1, outer_sum_inc_data = 1, outer_sum_inc_indices = 1;
    for (size_t i = axis + 1; i < rank; i++)
        max_inner_sum *= indices_shape[i];

    for (int i = 0; i < axis; i++)
        max_outer_sum *= indices_shape[i];

    for (size_t i = axis; i < rank; i++) {
        outer_sum_inc_data *= data_shape[i];
    }
    max_outer_sum *= outer_sum_inc_data;

    for (size_t i = axis; i < rank; i++) {
        outer_sum_inc_indices *= indices_shape[i];
    }

//     printf("max_inner_sum: %ld\n", max_inner_sum);
//     printf("outer_sum_inc_data: %ld\n",outer_sum_inc_data);
//     printf("max_inner_sum, max_outer_sum, outer_sum_inc_data: %d %d %d\n",max_inner_sum, max_outer_sum, outer_sum_inc);

// ========================================================================================

    size_t outer_sum = (out_idx / outer_sum_inc_indices) * outer_sum_inc_data;
    size_t inner_sum = out_idx % max_inner_sum;
    if (indices[out_idx] < 0 || indices[out_idx] >= data_shape[axis]) {
        printf("indices values of GatherElement exceed data size.\n");
        return;
    }
    uint idx = outer_sum + max_inner_sum * indices[out_idx] + inner_sum;
    uint tmp = outer_sum;

    INPUT0_TYPE val = data[idx];
    output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);

    // output[out_idx] = TO_OUTPUT_TYPE(axis);
    // output[out_idx] = axis;
// ========================================================================================

    // output[out_idx] = TO_OUTPUT_TYPE(out_idx);

// ========================================================================================

    // for (size_t outer_sum = 0, i = 0; outer_sum < max_outer_sum; outer_sum += outer_sum_inc_data) {
    //     for (size_t k = 0; k < indices_shape[axis]; k++) {
    //         for (size_t inner_sum = 0; inner_sum < max_inner_sum; inner_sum++) {
    //             if (indices[i] < 0 || indices[i] >= data_shape[axis])
    //             {
    //                 printf("indices values of GatherElement exceed data size.\n");
    //                 return;
    //             }

    //             // uint idx = outer_sum + max_inner_sum * indices[i] + inner_sum;
    //             uint idx = outer_sum;
    //             // uint idx = max_inner_sum * indices[i];
    //             // INPUT0_TYPE val = data[idx];
    //             // output[i] = ACTIVATION(val, ACTIVATION_PARAMS);
    //             output[i] = idx;
    //             // output[output_idx] = TO_OUTPUT_TYPE(val);
    //             i++;
    //         }
    //     }
    // }
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
