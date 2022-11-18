// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

KERNEL(fc)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
#if RANK == 2
    #define WEIGHT_COL_NUMBER OUTPUT_FEATURE_NUM
    #define INPUT_COL_NUMBER INPUT0_FEATURE_NUM

    const uint out_y = get_global_id(0);
    const uint out_x = get_global_id(1);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint in_x = 0; in_x < INPUT_COL_NUMBER; ++in_x) {
        uint input_idx = INPUT0_GET_INDEX(out_y, in_x, 0, 0);
        uint weights_idx = WEIGHT_COL_NUMBER * in_x + out_x;
        dotProd += input[input_idx] * weights[weights_idx];
    }

    uint output_idx = OUTPUT_GET_INDEX(out_y, out_x, 0, 0);
    output[output_idx] = TO_OUTPUT_TYPE(dotProd);

#elif RANK == 3
    #define WEIGHT_COL_NUMBER OUTPUT_SIZE_Y
    #define INPUT_COL_NUMBER INPUT0_SIZE_Y
    #define WEIGHT_BATCH_SIZE (FILTER_IFM_NUM * FILTER_SIZE_Y)

    const uint out_batch = get_global_id(0);
    const uint out_y = get_global_id(1);
    const uint out_x = get_global_id(2);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint in_x = 0; in_x < INPUT_COL_NUMBER; ++in_x) {
        uint input_idx = INPUT0_GET_INDEX(out_batch, out_y, in_x, 0);
        uint weights_idx = (out_batch * WEIGHT_BATCH_SIZE) + WEIGHT_COL_NUMBER * in_x + out_x;
        dotProd += input[input_idx] * weights[weights_idx];
    }

    uint output_idx = OUTPUT_GET_INDEX(out_batch, out_y, out_x, 0);
    output[output_idx] = TO_OUTPUT_TYPE(dotProd);

#elif RANK == 4
    #define WEIGHT_COL_NUMBER OUTPUT_SIZE_X
    #define INPUT_COL_NUMBER INPUT0_SIZE_X
    #define WEIGHT_BATCH2_SIZE (FILTER_SIZE_Y * FILTER_SIZE_X)
    #define WEIGHT_BATCH1_SIZE (WEIGHT_BATCH2_SIZE * FILTER_IFM_NUM)
    const uint batch = get_global_id(0);
    const uint out_batch1 = batch / INPUT0_FEATURE_NUM;
    const uint out_batch2 = batch % INPUT0_FEATURE_NUM;
    const uint out_y = get_global_id(1);
    const uint out_x = get_global_id(2);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint in_x = 0; in_x < INPUT_COL_NUMBER; ++in_x) {
        uint input_idx = INPUT0_GET_INDEX(out_batch1, out_batch2, out_y, in_x);
        uint weights_idx = (out_batch1 * WEIGHT_BATCH1_SIZE) + (out_batch2 * WEIGHT_BATCH2_SIZE) + WEIGHT_COL_NUMBER * in_x + out_x;
        dotProd += input[input_idx] * weights[weights_idx];
    }

    uint output_idx = OUTPUT_GET_INDEX(out_batch1, out_batch2, out_y, out_x);
    output[output_idx] = TO_OUTPUT_TYPE(dotProd);

#elif RANK == 5
    #define WEIGHT_COL_NUMBER OUTPUT_SIZE_X
    #define INPUT_COL_NUMBER INPUT0_SIZE_X

    #define WEIGHT_BATCH3_SIZE (FILTER_SIZE_Y * FILTER_SIZE_X)
    #define WEIGHT_BATCH2_SIZE (WEIGHT_BATCH3_SIZE * FILTER_SIZE_Z)
    #define WEIGHT_BATCH1_SIZE (WEIGHT_BATCH2_SIZE * FILTER_IFM_NUM)
    const uint batch = get_global_id(0);
    const uint out_batch1 = batch / INPUT0_FEATURE_NUM;
    const uint out_batch2 = batch % INPUT0_FEATURE_NUM;
    const uint batch_y = get_global_id(1);
    const uint out_batch3 = batch_y / INPUT0_SIZE_Y;
    const uint out_y = batch_y % INPUT0_SIZE_Y;
    const uint out_x = get_global_id(2);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint in_x = 0; in_x < INPUT_COL_NUMBER; ++in_x) {
        uint input_idx = INPUT0_GET_INDEX(out_batch1, out_batch2, out_batch3, out_y, in_x);
        uint weights_idx = (out_batch1 * WEIGHT_BATCH1_SIZE) +
                           (out_batch2 * WEIGHT_BATCH2_SIZE) +
                           (out_batch3 * WEIGHT_BATCH3_SIZE) +
                           WEIGHT_COL_NUMBER * in_x + out_x;
        dotProd += input[input_idx] * weights[weights_idx];
    }

    uint output_idx = OUTPUT_GET_INDEX(out_batch1, out_batch2, out_batch3, out_y, out_x);
    output[output_idx] = TO_OUTPUT_TYPE(dotProd);

#elif RANK == 6
    // Do we really need this?
#else
#error Invalid rank
#endif
}
