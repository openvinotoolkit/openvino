// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(swiglu_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
    ACCUMULATOR_TYPE res;
    for (uint b = 0; b < OUTPUT_BATCH_NUM; b++)
    {
        for (uint f = 0; f < OUTPUT_FEATURE_NUM; f++)
        {
            for (uint y = 0; y < OUTPUT_SIZE_Y; y++)
            {
                for (uint x = 0; x < OUTPUT_SIZE_X; x++)
                {
                    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
                    const uint gate_idx = INPUT0_GET_INDEX(b, f, y, x);
                    const uint input_idx = INPUT0_GET_INDEX(b, f, y, x) + SPLIT_LENGTH;
                    res = (ACCUMULATOR_TYPE)input[gate_idx];
                    res = (res / (ACCUMULATOR_VAL_ONE + (exp((-(ACCUMULATOR_VAL_ONE * res))))));
                    res = res * (ACCUMULATOR_TYPE)input[input_idx];
                    output[output_idx] = TO_OUTPUT_TYPE(res);
                }
            }
        }
    }
}
