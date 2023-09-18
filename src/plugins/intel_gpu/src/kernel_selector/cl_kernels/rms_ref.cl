// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(rms_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* gamma,
    __global OUTPUT_TYPE* output)
{
    for (uint b = 0; b < INPUT0_BATCH_NUM; b++)
    {
        for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
        {
            float rms = 0.f;
            for (uint y = 0; y < INPUT0_SIZE_Y; y++)
            {
                for (uint x = 0; x < INPUT0_SIZE_X; x++)
                {
                    const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);
                    rms += pow((float)input[input_idx], 2);
                }
            }
            rms /= INPUT0_SIZE_X * INPUT0_SIZE_Y;
            rms += (float)EPSILON;
            rms = pow(sqrt(rms), -1);

            for (uint y = 0; y < INPUT0_SIZE_Y; y++)
            {
                for (uint x = 0; x < INPUT0_SIZE_X; x++)
                {
                    const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);
                    const uint gamma_idx = INPUT1_GET_INDEX(b, y * INPUT0_SIZE_X + x, y, x);
                    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
                    OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms) * TO_OUTPUT_TYPE(input[input_idx]) * TO_OUTPUT_TYPE(gamma[gamma_idx]);
                    output[output_idx] = result;
                }
            }
        }
    }
}
