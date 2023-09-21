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
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);

    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;
    for (uint z = 0; z < INPUT0_SIZE_Z; z++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
#if INPUT0_DIMS == 4
                const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
                const uint input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
#endif
                rms += pow(TO_ACCUMULATOR_TYPE(input[input_idx]), 2);
            }
        }
    }

    rms /= INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z;
    rms = pow(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);

    for (uint z = 0; z < INPUT0_SIZE_Z; z++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
#if INPUT0_DIMS == 4
                const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);
                const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
                const uint gamma_idx = y;
#elif INPUT0_DIMS == 5
                const uint input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                const uint output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
                const uint gamma_idx = z;
#endif
                OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms) * TO_OUTPUT_TYPE(input[input_idx]) * TO_OUTPUT_TYPE(gamma[gamma_idx]);
                output[output_idx] = result;
            }
        }
    }
}
