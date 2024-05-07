// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#else
    #error
#endif

KERNEL(adaptive_pooling_gpu)(
        const __global INPUT0_TYPE* input,
        __global OUTPUT_TYPE* output
#if MAX_POOLING && defined(OUTPUT1_TYPE)
        , __global OUTPUT1_TYPE* output1
#endif
)
{
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;

    const uint x   = (uint)get_global_id(0);
#if OUTPUT_DIMS == 5
    const uint y   = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z   = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
#else
    const uint y   = (uint)get_global_id(1);
    const uint z   = 0;
#endif

    ACCUMULATOR_TYPE result = INIT_VAL;

#if MAX_POOLING
#if defined OUTPUT1_TYPE
    OUTPUT1_TYPE result_idx = 0;
#endif
#elif AVG_POOLING
    uint num_elements = 0;
#else
#error
#endif

#if OUTPUT_DIMS == 5
    uint z_start = z * INPUT0_SIZE_Z / OUTPUT_SIZE_Z;
    uint z_end = ((z + 1) * INPUT0_SIZE_Z) / OUTPUT_SIZE_Z;
    z_end += (((z + 1) * INPUT0_SIZE_Z) - OUTPUT_SIZE_Z * z_end != 0) ? 1 : 0;
#endif
    uint y_start = y * INPUT0_SIZE_Y / OUTPUT_SIZE_Y;
    uint y_end = ((y + 1) * INPUT0_SIZE_Y) / OUTPUT_SIZE_Y;
    y_end += (((y + 1) * INPUT0_SIZE_Y) - OUTPUT_SIZE_Y * y_end != 0) ? 1 : 0;
    uint x_start = x * INPUT0_SIZE_X / OUTPUT_SIZE_X;
    uint x_end = ((x + 1) * INPUT0_SIZE_X) / OUTPUT_SIZE_X;
    x_end += (((x + 1) * INPUT0_SIZE_X) - OUTPUT_SIZE_X * x_end != 0) ? 1 : 0;


#if OUTPUT_DIMS == 5
    for (uint k = z_start; k < z_end; ++k) {
        const uint z_offset = k * INPUT0_SIZE_Y * INPUT0_SIZE_X;
#else
    const uint z_offset = 0;
#endif
        for (uint j = y_start; j < y_end; ++j) {
            uint y_offset = z_offset + j * INPUT0_SIZE_X;

            for (uint i = x_start; i < x_end; ++i) {
                #if OUTPUT_DIMS == 5
                    const uint idx = INPUT0_GET_INDEX(b, f, k, j, i);
                #else
                    const uint idx = INPUT0_GET_INDEX(b, f, j, i);
                #endif

                const ACCUMULATOR_TYPE current_input_value = TO_ACCUMULATOR_TYPE(input[idx]);
#if MAX_POOLING
                if (current_input_value > result) {
                    result = current_input_value;
                    #if OUTPUT_DIMS == 5
                        result_idx = INPUT0_GET_INDEX(0, 0, k, j, i);
                    #else
                        result_idx = INPUT0_GET_INDEX(0, 0, j, i);
                    #endif
                }
#elif AVG_POOLING
                result += TO_ACCUMULATOR_TYPE(current_input_value);
                ++num_elements;
#else
#error
#endif
            }
        }
#if OUTPUT_DIMS == 5
    }
#endif

#if OUTPUT_DIMS == 5
    const uint output_pos = OUTPUT_GET_INDEX(b, f, z, y, x);
#else
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);
#endif

#if MAX_POOLING
#if defined OUTPUT1_TYPE
    #if OUTPUT_DIMS == 5
        const uint index_pos = OUTPUT1_GET_INDEX(b, f, z, y, x);
    #else
        const uint index_pos = OUTPUT1_GET_INDEX(b, f, y, x);
    #endif

    output[output_pos] = result;
    output1[index_pos] = result_idx;
#endif
#elif AVG_POOLING
    output[output_pos] = result / TO_ACCUMULATOR_TYPE(max(num_elements, (uint)1));
#else
#error
#endif
}

#undef INIT_VAL
