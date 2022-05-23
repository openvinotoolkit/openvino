// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
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
#if MAX_POOLING
        , __global INDICES_TYPE* indices
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

    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0, 0);
#else
    const uint y   = (uint)get_global_id(1);
    const uint z   = 0;

    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0);
#endif

    ACCUMULATOR_TYPE result = INIT_VAL;

#if MAX_POOLING
    INDICES_TYPE result_idx = 0;
#elif AVG_POOLING
    uint num_elements = 0;
#else
#error
#endif

#if OUTPUT_DIMS == 5
    uint z_start = z * INPUT0_SIZE_Z / OUTPUT_SIZE_Z;
    uint z_end = ceil((float)((z + 1) * INPUT0_SIZE_Z) / OUTPUT_SIZE_Z);
#endif
    uint y_start = y * INPUT0_SIZE_Y / OUTPUT_SIZE_Y;
    uint y_end = ceil((float)((y + 1) * INPUT0_SIZE_Y) / OUTPUT_SIZE_Y);
    uint x_start = x * INPUT0_SIZE_X / OUTPUT_SIZE_X;
    uint x_end = ceil((float)((x + 1) * INPUT0_SIZE_X) / OUTPUT_SIZE_X);


#if OUTPUT_DIMS == 5
    for (uint k = z_start; k < z_end; ++k) {
        const uint z_offset = k * INPUT0_SIZE_Y * INPUT0_SIZE_X;
#else
    const uint z_offset = 0;
#endif
        for (uint j = y_start; j < y_end; ++j) {
            uint y_offset = z_offset + j * INPUT0_SIZE_X;

            for (uint i = x_start; i < x_end; ++i) {
                uint idx_within_feature = y_offset + i;

                const current_input_value = TO_ACCUMULATOR_TYPE(input[batch_and_feature_offset + idx_within_feature]);
#if MAX_POOLING
                if (current_input_value > result) {
                    result = current_input_value;
                    result_idx = idx_within_feature;
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
    output[output_pos] = result;
    indices[output_pos] = result_idx;
#elif AVG_POOLING
    output[output_pos] = result / TO_ACCUMULATOR_TYPE(max(num_elements, (uint)1));
#else
#error
#endif
}

#undef INIT_VAL
