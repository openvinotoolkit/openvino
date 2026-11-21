// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if GROUPED_SPACE_TO_DEPTH
#define SPATIAL_BLOCK_SIZE (FACTOR_T*FACTOR_S*FACTOR_S)
#elif OUTPUT_DIMS == 5
#define SPATIAL_BLOCK_SIZE (BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE)
#else
#define SPATIAL_BLOCK_SIZE (BLOCK_SIZE*BLOCK_SIZE)
#endif

KERNEL(space_to_depth_ref)(const __global INPUT0_TYPE* input,
                                 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#if OUTPUT_DIMS == 5
    const uint z = ((uint)get_global_id(2) / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
    const uint y = ((uint)get_global_id(2) / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#else
    const uint z = 0;
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#endif

#if GROUPED_SPACE_TO_DEPTH
    const uint pad_begin_z = (FACTOR_T - INPUT0_SIZE_Z % FACTOR_T) % FACTOR_T;
    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;
    for (uint group_idx = 0; group_idx < GROUP_SIZE; ++group_idx) {
        const uint flat_feature = feature * GROUP_SIZE + group_idx;
        const uint input_feature = flat_feature / FACTOR_VOLUME;
        const uint factor_offset = flat_feature % FACTOR_VOLUME;
        const uint offset_z = factor_offset / (FACTOR_S * FACTOR_S);
        const uint offset_y = (factor_offset / FACTOR_S) % FACTOR_S;
        const uint offset_x = factor_offset % FACTOR_S;
        const uint padded_z = z * FACTOR_T + offset_z;
        if (padded_z >= pad_begin_z) {
            const uint input_z = padded_z - pad_begin_z;
            const uint input_y = y * FACTOR_S + offset_y;
            const uint input_x = x * FACTOR_S + offset_x;
            const uint input_index = INPUT0_GET_INDEX(batch, input_feature, input_z, input_y, input_x);
            acc += TO_ACCUMULATOR_TYPE(input[input_index]);
        }
    }
    INPUT0_TYPE in_val = TO_INPUT0_TYPE(acc / (ACCUMULATOR_TYPE)GROUP_SIZE);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
#else
#if BLOCKS_FIRST_MODE
    const uint input_offset = feature / INPUT0_FEATURE_NUM;
    const uint input_feature = feature % INPUT0_FEATURE_NUM;
#else
    const uint input_offset = feature % SPATIAL_BLOCK_SIZE;
    const uint input_feature = feature / SPATIAL_BLOCK_SIZE;
#endif

#if OUTPUT_DIMS == 5
    const uint input_z = (z * BLOCK_SIZE) + ((input_offset / BLOCK_SIZE) / BLOCK_SIZE);
    const uint input_y = (y * BLOCK_SIZE) + ((input_offset / BLOCK_SIZE) % BLOCK_SIZE);
    const uint input_x = (x * BLOCK_SIZE) + (input_offset % BLOCK_SIZE);
    const uint input_index = INPUT0_GET_INDEX(batch, input_feature, input_z, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
#else
    const uint input_z = 0;
    const uint input_y = (y * BLOCK_SIZE) + (input_offset / BLOCK_SIZE);
    const uint input_x = (x * BLOCK_SIZE) + (input_offset % BLOCK_SIZE);
    const uint input_index = INPUT0_GET_INDEX(batch, input_feature, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, y, x);
#endif

    INPUT0_TYPE in_val = input[input_index];
#endif
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_index] = FUSED_OPS_RESULT;
#else
    output[output_index] = ACTIVATION(in_val, ACTIVATION_PARAMS);
#endif
}
