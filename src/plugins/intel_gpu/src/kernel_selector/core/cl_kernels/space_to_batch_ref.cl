// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(space_to_batch_ref)(const __global INPUT0_TYPE* input,
                                 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#if OUTPUT_LAYOUT_BFYX || OUTPUT_LAYOUT_B_FS_YX_FSV16
    const uint w = 0;
    const uint z = 0;
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w = 0;
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint zyx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint w = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint yx = zyx % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = zyx / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#endif

    const uint input_batch = batch % INPUT0_BATCH_NUM;
    const uint offset_batch =  batch / INPUT0_BATCH_NUM;

    const int input_feature = feature * BLOCK_SHAPE_FEATURE - PADS_BEGIN_FEATURE +
                              offset_batch / (BLOCK_SHAPE_W * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);
    const uint offset_feature = offset_batch % (BLOCK_SHAPE_W * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);

    const int input_w = w * BLOCK_SHAPE_W - PADS_BEGIN_W + offset_feature / (BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);
    const uint offset_w = offset_feature % (BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);

    const int input_z = z * BLOCK_SHAPE_Z - PADS_BEGIN_Z + offset_w / (BLOCK_SHAPE_Y * BLOCK_SHAPE_X);
    const uint offset_z = offset_w % (BLOCK_SHAPE_Y * BLOCK_SHAPE_X);

    const int input_y = y * BLOCK_SHAPE_Y - PADS_BEGIN_Y + offset_z / BLOCK_SHAPE_X;
    const uint offset_y = offset_z % BLOCK_SHAPE_X;

    const int input_x = x * BLOCK_SHAPE_X - PADS_BEGIN_X + offset_y;

#if OUTPUT_DIMS == 4
    const int input_index = INPUT0_GET_INDEX(input_batch, input_feature, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, y, x);
#elif OUTPUT_DIMS == 5
    const int input_index = INPUT0_GET_INDEX(input_batch, input_feature, input_z, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
#elif OUTPUT_DIMS == 6
    const int input_index = INPUT0_GET_INDEX(input_batch, input_feature, input_w, input_z, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, w, z, y, x);
#endif

    const bool out_of_bounds = input_feature < 0 || input_feature >= INPUT0_FEATURE_NUM ||
                               input_w < 0 || input_w >= INPUT0_SIZE_W ||
                               input_z < 0 || input_z >= INPUT0_SIZE_Z ||
                               input_y < 0 || input_y >= INPUT0_SIZE_Y ||
                               input_x < 0 || input_x >= INPUT0_SIZE_X;

    INPUT0_TYPE in = out_of_bounds ? INPUT0_VAL_ZERO : input[input_index];
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_index] = FUSED_OPS_RESULT;
#else
    output[output_index] = ACTIVATION(in, ACTIVATION_PARAMS);
#endif
}
