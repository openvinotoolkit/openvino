// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(batch_to_space_ref)(const __global INPUT0_TYPE* input,
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
    const uint input_w = 0;
    const uint input_z = 0;
    const uint offset_w = 0;
    const uint offset_z = 0;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w = 0;
    const uint z = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint input_w = 0;
    const uint input_z = (z + CROPS_BEGIN_Z) / BLOCK_SHAPE_Z;
    const uint offset_w = 0;
    const uint offset_z = (z + CROPS_BEGIN_Z) % BLOCK_SHAPE_Z;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint zyx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint yx = zyx % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = zyx / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint input_w = (w + CROPS_BEGIN_W) / BLOCK_SHAPE_W;
    const uint input_z = (z + CROPS_BEGIN_Z) / BLOCK_SHAPE_Z;
    const uint offset_w = (w + CROPS_BEGIN_W) % BLOCK_SHAPE_W;
    const uint offset_z = (z + CROPS_BEGIN_Z) % BLOCK_SHAPE_Z;
#endif

    const uint input_feature = (feature + CROPS_BEGIN_FEATURE) / BLOCK_SHAPE_FEATURE;
    const uint offset_feature = (feature + CROPS_BEGIN_FEATURE) % BLOCK_SHAPE_FEATURE;

    const uint input_y = (y + CROPS_BEGIN_Y) / BLOCK_SHAPE_Y;
    const uint offset_y = (y + CROPS_BEGIN_Y) % BLOCK_SHAPE_Y;

    const uint input_x = (x + CROPS_BEGIN_X) / BLOCK_SHAPE_X;
    const uint offset_x = (x + CROPS_BEGIN_X) % BLOCK_SHAPE_X;

    const uint offset_batch = ((offset_feature * BLOCK_SHAPE_W * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y +
                               offset_w * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y +
                               offset_z * BLOCK_SHAPE_Y +
                               offset_y) * BLOCK_SHAPE_X +
                               offset_x) * OUTPUT_BATCH_NUM;
    const uint input_batch = batch + offset_batch;

#if OUTPUT_DIMS == 4
    const uint input_index = INPUT0_GET_INDEX(input_batch, input_feature, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, y, x);
#elif OUTPUT_DIMS == 5
    const uint input_index = INPUT0_GET_INDEX(input_batch, input_feature, input_z, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
#elif OUTPUT_DIMS == 6
    const uint input_index = INPUT0_GET_INDEX(input_batch, input_feature, input_w, input_z, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, w, z, y, x);
#endif

    INPUT0_TYPE result = input[input_index];
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_index] = FUSED_OPS_RESULT;
#else
    output[output_index] = ACTIVATION(result, ACTIVATION_PARAMS);
#endif
}
