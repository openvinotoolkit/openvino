// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(space_to_batch_ref)(const __global INPUT0_TYPE* input,
#ifdef BLOCK_TYPE
                          const __global BLOCK_TYPE* block,
#endif
#ifdef BEGIN_TYPE
                          const __global BEGIN_TYPE* begin,
#endif
#ifdef END_TYPE
                          const __global END_TYPE* end,
#endif
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


#ifdef BLOCK_TYPE
    const uint block_f = block[1];
    #if BLOCK_DIMS == 3
    const uint block_x = 1;
    const uint block_y = block[BLOCK_DIMS-1];
    const uint block_z = 1;
    const uint block_w = 1;
    #else
    const uint block_x = BLOCK_DIMS > 2 ? block[BLOCK_DIMS-1] : 1;
    const uint block_y = BLOCK_DIMS > 3 ? block[BLOCK_DIMS-2] : 1;
    const uint block_z = BLOCK_DIMS > 4 ? block[BLOCK_DIMS-3] : 1;
    const uint block_w = BLOCK_DIMS > 5 ? block[BLOCK_DIMS-4] : 1;
    #endif
#else
    const uint block_f = BLOCK_SHAPE_FEATURE;
    const uint block_x = BLOCK_SHAPE_X;
    const uint block_y = BLOCK_SHAPE_Y;
    const uint block_z = BLOCK_SHAPE_Z;
    const uint block_w = BLOCK_SHAPE_W;
#endif


#ifdef BEGIN_TYPE
    const uint begin_f = begin[1];
    #if BEGIN_DIMS == 3
    const uint begin_x = 0;
    const uint begin_y = begin[BEGIN_DIMS-1];
    const uint begin_z = 0;
    const uint begin_w = 0;
    #else
    const uint begin_x = BEGIN_DIMS > 2 ? begin[BEGIN_DIMS-1] : 0;
    const uint begin_y = BEGIN_DIMS > 3 ? begin[BEGIN_DIMS-2] : 0;
    const uint begin_z = BEGIN_DIMS > 4 ? begin[BEGIN_DIMS-3] : 0;
    const uint begin_w = BEGIN_DIMS > 5 ? begin[BEGIN_DIMS-4] : 0;
    #endif
#else
    const uint begin_f = PADS_BEGIN_FEATURE;
    const uint begin_x = PADS_BEGIN_X;
    const uint begin_y = PADS_BEGIN_Y;
    const uint begin_z = PADS_BEGIN_Z;
    const uint begin_w = PADS_BEGIN_W;
#endif

    const int input_feature = feature * block_f - begin_f +
                              offset_batch / (block_w * block_z * block_y * block_x);
    const uint offset_feature = offset_batch % (block_w * block_z * block_y * block_x);

    const int input_w = w * block_w - begin_w + offset_feature / (block_z * block_y * block_x);
    const uint offset_w = offset_feature % (block_z * block_y * block_x);

    const int input_z = z * block_z - begin_z + offset_w / (block_y * block_x);
    const uint offset_z = offset_w % (block_y * block_x);

    const int input_y = y * block_y - begin_y + offset_z / block_x;
    const uint offset_y = offset_z % block_x;

    const int input_x = x * block_x - begin_x + offset_y;

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
