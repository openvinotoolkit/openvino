// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"


KERNEL(broadcast_gpu_ref)(
    const __global INPUT0_TYPE* input,
    __global INPUT0_TYPE* output)
{
    // [CONSTEXPR]
    // Input sizes:
#ifdef OUTPUT_LAYOUT_BFZYX
    uint8 input_indices;
#else
    uint4 input_indices;
#endif
    input_indices[0] = INPUT0_BATCH_NUM;
    input_indices[1] = INPUT0_FEATURE_NUM;
#ifdef OUTPUT_LAYOUT_BFZYX
    input_indices[2] = INPUT0_SIZE_Z;
    input_indices[3] = INPUT0_SIZE_Y;
    input_indices[4] = INPUT0_SIZE_X;
#else
    input_indices[2] = INPUT0_SIZE_Y;
    input_indices[3] = INPUT0_SIZE_X;
#endif
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint in_sx = input_indices[BROADCAST_ORDER[4]];
    const uint in_sy = input_indices[BROADCAST_ORDER[3]];
    const uint in_sz = input_indices[BROADCAST_ORDER[2]];
#else
    const uint in_sx = input_indices[BROADCAST_ORDER[3]];
    const uint in_sy = input_indices[BROADCAST_ORDER[2]];
#endif
    const uint in_sf = input_indices[BROADCAST_ORDER[1]];
    const uint in_sb = input_indices[BROADCAST_ORDER[0]];

    const uint out_x  = (uint) get_global_id(0);
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint out_zy = (uint) get_global_id(1);
    const uint out_y  = out_zy % OUTPUT_SIZE_Y;
    const uint out_z  = out_zy / OUTPUT_SIZE_Y;
#else
    const uint out_y = (uint) get_global_id(1);
#endif

    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint in_z = out_z % in_sz;
#endif
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_z + in_sz * (in_f + in_sf * in_b)));
    const uint out_pos = GET_DATA_INDEX_5D(OUTPUT, out_b, out_f, out_z, out_y, out_x);
#else
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_f + in_sf * in_b));
    const uint out_pos = GET_DATA_INDEX(OUTPUT, out_b, out_f, out_y, out_x);
#endif
    output[out_pos] = input[in_pos];
}
