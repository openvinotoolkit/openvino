// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"


KERNEL(broadcast_gpu_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{

#if OUTPUT_DIMS >= 5
    uint8 input_indices;
#else
    uint4 input_indices;
#endif

    input_indices[0] = INPUT0_BATCH_NUM;
    input_indices[1] = INPUT0_FEATURE_NUM;
#if OUTPUT_DIMS == 6
    input_indices[2] = INPUT0_SIZE_W;
    input_indices[3] = INPUT0_SIZE_Z;
    input_indices[4] = INPUT0_SIZE_Y;
    input_indices[5] = INPUT0_SIZE_X;
#elif OUTPUT_DIMS == 5
    input_indices[2] = INPUT0_SIZE_Z;
    input_indices[3] = INPUT0_SIZE_Y;
    input_indices[4] = INPUT0_SIZE_X;
#else
    input_indices[2] = INPUT0_SIZE_Y;
    input_indices[3] = INPUT0_SIZE_X;
#endif

#if OUTPUT_DIMS == 6
    const uint in_sx = input_indices[BROADCAST_ORDER[5]];
    const uint in_sy = input_indices[BROADCAST_ORDER[4]];
    const uint in_sz = input_indices[BROADCAST_ORDER[3]];
    const uint in_sw = input_indices[BROADCAST_ORDER[2]];
#elif OUTPUT_DIMS == 5
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
#if OUTPUT_DIMS == 6
    const uint out_wzy = (uint) get_global_id(1);
    const uint out_y  = out_wzy % OUTPUT_SIZE_Y;
    const uint out_z  = (out_wzy / OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z;
    const uint out_w  = (out_wzy / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z;
#elif OUTPUT_DIMS == 5
    const uint out_zy = (uint) get_global_id(1);
    const uint out_y  = out_zy % OUTPUT_SIZE_Y;
    const uint out_z  = out_zy / OUTPUT_SIZE_Y;
    const uint out_w  = 0;
#else
    const uint out_y = (uint) get_global_id(1);
    const uint out_z = 0;
    const uint out_w = 0;
#endif

    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;

#if OUTPUT_DIMS == 6
    const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_w, out_z, out_y, out_x);
#elif OUTPUT_DIMS == 5
    const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_z, out_y, out_x);
#else
    const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_y, out_x);
#endif

#if INPUT0_DIMS == 6
    const uint in_w = out_w % in_sw;
    const uint in_z = out_z % in_sz;
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_z + in_sz * (in_w + in_sw * (in_f + in_sf * in_b))));
#elif INPUT0_DIMS == 5
    const uint in_z = out_z % in_sz;
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_z + in_sz * (in_f + in_sf * in_b)));
#else
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_f + in_sf * in_b));
#endif

    output[out_pos] = input[in_pos];
}
