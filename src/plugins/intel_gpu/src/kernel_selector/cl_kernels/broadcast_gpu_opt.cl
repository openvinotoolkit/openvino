// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)

#if INPUT0_DIMS == 4
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif INPUT0_DIMS == 5
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#elif INPUT0_DIMS == 6
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif


#define VLOAD CAT(vload, VEC_SIZE)
#define VSTORE CAT(vstore,VEC_SIZE)
#define INPUT0_VTYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)

KERNEL(broadcast_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
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

    const uint leftovers = INPUT0_SIZE_X % VEC_SIZE;

    const uint blockND[] = {INPUT0_BLOCK_ND};

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

    const uint gdim0 = (uint) get_global_id(0);
    uint offset = leftovers;
    if (gdim0 < leftovers) {
        offset = gdim0;
    }

    const uint out_x  = (uint) (get_global_id(0) * VEC_SIZE + offset);
    const uint out_y  = (uint) (get_global_id(1) * Y_BLOCK_SIZE);
#if OUTPUT_DIMS == 6
    const uint out_bfwz = (uint) get_global_id(2);
    const uint out_z  = (out_bfwz % OUTPUT_SIZE_Z);
    const uint out_w  = (out_bfwz / OUTPUT_SIZE_Z) % OUTPUT_SIZE_W;
    const uint out_f  = (out_bfwz / OUTPUT_SIZE_Z / OUTPUT_SIZE_W) % OUTPUT_FEATURE_NUM;
    const uint out_b  = (out_bfwz / OUTPUT_SIZE_Z / OUTPUT_SIZE_W / OUTPUT_FEATURE_NUM);
#elif OUTPUT_DIMS == 5
    const uint out_bfwz = (uint) get_global_id(2);
    const uint out_z  = (out_bfwz % OUTPUT_SIZE_Z);
    const uint out_w  = 0;
    const uint out_f  = (out_bfwz / OUTPUT_SIZE_Z) % OUTPUT_FEATURE_NUM;
    const uint out_b  = (out_bfwz / OUTPUT_SIZE_Z / OUTPUT_FEATURE_NUM);
#else
    const uint out_bfwz = (uint) get_global_id(2);
    const uint out_z  = 0;
    const uint out_w  = 0;
    const uint out_f  = (out_bfwz % OUTPUT_FEATURE_NUM);
    const uint out_b  = (out_bfwz / OUTPUT_FEATURE_NUM);
#endif

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;

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

    uint idx[INPUT0_DIMS] = {0};
    uint rmd = in_pos;
    unroll_for (int i = 0; i < INPUT0_DIMS; ++i)
    {
        idx[i] = rmd / blockND[i + 1];
        rmd %= blockND[i + 1];
    }
    const uint idx_b = idx[0];
    const uint idx_f = idx[1];
    #if INPUT0_DIMS == 4
        const uint idx_y = idx[2];
        const uint idx_x = idx[3];
    #elif INPUT0_DIMS == 5
        const uint idx_z = idx[2];
        const uint idx_y = idx[3];
        const uint idx_x = idx[4];
    #elif INPUT0_DIMS == 6
        const uint idx_w = idx[2];
        const uint idx_z = idx[3];
        const uint idx_y = idx[4];
        const uint idx_x = idx[5];
    #endif

    const uint idx_pos = GET_UPDATES_INDEX(INPUT0, IDX_ORDER);
#if OUTPUT_DIMS == 6
        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_w, out_z, out_y, out_x);
#elif OUTPUT_DIMS == 5
        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_z, out_y, out_x);
#else
        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_y, out_x);
#endif

    INPUT0_VTYPE input_vec = VLOAD(0, &input[idx_pos]);
    unroll_for(uint i = 0; i < Y_BLOCKS; i++) {
        VSTORE(input_vec, 0, &output[out_pos + i * OUTPUT_SIZE_X]);
    }

    if (gdim0 < leftovers) {
        output[out_pos + offset] = input[idx_pos + offset];
    }
}

#ifdef IDX_ORDER
#undef IDX_ORDER
#endif
