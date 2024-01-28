// Copyright (C) 2018-2024 Intel Corporation
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


#if OUTPUT_DIMS == 6
inline uint FUNC(get_idx_pos)(OPTIONAL_SHAPE_INFO_ARG uint out_b, uint out_f, uint out_w, uint out_z, uint out_y, uint out_x) {

    uint8 input_indices;

    input_indices[0] = INPUT0_BATCH_NUM;
    input_indices[1] = INPUT0_FEATURE_NUM;
    input_indices[2] = INPUT0_SIZE_W;
    input_indices[3] = INPUT0_SIZE_Z;
    input_indices[4] = INPUT0_SIZE_Y;
    input_indices[5] = INPUT0_SIZE_X;

    const uint in_sx = input_indices[BROADCAST_ORDER[5]];
    const uint in_sy = input_indices[BROADCAST_ORDER[4]];
    const uint in_sz = input_indices[BROADCAST_ORDER[3]];
    const uint in_sw = input_indices[BROADCAST_ORDER[2]];
    const uint in_sf = input_indices[BROADCAST_ORDER[1]];
    const uint in_sb = input_indices[BROADCAST_ORDER[0]];

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;
    const uint in_w = out_w % in_sw;
    const uint in_z = out_z % in_sz;
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_z + in_sz * (in_w + in_sw * (in_f + in_sf * in_b))));

    const uint blockND[] = {INPUT0_BLOCK_ND};
    uint idx[INPUT0_DIMS] = {0};
    uint rmd = in_pos;
    unroll_for (int i = 0; i < INPUT0_DIMS; ++i)
    {
        idx[i] = rmd / blockND[i + 1];
        rmd %= blockND[i + 1];
    }
    const uint idx_b = idx[0];
    const uint idx_f = idx[1];
    const uint idx_w = idx[2];
    const uint idx_z = idx[3];
    const uint idx_y = idx[4];
    const uint idx_x = idx[5];

    const uint idx_pos = GET_UPDATES_INDEX(INPUT0, IDX_ORDER);
    return idx_pos;
}
#elif OUTPUT_DIMS == 5
inline uint FUNC(get_idx_pos)(OPTIONAL_SHAPE_INFO_ARG uint out_b, uint out_f, uint out_z, uint out_y, uint out_x) {

    uint8 input_indices;

    input_indices[0] = INPUT0_BATCH_NUM;
    input_indices[1] = INPUT0_FEATURE_NUM;
    input_indices[2] = INPUT0_SIZE_Z;
    input_indices[3] = INPUT0_SIZE_Y;
    input_indices[4] = INPUT0_SIZE_X;

    const uint in_sx = input_indices[BROADCAST_ORDER[4]];
    const uint in_sy = input_indices[BROADCAST_ORDER[3]];
    const uint in_sz = input_indices[BROADCAST_ORDER[2]];
    const uint in_sf = input_indices[BROADCAST_ORDER[1]];
    const uint in_sb = input_indices[BROADCAST_ORDER[0]];

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;
    const uint in_z = out_z % in_sz;
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_z + in_sz * (in_f + in_sf * in_b)));

    const uint blockND[] = {INPUT0_BLOCK_ND};
    uint idx[INPUT0_DIMS] = {0};
    uint rmd = in_pos;
    unroll_for (int i = 0; i < INPUT0_DIMS; ++i)
    {
        idx[i] = rmd / blockND[i + 1];
        rmd %= blockND[i + 1];
    }
    const uint idx_b = idx[0];
    const uint idx_f = idx[1];
    const uint idx_z = idx[2];
    const uint idx_y = idx[3];
    const uint idx_x = idx[4];
    const uint idx_pos = GET_UPDATES_INDEX(INPUT0, IDX_ORDER);

    return idx_pos;
}
#else
inline uint FUNC(get_idx_pos)(OPTIONAL_SHAPE_INFO_ARG uint out_b, uint out_f, uint out_y, uint out_x) {

    uint4 input_indices;

    input_indices[0] = INPUT0_BATCH_NUM;
    input_indices[1] = INPUT0_FEATURE_NUM;
    input_indices[2] = INPUT0_SIZE_Y;
    input_indices[3] = INPUT0_SIZE_X;


    const uint in_sx = input_indices[BROADCAST_ORDER[3]];
    const uint in_sy = input_indices[BROADCAST_ORDER[2]];
    const uint in_sf = input_indices[BROADCAST_ORDER[1]];
    const uint in_sb = input_indices[BROADCAST_ORDER[0]];

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_f + in_sf * in_b));

    const uint blockND[] = {INPUT0_BLOCK_ND};
    uint idx[INPUT0_DIMS] = {0};
    uint rmd = in_pos;
    unroll_for(int i = 0; i < INPUT0_DIMS; ++i)
    {
        idx[i] = rmd / blockND[i + 1];
        rmd %= blockND[i + 1];
    }
    const uint idx_b = idx[0];
    const uint idx_f = idx[1];
    const uint idx_y = idx[2];
    const uint idx_x = idx[3];
    const uint idx_pos = GET_UPDATES_INDEX(INPUT0, IDX_ORDER);

    return idx_pos;
}
#endif

#define VLOAD CAT(vload, VEC_SIZE)
#define VSTORE CAT(vstore,VEC_SIZE)
#define INPUT0_VTYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)

KERNEL(broadcast_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
#if SAME_RANK_PLAIN_FORMAT == 1
    const bool use_opt_code = INPUT0_SIZE_X == OUTPUT_SIZE_X && INPUT0_SIZE_Y != OUTPUT_SIZE_Y
                        && INPUT0_BATCH_NUM == OUTPUT_BATCH_NUM && INPUT0_FEATURE_NUM == OUTPUT_FEATURE_NUM
                        && INPUT0_SIZE_W == OUTPUT_SIZE_W && INPUT0_SIZE_Z == OUTPUT_SIZE_Z;
#else
    const bool use_opt_code = false;
#endif

    if (use_opt_code) {
        const uint gdim0 = (uint) get_global_id(0);
        const uint gdim1 = (uint) get_global_id(1);

        const uint x_stride = min(VEC_SIZE, OUTPUT_SIZE_X);
        const uint x_leftovers = OUTPUT_SIZE_X % x_stride;
        const uint x_offset = min(x_leftovers, gdim0);
        const uint out_x  = (uint) (gdim0 * x_stride + x_offset);

        const uint y_stride = min(Y_BLOCKS, OUTPUT_SIZE_Y);
        const uint y_leftovers = OUTPUT_SIZE_Y % y_stride;
        const uint y_offset = min(y_leftovers, gdim1);
        const uint out_y  = (uint) (gdim1 * y_stride + y_offset);

#if OUTPUT_DIMS == 6
        const uint out_bfwz = (uint) get_global_id(2);
        const uint out_z  = (out_bfwz % OUTPUT_SIZE_Z);
        const uint out_w  = (out_bfwz / OUTPUT_SIZE_Z) % OUTPUT_SIZE_W;
        const uint out_f  = (out_bfwz / OUTPUT_SIZE_Z / OUTPUT_SIZE_W) % OUTPUT_FEATURE_NUM;
        const uint out_b  = (out_bfwz / OUTPUT_SIZE_Z / OUTPUT_SIZE_W / OUTPUT_FEATURE_NUM);

        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_w, out_z, out_y, out_x);
        const uint idx_pos = FUNC_CALL(get_idx_pos)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_w, out_z, out_y, out_x);
#elif OUTPUT_DIMS == 5
        const uint out_bfwz = (uint) get_global_id(2);
        const uint out_z  = (out_bfwz % OUTPUT_SIZE_Z);
        const uint out_w  = 0;
        const uint out_f  = (out_bfwz / OUTPUT_SIZE_Z) % OUTPUT_FEATURE_NUM;
        const uint out_b  = (out_bfwz / OUTPUT_SIZE_Z / OUTPUT_FEATURE_NUM);

        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_z, out_y, out_x);
        const uint idx_pos = FUNC_CALL(get_idx_pos)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_z, out_y, out_x);
#else
        const uint out_bfwz = (uint) get_global_id(2);
        const uint out_z  = 0;
        const uint out_w  = 0;
        const uint out_f  = (out_bfwz % OUTPUT_FEATURE_NUM);
        const uint out_b  = (out_bfwz / OUTPUT_FEATURE_NUM);

        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_y, out_x);
        const uint idx_pos = FUNC_CALL(get_idx_pos)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_y, out_x);
#endif

        uint y_nums = y_stride;
        if (gdim1 < y_leftovers)
            y_nums += 1;

        uint remained_y = OUTPUT_SIZE_Y - (out_y + y_nums);
        if (remained_y < y_stride)
            y_nums += remained_y;

        if (OUTPUT_SIZE_X < VEC_SIZE) {
            uint output_idx = out_pos;
            unroll_for(uint j = 0; j < y_nums; j++) {
                unroll_for(uint i = 0; i < x_stride; i++) {
                    output[output_idx + i] = input[idx_pos + i];
                }
                output_idx += OUTPUT_SIZE_X;
            }
        } else {
            uint output_idx = out_pos;
            INPUT0_VTYPE input_vec = VLOAD(0, &input[idx_pos]);
            unroll_for(uint i = 0; i < y_nums; i++) {
                VSTORE(input_vec, 0, &output[output_idx]);
                output_idx += OUTPUT_SIZE_X;
            }

            if (gdim0 < x_leftovers) {
                INPUT0_TYPE input_val = input[idx_pos + x_stride];

                output_idx = out_pos;
                unroll_for(uint i = 0; i < y_nums; i++) {
                    output[output_idx + x_stride] = input_val;
                    output_idx += OUTPUT_SIZE_X;
                }
            }
        }
    } else {
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

#if OUTPUT_DIMS == 6
        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_w, out_z, out_y, out_x);
        const uint idx_pos = FUNC_CALL(get_idx_pos)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_w, out_z, out_y, out_x);
#elif OUTPUT_DIMS == 5
        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_z, out_y, out_x);
        const uint idx_pos = FUNC_CALL(get_idx_pos)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_z, out_y, out_x);
#else
        const uint out_pos = OUTPUT_GET_INDEX(out_b, out_f, out_y, out_x);
        const uint idx_pos = FUNC_CALL(get_idx_pos)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_y, out_x);
#endif
        output[out_pos] = input[idx_pos];
    }
}

#ifdef IDX_ORDER
#undef IDX_ORDER
#endif
