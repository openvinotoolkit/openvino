// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)

#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
#endif

KERNEL(scatter_nd_update_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{

    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

#ifndef IS_SECOND_ITER // First kernel
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;

    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel

    const uint blockND[] = {INPUT_BLOCK_ND};
    const uint k = INDICES_LAST_DIM;
    const uint size_to_update = blockND[INDICES_LAST_DIM];
    const uint indices_idx = dim2;
    const uint indices_offset = indices_idx * k;
    uint dst_offset = 0;

    for (uint i = 0; i < k; i++) {
        INPUT1_TYPE idxValue = indices[indices_offset + i];
        dst_offset += idxValue * blockND[i + 1];
    }

    uint update_offset = indices_idx * size_to_update;

    for (int i = 0; i < size_to_update; i++) {
        uint dst_idx = dst_offset + i;
        uint up_idx = update_offset + i;
        INPUT2_TYPE val = updates[up_idx];

    #if HAS_FUSED_OPS
        #if OUTPUT_DIMS == 4
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint f_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint b_pitch = f_pitch * OUTPUT_FEATURE_NUM;

            const uint b_remain = dst_idx % b_pitch;
            const uint f_remain = b_remain % f_pitch;
            const uint y_remain = f_remain % y_pitch;

            const uint b = dst_idx / b_pitch;
            const uint f = b_remain / f_pitch;
            const uint y = f_remain / y_pitch;
            const uint x = y_remain;
        #elif OUTPUT_DIMS == 5
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint f_pitch = z_pitch * OUTPUT_SIZE_Z;
            const uint b_pitch = f_pitch * OUTPUT_FEATURE_NUM;

            const uint b_remain = dst_idx % b_pitch;
            const uint f_remain = b_remain % f_pitch;
            const uint z_remain = f_remain % z_pitch;
            const uint y_remain = z_remain % y_pitch;

            const uint b = dst_idx / b_pitch;
            const uint f = b_remain / f_pitch;
            const uint z = f_remain / z_pitch;
            const uint y = z_remain / y_pitch;
            const uint x = y_remain;
        #elif OUTPUT_DIMS == 6
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint w_pitch = z_pitch * OUTPUT_SIZE_Z;
            const uint f_pitch = w_pitch * OUTPUT_SIZE_W;
            const uint b_pitch = f_pitch * OUTPUT_FEATURE_NUM;

            const uint b_remain = dst_idx % b_pitch;
            const uint f_remain = b_remain % f_pitch;
            const uint w_remain = f_remain % w_pitch;
            const uint z_remain = w_remain % z_pitch;
            const uint y_remain = z_remain % y_pitch;

            const uint b = dst_idx / b_pitch;
            const uint f = b_remain / f_pitch;
            const uint w = f_remain / w_pitch;
            const uint z = w_remain / z_pitch;
            const uint y = z_remain / y_pitch;
            const uint x = y_remain;
        #endif

        FUSED_OPS_SECOND_KERNEL;
        output[dst_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[dst_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
    }
#endif

}

#ifdef GET_UPDATES_INDEX
#undef GET_UPDATES_INDEX
#endif

#ifdef GET_OUTPUT_INDEX
#undef GET_OUTPUT_INDEX
#endif

#ifdef ORDER
#undef ORDER
#endif
