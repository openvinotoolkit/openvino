
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#if INPUT2_DIMS == 4
    #define UPD_ORDER upd_b,upd_f,upd_y,upd_x
#elif INPUT2_DIMS == 5
    #define UPD_ORDER upd_b,upd_f,upd_z,upd_y,upd_x
#elif INPUT2_DIMS == 6
    #define UPD_ORDER upd_b,upd_f,upd_w,upd_z,upd_y,upd_x
#endif

#if INPUT1_DIMS == 4
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif INPUT1_DIMS == 5
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#elif INPUT1_DIMS == 6
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif

#define INDICES_MAX_DIM 6

KERNEL(scatter_nd_update_ref)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
#ifdef IS_SECOND_ITER
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
#endif
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

    const uint input_idx = GET_UPDATES_INDEX(INPUT0, ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[input_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel

    const uint dataND[] = {INPUT0_BLOCK_ND};
    const uint updatesND[] = {INPUT2_BLOCK_ND};
    const uint indicesND[] = {INPUT1_BLOCK_ND};
    const uint size_to_update = dataND[INDICES_LAST_DIM];

    #if INPUT1_DIMS == 4
        const uint indices_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #elif INPUT1_DIMS == 5
        const uint indices_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #elif INPUT1_DIMS == 6
        const uint indices_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #endif

    #if INPUT0_DIMS == 4
        const uint data_dim[INPUT0_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    #elif INPUT0_DIMS == 5
        const uint data_dim[INPUT0_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    #elif INPUT0_DIMS == 6
        const uint data_dim[INPUT0_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    #endif

    // Get indices index
    uint idx[INDICES_MAX_DIM] = {0};
    uint rmd_idx = dim2;
    for (int i = 0; i < INDICES_RANK - 1; ++i) {
        idx[i] = rmd_idx / indicesND[1 + i];
        rmd_idx %= indicesND[1 + i];
    }

    uint out[INDICES_MAX_DIM] = {0};
    for (int i = 0; i < indices_dim[INDICES_RANK - 1]; ++i) {
        idx[INDICES_RANK - 1] = i;
        const uint idx_b = idx[0];
        const uint idx_f = idx[1];
        #if INPUT1_DIMS == 4
            const uint idx_y = idx[2];
            const uint idx_x = idx[3];
        #elif INPUT1_DIMS == 5
            const uint idx_z = idx[2];
            const uint idx_y = idx[3];
            const uint idx_x = idx[4];
        #elif INPUT1_DIMS == 6
            const uint idx_w = idx[2];
            const uint idx_z = idx[3];
            const uint idx_y = idx[4];
            const uint idx_x = idx[5];
        #endif
        uint index = GET_UPDATES_INDEX(INPUT1, IDX_ORDER);
        out[i] = indices[index];

        // Check if tensor size is valid
        // ex) when data format = bfyx and data shape = { 3, 3, 4, 1 }, indices shape is { 2, 1 } with rank = 2, indices values are { 1.0, 4.0 },
        //     the second indices value is invalid as data shape has 'b' of size 3, and therefore 4 cannot be a correct index of data
        // If indices value is invalid, saturate value to max valid value (ex. 4.0 -> 2.0)
        if(out[i] >= data_dim[i])
            out[i] = data_dim[i] - 1;
    }

    for (int i = 0; i < size_to_update; ++i) {
        // Define updates index
        uint upd[INDICES_MAX_DIM] = {0};
        for (int j = 0; j < INDICES_RANK - 1; ++j) {
            upd[j] = idx[j];
        }
        uint data_rmd = i, updates_rmd = i;
        for (int j = indices_dim[INDICES_RANK - 1]; j < INPUT0_DIMS; ++j) {
            out[j] = data_rmd / dataND[j + 1];
            data_rmd %= dataND[j + 1];
        }
        for (int k = INDICES_RANK - 1; k < INPUT2_DIMS; ++k) {
            upd[k] = updates_rmd / updatesND[k + 1];
            updates_rmd %= updatesND[k + 1];
        }
        // Get update index
        const uint upd_b = upd[0];
        const uint upd_f = upd[1];
        #if INPUT2_DIMS == 4
            const uint upd_y = upd[2];
            const uint upd_x = upd[3];
        #elif INPUT2_DIMS == 5
            const uint upd_z = upd[2];
            const uint upd_y = upd[3];
            const uint upd_x = upd[4];
        #elif INPUT2_DIMS == 6
            const uint upd_w = upd[2];
            const uint upd_z = upd[3];
            const uint upd_y = upd[4];
            const uint upd_x = upd[5];
        #endif
        uint upd_idx = GET_UPDATES_INDEX(INPUT2, UPD_ORDER);

        // Get output index
        const uint b = out[0];
        const uint f = out[1];
        #if INPUT0_DIMS == 4
            const uint y = out[2];
            const uint x = out[3];
        #elif INPUT0_DIMS == 5
            const uint z = out[2];
            const uint y = out[3];
            const uint x = out[4];
        #elif INPUT0_DIMS == 6
            const uint w = out[2];
            const uint z = out[3];
            const uint y = out[4];
            const uint x = out[5];
        #endif
        uint out_idx = GET_OUTPUT_INDEX(ORDER);
        INPUT2_TYPE val = updates[upd_idx];

        #if HAS_FUSED_OPS
            FUSED_OPS_SECOND_KERNEL;
            output[out_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
        #else
            output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
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

#ifdef UPD_ORDER
#undef UPD_ORDER
#endif

#ifdef IDX_ORDER
#undef IDX_ORDER
#endif

#ifdef INDICES_MAX_DIM
#undef INDICES_MAX_DIM
#endif
