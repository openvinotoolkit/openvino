// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(out_order) OUTPUT_GET_INDEX(out_order)

#if INPUT0_DIMS == 4
    #define IN_ORDER in_b,in_f,in_y,in_x
#elif INPUT0_DIMS == 5
    #define IN_ORDER in_b,in_f,in_z,in_y,in_x
#else
    #define IN_ORDER in_b,in_f,in_w,in_z,in_y,in_x
#endif

#if INPUT1_DIMS == 4
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif INPUT1_DIMS == 5
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#else
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif

#if OUTPUT_DIMS == 4
    #define OUT_ORDER out_b,out_f,out_y,out_x
#elif OUTPUT_DIMS == 5
    #define OUT_ORDER out_b,out_f,out_z,out_y,out_x
#else
    #define OUT_ORDER out_b,out_f,out_w,out_z,out_y,out_x
#endif

#define INDICES_MAX_DIM 6

KERNEL(gather_nd_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* data,
    const __global INPUT1_TYPE* indices,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    // Calculate indice index
    const uint F_NUM = (INDICES_RANK == 2) ? 1 : INPUT1_FEATURE_NUM;
    const uint idx_f = dim2 % F_NUM;
    const uint idx_b = dim2 / F_NUM;

    #if INPUT1_DIMS == 4
        const uint idx_x = dim0;
        const uint idx_y = dim1;
        const uint idx_z = 0;
        const uint idx_w = 0;

        const uint idx_arr[INPUT1_DIMS*2] = {idx_b, idx_f, idx_y, idx_x, 0, 0, 0, 0};
        const uint idx_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #elif INPUT1_DIMS == 5
        const uint X_NUM = (INDICES_RANK == 5) ? 1 : INPUT1_SIZE_X;

        const uint idx_x = dim0 % X_NUM;
        const uint idx_y = dim0 / X_NUM;
        const uint idx_z = dim1;
        const uint idx_w = 0;

        const uint idx_arr[INPUT1_DIMS*2] = {idx_b, idx_f, idx_z, idx_y, idx_x, 0, 0, 0, 0, 0};
        const uint idx_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #else
        const uint X_NUM = (INDICES_RANK == 6) ? 1 : INPUT1_SIZE_X;
        const uint Z_NUM = (INDICES_RANK == 4) ? 1 : INPUT1_SIZE_Z;

        const uint idx_x = dim0 % X_NUM;
        const uint idx_y = dim0 / X_NUM;
        const uint idx_z = dim1 % Z_NUM;
        const uint idx_w = dim1 / Z_NUM;

        const uint idx_arr[INPUT1_DIMS*2] = {idx_b, idx_f, idx_w, idx_z, idx_y, idx_x, 0, 0, 0, 0, 0, 0};
        const uint idx_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #endif

#if IS_DYNAMIC
    uint wi_slice = 1;
    #if INPUT0_DIMS == 4
        uint input_dims[4] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    #elif INPUT0_DIMS == 5
        uint input_dims[5] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    #else
        uint input_dims[6] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    #endif
    const uint indices_last_dim = idx_dim[INDICES_RANK - 1];
    for (uint i = BATCH_DIMS + indices_last_dim; i < INPUT0_DIMS; i++)
        wi_slice *= input_dims[i];
#else
    const uint wi_slice = WI_SLICE_SIZE;
    const uint indices_last_dim = INDICES_LAST_DIM;
#endif

    const int idx = GET_UPDATES_INDEX(INPUT1, IDX_ORDER);

    // Calculate data index
    uint indices_val[INDICES_MAX_DIM + BATCH_DIMS];
    for (uint i = 0; i < INDICES_MAX_DIM + BATCH_DIMS; i++) {
        indices_val[i] = 0;
    }

    for (uint i = 0; i < BATCH_DIMS; i++) {
        indices_val[i] = idx_arr[i];
    }

    for (uint i = 0; i < indices_last_dim; i++) {
        indices_val[i + BATCH_DIMS] = indices[idx+i];
    }

    #if INPUT0_DIMS == 4
        const uint in_x = indices_val[3];
        const uint in_y = indices_val[2];
    #elif INPUT0_DIMS == 5
        const uint in_x = indices_val[4];
        const uint in_y = indices_val[3];
        const uint in_z = indices_val[2];
    #else
        const uint in_x = indices_val[5];
        const uint in_y = indices_val[4];
        const uint in_z = indices_val[3];
        const uint in_w = indices_val[2];
    #endif
    const uint in_f = indices_val[1];
    const uint in_b = indices_val[0];

    const uint data_idx = GET_UPDATES_INDEX(INPUT0, IN_ORDER);

    // Calculate output index
    #if BATCH_MERGED_OUTPUT && BATCH_DIMS > 1
        uint pitch_acc = 1;
        uint output_batch_size = 0;
        for (int i = BATCH_DIMS - 1; i >= 0; i--) {
            output_batch_size += (idx_arr[i] * pitch_acc);
            pitch_acc *= idx_dim[i];
        }

        #if OUTPUT_DIMS == 4
            const uint out_x = idx_arr[BATCH_DIMS+2];
            const uint out_y = idx_arr[BATCH_DIMS+1];
        #elif OUTPUT_DIMS == 5
            const uint out_x = idx_arr[BATCH_DIMS+3];
            const uint out_y = idx_arr[BATCH_DIMS+2];
            const uint out_z = idx_arr[BATCH_DIMS+1];
        #else
            const uint out_x = idx_arr[BATCH_DIMS+4];
            const uint out_y = idx_arr[BATCH_DIMS+3];
            const uint out_z = idx_arr[BATCH_DIMS+2];
            const uint out_w = idx_arr[BATCH_DIMS+1];
        #endif
        const uint out_f = idx_arr[BATCH_DIMS+0];
        const uint out_b = output_batch_size;
    #else
        #if OUTPUT_DIMS == 4
            const uint out_x = idx_arr[3];
            const uint out_y = idx_arr[2];
        #elif OUTPUT_DIMS == 5
            const uint out_x = idx_arr[4];
            const uint out_y = idx_arr[3];
            const uint out_z = idx_arr[2];
        #else
            const uint out_x = idx_arr[5];
            const uint out_y = idx_arr[4];
            const uint out_z = idx_arr[3];
            const uint out_w = idx_arr[2];
        #endif
        const uint out_f = idx_arr[1];
        const uint out_b = idx_arr[0];

    #endif

    const uint output_idx = GET_OUTPUT_INDEX(OUT_ORDER);

    // Copy data to output as slice size
    #if HAS_FUSED_OPS
        #if OUTPUT_DIMS == 4
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint f_pitch = y_pitch * OUTPUT_SIZE_Y;
        #elif OUTPUT_DIMS == 5
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint f_pitch = z_pitch * OUTPUT_SIZE_Z;
        #else
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint w_pitch = z_pitch * OUTPUT_SIZE_Z;
            const uint f_pitch = w_pitch * OUTPUT_SIZE_W;
        #endif
        const uint b_pitch = f_pitch * OUTPUT_FEATURE_NUM;
    #endif

    for (uint i = 0; i < wi_slice; i++) {
        uint dst_idx = output_idx + i;
        INPUT0_TYPE val = data[data_idx + i];

        #if HAS_FUSED_OPS
            const uint b_remain = dst_idx % b_pitch;
            const uint f_remain = b_remain % f_pitch;
            #if OUTPUT_DIMS == 4
                const uint y_remain = f_remain % y_pitch;

                const uint y = f_remain / y_pitch;
            #elif OUTPUT_DIMS == 5
                const uint z_remain = f_remain % z_pitch;
                const uint y_remain = z_remain % y_pitch;

                const uint z = f_remain / z_pitch;
                const uint y = z_remain / y_pitch;
            #else
                const uint w_remain = f_remain % w_pitch;
                const uint z_remain = w_remain % z_pitch;
                const uint y_remain = z_remain % y_pitch;

                const uint w = f_remain / w_pitch;
                const uint z = w_remain / z_pitch;
                const uint y = z_remain / y_pitch;
            #endif
            const uint b = dst_idx / b_pitch;
            const uint f = b_remain / f_pitch;
            const uint x = y_remain;

            #if FUSED_OPS_CAN_USE_PRELOAD
                FUSED_OPS_PRELOAD;
                FUSED_OPS_CALC;
            #else
                FUSED_OPS;
            #endif

            output[dst_idx] = FUSED_OPS_RESULT;
        #else
            output[dst_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
        #endif
    }
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
