// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define AXIS_B (0)
#define AXIS_F (1)
#define AXIS_W (2)
#define AXIS_Z (OUTPUT_DIMS - 3)
#define AXIS_Y (OUTPUT_DIMS - 2)
#define AXIS_X (OUTPUT_DIMS - 1)

#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#define GET_INPUT_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)

#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
#endif

#ifdef BLOCKED_LAYOUT
inline void FUNC(planar_to_bfyx)(const uint planar_index,
                                 const uint batch_num, const uint channel_num, const uint height, const uint width,
                                 uint* dst_b, uint* dst_f, uint* dst_y, uint* dst_x)
{
    const uint feature_size = height * width;
    const uint batch_size = channel_num * feature_size;

    *dst_b = planar_index / batch_size;
    const uint dst_fxy = planar_index % batch_size;
    *dst_f = dst_fxy / feature_size;
    const uint dst_xy = dst_fxy % feature_size;
    *dst_y = dst_xy / width;
    *dst_x = dst_xy % width;
}

#if INPUT2_DIMS == 5
inline void FUNC(planar_to_bfzyx)(const uint planar_index,
                                 const uint batch_num, const uint channel_num, const uint depth, const uint height, const uint width,
                                 uint* dst_b, uint* dst_f, uint* dst_z, uint* dst_y, uint* dst_x)
{
    const uint matrix_size = height * width;
    const uint feature_size = depth * matrix_size;
    const uint batch_size = channel_num * feature_size;

    *dst_b = planar_index / batch_size;
    const uint dst_fzxy = planar_index % batch_size;

    *dst_f = dst_fzxy / feature_size;
    const uint dst_zxy = dst_fzxy % feature_size;

    *dst_z = dst_zxy / matrix_size;
    const uint dst_xy = dst_zxy % matrix_size;

    *dst_y = dst_xy / width;
    *dst_x = dst_xy % width;
}
#elif INPUT2_DIMS == 6
inline void FUNC(planar_to_bfwzyx)(const uint planar_index,
                                 const uint batch_num, const uint channel_num, const uint w_depth, const uint depth, const uint height, const uint width,
                                 uint* dst_b, uint* dst_f, uint* dst_w, uint* dst_z, uint* dst_y, uint* dst_x)
{
    const uint matrix_size = height * width;
    const uint cube_size = depth * matrix_size;
    const uint feature_size = w_depth * cube_size;
    const uint batch_size = channel_num * feature_size;

    *dst_b = planar_index / batch_size;
    const uint dst_fwzxy = planar_index % batch_size;

    *dst_f = dst_fwzxy / feature_size;
    const uint dst_wzxy = dst_fwzxy % feature_size;

    *dst_w = dst_wzxy / cube_size;
    const uint dst_zxy = dst_wzxy % cube_size;

    *dst_z = dst_zxy / matrix_size;
    const uint dst_xy = dst_zxy % matrix_size;

    *dst_y = dst_xy / width;
    *dst_x = dst_xy % width;
}
#endif // INPUT2_DIMS
#endif // BLOCKED_LAYOUT

KERNEL(scatter_update_ref)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* dictionary,
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
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    const uint dict_idx = GET_INPUT_INDEX(ORDER);

    // Use input index instead of output index because output padding is not empty.
    INPUT0_TYPE val = dictionary[dict_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel
    #if (OUTPUT_DIMS == 4)
        // bf|y|x
        #if (AXIS_VALUE == AXIS_F)
            const uint b = dim2 / INDICES_SIZE;
            const uint f = dim2 % INDICES_SIZE;
        #else
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
        #endif
        const uint y = dim1;
        const uint x = dim0;
    #elif (OUTPUT_DIMS == 5)
        // bf|z|yx
        #if (AXIS_VALUE == AXIS_F)
            const uint b = dim2 / INDICES_SIZE;
            const uint f = dim2 % INDICES_SIZE;
        #else
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
        #endif
        const uint z = dim1;
        #if (AXIS_VALUE == AXIS_X)
            const uint y = dim0 / INDICES_SIZE;
            const uint x = dim0 % INDICES_SIZE;
        #else
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint x = dim0 % OUTPUT_SIZE_X;
        #endif
    #elif (OUTPUT_DIMS == 6)
        // bf|wz|yx
        #if (AXIS_VALUE == AXIS_F)
            const uint b = dim2 / INDICES_SIZE;
            const uint f = dim2 % INDICES_SIZE;
        #else
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
        #endif
        #if (AXIS_VALUE == AXIS_Z)
            const uint w = dim1 / INDICES_SIZE;
            const uint z = dim1 % INDICES_SIZE;
        #else
            const uint w = dim1 / OUTPUT_SIZE_Z;
            const uint z = dim1 % OUTPUT_SIZE_Z;
        #endif
        #if (AXIS_VALUE == AXIS_X)
            const uint y = dim0 / INDICES_SIZE;
            const uint x = dim0 % INDICES_SIZE;
        #else
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint x = dim0 % OUTPUT_SIZE_X;
        #endif
    #endif

    #ifdef BLOCKED_LAYOUT
        const uint planar_axis_idx = OUTPUT_INDEX_ON_AXIS;
        uint b_b, b_f, b_w, b_z, b_y, b_x;
        FUNC_CALL(planar_to_bfyx)(planar_axis_idx, INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X,
                       &b_b, &b_f, &b_y, &b_x);
        const uint axis_idx = INPUT1_GET_INDEX(b_b, b_f, b_y, b_x);
        const uint index_by_axis = convert_int(indices[axis_idx]);
    #else
        const uint index_by_axis = convert_int(indices[OUTPUT_INDEX_ON_AXIS]);
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);

    #ifdef BLOCKED_LAYOUT
        const uint planar_updates_idx = GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);

        #if INPUT2_DIMS == 4
            FUNC_CALL(planar_to_bfyx)(planar_updates_idx, INPUT2_BATCH_NUM, INPUT2_FEATURE_NUM, INPUT2_SIZE_Y, INPUT2_SIZE_X,
                           &b_b, &b_f, &b_y, &b_x);
            const uint updates_idx = INPUT2_GET_INDEX(b_b, b_f, b_y, b_x);
        #elif INPUT2_DIMS == 5
            FUNC_CALL(planar_to_bfzyx)(planar_updates_idx, INPUT2_BATCH_NUM, INPUT2_FEATURE_NUM,
                           INPUT2_SIZE_Z, INPUT2_SIZE_Y, INPUT2_SIZE_X,
                           &b_b, &b_f, &b_z, &b_y, &b_x);
            const uint updates_idx = INPUT2_GET_INDEX(b_b, b_f, b_z, b_y, b_x);
        #elif INPUT2_DIMS == 6
            FUNC_CALL(planar_to_bfwzyx)(planar_updates_idx, INPUT2_BATCH_NUM, INPUT2_FEATURE_NUM,
                           INPUT2_SIZE_W, INPUT2_SIZE_Z, INPUT2_SIZE_Y, INPUT2_SIZE_X,
                           &b_b, &b_f, &b_w, &b_z, &b_y, &b_x);
            const uint updates_idx = INPUT2_GET_INDEX(b_b, b_f, b_w, b_z, b_y, b_x);
        #else
            #error Unsupported updates rank
        #endif
    #else
        const uint updates_idx = GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);
    #endif

    INPUT2_TYPE val = updates[updates_idx];

    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
#endif
}

#undef GET_OUTPUT_INDEX
#undef GET_INPUT_INDEX
#undef ORDER
#undef AXIS_B
#undef AXIS_F
#undef AXIS_W
#undef AXIS_Z
#undef AXIS_Y
#undef AXIS_X
