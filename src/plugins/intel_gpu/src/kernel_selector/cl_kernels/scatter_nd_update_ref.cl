
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_INPUT_INDEX(idx_order)   INPUT0_GET_INDEX(idx_order)
#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order)  OUTPUT_GET_INDEX(idx_order)

#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
    #define TARGET_COORD_ORDER    target_coord[0],target_coord[1],target_coord[2],target_coord[3]
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #define TARGET_COORD_ORDER    target_coord[0],target_coord[1],target_coord[2],target_coord[3],target_coord[4]
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #define TARGET_COORD_ORDER    target_coord[0],target_coord[1],target_coord[2],target_coord[3],target_coord[4],target_coord[5]
#endif

#if INPUT2_DIMS == 4
    #define INPUT2_ORDER b,f,y,x
#elif INPUT2_DIMS == 5
    #define INPUT2_ORDER b,f,z,y,x
#elif INPUT2_DIMS == 6
    #define INPUT2_ORDER b,f,w,z,y,x
#endif

#define INDICES_MAX_DIM 6


#if INDICES_RANK == 1
    #define IND_ORDER  i,0,0,0
#elif INDICES_RANK == 2
    #if INPUT1_DIMS == 4
        #define IND_ORDER  b,i,0,0
    #elif INPUT1_DIMS == 5
        #define IND_ORDER  b,i,0,0,0
    #elif INPUT1_DIMS == 6
        #define IND_ORDER  b,i,0,0,0,0
    #endif
#elif INDICES_RANK == 3
#define IND_ORDER  b,f,0,i
#elif INDICES_RANK == 4
    #if INPUT1_DIMS == 4
        #if INPUT2_DIMS == 4
            #define IND_ORDER  b,f,y,i
        #elif INPUT2_DIMS == 5
            #define IND_ORDER  b,f,z,i
        #elif INPUT2_DIMS == 6
            #define IND_ORDER  b,f,w,i
        #endif
    #elif INPUT1_DIMS == 5
        #define IND_ORDER  b,f,y,i,0
    #endif
// #elif INDICES_RANK == 5
//         target_coord[i] = indices[INPUT1_GET_INDEX(b, f, z, y, i)];
// #elif INDICES_RANK == 6
//         target_coord[i] = indices[INPUT1_GET_INDEX(b, f, w, z, y, i)];
#endif



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

#ifdef IS_FIRST_ITER
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;

    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[input_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // IS_SECOND_ITER

#if INPUT2_DIMS == 4
    const uint x = dim0;
    const uint y = dim1;
    const uint f = dim2 % INPUT2_FEATURE_NUM;
    const uint b = dim2 / INPUT2_FEATURE_NUM;
#elif INPUT2_DIMS == 5
    const uint x = dim0;
    const uint y = dim1 % INPUT2_SIZE_Y;
    const uint z = dim1 / INPUT2_SIZE_Y;
    const uint f = dim2 % INPUT2_FEATURE_NUM;
    const uint b = dim2 / INPUT2_FEATURE_NUM;
#elif INPUT2_DIMS == 6
    const uint x = dim0 % INPUT2_SIZE_X;
    const uint y = dim0 / INPUT2_SIZE_X;
    const uint z = dim1 % INPUT2_SIZE_Z;
    const uint w = dim1 / INPUT2_SIZE_Z;
    const uint f = dim2 % INPUT2_FEATURE_NUM;
    const uint b = dim2 / INPUT2_FEATURE_NUM;
#endif

    INPUT1_TYPE target_coord[INDICES_MAX_DIM];
    INPUT1_TYPE g_coord[INDICES_MAX_DIM] = { INPUT2_ORDER };

#if INPUT1_LENGTH == 1 && INDICES_RANK == 1
    for (uint i = 0; i < OUTPUT_DIMS; ++i) {
        target_coord[i] = g_coord[i];
    }
#else
    for (uint i = 0; i < INDICES_LAST_DIM; ++i) {
        target_coord[i] = indices[GET_INDICES_INDEX(IND_ORDER)];
    }

    for (uint i = INDICES_LAST_DIM; i < OUTPUT_DIMS; ++i) {
        target_coord[i] = g_coord[INDICES_RANK - 1 - INDICES_LAST_DIM + i];
    }
#endif

    const uint output_idx = GET_OUTPUT_INDEX(TARGET_COORD_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(INPUT2_ORDER);

    INPUT2_TYPE val = updates[updates_idx];

    // printf("g_coord[%2d,%2d,%2d,%2d,%2d] target_coord[%2d,%2d,%2d,%2d,%2d] output_id(%2d) updates_idx(%2d) val(%.2f) INDICES_LAST_DIM(%d) OUTPUT_DIMS(%d)\n",
    //         g_coord[0], g_coord[1], g_coord[2], g_coord[3], g_coord[4],
    //         target_coord[0], target_coord[1], target_coord[2], target_coord[3], target_coord[4],
    //         output_idx, updates_idx, val, INDICES_LAST_DIM, OUTPUT_DIMS);

    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
#endif  // IS_SECOND_ITER
}

#ifdef ORDER
#undef ORDER
#endif

#ifdef INDICES_MAX_DIM
#undef INDICES_MAX_DIM
#endif
