
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_INPUT_INDEX(idx_order)   INPUT0_GET_INDEX(idx_order)
#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order)  OUTPUT_GET_INDEX(idx_order)

#if OUTPUT_DIMS <= 4
    #define ORDER b,f,y,x
    #define TARGET_COORD_ORDER    target_coord[0],target_coord[1],target_coord[2],target_coord[3]
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #define TARGET_COORD_ORDER    target_coord[0],target_coord[1],target_coord[2],target_coord[3],target_coord[4]
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #define TARGET_COORD_ORDER    target_coord[0],target_coord[1],target_coord[2],target_coord[3],target_coord[4],target_coord[5]
#endif

#if INPUT1_DIMS <= 4
    #define INPUT1_ORDER indices_coord[0],indices_coord[1],indices_coord[2],indices_coord[3]
#elif INPUT1_DIMS == 5
    #define INPUT1_ORDER indices_coord[0],indices_coord[1],indices_coord[2],indices_coord[3],indices_coord[4]
#elif INPUT1_DIMS == 6
    #define INPUT1_ORDER indices_coord[0],indices_coord[1],indices_coord[2],indices_coord[3],indices_coord[4],indices_coord[5]
#endif

#if INPUT2_DIMS <= 4
    #define INPUT2_ORDER upd_b,upd_f,upd_y,upd_x
#elif INPUT2_DIMS == 5
    #define INPUT2_ORDER upd_b,upd_f,upd_z,upd_y,upd_x
#elif INPUT2_DIMS == 6
    #define INPUT2_ORDER upd_b,upd_f,upd_w,upd_z,upd_y,upd_x
#endif

#define INDICES_MAX_DIM 6


KERNEL(scatter_nd_update_opt)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
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

#if INPUT2_DIMS <= 4
    const uint upd_x = dim0;
    const uint upd_y = dim1;
    const uint upd_f = dim2 % INPUT2_FEATURE_NUM;
    const uint upd_b = dim2 / INPUT2_FEATURE_NUM;
#elif INPUT2_DIMS == 5
    const uint upd_x = dim0;
    const uint upd_y = dim1 % INPUT2_SIZE_Y;
    const uint upd_z = dim1 / INPUT2_SIZE_Y;
    const uint upd_f = dim2 % INPUT2_FEATURE_NUM;
    const uint upd_b = dim2 / INPUT2_FEATURE_NUM;
#elif INPUT2_DIMS == 6
    const uint upd_x = dim0 % INPUT2_SIZE_X;
    const uint upd_y = dim0 / INPUT2_SIZE_X;
    const uint upd_z = dim1 % INPUT2_SIZE_Z;
    const uint upd_w = dim1 / INPUT2_SIZE_Z;
    const uint upd_f = dim2 % INPUT2_FEATURE_NUM;
    const uint upd_b = dim2 / INPUT2_FEATURE_NUM;
#endif

    uint indices_coord[INDICES_MAX_DIM] = { 0 };
    uint target_coord[INDICES_MAX_DIM];
    uint g_coord[INDICES_MAX_DIM] = { INPUT2_ORDER };

#if INPUT1_LENGTH == 1 && INDICES_RANK == 1
    for (uint i = 0; i < OUTPUT_DIMS; ++i) {
        target_coord[i] = g_coord[i];
    }
#else
    for (uint i = 0; i < INDICES_RANK; ++i) {
        indices_coord[i] = g_coord[i];
    }

    for (uint i = 0; i < INDICES_LAST_DIM; ++i) {
        indices_coord[INDICES_RANK - 1] = i;
        target_coord[i] = indices[GET_INDICES_INDEX(INPUT1_ORDER)];
    }

    for (uint i = INDICES_LAST_DIM; i < OUTPUT_DIMS; ++i) {
        target_coord[i] = g_coord[INDICES_RANK - 1 - INDICES_LAST_DIM + i];
    }
#endif

    const uint output_idx = GET_OUTPUT_INDEX(TARGET_COORD_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(INPUT2_ORDER);

    INPUT2_TYPE val = updates[updates_idx];

    #if HAS_FUSED_OPS
        const uint b = target_coord[0];
        const uint f = target_coord[1];
        #if INPUT0_DIMS <= 4
            const uint y = target_coord[2];
            const uint x = target_coord[3];
        #elif INPUT0_DIMS == 5
            const uint z = target_coord[2];
            const uint y = target_coord[3];
            const uint x = target_coord[4];
        #elif INPUT0_DIMS == 6
            const uint w = target_coord[2];
            const uint z = target_coord[3];
            const uint y = target_coord[4];
            const uint x = target_coord[5];
        #endif

        FUSED_OPS;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
    #else
        output[output_idx] = val;
    #endif
}

#ifdef GET_INPUT_INDEX
#undef GET_INPUT_INDEX
#endif

#ifdef GET_INDICES_INDEX
#undef GET_INDICES_INDEX
#endif

#ifdef GET_UPDATES_INDEX
#undef GET_UPDATES_INDEX
#endif

#ifdef GET_OUTPUT_INDEX
#undef GET_OUTPUT_INDEX
#endif

#ifdef ORDER
#undef ORDER
#endif

#ifdef TARGET_COORD_ORDER
#undef TARGET_COORD_ORDER
#endif

#ifdef INPUT1_ORDER
#undef INPUT1_ORDER
#endif

#ifdef INPUT2_ORDER
#undef INPUT2_ORDER
#endif

#ifdef INDICES_MAX_DIM
#undef INDICES_MAX_DIM
#endif
