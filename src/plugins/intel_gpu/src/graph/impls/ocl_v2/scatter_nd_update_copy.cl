
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)

#if OUTPUT_DIMS <= 4
    #define ORDER b,f,y,x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
#endif

KERNEL(scatter_nd_update_copy)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

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
        FUSED_OPS;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
    #else
        output[output_idx] = val;
    #endif
}

#ifdef GET_UPDATES_INDEX
#undef GET_UPDATES_INDEX
#endif

#ifdef GET_OUTPUT_INDEX
#undef GET_OUTPUT_INDEX
#endif
