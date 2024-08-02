// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_OUTPUT_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)

KERNEL(gather_elements_ref)(OPTIONAL_SHAPE_INFO_ARG
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
#if INPUT1_DIMS == 4
    #define ORDER b,f,y,x
    const uint x = dim0;
    const uint y = dim1;
#elif INPUT1_DIMS == 5
    #define ORDER b,f,z,y,x
    const uint x = dim0;
    const uint y = dim1 % OUTPUT_SIZE_Y;
    const uint z = dim1 / OUTPUT_SIZE_Y;
#else
    #define ORDER b,f,w,z,y,x
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
#endif
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;

    const uint out_idx = GET_OUTPUT_INDEX(INPUT1, ORDER);
    const uint input_idx = GET_OUTPUT_INDEX(INPUT0, DATA_INDEX_ORDER);

    INPUT0_TYPE val = data[input_idx];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[out_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef ORDER
#undef GET_OUTPUT_INDEX
