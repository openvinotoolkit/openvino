// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#ifdef INDEX_DIM
inline uint FUNC(get_positive_index)(int in)
{
    if(in < 0)
        return in + INDEX_DIM;
    else
        return in;
}
#define INPUT_AXIS_INDEX (uint)FUNC_CALL(get_positive_index)(indices[indices_idx])
#else
#define INPUT_AXIS_INDEX (uint)(indices[indices_idx])
#endif

#define GET_DICTIONARY_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)
#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)

KERNEL(gather_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
    const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
    const uint z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
    const uint w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
    const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #if OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
    #elif OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
    #endif

    const uint indices_idx = GET_INDICES_INDEX(INDICES_INDEX_ORDER);
    const uint dictionary_idx = GET_DICTIONARY_INDEX(DICTIONARY_INDEX_ORDER);
    const uint output_idx = GET_INDEX(OUTPUT,,ORDER);

    INPUT0_TYPE val = dictionary[dictionary_idx];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef GET_INDICES_INDEX
#undef GET_DICTIONARY_INDEX
#undef INPUT_AXIS_INDEX
