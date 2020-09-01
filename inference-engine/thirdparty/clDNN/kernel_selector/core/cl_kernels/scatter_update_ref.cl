// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
#endif

KERNEL(scatter_update_ref)(const __global INPUT0_TYPE* dictionary,
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
    INPUT0_TYPE val = dictionary[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        #if AXIS_VALUE == 0
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
        #else
            const uint f = dim2 / OUTPUT_BATCH_NUM;
            const uint b = dim2 % OUTPUT_BATCH_NUM;
        #endif
    #elif OUTPUT_DIMS == 5
        const uint z = dim1;
        #if AXIS_VALUE == 1
            const uint f = dim2 / OUTPUT_BATCH_NUM;
            const uint b = dim2 % OUTPUT_BATCH_NUM;
            const uint x = dim0 % OUTPUT_SIZE_X;
            const uint y = dim0 / OUTPUT_SIZE_X;
        #elif AXIS_VALUE == 4
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint x = dim0 / OUTPUT_SIZE_Y;
            const uint y = dim0 % OUTPUT_SIZE_Y;
        #else
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint x = dim0 % OUTPUT_SIZE_X;
            const uint y = dim0 / OUTPUT_SIZE_X;
        #endif
    #elif OUTPUT_DIMS == 6
        #if AXIS_VALUE == 1
            const uint f = dim2 / OUTPUT_BATCH_NUM;
            const uint b = dim2 % OUTPUT_BATCH_NUM;
            const uint x = dim0 % OUTPUT_SIZE_X;
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint z = dim1 % OUTPUT_SIZE_Z;
            const uint w = dim1 / OUTPUT_SIZE_Z;
        #elif AXIS_VALUE == 3
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint x = dim0 % OUTPUT_SIZE_X;
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint z = dim1 / OUTPUT_SIZE_W;
            const uint w = dim1 % OUTPUT_SIZE_W;
        #elif AXIS_VALUE == 5
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint x = dim0 / OUTPUT_SIZE_Y;
            const uint y = dim0 % OUTPUT_SIZE_Y;
            const uint z = dim1 % OUTPUT_SIZE_Z;
            const uint w = dim1 / OUTPUT_SIZE_Z;
        #else
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint x = dim0 % OUTPUT_SIZE_X;
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint z = dim1 % OUTPUT_SIZE_Z;
            const uint w = dim1 / OUTPUT_SIZE_Z;
        #endif
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(INPUT2, UPDATES_INDEX_ORDER);

    INPUT2_TYPE val = updates[updates_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
#endif
}

#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
