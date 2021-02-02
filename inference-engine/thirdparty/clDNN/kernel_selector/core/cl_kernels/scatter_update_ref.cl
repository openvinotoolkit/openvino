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

#define AXIS_B (0)
#define AXIS_F (1)
#define AXIS_W (2)
#define AXIS_Z (OUTPUT_DIMS - 3)
#define AXIS_Y (OUTPUT_DIMS - 2)
#define AXIS_X (OUTPUT_DIMS - 1)

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

    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);

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
#undef AXIS_B
#undef AXIS_F
#undef AXIS_W
#undef AXIS_Z
#undef AXIS_Y
#undef AXIS_X
