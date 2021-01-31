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
    #if AXIS_VALUE == AXIS_B
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_F_PITCH OUTPUT_FEATURE_PITCH
        #define UPDATES_B_PITCH OUTPUT_BATCH_PITCH
    #elif AXIS_VALUE == AXIS_F
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_F_PITCH OUTPUT_FEATURE_PITCH
        #define UPDATES_B_PITCH OUTPUT_FEATURE_PITCH * INDICES_SIZE
    #elif AXIS_VALUE == AXIS_Y
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_F_PITCH OUTPUT_Y_PITCH * INDICES_SIZE
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #elif AXIS_VALUE == AXIS_X
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_X_PITCH * INDICES_SIZE
        #define UPDATES_F_PITCH UPDATES_Y_PITCH * OUTPUT_SIZE_Y
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #endif
    #define GET_UPDATES_INDEX(idx_order) \
         (INPUT2_OFFSET) + \
         (x)*(UPDATES_X_PITCH) + \
         (y)*(UPDATES_Y_PITCH) + \
         (f)*(UPDATES_F_PITCH) + \
         (b)*(UPDATES_B_PITCH)
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #if AXIS_VALUE == AXIS_B
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_F_PITCH OUTPUT_FEATURE_PITCH
        #define UPDATES_B_PITCH OUTPUT_BATCH_PITCH
    #elif AXIS_VALUE == AXIS_F
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_F_PITCH OUTPUT_FEATURE_PITCH
        #define UPDATES_B_PITCH OUTPUT_FEATURE_PITCH * INDICES_SIZE
    #elif AXIS_VALUE == AXIS_Z
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_F_PITCH OUTPUT_Z_PITCH * INDICES_SIZE
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #elif AXIS_VALUE == AXIS_Y
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Y_PITCH * INDICES_SIZE
        #define UPDATES_F_PITCH UPDATES_Z_PITCH * OUTPUT_SIZE_Z
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #elif AXIS_VALUE == AXIS_X
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_X_PITCH * INDICES_SIZE
        #define UPDATES_Z_PITCH UPDATES_Y_PITCH * OUTPUT_SIZE_Y
        #define UPDATES_F_PITCH UPDATES_Z_PITCH * OUTPUT_SIZE_Z
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #endif
    #define GET_UPDATES_INDEX(idx_order) \
         (INPUT2_OFFSET) + \
         (x)*(UPDATES_X_PITCH) + \
         (y)*(UPDATES_Y_PITCH) + \
         (z)*(UPDATES_Z_PITCH) + \
         (f)*(UPDATES_F_PITCH) + \
         (b)*(UPDATES_B_PITCH)

#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #if AXIS_VALUE == AXIS_B
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_W_PITCH OUTPUT_W_PITCH
        #define UPDATES_F_PITCH OUTPUT_FEATURE_PITCH
        #define UPDATES_B_PITCH OUTPUT_BATCH_PITCH
    #elif AXIS_VALUE == AXIS_F
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_W_PITCH OUTPUT_W_PITCH
        #define UPDATES_F_PITCH OUTPUT_FEATURE_PITCH
        #define UPDATES_B_PITCH OUTPUT_FEATURE_PITCH * INDICES_SIZE
    #elif AXIS_VALUE == AXIS_W
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_W_PITCH OUTPUT_W_PITCH
        #define UPDATES_F_PITCH OUTPUT_W_PITCH * INDICES_SIZE
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #elif AXIS_VALUE == AXIS_Z
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Z_PITCH
        #define UPDATES_W_PITCH OUTPUT_Z_PITCH * INDICES_SIZE
        #define UPDATES_F_PITCH UPDATES_W_PITCH * OUTPUT_SIZE_W
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #elif AXIS_VALUE == AXIS_Y
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_Y_PITCH
        #define UPDATES_Z_PITCH OUTPUT_Y_PITCH  * INDICES_SIZE
        #define UPDATES_W_PITCH UPDATES_Z_PITCH * OUTPUT_SIZE_Z
        #define UPDATES_F_PITCH UPDATES_W_PITCH * OUTPUT_SIZE_W
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #elif AXIS_VALUE == AXIS_X
        #define UPDATES_X_PITCH OUTPUT_X_PITCH
        #define UPDATES_Y_PITCH OUTPUT_X_PITCH * INDICES_SIZE
        #define UPDATES_Z_PITCH UPDATES_Y_PITCH * OUTPUT_SIZE_Y
        #define UPDATES_W_PITCH UPDATES_Z_PITCH * OUTPUT_SIZE_Z
        #define UPDATES_F_PITCH UPDATES_W_PITCH * OUTPUT_SIZE_W
        #define UPDATES_B_PITCH UPDATES_F_PITCH * OUTPUT_FEATURE_NUM
    #endif
    #define GET_UPDATES_INDEX(idx_order) \
         (INPUT2_OFFSET) + \
         (x)*(UPDATES_X_PITCH) + \
         (y)*(UPDATES_Y_PITCH) + \
         (z)*(UPDATES_Z_PITCH) + \
         (w)*(UPDATES_W_PITCH) + \
         (f)*(UPDATES_F_PITCH) + \
         (b)*(UPDATES_B_PITCH)
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

#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef AXIS_B
#undef AXIS_F
#undef AXIS_W
#undef AXIS_Z
#undef AXIS_Y
#undef AXIS_X
