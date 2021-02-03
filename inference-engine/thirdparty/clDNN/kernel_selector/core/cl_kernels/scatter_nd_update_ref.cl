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
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif

#if OUTPUT_DIMS != INPUT2_DIMS
    #error "OUTPUT_DIMS is supposed to be same as INPUT2_DIMS"
#endif

KERNEL(scatter_nd_update_ref)(const __global INPUT0_TYPE* data,
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
    INPUT0_TYPE val = data[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel
    #if OUTPUT_DIMS == 4
        const uint idx_x = dim0;
        const uint idx_y = dim1;
        const uint idx_f = dim2 % INPUT2_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT2_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint idx_x = dim0 % INPUT2_SIZE_X;
        const uint idx_y = dim0 / INPUT2_SIZE_X;
        const uint idx_z = dim1;
        const uint idx_f = dim2 % INPUT2_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT2_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint idx_x = dim0 % INPUT2_SIZE_X;
        const uint idx_y = dim0 / INPUT2_SIZE_X;
        const uint idx_z = dim1 % INPUT2_SIZE_Z;
        const uint idx_w = dim1 / INPUT2_SIZE_Z;
        const uint idx_f = dim2 % INPUT2_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT2_FEATURE_NUM;
    #endif

    const uint updates_idx = GET_UPDATES_INDEX(INPUT2, IDX_ORDER);
    INPUT1_TYPE index = indices[(int)updates_idx];

    #if OUTPUT_DIMS == 4
    #if     AXIS_VALUE == 0
        const uint x = idx_x; const uint y = idx_y; const uint f = idx_f; const uint b = index;
    #elif   AXIS_VALUE == 1
        const uint x = idx_x; const uint y = idx_y; const uint f = index; const uint b = idx_b;
    #elif   AXIS_VALUE == 2
        const uint x = idx_x; const uint y = index; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 3
        const uint x = index; const uint y = idx_y; const uint f = idx_f; const uint b = idx_b;
    #endif  // AXIS_VALUE
    #elif OUTPUT_DIMS == 5
    #if     AXIS_VALUE == 0
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint f = idx_f; const uint b = index;
    #elif   AXIS_VALUE == 1
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint f = index; const uint b = idx_b;
    #elif   AXIS_VALUE == 2
        const uint x = idx_x; const uint y = idx_y; const uint z = index; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 3
        const uint x = idx_x; const uint y = index; const uint z = idx_z; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 4
        const uint x = index; const uint y = idx_y; const uint z = idx_z; const uint f = idx_f; const uint b = idx_b;
    #endif  // AXIS_VALUE
    #elif OUTPUT_DIMS == 6
    #if     AXIS_VALUE == 0
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint w = idx_w; const uint f = idx_f; const uint b = index;
    #elif   AXIS_VALUE == 1
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint w = idx_w; const uint f = index; const uint b = idx_b;
    #elif   AXIS_VALUE == 2
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint w = index; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 3
        const uint x = idx_x; const uint y = idx_y; const uint z = index; const uint w = idx_w; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 4
        const uint x = idx_x; const uint y = index; const uint z = idx_z; const uint w = idx_w; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 5
        const uint x = index; const uint y = idx_y; const uint z = idx_z; const uint w = idx_w; const uint f = idx_f; const uint b = idx_b;
    #endif  // AXIS_VALUE
    #endif
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);

    INPUT2_TYPE val = updates[(int)updates_idx];
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
#undef IDX_ORDER
#undef ORDER
