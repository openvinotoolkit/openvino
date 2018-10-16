// Copyright (c) 2018 Intel Corporation
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

#ifdef BATCH_AXIS
    #define GAP_SIZE (INPUT0_FEATURE_NUM * INPUT0_SIZE_X * INPUT0_SIZE_Y)
    #define VALUES_NUM INPUT0_BATCH_NUM
    #define FIRST_DIM_SIZE INPUT0_SIZE_X
    #define SECOND_DIM_SIZE INPUT0_SIZE_Y
    #define FIRST_DIM_MUL 1
    #define SECOND_DIM_MUL INPUT0_SIZE_X
    #define THIRD_DIM_MUL (INPUT0_SIZE_X * INPUT0_SIZE_Y)
#endif
#ifdef FEATURE_AXIS
    #define GAP_SIZE (INPUT0_SIZE_X * INPUT0_SIZE_Y)
    #define VALUES_NUM INPUT0_FEATURE_NUM
    #define FIRST_DIM_SIZE INPUT0_SIZE_X
    #define SECOND_DIM_SIZE INPUT0_SIZE_Y
    #define FIRST_DIM_MUL 1
    #define SECOND_DIM_MUL INPUT0_SIZE_X
    #define THIRD_DIM_MUL (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM)
#endif
#ifdef Y_AXIS
    #define GAP_SIZE INPUT0_SIZE_X
    #define VALUES_NUM INPUT0_SIZE_Y
    #define FIRST_DIM_SIZE INPUT0_SIZE_X
    #define SECOND_DIM_SIZE INPUT0_FEATURE_NUM
    #define FIRST_DIM_MUL 1
    #define SECOND_DIM_MUL (INPUT0_SIZE_Y * INPUT0_SIZE_X)
    #define THIRD_DIM_MUL (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM)
#endif
#ifdef X_AXIS
    #define GAP_SIZE 1
    #define VALUES_NUM INPUT0_SIZE_X
    #define FIRST_DIM_SIZE INPUT0_SIZE_Y
    #define SECOND_DIM_SIZE INPUT0_FEATURE_NUM
    #define FIRST_DIM_MUL INPUT0_SIZE_X
    #define SECOND_DIM_MUL (INPUT0_SIZE_Y * INPUT0_SIZE_X)
    #define THIRD_DIM_MUL (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM)
#endif


#include "include/include_all.cl"

KERNEL(lookup_table_axis)(const __global UNIT_TYPE* input0, const __global float* indices, __global UNIT_TYPE* output)
{
    const uint first_dim_id = (uint)get_global_id(0);
    const uint second_dim_id = (uint)get_global_id(1);
    const uint third_dim_id = (uint)get_global_id(2);
	const uint offset = first_dim_id * FIRST_DIM_MUL + second_dim_id * SECOND_DIM_MUL + third_dim_id * THIRD_DIM_MUL;
    const uint val_index = (first_dim_id + second_dim_id * FIRST_DIM_SIZE + third_dim_id * FIRST_DIM_SIZE * SECOND_DIM_SIZE) * VAL_NUM;
	for (uint i = 0; i < VAL_NUM; i++)
    {
        uint global_index = offset + (int)indices[val_index + i] * GAP_SIZE;
        output[val_index + i] = input0[global_index];
    }
}


#undef GAP_SIZE
#undef VALUES_NUM
#undef FIRST_DIM_SIZE
#undef SECOND_DIM_SIZE
#undef FIRST_DIM_MUL
#undef SECOND_DIM_MUL
#undef THIRD_DIM_MUL