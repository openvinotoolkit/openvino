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


#include "include/include_all.cl"
    
#define GLOBAL_SIZE 128
#define LOCAL_SIZE GLOBAL_SIZE

typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    UNIT_TYPE value; 
} iav_type;

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

#ifdef MAX_OUT
    #define COMPARE_SIGN <
    #define UNIT_FILL_VAL UNIT_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define UNIT_FILL_VAL UNIT_VAL_MAX    
#endif

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
KERNEL(arg_max_gpu_axis)(const __global UNIT_TYPE* input, __global float* output)
{
    uint results[TOP_K];
    __local iav_type scratch[LOCAL_SIZE];
    const uint first_dim_id = (uint)get_global_id(1);
    const uint second_third_dim_id = (uint)get_global_id(2);
    const uint second_dim_id = second_third_dim_id % SECOND_DIM_SIZE;
    const uint third_dim_id = second_third_dim_id / SECOND_DIM_SIZE;
    const uint output_index = (first_dim_id + second_dim_id * FIRST_DIM_SIZE + third_dim_id * FIRST_DIM_SIZE * SECOND_DIM_SIZE) * TOP_K;
    const uint offset = first_dim_id * FIRST_DIM_MUL + second_dim_id * SECOND_DIM_MUL + third_dim_id * THIRD_DIM_MUL;
    uint local_index = get_local_id(0);
    uint global_index = offset + local_index * GAP_SIZE;

    iav_type accumulator;

    uint temp_index = global_index;
    uint start_index = (global_index - offset) / GAP_SIZE;
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < TOP_K; i++)
    {
        accumulator.index = start_index;
        accumulator.value = input[global_index];
        for (int j = 0; j < i; j++)
        {
            if (accumulator.index == results[j])
                accumulator.value = UNIT_FILL_VAL;
        }
        global_index += GLOBAL_SIZE * GAP_SIZE;
        uint element_index = start_index + GLOBAL_SIZE;
        while (global_index < offset + VALUES_NUM * GAP_SIZE) 
        {
            iav_type element;
            element.value = input[global_index];
            element.index = element_index;
            for (int j = 0; j < i; j++){
                if (element.index == results[j])
                    element.value = UNIT_FILL_VAL;
            }
            if(accumulator.value COMPARE_SIGN element.value)
            {
                accumulator.value = element.value;
                accumulator.index = element.index;
            }
            element_index += GLOBAL_SIZE;
            global_index += GLOBAL_SIZE * GAP_SIZE;
        }
        if (local_index < VALUES_NUM)
            scratch[local_index] = accumulator;
        else
            scratch[local_index].value = UNIT_FILL_VAL;

        barrier(CLK_LOCAL_MEM_FENCE);

        __attribute__((opencl_unroll_hint))
        for(uint scratch_offset = LOCAL_SIZE / 2; scratch_offset > 0; scratch_offset /= 2) 
        {
            if (local_index < scratch_offset) 
            {
                iav_type other = scratch[local_index + scratch_offset];
                iav_type mine = scratch[local_index];

                if(mine.value COMPARE_SIGN other.value)
                {
                    scratch[local_index] = other;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_index == 0) 
        {
            output[output_index + i] = scratch[0].index;
        }
        global_index = temp_index;
        results[i] = scratch[0].index;
    }
}

#undef COMPARE_SIGN
#undef UNIT_FILL_VAL
#undef GAP_SIZE
#undef VALUES_NUM
#undef FIRST_DIM_SIZE
#undef SECOND_DIM_SIZE
#undef FIRST_DIM_MUL
#undef SECOND_DIM_MUL
#undef THIRD_DIM_MUL