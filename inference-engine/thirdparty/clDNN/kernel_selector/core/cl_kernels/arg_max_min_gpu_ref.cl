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


#include "include/common.cl"
#include "include/data_types.cl"
    
#define GLOBAL_SIZE 128
#define LOCAL_SIZE GLOBAL_SIZE

typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    UNIT_TYPE value; 
} iav_type;

#ifdef MAX_OUT
    #define COMPARE_SIGN <
    #define UNIT_FILL_VAL UNIT_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define UNIT_FILL_VAL UNIT_VAL_MAX    
#endif

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
KERNEL(arg_max_gpu_top_k)(const __global UNIT_TYPE* input, __global float* output)
{
    uint results[TOP_K];
    __local iav_type scratch[LOCAL_SIZE];

    const uint current_batch = (uint)get_global_id(1);
    uint local_index = get_local_id(0);
#ifdef INPUT0_LAYOUT_BFYX
    const uint size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM;
    const uint batch_offset = current_batch * size;
    uint global_index = batch_offset + local_index;
#else
    const uint size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM;
    const uint fyx_size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM;
    uint global_index = current_batch + local_index*INPUT0_BATCH_NUM;
#endif

    iav_type accumulator;

    uint temp_index = global_index;

    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < TOP_K; i++){
        accumulator.index = global_index;
        accumulator.value = input[global_index];
        for (int j = 0; j < i; j++){
            if (accumulator.index % size == results[j])
                accumulator.value = UNIT_FILL_VAL;
        }
        global_index += GLOBAL_SIZE;
#ifdef INPUT0_LAYOUT_BFYX
            while (global_index < size + batch_offset) 
#else
            while (global_index < size)
#endif   
        {
            iav_type element;
            element.value = input[global_index];
            element.index = global_index;
            for (int j = 0; j < i; j++){
                if (element.index % size == results[j])
                    element.value = UNIT_FILL_VAL;
            }
            if(accumulator.value COMPARE_SIGN element.value)
            {
                accumulator.value = element.value;
                accumulator.index = element.index;
            }
#ifdef INPUT0_LAYOUT_BFYX
            global_index += GLOBAL_SIZE;
#else
            global_index += GLOBAL_SIZE * INPUT0_BATCH_NUM;
#endif
        }
        
#ifdef INPUT0_LAYOUT_BFYX
        if (local_index < size)
            scratch[local_index] = accumulator;
        else
            scratch[local_index].value = UNIT_FILL_VAL;
#else
        if (local_index < fyx_size)
            scratch[local_index] = accumulator;
        else
            scratch[local_index].value = UNIT_FILL_VAL;
#endif
        

        barrier(CLK_LOCAL_MEM_FENCE);

        __attribute__((opencl_unroll_hint))
        for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
        {
            if (local_index < offset) 
            {
                iav_type other = scratch[local_index + offset];
                iav_type mine = scratch[local_index];

                if(mine.value COMPARE_SIGN other.value)
                {
                    scratch[local_index] = other;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
#ifdef INPUT0_LAYOUT_BFYX
        if (local_index == 0) 
        {
            output[current_batch * TOP_K + i] = scratch[0].index % size;
        }
        global_index = temp_index;
        results[i] = scratch[0].index % size;
#else
        if (local_index == 0) 
        {
            output[current_batch + i*INPUT0_BATCH_NUM] = scratch[0].index / INPUT0_BATCH_NUM;
        }
        global_index = temp_index;
        results[i] = scratch[0].index;
#endif
    }
}

#undef COMPARE_SIGN
#undef UNIT_FILL_VAL