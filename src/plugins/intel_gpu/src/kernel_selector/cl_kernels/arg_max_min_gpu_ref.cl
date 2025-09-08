// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define GLOBAL_SIZE 128
#define LOCAL_SIZE GLOBAL_SIZE

#ifdef MAX_OUT
    #define COMPARE_SIGN <
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
KERNEL(arg_max_gpu_top_k)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
#include "include/arg_max_min_common.cl"
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

    unroll_for(uint i = 0; i < TOP_K; i++){
        accumulator.index = global_index;
        accumulator.value = input[global_index];
        for (int j = 0; j < i; j++){
            if (accumulator.index % size == results[j])
                accumulator.value = INPUT0_FILL_VAL;
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
                    element.value = INPUT0_FILL_VAL;
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
            scratch[local_index].value = INPUT0_FILL_VAL;
#else
        if (local_index < fyx_size)
            scratch[local_index] = accumulator;
        else
            scratch[local_index].value = INPUT0_FILL_VAL;
#endif


        barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2)
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
#undef INPUT0_FILL_VAL
