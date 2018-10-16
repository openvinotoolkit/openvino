// Copyright (c) 2016-2017 Intel Corporation
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

UNIT_TYPE FUNC(find_max_value)(__local UNIT_TYPE* partial_max, const int global_id, const int idx, const int batch_offset, const int data_sets_count, const __global UNIT_TYPE* input)
{
    UNIT_TYPE value = -UNIT_VAL_MAX;
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        value = max(value, input[LWS * i + global_id]);
    }
    value = max(value, global_id < LEFTOVERS? input[LWS * ITEMS_NUM + global_id] : -UNIT_VAL_MAX);
    partial_max[global_id] = value;

    barrier(CLK_LOCAL_MEM_FENCE);
    if(global_id < data_sets_count)
    {
        for(int i = 1; i < LWS / data_sets_count; i++)
        {
            partial_max[batch_offset] = max(partial_max[batch_offset], partial_max[i*data_sets_count + batch_offset]);
        };
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return partial_max[batch_offset];
}

KERNEL (softmax_gpu_continoues_yxfb)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint data_sets_count = DATA_SETS_COUNT;   //how many data sets are in the processing payload     

    const int global_id = get_global_id(0);
    const int idx = global_id / data_sets_count;

    const int batch_offset = global_id % data_sets_count;

    __local UNIT_TYPE partial_max[LWS];
    const UNIT_TYPE max_value = FUNC_CALL(find_max_value)(partial_max, global_id, idx, batch_offset, data_sets_count, input);

    UNIT_TYPE tmp_vals[ITEMS_NUM + 1];
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        tmp_vals[i] = native_exp(input[LWS * i + global_id] - max_value);
    }
    tmp_vals[ITEMS_NUM] = global_id < LEFTOVERS ? native_exp(input[LWS * ITEMS_NUM + global_id] - max_value) : UNIT_VAL_ZERO;

    // accumulate all values;
    __local UNIT_TYPE partial_acc[LWS]; // all values accumulated;
    partial_acc[global_id] = UNIT_VAL_ZERO;
    for(int i = 0; i < ITEMS_NUM + 1; i++)
    {
        partial_acc[global_id] += tmp_vals[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // we must be sure that all threads calculated max of elements(we can remove it if simd32 and GWS <= 32
    if(global_id < data_sets_count)
    {
        for(int i = 1; i < LWS/data_sets_count; i++)
        {
            partial_acc[batch_offset] += partial_acc[i*data_sets_count + batch_offset];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < ITEMS_NUM; i++)
    {
        output[LWS * i + global_id] = ACTIVATION(tmp_vals[i] / partial_acc[batch_offset], NL_M ,NL_N);
    }
    if(global_id < LEFTOVERS)
        output[LWS * ITEMS_NUM + global_id] = ACTIVATION(tmp_vals[ITEMS_NUM] / partial_acc[batch_offset], NL_M ,NL_N);
}