// Copyright (c) 2018-2021 Intel Corporation
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
#include "include/fetch.cl"

__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL (mvn_gpu_bfyx_opt)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);     //in processing of which data set this WI participates?
    const uint workers_per_data_set = LWS;          //how many WI participates in processing of one data set
    const uint in_data_set_idx = get_global_id(0);  //this WI's id in group of items processing single data set
    const uint data_set_size = DATA_SET_SIZE;       //how many elements are in one data set
    const uint data_sets_count = DATA_SETS_COUNT;   //how many data sets are in the processing payload

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    float my_sum = 0;
    float tmp;

    __local float lg_storage[LWS];

    //each WI reads ITEMS_NUM consecutive items from batch*feature
    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        my_sum += (float)input[my_data_offset + i * workers_per_data_set];
    }

    if (in_data_set_idx < LEFTOVERS)
    {
        my_sum += (float)input[data_set_offset + workers_per_data_set * ITEMS_NUM + in_data_set_idx];
    }

    lg_storage[in_data_set_idx] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_set_idx == 0)
    {
        for (uint i=1; i<LWS; ++i)
            my_sum += lg_storage[i];

        lg_storage[0] = my_sum / data_set_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    my_sum = lg_storage[0];

#if NORMALIZE_VARIANCE == 0
    for (uint i=0; i<ITEMS_NUM; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(my_sum);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
    if (in_data_set_idx < LEFTOVERS) {
        uint iteration_in_data_set_offset = ITEMS_NUM * workers_per_data_set;
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(my_sum);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#else
    barrier(CLK_LOCAL_MEM_FENCE);

    float my_variance = 0.f;
    //each WI reads ITEMS_NUM consecutive items from batch*feature
    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        tmp = (float)input[my_data_offset + i * workers_per_data_set];
        tmp -= my_sum;
        my_variance = fma(tmp, tmp, my_variance);
    }

    if (in_data_set_idx < LEFTOVERS)
    {
        tmp = (float)input[data_set_offset + workers_per_data_set * ITEMS_NUM + in_data_set_idx];
        tmp -= my_sum;
        my_variance = fma(tmp, tmp, my_variance);
    }

    lg_storage[in_data_set_idx] = my_variance;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_set_idx == 0)
    {
        for (uint i=1; i<LWS; ++i)
            my_variance += lg_storage[i];

        my_variance /= data_set_size;

#   if defined EPS_OUTSIDE_SQRT
        lg_storage[0] = native_powr(native_sqrt(my_variance) + (float)EPSILON, -1.f);
#   elif defined EPS_INSIDE_SQRT
        lg_storage[0] = native_powr(my_variance + (float)EPSILON, -0.5f);
#   endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    my_variance = lg_storage[0];

    for (uint i=0; i<ITEMS_NUM; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(my_sum)) * TO_ACTIVATION_TYPE(my_variance);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
    if (in_data_set_idx < LEFTOVERS) {
        uint iteration_in_data_set_offset = ITEMS_NUM * workers_per_data_set;
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(my_sum)) * TO_ACTIVATION_TYPE(my_variance);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#endif
}

