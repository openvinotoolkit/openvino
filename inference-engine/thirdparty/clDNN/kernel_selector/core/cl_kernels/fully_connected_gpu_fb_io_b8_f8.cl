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
#include "include/sub_group.cl"

__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (fully_connected_gpu_xb_xb_b8_x8)(
    const __global float* input,
    __global float* output,
    const __global float* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias)
#else
    )
#endif
{
    const uint global_id = get_global_id(0);
    const int x = get_global_id(0);
    const uint batch_id = x % INPUT0_BATCH_NUM;

    uint neuronIdx = (x / INPUT0_BATCH_NUM) * NEURONS_PER_WORK_ITEM;

    const uint sub_group_id = get_local_id(0);
    const uint batch_num = INPUT0_BATCH_NUM;

    const int out_id = (global_id / batch_num) * NEURONS_PER_WORK_ITEM * batch_num + batch_id;

    const int ofm_offset = (global_id * NEURONS_PER_WORK_ITEM) / batch_num;

    float8 _data0 = 0.f;
#if NEURONS_PER_WORK_ITEM > 8
    float8 _data1 = 0.f;
#endif

    uint weight_offset = sub_group_id + neuronIdx;

    for(uint h = 0; h < INPUT0_ELEMENTS_COUNT; h++)
    {
        DOT_PRODUCT_8(_data0, input[h * batch_num + batch_id], weight[weight_offset])
#if NEURONS_PER_WORK_ITEM > 8
        DOT_PRODUCT_8(_data1, input[h * batch_num + batch_id], weight[weight_offset + 8])
#endif
        weight_offset+= FILTER_OFM_NUM;
    }

#if BIAS_TERM
    ADD_BIAS_8(_data0, bias[neuronIdx + sub_group_id]);
#if NEURONS_PER_WORK_ITEM > 8
    ADD_BIAS_8(_data1, bias[neuronIdx + sub_group_id + 8]);
#endif
#endif
    _data0 = ACTIVATION(_data0, NL_M, NL_N);
#if NEURONS_PER_WORK_ITEM > 8
    _data1 = ACTIVATION(_data1, NL_M, NL_N);
#endif

    intel_sub_group_block_write8((__global uint*)output + out_id, as_uint8(_data0));
#if NEURONS_PER_WORK_ITEM > 8
    intel_sub_group_block_write8((__global uint*)output + out_id + 8 * batch_num, as_uint8(_data1));
#endif
}
