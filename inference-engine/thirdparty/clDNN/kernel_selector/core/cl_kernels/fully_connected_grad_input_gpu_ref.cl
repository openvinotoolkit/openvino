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

KERNEL(fully_connected_grad_input_gpu_ref)(
    const __global INPUT0_TYPE* input_grad,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights,
    const __global INPUT1_TYPE* input
    )
{
    const uint x            = get_global_id(1);
    const uint y            = get_global_id(2);
    const uint b_f          = get_global_id(0);
    const uint batch_id     = b_f % INPUT0_BATCH_NUM;
    const uint feature_id   = b_f / INPUT0_BATCH_NUM;

    if(b_f >= INPUT1_FEATURE_NUM * INPUT0_BATCH_NUM)
        return;

    ACCUMULATOR_TYPE result = 0;

    for (uint ofm = 0; ofm < FILTER_OFM_NUM; ++ofm)
    {
        const uint input_grad_idx = GET_DATA_INDEX(INPUT0, batch_id, 0, 0, ofm);
        const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, ofm, feature_id, y, x);

        result += (ACCUMULATOR_TYPE)(input_grad[input_grad_idx] * weights[filter_idx]);
    }

    const uint output_idx = GET_DATA_INDEX(OUTPUT, batch_id, feature_id, y, x);
    output[output_idx] = result;
}