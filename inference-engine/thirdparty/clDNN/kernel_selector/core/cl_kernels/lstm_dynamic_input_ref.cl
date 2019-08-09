// Copyright (c) 2019 Intel Corporation
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

KERNEL(lstm_dynamic_input_ref)(
    const __global INPUT0_TYPE* input,
    const __global DYN_LENGTH_TYPE* dyn_lengths,
    __global OUTPUT_TYPE* output,
    const __global WEIGHTS_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
    const uint y        = get_global_id(0);
    const uint batch    = get_global_id(1) % INPUT0_BATCH_NUM;
    const uint dir      = get_global_id(1) / INPUT0_BATCH_NUM;
    const uint timestep = get_global_id(2);

    if(timestep > (uint)dyn_lengths[batch])
        return;

    ACCUMULATOR_TYPE dot_prod = 0;
    for(uint x = 0; x < INPUT0_SIZE_X; ++x )
    {
        const uint input_idx   = GET_DATA_INDEX(INPUT0, batch, timestep, dir, x);
        const uint weights_idx = GET_DATA_INDEX(WEIGHTS, 0, dir, y, x);
        dot_prod += (ACCUMULATOR_TYPE)(input[input_idx] * weights[weights_idx]);
    }

#if BIAS_TERM
    dot_prod += (ACCUMULATOR_TYPE)biases[GET_DATA_INDEX(BIAS, 0, 0, dir, y)];
#endif

    output[GET_DATA_INDEX(OUTPUT, batch, timestep, dir, y)] = (OUTPUT_TYPE)dot_prod;
}
