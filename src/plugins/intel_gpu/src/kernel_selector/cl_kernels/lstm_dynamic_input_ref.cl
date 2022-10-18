// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/acc_type.cl"

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
    const uint batch    = (uint)get_global_id(1) % INPUT0_BATCH_NUM;
    const uint dir      = (uint)get_global_id(1) / INPUT0_BATCH_NUM;
    const uint timestep = get_global_id(2);

    if(timestep > (uint)dyn_lengths[batch])
        return;

    ACCUMULATOR_TYPE dot_prod = 0;
    for(uint x = 0; x < INPUT0_SIZE_X; ++x )
    {
        const uint input_idx   = GET_DATA_INDEX(INPUT0, batch, timestep, dir, x);
        const uint weights_idx = GET_FILTER_INDEX(WEIGHTS, 0, 0, dir, y, x);
        dot_prod += (ACCUMULATOR_TYPE)(input[input_idx] * weights[weights_idx]);
    }

#if BIAS_TERM
    dot_prod += (ACCUMULATOR_TYPE)biases[GET_DATA_INDEX(BIAS, 0, 0, dir, y)];
#endif

    output[GET_DATA_INDEX(OUTPUT, batch, timestep, dir, y)] = (OUTPUT_TYPE)dot_prod;
}
