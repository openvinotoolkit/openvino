// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

#ifndef DIRECTION
#define DIRECTION 0
#endif

// input     = [    batch,  sequence,               1,      input_size ]
// weights   = [        1, direction, 4 * hidden_size,      input_size ]
// recurrent = [        1, direction, 4 * hidden_size,     hidden_size ]
// biases    = [        1,         1,       direction, 4 * hidden_size ] optional
// hidden    = [    batch, direction,               1,     hidden_size ] optional
// tempGEMM  = [    batch, direction,               1, 4 * hidden_size ] output
KERNEL(lstm_gemm)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global WEIGHTS_TYPE* weights
#if HIDDEN_TERM
    , const __global OUTPUT_TYPE* hidden,
    const __global RECURRENT_TYPE* recurrent
#endif
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
    const uint y = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE dotProd = 0;
    for(uint x = 0; x < INPUT0_SIZE_X; ++x ) {
      const uint input_idx     = GET_DATA_INDEX(INPUT0, b, 0, INPUT_DIRECTION, x);
      const uint weights_idx   = GET_DATA_INDEX(WEIGHTS, 0, DIRECTION, y, x);
      dotProd += (ACCUMULATOR_TYPE)(input[input_idx] * weights[weights_idx]);
    }

#if HIDDEN_TERM
    for(uint x = 0; x < HIDDEN_SIZE_X; ++x ) {
      const uint hidden_idx    = GET_DATA_INDEX(HIDDEN, b, 0, HIDDEN_DIRECTION, x);
      const uint recurrent_idx = GET_DATA_INDEX(RECURRENT, 0, DIRECTION, y, x);
      dotProd += (ACCUMULATOR_TYPE)(hidden[hidden_idx] * recurrent[recurrent_idx]);
    }
#endif

#if BIAS_TERM
    const uint bias_idx = GET_DATA_INDEX(BIAS, 0, 0, DIRECTION, y);
    dotProd += (ACCUMULATOR_TYPE)biases[bias_idx];
#endif
    const uint output_idx = GET_DATA_INDEX(OUTPUT, b, 0, 0, y);
    output[output_idx] = (OUTPUT_TYPE)dotProd;
}
