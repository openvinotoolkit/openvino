// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// input = [ batch_size, seq_length, input_size ]
// cell     = [ batch_size, num_directions, hidden_size ] optional
// output   = [ batch_size, num_directions, seq_len, hidden_size ] output
KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if CELL_TERM
    ,const __global OUTPUT_TYPE* cell
#endif
    )
{
}
