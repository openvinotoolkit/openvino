// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// initial_hidden_state
// initial_cell_state     
// sequence_lengths
// WR
// B
// output0  
//output1
//output2
KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* initial_hidden_state,
    const __global INPUT0_TYPE* initial_cell_state,
    const __global INPUT0_TYPE* sequence_lengths,
    const __global INPUT0_TYPE* WR,
    const __global INPUT0_TYPE* B,
    __global OUTPUT_TYPE* output0,
    __global OUTPUT_TYPE* output1,
    __global OUTPUT_TYPE* output2
)
{
}
