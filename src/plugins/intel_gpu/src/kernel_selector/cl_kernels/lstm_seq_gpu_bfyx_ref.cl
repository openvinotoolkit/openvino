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
    const __global INPUT0_TYPE* x,
    const __global INPUT0_TYPE* initial_hidden_state,
    const __global INPUT0_TYPE* initial_cell_state,
    const __global INPUT0_TYPE* W,
    const __global INPUT0_TYPE* R,
    const __global INPUT0_TYPE* B,
    __global OUTPUT_TYPE* output0,
    __global OUTPUT_TYPE* output1,
    __global OUTPUT_TYPE* cell_state
)
{
    const uint hidden_idx = get_global_id(0);
    const uint b = get_global_id(1);
    global ACCUMULATOR[BATCH_SIZE][HIDDEN_SIZE] hidden_result;
    global ACCUMULATOR[BATCH_SIZE][HIDDEN_SIZE] input_result;
    global ACCUMULATOR[BATCH_SIZE][HIDDEN_SIZE] forget_gate_output;
    for(int i=0;i<SEQ_LENGTH;i++){
        if( i == 0){
            for(int j=0;j<HIDDEN_SIZE;j++) {
                hidden_result[b][hidden_idx] += initial_hidden_state[INPUT1_GET_INDEX(b, hidden_idx, 0, 0)]*R[INPUT4_GET_INDEX(1, j, hidden_idx, 0)];
            }
            for(int j=0;j<INPUT_SIZE;j++) {
                input_result[b][hidden_idx] += x[INPUT0_GET_INDEX(b, hidden_idx, j)]*W[INPUT3_GET_INDEX(0, hidden_idx, j, 0)]
            }
            for(int j=0;j<HIDDEN_SIZE;j++){
                forget_gate_output[b][j] = hidden_result[b][j] + input_result[b][j] + B[INPUT5_GET_INDEX(0, hidden_idx, 0, 0)];
            }
            cell_state[OUTPUT2_GET_INDEX(b, hidden_idx, j)]
        }
    }
}
