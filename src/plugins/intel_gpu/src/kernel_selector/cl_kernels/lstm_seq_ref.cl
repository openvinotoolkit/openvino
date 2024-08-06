// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* x,
    const __global INPUT1_TYPE* initial_hidden_state,
    const __global INPUT2_TYPE* initial_cell_state,
    const __global INPUT3_TYPE* sequence_lengths,
    const __global INPUT4_TYPE* W,
    const __global INPUT5_TYPE* R,
    const __global INPUT6_TYPE* B,
    __global OUTPUT_TYPE* hidden_history,
    __global OUTPUT1_TYPE* hidden_state,
    __global OUTPUT2_TYPE* cell_state
)
{
    const uint hidden_idx = get_global_id(0);
    const uint b = get_global_id(1);
    const int weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O};
    const int gate_num = 4;
    ACCUMULATOR_TYPE hidden_result[gate_num];
    ACCUMULATOR_TYPE input_result[gate_num];
    ACCUMULATOR_TYPE gate_output[gate_num];
    ACCUMULATOR_TYPE temp_cell_state = 0;
    for(int k=0;k<gate_num;k++){
        gate_output[k] = 0;
    }

    const int real_seq_length = sequence_lengths[INPUT3_GET_INDEX_SAFE(b, 0, 0, 0)];
    for(int i=0;i<real_seq_length;i++){
        for(int k=0;k<gate_num;k++){
            hidden_result[k] = 0;
            input_result[k] = 0;
        }
        for(int k=0;k<gate_num;k++){
            for(int j=0;j<HIDDEN_SIZE;j++) {
                if(i==0){
                    hidden_result[k] += initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, 0, j, 0)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                }else{
                    int prev_idx = i-1;
                    if(DIRECTION == 1){ //reverse
                        prev_idx = real_seq_length - i ;
                    }
                    hidden_result[k] += hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, prev_idx, j)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                }
            }
            
            for(int j=0;j<INPUT_SIZE;j++) {
                if(DIRECTION == 1){ //reverse
                    input_result[k] += x[INPUT0_GET_INDEX_SAFE(b, real_seq_length-1-i, j, 0)]*W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                } else {
                    input_result[k] += x[INPUT0_GET_INDEX_SAFE(b, i, j, 0)]*W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                }
            }
            gate_output[k] = hidden_result[k] + input_result[k] + TO_ACCUMULATOR_TYPE(B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)]);
        
            switch(k){
                case 0:
                case 1:
                case 3:
                    gate_output[k] = ACTIVATION_F(ACTIVATION_CLIP(TO_OUTPUT_TYPE(gate_output[k]), ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
                    break;
                case 2:
                    gate_output[k] = ACTIVATION_G(ACTIVATION_CLIP(TO_OUTPUT_TYPE(gate_output[k]), ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);
                    break;
                default:
                    break;
            }
        }

        if (i==0){
            temp_cell_state = gate_output[0]*initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] + gate_output[1]*gate_output[2];
        }else{
            temp_cell_state *= gate_output[0];
            temp_cell_state += gate_output[1]*gate_output[2];
        }
        int cur_history_idx = i;
        if(DIRECTION == 1){ //reverse
            cur_history_idx = real_seq_length - 1 - i ;
        }
        hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state, ACTIVATION_PARAMS_H);
        barrier(CLK_LOCAL_MEM_FENCE);
        hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, cur_history_idx, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(i==real_seq_length-1){
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = temp_cell_state;
        }
    }   
}
