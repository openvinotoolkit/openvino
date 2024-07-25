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
    const uint hidden_idxl = get_local_id(0);
    const uint bl = get_local_id(1);
    const int gate_num = 4;
    __local float hidden_result[BATCH_SIZE][HIDDEN_SIZE][gate_num];
    __local float input_result[BATCH_SIZE][HIDDEN_SIZE][gate_num];
    __local float gate_output[BATCH_SIZE][HIDDEN_SIZE][gate_num];
    int weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O}; 

    for(int i=0;i<BATCH_SIZE;i++){
        for(int j=0;j<HIDDEN_SIZE;j++){
            for(int k=0;k<gate_num;k++){
                hidden_result[i][j][k] = 0;
                input_result[i][j][k] = 0;
                gate_output[i][j][k] = 0;
            }
        }
    }
    printf("kernel usage %d %d %d %d\n", hidden_idx, b, hidden_idxl, bl);
    for(int i=0;i<MAX_SEQ_LENGTH;i++){
        for(int k=0;k<gate_num;k++){            
            for(int j=0;j<HIDDEN_SIZE;j++) {
                if(i==0){
                    hidden_result[b][hidden_idx][k] += initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k],  j, 0)];
                    printf("mult %d %f %f\n", __LINE__, initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)], R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k],  j, 0)]);
                }else{
                    hidden_result[b][hidden_idx][k] += hidden_state[INPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                    printf("hidden_result[b][hidden_idx][k]   %f %f\n",hidden_state[INPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)], R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)]);
                }
            }
            
            for(int j=0;j<INPUT_SIZE;j++) {
                input_result[b][hidden_idx][k] += x[INPUT0_GET_INDEX_SAFE(b, hidden_idx, j, 0)]*W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                printf("input_result[b][hidden_idx][k] %f %f\n", x[INPUT0_GET_INDEX_SAFE(b, hidden_idx, j, 0)], W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)]);
            }
            for(int j=0;j<HIDDEN_SIZE;j++){
                gate_output[b][hidden_idx][k] = hidden_result[b][j][k] + input_result[b][j][k] + B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)];
                printf("gate_output[b][hidden_idx][k] %f %f %f\n", hidden_result[b][j][k], input_result[b][j][k], B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)]);
            }
            switch(k){
                case 0:
                case 3:
                    gate_output[b][hidden_idx][k] = ACTIVATION_F(ACTIVATION_CLIP(gate_output[b][hidden_idx][k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
                    printf("03 gate output is %f\n", gate_output[b][hidden_idx][k]);
                    break;
                case 1:
                    printf("1 input is %f\n", gate_output[b][hidden_idx][k]);
                    gate_output[b][hidden_idx][k] = ACTIVATION_G(ACTIVATION_CLIP(gate_output[b][hidden_idx][k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);
                    printf("1 gate output is %f\n", gate_output[b][hidden_idx][k]);
                    break;
                case 2:
                    gate_output[b][hidden_idx][k] = ACTIVATION_H(ACTIVATION_CLIP(gate_output[b][hidden_idx][k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_H);
                    printf("2 gate output is %f\n", gate_output[b][hidden_idx][k]);
                    break;
                default:
                    break;
            }
        }

        if (i==0){
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] = gate_output[b][hidden_idx][0]*initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
            printf("cell_stateeq %f %f\n" , gate_output[b][hidden_idx][0], initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)]);
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] += gate_output[b][hidden_idx][1]*gate_output[b][hidden_idx][2];
            printf("cell_stateplus %f %f OUTPUT2_GET_INDEX(b, hidden_idx, 0, 0) %d\n" , gate_output[b][hidden_idx][1], gate_output[b][hidden_idx][2], OUTPUT2_GET_INDEX(b, hidden_idx, 0, 0));
        }else{
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] = gate_output[b][hidden_idx][0]*cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
            printf("cell_stateeqq is %f\n", cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)]);
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] += gate_output[b][hidden_idx][1]*gate_output[b][hidden_idx][2];
            printf("cell_stateppliu is %f\n", cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] );
        }
        hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = gate_output[b][hidden_idx][3]*ACTIVATION_H(ACTIVATION_CLIP(cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 1)], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_H);
        printf("hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] is %f\n", hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)]);
        hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
        printf("hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)] is %f\n", hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)]);
    }
}
