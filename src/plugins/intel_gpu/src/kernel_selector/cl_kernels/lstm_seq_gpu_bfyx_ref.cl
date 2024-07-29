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
    for(int i=0;i<MAX_SEQ_LENGTH;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            //printf("x is %f\n", x[INPUT0_GET_INDEX_SAFE(b, i, j, 0)]);
        }
    }
    //printf("initial hidden state is %f\n", initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)]);
    //printf("initial cell state is %f\n", initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)]);
    //printf("seq lentghts are %d \n", sequence_lengths[INPUT3_GET_INDEX_SAFE(b, 0, 0, 0)]);
    //printf("INPUT SIZE is %d \n", INPUT_SIZE);
    for(int i=0;i<INPUT_SIZE;i++){
        for(int j=0;j<4;j++){
            //printf("j is %d hidden_idx is %d HIDDEN_SIZE is %d idx is %d  hidden_idx+j*HIDDEN_SIZE is %d\n", j, hidden_idx, HIDDEN_SIZE, INPUT4_GET_INDEX_SAFE(0, hidden_idx+j*HIDDEN_SIZE, i, 0), W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+j*HIDDEN_SIZE, i, 0)], hidden_idx+j*HIDDEN_SIZE);
            //W[INPUT4_GET_INDEX_SAFE(hidden_idx+j*HIDDEN_SIZE, 0, i, 0)], hidden_idx+j*HIDDEN_SIZE);
            //printf("W are %f for idx %d %d\n", W[INPUT4_GET_INDEX_SAFE(0, i, hidden_idx+j*HIDDEN_SIZE, 0)], hidden_idx+j*HIDDEN_SIZE, i);
            //printf("oj W are %f for idx %d %d\n", W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+j*HIDDEN_SIZE, i, 0)], hidden_idx+j*HIDDEN_SIZE, i);
        }
    }
    //printf("input0 pitches %d, %d, %d, %d \n", INPUT0_PITCHES[0], INPUT0_PITCHES[1], INPUT0_PITCHES[2], INPUT0_PITCHES[3]);
    //printf("input1 pitches %d, %d, %d, %d \n", INPUT1_PITCHES[0], INPUT1_PITCHES[1], INPUT1_PITCHES[2], INPUT1_PITCHES[3]);
    //printf("input2 pitches %d, %d, %d, %d \n", INPUT2_PITCHES[0], INPUT2_PITCHES[1], INPUT2_PITCHES[2], INPUT2_PITCHES[3]);
    //printf("input3 pitches %d, %d, %d, %d \n", INPUT3_PITCHES[0], INPUT3_PITCHES[1], INPUT3_PITCHES[2], INPUT3_PITCHES[3]);
    //printf("input4 pitches %d, %d, %d, %d \n", INPUT4_PITCHES[0], INPUT4_PITCHES[1], INPUT4_PITCHES[2], INPUT4_PITCHES[3]);
    //printf("input5 pitches %d, %d, %d, %d \n", INPUT5_PITCHES[0], INPUT5_PITCHES[1], INPUT5_PITCHES[2], INPUT5_PITCHES[3]);
    //printf("input6 pitches %d, %d, %d, %d \n", INPUT6_PITCHES[0], INPUT6_PITCHES[1], INPUT6_PITCHES[2], INPUT6_PITCHES[3]);
    //printf("output pitches %d, %d, %d, %d \n", OUTPUT_PITCHES[0], OUTPUT_PITCHES[1], OUTPUT_PITCHES[2], OUTPUT_PITCHES[3]);
    //printf("output1 pitches %d, %d, %d, %d \n", OUTPUT1_PITCHES[0], OUTPUT1_PITCHES[1], OUTPUT1_PITCHES[2], OUTPUT1_PITCHES[3]);
    //printf("output2 pitches %d, %d, %d, %d \n", OUTPUT2_PITCHES[0], OUTPUT2_PITCHES[1], OUTPUT2_PITCHES[2], OUTPUT2_PITCHES[3]);
    for(int i=0;i<HIDDEN_SIZE;i++){
        for(int j=0;j<4;j++){
            //printf("R are %f \n", R[INPUT5_GET_INDEX_SAFE(0,hidden_idx+j*HIDDEN_SIZE, 0, i)]);
        }
    }
    for(int j=0;j<4;j++){
        //printf("B are %f \n", B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+j*HIDDEN_SIZE, 0, 0)]);
    }
    const int weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O};
    const uint hidden_idxl = get_local_id(0);
    const uint bl = get_local_id(1);
    const int gate_num = 4;
    float hidden_result[gate_num];
    float input_result[BATCH_SIZE][HIDDEN_SIZE][gate_num];
    float gate_output[BATCH_SIZE][HIDDEN_SIZE][gate_num];
    for(int i=0;i<BATCH_SIZE;i++){
        for(int j=0;j<HIDDEN_SIZE;j++){
            for(int k=0;k<gate_num;k++){
                hidden_result[k] = 0;
                input_result[i][j][k] = 0;
                gate_output[i][j][k] = 0;
            }
        }
    }
    /*
    for(int i=0;i<4;i++) {
        printf("HIDDEN_SIZE is %d weight is %d MAX_SEQ_LENGTH %d INPUT_SIZE %d\n", HIDDEN_SIZE, weight_offsets[i], MAX_SEQ_LENGTH, INPUT_SIZE);
    }
    printf("offsets %d %d %d %d \n", GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O);

    printf("kernel usage %d seq len\n", MAX_SEQ_LENGTH);
    */
    for(int i=0;i<MAX_SEQ_LENGTH;i++){
        for(int k=0;k<gate_num;k++){
            hidden_result[k] = 0;
            //printf("begin hidden result %f for b%d i %d k %d\n", hidden_result[k], b, i, k);
            input_result[b][hidden_idx][k] = 0;
        }
        for(int k=0;k<gate_num;k++){
            for(int j=0;j<HIDDEN_SIZE;j++) {
                if(i==0){
                    hidden_result[k] += initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k],  j, 0)];
                    
                    //printf("mult %f %f \n", initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)], R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+GEMM_OFFSET_F,  0, 0)]);
                }else{
                    hidden_result[k] += hidden_state[OUTPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                    //printf("hidden_result[k]   %f %f\n",hidden_state[INPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)], R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)]);
                }
            }
            
            for(int j=0;j<INPUT_SIZE;j++) {
                input_result[b][hidden_idx][k] += x[INPUT0_GET_INDEX_SAFE(b, i, j, 0)]*W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                //printf("input_result[b][hidden_idx][k] %f %f\n", x[INPUT0_GET_INDEX_SAFE(b, i, j, 0)], W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)]);
            }
            for(int j=0;j<HIDDEN_SIZE;j++){
                gate_output[b][hidden_idx][k] = hidden_result[k] + input_result[b][j][k] + B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)];
                //printf("gate_output[b][hidden_idx][k] %f %f %f for b %d and k %d\n", hidden_result[k], input_result[b][j][k], B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)], b, k);
            }
            switch(k){
                case 0:
                case 1:
                case 3:
                    gate_output[b][hidden_idx][k] = ACTIVATION_F(ACTIVATION_CLIP(gate_output[b][hidden_idx][k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
                    //printf("03 gate output is %f\n", gate_output[b][hidden_idx][k]);
                    break;
                case 2:
                    gate_output[b][hidden_idx][k] = ACTIVATION_G(ACTIVATION_CLIP(gate_output[b][hidden_idx][k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);
                    //printf("2 gate output is %f\n", gate_output[b][hidden_idx][k]);
                    break;
                default:
                    break;
            }
        }

        if (i==0){
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = gate_output[b][hidden_idx][0]*initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, 0, 0)];
            //printf("cell_stateeq %f %f for b %d %d %d %d %d\n" , gate_output[b][hidden_idx][0], initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, 0, 0)], b, OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0), OUTPUT2_GET_INDEX_SAFE(0, b, 0, 0), OUTPUT2_GET_INDEX_SAFE(0, 0, b, 0), OUTPUT2_GET_INDEX_SAFE(0, 0, 0, b));
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] += gate_output[b][hidden_idx][1]*gate_output[b][hidden_idx][2];
            //printf("cell_stateplus %f %f OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0) %d for b %d \n" , gate_output[b][hidden_idx][1], gate_output[b][hidden_idx][2], OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0), b);
        }else{
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] *= gate_output[b][hidden_idx][0];
            //printf("cell_stateeqq is %f\n", cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)]);
            cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] += gate_output[b][hidden_idx][1]*gate_output[b][hidden_idx][2];
            //printf("cell_stateppliu is %f\n", cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] );
        }
        hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = gate_output[b][hidden_idx][3]*ACTIVATION_H(ACTIVATION_CLIP(cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_H);
        //printf("hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] is %f on b %d\n", hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)], b);
        hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
        //printf("hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)] is %f\n", hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)]);
    }
    //printf("cell state for %d is %f \n", OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0), cell_state[OUTPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)]);
    printf("R is %p B is %p ; %p out0 %p add for out1 for out2 %p batch %d\n", &R, &B, &hidden_history, &hidden_state, &cell_state, b);
}
