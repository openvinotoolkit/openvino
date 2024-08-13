// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#define MAX_LENGTH 100
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)


KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* x,
    const __global INPUT1_TYPE* initial_hidden_state,
    const __global INPUT2_TYPE* initial_cell_state,
    const __global INPUT3_TYPE* W,
    const __global INPUT4_TYPE* R,
    const __global INPUT5_TYPE* B,
#ifdef SEQUENCE
    const __global INPUT6_TYPE* sequence_lengths,
    __global OUTPUT_TYPE* hidden_history,
    __global OUTPUT1_TYPE* hidden_state,
    __global OUTPUT2_TYPE* cell_state
#else
    __global OUTPUT_TYPE* hidden_state,
    __global OUTPUT1_TYPE* cell_state
#endif
)
{
    const uint b = get_global_id(1);
    const uint local_idx = get_local_id(0);
    const uint local_hidden_size = get_local_size(0);
     
    const int weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O};
    const int gate_num = 4;
    ACCUMULATOR_TYPE hidden_result[gate_num][NUM_HIDDEN_TO_DO];
    ACCUMULATOR_TYPE input_result[gate_num][NUM_HIDDEN_TO_DO];
    ACCUMULATOR_TYPE gate_output[gate_num][NUM_HIDDEN_TO_DO];
    ACCUMULATOR_TYPE temp_cell_state[NUM_HIDDEN_TO_DO];
    for(int l=0;l<HIDDEN_SIZE;l++){
        for(int k=0;k<gate_num;k++){
            gate_output[k][HIDDEN_SIZE] = 0;
        }
    }
#ifdef SEQUENCE
    const int real_seq_length = sequence_lengths[INPUT6_GET_INDEX_SAFE(b, 0, 0, 0)];
#else
    const int real_seq_length = 1;
#endif
    for(int i=0;i<real_seq_length;i++){
        for(int l=0;l<HIDDEN_SIZE;l++){
            for(int k=0;k<gate_num;k++){
                hidden_result[k][l] = 0;
                input_result[k][l] = 0;
            }
        }
        for(int k=0;k<gate_num;k++){
            for(int l=0;l<NUM_HIDDEN_TO_DO;l++) { //kernel responsible for HIDDEN_SIZE
                const uint hidden_idx = local_idx*NUM_HIDDEN_TO_DO + l;
                if (hidden_idx >= HIDDEN_SIZE) {
                    continue;
                }
                for(int j=0;j<HIDDEN_SIZE;j++) {
                    if(i==0){
                        #ifdef SEQUENCE
                            hidden_result[k][l] += initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, 0, j, 0)]*R[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                        #else
                            hidden_result[k][l] += initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, j, 0, 0)]*R[INPUT4_GET_INDEX_SAFE(hidden_idx+weight_offsets[k], j, 0, 0)];
                        #endif
                    }else{
                        #ifdef SEQUENCE
                            int prev_idx = i-1;
                            if(DIRECTION == 1){ //reverse
                                prev_idx = real_seq_length - i ;
                            }
                            hidden_result[k][l] += hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, prev_idx, j)]*R[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                        #endif
                    }
                }
                
                for(int j=0;j<INPUT_SIZE;j++) {
                    if(DIRECTION == 1){ //reverse
                        input_result[k][l] += x[INPUT0_GET_INDEX_SAFE(b, real_seq_length-1-i, j, 0)]*W[INPUT3_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                    } else {
                        #ifdef SEQUENCE
                            input_result[k][l] += x[INPUT0_GET_INDEX_SAFE(b, i, j, 0)]*W[INPUT3_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                        #else
                            input_result[k][l] += x[INPUT0_GET_INDEX_SAFE(b, j, 0, 0)]*W[INPUT3_GET_INDEX_SAFE(hidden_idx+weight_offsets[k], j, 0, 0)];
                        #endif
                    }
                }
                #ifdef SEQUENCE
                    gate_output[k][l] = hidden_result[k][l] + input_result[k][l] + TO_ACCUMULATOR_TYPE(B[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)]);
                #else
                    gate_output[k][l] = hidden_result[k][l] + input_result[k][l] + TO_ACCUMULATOR_TYPE(B[INPUT5_GET_INDEX_SAFE(hidden_idx+weight_offsets[k], 0, 0, 0)]);
                #endif
                switch(k){
                    case 0:
                    case 1:
                    case 3:
                        gate_output[k][l] = ACTIVATION_F(ACTIVATION_CLIP(TO_OUTPUT_TYPE(gate_output[k][l]), ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
                        break;
                    case 2:
                        gate_output[k][l] = ACTIVATION_G(ACTIVATION_CLIP(TO_OUTPUT_TYPE(gate_output[k][l]), ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);
                        break;
                    default:
                        break;
                }
            }
        }
         for(int l=0;l<NUM_HIDDEN_TO_DO;l++) { //kernel responsible for HIDDEN_SIZE
            const uint hidden_idx = local_idx*NUM_HIDDEN_TO_DO + l;
            if (hidden_idx >= HIDDEN_SIZE) {
                continue;
            }
            if (i==0){
                #ifdef SEQUENCE
                    temp_cell_state[l] = gate_output[0][l]*initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] + gate_output[1][l]*gate_output[2][l];
                #else
                    temp_cell_state[l] = gate_output[0][l]*initial_cell_state[INPUT2_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] + gate_output[1][l]*gate_output[2][l];
                #endif
            }else{
                temp_cell_state[l] *= gate_output[0][l];
                temp_cell_state[l] += gate_output[1][l]*gate_output[2][l];
            }
            int cur_history_idx = i;
            if(DIRECTION == 1){ //reverse
                cur_history_idx = real_seq_length - 1 - i ;
            }
        #ifdef SEQUENCE
            hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = gate_output[3][l]*ACTIVATION_H(temp_cell_state[l], ACTIVATION_PARAMS_H);
        #else
            hidden_state[OUTPUT_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] = gate_output[3][l]*ACTIVATION_H(temp_cell_state[l], ACTIVATION_PARAMS_H);
        #endif
        #ifdef SEQUENCE
            barrier(CLK_LOCAL_MEM_FENCE);
            hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, cur_history_idx, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
        #endif
            barrier(CLK_LOCAL_MEM_FENCE);
            if(i==real_seq_length-1){
                #ifdef SEQUENCE
                    cell_state[OUTPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = temp_cell_state[l];
                #else
                    cell_state[OUTPUT1_GET_INDEX_SAFE(b, hidden_idx, 0, 0)] = temp_cell_state[l];
                #endif
            }
        }
    }   
}
