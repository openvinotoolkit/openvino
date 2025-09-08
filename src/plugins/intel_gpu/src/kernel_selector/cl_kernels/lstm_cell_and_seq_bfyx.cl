// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"

#define INPUT0_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define INPUT1_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT1_TYPE, VEC_SIZE)
#define INPUT3_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT3_TYPE, VEC_SIZE)
#define INPUT4_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT4_TYPE, VEC_SIZE)
#define OUTPUT_TYPE_VEC  MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define READ_VEC(offset, ptr) CAT(vload, VEC_SIZE)(offset, ptr)

#ifdef SEQUENCE
#define GET_IN0_IDX(b, f, y) INPUT0_GET_INDEX(b, f, y, 0)
    #if DIRECTION == 2
        #define GET_IN1_IDX(b, f, y) INPUT1_GET_INDEX(b, f, y, 0)
        #define GET_IN2_IDX(b, f, y) INPUT2_GET_INDEX(b, f, y, 0)
        #define GET_IN3_IDX(b, f, y) INPUT3_GET_INDEX(b, f, y, 0)
        #define GET_IN4_IDX(b, f, y) INPUT4_GET_INDEX(b, f, y, 0)
        #define GET_IN5_IDX(b, f)    INPUT5_GET_INDEX(b, f, 0, 0)
    #else
        #define GET_IN1_IDX(b, f, y) INPUT1_GET_INDEX(b, 0, y, 0)
        #define GET_IN2_IDX(b, f, y) INPUT2_GET_INDEX(b, 0, y, 0)
        #define GET_IN3_IDX(b, f, y) INPUT3_GET_INDEX(0, f, y, 0)
        #define GET_IN4_IDX(b, f, y) INPUT4_GET_INDEX(0, f, y, 0)
        #define GET_IN5_IDX(b, f)    INPUT5_GET_INDEX(0, f, 0, 0)
    #endif
#else
#define GET_IN0_IDX(b, f, y) INPUT0_GET_INDEX(b, y, 0, 0)
#define GET_IN1_IDX(b, f, y) INPUT1_GET_INDEX(b, y, 0, 0)
#define GET_IN2_IDX(b, f, y) INPUT2_GET_INDEX(b, y, 0, 0)
#define GET_IN3_IDX(b, f, y) INPUT3_GET_INDEX(f, y, 0, 0)
#define GET_IN4_IDX(b, f, y) INPUT4_GET_INDEX(f, y, 0, 0)
#define GET_IN5_IDX(b, f)    INPUT5_GET_INDEX(f, 0, 0, 0)
#endif

KERNEL(lstm_cell_and_seq_bfyx)(
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
    const uint weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O};
    #ifdef SEQUENCE
        const uint real_seq_length = sequence_lengths[INPUT6_GET_INDEX(b, 0, 0, 0)];
    #else
        const uint real_seq_length = 1;
    #endif
    #if DIRECTION == 2
    unroll_for(uint dir=0;dir<DIRECTION;dir++) {
    #else
    uint dir = DIRECTION;
    #endif
    unroll_for(uint i=0;i<real_seq_length;++i){
        #ifdef SEQUENCE
            uint prev_idx = i-1;
            if(dir == 1) {
               prev_idx = real_seq_length - i;
            }
            if(i>0){
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        #endif
        unroll_for(uint l=0;l<NUM_HIDDEN_TO_DO;++l) { //kernel responsible for HIDDEN_SIZE
            const uint hidden_idx = local_idx*NUM_HIDDEN_TO_DO + l;
            if (hidden_idx >= HIDDEN_SIZE) {
                continue;
            }
            ACCUMULATOR_TYPE gate_output[GATE_NUM];
            unroll_for(uint k=0;k<GATE_NUM;++k){
                ACCUMULATOR_TYPE hidden_result = 0;
                ACCUMULATOR_TYPE input_result = 0;
                const uint weight_idx = hidden_idx+weight_offsets[k];
                uint hblock_num = HIDDEN_SIZE/VEC_SIZE;
                uint idx_r = GET_IN4_IDX(dir, weight_idx, 0);
                #ifdef SEQUENCE
                    #if DIRECTION == 2
                        uint idx_hidden_history = OUTPUT_GET_INDEX(b, dir, prev_idx, 0);
                    #else
                        uint idx_hidden_history = OUTPUT_GET_INDEX(b, 0, prev_idx, 0);
                    #endif
                #endif
                unroll_for(uint j=0;j<hblock_num;++j) {
                    INPUT4_TYPE_VEC r_block = READ_VEC(0, &R[idx_r]);
                    idx_r += VEC_SIZE;
                    if(i==0){
                        INPUT1_TYPE_VEC initial_block = READ_VEC(0, &initial_hidden_state[GET_IN1_IDX(b, dir, j*VEC_SIZE)]);
                        hidden_result += dot(initial_block, r_block);
                    }else{
                        #ifdef SEQUENCE
                            #if DIRECTION == 2
                                OUTPUT_TYPE_VEC h_block = READ_VEC(0, &hidden_history[idx_hidden_history]);
                            #else
                                OUTPUT_TYPE_VEC h_block = READ_VEC(0, &hidden_history[idx_hidden_history]);
                            #endif
                            idx_hidden_history += VEC_SIZE;
                            hidden_result += dot(h_block, r_block);
                        #endif
                    }
                }
                unroll_for(uint j=hblock_num*VEC_SIZE;j<HIDDEN_SIZE;++j) {
                    if(i==0){
                        #if DIRECTION == 2
                            hidden_result += initial_hidden_state[GET_IN1_IDX(b, dir, j)]*R[GET_IN4_IDX(dir, weight_idx, j)];
                        #else
                            hidden_result += initial_hidden_state[GET_IN1_IDX(b, 0, j)]*R[GET_IN4_IDX(dir, weight_idx, j)];
                        #endif
                    }else{
                        #ifdef SEQUENCE
                            #if DIRECTION == 2
                                hidden_result += hidden_history[OUTPUT_GET_INDEX(b, dir, prev_idx, j)]*R[GET_IN4_IDX(dir, weight_idx, j)];
                            #else
                                hidden_result += hidden_history[OUTPUT_GET_INDEX(b, 0, prev_idx, j)]*R[GET_IN4_IDX(dir, weight_idx, j)];
                            #endif
                        #endif
                    }
                }

                uint block_num = INPUT_SIZE/VEC_SIZE;
                uint idx_x_block;
                uint idx_w_block = GET_IN3_IDX(dir, weight_idx, 0);
                if (dir == 1) {
                    idx_x_block = GET_IN0_IDX(b, real_seq_length-1-i, 0);
                } else {
                    idx_x_block = GET_IN0_IDX(b, i, 0);
                }
                unroll_for(uint j=0;j<block_num;++j) {
                    INPUT0_TYPE_VEC x_block = READ_VEC(0, &x[idx_x_block]);
                    INPUT3_TYPE_VEC w_block = READ_VEC(0, &W[idx_w_block]);
                    idx_x_block += VEC_SIZE;
                    idx_w_block += VEC_SIZE;
                    input_result += dot(x_block, w_block);
                }

                unroll_for(uint j=block_num*VEC_SIZE;j<INPUT_SIZE;++j) { //leftovers
                    if (dir == 1) {
                        input_result += x[GET_IN0_IDX(b, real_seq_length-1-i, j)]*W[GET_IN3_IDX(dir, weight_idx, j)];
                    } else {
                        input_result += x[GET_IN0_IDX(b, i, j)]*W[GET_IN3_IDX(dir, weight_idx, j)];
                    }
                }
                gate_output[k] = hidden_result + input_result + TO_ACCUMULATOR_TYPE(B[GET_IN5_IDX(dir, weight_idx)]);
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
            ACCUMULATOR_TYPE temp_cell_state;
            if (i==0){
                temp_cell_state = gate_output[0]*initial_cell_state[GET_IN2_IDX(b, dir, hidden_idx)] + gate_output[1]*gate_output[2];
            }else{
                temp_cell_state *= gate_output[0];
                temp_cell_state += gate_output[1]*gate_output[2];
            }
            uint cur_history_idx = i;
            if (dir == 1) {  //reverse
                cur_history_idx = real_seq_length - 1 - i ;
            }
            #ifdef SEQUENCE
                #if DIRECTION == 2
                    hidden_state[OUTPUT1_GET_INDEX(b, dir, hidden_idx, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state, ACTIVATION_PARAMS_H);
                #else
                    hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state, ACTIVATION_PARAMS_H);
                #endif
            #else
                hidden_state[OUTPUT_GET_INDEX(b, hidden_idx, 0, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state, ACTIVATION_PARAMS_H);
            #endif
            #ifdef SEQUENCE
                #if DIRECTION == 2
                    hidden_history[OUTPUT_GET_INDEX(b, dir, cur_history_idx, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX(b, dir, hidden_idx, 0)];
                #else // DIRECTION == 2
                    hidden_history[OUTPUT_GET_INDEX(b, 0, cur_history_idx, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)];
                #endif
            #endif
            if(i==real_seq_length-1){
                #ifdef SEQUENCE
                    #if DIRECTION == 2
                        cell_state[OUTPUT2_GET_INDEX(b, dir, hidden_idx, 0)] = temp_cell_state;
                    #else
                        cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 0)] = temp_cell_state;
                    #endif
                #else
                    cell_state[OUTPUT1_GET_INDEX(b, hidden_idx, 0, 0)] = temp_cell_state;
                #endif
            }
        }
    }
    #if DIRECTION == 2
    }
    #endif   
}
