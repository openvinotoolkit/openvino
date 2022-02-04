// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

#define ACTIVATION_LOGISTIC(input)        (UNIT_VAL_ONE/(UNIT_VAL_ONE + exp(-input)))
#define ACTIVATION_HYPERBOLIC_TAN(input)  (tanh(input))

KERNEL(lstm_dynamic_timeloop_ref)(
    const __global INPUT0_TYPE* input,
    const __global DYN_LENGTH_TYPE* dyn_lengths,
    __global OUTPUT_TYPE* output,
    const __global RECURRENT_TYPE* recurrent
#if INIT_HIDDEN_TERM
    , const __global INIT_HIDDEN_TYPE* hidden
#endif
#if INIT_CELL_TERM
    , const __global INIT_CELL_TYPE* cell
#endif
#if LAST_HIDDEN_TERM
    , __global LAST_HIDDEN_TYPE* last_hidden
#endif
#if LAST_CELL_TERM
    , __global LAST_CELL_TYPE* last_cell
#endif
    )
{
    const uint y_offset = (uint)get_global_id(0) * ELEMENTS_TO_COUNT;
    const uint b        = get_global_id(1);
    const uint dir      = get_global_id(2);
    uint unroll_timesteps = dyn_lengths[b];

    //if hidden_size is bigger then 256, then ELEMENTS_TO_COUNT will be hidden_size/256
    ACCUMULATOR_TYPE it[ELEMENTS_TO_COUNT];
    ACCUMULATOR_TYPE ot[ELEMENTS_TO_COUNT];
    ACCUMULATOR_TYPE zt[ELEMENTS_TO_COUNT];
    ACCUMULATOR_TYPE ft[ELEMENTS_TO_COUNT];
    ACCUMULATOR_TYPE eltiwse_vals[ELEMENTS_TO_COUNT];
    ACCUMULATOR_TYPE cell_vals[ELEMENTS_TO_COUNT];
    OUTPUT_TYPE      output_value = UNIT_VAL_ZERO;
    #if INIT_HIDDEN_TERM
    bool use_hidden = true;
    #else
    bool use_hidden = false;
    #endif //hidden_term

    #if INIT_CELL_TERM
    bool use_cell = true;
    #else
    bool use_cell = false;
    #endif //cell_term

    for(int timestep = 0; timestep < MAX_SEQUENCE_LENGTH; timestep++)
    {
        //not all workitems will do computations
        if(timestep < unroll_timesteps)
        {
            for(uint element_idx = 0; element_idx < ELEMENTS_TO_COUNT; element_idx++)
            {
                const uint y = y_offset + element_idx;
                // [f, i, z, o]
                ft[element_idx] = input[GET_DATA_INDEX(INPUT0, b, timestep, dir, y + GEMM_OFFSET_F)];
                it[element_idx] = input[GET_DATA_INDEX(INPUT0, b, timestep, dir, y + GEMM_OFFSET_I)];
                zt[element_idx] = input[GET_DATA_INDEX(INPUT0, b, timestep, dir, y + GEMM_OFFSET_Z)];
                ot[element_idx] = input[GET_DATA_INDEX(INPUT0, b, timestep, dir, y + GEMM_OFFSET_O)];
                if(use_hidden)
                {
                    for(uint x = 0; x < OUTPUT_SIZE_X; ++x)
                    {
                        if(timestep == 0)
                        {
                        #if INIT_HIDDEN_TERM
                            uint hidden_idx = GET_DATA_INDEX(INIT_HIDDEN, b, 0, dir, x);
                            ft[element_idx] += (ACCUMULATOR_TYPE)(hidden[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_F, x)]);
                            it[element_idx] += (ACCUMULATOR_TYPE)(hidden[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_I, x)]);
                            zt[element_idx] += (ACCUMULATOR_TYPE)(hidden[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_Z, x)]);
                            ot[element_idx] += (ACCUMULATOR_TYPE)(hidden[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_O, x)]);
                        #endif //INIT_HIDDEN_TERM
                        }
                        else
                        {
                            uint hidden_idx = GET_DATA_INDEX(OUTPUT, b, timestep - 1, dir, x);
                            ft[element_idx] += (ACCUMULATOR_TYPE)(output[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_F, x)]);
                            it[element_idx] += (ACCUMULATOR_TYPE)(output[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_I, x)]);
                            zt[element_idx] += (ACCUMULATOR_TYPE)(output[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_Z, x)]);
                            ot[element_idx] += (ACCUMULATOR_TYPE)(output[hidden_idx] * recurrent[GET_DATA_INDEX(RECURRENT, 0, dir, y + GEMM_OFFSET_O, x)]);
                        } //else timesteo ==0
                    }//for(uint x = 0; x < OUTPUT_SIZE_X; ++x)
                }//if(use_hidden)

                //eltwise operation
                eltiwse_vals[element_idx] = ACTIVATION_LOGISTIC(CLIP(it[element_idx])) * ACTIVATION_HYPERBOLIC_TAN(CLIP(zt[element_idx]));
                #if INPUT_FORGET
                eltiwse_vals[element_idx] *= ((ACCUMULATOR_TYPE)1 - ft[element_idx]);
                #endif //INPUT_FORGET

                if(use_cell)
                {
                    if(timestep == 0)
                    {
                    #if INIT_CELL_TERM
                        eltiwse_vals[element_idx] += cell[GET_DATA_INDEX(INIT_CELL, b, 0, dir, y)] * ACTIVATION_LOGISTIC(CLIP(ft[element_idx]));
                    #endif //INIT_CELL_TERM
                    }
                    else
                    {
                        eltiwse_vals[element_idx] += cell_vals[element_idx] * ACTIVATION_LOGISTIC(CLIP(ft[element_idx]));
                    }
                }
                //end of eltwise operation
            }//for(uint cell_element = 0; cell_element < ELEMENTS_TO_COUNT; cell_element++)
        } //first if(timestep < unroll_timesteps)

        //all workitems needs to hit the barrier before writing to global output memory
        barrier(CLK_GLOBAL_MEM_FENCE);

        //not all workitems will do computations
        if(timestep < unroll_timesteps)
        {
            for(uint element_idx = 0; element_idx < ELEMENTS_TO_COUNT; element_idx++)
            {
                const uint y = y_offset + element_idx;
                output_value = (OUTPUT_TYPE)(ACTIVATION_HYPERBOLIC_TAN(eltiwse_vals[element_idx]) * ACTIVATION_LOGISTIC(ot[element_idx])); // hidden
                output[GET_DATA_INDEX(OUTPUT, b, timestep, dir, y)] = output_value;
                #if LAST_HIDDEN_TERM
                if(timestep == unroll_timesteps - 1)
                {
                    last_hidden[GET_DATA_INDEX(LAST_HIDDEN, b, 0, dir, y)] = output_value;
                }
                #endif //LAST_HIDDEN_TERM
                cell_vals[element_idx] = (OUTPUT_TYPE)eltiwse_vals[element_idx];
                #if LAST_CELL_TERM
                if(timestep == unroll_timesteps - 1)
                {
                    last_cell[GET_DATA_INDEX(LAST_CELL, b, 0, dir, y)] = cell_vals[element_idx];
                }
                #endif //LAST_CELL_TERM
                //cleanup loop
                use_hidden = true;
                use_cell = true;
                eltiwse_vals[element_idx] = UNIT_VAL_ZERO;
            }
        } //second if(timestep < unroll_timesteps)

        //all workitems needs to hit the barrier after writing to global output memory
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
