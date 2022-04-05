// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

// tempGEMM = [ batch, 1, direction, 4 * hidden_size ]
// cell     = [ batch, 1, direction, hidden_size ] optional
// output   = [ batch, 1, direction, hidden_size ] output
KERNEL(lstm_elt)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if CELL_TERM
    ,const __global OUTPUT_TYPE* cell
#endif
    )
{
    const uint x = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE it = input[INPUT0_GET_INDEX(b, 0, 0, x + GEMM_OFFSET_I)];
    ACCUMULATOR_TYPE ot = input[INPUT0_GET_INDEX(b, 0, 0, x + GEMM_OFFSET_O)]; // pass constant offsets here
    ACCUMULATOR_TYPE zt = input[INPUT0_GET_INDEX(b, 0, 0, x + GEMM_OFFSET_Z)];

    ACCUMULATOR_TYPE val = ACTIVATION_F(ACTIVATION_CLIP(it, ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F) *
                           ACTIVATION_G(ACTIVATION_CLIP(zt, ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);

#if CELL_TERM || INPUT_FORGET
    ACCUMULATOR_TYPE ft = input[INPUT0_GET_INDEX(b, 0, 0, x + GEMM_OFFSET_F)];
#endif

#if INPUT_FORGET
    val *= ((ACCUMULATOR_TYPE)1 - ft);
#endif

#if CELL_TERM
    val += cell[CELL_GET_INDEX(b, 0, CELL_DIRECTION, x)] * ACTIVATION_F(ACTIVATION_CLIP(ft, ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
#endif
    // hidden
    output[OUTPUT_GET_INDEX(b, 0, 0, x)] = (OUTPUT_TYPE)(ACTIVATION_H(val, ACTIVATION_PARAMS_H) *
                                                         ACTIVATION_F(ACTIVATION_CLIP(ot, ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F));
    // cell
    output[OUTPUT_GET_INDEX(b, 1, 0, x)] = (OUTPUT_TYPE)val;
}
