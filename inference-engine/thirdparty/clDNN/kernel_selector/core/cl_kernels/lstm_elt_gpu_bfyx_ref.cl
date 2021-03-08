// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"

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
