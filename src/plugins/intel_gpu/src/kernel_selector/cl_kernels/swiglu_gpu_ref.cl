// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(swiglu_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
#if OUTPUT_DIMS == 5
    uint data_idx = (uint)get_global_id(GWS_YX);
    const uint x = data_idx % OUTPUT_SIZE_X;
    data_idx = data_idx / OUTPUT_SIZE_X;
    const uint y = data_idx % OUTPUT_SIZE_Y;
    data_idx = data_idx / OUTPUT_SIZE_Y;
    const uint z = data_idx % OUTPUT_SIZE_Z;
#else // 2D spatial
    const uint x = (uint)get_global_id(GWS_YX) % OUTPUT_SIZE_X;
    const uint y = (uint)get_global_id(GWS_YX) / OUTPUT_SIZE_X;
#endif
    const uint f = (uint)get_global_id(GWS_FEATURE);
    const uint b = (uint)get_global_id(GWS_BATCH);

#if OUTPUT_DIMS == 5
    const uint output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
    #if GLU_STRIDE == 2 // alternating
        #if GATE_IDX == 0
            const uint gate_idx = OUTPUT_GET_INDEX(b, f, z, y, x) * GLU_STRIDE;
            const uint input_idx = gate_idx + 1;
        #else
            const uint input_idx = OUTPUT_GET_INDEX(b, f, z, y, x) * GLU_STRIDE;
            const uint gate_idx = input_idx + 1;
        #endif
    #else // split
        #if GATE_IDX == 0
        const uint gate_idx = INPUT0_GET_INDEX(b, f, z, y, x);
        const uint input_idx = gate_idx + GLU_STRIDE;
        #else
        const uint input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
        const uint gate_idx = input_idx + GLU_STRIDE;
        #endif
    #endif
#else // 2D spatial
    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
    #if GLU_STRIDE == 2 // alternating
        #if GATE_IDX == 0
            const uint gate_idx = (OUTPUT_GET_INDEX(b, f, y, x)) * GLU_STRIDE;
            const uint input_idx = gate_idx + 1;
        #else
            const uint input_idx = (OUTPUT_GET_INDEX(b, f, y, x)) * GLU_STRIDE;
            const uint gate_idx = input_idx + 1;
        #endif
    #else // split
        #if GATE_IDX == 0
            const uint gate_idx = INPUT0_GET_INDEX(b, f, y, x);
            const uint input_idx = gate_idx + GLU_STRIDE;
        #else
            const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);
            const uint gate_idx = input_idx + GLU_STRIDE;
        #endif
    #endif
#endif
    ACCUMULATOR_TYPE gate = (ACCUMULATOR_TYPE) input[gate_idx];
    ACCUMULATOR_TYPE up = (ACCUMULATOR_TYPE) input[input_idx];
    #if GLU_TYPE == 0   // Swish
        #if defined(CLAMP_MIN) && defined(CLAMP_MAX)
        gate = ACCUMULATOR_MIN_FUNC(TO_OUTPUT_TYPE(CLAMP_MAX), gate);
        up = ACCUMULATOR_MAX_FUNC(TO_OUTPUT_TYPE(CLAMP_MIN), ACCUMULATOR_MIN_FUNC(up, TO_OUTPUT_TYPE(CLAMP_MAX)));
        #endif
        gate /= (ACCUMULATOR_VAL_ONE + exp(-SWISH_BETA * gate));
    #elif GLU_TYPE == 1 // Gelu
        gate = (GEGLU_HALF * gate * (ACCUMULATOR_VAL_ONE + (erf(gate * GEGLU_MULT))));
    #elif GLU_TYPE == 2 // Gelu_Tanh
        gate = (GEGLU_HALF * gate * (ACCUMULATOR_VAL_ONE + (tanh(GEGLU_SQUARE_2_OVER_PI * gate * (ACCUMULATOR_VAL_ONE + GEGLU_MULT * gate * gate)))));
    #endif
    ACCUMULATOR_TYPE res = ((ACCUMULATOR_TYPE)up + UP_ADD_VAL) * gate;

    output[output_idx] = TO_OUTPUT_TYPE(res);
}
