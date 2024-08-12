// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// From A&S Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
// Formula section 7.1.26

ACCUMULATOR_TYPE fast_erf(ACCUMULATOR_TYPE x) {
    const ACCUMULATOR_TYPE a1 = 0.254829592f;
    const ACCUMULATOR_TYPE a2 = -0.284496736f;
    const ACCUMULATOR_TYPE a3 = 1.421413741f;
    const ACCUMULATOR_TYPE a4 = -1.453152027f;
    const ACCUMULATOR_TYPE a5 = 1.061405429f;
    const ACCUMULATOR_TYPE p = 0.3275911f;

    int sign = (x >= 0) ? 1 : -1;
    x = fabs(x);

    ACCUMULATOR_TYPE t = ACCUMULATOR_VAL_ONE / (ACCUMULATOR_VAL_ONE + p * x);
    ACCUMULATOR_TYPE erf = ACCUMULATOR_VAL_ONE - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * native_exp(-(x * x));

    return erf * sign;
}

KERNEL(swiglu_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
    const unsigned int x = (uint)get_global_linear_id();
    const unsigned int y = x + ((x / SPLIT_LENGTH) * SPLIT_LENGTH);

#if SPLIT_TO_GLU_IDX == 0
    ACCUMULATOR_TYPE gate = input[y];
    ACCUMULATOR_TYPE value = input[y + SPLIT_LENGTH];
#else
    ACCUMULATOR_TYPE gate = input[y + SPLIT_LENGTH];
    ACCUMULATOR_TYPE value = input[y];
#endif

    #if GLU_TYPE == 0   // Swish
        gate /= ACCUMULATOR_VAL_ONE + native_exp(-(ACCUMULATOR_VAL_ONE * gate));
    #elif GLU_TYPE == 1 // Gelu
        gate = (GEGLU_HALF * gate * (ACCUMULATOR_VAL_ONE + (fast_erf(gate * GEGLU_MULT))));
    #elif GLU_TYPE == 2 // Gelu_Tanh
        gate = (GEGLU_HALF * gate * (ACCUMULATOR_VAL_ONE + (tanh(GEGLU_SQUARE_2_OVER_PI * gate * (ACCUMULATOR_VAL_ONE + GEGLU_MULT * gate * gate)))));
    #endif

    value *= gate;
    output[x] = TO_OUTPUT_TYPE(value);
}
