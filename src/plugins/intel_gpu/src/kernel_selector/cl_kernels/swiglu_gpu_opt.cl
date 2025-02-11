// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

float FUNC(fast_erf)(float x) {
    // If x is very big just straight up assume the result is +-1.0f
    if(x > 4.0f) return 1.0f;
    if(x < -4.0f) return -1.0f;

    float z = fabs(x);
    // Use Taylor expansion when x is close to 0
    if(z < 0.44593f) { // Cutoff where Taylor expansion is more precise than A&S formula
        // Taylor expansion with 5 terms
        const float a1 = 1.1283791670955126f;
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x5 = x3 * x2;
        const float x7 = x5 * x2;
        const float x9 = x7 * x2;

        return a1 * (x - (x3 / 3.0f) + (x5 / 10.0f) - (x7 / 42.0f) + (x9 / 216.0f));
    }
    // A&S formula has a high relative error when x is close to 0
    else {
        // From A&S Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
        // Formula section 7.1.26

        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        const float p = 0.3275911f;

        int sign = (x >= 0) ? 1 : -1;

        float t = 1.0f / (1.0f + p * z);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * native_exp(-(z * z));

        return y * sign; // A&S formula is only good for when x >= 0, however -erf(x) = erf(-x)
    }
}

#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
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
        gate = (GEGLU_HALF * gate * (ACCUMULATOR_VAL_ONE + (FUNC_CALL(fast_erf)(gate * GEGLU_MULT))));
    #elif GLU_TYPE == 2 // Gelu_Tanh
        gate = (GEGLU_HALF * gate * (ACCUMULATOR_VAL_ONE + (tanh(GEGLU_SQUARE_2_OVER_PI * gate * (ACCUMULATOR_VAL_ONE + GEGLU_MULT * gate * gate)))));
    #endif

    value *= gate;
    output[x] = TO_OUTPUT_TYPE(value);
}
