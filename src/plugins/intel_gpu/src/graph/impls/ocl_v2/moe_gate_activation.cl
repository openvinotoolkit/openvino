// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Common gate activation functions shared across MoE OpenCL kernels.
// Include this file after any JIT-defined macros (GATE_ACT_GELU_ERF,
// GATE_ACT_GELU_TANH, SWISH_BETA) have been set.
//
// Activation variants (selected by compile-time macros):
//   GATE_ACT_GELU_ERF  – GeGLU-ERF:  0.5·x·(1 + erf(x/√2))
//   GATE_ACT_GELU_TANH – GeGLU-Tanh: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
//   (default)          – SwiGLU:     x·σ(SWISH_BETA·x)
//
// Note: this file is inlined by the build-time codegen into each kernel template
// that includes it. When multiple templates are compiled in the same OpenCL batch
// (via join_strings), both definitions will appear; OpenCL C inline functions are
// safe to define identically more than once in the same compilation unit.

// Default Swish beta; JIT-compiled kernels may override before including this file.
#ifndef SWISH_BETA
#    define SWISH_BETA 1.0f
#endif

// Fast ERF approximation (Abramowitz & Stegun 7.1.26).
// |error| <= 1.5e-7 for all finite x; saturates at ±1 outside [-4, 4].
inline float moe_fast_erf(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float z = fabs(x);
    float t = 1.0f / (1.0f + p * z);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * native_exp(-(z * z));
    return (x >= 0.0f) ? y : -y;
}

// Gate activation dispatcher.
inline float moe_gate_activation(float x) {
#if defined(GATE_ACT_GELU_ERF)
    return 0.5f * x * (1.0f + moe_fast_erf(x * 0.7071067811865475f));
#elif defined(GATE_ACT_GELU_TANH)
    return 0.5f * x * (1.0f + tanh(0.79788458347320556640625f * x * (1.0f + 0.044715f * x * x)));
#else
    // SwiGLU: x * sigmoid(SWISH_BETA * x)
    return x / (1.0f + native_exp(-SWISH_BETA * x));
#endif
}
