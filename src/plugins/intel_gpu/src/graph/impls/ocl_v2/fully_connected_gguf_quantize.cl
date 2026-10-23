// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Quantizes the activation rows for fully_connected_gguf_gemv: int8 per 32 values, scale amax / 127.
// Row (GGUF_XPACK bytes): xq char[K] | xd float[K / 32] | Q5_K only: xs float[K / 16], the sums per 16 (min term).
// Dispatch: global = (K / 32, M).
KERNEL(fully_connected_gguf_quantize)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input, __global char* xpack) {
    const uint g = get_global_id(0);
    const uint t = get_global_id(1);
    const __global INPUT0_TYPE* x = input + (size_t)t * GGUF_K + g * 32;
    __global char* xq = xpack + (size_t)t * GGUF_XPACK;
    __global float* xd = (__global float*)(xq + GGUF_K);

    float v[32];
    float amax = 0.0f;
    float sum[2] = {0.0f, 0.0f};
    for (int j = 0; j < 32; j++) {
        v[j] = (float)x[j];
        amax = fmax(amax, fabs(v[j]));
        sum[j / 16] += v[j];
    }
#if GGUF_Q5_K
    __global float* xs = (__global float*)(xq + GGUF_K + GGUF_K / 8);
    xs[2 * g] = sum[0];
    xs[2 * g + 1] = sum[1];
#endif
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    for (int j = 0; j < 32; j++) {
        xq[g * 32 + j] = (char)clamp(rint(v[j] * id), -127.0f, 127.0f);
    }
    xd[g] = d;
}
