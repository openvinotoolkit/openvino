// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if RESAMPLE_PILLOW_STAGE == STAGE_CALC_HORIZONTAL_COEFFICIENTS || RESAMPLE_PILLOW_STAGE == STAGE_CALC_VERTICAL_COEFFICIENTS

#if ENABLE_BILINEAR_PILLOW_MODE == 1
#define PILLOW_SUPPORT 1.f
inline float FUNC(filter)(float x) {
    if (x < 0.f) {
        x = -x;
    }
    if (x < 1.f) {
        return 1.f - x;
    }
    return 0.f;
}
#elif ENABLE_BICUBIC_PILLOW_MODE == 1
#define PILLOW_SUPPORT 2.f
inline float FUNC(filter)(float x) {
    if (x < 0.f) {
        x = -x;
    }
    if (x < 1.f) {
        return ((CUBE_COEFF + 2.f) * x - (CUBE_COEFF + 3.f)) * x * x + 1.f;
    }
    if (x < 2.f) {
        return (((x - 5.f) * x + 8.f) * x - 4.f) * CUBE_COEFF;
    }
    return 0.f;
}
#endif

KERNEL (calculate_coefficients_gpu_ref)(__global float* coefficients, __global int* bounds)
{
    const int xx = get_global_id(0);
    if (xx >= OUT_DIM_SIZE)
        return;
    float center = IN_DIM_BEGIN + (xx + 0.5f) * SCALE;
    float ww = 0.f;
    float ss = 1.f / FILTER_SCALE;
    // Round the value
    int xmin = (int)(center - SUPPORT + 0.5f);
    if (xmin < 0) {
        xmin = 0;
    }
    // Round the value
    int xmax = (int)(center + SUPPORT + 0.5f);
    if (xmax > IN_DIM_SIZE) {
        xmax = IN_DIM_SIZE;
    }
    xmax -= xmin;
    float* k = coefficients + xx * KSIZE;
    int x = 0;
    for (; x < xmax; x++) {
        float w = FUNC_CALL(filter)((x + xmin - center + 0.5f) * ss);
        k[x] = w;
        ww += w;
    }
    for (x = 0; x < xmax; x++) {
        if (ww != 0.0) {
            k[x] /= ww;
        }
    }
    // Remaining values should stay empty if they are used despite of xmax.
    for (; x < KSIZE; x++) {
        k[x] = 0.f;
    }
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
}

#elif RESAMPLE_PILLOW_STAGE == STAGE_RESAMPLE_HORIZONTAL

KERNEL (resample_horizontal_gpu_ref)(  __global INPUT0_TYPE* input
                                     , __global float* coefficients
                                     , __global int* bounds
                                     , __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                                     , FUSED_OPS_DECLS
#endif
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
#if ENABLE_VERTICAL_PASS
    const int f = get_global_id(2) % INTERMEDIATE_BUF_FEATURE_NUM;
    const int b = get_global_id(2) / INTERMEDIATE_BUF_FEATURE_NUM;
    if (b >=  INTERMEDIATE_BUF_BATCH_NUM || f >= INTERMEDIATE_BUF_FEATURE_NUM ||
        y >= INTERMEDIATE_BUF_SIZE_Y || x >= INTERMEDIATE_BUF_SIZE_X)
        return;
#else
    const int f = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const int b = get_global_id(2) / OUTPUT_FEATURE_NUM;
    if (b >=  OUTPUT_BATCH_NUM || f >= OUTPUT_FEATURE_NUM || y >= OUTPUT_SIZE_Y || x >= OUTPUT_SIZE_X)
        return;
#endif
    int horizontal_min, horizontal_max;
    float* k;
    OUTPUT_TYPE ss = 0.f;
#if BATCH_IS_HORIZONTAL_AXIS == 1
    horizontal_min = bounds[b * 2 + 0];
    horizontal_max = bounds[b * 2 + 1];
    k = &coefficients[b * KSIZE];
#elif FEATURE_IS_HORIZONTAL_AXIS == 1
    horizontal_min = bounds[f * 2 + 0];
    horizontal_max = bounds[f * 2 + 1];
    k = &coefficients[f * KSIZE];
#elif Y_IS_HORIZONTAL_AXIS == 1
    horizontal_min = bounds[y * 2 + 0];
    horizontal_max = bounds[y * 2 + 1];
    k = &coefficients[y * KSIZE];
#elif X_IS_HORIZONTAL_AXIS == 1
    horizontal_min = bounds[x * 2 + 0];
    horizontal_max = bounds[x * 2 + 1];
    k = &coefficients[x * KSIZE];
#endif
    for (int horizontal_dim = 0; horizontal_dim < horizontal_max; horizontal_dim++) {
#if BATCH_IS_HORIZONTAL_AXIS == 1
        int b_no_padding = horizontal_dim + horizontal_min - BEGIN_PADDING_BATCH;
#else // BATCH_IS_HORIZONTAL_AXIS == 1
        int b_no_padding = b + BATCH_HORIZONTAL_OFFSET - BEGIN_PADDING_BATCH;
#endif // BATCH_IS_HORIZONTAL_AXIS == 1

#if FEATURE_IS_HORIZONTAL_AXIS == 1
        int f_no_padding = horizontal_dim + horizontal_min - BEGIN_PADDING_FEATURE;
#else // FEATURE_IS_HORIZONTAL_AXIS == 1
        int f_no_padding = f + FEATURE_HORIZONTAL_OFFSET - BEGIN_PADDING_FEATURE;
#endif // FEATURE_IS_HORIZONTAL_AXIS == 1

#if Y_IS_HORIZONTAL_AXIS == 1
        int y_no_padding = horizontal_dim + horizontal_min + INPUT_OFFSET - BEGIN_PADDING_Y;
#else // Y_IS_HORIZONTAL_AXIS == 1
        int y_no_padding = y + Y_HORIZONTAL_OFFSET - BEGIN_PADDING_Y;
#endif // Y_IS_HORIZONTAL_AXIS == 1

#if X_IS_HORIZONTAL_AXIS == 1
        int x_no_padding = horizontal_dim + horizontal_min - BEGIN_PADDING_X;
#else // X_IS_HORIZONTAL_AXIS == 1
        int x_no_padding = x + X_HORIZONTAL_OFFSET - BEGIN_PADDING_X;
#endif // X_IS_HORIZONTAL_AXIS == 1
        if (b_no_padding >= 0 && b_no_padding < INPUT0_BATCH_NUM &&
            f_no_padding >= 0 && f_no_padding < INPUT0_FEATURE_NUM &&
            y_no_padding >= 0 && y_no_padding < INPUT0_SIZE_Y &&
            x_no_padding >= 0 && x_no_padding < INPUT0_SIZE_X) {
            int in_idx = INPUT0_GET_INDEX(b_no_padding, f_no_padding, y_no_padding, x_no_padding);
            ss += input[in_idx] * k[horizontal_dim];
        }
    }
#if ENABLE_VERTICAL_PASS
    int out_idx = INTERMEDIATE_BUF_GET_INDEX(b, f, y, x);
#else
    int out_idx = OUTPUT_GET_INDEX(b, f, y, x);
#endif
    output[out_idx] = ss;
}

#else // RESAMPLE_PILLOW_STAGE == STAGE_RESAMPLE_VERTICAL

KERNEL (resample_vertical_gpu_ref)(  __global INPUT0_TYPE* input
                                     , __global float* coefficients
                                     , __global int* bounds
                                     , __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                          , FUSED_OPS_DECLS
#endif
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int f = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const int b = get_global_id(2) / OUTPUT_FEATURE_NUM;
    
    if (b >=  OUTPUT_BATCH_NUM || f >= OUTPUT_FEATURE_NUM || y >= OUTPUT_SIZE_Y || x >= OUTPUT_SIZE_X)
        return;
    
    int vertical_min, vertical_max;
    float* k;
#if BATCH_IS_VERTICAL_AXIS == 1
    k = &coefficients[b * KSIZE];
    vertical_min = bounds[b * 2 + 0];
    vertical_max = bounds[b * 2 + 1];
#elif FEATURE_IS_VERTICAL_AXIS == 1
    k = &coefficients[f * KSIZE];
    vertical_min = bounds[f * 2 + 0];
    vertical_max = bounds[f * 2 + 1];
#elif Y_IS_VERTICAL_AXIS == 1
    k = &coefficients[y * KSIZE];
    vertical_min = bounds[y * 2 + 0];
    vertical_max = bounds[y * 2 + 1];
#elif X_IS_VERTICAL_AXIS == 1
    k = &coefficients[x * KSIZE];
    vertical_min = bounds[x * 2 + 0];
    vertical_max = bounds[x * 2 + 1];
#endif

    OUTPUT_TYPE ss = 0.f;
    for (int vertical_dim = 0; vertical_dim < vertical_max; vertical_dim++) {
#if ENABLE_HORIZONTAL_PASS

#if BATCH_IS_VERTICAL_AXIS == 1
        int in_idx = INTERMEDIATE_BUF_GET_INDEX(vertical_dim + vertical_min, f, y, x);
#elif FEATURE_IS_VERTICAL_AXIS == 1
        int in_idx = INTERMEDIATE_BUF_GET_INDEX(b, vertical_dim + vertical_min, y, x);
#elif Y_IS_VERTICAL_AXIS == 1
        int in_idx = INTERMEDIATE_BUF_GET_INDEX(b, f, vertical_dim + vertical_min, x);
#elif X_IS_VERTICAL_AXIS == 1
        int in_idx = INTERMEDIATE_BUF_GET_INDEX(b, f, y, vertical_dim + vertical_min);
#endif
        ss += input[in_idx] * k[vertical_dim];
#else // ENABLE_HORIZONTAL_PASS

#if BATCH_IS_VERTICAL_AXIS == 1
        int b_no_padding = vertical_dim + vertical_min - BEGIN_PADDING_BATCH;
#else // BATCH_IS_VERTICAL_AXIS == 1
        int b_no_padding = b - BEGIN_PADDING_BATCH;
#endif // BATCH_IS_VERTICAL_AXIS == 1

#if FEATURE_IS_VERTICAL_AXIS == 1
        int f_no_padding = vertical_dim + vertical_min - BEGIN_PADDING_FEATURE;
#else // FEATURE_IS_VERTICAL_AXIS == 1
        int f_no_padding = f - BEGIN_PADDING_FEATURE;
#endif // FEATURE_IS_VERTICAL_AXIS == 1

#if Y_IS_VERTICAL_AXIS == 1
        int y_no_padding = vertical_dim + vertical_min - BEGIN_PADDING_Y;
#else // Y_IS_VERTICAL_AXIS == 1
        int y_no_padding = y - BEGIN_PADDING_Y;
#endif // Y_IS_VERTICAL_AXIS == 1

#if X_IS_VERTICAL_AXIS == 1
        int x_no_padding = vertical_dim + vertical_min - BEGIN_PADDING_X;
#else // X_IS_VERTICAL_AXIS == 1
        int x_no_padding = x - BEGIN_PADDING_X;
#endif // X_IS_VERTICAL_AXIS == 1
        if (b_no_padding >= 0 && b_no_padding < INPUT0_BATCH_NUM &&
            f_no_padding >= 0 && f_no_padding < INPUT0_FEATURE_NUM &&
            y_no_padding >= 0 && y_no_padding < INPUT0_SIZE_Y &&
            x_no_padding >= 0 && x_no_padding < INPUT0_SIZE_X) {
            int in_idx = INPUT0_GET_INDEX(b_no_padding, f_no_padding, y_no_padding, x_no_padding);
            ss += input[in_idx] * k[vertical_dim];
        }
#endif // ENABLE_HORIZONTAL_PASS
    }
    int out_idx = OUTPUT_GET_INDEX(b, f, y, x);
    output[out_idx] = ss;
}

#endif

#undef PILLOW_SUPPORT
