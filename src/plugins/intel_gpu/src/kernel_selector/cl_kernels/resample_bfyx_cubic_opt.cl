// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#pragma OPENCL FP_CONTRACT OFF

#ifdef RTE_OUTPUT
    #define TO_OUTPUT_TYPE(x)   CAT(CAT(convert_, OUTPUT_TYPE), _rte)(x)
#else
    #define TO_OUTPUT_TYPE(x)   CAT(convert_, OUTPUT_TYPE)(x)
#endif

inline float FUNC(ref_divide)(float numerator, float denominator)
{
    volatile float numerator_value = numerator;
    volatile float denominator_value = denominator;
    volatile float quotient = numerator_value / denominator_value;
    volatile float residual = numerator_value - quotient * denominator_value;
    return quotient + residual / denominator_value;
}

inline float FUNC(get_original_coordinate)(float num, float scale, int length_resized, int length_original)
{
    if (scale == 1.0f)
        return num;
#if defined(COORD_TRANS_MODE_HALF_PIXEL)
    return FUNC_CALL(ref_divide)(num + 0.5f, scale) - 0.5f;
#elif defined(COORD_TRANS_MODE_PYTORCH_HALF_PIXEL)
    return (length_resized > 1) ? FUNC_CALL(ref_divide)(num + 0.5f, scale) - 0.5f : 0.f;
#elif defined(COORD_TRANS_MODE_ASYMMETRIC)
    return FUNC_CALL(ref_divide)(num, scale);
#elif defined(COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN)
    return FUNC_CALL(ref_divide)(num + 0.5f, scale);
#elif defined(COORD_TRANS_MODE_ALIGN_CORNERS)
    if (length_resized == 1)
        return 0.f;
    if (num == 0.f || num == (float)(length_resized - 1))
        return num == 0.f ? 0.f : (float)(length_original - 1);
    return FUNC_CALL(ref_divide)((float)((int)num * (length_original - 1)), (float)(length_resized - 1));
#else
#error [clDNN resample_bfyx_cubic_opt.cl]: coordinate transformation mode - not supported
#endif
}

inline void FUNC(get_cubic_coeff)(float* cubic_coef, float coord, float coef)
{
    float abs_num = fabs(coord);
    float x0 = abs_num + 1.0f;
    float x1 = abs_num;
    float x2 = 1.0f - abs_num;
    float x3 = 2.0f - abs_num;
    float t0 = coef * x0;
    t0 = t0 - 5.0f * coef;
    t0 = t0 * x0;
    t0 = t0 + 8.0f * coef;
    t0 = t0 * x0;
    cubic_coef[0] = t0 - 4.0f * coef;
    float t1 = (coef + 2.0f) * x1;
    t1 = t1 - (coef + 3.0f);
    t1 = t1 * x1;
    t1 = t1 * x1;
    cubic_coef[1] = t1 + 1.0f;
    float t2 = (coef + 2.0f) * x2;
    t2 = t2 - (coef + 3.0f);
    t2 = t2 * x2;
    t2 = t2 * x2;
    cubic_coef[2] = t2 + 1.0f;
    float t3 = coef * x3;
    t3 = t3 - 5.0f * coef;
    t3 = t3 * x3;
    t3 = t3 + 8.0f * coef;
    t3 = t3 * x3;
    cubic_coef[3] = t3 - 4.0f * coef;
}

KERNEL (resample_bfyx_cubic_opt)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const int xy = get_global_id(0);
    const int out_x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int out_y = xy / X_BLOCKS;
    const int out_f = get_global_id(1);
    const int out_b = get_global_id(2);

    if (out_f >= OUTPUT_FEATURE_NUM || out_b >= OUTPUT_BATCH_NUM)
        return;

    // Compute Y coordinate mapping once (shared for all X in the block)
    const float orig_y = FUNC_CALL(get_original_coordinate)((float)out_y, SCALES[3], OUTPUT_SIZE_Y,
                                                            INPUT0_SIZE_Y + PADS_BEGIN[3] + PADS_END[3]) - PADS_BEGIN[3];
    const int iy = (int)floor(orig_y);
    const float y_frac = orig_y - (float)iy;
    float cy[4];
    FUNC_CALL(get_cubic_coeff)(cy, y_frac, CUBE_COEFF);

    // Pre-clamp Y indices
    int y_idx[4];
    unroll_for (int j = 0; j < 4; ++j) {
        y_idx[j] = clamp(iy + j - 1, -PADS_BEGIN[3], INPUT0_SIZE_Y + PADS_END[3] - 1);
    }

    unroll_for (uint bx = 0; bx < OUTPUT_X_BLOCK_SIZE; ++bx) {
        const int ox = out_x + bx;
        if (ox >= OUTPUT_SIZE_X)
            break;

        // Compute X coordinate mapping
        const float orig_x = FUNC_CALL(get_original_coordinate)((float)ox, SCALES[4], OUTPUT_SIZE_X,
                                                                INPUT0_SIZE_X + PADS_BEGIN[4] + PADS_END[4]) - PADS_BEGIN[4];
        const int ix = (int)floor(orig_x);
        const float x_frac = orig_x - (float)ix;
        float cx[4];
        FUNC_CALL(get_cubic_coeff)(cx, x_frac, CUBE_COEFF);

        // Pre-clamp X indices
        int x_idx[4];
        unroll_for (int k = 0; k < 4; ++k) {
            x_idx[k] = clamp(ix + k - 1, -PADS_BEGIN[4], INPUT0_SIZE_X + PADS_END[4] - 1);
        }

        ACCUMULATOR_TYPE interp_val = ACCUMULATOR_VAL_ZERO;

        // 4x4 bicubic interpolation with fused cy*cx coefficients
        unroll_for (int dy = 0; dy < 4; ++dy) {
            unroll_for (int dx = 0; dx < 4; ++dx) {
#if PADDING_USED == 1
                if (y_idx[dy] >= 0 && y_idx[dy] < INPUT0_SIZE_Y &&
                    x_idx[dx] >= 0 && x_idx[dx] < INPUT0_SIZE_X)
#endif
                {
                    const float term = cy[dy] * cx[dx] * (float)input[INPUT0_GET_INDEX(out_b, out_f, y_idx[dy], x_idx[dx])];
                    interp_val = interp_val + (ACCUMULATOR_TYPE)term;
                }
            }
        }

#if HAS_FUSED_OPS
        #define batch (out_b)
        #define OF_ID (out_f)
        #define oy (out_y)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        #undef batch
        #undef OF_ID
        #undef oy
#else
        OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif
        output[OUTPUT_GET_INDEX(out_b, out_f, out_y, ox)] = res;
    }
}

#undef TO_OUTPUT_TYPE
