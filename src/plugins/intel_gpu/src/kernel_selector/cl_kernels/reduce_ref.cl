// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

inline uint FUNC(calc_linear_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x)
{
    uint index = b * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U * OUTPUT_SIZE_V * OUTPUT_FEATURE_NUM +
                 f * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U * OUTPUT_SIZE_V +
                 v * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U +
                 u * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W +
                 w * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z +
                 z * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
                 y * OUTPUT_SIZE_X +
                 x;

    return index;
}

KERNEL(reduce_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint xy     = (uint)get_global_id(0);
    const uint wzuv   = (uint)get_global_id(1);
    const uint bf     = (uint)get_global_id(2);
    const uint x      = xy % OUTPUT_SIZE_X;
    const uint y      = xy / OUTPUT_SIZE_X;

    const uint b    = bf / OUTPUT_FEATURE_NUM;
    const uint f    = bf % OUTPUT_FEATURE_NUM;
#if INPUT0_DIMS == 4
    const uint w    = 0;
    const uint z    = 0;
    const uint u    = 0;
    const uint v    = 0;
#elif INPUT0_DIMS == 5
    const uint z    = wzuv % OUTPUT_SIZE_Z;
    const uint w    = 0;
    const uint u    = 0;
    const uint v    = 0;
#elif INPUT0_DIMS == 6
    const uint z    = wzuv % OUTPUT_SIZE_Z;
    const uint w    = wzuv / OUTPUT_SIZE_Z;
    const uint u    = 0;
    const uint v    = 0;
#elif INPUT0_DIMS == 7
    const uint z    = wzuv % OUTPUT_SIZE_Z;
    const uint w    = wzuv / OUTPUT_SIZE_Z % OUTPUT_SIZE_W;
    const uint u    = wzuv / (OUTPUT_SIZE_Z * OUTPUT_SIZE_W) % OUTPUT_SIZE_U;
    const uint v    = 0;
#elif INPUT0_DIMS == 8
    const uint z    = wzuv % OUTPUT_SIZE_Z;
    const uint w    = wzuv / OUTPUT_SIZE_Z % OUTPUT_SIZE_W;
    const uint u    = wzuv / (OUTPUT_SIZE_Z * OUTPUT_SIZE_W) % OUTPUT_SIZE_U;
    const uint v    = wzuv / (OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U);
#endif

#if OUTPUT_DIMS == 4
    const uint out_idx = OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    const uint out_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 6
    const uint out_idx = OUTPUT_GET_INDEX(b, f, w, z, y, x);
#elif OUTPUT_DIMS == 7
    const uint out_idx = OUTPUT_GET_INDEX(b, f, u, w, z, y, x);
#elif OUTPUT_DIMS == 8
    const uint out_idx = OUTPUT_GET_INDEX(b, f, v, u, w, z, y, x);
#endif

    const uint linear_idx = FUNC_CALL(calc_linear_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, v, u, w, z, y, x);
    if (linear_idx >= COMPUTATIONAL_OPERATIONS_NUMBER)
        return;

#ifdef REDUCE_BATCH
    const uint batch_out = 0;
    const uint batch_max_val = INPUT0_BATCH_NUM;
#else
    const uint batch_out = BATCH_NUM_IDX_COMP(linear_idx);
    const uint batch_max_val = batch_out + 1;
#endif

#ifdef REDUCE_FEATURE
    const uint feature_out = 0;
    const uint feature_max_val = INPUT0_FEATURE_NUM;
#else
    const uint feature_out = FEATURE_NUM_IDX_COMP(linear_idx);
    const uint feature_max_val = feature_out + 1;
#endif

#if INPUT0_DIMS >= 8
#ifdef REDUCE_V
    const uint v_out = 0;
    const uint v_max_val = INPUT0_SIZE_V;
#else
    const uint v_out = SIZE_V_IDX_COMP(linear_idx);
    const uint v_max_val = v_out + 1;
#endif
#else
    const uint v_out = 0;
    const uint v_max_val = 1;
#endif

#if INPUT0_DIMS >= 7
#ifdef REDUCE_U
    const uint u_out = 0;
    const uint u_max_val = INPUT0_SIZE_U;
#else
    const uint u_out = SIZE_U_IDX_COMP(linear_idx);
    const uint u_max_val = u_out + 1;
#endif
#else
    const uint u_out = 0;
    const uint u_max_val = 1;
#endif

#if INPUT0_DIMS >= 6
#ifdef REDUCE_W
    const uint w_out = 0;
    const uint w_max_val = INPUT0_SIZE_W;
#else
    const uint w_out = SIZE_W_IDX_COMP(linear_idx);
    const uint w_max_val = w_out + 1;
#endif
#else
    const uint w_out = 0;
    const uint w_max_val = 1;
#endif

#if INPUT0_DIMS >= 5
#ifdef REDUCE_Z
    const uint z_out = 0;
    const uint z_max_val = INPUT0_SIZE_Z;
#else
    const uint z_out = SIZE_Z_IDX_COMP(linear_idx);
    const uint z_max_val = z_out + 1;
#endif
#else
    const uint z_out = 0;
    const uint z_max_val = 1;
#endif

#ifdef REDUCE_Y
    const uint y_out = 0;
    const uint y_max_val = INPUT0_SIZE_Y;
#else
    const uint y_out = SIZE_Y_IDX_COMP(linear_idx);
    const uint y_max_val = y_out + 1;
#endif

#ifdef REDUCE_X
    const uint x_out = 0;
    const uint x_max_val = INPUT0_SIZE_X;
#else
    const uint x_out = SIZE_X_IDX_COMP(linear_idx);
    const uint x_max_val = x_out + 1;
#endif
    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;
    uint counter = 0;
    for (uint bi = batch_out; bi < batch_max_val; ++bi) {
        for (uint fi = feature_out; fi < feature_max_val; ++fi) {
            for (uint vi = v_out; vi < v_max_val; ++vi) {
                for (uint ui = u_out; ui < u_max_val; ++ui) {
                    for (uint wi = w_out; wi < w_max_val; ++wi) {
                        for (uint zi = z_out; zi < z_max_val; ++zi) {
                            for (uint yi = y_out; yi < y_max_val; ++yi) {
                                for (uint xi = x_out; xi < x_max_val; ++xi) {
#if INPUT0_DIMS == 8
                                    const uint input_idx = INPUT0_GET_INDEX(bi, fi, vi, ui, wi, zi, yi, xi);
#elif INPUT0_DIMS == 7
                                    const uint input_idx = INPUT0_GET_INDEX(bi, fi, ui, wi, zi, yi, xi);
#elif INPUT0_DIMS == 6
                                    const uint input_idx = INPUT0_GET_INDEX(bi, fi, wi, zi, yi, xi);
#elif INPUT0_DIMS == 5
                                    const uint input_idx = INPUT0_GET_INDEX(bi, fi, zi, yi, xi);
#else
                                    const uint input_idx = INPUT0_GET_INDEX(bi, fi, yi, xi);

#endif
#ifdef REDUCE_SUM_MODE
                                    acc += data[input_idx];
#elif REDUCE_MAX_MODE
                                    if (counter == 0)
                                        acc = data[input_idx];
                                    else
                                        acc = data[input_idx] > acc ? data[input_idx] : acc;
#elif REDUCE_MIN_MODE
                                    if (counter == 0)
                                        acc = data[input_idx];
                                    else
                                        acc = data[input_idx] < acc ? data[input_idx] : acc;
#elif REDUCE_MEAN_MODE
                                    acc += data[input_idx];
#elif REDUCE_PROD_MODE
                                    if (counter == 0)
                                        acc = data[input_idx];
                                    else
                                        acc *= data[input_idx];
#elif REDUCE_AND_MODE
                                    if (counter == 0)
                                        acc = data[input_idx];
                                    else
                                        acc = acc && data[input_idx];
#elif REDUCE_OR_MODE
                                    if (counter == 0)
                                        acc = data[input_idx];
                                    else
                                        acc = acc || data[input_idx];
#elif REDUCE_SUM_SQUARE_MODE
                                    acc += data[input_idx] * data[input_idx];
#elif REDUCE_L1_MODE
                                #if !INPUT0_IS_FP
                                    acc += TO_ACCUMULATOR_TYPE(fabs(TO_FINAL_ACCUMULATOR_TYPE(data[input_idx])));
                                #else
                                    acc += fabs(data[input_idx]);
                                #endif
#elif REDUCE_L2_MODE
                                    acc += data[input_idx] * data[input_idx];
#elif REDUCE_LOG_SUM_MODE
                                    acc += data[input_idx];
#elif REDUCE_LOG_SUM_EXP_MODE
                                #if !INPUT0_IS_FP
                                    acc += TO_ACCUMULATOR_TYPE(exp(TO_FINAL_ACCUMULATOR_TYPE(data[input_idx])));
                                #else
                                        acc += exp(data[input_idx]);
                                #endif
#endif
                                    counter++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    FINAL_ACCUMULATOR_TYPE final_acc = TO_FINAL_ACCUMULATOR_TYPE(acc);
#if REDUCE_MEAN_MODE
    if (counter != 0) final_acc /= counter;
#endif
#if REDUCE_L2_MODE
    final_acc = sqrt(final_acc);
#endif
#if REDUCE_LOG_SUM_MODE || REDUCE_LOG_SUM_EXP_MODE
    final_acc = log(final_acc);
#endif

    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE reduce_result = TO_ACTIVATION_TYPE(final_acc);
#if HAS_FUSED_OPS
    FUSED_OPS;
    final_result = FUSED_OPS_RESULT;
#else
    final_result = TO_OUTPUT_TYPE(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
#endif
    output[out_idx] = final_result;
}
