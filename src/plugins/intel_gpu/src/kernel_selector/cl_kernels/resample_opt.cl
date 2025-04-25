// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#if ANTIALIAS == 1
    #define TRIANGLE_COEFF(a, x) ( (a) * ACCUMULATOR_MAX_FUNC(ACCUMULATOR_VAL_ZERO, ACCUMULATOR_VAL_ONE - ACCUMULATOR_ABS_FUNC((a) * (x))))
#else
    #define TRIANGLE_COEFF(a, x) (ACCUMULATOR_MAX_FUNC(ACCUMULATOR_VAL_ZERO, ACCUMULATOR_VAL_ONE - ACCUMULATOR_ABS_FUNC(x)))
#endif

#define READ_FUNC(ptr, offset)          BLOCK_READN(INPUT0_TYPE, VEC_SIZE, ptr, offset)
#define WRITE_FUNC(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, VEC_SIZE, ptr, offset, val)

#define IN_VEC_TYPE                     MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define TO_IN_VEC_TYPE(x)               CAT(convert_, IN_VEC_TYPE)(x)
#define ACC_VEC_TYPE                    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE)
#define TO_ACC_VEC_TYPE(x)              CAT(convert_, ACC_VEC_TYPE)(x)
#define OUT_VEC_TYPE                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)

#ifdef RTE_OUTPUT
    #define TO_OUT_VEC_TYPE(x)          CAT(CAT(convert_, OUT_VEC_TYPE), _rte)(x)
    #define TO_OUTPUT_TYPE(x)           CAT(CAT(convert_, OUTPUT_TYPE), _rte)(x)
#else
    #define TO_OUT_VEC_TYPE(x)          CAT(convert_, OUT_VEC_TYPE)(x)
#endif

inline float FUNC(get_original_coordinate)(float num, float scale, int length_resized, int length_original)
{
    if (scale == 1.0f)
        return num;
#if defined(COORD_TRANS_MODE_HALF_PIXEL)
    return (num + 0.5f) * scale - 0.5f;
#elif defined(COORD_TRANS_MODE_PYTORCH_HALF_PIXEL)
    return (length_resized > 1) ? (num + 0.5f) * scale - 0.5f : 0.f;
#elif defined(COORD_TRANS_MODE_ASYMMETRIC)
    return num * scale;
#elif defined(COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN)
    return (num + 0.5f) * scale;
#elif defined(COORD_TRANS_MODE_ALIGN_CORNERS)
    return (length_resized != 1) ? num * (length_original - 1) / (length_resized - 1) : 0.f;
#else
#error [clDNN resample_opt.cl]: coordinate transformation mode - not supported
#endif
}

#ifdef SAMPLE_TYPE_CAFFE_INTERP
KERNEL (resample_opt)(__global INPUT0_TYPE* input,
                      __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                      , FUSED_OPS_DECLS
#endif
)
{
    const int in_size[4] = { INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X };
    const int out_size[4] = { OUTPUT_BATCH_NUM, OUTPUT_FEATURE_NUM, OUTPUT_SIZE_Y, OUTPUT_SIZE_X };

    const int ox = (int)get_global_id(0) % OUTPUT_SIZE_X;
    const int oy = (int)get_global_id(0) / OUTPUT_SIZE_X;
    const int feature_block_num = get_global_id(1);
    const int feature = feature_block_num * FEATURE_BLOCK_SIZE;

#if OUTPUT_DIMS <= 4
    const int batch = get_global_id(2);
#else
#error [clDNN resample_ref.cl]: Unsupported data dimension
#endif

    ACCUMULATOR_TYPE i_b = AXES_USED[0] ? FUNC_CALL(get_original_coordinate)(batch, SCALES[0], out_size[0], PADDED_B) : batch;
    ACCUMULATOR_TYPE i_f = AXES_USED[1] ? FUNC_CALL(get_original_coordinate)(feature, SCALES[1], out_size[1], PADDED_F) : feature;
    ACCUMULATOR_TYPE i_y = AXES_USED[3] ? FUNC_CALL(get_original_coordinate)(oy, SCALES[3], out_size[2], PADDED_Y) : oy;
    ACCUMULATOR_TYPE i_x = AXES_USED[4] ? FUNC_CALL(get_original_coordinate)(ox, SCALES[4], out_size[3], PADDED_X) : ox;

#if PADDING_USED == 1
    i_b -= PADS_BEGIN[0];
    i_f -= PADS_BEGIN[1];
    i_y -= PADS_BEGIN[3];
    i_x -= PADS_BEGIN[4];
#endif

    const int ib_r = (int)i_b;
    const int if_r = (int)i_f;
    const int iy_r = (int)i_y;
    const int ix_r = (int)i_x;

#if ANTIALIAS == 1
    const ACCUMULATOR_TYPE ab = 1.0f / SCALES[0];
    const ACCUMULATOR_TYPE af = 1.0f / SCALES[1];
    const ACCUMULATOR_TYPE ay = 1.0f / SCALES[3];
    const ACCUMULATOR_TYPE ax = 1.0f / SCALES[4];

    const int rb = (SCALES[0] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ab);
    const int rf = (SCALES[1] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / af);
    const int ry = (SCALES[3] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ay);
    const int rx = (SCALES[4] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ax);
#else
    const ACCUMULATOR_TYPE ab = 1.0f;
    const ACCUMULATOR_TYPE af = 1.0f;
    const ACCUMULATOR_TYPE ay = 1.0f;
    const ACCUMULATOR_TYPE ax = 1.0f;

    const int rb = (SCALES[0] < 1.0f) ? 1 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ab);
    const int rf = (SCALES[1] < 1.0f) ? 1 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / af);
    const int ry = (SCALES[3] < 1.0f) ? 1 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ay);
    const int rx = (SCALES[4] < 1.0f) ? 1 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ax);
#endif

    int const b_init = max(-PADS_BEGIN[0], ib_r - rb);
    int const f_init = max(-PADS_BEGIN[1], if_r - rf);
    int const y_init = max(-PADS_BEGIN[3], iy_r - ry);
    int const x_init = max(-PADS_BEGIN[4], ix_r - rx);

    int const b_max = min(PADS_END[0] + INPUT0_BATCH_NUM, ib_r + rb + 1);
    int const f_max = min(PADS_END[1] + INPUT0_FEATURE_NUM, if_r + rf + 1);
    int const y_max = min(PADS_END[3] + INPUT0_SIZE_Y, iy_r + ry + 1);
    int const x_max = min(PADS_END[4] + INPUT0_SIZE_X, ix_r + rx + 1);

    const int fp_max = FEATURE_BLOCK_SIZE;

    ACCUMULATOR_TYPE wb = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE wf = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE wy = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE wx = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE w  = ACCUMULATOR_VAL_ZERO;

    for (int fp = 0; fp < fp_max; fp+=VEC_BLOCK_SIZE) {
        MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_BLOCK_SIZE) sum = ACCUMULATOR_VAL_ZERO;
        ACCUMULATOR_TYPE wsum = ACCUMULATOR_VAL_ZERO;

        for (int b = b_init; b < b_max; b++) {
            wb = TRIANGLE_COEFF(ab, i_b - b);

            for (int f = f_init; f < f_max; f++) {
                wf = wb * TRIANGLE_COEFF(af, i_f - f);

                if (wf != 0) {
                    for (int y = y_init; y < y_max; y++) {
                        wy = wf * TRIANGLE_COEFF(ay, i_y - y);

                        if (wy != 0) {
                            for (int x = x_init; x < x_max; x++) {
                                wx = TRIANGLE_COEFF(ax, i_x - x);
                                w = wx * wy;

#if PADDING_USED == 1
                                bool isOutOfBounds = b < 0 || f < 0 || y < 0 || x < 0 ||
                                                    b >= in_size[0] || f >= in_size[1] ||
                                                    y >= in_size[2] || x >= in_size[3];
#endif
                                if (w != 0) {
                                    wsum += w;

#if PADDING_USED == 1
                                    if (!isOutOfBounds)
#endif
                                    {
#if VEC_BLOCK_SIZE == 8
                                        MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_BLOCK_SIZE) input_vec = vload8(0, &input[INPUT0_GET_INDEX(b, f+fp, y, x)]);
                                        sum = fma(convert_float8(input_vec), (float8)w, sum);
#else
                                        MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_BLOCK_SIZE) input_vec = vload16(0, &input[INPUT0_GET_INDEX(b, f+fp, y, x)]);
                                        sum = fma(convert_float16(input_vec), (float16)w, sum);
#endif
                                    }
                                }  // w != 0;
                            }  // for (int x = x_init; x < x_max; x++)
                        }
                    }  // for (int y = y_init; y < y_max; y++)
                }
            }  // for (int f = f_init; f < f_max; f++)
        }  // for (int b = b_init; b < b_max; b++)

        MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_BLOCK_SIZE) out;
        ACCUMULATOR_TYPE res;

        if (wsum == 0) {
            res = ACCUMULATOR_VAL_ZERO;
            for  (int f = 0; f < VEC_BLOCK_SIZE; f++) {
#if HAS_FUSED_OPS
                #define OF_ID (feature+fp+f)
                FUSED_OPS;
                out[f] = FUSED_OPS_RESULT;
                #undef OF_ID
#else
                out[f] = ACTIVATION(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS);
#endif
            }
        } else {
            for  (int f = 0; f < VEC_BLOCK_SIZE; f++) {
                res = sum[f] / wsum;
#if HAS_FUSED_OPS
                #define OF_ID (feature+fp+f)
                FUSED_OPS;
                out[f] = FUSED_OPS_RESULT;
                #undef OF_ID
#else
                out[f] = ACTIVATION(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS);
#endif
            }
        }

#if VEC_BLOCK_SIZE == 8
        vstore8(out, 0, &output[OUTPUT_GET_INDEX(batch, feature+fp, oy, ox)]);
#else
        vstore16(out, 0, &output[OUTPUT_GET_INDEX(batch, feature+fp, oy, ox)]);
#endif
    } // fp
}
#endif // SAMPLE_TYPE_CAFFE_INTERP

#ifndef SAMPLE_TYPE_CAFFE_INTERP
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL (resample_opt)(__global INPUT0_TYPE* input,
                      __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                      , FUSED_OPS_DECLS
#endif
)
{
#if OUTPUT_DIMS == 5
    const int xyz = get_global_id(0);
    const int z = xyz / (OUTPUT_SIZE_Y * X_BLOCKS);
    const int xy = xyz % (OUTPUT_SIZE_Y * X_BLOCKS);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);
#else
    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);
#endif
    const int f_block = get_group_id(1);
    const int b = get_global_id(2);
    const uint feature_block = f_block * FEATURE_SLICE_SIZE;

    typedef IN_VEC_TYPE in_vec_t;
    typedef ACC_VEC_TYPE acc_vec_t;

    const int in_size[5] = { INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X };

#ifdef SAMPLE_TYPE_NEAREST
    unroll_for (uint out_x = 0; out_x < OUTPUT_X_BLOCK_SIZE; out_x++) {
        const int ix = floor((x + out_x) * SCALES[4]);
        const int iy = floor(y * SCALES[3]);

#if OUTPUT_DIMS == 5
        const int iz = floor(z * SCALES[2]);
        in_vec_t res = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, iz, iy, ix));
#else
        in_vec_t res = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, iy, ix));
#endif
#elif defined(SAMPLE_TYPE_INTERP)
    unroll_for (uint out_x = 0; out_x < OUTPUT_X_BLOCK_SIZE; out_x++) {
        const ACCUMULATOR_TYPE ix = TO_ACCUMULATOR_TYPE(SCALES[4]) * (x + out_x);
        const ACCUMULATOR_TYPE iy = TO_ACCUMULATOR_TYPE(SCALES[3]) * y;

        const int top_y_index    = (int)(floor(iy));
        const int bottom_y_index = min((int)ceil(iy), INPUT0_SIZE_Y - 1);
        const int left_x_index   = (int)(floor(ix));
        const int right_x_index  = min((int)ceil(ix), INPUT0_SIZE_X - 1);

        const ACCUMULATOR_TYPE dx = ix - left_x_index;
        const ACCUMULATOR_TYPE dy = iy - top_y_index;

#if OUTPUT_DIMS == 5
        const in_vec_t top_left     = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, top_y_index, left_x_index));
        const in_vec_t top_right    = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, top_y_index, right_x_index));
        const in_vec_t bottom_left  = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, bottom_y_index, left_x_index));
        const in_vec_t bottom_right = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, bottom_y_index, right_x_index));
#else
        const in_vec_t top_left     = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, top_y_index, left_x_index));
        const in_vec_t top_right    = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, top_y_index, right_x_index));
        const in_vec_t bottom_left  = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, bottom_y_index, left_x_index));
        const in_vec_t bottom_right = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, bottom_y_index, right_x_index));
#endif

        const acc_vec_t top    = TO_ACC_VEC_TYPE(top_left) + (TO_ACC_VEC_TYPE(top_right) - TO_ACC_VEC_TYPE(top_left)) * dx;
        const acc_vec_t bottom = TO_ACC_VEC_TYPE(bottom_left) + (TO_ACC_VEC_TYPE(bottom_right) - TO_ACC_VEC_TYPE(bottom_left)) * dx;
        acc_vec_t res = top + (bottom - top) * dy;
#else // defined(SAMPLE_TYPE_LINEAR_ONNX)
#error [clDNN resample_opt.cl]: unsupported resample type
#endif

#if HAS_FUSED_OPS
        FUSED_OPS;
        OUT_VEC_TYPE out = FUSED_OPS_RESULT;
#else
        OUT_VEC_TYPE out = TO_OUT_VEC_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
#endif // #if HAS_FUSED_OPS

#if OUTPUT_DIMS == 5
        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_block, z, y, (x + out_x)), out);
#else
        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_block, y, (x + out_x)), out);
#endif
    }
}
#endif // !SAMPLE_TYPE_CAFFE_INTERP

#undef TRIANGLE_COEFF
#undef READ_FUNC
#undef WRITE_FUNC
