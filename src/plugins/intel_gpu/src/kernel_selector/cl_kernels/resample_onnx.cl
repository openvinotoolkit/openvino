// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define READ_FUNC(ptr, offset)          BLOCK_READN(INPUT0_TYPE, VEC_SIZE, ptr, offset)
#define WRITE_FUNC(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, VEC_SIZE, ptr, offset, val)

#define IN_VEC_TYPE                     MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define TO_IN_VEC_TYPE(x)               CAT(convert_, IN_VEC_TYPE)(x)
#define ACC_VEC_TYPE                    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE)
#define TO_ACC_VEC_TYPE(x)              CAT(convert_, ACC_VEC_TYPE)(x)
#define OUT_VEC_TYPE                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)

#ifdef RTE_OUTPUT
    #define TO_OUT_VEC_TYPE(x)          CAT(CAT(convert_, OUT_VEC_TYPE), _rte)(x)
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
#error [clDNN resample_onnx.cl]: coordinate transformation mode - not supported
#endif
}

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL (resample_onnx)(__global INPUT0_TYPE* input,
                      __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                      , FUSED_OPS_DECLS
#endif // #if HAS_FUSED_OPS_DECLS
)
{
    const int xyz = get_global_id(0);
    const int z = xyz / (OUTPUT_SIZE_Y * X_BLOCKS);
    const int xy = xyz % (OUTPUT_SIZE_Y * X_BLOCKS);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    const int f_block = get_group_id(1);
    const int b = get_global_id(2);
    int feature_num = f_block * FEATURE_SLICE_SIZE + get_sub_group_local_id();
    const uint feature_block = f_block * FEATURE_SLICE_SIZE;

    typedef IN_VEC_TYPE in_vec_t;
    typedef ACC_VEC_TYPE acc_vec_t;

    const int in_size[5] = { INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X };

    const int PADDED_Y = INPUT0_SIZE_Y + PADS_BEGIN[3] + PADS_END[3];
    const int PADDED_X = INPUT0_SIZE_X + PADS_BEGIN[4] + PADS_END[4];
    const ACCUMULATOR_TYPE iy = FUNC_CALL(get_original_coordinate)(y, SCALES[3], OUTPUT_SIZE_Y, PADDED_Y);

    float in_y = fmax(0, fmin(iy, PADDED_Y - 1));
    int in_y1 = min((int)in_y, PADDED_Y - 1);
    int in_y2 = min(in_y1 + 1, PADDED_Y - 1);
    const ACCUMULATOR_TYPE dy1 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y1)) : 0.5f;
    const ACCUMULATOR_TYPE dy2 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y2)) : 0.5f;

#if defined (THREE_SPATIAL_RESAMPLE)

    const int PADDED_Z = INPUT0_SIZE_Z + PADS_BEGIN[2] + PADS_END[2];
    const ACCUMULATOR_TYPE iz = FUNC_CALL(get_original_coordinate)(z, SCALES[2], OUTPUT_SIZE_Z, PADDED_Z);
    float in_z = fmax(0, fmin(iz, PADDED_Z - 1));
    int in_z1 = min((int)in_z, PADDED_Z - 1);
    int in_z2 = min(in_z1 + 1, PADDED_Z - 1);
    const ACCUMULATOR_TYPE dz1 = (in_z1 != in_z2) ? TO_ACCUMULATOR_TYPE(fabs(in_z - in_z1)) : 0.5f;
    const ACCUMULATOR_TYPE dz2 = (in_z1 != in_z2) ? TO_ACCUMULATOR_TYPE(fabs(in_z - in_z2)) : 0.5f;

#if PADDING_USED == 1
    const int saved_in_z1 = in_z1;
    const int saved_in_z2 = in_z2;
    const int saved_in_y1 = in_y1;
    const int saved_in_y2 = in_y2;
#endif // PADDING_USED == 1

    unroll_for (uint out_x = 0; out_x < OUTPUT_X_BLOCK_SIZE; out_x++) {
        const ACCUMULATOR_TYPE ix = FUNC_CALL(get_original_coordinate)(x + out_x, SCALES[4], OUTPUT_SIZE_X, PADDED_X);
        float in_x = fmax(0, fmin(ix, PADDED_X - 1));
        int in_x1 = min((int)in_x, PADDED_X - 1);
        int in_x2 = min(in_x1 + 1, PADDED_X - 1);
        const ACCUMULATOR_TYPE dx1 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x1)) : 0.5f;
        const ACCUMULATOR_TYPE dx2 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x2)) : 0.5f;
#if PADDING_USED == 1
        in_z1 = saved_in_z1;
        in_z2 = saved_in_z2;
        in_y1 = saved_in_y1;
        in_y2 = saved_in_y2;

        in_z1 -= PADS_BEGIN[2];
        in_z2 -= PADS_BEGIN[2];
        in_y1 -= PADS_BEGIN[3];
        in_y2 -= PADS_BEGIN[3];
        in_x1 -= PADS_BEGIN[4];
        in_x2 -= PADS_BEGIN[4];

        bool BackTopLOutOfBounds = in_z1 < 0 || in_z1 >= INPUT0_SIZE_Z || in_y1 < 0 || in_y1 >= INPUT0_SIZE_Y || in_x1 < 0|| in_x1 >= INPUT0_SIZE_X;
        bool BackTopROutOfBounds = in_z1 < 0 || in_z1 >= INPUT0_SIZE_Z || in_y1 < 0 || in_y1 >= INPUT0_SIZE_Y || in_x2 < 0 || in_x2 >= INPUT0_SIZE_X;
        bool BackBottomLOutOfBounds = in_z1 < 0 || in_z1 >= INPUT0_SIZE_Z || in_y2 < 0 || in_y2 >= INPUT0_SIZE_Y || in_x1 < 0 || in_x1 >= INPUT0_SIZE_X;
        bool BackBottomROutOfBounds = in_z1 < 0 || in_z1 >= INPUT0_SIZE_Z || in_y2 < 0 || in_y2 >= INPUT0_SIZE_Y || in_x2 < 0 || in_x2 >= INPUT0_SIZE_X;

        bool FrontTopLOutOfBounds = in_z2 < 0 || in_z2 >= INPUT0_SIZE_Z || in_y1 < 0 || in_y1 >= INPUT0_SIZE_Y || in_x1 < 0 || in_x1 >= INPUT0_SIZE_X;
        bool FrontTopROutOfBounds = in_z2 < 0 || in_z2 >= INPUT0_SIZE_Z || in_y1 < 0 || in_y1 >= INPUT0_SIZE_Y || in_x2 < 0 || in_x2 >= INPUT0_SIZE_X;
        bool FrontBottomLOutOfBounds = in_z2 < 0 || in_z2 >= INPUT0_SIZE_Z || in_y2 < 0 || in_y2 >= INPUT0_SIZE_Y || in_x1 < 0 || in_x1 >= INPUT0_SIZE_X;
        bool FrontBottomROutOfBounds = in_z2 < 0 || in_z2 >= INPUT0_SIZE_Z || in_y2 < 0 || in_y2 >= INPUT0_SIZE_Y || in_x2 < 0 || in_x2 >= INPUT0_SIZE_X;

        const acc_vec_t x111 = BackTopLOutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y1, in_x1)));
        const acc_vec_t x211 = BackTopROutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y1, in_x2)));
        const acc_vec_t x121 = BackBottomLOutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y2, in_x1)));
        const acc_vec_t x221 = BackBottomROutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y2, in_x2)));
        const acc_vec_t x112 = FrontTopLOutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y1, in_x1)));
        const acc_vec_t x212 = FrontTopROutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y1, in_x2)));
        const acc_vec_t x122 = FrontBottomLOutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y2, in_x1)));
        const acc_vec_t x222 = FrontBottomROutOfBounds ? INPUT0_VAL_ZERO : TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y2, in_x2)));
#else
        const acc_vec_t x111 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y1, in_x1)));
        const acc_vec_t x211 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y1, in_x2)));
        const acc_vec_t x121 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y2, in_x1)));
        const acc_vec_t x221 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z1, in_y2, in_x2)));
        const acc_vec_t x112 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y1, in_x1)));
        const acc_vec_t x212 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y1, in_x2)));
        const acc_vec_t x122 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y2, in_x1)));
        const acc_vec_t x222 = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_z2, in_y2, in_x2)));
#endif // PADDING_USED == 1

        acc_vec_t res = TO_ACC_VEC_TYPE(dx2 * dy2 * dz2 * x111) + TO_ACC_VEC_TYPE(dx1 * dy2 * dz2 * x211);
        res += TO_ACC_VEC_TYPE(dx2 * dy1 * dz2 * x121) + TO_ACC_VEC_TYPE(dx1 * dy1 * dz2 * x221);
        res += TO_ACC_VEC_TYPE(dx2 * dy2 * dz1 * x112) + TO_ACC_VEC_TYPE(dx1 * dy2 * dz1 * x212);
        res += TO_ACC_VEC_TYPE(dx2 * dy1 * dz1 * x122) + TO_ACC_VEC_TYPE(dx1 * dy1 * dz1 * x222);

#if HAS_FUSED_OPS
        FUSED_OPS;
        OUT_VEC_TYPE out = FUSED_OPS_RESULT;
#else
        OUT_VEC_TYPE out = TO_OUT_VEC_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
#endif // #if HAS_FUSED_OPS

        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_block, z, y, (x + out_x)), out);
    }
#else // #if defined (THREE_SPATIAL_RESAMPLE)

#if PADDING_USED == 1
    const int saved_in_y1 = in_y1;
    const int saved_in_y2 = in_y2;
#endif

    unroll_for (uint out_x = 0; out_x < OUTPUT_X_BLOCK_SIZE; out_x++) {
        const ACCUMULATOR_TYPE ix = FUNC_CALL(get_original_coordinate)(x + out_x, SCALES[4], OUTPUT_SIZE_X, PADDED_X);
        float in_x = fmax(0, fmin(ix, PADDED_X - 1));
        int in_x1 = min((int)in_x, PADDED_X - 1);
        int in_x2 = min(in_x1 + 1, PADDED_X - 1);
        const ACCUMULATOR_TYPE dx1 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x1)) : 0.5f;
        const ACCUMULATOR_TYPE dx2 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x2)) : 0.5f;

#if PADDING_USED == 1
        in_y1 = saved_in_y1;
        in_y2 = saved_in_y2;

        in_y1 -= PADS_BEGIN[3];
        in_y2 -= PADS_BEGIN[3];
        in_x1 -= PADS_BEGIN[4];
        in_x2 -= PADS_BEGIN[4];

        bool tlOutOfBounds = in_y1 < 0 || in_y1 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
        bool trOutOfBounds = in_y1 < 0 || in_y1 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
        bool blOutOfBounds = in_y2 < 0 || in_y2 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
        bool brOutOfBounds = in_y2 < 0 || in_y2 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
#endif // PADDING_USED == 1

#if OUTPUT_DIMS == 5
        acc_vec_t top_left     = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, in_y1, in_x1)));
        acc_vec_t top_right    = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, in_y1, in_x2)));
        acc_vec_t bottom_left  = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, in_y2, in_x1)));
        acc_vec_t bottom_right = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, z, in_y2, in_x2)));
#else
        acc_vec_t top_left     = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y1, in_x1)));
        acc_vec_t top_right    = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y1, in_x2)));
        acc_vec_t bottom_left  = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y2, in_x1)));
        acc_vec_t bottom_right = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y2, in_x2)));
#endif

#if PADDING_USED == 1
        if (tlOutOfBounds)
            top_left = TO_OUT_VEC_TYPE(INPUT0_VAL_ZERO);
        if (trOutOfBounds)
            top_right = TO_OUT_VEC_TYPE(INPUT0_VAL_ZERO);
        if (blOutOfBounds)
            bottom_left = TO_OUT_VEC_TYPE(INPUT0_VAL_ZERO);
        if (brOutOfBounds)
            bottom_right = TO_OUT_VEC_TYPE(INPUT0_VAL_ZERO);
#endif // PADDING_USED == 1
        acc_vec_t res = TO_ACC_VEC_TYPE(dx2 * dy2 * top_left) +
                        TO_ACC_VEC_TYPE(dx1 * dy2 * top_right) +
                        TO_ACC_VEC_TYPE(dx2 * dy1 * bottom_left) +
                        TO_ACC_VEC_TYPE(dx1 * dy1 * bottom_right);
#if HAS_FUSED_OPS
        FUSED_OPS;
        OUT_VEC_TYPE out = FUSED_OPS_RESULT;
#else
        OUT_VEC_TYPE out = TO_OUT_VEC_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
#endif

#if OUTPUT_DIMS == 5
        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_block, z, y, (x + out_x)), out);
#else
        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_block, y, (x + out_x)), out);
#endif
    }
#endif // #if defined (THREE_SPATIAL_RESAMPLE)
}

#undef TRIANGLE_COEFF
#undef READ_FUNC
#undef WRITE_FUNC
