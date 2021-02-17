// Copyright (C) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/include_all.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define READ_FUNC(ptr, offset)          BLOCK_READN(INPUT0_TYPE, VEC_SIZE, ptr, offset)
#define WRITE_FUNC(ptr, offset, val)    BLOCK_WRITEN(OUTPUT_TYPE, VEC_SIZE, ptr, offset, val)

#define IN_VEC_TYPE                     MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define TO_IN_VEC_TYPE(x)               CAT(convert_, IN_VEC_TYPE)(x)
#define ACC_VEC_TYPE                    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE)
#define TO_ACC_VEC_TYPE(x)              CAT(convert_, ACC_VEC_TYPE)(x)
#define OUT_VEC_TYPE                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define TO_OUT_VEC_TYPE(x)              CAT(convert_, OUT_VEC_TYPE)(x)

inline float FUNC(get_original_coordinate)(float num, float scale, int length_resized, int length_original)
{
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

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL (resample_opt)(__global INPUT0_TYPE* input,
                      __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                      , FUSED_OPS_DECLS
#endif
)
{
    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);
    const int f_block = get_group_id(1);
    const int b = get_global_id(2);
    const int feature_num = f_block * FEATURE_SLICE_SIZE + get_sub_group_local_id();
    const uint feature_block = f_block * FEATURE_SLICE_SIZE;

    typedef IN_VEC_TYPE in_vec_t;
    typedef ACC_VEC_TYPE acc_vec_t;

    if (feature_num >= OUTPUT_FEATURE_NUM)
        return;

    unroll_for (uint out_x = 0; out_x < OUTPUT_X_BLOCK_SIZE; out_x++) {
#ifdef SAMPLE_TYPE_NEAREST
        const int ix = floor((x + out_x) * SCALES[4]);
        const int iy = floor(y * SCALES[3]);

        in_vec_t res = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, iy, ix));
#elif defined(SAMPLE_TYPE_INTERP)
        const ACCUMULATOR_TYPE ix = TO_ACCUMULATOR_TYPE(SCALES[4]) * (x + out_x);
        const ACCUMULATOR_TYPE iy = TO_ACCUMULATOR_TYPE(SCALES[3]) * y;

        const int top_y_index    = (int)(floor(iy));
        const int bottom_y_index = min((int)ceil(iy), INPUT0_SIZE_Y - 1);
        const int left_x_index   = (int)(floor(ix));
        const int right_x_index  = min((int)ceil(ix), INPUT0_SIZE_X - 1);

        const ACCUMULATOR_TYPE dx = ix - left_x_index;
        const ACCUMULATOR_TYPE dy = iy - top_y_index;

        const in_vec_t top_left     = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, top_y_index, left_x_index));
        const in_vec_t top_right    = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, top_y_index, right_x_index));
        const in_vec_t bottom_left  = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, bottom_y_index, left_x_index));
        const in_vec_t bottom_right = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, bottom_y_index, right_x_index));

        const acc_vec_t top    = TO_ACC_VEC_TYPE(top_left) + (TO_ACC_VEC_TYPE(top_right) - TO_ACC_VEC_TYPE(top_left)) * dx;
        const acc_vec_t bottom = TO_ACC_VEC_TYPE(bottom_left) + (TO_ACC_VEC_TYPE(bottom_right) - TO_ACC_VEC_TYPE(bottom_left)) * dx;
        acc_vec_t res = top + (bottom - top) * dy;
#else // defined(SAMPLE_TYPE_LINEAR_ONNX)
        const int PADDED_Y = INPUT0_SIZE_Y + PADS_BEGIN[3] + PADS_END[3];
        const int PADDED_X = INPUT0_SIZE_X + PADS_BEGIN[4] + PADS_END[4];

        const ACCUMULATOR_TYPE ix = FUNC_CALL(get_original_coordinate)(x + out_x, SCALES[4], OUTPUT_SIZE_X, PADDED_X);
        const ACCUMULATOR_TYPE iy = FUNC_CALL(get_original_coordinate)(y, SCALES[3], OUTPUT_SIZE_Y, PADDED_Y);

        float in_y = fmax(0, fmin(iy, PADDED_Y - 1));
        float in_x = fmax(0, fmin(ix, PADDED_X - 1));
        int in_y1 = min((int)in_y, PADDED_Y - 1);
        int in_y2 = min(in_y1 + 1, PADDED_Y - 1);
        int in_x1 = min((int)in_x, PADDED_X - 1);
        int in_x2 = min(in_x1 + 1, PADDED_X - 1);
        const ACCUMULATOR_TYPE dx1 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x1)) : 0.5f;
        const ACCUMULATOR_TYPE dx2 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x2)) : 0.5f;
        const ACCUMULATOR_TYPE dy1 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y1)) : 0.5f;
        const ACCUMULATOR_TYPE dy2 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y2)) : 0.5f;
#if PADDING_USED == 1
        in_y1 -= PADS_BEGIN[3];
        in_y2 -= PADS_BEGIN[3];
        in_x1 -= PADS_BEGIN[4];
        in_x2 -= PADS_BEGIN[4];

        bool tlOutOfBounds = in_y1 < 0 || in_y1 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
        bool trOutOfBounds = in_y1 < 0 || in_y1 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
        bool blOutOfBounds = in_y2 < 0 || in_y2 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
        bool brOutOfBounds = in_y2 < 0 || in_y2 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
#endif // PADDING_USED == 1
        const acc_vec_t top_left     = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y1, in_x1)));
        const acc_vec_t top_right    = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y1, in_x2)));
        const acc_vec_t bottom_left  = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y2, in_x1)));
        const acc_vec_t bottom_right = TO_ACC_VEC_TYPE(READ_FUNC(input, INPUT0_GET_INDEX(b, feature_block, in_y2, in_x2)));
#if PADDING_USED == 1
        if (tlOutOfBounds)
            top_left = INPUT0_VAL_ZERO;
        if (trOutOfBounds)
            top_right = INPUT0_VAL_ZERO;
        if (blOutOfBounds)
            bottom_left = INPUT0_VAL_ZERO;
        if (brOutOfBounds)
            bottom_right = INPUT0_VAL_ZERO;
#endif // PADDING_USED == 1
        acc_vec_t res = TO_ACC_VEC_TYPE(dx2 * dy2 * top_left) +
                        TO_ACC_VEC_TYPE(dx1 * dy2 * top_right) +
                        TO_ACC_VEC_TYPE(dx2 * dy1 * bottom_left) +
                        TO_ACC_VEC_TYPE(dx1 * dy1 * bottom_right);
#endif
#if HAS_FUSED_OPS
        FUSED_OPS;
        OUT_VEC_TYPE out = FUSED_OPS_RESULT;
#else
        OUT_VEC_TYPE out = TO_OUT_VEC_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
#endif

        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_block, y, (x + out_x)), out);
    }
}

#undef unroll_for
#undef READ_FUNC
#undef WRITE_FUNC
