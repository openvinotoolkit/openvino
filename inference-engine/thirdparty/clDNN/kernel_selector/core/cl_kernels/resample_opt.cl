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
#else
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
