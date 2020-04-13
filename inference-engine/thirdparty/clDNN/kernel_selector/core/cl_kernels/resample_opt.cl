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
#include "include/unit_type.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for

#ifdef INPUT0_LAYOUT_FS_B_YX_FSV32
    #define READ_FUNC(ptr, offset) CAT(UNIT_BLOCK_READ, VEC_SIZE)(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) CAT(UNIT_BLOCK_WRITE, VEC_SIZE)(ptr, offset, val)
#else
    #define READ_FUNC(ptr, offset) UNIT_BLOCK_READ(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) UNIT_BLOCK_WRITE(ptr, offset, val)
#endif

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
#ifdef INPUT0_LAYOUT_FS_B_YX_FSV32
    typedef MAKE_VECTOR_TYPE(UNIT_TYPE, VEC_SIZE) unit_t;
#else
    typedef UNIT_TYPE unit_t;
#endif

    if (feature_num >= OUTPUT_FEATURE_NUM)
        return;

    unroll_for (uint out_x = 0; out_x < OUTPUT_X_BLOCK_SIZE; out_x++) {
#ifdef SAMPLE_TYPE_NEAREST
        const int ix = floor((x + out_x) * X_RATIO);
        const int iy = floor(y * Y_RATIO);

        unit_t res = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_num, iy, ix));
#else
        const UNIT_TYPE ix = TO_UNIT_TYPE(X_RATIO) * (x + out_x);
        const UNIT_TYPE iy = TO_UNIT_TYPE(Y_RATIO) * y;

        const int top_y_index    = (int)(floor(iy));
        const int bottom_y_index = (int)(min(ceil(iy), TO_UNIT_TYPE(INPUT0_SIZE_Y) - 1));
        const int left_x_index   = (int)(floor(ix));
        const int right_x_index  = (int)(min(ceil(ix), TO_UNIT_TYPE(INPUT0_SIZE_X) - 1));

        const UNIT_TYPE dx = ix - left_x_index;
        const UNIT_TYPE dy = iy - top_y_index;

        const unit_t top_left     = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_num, top_y_index, left_x_index));
        const unit_t top_right    = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_num, top_y_index, right_x_index));
        const unit_t bottom_left  = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_num, bottom_y_index, left_x_index));
        const unit_t bottom_right = READ_FUNC(input, INPUT0_GET_INDEX(b, feature_num, bottom_y_index, right_x_index));

        const unit_t top    = top_left + (top_right - top_left) * dx;
        const unit_t bottom = bottom_left + (bottom_right - bottom_left) * dx;
        unit_t res = top + (bottom - top) * dy;
#endif
#if HAS_FUSED_OPS
        FUSED_OPS;
        res = FUSED_OPS_RESULT;
#else
        res = ACTIVATION(res, ACTIVATION_PARAMS);
#endif

#if OUTPUT_IS_FP
        WRITE_FUNC(output, OUTPUT_GET_INDEX(b, feature_num, y, (x + out_x)), res);
#else
#if VEC_SIZE > 1
        for (uint i = 0; i < VEC_SIZE; i++)
            output[OUTPUT_GET_INDEX(b, feature_num + i*SUB_GROUP_SIZE, y, (x + out_x))] = res[i];
#else
            output[OUTPUT_GET_INDEX(b, feature_num, y, (x + out_x))] = res;
#endif

#endif
    }
}

#undef unroll_for
#undef READ_FUNC
#undef WRITE_FUNC
