// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/fetch_data.cl"

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)

#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)

// ======================================================================================
// Required JIT definitions:
// --------------------------------------------------------------------------------------
// SUB_GROUP_SIZE     - [int] sub-group/simd size; limited to 16
// FSV                - [int] feature slice size; limted to 32
// FSV_PER_THREAD     - [int] number of features from slice per thread;
//                            must be equal FSV / SUB_GROUP_SIZE
// ======================================================================================

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL (concatenation_gpu_fs_b_yx_fsv32)(__global INPUT0_TYPE* input,
                                         __global OUTPUT_TYPE* output,
                                         uint output_offset_in_concat_axis
)
{
    uint x = (uint)get_global_id(0);
    uint y = (uint)get_global_id(1);
    uint fs_b_id = get_group_id(2);
    uint sglid = get_sub_group_local_id();

    uint fs = fs_b_id / INPUT0_BATCH_NUM;
    uint b = fs_b_id - fs * INPUT0_BATCH_NUM;

    uint input_offset = 0;
    input_offset += (x + INPUT0_PAD_BEFORE_SIZE_X) * FSV;
    input_offset += (y + INPUT0_PAD_BEFORE_SIZE_Y) * INPUT0_SIZE_X_WITH_PADDING * FSV;
    input_offset += b * INPUT0_SIZE_X_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING * FSV;
    input_offset += fs * INPUT0_SIZE_X_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING * FSV * INPUT0_BATCH_NUM;

    MAKE_VECTOR_TYPE(INPUT0_TYPE, 2) in = DT_INPUT_BLOCK_READ2(input, input_offset);

    in = ACTIVATION(in, ACTIVATION_PARAMS);
#if ALIGNED
    const uint dst_index = OUTPUT_GET_INDEX(b, output_offset_in_concat_axis + fs * FSV, y, x);
    DT_OUTPUT_BLOCK_WRITE2(output, dst_index, in);
#else
    const uint dst_feature = fs * FSV + output_offset_in_concat_axis + sglid;
    if (dst_feature + SUB_GROUP_SIZE < OUTPUT_FEATURE_NUM) {
        output[OUTPUT_GET_INDEX(b, dst_feature, y, x)] = in.s0;
        output[OUTPUT_GET_INDEX(b, dst_feature + SUB_GROUP_SIZE, y, x)] = in.s1;
    } else {
        if (dst_feature < OUTPUT_FEATURE_NUM) {
            output[OUTPUT_GET_INDEX(b, dst_feature, y, x)] = in.s0;
        }
    }
#endif
}

#undef INPUT0_SIZE_X_WITH_PADDING
#undef INPUT0_SIZE_Y_WITH_PADDING

#undef OUTPUT_SIZE_X_WITH_PADDING
#undef OUTPUT_SIZE_Y_WITH_PADDING
