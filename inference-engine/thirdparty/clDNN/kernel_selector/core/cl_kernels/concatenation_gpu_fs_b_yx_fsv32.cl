// Copyright (c) 2019 Intel Corporation
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
#include "include/unit_type.cl"
#include "include/fetch.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for

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

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL (concatenation_gpu_fs_b_yx_fsv32)(__global UNIT_TYPE* input,
                                         __global UNIT_TYPE* output,
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

    UNIT_TYPE2 in = UNIT_BLOCK_READ2(input, input_offset);

    in = ACTIVATION(in, ACTIVATION_PARAMS);
#if ALIGNED
    const uint dst_index = OUTPUT_GET_INDEX(b, output_offset_in_concat_axis + fs * FSV, y, x);
    UNIT_BLOCK_WRITE2(output, dst_index, in);
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

#undef unroll_for

#undef INPUT0_SIZE_X_WITH_PADDING
#undef INPUT0_SIZE_Y_WITH_PADDING

#undef OUTPUT_SIZE_X_WITH_PADDING
#undef OUTPUT_SIZE_Y_WITH_PADDING
