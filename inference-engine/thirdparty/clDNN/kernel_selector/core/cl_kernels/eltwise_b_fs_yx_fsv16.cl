/*
// Copyright (c) 2020 Intel Corporation
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
*/

#include "include/include_all.cl"
#include "include/common.cl"
#include "include/data_types.cl"

#define FEATURE_SLICE_SIZE 16

#define OUTPUT_TYPE_BLOCK               MAKE_VECTOR_TYPE(OUTPUT_TYPE, BLOCK_SIZE)
#define TO_OUTPUT_TYPE_BLOCK(val)       CAT(convert_, OUTPUT_TYPE_BLOCK)(val)

#if BLOCK_SIZE != 1
    #define READ_FUNC(ptr, offset) CAT(DT_INPUT_BLOCK_READ, BLOCK_SIZE)(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, BLOCK_SIZE)(ptr, offset, val)
#else
    #define READ_FUNC(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
    #define WRITE_FUNC(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#endif

__attribute__((intel_reqd_sub_group_size(FEATURE_SLICE_SIZE)))
KERNEL(eltwise_b_fs_yx_fsv16)(INPUTS_DECLS
                              __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
, FUSED_OPS_DECLS
#endif
)
{
    const uint f_block = get_group_id(0);
    const uint y = (uint)get_global_id(1) / BLOCKS_COUNT;
    const uint x = ((uint)get_global_id(1) % BLOCKS_COUNT) * BLOCK_SIZE;
    const uint b = get_global_id(2);

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset = b * output_b_pitch +
                               (f_block + output_fs_pad_before) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

#if BLOCK_SIZE != 1
    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, BLOCK_SIZE) res;
#else
    ACCUMULATOR_TYPE res;
#endif

    DO_ELTWISE

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE_BLOCK out = TO_OUTPUT_TYPE_BLOCK(FUSED_OPS_RESULT);
#else
#if BLOCK_SIZE != 1
    OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_OUTPUT_TYPE_BLOCK(res), ACTIVATION_PARAMS_TYPED);
#else
    OUTPUT_TYPE out = ACTIVATION_TYPED(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS_TYPED);
#endif
#endif

#ifdef LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE > OUTPUT_FEATURE_NUM) {
        const uint sglid = get_sub_group_local_id();
        if (sglid < OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE) {
            for (uint block_x = 0; block_x < BLOCK_SIZE; block_x++) {
#if BLOCK_SIZE != 1
                output[output_offset + block_x * output_x_pitch + sglid] = out[block_x];
#else
                output[output_offset + block_x * output_x_pitch + sglid] = out;
#endif
            }
        }
    } else
#endif
    {
        WRITE_FUNC(output, output_offset, out);
    }

}
