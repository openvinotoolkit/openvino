// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

// ======================================================================================
// Optimized concatenation kernel for b_fs_yx_fsv32 format (feature-axis concat)
// Supports INT8/UINT8/FP16 data types.
//
// Required JIT definitions:
// --------------------------------------------------------------------------------------
// SUB_GROUP_SIZE     - [int] sub-group/simd size; limited to 16
// FSV                - [int] feature slice size; limited to 32
// FSV_PER_THREAD     - [int] number of features per thread = FSV / SUB_GROUP_SIZE
// ALIGNED            - [0/1] whether the output offset is aligned to FSV
// ======================================================================================

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(concatenation_gpu_b_fs_yx_fsv32)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    uint output_offset_in_concat_axis)
{
    const uint x = (uint)get_global_id(0);
    const uint y = (uint)get_global_id(1);
    const uint fs_b_id = get_group_id(2);
    const uint sglid = get_sub_group_local_id();

    const uint fs = fs_b_id / INPUT0_BATCH_NUM;
    const uint b = fs_b_id - fs * INPUT0_BATCH_NUM;

    const uint input_offset = INPUT0_GET_INDEX(b, fs * FSV, y, x);

    MAKE_VECTOR_TYPE(INPUT0_TYPE, 2) in = DT_INPUT_BLOCK_READ2(input, input_offset);

    in = ACTIVATION(in, ACTIVATION_PARAMS);

#if ALIGNED
    const uint dst_index = OUTPUT_GET_INDEX(b, output_offset_in_concat_axis + fs * FSV, y, x);

    // Full feature block: use block write for maximum throughput
    if (fs * FSV + FSV <= INPUT0_FEATURE_NUM) {
        DT_OUTPUT_BLOCK_WRITE2(output, dst_index, in);
    } else {
        // Last partial feature block: write only valid features
        if (sglid + fs * FSV < INPUT0_FEATURE_NUM) {
            output[dst_index + sglid] = in.s0;
        }
        if (sglid + SUB_GROUP_SIZE + fs * FSV < INPUT0_FEATURE_NUM) {
            output[dst_index + SUB_GROUP_SIZE + sglid] = in.s1;
        }
    }
#else
    // Unaligned case: use per-element writes with proper index computation
    const uint dst_feature = fs * FSV + output_offset_in_concat_axis + sglid;

    if (sglid + SUB_GROUP_SIZE + fs * FSV < INPUT0_FEATURE_NUM) {
        output[OUTPUT_GET_INDEX(b, dst_feature, y, x)] = in.s0;
        output[OUTPUT_GET_INDEX(b, dst_feature + SUB_GROUP_SIZE, y, x)] = in.s1;
    } else if (sglid + fs * FSV < INPUT0_FEATURE_NUM) {
        output[OUTPUT_GET_INDEX(b, dst_feature, y, x)] = in.s0;
    }
#endif
}
