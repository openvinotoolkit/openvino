// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/acc_type.cl"
#include "include/batch_headers/common.cl"


__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(softmax)(__global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if INPUT0_DIMS == 5
    const uint other0 = (uint)get_global_id(0) % INPUT0_OTHER0_SIZE;
    const uint other2 = (uint)get_global_id(0) / INPUT0_OTHER0_SIZE;
#else
    const uint other0 = get_global_id(0);
    const uint other2 = 0;
#endif
    const uint other1 = get_global_id(1);
    const uint other3  = get_global_id(2);

    const uint in_depth_offset  = other3*INPUT0_OTHER3_PITCH + other2*INPUT0_OTHER2_PITCH + other1*INPUT0_OTHER1_PITCH + other0*INPUT0_OTHER0_PITCH + INPUT0_OFFSET;
    const uint out_depth_offset = other3*OUTPUT_OTHER3_PITCH + other2*OUTPUT_OTHER2_PITCH + other1*OUTPUT_OTHER1_PITCH + other0*OUTPUT_OTHER0_PITCH + OUTPUT_OFFSET;

    UNIT_TYPE max_value = UNIT_VAL_MIN;
    UNIT_TYPE data[INPUT0_CLASS_NUM];

    for (uint cls = 0; cls < INPUT0_CLASS_NUM; ++cls)
    {
        const uint index = in_depth_offset + cls*INPUT0_CLASS_PITCH;
        UNIT_TYPE in = input[index];
        max_value = max(max_value, in);
        data[cls] = in;
    }

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    ACCUMULATOR_TYPE denominator = 0.0;
    for (uint cls = 0; cls < INPUT0_CLASS_NUM; ++cls)
    {
        data[cls] = native_exp(data[cls] - max_value);;
        denominator += data[cls];
    }

    for (uint cls = 0; cls < INPUT0_CLASS_NUM; ++cls)
    {
        const UNIT_TYPE res = data[cls] / (UNIT_TYPE)denominator;
        const uint output_idx = out_depth_offset + cls*OUTPUT_CLASS_PITCH;
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
    }
}
