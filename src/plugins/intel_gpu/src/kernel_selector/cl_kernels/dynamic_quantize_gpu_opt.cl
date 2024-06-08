// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    const uint bf = (uint)get_global_id(2);

    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();
    const uint num_sg = get_num_sub_groups();
    const uint group_size = (INPUT0_FEATURE_PITCH / 16 / num_sg);
    const uint offset = bf * INPUT0_FEATURE_PITCH + group_size * (sglid + 16 * sgid);
    __local half partial_max[32]; // FIXME: 16 is an arbitrary number
    half8 val;
    half max;
    half grp_max = 0.001h;

    unroll_for (int i = 0; i < group_size/8; ++i) {
        val = fabs(as_half8(vload8(0, input + offset + (i * 8))));

        max = fmax(fmax(fmax(val[0], val[1]), fmax(val[2], val[3])),
                                fmax(fmax(val[4], val[5]), fmax(val[6], val[7])));
        grp_max = fmax(grp_max, max);
    }

    half max_value = sub_group_reduce_max(grp_max);
    partial_max[sgid] = max_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate global max
    max_value = partial_max[0];
    for (int i = 1; i < num_sg; i++)
        max_value = fmax(max_value, partial_max[i]);

    half scale = 127.0h / max_value;

    unroll_for (int i = 0; i < group_size/8; ++i) {
        val = as_half8(vload8(0, input + offset + i*8));
        val *= scale;
        vstore8(convert_char8(val), 0, output + offset + i*8);
    }

    if (sglid == 0 && sgid == 0)
        output_scale[bf] = 1.0h / scale;
}
