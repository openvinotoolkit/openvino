// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void experimental_detectron_prior_grid_generator(
    __global const half *restrict input_priors,
    __global const half *restrict input_feature_map,
    __global const half *restrict input_rois,
    __global half *restrict output,
    int grid_h,
    int grid_w,
    float stride_h,
    float stride_w,
    int num_priors,
    int num_anchors_per_prior)
{
    __local half local_input_priors[8 * 1024];
    __local half local_output[8 * 1024];

    event_t e1 = async_work_group_copy(
        local_input_priors,
        input_priors,
        num_anchors_per_prior * num_priors,
        0);
    wait_group_events(1, &e1);

    int width_start = get_group_id(0) * get_local_size(0);
    int width_end   = min(width_start + get_local_size(0), (unsigned)grid_w);
    int width       = width_end - width_start;

    int h     = get_group_id(1);
    int w_idx = get_group_id(0) * get_local_size(0);
    for (int w = 0; w < width; ++w) {
        #pragma unroll 4
        for (int p = 0; p < num_priors; ++p) {
            local_output[(w * num_priors + p) * num_anchors_per_prior + 0] =
                local_input_priors[4 * p + 0]
                + convert_half(stride_w) * (convert_half(w_idx + w) + 0.5);
            local_output[(w * num_priors + p) * num_anchors_per_prior + 1] =
                local_input_priors[4 * p + 1] + convert_half(stride_h) * (convert_half(h) + 0.5);
            local_output[(w * num_priors + p) * num_anchors_per_prior + 2] =
                local_input_priors[4 * p + 2]
                + convert_half(stride_w) * (convert_half(w_idx + w) + 0.5);
            local_output[(w * num_priors + p) * num_anchors_per_prior + 3] =
                local_input_priors[4 * p + 3] + convert_half(stride_h) * (convert_half(h) + 0.5);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_2D2D(
        output + get_group_id(0) * get_local_size(0) * num_anchors_per_prior * num_priors
            + get_group_id(1) * get_local_size(1) * grid_w * num_anchors_per_prior
                  * num_priors, // dst
        local_output, // src
        width * num_anchors_per_prior * num_priors, // num_elements_per_line
        1, // num_lines
        (grid_w - width) * num_anchors_per_prior * num_priors, // src_line_stride
        (grid_w - width) * num_anchors_per_prior * num_priors, // dst_line_stride
        0);
    wait_group_events(1, &e2);
}
