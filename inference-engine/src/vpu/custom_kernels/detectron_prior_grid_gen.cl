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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void __dma_preload_experimental_detectron_prior_grid_generator(
    __global const half* restrict input_priors,
    __global const half* restrict input_feature_map,
    __global const half* restrict input_rois,
    __global half* restrict output,
    __local half* restrict local_input_priors,
    __local half* restrict local_output,
    int grid_h,
    int grid_w,
    float stride_h,
    float stride_w,
    int num_priors,
    int num_anchors_per_prior) {

    // Move input_priors to local memory.
    WorkGroupDmaCreateStrideTransaction(
        input_priors, // src
        local_input_priors, // dst
        num_anchors_per_prior * num_priors * sizeof(half), // src_width
        num_anchors_per_prior * num_priors * sizeof(half), // dst_width
        num_anchors_per_prior * num_priors * sizeof(half), // src_stride
        num_anchors_per_prior * num_priors * sizeof(half), // dst_stride
        num_anchors_per_prior * num_priors * sizeof(half), // total_size
        0);
}

__kernel void __dma_postwrite_experimental_detectron_prior_grid_generator(
    __global const half* restrict input_priors,
    __global const half* restrict input_feature_map,
    __global const half* restrict input_rois,
    __global half* restrict output,
    __local half* restrict local_input_priors,
    __local half* restrict local_output,
    int grid_h,
    int grid_w,
    float stride_h,
    float stride_w,
    int num_priors,
    int num_anchors_per_prior) {

    int local_width = get_local_size(0);
    int width_start = get_group_id(0) * get_local_size(0);
    int width_end = min(width_start + local_width, grid_w);
    int width = width_end - width_start;

    WorkGroupDmaCreateStrideTransaction(
        local_output,                                               // src
        output + get_group_id(0) * get_local_size(0) *
                 num_anchors_per_prior * num_priors
               + get_group_id(1) * get_local_size(1) * grid_w *
                 num_anchors_per_prior * num_priors,                // dst
        width * num_anchors_per_prior * num_priors * sizeof(half),  // src_width
        width * num_anchors_per_prior * num_priors * sizeof(half),  // dst_width
        grid_w * num_anchors_per_prior * num_priors * sizeof(half), // src_stride
        grid_w * num_anchors_per_prior * num_priors * sizeof(half), // dst_stride
        width * num_anchors_per_prior * num_priors * sizeof(half),  // total_size
        0);
}

__kernel void experimental_detectron_prior_grid_generator(
    __global const half* restrict input_priors,
    __global const half* restrict input_feature_map,
    __global const half* restrict input_rois,
    __global half* restrict output,
    __local half* restrict local_input_priors,
    __local half* restrict local_output,
    int grid_h,
    int grid_w,
    float stride_h,
    float stride_w,
    int num_priors,
    int num_anchors_per_prior) {

    int workgroup_width = get_local_size(0);
    int width_start = get_group_id(0) * workgroup_width;
    int width_end = min(width_start + workgroup_width, grid_w);
    int width = width_end - width_start;

    int h = get_group_id(1);
    int w_idx = get_group_id(0) * workgroup_width;
    for (int w = 0; w < width; ++w) {
        #pragma unroll 4
        for (int p = 0; p < num_priors; ++p) {
            local_output[(w * num_priors + p) * num_anchors_per_prior + 0] =
                local_input_priors[4 * p + 0] +
                convert_half(stride_w) * (convert_half(w_idx + w) + 0.5);
            local_output[(w * num_priors + p) * num_anchors_per_prior + 1] =
                local_input_priors[4 * p + 1] +
                convert_half(stride_h) * (convert_half(h) + 0.5);
            local_output[(w * num_priors + p) * num_anchors_per_prior + 2] =
                local_input_priors[4 * p + 2] +
                convert_half(stride_w) * (convert_half(w_idx + w) + 0.5);
            local_output[(w * num_priors + p) * num_anchors_per_prior + 3] =
                local_input_priors[4 * p + 3] +
                convert_half(stride_h) * (convert_half(h) + 0.5);
        }
    }
}
