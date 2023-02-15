// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(experimental_detectron_topk_rois_ref)(const __global INPUT0_TYPE* input_rois,
        const __global INPUT1_TYPE* topk_indices, __global OUTPUT_TYPE* output_rois)
{
    const uint b = get_global_id(0);
    const uint roi_idx = topk_indices[INPUT1_GET_INDEX(b, 0, 0, 0)];
    output_rois[OUTPUT_GET_INDEX(b, 0, 0, 0)] = input_rois[INPUT0_GET_INDEX(roi_idx, 0, 0, 0)];
    output_rois[OUTPUT_GET_INDEX(b, 1, 0, 0)] = input_rois[INPUT0_GET_INDEX(roi_idx, 1, 0, 0)];
    output_rois[OUTPUT_GET_INDEX(b, 2, 0, 0)] = input_rois[INPUT0_GET_INDEX(roi_idx, 2, 0, 0)];
    output_rois[OUTPUT_GET_INDEX(b, 3, 0, 0)] = input_rois[INPUT0_GET_INDEX(roi_idx, 3, 0, 0)];
}
