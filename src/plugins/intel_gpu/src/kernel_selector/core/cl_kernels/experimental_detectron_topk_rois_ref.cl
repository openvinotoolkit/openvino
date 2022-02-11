// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

KERNEL(experimental_detectron_topk_rois_ref)(const __global INPUT0_TYPE* input_rois,
        const __global INPUT1_TYPE* topk_indices, __global OUTPUT_TYPE* output_rois)
{
    const uint b = get_global_id(0);
    const uint output_idx = OUTPUT_GET_INDEX(b, 0, 0, 0);
    const uint roi_idx = topk_indices[b];
    const uint input_idx = INPUT0_GET_INDEX(roi_idx, 0, 0, 0);
    output_rois[output_idx] = input_rois[input_idx];
    output_rois[output_idx + 1] = input_rois[input_idx + 1];
    output_rois[output_idx + 2] = input_rois[input_idx + 2];
    output_rois[output_idx + 3] = input_rois[input_idx + 3];
}
