// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"

KERNEL(eddo_ref_stage_0)
(__global OUTPUT_TYPE* output_boxes,
 __global OUTPUT_INDICES_TYPE* output_classes,
 __global OUTPUT_TYPE* output_scores) {
    const size_t i = get_global_id(0);

    if (i == 0) {
        output_boxes[4 * i + 0] = 0.;
        output_boxes[4 * i + 1] = 0.892986;
        output_boxes[4 * i + 2] = 10.107;
        output_boxes[4 * i + 3] = 12.107;
        output_scores[i] = 0.9;
        output_classes[i] = 1;
    }

    // NB!!!! uncomment this line, and the problem disappears
    // barrier(CLK_GLOBAL_MEM_FENCE);
}
