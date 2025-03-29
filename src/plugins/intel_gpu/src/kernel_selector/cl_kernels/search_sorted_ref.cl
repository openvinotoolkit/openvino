// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if RIGHT_MODE == 0
#define CMP <=
#else
#define CMP <
#endif

OUTPUT_TYPE FUNC(binary_search_thread)(const INPUT0_TYPE search_val,
                                       const __global INPUT0_TYPE* restrict sorted, 
                                       OUTPUT_TYPE sorted_begin_idx, 
                                       OUTPUT_TYPE sorted_end_idx) {
    while(sorted_begin_idx != sorted_end_idx) {
        const OUTPUT_TYPE half_offset = (sorted_end_idx-sorted_begin_idx)/2;
        const OUTPUT_TYPE half_idx = sorted_begin_idx+half_offset;
        const INPUT0_TYPE half_val = sorted[half_idx];
        if ( search_val CMP half_val )
            sorted_end_idx = half_idx;
        else
            sorted_begin_idx = half_idx + 1;
    }

    return sorted_begin_idx;
}

#undef CMP

KERNEL(search_sorted_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict sorted, 
    const __global INPUT1_TYPE* restrict values,
    __global OUTPUT_TYPE* restrict output)
{
    // INPUT0_TYPE has to be egual to INPUT1_TYPE
    const int this_thread_idx = get_global_id(0);
    const INPUT0_TYPE search_val = values[this_thread_idx];

    const int SORTED_STRIDE = INPUT0_BATCH_NUM*INPUT0_FEATURE_NUM*INPUT0_SIZE_Y*INPUT0_SIZE_Z;

    // NOTE: SORTED_STRIDE-1 handles here a special case when sorted is actually 1D
    // tensor and values is ND tensor. In such case we effectively want sorted_offset 
    // to be 0.
    const int sorted_offset = min(this_thread_idx/INPUT1_SIZE_X, SORTED_STRIDE-1);

    OUTPUT_TYPE sorted_begin_idx = sorted_offset * INPUT0_SIZE_X;
    const OUTPUT_TYPE idx = FUNC_CALL(binary_search_thread)(search_val, 
                                                            sorted + sorted_begin_idx, 
                                                            0, 
                                                            INPUT0_SIZE_X);
    
    output[this_thread_idx] = idx;
}