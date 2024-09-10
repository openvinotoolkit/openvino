// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

KERNEL(concatenation_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    uint output_offset_in_concat_axis
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint d1 = (uint)get_global_id(0); // Y
    const uint d2 = (uint)get_global_id(1); // F
#ifdef CHECK_FEATURES
    if (d2 >= INPUT0_FEATURE_NUM)
        return;
#endif
    const uint d3 = (uint)get_global_id(2); // B

    for (size_t d0 = 0; d0 < INPUT0_SIZE_X; ++d0) // X
    {
        uint input_offset = GET_INDEX(INPUT0, INPUT_DIMS_ORDER);
        uint output_offset = GET_INDEX(OUTPUT, OUTPUT_DIMS_ORDER);

        INPUT0_TYPE result = input[input_offset];

#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_offset] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
        output[output_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif
    }
}
