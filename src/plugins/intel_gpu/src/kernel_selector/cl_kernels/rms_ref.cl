// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(rms_ref)(
    // OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* gamma,
    __global OUTPUT_TYPE* output)
{
    const uint in_data_idx = get_global_id(0);
    const uint data_idx = get_global_id(1);
    const uint data_size = DATA_SIZE;
    const uint data_count = DATA_COUNT;
    const uint items_num = VEC_SIZE;

    const uint data_offset = data_idx * data_size;
    const uint in_data_offset = data_offset + in_data_idx * items_num;
    const uint gamma_offset = in_data_idx * items_num;

    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;

    __local ACCUMULATOR_TYPE slm_buf[SLM_SIZE];

    INPUTVTYPE inputs = AS_INPUTVTYPE(VLOAD(0, input + in_data_offset));
    INPUTVTYPE square = pow(inputs, (INPUTVTYPE)(2));
    unroll_for (uint i = 0; i < VEC_SIZE; ++i) {
        rms += TO_ACCUMULATOR_TYPE(square[i]);
    }

    slm_buf[in_data_idx] = rms;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_idx == 0)
    {
        unroll_for (uint i = 1; i < SLM_SIZE; ++i)
            rms += slm_buf[i];

        rms = rms / data_size;
        slm_buf[0] = pow(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    rms = slm_buf[0];

    OUTPUTVTYPE results = (OUTPUTVTYPE)(rms) * AS_OUTPUTVTYPE(inputs) * AS_OUTPUTVTYPE(VLOAD(0, gamma + gamma_offset));
    VSTORE(results, 0, output + in_data_offset);
}
