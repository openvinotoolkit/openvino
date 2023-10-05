// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(rms_gpu_bfyx_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* gamma,
    __global OUTPUT_TYPE* output)
{
    const uint in_data_idx = get_global_id(0);
    const uint data_idx = get_global_id(1);
    const uint lws_size = LWS;
    const uint items_num = VEC_SIZE;
    const uint data_size = DATA_SIZE;
    const uint total_items_num = lws_size * items_num;
#if !IS_DYNAMIC
    const uint leftovers = LEFTOVERS;
#else
    const uint leftovers = data_size % items_num;
#endif

    const uint data_offset = data_idx * data_size;
    const uint in_data_offset = data_offset + in_data_idx * items_num;
    const uint gamma_offset = in_data_idx * items_num;

    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;

    __local ACCUMULATOR_TYPE slm_buf[SLM_SIZE];

    INPUT_VEC_TYPE inputs = AS_INPUT_VEC_TYPE(VLOAD(0, input + in_data_offset));
    ACCUMULATOR_VEC_TYPE square = pow(TO_ACCUMULATOR_VEC_TYPE(inputs), (ACCUMULATOR_VEC_TYPE)(2));
    unroll_for (uint i = 0; i < VEC_SIZE; ++i) {
        rms += square[i];
    }

    if (in_data_idx < leftovers)
    {
        const uint input_idx = data_offset + total_items_num + in_data_idx;
        rms += pow(TO_ACCUMULATOR_TYPE(input[input_idx]), 2);
    }

    slm_buf[in_data_idx] = rms;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_idx == 0)
    {
#if !IS_DYNAMIC
        unroll_for (uint i = 1; i < LWS; ++i)
#else
        for (uint i = 1; i < lws_size; ++i)
#endif
            rms += slm_buf[i];

        rms = rms / data_size;
        slm_buf[0] = pow(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    rms = slm_buf[0];

    OUTPUT_VEC_TYPE results = TO_OUTPUT_VEC_TYPE((ACCUMULATOR_VEC_TYPE)(rms) * TO_ACCUMULATOR_VEC_TYPE(inputs) * AS_ACCUMULATOR_VEC_TYPE(VLOAD(0, gamma + gamma_offset)));
    VSTORE(results, 0, output + in_data_offset);

    if (in_data_idx < leftovers)
    {
        const uint input_idx = data_offset + total_items_num + in_data_idx;
        const uint output_idx = data_offset + total_items_num + in_data_idx;
        const uint gamma_idx = total_items_num + in_data_idx;
        OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(input[input_idx]) * TO_ACCUMULATOR_TYPE(gamma[gamma_idx]));
        output[output_idx] = result;
    }
}
