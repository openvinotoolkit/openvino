// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// Required JIT constants:
//  - FP16_SUPPORTED        - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED        - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE             - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO         - Literal of current UNIT_TYPE that represents 0.
//  - INPUT0_BATCH_NUM      - [int] Number of elements from single spatial and single feature that are grouped in single batch in input.
//  - INPUT0_ELEMENTS_COUNT - [int] Cumulative number of elements from input that are processed in single batch.
//  - FILTER_OFM_NUM        - [int] Cumulative number of elements that are outputted in single batch.
//  - RELU                  - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE        - [float] Factor for negative output values (required when ReLU is specified).


KERNEL (fully_connected_gpu_bx_xb_from_fyxb)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = get_global_id(0);
    const uint batch_id = x / FILTER_OFM_NUM;

    const uint outXIdx = x % FILTER_OFM_NUM;
    UNIT_TYPE result = UNIT_VAL_ZERO;

    uint input_idx = batch_id * INPUT0_ELEMENTS_COUNT;
    input_idx = MULTIPLY_OFFSET(UNIT_TYPE, input_idx);
    uint weight_idx = MULTIPLY_OFFSET(UNIT_TYPE, outXIdx);
    for (uint i = 0; i < INPUT0_ELEMENTS_COUNT; i++)
    {
        UNIT_TYPE _in = *OFFSET_GLOBAL_PTR(UNIT_TYPE, input, input_idx);
        UNIT_TYPE _w =  *OFFSET_GLOBAL_PTR(UNIT_TYPE, weight, weight_idx);
        result += _in * _w;
        input_idx  += MULTIPLY_OFFSET(UNIT_TYPE, 1);
        weight_idx += MULTIPLY_OFFSET(UNIT_TYPE, FILTER_OFM_NUM);
    }
#if BIAS_TERM
    result += bias[outXIdx];
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;

    output[x] = res;
#else
    output[x] = ACTIVATION(result, ACTIVATION_PARAMS);
#endif
}
