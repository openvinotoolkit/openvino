// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/reshape_dims.cl"

// Required JIT constants:
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weights/biases.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - INPUT_BATCH_NUM      - [int] Number of elements from single spatial and single feature that are grouped in single batch in input.
//  - INPUT_ELEMENTS_COUNT - [int] Cumulative number of elements from input that are processed in single batch.
//  - FILTER_OFM_NUM       - [int] Cumulative number of elements that are outputted in single batch.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).

KERNEL (fully_connected_gpu_yxfn)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = get_global_id(0);
    const uint batch_id = x % INPUT0_BATCH_NUM;
    const uint neuronIdx = x / INPUT0_BATCH_NUM;

    UNIT_TYPE result = UNIT_VAL_ZERO;

    uint weight_offset = neuronIdx * FILTER_OFM_PITCH;
    for (uint k = 0; k < INPUT0_FEATURE_NUM; k++)
    {
        for (uint j = 0; j < INPUT0_SIZE_Y; j++)
        {
            for (uint i = 0; i < INPUT0_SIZE_X; i++)
            {
                // Due to reshape from DataTensor to WeightTensor reshape_dims function is called directly
                uint8 widx = FUNC_CALL(reshape_dims)(
                    batch_id, k, 0, 0, j, i,   // b, f, w, z, y, x
                    INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X,
                    FILTER_IFM_NUM, 1, FILTER_SIZE_Z, FILTER_SIZE_Y, FILTER_SIZE_X,
                    INPUT0_DIMS, FILTER_DIMS);
                uint weight_idx = weight_offset + widx[2] * FILTER_IFM_PITCH + widx[5] * FILTER_Y_PITCH + widx[6] * FILTER_X_PITCH;
                uint input_idx = INPUT0_OFFSET + k * INPUT0_FEATURE_PITCH + j * INPUT0_Y_PITCH + i * INPUT0_X_PITCH + batch_id * INPUT0_BATCH_PITCH;
                result += input[input_idx] * weights[weight_idx];
            }
        }
    }
    const uint output_idx = OUTPUT_OFFSET + batch_id * OUTPUT_BATCH_PITCH + neuronIdx * OUTPUT_FEATURE_PITCH;

#if BIAS_TERM
    result += biases[neuronIdx];
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;

    output[output_idx] = res;
#else
    output[output_idx] = ACTIVATION(result, ACTIVATION_PARAMS);
#endif
}
