// Copyright (c) 2016-2019 Intel Corporation
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

#include "include/common.cl"
#include "include/fetch.cl"

KERNEL(kernel_name)(
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights,
#if BIAS_TERM
    const __global BIAS_TYPE *biases,
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp,
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp,
#endif
#if COMPENSATION_TERM
    const __global COMPENSATION_TYPE *comp,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx
    )
{
    // Convolution part.
    const uint x = get_global_id(0);
#if  OUTPUT_DIMS > 4
    const uint y = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
#else
    const uint y = get_global_id(1);
    const uint z = 0;
#endif
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif

    ACCUMULATOR_TYPE dotProd = (ACCUMULATOR_TYPE)0;
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
#if  OUTPUT_DIMS > 4
    const int input_z = z * STRIDE_SIZE_Z - PADDING_SIZE_Z;
#else
    const int input_z = 0;
#endif

#if DEPTHWISE_SEPARABLE_OPT || GROUPED
    const uint g = (f / FILTER_OFM_NUM);
    const uint of = (f % FILTER_OFM_NUM);
#else
    const uint g = 0;
    const uint of = f;
#endif

    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
#if INPUT0_SIZE > 4
        for (uint l = 0; l < FILTER_SIZE_Z ; ++l)
        {
            const int input_offset_z = input_z + l * DILATION_SIZE_Z;
            const bool zero_z = input_offset_z >= INPUT0_SIZE_Z || input_offset_z < 0;
            if(!zero_z)
            {
#endif
                for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
                {
                    const int input_offset_y = input_y + j * DILATION_SIZE_Y;
                    const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

                    if(!zero_y)
                    {
                        for (uint i = 0; i < FILTER_SIZE_X ; ++i)
                        {
                            const int input_offset_x = input_x + i * DILATION_SIZE_X;
                            const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                            if(!zero_x)
                            {
#if INPUT0_SIZE <= 4
                                uint input_idx = INPUT0_GET_INDEX(b, (k + g*FILTER_IFM_NUM), input_offset_y, input_offset_x);
                                uint filter_idx = GET_FILTER_INDEX(FILTER, g, of, k, j, i);
#else
                                uint input_idx = INPUT0_GET_INDEX(b, (k + g*FILTER_IFM_NUM), input_offset_z, input_offset_y, input_offset_x);
                                uint filter_idx = GET_FILTER_INDEX_5D(FILTER, g, of, k, l, j, i);
#endif

#ifdef LOCAL_CONVOLUTION
                                filter_idx += FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z
                                    * (x + OUTPUT_SIZE_X * y + OUTPUT_SIZE_X * OUTPUT_SIZE_Y * z);
#endif
                                ACCUMULATOR_TYPE in = TO_ACCUMULATOR_TYPE(conv_input[input_idx]);
#if ASYMMETRIC_DATA_QUANTIZATION
                                in -= TO_ACCUMULATOR_TYPE(activations_zp[g * FILTER_IFM_NUM + k]);
#endif
                                ACCUMULATOR_TYPE wei = TO_ACCUMULATOR_TYPE(weights[filter_idx]);
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                                wei -= TO_ACCUMULATOR_TYPE(weights_zp[f]);
#endif
                                dotProd += in * wei;
                            }
                        }
                    }
                }
#if INPUT0_SIZE > 4
            }
        }
#endif
    }

#if BIAS_TERM
    #if GROUPED || DEPTHWISE_SEPARABLE_OPT
        const uint bias_offset = 0;
    #else
        const uint bias_offset = 0;
    #endif
    #if   BIAS_PER_OUTPUT
        const uint bias_index = bias_offset + GET_DATA_INDEX_5D(BIAS, b, f, z, y, x);
    #elif BIAS_PER_OFM
        const uint bias_index = bias_offset + f;
    #endif

    ACTIVATION_TYPE dequantized = dotProd + biases[bias_index];
#else
    ACTIVATION_TYPE dequantized = dotProd;
#endif


#if OUTPUT_SIZE <= 4
    const uint dst_index = OUTPUT_GET_INDEX(b, f, y, x);
#else
    const uint dst_index = OUTPUT_GET_INDEX(b, f, z, y, x);
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;

    output[dst_index] = res;
#else
    output[dst_index] = ACTIVATION_TYPED(dequantized, ACTIVATION_PARAMS_TYPED);
#endif
}
