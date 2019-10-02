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

#if QUANTIZATION_TERM || CALIBRATION_TERM || defined(O_QF)
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE float
#    define ACTIVATION_TYPE_BASE float
     // TODO: It's unclear for now what should be the rounding scheme. Might
     // happen that we will be required to round to nearest-even. Should it
     // become a customization point, or will we support just one rounding
     // scheme?
#    if OUTPUT_IS_FP
         // [U]INT8 -> float convolution with quantization/calibration.
#        define AFTER_CALIBRATION_ROUND(x) (x)
#    else
         // TODO: Do we need the round of the conv result in the fused
         // primitive?
#        define AFTER_CALIBRATION_ROUND(x) round(x)
#    endif
#elif defined(ACTIVATION_ELTW_TYPED) && (INT8_UNIT_USED || UINT8_UNIT_USED)
    // TODO: Get rid of INT8_UNIT_USED - its meaning doesn't look to be defined
    // properly. Better to check INPUT0_TYPE is [U]INT8 somehow...
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE int
#    define ACTIVATION_TYPE_BASE int
#    define AFTER_CALIBRATION_ROUND(x) (x)
#else
#    define ACCUMULATOR_TYPE INPUT0_TYPE
#    define TO_ACCUMULATOR_TYPE(x) TO_INPUT0_TYPE(x)
#    define ACTIVATION_TYPE INPUT0_TYPE
#    define ACTIVATION_TYPE_BASE INPUT0
#    define AFTER_CALIBRATION_ROUND(x) (x)
#endif

#if defined(ACTIVATION_ELTW_TYPED)
#    if OUTPUT_IS_FP
#        define AFTER_ELTW_CALIBRATION_ROUND(x) (x)
#    else
#        define AFTER_ELTW_CALIBRATION_ROUND(x) round(x)
#    endif
#endif

#if QUANTIZATION_TERM && (!defined(CALIBRATION_TERM) || CALIBRATION_TERM == 0 ) && !defined(O_QF)
    // To get proper type for the "before_activation" below.
#   define O_QF_LOCAL_DEFINE
#   define O_QF 1.0
#endif

KERNEL(kernel_name)(
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights,
#if BIAS_TERM
    const __global BIAS_TYPE *biases,
#endif
#if QUANTIZATION_TERM
    const __global float* quantizations,
#endif
#if CALIBRATION_TERM
    const __global float* calibrations,
#endif
    uint split_idx
#if defined(ACTIVATION_ELTW_TYPED)
#    if IN_OUT_OPT
    // The argument is always present (in this case it would be the same as
    // output), just ignore it.
    , const __global OUTPUT_TYPE *_ignore
#    else
    , const __global INPUT1_TYPE *eltw_input
#    endif
#    if ELTW_CALIBRATION_TERM
    , const __global float* eltw_output_calibrations
#    endif
#endif
    )
{
    // Convolution part.
    const uint x = get_global_id(0);
#if  OUTPUT_DIMS > 4
    const uint y = get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z = get_global_id(1) / OUTPUT_SIZE_Y;
#else
    const uint y = get_global_id(1);
    const uint z = 0;
#endif
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif

    ACCUMULATOR_TYPE dotProd = (ACCUMULATOR_TYPE)0;
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
#if  OUTPUT_DIMS > 4
    const int input_z = z * STRIDE_SIZE_Z - PADDING_SIZE_Z;
#else
    const int input_z = 0;
#endif

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
        for (uint l = 0; l < FILTER_SIZE_Z ; ++l)
        {
            const int input_offset_z = input_z + l * DILATION_SIZE_Z;
            const bool zero_z = input_offset_z >= INPUT0_SIZE_Z || input_offset_z < 0;

            if(!zero_z)
            {
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
                                uint input_idx =
                                    GET_DATA_INDEX_5D(
                                        INPUT0, b, k, input_offset_z, input_offset_y, input_offset_x)
                                    + in_split_offset;
                                uint filter_idx = GET_FILTER_INDEX_5D(FILTER, f, k, l, j, i);
#if GROUPED && !DEPTHWISE_SEPARABLE_OPT
                                filter_idx += split_idx * FILTER_LENGTH;
#endif
#ifdef LOCAL_CONVOLUTION
                                filter_idx += FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z
                                    * (x + OUTPUT_SIZE_X * y + OUTPUT_SIZE_X * OUTPUT_SIZE_Y * z);
#endif
                                dotProd += TO_ACCUMULATOR_TYPE(conv_input[input_idx]) * TO_ACCUMULATOR_TYPE(weights[filter_idx]);
                            }
                        }
                    }
                }
            }
        }
    }

#if BIAS_TERM
    #if GROUPED && !DEPTHWISE_SEPARABLE_OPT
        const uint bias_offset = split_idx * BIAS_LENGTH;
    #else
        const uint bias_offset = 0;
    #endif
    #if   BIAS_PER_OUTPUT
        const uint bias_index = bias_offset + GET_DATA_INDEX_5D(BIAS, b, f, z, y, x);
    #elif BIAS_PER_OFM
        const uint bias_index = bias_offset + f;
    #endif

    #if !DONT_DEQUANTIZE_BIAS
        dotProd += TO_ACCUMULATOR_TYPE(biases[bias_index]);
    #endif
#endif

    ACTIVATION_TYPE dequantized = dotProd;

#if QUANTIZATION_TERM
    // TODO: Do per-channel and per-tensor dequantization coeffecients
    // happen to co-exist really?
    #if GROUPED && !DEPTHWISE_SEPARABLE_OPT
        const uint quantization_offset = split_idx * OUTPUT_FEATURE_NUM;
    #else
        const uint quantization_offset = 0;
    #endif

    dequantized *= quantizations[quantization_offset + f] * I_QF;
#endif

#if DONT_DEQUANTIZE_BIAS
    #if !BIAS_TERM || !QUANTIZATION_TERM
        #error "DONT_DEQUANTIZE_BIAS is meaningless without BIAS_TERM and QUANTIZATION_TERM"
    #endif
    // assert(BIAS_TYPE == float);
    dequantized += biases[bias_index];
#endif

    ACTIVATION_TYPE after_activation =
        ACTIVATION_CONV_TYPED(ACTIVATION_TYPE_BASE, dequantized, ACTIVATION_PARAMS_CONV_TYPED);

#if CALIBRATION_TERM
    #if GROUPED && !DEPTHWISE_SEPARABLE_OPT
        const uint calibration_offset = split_idx * OUTPUT_FEATURE_NUM;
    #else
        const uint calibration_offset = 0;
    #endif

    ACTIVATION_TYPE after_output_calibration = after_activation * calibrations[calibration_offset + f];
#elif defined(O_QF)
    ACTIVATION_TYPE after_output_calibration = after_activation * O_QF;
#else
    ACTIVATION_TYPE after_output_calibration = after_activation;
#endif

    after_output_calibration = AFTER_CALIBRATION_ROUND(after_output_calibration);

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = GET_DATA_INDEX_5D(OUTPUT, b, f, z, y, x) + out_split_offset;

#if !defined(ACTIVATION_ELTW_TYPED)
    output[dst_index] = TO_OUTPUT_TYPE_SAT(after_output_calibration);
#else

#    if IN_OUT_OPT == 1
    OUTPUT_TYPE eltw_elem = output[dst_index];
#    else
    INPUT1_TYPE eltw_elem = eltw_input[GET_DATA_INDEX_5D(INPUT1, b, f, z * ELTW_STRIDE_Z, y * ELTW_STRIDE_Y, x * ELTW_STRIDE_X)];
#    endif

#    if defined(NON_CONV_SCALE)
    float eltw_elem_scaled = (float)eltw_elem * NON_CONV_SCALE;
#    else
    // Saturation isn't needed here probably. However, this is not a performant
    // kernel, so better be safe. A slightly better way would be to assert about
    // precision somehow, but it's not going to be easy/elegant in OpenCL :(
    ACTIVATION_TYPE eltw_elem_scaled = CAT(CAT(TO_, ACTIVATION_TYPE_BASE), _TYPE_SAT)(eltw_elem);
#    endif

    // TODO: Support other eltwise operations.
    ACTIVATION_TYPE before_eltw_activation = after_output_calibration + eltw_elem_scaled;
    ACTIVATION_TYPE after_eltw_activation =
        ACTIVATION_ELTW_TYPED(
            ACTIVATION_TYPE_BASE,
            before_eltw_activation,
            ACTIVATION_PARAMS_ELTW_TYPED);

    after_eltw_activation =
        AFTER_ELTW_CALIBRATION_ROUND(after_eltw_activation
#    if ELTW_CALIBRATION_TERM
                                     * eltw_output_calibrations[f]
#    endif
        );

    output[dst_index] = TO_OUTPUT_TYPE_SAT(after_eltw_activation);
#endif
}

#if defined(O_QF_LOCAL_DEFINE)
#    undef O_QF
#    undef O_QF_LOCAL_DEFINE
#endif

#if defined(ACTIVATION_ELTW_TYPED)
#    undef AFTER_CELTW_CALIBRATION_ROUND
#endif

#undef AFTER_CALIBRATION_ROUND
#undef ACTIVATION_TYPE_BASE
#undef ACTIVATION_TYPE
#undef TO_ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
