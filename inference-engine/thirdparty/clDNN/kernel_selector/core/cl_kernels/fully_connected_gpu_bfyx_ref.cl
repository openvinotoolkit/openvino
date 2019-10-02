// Copyright (c) 2016-2017 Intel Corporation
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

#include "include/include_all.cl"

KERNEL(fc)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if QUANTIZATION_TERM
    ,const __global float* quantizations
#endif
#if CALIBRATION_TERM
    ,const __global float* calibrations
#endif
    )
{
    const uint ofm = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint ifm = 0; ifm < INPUT0_FEATURE_NUM; ++ifm)
    {
       for (uint y = 0; y < INPUT0_SIZE_Y; ++y)
       {
           for(uint x = 0; x < INPUT0_SIZE_X; ++x )
           {
               const uint input0_idx = GET_DATA_INDEX(INPUT0, b, ifm, y, x);
               const uint filter_idx = GET_FILTER_INDEX(FILTER, ofm, ifm, y, x);
#if QUANTIZATION_TERM
               dotProd += (ACCUMULATOR_TYPE)input[input0_idx] * (ACCUMULATOR_TYPE)weights[filter_idx];
#else
               dotProd += (ACCUMULATOR_TYPE)(input[input0_idx] * weights[filter_idx]);
#endif
          }
       }
    }
    
    const uint output_idx = GET_DATA_INDEX(OUTPUT, b, ofm, 0, 0);

#if BIAS_TERM
    const uint bias_index = ofm;
#endif

#if BIAS_TERM && !DONT_DEQUANTIZE_BIAS
    dotProd += TO_ACCUMULATOR_TYPE(biases[bias_index]);
#endif

    ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(dotProd);

#if QUANTIZATION_TERM
    dequantized = dequantized * quantizations[ofm] * I_QF;
#endif

#if DONT_DEQUANTIZE_BIAS
    dequantized = dequantized + TO_ACTIVATION_TYPE(biases[bias_index]);
#endif

    dequantized = ACTIVATION_TYPED(ACTIVATION, dequantized, ACTIVATION_PARAMS_TYPED);

#if QUANTIZATION_TERM
#   if CALIBRATION_TERM
    dequantized = dequantized * calibrations[ofm];
#   else
    dequantized = dequantized * O_QF;
#   endif
#endif

    dequantized = AFTER_CALIBRATION_ROUND(dequantized);

    output[output_idx] = TO_OUTPUT_TYPE_SAT(dequantized);
}
