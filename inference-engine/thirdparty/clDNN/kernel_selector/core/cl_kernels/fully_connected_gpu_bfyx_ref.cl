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
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint ofm = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE dotProd = (ACCUMULATOR_TYPE)0;

    for (uint ifm = 0; ifm < INPUT0_FEATURE_NUM; ++ifm)
    {
       for (uint y = 0; y < INPUT0_SIZE_Y; ++y)
       {
           for(uint x = 0; x < INPUT0_SIZE_X; ++x )
           {
               const uint input0_idx = GET_DATA_INDEX(INPUT0, b, ifm, y, x);
               const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, ofm, ifm, y, x);
               dotProd += (ACCUMULATOR_TYPE)(input[input0_idx] * weights[filter_idx]);
          }
       }
    }

    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, ofm, 0, 0);

#if BIAS_TERM
    const uint bias_index = ofm;
    ACTIVATION_TYPE dequantized = dotProd + biases[bias_index];
#else
    ACTIVATION_TYPE dequantized = dotProd;
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[dst_index] = res;
#else
    output[dst_index] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(dequantized, ACTIVATION_PARAMS_TYPED));
#endif

}
