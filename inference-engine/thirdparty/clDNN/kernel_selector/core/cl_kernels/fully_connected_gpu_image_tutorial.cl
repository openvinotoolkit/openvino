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

KERNEL(fully_connected_gpu_image_tutorial)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    read_only image2d_t weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
    const uint ofm = get_global_id(0);
    const uint b = get_global_id(1);
    DECLARE_SAMPLER;
    
    ACCUMULATOR_TYPE dotProd = 0;

    for (uint iyx = 0; iyx < (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X + 3) / 4; ++iyx)
    {
        MAKE_VECTOR_TYPE(UNIT_TYPE, 4) weights_val = IMAGE_READ(weights, (int2)(iyx, ofm));
        const uint input0_idx = INPUT0_OFFSET + b * INPUT0_BATCH_PITCH + iyx * 4;
        
        dotProd += (ACCUMULATOR_TYPE)(input[input0_idx] * weights_val.x);
        if(iyx*4 + 1 >= INPUT0_BATCH_PITCH) break;
        dotProd += (ACCUMULATOR_TYPE)(input[input0_idx + 1] * weights_val.y);
        if(iyx*4 + 2 >= INPUT0_BATCH_PITCH) break;
        dotProd += (ACCUMULATOR_TYPE)(input[input0_idx + 2] * weights_val.z);
        if(iyx*4 + 3 >= INPUT0_BATCH_PITCH) break;
        dotProd += (ACCUMULATOR_TYPE)(input[input0_idx + 3] * weights_val.w);
    }
    
    const uint output_idx = GET_DATA_INDEX(OUTPUT, b, ofm, 0, 0);

#if BIAS_TERM
    dotProd += (ACCUMULATOR_TYPE)biases[ofm];
#endif

    output[output_idx] = ACTIVATION((UNIT_TYPE)dotProd, ACTIVATION_PARAMS);
    MAKE_VECTOR_TYPE(UNIT_TYPE, 4) weights_val = IMAGE_READ(weights, (int2)(1, 0));
}