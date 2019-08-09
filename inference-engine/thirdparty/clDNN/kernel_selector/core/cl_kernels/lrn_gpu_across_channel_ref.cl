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

#include "include/common.cl"
#include "include/data_types.cl"


#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (lrn_gpu_across_channel_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
#if   defined OUTPUT_LAYOUT_BFYX
    const uint x            = get_global_id(0);    
    const uint y            = get_global_id(1);
    const uint b_f          = get_global_id(2);
    const uint batch_id     = b_f / INPUT0_FEATURE_NUM;
    const uint feature_id   = b_f % INPUT0_FEATURE_NUM;
    
    if (x >= INPUT0_SIZE_X)
        return;
#else
    const uint b_f          = get_global_id(0);
    const uint x            = (uint)get_global_id(1);
    const uint y            = (uint)get_global_id(2);
    const uint feature_id   = b_f / INPUT0_BATCH_NUM;
    const uint batch_id     = b_f % INPUT0_BATCH_NUM;
#endif    

    const uint input_id = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

    UNIT_TYPE acc = UNIT_VAL_ZERO;

    int input_offset_f = feature_id - PADDING;
    int input_idx = (int)input_id - PADDING*INPUT0_FEATURE_PITCH;

    for (int i = 0; i < LOCAL_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;

        UNIT_TYPE value = zero ? UNIT_VAL_ZERO : UNIT_CVT_FUNC(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * input[input_idx];
        acc = mad(value, value, acc);

        input_offset_f++;
        input_idx += INPUT0_FEATURE_PITCH;
    }
    acc = mad(acc, UNIT_CVT_FUNC(ALPHA_DIV_BY_SIZE), UNIT_CVT_FUNC(K));
    acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

    const uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    output[output_idx] = ACTIVATION(acc * input[input_id], ACTIVATION_PARAMS);
}

#undef UNIT_CVT_FUNC
