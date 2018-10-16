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

#ifdef FORCE_SIMD_16
__attribute__((intel_reqd_sub_group_size(16)))
#endif
KERNEL (lrn_gpu_across_channel_multiple_features)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if   defined OUTPUT_LAYOUT_BFYX
// PERF NOTE: SIMD IS OVER global_id(0) so in SIMD global_id(1) and global_id(2) does not change, so we can use group_id to have SIMD1 instructions
    const uint x            = get_global_id(0);
    const uint y            = get_group_id(1);
    const uint b_f          = get_group_id(2);
    const uint batch_id     = (b_f * OFM_PER_SIMD) / INPUT0_FEATURE_NUM;
    const uint feature_id   = (b_f % (INPUT0_FEATURE_NUM / OFM_PER_SIMD)) * OFM_PER_SIMD;
    
    if (x >= INPUT0_SIZE_X)
        return;
#elif defined OUTPUT_LAYOUT_YXFB
    const uint b_f          = get_global_id(0);
    const uint x            = get_group_id(1);
    const uint y            = get_group_id(2);
    const uint feature_id   = (b_f / INPUT0_BATCH_NUM) * OFM_PER_SIMD;
    const uint batch_id     = b_f % INPUT0_BATCH_NUM;
#endif    

    uint input_id = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

    int input_offset_f = feature_id - PADDING;
    uint input_idx = input_id - PADDING*INPUT0_FEATURE_PITCH;

    input_idx =  MULTIPLY_OFFSET(UNIT_TYPE, input_idx);

    UNIT_TYPE vals[OFM_PER_SIMD];
    UNIT_TYPE results[OFM_PER_SIMD] = { UNIT_VAL_ZERO };

    // prefetch
    for(uint i = 0; i < OFM_PER_SIMD; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;
        vals[i] = zero ? UNIT_VAL_ZERO : TO_UNIT_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * (*OFFSET_GLOBAL_PTR(UNIT_TYPE, input, input_idx));
        input_offset_f++;
        input_idx += MULTIPLY_OFFSET(UNIT_TYPE, INPUT0_FEATURE_PITCH);
    }

    for (uint i = 0; i < LOCAL_SIZE-1; i++)
    {
        for(uint j = 0; j < OFM_PER_SIMD; j++)
        {
            results[j] = mad(vals[j], vals[j], results[j]);
        }
        for(uint j = 0; j < OFM_PER_SIMD-1; j++)
        {
            vals[j] = vals[j+1];
        }

        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;
        vals[OFM_PER_SIMD-1] = zero ? UNIT_VAL_ZERO : TO_UNIT_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * (*OFFSET_GLOBAL_PTR(UNIT_TYPE, input, input_idx));
        input_offset_f++;
        input_idx += MULTIPLY_OFFSET(UNIT_TYPE, INPUT0_FEATURE_PITCH);
    }

    for(uint j = 0; j < OFM_PER_SIMD; j++)
    {
        results[j] = mad(vals[j], vals[j], results[j]);
    }

    for(uint j = 0; j < OFM_PER_SIMD; j++)
    {
        results[j] = mad(results[j], TO_UNIT_TYPE(ALPHA_DIV_BY_SIZE), TO_UNIT_TYPE(K));
        results[j] = native_powr(results[j], -TO_UNIT_TYPE(BETA));
    }

    uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    for(uint j = 0; j < OFM_PER_SIMD; j++)
    {
        output[output_idx] = ACTIVATION(results[j] * input[input_id], NL_M ,NL_N);
        output_idx += OUTPUT_FEATURE_PITCH;
        input_id += INPUT0_FEATURE_PITCH;
    }
}