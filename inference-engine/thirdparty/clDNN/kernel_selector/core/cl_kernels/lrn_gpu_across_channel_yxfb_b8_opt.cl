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

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL (lrn_gpu_yxfb_b8)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    
    const uint batch_num_group  = (INPUT0_BATCH_NUM/SUB_GROUP_SIZE);
    const uint b_f              = get_global_id(0);
    const uint x                = (uint)get_global_id(1);
    const uint y                = (uint)get_global_id(2);
    const uint feature_id       = b_f / batch_num_group;
    const uint batch_id_group   = b_f % batch_num_group;
    const uint batch_id         = batch_id_group * SUB_GROUP_SIZE;

    const uint input_id = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;
    const uint input_id_group = input_id / SUB_GROUP_SIZE;

    int input_offset_f = feature_id - PADDING;
    
    const uint input_feature_pitch_group  = (INPUT0_FEATURE_PITCH/SUB_GROUP_SIZE);
    int input_idx_group = (int)input_id_group - PADDING*input_feature_pitch_group;
    
    float8 acc = 0;

    for (int i = 0; i < LOCAL_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;

        if(!zero)
        {
            float8 value = vload8(input_idx_group, input);
            acc = mad(value, value, acc);
        }

        input_offset_f++;
        input_idx_group += input_feature_pitch_group;
    }
    acc = mad(acc, UNIT_CVT_FUNC(ALPHA_DIV_BY_SIZE), UNIT_CVT_FUNC(K));
    acc = native_powr(acc, -UNIT_CVT_FUNC(BETA));

    const uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    const uint output_idx_group = output_idx / SUB_GROUP_SIZE;
    float8 _in = vload8(input_id_group, input);
    float8 res = ACTIVATION(acc * _in, NL_M ,NL_N);
    vstore8(res, output_idx_group, output);
}