/*
// Copyright (c) 2019 Intel Corporation
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
*/

#include "include/include_all.cl"
#include "include/unit_type.cl"

// Kernel works only for sub_group size of 16 with 32 features slice size and process 2 features per WI
#define REQD_SUB_GROUP_SIZE 16
#define REQD_FEATURE_SLICE_SIZE 32
#define REQD_FEATURES_PER_WORK_ITEM 2

//inputs_decls -> __global unit_type * input0, __global unit_type * input1

__attribute__((intel_reqd_sub_group_size(REQD_SUB_GROUP_SIZE)))
KERNEL(eltwise_mixed_byxf_and_fs_b_yx_fsv32)(
    INPUTS_DECLS
    __global UNIT_TYPE* output)
{
    const uint x   = get_global_id(0);
    const uint y   = get_global_id(1);
    const uint bf  = (uint)get_global_id(2);
    const uint bfs = bf / (REQD_FEATURE_SLICE_SIZE / REQD_FEATURES_PER_WORK_ITEM);

    const uint b  = bfs % INPUT0_BATCH_NUM;
    const uint fs = bfs / INPUT0_BATCH_NUM;
    const uint f0 = fs * REQD_FEATURE_SLICE_SIZE; //number of first feature in slice

    const uint input_0_offset = GET_INPUT_0_DATA_INDEX(b,f0,y,x);
    const uint input_1_offset = GET_INPUT_1_DATA_INDEX(b,f0,y,x);
    const uint output_offset  = GET_OUTPUT_DATA_INDEX(b,f0,y,x);

    UNIT_TYPE2 in1;
    UNIT_TYPE2 in2;
    UNIT_TYPE2 out;

    in1 = UNIT_BLOCK_READ2(input0,input_0_offset);
    in2 = UNIT_BLOCK_READ2(input1,input_1_offset);

    { 
        const UNIT_TYPE tmp_input_0 = in1.s0;
        const UNIT_TYPE tmp_input_1 = in2.s0;
        OPERATION0;
        out.s0 = tmp0;
    }
    {
        const UNIT_TYPE tmp_input_0 = in1.s1;
        const UNIT_TYPE tmp_input_1 = in2.s1;
        OPERATION0;
        out.s1 = tmp0;
    }

    out = ACTIVATION(out, ACTIVATION_PARAMS);

    UNIT_BLOCK_WRITE2(output,output_offset,out);
}



