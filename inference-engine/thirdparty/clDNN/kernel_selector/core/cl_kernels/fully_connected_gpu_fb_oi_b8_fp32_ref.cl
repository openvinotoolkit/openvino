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


__attribute__((reqd_work_group_size(8, 1, 1)))
KERNEL (fully_connected_gpu_xb_bx_b8)(
    const __global float* input,
    __global float* output,
    const __global float* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias)
#else
    )
#endif
{
    const uint batch_id = get_global_id(0);

    uint outXIdx = get_global_id(1);
    uint weight_offset = outXIdx * INPUT0_ELEMENTS_COUNT + batch_id;
#if BIAS_TERM
    float result = bias[outXIdx];
#else
    float result = 0.0f;
#endif

    float8 _data = 0.f;
    const uint sub_group_id = get_local_id(0);

    for(uint _i = 0; _i < INPUT0_ELEMENTS_COUNT/8; _i++)
    {
        uint i = _i * 8;
        const float weight_val = weight[weight_offset];
        const float8 _input = as_float8(intel_sub_group_block_read8((const __global uint*)input + i * INPUT0_BATCH_NUM + batch_id));
        _data.s0 = fma(_input.s0, intel_sub_group_shuffle(weight_val, 0), _data.s0);
        _data.s1 = fma(_input.s1, intel_sub_group_shuffle(weight_val, 1), _data.s1);
        _data.s2 = fma(_input.s2, intel_sub_group_shuffle(weight_val, 2), _data.s2);
        _data.s3 = fma(_input.s3, intel_sub_group_shuffle(weight_val, 3), _data.s3);
        _data.s4 = fma(_input.s4, intel_sub_group_shuffle(weight_val, 4), _data.s4);
        _data.s5 = fma(_input.s5, intel_sub_group_shuffle(weight_val, 5), _data.s5);
        _data.s6 = fma(_input.s6, intel_sub_group_shuffle(weight_val, 6), _data.s6);
        _data.s7 = fma(_input.s7, intel_sub_group_shuffle(weight_val, 7), _data.s7);
        weight_offset += 8;
    }
    for(uint i = INPUT0_ELEMENTS_COUNT - (INPUT0_ELEMENTS_COUNT % 8); i < INPUT0_ELEMENTS_COUNT; i++)
    {
        result += input[i * INPUT0_BATCH_NUM + batch_id] * weight[weight_offset++];
    }
    result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
              _data.s4 + _data.s5 + _data.s6 + _data.s7;

    output[outXIdx * INPUT0_BATCH_NUM + batch_id] = ACTIVATION(result, NL_M, NL_N);
}
