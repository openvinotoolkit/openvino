// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"


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

    for (uint _i = 0; _i < INPUT0_ELEMENTS_COUNT/8; _i++)
    {
        uint i = _i * 8;
        const float weight_val = weight[weight_offset];
        const float8 _input = as_float8(_sub_group_block_read8((const __global uint*)input + i * INPUT0_BATCH_NUM + batch_id));
        _data.s0 = fma(_input.s0, _sub_group_shuffle(weight_val, 0), _data.s0);
        _data.s1 = fma(_input.s1, _sub_group_shuffle(weight_val, 1), _data.s1);
        _data.s2 = fma(_input.s2, _sub_group_shuffle(weight_val, 2), _data.s2);
        _data.s3 = fma(_input.s3, _sub_group_shuffle(weight_val, 3), _data.s3);
        _data.s4 = fma(_input.s4, _sub_group_shuffle(weight_val, 4), _data.s4);
        _data.s5 = fma(_input.s5, _sub_group_shuffle(weight_val, 5), _data.s5);
        _data.s6 = fma(_input.s6, _sub_group_shuffle(weight_val, 6), _data.s6);
        _data.s7 = fma(_input.s7, _sub_group_shuffle(weight_val, 7), _data.s7);
        weight_offset += 8;
    }
    for (uint i = INPUT0_ELEMENTS_COUNT - (INPUT0_ELEMENTS_COUNT % 8); i < INPUT0_ELEMENTS_COUNT; i++)
    {
        result += input[i * INPUT0_BATCH_NUM + batch_id] * weight[weight_offset++];
    }
    result += _data.s0 + _data.s1 + _data.s2 + _data.s3 +
              _data.s4 + _data.s5 + _data.s6 + _data.s7;

    output[outXIdx * INPUT0_BATCH_NUM + batch_id] = ACTIVATION(result, ACTIVATION_PARAMS);
}
