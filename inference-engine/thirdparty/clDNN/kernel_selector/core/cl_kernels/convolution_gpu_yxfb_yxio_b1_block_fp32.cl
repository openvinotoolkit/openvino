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
#include "include/sub_group.cl"

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(convolution_gpu_yxfb_yxio_b1_block)(
    const __global float* input,
    __global float* output,
    const __global float* filter,
#if BIAS_TERM
    const __global float* bias,
#endif
    uint split_idx)
{
#ifdef USE_VECTOR_8
    #define VECTOR_FLOAT float8
    #define BLOCK_READ(IN) as_float8(intel_sub_group_block_read8((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) intel_sub_group_block_write8((__global uint*)OUT, as_uint8(DATA));
#endif
#ifdef USE_VECTOR_4
    #define VECTOR_FLOAT float4
    #define BLOCK_READ(IN) as_float4(intel_sub_group_block_read4((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) intel_sub_group_block_write4((__global uint*)OUT, as_uint4(DATA));
#endif
#ifdef USE_VECTOR_2
    #define VECTOR_FLOAT float2
    #define BLOCK_READ(IN) as_float2(intel_sub_group_block_read2((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) intel_sub_group_block_write2((__global uint*)OUT, as_uint2(DATA));
#endif

    const uint batch_num = INPUT0_BATCH_NUM;
    const uint linear_id_xy = get_group_id(1) + get_global_size(1) * get_group_id(2);
    uint global_id = (((uint)get_group_id(0) * LOCAL_WORK_GROUP_SIZE) / batch_num) * batch_num + (linear_id_xy * FILTER_ARRAY_NUM + split_idx) * (FILTER_OFM_NUM / OFM_PER_WORK_ITEM) * batch_num;

    const uint out_batch_id = (uint)get_local_id(0) % INPUT0_BATCH_NUM;
    const uint out_x = get_group_id(1);
    const uint out_y = get_group_id(2);

    const uint out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

    const uint ofm_offset = (global_id * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OFM_NUM;

    const uint sub_group_id = (uint)get_local_id(0) % INPUT0_BATCH_NUM;

    VECTOR_FLOAT _data0 = 0.f;

    const int x = (int)out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int y = (int)out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    for (uint i = 0; i < FILTER_SIZE_Y; i++)
    {
        const int input_offset_y = y + i * DILATION_SIZE_Y;
        const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

        if(!zero_y)
        {
            for (uint j = 0; j < FILTER_SIZE_X; j++)
            {
                const int input_offset_x = x + j * DILATION_SIZE_X;
                const bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                if(!zero)
                {
                    uint input_idx = input_offset_x*INPUT0_X_PITCH + input_offset_y*INPUT0_Y_PITCH;
                    input_idx += INPUT0_OFFSET + split_idx * FILTER_IFM_NUM * INPUT0_FEATURE_PITCH;
                    input_idx += out_batch_id;

                    uint filter_idx = ofm_offset + sub_group_id + i*FILTER_Y_PITCH + j*FILTER_X_PITCH;

#if INPUT0_BATCH_NUM == 1
                    for(uint h = 0; h < FILTER_IFM_NUM / 8; h++)
                    {
                        float _in = as_float(intel_sub_group_block_read((const __global uint*)input + input_idx));
                        float8 _input = TRANSPOSE_BLOCK_8(_in);

                        VECTOR_FLOAT _filter;
                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s0, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s1, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s2, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s3, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s4, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s5, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s6, _filter, _data0);

                        _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(_input.s7, _filter, _data0);

                        input_idx += 8 * INPUT0_FEATURE_PITCH;
                    }
                    for (uint h = FILTER_IFM_NUM - (FILTER_IFM_NUM % 8); h < FILTER_IFM_NUM; h++)
#else
                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
#endif
                    {
                        VECTOR_FLOAT _filter = BLOCK_READ(filter + filter_idx);
                        _data0 = mad(input[input_idx], _filter, _data0);
                        filter_idx += FILTER_IFM_PITCH;
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
                }
            }
        }
    }

#if BIAS_TERM
    _data0 += BLOCK_READ(bias + ofm_offset);
#endif
    _data0 = ACTIVATION(_data0, NL_M, NL_N);

    uint _out_id = OUTPUT_OFFSET + out_id;
    BLOCK_WRITE(output + _out_id, _data0);
#if defined(USE_VECTOR_8) || defined(USE_VECTOR_4) || defined(USE_VECTOR_2)
    #undef VECTOR_FLOAT
    #undef BLOCK_READ
    #undef BLOCK_WRITE
#endif
}