// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/sub_group.cl"

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(convolution_gpu_yxfb_yxio_b1_block_multiple_x)(
    const __global float* input,
    __global float* output,
    const __global float* filter
#if BIAS_TERM
    , const __global float* bias
#endif
)
{
#if USE_VECTOR == 8
    #define VECTOR_FLOAT float8
    #define BLOCK_READ(IN) as_float8(_sub_group_block_read8((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) _sub_group_block_write8((__global uint*)OUT, as_uint8(DATA));
#endif
#if USE_VECTOR == 4
    #define VECTOR_FLOAT float4
    #define BLOCK_READ(IN) as_float4(_sub_group_block_read4((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) _sub_group_block_write4((__global uint*)OUT, as_uint4(DATA));
#endif
#if USE_VECTOR == 2
    #define VECTOR_FLOAT float2
    #define BLOCK_READ(IN) as_float2(_sub_group_block_read2((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) _sub_group_block_write2((__global uint*)OUT, as_uint2(DATA));
#endif
#if USE_VECTOR == 1
    #define VECTOR_FLOAT float
    #define BLOCK_READ(IN) as_float(_sub_group_block_read((const __global uint*)IN))
    #define BLOCK_WRITE(OUT, DATA) _sub_group_block_write((__global uint*)OUT, as_uint(DATA));
#endif

    const uint batch_num = INPUT0_BATCH_NUM;
    const uint linear_id_xy = (uint)get_group_id(1) * X_PER_WORK_ITEM + OUTPUT_SIZE_X * (uint)get_group_id(2);
    uint global_id = (((uint)get_group_id(0) * LOCAL_WORK_GROUP_SIZE) / batch_num) * batch_num + ( linear_id_xy * FILTER_ARRAY_NUM ) * (FILTER_OFM_NUM / OFM_PER_WORK_ITEM) * batch_num;

    const uint out_batch_id = (uint)get_local_id(0) % INPUT0_BATCH_NUM;
    const uint out_x = (uint)get_group_id(1) * X_PER_WORK_ITEM;
    const uint out_y = get_group_id(2);

    uint out_id[X_PER_WORK_ITEM];
    for(uint i = 0; i < X_PER_WORK_ITEM; i++)
    {
        out_id[i] = OUTPUT_OFFSET + ( (global_id + i * FILTER_ARRAY_NUM * (FILTER_OFM_NUM / OFM_PER_WORK_ITEM) * INPUT0_BATCH_NUM) / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;
    }

    const uint ofm_offset = (global_id * (OFM_PER_WORK_ITEM / batch_num)) % FILTER_OFM_NUM;

    const uint sub_group_id = (uint)get_local_id(0) % INPUT0_BATCH_NUM;

    VECTOR_FLOAT _data[X_PER_WORK_ITEM];
    for(uint i = 0; i < X_PER_WORK_ITEM; i++)
    {
        _data[i] = 0.0f;
    }

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

                bool zero_x[X_PER_WORK_ITEM];
                for(int z = 0; z < X_PER_WORK_ITEM; z++)
                {
                    zero_x[z] = (input_offset_x + z * STRIDE_SIZE_X) >= INPUT0_SIZE_X || (input_offset_x + z * STRIDE_SIZE_X) < 0;
                }

                VECTOR_FLOAT _tmp[X_PER_WORK_ITEM];
                for(uint t = 0; t < X_PER_WORK_ITEM; t++)
                {
                    _tmp[t] = 0.f;
                }

                uint input_idx = input_offset_x*INPUT0_X_PITCH + input_offset_y*INPUT0_Y_PITCH;
                input_idx += INPUT0_OFFSET;
                input_idx += out_batch_id;

                uint filter_idx = ofm_offset + sub_group_id + i*FILTER_Y_PITCH + j*FILTER_X_PITCH;

#if FILTER_IFM_NUM >= 8
                for(uint h = 0; h < FILTER_IFM_NUM / 8; h++)
                {
                    float _in[X_PER_WORK_ITEM];
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _in[a] = as_float(_sub_group_block_read((const __global uint*)input + (input_idx + a * INPUT0_FEATURE_NUM * STRIDE_SIZE_X)));
                    }
                    float8 _input[X_PER_WORK_ITEM];
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _input[a] = TRANSPOSE_BLOCK_8(_in[a]);
                    }

                    VECTOR_FLOAT _filter;
                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s0, _filter, _tmp[a]);
                    }

                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s1, _filter, _tmp[a]);
                    }

                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s2, _filter, _tmp[a]);
                    }

                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s3, _filter, _tmp[a]);
                    }


                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s4, _filter, _tmp[a]);
                    }

                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s5, _filter, _tmp[a]);
                    }

                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s6, _filter, _tmp[a]);
                    }

                    _filter = BLOCK_READ(filter + filter_idx); filter_idx += FILTER_OFM_NUM;
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(_input[a].s7, _filter, _tmp[a]);
                    }

                    input_idx += 8 * INPUT0_FEATURE_PITCH;
                }
                for (uint h = FILTER_IFM_NUM - (FILTER_IFM_NUM % 8); h < FILTER_IFM_NUM; h++)
#else
                for (uint h = 0; h < FILTER_IFM_NUM; h++)
#endif
                {
                    VECTOR_FLOAT _filter = BLOCK_READ(filter + filter_idx);
                    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                    {
                        _tmp[a] = mad(input[input_idx + a * INPUT0_FEATURE_NUM * STRIDE_SIZE_X], _filter, _tmp[a]);
                    }
                    filter_idx += FILTER_IFM_PITCH;
                    input_idx += INPUT0_FEATURE_PITCH;
                }
                for(uint a = 0; a < X_PER_WORK_ITEM; a++)
                {
                    if(!zero_x[a])
                        _data[a] += _tmp[a];
                }
            }
        }
    }

#if BIAS_TERM
    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
    {
        _data[a] += BLOCK_READ(bias + ofm_offset);
    }
#endif
    for(uint a = 0; a < X_PER_WORK_ITEM; a++)
    {
        _data[a] = ACTIVATION(_data[a], ACTIVATION_PARAMS);
    }

    BLOCK_WRITE(output + out_id[0], _data[0]);
    for(uint a = 1; a < X_PER_WORK_ITEM; a++)
    {
        if(out_x + a < OUTPUT_SIZE_X)
        {
            BLOCK_WRITE(output + out_id[a], _data[a]);
        }
    }

#if defined(USE_VECTOR)
    #undef VECTOR_FLOAT
    #undef BLOCK_READ
    #undef BLOCK_WRITE
#endif
}
