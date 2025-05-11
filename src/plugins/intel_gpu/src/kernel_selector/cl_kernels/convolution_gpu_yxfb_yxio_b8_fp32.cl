// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/sub_group.cl"

__attribute__((reqd_work_group_size(LOCAL_WORK_GROUP_SIZE, 1, 1)))
KERNEL(convolution_gpu_yxfb_yxio_b8)(
    const __global float* input,
    __global float* output,
    const __global float* filter
#if BIAS_TERM
    , const __global float* bias
#endif
)
{
    const uint batch_num = INPUT0_BATCH_NUM;

    const uint linear_id_xy = (uint)get_global_id(1) + (uint)get_global_size(1) * (uint)get_global_id(2);
    // we're computing 8 OUTPUT_FEATURE_MAP so we must divide by 8, but we got 8 batches, so no division is needed.
    uint global_id = ((uint)get_global_id(0) / batch_num) * batch_num + (linear_id_xy * FILTER_ARRAY_NUM) * (FILTER_OFM_NUM / OFM_PER_WORK_ITEM) * batch_num;

    const uint out_batch_id = get_local_id(0);
    const uint out_x = get_global_id(1);
    const uint out_y = get_global_id(2);

    const uint out_id = (global_id / batch_num) * OFM_PER_WORK_ITEM * batch_num + out_batch_id;

    const uint ofm_offset = (global_id * OFM_PER_WORK_ITEM) / batch_num % FILTER_OFM_NUM;

    const uint sub_group_id = get_local_id(0);

    float8 _data0 = 0.f;
#if OFM_PER_WORK_ITEM == 16
    float8 _data1 = 0.f;
#endif

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
                    input_idx += INPUT0_OFFSET;
                    input_idx += out_batch_id;

                    //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                    uint filter_idx = ofm_offset + sub_group_id + i*FILTER_Y_PITCH + j*FILTER_X_PITCH;
#if OFM_PER_WORK_ITEM == 16
                    uint filter_idx2 = filter_idx + 8;
#endif
                    for (uint h = 0; h < FILTER_IFM_NUM / 8; h++)
                    {
                        float8 _input = as_float8(_sub_group_block_read8((const __global uint*)input + input_idx));

                        DOT_PRODUCT_8(_data0, _input.s0, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s0, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s1, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s1, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s2, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s2, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s3, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s3, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s4, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s4, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s5, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s5, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s6, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s6, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        DOT_PRODUCT_8(_data0, _input.s7, filter[filter_idx]) filter_idx += FILTER_OFM_NUM;
#if OFM_PER_WORK_ITEM == 16
                        DOT_PRODUCT_8(_data1, _input.s7, filter[filter_idx2]) filter_idx2 += FILTER_OFM_NUM;
#endif
                        input_idx += 8 * INPUT0_FEATURE_PITCH;
                    }
                    for (uint h = FILTER_IFM_NUM - (FILTER_IFM_NUM % 8); h < FILTER_IFM_NUM; h++)
                    {
                        float8 _filter = TRANSPOSE_BLOCK_8(filter[filter_idx]); filter_idx += FILTER_OFM_NUM;
                        _data0 = mad(input[input_idx], _filter, _data0);
#if OFM_PER_WORK_ITEM == 16
                        float8 _filter2 = TRANSPOSE_BLOCK_8(filter[filter_idx2]); filter_idx2 += FILTER_OFM_NUM;
                        _data1 = mad(input[input_idx], _filter2, _data1);
#endif
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
                }
            }
        }
    }

#if BIAS_TERM
    ADD_BIAS_8(_data0, bias[ofm_offset + sub_group_id]);
#if OFM_PER_WORK_ITEM == 16
    ADD_BIAS_8(_data1, bias[ofm_offset + sub_group_id + 8]);
#endif
#endif // #if BIAS_TERM
    _data0 = ACTIVATION(_data0, ACTIVATION_PARAMS);
#if OFM_PER_WORK_ITEM == 16
    _data1 = ACTIVATION(_data1, ACTIVATION_PARAMS);
#endif

    const uint _out_id = OUTPUT_OFFSET + out_id;
    _sub_group_block_write8((__global uint*)output + _out_id, as_uint8(_data0));
#if OFM_PER_WORK_ITEM == 16
    _sub_group_block_write8((__global uint*)output + _out_id + 8 * INPUT0_FEATURE_PITCH, as_uint8(_data1));
#endif
}
