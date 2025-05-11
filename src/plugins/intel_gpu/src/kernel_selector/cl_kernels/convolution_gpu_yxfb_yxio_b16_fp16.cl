// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/sub_group.cl"

REQD_SUB_GROUP_SIZE(16)
__attribute__((reqd_work_group_size(16, 1, 1)))
KERNEL(convolution_gpu_yxfb_yxio_b16)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter
#if BIAS_TERM
    , const __global UNIT_TYPE* bias
#endif
)
{
    // get_global_size(0) -> Number of work items needed to compute all features and all batches for single output spatial position
    //                       (single (x, y) point in output).
    // get_global_size(1) -> Output size in X-dimension.
    // get_global_size(2) -> Output size in Y-dimension.
    // get_global_id(0)   -> Id of work item computing single spatial point of output indicated by get_global_id(1), get_global_id(2).
    // get_group_id(1)   -> Current x-position in output.
    // get_group_id(2)   -> Current y-position in output.
    //
    // WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS -> Number of work items needed to compute entire one batch for at least one feature and one spatial point.
    //                                           (this number in current implementation computes also OFM_PER_WORK_ITEM output features at the same time).
    // FILTER_ARRAY_NUM                       -> Number of filters groups (split size).

    const uint out_x = get_group_id(1);
    const uint out_y = get_group_id(2);

    const uint output_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_x_size = OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X;
    const uint linear_id_xy = OUTPUT_PAD_BEFORE_SIZE_X + out_x + output_x_size * (out_y + OUTPUT_PAD_BEFORE_SIZE_Y);
    const uint of = (uint)get_global_id(0) / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS;
#if GROUPED
    const uint g = of / (FILTER_OFM_NUM / OFM_PER_WORK_ITEM);
    const uint f = of % (FILTER_OFM_NUM / OFM_PER_WORK_ITEM);
#else
    const uint g = 0;
    const uint f = of;
#endif
    const uint global_id = (f + linear_id_xy * (output_f_size / OFM_PER_WORK_ITEM)) * WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS;

    const uint sub_group_id = get_local_id(0);

#if defined(USE_BLOCK_READ_2) || defined(USE_BLOCK_READ_1)
    const uint chunk_size = sizeof(uint)/sizeof(UNIT_TYPE);
#else
    const uint chunk_size = 1;
#endif

    const uint out_batch_id = chunk_size * sub_group_id + LOCAL_WORK_GROUP_SIZE * BATCHES_PER_WORK_ITEM * ((uint)get_group_id(0) % LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS);

    const uint out_id = (g*FILTER_OFM_NUM + (global_id / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) * OFM_PER_WORK_ITEM) * OUTPUT_FEATURE_PITCH + OUTPUT_PAD_BEFORE_FEATURE_NUM * OUTPUT_FEATURE_PITCH + OUTPUT_PAD_BEFORE_BATCH_NUM + out_batch_id;

    const uint ofm_offset = g*FILTER_GROUPS_PITCH + ((global_id * OFM_PER_WORK_ITEM) / WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS) % output_f_size;

    // Each component of vector element contains computation for separate output feature.
    half16 _data[BATCHES_PER_WORK_ITEM];
    for(uint i = 0; i < BATCHES_PER_WORK_ITEM; i++)
    {
        _data[i] = UNIT_VAL_ZERO;
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
                const bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                if(!zero)
                {
                    uint input_idx = input_offset_x*INPUT0_X_PITCH + input_offset_y*INPUT0_Y_PITCH;
                    input_idx += INPUT0_OFFSET + g * FILTER_IFM_NUM * INPUT0_FEATURE_PITCH;
                    input_idx += out_batch_id;

                    //sub_group_id used as offset to make each workitem load different filter, and then shuffle it
                    // 2 * sub_group_id is used because we group 2 halfs as one uint element.
                    uint filter_idx = ofm_offset + 2*sub_group_id + i*FILTER_Y_PITCH + j*FILTER_X_PITCH;

                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
                    {
#if defined(USE_BLOCK_READ_2)
                        half4 _input = as_half4(_sub_group_block_read2((const __global uint*)(input + input_idx)));
                        uint filter_val_pair = *(const __global uint*)(filter + filter_idx);
                        half16 filter_transp = TRANSPOSE_BLOCK_16_FP16(filter_val_pair);
                        _data[0] = fma(_input.s0, filter_transp, _data[0]);
                        _data[1] = fma(_input.s1, filter_transp, _data[1]);
                        _data[2] = fma(_input.s2, filter_transp, _data[2]);
                        _data[3] = fma(_input.s3, filter_transp, _data[3]);
                        input_idx += INPUT0_FEATURE_PITCH;
#elif defined(USE_BLOCK_READ_1)
                        half2 _input = as_half2(_sub_group_block_read((const __global uint*)(input + input_idx)));
                        uint filter_val_pair = *(const __global uint*)(filter + filter_idx);
                        half16 filter_transp = TRANSPOSE_BLOCK_16_FP16(filter_val_pair);
                        _data[0] = fma(_input.s0, filter_transp, _data[0]);
                        _data[1] = fma(_input.s1, filter_transp, _data[1]);
                        input_idx += INPUT0_FEATURE_PITCH;
#else
                        uint filter_val_pair = *(const __global uint*)(filter + filter_idx);
                        half16 filter_transp = TRANSPOSE_BLOCK_16_FP16(filter_val_pair);
                        for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
                        {
                            _data[s] = fma(input[input_idx], filter_transp, _data[s]);
                            input_idx += LOCAL_WORK_GROUP_SIZE;
                        }
                        input_idx += INPUT0_FEATURE_PITCH - BATCHES_PER_WORK_ITEM * LOCAL_WORK_GROUP_SIZE;
#endif
                        filter_idx += FILTER_IFM_PITCH;
                    }
                }
            }
        }
    }

#if BIAS_TERM
    uint bias_val_pair = *(const __global uint*)(bias + (ofm_offset + 2 * sub_group_id));
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        ADD_BIAS_16_FP16(_data[s], bias_val_pair);
    }
#endif
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        _data[s] = ACTIVATION(_data[s], ACTIVATION_PARAMS);
    }

#if defined(USE_BLOCK_READ_2) || defined(USE_BLOCK_READ_1)
    #if BATCHES_PER_WORK_ITEM == 4
        uint _out_id = OUTPUT_VIEW_OFFSET + out_id;
        for(uint i = 0; i < 16; i++)
        {
            *(__global uint*)(output + _out_id) = as_uint((half2)(_data[0][i], _data[1][i]));
            *(__global uint*)(output + _out_id + 32) = as_uint((half2)(_data[2][i], _data[3][i]));
            _out_id += OUTPUT_FEATURE_PITCH;
        }
    #else
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM / 2; s++)
    {
        uint _out_id = OUTPUT_VIEW_OFFSET + out_id + chunk_size * s * LOCAL_WORK_GROUP_SIZE;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s0, _data[chunk_size * s + 1].s0)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s1, _data[chunk_size * s + 1].s1)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s2, _data[chunk_size * s + 1].s2)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s3, _data[chunk_size * s + 1].s3)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s4, _data[chunk_size * s + 1].s4)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s5, _data[chunk_size * s + 1].s5)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s6, _data[chunk_size * s + 1].s6)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s7, _data[chunk_size * s + 1].s7)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s8, _data[chunk_size * s + 1].s8)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].s9, _data[chunk_size * s + 1].s9)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].sa, _data[chunk_size * s + 1].sa)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].sb, _data[chunk_size * s + 1].sb)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].sc, _data[chunk_size * s + 1].sc)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].sd, _data[chunk_size * s + 1].sd)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].se, _data[chunk_size * s + 1].se)); _out_id += OUTPUT_FEATURE_PITCH;
        *(__global uint*)(output + _out_id) = as_uint((half2)(_data[chunk_size * s].sf, _data[chunk_size * s + 1].sf)); _out_id += OUTPUT_FEATURE_PITCH;
    }
    #endif
#else
    for(uint s = 0; s < BATCHES_PER_WORK_ITEM; s++)
    {
        uint _out_id = OUTPUT_VIEW_OFFSET + out_id + s * LOCAL_WORK_GROUP_SIZE;
        output[_out_id] = _data[s].s0; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s1; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s2; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s3; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s4; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s5; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s6; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s7; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s8; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].s9; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].sa; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].sb; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].sc; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].sd; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].se; _out_id += OUTPUT_FEATURE_PITCH;
        output[_out_id] = _data[s].sf; _out_id += OUTPUT_FEATURE_PITCH;
    }
#endif
}
