// Copyright (c) 2018-2019 Intel Corporation
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
#include "include/unit_type.cl"

#define FEATURE_SLICE_SIZE 16
#define X_BLOCK_SIZE 8

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_depthwise)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    FUSED_OPS_DECLS
    uint split_idx)
{
    const uint yx = get_global_id(0);
    const uint x = (yx % X_BLOCKS) * X_BLOCK_SIZE;
    const uint y = (yx / X_BLOCKS);
    const uint f_block = get_group_id(1);
    const int lid = get_local_id(1);
    const uint b = get_global_id(2);

    const uint filter_offset = f_block * FEATURE_SLICE_SIZE*FILTER_SIZE_X*FILTER_SIZE_Y;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X +  INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y +  INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                              INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch +
                              (f_block + input_fs_pad_before)*input_fs_pitch;
#if BIAS_TERM
    UNIT_TYPE8 dst = (UNIT_TYPE8)(UNIT_BLOCK_READ(biases, f_block * FEATURE_SLICE_SIZE));
#else
    UNIT_TYPE8 dst = (UNIT_TYPE8)(UNIT_VAL_ZERO);
 #endif

    UNIT_TYPE wei_00 = UNIT_BLOCK_READ(weights, filter_offset + 0*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 0*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_01 = UNIT_BLOCK_READ(weights, filter_offset + 0*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 1*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_02 = UNIT_BLOCK_READ(weights, filter_offset + 0*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 2*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_10 = UNIT_BLOCK_READ(weights, filter_offset + 1*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 0*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_11 = UNIT_BLOCK_READ(weights, filter_offset + 1*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 1*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_12 = UNIT_BLOCK_READ(weights, filter_offset + 1*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 2*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_20 = UNIT_BLOCK_READ(weights, filter_offset + 2*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 0*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_21 = UNIT_BLOCK_READ(weights, filter_offset + 2*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 1*FEATURE_SLICE_SIZE);
    UNIT_TYPE wei_22 = UNIT_BLOCK_READ(weights, filter_offset + 2*FILTER_Y_PITCH*FEATURE_SLICE_SIZE + 2*FEATURE_SLICE_SIZE);


#if STRIDE_SIZE_X == 1
    UNIT_TYPE8 src_block_0 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 0)*input_y_pitch + (input_x)*input_x_pitch);
    UNIT_TYPE8 src_block_1 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 1)*input_y_pitch + (input_x)*input_x_pitch);
    UNIT_TYPE8 src_block_2 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 2)*input_y_pitch + (input_x)*input_x_pitch);
    UNIT_TYPE src_tail_00 = UNIT_BLOCK_READ(input, input_offset + (input_y + 0)*input_y_pitch + (input_x + 8)*input_x_pitch);
    UNIT_TYPE src_tail_01 = UNIT_BLOCK_READ(input, input_offset + (input_y + 0)*input_y_pitch + (input_x + 9)*input_x_pitch);
    UNIT_TYPE src_tail_10 = UNIT_BLOCK_READ(input, input_offset + (input_y + 1)*input_y_pitch + (input_x + 8)*input_x_pitch);
    UNIT_TYPE src_tail_11 = UNIT_BLOCK_READ(input, input_offset + (input_y + 1)*input_y_pitch + (input_x + 9)*input_x_pitch);
    UNIT_TYPE src_tail_20 = UNIT_BLOCK_READ(input, input_offset + (input_y + 2)*input_y_pitch + (input_x + 8)*input_x_pitch);
    UNIT_TYPE src_tail_21 = UNIT_BLOCK_READ(input, input_offset + (input_y + 2)*input_y_pitch + (input_x + 9)*input_x_pitch);

    for (int i = 0; i < X_BLOCK_SIZE - 2; i++)
    {
        dst[i] = mad(src_block_0[i + 0], wei_00,dst[i]);
        dst[i] = mad(src_block_0[i + 1], wei_01,dst[i]);
        dst[i] = mad(src_block_0[i + 2], wei_02,dst[i]);

        dst[i] = mad(src_block_1[i + 0], wei_10,dst[i]);
        dst[i] = mad(src_block_1[i + 1], wei_11,dst[i]);
        dst[i] = mad(src_block_1[i + 2], wei_12,dst[i]);

        dst[i] = mad(src_block_2[i + 0], wei_20,dst[i]);
        dst[i] = mad(src_block_2[i + 1], wei_21,dst[i]);
        dst[i] = mad(src_block_2[i + 2], wei_22,dst[i]);
    }
    {
        dst[6] = mad(src_block_0[6], wei_00,dst[6]);
        dst[6] = mad(src_block_0[7], wei_01,dst[6]);
        dst[6] = mad(src_tail_00,    wei_02,dst[6]);

        dst[6] = mad(src_block_1[6], wei_10,dst[6]);
        dst[6] = mad(src_block_1[7], wei_11,dst[6]);
        dst[6] = mad(src_tail_10,    wei_12,dst[6]);

        dst[6] = mad(src_block_2[6], wei_20,dst[6]);
        dst[6] = mad(src_block_2[7], wei_21,dst[6]);
        dst[6] = mad(src_tail_20,    wei_22,dst[6]);
    }

    {
        dst[7] = mad(src_block_0[7], wei_00, dst[7]);
        dst[7] = mad(src_tail_00,    wei_01, dst[7]);
        dst[7] = mad(src_tail_01,    wei_02, dst[7]);

        dst[7] = mad(src_block_1[7], wei_10, dst[7]);
        dst[7] = mad(src_tail_10,    wei_11, dst[7]);
        dst[7] = mad(src_tail_11,    wei_12, dst[7]);

        dst[7] = mad(src_block_2[7], wei_20, dst[7]);
        dst[7] = mad(src_tail_20,    wei_21, dst[7]);
        dst[7] = mad(src_tail_21,    wei_22, dst[7]);
    }

#else
    UNIT_TYPE8 src_block_00 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 0) * input_y_pitch + (input_x)*input_x_pitch);
    UNIT_TYPE8 src_block_01 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 0) * input_y_pitch + (input_x + 8)*input_x_pitch);
    UNIT_TYPE8 src_block_10 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 1) * input_y_pitch + (input_x)*input_x_pitch);
    UNIT_TYPE8 src_block_11 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 1) * input_y_pitch + (input_x + 8)*input_x_pitch);
    UNIT_TYPE8 src_block_20 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 2) * input_y_pitch + (input_x)*input_x_pitch);
    UNIT_TYPE8 src_block_21 = UNIT_BLOCK_READ8(input, input_offset + (input_y + 2) * input_y_pitch + (input_x + 8)*input_x_pitch);
    UNIT_TYPE src_tail_0 = UNIT_BLOCK_READ(input, input_offset + (input_y + 0)*input_y_pitch + (input_x + 16)*input_x_pitch);
    UNIT_TYPE src_tail_1 = UNIT_BLOCK_READ(input, input_offset + (input_y + 1)*input_y_pitch + (input_x + 16)*input_x_pitch);
    UNIT_TYPE src_tail_2 = UNIT_BLOCK_READ(input, input_offset + (input_y + 2)*input_y_pitch + (input_x + 16)*input_x_pitch);

    for (int i = 0; i < 3; i++) {
        dst[i] = mad(src_block_00[2*i + 0], wei_00,dst[i]);
        dst[i] = mad(src_block_00[2*i + 1], wei_01,dst[i]);
        dst[i] = mad(src_block_00[2*i + 2], wei_02,dst[i]);
        dst[i] = mad(src_block_10[2*i + 0], wei_10,dst[i]);
        dst[i] = mad(src_block_10[2*i + 1], wei_11,dst[i]);
        dst[i] = mad(src_block_10[2*i + 2], wei_12,dst[i]);
        dst[i] = mad(src_block_20[2*i + 0], wei_20,dst[i]);
        dst[i] = mad(src_block_20[2*i + 1], wei_21,dst[i]);
        dst[i] = mad(src_block_20[2*i + 2], wei_22,dst[i]);
    }
    {
        dst[3] = mad(src_block_00[6], wei_00,dst[3]);
        dst[3] = mad(src_block_00[7], wei_01,dst[3]);
        dst[3] = mad(src_block_01[0], wei_02,dst[3]);

        dst[3] = mad(src_block_10[6], wei_10,dst[3]);
        dst[3] = mad(src_block_10[7], wei_11,dst[3]);
        dst[3] = mad(src_block_11[0], wei_12,dst[3]);

        dst[3] = mad(src_block_20[6], wei_20,dst[3]);
        dst[3] = mad(src_block_20[7], wei_21,dst[3]);
        dst[3] = mad(src_block_21[0], wei_22,dst[3]);
    }

    for (int i = 0; i < 3; i++) {
        dst[4+i] = mad(src_block_01[2*i + 0], wei_00,dst[4+i]);
        dst[4+i] = mad(src_block_01[2*i + 1], wei_01,dst[4+i]);
        dst[4+i] = mad(src_block_01[2*i + 2], wei_02,dst[4+i]);

        dst[4+i] = mad(src_block_11[2*i + 0], wei_10,dst[4+i]);
        dst[4+i] = mad(src_block_11[2*i + 1], wei_11,dst[4+i]);
        dst[4+i] = mad(src_block_11[2*i + 2], wei_12,dst[4+i]);

        dst[4+i] = mad(src_block_21[2*i + 0], wei_20,dst[4+i]);
        dst[4+i] = mad(src_block_21[2*i + 1], wei_21,dst[4+i]);
        dst[4+i] = mad(src_block_21[2*i + 2], wei_22,dst[4+i]);
    }
    {
        dst[7] = mad(src_block_01[6], wei_00,dst[7]);
        dst[7] = mad(src_block_01[7], wei_01,dst[7]);
        dst[7] = mad(src_tail_0, wei_02,dst[7]);

        dst[7] = mad(src_block_11[6], wei_10,dst[7]);
        dst[7] = mad(src_block_11[7], wei_11,dst[7]);
        dst[7] = mad(src_tail_1, wei_12,dst[7]);

        dst[7] = mad(src_block_21[6], wei_20,dst[7]);
        dst[7] = mad(src_block_21[7], wei_21,dst[7]);
        dst[7] = mad(src_tail_2, wei_22,dst[7]);
    }
#endif // STRIDE_SIZE_X == 1

    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;


    const uint output_offset =  b * output_b_pitch +
                                (f_block + output_fs_pad_before) * output_fs_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_Y + y) * output_y_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
        for (int i = 0; i < X_BLOCK_SIZE; i++) {
            FUSED_OPS_LOAD_DATA;
            DO_ELTWISE_FUSED_OPS;
            if ((x+i) < OUTPUT_SIZE_X && f_block*FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM)
                output[output_offset + (x+i)*output_x_pitch + lid] = dst[i];
        }
    }
    else
#endif
    {
        if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
        {
            FUSED_OPS_LOAD_DATA_VEC;
            DO_ELTWISE_FUSED_OPS_VEC;
            UNIT_BLOCK_WRITE8(output, output_offset + x*output_x_pitch, dst);
        }
        else
        {
            for (int i = 0; i < (OUTPUT_SIZE_X - x); i++)
            {
                FUSED_OPS_LOAD_DATA;
                DO_ELTWISE_FUSED_OPS;
                UNIT_BLOCK_WRITE(output, output_offset + (x+i)*output_x_pitch, dst[i]);
            }
        }
    }
}

#undef FEATURE_SLICE_SIZE
#undef X_BLOCK_SIZE
