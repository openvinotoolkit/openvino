// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define FEATURE_SLICE_SIZE 16

#define INPUT_TYPE        INPUT0_TYPE
#define INPUT_TYPE8       MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)

#define FILTER_TYPE2      MAKE_VECTOR_TYPE(FILTER_TYPE, 2)

#define OUTPUT_TYPE8      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8)

#define AS_INPUT_TYPE     CAT(as_, INPUT_TYPE)
#define AS_INPUT_TYPE8    CAT(as_, INPUT_TYPE8)

#define AS_FILTER_TYPE2   CAT(as_, FILTER_TYPE2)
#define TO_OUTPUT_TYPE8   CAT(convert_, OUTPUT_TYPE8)

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_gpu_bfyx_f16_depthwise)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint yx = (uint)get_global_id(0);
    const uint x = (yx % X_BLOCKS) * X_BLOCK_SIZE;
    const uint y = (yx / X_BLOCKS);
    const uint f_block = get_group_id(1);
    const uint lid = get_local_id(1);
    const uint b = (uint)get_global_id(2);

    const uint filter_offset = f_block * FEATURE_SLICE_SIZE * FILTER_SIZE_X * FILTER_SIZE_Y;

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
                              INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                              INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch +
                              (f_block + input_fs_pad_before) * input_fs_pitch;

#if BIAS_TERM
    INPUT_TYPE8 dst = (INPUT_TYPE8)(DT_INPUT_BLOCK_READ(biases, f_block * FEATURE_SLICE_SIZE));
#else
    INPUT_TYPE8 dst = (INPUT_TYPE8)(INPUT0_VAL_ZERO);
#endif

#if ((FILTER_SIZE_X == 3) && (FILTER_SIZE_Y == 3) && (STRIDE_SIZE_X == 1) && (DILATION_SIZE_X == 1) && (DILATION_SIZE_Y == 1))

    FILTER_TYPE wei_00 = DT_FILTER_BLOCK_READ(weights, filter_offset + 0 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 0 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_01 = DT_FILTER_BLOCK_READ(weights, filter_offset + 0 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 1 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_02 = DT_FILTER_BLOCK_READ(weights, filter_offset + 0 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 2 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_10 = DT_FILTER_BLOCK_READ(weights, filter_offset + 1 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 0 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_11 = DT_FILTER_BLOCK_READ(weights, filter_offset + 1 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 1 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_12 = DT_FILTER_BLOCK_READ(weights, filter_offset + 1 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 2 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_20 = DT_FILTER_BLOCK_READ(weights, filter_offset + 2 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 0 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_21 = DT_FILTER_BLOCK_READ(weights, filter_offset + 2 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 1 * FEATURE_SLICE_SIZE);
    FILTER_TYPE wei_22 = DT_FILTER_BLOCK_READ(weights, filter_offset + 2 * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + 2 * FEATURE_SLICE_SIZE);

    INPUT_TYPE8 src_block_0 = DT_INPUT_BLOCK_READ8(input, input_offset + (input_y + 0) * input_y_pitch + (input_x) * input_x_pitch);
    INPUT_TYPE8 src_block_1 = DT_INPUT_BLOCK_READ8(input, input_offset + (input_y + 1) * input_y_pitch + (input_x) * input_x_pitch);
    INPUT_TYPE8 src_block_2 = DT_INPUT_BLOCK_READ8(input, input_offset + (input_y + 2) * input_y_pitch + (input_x) * input_x_pitch);
    INPUT_TYPE src_tail_00 = DT_INPUT_BLOCK_READ(input, input_offset + (input_y + 0) * input_y_pitch + (input_x + 8) * input_x_pitch);
    INPUT_TYPE src_tail_01 = DT_INPUT_BLOCK_READ(input, input_offset + (input_y + 0) * input_y_pitch + (input_x + 9) * input_x_pitch);
    INPUT_TYPE src_tail_10 = DT_INPUT_BLOCK_READ(input, input_offset + (input_y + 1) * input_y_pitch + (input_x + 8) * input_x_pitch);
    INPUT_TYPE src_tail_11 = DT_INPUT_BLOCK_READ(input, input_offset + (input_y + 1) * input_y_pitch + (input_x + 9) * input_x_pitch);
    INPUT_TYPE src_tail_20 = DT_INPUT_BLOCK_READ(input, input_offset + (input_y + 2) * input_y_pitch + (input_x + 8) * input_x_pitch);
    INPUT_TYPE src_tail_21 = DT_INPUT_BLOCK_READ(input, input_offset + (input_y + 2) * input_y_pitch + (input_x + 9) * input_x_pitch);

#if X_BLOCK_SIZE == 8
    for (uint i = 0; i < X_BLOCK_SIZE - 2; i++)
    {
        dst[i] = mad(src_block_0[i + 0], wei_00, dst[i]);
        dst[i] = mad(src_block_0[i + 1], wei_01, dst[i]);
        dst[i] = mad(src_block_0[i + 2], wei_02, dst[i]);

        dst[i] = mad(src_block_1[i + 0], wei_10, dst[i]);
        dst[i] = mad(src_block_1[i + 1], wei_11, dst[i]);
        dst[i] = mad(src_block_1[i + 2], wei_12, dst[i]);

        dst[i] = mad(src_block_2[i + 0], wei_20, dst[i]);
        dst[i] = mad(src_block_2[i + 1], wei_21, dst[i]);
        dst[i] = mad(src_block_2[i + 2], wei_22, dst[i]);
    }
    {
        dst[6] = mad(src_block_0[6], wei_00, dst[6]);
        dst[6] = mad(src_block_0[7], wei_01, dst[6]);
        dst[6] = mad(src_tail_00,    wei_02, dst[6]);

        dst[6] = mad(src_block_1[6], wei_10, dst[6]);
        dst[6] = mad(src_block_1[7], wei_11, dst[6]);
        dst[6] = mad(src_tail_10,    wei_12, dst[6]);

        dst[6] = mad(src_block_2[6], wei_20, dst[6]);
        dst[6] = mad(src_block_2[7], wei_21, dst[6]);
        dst[6] = mad(src_tail_20,    wei_22, dst[6]);
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
#else // X_BLOCK_SIZE == 1
        dst[0] = mad(src_block_0[0], wei_00, dst[0]);
        dst[0] = mad(src_block_0[1], wei_01, dst[0]);
        dst[0] = mad(src_block_0[2], wei_02, dst[0]);

        dst[0] = mad(src_block_1[0], wei_10, dst[0]);
        dst[0] = mad(src_block_1[1], wei_11, dst[0]);
        dst[0] = mad(src_block_1[2], wei_12, dst[0]);

        dst[0] = mad(src_block_2[0], wei_20, dst[0]);
        dst[0] = mad(src_block_2[1], wei_21, dst[0]);
        dst[0] = mad(src_block_2[2], wei_22, dst[0]);
#endif

#else // ((FILTER_SIZE_X == 3) && (FILTER_SIZE_Y == 3) && (STRIDE_SIZE_X == 1))

    FILTER_TYPE wei[FILTER_SIZE_Y * FILTER_SIZE_X];
    FILTER_TYPE2 wei_temp;

    unroll_for (uint i = 0; i < FILTER_SIZE_Y; i++) {
        unroll_for (uint j = 0; j < FILTER_SIZE_X_DIV_2; j++) {
            wei_temp = DT_FILTER_BLOCK_READ2(weights, filter_offset + i * FILTER_Y_PITCH * FEATURE_SLICE_SIZE + j * 2 * FEATURE_SLICE_SIZE);
            wei[i * FILTER_SIZE_X + j * 2] = wei_temp.s0;
            wei[i * FILTER_SIZE_X + j * 2 + 1] = wei_temp.s1;
        }
#if (FILTER_SIZE_X % 2)
        wei[i * FILTER_SIZE_X + FILTER_SIZE_X - 1] = DT_FILTER_BLOCK_READ(weights, filter_offset +
                                                                                i * FILTER_Y_PITCH * FEATURE_SLICE_SIZE +
                                                                                (FILTER_SIZE_X - 1) * FEATURE_SLICE_SIZE);
#endif // (FILTER_SIZE_X % 2)
    }

    INPUT_TYPE src[X_BLOCK_SIZE * FILTER_SIZE_Y * FILTER_SIZE_X];

    unroll_for (uint k = 0; k < X_BLOCK_SIZE; k++) {
        unroll_for (uint i = 0; i < FILTER_SIZE_Y; i++) {
            unroll_for (uint j = 0; j < FILTER_SIZE_X; j++) {
                src[k * FILTER_SIZE_Y * FILTER_SIZE_X + i * FILTER_SIZE_X + j] = DT_INPUT_BLOCK_READ(input, input_offset +
                                                                                                         (input_y + (i * DILATION_SIZE_Y)) * input_y_pitch +
                                                                                                         (input_x + (j * DILATION_SIZE_X) + k * STRIDE_SIZE_X) * input_x_pitch);
            }
        }
    }

    unroll_for (uint k = 0; k < X_BLOCK_SIZE; k++) {
        unroll_for (uint i = 0; i < FILTER_SIZE_Y; i++) {
            unroll_for (uint j = 0; j < FILTER_SIZE_X; j++) {
                dst[k] = mad(src[k * FILTER_SIZE_Y * FILTER_SIZE_X + i * FILTER_SIZE_X + j], wei[i * FILTER_SIZE_X + j], dst[k]);
            }
        }
    }

#endif // ((FILTER_SIZE_X == 3) && (FILTER_SIZE_Y == 3) && (STRIDE_SIZE_X == 1))

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

#if X_BLOCK_SIZE == 8
    OUTPUT_TYPE8 res;
#if OUTPUT_LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
        for (uint i = 0; i < X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            res[i] = FUSED_OPS_RESULT_SCALAR;
#else
            res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif // HAS_FUSED_OPS
            if ((x + i) < OUTPUT_SIZE_X && f_block * FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM)
                output[output_offset + (x + i) * output_x_pitch + lid] = res[i];
        }
    }
    else
#endif // OUTPUT_LEFTOVERS
    {
        if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
        {
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            res = FUSED_OPS_RESULT_VEC;
#else
            res = TO_OUTPUT_TYPE8(dst);
#endif // HAS_FUSED_OPS
            DT_OUTPUT_BLOCK_WRITE8(output, output_offset + x * output_x_pitch, res);
        }
        else
        {
            for (uint i = 0; i < (OUTPUT_SIZE_X % X_BLOCK_SIZE); i++) {
#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                res[i] = FUSED_OPS_RESULT_SCALAR;
#else
                res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif // HAS_FUSED_OPS
                DT_OUTPUT_BLOCK_WRITE(output, output_offset + (x + i) * output_x_pitch, res[i]);
            }
        }
    }
#else // X_BLOCK_SIZE == 1
    OUTPUT_TYPE res;
#if OUTPUT_LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
#if HAS_FUSED_OPS
        uint i = 0;
        FUSED_OPS_SCALAR;
        res = FUSED_OPS_RESULT_SCALAR;
#else
        res = TO_OUTPUT_TYPE(dst[0]);
#endif // HAS_FUSED_OPS
        if (x < OUTPUT_SIZE_X && f_block * FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM)
            output[output_offset + x * output_x_pitch + lid] = res;
    }
    else
#endif // OUTPUT_LEFTOVERS
    {
#if HAS_FUSED_OPS
        uint i = 0;
        FUSED_OPS_SCALAR;
        res = FUSED_OPS_RESULT_SCALAR;
#else
        res = TO_OUTPUT_TYPE(dst[0]);
#endif // HAS_FUSED_OPS
        DT_OUTPUT_BLOCK_WRITE(output, output_offset + x * output_x_pitch, res);
    }
#endif
}

#undef FEATURE_SLICE_SIZE
#undef X_BLOCK_SIZE

#undef INPUT_TYPE
#undef INPUT_TYPE8

#undef FILTER_TYPE2

#undef OUTPUT_TYPE8

#undef AS_INPUT_TYPE
#undef AS_INPUT_TYPE8

#undef AS_FILTER_TYPE2
#undef TO_OUTPUT_TYPE8
