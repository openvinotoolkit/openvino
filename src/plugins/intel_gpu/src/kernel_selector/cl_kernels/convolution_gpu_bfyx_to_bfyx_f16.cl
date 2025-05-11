// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"

#define FEATURE_SLICE_SIZE 16

#define DT_OUTPUT_BLOCK_WRITEN(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE, ptr, offset, val)

#define OUTPUT_PACKED_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE)

#define TO_OUTPUT_PACKED_TYPE CAT(convert_, OUTPUT_PACKED_TYPE)

#if defined(BIAS_TYPE_SIZE) && FILTER_TYPE_SIZE != BIAS_TYPE_SIZE
#error "convolution_gpu_bfyx_to_bfyx_f16: Filter and bias has different data type."
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_gpu_bfyx_to_bfyx_f16)(
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
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // Input offset calculations:
    const uint input_x_pitch = INPUT0_X_PITCH;
    const uint input_y_pitch = INPUT0_Y_PITCH;
    const uint input_f_pitch = INPUT0_FEATURE_PITCH;
    const uint input_b_pitch = INPUT0_BATCH_PITCH;

    const uint input_offset = INPUT0_OFFSET +
                              b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset = b * output_b_pitch +
                               (f_block + output_fs_pad_before) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint filter_offset = f_block * filter_os_pitch;

    MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE) dst = INPUT0_VAL_ZERO;

    INPUT0_TYPE line_cache[INPUT0_FEATURE_NUM * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT0_FEATURE_NUM; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * SUB_GROUP_SIZE + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < INPUT0_SIZE_Y &&
                input_x + xb >= 0 && input_x + xb < INPUT0_SIZE_X)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = INPUT0_VAL_ZERO;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
    for (int kh = 0; kh < FILTER_SIZE_Y; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
        for (int kw = 0; kw < FILTER_SIZE_X; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;

            FILTER_TYPE wei[INPUT0_FEATURE_NUM];
            __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM)))
            for (int ic = 0; ic < INPUT0_FEATURE_NUM; ic++)
                wei[ic] = DT_FILTER_BLOCK_READ(weights, offset + ic * filter_isv_pitch);

            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++)
            {
                const uint buf_offset = (kw*DILATION_SIZE_X + STRIDE_SIZE_X * i + (kh) * INPUT_LINE_SIZE) / SUB_GROUP_SIZE;
                const uint buf_group  = (kw*DILATION_SIZE_X + STRIDE_SIZE_X * i + (kh) * INPUT_LINE_SIZE) % SUB_GROUP_SIZE;

                INPUT0_TYPE src[INPUT0_FEATURE_NUM];
                __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM)))
                for (int ic = 0; ic < INPUT0_FEATURE_NUM; ic++) {
                    src[ic] = _sub_group_shuffle(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src[ic], dst[i]);
                }
            }
        }
    }

#if BIAS_TERM
    uint bias_offset = f_block * FEATURE_SLICE_SIZE;

    dst += (MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE))(DT_BIAS_BLOCK_READ(biases, bias_offset));
#endif

    OUTPUT_PACKED_TYPE res;
#ifndef HAS_FUSED_OPS
    res = TO_OUTPUT_PACKED_TYPE(ACTIVATION(dst, ACTIVATION_PARAMS));
#endif

#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            res[i] = FUSED_OPS_RESULT_SCALAR;
#endif
            if ((f_block*FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X)
                output[output_offset + i * output_x_pitch + lid] = res[i];
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X) {
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            res = FUSED_OPS_RESULT_VEC;
#endif
            DT_OUTPUT_BLOCK_WRITEN(output, output_offset, res);
        } else {
            const int x_tail = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE;
            for (int i = 0; i < x_tail; i++) {
#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                res[i] = FUSED_OPS_RESULT_SCALAR;
#endif
                DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, res[i]);
            }
        }
    }
}

#undef FEATURE_SLICE_SIZE
