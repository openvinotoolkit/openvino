// Copyright (c) 2016-2019 Intel Corporation
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

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_bfyx_to_bfyx_f16)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    typedef MAKE_VECTOR_TYPE(UNIT_TYPE, OUTPUT_X_BLOCK_SIZE) vec_t;
    typedef MAKE_VECTOR_TYPE(UNIT_TYPE, 8) wei_t;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // Input offset calculations:
    const uint input_x_pitch = INPUT0_X_PITCH;
    const uint input_y_pitch = INPUT0_Y_PITCH;
    const uint input_f_pitch = INPUT0_FEATURE_PITCH;
    const uint input_b_pitch = INPUT0_BATCH_PITCH;

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f_block / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * FILTER_IFM_NUM * input_f_pitch;
#endif // DEPTHWISE_SEPARABLE_OPT

    const uint input_offset = in_split_offset +
                              INPUT0_OFFSET +
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

    const uint out_split_offset = split_idx * (OUTPUT_FEATURE_NUM / FEATURE_SLICE_SIZE) * output_fs_pitch;

    const uint output_offset = out_split_offset +
                               b * output_b_pitch +
                               (f_block + output_fs_pad_before) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

#if GROUPED && !DEPTHWISE_SEPARABLE_OPT
    const uint filter_offset = f_block * filter_os_pitch + split_idx * FILTER_LENGTH;
#else
    const uint filter_offset = f_block * filter_os_pitch;
#endif

#if BIAS_TERM
    uint bias_offset = f_block * FEATURE_SLICE_SIZE;

#   if GROUPED && !DEPTHWISE_SEPARABLE_OPT
    bias_offset += split_idx * BIAS_LENGTH;
#   endif

    vec_t dst = (vec_t)(UNIT_BLOCK_READ(biases, bias_offset));
#else
    vec_t dst = UNIT_VAL_ZERO;
#endif

    UNIT_TYPE line_cache[3 * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < 3; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (uint i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const uint in_elem = i * SUB_GROUP_SIZE + lid;
            const uint xb = in_elem % INPUT_LINE_SIZE;
            const uint yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < INPUT0_SIZE_Y &&
                input_x + xb >= 0 && input_x + xb < INPUT0_SIZE_X)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = UNIT_VAL_ZERO;
        }
    }


    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
    for (uint kh = 0; kh < FILTER_SIZE_Y; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
        for (uint kw = 0; kw < FILTER_SIZE_X; kw++)
        {
            MAKE_VECTOR_TYPE(UNIT_TYPE, 2) wei = UNIT_BLOCK_READ2(weights, filter_offset +
                                                                           kh * filter_y_pitch +
                                                                           kw * filter_x_pitch);
            UNIT_TYPE wei2 = UNIT_BLOCK_READ(weights, filter_offset +
                                                      kh * filter_y_pitch +
                                                      kw * filter_x_pitch +
                                                      2 * filter_isv_pitch);

            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++)
            {
                const uint buf_offset = (kw*DILATION_SIZE_X + STRIDE_SIZE_X * i + (kh) * INPUT_LINE_SIZE) / SUB_GROUP_SIZE;
                const uint buf_group  = (kw*DILATION_SIZE_X + STRIDE_SIZE_X * i + (kh) * INPUT_LINE_SIZE) % SUB_GROUP_SIZE;

                UNIT_TYPE src0 = intel_sub_group_shuffle(line_cache[0 * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                UNIT_TYPE src1 = intel_sub_group_shuffle(line_cache[1 * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                UNIT_TYPE src2 = intel_sub_group_shuffle(line_cache[2 * INPUT_BLOCK_SIZE + buf_offset], buf_group);

                dst[i] = mad(wei[0], src0, dst[i]);
                dst[i] = mad(wei[1], src1, dst[i]);
                dst[i] = mad(wei2, src2, dst[i]);
            }
        }
    }

    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
            FUSED_OPS_LOAD_DATA;
            DO_ELTWISE_FUSED_OPS;
            if ((f_block*FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X)
                output[output_offset + i * output_x_pitch + lid] = dst[i];
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X) {
            FUSED_OPS_LOAD_DATA_VEC;
            DO_ELTWISE_FUSED_OPS_VEC;
            // TODO Generalize for other block sizes
#if OUTPUT_X_BLOCK_SIZE == 8
            UNIT_BLOCK_WRITE8(output, output_offset, dst);
#elif OUTPUT_X_BLOCK_SIZE == 4
            UNIT_BLOCK_WRITE4(output, output_offset, dst);
#elif OUTPUT_X_BLOCK_SIZE == 2
            UNIT_BLOCK_WRITE2(output, output_offset, dst);
#elif OUTPUT_X_BLOCK_SIZE == 1
            UNIT_BLOCK_WRITE(output, output_offset, dst);
#else
#   error convolution_gpu_bfyx_to_bfyx_f16.cl: Unsupported output x block size.
#endif
        } else {
            const int x_tail = OUTPUT_SIZE_X - x;
            for (int i = 0; i < x_tail; i++) {
                FUSED_OPS_LOAD_DATA;
                DO_ELTWISE_FUSED_OPS;
                UNIT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, dst[i]);
            }
        }
    }
}

#undef FEATURE_SLICE_SIZE 16
