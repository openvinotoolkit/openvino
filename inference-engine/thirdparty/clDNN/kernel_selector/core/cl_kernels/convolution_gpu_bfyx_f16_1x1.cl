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

#define GET_SRC(data, id) AS_TYPE(MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE),                             \
                            intel_sub_group_shuffle(                                                     \
                                AS_TYPE(MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, X_BLOCK_SIZE), data),       \
                                id))

#define FEATURE_SLICE_SIZE 16


__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_b_fs_yx_fsv16_1x1)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx) {
    const int xy = get_global_id(0);
    const int f_block = get_group_id(1);
    const int b = get_global_id(2);
    const int lid = get_sub_group_local_id();

    const int x = (xy * X_BLOCK_SIZE) % OUTPUT_SIZE_X;
    const int y = (xy * X_BLOCK_SIZE) / OUTPUT_SIZE_X;

    const int input_x = x;
    const int input_y = y;

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                              INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch;

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset =  b * output_b_pitch +
                               (f_block + output_fs_pad_before) * output_fs_pitch +
                               (OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint filter_offset = f_block * filter_os_pitch;

    typedef MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE) vec_t;


#if BIAS_TERM
    vec_t dst = (vec_t)(UNIT_BLOCK_READ(biases, f_block * FEATURE_SLICE_SIZE));
#else
    vec_t dst = UNIT_VAL_ZERO;
#endif

    for (uint k = 0; k < IC_BLOCKS; ++k)
    {
        vec_t src = 0;
#if INPUT_LEFTOVERS
        if ((k+1)*FEATURE_SLICE_SIZE >= INPUT0_FEATURE_NUM)
        {
            if (k*FEATURE_SLICE_SIZE + lid < INPUT0_FEATURE_NUM)
            {
                __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
                for (int i = 0; i < X_BLOCK_SIZE; i++)
                {
                    const uint xb = (x + i) % INPUT0_SIZE_X;
                    const uint yb = y + (x + i) / INPUT0_SIZE_X;
                    const uint input_idx = input_offset + k * input_fs_pitch +
                                                          yb * input_y_pitch +
                                                          xb * input_x_pitch;
                    src[i] = input[input_idx + lid];
                }
            }
        }
        else
#endif
        {
#if PADDED_INPUT
            __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
            for (int i = 0; i < X_BLOCK_SIZE; i++)
            {
                const uint xb = (x + i) % INPUT0_SIZE_X;
                const uint yb = y + (x + i) / INPUT0_SIZE_X;
                const uint input_idx = input_offset + k * input_fs_pitch +
                                                      yb * input_y_pitch +
                                                      xb * input_x_pitch;
                src[i] = UNIT_BLOCK_READ(input, input_idx);
            }
#else
#if X_BLOCK_SIZE == 8
            src = UNIT_BLOCK_READ8(input, input_offset + k * input_fs_pitch +
                                                         input_y * input_y_pitch +
                                                         input_x * input_x_pitch);
#elif X_BLOCK_SIZE == 4
            src = UNIT_BLOCK_READ4(input, input_offset + k * input_fs_pitch +
                                                         input_y * input_y_pitch +
                                                         input_x * input_x_pitch);
#elif X_BLOCK_SIZE == 2
            src = UNIT_BLOCK_READ2(input, input_offset + k * input_fs_pitch +
                                                         input_y * input_y_pitch +
                                                         input_x * input_x_pitch);
#endif
#endif
        }


        UNIT_TYPE8 wei0 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch);
        UNIT_TYPE8 wei1 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch + 8 * filter_isv_pitch);
        const vec_t src0  = GET_SRC(src, 0);
        const vec_t src1  = GET_SRC(src, 1);
        const vec_t src2  = GET_SRC(src, 2);
        const vec_t src3  = GET_SRC(src, 3);
        const vec_t src4  = GET_SRC(src, 4);
        const vec_t src5  = GET_SRC(src, 5);
        const vec_t src6  = GET_SRC(src, 6);
        const vec_t src7  = GET_SRC(src, 7);
        const vec_t src8  = GET_SRC(src, 8);
        const vec_t src9  = GET_SRC(src, 9);
        const vec_t src10 = GET_SRC(src, 10);
        const vec_t src11 = GET_SRC(src, 11);
        const vec_t src12 = GET_SRC(src, 12);
        const vec_t src13 = GET_SRC(src, 13);
        const vec_t src14 = GET_SRC(src, 14);
        const vec_t src15 = GET_SRC(src, 15);

        dst = mad(wei0.s0, src0, dst);
        dst = mad(wei0.s1, src1, dst);
        dst = mad(wei0.s2, src2, dst);
        dst = mad(wei0.s3, src3, dst);
        dst = mad(wei0.s4, src4, dst);
        dst = mad(wei0.s5, src5, dst);
        dst = mad(wei0.s6, src6, dst);
        dst = mad(wei0.s7, src7, dst);
        dst = mad(wei1.s0, src8, dst);
        dst = mad(wei1.s1, src9, dst);
        dst = mad(wei1.s2, src10, dst);
        dst = mad(wei1.s3, src11, dst);
        dst = mad(wei1.s4, src12, dst);
        dst = mad(wei1.s5, src13, dst);
        dst = mad(wei1.s6, src14, dst);
        dst = mad(wei1.s7, src15, dst);
    }

    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
        for (int i = 0; i < X_BLOCK_SIZE; i++) {
            if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y ||
                f_block*FEATURE_SLICE_SIZE + lid >= OUTPUT_FEATURE_NUM)
                return;

            int xi = (x+i) % OUTPUT_SIZE_X;
            int yi = y + ((x+i) / OUTPUT_SIZE_X);

#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif

            output[output_offset + yi * output_y_pitch + xi * output_x_pitch + lid] = dst[i];
        }
    }
    else
#endif
    {
#if !PADDED_OUTPUT
        if (xy * X_BLOCK_SIZE + X_BLOCK_SIZE <= OUTPUT_SIZE_X * OUTPUT_SIZE_Y) {
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            dst = FUSED_OPS_RESULT_VEC;
#endif
#if X_BLOCK_SIZE == 8
            UNIT_BLOCK_WRITE8(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
#elif X_BLOCK_SIZE == 4
            UNIT_BLOCK_WRITE4(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
#elif X_BLOCK_SIZE == 2
            UNIT_BLOCK_WRITE2(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
#endif
        } else {
#else
        if (x * X_BLOCK_SIZE + X_BLOCK_SIZE <= OUTPUT_SIZE_X) {
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            dst = FUSED_OPS_RESULT_VEC;
#endif
#if X_BLOCK_SIZE == 8
            UNIT_BLOCK_WRITE8(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
#elif X_BLOCK_SIZE == 4
            UNIT_BLOCK_WRITE4(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
#elif X_BLOCK_SIZE == 2
            UNIT_BLOCK_WRITE2(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
#endif
        } else {
#endif
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
                    return;

                int xi = (x+i) % OUTPUT_SIZE_X;
                int yi = y + ((x+i) / OUTPUT_SIZE_X);

#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif

                UNIT_BLOCK_WRITE(output, output_offset + yi * output_y_pitch + xi * output_x_pitch, dst[i]);
            }
        }
    }
}

#undef GET_SRC
#undef FEATURE_SLICE_SIZE
