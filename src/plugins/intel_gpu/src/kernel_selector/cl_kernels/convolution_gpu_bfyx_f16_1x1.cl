// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/unit_type.cl"

#if X_BLOCK_SIZE > 1
#   define GET_SRC(data, id)    AS_TYPE(MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE),                             \
                                    _sub_group_shuffle(                                                   \
                                    AS_TYPE(MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, X_BLOCK_SIZE), data),         \
                                    id))
#else
#   define GET_SRC(data, id)    AS_TYPE(UNIT_TYPE, _sub_group_shuffle(AS_TYPE(UNIT_BLOCK_RW_TYPE, data), id))
#endif

#define FEATURE_SLICE_SIZE 16

#if X_BLOCK_SIZE > 1
#   define UNIT_BLOCK_READ_VEC(ptr, offset)         CAT(UNIT_BLOCK_READ, X_BLOCK_SIZE)(ptr, offset)
#   define UNIT_BLOCK_WRITE_VEC(ptr, offset, val)   CAT(UNIT_BLOCK_WRITE, X_BLOCK_SIZE)(ptr, offset, val)
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_b_fs_yx_fsv16_1x1)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if X_BLOCK_SIZE > 1
    const uint xy = (int)get_global_id(0);
    const uint x = (xy * X_BLOCK_SIZE) % OUTPUT_SIZE_X;
    const uint y = (xy * X_BLOCK_SIZE) / OUTPUT_SIZE_X;

    const uint input_x = x;
    const uint input_y = y;
#endif
    const uint b = (int)get_global_id(2);
    const uint sglid = (int)get_sub_group_local_id();

    const uint lid1 = (int)get_local_id(1);
    const uint feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const uint feature_sub_block = lid1 / feature_per_wg;
    const uint feature_block = (int)get_group_id(1);

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

    const uint output_offset = b * output_b_pitch +
                               (feature_block + output_fs_pad_before) * output_fs_pitch +
                               (OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint filter_offset = feature_block * filter_os_pitch;

#if X_BLOCK_SIZE > 1
    typedef MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE) vec_t;
#else
    typedef UNIT_TYPE vec_t;
#endif

#if BIAS_TERM
#if SLM_DIV_FACTOR == 1
    vec_t dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
    vec_t dst;

    if (feature_sub_block == 0) {
        dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
    } else {
        dst = UNIT_VAL_ZERO;
    }
#endif // SLM_DIV_FACTOR == 1
#else
    vec_t dst = UNIT_VAL_ZERO;
#endif // BIAS_TERM

#if SLM_DIV_FACTOR > 1
    __local vec_t partial_summ[WORK_GROUP_SIZE];

    for (uint k = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; k < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; k++)
    {
#else
    for (uint k = 0; k < IC_BLOCKS; k++)
    {
#endif // SLM_DIV_FACTOR > 1
        vec_t src = 0;
#if INPUT_LEFTOVERS
        if ((k + 1) * FEATURE_SLICE_SIZE >= INPUT0_FEATURE_NUM)
        {
            if (k * FEATURE_SLICE_SIZE + sglid < INPUT0_FEATURE_NUM)
            {
#if X_BLOCK_SIZE > 1
                __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
                for (int i = 0; i < X_BLOCK_SIZE; i++)
                {
                    const uint xb = (x + i) % INPUT0_SIZE_X;
                    const uint yb = y + (x + i) / INPUT0_SIZE_X;
                    const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;

                    src[i] = input[input_idx + sglid];
                }
#else
                src = input[input_offset + k * input_fs_pitch + sglid];
#endif // X_BLOCK_SIZE > 1
            }
        }
        else
#endif // INPUT_LEFTOVERS
        {
#if PADDED_INPUT
#if X_BLOCK_SIZE > 1
            __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
            for (int i = 0; i < X_BLOCK_SIZE; i++)
            {
                const uint xb = (x + i) % INPUT0_SIZE_X;
                const uint yb = y + (x + i) / INPUT0_SIZE_X;
                const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;

                src[i] = UNIT_BLOCK_READ(input, input_idx);
            }
#else
            src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif // X_BLOCK_SIZE > 1

#else // PADDED_INPUT

#if X_BLOCK_SIZE > 1
            src = UNIT_BLOCK_READ_VEC(input, input_offset + k * input_fs_pitch + input_y * input_y_pitch + input_x * input_x_pitch);
#else
            src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif // X_BLOCK_SIZE > 1
#endif // PADDED_INPUT
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

#if SLM_DIV_FACTOR > 1
    partial_summ[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif // SLM_DIV_FACTOR > 1

    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

#if OUTPUT_LEFTOVERS
    if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
#if X_BLOCK_SIZE > 1
        for (int i = 0; i < X_BLOCK_SIZE; i++) {
            if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
                return;

            int xi = (x + i) % OUTPUT_SIZE_X;
            int yi = y + ((x + i) / OUTPUT_SIZE_X);

#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif

            output[output_offset + yi * output_y_pitch + xi * output_x_pitch + sglid] = dst[i];
        }
#else // X_BLOCK_SIZE > 1
        if (feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
            return;

#if HAS_FUSED_OPS
        FUSED_OPS_SCALAR_B1;
        dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif

        output[output_offset + sglid] = dst;
#endif // X_BLOCK_SIZE > 1
    }
    else
#endif // OUTPUT_LEFTOVERS

#if X_BLOCK_SIZE > 1
    {
#if !PADDED_OUTPUT && !NON_UNIT_FUSED_OP_SPATIAL
        if (xy * X_BLOCK_SIZE + X_BLOCK_SIZE <= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % X_BLOCK_SIZE == 0) {
#else
        if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % X_BLOCK_SIZE == 0) {
#endif
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            dst = FUSED_OPS_RESULT_VEC;
#endif
            UNIT_BLOCK_WRITE_VEC(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
        } else {
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
                    return;

                int xi = (x + i) % OUTPUT_SIZE_X;
                int yi = y + ((x + i) / OUTPUT_SIZE_X);

#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif

                UNIT_BLOCK_WRITE(output, output_offset + yi * output_y_pitch + xi * output_x_pitch, dst[i]);
            }
        }
    }
#else // X_BLOCK_SIZE > 1
    {
#if HAS_FUSED_OPS
        FUSED_OPS_SCALAR_B1;
        dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
        UNIT_BLOCK_WRITE(output, output_offset, dst);
    }
#endif // X_BLOCK_SIZE > 1

#if SLM_DIV_FACTOR > 1
    }
#endif
}

#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef UNIT_BLOCK_READ_VEC
#undef UNIT_BLOCK_WRITE_VEC
