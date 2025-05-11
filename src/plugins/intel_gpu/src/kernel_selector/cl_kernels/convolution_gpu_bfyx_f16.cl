// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"

#define INPUT_TYPE        INPUT0_TYPE
#define INPUT_TYPE2       MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8       MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)

#define FILTER_TYPE8      MAKE_VECTOR_TYPE(FILTER_TYPE, 8)

#define AS_INPUT_TYPE     CAT(as_, INPUT_TYPE)
#define AS_INPUT_TYPE2    CAT(as_, INPUT_TYPE2)
#define AS_INPUT_TYPE4    CAT(as_, INPUT_TYPE4)
#define AS_INPUT_TYPE8    CAT(as_, INPUT_TYPE8)

#define AS_FILTER_TYPE8   CAT(as_, FILTER_TYPE8)


#if OUTPUT_FORMAT_BFYX
#   define OUTPUTVTYPE(n)       CAT(OUTPUT_TYPE, n)
#   define TO_OUTPUTVTYPE       CAT(convert_, OUTPUTVTYPE(OUTPUT_X_BLOCK_SIZE))
#   define VSTORE               CAT(vstore, OUTPUT_X_BLOCK_SIZE)
#endif  // OUTPUT_FORMAT_BFYX

#if INPUT0_TYPE_SIZE == 2
#   define AS_INPUT_SRC         CAT(as_, MAKE_VECTOR_TYPE(INPUT_TYPE, OUTPUT_X_BLOCK_SIZE))
#   define AS_US_SRC            CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
#   define GET_SRC(data, id)    AS_INPUT_SRC(_sub_group_shuffle(AS_US_SRC(data), id))
#else
#   define GET_SRC(data, id)    _sub_group_shuffle(data, id)
#endif

#define FEATURE_SLICE_SIZE 16

#define FILTER_OFM_NUM_ALIGNED (((FILTER_OFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)
#define FILTER_IFM_NUM_ALIGNED (((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_bfyx_f16)(
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
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);

#if GROUPED
    const int group = (feature_block * FEATURE_SLICE_SIZE) / FILTER_OFM_NUM;
    const int prev_group_leftover = (FILTER_OFM_NUM * (group + 1)) - (feature_block * FEATURE_SLICE_SIZE);
    int groups_per_sub_group = 1;
    if (prev_group_leftover < 16)
        groups_per_sub_group += ((FEATURE_SLICE_SIZE - prev_group_leftover - 1) / FILTER_OFM_NUM) + 1;
#else
    const int group = 0;
    const int groups_per_sub_group = 1;
#endif  // GROUPED

    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE) vec_t;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_Y + input_y) * input_y_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch;

    // Output offset calculations:

#if OUTPUT_FORMAT_BFYX
    const uint output_y_pitch = (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * (OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM);

    const uint output_offset = b * output_b_pitch +
                               feature_block * (output_fs_pitch * FEATURE_SLICE_SIZE) +
                               (sglid + OUTPUT_PAD_BEFORE_FEATURE_NUM) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X);
#else
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset = b * output_b_pitch +
                               (feature_block + output_fs_pad_before) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;
#endif

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

#if BIAS_TERM
#if SLM_DIV_FACTOR == 1
    vec_t dst = (vec_t)(DT_INPUT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
    vec_t dst;

    if (feature_sub_block == 0) {
        dst = (vec_t)(DT_INPUT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
    } else {
        dst = INPUT0_VAL_ZERO;
    }
#endif // SLM_DIV_FACTOR == 1
#else
    vec_t dst = INPUT0_VAL_ZERO;
#endif // BIAS_TERM

#if SLM_DIV_FACTOR > 1
    __local vec_t partial_summ[WORK_GROUP_SIZE];
#endif

#if MULTIPLE_GROUPS_INPUT_PRELOAD
    const uint in_split_offset = feature_block * input_fs_pitch;
    const uint g = sglid / (FEATURE_SLICE_SIZE / groups_per_sub_group);
    const uint ofm_in_group = sglid % (FEATURE_SLICE_SIZE / groups_per_sub_group);
    const uint grouped_filter_offset = (group + g) * FILTER_GROUPS_PITCH;
#else
#if GROUPED
    for (uint g = group; g < group + groups_per_sub_group; g++) {
        const uint in_split_offset = g * input_fs_pitch * (FILTER_IFM_NUM / FEATURE_SLICE_SIZE);
        const uint filter_split_offset = g * FILTER_GROUPS_PITCH;
        const uint filter_offset = (feature_block % (FILTER_OFM_NUM / FEATURE_SLICE_SIZE)) * filter_os_pitch;
#else
        const uint in_split_offset = 0;
        const uint filter_split_offset = 0;
        const uint filter_offset = feature_block * filter_os_pitch;
#endif  // GROUPED
        const uint grouped_filter_offset = filter_offset + filter_split_offset;
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD

        const uint grouped_input_offset = input_offset + in_split_offset;

#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif // SLM_DIV_FACTOR > 1
            __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
            for (int kh = 0; kh < FILTER_SIZE_Y; kh++) {
                if (input_y + kh * DILATION_SIZE_Y < 0 || input_y + kh * DILATION_SIZE_Y >= INPUT0_SIZE_Y)
                    continue;

                INPUT_TYPE line_cache[INPUT_LINE_SIZE];

#if INPUT_LEFTOVERS
                if ((icb + 1) * FEATURE_SLICE_SIZE >= FILTER_IFM_NUM)
                {
                    for (int xb = 0; xb < INPUT_LINE_SIZE; xb++)
                    {
                        if (icb * FEATURE_SLICE_SIZE + sglid >= FILTER_IFM_NUM)
                            line_cache[xb] = 0;
                        else
                            line_cache[xb] = input[grouped_input_offset +
                                                   icb * input_fs_pitch +
                                                   kh * DILATION_SIZE_Y * input_y_pitch +
                                                   xb * input_x_pitch +
                                                   sglid];
                    }
                }
                else
#endif  // INPUT_LEFTOVERS
                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        INPUT_TYPE8 vv = DT_INPUT_BLOCK_READ8(input, grouped_input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_SIZE_Y * input_y_pitch +
                                                                  xb * input_x_pitch);

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                        line_cache[xb + 4] = vv[4];
                        line_cache[xb + 5] = vv[5];
                        line_cache[xb + 6] = vv[6];
                        line_cache[xb + 7] = vv[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        INPUT_TYPE4 vv = DT_INPUT_BLOCK_READ4(input, grouped_input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_SIZE_Y * input_y_pitch +
                                                                  xb * input_x_pitch);

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = DT_INPUT_BLOCK_READ(input, grouped_input_offset +
                                                                 icb * input_fs_pitch +
                                                                 kh * DILATION_SIZE_Y * input_y_pitch +
                                                                 xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                for (int kw = 0; kw < FILTER_SIZE_X; kw++) {
                    vec_t src;
                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
                    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * i];
#endif  // FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                    }
#if MULTIPLE_GROUPS_INPUT_PRELOAD
                    typedef MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_IFM_NUM) ifm_vec_t;

                    ifm_vec_t wei0 = FILTER_VAL_ZERO;
                    for (int ifm = 0; ifm < FILTER_IFM_NUM; ifm++)
                        wei0[ifm] = weights[grouped_filter_offset +
                                            ofm_in_group +
                                            ifm * filter_isv_pitch +
                                            kh * filter_y_pitch +
                                            kw * filter_x_pitch];

#if FILTER_IFM_NUM == 2
                        const vec_t src0  = GET_SRC(src, g * FILTER_IFM_NUM + 0);
                        const vec_t src1  = GET_SRC(src, g * FILTER_IFM_NUM + 1);

                        dst = mad(wei0.s0, src0,  dst);
                        dst = mad(wei0.s1, src1,  dst);
#elif FILTER_IFM_NUM == 4
                        const vec_t src0  = GET_SRC(src, g * FILTER_IFM_NUM + 0);
                        const vec_t src1  = GET_SRC(src, g * FILTER_IFM_NUM + 1);
                        const vec_t src2  = GET_SRC(src, g * FILTER_IFM_NUM + 2);
                        const vec_t src3  = GET_SRC(src, g * FILTER_IFM_NUM + 3);

                        dst = mad(wei0.s0, src0,  dst);
                        dst = mad(wei0.s1, src1,  dst);
                        dst = mad(wei0.s2, src2,  dst);
                        dst = mad(wei0.s3, src3,  dst);
#elif FILTER_IFM_NUM == 8
                        const vec_t src0  = GET_SRC(src, g * FILTER_IFM_NUM + 0);
                        const vec_t src1  = GET_SRC(src, g * FILTER_IFM_NUM + 1);
                        const vec_t src2  = GET_SRC(src, g * FILTER_IFM_NUM + 2);
                        const vec_t src3  = GET_SRC(src, g * FILTER_IFM_NUM + 3);
                        const vec_t src4  = GET_SRC(src, g * FILTER_IFM_NUM + 4);
                        const vec_t src5  = GET_SRC(src, g * FILTER_IFM_NUM + 5);
                        const vec_t src6  = GET_SRC(src, g * FILTER_IFM_NUM + 6);
                        const vec_t src7  = GET_SRC(src, g * FILTER_IFM_NUM + 7);

                        dst = mad(wei0.s0, src0,  dst);
                        dst = mad(wei0.s1, src1,  dst);
                        dst = mad(wei0.s2, src2,  dst);
                        dst = mad(wei0.s3, src3,  dst);
                        dst = mad(wei0.s4, src4,  dst);
                        dst = mad(wei0.s5, src5,  dst);
                        dst = mad(wei0.s6, src6,  dst);
                        dst = mad(wei0.s7, src7,  dst);
#else
#   error convolution_gpu_bfyx_f16.cl: unsupported input feature size for multiple groups input preload
#endif  // FILTER_IFM_NUM
#else
                    FILTER_TYPE8 wei0 = DT_FILTER_BLOCK_READ8(weights, grouped_filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch);
                    FILTER_TYPE8 wei1 = DT_FILTER_BLOCK_READ8(weights, grouped_filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch);
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

                    dst = mad(wei0.s0, src0,  dst);
                    dst = mad(wei0.s1, src1,  dst);
                    dst = mad(wei0.s2, src2,  dst);
                    dst = mad(wei0.s3, src3,  dst);
                    dst = mad(wei0.s4, src4,  dst);
                    dst = mad(wei0.s5, src5,  dst);
                    dst = mad(wei0.s6, src6,  dst);
                    dst = mad(wei0.s7, src7,  dst);
                    dst = mad(wei1.s0, src8,  dst);
                    dst = mad(wei1.s1, src9,  dst);
                    dst = mad(wei1.s2, src10, dst);
                    dst = mad(wei1.s3, src11, dst);
                    dst = mad(wei1.s4, src12, dst);
                    dst = mad(wei1.s5, src13, dst);
                    dst = mad(wei1.s6, src14, dst);
                    dst = mad(wei1.s7, src15, dst);
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD
                }
            }
        }
#if GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD
    }
#endif  // GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD

#if SLM_DIV_FACTOR > 1
    partial_summ[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif // SLM_DIV_FACTOR > 1

    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) out_vec_t;
    out_vec_t res;

#if OUTPUT_LEFTOVERS
    if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {

#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
#   if OUTPUT_FORMAT_BFYX
            res[i] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SCALAR);
#   else
            res[i] = FUSED_OPS_RESULT_SCALAR;
#   endif
#else
            res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif

#if OUTPUT_FORMAT_BFYX
            if ((feature_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
                output[output_offset + i] = res[i];
            }
#else
            if ((feature_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
                output[output_offset + i * output_x_pitch + sglid] = res[i];
            }
#endif
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0) {
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
#   if OUTPUT_FORMAT_BFYX
            res = TO_OUTPUTVTYPE(FUSED_OPS_RESULT_VEC);
#   else
            res = FUSED_OPS_RESULT_VEC;
#   endif
#else
#   if OUTPUT_FORMAT_BFYX
            res = TO_OUTPUTVTYPE(dst);
#   else
            res = dst;
#   endif
#endif
            // TODO Generalize for other block sizes
#if OUTPUT_FORMAT_BFYX
    #if OUTPUT_X_BLOCK_SIZE == 2 || OUTPUT_X_BLOCK_SIZE == 4 || OUTPUT_X_BLOCK_SIZE == 8
            VSTORE(res, 0, output + output_offset);
    #elif OUTPUT_X_BLOCK_SIZE == 1
            output[output_offset] = res[0];
    #else
    #   error convolution_gpu_bfyx_f16.cl: unsupported output x block size
    #endif
#else
    #if OUTPUT_X_BLOCK_SIZE == 8
            DT_OUTPUT_BLOCK_WRITE8(output, output_offset, res);
    #elif OUTPUT_X_BLOCK_SIZE == 4
            DT_OUTPUT_BLOCK_WRITE4(output, output_offset, res);
    #elif OUTPUT_X_BLOCK_SIZE == 2
            DT_OUTPUT_BLOCK_WRITE2(output, output_offset, res);
    #elif OUTPUT_X_BLOCK_SIZE == 1
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, res);
    #else
    #   error convolution_gpu_bfyx_f16.cl: unsupported output x block size
    #endif
#endif  // OUTPUT_FORMAT_BFYX
        } else {
            for (int i = 0; i < OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
#   if OUTPUT_FORMAT_BFYX
                res[i] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SCALAR);
#   else
                res[i] = FUSED_OPS_RESULT_SCALAR;
#   endif
#else
                res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif

#if OUTPUT_FORMAT_BFYX
                output[output_offset + i] = res[i];
#else
                DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, res[i]);
#endif
            }
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}

#undef AS_INPUT_SRC
#undef AS_US_SRC
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef FILTER_OFM_NUM_ALIGNED
#undef FILTER_IFM_NUM_ALIGNED

#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8

#undef FILTER_TYPE8

#undef AS_INPUT_TYPE
#undef AS_INPUT_TYPE2
#undef AS_INPUT_TYPE4
#undef AS_INPUT_TYPE8

#undef AS_FILTER_TYPE8

#if OUTPUT_FORMAT_BFYX
#   undef OUTPUTVTYPE
#   undef TO_OUTPUTVTYPE
#   undef VSTORE
#endif  // OUTPUT_FORMAT_BFYX
