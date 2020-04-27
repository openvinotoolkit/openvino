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
#include "include/mmad.cl"

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

#if INPUT0_TYPE_SIZE == 2
#   define INPUT_BLOCK_READ(ptr, offset)    AS_INPUT_TYPE(intel_sub_group_block_read_us((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ2(ptr, offset)   AS_INPUT_TYPE2(intel_sub_group_block_read_us2((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ4(ptr, offset)   AS_INPUT_TYPE4(intel_sub_group_block_read_us4((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ8(ptr, offset)   AS_INPUT_TYPE8(intel_sub_group_block_read_us8((__global ushort*)(ptr) + (offset)))
#elif INPUT0_TYPE_SIZE == 4
#   define INPUT_BLOCK_READ(ptr, offset)    AS_INPUT_TYPE(intel_sub_group_block_read((__global uint*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ2(ptr, offset)   AS_INPUT_TYPE2(intel_sub_group_block_read2((__global uint*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ4(ptr, offset)   AS_INPUT_TYPE4(intel_sub_group_block_read4((__global uint*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ8(ptr, offset)   AS_INPUT_TYPE8(intel_sub_group_block_read8((__global uint*)(ptr) + (offset)))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported input type.
#endif

#if FILTER_TYPE_SIZE == 2
#   define FILTER_BLOCK_READ8(ptr, offset) AS_FILTER_TYPE8(intel_sub_group_block_read_us8((__global ushort*)(ptr) + (offset)))
#elif FILTER_TYPE_SIZE == 4
#   define FILTER_BLOCK_READ8(ptr, offset) AS_FILTER_TYPE8(intel_sub_group_block_read8((__global uint*)(ptr) + (offset)))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported filter type.
#endif

#if OUTPUT_TYPE_SIZE == 1
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)    BLOCK_WRITE_UC_1((__global uchar*)(ptr) + (offset), as_uchar(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   BLOCK_WRITE_UC_2((__global uchar*)(ptr) + (offset), as_uchar2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   BLOCK_WRITE_UC_4((__global uchar*)(ptr) + (offset), as_uchar4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   BLOCK_WRITE_UC_8((__global uchar*)(ptr) + (offset), as_uchar8(val))
#elif OUTPUT_TYPE_SIZE == 2
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write_us((__global ushort*)(ptr) + (offset), as_ushort(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write_us2((__global ushort*)(ptr) + (offset), as_ushort2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write_us4((__global ushort*)(ptr) + (offset), as_ushort4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write_us8((__global ushort*)(ptr) + (offset), as_ushort8(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write((__global uint*)(ptr) + (offset), as_uint(val))
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write2((__global uint*)(ptr) + (offset), as_uint2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write4((__global uint*)(ptr) + (offset), as_uint4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write8((__global uint*)(ptr) + (offset), as_uint8(val))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported output type.
#endif

#if INPUT0_TYPE_SIZE == 2
#   define AS_INPUT_SRC         CAT(as_, MAKE_VECTOR_TYPE(INPUT_TYPE, OUTPUT_X_BLOCK_SIZE))
#   define AS_US_SRC            CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
#   define GET_SRC(data, id)    AS_INPUT_SRC(intel_sub_group_shuffle(AS_US_SRC(data), id))
#else
#   define GET_SRC(data, id)    intel_sub_group_shuffle(data, id)
#endif
#define FEATURE_SLICE_SIZE 16
#define FILTER_OFM_NUM_ALIGNED (((FILTER_OFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)
#define FILTER_IFM_NUM_ALIGNED (((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
KERNEL(convolution_bfyx_f16)(
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
#if GROUPED
    const int f_block = get_group_id(1);
    const int group = (f_block * FEATURE_SLICE_SIZE) / FILTER_OFM_NUM;
    const int prev_group_leftover = (FILTER_OFM_NUM * (group + 1)) - (f_block * FEATURE_SLICE_SIZE);
    int groups_per_sub_group = 1;
    if (prev_group_leftover < 16)
        groups_per_sub_group += ((FEATURE_SLICE_SIZE - prev_group_leftover - 1) / FILTER_OFM_NUM) + 1;
#else
    const int f_block = get_group_id(1);
    const int group = split_idx;
    const int groups_per_sub_group = 1;
#endif  // GROUPED

    const int lid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

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

#if BIAS_TERM
    uint bias_offset = f_block * FEATURE_SLICE_SIZE;
    vec_t dst = (vec_t)(INPUT_BLOCK_READ(biases, bias_offset));
#else
    vec_t dst = INPUT0_VAL_ZERO;
#endif  // BIAS_TERM

#ifndef MULTIPLE_GROUPS_INPUT_PRELOAD
    for (uint g = group; g < group + groups_per_sub_group; g++) {
#if GROUPED
        const uint in_split_offset = g * input_fs_pitch * (FILTER_IFM_NUM / FEATURE_SLICE_SIZE);
        const uint filter_split_offset = g * FILTER_GROUPS_PITCH;
        const uint filter_offset = (f_block % (FILTER_OFM_NUM / FEATURE_SLICE_SIZE)) * filter_os_pitch;
#else
        const uint in_split_offset = 0;
        const uint filter_split_offset = 0;
        const uint filter_offset = f_block * filter_os_pitch;
#endif  // GROUPED
        const uint grouped_filter_offset = filter_offset + filter_split_offset;
#else
        const uint in_split_offset = f_block * input_fs_pitch;
        const uint g = lid / (FEATURE_SLICE_SIZE / groups_per_sub_group);
        const uint ofm_in_group = lid % (FEATURE_SLICE_SIZE / groups_per_sub_group);
        const uint grouped_filter_offset = (group + g) * FILTER_GROUPS_PITCH;
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD

        const uint grouped_input_offset = input_offset + in_split_offset;

        for (uint icb = 0; icb < IC_BLOCKS; icb++) {
            __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
            for (int kh = 0; kh < FILTER_SIZE_Y; kh++) {
                if (input_y + kh*DILATION_SIZE_Y < 0 || input_y + kh*DILATION_SIZE_Y >= INPUT0_SIZE_Y)
                    continue;

                INPUT_TYPE line_cache[INPUT_LINE_SIZE];

#if INPUT_LEFTOVERS
                if ((icb+1)*FEATURE_SLICE_SIZE >= FILTER_IFM_NUM)
                {
                    for (int xb = 0; xb < INPUT_LINE_SIZE; xb++)
                    {
                        if (icb*FEATURE_SLICE_SIZE + lid >= FILTER_IFM_NUM)
                            line_cache[xb] = 0;
                        else
                            line_cache[xb] = input[grouped_input_offset +
                                                   icb * input_fs_pitch +
                                                   kh * DILATION_SIZE_Y * input_y_pitch +
                                                   xb * input_x_pitch +
                                                   lid];
                    }
                }
                else
#endif  // INPUT_LEFTOVERS
                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        INPUT_TYPE8 vv = INPUT_BLOCK_READ8(input, grouped_input_offset +
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
                        INPUT_TYPE4 vv = INPUT_BLOCK_READ4(input, grouped_input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_SIZE_Y * input_y_pitch +
                                                                  xb * input_x_pitch);

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = INPUT_BLOCK_READ(input, grouped_input_offset +
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
                        src[i] = line_cache[kw*DILATION_SIZE_X + STRIDE_SIZE_X*i];
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
#error Unsupported input feature size for multiple groups input preload
#endif  // FILTER_IFM_NUM
#else
                    FILTER_TYPE8 wei0 = FILTER_BLOCK_READ8(weights, grouped_filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch);
                    FILTER_TYPE8 wei1 = FILTER_BLOCK_READ8(weights, grouped_filter_offset +
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
#ifndef MULTIPLE_GROUPS_INPUT_PRELOAD
    }
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD
    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) out_vec_t;
    out_vec_t res;

#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            res[i] = FUSED_OPS_RESULT_SCALAR;
#else
            res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif
            if ((f_block*FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
                output[output_offset + i * output_x_pitch + lid] = res[i];
            }
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X) {
#if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            res = FUSED_OPS_RESULT_VEC;
#else
            res = dst;
#endif
            // TODO Generalize for other block sizes
#if OUTPUT_X_BLOCK_SIZE == 8
            OUTPUT_BLOCK_WRITE8(output, output_offset, res);
#elif OUTPUT_X_BLOCK_SIZE == 4
            OUTPUT_BLOCK_WRITE4(output, output_offset, res);
#elif OUTPUT_X_BLOCK_SIZE == 2
            OUTPUT_BLOCK_WRITE2(output, output_offset, res);
#elif OUTPUT_X_BLOCK_SIZE == 1
            OUTPUT_BLOCK_WRITE(output, output_offset, res);
#else
#   error convolution_gpu_bfyx_f16.cl: Unsupported output x block size.
#endif
        } else {
            const int x_tail = OUTPUT_SIZE_X - x;
            for (int i = 0; i < x_tail; i++) {
#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                res[i] = FUSED_OPS_RESULT_SCALAR;
#else
                res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif
                OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, res[i]);
            }
        }
    }
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

#undef INPUT_BLOCK_READ
#undef INPUT_BLOCK_READ2
#undef INPUT_BLOCK_READ4
#undef INPUT_BLOCK_READ8

#undef FILTER_BLOCK_READ8

#undef OUTPUT_BLOCK_WRITE
#undef OUTPUT_BLOCK_WRITE2
#undef OUTPUT_BLOCK_WRITE4
#undef OUTPUT_BLOCK_WRITE8
