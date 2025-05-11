// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/unit_type.cl"
#include "include/batch_headers/fetch_data.cl"

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)

#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)
#define OUTPUT_SIZE_B_WITH_PADDING (OUTPUT_PAD_BEFORE_BATCH_NUM + OUTPUT_BATCH_NUM + OUTPUT_PAD_AFTER_BATCH_NUM)

// In some cases input padding may be bigger than needed, those variables describe the offset into padding.
#define INPUT0_PADDING_OFFSET_SIZE_X (INPUT0_PAD_BEFORE_SIZE_X - PADDING_SIZE_X)
#define INPUT0_PADDING_OFFSET_SIZE_Y (INPUT0_PAD_BEFORE_SIZE_Y - PADDING_SIZE_Y)

// ======================================================================================
// Required JIT definitions:
// --------------------------------------------------------------------------------------
// SUB_GROUP_SIZE      - [int] sub-group/simd size; limited to 16
// FSV                 - [int] feature slice size; limted to 32
// FSV_PER_THREAD      - [int] number of features from slice per thread;
//                             must be equal FSV / SUB_GROUP_SIZE
// OUTPUT_BLOCK_WIDTH  - [int] number of elements calculated in x dimension by one thread
// OUTPUT_BLOCK_HEIGHT - [int] number of output rows calculated by one thread
// INPUT_BLOCK_WIDTH   - [int] number of continous input elements to calculate output,
//                             rounded to multiple of sub-group size
// INPUT_BLOCK_HEIGHT  - [int] number of input rows needed to calculate output
// ======================================================================================

// For loading simd will be aligned along x dimension, this macro gives the number of
// elements to load for each simd-lane
#define INPUT_BLOCK_WIDTH_EL_CNT ((INPUT_BLOCK_WIDTH) / (SUB_GROUP_SIZE))

// In order to use block reads input offset must be aligned to 4 bytes
// To ensure this:
// 1. Every offset move to next row must be aligned,
//    so INPUT0_SIZE_X_WITH_PADDING must be aligned
// 2. Starting offset must be aligned, for this alignment must hold for:
//    OUTPUT_BLOCK_WIDTH * STRIDE_SIZE_X
//    INPUT0_PADDING_OFFSET_SIZE_X
//    OUTPUT_BLOCK_HEIGHT * STRIDE_SIZE_Y * INPUT0_SIZE_X_WITH_PADDING (covered by 1.)
//    INPUT0_PADDING_OFFSET_SIZE_Y * INPUT0_SIZE_X_WITH_PADDING        (covered by 1.)
//    INPUT0_SIZE_X_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING * FILTER_IFM_NUM (covered by 1.)
// 3. Every offset move to next feature by
//    INPUT0_SIZE_X_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING must be aligned (covered by 1.)

#define CAN_USE_BLOCK_READ                                          \
    (STRIDE_SIZE_X * OUTPUT_BLOCK_WIDTH * UNIT_TYPE_SIZE) % 4 == 0  \
    && (INPUT0_SIZE_X_WITH_PADDING * UNIT_TYPE_SIZE) % 4 == 0       \
    && (INPUT0_PADDING_OFFSET_SIZE_X * UNIT_TYPE_SIZE) % 4 == 0     \
    && (INPUT0_PAD_BEFORE_FEATURE_NUM * UNIT_TYPE_SIZE) % 4 == 0

#define ALIGNED_IFM_NUM (((FILTER_IFM_NUM + FSV - 1) / FSV) * FSV)

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_bfyx_to_fs_byx_fsv32)(
    __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    __global UNIT_TYPE* weights
#if BIAS_TERM
    , __global UNIT_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    uint oc = (uint)get_global_id(0) * OUTPUT_BLOCK_WIDTH;
    uint or = (uint)get_global_id(1) * OUTPUT_BLOCK_HEIGHT;
    uint fs_b_id = get_group_id(2);
    uint sglid = get_sub_group_local_id();


    uint fs = fs_b_id / INPUT0_BATCH_NUM;
    uint b = fs_b_id - fs * INPUT0_BATCH_NUM;

    UNIT_TYPE in[INPUT_BLOCK_HEIGHT * INPUT_BLOCK_WIDTH_EL_CNT];
    UNIT_TYPE w[FSV_PER_THREAD];
    UNIT_TYPE out[OUTPUT_BLOCK_HEIGHT * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD];

    for (uint out_i = 0; out_i < OUTPUT_BLOCK_HEIGHT * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD; ++out_i)
    {
        out[out_i] = UNIT_VAL_ZERO;
    }

    uint input_offset = oc * STRIDE_SIZE_X + INPUT0_PADDING_OFFSET_SIZE_X;
    input_offset += (or * STRIDE_SIZE_Y + INPUT0_PADDING_OFFSET_SIZE_Y) * INPUT0_SIZE_X_WITH_PADDING;
    input_offset += INPUT0_PAD_BEFORE_FEATURE_NUM * INPUT0_FEATURE_PITCH;
    input_offset += (b + INPUT0_PAD_BEFORE_BATCH_NUM) * INPUT0_BATCH_PITCH;

    uint weight_offset = 0;
    weight_offset += fs * FILTER_SIZE_X * FILTER_SIZE_Y * ALIGNED_IFM_NUM * FSV;

    for (uint ifi = 0; ifi < FILTER_IFM_NUM; ++ifi)
    {
        uint tmp_input_offset = input_offset;
        // ====================================================================
        // Load input:
        unroll_for (uint in_y = 0; in_y < INPUT_BLOCK_HEIGHT; ++in_y)
        {
            uint in_x = 0;

#if CAN_USE_BLOCK_READ
            unroll_for (; in_x + 4 <= INPUT_BLOCK_WIDTH_EL_CNT; in_x += 4)
            {
                UNIT_TYPE4 tmp_read = UNIT_BLOCK_READ4(input, tmp_input_offset + in_x * SUB_GROUP_SIZE);
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 0] = tmp_read.s0;
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 1] = tmp_read.s1;
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 2] = tmp_read.s2;
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 3] = tmp_read.s3;
            }
            unroll_for (; in_x + 2 <= INPUT_BLOCK_WIDTH_EL_CNT; in_x += 2)
            {
                UNIT_TYPE2 tmp_read = UNIT_BLOCK_READ2(input, tmp_input_offset + in_x * SUB_GROUP_SIZE);
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 0] = tmp_read.s0;
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 1] = tmp_read.s1;
            }
            unroll_for (; in_x + 1 <= INPUT_BLOCK_WIDTH_EL_CNT; in_x += 1)
            {
                UNIT_TYPE tmp_read = UNIT_BLOCK_READ(input, tmp_input_offset + in_x * SUB_GROUP_SIZE);
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 0] = tmp_read;
            }
#else // CAN_USE_BLOCK_READ
            // TODO Optimize this path by using vload's
            unroll_for (; in_x + 1 <= INPUT_BLOCK_WIDTH_EL_CNT; in_x += 1)
            {
                UNIT_TYPE tmp_read = input[tmp_input_offset + in_x * SUB_GROUP_SIZE + sglid];
                in[in_y * INPUT_BLOCK_WIDTH_EL_CNT + in_x + 0] = tmp_read;
            }
#endif // CAN_USE_BLOCK_READ

            // Move temporary input offset to next row
            tmp_input_offset += INPUT0_SIZE_X_WITH_PADDING;
        }
        // ====================================================================

        // ====================================================================
        // Perform convolution on loaded input
        unroll_for (uint f_y = 0; f_y < FILTER_SIZE_Y; ++f_y)
        {
            unroll_for (uint f_x = 0; f_x < FILTER_SIZE_X; ++f_x)
            {
                // Load weights
                UNIT_TYPE2 tmp_read = UNIT_BLOCK_READ2(weights, weight_offset);
                w[0] = tmp_read.s0;
                w[1] = tmp_read.s1;

                weight_offset += FSV;
                // Actual convolution
                unroll_for (uint out_y = 0; out_y < OUTPUT_BLOCK_HEIGHT; ++out_y)
                {
                    unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
                    {
                        unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
                        {
                            // Value to convolve is at 2D index in[out_y * STRIDE_SIZE_Y + f_y * DILATION_SIZE_Y][out_x * STRIDE_SIZE_X + f_x * DILATION_SIZE_X].
                            // With simd along x dimension:
                            // (out_x * STRIDE_SIZE_X + f_x * DILATION_SIZE_X) / SUB_GROUP_SIZE - element number in simd-lane;
                            // (out_x * STRIDE_SIZE_X + f_x * DILATION_SIZE_X) % SUB_GROUP_SIZE - simd-lane with that element.
                            UNIT_TYPE in_val = _sub_group_shuffle(
                                in[(out_y * STRIDE_SIZE_Y + f_y * DILATION_SIZE_Y) * INPUT_BLOCK_WIDTH_EL_CNT +
                                   (out_x * STRIDE_SIZE_X + f_x * DILATION_SIZE_X) / SUB_GROUP_SIZE],
                                (out_x * STRIDE_SIZE_X + f_x * DILATION_SIZE_X) % SUB_GROUP_SIZE);



                            const uint out_idx = out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + out_f;
                            out[out_idx] = mad(in_val, w[out_f], out[out_idx]);
                        }
                    }
                }
            }
            // ====================================================================
        }

        // Move input offset to next input feature
        input_offset += INPUT0_SIZE_X_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING;
        // No need to move weight offset as it has already been moved to next
        // input filter feature by FILTER_SIZE_Y * FILTER_SIZE_X times moving
        // by FSV.
    }

    // ========================================================================
    // Bias
#if BIAS_TERM
    unroll_for (uint out_y = 0; out_y < OUTPUT_BLOCK_HEIGHT; ++out_y)
    {
        unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
        {
#   if BIAS_PER_OUTPUT
            // TODO Change bias format to use block reads
            unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
            {
                const uint bias_index = (fs * FSV + out_f * SUB_GROUP_SIZE + sglid) * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
                                        (or + out_y) * OUTPUT_SIZE_X +
                                        (oc + out_x);
                out[out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + out_f]
                    += biases[bias_index];
            }
#   else
            const uint bias_index = fs * FSV;
            UNIT_TYPE2 bias_read = UNIT_BLOCK_READ2(biases, bias_index);
            out[out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + 0] += bias_read.s0;
            out[out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + 1] += bias_read.s1;
#   endif

        }
    }
#endif // BIAS_TERM
    // ========================================================================

    // ========================================================================
    // Activation
    unroll_for (uint out_y = 0; out_y < OUTPUT_BLOCK_HEIGHT; ++out_y)
    {
        unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
        {
            unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
            {
                const uint out_idx = out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + out_f;
                out[out_idx] = ACTIVATION(out[out_idx], ACTIVATION_PARAMS);
            }
        }
    }
    // ========================================================================

    // ========================================================================
    // Store results:
    // Calculate offset to first output element
    const uint out_pitch_x = FSV;
    const uint out_pitch_y = out_pitch_x * OUTPUT_SIZE_X_WITH_PADDING;
    const uint out_pitch_b = out_pitch_y * OUTPUT_SIZE_Y_WITH_PADDING;
    const uint out_pitch_fs = out_pitch_b * OUTPUT_SIZE_B_WITH_PADDING;

    const uint pad_before_fs = (OUTPUT_PAD_BEFORE_FEATURE_NUM / FSV);

    uint output_offset = 0;
    output_offset += (oc + OUTPUT_PAD_BEFORE_SIZE_X) * out_pitch_x;
    output_offset += (or + OUTPUT_PAD_BEFORE_SIZE_Y) * out_pitch_y;
    output_offset += (b + OUTPUT_PAD_BEFORE_BATCH_NUM) * out_pitch_b;
    output_offset += (pad_before_fs + fs) * out_pitch_fs;

    const bool full_f = OUTPUT_FEATURE_NUM % FSV == 0 || fs * FSV + FSV <= OUTPUT_FEATURE_NUM;
    const bool full_x = OUTPUT_SIZE_X % OUTPUT_BLOCK_WIDTH == 0 || oc + OUTPUT_BLOCK_WIDTH <= OUTPUT_SIZE_X;
    const bool full_y = OUTPUT_SIZE_Y % OUTPUT_BLOCK_HEIGHT == 0 || or + OUTPUT_BLOCK_HEIGHT <= OUTPUT_SIZE_Y;

    if (full_f && full_x && full_y)
    {
        // Case without bounds checking
        unroll_for (uint out_y = 0; out_y < OUTPUT_BLOCK_HEIGHT; ++out_y)
        {
            unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
            {
                UNIT_TYPE2 tmp_write = (UNIT_TYPE2)(out[out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + 0],
                                                    out[out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + 1]);
#if HAS_FUSED_OPS
                unroll_for (uint out_f = 0; out_f < 2; ++out_f)
                {
                    { FUSED_OPS_VEC_ELEM; tmp_write[out_f] = FUSED_OPS_RESULT_VEC_ELEM; }
                }
#endif
                UNIT_BLOCK_WRITE2(output, output_offset + out_x * FSV, tmp_write);
            }
            // Move output offset to next row
            output_offset += FSV * OUTPUT_SIZE_X_WITH_PADDING;
        }
    }
    else
    {
        unroll_for (uint out_y = 0; out_y < OUTPUT_BLOCK_HEIGHT; ++out_y)
        {
            unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
            {
                unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
                {
                    if (oc + out_x < OUTPUT_SIZE_X
                     && or + out_y < OUTPUT_SIZE_Y
                     && fs * FSV + sglid + out_f * SUB_GROUP_SIZE < OUTPUT_FEATURE_NUM)
                    {
                        const uint out_idx = out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + out_f;
#if HAS_FUSED_OPS
                        { FUSED_OPS_SCALAR; out[out_idx] = FUSED_OPS_RESULT_SCALAR; }
#endif
                        output[output_offset + out_x * FSV + out_f * SUB_GROUP_SIZE + sglid] = out[out_idx];
                    }
                }
            }
            // Move output offset to next row
            output_offset += FSV * OUTPUT_SIZE_X_WITH_PADDING;
        }
    }
    // ========================================================================
}

#undef INPUT0_SIZE_X_WITH_PADDING
#undef INPUT0_SIZE_Y_WITH_PADDING

#undef OUTPUT_SIZE_X_WITH_PADDING
#undef OUTPUT_SIZE_Y_WITH_PADDING
#undef OUTPUT_SIZE_B_WITH_PADDING

#undef INPUT_BLOCK_WIDTH_EL_CNT
