// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/unit_type.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)
#define INPUT0_SIZE_B_WITH_PADDING (INPUT0_PAD_BEFORE_BATCH_NUM + INPUT0_BATCH_NUM + INPUT0_PAD_AFTER_BATCH_NUM)

#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)
#define OUTPUT_SIZE_B_WITH_PADDING (OUTPUT_PAD_BEFORE_BATCH_NUM + OUTPUT_BATCH_NUM + OUTPUT_PAD_AFTER_BATCH_NUM)

// In some cases input padding may be bigger than needed, those variables describe the offset into padding.
#define INPUT0_PADDING_OFFSET_SIZE_X (INPUT0_PAD_BEFORE_SIZE_X - PADDING_SIZE_X)
#define INPUT0_PADDING_OFFSET_SIZE_Y (INPUT0_PAD_BEFORE_SIZE_Y - PADDING_SIZE_Y)

#define ALIGNED_IFM_NUM (((FILTER_IFM_NUM + FSV - 1) / FSV) * FSV)

// ======================================================================================
// Required JIT definitions:
// --------------------------------------------------------------------------------------
// SUB_GROUP_SIZE      - [int] sub-group/simd size; limited to 16
// FSV                 - [int] feature slice size; limted to 32
// FSV_PER_THREAD      - [int] number of features from slice per thread;
//                             must be equal FSV / SUB_GROUP_SIZE
// OUTPUT_BLOCK_WIDTH  - [int] number of elements calculated in x dimension by one thread
// OUTPUT_BLOCK_HEIGHT - [int] number of elements calculated in y dimension by one thread
// ======================================================================================

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_fs_byx_fsv32_1x1)(
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

    UNIT_TYPE in[FSV_PER_THREAD];
    UNIT_TYPE w[FSV * FSV_PER_THREAD];
    UNIT_TYPE out[OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT * FSV_PER_THREAD];

    for (uint out_i = 0; out_i < OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT * FSV_PER_THREAD; ++out_i)
    {
        out[out_i] = UNIT_VAL_ZERO;
    }

    // Calculate offset to first input data element
    const uint in_pitch_x = FSV;
    const uint in_pitch_y = in_pitch_x * INPUT0_SIZE_X_WITH_PADDING;
    const uint in_pitch_b = in_pitch_y * INPUT0_SIZE_Y_WITH_PADDING;
    const uint in_pitch_fs = in_pitch_b * INPUT0_SIZE_B_WITH_PADDING;

    uint input_offset = 0;
    input_offset += (oc * STRIDE_SIZE_X + INPUT0_PADDING_OFFSET_SIZE_X) * in_pitch_x;
    input_offset += (or * STRIDE_SIZE_Y + INPUT0_PADDING_OFFSET_SIZE_Y) * in_pitch_y;
    input_offset += (b + INPUT0_PAD_BEFORE_BATCH_NUM) * in_pitch_b;
    input_offset += (INPUT0_PAD_BEFORE_FEATURE_NUM / FSV) * in_pitch_fs;

    uint weight_offset = 0;
    weight_offset += fs * ALIGNED_IFM_NUM * FSV;

    for (uint ifi_32 = 0; ifi_32 < (FILTER_IFM_NUM + FSV - 1) / FSV; ++ifi_32)
    {
        // ========================================================================
        // Load weights for all input features:
        unroll_for (uint wi = 0; wi + 2 <= FSV; wi += 2)
        {
            UNIT_TYPE4 tmp_read = UNIT_BLOCK_READ4(weights, weight_offset + wi * FSV);
            w[wi * FSV_PER_THREAD + 0] = tmp_read.s0;
            w[wi * FSV_PER_THREAD + 1] = tmp_read.s1;
            w[wi * FSV_PER_THREAD + 2] = tmp_read.s2;
            w[wi * FSV_PER_THREAD + 3] = tmp_read.s3;
        }
        // Move weight offset by FSV output features x FSV input features
        weight_offset += FSV * FSV;
        // ========================================================================

        uint tmp_input_offset = input_offset;
        // ========================================================================
        // Convolution across output block
        unroll_for (uint out_y = 0; out_y < OUTPUT_BLOCK_HEIGHT; ++out_y)
        {
            unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
            {
                // Load input
                UNIT_TYPE2 input_read = UNIT_BLOCK_READ2(input, tmp_input_offset + out_x * STRIDE_SIZE_X * FSV);

                unroll_for (uint in_f = 0; in_f < FSV; ++in_f)
                {
                    unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
                    {
                        UNIT_TYPE in_val = _sub_group_shuffle(input_read[in_f / SUB_GROUP_SIZE], in_f % SUB_GROUP_SIZE);

                        const uint out_idx = out_y * OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD + out_x * FSV_PER_THREAD + out_f;
                        out[out_idx] = mad(w[in_f * FSV_PER_THREAD + out_f], in_val, out[out_idx]);
                    }
                }
            }
            // Move temporary input offset to next strided row
            tmp_input_offset += in_pitch_y * STRIDE_SIZE_Y;
        }
        // ========================================================================
        // Move input offset to next input feature slice
        input_offset += in_pitch_fs;

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
                out[out_y * FSV_PER_THREAD * OUTPUT_BLOCK_WIDTH + out_x * FSV_PER_THREAD + out_f] += biases[bias_index];
            }
#   else // BIAS_PER_OUTPUT
            const uint bias_index = fs * FSV;
            UNIT_TYPE2 bias_read = UNIT_BLOCK_READ2(biases, bias_index);
            out[out_y * FSV_PER_THREAD * OUTPUT_BLOCK_WIDTH + out_x * FSV_PER_THREAD + 0] += bias_read.s0;
            out[out_y * FSV_PER_THREAD * OUTPUT_BLOCK_WIDTH + out_x * FSV_PER_THREAD + 1] += bias_read.s1;
#   endif // BIAS_PER_OUTPUT
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
    output_offset += (b + OUTPUT_PAD_BEFORE_BATCH_NUM)  * out_pitch_b;
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
            output_offset += out_pitch_y;
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
            output_offset += out_pitch_y;
        }
    }
    // ========================================================================
}

#undef INPUT0_SIZE_X_WITH_PADDING
#undef INPUT0_SIZE_Y_WITH_PADDING
#undef INPUT0_SIZE_B_WITH_PADDING

#undef OUTPUT_SIZE_X_WITH_PADDING
#undef OUTPUT_SIZE_Y_WITH_PADDING
#undef OUTPUT_SIZE_B_WITH_PADDING
