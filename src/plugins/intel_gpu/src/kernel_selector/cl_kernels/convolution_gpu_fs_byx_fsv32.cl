// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#define INPUT_TYPE2 MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define BIAS_TYPE2 MAKE_VECTOR_TYPE(BIAS_TYPE, 2)
#define FILTER_TYPE2 MAKE_VECTOR_TYPE(FILTER_TYPE, 2)
#define ACTIVATION_TYPE2 MAKE_VECTOR_TYPE(ACTIVATION_TYPE, 2)
#define OUTPUT_TYPE2 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2)
#define TO_OUTPUT_TYPE2 CAT(convert_, OUTPUT_TYPE2)

// ======================================================================================
// Required JIT definitions:
// --------------------------------------------------------------------------------------
// SUB_GROUP_SIZE     - [int] sub-group/simd size; limited to 16
// FSV                - [int] feature slice size; limted to 32
// FSV_PER_THREAD     - [int] number of features from slice per thread;
//                            must be equal FSV / SUB_GROUP_SIZE
// OUTPUT_BLOCK_WIDTH - [int] number of elements calculated in x dimension by one thread
// INPUT_BLOCK_WIDTH  - [int] number of continous input elements to calculate output
// ======================================================================================


REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_fs_byx_fsv32)(
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
    uint oc = (uint)get_global_id(0) * OUTPUT_BLOCK_WIDTH;
    uint or = get_global_id(1);
    uint fs_b_id = get_group_id(2);
    uint sglid = get_sub_group_local_id();

    uint fs = fs_b_id / INPUT0_BATCH_NUM;
    uint b = fs_b_id - fs * INPUT0_BATCH_NUM;

    INPUT0_TYPE in[INPUT_BLOCK_WIDTH * FSV_PER_THREAD];
    FILTER_TYPE w[FSV_PER_THREAD];
    ACCUMULATOR_TYPE out[OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD] = { ACCUMULATOR_VAL_ZERO };

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
    weight_offset += fs * FILTER_SIZE_X * FILTER_SIZE_Y * ALIGNED_IFM_NUM * FSV;

    for (uint ifi_32 = 0; ifi_32 < (FILTER_IFM_NUM + FSV - 1) / FSV; ++ifi_32)
    {

        uint tmp_input_offset = input_offset;
        for (uint in_y = 0; in_y < FILTER_SIZE_Y; ++in_y)
        {
            // ====================================================================
            // Load input:
            uint in_x = 0;
            unroll_for (; in_x + 2 <= INPUT_BLOCK_WIDTH; in_x += 2)
            {
                INPUT_TYPE4 tmp_read = DT_INPUT_BLOCK_READ4(input, tmp_input_offset + in_x * FSV);
                in[in_x * FSV_PER_THREAD + 0] = tmp_read.s0;
                in[in_x * FSV_PER_THREAD + 1] = tmp_read.s1;
                in[in_x * FSV_PER_THREAD + 2] = tmp_read.s2;
                in[in_x * FSV_PER_THREAD + 3] = tmp_read.s3;
            }
            unroll_for (; in_x < INPUT_BLOCK_WIDTH; ++in_x)
            {
                INPUT_TYPE2 tmp_read = DT_INPUT_BLOCK_READ2(input, tmp_input_offset + in_x * FSV);
                in[in_x * FSV_PER_THREAD + 0] = tmp_read.s0;
                in[in_x * FSV_PER_THREAD + 1] = tmp_read.s1;
            }
            // ====================================================================

            // Move temporary input offset to next row
            tmp_input_offset += DILATION_SIZE_Y * in_pitch_y;

            uint tmp_weight_offset = weight_offset;

            // ====================================================================
            // Perform convolutions with loaded input features
            unroll_for (uint ifii = 0; ifii < FSV; ++ifii)
            {

                unroll_for (uint f_x = 0; f_x < FILTER_SIZE_X; ++f_x)
                {
                    // Load weights
                    FILTER_TYPE2 tmp_read = DT_FILTER_BLOCK_READ2(weights, tmp_weight_offset + f_x * FSV);
                    w[0] = tmp_read.s0;
                    w[1] = tmp_read.s1;

                    unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
                    {
                        unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
                        {
                            INPUT0_TYPE in_val = _sub_group_shuffle(
                                in[(out_x * STRIDE_SIZE_X + f_x * DILATION_SIZE_X) * FSV_PER_THREAD + ifii / SUB_GROUP_SIZE],
                                ifii % SUB_GROUP_SIZE);

                            const uint out_idx = out_x * FSV_PER_THREAD + out_f;
                            out[out_idx] = mad(TO_ACCUMULATOR_TYPE(in_val), TO_ACCUMULATOR_TYPE(w[out_f]), out[out_idx]);
                        }
                    }

                }
                // Move temporary weight offset to next input feature
                tmp_weight_offset += FILTER_SIZE_Y * FILTER_SIZE_X * FSV;
            }
            // ====================================================================
            // Move weight offset to next row
            weight_offset += FILTER_SIZE_X * FSV;
        }
        // Move input offset to next input feature slice
        input_offset += in_pitch_fs;
        // Move weight offset to next input feature slice (FSV input features)
        //  minus offset added by moving FILTER_SIZE_Y times to new row
        weight_offset += FSV * FILTER_SIZE_Y * FILTER_SIZE_X * FSV // FSV * input filter feature pitch
                       - FILTER_SIZE_Y * FILTER_SIZE_X * FSV;      // FILTER_SIZE_Y * filter y pitch
    }
    // ========================================================================
    // Bias
#if BIAS_TERM
    unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
    {
#   if BIAS_PER_OUTPUT
        // TODO Change bias format to use block reads
        unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
        {
            const uint bias_index = (fs * FSV + out_f * SUB_GROUP_SIZE + sglid) * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
                                    or * OUTPUT_SIZE_X +
                                    (oc + out_x);
            out[out_x * FSV_PER_THREAD + out_f] += TO_ACCUMULATOR_TYPE(biases[bias_index]);
        }
#   else // BIAS_PER_OUTPUT
        const uint bias_index = fs * FSV;
        BIAS_TYPE2 bias_read = DT_BIAS_BLOCK_READ2(biases, bias_index);
        out[out_x * FSV_PER_THREAD + 0] += TO_ACCUMULATOR_TYPE(bias_read.s0);
        out[out_x * FSV_PER_THREAD + 1] += TO_ACCUMULATOR_TYPE(bias_read.s1);
#   endif // BIAS_PER_OUTPUT
    }
#endif // BIAS_TERM
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
    output_offset += (fs + pad_before_fs) * out_pitch_fs;

    const bool full_f = OUTPUT_FEATURE_NUM % FSV == 0 || fs * FSV + FSV <= OUTPUT_FEATURE_NUM;
    const bool full_x = OUTPUT_SIZE_X % OUTPUT_BLOCK_WIDTH == 0 || oc + OUTPUT_BLOCK_WIDTH <= OUTPUT_SIZE_X;

    ACTIVATION_TYPE res[OUTPUT_BLOCK_WIDTH * FSV_PER_THREAD] = { ACTIVATION_VAL_ZERO };

    if (full_f && full_x)
    {
        // Case without bounds checking
        unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
        {
            ACTIVATION_TYPE2 tmp_write = (ACTIVATION_TYPE2)(out[out_x * FSV_PER_THREAD + 0],
                                                            out[out_x * FSV_PER_THREAD + 1]);
            OUTPUT_TYPE2 final_result;
#if HAS_FUSED_OPS
            unroll_for (uint out_f = 0; out_f < 2; ++out_f)
            {
                { FUSED_OPS_VEC_ELEM; final_result[out_f] = FUSED_OPS_RESULT_VEC_ELEM; }
            }
#else
            final_result = TO_OUTPUT_TYPE2(ACTIVATION(tmp_write, ACTIVATION_PARAMS));
#endif
            DT_OUTPUT_BLOCK_WRITE2(output, output_offset, final_result);
            output_offset += FSV;
        }
    }
    else
    {
        unroll_for (uint out_x = 0; out_x < OUTPUT_BLOCK_WIDTH; ++out_x)
        {
            unroll_for (uint out_f = 0; out_f < FSV_PER_THREAD; ++out_f)
            {
                if (oc + out_x < OUTPUT_SIZE_X && fs * FSV + sglid + out_f * SUB_GROUP_SIZE < OUTPUT_FEATURE_NUM)
                {
                    const uint out_idx = out_x * FSV_PER_THREAD + out_f;
                    ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(out[out_idx]);
                    OUTPUT_TYPE final_result;
#if HAS_FUSED_OPS
                    { FUSED_OPS_SCALAR; final_result = FUSED_OPS_RESULT_SCALAR; }
#else
                    final_result = TO_OUTPUT_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
#endif
                    output[output_offset + sglid] = final_result;
                }
                output_offset += SUB_GROUP_SIZE;
            }
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

#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef BIAS_TYPE2
#undef FILTER_TYPE2
#undef ACTIVATION_TYPE2
#undef OUTPUT_TYPE2
#undef TO_OUTPUT_TYPE2
