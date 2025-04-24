// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/unit_type.cl"

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)

// ============================================================================
// Required JIT definitions:
// ----------------------------------------------------------------------------
// SUB_GROUP_SIZE      - [int] sub-group/simd size. Limited to 16.
// WG_HEIGHT           - [int] number of sub-groups in work-group along b dimension.
// OUTPUT_BLOCK_SIZE_B - [int] number of batches to process by one thread.
// ============================================================================

#define FSV 32
#define OUTPUT_BLOCK_SIZE_F 32
#define OUTPUT_BLOCK_SIZE_F_EL_CNT 2
#define ALIGNED_INPUT0_FEATURE_NUM (((INPUT0_FEATURE_NUM + FSV - 1) / FSV) * FSV)
#define MEMORY_ALIGN 16

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, WG_HEIGHT, SUB_GROUP_SIZE)))
KERNEL(fully_connected_gpu_fs_byx_fsv32)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weights
#if BIAS_TERM
    , const __global UNIT_TYPE* biases
#endif
)
{
    const uint of = (uint)get_group_id(0) * OUTPUT_BLOCK_SIZE_F;
#if WG_HEIGHT == 1
    const uint ob = (uint)get_group_id(1) * OUTPUT_BLOCK_SIZE_B;
#else
    const uint ob = ((uint)get_group_id(1) * WG_HEIGHT + (uint)get_sub_group_id()) * OUTPUT_BLOCK_SIZE_B;

    // In case batch number is not evenly divisible by work-group processed batch number,
    // early return threads from work-group that target batches outside.
    if (OUTPUT_BATCH_NUM % (WG_HEIGHT * OUTPUT_BLOCK_SIZE_B) != 0 && ob > OUTPUT_BATCH_NUM)
        return;
#endif
    const uint sglid = get_sub_group_local_id();

    UNIT_TYPE2 in[OUTPUT_BLOCK_SIZE_B] = { };
    UNIT_TYPE2 out[OUTPUT_BLOCK_SIZE_B] = { };

    unroll_for (uint oi = 0; oi < OUTPUT_BLOCK_SIZE_B; ++oi)
    {
        out[oi] = UNIT_VAL_ZERO;
    }
    // ========================================================================
    // [constexpr] Input and weight pitches
    const uint input_x_pitch = FSV;
    const uint input_y_pitch = input_x_pitch * INPUT0_SIZE_X_WITH_PADDING;
    const uint input_b_pitch = input_y_pitch * INPUT0_SIZE_Y_WITH_PADDING;
    const uint input_fs_pitch = input_b_pitch * INPUT0_BATCH_NUM;

    const uint weights_x_pitch = FSV;
    const uint weights_y_pitch = weights_x_pitch * INPUT0_SIZE_X;
    const uint weights_i_pitch = weights_y_pitch * INPUT0_SIZE_Y;
    const uint weights_os_pitch = weights_i_pitch * ALIGNED_INPUT0_FEATURE_NUM;
    // ========================================================================

    // Input offset adjustement by padding
    const uint input_base_offset = INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch
                                 + INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch;
    uint input_offset = input_base_offset + ob * input_b_pitch;
    uint weights_offset = (of / FSV) * weights_os_pitch;

    // Loop over FSV input features from one slice in one iteration
    for (uint fi = 0; fi < (INPUT0_FEATURE_NUM + FSV - 1) / FSV; ++fi)
    {
        uint tmp_input_offset = input_offset;
        uint tmp_weights_offset = weights_offset;

        // Loop over spatial dimensions of input
        for (uint fii = 0; fii < INPUT0_SIZE_X * INPUT0_SIZE_Y; ++fii)
        {
            // Read input - 32 in_f x OUTPUT_BLOCK_SIZE_B b
            unroll_for (uint obi = 0; obi < OUTPUT_BLOCK_SIZE_B; ++obi)
            {
                const uint batched_input_offset =
                    tmp_input_offset + obi * input_b_pitch;
                in[obi] = UNIT_BLOCK_READ2(input, batched_input_offset);
            }

            unroll_for (uint ofi = 0; ofi < FSV; ++ofi)
            {
                // Read weights - OUTPUT_BLOCK_SIZE_F out_f
                const uint feature_weights_offset = tmp_weights_offset + ofi * weights_i_pitch;
                UNIT_TYPE2 w = UNIT_BLOCK_READ2(weights, feature_weights_offset);

                unroll_for (uint obi = 0; obi < OUTPUT_BLOCK_SIZE_B; ++obi)
                {
                    UNIT_TYPE in_val = _sub_group_shuffle(in[obi][ofi / SUB_GROUP_SIZE],
                                                               ofi % SUB_GROUP_SIZE);
                    out[obi] = mad(w, in_val, out[obi]);
                }
            }
            // Move temporary offsets to next spatial (x/y)
            tmp_input_offset += FSV;
            tmp_weights_offset += FSV;
        }
        // Move input offset to next feature slice (FSV features)
        input_offset += input_fs_pitch;
        weights_offset += FSV * weights_i_pitch;
    }

    // ========================================================================
    // Bias
#if BIAS_TERM
    unroll_for (uint obi = 0; obi < OUTPUT_BLOCK_SIZE_B; ++obi)
    {
        const uint bias_index = of;

        UNIT_TYPE2 bias = UNIT_BLOCK_READ2(biases, bias_index);
        out[obi] += bias;
    }
#endif
    // ========================================================================

    // ========================================================================
    // Activation
    unroll_for (uint obi = 0; obi < OUTPUT_BLOCK_SIZE_B; ++obi)
    {
        out[obi] = ACTIVATION(out[obi], ACTIVATION_PARAMS);
    }
    // ========================================================================

    // ========================================================================
    // Write output
    uint output_offset = of + ob * OUTPUT_FEATURE_NUM;

#if OUTPUT_FEATURE_NUM % OUTPUT_BLOCK_SIZE_F != 0
    const bool full_of = OUTPUT_FEATURE_NUM % OUTPUT_BLOCK_SIZE_F == 0 || of + OUTPUT_BLOCK_SIZE_F <= OUTPUT_FEATURE_NUM;
    const bool full_ob = OUTPUT_BATCH_NUM % OUTPUT_BLOCK_SIZE_B == 0 || ob + OUTPUT_BLOCK_SIZE_B <= OUTPUT_BATCH_NUM;

    unroll_for (uint obi = 0; obi < OUTPUT_BLOCK_SIZE_B; ++obi)
    {
        const bool correct_output_offset = output_offset % (MEMORY_ALIGN / OUTPUT_TYPE_SIZE) == 0;
        if (full_of && full_ob && correct_output_offset)
        {
            UNIT_BLOCK_WRITE2(output, output_offset, out[obi]);
            // Move output offset to next batch
            output_offset += OUTPUT_FEATURE_NUM;
        }
        else
        {
            unroll_for (uint ofi = 0; ofi < OUTPUT_BLOCK_SIZE_F_EL_CNT; ++ofi)
            {
                if (ob + obi < OUTPUT_BATCH_NUM && of + ofi * SUB_GROUP_SIZE + sglid < OUTPUT_FEATURE_NUM)
                {
                    const uint feature_output_offset = output_offset + ofi * SUB_GROUP_SIZE + sglid;
                    output[feature_output_offset] = out[obi][ofi];
                }
            }
            // Move output offset to next batch
            output_offset += OUTPUT_FEATURE_NUM;
        }
    }
#else
#if OUTPUT_BATCH_NUM % OUTPUT_BLOCK_SIZE_B != 0
    if (ob + OUTPUT_BLOCK_SIZE_B > OUTPUT_BATCH_NUM) {
        unroll_for (uint obi = 0; obi < OUTPUT_BATCH_NUM - ob; ++obi)
        {
            UNIT_BLOCK_WRITE2(output, output_offset, out[obi]);
            // Move output offset to next batch
            output_offset += OUTPUT_FEATURE_NUM;
        }
    }
    else
#endif
    {
        // Case without bounds checking
        unroll_for (uint obi = 0; obi < OUTPUT_BLOCK_SIZE_B; ++obi)
        {
            UNIT_BLOCK_WRITE2(output, output_offset, out[obi]);
            // Move output offset to next batch
            output_offset += OUTPUT_FEATURE_NUM;
        }
    }
#endif  // OUTPUT_FEATURE_NUM % OUTPUT_BLOCK_SIZE_F != 0
    // ========================================================================
}
