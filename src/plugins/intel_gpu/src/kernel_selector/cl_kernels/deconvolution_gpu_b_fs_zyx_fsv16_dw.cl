// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/fetch_data.cl"

#include "deconvolution_gpu_imad_common.cl"

#define FEATURE_SLICE_SIZE 16

#if X_BLOCK_SIZE == 1
    #define GET_VEC_ELEM(var, idx) var
#else
    #define GET_VEC_ELEM(var, idx) var[idx]
#endif

#define ACCUMULATOR_BLOCK_TYPE      MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, X_BLOCK_SIZE)
#define ACTIVATION_BLOCK_TYPE       MAKE_VECTOR_TYPE(ACTIVATION_TYPE, X_BLOCK_SIZE)
#define OUTPUT_BLOCK_TYPE           MAKE_VECTOR_TYPE(OUTPUT_TYPE, X_BLOCK_SIZE)

#define TO_ACTIVATION_BLOCK_TYPE(x)     CAT(convert_, ACTIVATION_BLOCK_TYPE)(x)
#define TO_OUTPUT_BLOCK_TYPE(x)         CAT(convert_, OUTPUT_BLOCK_TYPE)(x)

#if FILTER_TYPE_SIZE == 1
DECLARE_READ_BLOCK_16(load_weights, FILTER_TYPE)
#elif FILTER_TYPE_SIZE == 2
DECLARE_READ_BLOCK_8(load_weights, FILTER_TYPE)
#else
DECLARE_READ_BLOCK_4(load_weights, FILTER_TYPE)
#endif

#if OUTPUT_TYPE_SIZE == 1
DECLARE_STORE_BLOCK_16(store_output, OUTPUT_TYPE)
#else
DECLARE_STORE_BLOCK_8(store_output, OUTPUT_TYPE)
#endif

#if PRELOAD_INPUT_LINE
#   if INPUT0_TYPE_SIZE
DECLARE_READ_BLOCK_16(preload_input, INPUT0_TYPE)
#   else
DECLARE_READ_BLOCK_8(preload_input, INPUT0_TYPE)
#   endif
#endif

#if PRELOAD_WEIGHTS_LINE
#   if FILTER_TYPE_SIZE == 1
DECLARE_READ_BLOCK_16(preload_weights, FILTER_TYPE)
#   else
DECLARE_READ_BLOCK_8(preload_weights, FILTER_TYPE)
#   endif
#endif

REQD_SUB_GROUP_SIZE(FEATURE_SLICE_SIZE) // attr:no-format
__attribute__((reqd_work_group_size(1, FEATURE_SLICE_SIZE, 1)))
KERNEL(deconvolution_gpu_b_fs_zyx_fsv16_dw)(
        const  __global INPUT0_TYPE *input,
        __global OUTPUT_TYPE *output,
        const __global FILTER_TYPE *weights
#if BIAS_TERM
        , const __global BIAS_TYPE *bias
#endif
#if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
#endif

        )
{
    const uint zyx = (uint)get_global_id(0);
    const uint x = (zyx % (CEIL_DIV(OUTPUT_SIZE_X, X_BLOCK_SIZE))) * X_BLOCK_SIZE;
#if OUTPUT_DIMS <= 4
    const uint y = zyx / (CEIL_DIV(OUTPUT_SIZE_X, X_BLOCK_SIZE));
    const uint z = 0;
#else
    const uint zy = zyx / (CEIL_DIV(OUTPUT_SIZE_X, X_BLOCK_SIZE));
    const uint y = zy % OUTPUT_SIZE_Y;
    const uint z = zy / OUTPUT_SIZE_Y;
#endif
    const uint f_block = get_group_id(1);
    const uint sglid = get_sub_group_local_id();
    const uint fg = f_block * FEATURE_SLICE_SIZE;
    const uint f = fg + sglid;
    const uint b = (uint)get_global_id(2);

    const int input_x = x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int input_y = y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);
    const int input_z = z + PADDING_SIZE_Z - (FILTER_SIZE_Z - 1);

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X +  INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_z_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y +  INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_fs_pitch = input_z_pitch * (INPUT0_PAD_BEFORE_SIZE_Z +  INPUT0_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Z * input_z_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                              INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch +
                              (f_block + input_fs_pad_before) * input_fs_pitch;

    const uint filter_offset = f_block * FEATURE_SLICE_SIZE * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z;

#if BIAS_TERM && ACCUMULATOR_IS_FP
    ACCUMULATOR_BLOCK_TYPE dst = (ACCUMULATOR_BLOCK_TYPE)(DT_BIAS_BLOCK_READ(bias, f_block * FEATURE_SLICE_SIZE));
#else
    ACCUMULATOR_BLOCK_TYPE dst = (ACCUMULATOR_BLOCK_TYPE)(ACCUMULATOR_VAL_ZERO);
#endif

#if PRELOAD_WEIGHTS
    FILTER_TYPE wei[FILTER_SIZE_Z * FILTER_SIZE_Y * FILTER_SIZE_X];

    FUNC_CALL(load_weights)(weights, filter_offset, FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z, wei);
#endif

    INPUT0_TYPE src_val = INPUT0_VAL_ZERO;

#if PRELOAD_INPUT_LINE
    int first_input_x = input_x;
    if (first_input_x % STRIDE_SIZE_X != 0) {
        if (first_input_x >= 0)
            first_input_x = ALIGN(first_input_x, STRIDE_SIZE_X);
        else
            first_input_x = first_input_x / STRIDE_SIZE_X * STRIDE_SIZE_X;
    }
    first_input_x = first_input_x / STRIDE_SIZE_X;

    unroll_for (uint k_z = 0; k_z < FILTER_SIZE_Z; k_z++) {
        const int input_offset_z = input_z + k_z;
        const bool zero_z = (input_offset_z >= INPUT0_SIZE_Z * STRIDE_SIZE_Z) || (input_offset_z < 0) || ((input_offset_z % STRIDE_SIZE_Z) != 0);
        unroll_for (uint k_y = 0; k_y < FILTER_SIZE_Y; k_y++) {
            const int input_offset_y = input_y + k_y;
            const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);
            if (!zero_y && !zero_z) {
                INPUT0_TYPE input_line[INPUT_BLOCK_SIZE_X] = { };
                uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                uint fixed_input_offset_z = (uint)input_offset_z / STRIDE_SIZE_Z;
                uint preload_input_offset = input_offset + fixed_input_offset_z * input_z_pitch +
                                                           fixed_input_offset_y * input_y_pitch;

                if (first_input_x >= 0) {
                    FUNC_CALL(preload_input)(input, preload_input_offset + first_input_x * input_x_pitch, INPUT_BLOCK_SIZE_X, input_line);
                } else {
                    unroll_for (uint xi = 0; xi < INPUT_BLOCK_SIZE_X; ++xi) {
                        if (first_input_x + xi >= 0) {
                            input_line[xi] = DT_INPUT_BLOCK_READ(input, preload_input_offset + first_input_x * input_x_pitch + xi * input_x_pitch);
                        } else {
                            input_line[xi] = 0;
                        }
                    }
                }

#if PRELOAD_WEIGHTS_LINE
            FILTER_TYPE wei[FILTER_SIZE_X] = { };
            FUNC_CALL(preload_weights)(weights,
                                       filter_offset + (FILTER_SIZE_Z - k_z - 1) * FILTER_Z_PITCH * FEATURE_SLICE_SIZE
                                                     + (FILTER_SIZE_Y - k_y - 1) * FILTER_Y_PITCH * FEATURE_SLICE_SIZE,
                                       FILTER_SIZE_X,
                                       wei);
#endif

                unroll_for (uint k_x = 0; k_x < FILTER_SIZE_X; k_x++) {
#   if PRELOAD_WEIGHTS
                    const uint in_idx = (FILTER_SIZE_Z - k_z - 1) * FILTER_Z_PITCH + (FILTER_SIZE_Y - k_y - 1) * FILTER_Y_PITCH + (FILTER_SIZE_X - k_x - 1);
                    FILTER_TYPE wei_val = wei[in_idx];
#   elif PRELOAD_WEIGHTS_LINE
                    FILTER_TYPE wei_val = wei[(FILTER_SIZE_X - k_x - 1)];
#   else
                    const uint in_idx = (FILTER_SIZE_Z - k_z - 1) * FILTER_Z_PITCH + (FILTER_SIZE_Y - k_y - 1) * FILTER_Y_PITCH + (FILTER_SIZE_X - k_x - 1);
                    FILTER_TYPE wei_val = DT_FILTER_BLOCK_READ(weights, filter_offset + in_idx * FEATURE_SLICE_SIZE);
#   endif
                    unroll_for (uint x_block = 0; x_block < X_BLOCK_SIZE; x_block++) {
                        const int input_offset_x = input_x + k_x + x_block;
                        const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);
                        if (!zero_x) {
                            src_val = input_line[(x_block + k_x) / STRIDE_SIZE_X];
                            GET_VEC_ELEM(dst, x_block) += src_val * wei_val;
                        }  // if !zero_x
                    }  // for X_BLOCK_SIZE
                }  // for FILTER_SIZE_X
            }  // if !zero_y && !zero_z
        }  // for FILTER_SIZE_Y
    }  // for FILTER_SIZE_Z
#else
    unroll_for (uint x_block = 0; x_block < X_BLOCK_SIZE; x_block++) {
        unroll_for (uint k_z = 0; k_z < FILTER_SIZE_Z; k_z++) {
            const int input_offset_z = input_z + k_z;
            const bool zero_z = (input_offset_z >= INPUT0_SIZE_Z * STRIDE_SIZE_Z) || (input_offset_z < 0) || ((input_offset_z % STRIDE_SIZE_Z) != 0);
            unroll_for (uint k_y = 0; k_y < FILTER_SIZE_Y; k_y++) {
                const int input_offset_y = input_y + k_y;
                const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);
                unroll_for (uint k_x = 0; k_x < FILTER_SIZE_X; k_x++) {
                    const int input_offset_x = input_x + k_x + x_block;
                    const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);
                    const uint in_idx = (FILTER_SIZE_Z - k_z - 1) * FILTER_Z_PITCH + (FILTER_SIZE_Y - k_y - 1) * FILTER_Y_PITCH + (FILTER_SIZE_X - k_x - 1);
                    if (!zero_z && !zero_y && !zero_x) {
                        uint fixed_input_offset_x = (uint)input_offset_x / STRIDE_SIZE_X;
                        uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                        uint fixed_input_offset_z = (uint)input_offset_z / STRIDE_SIZE_Z;

                        src_val = DT_INPUT_BLOCK_READ(input, input_offset +
                                                             fixed_input_offset_z * input_z_pitch +
                                                             fixed_input_offset_y * input_y_pitch +
                                                             fixed_input_offset_x * input_x_pitch);
#   if PRELOAD_WEIGHTS
                        FILTER_TYPE wei_val = wei[in_idx];
#   else
                        FILTER_TYPE wei_val = DT_FILTER_BLOCK_READ(weights, filter_offset + in_idx * FEATURE_SLICE_SIZE);
#   endif
                        GET_VEC_ELEM(dst, x_block) += src_val * wei_val;
                    }  // if !zero_z && !zero_y && !zero_x
                }  // for FILTER_SIZE_X
            }  // for FILTER_SIZE_Y
        }  // for FILTER_SIZE_Z
    }  // for X_BLOCK_SIZE
#endif

    ACTIVATION_BLOCK_TYPE dequantized = TO_ACTIVATION_BLOCK_TYPE(dst);
#if BIAS_TERM && !ACCUMULATOR_IS_FP
    dequantized += TO_ACTIVATION_TYPE(DT_BIAS_BLOCK_READ(bias, f_block * FEATURE_SLICE_SIZE));
#endif

    OUTPUT_BLOCK_TYPE result;
#if HAS_FUSED_OPS
    FUSED_OPS;
    result = FUSED_OPS_RESULT;
#else
    result = TO_OUTPUT_BLOCK_TYPE(ACTIVATION(dequantized, ACTIVATION_PARAMS));
#endif

    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_z_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_z_pitch * (OUTPUT_PAD_BEFORE_SIZE_Z +  OUTPUT_SIZE_Z + OUTPUT_PAD_AFTER_SIZE_Z);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset =  b * output_b_pitch +
                                (f_block + output_fs_pad_before) * output_fs_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_Z + z) * output_z_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_Y + y) * output_y_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

#if OUTPUT_LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
        unroll_for (uint x_block = 0; x_block < X_BLOCK_SIZE; x_block++) {
            if (OUTPUT_SIZE_X % X_BLOCK_SIZE != 0 && x + X_BLOCK_SIZE >= OUTPUT_SIZE_X && x_block >= OUTPUT_SIZE_X % X_BLOCK_SIZE)
                break;
            if (f_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM)
                output[output_offset + (x + x_block) * output_x_pitch + sglid] = GET_VEC_ELEM(result, x_block);
        }
    }
    else
#endif //  OUTPUT_LEFTOVERS
#if OUTPUT_SIZE_X % X_BLOCK_SIZE != 0
    if (x + X_BLOCK_SIZE >= OUTPUT_SIZE_X) {
        FUNC_CALL(store_output)(output, output_offset + x * output_x_pitch, OUTPUT_SIZE_X % X_BLOCK_SIZE, (OUTPUT_TYPE *)&result);
    } else
#endif
    {
        FUNC_CALL(store_output)(output, output_offset + x * output_x_pitch, X_BLOCK_SIZE, (OUTPUT_TYPE *)&result);
    }
}

#undef FEATURE_SLICE_SIZE

#undef GET_VEC_ELEM

#undef ACCUMULATOR_BLOCK_TYPE
#undef ACTIVATION_BLOCK_TYPE
#undef OUTPUT_BLOCK_TYPE

#undef TO_ACTIVATION_BLOCK_TYPE
#undef TO_OUTPUT_BLOCK_TYPE
