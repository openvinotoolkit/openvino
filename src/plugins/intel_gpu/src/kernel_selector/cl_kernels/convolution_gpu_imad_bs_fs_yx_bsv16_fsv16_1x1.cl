// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/imad.cl"
#if QUANTIZATION_TERM
#define ACCUMULATOR_TYPE int
#define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#define ACTIVATION_TYPE float
#define TO_ACTIVATION_TYPE(x) convert_float(x)
#else
#define ACCUMULATOR_TYPE INPUT0_TYPE
#define TO_ACCUMULATOR_TYPE(x) TO_INPUT0_TYPE(x)
#define ACTIVATION_TYPE INPUT0_TYPE
#define TO_ACTIVATION_TYPE(x) TO_INPUT0_TYPE(x)
#endif

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)
#define OUTPUT_TYPE16 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 16)
#define BATCH_SLICE_SIZE 16
#define FEATURE_SLICE_SIZE 16

REQD_SUB_GROUP_SIZE(16)
KERNEL(convolution_gpu_imad_bs_fs_yx_bsv16_fsv16_1x1)(
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights
#if BIAS_TERM
    , const __global BIAS_TYPE *biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint out_x = (uint)get_global_id(0);
    const uint out_y = (uint)get_global_id(1);
    const uint out_f = (uint)get_group_id(2) * 32 % OUTPUT_FEATURE_NUM;
    const uint out_b = ((uint)get_group_id(2) * 32 / OUTPUT_FEATURE_NUM) * 16 + get_sub_group_local_id();

    ACCUMULATOR_TYPE dotProd[32] = {0};
    const int input_x = out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint weights_x_pitch = BATCH_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint input_x_pitch = BATCH_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);

    uint filter_idx = GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, out_f + get_sub_group_local_id(), 0, 0, 0);
    uint filter_idx2 = GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, out_f + 16 + get_sub_group_local_id(), 0, 0, 0);

    uint input_idx = GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(INPUT0, out_b, 0, input_y, input_x);
    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < INPUT0_FEATURE_NUM / 16; k++) {
        uint4 input_val0 = vload4(0, (__global uint *)(conv_input + input_idx));
        uint4 weights_val = vload4(0, (__global uint *)(weights + filter_idx));
        uint4 weights_val2 = vload4(0, (__global uint *)(weights + filter_idx2));

        __attribute__((opencl_unroll_hint(16)))
        for (uint j = 0; j < 16; j++) {
            dotProd[j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[j], AS_INPUT0_TYPE_4(input_val0.s0), as_char4(_sub_group_shuffle(weights_val.s0, j))));
            dotProd[j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[j], AS_INPUT0_TYPE_4(input_val0.s1), as_char4(_sub_group_shuffle(weights_val.s1, j))));
            dotProd[j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[j], AS_INPUT0_TYPE_4(input_val0.s2), as_char4(_sub_group_shuffle(weights_val.s2, j))));
            dotProd[j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[j], AS_INPUT0_TYPE_4(input_val0.s3), as_char4(_sub_group_shuffle(weights_val.s3, j))));

            dotProd[16 + j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[16 + j], AS_INPUT0_TYPE_4(input_val0.s0), as_char4(_sub_group_shuffle(weights_val2.s0, j))));
            dotProd[16 + j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[16 + j], AS_INPUT0_TYPE_4(input_val0.s1), as_char4(_sub_group_shuffle(weights_val2.s1, j))));
            dotProd[16 + j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[16 + j], AS_INPUT0_TYPE_4(input_val0.s2), as_char4(_sub_group_shuffle(weights_val2.s2, j))));
            dotProd[16 + j] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[16 + j], AS_INPUT0_TYPE_4(input_val0.s3), as_char4(_sub_group_shuffle(weights_val2.s3, j))));
        }
        filter_idx += weights_x_pitch;
        filter_idx2 += weights_x_pitch;
        input_idx += input_fs_pitch;
    }

    OUTPUT_TYPE16 results = 0;

    __attribute__((opencl_unroll_hint(2)))
    for (uint j = 0; j < 2; j++) {
        const uint dst_index = GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(OUTPUT, out_b, 16 * j + out_f, out_y, out_x);
#if BIAS_TERM
        ACTIVATION_TYPE bias = biases[out_f + 16 * j + get_sub_group_local_id()];
#endif
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_PRELOAD
#endif
        __attribute__((opencl_unroll_hint(16)))
        for (uint i = 0; i < 16; i++) {

            ACTIVATION_TYPE dequantized = (ACTIVATION_TYPE)0;
#if BIAS_TERM
            dequantized = (ACTIVATION_TYPE)dotProd[16 * j + i] + _sub_group_shuffle(bias, i);
#else
            dequantized = (ACTIVATION_TYPE)dotProd[16 * j + i];
#endif
#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC
#else
            FUSED_OPS
#endif
            OUTPUT_TYPE res = FUSED_OPS_RESULT;
            results[i] = res;
#else
            results[i] = TO_OUTPUT_TYPE(dequantized);
#endif
        }

#if OUTPUT_TYPE_SIZE == 1
        vstore4(as_uint4(results), 0, ((__global uint *)(output + dst_index)));
#else
        __attribute__((opencl_unroll_hint(16)))
        for (uint z = 0; z < 16; z++) {
            output[dst_index + z] = results[z];
        }

#endif
    }
}

#undef BLOCK_LOAD_INPUTS
#undef IN_BLOCK_WIDTH
#undef IN_BLOCK_HEIGHT
#undef PACK
#undef AS_TYPE_N_
#undef AS_TYPE_N
#undef AS_INPUT0_TYPE_4
#undef NUM_FILTERS
