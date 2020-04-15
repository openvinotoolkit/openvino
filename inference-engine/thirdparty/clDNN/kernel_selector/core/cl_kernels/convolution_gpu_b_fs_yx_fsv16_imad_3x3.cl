// Copyright (c) 2018-2019 Intel Corporation
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

#include "include/common.cl"
#include "include/fetch.cl"
#include "include/imad.cl"
#include "include/mmad.cl"

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

#define MAKE_VECTOR_TYPE(elem_type, size) CAT(elem_type, size)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// int8 conv_input and weights data is packed to int32 "batches",
// int/uint pointers here instead of INPUT0_TYPE/FILTER_TYPE for convenience
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(1, 1, 16)))
KERNEL(convolution_gpu_b_fs_yx_fsv16_imad_3x3)(
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights,
#if BIAS_TERM
    const __global BIAS_TYPE *biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx) {

    #define LUT_VALUE_CLAMP(x) ((x) < (OUT_BLOCK_WIDTH - 1) * STRIDE_SIZE_X + FILTER_SIZE_X ? (x) : 0)
    const int tmp[16] = {
        LUT_VALUE_CLAMP(0),
        LUT_VALUE_CLAMP(1),
        LUT_VALUE_CLAMP(2),
        LUT_VALUE_CLAMP(3),
        LUT_VALUE_CLAMP(4),
        LUT_VALUE_CLAMP(5),
        LUT_VALUE_CLAMP(6),
        LUT_VALUE_CLAMP(7),
        LUT_VALUE_CLAMP(8),
        LUT_VALUE_CLAMP(9),
        LUT_VALUE_CLAMP(10),
        LUT_VALUE_CLAMP(11),
        LUT_VALUE_CLAMP(12),
        LUT_VALUE_CLAMP(13),
        LUT_VALUE_CLAMP(14),
        LUT_VALUE_CLAMP(15)
    };
    #undef LUT_VALUE_CLAMP

    const uint out_x = (uint)get_global_id(0) * OUT_BLOCK_WIDTH;
    const uint out_y = get_global_id(1);
    const uint out_b = (uint)(get_group_id(2) * OFM_SIZE_PER_SIMD) / ALIGN(OUTPUT_FEATURE_NUM, OFM_SIZE_PER_SIMD);
    const uint out_fg = (uint)(get_group_id(2) * OFM_SIZE_PER_SIMD) % ALIGN(OUTPUT_FEATURE_NUM, OFM_SIZE_PER_SIMD);
    const uint out_f = out_fg + get_sub_group_local_id();
    ACCUMULATOR_TYPE dotProd[OUT_BLOCK_WIDTH * OFM_BLOCKS_PER_SIMD] = {0};
    const int input_x = out_x * STRIDE_SIZE_X - PADDING_SIZE_X;

    const int input_y = out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    uint filter_idx  = GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, out_f, 0, 0, 0);
#if OFM_BLOCKS_PER_SIMD == 2
    uint filter_idx2 = GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, out_f + 16, 0, 0, 0);
#endif

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < CEIL_DIV(INPUT0_FEATURE_NUM, 16); k++) {
        __attribute__((opencl_unroll_hint(1)))
        for (uint j = 0; j < FILTER_SIZE_Y; j++) {
            uint input_idx = GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, out_b, k * 16, input_y + j, input_x + tmp[get_sub_group_local_id()]);
            uint4 input_val0 = vload4(0, (__global uint *)(conv_input + input_idx));

            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (uint i = 0; i < FILTER_SIZE_X; i++) {

                uint4 weights_val = vload4(0, (__global uint *)(weights + filter_idx));
#if OFM_BLOCKS_PER_SIMD == 2
                uint4 weights_val3 = vload4(0, (__global uint *)(weights + filter_idx2));
#endif

                __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
                for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ow++) {
                    const uint ow_offset = ow + OUT_BLOCK_WIDTH;
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s0, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s0)));
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s1, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s1)));
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s2, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s2)));
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s3, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s3)));

#if OFM_BLOCKS_PER_SIMD == 2
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s0, ow * STRIDE_SIZE_X + i)),  as_char4(weights_val3.s0)));
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s1, ow * STRIDE_SIZE_X + i)),  as_char4(weights_val3.s1)));
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s2, ow * STRIDE_SIZE_X + i)),  as_char4(weights_val3.s2)));
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s3, ow * STRIDE_SIZE_X + i)),  as_char4(weights_val3.s3)));
#endif
                }
                filter_idx += 16 * 16;
#if OFM_BLOCKS_PER_SIMD == 2
                filter_idx2 += 16 * 16;
#endif
            }
        } 
    }

#if BIAS_TERM
    BIAS_TYPE bias[OFM_BLOCKS_PER_SIMD] = { biases[out_f]
#if OFM_BLOCKS_PER_SIMD == 2
        , biases[out_f + 16]
#endif
    };
#endif
    __attribute__((opencl_unroll_hint(OFM_BLOCKS_PER_SIMD)))
    for (uint j = 0; j < OFM_BLOCKS_PER_SIMD; j++) {
        const uint dst_index = GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, out_b, out_f + j * 16, out_y, out_x);
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_PRELOAD;
#endif
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for (uint i = 0; i < OUT_BLOCK_WIDTH; i++) {

#if OUTPUT_SIZE_X % OUT_BLOCK_WIDTH != 0
            if (out_x + OUT_BLOCK_WIDTH > OUTPUT_SIZE_X && i >= OUTPUT_SIZE_X % OUT_BLOCK_WIDTH)
                break;
#endif
            ACTIVATION_TYPE dequantized = (ACTIVATION_TYPE)0;
#if BIAS_TERM
            dequantized = (ACTIVATION_TYPE)dotProd[OUT_BLOCK_WIDTH * j + i] + bias[j];
#else
            dequantized = (ACTIVATION_TYPE)dotProd[OUT_BLOCK_WIDTH * j + i];
#endif
            OUTPUT_TYPE result;
#if HAS_FUSED_OPS
    #if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC;
    #else
            FUSED_OPS;
    #endif
            result = FUSED_OPS_RESULT;
#else
            result = TO_OUTPUT_TYPE(dequantized);
#endif

#if OUTPUT_FEATURE_NUM % 16 != 0
            if (out_fg + j * 16 + 16 > OUTPUT_FEATURE_NUM && get_sub_group_local_id() >= OUTPUT_FEATURE_NUM % 16)
                result = (OUTPUT_TYPE)0;
#endif
            output[dst_index + i * 16] = result;
        }
    }
}

#undef AS_INPUT0_TYPE_4
#undef AS_TYPE_N
#undef AS_TYPE_N_
#undef MAKE_VECTOR_TYPE
#undef TO_ACTIVATION_TYPE
#undef ACTIVATION_TYPE
#undef TO_ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE

#undef CEIL_DIV
#undef ALIGN
