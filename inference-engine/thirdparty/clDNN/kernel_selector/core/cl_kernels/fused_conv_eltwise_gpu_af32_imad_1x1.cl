// Copyright (c) 2019 Intel Corporation
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

#if QUANTIZATION_TERM || CALIBRATION_TERM || defined(O_QF)
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE float
#    define ACTIVATION_TYPE_BASE float
     // TODO: It's unclear for now what should be the rounding scheme. Might
     // happen that we will be required to round to nearest-even. Should it
     // become a customization point, or will we support just one rounding
     // scheme?
#    if OUTPUT_IS_FP
         // [U]INT8 -> float convolution with quantization/calibration.
#        define AFTER_CALIBRATION_ROUND(x) (x)
#    else
         // TODO: Do we need the round of the conv result in the fused
         // primitive?
#        define AFTER_CALIBRATION_ROUND(x) round(x)
#    endif
#elif defined(ACTIVATION_ELTW_TYPED) && (INT8_UNIT_USED || UINT8_UNIT_USED)
    // TODO: Get rid of INT8_UNIT_USED - its meaning doesn't look to be defined
    // properly. Better to check INPUT0_TYPE is [U]INT8 somehow...
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE int
#    define ACTIVATION_TYPE_BASE int
#    define AFTER_CALIBRATION_ROUND(x) (x)
#else
#    define ACCUMULATOR_TYPE INPUT0_TYPE
#    define TO_ACCUMULATOR_TYPE(x) TO_INPUT0_TYPE(x)
#    define ACTIVATION_TYPE INPUT0_TYPE
#    define ACTIVATION_TYPE_BASE INPUT0
#    define AFTER_CALIBRATION_ROUND(x) (x)
#endif

#if defined(ACTIVATION_ELTW)
#    if OUTPUT_IS_FP
#        define AFTER_ELTW_CALIBRATION_ROUND(x) (x)
#    else
#        define AFTER_ELTW_CALIBRATION_ROUND(x) round(x)
#    endif
#endif

#if QUANTIZATION_TERM && (!defined(CALIBRATION_TERM) || CALIBRATION_TERM == 0 ) && !defined(O_QF)
    // To get proper type for the "before_activation" below.
#   define O_QF_LOCAL_DEFINE
#   define O_QF 1.0
#endif

#define FILTER_IFM_SLICES_NUM ((FILTER_IFM_NUM + 31) / 32)
#define FILTER_OFM_NUM_ALIGNED ((FILTER_OFM_NUM + SUB_GROUP_SIZE - 1) / SUB_GROUP_SIZE * SUB_GROUP_SIZE)

// we are packing 4 8bit activations per 32 bit
#define PACK 4

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(fused_conv_eltwise_gpu_af32_imad_1x1)(
    const __global uint* input,
    __global OUTPUT_TYPE* restrict output,
    const __global uint* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if QUANTIZATION_TERM
    const __global float* quantizations,
#endif
#if CALIBRATION_TERM
    const __global float* calibrations,
#endif
    uint split_idx
// one kernel for both convolution and fused_conv_eltwise
#ifdef ACTIVATION_ELTW //defined for fused conv+eltwise
    , const __global OUTPUT_TYPE *eltw_input
    #if ELTW_CALIBRATION_TERM
    , const __global float       *eltw_calibrations
    #endif
#endif
)
{
    const uint sg_channel = get_sub_group_local_id();

    const uint x = get_global_id(0) * TILE_LENGTH % OUTPUT_SIZE_X;
    const uint y = get_global_id(0) * TILE_LENGTH / OUTPUT_SIZE_X;
    const uint f = ((get_global_id(1) * TILE_DEPTH) % FILTER_OFM_NUM_ALIGNED) / (TILE_DEPTH * SUB_GROUP_SIZE) * (TILE_DEPTH * SUB_GROUP_SIZE);
    const uint b = (get_global_id(1) * TILE_DEPTH) / FILTER_OFM_NUM_ALIGNED;
    const uint lid = get_local_id(1);

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    int input_slice[TILE_LENGTH];
    int8 weights_slice;
    int accu[TILE_LENGTH][TILE_DEPTH] = {0};

    uint filter_idx = f * FILTER_IFM_SLICES_NUM * 32 / PACK;
    uint in_addr = (GET_DATA_INDEX(INPUT0, b, 0, input_y, input_x)) / PACK;

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < FILTER_IFM_SLICES_NUM; ++k)
    {
        // Read 32 input features for each pixel in the tile. 4 features in each int, 8 ints across SIMD
        __attribute__((opencl_unroll_hint(TILE_LENGTH)))
        for (uint i = 0; i < TILE_LENGTH; ++i)
        {
            uint tmp_addr = in_addr + i * INPUT0_X_PITCH * STRIDE_SIZE_X / PACK;
            input_slice[i] = as_int(intel_sub_group_block_read(input + tmp_addr));
        }

        // Loop through TILE_DEPTH output features
        __attribute__((opencl_unroll_hint(TILE_DEPTH)))
        for (uint of = 0; of < TILE_DEPTH; ++of)
        {
            // Read 32 weights. 8 ints, 4 weights in each int, each SIMD lane has own weghts
            weights_slice = as_int8(intel_sub_group_block_read8(weights + filter_idx));

            __attribute__((opencl_unroll_hint(TILE_LENGTH)))
            for (uint i = 0; i < TILE_LENGTH; ++i)
            {
                int A_scalar;
                A_scalar = sub_group_broadcast(input_slice[i], 0); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s0));
                A_scalar = sub_group_broadcast(input_slice[i], 1); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s1));
                A_scalar = sub_group_broadcast(input_slice[i], 2); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s2));
                A_scalar = sub_group_broadcast(input_slice[i], 3); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s3));
                A_scalar = sub_group_broadcast(input_slice[i], 4); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s4));
                A_scalar = sub_group_broadcast(input_slice[i], 5); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s5));
                A_scalar = sub_group_broadcast(input_slice[i], 6); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s6));
                A_scalar = sub_group_broadcast(input_slice[i], 7); accu[i][of] = IMAD(accu[i][of], AS_INPUT0_TYPE_4(A_scalar), as_char4(weights_slice.s7));
            }

            filter_idx += 32 * 8 / 4; // 32 features per channel * 8 SIMD channels / sizeof(int)
        }
        in_addr += 4 * 8 / 4; // 4 features per channel * 8 SIMD channels / sizeof(int) -> next 32 input features
    }

#if TILE_DEPTH == 8
    char8 result[TILE_LENGTH];
#elif TILE_DEPTH == 4
    char4 result[TILE_LENGTH];
#endif

    uint dst_index = (GET_DATA_INDEX(OUTPUT, b, f, y, x)) / PACK;

    __attribute__((opencl_unroll_hint(TILE_LENGTH)))
    for (uint i = 0; i < TILE_LENGTH; ++i)
    {
#ifdef ACTIVATION_ELTW //defined for fused conv+eltwise
    #if IN_OUT_OPT != 0
        #if TILE_DEPTH == 8
        result[i] = as_char8(intel_sub_group_block_read2((__global uint*)output + dst_index + i * OUTPUT_X_PITCH / PACK));
        #elif TILE_DEPTH == 4
        result[i] = as_char4(intel_sub_group_block_read((__global uint*)output + dst_index + i * OUTPUT_X_PITCH / PACK));
        #endif
    #endif
#endif
        __attribute__((opencl_unroll_hint(TILE_DEPTH)))
        for (uint j = 0; j < TILE_DEPTH; ++j)
        {
            const uint f2 = f + lid * 4 + (j % 4) + (j / 4 * 32);
            int dotProd = accu[i][j];
#if BIAS_TERM
    #if BIAS_PER_OUTPUT
            const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
            const uint bias_index = f2;
    #endif

    #if !DONT_DEQUANTIZE_BIAS
        dotProd += TO_ACCUMULATOR_TYPE(biases[bias_index]);
    #endif
#endif //BIAS_TERM
        
    #if QUANTIZATION_TERM
        ACTIVATION_TYPE before_activation = dotProd * quantizations[f2] * I_QF;
    #else
        ACTIVATION_TYPE before_activation = dotProd;
    #endif

#if DONT_DEQUANTIZE_BIAS
    #if !BIAS_TERM || !QUANTIZATION_TERM
        #error "DONT_DEQUANTIZE_BIAS is meaningless without BIAS_TERM and QUANTIZATION_TERM"
    #endif
            before_activation += biases[bias_index];
#endif

    ACTIVATION_TYPE after_activation = ACTIVATION_CONV_TYPED(ACTIVATION_TYPE, before_activation, ACTIVATION_PARAMS_CONV_TYPED);

#if CALIBRATION_TERM
    after_activation = after_activation * calibrations[f2];
#elif defined(O_QF)
    after_activation = after_activation * O_QF;
#endif
    after_activation = AFTER_CALIBRATION_ROUND(after_activation);

#ifdef ACTIVATION_ELTW //defined for fused conv+eltwise
    #if IN_OUT_OPT == 0
            const uint eltw_idx = GET_DATA_INDEX(INPUT1, b, f2, (y + (x+i) / OUTPUT_SIZE_X)*ELTW_STRIDE_Y, ((x+i) % OUTPUT_SIZE_X)*ELTW_STRIDE_X);
            const INPUT1_TYPE/*int*/ current_eltw_input = (int)eltw_input[eltw_idx];
    #else
            const OUTPUT_TYPE/*int*/ current_eltw_input = (int)result[i][j];
    #endif

    #if defined(NON_CONV_SCALE)
        float eltw_input_scaled = (float)current_eltw_input * NON_CONV_SCALE;
    #else
        ACTIVATION_TYPE eltw_input_scaled = CAT(CAT(TO_, ACTIVATION_TYPE_BASE), _TYPE_SAT)(current_eltw_input);
    #endif
        after_activation = eltw_input_scaled + after_activation;

        after_activation = ACTIVATION_ELTW_TYPED(ACTIVATION_TYPE, after_activation, ACTIVATION_PARAMS_ELTW_TYPED);

        after_activation = AFTER_ELTW_CALIBRATION_ROUND(after_activation
    #if ELTW_CALIBRATION_TERM
                                         *eltw_calibrations[f2]
    #endif
            );
#endif
            result[i][j] = TO_OUTPUT_TYPE_SAT(after_activation);
        }
    }

    __attribute__((opencl_unroll_hint(TILE_LENGTH)))
    for (uint i = 0; i < TILE_LENGTH; ++i)
    {
#if TILE_DEPTH == 8
        intel_sub_group_block_write2((__global uint*)output + dst_index + i * OUTPUT_X_PITCH / PACK, as_uint2(result[i]));
#elif TILE_DEPTH == 4
        intel_sub_group_block_write((__global uint*)output + dst_index + i * OUTPUT_X_PITCH / PACK, as_uint(result[i]));
#endif
    }
}
#undef FILTER_IFM_SLICES_NUM
#undef FILTER_OFM_NUM_ALIGNED
