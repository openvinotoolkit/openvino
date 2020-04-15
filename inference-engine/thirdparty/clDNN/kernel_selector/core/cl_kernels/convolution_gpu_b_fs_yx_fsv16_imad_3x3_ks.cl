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

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(convolution_gpu_b_fs_yx_fsv16_3x3_ks)(
    const __global INPUT0_TYPE   *conv_input,
    __global OUTPUT_TYPE         *output,
    const __global FILTER_TYPE    *weights,
#if BIAS_TERM
    const __global BIAS_TYPE     *biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
#if OUT_BLOCK_WIDTH == 7 && STRIDE_SIZE_X == 1
    const int tmp[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0};
#elif OUT_BLOCK_WIDTH == 7 && STRIDE_SIZE_X == 2
    const int tmp[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0};
#elif OUT_BLOCK_WIDTH == 8 && STRIDE_SIZE_X == 1
    const int tmp[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0};
#else  // OUT_BLOCK_WIDTH == 8 && STRIDE_SIZE_X == 2
    const int tmp[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
#endif

    const uint out_x = (uint)get_global_id(0) * OUT_BLOCK_WIDTH;
    const uint out_y = get_global_id(1);
    const uint out_f = (uint)(get_group_id(2) * 32 + get_sub_group_local_id());
    const uint subgroup_id = get_sub_group_id();
    const uint subgroup_local_id = get_sub_group_local_id();
    const uint feature_offset = subgroup_id * INPUT0_FEATURE_NUM / 4;
    const uint out_b = (uint)(get_group_id(2) * 32) / OUTPUT_FEATURE_NUM;

    ACCUMULATOR_TYPE dotProd[OUT_BLOCK_WIDTH * 2] = { 0 };
    const int input_x = out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    uint filter_idx = GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, out_f, feature_offset, 0, 0);
    uint diff_filter_idx = 16*3*3*FILTER_IFM_NUM;

    __attribute__((opencl_unroll_hint(1)))
    for(uint k = 0; k < CEIL_DIV(INPUT0_FEATURE_NUM, 16)/4; k++ ) {
        __attribute__((opencl_unroll_hint(1)))
        for(uint j = 0; j < FILTER_SIZE_Y; j++) {
            uint input_idx = GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, out_b, feature_offset + k * 16, input_y + j, input_x + tmp[subgroup_local_id]);
            uint4 input_val0 = vload4(0, (__global uint *)(conv_input + input_idx));
            
             __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for(uint i = 0; i < FILTER_SIZE_X; i++) {

                uint4 weights_val = vload4(0, (__global uint*)(weights + filter_idx));                
                uint4 weights_val3 = vload4(0, (__global uint *)(weights + filter_idx + diff_filter_idx));

                __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
                for(uint ow = 0; ow < OUT_BLOCK_WIDTH; ow++) {
                    const uint ow_offset = ow + OUT_BLOCK_WIDTH;
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s0, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s0)));
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s1, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s1)));
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s2, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s2)));
                    dotProd[ow] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s3, ow * STRIDE_SIZE_X + i)), as_char4(weights_val.s3)));

                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s0, ow * STRIDE_SIZE_X + i)), as_char4(weights_val3.s0)));
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s1, ow * STRIDE_SIZE_X + i)), as_char4(weights_val3.s1)));
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s2, ow * STRIDE_SIZE_X + i)), as_char4(weights_val3.s2)));
                    dotProd[ow_offset] = TO_ACCUMULATOR_TYPE(IMAD(dotProd[ow_offset], AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val0.s3, ow * STRIDE_SIZE_X + i)), as_char4(weights_val3.s3)));
                }
                filter_idx += 16 * 16;
            }
        }
    }

    //k slicing summing up with SLM
    __local ACCUMULATOR_TYPE partial_acc[16 * OUT_BLOCK_WIDTH * 6];
    if(subgroup_id == 0)
    {
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for(uint i = 0; i < OUT_BLOCK_WIDTH; i++)
        {
            partial_acc[16 * OUT_BLOCK_WIDTH + i * 16 + subgroup_local_id] = dotProd[i + OUT_BLOCK_WIDTH];
        }       
    }
    else if(subgroup_id == 1)
    {
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for(uint i = 0; i < OUT_BLOCK_WIDTH; i++)
        {
            partial_acc[i * 16 + subgroup_local_id] = dotProd[i];
            dotProd[i] = dotProd[i + OUT_BLOCK_WIDTH];
        }
    }
    else if (subgroup_id == 2)
    {
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for (uint i = 0; i < OUT_BLOCK_WIDTH; i++)
        {
            partial_acc[2 * 16 * OUT_BLOCK_WIDTH + i * 16 + subgroup_local_id] = dotProd[i];
            partial_acc[3 * 16 * OUT_BLOCK_WIDTH + i * 16 + subgroup_local_id] = dotProd[i + OUT_BLOCK_WIDTH];

        }
    }
    else if (subgroup_id == 3)
    {
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for (uint i = 0; i < OUT_BLOCK_WIDTH; i++)
        {
            partial_acc[4 * 16 * OUT_BLOCK_WIDTH + i * 16 + subgroup_local_id] = dotProd[i];
            partial_acc[5 * 16 * OUT_BLOCK_WIDTH + i * 16 + subgroup_local_id] = dotProd[i + OUT_BLOCK_WIDTH];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (subgroup_id < 2) {
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for (uint i = 0; i < OUT_BLOCK_WIDTH; i++)
        {
            dotProd[i] += partial_acc[(i + subgroup_id * OUT_BLOCK_WIDTH) * 16 + subgroup_local_id];
            dotProd[i] += partial_acc[(i + (subgroup_id + 2) * OUT_BLOCK_WIDTH) * 16 + subgroup_local_id];
            dotProd[i] += partial_acc[(i + (subgroup_id + 4) * OUT_BLOCK_WIDTH) * 16 + subgroup_local_id];
        }
#if BIAS_TERM
    BIAS_TYPE bias = biases[out_f + get_sub_group_id() * 16];
#endif

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif
    const uint dst_index = GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, out_b, out_f + subgroup_id * 16, out_y, out_x);
     __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
    for (uint i = 0; i < OUT_BLOCK_WIDTH; i++)
    {
        ACTIVATION_TYPE dequantized = (ACTIVATION_TYPE)0;
#if BIAS_TERM
        dequantized = (ACTIVATION_TYPE)dotProd[i] + bias;
#else
        dequantized = (ACTIVATION_TYPE)dotProd[i];
#endif
#if HAS_FUSED_OPS
    #if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC;
    #else
        FUSED_OPS;
    #endif
        output[dst_index + i * 16] = FUSED_OPS_RESULT;
#else
        output[dst_index + i * 16] = TO_OUTPUT_TYPE(dequantized);
#endif
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
