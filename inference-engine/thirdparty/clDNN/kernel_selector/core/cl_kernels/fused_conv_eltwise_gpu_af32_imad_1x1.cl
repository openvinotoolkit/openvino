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

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/imad.cl"

#if defined(ACCUMULATOR_TYPE)
#undef ACCUMULATOR_TYPE
#endif

#if QUANTIZATION_TERM
#    define ACCUMULATOR_TYPE int
#    define ACTIVATION_TYPE float
#    define TO_ACTIVATION_TYPE(x) convert_float(x)
#else
#    define ACCUMULATOR_TYPE INPUT0_TYPE
#    define ACTIVATION_TYPE INPUT0_TYPE
#    define TO_ACTIVATION_TYPE(x) TO_INPUT0_TYPE(x)
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
    const __global PACKED_TYPE* input,
    __global OUTPUT_TYPE* restrict output,
    const __global uint* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
    const uint x = (uint)get_global_id(0) * TILE_LENGTH % OUTPUT_SIZE_X;
    const uint y = (uint)get_global_id(0) * TILE_LENGTH / OUTPUT_SIZE_X;
    const uint f = (((uint)get_global_id(1) * TILE_DEPTH) % FILTER_OFM_NUM_ALIGNED) / (TILE_DEPTH * SUB_GROUP_SIZE) * (TILE_DEPTH * SUB_GROUP_SIZE);
    const uint b = ((uint)get_global_id(1) * TILE_DEPTH) / FILTER_OFM_NUM_ALIGNED;
    const uint lid = get_sub_group_local_id();

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    PACKED_TYPE input_slice[TILE_LENGTH];
    int8 weights_slice;
    ACCUMULATOR_TYPE accu[TILE_LENGTH][TILE_DEPTH] = {0};

    uint filter_idx = f * FILTER_IFM_SLICES_NUM * 32 / PACK;
    uint in_addr = (INPUT0_GET_INDEX(b, 0, input_y, input_x)) / PACK;

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < FILTER_IFM_SLICES_NUM; ++k)
    {
        // Read 32 input features for each pixel in the tile. 4 features in each int, 8 ints across SIMD
        __attribute__((opencl_unroll_hint(TILE_LENGTH)))
        for (uint i = 0; i < TILE_LENGTH; ++i)
        {
            uint tmp_addr = in_addr + i * INPUT0_X_PITCH * STRIDE_SIZE_X / PACK;
            input_slice[i] = AS_PACKED_TYPE(intel_sub_group_block_read((const __global uint*)input + tmp_addr));
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
                PACKED_TYPE A_scalar;
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
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) result[TILE_LENGTH];
#elif TILE_DEPTH == 4
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) result[TILE_LENGTH];
#endif

    uint dst_index = (OUTPUT_GET_INDEX(b, f, y, x)) / PACK;

    __attribute__((opencl_unroll_hint(TILE_LENGTH)))
    for (uint i = 0; i < TILE_LENGTH; ++i)
    {

        __attribute__((opencl_unroll_hint(TILE_DEPTH)))
        for (uint j = 0; j < TILE_DEPTH; ++j)
        {
            const uint f2 = f + lid * 4 + (j % 4) + (j / 4 * 32);
            ACCUMULATOR_TYPE dotProd = accu[i][j];
#if BIAS_TERM
    #if BIAS_PER_OUTPUT
            const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
            const uint bias_index = f2;
    #endif
            ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(dotProd)  + TO_ACTIVATION_TYPE(biases[bias_index]);
#else
            ACTIVATION_TYPE res = TO_ACTIVATION_TYPE(dotProd);
#endif //BIAS_TERM

        #if HAS_FUSED_OPS
            FUSED_OPS;
            result[i][j] = FUSED_OPS_RESULT;
        #else
            result[i][j] = TO_OUTPUT_TYPE(res);
        #endif
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
#undef ACCUMULATOR_TYPE
#undef ACTIVATION_TYPE
#undef TO_ACTIVATION_TYPE
