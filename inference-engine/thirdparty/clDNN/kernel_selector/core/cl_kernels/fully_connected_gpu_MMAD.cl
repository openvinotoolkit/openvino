// Copyright (c) 2016-2020 Intel Corporation
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
#include "include/mmad.cl"

#define INPUT_PACKED_TYPE_8  CAT(INPUT_PACKED_TYPE, 8)
#define FILTER_PACKED_TYPE_8 CAT(FILTER_PACKED_TYPE, 8)

#define AS_TYPE(type, val) CAT(as_, type)(val)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(fully_connected_gpu_MMAD)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint lid0 = (uint)get_local_id(0);
    const uint feature_per_wg = (uint)get_local_size(0) / SLM_DIV_FACTOR;
    const uint feature = (uint)get_group_id(0) * feature_per_wg + (uint)get_global_id(0) % feature_per_wg;
    const uint feature_block = lid0 / feature_per_wg;
    const uint batch = (uint)get_global_id(1);

    int dotProd = 0;

    const uint filter_offset = FILTER_GET_OFFSET(feature);
#if INPUT0_DIMS == 5
    const uint input_offset = INPUT0_GET_INDEX(batch, 0, 0, 0, 0);
#else
    const uint input_offset = INPUT0_GET_INDEX(batch, 0, 0, 0);
#endif

#if SLM_DIV_FACTOR > 1
    __local int partial_summ[WORK_GROUP_SIZE];
#endif

#if SPATIAL_MAJOR

#if FULL_UNROLL_FACTOR < 2
    for (uint k = feature_block * FULL_UNROLL_FACTOR; k < (feature_block + 1) * FULL_UNROLL_FACTOR; ++k)
#elif UNROLL_FACTOR == FULL_UNROLL_FACTOR
    uint k = feature_block * FULL_UNROLL_FACTOR;
#else
    for (uint k = feature_block * FULL_UNROLL_FACTOR; k + UNROLL_FACTOR <= (feature_block + 1) * FULL_UNROLL_FACTOR; k += UNROLL_FACTOR)
#endif
    {
#   if !SPLIT_SPATIAL
        for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#   else
        for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
        for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
        for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
            const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif
#else  // SPATIAL_MAJOR
#   if !SPLIT_SPATIAL
    for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#   else
    for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
    for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
    for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
        const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#   endif

#if FULL_UNROLL_FACTOR < 2
        for (uint k = feature_block * FULL_UNROLL_FACTOR; k < (feature_block + 1) * FULL_UNROLL_FACTOR; ++k)
#elif UNROLL_FACTOR == FULL_UNROLL_FACTOR
        uint k = feature_block * FULL_UNROLL_FACTOR;
#else
        for (uint k = feature_block * FULL_UNROLL_FACTOR; k + UNROLL_FACTOR <= (feature_block + 1) * FULL_UNROLL_FACTOR; k += UNROLL_FACTOR)
#endif
        {
#endif
#if !SPLIT_SPATIAL
            uint input_idx = input_offset + spatial * MMAD_INPUT_SPATIAL_PITCH + k * MMAD_INPUT_FBLOCK_PITCH;
#else
            uint input_idx = input_offset + k * MMAD_INPUT_FBLOCK_PITCH + zi * MMAD_INPUT_Z_PITCH + yi * MMAD_INPUT_Y_PITCH + xi * MMAD_INPUT_X_PITCH;
#endif
            uint filter_idx = filter_offset + spatial * MMAD_FILTER_SPATIAL_PITCH + k * MMAD_FILTER_FBLOCK_PITCH;

#if UNROLL_FACTOR < 2
            uint input_data_u = intel_sub_group_block_read((const __global uint*)(input + input_idx));
            INPUT_PACKED_TYPE input_data = AS_TYPE(INPUT_PACKED_TYPE, input_data_u);

            INPUT_PACKED_TYPE_8 activations;

            activations.s0 = sub_group_broadcast(input_data, 0);
            activations.s1 = sub_group_broadcast(input_data, 1);
            activations.s2 = sub_group_broadcast(input_data, 2);
            activations.s3 = sub_group_broadcast(input_data, 3);
            activations.s4 = sub_group_broadcast(input_data, 4);
            activations.s5 = sub_group_broadcast(input_data, 5);
            activations.s6 = sub_group_broadcast(input_data, 6);
            activations.s7 = sub_group_broadcast(input_data, 7);

            uint8 weights_data_u = intel_sub_group_block_read8((const __global uint*)(weights + filter_idx));
            FILTER_PACKED_TYPE_8 weights_data = AS_TYPE(FILTER_PACKED_TYPE_8, weights_data_u);

            dotProd = MMAD_8(activations, weights_data, dotProd);
#else
            INPUT_PACKED_TYPE input_data[UNROLL_FACTOR];
            FILTER_PACKED_TYPE_8 weights_data[UNROLL_FACTOR];

            __attribute__((opencl_unroll_hint))
            for (uint kb = 0; kb < UNROLL_FACTOR; kb++) {
                input_data[kb] = AS_TYPE(INPUT_PACKED_TYPE, intel_sub_group_block_read((const __global uint*)(input +
                                         input_idx  + kb * MMAD_INPUT_FBLOCK_PITCH)));

                uint8 weights_data_u0 = intel_sub_group_block_read8((const __global uint*)(weights + filter_idx + kb * MMAD_FILTER_FBLOCK_PITCH));
                weights_data[kb] = AS_TYPE(FILTER_PACKED_TYPE_8, weights_data_u0);
            }

            __attribute__((opencl_unroll_hint))
            for (uint kb = 0; kb < UNROLL_FACTOR; kb++) {
                INPUT_PACKED_TYPE_8 in;

                in.s0 = sub_group_broadcast(input_data[kb], 0);
                in.s1 = sub_group_broadcast(input_data[kb], 1);
                in.s2 = sub_group_broadcast(input_data[kb], 2);
                in.s3 = sub_group_broadcast(input_data[kb], 3);
                in.s4 = sub_group_broadcast(input_data[kb], 4);
                in.s5 = sub_group_broadcast(input_data[kb], 5);
                in.s6 = sub_group_broadcast(input_data[kb], 6);
                in.s7 = sub_group_broadcast(input_data[kb], 7);

                dotProd = MMAD_8(in, weights_data[kb], dotProd);
            }
#endif // UNROLL_FACTOR < 2
        }
    }

#if SLM_DIV_FACTOR > 1
    partial_summ[lid0] = dotProd;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_block == 0) {
        __attribute__((opencl_unroll_hint))
        for (uint i = 1; i < SLM_DIV_FACTOR; i++)
            dotProd += partial_summ[lid0 % feature_per_wg + i * feature_per_wg];
#endif // SLM_DIV_FACTOR > 1

#if HAS_FEATURE_LEFTOVERS
        const uint sglid = get_sub_group_local_id();
#if SPATIAL_MAJOR
#if !SPLIT_SPATIAL
        for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#else
        for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
        for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
        for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
            const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif  // !SPLIT_SPATIAL

#else  // SPATIAL_MAJOR
#if !SPLIT_SPATIAL
    for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#else  // !SPLIT_SPATIAL
    for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
        for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
        for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
            const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif  // !SPLIT_SPATIAL
#endif  // SPATIAL_MAJOR

#if !SPLIT_SPATIAL
            uint input_idx = input_offset + spatial * MMAD_INPUT_SPATIAL_PITCH + FEATURE_BLOCKS_COUNT * INPUT0_FEATURE_PITCH;
#else  // !SPLIT_SPATIAL
            uint input_idx = input_offset + FEATURE_BLOCKS_COUNT * INPUT0_FEATURE_PITCH + zi * MMAD_INPUT_Z_PITCH + yi * MMAD_INPUT_Y_PITCH + xi * MMAD_INPUT_X_PITCH;
#endif  // !SPLIT_SPATIAL
            uint filter_idx = filter_offset + spatial * MMAD_FILTER_SPATIAL_PITCH + FEATURE_BLOCKS_COUNT * MMAD_FILTER_FBLOCK_PITCH;

            MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) input_data_u = (0, 0, 0, 0);
            for (uint i = 0; i < 4; i++) {
                if (FEATURE_BLOCKS_COUNT * 32 + sglid * 4 + i < INPUT0_FEATURE_NUM) {
                    input_data_u[i] = input[input_idx + (sglid * 4 + i) * INPUT0_FEATURE_PITCH];
                }
            }
            INPUT_PACKED_TYPE input_data = AS_TYPE(INPUT_PACKED_TYPE, input_data_u);

            INPUT_PACKED_TYPE_8 activations;  //activations of all lanes
            activations.s0 = sub_group_broadcast(input_data, 0);
            activations.s1 = sub_group_broadcast(input_data, 1);
            activations.s2 = sub_group_broadcast(input_data, 2);
            activations.s3 = sub_group_broadcast(input_data, 3);
            activations.s4 = sub_group_broadcast(input_data, 4);
            activations.s5 = sub_group_broadcast(input_data, 5);
            activations.s6 = sub_group_broadcast(input_data, 6);
            activations.s7 = sub_group_broadcast(input_data, 7);

            uint8 weights_data_u = intel_sub_group_block_read8((const __global uint*)(weights + filter_idx));
            FILTER_PACKED_TYPE_8 weights_data = AS_TYPE(FILTER_PACKED_TYPE_8, weights_data_u);

            dotProd = MMAD_8(activations, weights_data, dotProd);
        }
#endif  // HAS_FEATURE_LEFTOVERS

    if (OUTPUT_FEATURE_NUM % SUB_GROUP_SIZE != 0 && feature >= OUTPUT_FEATURE_NUM)
        return;

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, batch, feature, 0, 0);
#elif BIAS_PER_OFM
    const uint bias_index = feature;
#endif

    float dequantized = (float)dotProd + biases[bias_index];
#else  // BIAS_TERM
    float dequantized = (float)dotProd;
#endif

    const uint out_idx = OUTPUT_GET_INDEX(batch, feature, 0, 0);

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;

    output[out_idx] = res;
#else
    output[out_idx] = TO_OUTPUT_TYPE(dequantized);
#endif

#if SLM_DIV_FACTOR > 1
    }
#endif
}

#undef INPUT_PACKED_TYPE_8
#undef FILTER_PACKED_TYPE_8
#undef AS_TYPE
