// Copyright (c) 2016-2017 Intel Corporation
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
#if OUTPUT_BATCH_NUM == 1
    const uint f = (uint)get_global_id(0);
    const uint b = 0;
#else
    const uint f = (uint)get_global_id(0);
    const uint b = (uint)get_global_id(1);
#endif

    int dotProd = 0;

    const uint filter_offset = FILTER_GET_OFFSET(f);
#if INPUT0_DIMS == 5
    const uint input_offset = INPUT0_GET_INDEX(b, 0, 0, 0, 0);
#else
    const uint input_offset = INPUT0_GET_INDEX(b, 0, 0, 0);
#endif

#if SPATIAL_MAJOR
    for (uint k = 0; k < (FILTER_IFM_NUM + 31) / 32; ++k) {
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
        for (uint k = 0; k < (FILTER_IFM_NUM + 31) / 32; ++k) {
#endif
#if !SPLIT_SPATIAL
            uint input_idx = input_offset + spatial * MMAD_INPUT_SPATIAL_PITCH + k * MMAD_INPUT_FBLOCK_PITCH;
#else
            uint input_idx = input_offset + k * MMAD_INPUT_FBLOCK_PITCH + zi * MMAD_INPUT_Z_PITCH + yi * MMAD_INPUT_Y_PITCH + xi * MMAD_INPUT_X_PITCH;
#endif
            uint filter_idx = filter_offset + spatial * MMAD_FILTER_SPATIAL_PITCH + k * MMAD_FILTER_FBLOCK_PITCH;

            uint input_data_u = intel_sub_group_block_read((const __global uint*)(input + input_idx));
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
    }

    if (OUTPUT_FEATURE_NUM % SUB_GROUP_SIZE != 0 && f >= OUTPUT_FEATURE_NUM)
        return;

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, 0, 0);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif

    float dequantized = (float)dotProd + biases[bias_index];
#else  // BIAS_TERM
    float dequantized = (float)dotProd;
#endif

    const uint out_idx = OUTPUT_GET_INDEX(b, f, 0, 0);

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FINAL_NAME;

    output[out_idx] = res;
#else
    output[out_idx] = TO_OUTPUT_TYPE(dequantized);
#endif
}

#undef INPUT_PACKED_TYPE_8
#undef FILTER_PACKED_TYPE_8
#undef AS_TYPE
