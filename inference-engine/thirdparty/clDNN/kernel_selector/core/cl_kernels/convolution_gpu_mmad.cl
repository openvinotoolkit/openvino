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
#include "include/mmad.cl"

#define FILTER_IFM_MMAD_NUM ((FILTER_IFM_NUM + 31) / 32)
#define FILTER_OFM_MMAD_NUM ((FILTER_OFM_NUM + 7) / 8)
#define FILTER_IFM_ALIGNED (FILTER_IFM_MMAD_NUM * 32)
#define FILTER_OFM_ALIGNED (FILTER_OFM_MMAD_NUM * 8)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(convolution_MMAD)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = (uint)get_global_id(2) % FILTER_OFM_ALIGNED;
    const uint b = (uint)get_global_id(2) / FILTER_OFM_ALIGNED;
#endif

    int dotProd = 0;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint filter_offset = ((uint)get_group_id(2) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint k = 0; k < FILTER_IFM_MMAD_NUM; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

            if(!zero_y)
            {
                for (uint i = 0; i < FILTER_SIZE_X ; ++i)
                {
                    const int input_offset_x = input_x + i * DILATION_SIZE_X;
                    const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                    if(!zero_x)
                    {
                        uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH + k*32;
                        uint filter_idx = filter_offset + k*FILTER_Y_PITCH * FILTER_SIZE_Y + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;

                        PACKED_TYPE input_data = AS_PACKED_TYPE(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
                        MAKE_VECTOR_TYPE(PACKED_TYPE, 8) activations;  //activations of all lanes
                        activations.s0 = sub_group_broadcast(input_data, 0);
                        activations.s1 = sub_group_broadcast(input_data, 1);
                        activations.s2 = sub_group_broadcast(input_data, 2);
                        activations.s3 = sub_group_broadcast(input_data, 3);
                        activations.s4 = sub_group_broadcast(input_data, 4);
                        activations.s5 = sub_group_broadcast(input_data, 5);
                        activations.s6 = sub_group_broadcast(input_data, 6);
                        activations.s7 = sub_group_broadcast(input_data, 7);

                        int8 weights_data = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + filter_idx)));

                        dotProd = MMAD_8(activations, weights_data, dotProd);
                    }
                }
            }
        }
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif
    float res = (float)dotProd + biases[bias_index];
#else
    float res = (float)dotProd;
#endif // BIAS_TERM

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE result = FUSED_OPS_RESULT;
#else
    OUTPUT_TYPE result = TO_OUTPUT_TYPE(res);
#endif

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = OUTPUT_GET_INDEX(b, f, y, x) + out_split_offset;
    output[dst_index] = result;
}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED
