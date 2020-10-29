// Copyright (c) 2019-2020 Intel Corporation
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

#define SIMD_SIZE         16
#define BYTES_PER_READ    (sizeof(int))
#define BYTES_PER_READ8   (8 * BYTES_PER_READ)

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL(fully_connected_gpu_IMAD)(
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
    // This kernel works with linearized data w/o strides and padding
    // so only one dimension 'F' is required
    const uint f = get_global_id(0);
    const uint b = get_global_id(1);

    if (f >= OUTPUT_FEATURE_NUM) {
        return;
    }

    int dotProd = 0;

    uint idx_w = ((f / SIMD_SIZE) * SIMD_SIZE) * INPUT0_FEATURE_NUM;
    const __global INPUT0_TYPE* current_input = &input[GET_DATA_INDEX(INPUT0, b, 0, 0, 0)];

    for (uint idx_i = 0; idx_i < INPUT0_FEATURE_NUM; idx_i += BYTES_PER_READ8) {
        int input_data = as_int(intel_sub_group_block_read((const __global uint*)(current_input + idx_i)));
        int8 activations;  //activations of all lanes
        activations.s0 = sub_group_broadcast(input_data, 0);
        activations.s1 = sub_group_broadcast(input_data, 1);
        activations.s2 = sub_group_broadcast(input_data, 2);
        activations.s3 = sub_group_broadcast(input_data, 3);
        activations.s4 = sub_group_broadcast(input_data, 4);
        activations.s5 = sub_group_broadcast(input_data, 5);
        activations.s6 = sub_group_broadcast(input_data, 6);
        activations.s7 = sub_group_broadcast(input_data, 7);

        int8 weights_data = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w)));
        idx_w += SIMD_SIZE * BYTES_PER_READ8;

        for (int i = 0; i < 8; i++) {
            dotProd = IMAD(dotProd, AS_INPUT0_TYPE_4(activations[i]), as_char4(weights_data[i]));
        }
    }

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
        const uint bias_index = GET_DATA_INDEX(BIAS, b, f, 0, 0);
    #elif BIAS_PER_OFM
        const uint bias_index = f;
    #endif
    float dequantized = (float)dotProd + biases[bias_index];
#elif
    float dequantized = (float)dotProd;
#endif

    const uint out_index = GET_DATA_INDEX(OUTPUT, b, f, 0, 0);
#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;

    output[out_index] = res;
#else
    output[out_index] = TO_OUTPUT_TYPE(dequantized);
#endif
}

#undef SIMD_SIZE
#undef BYTES_PER_READ
#undef BYTES_PER_READ8
