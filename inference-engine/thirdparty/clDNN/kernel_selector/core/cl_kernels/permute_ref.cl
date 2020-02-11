// Copyright (c) 2017-2019 Intel Corporation
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

#include "include/include_all.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX(b, f, w, z, y, x);
#else
#error permute_ref.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////
inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 6
    return OUTPUT_GET_INDEX(b, f, w, z, y, x);
#else
#error permute_ref.cl: output format - not supported
#endif
}

KERNEL (permute_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    uint8 input_indices, output_indices;

    //gws(y * z * w, x, b*f)
    //input_indices[b, f, x, y, z, w]
    const uint gid_0 = get_global_id(0);
    input_indices[5] = gid_0 / (INPUT0_SIZE_Y * INPUT0_SIZE_Z) % INPUT0_SIZE_W;
    input_indices[4] = gid_0 / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    input_indices[3] = gid_0 % INPUT0_SIZE_Y;
    input_indices[2] = get_global_id(1);
    input_indices[1] = (uint)get_global_id(2) % INPUT0_FEATURE_NUM;
    input_indices[0] = (uint)get_global_id(2) / INPUT0_FEATURE_NUM;

    //PERMUTE_ORDER[b, f, x, y, z, w]
    //output_indices[b, f, x, y, z, w]
    __attribute__((opencl_unroll_hint(PERMUTE_ORDER_SIZE)))
    for (uint idx = 0; idx < PERMUTE_ORDER_SIZE; ++idx)
    {
        output_indices[idx] = input_indices[PERMUTE_ORDER[idx]];
    }

    uint input_offset;
    uint output_offset;

    input_offset =  FUNC_CALL(get_input_index)(input_indices[0], input_indices[1], input_indices[5], input_indices[4], input_indices[3], input_indices[2]);
    output_offset = FUNC_CALL(get_output_index)(output_indices[0], output_indices[1], output_indices[5], output_indices[4], output_indices[3], output_indices[2]);
    output[output_offset] = ACTIVATION(input[input_offset], ACTIVATION_PARAMS);
}
