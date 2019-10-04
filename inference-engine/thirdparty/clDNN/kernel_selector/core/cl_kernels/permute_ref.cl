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
#if   INPUT0_SIMPLE && INPUT0_DIMS < 5
    return GET_DATA_INDEX(INPUT0, b, f, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_DATA_INDEX_5D(INPUT0, b, f, z, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 6
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#elif defined INPUT0_LAYOUT_BS_F_BSV8__AF8  || \
      defined INPUT0_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(INPUT0, b, f, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_BF8_XY16
    return GET_DATA_BF8_XY16_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BFYX_F16
    return GET_DATA_BFYX_F16_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BYXF_AF32
    return GET_DATA_BYXF_AF32_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BYX8_F4
    return GET_DATA_BYX8_F4_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_FS_BS_YX_BSV4_FSV32
    return GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_B_FS_YX_FSV4
    return GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_FS_B_YX_FSV32
    return GET_DATA_FS_B_YX_FSV32_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BFZYX_F16
    return GET_DATA_BFZYX_F16_INDEX(INPUT0, b, f, z, y, x);
#else
#error permute_ref.cl: input format - not supported
#endif
}

inline uint FUNC(get_input3d_index)(uint b, uint f, uint z, uint y, uint x)
{
    return GET_DATA_INDEX_5D(INPUT0, b, f, z, y, x);
}

///////////////////////// Output Index /////////////////////////
inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if   OUTPUT_SIMPLE && OUTPUT_DIMS < 5
    return GET_DATA_INDEX(OUTPUT, b, f, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_DATA_INDEX_5D(OUTPUT, b, f, z, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 6
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, y, x);
#elif defined OUTPUT_LAYOUT_BS_F_BSV8__AF8  || \
      defined OUTPUT_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(OUTPUT, b, f, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_BF8_XY16
    return GET_DATA_BF8_XY16_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BFYX_F16
    return GET_DATA_BFYX_F16_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BYXF_AF32
    return GET_DATA_BYXF_AF32_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BYX8_F4
    return GET_DATA_BYX8_F4_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_FS_BS_YX_BSV4_FSV32
    return GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_B_FS_YX_FSV4
    return GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_FS_B_YX_FSV32
    return GET_DATA_FS_B_YX_FSV32_INDEX(OUTPUT, b, f, y, x);
#elif defined INPUT0_LAYOUT_BFZYX_F16
    return GET_DATA_BFZYX_F16_INDEX(OUTPUT, b, f, z, y, x);
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
    input_indices[1] = get_global_id(2) % INPUT0_FEATURE_NUM;
    input_indices[0] = get_global_id(2) / INPUT0_FEATURE_NUM;

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
