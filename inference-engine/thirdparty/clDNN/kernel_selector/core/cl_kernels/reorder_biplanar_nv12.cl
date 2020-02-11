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


#include "include/reshape_dims.cl"
#include "include/fetch.cl"

#include "include/data_types.cl"
///////////////////////// Output Index /////////////////////////
inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if   OUTPUT_SIMPLE && OUTPUT_DIMS < 6
    return GET_DATA_INDEX(OUTPUT, b, f, y, x);
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
#else
#error reorder_biplanar_nv12.cl: output format - not supported
#endif
}
KERNEL(reorder_biplanar_nv12)(
    read_only image2d_t input,
    read_only image2d_t input_uv,
    __global OUTPUT_REORDER_TYPE* output
     #ifdef MEAN_SUBTRACT_IN_BUFFER
     , __global MEAN_SUBTRACT_TYPE* mean_subtract
     #endif
     )
     {
    const uint b = get_global_id(GWS_BATCH);
    const uint f = get_global_id(GWS_FEATURE);
#if   INPUT0_DIMS == 2
    const uint y = 0;
    const uint x = 0;
    const uint z = 0;
    const uint w = 0;
#elif INPUT0_DIMS == 4
    const uint y = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
    const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
    const uint z = 0;
    const uint w = 0;
#elif INPUT0_DIMS == 5
    uint data_idx = get_global_id(GWS_YX);
    uint tmp_data_idx = data_idx / INPUT0_SIZE_X;
    const uint x = data_idx - tmp_data_idx * INPUT0_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_SIZE_Z;
    const uint z = data_idx - tmp_data_idx * INPUT0_SIZE_Z;
    const uint w = 0;
#elif INPUT0_DIMS == 6
    const uint gid_yx = (uint)(get_global_id(GWS_YX));
    const uint x = gid_yx % INPUT0_SIZE_X;
    const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
#endif

    float4 Y = read_imagef(input, (int2)(x, y));
    float4 UV = read_imagef(input_uv, (int2)(x / 2, y / 2));

    float Ycomponent = Y.x * 255;
    float Ucomponent = UV.x * 255;
    float Vcomponent = UV.y * 255;
    uchar R = convert_uchar_sat(1.164f * (Ycomponent - 16) + 1.596f * (Vcomponent - 128));
    uchar G = convert_uchar_sat(1.164f * (Ycomponent - 16) - 0.813f * (Vcomponent - 128) - 0.391f * (Ucomponent - 128));
    uchar B = convert_uchar_sat(1.164f * (Ycomponent - 16) + 2.018f * (Ucomponent - 128));

    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, w, z, y, x);
    uint output_idx = FUNC_CALL(get_output_index)(ov[0], ov[1], ov[2], ov[3], ov[4], ov[5]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(R), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[0], ov[1], ov[2], ov[3], ov[4], ov[5]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(G), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[0], ov[1], ov[2], ov[3], ov[4], ov[5]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(B), NL_M, NL_N);


    }
