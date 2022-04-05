// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/reshape_dims.cl"
#include "include/batch_headers/fetch_data.cl"

#include "include/batch_headers/data_types.cl"
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
#elif defined OUTPUT_LAYOUT_B_FS_YX_FSV16
    return GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x);
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

    float Ycomponent = mad(Y.x, 296.82f, -18.624f);
    float Ucomponent = mad(UV.x, 255.0f, -128.f);
    float Vcomponent = mad(UV.y, 255.0f, -128.f);

    float B = clamp(mad(Vcomponent, 1.596f, Ycomponent), 0.f, 255.f);
    float R = clamp(mad(Ucomponent, 2.018f, Ycomponent), 0.f, 255.f);
    float G = clamp(mad(Vcomponent, -0.813f, mad(Ucomponent, -0.391f, Ycomponent)), 0.f, 255.f);

#if defined MEAN_SUBTRACT_INSIDE_PARAMS
    R -= VALUE_TO_SUBTRACT[0];
    G -= VALUE_TO_SUBTRACT[1];
    B -= VALUE_TO_SUBTRACT[2];
#elif defined MEAN_SUBTRACT_IN_BUFFER
    uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, 0, w, z, y, x);
    R -= mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], msv[5], msv[6])];

    msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, 1, w, z, y, x);
    G -= mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], msv[5], msv[6])];

    msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, 2, w, z, y, x);
    B -= mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], msv[5], msv[6])];
#endif

    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, w, z, y, x);
    uint output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(R), NL_M, NL_N);

    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(G), NL_M, NL_N);

    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(B), NL_M, NL_N);


    }
