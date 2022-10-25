// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/reshape_dims.cl"
#include "include/batch_headers/fetch_data.cl"

#include "include/batch_headers/data_types.cl"
#include "include/image_data.cl"

#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT_REORDER_TYPE, 4)
#define OUTPUT_TYPE4 MAKE_VECTOR_TYPE(OUTPUT_REORDER_TYPE, 4)

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 6
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#else
#error reorder_data.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 6
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, y, x);
#else
#error reorder_data.cl: output format - not supported
#endif
}

KERNEL (reorder_data)(
#if INPUT0_LAYOUT_NV12 || INPUT0_LAYOUT_IMAGE_2D_RGBA
    read_only image2d_t input,
#else
    const __global INPUT_REORDER_TYPE* input,
#endif
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
    write_only image2d_t output
#else
    __global OUTPUT_REORDER_TYPE* output
#endif
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

    tmp_data_idx  = data_idx / INPUT0_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / INPUT0_SIZE_Z;
    const uint z = data_idx - tmp_data_idx * INPUT0_SIZE_Z;
    const uint w = 0;
#elif INPUT0_DIMS == 6
    const uint gid_yx = (uint)(get_global_id(GWS_YX));
    const uint x = gid_yx % INPUT0_SIZE_X;
    const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
#endif

#if defined INPUT0_LAYOUT_NV12
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
    float4 colorVYU = read_imagef(input, sampler, (int2)(x, y));

    float Ycomponent = mad(colorVYU.s1, 296.82f, -18.624f);
    float Ucomponent = mad(colorVYU.s2, 255.0f, -128.f);
    float Vcomponent = mad(colorVYU.s0, 255.0f, -128.f);

    float B = clamp(mad(Vcomponent, 1.596f, Ycomponent), 0.f, 255.f);
    float R = clamp(mad(Ucomponent, 2.018f, Ycomponent), 0.f, 255.f);
    float G = clamp(mad(Vcomponent, -0.813f, mad(Ucomponent, -0.391f, Ycomponent)), 0.f, 255.f);
#elif defined INPUT0_LAYOUT_IMAGE_2D_RGBA
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
    OUTPUT_TYPE4 colorRGBA = IMAGE_READ(input, (int2)(x, y));
#elif defined OUTPUT_LAYOUT_IMAGE_2D_RGBA
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, w, z, y, x);
    const uint input_idx_R  = FUNC_CALL(get_input_index)(b, 0, w, z, y, x);
    const uint input_idx_G  = FUNC_CALL(get_input_index)(b, 1, w, z, y, x);
    const uint input_idx_B  = FUNC_CALL(get_input_index)(b, 2, w, z, y, x);
#if OUTPUT_FEATURE_NUM == 3
    INPUT_TYPE4 colorRGBA = { TO_INPUT_REORDER_TYPE(input[input_idx_R]), TO_INPUT_REORDER_TYPE(input[input_idx_G]), TO_INPUT_REORDER_TYPE(input[input_idx_B]), TO_INPUT_REORDER_TYPE(0.f) };
#else
    const uint input_idx_A  = FUNC_CALL(get_input_index)(b, 3, w, z, y, x);
    INPUT_TYPE4 colorRGBA = { TO_INPUT_REORDER_TYPE(input[input_idx_R]), TO_INPUT_REORDER_TYPE(input[input_idx_G]), TO_INPUT_REORDER_TYPE(input[input_idx_B]), TO_INPUT_REORDER_TYPE(input[input_idx_A]) };
#endif
#else
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, w, z, y, x);
    const uint input_idx  = FUNC_CALL(get_input_index)(b, f, w, z, y, x);
    const uint output_idx = FUNC_CALL(get_output_index)(ov[1],ov[2],ov[3],ov[4], ov[5], ov[6]);

#if defined MEAN_SUBTRACT_INSIDE_PARAMS
    float res = TO_MEAN_TYPE(input[input_idx]);
    res = MEAN_OP(res, VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
#elif defined MEAN_SUBTRACT_IN_BUFFER
#if defined MEAN_PER_FEATURE
    MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
    res = MEAN_OP(res, mean_subtract[f]);
#else
    // TODO Add support for 6D mean
    MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
    uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, w, z, y, x);
    res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], /*msv[3], msv[4],*/ msv[5], msv[6])]);
#endif
#else
    CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
#endif
#endif

#if defined INPUT0_LAYOUT_NV12
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, w, z, y, x);
    uint output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(R), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(G), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(B), NL_M, NL_N);
#elif INPUT0_LAYOUT_IMAGE_2D_RGBA
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, w, z, y, x);
    uint output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s0), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s1), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s2), NL_M, NL_N);
#if INPUT0_FEATURE_NUM == 4
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 3, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[1], ov[2], ov[3], ov[4], ov[5], ov[6]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s3), NL_M, NL_N);
#endif
#elif OUTPUT_LAYOUT_IMAGE_2D_RGBA
    IMAGE_WRITE(output, (int2)(x, y), colorRGBA);
#else
#if INPUT0_IS_FP && !OUTPUT_IS_FP
    output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res), ACTIVATION_PARAMS_TYPED);
#else
    output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(res), ACTIVATION_PARAMS_TYPED);
#endif
#endif
}

#undef INPUT_TYPE4
#undef OUTPUT_TYPE4
