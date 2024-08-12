// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/reshape_dims.cl"
#include "include/fetch_utils.cl"

#include "include/image_data.cl"

#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT_REORDER_TYPE, 4)
#define OUTPUT_TYPE4 MAKE_VECTOR_TYPE(OUTPUT_REORDER_TYPE, 4)

KERNEL (reorder_data)(
    OPTIONAL_SHAPE_INFO_ARG
#if INPUT0_LAYOUT_NV12 || INPUT0_LAYOUT_IMAGE_2D_RGBA || SURFACE_INPUT
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
#if INPUT0_LAYOUT_BYFX
    // GWS_FEATURE takes Y for byfx format
    const uint b = get_global_id(GWS_BATCH);
    const uint y = get_global_id(GWS_FEATURE);
#else
    const uint b = get_global_id(GWS_BATCH);
    const uint f = get_global_id(GWS_FEATURE);
#endif

#if   INPUT0_DIMS == 2
    const uint y = 0;
    const uint x = 0;
    const uint z = 0;
    const uint w = 0;
    const uint u = 0;
    const uint v = 0;
#elif INPUT0_DIMS == 4
    #if INPUT0_LAYOUT_BYFX
        // GWS_YX takes (F and X) axes for byfx format
        const uint f = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
        const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
    #else
        const uint y = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
        const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
    #endif
    const uint z = 0;
    const uint w = 0;
    const uint u = 0;
    const uint v = 0;
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
    const uint u = 0;
    const uint v = 0;
#elif INPUT0_DIMS == 6
    const uint gid_yx = (uint)(get_global_id(GWS_YX));
    const uint x = gid_yx % INPUT0_SIZE_X;
    const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
    const uint u = 0;
    const uint v = 0;
#elif INPUT0_DIMS == 7
    const uint gid_yx = (uint)(get_global_id(GWS_YX));
    const uint x = gid_yx % INPUT0_SIZE_X;
    const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
    const uint u = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z / INPUT0_SIZE_W % INPUT0_SIZE_U;
    const uint v = 0;
#elif INPUT0_DIMS == 8
    const uint gid_yx = (uint)(get_global_id(GWS_YX));
    const uint x = gid_yx % INPUT0_SIZE_X;
    const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
    const uint u = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z / INPUT0_SIZE_W % INPUT0_SIZE_U;
    const uint v = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z / INPUT0_SIZE_W / INPUT0_SIZE_U % INPUT0_SIZE_V;
#endif

#if defined INPUT0_LAYOUT_NV12 && !SURFACE_INPUT
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
    float4 colorVYU = read_imagef(input, sampler, (int2)(y, f));

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
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, v, u, w, z, y, x);
    const uint input_idx_R  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 0, v, u, w, z, y, x);
    const uint input_idx_G  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 1, v, u, w, z, y, x);
    const uint input_idx_B  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 2, v, u, w, z, y, x);
#if OUTPUT_FEATURE_NUM == 3
    INPUT_TYPE4 colorRGBA = { TO_INPUT_REORDER_TYPE(input[input_idx_R]), TO_INPUT_REORDER_TYPE(input[input_idx_G]), TO_INPUT_REORDER_TYPE(input[input_idx_B]), TO_INPUT_REORDER_TYPE(0.f) };
#else
    const uint input_idx_A  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 3, v, u, w, z, y, x);
    INPUT_TYPE4 colorRGBA = { TO_INPUT_REORDER_TYPE(input[input_idx_R]), TO_INPUT_REORDER_TYPE(input[input_idx_G]), TO_INPUT_REORDER_TYPE(input[input_idx_B]), TO_INPUT_REORDER_TYPE(input[input_idx_A]) };
#endif
#else
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, v, u, w, z, y, x);
    const uint input_idx  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, v, u, w, z, y, x);
    const uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7);

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
    uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, v, u, w, z, y, x);
    res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv.s0, msv.s1, /*msv.s2, msv.s3, msv.s4,msv.s5,*/ msv.s6, msv.s7)]);
#endif
#elif SURFACE_INPUT
    float4 Y = read_imagef(input, (int2)(y, f));
    float Ycomponent = mad(Y.x, 296.82f, -18.624f);
    float res = clamp(Ycomponent, 0.f, 255.f);
#else
    #ifdef BF16_INPUT
        CALC_TYPE res = TO_CALC_TYPE(_convert_as_bfloat16_float(input[input_idx]));
    #else
        CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
    #endif
#endif
#endif

#if defined INPUT0_LAYOUT_NV12 && !SURFACE_INPUT
    uint8 ov0 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, v, u, w, z, y, x);
    uint8 ov1 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, v, u, w, z, y, x);
    uint8 ov2 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, v, u, w, z, y, x);
    uint output_idx_R = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov0.s0, ov0.s1, ov0.s2, ov0.s3, ov0.s4, ov0.s5, ov0.s6, ov0.s7);
    uint output_idx_G = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov1.s0, ov1.s1, ov1.s2, ov1.s3, ov1.s4, ov1.s5, ov1.s6, ov1.s7);
    uint output_idx_B = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov2.s0, ov2.s1, ov2.s2, ov2.s3, ov2.s4, ov2.s5, ov2.s6, ov2.s7);
    #if HAS_FUSED_OPS
        res = TO_OUTPUT_REORDER_TYPE(R);
        FUSED_OPS;
        output[output_idx_R] = FUSED_OPS_RESULT;
        res = TO_OUTPUT_REORDER_TYPE(G);
        FUSED_OPS;
        output[output_idx_G] = FUSED_OPS_RESULT;
        res = TO_OUTPUT_REORDER_TYPE(B);
        FUSED_OPS;
        output[output_idx_B] = FUSED_OPS_RESULT;
    #else
        output[output_idx_R] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(R), NL_M, NL_N);
        output[output_idx_G] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(G), NL_M, NL_N);
        output[output_idx_B] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(B), NL_M, NL_N);
    #endif
#elif INPUT0_LAYOUT_IMAGE_2D_RGBA
    uint8 ov0 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, v, u, w, z, y, x);
    uint8 ov1 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, v, u, w, z, y, x);
    uint8 ov2 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, v, u, w, z, y, x);
    uint output_idx_0 = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov0.s0, ov0.s1, ov0.s2, ov0.s3, ov0.s4, ov0.s5, ov0.s6, ov0.s7);
    uint output_idx_1 = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov1.s0, ov1.s1, ov1.s2, ov1.s3, ov1.s4, ov1.s5, ov1.s6, ov1.s7);
    uint output_idx_2 = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov2.s0, ov2.s1, ov2.s2, ov2.s3, ov2.s4, ov2.s5, ov2.s6, ov2.s7);
    #if HAS_FUSED_OPS
        res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s0);
        FUSED_OPS;
        output[output_idx_0] = FUSED_OPS_RESULT;
        res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s1);
        FUSED_OPS;
        output[output_idx_1] = FUSED_OPS_RESULT;
        res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s2);
        FUSED_OPS;
        output[output_idx_2] = FUSED_OPS_RESULT;
    #else
        output[output_idx_0] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s0), NL_M, NL_N);
        output[output_idx_1] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s1), NL_M, NL_N);
        output[output_idx_2] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s2), NL_M, NL_N);
    #endif
    #if INPUT0_FEATURE_NUM == 4
        uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 3, v, u, w, z, y, x);
        uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7);
        #if HAS_FUSED_OPS
            res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s3);
            FUSED_OPS;
            output[output_idx] = FUSED_OPS_RESULT;
        #else
            output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s3), NL_M, NL_N);
        #endif
    #endif
#elif OUTPUT_LAYOUT_IMAGE_2D_RGBA
    IMAGE_WRITE(output, (int2)(x, y), colorRGBA);
#else
    #if INPUT0_IS_FP && !OUTPUT_IS_FP
    #if CONVERT_TRUNCATE
        #define __TO_OUTPUT_REORDER_TYPE(res) TO_OUTPUT_REORDER_TYPE(convert_long(res))
    #else
        #define __TO_OUTPUT_REORDER_TYPE(res) TO_OUTPUT_REORDER_TYPE_SAT(res)
    #endif
    #else
        #define __TO_OUTPUT_REORDER_TYPE(res) TO_OUTPUT_REORDER_TYPE(res)
    #endif

    #if HAS_FUSED_OPS
        res = __TO_OUTPUT_REORDER_TYPE(res);
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
    #else
        output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, __TO_OUTPUT_REORDER_TYPE(res), ACTIVATION_PARAMS_TYPED);
    #endif
#undef __TO_OUTPUT_REORDER_TYPE
#endif
}

#undef INPUT_TYPE4
#undef OUTPUT_TYPE4
