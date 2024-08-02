// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"
#include "include/image_data.cl"

#define DECLARE_SAMPLER const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST

#if FP16_UNIT_USED
    #define IMAGE_READ(image, coord) read_imageh((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imageh((image), (coord), (val))
#else
    #define IMAGE_READ(image, coord) read_imagef((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imagef((image), (coord), (val))
#endif

inline uint8 FUNC(reshape_2_to_4)(uint o, uint i, uint y, uint x, uint dst_size_y, uint dst_size_x)
{
    uint _i  = i / (dst_size_y*dst_size_x);
    uint _yx = i % (dst_size_y*dst_size_x);
    uint _y = _yx / dst_size_x;
    uint _x = _yx % dst_size_x;
    return (uint8)(0, o, _i, 0, 0, _y,_x, 0);
}

inline uint8 FUNC(reshape_4_to_2)(uint o, uint i, uint y, uint x, uint src_size_y, uint src_size_x)
{
    uint _i = i*src_size_y*src_size_x + y*src_size_x + x;
    return (uint8)(0, o, _i, 0, 0, 0, 0, 0);
}

inline uint8 FUNC(reshape_4_to_5)(uint o, uint i, uint y, uint x,
    uint src_size_f, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_f = src_pitch_y * src_size_y;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_z = flat_idx % dst_size_z;
    flat_idx /= dst_size_z;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(0, dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0);
}

inline uint8 FUNC(reshape_5_to_4)(uint o, uint i, uint z, uint y, uint x,
    uint src_size_f, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_z = src_pitch_y * src_size_y;
    const uint src_pitch_f = src_pitch_z * src_size_z;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(0, dst_b, dst_f, 0, 0, dst_y, dst_x, 0);
}

inline uint8 FUNC(reshape_4_to_6)(uint o, uint i, uint y, uint x,
    uint src_size_f, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_f = src_pitch_y * src_size_y;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_z = flat_idx % dst_size_z;
    flat_idx /= dst_size_z;
    uint dst_w = flat_idx % dst_size_w;
    flat_idx /= dst_size_w;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(0, dst_b, dst_f, dst_w, dst_z, dst_y, dst_x, 0);
}

inline uint8 FUNC(reshape_6_to_4)(uint o, uint i, uint w, uint z, uint y, uint x,
    uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_z = src_pitch_y * src_size_y;
    const uint src_pitch_w = src_pitch_z * src_size_z;
    const uint src_pitch_f = src_pitch_w * src_size_w;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(0, dst_b, dst_f, 0, 0, dst_y, dst_x, 0);
}

inline uint8 FUNC(reshape_6_to_5)(uint o, uint i, uint w, uint z, uint y, uint x,
    uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_z = src_pitch_y * src_size_y;
    const uint src_pitch_w = src_pitch_z * src_size_z;
    const uint src_pitch_f = src_pitch_w * src_size_w;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_z = flat_idx % dst_size_z;
    flat_idx /= dst_size_z;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(0, dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0);
}

inline uint8 FUNC(reshape_grouped)(uint g, uint o, uint i, uint z, uint y, uint x, uint src_size_ofm, uint dst_size_ofm)
{
    const uint flat_ofm = g * src_size_ofm + o;
    const uint dst_ofm = flat_ofm % dst_size_ofm;
    const uint dst_g = flat_ofm / dst_size_ofm;
    const uint dst_ifm = i;
    const uint dst_z = z;
    const uint dst_y = y;
    const uint dst_x = x;
    return (uint8)(dst_g, dst_ofm, dst_ifm, 0, dst_z, dst_y, dst_x, 0);
}

inline uint8 FUNC(reshape_weights)(
    uint o, uint i, uint w, uint z, uint y, uint x,
    uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
    uint src_dims, uint dst_dims)
{
    if (src_dims == 4 && dst_dims == 2)
    {
        return FUNC_CALL(reshape_4_to_2)(o,i,y,x,src_size_y,src_size_x);
    }
    else if (src_dims == 2 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_2_to_4)(o,i,y,x,dst_size_y,dst_size_x);
    }
    else if (src_dims == 4 && dst_dims == 6)
    {
        return FUNC_CALL(reshape_4_to_6)(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_w, dst_size_z, dst_size_y, dst_size_x);
    }
    else if (src_dims == 6 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_6_to_4)(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
    }
    else if (src_dims == 4 && dst_dims == 5)
    {
        return FUNC_CALL(reshape_4_to_5)(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
    }
    else if (src_dims == 5 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_5_to_4)(o, i, z, y, x, src_size_f, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
    }
    else if (src_dims == 6 && dst_dims == 5)
    {
        return FUNC_CALL(reshape_6_to_5)(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
    }

    return (uint8)(0, o, i, w, z, y, x, 0);
}

inline uint8 FUNC(reshape_dims_with_groups)(
    uint g, uint o, uint i, uint w, uint z, uint y, uint x,
    uint src_size_ofm, uint src_size_ifm, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_ofm, uint dst_size_ifm, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
    uint src_dims, uint dst_dims, uint src_size_groups, uint dst_size_groups)
{
    if (src_dims == 5 && dst_dims == 4)  // goiyx -> oiyx
    {
        return FUNC_CALL(reshape_grouped)(g, o, i, 0, y, x, src_size_ofm, dst_size_ofm);
    }
    else if (src_dims == 6 && dst_dims == 5)  // goizyx -> oizyx or goizyx -> goiyx
    {
        return FUNC_CALL(reshape_grouped)(g, o, i, z, y, x, src_size_ofm, dst_size_ofm);
    }
    else if (src_dims == 6 && dst_dims == 4) // goizyx -> oiyx
    {
        return FUNC_CALL(reshape_grouped)(g, o, i, 0, y, x, src_size_ofm, dst_size_ofm);
    }

    return (uint8)(g, o, i, w, z, y, x, 0);
}

// ======================================================================
// Reshape indices between WeightTensors.
// Weight tensor has _IFM in place of _FEATURE_NUM and does not support 6D,
// so macro for DataTensors cannot be used directly.
#define RESHAPE_WEIGHT_DIMS(src_prefix, dst_prefix, o, i, w, z, y, x)   \
    FUNC_CALL(reshape_weights)(                                         \
        o, i, w, z, y, x,                                               \
        CAT(src_prefix, _IFM_NUM),                                      \
        1,                                                              \
        CAT(src_prefix, _SIZE_Z),                                       \
        CAT(src_prefix, _SIZE_Y),                                       \
        CAT(src_prefix, _SIZE_X),                                       \
        CAT(dst_prefix, _IFM_NUM),                                      \
        1,                                                              \
        CAT(dst_prefix, _SIZE_Z),                                       \
        CAT(dst_prefix, _SIZE_Y),                                       \
        CAT(dst_prefix, _SIZE_X),                                       \
        CAT(src_prefix, _DIMS),                                         \
        CAT(dst_prefix, _DIMS))

#define RESHAPE_WEIGHT_DIMS_WITH_GROUPS(src_prefix, dst_prefix, g, o, i, w, z, y, x)\
    FUNC_CALL(reshape_dims_with_groups)(                                            \
        g, o, i, w, z, y, x,                                                        \
        CAT(src_prefix, _OFM_NUM),                                                  \
        CAT(src_prefix, _IFM_NUM),                                                  \
        1,                                                                          \
        CAT(src_prefix, _SIZE_Z),                                                   \
        CAT(src_prefix, _SIZE_Y),                                                   \
        CAT(src_prefix, _SIZE_X),                                                   \
        CAT(dst_prefix, _OFM_NUM),                                                  \
        CAT(dst_prefix, _IFM_NUM),                                                  \
        1,                                                                          \
        CAT(dst_prefix, _SIZE_Z),                                                   \
        CAT(dst_prefix, _SIZE_Y),                                                   \
        CAT(dst_prefix, _SIZE_X),                                                   \
        CAT(src_prefix, _DIMS),                                                     \
        CAT(dst_prefix, _DIMS),                                                     \
        CAT(src_prefix, _GROUPS_NUM),                                               \
        CAT(dst_prefix, _GROUPS_NUM))


///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if   INPUT0_SIMPLE && INPUT0_DIMS <= 4
    return GET_FILTER_INDEX(INPUT0, 0, o, i, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_FILTER_INDEX_5D(INPUT0, 0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16    || \
      defined INPUT0_LAYOUT_OS_I_OSV16      || \
      defined INPUT0_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV_INDEX(INPUT0, o, i, y, x, 16);
#elif defined INPUT0_LAYOUT_OS_I_OSV8__AI8
    return GET_FILTER_OS_IYX_OSV_INDEX(INPUT0, o, i, y, x, 8);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV_INDEX(INPUT0, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV_INDEX(INPUT0, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV32__AI32
    return GET_FILTER_OS_IYX_OSV_INDEX(INPUT0, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_O_IS_YX_ISV4
    return GET_FILTER_O_IS_ZYX_ISV16_INDEX(INPUT0, o, i, 0, y, x, 4);
#elif defined INPUT0_LAYOUT_O_IS_YX_ISV16
    return GET_FILTER_O_IS_ZYX_ISV16_INDEX(INPUT0, o, i, 0, y, x, 16);
#elif defined INPUT0_LAYOUT_IYX_OSV64
    return GET_FILTER_OS_IYX_OSV_INDEX(INPUT0, o, i, y, x, 64);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV_ROTATE_180_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_I_YXS_OS_YXSV2_OSV16
    return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
    #error - not supported yet
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISA8_OSV16_ISV4
    return GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISA8_OSV16_ISV4
    return GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISV16_OSV16
    return GET_FILTER_OS_IS_YX_ISV_OSV_INDEX(INPUT0, o, i, y, x, 16, 16);
#elif defined INPUT0_LAYOUT_OIYX_O16
    return GET_FILTER_OIYX_O16(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISV16_OSV16
    return GET_FILTER_OS_IS_ZYX_ISV_OSV_INDEX(INPUT0, o, i, z, y, x, 16, 16);
#elif defined INPUT0_LAYOUT_IS_OS_ZYX_ISV16_OSV16
    return GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IS_OS_YX_ISV16_OSV16
    return GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_ZYXI_OSV16
    return GET_FILTER_OS_ZYXI_OSV16(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_GOIZYX
    return GET_FILTER_GOIZYX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_GIOZYX
    return GET_FILTER_GIOZYX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV8
    return GET_FILTER_OS_IYX_OSV16(INPUT0, o, i, y, x, 8);
#elif defined INPUT0_LAYOUT_G_OS_IYX_OSV8
    return GET_FILTER_G_OS_IYX_OSV16(INPUT0, g, o, i, y, x, 8);
#elif defined INPUT0_LAYOUT_G_OS_IYX_OSV16
    return GET_FILTER_G_OS_IYX_OSV16(INPUT0, g, o, i, y, x, 16);
#elif defined INPUT0_LAYOUT_G_OS_IYX_OSV32
    return GET_FILTER_G_OS_IYX_OSV16(INPUT0, g, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_GS_OIYX_GSV16
    return GET_FILTER_GS_OIYX_GSV16(INPUT0, g, o, i, y, x, 16);
#elif defined INPUT0_LAYOUT_GS_OIZYX_GSV16
    return GET_FILTER_GS_OIZYX_GSV16(INPUT0, g, o, i, z, y, x, 16);
#elif defined INPUT0_LAYOUT_GS_OIYX_GSV32
    return GET_FILTER_GS_OIYX_GSV16(INPUT0, g, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_GYXIO || \
      defined INPUT0_LAYOUT_GOIYX || \
      defined INPUT0_LAYOUT_GIOYX
    return GET_FILTER_GOIYX(INPUT0, g, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_OSV16_ISV16
    return GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_OSV16_ISV16
    return GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_OSV8_ISV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_G_OS_IS_ZYX_OSV16_ISV16
    return GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_OSV32_ISV16
    return GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_OSV64_ISV16
    return GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_GS_OI_YXS_GSV16_YXSV4
    return GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(INPUT0, g, o, i, y, x);
#elif defined INPUT0_LAYOUT_GS_OI_YXS_GSV32_YXSV4
    return GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(INPUT0, g, o, i, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV16_ISV4
    return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV16_ISV16
    return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV16_ISV32
    return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV32_ISV4
    return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV32_ISV16
    return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV32_ISV32
    return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(INPUT0, g, o, i, z, y, x);
#else
#error reorder_weights.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if   OUTPUT_SIMPLE && OUTPUT_DIMS <= 4
    return GET_FILTER_INDEX(OUTPUT, 0, o, i, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 5
    return GET_FILTER_INDEX_5D(OUTPUT, 0, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16    || \
      defined OUTPUT_LAYOUT_OS_I_OSV16      || \
      defined OUTPUT_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, y, x, 16);
#elif defined OUTPUT_LAYOUT_OS_I_OSV8__AI8
    return GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, y, x, 8);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32__AI32
    return GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV64
    return GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, y, x, 64);
#elif defined OUTPUT_LAYOUT_O_IS_YX_ISV4
    return GET_FILTER_O_IS_ZYX_ISV16_INDEX(OUTPUT, o, i, 0, y, x, 4);
#elif defined OUTPUT_LAYOUT_O_IS_YX_ISV16
    return GET_FILTER_O_IS_ZYX_ISV16_INDEX(OUTPUT, o, i, 0, y, x, 16);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV_ROTATE_180_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_I_YXS_OS_YXSV2_OSV16
    return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
    return 0; //will not be used for images
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV16_ISV4
    return GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISA8_OSV16_ISV4
    return GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV16_ISV4
    return GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2
    return GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4
    return GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV32_ISV4
    return GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV16_OSV16
    return GET_FILTER_OS_IS_YX_ISV_OSV_INDEX(OUTPUT, o, i, y, x, 16, 16);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISV16_OSV16
    return GET_FILTER_OS_IS_ZYX_ISV_OSV_INDEX(OUTPUT, o, i, z, y, x, 16, 16);
#elif defined OUTPUT_LAYOUT_IS_OS_ZYX_ISV16_OSV16
    return GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IS_OS_YX_ISV16_OSV16
    return GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_ZYXI_OSV16
    return GET_FILTER_OS_ZYXI_OSV16(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_GOIZYX || defined OUTPUT_LAYOUT_GIOZYX
    return GET_FILTER_INDEX_5D(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV8
    return GET_FILTER_OS_IYX_OSV16(OUTPUT, o, i, y, x, 8);
#elif defined OUTPUT_LAYOUT_G_OS_IYX_OSV8
    return GET_FILTER_G_OS_IYX_OSV16(OUTPUT, g, o, i, y, x, 8);
#elif defined OUTPUT_LAYOUT_G_OS_IYX_OSV16
    return GET_FILTER_G_OS_IYX_OSV16(OUTPUT, g, o, i, y, x, 16);
#elif defined OUTPUT_LAYOUT_G_OS_IYX_OSV32
    return GET_FILTER_G_OS_IYX_OSV16(OUTPUT, g, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_GS_OIYX_GSV16
    return GET_FILTER_GS_OIYX_GSV16(OUTPUT, g, o, i, y, x, 16);
#elif defined OUTPUT_LAYOUT_GS_OIZYX_GSV16
    return GET_FILTER_GS_OIZYX_GSV16(OUTPUT, g, o, i, z, y, x, 16);
#elif defined OUTPUT_LAYOUT_GS_OIYX_GSV32
    return GET_FILTER_GS_OIYX_GSV16(OUTPUT, g, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_GYXIO || \
      defined OUTPUT_LAYOUT_GOIYX || \
      defined OUTPUT_LAYOUT_GIOYX
    return GET_FILTER_GOIYX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_GI_YXS_OS_YXSV2_OSV16
    return GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_IS_OS_ZYX_ISV16_OSV16
    return GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_IS_OS_YX_ISV16_OSV16
    return GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_OS_IS_ZYX_ISV16_OSV16
    return GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_ISV8_OSV16_ISV2
    return GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_OS_IS_ZYX_ISV8_OSV16_ISV2
    return GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(OUTPUT, g, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_GIY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_GIY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_GS_OI_YXS_GSV4_YXSV4
    return GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_ISV16_OSV16
    return GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV16_ISV16
    return GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV16_ISV16
    return GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_ZYX_OSV16_ISV16
    return GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV32_ISV16
    return GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV64_ISV16
    return GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_GS_OI_YXS_GSV16_YXSV4
    return GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_GS_OI_YXS_GSV32_YXSV4
    return GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_OSV16_ISV4
    return GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV16_ISV4
    return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV16_ISV16
    return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV16_ISV32
    return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV32_ISV4
    return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV32_ISV16
    return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV32_ISV32
    return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(OUTPUT, g, o, i, z, y, x);
#else
#error reorder_weights.cl: output format - not supported
#endif
}

#if OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
    const unsigned o = get_global_id(0);
    const unsigned iyx = get_global_id(1);
    const unsigned x = iyx % INPUT0_SIZE_X;
    const unsigned y = (iyx / INPUT0_SIZE_X) % INPUT0_SIZE_Y;
    const unsigned i = (iyx / INPUT0_SIZE_X) / INPUT0_SIZE_Y;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 4) input_val = (MAKE_VECTOR_TYPE(UNIT_TYPE, 4))(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    const int2 coord = (int2)(o, iyx);
    uint8 ir = RESHAPE_WEIGHT_DIMS(OUTPUT, INPUT0, o, i, 0, 0, y, x);
    input_val.s0 = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir.s0,ir.s1,ir.s2,ir.s4,ir.s5,ir.s6)]);
    IMAGE_WRITE(output, coord, input_val);
}
#else
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if OUTPUT_GROUPS_NUM > 1
    const unsigned g = (uint)get_global_id(0) / OUTPUT_OFM_NUM;
    const unsigned o = (uint)get_global_id(0) % OUTPUT_OFM_NUM;
#else
    const unsigned g = 0;
    const unsigned o = (uint)get_global_id(0);
#endif

    const unsigned i = (uint)get_global_id(1);

#if   OUTPUT_DIMS == 2 || (OUTPUT_DIMS == 3 && OUTPUT_GROUPED)
    const unsigned x = 0;
    const unsigned y = 0;
    const unsigned z = 0;
#elif OUTPUT_DIMS == 4 || (OUTPUT_DIMS == 5 && OUTPUT_GROUPED)
    const unsigned x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
    const unsigned y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const unsigned z = 0;
#elif OUTPUT_DIMS == 5 || (OUTPUT_DIMS == 6 && OUTPUT_GROUPED)
    const unsigned zyx = get_global_id(2);
    const unsigned x = zyx % OUTPUT_SIZE_X;
    const unsigned y = (zyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const unsigned z = (zyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
#endif

#if OUTPUT_GROUPS_NUM > 1 //  Add grouped macro instead this check
    uint8 ir = RESHAPE_WEIGHT_DIMS_WITH_GROUPS(OUTPUT, INPUT0, g, o, i, 0, z, y, x);
#else
    uint8 ir = RESHAPE_WEIGHT_DIMS(OUTPUT, INPUT0, o, i, 0, z, y, x);
#endif

    uint input_idx = FUNC_CALL(get_input_index)(ir.s0,ir.s1,ir.s2,ir.s4,ir.s5,ir.s6);
#if !REORDER_ROTATE
    uint output_idx = FUNC_CALL(get_output_index)(g, o, i, z, y, x);
#else
    uint output_idx = FUNC_CALL(get_output_index)(g, o, i, OUTPUT_SIZE_Z - z - 1, OUTPUT_SIZE_Y - y - 1, OUTPUT_SIZE_X - x - 1);
#endif
#ifdef BF16_INPUT
    output[output_idx] = TO_OUTPUT_TYPE(_convert_as_bfloat16_float(input[input_idx]));
#else
    output[output_idx] = TO_OUTPUT_TYPE(input[input_idx]);
#endif
}
#endif
