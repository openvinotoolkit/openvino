// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_weights.cl"
#include "include/reshape_dims.cl"
#include "include/data_types.cl"

#define DECLARE_SAMPLER const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST

#if FP16_UNIT_USED
    #define IMAGE_READ(image, coord) read_imageh((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imageh((image), (coord), (val))
#else
    #define IMAGE_READ(image, coord) read_imagef((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imagef((image), (coord), (val))
#endif

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if   INPUT0_SIMPLE && INPUT0_DIMS <= 4
    return GET_FILTER_INDEX(INPUT0, 0, o, i, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_FILTER_INDEX_5D(INPUT0, 0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16    || \
      defined INPUT0_LAYOUT_OS_I_OSV16      || \
      defined INPUT0_LAYOUT_OS_I_OSV8__AI8  || \
      defined INPUT0_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV32__AI32
    return GET_FILTER_OS_IYX_OSV32__AI32_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_O_IS_YX_ISV16
    return GET_FILTER_O_IS_YX_ISV16_INDEX(INPUT0, o, i, y, x, 16);
#elif defined INPUT0_LAYOUT_IYX_OSV64
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, 64);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
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
#elif defined INPUT0_LAYOUT_IS_O_YX_ISV32
    return GET_FILTER_IS_O_YX_ISV32(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_IS_O32_YX_ISV32_SWIZZLED_BY_4
    return GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_Y_X8_OSV8_ISV4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISV16_OSV16
    return GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OIYX_O16
    return GET_FILTER_OIYX_O16(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISV16_OSV16
    return GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IS_OS_ZYX_ISV16_OSV16
    return GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IS_OS_YX_ISV16_OSV16
    return GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_OSV32_ISV32_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_ZYXI_OSV16
    return GET_FILTER_OS_ZYXI_OSV16(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_I_YXS_OSV4_YXSV4
    return GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_GOIZYX
    return GET_FILTER_GOIZYX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_GIOZYX
    return GET_FILTER_GIOZYX(INPUT0, g, o, i, z, y, x);
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
#elif defined INPUT0_LAYOUT_OS_IS_YX_OSV8_ISV2
    return GET_FILTER_OS_IS_YX_OSV8_ISV2_INDEX(INPUT0, o, i, y, x);
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
      defined OUTPUT_LAYOUT_OS_I_OSV8__AI8  || \
      defined OUTPUT_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32__AI32
    return GET_FILTER_OS_IYX_OSV32__AI32_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV64
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, 64);
#elif defined OUTPUT_LAYOUT_O_IS_YX_ISV16
    return GET_FILTER_O_IS_YX_ISV16_INDEX(OUTPUT, o, i, y, x, 16);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
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
#elif defined OUTPUT_LAYOUT_IS_O_YX_ISV32
    return GET_FILTER_IS_O_YX_ISV32(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_IS_O32_YX_ISV32_SWIZZLED_BY_4
    return GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_Y_X8_OSV8_ISV4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV16_ISV4
    return GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV8_ISV2
    return GET_FILTER_OS_IS_YX_OSV8_ISV2_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2
    return GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4
    return GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV32_ISV4
    return GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_OSA4_ISA8_OSV8_ISV2
    return GET_FILTER_G_OS_IS_YX_OSA4_ISA8_OSV8_ISV2_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_OSA4_ISA8_OSV8_ISV4
    return GET_FILTER_G_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV2
    return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV2_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV16_OSV16
    return GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_YXI_OSV16
    return GET_FILTER_OS_YXI_OSV16(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISV16_OSV16
    return GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IS_OS_ZYX_ISV16_OSV16
    return GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IS_OS_YX_ISV16_OSV16
    return GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_OSV32_ISV32_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISV8_OSV16_ISV2
    return GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_ZYXI_OSV16
    return GET_FILTER_OS_ZYXI_OSV16(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_I_YXS_OSV4_YXSV4
    return GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_GOIZYX || defined OUTPUT_LAYOUT_GIOZYX
    return GET_FILTER_INDEX_5D(OUTPUT, g, o, i, z, y, x);
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
    input_val.s0 = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[4],ir[5],ir[6])]);
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

    uint input_idx = FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[4],ir[5],ir[6]);
#if !REORDER_ROTATE
    uint output_idx = FUNC_CALL(get_output_index)(g, o, i, z, y, x);
#else
    uint output_idx = FUNC_CALL(get_output_index)(g, o, i, OUTPUT_SIZE_Z - z - 1, OUTPUT_SIZE_Y - y - 1, OUTPUT_SIZE_X - x - 1);
#endif

    output[output_idx] = TO_OUTPUT_TYPE(input[input_idx]);
}
#endif
