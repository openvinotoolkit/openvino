/*
// Copyright (c) 2016 Intel Corporation
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
*/

#include "common.cl"

#define GET_DATA_INDEX(prefix, b, f, y, x)  \
    CAT(prefix, _OFFSET) +                  \
    (x)*CAT(prefix, _X_PITCH) +             \
    (y)*CAT(prefix, _Y_PITCH) +             \
    (f)*CAT(prefix, _FEATURE_PITCH) +       \
    (b)*CAT(prefix, _BATCH_PITCH)
    
#define GET_DATA_INDEX_SAFE(prefix, b, f, y, x)                     \
    CAT(prefix, _OFFSET) +                                          \
    (x % CAT(prefix, _SIZE_X     ))*CAT(prefix, _X_PITCH) +         \
    (y % CAT(prefix, _SIZE_Y     ))*CAT(prefix, _Y_PITCH) +         \
    (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) +   \
    (b % CAT(prefix, _BATCH_NUM  ))*CAT(prefix, _BATCH_PITCH)
    
    
#define GET_DATA_BS_FYX_BSV8_INDEX(prefix, b, f, y, x, sub_group_size)  \
    CAT(prefix, _OFFSET) +                                              \
    ((b) % (sub_group_size)) +                                          \
    (sub_group_size)*(                                                  \
        (x)*CAT(prefix, _X_PITCH) +                                     \
        (y)*CAT(prefix, _Y_PITCH) +                                     \
        (f)*CAT(prefix, _FEATURE_PITCH) +                               \
        ((b) / (sub_group_size))*CAT(prefix, _BATCH_PITCH)              \
    )

inline uint FUNC(get_bf8_xy16_index)(uint b, uint f, uint y, uint x, uint x_size, uint y_size, uint f_size, uint offset)
{
    const uint xy_idx = x + y * x_size;
    const uint xy_offset = (xy_idx % 16) + (xy_idx / 16) * 16 * 8;
    const uint xy_block_num = (x_size * y_size + 16 - 1) / 16;
    const uint f_offset = (f % 8) * 16 + (f / 8) * xy_block_num * 16 * 8;
    const uint f_block_num = (f_size + 8 - 1) / 8;
    const uint b_offset = b * f_block_num * xy_block_num * 128;

    const size_t idx = offset + xy_offset + f_offset + b_offset;

    return idx;
}

inline uint FUNC(get_byxf_af32_index)(uint b, uint f, uint y, uint x, uint y_pitch, uint b_pitch, uint f_size, uint offset)
{
	const uint f_aligned_to_32 = ((f_size + 31) / 32) * 32;
	const uint b_offset = b * b_pitch;
	const uint xy_offset = f_aligned_to_32 * x + y_pitch * y;
	const uint f_offset = f;
	const size_t idx = offset + xy_offset + b_offset + f_offset;
	return idx;
}

#define GET_DATA_BYXF_AF32_INDEX(prefix, b, f, y, x)\
	FUNC_CALL(get_byxf_af32_index)(                 \
		b, f, y, x, CAT(prefix, _Y_PITCH),          \
		CAT(prefix, _BATCH_PITCH),                      \
		CAT(prefix, _FEATURE_NUM),                 \
		CAT(prefix, _OFFSET))

#define GET_DATA_BF8_XY16_INDEX(prefix, b, f, y, x)     \
    FUNC_CALL(get_bf8_xy16_index)(                      \
        b, f, y, x, CAT(prefix, _SIZE_X ),              \
        CAT(prefix, _SIZE_Y),                           \
        CAT(prefix, _FEATURE_NUM),                      \
        CAT(prefix, _OFFSET))

inline uint FUNC(get_fs_bs_yx_bsv4_fsv32_index)(uint b, uint f, uint y, uint x,
    uint x_pad_before, uint x_size, uint x_pad_after,
    uint y_pad_before, uint y_size, uint y_pad_after,
    uint size_f, uint size_b)
{
    const uint f_32_aligned = ((size_f + 31)/32) * 32;
    const uint b_4_aligned = ((size_b + 3)/4) * 4;
    const uint fsv_idx = f % 32;
    const uint bsv_idx = b % 4;
    const uint fs_idx = f / 32;
    const uint bs_idx = b / 4;

    const uint x_pitch = 32 * 4;
    const uint y_pitch = 32 * 4 * (x_pad_before + x_size + x_pad_after);
    const uint bs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
    const uint fs_pitch = bs_pitch * (b_4_aligned / 4);
    uint offset = x_pitch * x_pad_before + y_pitch * y_pad_before;

    size_t idx = offset + fsv_idx + bsv_idx * 32;
    idx += 32*4 * x;
    idx += y * y_pitch;
    idx += bs_idx * bs_pitch;
    idx += fs_idx * fs_pitch;

    return idx;
}

#define GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(prefix, b, f, y, x)\
	FUNC_CALL(get_fs_bs_yx_bsv4_fsv32_index)(       \
		b, f, y, x,                                 \
        CAT(prefix, _PAD_BEFORE_SIZE_X),            \
        CAT(prefix, _SIZE_X),                       \
        CAT(prefix, _PAD_AFTER_SIZE_X),             \
        CAT(prefix, _PAD_BEFORE_SIZE_Y),            \
        CAT(prefix, _SIZE_Y),                       \
        CAT(prefix, _PAD_AFTER_SIZE_Y),             \
		CAT(prefix, _FEATURE_NUM),                  \
        CAT(prefix, _BATCH_NUM))

#define GET_FILTER_INDEX(prefix, o, i, y, x)    \
    CAT(prefix, _OFFSET) +                      \
    (x)*CAT(prefix, _X_PITCH) +                 \
    (y)*CAT(prefix, _Y_PITCH) +                 \
    (i)*CAT(prefix, _IFM_PITCH) +               \
    (o)*CAT(prefix, _OFM_PITCH)
    
#define GET_FILTER_INDEX_SAFE(prefix, o, i, y, x)           \
    CAT(prefix, _OFFSET) +                                  \
    (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) +     \
    (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) +     \
    (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) +   \
    (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH)
    
#define GET_FILTER_OS_IYX_OSV8_INDEX(prefix, o, i, y, x, sub_group_size)    \
    CAT(prefix, _OFFSET) +                                                  \
    ((o) % (sub_group_size)) +                                              \
    (sub_group_size)*(                                                      \
        (x)*CAT(prefix, _X_PITCH) +                                         \
        (y)*CAT(prefix, _Y_PITCH) +                                         \
        (i)*CAT(prefix, _IFM_PITCH) +                                       \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                    \
    )

#define GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(prefix, o, i, y, x, sub_group_size)    \
    CAT(prefix, _OFFSET) +                                                  \
    ((o) % (sub_group_size)) +                                              \
    (sub_group_size)*(                                                      \
        (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) +             \
        (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) +             \
        (i)*CAT(prefix, _IFM_PITCH) +                                       \
        ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH)                    \
    )

inline uint FUNC(get_i_yxs_os_yxsv2_osv_index)(uint o, uint i, uint y, uint x, uint x_size, uint i_pitch, uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
    const uint aligned_ofm_line = x_pitch;
    const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
    const uint dst_height = i*ifm_height_pitch + y*x_size + x;
    const uint base_filter_index = y*x_size + x;

    const uint aligned_height = dst_height & 0xfffffffe;
    const uint base_filter_odd = (base_filter_index & 0x1);

    uint slice_id = o / sub_group_size;
    uint id_in_slice = o % sub_group_size;
    uint slice_pitch = 2*sub_group_size;
    uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
    
    const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
    const size_t idx = offset + aligned_height*aligned_ofm_line + in_line;

    return idx;
}

#define GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
    FUNC_CALL(get_i_yxs_os_yxsv2_osv_index)(                                    \
        o, i, y, x, CAT(prefix, _SIZE_X ),                                      \
        CAT(prefix, _IFM_PITCH),                                                \
        CAT(prefix, _Y_PITCH),                                                  \
        CAT(prefix, _X_PITCH),                                                  \
        CAT(prefix, _OFFSET),                                                   \
        sub_group_size)

inline uint FUNC(get_iy_xs_os_xsv2_osv_index)(uint o, uint i, uint y, uint x, uint x_size, uint i_pitch, uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
    const uint aligned_ofm_line = x_pitch;
    const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
    const uint aligned_x_line = y_pitch / x_pitch;
    const uint dst_height = i*ifm_height_pitch + y*aligned_x_line + x;
    const uint base_filter_index = x;

    const uint aligned_height = dst_height & 0xfffffffe;
    const uint base_filter_odd = (base_filter_index & 0x1);

    uint slice_id = o / sub_group_size;
    uint id_in_slice = o % sub_group_size;
    uint slice_pitch = 2*sub_group_size;
    uint offset_in_slice = (int)(sub_group_size*base_filter_odd);

    const bool last_line_in_base_filter = (x == (x_size - 1));
    if (last_line_in_base_filter && base_filter_odd == 0)
    {
        const uint element_in_slice = 32;
        slice_id = o / element_in_slice;
        id_in_slice = o % element_in_slice;
        slice_pitch = 2*element_in_slice;
        offset_in_slice = 0;
    }
    
    const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
    const size_t idx = offset + aligned_height*aligned_ofm_line + in_line;

    return idx;
}

#define GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size)  \
    FUNC_CALL(get_iy_xs_os_xsv2_osv_index)(                                     \
        o, i, y, x, CAT(prefix, _SIZE_X ),                                      \
        CAT(prefix, _IFM_PITCH),                                                \
        CAT(prefix, _Y_PITCH),                                                  \
        CAT(prefix, _X_PITCH),                                                  \
        CAT(prefix, _OFFSET),                                                   \
        sub_group_size)

inline uint FUNC(get_os_is_yx_isa8_osv8_isv4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
    const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
	const uint isv2_idx = i % 4;
	const uint osv_idx = o % 8;
	const uint isv1_idx = (i / 4) % 8;
	const uint is_idx = i / 32;
	const uint os_idx = o / 8;

	size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
	idx += x * 4 * 8 * 8;
	idx += y * size_x * 4 * 8 * 8;
	idx += is_idx * size_y * size_x * 4 * 8 * 8;
	idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 8;

    return idx;
}

#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4(prefix, o, i, y, x) \
	FUNC_CALL(get_os_is_yx_isa8_osv8_isv4_index)(                               \
        o, i, y, x, CAT(prefix, _SIZE_X ),                                      \
        CAT(prefix, _SIZE_Y),                                                \
        CAT(prefix, _IFM_NUM),                                                  \
        CAT(prefix, _OFM_NUM),                                                  \
        CAT(prefix, _OFFSET))


inline uint FUNC(get_is_o_yx_isv32_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
    const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
    const uint i_val = i % 32;
    const uint i_slice = i / 32;
    const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o + o_size * i_slice) ) );
    return idx;
}

#define GET_FILTER_IS_O_YX_ISV32(prefix, o, i, y, x)\
    FUNC_CALL(get_is_o_yx_isv32_index)(\
        o, i, y, x, CAT(prefix, _IFM_NUM),\
        CAT(prefix, _OFM_NUM),\
        CAT(prefix, _SIZE_X),\
        CAT(prefix, _SIZE_Y))

#define DECLARE_SAMPLER const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST

#if FP16_UNIT_USED
    #define IMAGE_READ(image, coord) read_imageh((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imageh((image), (coord), (val))
#else
    #define IMAGE_READ(image, coord) read_imagef((image), imageSampler, (coord))
    #define IMAGE_WRITE(image, coord, val) write_imagef((image), (coord), (val))
#endif